/*
 * ATI Radeon HD 5770 (Juniper) command processor (accel on).
 *
 * The microcode the driver uploads (PFP, ME, RLC) is stored, not run;
 * the CP is modelled at packet level. A CP_RB_WPTR write runs the ring
 * from the read pointer to the write pointer on the calling thread, with
 * the BQL: type-0 packets and SET_CONFIG_REG write registers, the other
 * SET_* packets fill the register file at 0x28000-0x3ffff (and its
 * shadow in memory when CONTEXT_CONTROL enables it), LOAD_* fill it from
 * memory, indirect buffers nest two deep. Fences, timestamps, memory
 * writes and semaphores complete at once; EVENT_WRITE_EOP raises IH
 * source 181 when CP_INT_CNTL enables time stamp interrupts. A
 * WAIT_REG_MEM or a semaphore wait whose condition is false stalls the
 * CP at that packet (GRBM_STATUS busy) and a timer polls it again.
 * Draws and dispatches are counted, nothing is rendered.
 *
 * Packets and registers the model does not act on are counted in the
 * `gaps` property; counters and a census of the state WindowServer
 * programs are in `cp-stats`.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/bswap.h"
#include "qemu/timer.h"
#include "hw/pci/pci_device.h"
#include "ati_hd5770_int.h"
#include "ati_hd5770_regs.h"
#include "trace.h"

#define CP_POLL_NS              (50 * SCALE_US)
#define GPU_CLOCK_HZ            27000000ull
#define NUM_BACKENDS            4       /* Juniper render backends */

enum {
    PKT_DONE,
    PKT_STALL,                  /* condition false: retry this packet */
    PKT_IB,                     /* entered an indirect buffer */
    PKT_SKIP,                   /* discard the next cp_pending_skip dwords */
    PKT_END,                    /* INDIRECT_BUFFER_END */
};

/* type-3 opcodes */
enum {
    OP_NOP = 0x10, OP_SET_BASE = 0x11, OP_CLEAR_STATE = 0x12,
    OP_INDEX_BUFFER_SIZE = 0x13, OP_DISPATCH_DIRECT = 0x15,
    OP_DISPATCH_INDIRECT = 0x16, OP_INDIRECT_BUFFER_END = 0x17,
    OP_MODE_CONTROL = 0x18, OP_SET_PREDICATION = 0x20, OP_REG_RMW = 0x21,
    OP_COND_EXEC = 0x22, OP_PRED_EXEC = 0x23, OP_DRAW_INDIRECT = 0x24,
    OP_DRAW_INDEX_INDIRECT = 0x25, OP_INDEX_BASE = 0x26,
    OP_DRAW_INDEX_2 = 0x27, OP_CONTEXT_CONTROL = 0x28,
    OP_DRAW_INDEX_OFFSET = 0x29, OP_INDEX_TYPE = 0x2a, OP_DRAW_INDEX = 0x2b,
    OP_DRAW_INDEX_AUTO = 0x2d, OP_DRAW_INDEX_IMMD = 0x2e,
    OP_NUM_INSTANCES = 0x2f, OP_DRAW_INDEX_MULTI_AUTO = 0x30,
    OP_INDIRECT_BUFFER = 0x32, OP_STRMOUT_BUFFER_UPDATE = 0x34,
    OP_DRAW_INDEX_OFFSET_2 = 0x35, OP_DRAW_INDEX_MULTI_ELEMENT = 0x36,
    OP_MEM_SEMAPHORE = 0x39, OP_COPY_DW = 0x3b, OP_WAIT_REG_MEM = 0x3c,
    OP_MEM_WRITE = 0x3d, OP_INDIRECT_BUFFER_3F = 0x3f, OP_CP_DMA = 0x41,
    OP_PFP_SYNC_ME = 0x42, OP_SURFACE_SYNC = 0x43, OP_ME_INITIALIZE = 0x44,
    OP_COND_WRITE = 0x45,
    OP_EVENT_WRITE = 0x46, OP_EVENT_WRITE_EOP = 0x47,
    OP_EVENT_WRITE_EOS = 0x48, OP_PREAMBLE_CNTL = 0x4a,
    OP_ONE_REG_WRITE = 0x57, OP_LOAD_FIRST = 0x60, OP_LOAD_LAST = 0x67,
    OP_SET_CONFIG_REG = 0x68, OP_SET_CONTEXT_REG = 0x69,
    OP_SET_ALU_CONST = 0x6a, OP_SET_CTL_CONST = 0x6f,
    OP_SET_RESOURCE_OFFSET = 0x70, OP_SET_ALU_CONST_VS = 0x71,
    OP_SET_ALU_CONST_DI = 0x72, OP_SET_APPEND_CNT = 0x75,
};

/*
 * Register classes of SET_* (0x68 + i) and LOAD_* (0x60 + i): byte base
 * of the class, and its bit in CONTEXT_CONTROL. ALU constants have no
 * register file on Evergreen.
 */
static const uint32_t class_base[8] = {
    EG_CONFIG_REG_FIRST, EG_CONTEXT_REG_FIRST, 0, EG_BOOL_CONST_FIRST,
    EG_LOOP_CONST_FIRST, EG_RESOURCE_FIRST, EG_SAMPLER_FIRST,
    EG_CTL_CONST_FIRST,
};

/* opcodes the model acts on or knows to need nothing */
static const bool op_known[256] = {
    [OP_NOP] = true, [OP_SET_BASE] = true, [OP_CLEAR_STATE] = true,
    [OP_INDEX_BUFFER_SIZE] = true, [OP_DISPATCH_DIRECT] = true,
    [OP_DISPATCH_INDIRECT] = true, [OP_INDIRECT_BUFFER_END] = true,
    [OP_MODE_CONTROL] = true, [OP_SET_PREDICATION] = true,
    [OP_REG_RMW] = true, [OP_COND_EXEC] = true, [OP_PRED_EXEC] = true,
    [OP_DRAW_INDIRECT] = true, [OP_DRAW_INDEX_INDIRECT] = true,
    [OP_INDEX_BASE] = true, [OP_DRAW_INDEX_2] = true,
    [OP_CONTEXT_CONTROL] = true, [OP_DRAW_INDEX_OFFSET] = true,
    [OP_INDEX_TYPE] = true, [OP_DRAW_INDEX] = true,
    [OP_DRAW_INDEX_AUTO] = true, [OP_DRAW_INDEX_IMMD] = true,
    [OP_NUM_INSTANCES] = true, [OP_DRAW_INDEX_MULTI_AUTO] = true,
    [OP_INDIRECT_BUFFER] = true, [OP_INDIRECT_BUFFER_3F] = true,
    [OP_STRMOUT_BUFFER_UPDATE] = true,
    [OP_DRAW_INDEX_OFFSET_2] = true, [OP_DRAW_INDEX_MULTI_ELEMENT] = true,
    [OP_MEM_SEMAPHORE] = true, [OP_COPY_DW] = true,
    [OP_WAIT_REG_MEM] = true, [OP_MEM_WRITE] = true, [OP_CP_DMA] = true,
    [OP_PFP_SYNC_ME] = true, [OP_SURFACE_SYNC] = true,
    [OP_ME_INITIALIZE] = true, [OP_COND_WRITE] = true,
    [OP_EVENT_WRITE] = true, [OP_EVENT_WRITE_EOP] = true,
    [OP_EVENT_WRITE_EOS] = true, [OP_PREAMBLE_CNTL] = true,
    [OP_LOAD_FIRST ... OP_LOAD_LAST] = true,
    [OP_SET_CONFIG_REG ... OP_SET_CTL_CONST] = true,
    [OP_SET_RESOURCE_OFFSET] = true, [OP_SET_ALU_CONST_VS] = true,
    [OP_SET_ALU_CONST_DI] = true, [OP_SET_APPEND_CNT] = true,
};

static bool op_is_draw(unsigned op)
{
    switch (op) {
    case OP_DISPATCH_DIRECT:
    case OP_DISPATCH_INDIRECT:
    case OP_DRAW_INDIRECT:
    case OP_DRAW_INDEX_INDIRECT:
    case OP_DRAW_INDEX_2:
    case OP_DRAW_INDEX_OFFSET:
    case OP_DRAW_INDEX:
    case OP_DRAW_INDEX_AUTO:
    case OP_DRAW_INDEX_IMMD:
    case OP_DRAW_INDEX_MULTI_AUTO:
    case OP_DRAW_INDEX_OFFSET_2:
    case OP_DRAW_INDEX_MULTI_ELEMENT:
        return true;
    default:
        return false;
    }
}

uint64_t ati_hd5770_gpu_clock(void)
{
    return muldiv64(qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL), GPU_CLOCK_HZ,
                    NANOSECONDS_PER_SECOND);
}

void ati_hd5770_census_add(HD5770Census *c, uint32_t val)
{
    uint32_t i;

    for (i = 0; i < c->used; i++) {
        if (c->val[i] == val) {
            c->cnt[i]++;
            return;
        }
    }
    if (c->used < HD5770_CENSUS_SLOTS) {
        c->val[c->used] = val;
        c->cnt[c->used++] = 1;
    } else {
        c->other++;
    }
}

static uint64_t addr40(uint32_t lo, uint32_t hi, uint32_t align_mask)
{
    return ((uint64_t)(hi & 0xff) << 32) | (lo & ~align_mask);
}

/* ---------------------------------------------------------------- */
/* Register file                                                    */

static void gfx_write(ATIHD5770State *s, uint32_t off, uint32_t val)
{
    uint32_t i = (off - EG_GFX_FIRST) >> 2;

    s->gfx[i] = val;
    if (off < EG_CONTEXT_REG_END) {
        s->census_ctx_writes[i]++;
        if (off == EG_CB_COLOR0_INFO) {
            ati_hd5770_census_add(&s->census_cb0, val);
        } else if (off == EG_SQ_PGM_START_PS) {
            ati_hd5770_census_add(&s->census_ps, val);
        } else if (off == EG_SQ_PGM_START_VS) {
            ati_hd5770_census_add(&s->census_vs, val);
        }
    }
}

/* a register write from a command stream */
static void cp_reg_write(ATIHD5770State *s, uint32_t off, uint32_t val)
{
    if (off < ATI_HD5770_MMIO_SIZE) {
        ati_hd5770_reg_write(s, off, val);
    } else if (off >= EG_GFX_FIRST && off < EG_GFX_END) {
        gfx_write(s, off, val);
    }
}

static uint32_t cp_reg_read(ATIHD5770State *s, uint32_t off)
{
    if (off < ATI_HD5770_MMIO_SIZE) {
        return ati_hd5770_reg_read(s, off);
    } else if (off >= EG_GFX_FIRST && off < EG_GFX_END) {
        return s->gfx[(off - EG_GFX_FIRST) >> 2];
    }
    return 0;
}

static void scratch_writeback(ATIHD5770State *s, unsigned n, uint32_t val)
{
    if (s->regs[EG_SCRATCH_UMSK >> 2] & (1u << n)) {
        ati_hd5770_mc_wr32(s, ((uint64_t)s->regs[EG_SCRATCH_ADDR >> 2] << 8) +
                           4 * n, val);
        s->cp.scratch_wb++;
    }
}

/* ---------------------------------------------------------------- */
/* Interrupts and fences                                            */

static void cp_irq(ATIHD5770State *s, uint32_t stat, uint32_t src)
{
    s->regs[EG_CP_INT_STATUS >> 2] |= stat;
    if (s->regs[EG_CP_INT_CNTL >> 2] & stat) {
        ati_hd5770_ih_push(s, src, 0);
    }
}

static bool compare(uint32_t func, uint32_t val, uint32_t ref)
{
    switch (func & 7) {
    case 0: return true;
    case 1: return val < ref;
    case 2: return val <= ref;
    case 3: return val == ref;
    case 4: return val != ref;
    case 5: return val >= ref;
    case 6: return val > ref;
    default: return false;
    }
}

static void eop(ATIHD5770State *s, const uint32_t *b)
{
    uint64_t addr = addr40(b[2], b[3], 3);
    uint32_t dsel = b[3] >> 29, isel = (b[3] >> 24) & 3;

    switch (dsel) {
    case 1:
        ati_hd5770_mc_wr32(s, addr, b[4]);
        break;
    case 2:
        ati_hd5770_mc_wr64(s, addr, b[4] | (uint64_t)b[5] << 32);
        break;
    case 3:
    case 4:
        ati_hd5770_mc_wr64(s, addr, ati_hd5770_gpu_clock());
        break;
    default:
        break;
    }
    s->cp.eop++;
    trace_ati_hd5770_eop(b[1], addr, dsel, isel, b[4]);
    if (isel) {
        if (s->regs[EG_CP_INT_CNTL >> 2] & EG_CP_TIME_STAMP_INT) {
            s->cp.eop_irq++;
        } else {
            s->cp.eop_masked++;
        }
        cp_irq(s, EG_CP_TIME_STAMP_INT, HD5770_IH_SRC_CP_EOP);
    }
}

/* EVENT_WRITE with an address: occlusion counts read as zero, valid */
static void event_write(ATIHD5770State *s, const uint32_t *b, uint32_t n)
{
    uint32_t index = (b[1] >> 8) & 0xf;
    uint64_t addr;
    unsigned i;

    if (n < 4 || index < 1 || index > 3) {
        return;
    }
    addr = addr40(b[2], b[3], 7);
    if (index == 1) {
        for (i = 0; i < NUM_BACKENDS; i++) {
            ati_hd5770_mc_wr64(s, addr + 16 * i, 1ull << 63);
        }
    } else {
        /* pipeline / streamout statistics: not written */
        s->cp.gap_op[OP_EVENT_WRITE]++;
    }
}

/* ---------------------------------------------------------------- */
/* Packets                                                          */

static void set_regs(ATIHD5770State *s, unsigned cls, const uint32_t *b,
                     uint32_t n)
{
    uint32_t base = class_base[cls], i;
    uint64_t shadow = s->cp_shadow_base[cls];
    bool shadowing = (s->cp_shadow_ctl & (1u << cls)) && shadow;

    if (!base || n < 2) {
        return;
    }
    for (i = 2; i < n; i++) {
        uint32_t idx = (b[1] & 0xffff) + (i - 2);

        cp_reg_write(s, base + idx * 4, b[i]);
        if (shadowing) {
            ati_hd5770_mc_wr32(s, shadow + idx * 4, b[i]);
            s->cp.shadow_writes++;
        }
    }
    s->cp.set_regs += n - 2;
}

static void load_regs(ATIHD5770State *s, unsigned cls, const uint32_t *b,
                      uint32_t n)
{
    uint32_t base = class_base[cls], i, j;
    uint64_t mem;

    if (n < 3) {
        return;
    }
    mem = addr40(b[1], b[2], 3);
    s->cp_shadow_base[cls] = mem;
    if (!base || !(s->cp_load_ctl & (1u << cls))) {
        return;
    }
    for (i = 3; i + 1 < n; i += 2) {
        uint32_t off = b[i] & 0xffff, cnt = b[i + 1] & 0x3fff;

        for (j = 0; j < cnt; j++) {
            cp_reg_write(s, base + (off + j) * 4,
                         ati_hd5770_mc_rd32(s, mem + (off + j) * 4));
        }
        s->cp.load_regs += cnt;
    }
}

static int cp_dma(ATIHD5770State *s, const uint32_t *b)
{
    uint32_t cmd = b[5], bytes = cmd & 0x1fffff;
    uint32_t src_sel = (b[2] >> 29) & 3;
    uint64_t src = addr40(b[1], b[2], 0), dst = addr40(b[3], b[4], 0);
    uint8_t buf[4096];

    if (cmd & ((1u << 26) | (1u << 27))) {
        return -1;              /* register source or destination */
    }
    if (src_sel == 2) {
        uint32_t i;

        for (i = 0; i + 4 <= bytes; i += 4) {
            ati_hd5770_mc_wr32(s, dst + i, b[1]);
        }
        return 0;
    }
    if (src_sel != 0) {
        return -1;
    }
    while (bytes) {
        uint32_t c = MIN(bytes, (uint32_t)sizeof(buf));

        ati_hd5770_mc_read(s, src, buf, c);
        ati_hd5770_mc_write(s, dst, buf, c);
        src += c;
        dst += c;
        bytes -= c;
    }
    return 0;
}

static void gap(ATIHD5770State *s, unsigned op)
{
    s->cp.gap_op[op & 0xff]++;
    trace_ati_hd5770_cp_gap(op);
}

/* one type-3 packet: b[0] header, n dwords in all */
static int cp_type3(ATIHD5770State *s, const uint32_t *b, uint32_t n,
                    unsigned level, uint32_t *skip)
{
    unsigned op = (b[0] >> 8) & 0xff;

    s->cp.type3++;
    s->cp.op[op]++;
    trace_ati_hd5770_cp_packet(level, op, n);
    if (op_is_draw(op)) {
        s->cp.draws++;
        ati_hd5770_census_add(&s->census_prim,
                              s->regs[EG_VGT_PRIMITIVE_TYPE >> 2] & 0x3f);
        return PKT_DONE;
    }
    switch (op) {
    case OP_INDIRECT_BUFFER:
    case OP_INDIRECT_BUFFER_3F: {   /* the blit manager's form */
        HD5770Stream *ib;

        if (n < 4 || level >= HD5770_CP_IB_LEVELS) {
            gap(s, op);
            return PKT_DONE;
        }
        ib = &s->cp_ib[level];
        ib->base = addr40(b[1], b[2], 3);
        ib->size = b[3] & 0xfffff;
        ib->pos = 0;
        s->cp.ibs++;
        return PKT_IB;
    }
    case OP_INDIRECT_BUFFER_END:
        return PKT_END;
    case OP_EVENT_WRITE_EOP:
        if (n >= 6) {
            eop(s, b);
        }
        return PKT_DONE;
    case OP_EVENT_WRITE:
        event_write(s, b, n);
        return PKT_DONE;
    case OP_EVENT_WRITE_EOS:
        if (n >= 5 && (b[3] >> 29) == 2) {
            ati_hd5770_mc_wr32(s, addr40(b[2], b[3], 3), b[4]);
        }
        return PKT_DONE;
    case OP_MEM_WRITE:
        if (n >= 5) {
            uint64_t a = addr40(b[1], b[2], 3);

            if (b[2] & (1u << 16)) {
                ati_hd5770_mc_wr64(s, a, ati_hd5770_gpu_clock());
            } else if (b[2] & (1u << 18)) {
                ati_hd5770_mc_wr32(s, a, b[3]);
            } else {
                ati_hd5770_mc_wr64(s, a, b[3] | (uint64_t)b[4] << 32);
            }
        }
        return PKT_DONE;
    case OP_WAIT_REG_MEM: {
        uint32_t v;

        if (n < 7) {
            return PKT_DONE;
        }
        v = (b[1] & (1u << 4)) ? ati_hd5770_mc_rd32(s, addr40(b[2], b[3], 3))
                               : cp_reg_read(s, (b[2] & 0xffff) << 2);
        if (!compare(b[1], v & b[5], b[4])) {
            trace_ati_hd5770_cp_wait(b[1], b[2], v, b[4], b[5]);
            return PKT_STALL;
        }
        s->cp.wait_ok++;
        return PKT_DONE;
    }
    case OP_COND_WRITE: {
        uint32_t v;

        if (n < 9) {
            return PKT_DONE;
        }
        v = (b[1] & (1u << 4)) ? ati_hd5770_mc_rd32(s, addr40(b[2], b[3], 3))
                               : cp_reg_read(s, (b[2] & 0xffff) << 2);
        if (compare(b[1], v & b[5], b[4])) {
            if (b[1] & (1u << 8)) {
                ati_hd5770_mc_wr32(s, addr40(b[6], b[7], 3), b[8]);
            } else {
                cp_reg_write(s, (b[6] & 0xffff) << 2, b[8]);
            }
        }
        return PKT_DONE;
    }
    case OP_MEM_SEMAPHORE: {
        uint64_t a;
        uint32_t v;

        if (n < 3) {
            return PKT_DONE;
        }
        a = addr40(b[1], b[2], 7);
        v = ati_hd5770_mc_rd32(s, a);
        if ((b[2] >> 29) == 7) {
            if (!v) {
                return PKT_STALL;
            }
            ati_hd5770_mc_wr32(s, a, v - 1);
            s->cp.sem_wait++;
        } else {
            ati_hd5770_mc_wr32(s, a, (b[2] & (1u << 20)) ? 1 : v + 1);
            s->cp.sem_signal++;
        }
        return PKT_DONE;
    }
    case OP_COND_EXEC:
        if (n >= 4 && !ati_hd5770_mc_rd32(s, addr40(b[1], b[2], 3))) {
            *skip = b[n - 1] & 0x3fff;
            return PKT_SKIP;
        }
        return PKT_DONE;
    case OP_PRED_EXEC:
        /* a single device: every device select includes it */
        return PKT_DONE;
    case OP_CONTEXT_CONTROL:
        if (n >= 3) {
            if (b[1] & (1u << 31)) {
                s->cp_load_ctl = b[1] & 0x1fff;
            }
            if (b[2] & (1u << 31)) {
                s->cp_shadow_ctl = b[2] & 0x1fff;
            }
        }
        return PKT_DONE;
    case OP_LOAD_FIRST ... OP_LOAD_LAST:
        load_regs(s, op - OP_LOAD_FIRST, b, n);
        return PKT_DONE;
    case OP_SET_CONFIG_REG ... OP_SET_CTL_CONST:
        set_regs(s, op - OP_SET_CONFIG_REG, b, n);
        return PKT_DONE;
    case OP_REG_RMW:
        if (n >= 4) {
            uint32_t off = (b[1] & 0xffff) << 2;

            cp_reg_write(s, off, (cp_reg_read(s, off) & b[2]) | b[3]);
        }
        return PKT_DONE;
    case OP_COPY_DW:
        if (n >= 6) {
            uint32_t v = (b[1] & 1)
                ? ati_hd5770_mc_rd32(s, addr40(b[2], b[3], 3))
                : cp_reg_read(s, (b[2] & 0xffff) << 2);

            if (b[1] & 2) {
                ati_hd5770_mc_wr32(s, addr40(b[4], b[5], 3), v);
            } else {
                cp_reg_write(s, (b[4] & 0xffff) << 2, v);
            }
        }
        return PKT_DONE;
    case OP_CP_DMA:
        if (n >= 6 && cp_dma(s, b) < 0) {
            gap(s, op);
        }
        return PKT_DONE;
    case OP_STRMOUT_BUFFER_UPDATE:
        if (n >= 4 && (b[1] & 1)) {
            ati_hd5770_mc_wr32(s, addr40(b[2], b[3], 3), 0);
        }
        return PKT_DONE;
    default:
        if (!op_known[op]) {
            trace_ati_hd5770_cp_gap(op);
        }
        return PKT_DONE;
    }
}

/* packet length in dwords from its header, 0 if malformed */
static uint32_t pkt_len(uint32_t h)
{
    switch (h >> 30) {
    case 0:
    case 3:
        return ((h >> 16) & 0x3fff) + 2;
    case 2:
        return 1;
    default:
        return 0;
    }
}

static int cp_exec(ATIHD5770State *s, const uint32_t *b, uint32_t n,
                   unsigned level, uint32_t *skip)
{
    uint32_t i;

    s->cp.dwords += n;
    switch (b[0] >> 30) {
    case 0: {
        uint32_t reg = (b[0] & 0xffff) << 2;

        s->cp.type0++;
        trace_ati_hd5770_cp_type0(level, b[0], n > 1 ? b[1] : 0);
        for (i = 1; i < n; i++) {
            cp_reg_write(s, reg + (i - 1) * 4, b[i]);
        }
        s->cp.type0_regs += n - 1;
        return PKT_DONE;
    }
    case 2:
        s->cp.type2++;
        return PKT_DONE;
    default:
        return cp_type3(s, b, n, level, skip);
    }
}

/* ---------------------------------------------------------------- */
/* Ring and indirect buffers                                        */

static uint32_t ring_dwords(ATIHD5770State *s)
{
    uint32_t sz = s->regs[EG_CP_RB_CNTL >> 2] & EG_CP_RB_BUFSZ_MASK;

    return 2u << MIN(sz, 22u);
}

static void fetch(ATIHD5770State *s, uint64_t base, uint32_t idx,
                  uint32_t mask, uint32_t n, uint32_t *buf)
{
    bool swap = ((s->regs[EG_CP_RB_CNTL >> 2] >> 16) & 3) == 2;
    uint32_t i, done = 0;

    while (done < n) {
        uint32_t c = n - done;

        if (mask) {
            c = MIN(c, mask + 1 - idx);
        }
        ati_hd5770_mc_read(s, base + (uint64_t)idx * 4, buf + done, c * 4);
        done += c;
        idx = mask ? (idx + c) & mask : idx + c;
    }
    for (i = 0; i < n; i++) {
        buf[i] = swap ? bswap32(le32_to_cpu(buf[i])) : le32_to_cpu(buf[i]);
    }
}

static void rptr_writeback(ATIHD5770State *s)
{
    if (s->regs[EG_CP_RB_CNTL >> 2] & EG_CP_RB_NO_UPDATE) {
        return;
    }
    ati_hd5770_mc_wr32(s, addr40(s->regs[EG_CP_RB_RPTR_ADDR >> 2],
                                 s->regs[EG_CP_RB_RPTR_ADDR_HI >> 2], 3),
                       s->cp_rptr);
    s->cp.rptr_wb++;
}

/* the innermost indirect buffer until it ends, stalls or nests */
static int ib_step(ATIHD5770State *s)
{
    unsigned level = s->cp_depth;
    HD5770Stream *ib = &s->cp_ib[level - 1];
    uint32_t h, n, skip = 0;
    int r;

    if (ib->pos >= ib->size) {
        return PKT_END;
    }
    fetch(s, ib->base, ib->pos, 0, 1, &h);
    n = pkt_len(h);
    if (!n) {
        s->cp.bad_header++;
        ib->pos++;
        return PKT_DONE;
    }
    if (ib->pos + n > ib->size) {
        s->cp.ib_overrun++;
        return PKT_END;
    }
    fetch(s, ib->base, ib->pos, 0, n, s->cp_pkt);
    r = cp_exec(s, s->cp_pkt, n, level, &skip);
    if (r == PKT_STALL) {
        return r;
    }
    ib->pos += n;
    if (r == PKT_SKIP) {
        ib->pos = MIN(ib->pos + skip, ib->size);
        s->cp.skipped += skip;
    } else if (r == PKT_IB) {
        s->cp_depth++;
    }
    return r == PKT_END ? PKT_END : PKT_DONE;
}

static void cp_run(ATIHD5770State *s)
{
    uint32_t size, mask, wptr, rptr0 = s->cp_rptr;
    uint64_t base;

    if (s->cp_running ||
        (s->regs[EG_CP_ME_CNTL >> 2] & EG_CP_ME_HALT) ||
        !s->regs[EG_CP_RB_BASE >> 2]) {
        return;
    }
    s->cp_running = true;
    s->cp_stalled = false;
    size = ring_dwords(s);
    mask = size - 1;
    base = (uint64_t)s->regs[EG_CP_RB_BASE >> 2] << 8;
    wptr = s->regs[EG_CP_RB_WPTR >> 2] & mask;
    s->cp_rptr &= mask;

    for (;;) {
        uint32_t avail, h, n, skip = 0;
        int r;

        if (s->cp_depth) {
            r = ib_step(s);
            if (r == PKT_STALL) {
                s->cp_stalled = true;
                break;
            }
            if (r == PKT_END) {
                s->cp_depth--;
            }
            continue;
        }
        avail = (wptr - s->cp_rptr) & mask;
        if (!avail) {
            break;
        }
        if (s->cp_skip) {
            uint32_t c = MIN(s->cp_skip, avail);

            s->cp_rptr = (s->cp_rptr + c) & mask;
            s->cp_skip -= c;
            continue;
        }
        fetch(s, base, s->cp_rptr, mask, 1, &h);
        n = pkt_len(h);
        if (!n) {
            s->cp.bad_header++;
            s->cp_rptr = (s->cp_rptr + 1) & mask;
            continue;
        }
        if (n > avail) {
            break;              /* the rest is not written yet */
        }
        fetch(s, base, s->cp_rptr, mask, n, s->cp_pkt);
        r = cp_exec(s, s->cp_pkt, n, 0, &skip);
        if (r == PKT_STALL) {
            s->cp_stalled = true;
            break;
        }
        s->cp_rptr = (s->cp_rptr + n) & mask;
        if (r == PKT_SKIP) {
            s->cp_skip = skip;
            s->cp.skipped += skip;
        } else if (r == PKT_IB) {
            s->cp_depth = 1;
        }
    }
    if (s->cp_rptr != rptr0) {
        rptr_writeback(s);
    }
    s->cp_running = false;
    if (s->cp_stalled) {
        s->cp.stalls++;
        timer_mod(s->cp_timer, qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) +
                  CP_POLL_NS);
    }
}

static void cp_timer_cb(void *opaque)
{
    ATIHD5770State *s = opaque;

    if (s->cp_stalled) {
        cp_run(s);
    }
    ati_hd5770_dma_run(s);
}

/* ---------------------------------------------------------------- */
/* Registers                                                        */

/* the register ranges of the engines, known with accel on */
bool ati_hd5770_cp_known(uint32_t off)
{
    return (off >= EG_CONFIG_REG_FIRST && off < EG_CONFIG_REG_END) ||
           (off >= 0xc100 && off < 0xc200) ||
           (off >= EG_DMA_FIRST && off <= EG_DMA_LAST);
}

bool ati_hd5770_cp_read(ATIHD5770State *s, uint32_t off, uint32_t *val)
{
    switch (off) {
    case EG_CP_RB_RPTR:
        *val = s->cp_rptr;
        return true;
    case EG_GRBM_STATUS:
        *val = s->cp_stalled ? EG_GRBM_STATUS_GUI_ACTIVE | EG_GRBM_CP_BUSY : 0;
        return true;
    case EG_CP_STAT:
        *val = s->cp_stalled ? 0x80000000u : 0;
        return true;
    case EG_CP_COHER_STATUS:
        *val = 0;
        return true;
    case EG_CP_PFP_UCODE_DATA: {
        uint32_t a = s->regs[EG_CP_PFP_UCODE_ADDR >> 2];

        *val = s->pfp_ucode[a % HD5770_UCODE_DWORDS];
        s->regs[EG_CP_PFP_UCODE_ADDR >> 2] = a + 1;
        return true;
    }
    case EG_CP_ME_RAM_DATA: {
        uint32_t a = s->regs[EG_CP_ME_RAM_RADDR >> 2];

        *val = s->me_ucode[a % HD5770_UCODE_DWORDS];
        s->regs[EG_CP_ME_RAM_RADDR >> 2] = a + 1;
        return true;
    }
    case EG_RLC_GPU_CLOCK_COUNT_LSB:
        *val = s->gpu_clock_capture;
        return true;
    case EG_RLC_GPU_CLOCK_COUNT_MSB:
        *val = s->gpu_clock_capture >> 32;
        return true;
    case EG_GB_ADDR_CONFIG:
        *val = s->regs[off >> 2] ? s->regs[off >> 2]
                                 : EG_GB_ADDR_CONFIG_JUNIPER;
        return true;
    default:
        return ati_hd5770_dma_read(s, off, val);
    }
}

bool ati_hd5770_cp_write(ATIHD5770State *s, uint32_t off, uint32_t val)
{
    if (ati_hd5770_mc_reg_write(s, off, val)) {
        return true;
    }
    if (off >= EG_SCRATCH_REG0 && off <= EG_SCRATCH_REG7) {
        s->regs[off >> 2] = val;
        scratch_writeback(s, (off - EG_SCRATCH_REG0) >> 2, val);
        return true;
    }
    switch (off) {
    case EG_CP_RB_WPTR:
        s->regs[off >> 2] = val;
        s->cp.kicks++;
        cp_run(s);
        return true;
    case EG_CP_RB_RPTR_WR:
        s->regs[off >> 2] = val;
        if (s->regs[EG_CP_RB_CNTL >> 2] & EG_CP_RB_RPTR_WR_ENA) {
            s->cp_rptr = val & 0xfffff;
            s->cp_skip = 0;
        }
        return true;
    case EG_CP_ME_CNTL:
        s->regs[off >> 2] = val;
        if (!(val & EG_CP_ME_HALT)) {
            cp_run(s);
        }
        return true;
    case EG_GRBM_SOFT_RESET:
        s->regs[off >> 2] = val;
        if (val & EG_GRBM_SOFT_RESET_CP) {
            s->cp_depth = 0;
            s->cp_skip = 0;
            s->cp_stalled = false;
            s->cp.resets++;
        }
        return true;
    case EG_CP_INT_STATUS:
        if (s->cp_running) {
            /* from a command stream: raise the interrupt the bit names */
            if (val & EG_CP_RB_INT) {
                cp_irq(s, EG_CP_RB_INT, HD5770_IH_SRC_CP_RB);
                s->cp.rb_irq++;
            }
            if (val & EG_CP_IB1_INT) {
                cp_irq(s, EG_CP_IB1_INT, HD5770_IH_SRC_CP_IB1);
            }
            if (val & EG_CP_IB2_INT) {
                cp_irq(s, EG_CP_IB2_INT, HD5770_IH_SRC_CP_IB2);
            }
        } else {
            s->regs[off >> 2] &= ~val;      /* acknowledge */
        }
        return true;
    case EG_CP_PFP_UCODE_DATA: {
        uint32_t a = s->regs[EG_CP_PFP_UCODE_ADDR >> 2];

        s->pfp_ucode[a % HD5770_UCODE_DWORDS] = val;
        s->regs[EG_CP_PFP_UCODE_ADDR >> 2] = a + 1;
        s->cp.ucode_pfp++;
        return true;
    }
    case EG_CP_ME_RAM_DATA: {
        uint32_t a = s->regs[EG_CP_ME_RAM_WADDR >> 2];

        s->me_ucode[a % HD5770_UCODE_DWORDS] = val;
        s->regs[EG_CP_ME_RAM_WADDR >> 2] = a + 1;
        s->cp.ucode_me++;
        return true;
    }
    case EG_RLC_UCODE_DATA: {
        uint32_t a = s->regs[EG_RLC_UCODE_ADDR >> 2];

        s->rlc_ucode[a % HD5770_UCODE_DWORDS] = val;
        s->regs[EG_RLC_UCODE_ADDR >> 2] = a + 1;
        s->cp.ucode_rlc++;
        return true;
    }
    case EG_RLC_CAPTURE_GPU_CLOCK_COUNT:
        s->regs[off >> 2] = val;
        s->gpu_clock_capture = ati_hd5770_gpu_clock();
        return true;
    default:
        return ati_hd5770_dma_write(s, off, val);
    }
}

/* ---------------------------------------------------------------- */
/* Reports                                                          */

void ati_hd5770_cp_gaps(ATIHD5770State *s, GString *out)
{
    unsigned i;

    for (i = 0; i < 256; i++) {
        if (s->cp.op[i] && !op_known[i]) {
            g_string_append_printf(out, "%spacket3 0x%02x: %" PRIu64
                                   " (not modelled)", out->len ? "\n" : "",
                                   i, s->cp.op[i]);
        } else if (s->cp.gap_op[i]) {
            g_string_append_printf(out, "%spacket3 0x%02x: %" PRIu64
                                   " of %" PRIu64 " with an unmodelled mode",
                                   out->len ? "\n" : "", i, s->cp.gap_op[i],
                                   s->cp.op[i]);
        }
    }
    if (s->cp.bad_header || s->cp.ib_overrun) {
        g_string_append_printf(out, "%sbad packet headers %" PRIu64
                               ", indirect buffer overruns %" PRIu64,
                               out->len ? "\n" : "", s->cp.bad_header,
                               s->cp.ib_overrun);
    }
    if (s->mc.gart_invalid || s->mc.fault_reads || s->mc.fault_writes) {
        g_string_append_printf(out, "%sGART invalid entries %" PRIu64
                               ", dropped reads %" PRIu64 ", writes %" PRIu64,
                               out->len ? "\n" : "", s->mc.gart_invalid,
                               s->mc.fault_reads, s->mc.fault_writes);
    }
    if (s->dma.sub_gap) {
        g_string_append_printf(out, "%sDMA packets not modelled %" PRIu64,
                               out->len ? "\n" : "", s->dma.sub_gap);
    }
}

static void census_print(GString *out, const char *name, HD5770Census *c)
{
    uint32_t i;

    g_string_append_printf(out, "%s:", name);
    for (i = 0; i < c->used; i++) {
        g_string_append_printf(out, " 0x%x x%" PRIu64, c->val[i], c->cnt[i]);
    }
    if (c->other) {
        g_string_append_printf(out, " other x%" PRIu64, c->other);
    }
    g_string_append_c(out, '\n');
}

void ati_hd5770_cp_stats(ATIHD5770State *s, GString *out)
{
    HD5770CPStats *c = &s->cp;
    unsigned i, n;

    g_string_append_printf(out,
        "CP: kicks %" PRIu64 " dwords %" PRIu64 " type0 %" PRIu64
        " type2 %" PRIu64 " type3 %" PRIu64 " ibs %" PRIu64 "\n"
        "ring: base 0x%08x cntl 0x%08x rptr 0x%x wptr 0x%x depth %u "
        "stalled %d me_cntl 0x%08x int_cntl 0x%08x\n"
        "fences: eop %" PRIu64 " eop_irq %" PRIu64 " eop_masked %" PRIu64
        " rb_irq %" PRIu64 " scratch_wb %" PRIu64 " rptr_wb %" PRIu64 "\n"
        "waits ok %" PRIu64 " stalls %" PRIu64 " sem signal %" PRIu64
        " wait %" PRIu64 " skipped %" PRIu64 " resets %" PRIu64 "\n"
        "draws %" PRIu64 " regs: type0 %" PRIu64 " set %" PRIu64
        " load %" PRIu64 " shadow %" PRIu64 "\n"
        "ucode dwords: pfp %" PRIu64 " me %" PRIu64 " rlc %" PRIu64 "\n"
        "MC: gart %" PRIu64 " invalid %" PRIu64 " tlb_inval %" PRIu64 "\n"
        "scratch:",
        c->kicks, c->dwords, c->type0, c->type2, c->type3, c->ibs,
        s->regs[EG_CP_RB_BASE >> 2], s->regs[EG_CP_RB_CNTL >> 2],
        s->cp_rptr, s->regs[EG_CP_RB_WPTR >> 2], s->cp_depth, s->cp_stalled,
        s->regs[EG_CP_ME_CNTL >> 2], s->regs[EG_CP_INT_CNTL >> 2],
        c->eop, c->eop_irq, c->eop_masked, c->rb_irq, c->scratch_wb,
        c->rptr_wb, c->wait_ok, c->stalls, c->sem_signal, c->sem_wait,
        c->skipped, c->resets, c->draws, c->type0_regs, c->set_regs,
        c->load_regs, c->shadow_writes, c->ucode_pfp, c->ucode_me,
        c->ucode_rlc, s->mc.gart, s->mc.gart_invalid, s->mc.tlb_inval);
    for (i = 0; i < 8; i++) {
        g_string_append_printf(out, " 0x%x",
                               s->regs[(EG_SCRATCH_REG0 >> 2) + i]);
    }
    g_string_append(out, "\npackets:");
    for (i = 0; i < 256; i++) {
        if (c->op[i]) {
            g_string_append_printf(out, " %02x:%" PRIu64, i, c->op[i]);
        }
    }
    g_string_append_c(out, '\n');
    census_print(out, "CB_COLOR0_INFO", &s->census_cb0);
    census_print(out, "SQ_PGM_START_PS", &s->census_ps);
    census_print(out, "SQ_PGM_START_VS", &s->census_vs);
    census_print(out, "primitive types", &s->census_prim);
    g_string_append(out, "context registers written:");
    for (i = 0, n = 0; i < ARRAY_SIZE(s->census_ctx_writes); i++) {
        if (s->census_ctx_writes[i]) {
            g_string_append_printf(out, "%s0x%05x:%" PRIu64,
                                   n++ % 8 ? " " : "\n ",
                                   EG_CONTEXT_REG_FIRST + i * 4,
                                   s->census_ctx_writes[i]);
        }
    }
    g_string_append_c(out, '\n');
    ati_hd5770_dma_stats(s, out);
}

/* ---------------------------------------------------------------- */

void ati_hd5770_cp_reset(ATIHD5770State *s)
{
    if (s->cp_timer) {
        timer_del(s->cp_timer);
    }
    memset(s->gfx, 0, HD5770_GFX_DWORDS * 4);
    memset(s->pfp_ucode, 0, sizeof(s->pfp_ucode));
    memset(s->me_ucode, 0, sizeof(s->me_ucode));
    memset(s->rlc_ucode, 0, sizeof(s->rlc_ucode));
    s->cp_rptr = 0;
    s->cp_skip = 0;
    s->cp_depth = 0;
    s->cp_stalled = false;
    s->cp_running = false;
    s->cp_load_ctl = 0;
    s->cp_shadow_ctl = 0;
    memset(s->cp_shadow_base, 0, sizeof(s->cp_shadow_base));
    ati_hd5770_dma_reset(s);
}

void ati_hd5770_cp_init(ATIHD5770State *s)
{
    s->gfx = g_new0(uint32_t, HD5770_GFX_DWORDS);
    s->gfx_dwords = HD5770_GFX_DWORDS;
    s->cp_pkt = g_new0(uint32_t, HD5770_CP_MAX_PKT);
    s->cp_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL, cp_timer_cb, s);
}

void ati_hd5770_cp_fini(ATIHD5770State *s)
{
    timer_free(s->cp_timer);
    s->cp_timer = NULL;
    g_free(s->gfx);
    s->gfx = NULL;
    g_free(s->cp_pkt);
    s->cp_pkt = NULL;
}
