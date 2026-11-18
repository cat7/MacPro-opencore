/*
 * ATI Radeon HD 5770 (Juniper) command processor (accel on).
 *
 * The microcode the driver uploads (PFP, ME, RLC) is stored, not run;
 * the CP is modelled at packet level. A CP_RB_WPTR write runs the ring
 * from the read pointer to the write pointer (on the command processor
 * thread, see ENGINE THREAD below, or with cp-thread=off on the calling
 * thread), with reg_lock: type-0 packets and SET_CONFIG_REG write
 * registers, the other SET_* packets fill the register file at
 * 0x28000-0x3ffff (and its shadow in memory when CONTEXT_CONTROL enables
 * it), LOAD_* fill it from memory, indirect buffers nest two deep. Fences, timestamps, memory
 * writes and semaphores complete at once; EVENT_WRITE_EOP raises IH
 * source 181 when CP_INT_CNTL enables time stamp interrupts. A
 * WAIT_REG_MEM or a semaphore wait whose condition is false stalls the
 * CP at that packet (GRBM_STATUS busy) and a timer polls it again.
 * Draws go to the 3D engine (ati_hd5770_draw.c) at once; SET_ALU_CONST
 * fills the DX9 constant file the shaders read as constant buffer 0.
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
#include "qemu/main-loop.h"
#include "qemu/rcu.h"
#include "system/runstate.h"
#include "qemu/error-report.h"
#include "system/ramblock.h"
#include "system/physmem.h"
#include "hw/pci/pci_device.h"
#include "hw/core/cpu.h"
#include "ati_hd5770_int.h"
#include "ati_hd5770_gpu.h"
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
 * of the class, and its bit in CONTEXT_CONTROL. ALU constants go to the
 * constant file instead.
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

/* packets that write memory or raise interrupts */
static bool op_signals(unsigned op)
{
    switch (op) {
    case OP_EVENT_WRITE:
    case OP_EVENT_WRITE_EOP:
    case OP_EVENT_WRITE_EOS:
    case OP_MEM_WRITE:
    case OP_COND_WRITE:
    case OP_COPY_DW:
    case OP_MEM_SEMAPHORE:
    case OP_CP_DMA:
    case OP_STRMOUT_BUFFER_UPDATE:
        return true;
    default:
        return false;
    }
}

/* this thread is running a command stream */
static __thread bool cp_in_stream;

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

/* this thread holds reg_lock; one device per thread at a time */
static __thread bool reg_held;

void ati_hd5770_reg_lock(ATIHD5770State *s)
{
    qemu_mutex_lock(&s->reg_lock);
    reg_held = true;
}

void ati_hd5770_reg_unlock(ATIHD5770State *s)
{
    reg_held = false;
    qemu_mutex_unlock(&s->reg_lock);
}

bool ati_hd5770_reg_held(void)
{
    return reg_held;
}

/* the registers of the engines, under reg_lock: config, GRBM, CP, RLC */
bool ati_hd5770_engine_reg(uint32_t off)
{
    return (off >= EG_CONFIG_REG_FIRST && off < EG_CONFIG_REG_END) ||
           (off >= 0xc100 && off < 0xc200) ||
           (off >= EG_RLC_FIRST && off <= EG_RLC_LAST);
}

/*
 * The BQL for the command processor: reg_lock is released first (LOCK
 * ORDER in ati_hd5770.c) and taken back after. Returns whether the BQL
 * was taken here.
 */
static bool cp_bql_enter(ATIHD5770State *s)
{
    if (bql_locked()) {
        return false;
    }
    assert(reg_held);
    ati_hd5770_reg_unlock(s);
    bql_lock();
    return true;
}

static void cp_bql_leave(ATIHD5770State *s, bool taken)
{
    if (taken) {
        bql_unlock();
        ati_hd5770_reg_lock(s);
    }
}

static void gfx_write(ATIHD5770State *s, uint32_t off, uint32_t val)
{
    uint32_t i = (off - EG_GFX_FIRST) >> 2;

    s->gfx[i] = val;
    ati_hd5770_note_write(s, off, val);
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
        if (ati_hd5770_engine_reg(off)) {
            ati_hd5770_reg_write(s, off, val);
        } else {
            bool bql = cp_bql_enter(s);

            ati_hd5770_reg_write(s, off, val);
            s->cp.bql_regs += bql;
            s->cp.bql_reg_last = off;
            cp_bql_leave(s, bql);
        }
    } else if (off >= EG_GFX_FIRST && off < EG_GFX_END) {
        gfx_write(s, off, val);
    }
}

static uint32_t cp_reg_read(ATIHD5770State *s, uint32_t off)
{
    if (off < ATI_HD5770_MMIO_SIZE) {
        uint32_t v;
        bool bql;

        if (ati_hd5770_engine_reg(off)) {
            return ati_hd5770_reg_read(s, off);
        }
        bql = cp_bql_enter(s);
        v = ati_hd5770_reg_read(s, off);
        s->cp.bql_regs += bql;
        s->cp.bql_reg_last = off;
        cp_bql_leave(s, bql);
        return v;
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

/*
 * FENCES. What a command stream lets the guest learn -- a fence or a
 * timestamp in memory, a semaphore signal, a scratch register, an
 * interrupt -- follows the draws queued before it. With a backend that
 * completes fences (FENCES in ati_hd5770_gpu.h) and the engine thread,
 * a packet with such effects issued while draws are outstanding records
 * them and the command processor runs on; the backend retires the
 * record from its own thread once the draws are in memory. Records are
 * applied in ring order, with the BQL: by the backend's thread when the
 * BQL is free, else by fx_bh on the main loop or by a thread waiting for
 * them. A packet that reads what a record may change waits for the
 * records first (fx_settle()). eng_busy stays set until every record is
 * applied; a run that leaves draws queued ends with an empty record.
 * Without fences the command processor waits for the host GPU before
 * such a packet and applies its effects itself.
 *
 * fx_q and fx_cond under fx_lock; fx_retired (the last ticket the
 * backend retired) and fx_out (records not yet applied) are atomics.
 * LOCK ORDER: eng_lock -> fx_lock; the backend's thread takes the BQL
 * only by trylock.
 */
enum { FX_WR32, FX_WR64, FX_CLOCK, FX_SEM, FX_SCRATCH, FX_IRQ, FX_FLIP };

#define FX_MAX                  8

typedef struct HD5770Effect {
    uint8_t kind;
    uint32_t a;                 /* FX_SEM set to 1; FX_SCRATCH, FX_IRQ, FX_FLIP */
    uint64_t addr, val;
} HD5770Effect;

struct HD5770Fence {
    uint64_t ticket;            /* 0: nothing was outstanding */
    unsigned n;
    HD5770Effect e[FX_MAX];
};

/* with the BQL, or on the command processor with reg_lock */
static void fx_apply(ATIHD5770State *s, const HD5770Effect *e)
{
    uint8_t b[8];
    uint32_t v;

    switch (e->kind) {
    case FX_WR32:
        stl_le_p(b, e->val);
        ati_hd5770_mc_post(s, e->addr, b, 4);
        break;
    case FX_WR64:
        stq_le_p(b, e->val);
        ati_hd5770_mc_post(s, e->addr, b, 8);
        break;
    case FX_CLOCK:
        stq_le_p(b, ati_hd5770_gpu_clock());
        ati_hd5770_mc_post(s, e->addr, b, 8);
        break;
    case FX_SEM:
        ati_hd5770_mc_peek(s, e->addr, b, 4);
        v = e->a ? 1 : ldl_le_p(b) + 1;
        stl_le_p(b, v);
        ati_hd5770_mc_post(s, e->addr, b, 4);
        break;
    case FX_SCRATCH:
        qatomic_set(&s->regs[(EG_SCRATCH_REG0 >> 2) + e->a], e->val);
        if (qatomic_read(&s->regs[EG_SCRATCH_UMSK >> 2]) & (1u << e->a)) {
            stl_le_p(b, e->val);
            ati_hd5770_mc_post(s, ((uint64_t)s->regs[EG_SCRATCH_ADDR >> 2]
                                   << 8) + 4 * e->a, b, 4);
            s->cp.scratch_wb++;
        }
        break;
    case FX_IRQ:
        qatomic_or(&s->regs[EG_CP_INT_STATUS >> 2], e->a);
        if (qatomic_read(&s->regs[EG_CP_INT_CNTL >> 2]) & e->a) {
            if (bql_locked()) {
                ati_hd5770_ih_push(s, e->val, 0);
            } else {
                bool bql = cp_bql_enter(s);

                ati_hd5770_ih_push(s, e->val, 0);
                cp_bql_leave(s, bql);
            }
        }
        break;
    case FX_FLIP:
        qatomic_set(&s->flip_done, e->a);
        break;
    }
}

/* an effect of the current packet: recorded, or applied now */
static void fx_add(ATIHD5770State *s, uint8_t kind, uint64_t addr,
                   uint64_t val, uint32_t a)
{
    HD5770Effect e = { .kind = kind, .a = a, .addr = addr, .val = val };
    struct HD5770Fence *f = s->fx_cur;

    if (f) {
        assert(f->n < FX_MAX);
        f->e[f->n++] = e;
    } else {
        fx_apply(s, &e);
    }
}

static bool fx_ready(ATIHD5770State *s)
{
    struct HD5770Fence *f;
    bool r;

    qemu_mutex_lock(&s->fx_lock);
    f = g_queue_peek_head(&s->fx_q);
    r = f && f->ticket <= qatomic_read(&s->fx_retired);
    qemu_mutex_unlock(&s->fx_lock);
    return r;
}

/* eng_busy off once the thread is out of work and every record applied */
static void cp_idle_check(ATIHD5770State *s)
{
    qemu_mutex_lock(&s->eng_lock);
    if (!s->eng_running && !s->eng_kick && !qatomic_read(&s->fx_out) &&
        s->eng_busy) {
        qatomic_set(&s->eng_busy, false);
        qemu_cond_broadcast(&s->eng_idle_cond);
    }
    qemu_mutex_unlock(&s->eng_lock);
}

/* the retired records at the head of the queue, applied; with the BQL */
static void fx_drain(ATIHD5770State *s)
{
    struct HD5770Fence *f;
    unsigned i;

    for (;;) {
        qemu_mutex_lock(&s->fx_lock);
        f = g_queue_peek_head(&s->fx_q);
        if (!f || f->ticket > qatomic_read(&s->fx_retired)) {
            qemu_mutex_unlock(&s->fx_lock);
            return;
        }
        g_queue_pop_head(&s->fx_q);
        qemu_mutex_unlock(&s->fx_lock);
        for (i = 0; i < f->n; i++) {
            fx_apply(s, &f->e[i]);
        }
        g_free(f);
        qemu_mutex_lock(&s->fx_lock);
        qatomic_dec(&s->fx_out);
        qemu_cond_broadcast(&s->fx_cond);
        qemu_mutex_unlock(&s->fx_lock);
        if (!qatomic_read(&s->fx_out)) {
            cp_idle_check(s);
        }
    }
}

static void fx_bh_cb(void *opaque)
{
    ATIHD5770State *s = opaque;

    s->fx_bh_runs++;
    fx_drain(s);
}

/* the backend's thread: FENCES in ati_hd5770_gpu.h */
static void gpu_retired_cb(void *opaque, uint64_t ticket)
{
    ATIHD5770State *s = opaque;

    qatomic_set(&s->fx_retired, ticket);
    if (bql_trylock()) {
        fx_drain(s);
        bql_unlock();
        return;
    }
    qemu_bh_schedule(s->fx_bh);
    qemu_mutex_lock(&s->fx_lock);
    qemu_cond_broadcast(&s->fx_cond);
    qemu_mutex_unlock(&s->fx_lock);
    qemu_mutex_lock(&s->eng_lock);
    qemu_cond_broadcast(&s->eng_idle_cond);
    qemu_mutex_unlock(&s->eng_lock);
}

static void gpu_thread_cb(void *opaque, bool enter)
{
    if (enter) {
        rcu_register_thread();
    } else {
        rcu_unregister_thread();
    }
}

/* the record of a packet, issued as a fence of the backend */
static void fx_issue(ATIHD5770State *s, struct HD5770Fence *f)
{
    f->ticket = hd5770_gfx_fence(&s->gfx3d);
    if (!f->ticket && !qatomic_read(&s->fx_out)) {
        /* nothing outstanding after all */
        unsigned i;

        for (i = 0; i < f->n; i++) {
            fx_apply(s, &f->e[i]);
        }
        g_free(f);
        s->fx_now++;
        return;
    }
    qemu_mutex_lock(&s->fx_lock);
    qatomic_inc(&s->fx_out);
    g_queue_push_tail(&s->fx_q, f);
    qemu_mutex_unlock(&s->fx_lock);
    s->fx_records++;
    if (!f->ticket) {
        bool bql = cp_bql_enter(s);

        fx_drain(s);
        cp_bql_leave(s, bql);
    }
}

/*
 * Around a packet that may have effects: they are recorded while draws
 * or records are outstanding. Nests; without fences, waits for the host
 * GPU instead.
 */
static void fx_begin(ATIHD5770State *s)
{
    if (s->fx_depth++) {
        return;
    }
    if (!s->fx_on) {
        ati_hd5770_gpu_sync(s);
        return;
    }
    if (hd5770_gfx_busy(&s->gfx3d) || qatomic_read(&s->fx_out)) {
        s->fx_cur = g_new0(struct HD5770Fence, 1);
    }
}

static void fx_end(ATIHD5770State *s)
{
    struct HD5770Fence *f = s->fx_cur;

    if (--s->fx_depth || !f) {
        return;
    }
    s->fx_cur = NULL;
    if (!f->n) {
        g_free(f);
        return;
    }
    fx_issue(s, f);
}

/*
 * Before a packet reads memory or registers a record may change: every
 * record applied (reg_lock released while waiting).
 */
static void fx_settle(ATIHD5770State *s)
{
    if (!qatomic_read(&s->fx_out)) {
        return;
    }
    s->fx_settles++;
    ati_hd5770_reg_unlock(s);
    while (qatomic_read(&s->fx_out)) {
        bool ready;

        qemu_mutex_lock(&s->fx_lock);
        ready = g_queue_peek_head(&s->fx_q) &&
                ((struct HD5770Fence *)g_queue_peek_head(&s->fx_q))->ticket <=
                qatomic_read(&s->fx_retired);
        if (!ready && qatomic_read(&s->fx_out)) {
            qemu_cond_wait(&s->fx_cond, &s->fx_lock);
        }
        qemu_mutex_unlock(&s->fx_lock);
        if (ready) {
            bql_lock();
            fx_drain(s);
            bql_unlock();
        }
    }
    ati_hd5770_reg_lock(s);
}

/* the same before a register read: the registers records write */
static void fx_settle_reg(ATIHD5770State *s, uint32_t off)
{
    if ((off >= EG_SCRATCH_REG0 && off <= EG_SCRATCH_REG7) ||
        off == EG_CP_INT_STATUS) {
        fx_settle(s);
    }
}

/* the end of a run: the engine stays busy until its draws are in memory */
static void fx_run_end(ATIHD5770State *s)
{
    if (!s->fx_on) {
        ati_hd5770_gpu_sync(s);
    } else if (hd5770_gfx_busy(&s->gfx3d)) {
        fx_issue(s, g_new0(struct HD5770Fence, 1));
    }
}

/*
 * SCANOUT FLIPS. The surface CRTC 0 turns to may be one the backend
 * keeps newer than memory. The display goes on showing the surface
 * before it (ati_hd5770.c) until flip_done equals flip_req: the engine
 * has run every packet written to the ring before the flip, and the
 * fence issued at that point, which writes what lies over the scan
 * range, has retired.
 *
 * flip_req is odd while the scan range and flip_at (the ring's write
 * pointer at the flip) are being set.
 */

/*
 * Between two packets of the ring, read pointer rptr: the flip asked
 * for is recorded when the engine stands where the ring ended at the
 * flip (any: wherever it stands)
 */
static void cp_flip(ATIHD5770State *s, uint32_t rptr, uint32_t mask, bool any)
{
    uint32_t r = qatomic_read(&s->flip_req), at;
    HD5770Effect e = { .kind = FX_FLIP, .a = r };

    if (r == s->flip_seen || (r & 1)) {
        return;
    }
    at = qatomic_read(&s->flip_at);
    if (qatomic_read(&s->flip_req) != r || (!any && (at & mask) != rptr)) {
        return;
    }
    s->flip_seen = r;
    s->cp.flips++;
    if (!s->fx_on) {
        ati_hd5770_gpu_sync(s);
        fx_apply(s, &e);
    } else if (hd5770_gfx_busy(&s->gfx3d) || qatomic_read(&s->fx_out)) {
        struct HD5770Fence *f = g_new0(struct HD5770Fence, 1);

        f->e[f->n++] = e;
        fx_issue(s, f);
    } else {
        fx_apply(s, &e);
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
        fx_add(s, FX_WR32, addr, b[4], 0);
        break;
    case 2:
        fx_add(s, FX_WR64, addr, b[4] | (uint64_t)b[5] << 32, 0);
        break;
    case 3:
    case 4:
        fx_add(s, FX_CLOCK, addr, 0, 0);
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
        fx_add(s, FX_IRQ, 0, HD5770_IH_SRC_CP_EOP, EG_CP_TIME_STAMP_INT);
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
            fx_add(s, FX_WR64, addr + 16 * i, 1ull << 63, 0);
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

    if (cls == 2) {
        /* the DX9 constant file the CP keeps: PS 0-1023, VS 1024-2047 */
        for (i = 2; i < n; i++) {
            uint32_t idx = (b[1] & 0xffff) + (i - 2);

            if (idx < HD5770_ALU_CONST_DWORDS) {
                s->alu_const[idx] = b[i];
            }
        }
        s->cp.set_regs += n - 2;
        return;
    }
    if (!base || n < 2) {
        return;
    }
    for (i = 2; i < n; i++) {
        uint32_t idx = (b[1] & 0xffff) + (i - 2);

        cp_reg_write(s, base + idx * 4, b[i]);
        if (shadowing) {
            /* the CP's copy of the register: no wait for draws */
            uint8_t v[4];

            stl_le_p(v, b[i]);
            ati_hd5770_mc_post(s, shadow + idx * 4, v, 4);
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

/*
 * A wait for the host GPU: at a packet boundary, a memory access, when the
 * engine goes idle. It runs without reg_lock, and on the engine thread or
 * a vCPU without the BQL, as a draw does: the vCPUs and the main loop are
 * not held for the GPU's execution time.
 */
void ati_hd5770_gpu_sync(ATIHD5770State *s)
{
    bool reg = reg_held;
    bool bql = bql_locked() &&
               ((s->eng_on && qemu_thread_is_self(&s->eng_thread)) ||
                current_cpu);

    if (reg) {
        ati_hd5770_reg_unlock(s);
    }
    if (bql) {
        bql_unlock();
    }
    hd5770_gfx_sync(&s->gfx3d);
    if (bql) {
        bql_lock();
    }
    if (reg) {
        ati_hd5770_reg_lock(s);
    }
}

static void gap(ATIHD5770State *s, unsigned op)
{
    s->cp.gap_op[op & 0xff]++;
    trace_ati_hd5770_cp_gap(op);
}

static uint32_t pkt_len(uint32_t h);
static void fetch(ATIHD5770State *s, uint64_t base, uint32_t idx,
                  uint32_t mask, uint32_t n, uint32_t *buf);

/*
 * BUILD AHEAD (ati_hd5770_gfx.h): a draw of indirect buffer level built
 * a program. The packets after it (n dwords long) to the buffer's end
 * are walked with copies of the register files, so that what their
 * draws need is built at once and not draw by draw. Only state and
 * draws count; a packet whose outcome depends on memory is taken as
 * executed. Engine thread, no reg_lock.
 */
static void cp_ahead(ATIHD5770State *s, unsigned level, uint32_t n)
{
    HD5770Stream *ib = &s->cp_ib[level - 1];
    uint32_t *ctx, *cfg, *alu, *pkt;
    HD5770GfxAhead *a;
    uint32_t pos, h, i;

    ctx = g_memdup2(s->gfx, EG_GFX_END - EG_GFX_FIRST);
    cfg = g_memdup2(s->regs, sizeof(s->regs));
    alu = g_memdup2(s->alu_const, HD5770_ALU_CONST_DWORDS * 4);
    a = hd5770_gfx_ahead_begin(&s->gfx3d, ctx, cfg, alu);
    if (a) {
        pkt = g_new(uint32_t, 0x4002);
        s->cp.ahead++;
        for (pos = ib->pos + n; pos < ib->size; pos += n) {
            unsigned op;
            uint32_t reg = 0, first = 1;

            fetch(s, ib->base, pos, 0, 1, &h);
            n = pkt_len(h);
            if (!n || pos + n > ib->size) {
                break;
            }
            fetch(s, ib->base, pos, 0, n, pkt);
            op = (h >> 8) & 0xff;
            if (h >> 30 == 0) {
                reg = (h & 0xffff) << 2;
            } else if (h >> 30 != 3) {
                continue;
            } else if (op_is_draw(op)) {
                hd5770_gfx_draw(&s->gfx3d, pkt, n);
                ati_hd5770_mc_unmap_all(s);
                s->cp.ahead_draws++;
                continue;
            } else if (op == OP_INDIRECT_BUFFER ||
                       op == OP_INDIRECT_BUFFER_3F ||
                       op == OP_INDIRECT_BUFFER_END) {
                break;
            } else if (op >= OP_SET_CONFIG_REG && op <= OP_SET_CTL_CONST &&
                       n >= 2) {
                unsigned cls = op - OP_SET_CONFIG_REG;

                first = 2;
                if (cls == 2) {
                    for (i = 2; i < n; i++) {
                        uint32_t idx = (pkt[1] & 0xffff) + (i - 2);

                        if (idx < HD5770_ALU_CONST_DWORDS) {
                            alu[idx] = pkt[i];
                        }
                    }
                    continue;
                }
                if (!class_base[cls]) {
                    continue;
                }
                reg = class_base[cls] + (pkt[1] & 0xffff) * 4;
            } else {
                hd5770_gfx_state_packet(&s->gfx3d, pkt, n);
                continue;
            }
            for (i = first; i < n; i++) {
                uint32_t off = reg + (i - first) * 4;

                if (off >= EG_GFX_FIRST && off < EG_GFX_END) {
                    ctx[(off - EG_GFX_FIRST) >> 2] = pkt[i];
                } else if (off >= EG_CONFIG_REG_FIRST &&
                           off < EG_CONFIG_REG_END) {
                    cfg[off >> 2] = pkt[i];
                }
            }
        }
        hd5770_gfx_ahead_end(&s->gfx3d, a);
        g_free(pkt);
    }
    g_free(ctx);
    g_free(cfg);
    g_free(alu);
}

/* one type-3 packet: b[0] header, n dwords in all */
static int cp_type3_op(ATIHD5770State *s, const uint32_t *b, uint32_t n,
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
        if (s->eng_on) {
            int64_t t0 = get_clock();

            /* register accesses go on meanwhile */
            ati_hd5770_reg_unlock(s);
            hd5770_gfx_draw(&s->gfx3d, b, n);
            if (level && hd5770_gfx_built(&s->gfx3d)) {
                cp_ahead(s, level, n);
            }
            ati_hd5770_reg_lock(s);
            s->eng_draw_ns += get_clock() - t0;
        } else {
            hd5770_gfx_draw(&s->gfx3d, b, n);
        }
        ati_hd5770_mc_unmap_all(s);
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
            fx_add(s, FX_WR32, addr40(b[2], b[3], 3), b[4], 0);
        }
        return PKT_DONE;
    case OP_MEM_WRITE:
        if (n >= 5) {
            uint64_t a = addr40(b[1], b[2], 3);

            if (b[2] & (1u << 16)) {
                fx_add(s, FX_CLOCK, a, 0, 0);
            } else if (b[2] & (1u << 18)) {
                fx_add(s, FX_WR32, a, b[3], 0);
            } else {
                fx_add(s, FX_WR64, a, b[3] | (uint64_t)b[4] << 32, 0);
            }
        }
        return PKT_DONE;
    case OP_WAIT_REG_MEM: {
        uint32_t v;

        if (n < 7) {
            return PKT_DONE;
        }
        if (b[1] & (1u << 4)) {
            fx_settle(s);
            v = ati_hd5770_mc_rd32(s, addr40(b[2], b[3], 3));
        } else {
            fx_settle_reg(s, (b[2] & 0xffff) << 2);
            v = cp_reg_read(s, (b[2] & 0xffff) << 2);
        }
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
        if (b[1] & (1u << 4)) {
            fx_settle(s);
            v = ati_hd5770_mc_rd32(s, addr40(b[2], b[3], 3));
        } else {
            fx_settle_reg(s, (b[2] & 0xffff) << 2);
            v = cp_reg_read(s, (b[2] & 0xffff) << 2);
        }
        if (compare(b[1], v & b[5], b[4])) {
            if (b[1] & (1u << 8)) {
                fx_add(s, FX_WR32, addr40(b[6], b[7], 3), b[8], 0);
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
        if ((b[2] >> 29) == 7) {
            fx_settle(s);
            v = ati_hd5770_mc_rd32(s, a);
            if (!v) {
                return PKT_STALL;
            }
            ati_hd5770_mc_wr32(s, a, v - 1);
            s->cp.sem_wait++;
        } else {
            fx_add(s, FX_SEM, a, 0, !!(b[2] & (1u << 20)));
            s->cp.sem_signal++;
        }
        return PKT_DONE;
    }
    case OP_COND_EXEC:
        fx_settle(s);
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
            uint32_t v;

            if (b[1] & 1) {
                fx_settle(s);
                v = ati_hd5770_mc_rd32(s, addr40(b[2], b[3], 3));
            } else {
                fx_settle_reg(s, (b[2] & 0xffff) << 2);
                v = cp_reg_read(s, (b[2] & 0xffff) << 2);
            }
            if (b[1] & 2) {
                fx_add(s, FX_WR32, addr40(b[4], b[5], 3), v, 0);
            } else {
                cp_reg_write(s, (b[4] & 0xffff) << 2, v);
            }
        }
        return PKT_DONE;
    case OP_CP_DMA:
        /* a copy the guest sees: after the draws and records before it */
        fx_settle(s);
        ati_hd5770_gpu_sync(s);
        if (n >= 6 && cp_dma(s, b) < 0) {
            gap(s, op);
        }
        return PKT_DONE;
    case OP_INDEX_BUFFER_SIZE:
    case OP_INDEX_BASE:
    case OP_INDEX_TYPE:
    case OP_NUM_INSTANCES:
    case OP_SET_BASE:
        hd5770_gfx_state_packet(&s->gfx3d, b, n);
        return PKT_DONE;
    case OP_STRMOUT_BUFFER_UPDATE:
        if (n >= 4 && (b[1] & 1)) {
            fx_add(s, FX_WR32, addr40(b[2], b[3], 3), 0, 0);
        }
        return PKT_DONE;
    default:
        if (!op_known[op]) {
            trace_ati_hd5770_cp_gap(op);
        }
        return PKT_DONE;
    }
}

static int cp_type3(ATIHD5770State *s, const uint32_t *b, uint32_t n,
                    unsigned level, uint32_t *skip)
{
    unsigned op = (b[0] >> 8) & 0xff;
    int r;

    if (!op_signals(op)) {
        return cp_type3_op(s, b, n, level, skip);
    }
    /* what the guest may learn from it follows the draws before it */
    fx_begin(s);
    r = cp_type3_op(s, b, n, level, skip);
    fx_end(s);
    return r;
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
        ati_hd5770_mc_fetch(s, base + (uint64_t)idx * 4, buf + done, c * 4);
        done += c;
        idx = mask ? (idx + c) & mask : idx + c;
    }
    for (i = 0; i < n; i++) {
        buf[i] = swap ? bswap32(le32_to_cpu(buf[i])) : le32_to_cpu(buf[i]);
    }
}

/* what the CP has fetched: no wait for draws */
static void rptr_writeback(ATIHD5770State *s)
{
    uint8_t b[4];

    if (s->regs[EG_CP_RB_CNTL >> 2] & EG_CP_RB_NO_UPDATE) {
        return;
    }
    stl_le_p(b, s->cp_rptr);
    ati_hd5770_mc_post(s, addr40(s->regs[EG_CP_RB_RPTR_ADDR >> 2],
                                 s->regs[EG_CP_RB_RPTR_ADDR_HI >> 2], 3),
                       b, 4);
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

    if (s->cp_running) {
        return;
    }
    if ((s->regs[EG_CP_ME_CNTL >> 2] & EG_CP_ME_HALT) ||
        !s->regs[EG_CP_RB_BASE >> 2]) {
        cp_flip(s, 0, 0, true);
        return;
    }
    s->cp_running = true;
    cp_in_stream = true;
    s->cp_stalled = false;
    size = ring_dwords(s);
    mask = size - 1;
    base = (uint64_t)s->regs[EG_CP_RB_BASE >> 2] << 8;
    wptr = s->regs[EG_CP_RB_WPTR >> 2] & mask;
    s->cp_rptr &= mask;

    for (;;) {
        uint32_t avail, h, n, skip = 0;
        int r;

        if (qatomic_read(&s->eng_hold)) {
            /* stopped between two packets; a kick resumes */
            break;
        }
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
        cp_flip(s, s->cp_rptr, mask, false);
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
    if (!s->cp_depth && !s->cp_stalled && !qatomic_read(&s->eng_hold)) {
        /* a flip whose ring position was not met: at the ring's end */
        cp_flip(s, s->cp_rptr, mask,
                (s->regs[EG_CP_RB_WPTR >> 2] & mask) == s->cp_rptr);
    }
    /* the engine goes idle with its draws in memory */
    fx_run_end(s);
    if (s->cp_rptr != rptr0) {
        rptr_writeback(s);
    }
    s->cp_running = false;
    cp_in_stream = false;
    if (s->cp_stalled) {
        s->cp.stalls++;
        timer_mod(s->cp_timer, qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) +
                  CP_POLL_NS);
    }
}

/*
 * ENGINE THREAD (cp-thread=on). The ring runs on a thread of its own: a
 * kick (CP_RB_WPTR, ME_CNTL, the poll of a stalled wait) only wakes it.
 * The thread runs the ring with reg_lock, not the BQL, and drops it for
 * the length of each draw and of each wait for the host GPU, so packets
 * and their side effects -- registers, memory writes, fences, interrupts
 * -- happen in ring order, and a fence written after a draw is written
 * once the draw is in memory. A register the BQL owns (REGISTER ACCESS
 * in ati_hd5770.c) and an interrupt take the BQL, reg_lock released
 * first. GRBM_STATUS and CP_STAT read busy until the thread has run out
 * of work.
 *
 * eng_hold (set under the BQL) stops the ring at the next packet
 * boundary: set while the VM is stopped and for a reset, it leaves read
 * pointer and indirect buffer state consistent for migration.
 * cp_wait_idle() waits for the thread with the BQL released.
 *
 * LOCK ORDER: BQL -> reg_lock -> eng_lock; the thread holds no eng_lock
 * while it takes reg_lock.
 */
static void cp_kick(ATIHD5770State *s)
{
    if (!s->eng_on) {
        cp_run(s);
        return;
    }
    qemu_mutex_lock(&s->eng_lock);
    s->eng_kick = true;
    qatomic_set(&s->eng_busy, true);
    qemu_cond_signal(&s->eng_cond);
    qemu_mutex_unlock(&s->eng_lock);
}

/* SCANOUT FLIPS above; a run of the engine follows */
void ati_hd5770_cp_scan(ATIHD5770State *s, uint64_t lo, uint64_t hi)
{
    HD5770Gpu *gpu = s->gfx3d.gpu;

    qatomic_inc(&s->flip_req);
    qatomic_set(&gpu->scan_lo, lo);
    qatomic_set(&gpu->scan_hi, hi);
    qatomic_set(&s->flip_at, s->regs[EG_CP_RB_WPTR >> 2]);
    qatomic_inc(&s->flip_req);
    s->cp.scan_flush++;
    if (s->eng_on) {
        cp_kick(s);
    } else {
        qatomic_set(&s->flip_done, qatomic_read(&s->flip_req));
    }
}

/*
 * With the BQL, which is released while waiting; retired records are
 * applied here meanwhile (FENCES).
 */
static void cp_wait_idle(ATIHD5770State *s)
{
    if (!s->eng_on || qemu_thread_is_self(&s->eng_thread)) {
        return;
    }
    for (;;) {
        fx_drain(s);
        qemu_mutex_lock(&s->eng_lock);
        if (!s->eng_busy) {
            qemu_mutex_unlock(&s->eng_lock);
            return;
        }
        qemu_mutex_unlock(&s->eng_lock);
        bql_unlock();
        qemu_mutex_lock(&s->eng_lock);
        while (s->eng_busy && !fx_ready(s)) {
            qemu_cond_wait(&s->eng_idle_cond, &s->eng_lock);
        }
        s->eng_waits++;
        qemu_mutex_unlock(&s->eng_lock);
        bql_lock();
    }
}

/*
 * Before a CPU access through an HDP window the engine finishes what is
 * queued, so the access lands after the draws submitted before it (with
 * the BQL, which is released while waiting).
 */
void ati_hd5770_cp_drain(ATIHD5770State *s)
{
    if (qatomic_read(&s->eng_busy)) {
        cp_wait_idle(s);
    }
}

static void *cp_thread_fn(void *opaque)
{
    ATIHD5770State *s = opaque;

    rcu_register_thread();
    qemu_mutex_lock(&s->eng_lock);
    for (;;) {
        while (!s->eng_kick && !s->eng_quit) {
            qemu_cond_wait(&s->eng_cond, &s->eng_lock);
        }
        if (s->eng_quit) {
            break;
        }
        s->eng_kick = false;
        s->eng_running = true;
        qemu_mutex_unlock(&s->eng_lock);

        ati_hd5770_reg_lock(s);
        if (!qatomic_read(&s->eng_hold)) {
            cp_run(s);
            s->eng_runs++;
        }
        ati_hd5770_reg_unlock(s);

        qemu_mutex_lock(&s->eng_lock);
        s->eng_running = false;
        if (!s->eng_kick && !qatomic_read(&s->fx_out)) {
            qatomic_set(&s->eng_busy, false);
            qemu_cond_broadcast(&s->eng_idle_cond);
        }
    }
    qatomic_set(&s->eng_busy, false);
    qemu_cond_broadcast(&s->eng_idle_cond);
    qemu_mutex_unlock(&s->eng_lock);
    rcu_unregister_thread();
    return NULL;
}

static void cp_vm_state(void *opaque, bool running, RunState state)
{
    ATIHD5770State *s = opaque;

    if (running) {
        qatomic_set(&s->eng_hold, false);
        cp_kick(s);
    } else {
        qatomic_set(&s->eng_hold, true);
        cp_wait_idle(s);
    }
}

static void cp_timer_cb(void *opaque)
{
    ATIHD5770State *s = opaque;

    if (s->cp_stalled) {
        cp_kick(s);
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
        *val = qatomic_read(&s->cp_rptr);
        return true;
    case EG_GRBM_STATUS:
        *val = s->cp_stalled || qatomic_read(&s->eng_busy)
               ? EG_GRBM_STATUS_GUI_ACTIVE | EG_GRBM_CP_BUSY : 0;
        return true;
    case EG_CP_STAT:
        *val = s->cp_stalled || qatomic_read(&s->eng_busy) ? 0x80000000u : 0;
        return true;
    case EG_THREADS_ACTIVE:
        /*
         * Bit 30: work in progress. A guest driver whose timestamp is
         * late goes on waiting while it reads set and reports a hang
         * when it reads clear; set while the engine runs packets.
         */
        *val = qatomic_read(&s->eng_busy) && !s->cp_stalled ? 1u << 30 : 0;
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
        if (cp_in_stream) {
            /* from a command stream: after the draws before it */
            fx_begin(s);
            fx_add(s, FX_SCRATCH, 0, val, (off - EG_SCRATCH_REG0) >> 2);
            fx_end(s);
        } else {
            s->regs[off >> 2] = val;
            scratch_writeback(s, (off - EG_SCRATCH_REG0) >> 2, val);
        }
        return true;
    }
    switch (off) {
    case EG_CP_RB_WPTR:
        s->regs[off >> 2] = val;
        s->cp.kicks++;
        cp_kick(s);
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
            cp_kick(s);
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
        if (cp_in_stream) {
            /* from a command stream: raise the interrupt the bit names */
            fx_begin(s);
            if (val & EG_CP_RB_INT) {
                fx_add(s, FX_IRQ, 0, HD5770_IH_SRC_CP_RB, EG_CP_RB_INT);
                s->cp.rb_irq++;
            }
            if (val & EG_CP_IB1_INT) {
                fx_add(s, FX_IRQ, 0, HD5770_IH_SRC_CP_IB1, EG_CP_IB1_INT);
            }
            if (val & EG_CP_IB2_INT) {
                fx_add(s, FX_IRQ, 0, HD5770_IH_SRC_CP_IB2, EG_CP_IB2_INT);
            }
            fx_end(s);
        } else {
            /* acknowledge */
            qatomic_and(&s->regs[off >> 2], ~val);
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
    hd5770_gfx_gaps(&s->gfx3d, out);
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

    g_string_append_printf(out, "CP: runs for a scanout that moved %" PRIu64
                           ", built ahead %" PRIu64 " times over %" PRIu64
                           " draws\n", c->scan_flush, c->ahead,
                           c->ahead_draws);
    g_string_append_printf(out, "display: refreshes %" PRIu64 ", flips %"
                           PRIu64 " recorded, refreshes that kept the "
                           "surface before %" PRIu64 ", showed a flip not "
                           "recorded %" PRIu64 "\n", s->refreshes, c->flips,
                           s->flip_held, s->flip_late);
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
        "engine:",
        c->kicks, c->dwords, c->type0, c->type2, c->type3, c->ibs,
        s->regs[EG_CP_RB_BASE >> 2], s->regs[EG_CP_RB_CNTL >> 2],
        s->cp_rptr, s->regs[EG_CP_RB_WPTR >> 2], s->cp_depth, s->cp_stalled,
        s->regs[EG_CP_ME_CNTL >> 2], s->regs[EG_CP_INT_CNTL >> 2],
        c->eop, c->eop_irq, c->eop_masked, c->rb_irq, c->scratch_wb,
        c->rptr_wb, c->wait_ok, c->stalls, c->sem_signal, c->sem_wait,
        c->skipped, c->resets, c->draws, c->type0_regs, c->set_regs,
        c->load_regs, c->shadow_writes, c->ucode_pfp, c->ucode_me,
        c->ucode_rlc, s->mc.gart, s->mc.gart_invalid, s->mc.tlb_inval);
    g_string_append_printf(out, " lockless mmio %s, stream register "
                           "accesses with the BQL %" PRIu64 " (last 0x%x)\n"
                           " fences %s: records %" PRIu64 " outstanding %d "
                           "at once %" PRIu64 " waited for %" PRIu64
                           " applied on the main loop %" PRIu64 "\n",
                           s->mmio.lockless_io ? "on" : "off",
                           s->cp.bql_regs, s->cp.bql_reg_last,
                           s->fx_on ? "backend" : "device", s->fx_records,
                           qatomic_read(&s->fx_out), s->fx_now,
                           s->fx_settles, s->fx_bh_runs);
    g_string_append_printf(out, " thread %s runs %" PRIu64 " idle waits %"
                           PRIu64 " draw ms %" PRIu64 " scanout redraws %"
                           PRIu64 " (async %s, frames %" PRIu64
                           " deferred %" PRIu64 ")\nhdp: window 0x%" PRIx64
                           "-0x%" PRIx64 " cpu accesses %" PRIu64 "\nscratch:",
                           s->eng_on ? "on" : "off",
                           s->eng_runs, s->eng_waits, s->eng_draw_ns / 1000000,
                           s->redraws, s->disp_on ? "on" : "off",
                           s->disp_frames, s->disp_deferred, s->hdp_lo[0],
                           s->hdp_hi[0], s->hdp_accesses);
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
    hd5770_gfx_stats(&s->gfx3d, out);
    hd5770_gfx_gpu_stats(&s->gfx3d, out);
    ati_hd5770_dma_stats(s, out);
}

/* ---------------------------------------------------------------- */

void ati_hd5770_cp_reset(ATIHD5770State *s)
{
    bool hold = s->eng_hold;

    qatomic_set(&s->eng_hold, true);
    cp_wait_idle(s);
    ati_hd5770_reg_lock(s);
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
    memset(s->alu_const, 0, HD5770_ALU_CONST_DWORDS * 4);
    ati_hd5770_dma_reset(s);
    ati_hd5770_reg_unlock(s);
    qatomic_set(&s->eng_hold, hold);
}

static void gpu_lock_cb(HD5770Gfx *g, bool on)
{
    ATIHD5770State *s = g->lock_opaque;

    if (on) {
        qemu_rec_mutex_lock(&s->gpu_mutex);
    } else {
        qemu_rec_mutex_unlock(&s->gpu_mutex);
    }
}

static void gpu_done_cb(void *opaque, uint64_t lo, uint64_t hi)
{
    ATIHD5770State *s = opaque;

    memory_region_set_dirty(&s->vram, lo, hi - lo);
}

static void gpu_done_ram_cb(void *opaque, uint64_t lo, uint64_t hi)
{
    ATIHD5770State *s = opaque;

    memory_region_set_dirty(s->gpu_ram->mr, lo, hi - lo);
}

/*
 * HD5770Gpu.written: the pages the refresh took from the dirty bitmap
 * (gpu_wlog) and the ones still in it. A refresh between the two reads
 * can hide a page from both: then every page counts as written.
 */
static uint64_t gpu_written_cb(void *opaque, uint64_t off)
{
    ATIHD5770State *s = opaque;
    ram_addr_t ram = memory_region_get_ram_addr(&s->vram);
    uint64_t p0 = off >> 12, n, bits = 0, i;
    uint32_t seq;

    if (off >= s->vram_size) {
        return 0;
    }
    n = MIN(64, (s->vram_size >> 12) - p0);
    smp_mb();
    seq = qatomic_read(&s->gpu_wseq);
    for (i = 0; i < n; i++) {
        if (qatomic_read(&s->gpu_wlog[p0 + i]) &&
            qatomic_xchg(&s->gpu_wlog[p0 + i], 0)) {
            bits |= 1ull << i;
        }
    }
    for (i = 0; i < n; i++) {
        if (physical_memory_get_dirty_flag(ram + off + (i << 12),
                                           DIRTY_MEMORY_VGA)) {
            bits |= 1ull << i;
        }
    }
    smp_mb();
    if ((seq & 1) || qatomic_read(&s->gpu_wseq) != seq) {
        bits = ~0ull;
    }
    return bits;
}

/* host page p of guest RAM: its RAM block given to the backend, once */
static bool gpu_ram_for(HD5770Gfx *g, uint8_t *p)
{
    ATIHD5770State *s = g->lock_opaque;
    ram_addr_t off;
    RAMBlock *rb;

    if (g->ram && p >= g->ram && p < g->ram + g->ram_size) {
        return true;
    }
    if (s->gpu_ram) {
        return false;
    }
    rb = qemu_ram_block_from_host(p, false, &off);
    if (!rb || !g->gpu->ops->ram ||
        !g->gpu->ops->ram(g->gpu->b, qemu_ram_get_host_addr(rb),
                          qemu_ram_get_used_length(rb))) {
        return false;
    }
    s->gpu_ram = rb;
    g->ram = qemu_ram_get_host_addr(rb);
    g->ram_size = qemu_ram_get_used_length(rb);
    return true;
}

/* the host GPU backend gpu=on|verify names, if the host has one */
static void gpu_open(ATIHD5770State *s)
{
    const char *mode = s->gpu_prop ? s->gpu_prop : "on";
    HD5770Gpu *gp = NULL;
    const char *err = "no host GPU backend on this host";

    if (!s->accel || !strcmp(mode, "off")) {
        return;
    }
#if defined(CONFIG_DARWIN)
    gp = !strcmp(mode, "native")
         ? hd5770_mtlr_open(memory_region_get_ram_ptr(&s->vram), s->vram_size,
                            &err)
         : hd5770_mtl_open(memory_region_get_ram_ptr(&s->vram), s->vram_size,
                           &err);
#elif defined(CONFIG_WIN32)
    gp = hd5770_gl_open(memory_region_get_ram_ptr(&s->vram), s->vram_size,
                        &err);
#endif
    if (!gp) {
        warn_report("ati-hd5770: gpu=%s: %s; drawing on the CPU", mode, err);
        return;
    }
    gp->done = gpu_done_cb;
    gp->done_ram = gpu_done_ram_cb;
    gp->opaque = s;
    if (!strcmp(mode, "native")) {
        s->gpu_wlog = g_malloc0((s->vram_size >> 12) + 1);
        gp->written = gpu_written_cb;
    }
    s->gfx3d.ram_for = gpu_ram_for;
    s->gfx3d.gpu = gp;
    s->gfx3d.vram = memory_region_get_ram_ptr(&s->vram);
    s->gfx3d.vram_size = s->vram_size;
    s->gfx3d.gpu_mode = !strcmp(mode, "verify") ? HD5770_GPU_VERIFY
                                                : HD5770_GPU_ON;
}

void ati_hd5770_cp_init(ATIHD5770State *s)
{
    s->gfx = g_new0(uint32_t, HD5770_GFX_DWORDS);
    s->gfx_dwords = HD5770_GFX_DWORDS;
    s->cp_pkt = g_new0(uint32_t, HD5770_CP_MAX_PKT);
    s->cp_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL, cp_timer_cb, s);
    s->alu_const = g_new0(uint32_t, HD5770_ALU_CONST_DWORDS);
    s->alu_const_dwords = HD5770_ALU_CONST_DWORDS;
    s->gfx_stats = g_new0(HD5770GfxStats, 1);
    s->gfx3d.ctx = s->gfx;
    s->gfx3d.cfg = s->regs;
    s->gfx3d.alu = s->alu_const;
    s->gfx3d.st = s->gfx_stats;
    hd5770_gfx_init(&s->gfx3d);
    ati_hd5770_mc_gfx_init(s);
    qemu_rec_mutex_init(&s->gpu_mutex);
    qemu_mutex_init(&s->reg_lock);
    s->gfx3d.lock = gpu_lock_cb;
    s->gfx3d.lock_opaque = s;
    gpu_open(s);
    qemu_mutex_init(&s->fx_lock);
    qemu_cond_init(&s->fx_cond);
    g_queue_init(&s->fx_q);
    if (s->accel && s->cp_thread) {
        s->eng_on = true;
        if (s->gfx3d.gpu && s->gfx3d.gpu->ops->fence) {
            /* FENCES above */
            s->fx_on = true;
            s->fx_bh = qemu_bh_new(fx_bh_cb, s);
            s->gfx3d.gpu->retired = gpu_retired_cb;
            s->gfx3d.gpu->thread = gpu_thread_cb;
        }
        s->eng_hold = !runstate_is_running();
        qemu_mutex_init(&s->eng_lock);
        qemu_cond_init(&s->eng_cond);
        qemu_cond_init(&s->eng_idle_cond);
        s->eng_vmse = qemu_add_vm_change_state_handler(cp_vm_state, s);
        qemu_thread_create(&s->eng_thread, "ati-hd5770-cp", cp_thread_fn, s,
                           QEMU_THREAD_JOINABLE);
    }
}

void ati_hd5770_cp_fini(ATIHD5770State *s)
{
    if (s->eng_on) {
        qatomic_set(&s->eng_hold, true);
        cp_wait_idle(s);
        qemu_mutex_lock(&s->eng_lock);
        s->eng_quit = true;
        qemu_cond_signal(&s->eng_cond);
        qemu_mutex_unlock(&s->eng_lock);
        qemu_thread_join(&s->eng_thread);
        qemu_del_vm_change_state_handler(s->eng_vmse);
        qemu_cond_destroy(&s->eng_cond);
        qemu_cond_destroy(&s->eng_idle_cond);
        qemu_mutex_destroy(&s->eng_lock);
        s->eng_on = false;
    }
    qemu_mutex_destroy(&s->reg_lock);
    if (s->gfx3d.gpu) {
        hd5770_gfx_sync(&s->gfx3d);
        s->gfx3d.gpu->ops->close(s->gfx3d.gpu->b);
        s->gfx3d.gpu = NULL;
        g_free(s->gpu_wlog);
        s->gpu_wlog = NULL;
    }
    if (s->fx_bh) {
        qemu_bh_delete(s->fx_bh);
        s->fx_bh = NULL;
    }
    s->fx_on = false;
    qemu_cond_destroy(&s->fx_cond);
    qemu_mutex_destroy(&s->fx_lock);
    timer_free(s->cp_timer);
    s->cp_timer = NULL;
    g_free(s->gfx);
    s->gfx = NULL;
    g_free(s->cp_pkt);
    s->cp_pkt = NULL;
    hd5770_gfx_fini(&s->gfx3d);
    g_free(s->gfx_stats);
    g_free(s->alu_const);
    g_free(s->maps);
}
