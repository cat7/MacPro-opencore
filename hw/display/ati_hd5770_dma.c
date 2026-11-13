/*
 * ATI Radeon HD 5770 (Juniper) asynchronous DMA engine (accel on).
 *
 * A DMA_RB_WPTR write runs the ring (byte pointers, DMA_RB_CNTL size in
 * log2 dwords) on the calling thread: linear writes, dword and byte
 * linear copies (also broadcast), constant fills, fences, traps (IH
 * source 224 with DMA_CNTL TRAP_ENABLE), semaphores, SRBM register
 * writes and indirect buffers. Tiled copies and writes are counted as
 * gaps and skipped. A semaphore wait on zero stalls the ring; the CP's
 * poll timer retries it.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/bswap.h"
#include "ati_hd5770_int.h"
#include "ati_hd5770_regs.h"
#include "trace.h"

enum {
    DMA_WRITE = 0x2, DMA_COPY = 0x3, DMA_IB = 0x4, DMA_SEMAPHORE = 0x5,
    DMA_FENCE = 0x6, DMA_TRAP = 0x7, DMA_SRBM_WRITE = 0x9,
    DMA_CONSTANT_FILL = 0xd, DMA_NOP = 0xf,
};

#define DMA_MAX_PKT     16      /* dwords of a packet header part */

static uint64_t addr40(uint32_t lo, uint32_t hi)
{
    return ((uint64_t)(hi & 0xff) << 32) | lo;
}

static void copy(ATIHD5770State *s, uint64_t dst, uint64_t src,
                 uint64_t bytes)
{
    uint8_t buf[4096];

    s->dma.copy_bytes += bytes;
    while (bytes) {
        uint32_t c = MIN(bytes, sizeof(buf));

        ati_hd5770_mc_read(s, src, buf, c);
        ati_hd5770_mc_write(s, dst, buf, c);
        src += c;
        dst += c;
        bytes -= c;
    }
}

/*
 * The dwords a packet occupies beyond its header part, and the header
 * part's length, from the header: 0 header length if unknown.
 */
static void pkt_shape(uint32_t h, uint32_t *head, uint32_t *tail)
{
    uint32_t cmd = h >> 28, sub = (h >> 20) & 0xff, count = h & 0xfffff;

    *tail = 0;
    switch (cmd) {
    case DMA_WRITE:
        *head = (sub & 8) ? 7 : 3;
        *tail = count;
        break;
    case DMA_COPY:
        switch (sub) {
        case 0x00: case 0x40: *head = 5; break;
        case 0x44: *head = 7; break;
        case 0x08: case 0x41: case 0x4c: *head = 9; break;
        case 0x48: case 0x4b: case 0x4f: *head = 10; break;
        case 0x49: *head = 12; break;
        case 0x4d: *head = 13; break;
        default: *head = 0; break;
        }
        break;
    case DMA_IB:
        *head = 3;
        break;
    case DMA_SEMAPHORE:
        *head = 3;
        break;
    case DMA_FENCE:
        *head = 4;
        break;
    case DMA_TRAP:
    case DMA_NOP:
        *head = 1;
        break;
    case DMA_SRBM_WRITE:
        *head = 3;
        break;
    case DMA_CONSTANT_FILL:
        *head = 4;
        break;
    default:
        *head = 0;
        break;
    }
}

enum { DMA_DONE, DMA_STALL, DMA_ENTER_IB };

/* one packet; b holds its header part, its tail is read from src */
static int dma_exec(ATIHD5770State *s, const uint32_t *b, uint64_t tail_mc,
                    uint64_t *ib_base, uint32_t *ib_size)
{
    uint32_t cmd = b[0] >> 28, sub = (b[0] >> 20) & 0xff;
    uint32_t count = b[0] & 0xfffff;

    s->dma.packets++;
    s->dma.cmd[cmd]++;
    trace_ati_hd5770_dma_packet(b[0]);
    switch (cmd) {
    case DMA_WRITE:
        if (sub & 8) {
            s->dma.sub_gap++;
        } else {
            copy(s, addr40(b[1] & ~3u, b[2]), tail_mc, (uint64_t)count * 4);
            s->dma.write_dw += count;
        }
        return DMA_DONE;
    case DMA_COPY:
        switch (sub) {
        case 0x00:
            copy(s, addr40(b[1] & ~3u, b[3]), addr40(b[2] & ~3u, b[4]),
                 (uint64_t)count * 4);
            break;
        case 0x40:
            copy(s, addr40(b[1], b[3]), addr40(b[2], b[4]), count);
            break;
        case 0x44:
            copy(s, addr40(b[1] & ~3u, b[4]), addr40(b[3] & ~3u, b[6]),
                 (uint64_t)count * 4);
            copy(s, addr40(b[2] & ~3u, b[5]), addr40(b[3] & ~3u, b[6]),
                 (uint64_t)count * 4);
            break;
        default:
            s->dma.sub_gap++;
            break;
        }
        return DMA_DONE;
    case DMA_CONSTANT_FILL: {
        uint64_t dst = ((uint64_t)(b[3] & 0x00ff0000) << 16) | (b[1] & ~3u);
        uint32_t i;

        for (i = 0; i < count; i++) {
            ati_hd5770_mc_wr32(s, dst + i * 4, b[2]);
        }
        s->dma.fill_bytes += (uint64_t)count * 4;
        return DMA_DONE;
    }
    case DMA_FENCE:
        ati_hd5770_mc_wr32(s, addr40(b[1] & ~3u, b[2]), b[3]);
        s->dma.fences++;
        return DMA_DONE;
    case DMA_TRAP:
        if (s->regs[EG_DMA_CNTL >> 2] & EG_DMA_TRAP_ENABLE) {
            ati_hd5770_ih_push(s, HD5770_IH_SRC_DMA_TRAP, 0);
            s->dma.traps++;
        } else {
            s->dma.traps_masked++;
        }
        return DMA_DONE;
    case DMA_SEMAPHORE: {
        uint64_t a = addr40(b[1] & ~7u, b[2]);
        uint32_t v = ati_hd5770_mc_rd32(s, a);

        if (b[0] & (1u << 22)) {
            ati_hd5770_mc_wr32(s, a, v + 1);
        } else if (!v) {
            return DMA_STALL;
        } else {
            ati_hd5770_mc_wr32(s, a, v - 1);
        }
        return DMA_DONE;
    }
    case DMA_SRBM_WRITE:
        ati_hd5770_reg_write(s, (b[1] & 0xffff) << 2, b[2]);
        return DMA_DONE;
    case DMA_IB:
        *ib_base = addr40(b[1] & ~0x1fu, b[2]);
        *ib_size = (b[2] >> 12) & 0xfffff;
        s->dma.ibs++;
        return DMA_ENTER_IB;
    default:
        return DMA_DONE;
    }
}

static uint32_t rd_ring(ATIHD5770State *s, uint64_t base, uint32_t mask,
                        uint32_t byte)
{
    return ati_hd5770_mc_rd32(s, base + (byte & mask));
}

/* an indirect buffer: packets only, no nesting, no stalls */
static void dma_ib(ATIHD5770State *s, uint64_t base, uint32_t size)
{
    uint32_t pos = 0, b[DMA_MAX_PKT];

    while (pos < size) {
        uint32_t head, tail, i;
        uint64_t ib = 0;
        uint32_t ibs = 0;

        b[0] = ati_hd5770_mc_rd32(s, base + pos * 4);
        pkt_shape(b[0], &head, &tail);
        if (!head || pos + head + tail > size) {
            s->dma.sub_gap++;
            return;
        }
        for (i = 1; i < head; i++) {
            b[i] = ati_hd5770_mc_rd32(s, base + (pos + i) * 4);
        }
        if (dma_exec(s, b, base + (pos + head) * 4, &ib, &ibs) != DMA_DONE) {
            s->dma.sub_gap++;
        }
        pos += head + tail;
    }
}

void ati_hd5770_dma_run(ATIHD5770State *s)
{
    uint32_t cntl = s->regs[EG_DMA_RB_CNTL >> 2];
    uint32_t bytes, mask, wptr, rptr0;
    uint64_t base;

    if (s->dma_running || !(cntl & EG_DMA_RB_ENABLE)) {
        return;
    }
    s->dma_running = true;
    s->dma_stalled = false;
    bytes = 4u << MIN((cntl >> 1) & 0x1f, 22u);
    mask = bytes - 1;
    base = (uint64_t)s->regs[EG_DMA_RB_BASE >> 2] << 8;
    wptr = s->regs[EG_DMA_RB_WPTR >> 2] & mask & ~3u;
    rptr0 = s->dma_rptr;
    s->dma_rptr &= mask & ~3u;

    while (s->dma_rptr != wptr) {
        uint32_t avail = (wptr - s->dma_rptr) & mask;
        uint32_t b[DMA_MAX_PKT], head, tail, i, ibs = 0;
        uint64_t ib = 0;
        int r;

        b[0] = rd_ring(s, base, mask, s->dma_rptr);
        pkt_shape(b[0], &head, &tail);
        if (!head) {
            /* unknown packet: its length is unknown, give up the ring */
            s->dma.sub_gap++;
            s->dma_rptr = wptr;
            break;
        }
        if ((head + tail) * 4 > avail) {
            break;
        }
        for (i = 1; i < head; i++) {
            b[i] = rd_ring(s, base, mask, s->dma_rptr + i * 4);
        }
        if (tail && ((s->dma_rptr + head * 4) & mask) + tail * 4 > bytes) {
            /* a write payload that wraps: copy it in two parts */
            uint32_t first = bytes - ((s->dma_rptr + head * 4) & mask);
            uint64_t dst = addr40(b[1] & ~3u, b[2]);

            s->dma.packets++;
            s->dma.cmd[b[0] >> 28]++;
            copy(s, dst, base + ((s->dma_rptr + head * 4) & mask), first);
            copy(s, dst + first, base, tail * 4 - first);
            s->dma.write_dw += tail;
            r = DMA_DONE;
        } else {
            r = dma_exec(s, b, base + ((s->dma_rptr + head * 4) & mask),
                         &ib, &ibs);
        }
        if (r == DMA_STALL) {
            s->dma_stalled = true;
            s->dma.stalls++;
            break;
        }
        if (r == DMA_ENTER_IB) {
            dma_ib(s, ib, ibs);
        }
        s->dma_rptr = (s->dma_rptr + (head + tail) * 4) & mask;
    }
    if (s->dma_rptr != rptr0 && (cntl & EG_DMA_RPTR_WRITEBACK_ENABLE)) {
        ati_hd5770_mc_wr32(s, addr40(s->regs[EG_DMA_RB_RPTR_ADDR_LO >> 2] & ~3u,
                                     s->regs[EG_DMA_RB_RPTR_ADDR_HI >> 2]),
                           s->dma_rptr);
    }
    s->dma_running = false;
    if (s->dma_stalled) {
        timer_mod(s->cp_timer, qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) +
                  50 * SCALE_US);
    }
}

bool ati_hd5770_dma_read(ATIHD5770State *s, uint32_t off, uint32_t *val)
{
    switch (off) {
    case EG_DMA_RB_RPTR:
        *val = s->dma_rptr;
        return true;
    case EG_DMA_STATUS:
        *val = s->dma_stalled ? 0 : EG_DMA_IDLE;
        return true;
    default:
        return false;
    }
}

bool ati_hd5770_dma_write(ATIHD5770State *s, uint32_t off, uint32_t val)
{
    switch (off) {
    case EG_DMA_RB_WPTR:
        s->regs[off >> 2] = val;
        s->dma.kicks++;
        ati_hd5770_dma_run(s);
        return true;
    case EG_DMA_RB_RPTR:
        s->regs[off >> 2] = val;
        s->dma_rptr = val & 0x3fffc;
        return true;
    case EG_SRBM_SOFT_RESET:
        s->regs[off >> 2] = val;
        if (val & EG_SRBM_SOFT_RESET_DMA) {
            s->dma_stalled = false;
        }
        return true;
    default:
        return false;
    }
}

void ati_hd5770_dma_stats(ATIHD5770State *s, GString *out)
{
    HD5770DMAStats *d = &s->dma;
    unsigned i;

    g_string_append_printf(out,
        "DMA: cntl 0x%08x base 0x%08x rptr 0x%x wptr 0x%x stalled %d "
        "kicks %" PRIu64 " packets %" PRIu64 " ibs %" PRIu64 "\n"
        "fences %" PRIu64 " traps %" PRIu64 " masked %" PRIu64
        " write dwords %" PRIu64 " copied %" PRIu64 " filled %" PRIu64
        " not modelled %" PRIu64 " stalls %" PRIu64 "\ncommands:",
        s->regs[EG_DMA_RB_CNTL >> 2], s->regs[EG_DMA_RB_BASE >> 2],
        s->dma_rptr, s->regs[EG_DMA_RB_WPTR >> 2], s->dma_stalled,
        d->kicks, d->packets, d->ibs, d->fences, d->traps, d->traps_masked,
        d->write_dw, d->copy_bytes, d->fill_bytes, d->sub_gap, d->stalls);
    for (i = 0; i < 16; i++) {
        if (d->cmd[i]) {
            g_string_append_printf(out, " %x:%" PRIu64, i, d->cmd[i]);
        }
    }
    g_string_append_c(out, '\n');
}

void ati_hd5770_dma_reset(ATIHD5770State *s)
{
    s->dma_rptr = 0;
    s->dma_stalled = false;
    s->dma_running = false;
}
