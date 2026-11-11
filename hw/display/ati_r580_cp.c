/*
 * QEMU ATI Radeon X1900 XT (R580) emulation: PM4 command processor,
 * engine thread and memory-controller routing -- see ati_r580_int.h.
 *
 * This work is licensed under the GNU GPL license version 2 or later.
 */

#include "qemu/osdep.h"
#include "qemu/bswap.h"
#include "qemu/error-report.h"
#include "qemu/log.h"
#include "qemu/module.h"
#include "qemu/units.h"
#include "hw/pci/pci_device.h"
#include "hw/pci/pci_bus.h"
#include "hw/core/qdev-properties.h"
#include "migration/vmstate.h"
#include "qapi/error.h"
#include "system/memory.h"
#include "ui/console.h"
#include "qom/object.h"
#include "hw/i2c/i2c.h"

#include "exec/target_page.h"
#include "system/physmem.h"
#include "system/qtest.h"
#include "system/runstate.h"
#include "qemu/main-loop.h"
#include "qemu/rcu.h"
#include "qemu/lockable.h"
#include "ati_r580_int.h"
#include "ati_r580_regs.h"
#include "ati_r580_gl.h"
#include "trace.h"

/*
 * Read one dword from the PM4 ring at the current read pointer and
 * advance it, wrapping at the ring size. The ring lives in VRAM (see
 * the comment on R580_PM4_BUFFER_OFFSET) so this is a direct pointer
 * read, not a DMA -- consistent with every other VRAM access in this
 * file.
 */
/*
 * Read one little-endian dword from card address space: local VRAM
 * below ATI_R580_VRAM_SIZE, otherwise the 32MB GART-translated
 * "VM" window (see the R580_PCIGART_TABLE_ENTRIES comment in
 * ati_r580_regs.h). Command streams are little-endian regardless
 * of guest CPU endianness -- big-endian Mac drivers byte-swap their
 * command buffers on the way out, exactly like the Linux driver's
 * cpu_to_le32() (verified live: PIO-FIFO dwords, which arrive
 * pre-swapped through the LE register aperture, decode with the
 * identical packet layout).
 *
 * The GART window's card-space base isn't modeled explicitly: the
 * window is exactly 32MB and its base is 32MB-aligned, so masking the
 * page index into the 8192-entry table is base-agnostic.
 */
/*
 * Card ("MC") address space. Everything the engine and the command
 * processor fetch or draw to is a 32-bit card address that the memory
 * controller routes by window: MC_FB_LOCATION [31:16] = top, [15:0] =
 * base of the frame buffer window (in 64KB units), MC_AGP_LOCATION
 * likewise for the AGP window (translated by AGP_BASE + the host
 * bridge's GART), and anything outside both goes straight to the PCI
 * bus as a host address. The Mac ROMs put the frame buffer window at
 * the card's PCI aperture address, so a frame-buffer card address is
 * numerically the same as its PCI bus address -- either way the window
 * test below resolves it locally first.
 */
bool ati_r580_mc_to_vram(ATIR580State *s, uint32_t addr,
                                uint32_t *off)
{
    uint32_t loc = s->regs[R580_MC_FB_LOCATION >> 2];
    uint32_t lo = (loc & 0xffff) << 16;
    uint32_t hi = (loc & 0xffff0000) | 0xffff;

    if (addr >= lo && addr <= hi && addr - lo < ATI_R580_VRAM_SIZE) {
        *off = addr - lo;
        return true;
    }
    return false;
}

/*
 * The card's PCI GART: AIC_LO_ADDR..AIC_HI_ADDR maps page by page
 * through a table in host memory whose bus address the Mac driver
 * programs at 0xab0.
 */
/*
 * The R5xx PCIe GART: PCIE_TX_GART_START_LO..END_LO maps page by page
 * through a table of 32-bit entries at card address PCIE_TX_GART_BASE.
 * An entry holds bus address bits 39:8 (low nibble = flags).
 */
static bool ati_r580_mc_to_pcie_gart(ATIR580State *s, uint32_t addr,
                                     dma_addr_t *bus)
{
    uint32_t start = s->pcie_ind[R5XX_PCIE_TX_GART_START_LO] & ~0xfffu;
    uint32_t end = s->pcie_ind[R5XX_PCIE_TX_GART_END_LO];
    uint32_t base = s->pcie_ind[R5XX_PCIE_TX_GART_BASE];
    uint32_t off, entry;

    if (!(s->pcie_ind[R5XX_PCIE_TX_GART_CNTL] & R5XX_PCIE_TX_GART_EN) ||
        addr < start || addr > end) {
        return false;
    }
    base += ((addr - start) >> 12) * 4;
    if (ati_r580_mc_to_vram(s, base, &off)) {
        if (off + 4 > ATI_R580_VRAM_SIZE) {
            return false;
        }
        entry = ldl_le_p((uint8_t *)memory_region_get_ram_ptr(&s->vram) +
                         off);
    } else if (pci_dma_read(PCI_DEVICE(s), base, &entry,
                            sizeof(entry)) != MEMTX_OK) {
        return false;
    } else {
        entry = le32_to_cpu(entry);
    }
    *bus = (((dma_addr_t)entry & ~0xfull) << 8) | (addr & 0xfff);
    return true;
}

static bool ati_r580_mc_to_aic(ATIR580State *s, uint32_t addr,
                               dma_addr_t *bus)
{
    uint32_t lo = s->regs[R580_AIC_LO_ADDR >> 2] & ~0xfffu;
    uint32_t hi = s->regs[R580_AIC_HI_ADDR >> 2];
    uint32_t entry;

    if (ati_r580_mc_to_pcie_gart(s, addr, bus)) {
        return true;
    }
    if (!(s->regs[R580_AIC_CNTL >> 2] & R580_PCIGART_TRANSLATE_EN) ||
        addr < lo || addr > hi) {
        return false;
    }
    if (pci_dma_read(PCI_DEVICE(s),
                     (s->regs[R580_AIC_PT_BASE_R300 >> 2] & ~0xfffu) +
                     ((addr - lo) >> 12) * 4,
                     &entry, sizeof(entry)) != MEMTX_OK) {
        return false;
    }
    *bus = (le32_to_cpu(entry) & ~0xfffu) | (addr & 0xfff);
    return true;
}

static bool ati_r580_mc_to_agp(ATIR580State *s, uint32_t addr,
                               dma_addr_t *bus)
{
    uint32_t loc = s->regs[R580_MC_AGP_LOCATION >> 2];
    uint32_t lo = (loc & 0xffff) << 16;
    uint32_t hi = (loc & 0xffff0000) | 0xffff;

    if (lo <= hi && addr >= lo && addr <= hi) {
        *bus = (s->regs[R580_AGP_BASE >> 2] & ~0xfffu) + (addr - lo);
        return true;
    }
    return false;
}

const char *ati_r580_mc_describe(ATIR580State *s, uint32_t addr,
                                 uint64_t *target)
{
    uint32_t off;
    dma_addr_t bus;

    if (ati_r580_mc_to_vram(s, addr, &off)) {
        *target = off;
        return "vram";
    }
    if (ati_r580_mc_to_aic(s, addr, &bus)) {
        *target = bus;
        return "gart";
    }
    if (ati_r580_mc_to_agp(s, addr, &bus)) {
        *target = bus;
        return "agp";
    }
    *target = addr;
    return "bus";
}

/*
 * Read one little-endian dword from card address space. Command streams
 * are little-endian regardless of guest CPU endianness (big-endian Mac
 * drivers byte-swap on the way out, or stage them through a swapped
 * surface -- which ati_r580_vram_ld32() honours).
 */
uint32_t ati_r580_mc_read32(ATIR580State *s, uint32_t addr)
{
    uint32_t off, val = 0;
    dma_addr_t bus;

    if (ati_r580_mc_to_vram(s, addr, &off)) {
        ati_r580_gl_touch(s, off, 4);
        return ati_r580_vram_ld32(s, off);
    }
    if (ati_r580_mc_to_aic(s, addr, &bus)) {
        /* bus address */
    } else if (!ati_r580_mc_to_agp(s, addr, &bus)) {
        bus = addr;
    } else if (s->agp_as_valid) {
        address_space_read(&s->agp_as, bus, MEMTXATTRS_UNSPECIFIED,
                           &val, sizeof(val));
        return le32_to_cpu(val);
    }
    pci_dma_read(PCI_DEVICE(s), bus, &val, sizeof(val));
    return le32_to_cpu(val);
}

/* the registers ati_r580_mc_route() depends on */
static const uint16_t ati_r580_mc_map_regs[] = {
    R580_MC_FB_LOCATION, R580_MC_AGP_LOCATION, R580_AGP_BASE,
    R580_AIC_CNTL, R580_AIC_LO_ADDR, R580_AIC_HI_ADDR, R580_AIC_PT_BASE_R300,
};

/* and the PCIe GART registers it depends on */
static const uint8_t ati_r580_mc_map_pcie[] = {
    R5XX_PCIE_TX_GART_CNTL, R5XX_PCIE_TX_GART_BASE,
    R5XX_PCIE_TX_GART_START_LO, R5XX_PCIE_TX_GART_END_LO,
};

#define R580_MC_MAP_FRONT ARRAY_SIZE(ati_r580_mc_map_regs)
QEMU_BUILD_BUG_ON(ARRAY_SIZE(ati_r580_mc_map_regs) +
                  ARRAY_SIZE(ati_r580_mc_map_pcie) != R580_MC_MAP_REGS);

static uint32_t ati_r580_mc_map_val(ATIR580State *s, unsigned k)
{
    return k < R580_MC_MAP_FRONT ? s->regs[ati_r580_mc_map_regs[k] >> 2]
           : s->pcie_ind[ati_r580_mc_map_pcie[k - R580_MC_MAP_FRONT]];
}

void ati_r580_mc_map_save(ATIR580State *s, uint32_t *map)
{
    unsigned k;

    for (k = 0; k < R580_MC_MAP_REGS; k++) {
        map[k] = ati_r580_mc_map_val(s, k);
    }
}

bool ati_r580_mc_map_same(ATIR580State *s, const uint32_t *map)
{
    unsigned k;

    for (k = 0; k < R580_MC_MAP_REGS; k++) {
        if (map[k] != ati_r580_mc_map_val(s, k)) {
            return false;
        }
    }
    return true;
}

/* where a card address goes: VRAM, a PCI bus address, or the AGP space */
enum { R580_MC_VRAM, R580_MC_PCI, R580_MC_AGP };

static int ati_r580_mc_route(ATIR580State *s, uint32_t addr, dma_addr_t *bus)
{
    uint32_t off;

    if (ati_r580_mc_to_vram(s, addr, &off)) {
        *bus = off;
        return R580_MC_VRAM;
    }
    if (ati_r580_mc_to_aic(s, addr, bus)) {
        return R580_MC_PCI;
    }
    if (!ati_r580_mc_to_agp(s, addr, bus)) {
        *bus = addr;
        return R580_MC_PCI;
    }
    return s->agp_as_valid ? R580_MC_AGP : R580_MC_PCI;
}

/*
 * `n` consecutive dwords of card address space, as ati_r580_mc_read32()
 * reads them one at a time. A run inside one 4 KiB page whose two ends
 * take the same route to contiguous bus addresses is read with one
 * translation; VRAM, whose reads have a coherency hook, and anything
 * else go dword by dword.
 */
void ati_r580_mc_read_block(ATIR580State *s, uint32_t addr, uint32_t *dst,
                            unsigned n)
{
    while (n) {
        unsigned run = (addr & 3) ? 1
                       : MIN(n, (0x1000 - (addr & 0xfff)) / 4);
        dma_addr_t b0 = 0, b1 = 0;
        int k0 = ati_r580_mc_route(s, addr, &b0);
        int k1 = ati_r580_mc_route(s, addr + (run - 1) * 4, &b1);
        unsigned i;

        if (k0 == R580_MC_VRAM || k0 != k1 ||
            b1 - b0 != (dma_addr_t)(run - 1) * 4) {
            for (i = 0; i < run; i++) {
                dst[i] = ati_r580_mc_read32(s, addr + i * 4);
            }
        } else {
            memset(dst, 0, run * 4);
            if (k0 == R580_MC_AGP) {
                address_space_read(&s->agp_as, b0, MEMTXATTRS_UNSPECIFIED,
                                   dst, run * 4);
            } else {
                pci_dma_read(PCI_DEVICE(s), b0, dst, run * 4);
            }
            for (i = 0; i < run; i++) {
                dst[i] = le32_to_cpu(dst[i]);
            }
        }
        dst += run;
        addr += run * 4;
        n -= run;
    }
}

void ati_r580_mc_write32(ATIR580State *s, uint32_t addr, uint32_t val)
{
    uint32_t off;
    dma_addr_t bus;

    if (ati_r580_mc_to_vram(s, addr, &off)) {
        uint8_t *vram = memory_region_get_ram_ptr(&s->vram);

        if (off + 4 <= ATI_R580_VRAM_SIZE) {
            ati_r580_gl_dirty(s, off, 4);
            stl_le_p(vram + off, val);
            memory_region_set_dirty(&s->vram, off, 4);
        }
        return;
    }
    val = cpu_to_le32(val);
    if (ati_r580_mc_to_aic(s, addr, &bus)) {
        /* bus address */
    } else if (!ati_r580_mc_to_agp(s, addr, &bus)) {
        bus = addr;
    } else if (s->agp_as_valid) {
        address_space_write(&s->agp_as, bus, MEMTXATTRS_UNSPECIFIED,
                            &val, sizeof(val));
        return;
    }
    pci_dma_write(PCI_DEVICE(s), bus, &val, sizeof(val));
}

/*
 * The CP's memory write-backs: the ring read pointer (CP_RB_RPTR_ADDR,
 * unless RB_NO_UPDATE) and any scratch register whose SCRATCH_UMSK bit
 * is set (to SCRATCH_ADDR + 4*n). Drivers poll these in memory rather
 * than reading the registers, so a missing write-back is a silent
 * driver hang.
 */
/*
 * RB_RPTR_SWAP (CP_RB_RPTR_ADDR bits 1:0) swaps the write-back like
 * BUF_SWAP: 1 = bytes in each half, 2 = whole dword, 3 = halves. The
 * Panther driver asks for 2 and reads the pointer with a plain load.
 */
static void ati_r580_cp_rptr_writeback(ATIR580State *s)
{
    uint32_t cntl = s->regs[R580_CP_RB_CNTL >> 2];
    uint32_t reg = s->regs[R580_CP_RB_RPTR_ADDR >> 2];
    uint32_t addr = reg & ~R580_RB_RPTR_SWAP_MASK;
    uint32_t val = s->pm4_rptr;

    if (!(cntl & R580_RB_NO_UPDATE) && addr) {
        switch (reg & R580_RB_RPTR_SWAP_MASK) {
        case 1:
            val = ((val & 0x00ff00ffu) << 8) | ((val >> 8) & 0x00ff00ffu);
            break;
        case 2:
            val = bswap32(val);
            break;
        case 3:
            val = (val << 16) | (val >> 16);
            break;
        }
        ati_r580_mc_write32(s, addr, val);
    }
}

/*
 * With CP_RB_CNTL.RB_RPTR_WR_ENA set, a CP_RB_WPTR write first moves
 * CP_RB_RPTR_WR into the read pointer: the ring reset sequence.
 */
void ati_r580_cp_rptr_load(ATIR580State *s)
{
    uint32_t old = s->pm4_rptr;
    uint32_t val = s->regs[R580_CP_RB_RPTR_WR >> 2];

    if (!(s->regs[R580_CP_RB_CNTL >> 2] & R580_RB_RPTR_WR_ENA)) {
        return;
    }
    qatomic_set(&s->pm4_rptr, s->pm4_ring_dwords ?
                val & (s->pm4_ring_dwords - 1) : val);
    /* a packet split across the old position is not continued */
    s->pm4_ring.remaining = 0;
    trace_ati_r580_cp_rptr_load(old, s->pm4_rptr);
    ati_r580_cp_rptr_writeback(s);
}

void ati_r580_scratch_writeback(ATIR580State *s, unsigned n)
{
    uint32_t umsk = s->regs[R580_SCRATCH_UMSK >> 2];
    uint32_t addr = s->regs[R580_SCRATCH_ADDR >> 2] & ~3u;

    if ((umsk & (1u << n)) && addr) {
        ati_r580_mc_write32(s, addr + n * 4,
                            s->regs[(R580_SCRATCH_REG_BASE >> 2) + n]);
    }
}

static uint32_t ati_r580_pm4_read_ring(ATIR580State *s)
{
    uint32_t base = s->regs[R580_CP_RB_BASE >> 2] & ~3u;
    uint32_t val;

    val = ati_r580_mc_read32(s, base + s->pm4_rptr * 4);
    /*
     * BUF_SWAP_32BIT: the driver writes the ring in its own (big-endian)
     * byte order and the CP swaps each dword on fetch.
     */
    if ((s->regs[R580_CP_RB_CNTL >> 2] & R580_BUF_SWAP_MASK) ==
        R580_BUF_SWAP_32BIT) {
        val = bswap32(val);
    }
    qatomic_set(&s->pm4_rptr, (s->pm4_rptr + 1) & (s->pm4_ring_dwords - 1));
    return val;
}

/*
 * Consume PM4 ring entries from the read pointer up to the write
 * pointer, synchronously (matching the BM engine's own synchronous
 * model). Packet format: see the comment on R580_PM4_BUFFER_OFFSET in
 * ati_r580_regs.h.
 */
void ati_r580_pm4_run(ATIR580State *s)
{
    /* a CP_RB_WPTR write from the command stream extends the running loop */
    if (!s->pm4_ring_dwords || s->pm4_in_ring) {
        return;
    }
    s->pm4_in_ring = true;
    ati_r580_pm4_run_ring(s);
    s->pm4_in_ring = false;
    /*
     * A whole ring is drained inside the guest store that kicked it, so
     * this is the first moment the guest CPU could look at VRAM again --
     * and its loads through the frame-buffer BAR cannot be trapped. See
     * "GL-OWNED RENDER TARGET" in ati_r580_3d.c. The target goes back
     * before the read pointer does: the other vCPU, or the guest behind
     * the command processor thread, may act on the write-back at once.
     */
    ati_r580_gl_release(s, R580_GLR_RING);
    ati_r580_zpass_settle(s);
    ati_r580_cp_rptr_writeback(s);
}

/*
 * Every ring dword goes through the same packet state machine as the
 * PIO FIFO and the indirect buffers, so a packet behaves identically
 * no matter how it was submitted. The parser state persists in
 * s->pm4_ring because the driver can bump the write pointer
 * mid-packet: the remainder arrives with a later WPTR write. (The
 * previous inline ring parser instead read payload dwords straight
 * past the write pointer, consuming ring slots the driver had not
 * written yet.)
 */
void ati_r580_pm4_run_ring(ATIR580State *s)
{
    int guard;

    s->engine_rptr_wb = 0;
    for (guard = 0; guard < 1000000 &&
                    s->pm4_rptr != qatomic_load_acquire(&s->pm4_wptr);
         guard++) {
        uint32_t pos = s->pm4_rptr;
        uint32_t val = ati_r580_pm4_read_ring(s);

        trace_ati_r580_pm4_ring_dword(pos, val);
        ati_r580_pm4_parse(s, &s->pm4_ring, val);
        /*
         * On the thread the guest refills the ring while it runs: tell
         * it what has been consumed, between packets and with nothing
         * left on the GPU.
         */
        if (ati_r580_on_engine() && ++s->engine_rptr_wb >= 256 &&
            !s->pm4_ring.remaining && !s->gl_res) {
            s->engine_rptr_wb = 0;
            ati_r580_cp_rptr_writeback(s);
        }
    }
}

/*
 * PIO alternative to the ring above: the real driver actually pushes
 * its command stream straight through PM4_FIFO_DATA_EVEN/ODD (see the
 * comment on those in ati_r580_regs.h), one dword per write. Same
 * packet format as the ring, just delivered by MMIO write instead of
 * being pulled from VRAM, so the state (in-flight packet type/count/
 * running register) has to live across calls instead of a loop index.
 */
void ati_r580_pm4_parse(ATIR580State *s,
                        ATIR580PM4Parser *p, uint32_t val)
{
    if (p->remaining == 0) {
        /* val is a new packet header */
        p->type = R580_PM4_PACKET_TYPE(val);

        switch (p->type) {
        case 0:
            p->reg = R580_PM4_PACKET0_REG(val);
            p->one_reg = R580_PM4_PACKET0_ONE_REG(val);
            p->remaining = R580_PM4_PACKET_COUNT(val);
            break;
        case 1:
            p->p1_reg1 = R580_PM4_PACKET1_REG1(val);
            p->p1_reg2 = R580_PM4_PACKET1_REG2(val);
            p->remaining = 2;
            break;
        case 2:
            /* NOP/padding, no data words */
            break;
        case 3:
            p->remaining = R580_PM4_PACKET_COUNT(val);
            p->p3_opcode = R580_PM4_PACKET3_OPCODE(val);
            p->p3_param_idx = 0;
            p->p3_total = p->remaining;
            trace_ati_r580_pm4_p3_hdr(p->p3_opcode, p->remaining);
            if (p->p3_opcode != R580_PM4_OPCODE_PAINT &&
                p->p3_opcode != R580_PM4_OPCODE_PAINT_MULTI &&
                p->p3_opcode != R580_PM4_OPCODE_BITBLT &&
                p->p3_opcode != R580_PM4_OPCODE_BITBLT_RECT &&
                p->p3_opcode != R580_PM4_OPCODE_BITBLT_MULTI &&
                p->p3_opcode != R580_PM4_OPCODE_HOSTDATA_BLT &&
                p->p3_opcode != R580_PM4_OPCODE_SCALING &&
                p->p3_opcode != R300_PM4_OPCODE_NOP3 &&
                p->p3_opcode != R300_PM4_OPCODE_CLEAR_ZMASK &&
                p->p3_opcode != R300_PM4_OPCODE_CLEAR_HIZ &&
                p->p3_opcode != R300_PM4_OPCODE_CLEAR_CMASK &&
                p->p3_opcode != R300_PM4_OPCODE_LOAD_VBPNTR &&
                p->p3_opcode != R300_PM4_OPCODE_DRAW_IMMD_2 &&
                p->p3_opcode != R300_PM4_OPCODE_DRAW_INDX_2 &&
                p->p3_opcode != R300_PM4_OPCODE_INDX_BUFFER &&
                p->p3_opcode != R300_PM4_OPCODE_DRAW_VBUF_2) {
                trace_ati_r580_pm4_unimp(p->p3_opcode, p->remaining);
                ati_r580_note_gap(s, R580_GAP_P3_OPCODE, p->p3_opcode);
            }
            break;
        }
        return;
    }

    /* val is a data dword for the packet currently in flight */
    switch (p->type) {
    case 0:
        if (trace_event_get_state_backends(TRACE_ATI_R580_PM4_REG)) {
            trace_ati_r580_pm4_reg(p->reg & 0xfffc,
                                   ati_r580_reg_name(p->reg & 0xfffc), val);
        }
        ati_r580_reg_write32(s, p->reg & 0xfffc, val);
        if (!p->one_reg) {
            p->reg += 4;
        }
        break;
    case 1:
        if (p->remaining == 2) {
            if (trace_event_get_state_backends(TRACE_ATI_R580_PM4_REG)) {
                trace_ati_r580_pm4_reg(p->p1_reg1 & 0xfffc,
                    ati_r580_reg_name(p->p1_reg1 & 0xfffc), val);
            }
            ati_r580_reg_write32(s, p->p1_reg1 & 0xfffc, val);
        } else {
            if (trace_event_get_state_backends(TRACE_ATI_R580_PM4_REG)) {
                trace_ati_r580_pm4_reg(p->p1_reg2 & 0xfffc,
                    ati_r580_reg_name(p->p1_reg2 & 0xfffc), val);
            }
            ati_r580_reg_write32(s, p->p1_reg2 & 0xfffc, val);
        }
        break;
    case 3:
        trace_ati_r580_p3_payload(p->p3_opcode, p->p3_param_idx, val);
        switch (p->p3_opcode) {
        case R580_PM4_OPCODE_PAINT:
        {
            /*
             * Two forms, told apart by GMC_LD_BRUSH_Y_X (bit 31 of the
             * context dword, which is always the first payload dword):
             *
             *   clear, 4 dwords: context, colour, top-left corner,
             *                    bottom-right corner
             *   set,   7 dwords: context, colour, BRUSH_DATA0,
             *                    BRUSH_DATA1, BRUSH_Y_X, top-left,
             *                    bottom-right
             *
             * "Load brush Y/X" means the packet carries the pattern
             * inline instead of the driver pre-loading the BRUSH_DATA
             * registers (which is how Linux's DRM r128 does it, via a
             * type-0 write to BRUSH_DATA0). Mac OS uses the inline form
             * for every marquee segment: 4668 consecutive packets in one
             * live capture, all
             *   f2550610 00ffffff aa55aa55 aa55aa55 00000000 <tl> <br>
             * i.e. DSTINVERT through a 50% checkerboard -- marching
             * ants. Reading only four dwords took BRUSH_DATA0/1 for the
             * two corners, which decode to a degenerate rectangle, so
             * every segment was silently dropped and no selection
             * rectangle was ever drawn.
             */
            unsigned want = (p->p3_params[0] & R580_GMC_LD_BRUSH_Y_X) &&
                            p->p3_total >= 7 ? 7 : 4;

            if (p->p3_param_idx < 7) {
                p->p3_params[p->p3_param_idx++] = val;
            }
            if (p->p3_total >= 4) {
                if (p->p3_param_idx == want) {
                    uint32_t tl = p->p3_params[want - 2];
                    uint32_t br = p->p3_params[want - 1];
                    int x1 = tl & 0x3fff, y1 = (tl >> 16) & 0x3fff;
                    int x2 = br & 0x3fff, y2 = (br >> 16) & 0x3fff;

                    ati_r580_reg_write32(s, R580_DP_GUI_MASTER_CNTL,
                                            p->p3_params[0]);
                    ati_r580_reg_write32(s, R580_DP_BRUSH_FRGD_CLR,
                                            p->p3_params[1]);
                    if (want == 7) {
                        ati_r580_reg_write32(s, R580_BRUSH_DATA0,
                                                p->p3_params[2]);
                        ati_r580_reg_write32(s, R580_BRUSH_DATA0 + 4,
                                                p->p3_params[3]);
                        ati_r580_reg_write32(s, R580_BRUSH_Y_X,
                                                p->p3_params[4]);
                    }
                    if (x2 > x1 && y2 > y1) {
                        s->dst_x = x1;
                        s->dst_y = y1;
                        s->dst_width = x2 - x1;
                        s->dst_height = y2 - y1;
                        trace_ati_r580_paint_multi(0, tl, br, s->dst_x,
                                                      s->dst_y, s->dst_width,
                                                      s->dst_height);
                        ati_r580_2d_blt(s);
                    }
                }
            } else if (p->p3_param_idx == 2) {
                s->dst_x = p->p3_params[0] & 0x3fff;
                s->dst_y = (p->p3_params[0] >> 16) & 0x3fff;
                s->dst_width = p->p3_params[1] & 0x3fff;
                s->dst_height = (p->p3_params[1] >> 16) & 0x3fff;
                ati_r580_2d_blt(s);
            }
            break;
        }
        case R580_PM4_OPCODE_PAINT_MULTI:
            /*
             * [0] context, [1] colour, then (DST_X_Y, DST_WIDTH_HEIGHT)
             * pairs -- and unlike PAINT's corners, X and WIDTH sit in
             * the HIGH half with Y and HEIGHT in the low one, matching
             * the registers of the same names. Established from a live
             * capture of Mac OS drawing a dialog: successive packets
             * walk y = 0,1,2,3 with widths 5,3,2,1 at x = 0 and
             * x = 1019..1023 on a 1024-wide screen -- a window's
             * rounded corners, pixel row by pixel row. Reading the
             * halves the other way round made every one of them
             * zero-sized, which is why button frames never appeared.
             */
            if (p->p3_param_idx == 0) {
                /*
                 * A context with DST_PITCH_OFFSET_CNTL set is followed
                 * by the destination pitch/offset dword, then the
                 * colour (Linux's r128 DRM clear packet); Mac OS 9's
                 * context (0x72f036d0) has the bit clear and goes
                 * straight to the colour.
                 */
                ati_r580_reg_write32(s, R580_DP_GUI_MASTER_CNTL, val);
                p->p3_params[3] = (val & R580_GMC_DST_PITCH_OFFSET_CNTL) ?
                                  1 : 0;
                p->p3_param_idx = 1;
            } else if (p->p3_param_idx == 1 && p->p3_params[3]) {
                ati_r580_reg_write32(s, R580_DST_PITCH_OFFSET, val);
                p->p3_params[3] = 0;
            } else if (p->p3_param_idx == 1) {
                ati_r580_reg_write32(s, R580_DP_BRUSH_FRGD_CLR, val);
                p->p3_param_idx = 2;
            } else if (p->p3_param_idx == 2) {
                p->p3_params[2] = val;
                p->p3_param_idx = 3;
            } else {
                uint32_t dst_x_y = p->p3_params[2];

                s->dst_y = dst_x_y & 0x3fff;
                s->dst_x = (dst_x_y >> 16) & 0x3fff;
                s->dst_height = val & 0x3fff;
                s->dst_width = (val >> 16) & 0x3fff;
                trace_ati_r580_paint_multi(0, dst_x_y, val, s->dst_x,
                                              s->dst_y, s->dst_width,
                                              s->dst_height);
                if (s->dst_width && s->dst_height) {
                    ati_r580_2d_blt(s);
                }
                p->p3_param_idx = 2;   /* next rectangle pair */
            }
            break;
        case R580_PM4_OPCODE_BITBLT_MULTI:
            /*
             * The save-under half of a window drag: a context dword,
             * the pitch/offset dwords the context itself announces
             * (its SRC/DST_PITCH_OFFSET_CNTL bits, in SRC-then-DST
             * order, as Linux's r128 DRM builds its swap packet), and
             * then a RUN of three-dword rectangles -- the MULTI is
             * not decoration: iTunes tiles a rounded-corner window
             * with nine rectangles in one 29-dword packet, and
             * stopping after the first copied a 1-pixel strip and
             * dropped the 600x390 body. Mac OS X 10.4 sets both
             * pitch/offset bits for its 16x16 pointer save/restore;
             * reading that as one dword plus a rectangle turned the
             * restore into "corruption growing from the top-left
             * towards the mouse".
             *
             * The rectangle run is longer than p3_params, so each
             * three-dword rectangle is gathered in place and issued as
             * soon as it completes rather than buffering the packet.
             * p3_params[3] counts the pitch/offset dwords the context
             * announced, so the rectangle run is known to start at
             * index 1 + that count.
             */
            if (p->p3_param_idx == 0) {
                ati_r580_reg_write32(s, R580_DP_GUI_MASTER_CNTL, val);
                p->p3_params[3] = 0;
                if (val & R580_GMC_SRC_PITCH_OFFSET_CNTL) {
                    p->p3_params[3]++;
                }
                if (val & R580_GMC_DST_PITCH_OFFSET_CNTL) {
                    p->p3_params[3]++;
                }
                p->p3_params[4] = val;
            } else if (p->p3_param_idx <= p->p3_params[3]) {
                bool src_first = p->p3_params[4] &
                                 R580_GMC_SRC_PITCH_OFFSET_CNTL;

                if (p->p3_param_idx == 1 && src_first) {
                    ati_r580_reg_write32(s, R580_SRC_PITCH_OFFSET, val);
                } else {
                    ati_r580_reg_write32(s, R580_DST_PITCH_OFFSET, val);
                }
            } else {
                unsigned slot = (p->p3_param_idx - 1 - p->p3_params[3]) % 3;

                p->p3_params[slot] = val;
                if (slot == 2) {
                    s->src_y = p->p3_params[0] & 0x3fff;
                    s->src_x = (p->p3_params[0] >> 16) & 0x3fff;
                    s->dst_y = p->p3_params[1] & 0x3fff;
                    s->dst_x = (p->p3_params[1] >> 16) & 0x3fff;
                    s->dst_height = p->p3_params[2] & 0x3fff;
                    s->dst_width = (p->p3_params[2] >> 16) & 0x3fff;
                    ati_r580_2d_blt(s);
                }
            }
            p->p3_param_idx++;
            break;

        case R580_PM4_OPCODE_BITBLT:
            /*
             * Context, then the pitch/offset dwords the context
             * announces (SRC_PITCH_OFFSET_CNTL, DST_PITCH_OFFSET_CNTL,
             * in that order), then SRC_X_Y, DST_X_Y, DST_WIDTH_HEIGHT
             * -- and, like the registers of those names (and unlike
             * PAINT's corners), X and WIDTH sit in the HIGH half with Y
             * and HEIGHT in the low one. The packet length says how many
             * dwords precede the rectangle; Mac OS X 10.5 scrolls a view
             * in place with a five-dword packet carrying the source
             * pitch/offset.
             */
            {
                uint32_t nopt = p->p3_total > 4 ? p->p3_total - 4 : 0;
                uint32_t idx = p->p3_param_idx++;

                if (idx == 0) {
                    p->p3_params[0] = val;
                    ati_r580_reg_write32(s, R580_DP_GUI_MASTER_CNTL, val);
                } else if (idx <= nopt) {
                    bool src = p->p3_params[0] &
                               R580_GMC_SRC_PITCH_OFFSET_CNTL;
                    bool dst = p->p3_params[0] &
                               R580_GMC_DST_PITCH_OFFSET_CNTL;

                    if (idx == 1 && src) {
                        ati_r580_reg_write32(s, R580_SRC_PITCH_OFFSET, val);
                    } else if (idx == 1 + src && dst) {
                        ati_r580_reg_write32(s, R580_DST_PITCH_OFFSET, val);
                    }
                } else if (idx <= nopt + 3) {
                    p->p3_params[idx - nopt] = val;
                    if (idx == nopt + 3) {
                        s->src_y = p->p3_params[1] & 0x3fff;
                        s->src_x = (p->p3_params[1] >> 16) & 0x3fff;
                        s->dst_y = p->p3_params[2] & 0x3fff;
                        s->dst_x = (p->p3_params[2] >> 16) & 0x3fff;
                        s->dst_height = p->p3_params[3] & 0x3fff;
                        s->dst_width = (p->p3_params[3] >> 16) & 0x3fff;
                        ati_r580_2d_blt(s);
                    }
                }
            }
            break;
        case R580_PM4_OPCODE_BITBLT_RECT:
            /*
             * One SRC_X_Y/DST_X_Y/DST_WIDTH_HEIGHT rectangle in
             * BITBLT's field layout, drawing context inherited from
             * the registers -- see the define for the capture that
             * established it. No GMC dword, so no register writes
             * here.
             */
            if (p->p3_param_idx < 3) {
                p->p3_params[p->p3_param_idx++] = val;
                if (p->p3_param_idx == 3) {
                    s->src_y = p->p3_params[0] & 0x3fff;
                    s->src_x = (p->p3_params[0] >> 16) & 0x3fff;
                    s->dst_y = p->p3_params[1] & 0x3fff;
                    s->dst_x = (p->p3_params[1] >> 16) & 0x3fff;
                    s->dst_height = p->p3_params[2] & 0x3fff;
                    s->dst_width = (p->p3_params[2] >> 16) & 0x3fff;
                    ati_r580_2d_blt(s);
                }
            }
            break;
        case R580_PM4_OPCODE_SCALING:
            /* fixed-length packet: see R580_SCALE_PKT_* in the header */
            if (p->p3_param_idx < R580_SCALE_PKT_DWORDS) {
                p->p3_scale[p->p3_param_idx++] = val;
                if (p->p3_param_idx == R580_SCALE_PKT_DWORDS) {
                    ati_r580_reg_write32(s, R580_DP_GUI_MASTER_CNTL,
                                    p->p3_scale[R580_SCALE_PKT_GMC]);
                    ati_r580_reg_write32(s, R580_SC_TOP_LEFT,
                                    p->p3_scale[R580_SCALE_PKT_SC_TL]);
                    ati_r580_reg_write32(s, R580_SC_BOTTOM_RIGHT,
                                    p->p3_scale[R580_SCALE_PKT_SC_BR]);
                    ati_r580_2d_scale(s, p->p3_scale);
                }
            }
            break;
        case R580_PM4_OPCODE_HOSTDATA_BLT:
        {
            /*
             * Header dwords before the pixel data. Two long layouts
             * exist -- Linux's r128 DRM emits seven header dwords
             * (context, pitch/offset, write mask, clip, position,
             * size, count), the Mac driver eight, with one extra
             * dword before the position -- and the last header dword
             * is the pixel-dword count, which must equal what is left
             * of the packet: that identity tells the two apart
             * without guessing.
             */
            uint32_t nhdr = p->p3_total >= 8 ? 8 : 2;

            if (p->p3_param_idx < nhdr) {
                p->p3_params[p->p3_param_idx++] = val;
                if (p->p3_param_idx == nhdr) {
                    uint32_t real = nhdr;
                    uint32_t yx, hw;

                    if (nhdr == 8) {
                        real = p->p3_params[7] == p->p3_total - 8 ? 8 :
                               p->p3_params[6] == p->p3_total - 7 ? 7 : 8;
                        ati_r580_reg_write32(s, R580_DP_GUI_MASTER_CNTL,
                                                p->p3_params[0]);
                    }
                    if (real == 8) {
                        /*
                         * Context first, then the clip: the driver
                         * pads the blit width up to a 4-pixel boundary
                         * and expects the clip to discard the surplus,
                         * and a GMC write with DST_CLIPPING clear
                         * resets the scissors.
                         */
                        ati_r580_reg_write32(s, R580_SC_TOP_LEFT,
                                                p->p3_params[1]);
                        ati_r580_reg_write32(s, R580_SC_BOTTOM_RIGHT,
                                                p->p3_params[2]);
                    }
                    yx = nhdr == 2 ? p->p3_params[0]
                                   : p->p3_params[real - 3];
                    hw = nhdr == 2 ? p->p3_params[1]
                                   : p->p3_params[real - 2];
                    s->dst_x = yx & 0x3fff;
                    s->dst_y = (yx >> 16) & 0x3fff;
                    s->dst_width = hw & 0x3fff;
                    s->dst_height = (hw >> 16) & 0x3fff;
                    ati_r580_2d_blt(s); /* enters host-data mode */
                    if (nhdr == 8 && real == 7 && s->host_data_active) {
                        /* the 8th dword already read is pixel data */
                        s->host_data_acc[0] = p->p3_params[7];
                        s->host_data_next = 1;
                    }
                }
            } else if (s->host_data_active) {
                s->host_data_acc[s->host_data_next++] = val;
                if (s->host_data_next >= 4) {
                    ati_r580_host_data_flush(s);
                    s->host_data_next = 0;
                }
            }
            break;
        }
        case R300_PM4_OPCODE_DRAW_IMMD_2:
        case R300_PM4_OPCODE_DRAW_INDX_2:
            if (p->p3_param_idx < ARRAY_SIZE(s->r300_immd)) {
                s->r300_immd[p->p3_param_idx] = val;
            }
            p->p3_param_idx++;
            break;
        case R300_PM4_OPCODE_CLEAR_ZMASK:
            /* first dword, dword count, value */
            if (p->p3_param_idx < 3) {
                p->p3_params[p->p3_param_idx] = val;
                if (p->p3_param_idx == 2) {
                    ati_r580_r300_clear_zmask(s, p->p3_params[0],
                                              p->p3_params[1], val);
                }
            }
            p->p3_param_idx++;
            break;
        case R300_PM4_OPCODE_INDX_BUFFER:
            /* port register, card address, dword count */
            if (p->p3_param_idx < 3) {
                p->p3_params[p->p3_param_idx] = val;
                if (p->p3_param_idx == 2) {
                    ati_r580_r300_indx_buffer(s, p->p3_params[0],
                                              p->p3_params[1], val);
                }
            }
            p->p3_param_idx++;
            break;
        case R300_PM4_OPCODE_DRAW_VBUF_2:
            /* single payload dword: VAP_VF_CNTL for an AOS-array draw */
            if (p->p3_param_idx++ == 0) {
                ati_r580_r300_draw_vbuf(s, val);
            }
            break;
        case R300_PM4_OPCODE_LOAD_VBPNTR:
            /*
             * Same data in the same order as type-0 writes to the
             * VAP_VTX_AOS block: the array count, then for each PAIR
             * of arrays a packed size|stride control dword followed by
             * the two addresses (Linux r100_packet3_load_vbpntr). The
             * packet's dwords map one for one onto the registers from
             * VAP_VTX_AOS_CNT up, so writing them through is all this
             * needs to do -- but only for as many arrays as the draw
             * engine fetches. Stopping after four dwords covered a
             * two-array binding and silently dropped the third array
             * of anything that bound more.
             */
            if (p->p3_param_idx < 1 + (R300_AOS_MAX + 1) / 2 * 3) {
                ati_r580_reg_write32(s, R300_VAP_VTX_AOS_CNT +
                                        p->p3_param_idx * 4, val);
            }
            p->p3_param_idx++;
            break;
        default:
            /*
             * Payload of an opcode we do not model. Still advance the
             * parameter index: it is what the payload trace reports, and
             * leaving it pinned at zero made every dword of such a packet
             * look like a fresh one-dword packet, which turned a single
             * 960-dword CNTL_SCALING into 960 phantom packets in the
             * offline analyser. The data itself is discarded as before.
             */
            if (p->p3_param_idx < 0xffff) {
                p->p3_param_idx++;
            }
            break;
        }
        break;
    default:
        break;
    }
    p->remaining--;
    if (p->remaining == 0 && p->type == 3 &&
        p->p3_opcode == R580_PM4_OPCODE_HOSTDATA_BLT &&
        s->host_data_active) {
        ati_r580_host_data_flush(s);
        s->host_data_active = false;
        s->host_data_next = 0;
    }
    if (p->remaining == 0 && p->type == 3 &&
        p->p3_opcode == R300_PM4_OPCODE_DRAW_IMMD_2 &&
        p->p3_total >= 1 && p->p3_total <= ARRAY_SIZE(s->r300_immd)) {
        ati_r580_r300_draw_immd(s, s->r300_immd, p->p3_total);
    }
    if (p->remaining == 0 && p->type == 3 &&
        p->p3_opcode == R300_PM4_OPCODE_DRAW_INDX_2 &&
        p->p3_total >= 1 && p->p3_total <= ARRAY_SIZE(s->r300_immd)) {
        ati_r580_r300_draw_indx(s, s->r300_immd, p->p3_total);
    }
}

void ati_r580_pm4_fifo_push(ATIR580State *s, uint32_t val)
{
    ati_r580_pm4_parse(s, &s->pm4_fifo, val);
    ati_r580_gl_release(s, R580_GLR_FIFO);
    ati_r580_zpass_settle(s);
}

/*
 * Indirect-buffer dispatch (PM4_IW_INDOFF/INDSIZE): the engine
 * fetches `dwords` command dwords from card address `offset` --
 * bus-master command submission from GART-translated system memory
 * (or VRAM). This is how Mac OS X's driver, running the CCE in mode
 * 7 (64PIO_64VCBM_64INDBM), submits its actual drawing/mode-set
 * command buffers; the Linux driver does the same via
 * r128_cce_dispatch_indirect(). Each fetched dword goes through the
 * same packet state machine as the PIO FIFO, so every packet type the
 * FIFO path understands works identically here.
 */
void ati_r580_pm4_indirect(ATIR580State *s, uint32_t offset,
                          uint32_t dwords)
{
    /*
     * The IND queue is a bus-master fetcher by definition (the CNTL
     * mode names literally call it INDBM): with a GART configured,
     * IW_INDOFF addresses are offsets into the GART-translated VM
     * space (guest system memory), even small ones -- confirmed live:
     * treating small offsets as VRAM read all-zero/stale bytes, while
     * the guest's real command buffers sit in the GART pages.
     */
    ATIR580PM4Parser parser = { 0 };
    bool swap = (s->regs[R580_CP_RB_CNTL >> 2] & R580_BUF_SWAP_MASK) ==
                R580_BUF_SWAP_32BIT;
    uint32_t buf[1024], map[R580_MC_MAP_REGS];
    uint32_t i, have = 0, base = 0;

    trace_ati_r580_pm4_indirect(offset, dwords,
        dwords ? ati_r580_mc_read32(s, offset) : 0);
    s->engine_ibs++;
    for (i = 0; i < dwords; i++) {
        uint32_t val;

        /*
         * Fetched ahead a chunk at a time. A packet that reprograms the
         * memory controller's windows discards what was fetched under
         * the old ones.
         */
        if (i - base >= have || !ati_r580_mc_map_same(s, map)) {
            uint32_t off;

            base = i;
            have = MIN(dwords - i, (uint32_t)ARRAY_SIZE(buf));
            if (ati_r580_mc_to_vram(s, offset + i * 4, &off)) {
                have = 1;       /* VRAM reads are coherency points */
            }
            ati_r580_mc_map_save(s, map);
            ati_r580_mc_read_block(s, offset + i * 4, buf, have);
        }
        val = buf[i - base];

        /*
         * CP_RB_CNTL's BUF_SWAP swapper sits on the CP's fetch port,
         * not on the ring specifically: indirect-buffer fetches go
         * through the same dword swapper as ring fetches. OS X's
         * driver stores IBs with native big-endian stores and relies
         * on this (an unswapped fetch reads its 0x80000000 NOP
         * padding as 0x00000080 and the whole IB parses as garbage).
         */
        if (swap) {
            val = bswap32(val);
        }
        trace_ati_r580_pm4_ib_dword(i, val);
        ati_r580_pm4_parse(s, &parser, val);
    }
    /*
     * Inside the guest store that issued it this is the guest's next
     * chance to look at VRAM. On the command processor thread it is not:
     * the guest learns what has finished from a scratch register or the
     * read pointer, and each of those hands the target back first.
     */
    if (!ati_r580_on_engine()) {
        ati_r580_gl_release(s, R580_GLR_IB);
        ati_r580_zpass_settle(s);
    }
}

/*
 * All register apertures allow 1/2/4-byte access at any offset
 * (RRG-G04500-C: "access: 8/16/32"). Sub-dword accesses are folded
 * onto the 32-bit register via read-modify-write against the raw
 * stored value.
 */
/*
 * COMMAND PROCESSOR THREAD.
 *
 * A CPU write of CP_RB_WPTR records the pointer and wakes a thread that
 * drains the ring without the BQL, so the guest runs on while its
 * commands draw, as it does beside the chip's own CP. Indirect buffers
 * the ring names run on the same thread, and so does the GL backend:
 * its context is made current wherever a draw runs.
 *
 * Registers split in two. The engine's own -- the CP, the 2D and 3D
 * blocks, and the memory-controller windows, surfaces and bus master it
 * reads while it runs (ati_r580_engine_reg()) -- are what a CPU access
 * waits for it to go idle before touching, the BQL released while it
 * waits; so commands complete in order and nothing it uses moves under
 * it. The read pointer, RBBM_STATUS, GUI_STAT, CP_STAT and the scratch
 * registers answer at once with its progress. Everything else --
 * display, cursor, interrupts, clocks -- belongs to the BQL, which a
 * command stream write to it takes.
 *
 * The GL target is the engine's while it runs: a hook on another thread
 * finds it busy and leaves the target alone (ati_r580_gl_enter()). The
 * ring and each indirect buffer end with the target handed back, and a
 * scratch register write or a read pointer write-back hands it back
 * first, so whatever the guest learns has finished is in VRAM.
 */
static __thread bool ati_r580_engine_ctx;

/*
 * LOCKLESS REGISTER ACCESS (lockless-mmio=on, with the engine thread).
 *
 * The register BAR is dispatched without the BQL, and an access takes
 * only what its register needs:
 *
 * - The progress registers (ati_r580_engine_status_reg(), RBBM_STATUS)
 *   are read with no lock: they are atomics the command processor
 *   publishes as it goes.
 * - Any other engine register (ati_r580_engine_reg()) is accessed by a
 *   thread that has CLAIMED the engine: engine_busy set for it, just as
 *   the command processor holds it while it runs, so the two and any
 *   other claimant exclude each other. A register the command processor
 *   itself writes without the BQL (ati_r580_engine_private_reg()) needs
 *   nothing more. The others, and the writes that run a command stream
 *   on the calling thread (the PIO FIFO, CP_IB_BUFSZ, a CP_RB_WPTR write
 *   narrower than the register), take the BQL inside the claim
 *   (ati_r580_engine_bql_reg()).
 * - Everything else -- display, cursor, interrupts, clocks, I2C/DDC, the
 *   PCI config mirror -- takes the BQL, as before.
 * - A full CP_RB_WPTR write kicks the command processor without the BQL.
 *   It waits only for a claimant to finish, never for the command
 *   processor; with RB_RPTR_WR_ENA it claims the engine, loads the read
 *   pointer and hands the claim over as the kick.
 *
 * A claim is what the BQL used to guarantee a vCPU that had seen the
 * engine idle. A thread that uses the GL target outside one (the display
 * refresh, the cursor, a display register write) takes the engine only
 * if it is idle (ati_r580_gl_enter()), and leaves the target alone
 * otherwise, as it always did while the command processor ran.
 *
 * LOCK ORDER: BQL -> engine_lock. engine_lock is never held while taking
 * the BQL or waiting for anything but engine_cond/engine_claim_cond. A
 * claim is not a lock: its holder may take the BQL (a register that
 * needs it; DMA that lands on a device region), and so no thread waits
 * for a claim, or for the command processor, with the BQL held --
 * ati_r580_engine_claim(), ati_r580_engine_wait() and the kick release
 * it while they wait, and the main loop only tries. gl_tex_lock and
 * scan_lock are taken inside a claim and never held while waiting for
 * one. The refresh's dirty snapshot, which may wait for a vCPU, holds no
 * claim.
 *
 * With lockless-mmio=off the BAR keeps the BQL and none of this runs.
 */
static __thread unsigned ati_r580_claim_depth;
static __thread bool ati_r580_claim_kick;

bool ati_r580_on_engine(void)
{
    return ati_r580_engine_ctx;
}

bool ati_r580_engine_reg(uint32_t base)
{
    if (ati_r580_display_reg(base)) {
        return false;
    }
    return base < R580_CLOCK_CNTL_INDEX ||           /* MM_INDEX/DATA */
           base == R5XX_MC_IND_INDEX || base == R5XX_MC_IND_DATA ||
           base == R5XX_PCIE_INDEX || base == R5XX_PCIE_DATA ||
           (base >= R580_MC_FB_LOCATION && base <= R580_MC_AGP_LOCATION) ||
           (base >= R580_AGP_BASE && base <= R580_PCI_GART_PAGE) ||
           (base >= R580_AIC_CNTL && base <= R580_AIC_HI_ADDR) ||
           (base >= 0x0700 && base < 0x0800) ||      /* CP */
           (base >= 0x0a00 && base < 0x0c00) ||      /* BM, GART, surfaces */
           base >= R580_PM4_FIFO_DATA_EVEN;          /* FIFO, 2D, 3D */
}

bool ati_r580_engine_status_reg(uint32_t base)
{
    return base == R580_CP_RB_RPTR || base == R580_CP_RB_WPTR ||
           base == R580_CP_STAT || base == R580_CP_CSQ_STAT ||
           base == R580_GUI_STAT ||
           (base >= R580_SCRATCH_REG_BASE && base <= R580_SCRATCH_REG_LAST);
}

/* registers the command processor writes without the BQL */
bool ati_r580_engine_private_reg(uint32_t base)
{
    return (base >= 0x0700 && base < 0x0800) ||
           (base >= R580_PM4_FIFO_DATA_EVEN && !ati_r580_display_reg(base));
}

/* engine registers a claimant accesses with the BQL as well */
bool ati_r580_engine_bql_reg(uint32_t base)
{
    return !ati_r580_engine_private_reg(base) ||
           (base >= R580_PM4_FIFO_DATA_EVEN &&
            base <= R580_PM4_FIFO_APER_END) ||
           base == R580_CP_IB_BUFSZ || base == R580_CP_RB_WPTR;
}

/*
 * Wait for the engine to go idle. Called with the BQL, which is released.
 * A thread that has claimed the engine has it idle already.
 */
void ati_r580_engine_wait(ATIR580State *s)
{
    int64_t t0;

    if (!s->engine_on || ati_r580_engine_ctx || ati_r580_claim_depth ||
        !qatomic_load_acquire(&s->engine_busy)) {
        return;
    }
    t0 = get_clock();
    do {
        bql_unlock();
        qemu_event_wait(&s->engine_idle);
        bql_lock();
    } while (qatomic_load_acquire(&s->engine_busy));
    s->engine_waits++;
    s->engine_wait_us += (get_clock() - t0) / 1000;
}

void ati_r580_engine_kick(ATIR580State *s)
{
    qemu_mutex_lock(&s->engine_lock);
    s->engine_kick = true;
    s->engine_kicks++;
    if (!s->engine_busy) {
        qemu_event_reset(&s->engine_idle);
        qatomic_store_release(&s->engine_busy, true);
    }
    qemu_cond_signal(&s->engine_cond);
    qemu_mutex_unlock(&s->engine_lock);
}

/* the engine is idle and engine_lock held: it is this thread's now */
static void ati_r580_engine_take(ATIR580State *s)
{
    qemu_event_reset(&s->engine_idle);
    s->engine_claimed = true;
    s->engine_claims++;
    qatomic_store_release(&s->engine_busy, true);
    ati_r580_claim_depth = 1;
}

/*
 * Claim the engine, waiting until neither the command processor nor
 * another claimant has it; the BQL, if held, is released meanwhile.
 * Nests, and is a no-op on the engine thread.
 */
void ati_r580_engine_claim(ATIR580State *s)
{
    bool bql;
    int64_t t0;

    if (ati_r580_engine_ctx) {
        return;
    }
    if (ati_r580_claim_depth) {
        ati_r580_claim_depth++;
        return;
    }
    bql = bql_locked();
    qemu_mutex_lock(&s->engine_lock);
    if (s->engine_busy) {
        t0 = get_clock();
        do {
            qemu_mutex_unlock(&s->engine_lock);
            if (bql) {
                bql_unlock();
            }
            qemu_event_wait(&s->engine_idle);
            if (bql) {
                bql_lock();
            }
            qemu_mutex_lock(&s->engine_lock);
        } while (s->engine_busy);
        s->engine_claim_waits++;
        s->engine_claim_wait_us += (get_clock() - t0) / 1000;
    }
    ati_r580_engine_take(s);
    qemu_mutex_unlock(&s->engine_lock);
}

/* Claim the engine only if it is idle; never waits. */
static bool ati_r580_engine_try_claim(ATIR580State *s)
{
    qemu_mutex_lock(&s->engine_lock);
    if (s->engine_busy) {
        s->engine_claim_busy++;
        qemu_mutex_unlock(&s->engine_lock);
        return false;
    }
    ati_r580_engine_take(s);
    qemu_mutex_unlock(&s->engine_lock);
    return true;
}

/*
 * Give the engine back, or with `kick` hand it to the command processor
 * as a ring kick. A kick requested inside a nested claim is carried to
 * the outermost one.
 */
void ati_r580_engine_unclaim(ATIR580State *s, bool kick)
{
    if (ati_r580_engine_ctx) {
        return;
    }
    if (--ati_r580_claim_depth) {
        ati_r580_claim_kick |= kick;
        return;
    }
    kick |= ati_r580_claim_kick;
    ati_r580_claim_kick = false;
    qemu_mutex_lock(&s->engine_lock);
    s->engine_claimed = false;
    if (kick) {
        s->engine_kick = true;
        s->engine_kicks++;
        qemu_cond_signal(&s->engine_cond);
    } else {
        qatomic_store_release(&s->engine_busy, false);
        qemu_event_set(&s->engine_idle);
    }
    qemu_cond_broadcast(&s->engine_claim_cond);
    qemu_mutex_unlock(&s->engine_lock);
}

bool ati_r580_gl_enter(ATIR580State *s, bool *claimed)
{
    *claimed = false;
    if (!s->engine_on || ati_r580_engine_ctx || ati_r580_claim_depth) {
        return true;
    }
    if (!s->lockless) {
        return !qatomic_read(&s->engine_busy);
    }
    *claimed = ati_r580_engine_try_claim(s);
    return *claimed;
}

void ati_r580_gl_leave(ATIR580State *s, bool claimed)
{
    if (claimed) {
        ati_r580_engine_unclaim(s, false);
    }
}

static void ati_r580_wptr_store(ATIR580State *s, uint32_t val)
{
    s->regs[R580_CP_RB_WPTR >> 2] = val;
    qatomic_store_release(&s->pm4_wptr, s->pm4_ring_dwords ?
                          val & (s->pm4_ring_dwords - 1) : val);
}

/* A full CP_RB_WPTR write with lockless register access. */
void ati_r580_engine_wptr(ATIR580State *s, uint32_t val)
{
    bool bql;

    if (ati_r580_engine_ctx || ati_r580_claim_depth) {
        /* this thread has the engine; the kick follows its release */
        ati_r580_cp_rptr_load(s);
        ati_r580_wptr_store(s, val);
        if (ati_r580_engine_ctx) {
            ati_r580_engine_kick(s);
        } else {
            ati_r580_claim_kick = true;
        }
        return;
    }
    if (qatomic_read(&s->regs[R580_CP_RB_CNTL >> 2]) & R580_RB_RPTR_WR_ENA) {
        ati_r580_engine_claim(s);
        ati_r580_cp_rptr_load(s);
        ati_r580_wptr_store(s, val);
        ati_r580_engine_unclaim(s, true);
        return;
    }
    bql = bql_locked();
    qemu_mutex_lock(&s->engine_lock);
    while (s->engine_claimed) {
        if (bql) {
            qemu_mutex_unlock(&s->engine_lock);
            bql_unlock();
            qemu_mutex_lock(&s->engine_lock);
        }
        while (s->engine_claimed) {
            qemu_cond_wait(&s->engine_claim_cond, &s->engine_lock);
        }
        if (bql) {
            qemu_mutex_unlock(&s->engine_lock);
            bql_lock();
            qemu_mutex_lock(&s->engine_lock);
        }
    }
    ati_r580_wptr_store(s, val);
    s->engine_kick = true;
    s->engine_kicks++;
    if (!s->engine_busy) {
        qemu_event_reset(&s->engine_idle);
        qatomic_store_release(&s->engine_busy, true);
    }
    qemu_cond_signal(&s->engine_cond);
    qemu_mutex_unlock(&s->engine_lock);
}

static void *ati_r580_engine_thread(void *opaque)
{
    ATIR580State *s = opaque;

    rcu_register_thread();
    ati_r580_engine_ctx = true;
    ati_r580_gl_hold(true);
    qemu_mutex_lock(&s->engine_lock);
    for (;;) {
        while (!s->engine_kick && !s->engine_quit) {
            qemu_cond_wait(&s->engine_cond, &s->engine_lock);
        }
        if (s->engine_quit) {
            break;
        }
        s->engine_kick = false;
        qemu_mutex_unlock(&s->engine_lock);

        ati_r580_pm4_run(s);

        qemu_mutex_lock(&s->engine_lock);
        if (!s->engine_kick) {
            /* the GL context is free before another thread may take it */
            ati_r580_gl_unbind(s->gl_ctx);
            qatomic_store_release(&s->engine_busy, false);
            qemu_event_set(&s->engine_idle);
        }
    }
    qemu_mutex_unlock(&s->engine_lock);
    rcu_unregister_thread();
    return NULL;
}

static void ati_r580_engine_vm_state(void *opaque, bool running,
                                     RunState state)
{
    if (!running) {
        ati_r580_engine_wait(opaque);
    }
}

void ati_r580_engine_init(ATIR580State *s)
{
    qemu_rec_mutex_init(&s->gl_tex_lock);
    qemu_mutex_init(&s->scan_lock);
    s->engine_on = s->engine_async == ON_OFF_AUTO_ON ||
                   (s->engine_async == ON_OFF_AUTO_AUTO && !qtest_enabled());
    if (!s->engine_on) {
        return;
    }
    qemu_mutex_init(&s->engine_lock);
    qemu_cond_init(&s->engine_cond);
    qemu_cond_init(&s->engine_claim_cond);
    qemu_event_init(&s->engine_idle, true);
    s->engine_vmse = qemu_add_vm_change_state_handler(ati_r580_engine_vm_state,
                                                      s);
    qemu_thread_create(&s->engine_thread, "ati-r580-cp",
                       ati_r580_engine_thread, s, QEMU_THREAD_JOINABLE);
}

void ati_r580_engine_fini(ATIR580State *s)
{
    if (!s->engine_on) {
        return;
    }
    ati_r580_engine_wait(s);
    qemu_mutex_lock(&s->engine_lock);
    s->engine_quit = true;
    qemu_cond_signal(&s->engine_cond);
    qemu_mutex_unlock(&s->engine_lock);
    qemu_thread_join(&s->engine_thread);
    qemu_del_vm_change_state_handler(s->engine_vmse);
    s->engine_on = false;
}
