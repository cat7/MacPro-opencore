/*
 * QEMU ATI Radeon HD 5770 (Juniper) emulation -- see ati_hd5770_int.h.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Adapted from ati_r580.c.
 */

#include "qemu/osdep.h"
#include "qemu/bswap.h"
#include "qemu/error-report.h"
#include "qemu/log.h"
#include "qemu/module.h"
#include "qemu/units.h"
#include "hw/pci/pci_device.h"
#include "hw/pci/pci_bus.h"
#include "hw/pci/pcie.h"
#include "hw/pci/msi.h"
#include "hw/core/qdev-properties.h"
#include "migration/vmstate.h"
#include "qapi/error.h"
#include "system/dma.h"
#include "system/memory.h"
#include "ui/console.h"
#include "ati_hd5770_int.h"
#include "ati_hd5770_regs.h"
#include "trace.h"

/* ---------------------------------------------------------------- */
/* Memory controller                                                */

/* the FB window: MC addresses [base, top] are VRAM from offset 0 */
static void fb_window(ATIHD5770State *s, uint64_t *base, uint64_t *top)
{
    uint32_t loc = s->regs[EG_MC_VM_FB_LOCATION >> 2];

    *base = (uint64_t)(loc & 0xffff) << 24;
    *top = ((uint64_t)(loc >> 16) << 24) | 0xffffff;
}

bool ati_hd5770_mc_to_vram(ATIHD5770State *s, uint64_t mc, uint64_t *off)
{
    uint64_t base, top;

    fb_window(s, &base, &top);
    if (mc < base || mc > top || mc - base >= s->vram_size) {
        return false;
    }
    *off = mc - base;
    return true;
}

/* a write by the chip: VRAM inside the FB window, else system memory */
static void mc_write(ATIHD5770State *s, uint64_t mc, const void *buf,
                     unsigned len)
{
    uint64_t off;

    if (ati_hd5770_mc_to_vram(s, mc, &off) && off + len <= s->vram_size) {
        memcpy((uint8_t *)memory_region_get_ram_ptr(&s->vram) + off, buf,
               len);
        memory_region_set_dirty(&s->vram, off, len);
        return;
    }
    pci_dma_write(PCI_DEVICE(s), mc, buf, len);
}

/* ---------------------------------------------------------------- */
/* Interrupt handler ring                                           */

static uint32_t ih_size(ATIHD5770State *s)
{
    uint32_t log2dw = (s->regs[EG_IH_RB_CNTL >> 2] >> 1) & 0x1f;

    return 4u << MIN(log2dw, 20);
}

static bool ih_pending(ATIHD5770State *s)
{
    uint32_t wptr = s->regs[EG_IH_RB_WPTR >> 2] & ~EG_IH_RB_WPTR_OVERFLOW;

    return (s->regs[EG_IH_CNTL >> 2] & EG_IH_ENABLE_INTR) &&
           (s->regs[EG_IH_RB_CNTL >> 2] & EG_IH_RB_ENABLE) &&
           (s->regs[EG_IH_RB_RPTR >> 2] & (ih_size(s) - 1)) != wptr;
}

/*
 * INTx follows "ring not empty". MSI is an edge: one message per new
 * entry, and one more when the driver writes the read pointer while
 * entries are left.
 */
void ati_hd5770_update_irq(ATIHD5770State *s)
{
    PCIDevice *d = PCI_DEVICE(s);
    bool level = ih_pending(s);

    if (msi_enabled(d)) {
        if (level && !s->ih_level) {
            msi_notify(d, 0);
        }
        pci_set_irq(d, 0);
    } else {
        pci_set_irq(d, level);
    }
    s->ih_level = level;
}

/* a 16-byte vector: source id, source data, ring id 0, reserved */
void ati_hd5770_ih_push(ATIHD5770State *s, uint32_t src_id, uint32_t src_data)
{
    uint32_t cntl = s->regs[EG_IH_RB_CNTL >> 2];
    uint32_t size = ih_size(s), wptr, rptr, next;
    uint8_t v[16];

    if (!(cntl & EG_IH_RB_ENABLE)) {
        return;
    }
    wptr = s->regs[EG_IH_RB_WPTR >> 2] & (size - 1) & ~15u;
    rptr = s->regs[EG_IH_RB_RPTR >> 2] & (size - 1);
    next = (wptr + 16) & (size - 1);
    if (next == rptr) {
        if (cntl & EG_IH_WPTR_OVERFLOW_ENABLE) {
            s->regs[EG_IH_RB_WPTR >> 2] |= EG_IH_RB_WPTR_OVERFLOW;
        }
        trace_ati_hd5770_ih_overflow(src_id, src_data, wptr, rptr);
        return;
    }
    stl_le_p(v, src_id & 0xff);
    stl_le_p(v + 4, src_data & 0xfffffff);
    stl_le_p(v + 8, 0);
    stl_le_p(v + 12, 0);
    mc_write(s, ((uint64_t)s->regs[EG_IH_RB_BASE >> 2] << 8) + wptr, v, 16);
    s->regs[EG_IH_RB_WPTR >> 2] = next |
        (s->regs[EG_IH_RB_WPTR >> 2] & EG_IH_RB_WPTR_OVERFLOW);
    if (cntl & EG_IH_WPTR_WRITEBACK_ENABLE) {
        uint8_t w[4];
        uint64_t a = ((uint64_t)(s->regs[EG_IH_RB_WPTR_ADDR_HI >> 2] & 0xff)
                      << 32) | (s->regs[EG_IH_RB_WPTR_ADDR_LO >> 2] & ~3u);

        stl_le_p(w, s->regs[EG_IH_RB_WPTR >> 2]);
        mc_write(s, a, w, 4);
    }
    trace_ati_hd5770_ih_push(src_id, src_data, wptr);
    if (msi_enabled(PCI_DEVICE(s)) && s->ih_level) {
        msi_notify(PCI_DEVICE(s), 0);
    }
    ati_hd5770_update_irq(s);
}

/* ---------------------------------------------------------------- */
/* Register map                                                     */

/*
 * Registers the model knows: behaviour, or storage the M1 display path
 * does not need to act on (marked). Anything else is stored too, but
 * traced and counted (`gaps`).
 */
typedef struct RegRange {
    uint32_t lo, hi;            /* [lo, hi) */
    bool per_crtc;              /* offsets of CRTC 0, repeated per CRTC */
} RegRange;

static const RegRange known_regs[] = {
    { 0x0000, 0x0020 },         /* MM index/data, PIF PHY index/data */
    { 0x0030, 0x0040 },         /* PCIE and PCIE_PORT index/data */
    { 0x00a8, 0x00b0 },         /* ROM index/data */
    { 0x0100, 0x0108 },         /* RCU index/data: stored */
    { 0x0200, 0x0240 },         /* SMC SRAM, I/O, messages, ISR vectors */
    { 0x0400, 0x0500 },         /* PPLLs: stored */
    { 0x05ac, 0x05d0 },         /* audio DTO: stored */
    { 0x0600, 0x0800 },         /* clock generator, power, thermal: stored */
    { 0x08f8, 0x0900 },         /* CG index/data */
    { 0x0e50, 0x0e54 },         /* SRBM_STATUS: idle */
    { 0x0e60, 0x0e64 },         /* SRBM_SOFT_RESET: stored */
    { 0x0ec4, 0x0ec8 },         /* SRBM_STATUS2: idle */
    { 0x1400, 0x1600 },         /* VM L2 / contexts: stored, no GART */
    { 0x1700, 0x1800 },         /* GPIO pads, thermal: stored */
    { 0x2000, 0x3000 },         /* MC, HDP: FB window modelled, rest stored */
    { 0x3e00, 0x3e20 },         /* IH ring */
    { 0x3f00, 0x4000 },         /* RLC: stored, no microcode runs */
    { 0x5428, 0x5438 },         /* CONFIG_MEMSIZE, APER_SIZE, FB_OFFSET */
    { 0x5480, 0x5484 },         /* HDP flush: stored */
    { 0x5564, 0x5568 },         /* chip id strap */
    { 0x5e00, 0x6000 },         /* Azalia endpoints: stored */
    { 0x6000, 0x6018 },         /* DC misc: stored */
    { 0x6018, 0x6070 },         /* HPD 1-6 */
    { 0x6064, 0x60d0 },         /* DC_I2C engine */
    { 0x60f4, 0x6104 },         /* DISP_INTERRUPT_STATUS(_CONTINUE*) */
    { 0x614c, 0x6154 },
    { 0x6400, 0x6500 },         /* DC GPIO: DDC lines, HPD pins */
    { 0x6500, 0x6600 },         /* DC GPIO and DCCG misc: stored */
    { 0x6600, 0x6800 },         /* UNIPHY PHYs: stored */
    { 0x7000, 0x7400 },         /* DIG0, HDMI0, DP0: stored */
    { 0x7c00, 0x8000 },         /* DIG1 */
    { 0x10800, 0x10c00 },       /* DIG2 */
    { 0x11400, 0x11800 },       /* DIG3 */
    { 0x12000, 0x12400 },       /* DIG4 */
    { 0x12c00, 0x13000 },       /* DIG5 */
    { 0x8010, 0x8024 },         /* GRBM status, soft reset: idle */
    { 0x8680, 0x8684 },         /* CP_STAT: idle */
    { 0xd034, 0xd038 },         /* DMA_STATUS: idle */
    { 0xe020, 0xe024 },         /* UVD status: idle */
    { HD5770_ATOM_MAILBOX, HD5770_ATOM_DOORBELL + 4 },
    /* per CRTC */
    { 0x6800, 0x6850, true },   /* GRPH */
    { 0x6880, 0x6900, true },   /* overlay: stored */
    { 0x6940, 0x6998, true },   /* colour space, gamut, dither: stored */
    { 0x6998, 0x69c4, true },   /* cursor */
    { 0x69c4, 0x69e0, true },   /* icon: stored */
    { 0x69e0, 0x6a1c, true },   /* LUT */
    { 0x6a1c, 0x6b00, true },   /* LUT and regamma control: stored */
    { 0x6b00, 0x6bb8, true },   /* line buffer, INT_MASK */
    { 0x6bb8, 0x6c00, true },   /* VLINE/VBLANK_STATUS, LB misc */
    { 0x6c00, 0x6d00, true },   /* input gamma, regamma: stored */
    { 0x6d00, 0x6d80, true },   /* scaler, viewport: viewport used */
    { 0x6e00, 0x6f00, true },   /* CRTC timing and status */
    { 0x6f00, 0x7000, true },   /* FMT: stored */
};

static bool reg_known(uint32_t off)
{
    unsigned i, c;

    for (i = 0; i < ARRAY_SIZE(known_regs); i++) {
        const RegRange *r = &known_regs[i];

        if (!r->per_crtc) {
            if (off >= r->lo && off < r->hi) {
                return true;
            }
            continue;
        }
        for (c = 0; c < ATI_HD5770_NUM_CRTC; c++) {
            uint32_t o = ati_hd5770_crtc_offset[c];

            if (off >= r->lo + o && off < r->hi + o) {
                return true;
            }
        }
    }
    return false;
}

static void note_unknown(ATIHD5770State *s, uint32_t off, bool write,
                         uint32_t val)
{
    uint32_t *cnt = write ? s->unk_wr : s->unk_rd;

    if (reg_known(off)) {
        return;
    }
    if (write) {
        trace_ati_hd5770_unk_write(off, val);
    } else {
        trace_ati_hd5770_unk_read(off, val);
    }
    if (cnt[off >> 2] != UINT32_MAX) {
        cnt[off >> 2]++;
    }
}

/* the DCE blocks: global, CRTC 0-1 and DIG 0-1 below 0x8000, the rest
 * from 0x10000 */
static bool display_reg(uint32_t off)
{
    return (off >= EG_DC_FIRST && off < 0x8000) ||
           (off >= 0x10000 && off < 0x13000);
}

static uint32_t rom_byte(ATIHD5770State *s, uint32_t off)
{
    PCIDevice *d = PCI_DEVICE(s);
    uint64_t size;

    if (!d->has_rom) {
        return 0xff;
    }
    size = memory_region_size(&d->rom);
    if (off >= size) {
        return 0xff;
    }
    return ((const uint8_t *)memory_region_get_ram_ptr(&d->rom))[off];
}

static uint32_t chip_id(ATIHD5770State *s)
{
    return ((s->device_id & 0xffff) << 4) | ((s->revision & 0xf) << 28);
}

static uint32_t pcie_port_read(ATIHD5770State *s, unsigned idx)
{
    switch (idx) {
    case 0xa2:                  /* PCIE_LC_LINK_WIDTH_CNTL: x16 */
        return (s->pcie_port[idx] & ~0x70u) | (6u << 4);
    default:
        return s->pcie_port[idx];
    }
}

/*
 * The SMC as the power management library sees it: an SRAM the driver
 * loads (firmware, tables) and reads back, a reset/clock control, and a
 * message register whose every message is answered at once with
 * PPSMC_Result_OK. A halted SMC reports stop mode.
 */
#define PPSMC_RESULT_OK     1

static uint32_t *smc_sram_word(ATIHD5770State *s, bool step)
{
    uint32_t a = s->regs[EG_SMC_SRAM_ADDR >> 2];
    uint32_t *w = &s->smc_sram[(a & (ATI_HD5770_SMC_SRAM_SIZE - 1)) >> 2];

    if (step && !(a & EG_SMC_SRAM_AUTO_INC_DIS)) {
        s->regs[EG_SMC_SRAM_ADDR >> 2] = (a & ~0xffffu) | ((a + 4) & 0xffff);
    }
    return w;
}

static uint32_t reg_read(ATIHD5770State *s, uint32_t off)
{
    uint32_t val;

    off &= ATI_HD5770_MMIO_SIZE - 4;
    if (display_reg(off)) {
        ati_hd5770_dce4_read(s, off, &val);
    } else {
        switch (off) {
        case EG_PCIE_DATA:
            val = s->pcie_ind[s->regs[EG_PCIE_INDEX >> 2] & 0xff];
            break;
        case EG_PCIE_PORT_DATA:
            val = pcie_port_read(s, s->regs[EG_PCIE_PORT_INDEX >> 2] & 0xff);
            break;
        case EG_CG_IND_DATA:
            val = s->cg_ind[s->regs[EG_CG_IND_ADDR >> 2] & 0xff];
            break;
        case EG_ROM_DATA: {
            uint32_t i = s->regs[EG_ROM_INDEX >> 2];

            val = rom_byte(s, i) | rom_byte(s, i + 1) << 8 |
                  rom_byte(s, i + 2) << 16 | rom_byte(s, i + 3) << 24;
            break;
        }
        case EG_SMC_SRAM_DATA:
            val = *smc_sram_word(s, true);
            break;
        case EG_SMC_IO:
            val = s->regs[off >> 2] | EG_SMC_STOP_MODE;
            break;
        case EG_CG_SPLL_STATUS:
            val = s->regs[off >> 2] | EG_SPLL_CHG_STATUS;
            break;
        case EG_CG_MULT_THERMAL_STATUS:
            val = (s->regs[off >> 2] & ~0x07ff0000u) | (40u << 16);   /* 40 C */
            break;
        case EG_CONFIG_MEMSIZE:
            val = s->vram_mb;
            break;
        case EG_CONFIG_APER_SIZE:
            val = s->aper_mb * MiB;
            break;
        case EG_CONFIG_FB_OFFSET:
            val = 0;
            break;
        case EG_CHIP_ID:
            val = chip_id(s);
            break;
        /* stubs: no engine ever runs, all of them read idle */
        case EG_GRBM_STATUS:
        case EG_GRBM_STATUS_SE0:
        case EG_GRBM_STATUS_SE1:
        case EG_SRBM_STATUS:
        case EG_SRBM_STATUS2:
        case EG_CP_STAT:
        case EG_DMA_STATUS:
        case EG_UVD_STATUS:
            val = 0;
            break;
        default:
            val = s->regs[off >> 2];
            break;
        }
    }
    note_unknown(s, off, false, val);
    trace_ati_hd5770_reg_read(off, val);
    return val;
}

static void reg_write(ATIHD5770State *s, uint32_t off, uint32_t val)
{
    off &= ATI_HD5770_MMIO_SIZE - 4;
    trace_ati_hd5770_reg_write(off, val);
    note_unknown(s, off, true, val);
    if (ati_hd5770_atom_write(s, off, val)) {
        return;
    }
    if (display_reg(off)) {
        ati_hd5770_dce4_write(s, off, val);
        return;
    }
    switch (off) {
    case EG_PCIE_DATA:
        s->pcie_ind[s->regs[EG_PCIE_INDEX >> 2] & 0xff] = val;
        return;
    case EG_PCIE_PORT_DATA:
        s->pcie_port[s->regs[EG_PCIE_PORT_INDEX >> 2] & 0xff] = val;
        return;
    case EG_CG_IND_DATA:
        s->cg_ind[s->regs[EG_CG_IND_ADDR >> 2] & 0xff] = val;
        return;
    case EG_CONFIG_MEMSIZE:
    case EG_CONFIG_APER_SIZE:
    case EG_CONFIG_FB_OFFSET:
    case EG_CHIP_ID:
    case EG_ROM_DATA:
        return;                 /* read only */
    case EG_MC_VM_FB_LOCATION:
        s->regs[off >> 2] = val;
        s->force_redraw = true;
        return;
    case EG_SMC_SRAM_DATA:
        *smc_sram_word(s, true) = val;
        return;
    case EG_SMC_MSG:
        s->regs[off >> 2] = (val & 0xff) | (PPSMC_RESULT_OK << 8);
        trace_ati_hd5770_smc_msg(val & 0xff);
        return;
    case EG_MC_SEQ_CG: {
        /* the MC sequencer answers each request at once; DRAMCONF_S1
         * puts it in power state 1, S0 back in 0 */
        uint32_t req = val & 0xff;

        s->regs[off >> 2] = (val & ~0xff00u) | (req << 8);
        if (req == EG_MC_CG_SEQ_DRAMCONF_S1) {
            s->regs[EG_MC_SEQ_STATUS_M >> 2] |= EG_PMG_PWRSTATE;
        } else if (req == EG_MC_CG_SEQ_DRAMCONF_S0) {
            s->regs[EG_MC_SEQ_STATUS_M >> 2] &= ~EG_PMG_PWRSTATE;
        }
        return;
    }
    case EG_IH_RB_CNTL:
        if (val & EG_IH_WPTR_OVERFLOW_CLEAR) {
            s->regs[EG_IH_RB_WPTR >> 2] &= ~EG_IH_RB_WPTR_OVERFLOW;
        }
        s->regs[off >> 2] = val & ~EG_IH_WPTR_OVERFLOW_CLEAR;
        trace_ati_hd5770_ih_setup(s->regs[EG_IH_RB_CNTL >> 2],
                                  s->regs[EG_IH_RB_BASE >> 2],
                                  s->regs[EG_IH_CNTL >> 2]);
        ati_hd5770_update_irq(s);
        return;
    case EG_IH_RB_RPTR:
        s->regs[off >> 2] = val & ~15u;
        /* the driver re-arms MSI by writing the read pointer */
        if (msi_enabled(PCI_DEVICE(s)) && ih_pending(s)) {
            msi_notify(PCI_DEVICE(s), 0);
        }
        ati_hd5770_update_irq(s);
        return;
    case EG_IH_RB_WPTR:
        s->regs[off >> 2] = val & ~15u;
        ati_hd5770_update_irq(s);
        return;
    case EG_IH_CNTL:
        s->regs[off >> 2] = val;
        ati_hd5770_update_irq(s);
        return;
    default:
        s->regs[off >> 2] = val;
        return;
    }
}

/* MM_INDEX / MM_DATA reach the whole register file through 0x00/0x04 */
static uint64_t ati_hd5770_mmio_read(void *opaque, hwaddr addr, unsigned size)
{
    ATIHD5770State *s = opaque;
    uint32_t base = addr & ~3u, val;

    if (base == EG_MM_DATA) {
        uint32_t idx = s->regs[EG_MM_INDEX >> 2] & 0x7fffffff;

        val = idx >= 8 && idx < ATI_HD5770_MMIO_SIZE ? reg_read(s, idx) : 0;
    } else {
        val = reg_read(s, base);
    }
    return extract64(val, (addr & 3) * 8, size * 8);
}

static void ati_hd5770_mmio_write(void *opaque, hwaddr addr, uint64_t data,
                                  unsigned size)
{
    ATIHD5770State *s = opaque;
    uint32_t base = addr & ~3u, val = data;

    if (size < 4) {
        uint32_t old = base == EG_MM_DATA ? 0 : s->regs[base >> 2];

        val = deposit32(old, (addr & 3) * 8, size * 8, data);
    }
    if (base == EG_MM_DATA) {
        uint32_t idx = s->regs[EG_MM_INDEX >> 2] & 0x7fffffff;

        if (idx >= 8 && idx < ATI_HD5770_MMIO_SIZE) {
            reg_write(s, idx, val);
        }
        return;
    }
    reg_write(s, base, val);
}

static const MemoryRegionOps ati_hd5770_mmio_ops = {
    .read = ati_hd5770_mmio_read,
    .write = ati_hd5770_mmio_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {
        .min_access_size = 1,
        .max_access_size = 4,
    },
    .impl = {
        .min_access_size = 1,
        .max_access_size = 4,
    },
};

/* ---------------------------------------------------------------- */
/* Display                                                          */

static uint32_t *draw_row(DisplaySurface *ds, int y)
{
    return (uint32_t *)((uint8_t *)surface_data(ds) + y * surface_stride(ds));
}

/*
 * CRTC 0's surface into the console. The LUT is a per-channel gamma
 * table in every depth; 8bpp indexes it with the pixel.
 */
static void ati_hd5770_draw(ATIHD5770State *s, DisplaySurface *ds,
                            const ATIHD5770Mode *m)
{
    const uint8_t *src = (const uint8_t *)memory_region_get_ram_ptr(&s->vram) +
                         m->fb_offset;
    const uint8_t (*pal)[3] = (const uint8_t (*)[3])s->palette;
    uint32_t x, y;

    for (y = 0; y < m->height; y++) {
        uint32_t *dst = draw_row(ds, y);

        for (x = 0; x < m->width; x++) {
            uint8_t r, g, b;

            if (m->bpp == 1) {
                r = pal[src[x]][0];
                g = pal[src[x]][1];
                b = pal[src[x]][2];
            } else if (m->bpp == 2) {
                uint16_t p = lduw_le_p(src + 2 * x);

                if (m->format == 1) {           /* RGB565 */
                    r = (p >> 11) << 3;
                    g = ((p >> 5) & 0x3f) << 2;
                } else {                        /* ARGB1555 */
                    r = ((p >> 10) & 0x1f) << 3;
                    g = ((p >> 5) & 0x1f) << 3;
                }
                b = (p & 0x1f) << 3;
                r = pal[r | r >> 5][0];
                g = pal[g | g >> (m->format == 1 ? 6 : 5)][1];
                b = pal[b | b >> 5][2];
            } else if (m->format == 1) {        /* ARGB2101010 */
                uint32_t p = ldl_le_p(src + 4 * x);

                r = pal[(p >> 22) & 0xff][0];
                g = pal[(p >> 12) & 0xff][1];
                b = pal[(p >> 2) & 0xff][2];
            } else {                            /* ARGB8888: B,G,R,A */
                r = pal[src[4 * x + 2]][0];
                g = pal[src[4 * x + 1]][1];
                b = pal[src[4 * x]][2];
            }
            dst[x] = 0xff000000u | ((uint32_t)r << 16) |
                     ((uint32_t)g << 8) | b;
        }
        src += m->pitch;
    }
}

static bool ati_hd5770_update_display(void *opaque)
{
    ATIHD5770State *s = opaque;
    ATIHD5770Mode mode;
    DisplaySurface *ds;
    DirtyBitmapSnapshot *snap;
    bool redraw;
    uint64_t len;

    snap = memory_region_snapshot_and_clear_dirty(&s->vram, 0, s->vram_size,
                                                   DIRTY_MEMORY_VGA);
    if (!ati_hd5770_dce4_scanout(s, &mode)) {
        /* CRTC 0 off or blanked: the last frame stays */
        g_free(snap);
        return true;
    }
    memcpy(s->palette, ati_hd5770_dce4_lut(s), sizeof(s->palette));
    if (s->regs[EG_GRPH_LUT_10BIT_BYPASS_CONTROL >> 2] &
        EG_LUT_10BIT_BYPASS_EN) {
        unsigned i;

        for (i = 0; i < 256; i++) {
            s->palette[i][0] = s->palette[i][1] = s->palette[i][2] = i;
        }
    }
    redraw = s->force_redraw || s->lut_drawn != s->lut_gen;
    s->lut_drawn = s->lut_gen;
    ds = qemu_console_surface(s->con);
    if (!ds || surface_width(ds) != (int)mode.width ||
        surface_height(ds) != (int)mode.height) {
        trace_ati_hd5770_surface(mode.width, mode.height, mode.bpp,
                                 mode.pitch, mode.fb_offset);
        ds = qemu_create_displaysurface(mode.width, mode.height);
        qemu_console_set_surface(s->con, ds);
        redraw = true;
    }
    if (memcmp(&s->mode, &mode, sizeof(mode))) {
        redraw = true;
    }
    len = (uint64_t)mode.pitch * mode.height;
    if (memory_region_snapshot_get_dirty(&s->vram, snap, mode.fb_offset,
                                         len)) {
        redraw = true;
    }
    g_free(snap);
    s->mode = mode;
    if (redraw) {
        s->force_redraw = false;
        ati_hd5770_draw(s, ds, &mode);
        ati_hd5770_dce4_cursor_apply(s);
        qemu_console_update_full(s->con);
    } else {
        ati_hd5770_cursor_update(s);
    }
    return true;
}

static const GraphicHwOps ati_hd5770_gfx_ops = {
    .gfx_update = ati_hd5770_update_display,
};

static void ati_hd5770_cursor_timer(void *opaque)
{
    ati_hd5770_dce4_cursor_apply(opaque);
}

/*
 * A cursor move is a burst of register writes (position, hot spot,
 * surface, control). Apply them once, a moment after the first write,
 * so the burst is seen whole; later writes do not re-arm the timer.
 */
void ati_hd5770_cursor_update(ATIHD5770State *s)
{
    if (!timer_pending(s->cursor_timer)) {
        timer_mod(s->cursor_timer,
                  qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) + SCALE_MS);
    }
}

/* ---------------------------------------------------------------- */
/* Device                                                           */

static void ati_hd5770_reset_hold(Object *obj, ResetType type)
{
    ATIHD5770State *s = ATI_HD5770(obj);
    uint64_t top = s->vram_size - 1;

    memset(s->regs, 0, sizeof(s->regs));
    memset(s->pcie_ind, 0, sizeof(s->pcie_ind));
    memset(s->pcie_port, 0, sizeof(s->pcie_port));
    memset(s->cg_ind, 0, sizeof(s->cg_ind));
    memset(s->smc_sram, 0, sizeof(s->smc_sram));
    /* VRAM at MC address 0 */
    s->regs[EG_MC_VM_FB_LOCATION >> 2] = (uint32_t)(top >> 24) << 16;
    memset(&s->mode, 0, sizeof(s->mode));
    s->force_redraw = true;
    if (s->hw_cursor_on) {
        s->hw_cursor_on = false;
        qemu_console_set_mouse(s->con, 0, 0, false);
    }
    s->hw_cursor_sum = 0;
    s->ih_level = false;
    ati_hd5770_dce4_reset(s);
    ati_hd5770_update_irq(s);
}

static void ati_hd5770_realize(PCIDevice *dev, Error **errp)
{
    ATIHD5770State *s = ATI_HD5770(dev);
    Object *obj = OBJECT(dev);
    uint8_t bar_type;

    if (s->device_id > 0xffff || s->subsys_vendor_id > 0xffff ||
        s->subsys_id > 0xffff || s->revision > 0xff) {
        error_setg(errp, "device-id, subsys-vendor-id and subsys-id are "
                   "16 bit, revision 8 bit");
        return;
    }
    if (s->vram_mb < 64 || s->vram_mb > 2048 || !is_power_of_2(s->vram_mb) ||
        s->aper_mb < 64 || s->aper_mb > s->vram_mb ||
        !is_power_of_2(s->aper_mb)) {
        error_setg(errp, "vram-mb (64-2048) and aper-mb (64 up to vram-mb) "
                   "are powers of two");
        return;
    }
    if (s->mon_ddc < 1 || s->mon_ddc > ATI_HD5770_NUM_DDC ||
        s->mon_hpd < 1 || s->mon_hpd > ATI_HD5770_NUM_HPD) {
        error_setg(errp, "mon-ddc and mon-hpd are 1-6");
        return;
    }
    s->vram_size = (uint64_t)s->vram_mb * MiB;
    pci_set_word(dev->config + PCI_DEVICE_ID, s->device_id);
    pci_set_word(dev->config + PCI_SUBSYSTEM_VENDOR_ID, s->subsys_vendor_id);
    pci_set_word(dev->config + PCI_SUBSYSTEM_ID, s->subsys_id);
    pci_set_byte(dev->config + PCI_REVISION_ID, s->revision);

    s->con = qemu_graphic_console_create(DEVICE(dev), 0, &ati_hd5770_gfx_ops,
                                         s);
    memory_region_init_ram(&s->vram, obj, "ati-hd5770-vram", s->vram_size,
                           &error_fatal);
    memory_region_set_log(&s->vram, true, DIRTY_MEMORY_VGA);
    /* BAR 0 shows the first aper-mb of VRAM */
    memory_region_init_alias(&s->aper, obj, "ati-hd5770-aper", &s->vram, 0,
                             (uint64_t)s->aper_mb * MiB);
    memory_region_init_io(&s->mmio, obj, &ati_hd5770_mmio_ops, s,
                          "ati-hd5770-mmio", ATI_HD5770_MMIO_SIZE);
    memory_region_init_io(&s->io, obj, &ati_hd5770_mmio_ops, s,
                          "ati-hd5770-io", ATI_HD5770_IO_SIZE);

    /*
     * BAR 0 VRAM aperture (prefetchable), BAR 2 registers, BAR 4 I/O, as
     * on the card. 32-bit BARs by default: the 32-bit kernels map them
     * below 4 GiB.
     */
    bar_type = PCI_BASE_ADDRESS_SPACE_MEMORY |
               (s->bar64 ? PCI_BASE_ADDRESS_MEM_TYPE_64 : 0);
    pci_register_bar(dev, 0, bar_type | PCI_BASE_ADDRESS_MEM_PREFETCH,
                     &s->aper);
    pci_register_bar(dev, 2, bar_type, &s->mmio);
    pci_register_bar(dev, 4, PCI_BASE_ADDRESS_SPACE_IO, &s->io);
    pci_config_set_interrupt_pin(dev->config, 1);

    if (pci_bus_is_express(pci_get_bus(dev))) {
        if (pcie_endpoint_cap_init(dev, 0x80) < 0) {
            error_setg(errp, "cannot add the PCI Express capability");
            return;
        }
    } else {
        dev->cap_present &= ~QEMU_PCI_CAP_EXPRESS;
    }
    if (s->msi) {
        Error *err = NULL;

        if (msi_init(dev, 0x50, 1, true, false, &err)) {
            error_propagate(errp, err);
            return;
        }
    }

    s->cursor_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL,
                                   ati_hd5770_cursor_timer, s);
    ati_hd5770_dce4_init(s);
}

static void ati_hd5770_exit(PCIDevice *dev)
{
    ATIHD5770State *s = ATI_HD5770(dev);

    ati_hd5770_dce4_fini(s);
    timer_free(s->cursor_timer);
    s->cursor_timer = NULL;
    msi_uninit(dev);
    qemu_graphic_console_close(s->con);
}

static int ati_hd5770_post_load(void *opaque, int version_id)
{
    ATIHD5770State *s = opaque;

    s->force_redraw = true;
    s->lut_gen++;
    ati_hd5770_update_irq(s);
    return 0;
}

static const VMStateDescription vmstate_ati_hd5770 = {
    .name = "ati-radeon-hd5770",
    .version_id = 1,
    .minimum_version_id = 1,
    .post_load = ati_hd5770_post_load,
    .fields = (const VMStateField[]) {
        VMSTATE_PCI_DEVICE(parent_obj, ATIHD5770State),
        VMSTATE_UINT32_ARRAY(regs, ATIHD5770State, ATI_HD5770_NUM_REGS),
        VMSTATE_UINT32_ARRAY(pcie_ind, ATIHD5770State, 256),
        VMSTATE_UINT32_ARRAY(pcie_port, ATIHD5770State, 256),
        VMSTATE_UINT32_ARRAY(cg_ind, ATIHD5770State, 256),
        VMSTATE_UINT32_ARRAY(smc_sram, ATIHD5770State,
                             ATI_HD5770_SMC_SRAM_SIZE / 4),
        VMSTATE_BUFFER_UNSAFE(lut, ATIHD5770State, 0,
                              ATI_HD5770_NUM_CRTC * 256 * 3),
        VMSTATE_BOOL_ARRAY(vblank_occurred, ATIHD5770State,
                           ATI_HD5770_NUM_CRTC),
        VMSTATE_BOOL_ARRAY(vline_occurred, ATIHD5770State,
                           ATI_HD5770_NUM_CRTC),
        VMSTATE_BOOL_ARRAY(hpd_pending, ATIHD5770State, ATI_HD5770_NUM_HPD),
        VMSTATE_BOOL_ARRAY(hpd_cond, ATIHD5770State, ATI_HD5770_NUM_HPD),
        VMSTATE_UINT32_ARRAY(pixclk_10khz, ATIHD5770State,
                             ATI_HD5770_NUM_CRTC),
        VMSTATE_UINT32(frame_count, ATIHD5770State),
        VMSTATE_UINT8_ARRAY(i2c_buf, ATIHD5770State, 256),
        VMSTATE_UINT8(i2c_index, ATIHD5770State),
        VMSTATE_UINT32(i2c_status, ATIHD5770State),
        VMSTATE_END_OF_LIST()
    }
};

static const Property ati_hd5770_properties[] = {
    DEFINE_PROP_UINT32("vram-mb", ATIHD5770State, vram_mb, 1024),
    DEFINE_PROP_UINT32("aper-mb", ATIHD5770State, aper_mb, 256),
    DEFINE_PROP_BOOL("bar64", ATIHD5770State, bar64, false),
    DEFINE_PROP_BOOL("msi", ATIHD5770State, msi, true),
    DEFINE_PROP_UINT32("device-id", ATIHD5770State, device_id,
                       PCI_DEVICE_ID_ATI_JUNIPER_XT),
    DEFINE_PROP_UINT32("subsys-vendor-id", ATIHD5770State, subsys_vendor_id,
                       ATI_HD5770_SUBSYS_VENDOR),
    DEFINE_PROP_UINT32("subsys-id", ATIHD5770State, subsys_id,
                       ATI_HD5770_SUBSYS_ID),
    DEFINE_PROP_UINT32("revision", ATIHD5770State, revision, 0),
    /* the DVI port of the Mac Pro card (ATY,Langur): DDC3, HPD5 */
    DEFINE_PROP_UINT32("mon-ddc", ATIHD5770State, mon_ddc, 3),
    DEFINE_PROP_UINT32("mon-hpd", ATIHD5770State, mon_hpd, 5),
    DEFINE_EDID_PROPERTIES(ATIHD5770State, edid_info),
};

/*
 * `gaps` property: registers outside the model's map the guest touched,
 * with read and write counts.
 */
static char *ati_hd5770_get_gaps(Object *obj, Error **errp)
{
    ATIHD5770State *s = ATI_HD5770(obj);
    GString *out = g_string_new(NULL);
    unsigned i;

    for (i = 0; i < ATI_HD5770_NUM_REGS; i++) {
        if (s->unk_rd[i] || s->unk_wr[i]) {
            g_string_append_printf(out, "%sreg 0x%05x: %u reads, %u writes, "
                                   "value 0x%08x", out->len ? "\n" : "",
                                   i << 2, s->unk_rd[i], s->unk_wr[i],
                                   s->regs[i]);
        }
    }
    if (!out->len) {
        g_string_append(out, "none");
    }
    return g_string_free(out, FALSE);
}

/* `atom-calls` property: AtomBIOS command tables run, with counts */
static char *ati_hd5770_get_atom_calls(Object *obj, Error **errp)
{
    ATIHD5770State *s = ATI_HD5770(obj);
    GString *out = g_string_new(NULL);
    unsigned i;

    for (i = 0; i < ARRAY_SIZE(s->atom_calls); i++) {
        if (s->atom_calls[i]) {
            g_string_append_printf(out, "%stable %u: %u",
                                   out->len ? "\n" : "", i, s->atom_calls[i]);
        }
    }
    if (!out->len) {
        g_string_append(out, "none");
    }
    return g_string_free(out, FALSE);
}

/* `scanout` property: what the last refresh drew and its registers */
static char *ati_hd5770_get_scanout(Object *obj, Error **errp)
{
    ATIHD5770State *s = ATI_HD5770(obj);
    GString *out = g_string_new(NULL);
    unsigned c;

    g_string_append_printf(out,
        "drawn: %ux%u bpp=%u fmt=%u pitch=%u fb_offset=0x%" PRIx64 "\n"
        "MC_VM_FB_LOCATION=0x%08x frame_period_ns=%" PRId64 " frames=%u\n"
        "IH_RB_CNTL=0x%08x BASE=0x%08x RPTR=0x%x WPTR=0x%x IH_CNTL=0x%08x "
        "msi=%d\n",
        s->mode.width, s->mode.height, s->mode.bpp, s->mode.format,
        s->mode.pitch, s->mode.fb_offset,
        s->regs[EG_MC_VM_FB_LOCATION >> 2], s->frame_period_ns,
        s->frame_count, s->regs[EG_IH_RB_CNTL >> 2],
        s->regs[EG_IH_RB_BASE >> 2], s->regs[EG_IH_RB_RPTR >> 2],
        s->regs[EG_IH_RB_WPTR >> 2], s->regs[EG_IH_CNTL >> 2],
        msi_enabled(PCI_DEVICE(s)));
    for (c = 0; c < ATI_HD5770_NUM_CRTC; c++) {
        uint32_t o = ati_hd5770_crtc_offset[c];
#define R(x) s->regs[((x) + o) >> 2]
        g_string_append_printf(out,
            "CRTC%u: CONTROL=0x%08x H_TOTAL=%u V_TOTAL=%u GRPH_ENABLE=%u "
            "GRPH_CONTROL=0x%08x SURFACE=0x%08x:%08x PITCH=%u X %u..%u "
            "Y %u..%u VIEWPORT=0x%08x CUR_CONTROL=0x%08x CUR_POSITION=0x%08x "
            "INT_MASK=0x%x pixclk=%u x 10 kHz\n",
            c, R(EG_CRTC_CONTROL), R(EG_CRTC_H_TOTAL) + 1,
            R(EG_CRTC_V_TOTAL) + 1, R(EG_GRPH_ENABLE), R(EG_GRPH_CONTROL),
            R(EG_GRPH_PRIMARY_SURFACE_ADDRESS_HIGH),
            R(EG_GRPH_PRIMARY_SURFACE_ADDRESS), R(EG_GRPH_PITCH),
            R(EG_GRPH_X_START), R(EG_GRPH_X_END), R(EG_GRPH_Y_START),
            R(EG_GRPH_Y_END), R(EG_VIEWPORT_SIZE), R(EG_CUR_CONTROL),
            R(EG_CUR_POSITION), R(EG_INT_MASK), s->pixclk_10khz[c]);
#undef R
    }
    return g_string_free(out, FALSE);
}

/* `edid` property: the monitor's EDID, hex */
static char *ati_hd5770_get_edid(Object *obj, Error **errp)
{
    ATIHD5770State *s = ATI_HD5770(obj);
    GString *out = g_string_new(NULL);
    unsigned i;

    for (i = 0; s->edid && i < ATI_R580_EDID_SIZE; i++) {
        g_string_append_printf(out, "%02x", s->edid->blob[i]);
    }
    return g_string_free(out, FALSE);
}

/* `palette` property: CRTC 0's gamma LUT as the scanout reads it */
static char *ati_hd5770_get_palette(Object *obj, Error **errp)
{
    ATIHD5770State *s = ATI_HD5770(obj);
    GString *out = g_string_new(NULL);
    unsigned i, n = 0;

    for (i = 0; i < 256; i++) {
        if (s->palette[i][0] != i || s->palette[i][1] != i ||
            s->palette[i][2] != i) {
            n++;
        }
    }
    g_string_append_printf(out, "%u of 256 entries differ from the identity",
                           n);
    for (i = 0; i < 256; i += 16) {
        g_string_append_printf(out, "\n[%3u] %3u %3u %3u", i,
                               s->palette[i][0], s->palette[i][1],
                               s->palette[i][2]);
    }
    return g_string_free(out, FALSE);
}

static void ati_hd5770_instance_init(Object *obj)
{
    PCI_DEVICE(obj)->cap_present |= QEMU_PCI_CAP_EXPRESS;
}

static void ati_hd5770_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    ResettableClass *rc = RESETTABLE_CLASS(klass);
    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);

    object_class_property_add_str(klass, "gaps", ati_hd5770_get_gaps, NULL);
    object_class_property_add_str(klass, "atom-calls",
                                  ati_hd5770_get_atom_calls, NULL);
    object_class_property_add_str(klass, "scanout", ati_hd5770_get_scanout,
                                  NULL);
    object_class_property_add_str(klass, "edid", ati_hd5770_get_edid, NULL);
    object_class_property_add_str(klass, "palette", ati_hd5770_get_palette,
                                  NULL);

    k->class_id  = PCI_CLASS_DISPLAY_VGA;
    k->vendor_id = PCI_VENDOR_ID_ATI;
    k->device_id = PCI_DEVICE_ID_ATI_JUNIPER_XT;
    k->subsystem_vendor_id = ATI_HD5770_SUBSYS_VENDOR;
    k->subsystem_id = ATI_HD5770_SUBSYS_ID;
    k->revision  = 0x00;
    /* the synthetic AtomBIOS; romfile= replaces it, e.g. with a GOP added */
    k->romfile   = "ati-hd5770-atombios.rom";
    k->realize   = ati_hd5770_realize;
    k->exit      = ati_hd5770_exit;
    dc->desc     = "ATI Radeon HD 5770 (Juniper)";
    dc->vmsd     = &vmstate_ati_hd5770;
    device_class_set_props(dc, ati_hd5770_properties);
    rc->phases.hold = ati_hd5770_reset_hold;
    set_bit(DEVICE_CATEGORY_DISPLAY, dc->categories);
}

static const TypeInfo ati_hd5770_type_info = {
    .name           = TYPE_ATI_HD5770,
    .parent         = TYPE_PCI_DEVICE,
    .instance_size  = sizeof(ATIHD5770State),
    .instance_init  = ati_hd5770_instance_init,
    .class_init     = ati_hd5770_class_init,
    .interfaces     = (const InterfaceInfo[]) {
        { INTERFACE_PCIE_DEVICE },
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { },
    },
};

static void ati_hd5770_register_types(void)
{
    type_register_static(&ati_hd5770_type_info);
}

type_init(ati_hd5770_register_types)
