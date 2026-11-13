/*
 * QEMU ATI Radeon HD 5770 (Juniper) emulation: internal interface.
 *
 * Milestone 1 of the card: PCI identity, synthetic AtomBIOS + UEFI GOP
 * option ROM, the memory controller's FB window and HDP, the interrupt
 * handler ring, and the DCE 4 display engine (six CRTCs, graphics
 * surface, cursor, LUT, hot plug, DDC). No command processor, GART or 3D:
 * their status registers read idle.
 *
 * Everything runs on the BQL.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef ATI_HD5770_INT_H
#define ATI_HD5770_INT_H

#include "qemu/units.h"
#include "hw/pci/pci_device.h"
#include "hw/i2c/bitbang_i2c.h"
#include "ui/console.h"
#include "hw/display/edid.h"
#include "qemu/timer.h"
#include "qom/object.h"
#include "ati_r580_edid.h"

#define TYPE_ATI_HD5770 "ati-radeon-hd5770"
OBJECT_DECLARE_SIMPLE_TYPE(ATIHD5770State, ATI_HD5770)

#define PCI_VENDOR_ID_ATI               0x1002
#define PCI_DEVICE_ID_ATI_JUNIPER_XT    0x68b8
#define ATI_HD5770_SUBSYS_VENDOR        0x106b
#define ATI_HD5770_SUBSYS_ID            0x0000

#define ATI_HD5770_MMIO_SIZE            (128 * KiB)
#define ATI_HD5770_NUM_REGS             (ATI_HD5770_MMIO_SIZE / 4)
#define ATI_HD5770_IO_SIZE              256
#define ATI_HD5770_ROM_BAR_SIZE         (128 * KiB)

#define ATI_HD5770_SMC_SRAM_SIZE        (64 * KiB)

#define ATI_HD5770_NUM_CRTC             6
#define ATI_HD5770_NUM_DDC              6
#define ATI_HD5770_NUM_HPD              6

/* AtomBIOS mailbox of the synthetic VBIOS (ati_hd5770_atombios.py) */
#define HD5770_ATOM_MAILBOX             0x1ff00
#define HD5770_ATOM_MAILBOX_DWORDS      16
#define HD5770_ATOM_DOORBELL            0x1ff7c

/* IH source ids (Evergreen) */
#define HD5770_IH_SRC_D1                1   /* D1..D6 = 1..6, data 0 vblank 1 vline */
#define HD5770_IH_SRC_HPD               42  /* data = HPD pin - 1 */

typedef struct ATIHD5770Mode {
    uint32_t width, height, bpp, pitch;
    uint32_t depth, format;
    uint64_t fb_offset;
} ATIHD5770Mode;

struct ATIHD5770State {
    PCIDevice parent_obj;

    MemoryRegion vram;
    MemoryRegion aper;
    MemoryRegion mmio;
    MemoryRegion io;
    QemuConsole *con;

    /* properties */
    uint32_t vram_mb;
    uint32_t aper_mb;
    bool bar64;
    bool msi;
    uint32_t device_id;
    uint32_t subsys_vendor_id;
    uint32_t subsys_id;
    uint32_t revision;
    uint32_t mon_ddc;           /* DDC line of the monitor, 1-6 */
    uint32_t mon_hpd;           /* its hot plug pin, 1-6 */
    qemu_edid_info edid_info;

    uint64_t vram_size;
    uint32_t regs[ATI_HD5770_NUM_REGS];
    uint32_t pcie_ind[256];
    uint32_t pcie_port[256];
    uint32_t cg_ind[256];

    /* SMC: its SRAM, loaded and read by the driver; no firmware runs */
    uint32_t smc_sram[ATI_HD5770_SMC_SRAM_SIZE / 4];

    /* display timing */
    QEMUTimer *vblank_timer;
    int64_t frame_start_ns;
    int64_t frame_period_ns;
    uint32_t frame_count;
    uint32_t pixclk_10khz[ATI_HD5770_NUM_CRTC];
    bool vblank_occurred[ATI_HD5770_NUM_CRTC];
    bool vline_occurred[ATI_HD5770_NUM_CRTC];
    bool hpd_pending[ATI_HD5770_NUM_HPD];
    bool hpd_cond[ATI_HD5770_NUM_HPD];

    /* gamma LUT per CRTC, 8 bits per channel */
    uint8_t lut[ATI_HD5770_NUM_CRTC][256][3];
    uint8_t lut_seq[ATI_HD5770_NUM_CRTC];
    uint32_t lut_gen;
    uint32_t lut_drawn;
    uint8_t palette[256][3];

    /* scanout of CRTC 0 */
    ATIHD5770Mode mode;
    bool force_redraw;
    uint32_t xres, yres;

    /* hardware cursor of CRTC 0 */
    QEMUTimer *cursor_timer;
    bool hw_cursor_on;
    uint32_t hw_cursor_sum;
    int hw_cursor_x, hw_cursor_y;

    /* DDC: GPIO bit-bang and the DC_I2C engine share each line */
    I2CBus *ddc_bus[ATI_HD5770_NUM_DDC];
    bitbang_i2c_interface ddc_bb[ATI_HD5770_NUM_DDC];
    uint32_t ddc_y[ATI_HD5770_NUM_DDC];
    ATIR580EdidState *edid;
    uint8_t i2c_buf[256];
    uint8_t i2c_index;
    uint32_t i2c_status;

    /* interrupt handler ring: pending interrupt as last signalled */
    bool ih_level;

    /* unknown register accesses: count per dword, reads and writes */
    uint32_t unk_rd[ATI_HD5770_NUM_REGS];
    uint32_t unk_wr[ATI_HD5770_NUM_REGS];
    uint32_t atom_calls[128];
};

/* ati_hd5770.c */
void ati_hd5770_ih_push(ATIHD5770State *s, uint32_t src_id, uint32_t src_data);
void ati_hd5770_update_irq(ATIHD5770State *s);
bool ati_hd5770_mc_to_vram(ATIHD5770State *s, uint64_t mc, uint64_t *off);
void ati_hd5770_cursor_update(ATIHD5770State *s);

/* ati_hd5770_dce4.c */
extern const uint32_t ati_hd5770_crtc_offset[ATI_HD5770_NUM_CRTC];
void ati_hd5770_dce4_init(ATIHD5770State *s);
void ati_hd5770_dce4_fini(ATIHD5770State *s);
void ati_hd5770_dce4_reset(ATIHD5770State *s);
bool ati_hd5770_dce4_read(ATIHD5770State *s, uint32_t off, uint32_t *val);
bool ati_hd5770_dce4_write(ATIHD5770State *s, uint32_t off, uint32_t val);
bool ati_hd5770_dce4_scanout(ATIHD5770State *s, ATIHD5770Mode *m);
void ati_hd5770_dce4_cursor_apply(ATIHD5770State *s);
const uint8_t (*ati_hd5770_dce4_lut(ATIHD5770State *s))[3];
void ati_hd5770_dce4_set_pixclk(ATIHD5770State *s, unsigned crtc,
                                uint32_t clk_10khz);

/* ati_hd5770_atom.c */
bool ati_hd5770_atom_write(ATIHD5770State *s, uint32_t off, uint32_t val);

#endif
