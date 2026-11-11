/*
 * ATI R580 (Radeon X1900 XT): R5xx bus front end and AVIVO display
 * controller registers.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Register offsets and bits from rv530-qemu 0b06c9f
 * (qemu/hw/display/rv530/rv530.h).
 */

#ifndef ATI_R580_AVIVO_H
#define ATI_R580_AVIVO_H

/* R5xx PCI identity */
#define PCI_DEVICE_ID_ATI_R580          0x7249
#define ATI_R580_SUBSYS_VENDOR          0x106b
#define ATI_R580_SUBSYS_ID              0x0000

/* bus interface: indirect register spaces */
#define R5XX_PCIE_INDEX                 0x0030
#define R5XX_PCIE_DATA                  0x0034
#define R5XX_MC_IND_INDEX               0x0070
#define R5XX_MC_IND_DATA                0x0074
#define R5XX_MC_IND_ADDR_MASK           0xff
#define R5XX_MC_IND_WR_EN               ((1u << 23) | (1u << 24))
#define R5XX_HDP_FB_LOCATION            0x0134
#define R5XX_MC_STATUS_LEGACY           0x0150
#define R5XX_MSI_REARM_EN               0x0160
#define R5XX_P1PLL_CNTL                 0x0450
#define R5XX_P2PLL_CNTL                 0x0454
#define R5XX_PPLL_LOCKED                (0x30u << 16)
#define R5XX_EXT1_PPLL_REF_DIV          0x0404
#define R5XX_EXT1_PPLL_FB_DIV           0x0430
#define R5XX_EXT1_PPLL_POST_DIV         0x043c
#define R5XX_VGA_RENDER_CONTROL         0x0300
#define R5XX_D1VGA_CONTROL              0x0330
#define R5XX_D2VGA_CONTROL              0x0338

/* PLL index space */
#define R5XX_PLL_ADDR_MASK              0x3f
#define R5XX_PLL_WR_EN                  (1u << 7)
#define R5XX_PLL_SPLL_FUNC_CNTL         0x00
#define R5XX_PLL_MPLL_FUNC_CNTL         0x04
#define R5XX_PLL_CHIP_ID                0x34

/* MC index space (R520 map) */
#define R5XX_MC_STATUS                  0x00
#define R5XX_MC_STATUS_IDLE             (1u << 1)
#define R5XX_MC_STATUS_IDLE_RV515       (1u << 4)
#define R5XX_MC_FB_LOCATION             0x04
#define R5XX_MC_AGP_LOCATION            0x05
#define R5XX_MC_AGP_BASE                0x06
#define R5XX_MC_AGP_BASE_2              0x07
#define R5XX_MC_CNTL0                   0x08
#define R5XX_MC_CNTL0_256BIT            (3u << 24)

/* PCIE index space */
#define R5XX_PCIE_TX_GART_CNTL          0x10
#define R5XX_PCIE_TX_GART_INVALIDATE    (1u << 8)
#define R5XX_PCIE_TX_GART_ERROR         0x18
#define R5XX_PCIE_TX_GART_EN            (1u << 0)
#define R5XX_PCIE_TX_GART_BASE          0x13
#define R5XX_PCIE_TX_GART_START_LO      0x14
#define R5XX_PCIE_TX_GART_END_LO        0x16

/* GEN_INT_STATUS: display interrupt summary */
#define R5XX_GEN_INT_DISPLAY_STAT       (1u << 0)

/* AVIVO display controller 1; D2 = D1 + R5XX_CRTC_STRIDE */
#define R5XX_CRTC_STRIDE                0x0800
#define R5XX_D1CRTC_H_TOTAL             0x6000
#define R5XX_D1CRTC_H_BLANK_START_END   0x6004
#define R5XX_D1CRTC_H_SYNC_A            0x6008
#define R5XX_D1CRTC_H_SYNC_A_CNTL       0x600c
#define R5XX_D1CRTC_V_TOTAL             0x6020
#define R5XX_D1CRTC_V_BLANK_START_END   0x6024
#define R5XX_D1CRTC_V_SYNC_A            0x6028
#define R5XX_D1CRTC_V_SYNC_A_CNTL       0x602c
#define R5XX_D1CRTC_CONTROL             0x6080
#define R5XX_D1CRTC_BLANK_CONTROL       0x6084
#define R5XX_D1CRTC_INTERLACE_CONTROL   0x6088
#define R5XX_D1CRTC_STATUS              0x609c
#define R5XX_D1CRTC_STATUS_POSITION     0x60a0
#define R5XX_D1CRTC_FRAME_COUNT         0x60a4
#define R5XX_D1CRTC_STATUS_HV_COUNT     0x60ac
#define R5XX_D1MODE_MASTER_UPDATE_LOCK  0x60e0
#define R5XX_D1MODE_MASTER_UPDATE_MODE  0x60e4
#define R5XX_D1CRTC_UPDATE_LOCK         0x60e8
#define R5XX_DC_CRTC_MASTER_EN          0x60f8
#define R5XX_D1GRPH_ENABLE              0x6100
#define R5XX_D1GRPH_CONTROL             0x6104
#define R5XX_D1GRPH_LUT_SEL             0x6108
#define R5XX_D1GRPH_PRIMARY_SURFACE_ADDRESS 0x6110
#define R5XX_D1GRPH_SECONDARY_SURFACE_ADDRESS 0x6118
#define R5XX_D1GRPH_PITCH               0x6120
#define R5XX_D1GRPH_SURFACE_OFFSET_X    0x6124
#define R5XX_D1GRPH_SURFACE_OFFSET_Y    0x6128
#define R5XX_D1GRPH_X_START             0x612c
#define R5XX_D1GRPH_Y_START             0x6130
#define R5XX_D1GRPH_X_END               0x6134
#define R5XX_D1GRPH_Y_END               0x6138
#define R5XX_D1GRPH_UPDATE              0x6144
#define R5XX_D1GRPH_FLIP_CONTROL        0x6148
#define R5XX_D1CUR_CONTROL              0x6400
#define R5XX_D1CUR_SURFACE_ADDRESS      0x6408
#define R5XX_D1CUR_SIZE                 0x6410
#define R5XX_D1CUR_POSITION             0x6414
#define R5XX_D1CUR_HOT_SPOT             0x6418
#define R5XX_D1CUR_UPDATE               0x6424
#define R5XX_DC_LUT_RW_SELECT           0x6480
#define R5XX_DC_LUT_RW_MODE             0x6484
#define R5XX_DC_LUT_RW_INDEX            0x6488
#define R5XX_DC_LUT_SEQ_COLOR           0x648c
#define R5XX_DC_LUT_PWL_DATA            0x6490
#define R5XX_DC_LUT_30_COLOR            0x6494
#define R5XX_DC_LUT_WRITE_EN_MASK       0x649c
#define R5XX_DC_LUT_AUTOFILL            0x64a0
#define R5XX_DC_LUT_AUTOFILL_DONE       (1u << 1)
#define R5XX_DC_LB_MEMORY_SPLIT         0x6520
#define R5XX_D1MODE_DATA_FORMAT         0x6528
#define R5XX_D1MODE_DESKTOP_HEIGHT      0x652c
#define R5XX_D1MODE_VBLANK_STATUS       0x6534
#define R5XX_D1MODE_VLINE_START_END     0x6538
#define R5XX_D1MODE_VLINE_STATUS        0x653c
#define R5XX_DxMODE_INT_MASK            0x6540
#define R5XX_D1MODE_VIEWPORT_START      0x6580
#define R5XX_D1MODE_VIEWPORT_SIZE       0x6584
#define R5XX_DACA_ENABLE                0x7800
#define R5XX_DACA_COMPARATOR_OUTPUT     0x7840
#define R5XX_TMDSA_CNTL                 0x7880
#define R5XX_DACB_COMPARATOR_OUTPUT     0x7a40
#define R5XX_DC_HOT_PLUG_DETECT1_CONTROL     0x7d00
#define R5XX_DC_HOT_PLUG_DETECT1_INT_STATUS  0x7d04
#define R5XX_DC_HOT_PLUG_DETECT1_INT_CONTROL 0x7d08
#define R5XX_DC_HOT_PLUG_DETECT2_INT_STATUS  0x7d14
#define R5XX_DC_GPIO_DDC1_MASK          0x7e40
#define R5XX_DC_GPIO_DDC2_MASK          0x7e50
#define R5XX_DC_GPIO_DDC3_MASK          0x7e60
#define R5XX_DISP_INTERRUPT_STATUS      0x7edc

/* bits */
#define R5XX_CRTC_EN                    (1u << 0)
#define R5XX_CRTC_DISP_READ_REQUEST_DISABLE (1u << 24)
#define R5XX_CRTC_CURRENT_MASTER_EN_STATE (1u << 16)
#define R5XX_CRTC_BLANK_DATA_EN         (1u << 8)
#define R5XX_D1CRTC_V_BLANK             (1u << 0)
#define R5XX_VBLANK_OCCURRED            (1u << 0)
#define R5XX_VBLANK_ACK                 (1u << 4)
#define R5XX_VBLANK_STAT                (1u << 12)
#define R5XX_VBLANK_INTERRUPT           (1u << 16)
#define R5XX_D1MODE_VBLANK_INT_MASK     (1u << 0)
#define R5XX_D2MODE_VBLANK_INT_MASK     (1u << 8)
#define R5XX_DISP_LB_D1_VBLANK_INT      (1u << 4)
#define R5XX_DISP_LB_D2_VBLANK_INT      (1u << 5)
#define R5XX_DISP_HPD1_INT              (1u << 18)
#define R5XX_HPD_INT_EN                 (1u << 16)
#define R5XX_HPD_INT_ACK                (1u << 0)
#define R5XX_HPD_INT_STATUS             (1u << 0)
#define R5XX_HPD_SENSE                  (1u << 1)
#define R5XX_D1CURSOR_EN                (1u << 0)
#define R5XX_D1CUR_UPDATE_LOCK          (1u << 16)
#define R5XX_D1GRPH_SURFACE_UPDATE_PENDING (1u << 2)
#define R5XX_D1GRPH_LUT_SEL_B           (1u << 0)
#define R5XX_D1GRPH_TILED               (1u << 20)
#define R5XX_D1GRPH_MACRO               (1u << 21)

/* AtomBIOS mailbox of the synthetic VBIOS (ati_r580_atombios.py) */
#define R5XX_ATOM_MAILBOX               0xff00
#define R5XX_ATOM_MAILBOX_DWORDS        16
#define R5XX_ATOM_DOORBELL              0xff7c

#define ATI_R580_NUM_CRTC               2
#define ATI_R580_NUM_DDC                3

/* registers served on the BQL, never by an engine claim */
static inline bool ati_r580_display_reg(uint32_t base)
{
    return (base >= 0x6000 && base < 0x8000) ||
           (base >= R5XX_ATOM_MAILBOX && base <= R5XX_ATOM_DOORBELL);
}

#endif
