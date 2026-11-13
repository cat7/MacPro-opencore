/*
 * ATI Radeon HD 5770 (Juniper, Evergreen, DCE 4) register offsets.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Names and offsets from Linux drivers/gpu/drm/radeon evergreen_reg.h and
 * evergreend.h.
 */

#ifndef ATI_HD5770_REGS_H
#define ATI_HD5770_REGS_H

/* bus interface */
#define EG_MM_INDEX                     0x0000
#define EG_MM_DATA                      0x0004
#define EG_MM_INDEX_HI                  0x0018
#define EG_PCIE_INDEX                   0x0030
#define EG_PCIE_DATA                    0x0034
#define EG_PCIE_PORT_INDEX              0x0038
#define EG_PCIE_PORT_DATA               0x003c
#define EG_ROM_INDEX                    0x00a8
#define EG_ROM_DATA                     0x00ac

/* SMC (power management microcontroller) */
#define EG_SMC_SRAM_ADDR                0x0200
#define EG_SMC_SRAM_AUTO_INC_DIS        (1u << 16)
#define EG_SMC_SRAM_DATA                0x0204
#define EG_SMC_IO                       0x0208
#define EG_SMC_RST_N                    (1u << 0)
#define EG_SMC_STOP_MODE                (1u << 2)
#define EG_SMC_CLK_EN                   (1u << 11)
#define EG_SMC_MSG                      0x020c
#define EG_SMC_FIRST                    0x0200
#define EG_SMC_LAST                     0x02ff

/* clocks and power */
#define EG_PPLL_FIRST                   0x0400
#define EG_PPLL_LAST                    0x04ff
#define EG_CG_SPLL_STATUS               0x060c
#define EG_SPLL_CHG_STATUS              (1u << 1)
#define EG_GENERAL_PWRMGT               0x063c
#define EG_TARGET_AND_CURRENT_PROFILE_INDEX 0x066c
#define EG_CG_MULT_THERMAL_STATUS       0x0740
#define EG_CG_IND_ADDR                  0x08f8
#define EG_CG_IND_DATA                  0x08fc

/* system blocks */
#define EG_SRBM_STATUS                  0x0e50
#define EG_SRBM_SOFT_RESET              0x0e60
#define EG_VM_L2_FIRST                  0x1400
#define EG_VM_L2_LAST                   0x15ff
#define EG_GPIOPAD_FIRST                0x1780
#define EG_GPIOPAD_LAST                 0x17ff
#define EG_MC_VM_FB_LOCATION            0x2024
#define EG_MC_FIRST                     0x2000
#define EG_MC_LAST                      0x2fff
#define EG_MC_SEQ_STATUS_M              0x29f4
#define EG_PMG_PWRSTATE                 (1u << 16)
#define EG_MC_SEQ_CG                    0x2a68
#define EG_MC_CG_SEQ_DRAMCONF_S0        0x05
#define EG_MC_CG_SEQ_DRAMCONF_S1        0x06
#define EG_HDP_HOST_PATH_CNTL           0x2c00
#define EG_HDP_NONSURFACE_BASE          0x2c04
#define EG_HDP_NONSURFACE_INFO          0x2c08
#define EG_IH_RB_CNTL                   0x3e00
#define EG_IH_RB_ENABLE                 (1u << 0)
#define EG_IH_WPTR_WRITEBACK_ENABLE     (1u << 8)
#define EG_IH_WPTR_OVERFLOW_ENABLE      (1u << 16)
#define EG_IH_WPTR_OVERFLOW_CLEAR       (1u << 31)
#define EG_IH_RB_BASE                   0x3e04
#define EG_IH_RB_RPTR                   0x3e08
#define EG_IH_RB_WPTR                   0x3e0c
#define EG_IH_RB_WPTR_OVERFLOW          (1u << 0)
#define EG_IH_RB_WPTR_ADDR_HI           0x3e10
#define EG_IH_RB_WPTR_ADDR_LO           0x3e14
#define EG_IH_CNTL                      0x3e18
#define EG_IH_ENABLE_INTR               (1u << 0)
#define EG_IH_FIRST                     0x3e00
#define EG_IH_LAST                      0x3eff
#define EG_CONFIG_MEMSIZE               0x5428
#define EG_CONFIG_APER_SIZE             0x5430
#define EG_CONFIG_FB_OFFSET             0x5434
#define EG_HDP_MEM_COHERENCY_FLUSH_CNTL 0x5480
#define EG_CHIP_ID                      0x5564
#define EG_AUDIO_FIRST                  0x5e00
#define EG_AUDIO_LAST                   0x5fff
#define EG_GRBM_STATUS                  0x8010
#define EG_GRBM_STATUS_SE0              0x8014
#define EG_GRBM_STATUS_SE1              0x8018
#define EG_GRBM_SOFT_RESET              0x8020
#define EG_GRBM_STATUS_GUI_ACTIVE       (1u << 31)
#define EG_DMA_STATUS                   0xd034
#define EG_SRBM_STATUS2                 0x0ec4
#define EG_RLC_FIRST                    0x3f00
#define EG_RLC_LAST                     0x3fff
#define EG_CP_STAT                      0x8680
#define EG_UVD_STATUS                   0xe020

/* display controller: global */
#define EG_DC_FIRST                     0x6000
#define EG_DC_LAST                      0x67ff
#define EG_DC_HPD1_INT_STATUS           0x601c
#define EG_DC_HPD1_INT_CONTROL          0x6020
#define EG_DC_HPD1_CONTROL              0x6024
#define EG_DC_HPD_STRIDE                0x000c
#define EG_DC_HPDx_INT_STATUS           (1u << 0)
#define EG_DC_HPDx_SENSE                (1u << 1)
#define EG_DC_HPDx_RX_INT_STATUS        (1u << 8)
#define EG_DC_HPDx_INT_ACK              (1u << 0)
#define EG_DC_HPDx_INT_POLARITY         (1u << 8)
#define EG_DC_HPDx_INT_EN               (1u << 16)
#define EG_DC_HPDx_RX_INT_ACK           (1u << 20)
#define EG_DC_HPDx_RX_INT_EN            (1u << 24)
#define EG_DC_I2C_CONTROL               0x6064
#define EG_DC_I2C_GO                    (1u << 0)
#define EG_DC_I2C_SOFT_RESET            (1u << 1)
#define EG_DC_I2C_SW_STATUS_RESET       (1u << 3)
#define EG_DC_I2C_ARBITRATION           0x6068
#define EG_DC_I2C_INTERRUPT_CONTROL     0x606c
#define EG_DC_I2C_SW_DONE_INT           (1u << 0)
#define EG_DC_I2C_SW_DONE_ACK           (1u << 1)
#define EG_DC_I2C_SW_STATUS             0x6070
#define EG_DC_I2C_SW_DONE               (1u << 2)
#define EG_DC_I2C_SW_STOPPED_ON_NACK    (1u << 8)
#define EG_DC_I2C_DDC1_HW_STATUS        0x6074
#define EG_DC_I2C_DDC1_SPEED            0x608c
#define EG_DC_I2C_TRANSACTION0          0x60bc
#define EG_DC_I2C_TRANSACTION1          0x60c0
#define EG_DC_I2C_TRANSACTION2          0x60c4
#define EG_DC_I2C_TRANSACTION3          0x60c8
#define EG_DC_I2C_RW                    (1u << 0)
#define EG_DC_I2C_STOP                  (1u << 13)
#define EG_DC_I2C_DATA                  0x60cc
#define EG_DC_I2C_DATA_RW               (1u << 0)
#define EG_DC_I2C_INDEX_WRITE           (1u << 31)
#define EG_DISP_INTERRUPT_STATUS        0x60f4
#define EG_DISP_INTERRUPT_STATUS_CONTINUE  0x60f8
#define EG_DISP_INTERRUPT_STATUS_CONTINUE2 0x60fc
#define EG_DISP_INTERRUPT_STATUS_CONTINUE3 0x6100
#define EG_DISP_INTERRUPT_STATUS_CONTINUE4 0x614c
#define EG_DISP_INTERRUPT_STATUS_CONTINUE5 0x6150
#define EG_LB_VLINE_INTERRUPT           (1u << 2)
#define EG_LB_VBLANK_INTERRUPT          (1u << 3)
#define EG_DC_HPD_INTERRUPT             (1u << 17)
#define EG_DC_HPD_RX_INTERRUPT          (1u << 18)
/* DDC GPIO pairs: line n (1..6) at 0x6420 + 0x10 n: MASK, A, EN, Y */
#define EG_DC_GPIO_DDC_FIRST            0x6430
#define EG_DC_GPIO_DDC_LAST             0x648f
#define EG_DC_GPIO_HPD_MASK             0x64b0
#define EG_DC_GPIO_HPD_A                0x64b4
#define EG_DC_GPIO_HPD_EN               0x64b8
#define EG_DC_GPIO_HPD_Y                0x64bc

/* display controller: per CRTC, CRTC0 offsets */
#define EG_CRTC_BLOCK_FIRST             0x6800
#define EG_CRTC_BLOCK_END               0x7000
#define EG_GRPH_ENABLE                  0x6800
#define EG_GRPH_CONTROL                 0x6804
#define EG_GRPH_LUT_10BIT_BYPASS_CONTROL 0x6808
#define EG_LUT_10BIT_BYPASS_EN          (1u << 8)
#define EG_GRPH_SWAP_CONTROL            0x680c
#define EG_GRPH_PRIMARY_SURFACE_ADDRESS 0x6810
#define EG_GRPH_SECONDARY_SURFACE_ADDRESS 0x6814
#define EG_GRPH_PITCH                   0x6818
#define EG_GRPH_PRIMARY_SURFACE_ADDRESS_HIGH 0x681c
#define EG_GRPH_SURFACE_OFFSET_X        0x6824
#define EG_GRPH_SURFACE_OFFSET_Y        0x6828
#define EG_GRPH_X_START                 0x682c
#define EG_GRPH_Y_START                 0x6830
#define EG_GRPH_X_END                   0x6834
#define EG_GRPH_Y_END                   0x6838
#define EG_GRPH_UPDATE                  0x6844
#define EG_GRPH_SURFACE_UPDATE_PENDING  (1u << 2)
#define EG_CUR_CONTROL                  0x6998
#define EG_CURSOR_EN                    (1u << 0)
#define EG_CUR_SURFACE_ADDRESS          0x699c
#define EG_CUR_SIZE                     0x69a0
#define EG_CUR_SURFACE_ADDRESS_HIGH     0x69a4
#define EG_CUR_POSITION                 0x69a8
#define EG_CUR_HOT_SPOT                 0x69ac
#define EG_CUR_UPDATE                   0x69b8
#define EG_CURSOR_UPDATE_PENDING        (1u << 0)
#define EG_CURSOR_UPDATE_LOCK           (1u << 16)
#define EG_DC_LUT_RW_MODE               0x69e0
#define EG_DC_LUT_RW_INDEX              0x69e4
#define EG_DC_LUT_SEQ_COLOR             0x69e8
#define EG_DC_LUT_PWL_DATA              0x69ec
#define EG_DC_LUT_30_COLOR              0x69f0
#define EG_DC_LUT_WRITE_EN_MASK         0x69f8
#define EG_DC_LUT_AUTOFILL              0x69fc
#define EG_DC_LUT_AUTOFILL_DONE         (1u << 1)
#define EG_DC_LUT_CONTROL               0x6a00
#define EG_INT_MASK                     0x6b40
#define EG_VBLANK_INT_MASK              (1u << 0)
#define EG_VLINE_INT_MASK               (1u << 4)
#define EG_VLINE_STATUS                 0x6bb8
#define EG_VBLANK_STATUS                0x6bbc
#define EG_xBLANK_OCCURRED              (1u << 0)
#define EG_xBLANK_ACK                   (1u << 4)
#define EG_xBLANK_STAT                  (1u << 12)
#define EG_xBLANK_INT                   (1u << 16)
#define EG_VIEWPORT_START               0x6d70
#define EG_VIEWPORT_SIZE                0x6d74
#define EG_CRTC_H_TOTAL                 0x6e00
#define EG_CRTC_H_BLANK_START_END       0x6e04
#define EG_CRTC_V_TOTAL                 0x6e1c
#define EG_CRTC_V_BLANK_START_END       0x6e34
#define EG_CRTC_CONTROL                 0x6e70
#define EG_CRTC_MASTER_EN               (1u << 0)
#define EG_CRTC_CURRENT_MASTER_EN_STATE (1u << 16)
#define EG_CRTC_DISP_READ_REQUEST_DISABLE (1u << 24)
#define EG_CRTC_BLANK_CONTROL           0x6e74
#define EG_CRTC_BLANK_DATA_EN           (1u << 8)
#define EG_CRTC_INTERLACE_CONTROL       0x6e78
#define EG_CRTC_STATUS                  0x6e8c
#define EG_CRTC_V_BLANK                 (1u << 0)
#define EG_CRTC_STATUS_POSITION         0x6e90
#define EG_CRTC_STATUS_FRAME_COUNT      0x6e98
#define EG_CRTC_STATUS_VF_COUNT         0x6e9c
#define EG_CRTC_STATUS_HV_COUNT         0x6ea0
#define EG_CRTC_UPDATE_LOCK             0x6ed4
#define EG_CRTC_DOUBLE_BUFFER_CONTROL   0x6ed8
#define EG_MASTER_UPDATE_LOCK           0x6ef4
#define EG_MASTER_UPDATE_MODE           0x6ef8

/* digital front/back ends (DIG0-5), with HDMI and DP blocks: stored only */
#define EG_DIG_BLOCK_FIRST              0x7000
#define EG_DIG_BLOCK_END                0x7400

#endif
