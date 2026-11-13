/*
 * ATI Radeon HD 5770 AtomBIOS mailbox: the device side of the synthetic
 * VBIOS built by ati_hd5770_atombios.py.
 *
 * Every command table of that ROM copies its parameter block into
 * HD5770_ATOM_MAILBOX (16 dwords) and writes "table index | dwords << 8"
 * to HD5770_ATOM_DOORBELL; tables that return values copy the mailbox
 * back to their parameter space. Parameter layouts are the DCE 4 versions
 * of the atombios.h structs (little endian).
 *
 * The display driver programs CRTC timing, surfaces and the DDC engine
 * itself; of the tables it runs, these act: SetPixelClock (the CRTC's
 * pixel clock, for the frame rate), EnableCRTC, BlankCRTC,
 * EnableCRTCMemReq, AdjustDisplayPll and the clock queries (they return
 * values). ProcessAuxChannelTransaction answers every request with a
 * timeout: no DisplayPort sink is attached. The others succeed and do
 * nothing.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Adapted from ati_r580_atom.c.
 */

#include "qemu/osdep.h"
#include "qemu/bswap.h"
#include "ati_hd5770_int.h"
#include "ati_hd5770_regs.h"
#include "trace.h"

/* command table indices (ATOM_MASTER_LIST_OF_COMMAND_TABLES) */
enum {
    ATOM_ASIC_Init = 0,
    ATOM_EnableCRTCMemReq = 6,
    ATOM_SetPixelClock = 12,
    ATOM_AdjustDisplayPll = 17,
    ATOM_BlankCRTC = 34,
    ATOM_EnableCRTC = 35,
    ATOM_GetMemoryClock = 47,
    ATOM_GetEngineClock = 48,
    ATOM_ComputeMemoryEnginePLL = 60,
    ATOM_ProcessAuxChannelTransaction = 78,
};

/* engine and memory clocks reported, 10 kHz units */
#define ATOM_ENGINE_CLOCK       85000
#define ATOM_MEMORY_CLOCK       120000

/* ProcessAuxChannelTransaction ucReplyStatus */
#define ATOM_AUX_TIMEOUT        1

static uint32_t mb_u8(const uint8_t *p, int i)
{
    return p[i];
}

static uint32_t mb_u16(const uint8_t *p, int i)
{
    return lduw_le_p(p + i);
}

static uint32_t *crtc_reg(ATIHD5770State *s, uint32_t crtc, uint32_t off)
{
    return &s->regs[(off + ati_hd5770_crtc_offset[crtc]) >> 2];
}

static void atom_call(ATIHD5770State *s, uint32_t table, uint32_t ndw)
{
    uint8_t p[HD5770_ATOM_MAILBOX_DWORDS * 4];
    uint32_t *mb = &s->regs[HD5770_ATOM_MAILBOX >> 2];
    uint32_t i, crtc;

    for (i = 0; i < HD5770_ATOM_MAILBOX_DWORDS; i++) {
        stl_le_p(p + 4 * i, i < ndw ? mb[i] : 0);
    }
    trace_ati_hd5770_atom_call(table, ndw, ldl_le_p(p), ldl_le_p(p + 4),
                               ldl_le_p(p + 8), ldl_le_p(p + 12));
    if (table < ARRAY_SIZE(s->atom_calls)) {
        s->atom_calls[table]++;
    }

    switch (table) {
    case ATOM_SetPixelClock:
        /* v5: CRTC, -, clock, ..., PPLL at 8; DCPLL calls name no CRTC */
        ati_hd5770_dce4_set_pixclk(s, mb_u8(p, 0), mb_u16(p, 2));
        break;
    case ATOM_AdjustDisplayPll:
        /* v3: clock in; the same clock out, no fixed dividers */
        mb[0] = mb_u16(p, 0);
        mb[1] = 0;
        break;
    case ATOM_EnableCRTC:
        crtc = mb_u8(p, 0);
        if (crtc < ATI_HD5770_NUM_CRTC) {
            uint32_t *r = crtc_reg(s, crtc, EG_CRTC_CONTROL);

            if (mb_u8(p, 1)) {
                *r = (*r | EG_CRTC_MASTER_EN) &
                     ~EG_CRTC_DISP_READ_REQUEST_DISABLE;
            } else {
                *r = (*r & ~EG_CRTC_MASTER_EN) |
                     EG_CRTC_DISP_READ_REQUEST_DISABLE;
            }
        }
        break;
    case ATOM_EnableCRTCMemReq:
        crtc = mb_u8(p, 0);
        if (crtc < ATI_HD5770_NUM_CRTC) {
            uint32_t *r = crtc_reg(s, crtc, EG_CRTC_CONTROL);

            *r = (*r & ~EG_CRTC_DISP_READ_REQUEST_DISABLE) |
                 (mb_u8(p, 1) ? 0 : EG_CRTC_DISP_READ_REQUEST_DISABLE);
        }
        break;
    case ATOM_BlankCRTC:
        /* CRTC, blanking; the black colour is not modelled */
        crtc = mb_u8(p, 0);
        if (crtc < ATI_HD5770_NUM_CRTC) {
            *crtc_reg(s, crtc, EG_CRTC_BLANK_CONTROL) =
                mb_u8(p, 1) ? EG_CRTC_BLANK_DATA_EN : 0;
        }
        break;
    case ATOM_GetEngineClock:
        mb[0] = ATOM_ENGINE_CLOCK;
        break;
    case ATOM_GetMemoryClock:
        mb[0] = ATOM_MEMORY_CLOCK;
        break;
    case ATOM_ComputeMemoryEnginePLL:
        /* clock in, post divider out */
        mb[1] = 1;
        break;
    case ATOM_ProcessAuxChannelTransaction:
        mb[1] = (mb[1] & ~0xff00u) | (ATOM_AUX_TIMEOUT << 8);
        break;
    default:
        break;
    }
    s->force_redraw = true;
}

bool ati_hd5770_atom_write(ATIHD5770State *s, uint32_t off, uint32_t val)
{
    if (off >= HD5770_ATOM_MAILBOX &&
        off < HD5770_ATOM_MAILBOX + 4 * HD5770_ATOM_MAILBOX_DWORDS) {
        s->regs[off >> 2] = val;
        return true;
    }
    if (off == HD5770_ATOM_DOORBELL) {
        atom_call(s, val & 0xff,
                  MIN((val >> 8) & 0xff, HD5770_ATOM_MAILBOX_DWORDS));
        return true;
    }
    return false;
}
