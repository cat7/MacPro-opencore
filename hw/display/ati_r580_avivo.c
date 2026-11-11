/*
 * ATI R580 (Radeon X1900 XT) AVIVO display engine: two display
 * controllers (D1, D2) with surface, CRTC timing, vblank, cursor and the
 * shared gamma LUTs, the display interrupt, hot plug and GPIO DDC.
 *
 * D2 registers are the D1 block plus 0x800. D1 is the console head; D2
 * keeps its registers and raises its vblank, nothing is shown from it.
 * All of this runs on the BQL.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Adapted from rv530-qemu 0b06c9f (qemu/hw/display/rv530/r5xx_avivo.c).
 */

#include "qemu/osdep.h"
#include "qemu/bswap.h"
#include "qemu/log.h"
#include "qemu/timer.h"
#include "qapi/error.h"
#include "hw/i2c/i2c.h"
#include "hw/i2c/bitbang_i2c.h"
#include "ui/console.h"
#include "ati_r580_int.h"
#include "ati_r580_regs.h"
#include "ati_r580_avivo.h"
#include "ati_r580_edid.h"
#include "trace.h"

#define FRAME_HZ        60
#define DEFAULT_LINES   806         /* until a mode is programmed */
#define REF_CLOCK_KHZ   27000

#define CREG(s, crtc, off) \
    ((s)->regs[((off) + ((crtc) ? R5XX_CRTC_STRIDE : 0)) >> 2])

/* ---------------------------------------------------------------- */
/* Timing                                                           */

static uint32_t crtc_lines(ATIR580State *s, int crtc)
{
    uint32_t vt = CREG(s, crtc, R5XX_D1CRTC_V_TOTAL) & 0x1fff;

    return vt ? vt + 1 : DEFAULT_LINES;
}

/* vblank window [start, end) in CRTC lines; may wrap */
static void crtc_vbl(ATIR580State *s, int crtc, uint32_t *start,
                     uint32_t *end)
{
    uint32_t vbl = CREG(s, crtc, R5XX_D1CRTC_V_BLANK_START_END);
    uint32_t lines = crtc_lines(s, crtc);

    *start = vbl & 0x1fff;
    *end = (vbl >> 16) & 0x1fff;
    if (!vbl || *start >= lines || *end >= lines) {
        *start = lines - lines / 25;
        *end = 0;
    }
}

/* each frame starts at the vblank tick, at the first vblank line */
static uint32_t crtc_vcount(ATIR580State *s, int crtc)
{
    int64_t t = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) - s->frame_start_ns;
    uint32_t lines = crtc_lines(s, crtc), start, end;

    t = MAX(t, 0);
    t = MIN(t, s->frame_period_ns - 1);
    crtc_vbl(s, crtc, &start, &end);
    return (start + (uint32_t)(t * lines / s->frame_period_ns)) % lines;
}

static bool crtc_in_vblank(ATIR580State *s, int crtc)
{
    uint32_t v, start, end;

    if (!(CREG(s, crtc, R5XX_D1CRTC_CONTROL) & R5XX_CRTC_EN)) {
        return true;
    }
    v = crtc_vcount(s, crtc);
    crtc_vbl(s, crtc, &start, &end);
    if (start <= end) {
        return v >= start && v < end;
    }
    return v >= start || v < end;
}

bool ati_r580_avivo_irq_pending(ATIR580State *s)
{
    uint32_t mask = s->regs[R5XX_DxMODE_INT_MASK >> 2];

    return (s->vblank_occurred[0] && (mask & R5XX_D1MODE_VBLANK_INT_MASK)) ||
           (s->vblank_occurred[1] && (mask & R5XX_D2MODE_VBLANK_INT_MASK)) ||
           (s->hpd1_pending &&
            (s->regs[R5XX_DC_HOT_PLUG_DETECT1_INT_CONTROL >> 2] &
             R5XX_HPD_INT_EN));
}

/*
 * D1's frame period: H_TOTAL x V_TOTAL / pixel clock, the clock from the
 * P1PLL dividers (27 MHz x fb.frac / (ref x post)). 60 Hz while those do
 * not give 24-240 Hz.
 */
static int64_t crtc_period_ns(ATIR580State *s)
{
    uint32_t fb = s->regs[R5XX_EXT1_PPLL_FB_DIV >> 2];
    uint32_t ref = s->regs[R5XX_EXT1_PPLL_REF_DIV >> 2] & 0x3ff;
    uint32_t post = s->regs[R5XX_EXT1_PPLL_POST_DIV >> 2] & 0x7f;
    uint64_t ht = (s->regs[R5XX_D1CRTC_H_TOTAL >> 2] & 0x1fff) + 1;
    uint64_t vt = (s->regs[R5XX_D1CRTC_V_TOTAL >> 2] & 0x1fff) + 1;
    /* pixel clock in 100 Hz units */
    uint64_t clk = ref && post ?
        (uint64_t)REF_CLOCK_KHZ * (((fb >> 16) & 0x7ff) * 10 + (fb & 0xf)) /
        (ref * post) : 0;
    int64_t period;

    if (!clk || ht < 64 || vt < 64) {
        return NANOSECONDS_PER_SECOND / FRAME_HZ;
    }
    period = ht * vt * 10000000 / clk;
    if (period < NANOSECONDS_PER_SECOND / 240 ||
        period > NANOSECONDS_PER_SECOND / 24) {
        return NANOSECONDS_PER_SECOND / FRAME_HZ;
    }
    return period;
}

/*
 * A frame starts at its deadline, not when the callback runs, so a late
 * callback does not delay the frames after it. After a stall of a whole
 * frame or more the timing restarts.
 */
static void ati_r580_avivo_vblank(void *opaque)
{
    ATIR580State *s = opaque;
    int64_t now = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
    int64_t start = s->frame_start_ns + s->frame_period_ns;
    int i;

    if (now - start >= s->frame_period_ns) {
        start = now;
    }
    s->frame_count++;
    s->frame_start_ns = start;
    s->frame_period_ns = crtc_period_ns(s);
    for (i = 0; i < ATI_R580_NUM_CRTC; i++) {
        if (CREG(s, i, R5XX_D1CRTC_CONTROL) & R5XX_CRTC_EN) {
            s->vblank_occurred[i] = true;
        }
    }
    ati_r580_update_irq(s);
    timer_mod(s->vblank_timer, start + s->frame_period_ns);
}

/* ---------------------------------------------------------------- */
/* DDC over GPIO: SCL on bit 0, SDA on bit 8 of A/EN/Y               */

static int ddc_for(uint32_t off)
{
    if (off >= R5XX_DC_GPIO_DDC1_MASK && off < R5XX_DC_GPIO_DDC3_MASK + 0x10) {
        return (off - R5XX_DC_GPIO_DDC1_MASK) / 0x10;
    }
    return -1;
}

static void ddc_update(ATIR580State *s, int n)
{
    uint32_t base = R5XX_DC_GPIO_DDC1_MASK + n * 0x10;
    uint32_t a = s->regs[(base + 4) >> 2];
    uint32_t en = s->regs[(base + 8) >> 2];
    /* open drain: a pin driven low pulls its line low, else it floats high */
    int scl = ((en & 1) && !(a & 1)) ? 0 : 1;
    int sda = ((en & 0x100) && !(a & 0x100)) ? 0 : 1;
    int sda_in = sda;

    if (s->ddc_slave[n]) {
        bitbang_i2c_set(&s->ddc_i2c[n], BITBANG_I2C_SCL, scl);
        sda_in = bitbang_i2c_set(&s->ddc_i2c[n], BITBANG_I2C_SDA, sda);
    }
    s->ddc_y[n] = (scl ? 1 : 0) | (sda_in ? 0x100 : 0);
}

/* ---------------------------------------------------------------- */
/* Scanout                                                          */

static int lut_index(ATIR580State *s)
{
    return (s->regs[R5XX_D1GRPH_LUT_SEL >> 2] & R5XX_D1GRPH_LUT_SEL_B) ? 1 : 0;
}

const uint8_t (*ati_r580_avivo_lut(ATIR580State *s))[3]
{
    return (const uint8_t (*)[3])s->lut[lut_index(s)];
}

/*
 * The D1 graphics surface as a mode for the display path. Linear
 * surfaces only; a tiled one is read as if linear.
 */
bool ati_r580_avivo_scanout(ATIR580State *s, ATIR580Mode *m)
{
    uint32_t ctl = s->regs[R5XX_D1GRPH_CONTROL >> 2];
    uint32_t depth = ctl & 3, fmt = (ctl >> 8) & 7;
    uint32_t xs = s->regs[R5XX_D1GRPH_X_START >> 2];
    uint32_t xe = s->regs[R5XX_D1GRPH_X_END >> 2];
    uint32_t ys = s->regs[R5XX_D1GRPH_Y_START >> 2];
    uint32_t ye = s->regs[R5XX_D1GRPH_Y_END >> 2];
    uint32_t vp = s->regs[R5XX_D1MODE_VIEWPORT_SIZE >> 2];
    uint32_t vps = s->regs[R5XX_D1MODE_VIEWPORT_START >> 2];
    uint32_t crtc = s->regs[R5XX_D1CRTC_CONTROL >> 2];
    uint32_t x0, y0, off;
    uint64_t end;

    memset(m, 0, sizeof(*m));
    if (!(crtc & R5XX_CRTC_EN) || (crtc & R5XX_CRTC_DISP_READ_REQUEST_DISABLE) ||
        !(s->regs[R5XX_D1GRPH_ENABLE >> 2] & 1) ||
        (s->regs[R5XX_D1CRTC_BLANK_CONTROL >> 2] & R5XX_CRTC_BLANK_DATA_EN)) {
        return false;
    }
    switch (depth) {
    case 0:
        m->bpp = 1;
        m->pix_width = R580_PIX_WIDTH_8BPP;
        break;
    case 1:
        m->bpp = 2;
        if (fmt == 0) {
            m->pix_width = R580_PIX_WIDTH_15BPP;
        } else if (fmt == 1) {
            m->pix_width = R580_PIX_WIDTH_16BPP;
        } else {
            return false;
        }
        break;
    case 2:
        if (fmt != 0) {
            return false;
        }
        m->bpp = 4;
        m->pix_width = R580_PIX_WIDTH_32BPP;
        break;
    default:
        return false;
    }
    m->width = xe > xs ? xe - xs : 0;
    m->height = ye > ys ? ye - ys : 0;
    if (vp) {
        uint32_t vw = (vp >> 16) & 0x3fff, vh = vp & 0x3fff;

        if (vw && vw < m->width) {
            m->width = vw;
        }
        if (vh && vh < m->height) {
            m->height = vh;
        }
    }
    if (m->width < 16 || m->height < 16 || m->width > 4096 ||
        m->height > 4096) {
        return false;
    }
    m->pitch = (s->regs[R5XX_D1GRPH_PITCH >> 2] & 0x3fff) * m->bpp;
    if (m->pitch < m->width * m->bpp) {
        return false;
    }
    if (!ati_r580_mc_to_vram(s,
            s->regs[R5XX_D1GRPH_PRIMARY_SURFACE_ADDRESS >> 2], &off)) {
        return false;
    }
    x0 = s->regs[R5XX_D1GRPH_SURFACE_OFFSET_X >> 2] + xs + ((vps >> 16) & 0x3fff);
    y0 = s->regs[R5XX_D1GRPH_SURFACE_OFFSET_Y >> 2] + ys + (vps & 0x3fff);
    end = (uint64_t)off + (uint64_t)y0 * m->pitch + (uint64_t)x0 * m->bpp;
    if (end + (uint64_t)m->pitch * m->height > ATI_R580_VRAM_SIZE) {
        return false;
    }
    m->fb_offset = end;
    return true;
}

/* ---------------------------------------------------------------- */
/* Cursor: 64x64 ARGB8888, 256-byte rows                            */

void ati_r580_avivo_cursor_apply(ATIR580State *s)
{
    uint32_t ctl = s->regs[R5XX_D1CUR_CONTROL >> 2];
    uint32_t size = s->regs[R5XX_D1CUR_SIZE >> 2];
    uint32_t pos = s->regs[R5XX_D1CUR_POSITION >> 2];
    uint32_t hot = s->regs[R5XX_D1CUR_HOT_SPOT >> 2];
    uint32_t w = ((size >> 16) & 0x7f) + 1, h = (size & 0x7f) + 1;
    uint32_t hx = (hot >> 16) & 0x3f, hy = hot & 0x3f;
    const uint8_t *vram = memory_region_get_ram_ptr(&s->vram);
    uint32_t off, sum = 0, x, y;
    QEMUCursor *c;

    if (!s->con ||
        (s->regs[R5XX_D1CUR_UPDATE >> 2] & R5XX_D1CUR_UPDATE_LOCK)) {
        return;
    }
    if (!(ctl & R5XX_D1CURSOR_EN) || w > 64 || h > 64 ||
        !ati_r580_mc_to_vram(s, s->regs[R5XX_D1CUR_SURFACE_ADDRESS >> 2],
                             &off) ||
        off + 64 * 256 > ATI_R580_VRAM_SIZE) {
        if (s->hw_cursor_on) {
            s->hw_cursor_on = false;
            s->hw_cursor_sum = 0;
            qemu_console_set_mouse(s->con, 0, 0, false);
        }
        trace_ati_r580_avivo_cursor(ctl & R5XX_D1CURSOR_EN, 0, 0, w, h, 0);
        return;
    }
    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            sum = sum * 31 + ldl_le_p(vram + off + y * 256 + x * 4);
        }
    }
    sum ^= (w << 24) ^ (h << 16) ^ (hx << 8) ^ hy;
    if (!s->hw_cursor_on || sum != s->hw_cursor_sum) {
        c = cursor_alloc(w, h);
        for (y = 0; y < h; y++) {
            for (x = 0; x < w; x++) {
                c->data[y * w + x] = ldl_le_p(vram + off + y * 256 + x * 4);
            }
        }
        c->hot_x = hx;
        c->hot_y = hy;
        qemu_console_set_cursor(s->con, c);
        cursor_unref(c);
        s->hw_cursor_sum = sum;
    }
    s->hw_cursor_on = true;
    s->hw_cursor_x = (pos >> 16) & 0x3fff;
    s->hw_cursor_y = pos & 0x3fff;
    trace_ati_r580_avivo_cursor(1, s->hw_cursor_x, s->hw_cursor_y, w, h, off);
    qemu_console_set_mouse(s->con, s->hw_cursor_x, s->hw_cursor_y, true);
}

/* ---------------------------------------------------------------- */
/* Registers                                                        */

static bool is_global_dc(uint32_t off)
{
    return (off >= R5XX_DC_LUT_RW_SELECT && off <= R5XX_DC_LUT_AUTOFILL) ||
           off == R5XX_DxMODE_INT_MASK || off == R5XX_DC_CRTC_MASTER_EN ||
           off == R5XX_DC_LB_MEMORY_SPLIT;
}

/* the CRTC a per-CRTC register belongs to, -1 for the others */
static int crtc_for(uint32_t off, uint32_t *rel)
{
    int crtc;

    if (off < 0x6000 || off >= 0x7000 || is_global_dc(off)) {
        return -1;
    }
    crtc = (off - 0x6000) / R5XX_CRTC_STRIDE;
    *rel = off - crtc * R5XX_CRTC_STRIDE;
    return crtc;
}

static void lut_set(ATIR580State *s, int sel, unsigned i, uint8_t r,
                    uint8_t g, uint8_t b)
{
    if (s->lut[sel][i][0] != r || s->lut[sel][i][1] != g ||
        s->lut[sel][i][2] != b) {
        s->lut[sel][i][0] = r;
        s->lut[sel][i][1] = g;
        s->lut[sel][i][2] = b;
        s->lut_gen++;
    }
}

bool ati_r580_avivo_read(ATIR580State *s, uint32_t off, uint32_t *val)
{
    uint32_t rel;
    int crtc, n;

    switch (off) {
    case R5XX_DISP_INTERRUPT_STATUS: {
        uint32_t v = 0, mask = s->regs[R5XX_DxMODE_INT_MASK >> 2];

        if (s->vblank_occurred[0] && (mask & R5XX_D1MODE_VBLANK_INT_MASK)) {
            v |= R5XX_DISP_LB_D1_VBLANK_INT;
        }
        if (s->vblank_occurred[1] && (mask & R5XX_D2MODE_VBLANK_INT_MASK)) {
            v |= R5XX_DISP_LB_D2_VBLANK_INT;
        }
        if (s->hpd1_pending) {
            v |= R5XX_DISP_HPD1_INT;
        }
        *val = v;
        return true;
    }
    case R5XX_DC_HOT_PLUG_DETECT1_INT_STATUS:
        *val = R5XX_HPD_SENSE | (s->hpd1_pending ? R5XX_HPD_INT_STATUS : 0);
        return true;
    case R5XX_DC_HOT_PLUG_DETECT2_INT_STATUS:
        *val = 0;
        return true;
    case R5XX_DC_LUT_AUTOFILL:
        *val = s->regs[off >> 2] | R5XX_DC_LUT_AUTOFILL_DONE;
        return true;
    case R5XX_DC_LUT_30_COLOR: {
        int sel = s->regs[R5XX_DC_LUT_RW_SELECT >> 2] & 1;
        unsigned i = s->regs[R5XX_DC_LUT_RW_INDEX >> 2] & 0xff;

        *val = ((uint32_t)s->lut[sel][i][0] << 22) |
               ((uint32_t)s->lut[sel][i][1] << 12) |
               ((uint32_t)s->lut[sel][i][2] << 2);
        return true;
    }
    case R5XX_DACA_COMPARATOR_OUTPUT:
    case R5XX_DACB_COMPARATOR_OUTPUT:
        *val = 0;                   /* nothing on the analog outputs */
        return true;
    default:
        break;
    }

    crtc = crtc_for(off, &rel);
    if (crtc >= 0) {
        switch (rel) {
        case R5XX_D1CRTC_CONTROL:
            *val = s->regs[off >> 2] & ~R5XX_CRTC_CURRENT_MASTER_EN_STATE;
            if (*val & R5XX_CRTC_EN) {
                *val |= R5XX_CRTC_CURRENT_MASTER_EN_STATE;
            }
            return true;
        case R5XX_D1CRTC_STATUS:
            *val = crtc_in_vblank(s, crtc) ? R5XX_D1CRTC_V_BLANK : 0;
            return true;
        case R5XX_D1CRTC_STATUS_POSITION:
        case R5XX_D1CRTC_STATUS_HV_COUNT:
            *val = crtc_vcount(s, crtc) & 0x1fff;
            return true;
        case R5XX_D1CRTC_FRAME_COUNT:
            *val = (CREG(s, crtc, R5XX_D1CRTC_CONTROL) & R5XX_CRTC_EN) ?
                   s->frame_count & 0xffffff : 0;
            return true;
        case R5XX_D1MODE_VBLANK_STATUS: {
            uint32_t bit = crtc ? R5XX_D2MODE_VBLANK_INT_MASK
                                : R5XX_D1MODE_VBLANK_INT_MASK;

            *val = 0;
            if (s->vblank_occurred[crtc]) {
                *val |= R5XX_VBLANK_OCCURRED;
                if (s->regs[R5XX_DxMODE_INT_MASK >> 2] & bit) {
                    *val |= R5XX_VBLANK_INTERRUPT;
                }
            }
            if (crtc_in_vblank(s, crtc)) {
                *val |= R5XX_VBLANK_STAT;
            }
            return true;
        }
        case R5XX_D1GRPH_UPDATE:
            /* surface updates apply at once: never pending */
            *val = s->regs[off >> 2] & ~(R5XX_D1GRPH_SURFACE_UPDATE_PENDING | 1u);
            return true;
        default:
            break;
        }
    }

    n = ddc_for(off);
    if (n >= 0 && (off & 0xc) == 0xc) {
        *val = s->ddc_y[n];
        return true;
    }
    *val = s->regs[off >> 2];
    return true;
}

bool ati_r580_avivo_write(ATIR580State *s, uint32_t off, uint32_t val)
{
    uint32_t idx = off >> 2, rel;
    int crtc, n;

    trace_ati_r580_avivo_write(off, val);
    switch (off) {
    case R5XX_DxMODE_INT_MASK:
        s->regs[idx] = val;
        ati_r580_update_irq(s);
        return true;
    case R5XX_DC_HOT_PLUG_DETECT1_INT_CONTROL:
        s->regs[idx] = val & ~R5XX_HPD_INT_ACK;
        if (val & R5XX_HPD_INT_ACK) {
            s->hpd1_pending = false;
        }
        ati_r580_update_irq(s);
        return true;
    case R5XX_DC_LUT_AUTOFILL:
        s->regs[idx] = val;
        if (val & 1) {
            int sel = s->regs[R5XX_DC_LUT_RW_SELECT >> 2] & 1;
            unsigned i;

            for (i = 0; i < 256; i++) {
                lut_set(s, sel, i, i, i, i);
            }
        }
        return true;
    case R5XX_DC_LUT_RW_INDEX:
        s->regs[idx] = val & 0xff;
        s->lut_seq = 0;
        return true;
    case R5XX_DC_LUT_30_COLOR: {
        int sel = s->regs[R5XX_DC_LUT_RW_SELECT >> 2] & 1;
        unsigned i = s->regs[R5XX_DC_LUT_RW_INDEX >> 2] & 0xff;

        lut_set(s, sel, i, (val >> 22) & 0xff, (val >> 12) & 0xff,
                (val >> 2) & 0xff);
        s->regs[R5XX_DC_LUT_RW_INDEX >> 2] = (i + 1) & 0xff;
        return true;
    }
    case R5XX_DC_LUT_SEQ_COLOR: {
        /* 8 bits per write, R then G then B, then the next entry */
        int sel = s->regs[R5XX_DC_LUT_RW_SELECT >> 2] & 1;
        unsigned i = s->regs[R5XX_DC_LUT_RW_INDEX >> 2] & 0xff;
        uint8_t c[3];

        memcpy(c, s->lut[sel][i], 3);
        c[s->lut_seq] = val & 0xff;
        lut_set(s, sel, i, c[0], c[1], c[2]);
        if (++s->lut_seq == 3) {
            s->lut_seq = 0;
            s->regs[R5XX_DC_LUT_RW_INDEX >> 2] = (i + 1) & 0xff;
        }
        return true;
    }
    default:
        break;
    }

    crtc = crtc_for(off, &rel);
    if (crtc >= 0) {
        switch (rel) {
        case R5XX_D1MODE_VBLANK_STATUS:
            if (val & R5XX_VBLANK_ACK) {
                s->vblank_occurred[crtc] = false;
                ati_r580_update_irq(s);
            }
            return true;
        case R5XX_D1CRTC_CONTROL:
            s->regs[idx] = val & ~R5XX_CRTC_CURRENT_MASTER_EN_STATE;
            if (!(val & R5XX_CRTC_EN)) {
                s->vblank_occurred[crtc] = false;
                ati_r580_update_irq(s);
            }
            return true;
        case R5XX_D1CRTC_STATUS:
        case R5XX_D1CRTC_STATUS_POSITION:
        case R5XX_D1CRTC_FRAME_COUNT:
        case R5XX_D1CRTC_STATUS_HV_COUNT:
            return true;            /* read only */
        case R5XX_D1GRPH_LUT_SEL:
            s->regs[idx] = val;
            s->lut_gen++;
            return true;
        case R5XX_D1CUR_CONTROL:
        case R5XX_D1CUR_SURFACE_ADDRESS:
        case R5XX_D1CUR_SIZE:
        case R5XX_D1CUR_POSITION:
        case R5XX_D1CUR_HOT_SPOT:
        case R5XX_D1CUR_UPDATE:
            s->regs[idx] = val;
            if (crtc == 0) {
                ati_r580_cursor_update(s);
            }
            return true;
        default:
            s->regs[idx] = val;
            return true;
        }
    }

    n = ddc_for(off);
    if (n >= 0) {
        if ((off & 0xc) == 0xc) {
            return true;            /* Y is read only */
        }
        s->regs[idx] = val;
        if ((off & 0xc) == 0x4 || (off & 0xc) == 0x8) {
            ddc_update(s, n);
        }
        return true;
    }
    s->regs[idx] = val;
    return true;
}

/* ---------------------------------------------------------------- */
/* Init / reset                                                     */

/*
 * The card resets posted: D1 enabled, scanning out a linear 32bpp
 * surface at VRAM offset 0 in the EDID's preferred mode. D2 off.
 */
void ati_r580_avivo_reset(ATIR580State *s)
{
    uint32_t xres = s->xres, yres = s->yres;
    unsigned i, j;

    s->frame_count = 0;
    s->hpd1_pending = false;
    s->vblank_occurred[0] = s->vblank_occurred[1] = false;
    s->frame_period_ns = NANOSECONDS_PER_SECOND / FRAME_HZ;
    s->frame_start_ns = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
    timer_mod(s->vblank_timer, s->frame_start_ns + s->frame_period_ns);
    for (j = 0; j < 2; j++) {
        for (i = 0; i < 256; i++) {
            s->lut[j][i][0] = s->lut[j][i][1] = s->lut[j][i][2] = i;
        }
    }
    s->lut_seq = 0;
    s->lut_gen++;
    for (i = 0; i < ATI_R580_NUM_DDC; i++) {
        s->ddc_y[i] = 0x101;
    }

    s->regs[R5XX_D1CRTC_CONTROL >> 2] = R5XX_CRTC_EN;
    s->regs[R5XX_D1GRPH_ENABLE >> 2] = 1;
    s->regs[R5XX_D1GRPH_CONTROL >> 2] = 2;              /* ARGB8888 */
    s->regs[R5XX_D1GRPH_PRIMARY_SURFACE_ADDRESS >> 2] = 0;
    s->regs[R5XX_D1GRPH_SECONDARY_SURFACE_ADDRESS >> 2] = 0;
    s->regs[R5XX_D1GRPH_PITCH >> 2] = xres;
    s->regs[R5XX_D1GRPH_X_END >> 2] = xres;
    s->regs[R5XX_D1GRPH_Y_END >> 2] = yres;
    s->regs[R5XX_D1MODE_VIEWPORT_SIZE >> 2] = (xres << 16) | yres;
    s->regs[R5XX_D1MODE_DESKTOP_HEIGHT >> 2] = yres;
    s->regs[R5XX_D1CRTC_H_TOTAL >> 2] = xres + 320 - 1;
    s->regs[R5XX_D1CRTC_V_TOTAL >> 2] = yres + 38 - 1;
    s->regs[R5XX_D1CRTC_H_BLANK_START_END >> 2] = (xres + 296) << 16 | 160;
    s->regs[R5XX_D1CRTC_V_BLANK_START_END >> 2] = (yres + 33) << 16 | 29;
    s->regs[R5XX_DC_CRTC_MASTER_EN >> 2] = 1;
    s->regs[R5XX_TMDSA_CNTL >> 2] = 1;
    s->regs[R5XX_D1MODE_MASTER_UPDATE_MODE >> 2] = 3;
    s->regs[(R5XX_D1MODE_MASTER_UPDATE_MODE + R5XX_CRTC_STRIDE) >> 2] = 3;
    s->regs[R5XX_DC_HOT_PLUG_DETECT1_CONTROL >> 2] = 1;
}

void ati_r580_avivo_init(ATIR580State *s)
{
    uint32_t hz = s->edid_info.refresh_rate ?
                  s->edid_info.refresh_rate / 1000 : FRAME_HZ;
    int i;

    if (!s->edid_info.prefx || !s->edid_info.prefy) {
        s->edid_info.prefx = 1024;
        s->edid_info.prefy = 768;
    }
    if (!s->edid_info.maxx || !s->edid_info.maxy) {
        s->edid_info.maxx = 2560;
        s->edid_info.maxy = 1600;
    }
    s->xres = MIN(s->edid_info.prefx, 4096) & ~7u;
    s->yres = MIN(s->edid_info.prefy, 4096);

    s->vblank_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL,
                                   ati_r580_avivo_vblank, s);
    for (i = 0; i < ATI_R580_NUM_DDC; i++) {
        char name[32];
        I2CBus *bus;

        snprintf(name, sizeof(name), "ati-r580.ddc%d", i + 1);
        bus = i2c_init_bus(DEVICE(s), name);
        bitbang_i2c_init(&s->ddc_i2c[i], bus);
        s->ddc_slave[i] = NULL;
        /* the monitor sits on DDC1, the DVI connector */
        if (i == 0) {
            I2CSlave *slv = i2c_slave_create_simple(bus, TYPE_ATI_R580_EDID,
                                                    0x50);

            s->ddc_slave[i] = ATI_R580_EDID(slv);
            ati_r580_edid_build(s->ddc_slave[i]->blob, s->xres, s->yres,
                                s->edid_info.maxx, s->edid_info.maxy, 0, hz);
        }
    }
}

void ati_r580_avivo_fini(ATIR580State *s)
{
    timer_free(s->vblank_timer);
    s->vblank_timer = NULL;
}
