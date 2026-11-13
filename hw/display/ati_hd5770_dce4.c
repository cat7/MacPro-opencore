/*
 * ATI Radeon HD 5770 (Juniper) DCE 4 display engine: six CRTCs with their
 * graphics surface, timing, vblank and vline interrupts, cursor and gamma
 * LUT; six hot plug pins; six DDC lines, each reachable by GPIO bit-bang
 * and by the DC_I2C engine.
 *
 * CRTC 0 is the console head; the others keep their registers and raise
 * their interrupts, nothing is shown from them. One monitor is attached,
 * on the DDC line and hot plug pin the mon-ddc / mon-hpd properties name;
 * the other ports read as empty.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Adapted from ati_r580_avivo.c.
 */

#include "qemu/osdep.h"
#include "qemu/bswap.h"
#include "qemu/timer.h"
#include "qemu/units.h"
#include "hw/i2c/i2c.h"
#include "hw/i2c/bitbang_i2c.h"
#include "ui/console.h"
#include "ati_hd5770_int.h"
#include "ati_hd5770_regs.h"
#include "trace.h"

#define FRAME_HZ        60
#define DEFAULT_LINES   806         /* until a mode is programmed */
#define MONITOR_SERIAL  0x5770
#define POSTED_X        1024
#define POSTED_Y        768

const uint32_t ati_hd5770_crtc_offset[ATI_HD5770_NUM_CRTC] = {
    0x0000, 0x0c00, 0x9800, 0xa400, 0xb000, 0xbc00,
};

/* DISP_INTERRUPT_STATUS register of each CRTC / HPD pin */
static const uint32_t disp_int_status[ATI_HD5770_NUM_CRTC] = {
    EG_DISP_INTERRUPT_STATUS, EG_DISP_INTERRUPT_STATUS_CONTINUE,
    EG_DISP_INTERRUPT_STATUS_CONTINUE2, EG_DISP_INTERRUPT_STATUS_CONTINUE3,
    EG_DISP_INTERRUPT_STATUS_CONTINUE4, EG_DISP_INTERRUPT_STATUS_CONTINUE5,
};

/* DC_GPIO_HPD_A/EN/Y bit of each hot plug pin */
static const uint8_t hpd_gpio_shift[ATI_HD5770_NUM_HPD] = { 0, 8, 16, 24, 26, 28 };

#define CREG(s, crtc, off) \
    ((s)->regs[((off) + ati_hd5770_crtc_offset[crtc]) >> 2])

/* the CRTC a per-CRTC register belongs to, -1 for the others */
static int crtc_for(uint32_t off, uint32_t *rel)
{
    int c;

    for (c = 0; c < ATI_HD5770_NUM_CRTC; c++) {
        uint32_t r = off - ati_hd5770_crtc_offset[c];

        if (off >= ati_hd5770_crtc_offset[c] &&
            r >= EG_CRTC_BLOCK_FIRST && r < EG_CRTC_BLOCK_END) {
            *rel = r;
            return c;
        }
    }
    return -1;
}

/* ---------------------------------------------------------------- */
/* Timing                                                           */

static uint32_t crtc_lines(ATIHD5770State *s, int crtc)
{
    uint32_t vt = CREG(s, crtc, EG_CRTC_V_TOTAL) & 0x3fff;

    return vt ? vt + 1 : DEFAULT_LINES;
}

/* vblank window [start, end) in CRTC lines; may wrap */
static void crtc_vbl(ATIHD5770State *s, int crtc, uint32_t *start,
                     uint32_t *end)
{
    uint32_t vbl = CREG(s, crtc, EG_CRTC_V_BLANK_START_END);
    uint32_t lines = crtc_lines(s, crtc);

    *start = vbl & 0x3fff;
    *end = (vbl >> 16) & 0x3fff;
    if (!vbl || *start >= lines || *end >= lines) {
        *start = lines - lines / 25;
        *end = 0;
    }
}

/* each frame starts at the vblank tick, at the first vblank line */
static uint32_t crtc_vcount(ATIHD5770State *s, int crtc)
{
    int64_t t = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) - s->frame_start_ns;
    uint32_t lines = crtc_lines(s, crtc), start, end;

    t = MAX(t, 0);
    t = MIN(t, s->frame_period_ns - 1);
    crtc_vbl(s, crtc, &start, &end);
    return (start + (uint32_t)(t * lines / s->frame_period_ns)) % lines;
}

static bool crtc_on(ATIHD5770State *s, int crtc)
{
    return CREG(s, crtc, EG_CRTC_CONTROL) & EG_CRTC_MASTER_EN;
}

static bool crtc_in_vblank(ATIHD5770State *s, int crtc)
{
    uint32_t v, start, end;

    if (!crtc_on(s, crtc)) {
        return true;
    }
    v = crtc_vcount(s, crtc);
    crtc_vbl(s, crtc, &start, &end);
    if (start <= end) {
        return v >= start && v < end;
    }
    return v >= start || v < end;
}

void ati_hd5770_dce4_set_pixclk(ATIHD5770State *s, unsigned crtc,
                                uint32_t clk_10khz)
{
    if (crtc < ATI_HD5770_NUM_CRTC) {
        s->pixclk_10khz[crtc] = clk_10khz;
    }
}

/*
 * CRTC 0's frame period: H_TOTAL x V_TOTAL / pixel clock, the clock as
 * SetPixelClock last set it. 60 Hz while that does not give 24-240 Hz.
 */
static int64_t crtc_period_ns(ATIHD5770State *s)
{
    uint64_t ht = (CREG(s, 0, EG_CRTC_H_TOTAL) & 0x3fff) + 1;
    uint64_t vt = (CREG(s, 0, EG_CRTC_V_TOTAL) & 0x3fff) + 1;
    uint64_t clk = s->pixclk_10khz[0];
    int64_t period;

    if (!clk || ht < 64 || vt < 64) {
        return NANOSECONDS_PER_SECOND / FRAME_HZ;
    }
    period = ht * vt * 100000 / clk;
    if (period < NANOSECONDS_PER_SECOND / 240 ||
        period > NANOSECONDS_PER_SECOND / 24) {
        return NANOSECONDS_PER_SECOND / FRAME_HZ;
    }
    return period;
}

/*
 * A frame starts at its deadline, not when the callback runs, so a late
 * callback does not delay the frames after it. After a stall of a whole
 * frame or more the timing restarts. An interrupt goes into the ring when
 * its status bit sets and its mask bit is on.
 */
static void ati_hd5770_vblank(void *opaque)
{
    ATIHD5770State *s = opaque;
    int64_t now = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
    int64_t start = s->frame_start_ns + s->frame_period_ns;
    int i;

    if (now - start >= s->frame_period_ns) {
        start = now;
    }
    s->frame_count++;
    s->frame_start_ns = start;
    s->frame_period_ns = crtc_period_ns(s);
    for (i = 0; i < ATI_HD5770_NUM_CRTC; i++) {
        uint32_t mask = CREG(s, i, EG_INT_MASK);

        if (!crtc_on(s, i)) {
            continue;
        }
        if (!s->vblank_occurred[i]) {
            s->vblank_occurred[i] = true;
            if (mask & EG_VBLANK_INT_MASK) {
                ati_hd5770_ih_push(s, HD5770_IH_SRC_D1 + i, 0);
            }
        }
        if (!s->vline_occurred[i]) {
            s->vline_occurred[i] = true;
            if (mask & EG_VLINE_INT_MASK) {
                ati_hd5770_ih_push(s, HD5770_IH_SRC_D1 + i, 1);
            }
        }
    }
    timer_mod(s->vblank_timer, start + s->frame_period_ns);
}

/* ---------------------------------------------------------------- */
/* Hot plug                                                         */

static bool hpd_connected(ATIHD5770State *s, int n)
{
    return s->mon_hpd == n + 1;
}

static uint32_t hpd_reg(int n, uint32_t first)
{
    return first + n * EG_DC_HPD_STRIDE;
}

/*
 * A pin interrupts when its sense level matches the polarity bit (set:
 * connected), once per match; the driver flips the polarity to wait for
 * the next change.
 */
static void hpd_eval(ATIHD5770State *s, int n)
{
    uint32_t ctl = s->regs[hpd_reg(n, EG_DC_HPD1_INT_CONTROL) >> 2];
    bool cond = hpd_connected(s, n) == !!(ctl & EG_DC_HPDx_INT_POLARITY);

    if (cond && !s->hpd_cond[n] && !s->hpd_pending[n]) {
        s->hpd_pending[n] = true;
        trace_ati_hd5770_hpd(n + 1, hpd_connected(s, n), ctl);
        if (ctl & EG_DC_HPDx_INT_EN) {
            ati_hd5770_ih_push(s, HD5770_IH_SRC_HPD, n);
        }
    }
    s->hpd_cond[n] = cond;
}

/* the hot plug register a DC offset is, -1 for none */
static int hpd_for(uint32_t off, uint32_t *rel)
{
    if (off >= EG_DC_HPD1_INT_STATUS &&
        off < EG_DC_HPD1_INT_STATUS + ATI_HD5770_NUM_HPD * EG_DC_HPD_STRIDE) {
        int n = (off - EG_DC_HPD1_INT_STATUS) / EG_DC_HPD_STRIDE;

        *rel = off - n * EG_DC_HPD_STRIDE;
        return n;
    }
    return -1;
}

/* ---------------------------------------------------------------- */
/* DDC: GPIO bit-bang, SCL on bit 0 and SDA on bit 8 of A/EN/Y      */

static int ddc_for(uint32_t off)
{
    if (off >= EG_DC_GPIO_DDC_FIRST && off <= EG_DC_GPIO_DDC_LAST) {
        return (off - EG_DC_GPIO_DDC_FIRST) / 0x10;
    }
    return -1;
}

static void ddc_update(ATIHD5770State *s, int n)
{
    uint32_t base = EG_DC_GPIO_DDC_FIRST + n * 0x10;
    uint32_t a = s->regs[(base + 4) >> 2];
    uint32_t en = s->regs[(base + 8) >> 2];
    /* open drain: a pin driven low pulls its line low, else it floats */
    int scl = ((en & 1) && !(a & 1)) ? 0 : 1;
    int sda = ((en & 0x100) && !(a & 0x100)) ? 0 : 1;
    int sda_in;

    bitbang_i2c_set(&s->ddc_bb[n], BITBANG_I2C_SCL, scl);
    sda_in = bitbang_i2c_set(&s->ddc_bb[n], BITBANG_I2C_SDA, sda);
    s->ddc_y[n] = (scl ? 1 : 0) | (sda_in ? 0x100 : 0);
}

/* ---------------------------------------------------------------- */
/* DC_I2C engine: up to four transactions over a 256-byte buffer;   */
/* each takes the slave address from the buffer, then writes or     */
/* reads COUNT bytes there                                           */

static void i2c_run(ATIHD5770State *s)
{
    uint32_t ctl = s->regs[EG_DC_I2C_CONTROL >> 2];
    unsigned line = (ctl >> 8) & 7, ntr = ((ctl >> 20) & 3) + 1, t, i;
    unsigned idx = 0;
    uint32_t status = EG_DC_I2C_SW_DONE;
    I2CBus *bus = line < ATI_HD5770_NUM_DDC ? s->ddc_bus[line] : NULL;
    bool open = false;

    for (t = 0; t < ntr; t++) {
        uint32_t tr = s->regs[(EG_DC_I2C_TRANSACTION0 + 4 * t) >> 2];
        unsigned count = (tr >> 16) & 0xff;
        bool rd = tr & EG_DC_I2C_RW;
        uint8_t addr = s->i2c_buf[idx++ & 0xff];

        if (!bus || i2c_start_transfer(bus, addr >> 1, rd)) {
            status |= EG_DC_I2C_SW_STOPPED_ON_NACK;
            trace_ati_hd5770_i2c(line + 1, t, addr, count, rd, 1);
            break;
        }
        open = true;
        trace_ati_hd5770_i2c(line + 1, t, addr, count, rd, 0);
        for (i = 0; i < count; i++) {
            if (rd) {
                s->i2c_buf[idx++ & 0xff] = i2c_recv(bus);
            } else {
                i2c_send(bus, s->i2c_buf[idx++ & 0xff]);
            }
        }
        if (tr & EG_DC_I2C_STOP) {
            i2c_end_transfer(bus);
            open = false;
        }
    }
    if (open) {
        i2c_end_transfer(bus);
    }
    s->i2c_status = status;
    s->regs[EG_DC_I2C_INTERRUPT_CONTROL >> 2] |= EG_DC_I2C_SW_DONE_INT;
}

static void i2c_data_write(ATIHD5770State *s, uint32_t val)
{
    if (val & EG_DC_I2C_INDEX_WRITE) {
        s->i2c_index = (val >> 16) & 0xff;
    }
    if (!(val & EG_DC_I2C_DATA_RW)) {
        s->i2c_buf[s->i2c_index++] = (val >> 8) & 0xff;
    }
}

static uint32_t i2c_data_read(ATIHD5770State *s)
{
    uint32_t v = ((uint32_t)s->i2c_index << 16) |
                 ((uint32_t)s->i2c_buf[s->i2c_index] << 8);

    s->i2c_index++;
    return v;
}

/* ---------------------------------------------------------------- */
/* Gamma LUT                                                        */

const uint8_t (*ati_hd5770_dce4_lut(ATIHD5770State *s))[3]
{
    return (const uint8_t (*)[3])s->lut[0];
}

static void lut_set(ATIHD5770State *s, int crtc, unsigned i, uint8_t r,
                    uint8_t g, uint8_t b)
{
    uint8_t *e = s->lut[crtc][i];

    if (e[0] != r || e[1] != g || e[2] != b) {
        e[0] = r;
        e[1] = g;
        e[2] = b;
        if (crtc == 0) {
            s->lut_gen++;
        }
    }
}

/* ---------------------------------------------------------------- */
/* Scanout                                                          */

/*
 * CRTC 0's graphics surface as a mode for the display path. Linear
 * surfaces only; a tiled one is read as if linear.
 */
bool ati_hd5770_dce4_scanout(ATIHD5770State *s, ATIHD5770Mode *m)
{
    uint32_t ctl = CREG(s, 0, EG_GRPH_CONTROL);
    uint32_t xs = CREG(s, 0, EG_GRPH_X_START), xe = CREG(s, 0, EG_GRPH_X_END);
    uint32_t ys = CREG(s, 0, EG_GRPH_Y_START), ye = CREG(s, 0, EG_GRPH_Y_END);
    uint32_t vp = CREG(s, 0, EG_VIEWPORT_SIZE);
    uint32_t vps = CREG(s, 0, EG_VIEWPORT_START);
    uint32_t crtc = CREG(s, 0, EG_CRTC_CONTROL);
    uint64_t mc, off, x0, y0;

    memset(m, 0, sizeof(*m));
    if (!(crtc & EG_CRTC_MASTER_EN) ||
        (crtc & EG_CRTC_DISP_READ_REQUEST_DISABLE) ||
        !(CREG(s, 0, EG_GRPH_ENABLE) & 1) ||
        (CREG(s, 0, EG_CRTC_BLANK_CONTROL) & EG_CRTC_BLANK_DATA_EN)) {
        return false;
    }
    m->depth = ctl & 3;
    m->format = (ctl >> 8) & 7;
    switch (m->depth) {
    case 0:
        m->bpp = 1;
        break;
    case 1:
        if (m->format > 1) {
            return false;
        }
        m->bpp = 2;
        break;
    case 2:
        if (m->format > 1) {
            return false;
        }
        m->bpp = 4;
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
    if (m->width < 16 || m->height < 16 || m->width > 8192 ||
        m->height > 8192) {
        return false;
    }
    m->pitch = (CREG(s, 0, EG_GRPH_PITCH) & 0x7fff) * m->bpp;
    if (m->pitch < m->width * m->bpp) {
        return false;
    }
    mc = ((uint64_t)(CREG(s, 0, EG_GRPH_PRIMARY_SURFACE_ADDRESS_HIGH) & 0xff)
          << 32) | (CREG(s, 0, EG_GRPH_PRIMARY_SURFACE_ADDRESS) & ~0xffu);
    if (!ati_hd5770_mc_to_vram(s, mc, &off)) {
        return false;
    }
    x0 = (uint64_t)CREG(s, 0, EG_GRPH_SURFACE_OFFSET_X) + xs +
         ((vps >> 16) & 0x3fff);
    y0 = (uint64_t)CREG(s, 0, EG_GRPH_SURFACE_OFFSET_Y) + ys + (vps & 0x3fff);
    off += y0 * m->pitch + x0 * m->bpp;
    if (off + (uint64_t)m->pitch * m->height > s->vram_size) {
        return false;
    }
    m->fb_offset = off;
    return true;
}

/* ---------------------------------------------------------------- */
/* Cursor: up to 64x64 ARGB8888, 256-byte rows                      */

void ati_hd5770_dce4_cursor_apply(ATIHD5770State *s)
{
    uint32_t ctl = CREG(s, 0, EG_CUR_CONTROL);
    uint32_t size = CREG(s, 0, EG_CUR_SIZE);
    uint32_t pos = CREG(s, 0, EG_CUR_POSITION);
    uint32_t hot = CREG(s, 0, EG_CUR_HOT_SPOT);
    uint32_t w = ((size >> 16) & 0x7f) + 1, h = (size & 0x7f) + 1;
    uint32_t hx = (hot >> 16) & 0x7f, hy = hot & 0x7f;
    const uint8_t *vram = memory_region_get_ram_ptr(&s->vram);
    uint64_t mc = ((uint64_t)(CREG(s, 0, EG_CUR_SURFACE_ADDRESS_HIGH) & 0xff)
                   << 32) | CREG(s, 0, EG_CUR_SURFACE_ADDRESS);
    uint64_t off;
    uint32_t sum = 0, x, y;
    QEMUCursor *c;

    if (!s->con || (CREG(s, 0, EG_CUR_UPDATE) & EG_CURSOR_UPDATE_LOCK)) {
        return;
    }
    if (!(ctl & EG_CURSOR_EN) || w > 64 || h > 64 ||
        !(CREG(s, 0, EG_CRTC_CONTROL) & EG_CRTC_MASTER_EN) ||
        !ati_hd5770_mc_to_vram(s, mc, &off) ||
        off + 64 * 256 > s->vram_size) {
        if (s->hw_cursor_on) {
            s->hw_cursor_on = false;
            s->hw_cursor_sum = 0;
            qemu_console_set_mouse(s->con, 0, 0, false);
        }
        trace_ati_hd5770_cursor(ctl & EG_CURSOR_EN, 0, 0, w, h, 0);
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
    trace_ati_hd5770_cursor(1, s->hw_cursor_x, s->hw_cursor_y, w, h, off);
    qemu_console_set_mouse(s->con, s->hw_cursor_x, s->hw_cursor_y, true);
}

/* ---------------------------------------------------------------- */
/* Registers                                                        */

static uint32_t disp_status(ATIHD5770State *s, int n)
{
    uint32_t v = 0;

    if (s->vblank_occurred[n]) {
        v |= EG_LB_VBLANK_INTERRUPT;
    }
    if (s->vline_occurred[n]) {
        v |= EG_LB_VLINE_INTERRUPT;
    }
    if (s->hpd_pending[n]) {
        v |= EG_DC_HPD_INTERRUPT;
    }
    return v;
}

bool ati_hd5770_dce4_read(ATIHD5770State *s, uint32_t off, uint32_t *val)
{
    uint32_t rel;
    int crtc, n;

    for (n = 0; n < ATI_HD5770_NUM_CRTC; n++) {
        if (off == disp_int_status[n]) {
            *val = disp_status(s, n);
            return true;
        }
    }
    switch (off) {
    case EG_DC_I2C_SW_STATUS:
        *val = s->i2c_status;
        return true;
    case EG_DC_I2C_DATA:
        *val = i2c_data_read(s);
        return true;
    case EG_DC_I2C_CONTROL:
        *val = s->regs[off >> 2] & ~EG_DC_I2C_GO;
        return true;
    case EG_DC_GPIO_HPD_Y: {
        uint32_t v = 0;

        for (n = 0; n < ATI_HD5770_NUM_HPD; n++) {
            if (hpd_connected(s, n)) {
                v |= 1u << hpd_gpio_shift[n];
            }
        }
        *val = v;
        return true;
    }
    default:
        break;
    }
    if (off >= EG_DC_I2C_DDC1_HW_STATUS && off < EG_DC_I2C_DDC1_SPEED) {
        *val = 0;                   /* the hardware-driven DDC is idle */
        return true;
    }

    n = hpd_for(off, &rel);
    if (n >= 0 && rel == EG_DC_HPD1_INT_STATUS) {
        *val = (hpd_connected(s, n) ? EG_DC_HPDx_SENSE : 0) |
               (s->hpd_pending[n] ? EG_DC_HPDx_INT_STATUS : 0);
        return true;
    }

    n = ddc_for(off);
    if (n >= 0 && (off & 0xc) == 0xc) {
        *val = s->ddc_y[n];
        return true;
    }

    crtc = crtc_for(off, &rel);
    if (crtc >= 0) {
        switch (rel) {
        case EG_CRTC_CONTROL:
            *val = s->regs[off >> 2] & ~EG_CRTC_CURRENT_MASTER_EN_STATE;
            if (*val & EG_CRTC_MASTER_EN) {
                *val |= EG_CRTC_CURRENT_MASTER_EN_STATE;
            }
            return true;
        case EG_CRTC_STATUS:
            *val = crtc_in_vblank(s, crtc) ? EG_CRTC_V_BLANK : 0;
            return true;
        case EG_CRTC_STATUS_POSITION:
        case EG_CRTC_STATUS_HV_COUNT:
            *val = crtc_vcount(s, crtc) & 0x3fff;
            return true;
        case EG_CRTC_STATUS_FRAME_COUNT:
        case EG_CRTC_STATUS_VF_COUNT:
            *val = crtc_on(s, crtc) ? s->frame_count & 0xffffff : 0;
            return true;
        case EG_VBLANK_STATUS:
        case EG_VLINE_STATUS: {
            bool occ = rel == EG_VBLANK_STATUS ? s->vblank_occurred[crtc]
                                               : s->vline_occurred[crtc];
            uint32_t bit = rel == EG_VBLANK_STATUS ? EG_VBLANK_INT_MASK
                                                   : EG_VLINE_INT_MASK;

            *val = 0;
            if (occ) {
                *val |= EG_xBLANK_OCCURRED;
                if (CREG(s, crtc, EG_INT_MASK) & bit) {
                    *val |= EG_xBLANK_INT;
                }
            }
            if (crtc_in_vblank(s, crtc)) {
                *val |= EG_xBLANK_STAT;
            }
            return true;
        }
        case EG_GRPH_UPDATE:
            /* surface updates apply at once: never pending */
            *val = s->regs[off >> 2] & ~EG_GRPH_SURFACE_UPDATE_PENDING;
            return true;
        case EG_CUR_UPDATE:
            *val = s->regs[off >> 2] & ~EG_CURSOR_UPDATE_PENDING;
            return true;
        case EG_DC_LUT_AUTOFILL:
            *val = s->regs[off >> 2] | EG_DC_LUT_AUTOFILL_DONE;
            return true;
        case EG_DC_LUT_30_COLOR: {
            unsigned i = CREG(s, crtc, EG_DC_LUT_RW_INDEX) & 0xff;

            *val = ((uint32_t)s->lut[crtc][i][0] << 22) |
                   ((uint32_t)s->lut[crtc][i][1] << 12) |
                   ((uint32_t)s->lut[crtc][i][2] << 2);
            return true;
        }
        default:
            break;
        }
    }
    *val = s->regs[off >> 2];
    return true;
}

bool ati_hd5770_dce4_write(ATIHD5770State *s, uint32_t off, uint32_t val)
{
    uint32_t idx = off >> 2, rel;
    int crtc, n;

    switch (off) {
    case EG_DC_I2C_CONTROL:
        if (val & (EG_DC_I2C_SOFT_RESET | EG_DC_I2C_SW_STATUS_RESET)) {
            s->i2c_status = 0;
        }
        s->regs[idx] = val & ~EG_DC_I2C_GO;
        if (val & EG_DC_I2C_GO) {
            i2c_run(s);
        }
        return true;
    case EG_DC_I2C_INTERRUPT_CONTROL:
        s->regs[idx] = (s->regs[idx] & EG_DC_I2C_SW_DONE_INT) |
                       (val & ~(EG_DC_I2C_SW_DONE_INT | EG_DC_I2C_SW_DONE_ACK));
        if (val & EG_DC_I2C_SW_DONE_ACK) {
            s->regs[idx] &= ~EG_DC_I2C_SW_DONE_INT;
        }
        return true;
    case EG_DC_I2C_DATA:
        i2c_data_write(s, val);
        return true;
    case EG_DC_I2C_SW_STATUS:
    case EG_DC_GPIO_HPD_Y:
        return true;                /* read only */
    default:
        break;
    }

    n = hpd_for(off, &rel);
    if (n >= 0) {
        switch (rel) {
        case EG_DC_HPD1_INT_STATUS:
            return true;
        case EG_DC_HPD1_INT_CONTROL:
            s->regs[idx] = val & ~(EG_DC_HPDx_INT_ACK | EG_DC_HPDx_RX_INT_ACK);
            if (val & EG_DC_HPDx_INT_ACK) {
                s->hpd_pending[n] = false;
            }
            hpd_eval(s, n);
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

    crtc = crtc_for(off, &rel);
    if (crtc >= 0) {
        switch (rel) {
        case EG_VBLANK_STATUS:
            if (val & EG_xBLANK_ACK) {
                s->vblank_occurred[crtc] = false;
            }
            return true;
        case EG_VLINE_STATUS:
            if (val & EG_xBLANK_ACK) {
                s->vline_occurred[crtc] = false;
            }
            return true;
        case EG_CRTC_CONTROL:
            s->regs[idx] = val & ~EG_CRTC_CURRENT_MASTER_EN_STATE;
            if (!(val & EG_CRTC_MASTER_EN)) {
                s->vblank_occurred[crtc] = false;
                s->vline_occurred[crtc] = false;
            }
            trace_ati_hd5770_crtc(crtc, "CRTC_CONTROL", val);
            return true;
        case EG_CRTC_STATUS:
        case EG_CRTC_STATUS_POSITION:
        case EG_CRTC_STATUS_FRAME_COUNT:
        case EG_CRTC_STATUS_VF_COUNT:
        case EG_CRTC_STATUS_HV_COUNT:
            return true;            /* read only */
        case EG_DC_LUT_AUTOFILL:
            s->regs[idx] = val;
            if (val & 1) {
                unsigned i;

                for (i = 0; i < 256; i++) {
                    lut_set(s, crtc, i, i, i, i);
                }
            }
            return true;
        case EG_DC_LUT_RW_INDEX:
            s->regs[idx] = val & 0xff;
            s->lut_seq[crtc] = 0;
            return true;
        case EG_DC_LUT_30_COLOR: {
            unsigned i = CREG(s, crtc, EG_DC_LUT_RW_INDEX) & 0xff;

            lut_set(s, crtc, i, (val >> 22) & 0xff, (val >> 12) & 0xff,
                    (val >> 2) & 0xff);
            CREG(s, crtc, EG_DC_LUT_RW_INDEX) = (i + 1) & 0xff;
            return true;
        }
        case EG_DC_LUT_SEQ_COLOR: {
            /* 8 bits per write, R then G then B, then the next entry */
            unsigned i = CREG(s, crtc, EG_DC_LUT_RW_INDEX) & 0xff;
            uint8_t c[3];

            memcpy(c, s->lut[crtc][i], 3);
            c[s->lut_seq[crtc]] = val & 0xff;
            lut_set(s, crtc, i, c[0], c[1], c[2]);
            if (++s->lut_seq[crtc] == 3) {
                s->lut_seq[crtc] = 0;
                CREG(s, crtc, EG_DC_LUT_RW_INDEX) = (i + 1) & 0xff;
            }
            return true;
        }
        case EG_GRPH_LUT_10BIT_BYPASS_CONTROL:
        case EG_DC_LUT_CONTROL:
            s->regs[idx] = val;
            if (crtc == 0) {
                s->lut_gen++;
            }
            return true;
        case EG_CUR_CONTROL:
        case EG_CUR_SURFACE_ADDRESS:
        case EG_CUR_SURFACE_ADDRESS_HIGH:
        case EG_CUR_SIZE:
        case EG_CUR_POSITION:
        case EG_CUR_HOT_SPOT:
        case EG_CUR_UPDATE:
            s->regs[idx] = val;
            if (crtc == 0) {
                ati_hd5770_cursor_update(s);
            }
            return true;
        case EG_GRPH_ENABLE:
        case EG_GRPH_CONTROL:
        case EG_GRPH_PRIMARY_SURFACE_ADDRESS:
        case EG_GRPH_PITCH:
        case EG_GRPH_X_END:
        case EG_GRPH_Y_END:
        case EG_VIEWPORT_SIZE:
        case EG_CRTC_H_TOTAL:
        case EG_CRTC_V_TOTAL:
        case EG_CRTC_BLANK_CONTROL:
            s->regs[idx] = val;
            trace_ati_hd5770_crtc(crtc, "mode", (rel << 16) ^ val);
            return true;
        default:
            s->regs[idx] = val;
            return true;
        }
    }
    s->regs[idx] = val;
    return true;
}

/* ---------------------------------------------------------------- */
/* Init / reset                                                     */

/*
 * The card resets posted, as the VBIOS leaves it: CRTC 0 scanning out a
 * linear 32bpp 1024x768 surface at VRAM offset 0 (the firmware's mode;
 * the driver sets the EDID's), the hot plug pins enabled. The other CRTCs
 * are off.
 */
void ati_hd5770_dce4_reset(ATIHD5770State *s)
{
    uint32_t xres = POSTED_X, yres = POSTED_Y;
    unsigned i, j;

    s->frame_count = 0;
    memset(s->vblank_occurred, 0, sizeof(s->vblank_occurred));
    memset(s->vline_occurred, 0, sizeof(s->vline_occurred));
    memset(s->hpd_pending, 0, sizeof(s->hpd_pending));
    memset(s->hpd_cond, 0, sizeof(s->hpd_cond));
    memset(s->pixclk_10khz, 0, sizeof(s->pixclk_10khz));
    s->frame_period_ns = NANOSECONDS_PER_SECOND / FRAME_HZ;
    s->frame_start_ns = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
    timer_mod(s->vblank_timer, s->frame_start_ns + s->frame_period_ns);
    for (j = 0; j < ATI_HD5770_NUM_CRTC; j++) {
        for (i = 0; i < 256; i++) {
            s->lut[j][i][0] = s->lut[j][i][1] = s->lut[j][i][2] = i;
        }
        s->lut_seq[j] = 0;
    }
    s->lut_gen++;
    for (i = 0; i < ATI_HD5770_NUM_DDC; i++) {
        s->ddc_y[i] = 0x101;
    }
    memset(s->i2c_buf, 0, sizeof(s->i2c_buf));
    s->i2c_index = 0;
    s->i2c_status = 0;

    CREG(s, 0, EG_CRTC_CONTROL) = EG_CRTC_MASTER_EN;
    CREG(s, 0, EG_GRPH_ENABLE) = 1;
    CREG(s, 0, EG_GRPH_CONTROL) = 2;                /* ARGB8888, linear */
    CREG(s, 0, EG_GRPH_PITCH) = xres;
    CREG(s, 0, EG_GRPH_X_END) = xres;
    CREG(s, 0, EG_GRPH_Y_END) = yres;
    CREG(s, 0, EG_VIEWPORT_SIZE) = (xres << 16) | yres;
    CREG(s, 0, EG_CRTC_H_TOTAL) = xres + 160 - 1;
    CREG(s, 0, EG_CRTC_V_TOTAL) = yres + 30 - 1;
    CREG(s, 0, EG_CRTC_H_BLANK_START_END) = (xres + 80) << 16 | 80;
    CREG(s, 0, EG_CRTC_V_BLANK_START_END) = (yres + 27) << 16 | 27;
    for (i = 0; i < ATI_HD5770_NUM_CRTC; i++) {
        CREG(s, i, EG_MASTER_UPDATE_MODE) = 3;
    }
    for (i = 0; i < ATI_HD5770_NUM_HPD; i++) {
        s->regs[hpd_reg(i, EG_DC_HPD1_CONTROL) >> 2] = 1u << 28;
    }
}

void ati_hd5770_dce4_init(ATIHD5770State *s)
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

    s->vblank_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL, ati_hd5770_vblank, s);
    for (i = 0; i < ATI_HD5770_NUM_DDC; i++) {
        char name[32];

        snprintf(name, sizeof(name), "ati-hd5770.ddc%d", i + 1);
        s->ddc_bus[i] = i2c_init_bus(DEVICE(s), name);
        bitbang_i2c_init(&s->ddc_bb[i], s->ddc_bus[i]);
        if (i + 1 == s->mon_ddc) {
            I2CSlave *slv = i2c_slave_create_simple(s->ddc_bus[i],
                                                    TYPE_ATI_R580_EDID, 0x50);

            s->edid = ATI_R580_EDID(slv);
            /* its own serial: a monitor the guest has not seen on
             * another card, so it starts in the preferred mode */
            ati_r580_edid_build(s->edid->blob, s->xres, s->yres,
                                s->edid_info.maxx, s->edid_info.maxy,
                                MONITOR_SERIAL, hz);
        }
    }
}

void ati_hd5770_dce4_fini(ATIHD5770State *s)
{
    timer_free(s->vblank_timer);
    s->vblank_timer = NULL;
}
