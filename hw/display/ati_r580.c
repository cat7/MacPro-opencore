/*
 * QEMU ATI Radeon X1900 XT (R580) emulation -- see ati_r580_int.h.
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
#include "hw/pci/pcie.h"
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
#include "ati_r580_avivo.h"
#include "ati_r580_edid.h"
#include "trace.h"


/* ---------------------------------------------------------------- */
/* Display                                                          */

/* the frame buffer aperture maps VRAM linearly, without byte swapping */
unsigned ati_r580_aper_xor(ATIR580State *s, uint32_t off)
{
    return 0;
}

/* a host access of `size` bytes at VRAM offset `off`, lanes XOR `xr` */
static uint64_t ati_r580_aper_ld(ATIR580State *s, uint32_t off,
                                 unsigned size, unsigned xr)
{
    const uint8_t *vram = memory_region_get_ram_ptr(&s->vram);
    unsigned i;
    uint64_t val = 0;

    if (off + size > ATI_R580_VRAM_SIZE) {
        return 0;
    }
    for (i = 0; i < size; i++) {
        val |= (uint64_t)vram[(off + i) ^ xr] << (8 * i);
    }
    return val;
}

static void ati_r580_aper_st(ATIR580State *s, uint32_t off, uint64_t data,
                             unsigned size, unsigned xr)
{
    uint8_t *vram = memory_region_get_ram_ptr(&s->vram);
    unsigned i;

    if (off + size > ATI_R580_VRAM_SIZE) {
        return;
    }
    for (i = 0; i < size; i++) {
        vram[(off + i) ^ xr] = data >> (8 * i);
    }
    memory_region_set_dirty(&s->vram, off & ~3u, ((off + size + 3) & ~3u) -
                            (off & ~3u));
}

uint32_t ati_r580_vram_ld32(ATIR580State *s, uint32_t off)
{
    if (off + 4 > ATI_R580_VRAM_SIZE) {
        return 0;
    }
    return ldl_le_p((const uint8_t *)memory_region_get_ram_ptr(&s->vram) +
                    off);
}

static uint32_t *ati_r580_draw_row(const ATIR580DispJob *j, int y)
{
    return (uint32_t *)((uint8_t *)surface_data(j->ds) +
                        y * surface_stride(j->ds));
}

static void ati_r580_draw_8bpp(const ATIR580DispJob *j)
{
    const ATIR580Mode *mode = &j->mode;
    const uint8_t *src = j->vram + mode->fb_offset;
    const uint8_t (*pal)[3] = j->pal;
    uint32_t *dst;
    int x, y;

    for (y = 0; y < mode->height; y++) {
        dst = ati_r580_draw_row(j, y);
        for (x = 0; x < mode->width; x++) {
            uint8_t idx = src[x];
            dst[x] = 0xff000000u |
                     ((uint32_t)pal[idx][0] << 16) |
                     ((uint32_t)pal[idx][1] << 8) |
                     pal[idx][2];
        }
        src += mode->pitch;
    }
}

static bool ati_r580_palette_identity(const uint8_t (*pal)[3])
{
    int i;

    for (i = 0; i < 256; i++) {
        if (pal[i][0] != i || pal[i][1] != i || pal[i][2] != i) {
            return false;
        }
    }
    return true;
}

/*
 * 15/16bpp look each component up in the palette too: a 5-bit one at
 * entry c << 3, a 6-bit one at c << 2. With the identity ramp the plain
 * bit expansion is kept.
 */
static void ati_r580_draw_16bpp(const ATIR580DispJob *j, bool rgb565)
{
    const ATIR580Mode *mode = &j->mode;
    const uint8_t *src = j->vram + mode->fb_offset;
    const uint8_t (*pal)[3] = j->pal;
    bool lut = !ati_r580_palette_identity(pal);
    uint32_t *dst;
    int x, y;

    for (y = 0; y < mode->height; y++) {
        dst = ati_r580_draw_row(j, y);
        for (x = 0; x < mode->width; x++) {
            uint16_t pixel = ((uint16_t)src[2 * x + 1] << 8) |
                             src[2 * x];
            uint8_t r, g, b;

            if (rgb565) {
                r = ((pixel >> 11) & 0x1f) << 3;
                g = ((pixel >> 5) & 0x3f) << 2;
                b = (pixel & 0x1f) << 3;
            } else {                        /* RGB555 */
                r = ((pixel >> 10) & 0x1f) << 3;
                g = ((pixel >> 5) & 0x1f) << 3;
                b = (pixel & 0x1f) << 3;
            }
            if (lut) {
                r = pal[r][0];
                g = pal[g][1];
                b = pal[b][2];
            } else {
                g |= rgb565 ? g >> 6 : g >> 5;
                r |= r >> 5;
                b |= b >> 5;
            }
            dst[x] = 0xff000000u | ((uint32_t)r << 16) |
                     ((uint32_t)g << 8) | b;
        }
        src += mode->pitch;
    }
}

static void ati_r580_draw_32bpp(const ATIR580DispJob *j)
{
    const ATIR580Mode *mode = &j->mode;
    const uint8_t *src = j->vram + mode->fb_offset;
    const uint8_t (*pal)[3] = j->pal;
    uint32_t *dst;
    int x, y;

    for (y = 0; y < mode->height; y++) {
        dst = ati_r580_draw_row(j, y);
        for (x = 0; x < mode->width; x++) {
            /*
             * Chip-native little-endian: B,G,R,X in VRAM. The DAC's
             * palette RAM acts as a per-channel gamma LUT in direct-
             * colour modes too -- OS X renders in linear light and
             * relies on the ramp it loads via PALETTE_30_DATA (the
             * accelerated desktop backdrop arrives as (8,21,55) and
             * only becomes the Aqua blue through the LUT).
             */
            dst[x] = 0xff000000u |
                     ((uint32_t)pal[src[4 * x + 2]][0] << 16) |
                     ((uint32_t)pal[src[4 * x + 1]][1] << 8) |
                     pal[src[4 * x]][2];
        }
        src += mode->pitch;
    }
}

static void ati_r580_draw_24bpp(const ATIR580DispJob *j)
{
    const ATIR580Mode *mode = &j->mode;
    const uint8_t *src = j->vram + mode->fb_offset;
    const uint8_t (*pal)[3] = j->pal;
    uint32_t *dst;
    int x, y;

    for (y = 0; y < mode->height; y++) {
        dst = ati_r580_draw_row(j, y);
        for (x = 0; x < mode->width; x++) {
            /* chip-native little-endian: B,G,R in VRAM; palette RAM
             * doubles as the per-channel gamma LUT, as in 32bpp */
            dst[x] = 0xff000000u |
                     ((uint32_t)pal[src[3 * x + 2]][0] << 16) |
                     ((uint32_t)pal[src[3 * x + 1]][1] << 8) |
                     pal[src[3 * x]][2];
        }
        src += mode->pitch;
    }
}

/* convert the job's frame into its surface; touches nothing else */
static void ati_r580_draw(const ATIR580DispJob *j)
{
    switch (j->mode.pix_width) {
    case R580_PIX_WIDTH_8BPP:
        ati_r580_draw_8bpp(j);
        break;
    case R580_PIX_WIDTH_15BPP:
        ati_r580_draw_16bpp(j, false);
        break;
    case R580_PIX_WIDTH_16BPP:
        ati_r580_draw_16bpp(j, true);
        break;
    case R580_PIX_WIDTH_24BPP:
        ati_r580_draw_24bpp(j);
        break;
    case R580_PIX_WIDTH_32BPP:
        ati_r580_draw_32bpp(j);
        break;
    default:
        break;
    }
}

/*
 * Consume the VRAM dirty bitmap once per refresh and return the snapshot
 * so the caller can ask whether the range it is about to display changed.
 */
static DirtyBitmapSnapshot *ati_r580_take_dirty(ATIR580State *s)
{
    DirtyBitmapSnapshot *snap;

    /*
     * Not under gl_tex_lock, which a draw holds for as long as it runs:
     * the refresh holds the BQL, and waiting for a draw there stalls
     * every vCPU's device access and every main-loop timer with it.
     * Nor under any lock the draw path waits for: under TCG the
     * snapshot waits for a vCPU, which may be waiting for the command
     * processor thread.
     *
     * The clear is the only thing that resets DIRTY_MEMORY_VGA, and the
     * decoded-texture cache reads those bits to know whether the guest
     * CPU wrote over a texture, so the pages it found are handed over --
     * the sole record of what was set. scan_seq is odd from before the
     * clear until they are, and a lookup that finds it odd decodes
     * afresh.
     */
    qatomic_inc(&s->scan_seq);
    snap = memory_region_snapshot_and_clear_dirty(&s->vram, 0,
                                                   ATI_R580_VRAM_SIZE,
                                                   DIRTY_MEMORY_VGA);
    qemu_mutex_lock(&s->scan_lock);
    ati_r580_gl_epoch(s, snap);
    qatomic_set(&s->scan_pending, true);
    qemu_mutex_unlock(&s->scan_lock);
    qatomic_inc(&s->scan_seq);
    return snap;
}

/*
 * ADMITTING A TEXTURE RANGE to the decoded-texture cache: take
 * responsibility for the VGA dirty bits over it, so that from here on
 * any set bit is a write that happened since. See "TEXTURE CACHE
 * LIFETIME" in ati_r580_3d.c for why admission has to be a clean slate
 * rather than a recorded baseline.
 *
 * Those bits belong to the scanout, whose redraw test reads only the
 * DISPLAYED range, so a range overlapping it is refused outright --
 * rounded outwards by a page, because the clear below is page-granular.
 *
 * Returns false when the range may not be claimed, and the caller then
 * decodes without caching.
 */
bool ati_r580_gl_admit(ATIR580State *s, uint32_t off, uint32_t len)
{
    uint64_t pg = qemu_target_page_size();
    uint64_t lo = off & ~(pg - 1);
    uint64_t hi = (off + len + pg - 1) & ~(pg - 1);
    uint64_t fb = qatomic_read(&s->scan_fb);
    uint64_t fb_off = fb >> 32, fb_len = (uint32_t)fb;

    if (hi > ATI_R580_VRAM_SIZE) {
        return false;
    }
    if (fb_len && lo < fb_off + fb_len && hi > fb_off) {
        return false;               /* those bits decide the redraw */
    }
    memory_region_reset_dirty(&s->vram, lo, hi - lo, DIRTY_MEMORY_VGA);
    return true;
}

/*
 * The refresh's decisions, BQL held: consume the dirty bitmap, settle
 * the mode, give the console a surface of its size, and say whether the
 * picture changed. Returns false when there is nothing to show at all.
 */
static bool ati_r580_refresh_prepare(ATIR580State *s, ATIR580Mode *out,
                                     bool *out_redraw)
{
    ATIR580Mode mode;
    DisplaySurface *ds;
    DirtyBitmapSnapshot *snap;
    bool valid, redraw;
    uint64_t fb_len;

    snap = ati_r580_take_dirty(s);
    valid = ati_r580_avivo_scanout(s, &mode);
    trace_ati_r580_update(mode.width, mode.height, mode.bpp, valid,
                          mode.fb_offset);
    if (!valid) {
        /* D1 off or blanked: the last frame stays */
        g_free(snap);
        return false;
    }
    memcpy(s->palette, ati_r580_avivo_lut(s), sizeof(s->palette));
    if (s->lut_drawn != s->lut_gen) {
        s->lut_drawn = s->lut_gen;
        s->force_redraw = true;
    }

    /*
     * The surface is reallocated only when its size changes; a new
     * offset or pitch (a flip) only changes where the refresh reads.
     */
    ds = qemu_console_surface(s->con);
    redraw = s->force_redraw;
    if (!ds || surface_width(ds) != (int)mode.width ||
        surface_height(ds) != (int)mode.height) {
        trace_ati_r580_surface_realloc(ds ? surface_width(ds) : 0,
                                          ds ? surface_height(ds) : 0,
                                          mode.width, mode.height, mode.bpp,
                                          mode.fb_offset, mode.pitch);
        ds = qemu_create_displaysurface(mode.width, mode.height);
        qemu_console_set_surface(s->con, ds);
        redraw = true;
    }
    /*
     * Only redraw -- and only hand the UI a new frame -- when something
     * that shapes the picture changed: the mode/framebuffer geometry, the
     * palette (force_redraw), or the displayed VRAM range itself (CPU
     * stores through either aperture, engine-drawn pixels and DMA all
     * land in the dirty bitmap). Re-pushing an identical 1024x768 frame
     * on every refresh tick was pure host load, and on macOS the stream
     * of redundant full-window redraws is what the intermittent darker
     * frames ("screen flicker") were made of.
     */
    if (memcmp(&s->mode, &mode, sizeof(mode)) != 0) {
        redraw = true;
    }
    fb_len = (uint64_t)mode.pitch * mode.height;
    if (fb_len && (uint64_t)mode.fb_offset + fb_len <= ATI_R580_VRAM_SIZE &&
        memory_region_snapshot_get_dirty(&s->vram, snap, mode.fb_offset,
                                         fb_len)) {
        redraw = true;
    }
    g_free(snap);
    s->mode = mode;
    /* ati_r580_gl_admit() reads it */
    qatomic_set(&s->scan_fb, (uint64_t)mode.fb_offset << 32 |
                (fb_len <= UINT32_MAX ? fb_len : UINT32_MAX));
    *out = mode;
    *out_redraw = redraw;
    return true;
}

static void ati_r580_vram_peek(ATIR580State *s, const ATIR580Mode *mode)
{
    if (trace_event_get_state_backends(TRACE_ATI_R580_VRAM_PEEK)) {
        uint8_t *vp = (uint8_t *)memory_region_get_ram_ptr(&s->vram);
        uint32_t o5 = mode->fb_offset + 5 * mode->pitch;
        uint32_t o30 = mode->fb_offset + 30 * mode->pitch;
        trace_ati_r580_vram_peek(5, o5, ldl_le_p(vp + o5),
                                    ldl_le_p(vp + o5 + 4),
                                    ldl_le_p(vp + o5 + 8),
                                    ldl_le_p(vp + o5 + 12));
        trace_ati_r580_vram_peek(30, o30, ldl_le_p(vp + o30),
                                    ldl_le_p(vp + o30 + 4),
                                    ldl_le_p(vp + o30 + 8),
                                    ldl_le_p(vp + o30 + 12));
    }
}

/*
 * ASYNC DISPLAY (async-display=on). The refresh's per-pixel work -- the
 * conversion of the scanned-out VRAM into the console's 32bpp surface --
 * runs on a worker thread; the main loop keeps everything that decides
 * or publishes:
 *
 * - main loop, BQL: the GL target release, the dirty-bitmap snapshot,
 *   the mode and framebuffer choice, surface reallocation, the cursor;
 *   then a job -- mode, surface, xor, a copy of the palette -- that the
 *   worker converts straight into the console's surface;
 * - worker: reads VRAM and the job, writes the surface, nothing else;
 * - hand-over, a bottom half on the main loop: the cursor and
 *   dpy_gfx_update, then any refresh that came in meanwhile.
 *
 * One job at most is in flight. A refresh tick that finds the worker
 * busy takes no snapshot -- the dirty bits it would have consumed stay
 * in the bitmap -- and the hand-over runs that tick when the frame is
 * in, so no write is lost and a busy worker only lowers the frame rate.
 * The surface is replaced only by the refresh, and only while the worker
 * is idle, so the surface a job names lives until its hand-over. A
 * guest write landing during the conversion is in the next snapshot,
 * so a frame can be torn for one refresh at most -- as it could before,
 * since vCPUs and the command processor never stopped for the refresh,
 * and a UI that draws on its own thread could always read a surface
 * mid-conversion.
 *
 * LOCK ORDER: BQL -> disp_lock, and disp_lock is a leaf: nothing else
 * is taken under it. The worker takes neither the BQL nor a claim nor
 * any other lock while converting, and waits for nothing but its
 * condition variable, so the main loop may wait for it with the BQL
 * held (unrealize); it never does so on a refresh.
 */
static void ati_r580_disp_post(ATIR580State *s, const ATIR580Mode *mode)
{
    ATIR580DispJob *j = &s->disp_job;

    j->mode = *mode;
    j->ds = qemu_console_surface(s->con);
    j->vram = memory_region_get_ram_ptr(&s->vram);
    memcpy(j->palette, s->palette, sizeof(j->palette));
    j->pal = (const uint8_t (*)[3])j->palette;
    qemu_mutex_lock(&s->disp_lock);
    s->disp_state = ATI_R580_DISP_QUEUED;
    qemu_cond_signal(&s->disp_cond);
    qemu_mutex_unlock(&s->disp_lock);
}

static bool ati_r580_update_async(ATIR580State *s)
{
    ATIR580Mode mode;
    bool redraw;
    unsigned state;

    qemu_mutex_lock(&s->disp_lock);
    state = s->disp_state;
    qemu_mutex_unlock(&s->disp_lock);
    if (state != ATI_R580_DISP_IDLE) {
        s->disp_tick_missed = true;
        s->disp_deferred++;
        ati_r580_cursor_update(s);
        return false;
    }
    /* see ati_r580_update_display() */
    ati_r580_gl_release(s, R580_GLR_SCANOUT);
    if (!ati_r580_refresh_prepare(s, &mode, &redraw)) {
        return true;
    }
    if (!redraw) {
        ati_r580_cursor_update(s);
        return true;
    }
    s->force_redraw = false;
    ati_r580_vram_peek(s, &mode);
    ati_r580_disp_post(s, &mode);
    return false;
}

/* the frame is in: publish it, then run a tick that found the worker busy */
static void ati_r580_disp_handover(void *opaque)
{
    ATIR580State *s = opaque;

    qemu_mutex_lock(&s->disp_lock);
    if (s->disp_state != ATI_R580_DISP_DONE) {
        qemu_mutex_unlock(&s->disp_lock);
        return;
    }
    s->disp_state = ATI_R580_DISP_IDLE;
    qemu_mutex_unlock(&s->disp_lock);

    if (qemu_console_surface(s->con) == s->disp_job.ds) {
        s->disp_frames++;
        ati_r580_avivo_cursor_apply(s);
        qemu_console_update_full(s->con);
    } else {
        s->force_redraw = true;
    }
    if (s->disp_tick_missed) {
        s->disp_tick_missed = false;
        if (!ati_r580_update_async(s)) {
            return;                 /* that job's hand-over finishes it */
        }
    }
    qemu_console_hw_update_done(s->con);
}

static void *ati_r580_disp_thread(void *opaque)
{
    ATIR580State *s = opaque;

    qemu_mutex_lock(&s->disp_lock);
    for (;;) {
        while (s->disp_state != ATI_R580_DISP_QUEUED && !s->disp_quit) {
            qemu_cond_wait(&s->disp_cond, &s->disp_lock);
        }
        if (s->disp_quit) {
            break;
        }
        s->disp_state = ATI_R580_DISP_RUNNING;
        qemu_mutex_unlock(&s->disp_lock);

        ati_r580_draw(&s->disp_job);

        qemu_mutex_lock(&s->disp_lock);
        s->disp_state = ATI_R580_DISP_DONE;
        qemu_bh_schedule(s->disp_bh);
    }
    qemu_mutex_unlock(&s->disp_lock);
    return NULL;
}

static void ati_r580_disp_init(ATIR580State *s)
{
    s->disp_on = s->async_display == ON_OFF_AUTO_ON ||
                 (s->async_display == ON_OFF_AUTO_AUTO && !qtest_enabled());
    if (!s->disp_on) {
        return;
    }
    qemu_mutex_init(&s->disp_lock);
    qemu_cond_init(&s->disp_cond);
    s->disp_state = ATI_R580_DISP_IDLE;
    s->disp_bh = qemu_bh_new(ati_r580_disp_handover, s);
    qemu_thread_create(&s->disp_thread, "ati-r580-disp",
                       ati_r580_disp_thread, s, QEMU_THREAD_JOINABLE);
}

/* waits for a frame in flight; the worker needs no lock to finish it */
static void ati_r580_disp_fini(ATIR580State *s)
{
    if (!s->disp_on) {
        return;
    }
    qemu_mutex_lock(&s->disp_lock);
    s->disp_quit = true;
    qemu_cond_signal(&s->disp_cond);
    qemu_mutex_unlock(&s->disp_lock);
    qemu_thread_join(&s->disp_thread);
    qemu_bh_delete(s->disp_bh);
    s->disp_bh = NULL;
    qemu_cond_destroy(&s->disp_cond);
    qemu_mutex_destroy(&s->disp_lock);
    s->disp_on = false;
}

static bool ati_r580_update_display(void *opaque)
{
    ATIR580State *s = opaque;
    ATIR580DispJob j;
    ATIR580Mode mode;
    bool redraw;

    if (s->disp_on) {
        return ati_r580_update_async(s);
    }
    /*
     * Before the dirty snapshot, not after: a flush marks the rows it
     * writes, and those marks are how this function decides to redraw
     * at all. Releasing later would let a frame the GPU rendered sit
     * unnoticed until something else dirtied the same page.
     */
    ati_r580_gl_release(s, R580_GLR_SCANOUT);
    if (!ati_r580_refresh_prepare(s, &mode, &redraw)) {
        return true;
    }
    if (!redraw) {
        ati_r580_cursor_update(s);
        return true;
    }
    s->force_redraw = false;
    ati_r580_vram_peek(s, &mode);
    j.mode = mode;
    j.ds = qemu_console_surface(s->con);
    j.vram = memory_region_get_ram_ptr(&s->vram);
    j.pal = (const uint8_t (*)[3])s->palette;
    ati_r580_draw(&j);
    ati_r580_avivo_cursor_apply(s);
    qemu_console_update_full(s->con);

    return true;
}

static const GraphicHwOps ati_r580_gfx_ops = {
    .gfx_update = ati_r580_update_display,
};

/* ---------------------------------------------------------------- */
/*
 * The interrupt line: GEN_INT_STATUS & GEN_INT_CNTL, or a display
 * interrupt (vblank, hot plug) that DxMODE_INT_MASK lets through.
 */
void ati_r580_update_irq(ATIR580State *s)
{
    uint32_t pending = s->regs[R580_GEN_INT_STATUS >> 2] &
                       s->regs[R580_GEN_INT_CNTL >> 2] &
                       R580_GEN_INT_ACK_MASK;

    pci_set_irq(PCI_DEVICE(s), pending || ati_r580_avivo_irq_pending(s));
}

/* ---------------------------------------------------------------- */
/* R5xx bus interface: PLL, MC and PCIE index spaces, configuration */

static uint32_t ati_r580_pll_read(ATIR580State *s, unsigned idx)
{
    uint32_t val = s->plls[idx];

    if (idx == R5XX_PLL_CHIP_ID) {
        /* chip id strap */
        val = (s->device_id & 0xffff) | ((s->revision & 0xf) << 24);
    }
    trace_ati_r580_pll_read(idx, val);
    return val;
}

static uint32_t ati_r580_mc_ind_read(ATIR580State *s, unsigned idx)
{
    switch (idx) {
    case R5XX_MC_STATUS:
        return R5XX_MC_STATUS_IDLE | R5XX_MC_STATUS_IDLE_RV515;
    case R5XX_MC_FB_LOCATION:
        return s->regs[R580_MC_FB_LOCATION >> 2];
    case R5XX_MC_AGP_LOCATION:
        return s->regs[R580_MC_AGP_LOCATION >> 2];
    default:
        return s->mc_ind[idx];
    }
}

static void ati_r580_mc_ind_write(ATIR580State *s, unsigned idx,
                                  uint32_t val)
{
    trace_ati_r580_mc_ind(idx, val);
    switch (idx) {
    case R5XX_MC_STATUS:
        break;
    case R5XX_MC_FB_LOCATION:
        /* the engines route card addresses through the R300 window */
        s->regs[R580_MC_FB_LOCATION >> 2] = val;
        s->force_redraw = true;
        break;
    case R5XX_MC_AGP_LOCATION:
        s->regs[R580_MC_AGP_LOCATION >> 2] = val;
        break;
    default:
        s->mc_ind[idx] = val;
        break;
    }
}

static void ati_r580_pcie_write(ATIR580State *s, unsigned idx, uint32_t val)
{
    trace_ati_r580_pcie_ind(idx, val);
    switch (idx) {
    case R5XX_PCIE_TX_GART_CNTL:
        s->pcie_ind[idx] = val & ~R5XX_PCIE_TX_GART_INVALIDATE;
        break;
    case R5XX_PCIE_TX_GART_ERROR:
        s->pcie_ind[idx] = 0;
        break;
    default:
        s->pcie_ind[idx] = val;
        break;
    }
}

/* the BAR 0 bus address: the driver maps VRAM from it */
static uint32_t ati_r580_aper_base(ATIR580State *s)
{
    pcibus_t bar = pci_get_bar_addr(PCI_DEVICE(s), 0);

    return bar == PCI_BAR_UNMAPPED ? 0 : bar;
}

/* legacy display registers the R5xx keeps only as storage */
static bool ati_r580_legacy_display_reg(uint32_t base)
{
    return (base >= R580_CRTC_GEN_CNTL && base <= R580_GPIO_MONIDB) ||
           (base >= R580_I2C_CNTL_0 && base <= R580_PALETTE_30_DATA) ||
           (base >= R580_CRTC_H_TOTAL_DISP && base < R5XX_VGA_RENDER_CONTROL) ||
           base == R580_CRTC2_GEN_CNTL || base == R580_MC_FB_LOCATION ||
           base == R580_MC_AGP_LOCATION;
}

/*
 * The R5xx registers in front of the R300 register file: true when `base`
 * is answered here.
 */
static bool ati_r580_front_read(ATIR580State *s, uint32_t base, uint32_t *val)
{
    if (ati_r580_display_reg(base)) {
        if (base >= R5XX_ATOM_MAILBOX) {
            *val = s->regs[base >> 2];
            return true;
        }
        return ati_r580_avivo_read(s, base, val);
    }
    switch (base) {
    case R580_CLOCK_CNTL_DATA:
        *val = ati_r580_pll_read(s, s->regs[R580_CLOCK_CNTL_INDEX >> 2] &
                                    R5XX_PLL_ADDR_MASK);
        return true;
    case R5XX_PCIE_DATA:
        *val = s->pcie_ind[s->regs[R5XX_PCIE_INDEX >> 2] & 0xff];
        return true;
    case R5XX_MC_IND_DATA:
        *val = ati_r580_mc_ind_read(s, s->regs[R5XX_MC_IND_INDEX >> 2] &
                                       R5XX_MC_IND_ADDR_MASK);
        return true;
    case R580_GEN_INT_STATUS:
        *val = s->regs[base >> 2] &
               ~(R580_CRTC_VBLANK_INT | R580_CRTC_VSYNC_INT);
        if (ati_r580_avivo_irq_pending(s)) {
            *val |= R5XX_GEN_INT_DISPLAY_STAT;
        }
        return true;
    case R580_CONFIG_MEMSIZE:
        *val = ATI_R580_VRAM_SIZE;
        return true;
    case R580_CONFIG_APER_0_BASE:
    case R580_CONFIG_APER_1_BASE:
        *val = ati_r580_aper_base(s);
        return true;
    case R580_CONFIG_APER_SIZE:
        *val = ATI_R580_APER_SIZE;
        return true;
    case R580_CONFIG_REG_APER_SIZE:
        *val = ATI_R580_MMIO_SIZE;
        return true;
    case R5XX_MC_STATUS_LEGACY:
        *val = 1u << 2;             /* MC idle */
        return true;
    case R5XX_P1PLL_CNTL:
    case R5XX_P2PLL_CNTL:
        /* the pixel PLLs lock at once */
        *val = s->regs[base >> 2] | R5XX_PPLL_LOCKED;
        return true;
    case R5XX_PCIE_INDEX:
    case R5XX_MC_IND_INDEX:
        *val = s->regs[base >> 2];
        return true;
    default:
        break;
    }
    if (ati_r580_legacy_display_reg(base)) {
        *val = s->regs[base >> 2];
        return true;
    }
    return false;
}

static bool ati_r580_front_write(ATIR580State *s, uint32_t base, uint32_t val)
{
    if (ati_r580_display_reg(base)) {
        if (base >= R5XX_ATOM_MAILBOX) {
            if (!ati_r580_atom_write(s, base, val)) {
                s->regs[base >> 2] = val;
            }
            return true;
        }
        return ati_r580_avivo_write(s, base, val);
    }
    switch (base) {
    case R580_CLOCK_CNTL_DATA:
        if (s->regs[R580_CLOCK_CNTL_INDEX >> 2] & R5XX_PLL_WR_EN) {
            unsigned idx = s->regs[R580_CLOCK_CNTL_INDEX >> 2] &
                           R5XX_PLL_ADDR_MASK;

            s->plls[idx] = val;
            trace_ati_r580_pll_write(idx, val);
        }
        return true;
    case R5XX_PCIE_DATA:
        ati_r580_pcie_write(s, s->regs[R5XX_PCIE_INDEX >> 2] & 0xff, val);
        return true;
    case R5XX_MC_IND_DATA:
        if (s->regs[R5XX_MC_IND_INDEX >> 2] & R5XX_MC_IND_WR_EN) {
            ati_r580_mc_ind_write(s, s->regs[R5XX_MC_IND_INDEX >> 2] &
                                     R5XX_MC_IND_ADDR_MASK, val);
        }
        return true;
    case R580_CONFIG_MEMSIZE:
    case R580_CONFIG_APER_0_BASE:
    case R580_CONFIG_APER_1_BASE:
    case R580_CONFIG_APER_SIZE:
    case R580_CONFIG_REG_APER_SIZE:
    case R580_MC_FB_LOCATION:
    case R580_MC_AGP_LOCATION:
        return true;                /* read only here */
    case R5XX_PCIE_INDEX:
    case R5XX_MC_IND_INDEX:
        s->regs[base >> 2] = val;
        return true;
    default:
        break;
    }
    if (ati_r580_legacy_display_reg(base)) {
        s->regs[base >> 2] = val;
        return true;
    }
    return false;
}

static uint32_t ati_r580_reg_read32(ATIR580State *s, uint32_t base)
{
    PCIDevice *dev = PCI_DEVICE(s);
    uint32_t val = s->regs[base >> 2];

    if (ati_r580_front_read(s, base, &val)) {
        return val;
    }
    switch (base) {
    case R300_ZB_ZPASS_DATA:
        val = ati_r580_zpass_read(s);
        break;
    case R580_MM_DATA:
        val = 0;
        if (s->regs[R580_MM_INDEX >> 2] & R580_MM_APER) {
            uint32_t off = s->regs[R580_MM_INDEX >> 2] & 0x7ffffffc;

            if (off + 4 <= ATI_R580_VRAM_SIZE) {
                ati_r580_gl_touch(s, off, 4);
                val = ldl_le_p((uint8_t *)memory_region_get_ram_ptr(&s->vram)
                               + off);
            }
        } else if ((s->regs[R580_MM_INDEX >> 2] & 0xfffc) > R580_MM_DATA) {
            /* MM_INDEX 0 and 4 would point back at the pair itself */
            val = ati_r580_reg_read32(s,
                s->regs[R580_MM_INDEX >> 2] & 0xfffc);
        }
        break;
    case R580_CP_RB_BASE:
        val = s->pm4_buffer_addr;
        break;
    case R580_CP_RB_CNTL:
        val = s->pm4_buffer_cntl;
        break;
    case R580_CP_RB_RPTR:
        val = qatomic_read(&s->pm4_rptr);
        break;
    case R580_CP_RB_WPTR:
        val = qatomic_read(&s->pm4_wptr);
        break;
    case R580_PM4_MICROCODE_DATAH:
        val = s->pm4_microcode[s->pm4_ucode_raddr][0];
        break;
    case R580_PM4_MICROCODE_DATAL:
        val = s->pm4_microcode[s->pm4_ucode_raddr][1];
        s->pm4_ucode_raddr = (s->pm4_ucode_raddr + 1) &
                             (R580_PM4_MICROCODE_WORDS - 1);
        break;
    case R580_CP_STAT:
    case R580_CP_CSQ_STAT:
        /* command processor idle, queues empty */
        val = 0;
        break;
    case R580_RBBM_STATUS:
        /* FIFO fully free; GUI_ACTIVE while the command processor runs */
        val = 0x40 & R580_RBBM_FIFOCNT_MASK;
        if (qatomic_read(&s->engine_busy)) {
            val |= 1u << 31;
        }
        break;
    case R580_RB2D_DSTCACHE_CTLSTAT:
    case R580_RB3D_DSTCACHE_CTLSTAT:
    case R580_RB3D_ZCACHE_CTLSTAT:
        /* cache flush/idle handshakes: never busy (bit 31) */
        val = s->regs[base >> 2] & 0x7fffffff;
        break;
    case R580_CONFIG_REG_1_BASE:
        val = pci_get_long(dev->config + PCI_BASE_ADDRESS_2) &
              PCI_BASE_ADDRESS_MEM_MASK;
        break;
    case R580_CONFIG_XSTRAP:
        val = R580_XSTRAP_ADDIN_CARD;
        break;
    case R580_GEN_STATUS:
    case R580_CONFIG_BONDS:
        val = 0;
        break;
    case R580_SW_SEMAPHORE:
        /* report all 8 semaphores free (bit reads as acquired-ok) */
        val = 0xff;
        break;
    case R580_MEM_STR_CNTL:
        val = s->regs[base >> 2];
        break;
    case R580_PC_NGUI_CTLSTAT:
        /* pixel cache idle: BUSY (bit 31) clear */
        val = s->regs[base >> 2] & 0x3fffffff;
        break;
    case R580_GUI_STAT:
        /* all 64 command FIFO entries free; GUI_ACTIVE while it runs */
        val = 0x40;
        if (qatomic_read(&s->engine_busy)) {
            val |= 1u << 31;
        }
        break;
    case R580_DST_OFFSET:
        val = s->dst_offset_reg;
        break;
    case R580_DST_PITCH:
        val = s->dst_pitch_reg | (s->dst_tile_reg << 16);
        break;
    case R580_DST_WIDTH:
        val = s->dst_width;
        break;
    case R580_DST_HEIGHT:
        val = s->dst_height;
        break;
    case R580_SRC_X:
        val = s->src_x;
        break;
    case R580_SRC_Y:
        val = s->src_y;
        break;
    case R580_DST_X:
        val = s->dst_x;
        break;
    case R580_DST_Y:
        val = s->dst_y;
        break;
    case R580_DP_GUI_MASTER_CNTL:
        /* aliases fields from DP_MIX and DP_DATATYPE -- see the write case */
        val = s->dp_gui_master_cntl |
              ((s->dp_datatype & R580_DP_BRUSH_DATATYPE) >> 4) |
              ((s->dp_datatype & R580_DP_DST_DATATYPE) << 8) |
              ((s->dp_datatype & R580_DP_SRC_DATATYPE) >> 4) |
              (s->dp_mix & R580_DP_ROP3) |
              ((s->dp_mix & R580_DP_SRC_SOURCE) << 16);
        break;
    /*
     * The packed PITCH_OFFSET form and the separate PITCH/OFFSET
     * registers are two views of one piece of state on real silicon, so
     * a read through either has to see what was written through the
     * other. Mac OS X programs the separate registers exclusively, which
     * left the packed ones reading a permanent zero.
     */
    case R580_SRC_PITCH_OFFSET:
        val = (s->src_offset_reg >> 5) |
              (s->src_pitch_reg << R580_PITCH_OFFSET_PITCH_SHIFT) |
              (s->src_tile_reg << 31);
        break;
    case R580_DST_PITCH_OFFSET:
    case R580_DST_PITCH_OFFSET_C:
        val = (s->dst_offset_reg >> 5) |
              (s->dst_pitch_reg << R580_PITCH_OFFSET_PITCH_SHIFT) |
              (s->dst_tile_reg << 31);
        break;
    case R580_SRC_OFFSET:
        val = s->src_offset_reg;
        break;
    case R580_SRC_PITCH:
        val = s->src_pitch_reg | (s->src_tile_reg << 16);
        break;
    case R580_DP_BRUSH_BKGD_CLR:
        val = s->dp_brush_bkgd_clr;
        break;
    case R580_DP_BRUSH_FRGD_CLR:
        val = s->dp_brush_frgd_clr;
        break;
    case R580_DP_SRC_FRGD_CLR:
        val = s->dp_src_frgd_clr;
        break;
    case R580_DP_SRC_BKGD_CLR:
        val = s->dp_src_bkgd_clr;
        break;
    case R580_DP_CNTL_XDIR_YDIR_YMAJOR:
        /*
         * The same two directions as DP_CNTL, in different bit
         * positions. Keep DP_CNTL as the canonical copy so the 2D engine
         * has one place to look.
         */
        s->dp_cntl &= ~(R580_DST_X_LEFT_TO_RIGHT | R580_DST_Y_TOP_TO_BOTTOM);
        if (val & R580_DST_X_DIR_LEFT_TO_RIGHT) {
            s->dp_cntl |= R580_DST_X_LEFT_TO_RIGHT;
        }
        if (val & R580_DST_Y_DIR_TOP_TO_BOTTOM) {
            s->dp_cntl |= R580_DST_Y_TOP_TO_BOTTOM;
        }
        break;
    case R580_DP_CNTL:
        val = s->dp_cntl;
        break;
    case R580_DP_DATATYPE:
        val = s->dp_datatype;
        break;
    case R580_DP_MIX:
        val = s->dp_mix;
        break;
    case R580_DP_WRITE_MASK:
        val = s->dp_write_mask;
        break;
    case R580_DEFAULT_OFFSET:
        val = s->default_offset;
        break;
    case R580_DEFAULT_PITCH:
        val = s->default_pitch;
        break;
    case R580_DEFAULT_SC_BOTTOM_RIGHT:
        val = s->default_sc_right | (s->default_sc_bottom << 16);
        break;
    case R580_SC_TOP:
        val = s->sc_top_reg;
        break;
    case R580_SC_LEFT:
        val = s->sc_left_reg;
        break;
    case R580_SC_BOTTOM:
        val = s->sc_bottom_reg;
        break;
    case R580_SC_RIGHT:
        val = s->sc_right_reg;
        break;
    case R580_SRC_SC_BOTTOM:
        val = s->src_sc_bottom_reg;
        break;
    case R580_SRC_SC_RIGHT:
        val = s->src_sc_right_reg;
        break;
    case R580_CFG_MIRROR_BASE ... R580_CFG_MIRROR_END:
        val = pci_default_read_config(dev, base - R580_CFG_MIRROR_BASE, 4);
        break;
    default:
        break;
    }
    return val;
}

static void ati_r580_bm_gui_run(ATIR580State *s, uint32_t table);

/*
 * Re-derive the effective 2D pitch/offset and scissors from the register
 * values, honouring DP_GUI_MASTER_CNTL's per-operation source selects.
 *
 * Those GMC bits pick where each value comes from; they do not move it.
 * Applying them destructively (writing DEFAULT_* into the effective field
 * on a GMC write) loses the register underneath, so whether a blit saw the
 * right pitch came down to whether the driver wrote SRC_PITCH before or
 * after the GMC dword. Recomputing instead means either order works, and a
 * surface's pitch can no longer leak into the next, differently sized one.
 */
static void ati_r580_resolve_gui_context(ATIR580State *s)
{
    uint32_t gmc = s->dp_gui_master_cntl;

    if (gmc & R580_GMC_SRC_PITCH_OFFSET_CNTL) {
        s->src_offset = s->src_offset_reg;
        s->src_pitch = s->src_pitch_reg;
        s->src_pitch_bytes = s->src_pitch_bytes_reg;
        s->src_tile = s->src_tile_reg;
    } else {
        s->src_offset = s->default_offset;
        s->src_pitch = s->default_pitch;
        s->src_pitch_bytes = true;
        s->src_tile = 0;
    }

    if (gmc & R580_GMC_DST_PITCH_OFFSET_CNTL) {
        s->dst_offset = s->dst_offset_reg;
        s->dst_pitch = s->dst_pitch_reg;
        s->dst_pitch_bytes = s->dst_pitch_bytes_reg;
        s->dst_tile = s->dst_tile_reg;
    } else {
        s->dst_offset = s->default_offset;
        s->dst_pitch = s->default_pitch;
        s->dst_pitch_bytes = true;
        s->dst_tile = 0;
    }

    if (gmc & R580_GMC_SRC_CLIPPING) {
        s->src_sc_right = s->src_sc_right_reg;
        s->src_sc_bottom = s->src_sc_bottom_reg;
    } else {
        s->src_sc_right = s->default_sc_right;
        s->src_sc_bottom = s->default_sc_bottom;
    }

    if (gmc & R580_GMC_DST_CLIPPING) {
        s->sc_top = s->sc_top_reg;
        s->sc_left = s->sc_left_reg;
        s->sc_right = s->sc_right_reg;
        s->sc_bottom = s->sc_bottom_reg;
    } else {
        s->sc_top = 0;
        s->sc_left = 0;
        s->sc_right = s->default_sc_right;
        s->sc_bottom = s->default_sc_bottom;
    }
}

void ati_r580_reg_write32(ATIR580State *s, uint32_t base,
                         uint32_t val)
{
    if (ati_r580_on_engine() && !ati_r580_engine_private_reg(base) &&
        !bql_locked()) {
        /* display, interrupt, clock and window state belong to the BQL */
        bql_lock();
        s->engine_bql_writes++;
        ati_r580_reg_write32(s, base, val);
        bql_unlock();
        return;
    }
    ati_r580_audit_reg_write(s, base);
    if (ati_r580_front_write(s, base, val)) {
        return;
    }

    /*
     * Diagnostic: the 2D source/destination context. Mac OS X programs
     * essentially all of this through the CCE (a whole window drag issues
     * ~180 MMIO register writes, nearly all of them CUR_OFFSET), and PM4
     * type-0 packets reach the register file through this function rather
     * than through the MMIO ops -- so the MMIO trace never sees them.
     * Tracing here catches both paths.
     */
    switch (base) {
    case R580_SRC_OFFSET:
    case R580_SRC_PITCH:
    case R580_SRC_PITCH_OFFSET:
    case R580_DST_OFFSET:
    case R580_DST_PITCH:
    case R580_DST_PITCH_OFFSET:
    case R580_DST_PITCH_OFFSET_C:
    case R580_DEFAULT_OFFSET:
    case R580_DEFAULT_PITCH:
    case R580_DP_GUI_MASTER_CNTL:
    case R580_DP_GUI_MASTER_CNTL_C:
        trace_ati_r580_ctx_write(ati_r580_reg_name(base), base, val);
        break;
    default:
        break;
    }

    switch (base) {
    case R580_MM_INDEX:
        s->regs[base >> 2] = val;
        break;
    case R580_MM_DATA:
        if (s->regs[R580_MM_INDEX >> 2] & R580_MM_APER) {
            uint32_t off = s->regs[R580_MM_INDEX >> 2] & 0x7ffffffc;
            uint8_t *vram = memory_region_get_ram_ptr(&s->vram);

            if (off + 4 <= ATI_R580_VRAM_SIZE) {
                ati_r580_gl_dirty(s, off, 4);
                stl_le_p(vram + off, val);
                memory_region_set_dirty(&s->vram, off, 4);
            }
        } else if ((s->regs[R580_MM_INDEX >> 2] & 0xfffc) > R580_MM_DATA) {
            /* MM_INDEX 0 and 4 would point back at the pair itself */
            ati_r580_reg_write32(s, s->regs[R580_MM_INDEX >> 2] & 0xfffc,
                                    val);
        }
        break;
    case R580_CLOCK_CNTL_INDEX:
        s->regs[base >> 2] = val;
        break;
    case R580_GEN_INT_CNTL:
        s->regs[base >> 2] = val;
        ati_r580_update_irq(s);
        break;
    case R580_GEN_INT_STATUS:
        /* write-1-to-acknowledge */
        s->regs[base >> 2] &= ~(val & R580_GEN_INT_ACK_MASK);
        /*
         * SW_INT_FIRE is a command, not a status bit to clear: the
         * command stream writes it to raise the software interrupt
         * that tells the driver its submitted work is done.
         */
        if (val & R580_SW_INT_FIRE) {
            s->regs[base >> 2] |= R580_SW_INT;
        }
        trace_ati_r580_int_ack(val, s->regs[base >> 2]);
        ati_r580_update_irq(s);
        break;
    case R580_GEN_RESET_CNTL:
        s->regs[base >> 2] = val;
        if (val & R580_SOFT_RESET_GUI) {
            /* engine soft reset also aborts any half-parsed command
             * stream state (matches the driver's reset-then-restart
             * expectation) */
            s->pm4_fifo.remaining = 0;
            s->pm4_ring.remaining = 0;
            s->pm4_rptr = 0;
            s->pm4_wptr = 0;
        }
        break;
    case R580_BM_GUI_TABLE:
        s->regs[base >> 2] = val;
        /* a descriptor that targets this register does not start a new walk */
        if (!s->bm_busy) {
            s->bm_busy = true;
            ati_r580_bm_gui_run(s, val);
            s->bm_busy = false;
        }
        break;
    case R580_CP_RB_BASE:
        s->pm4_buffer_addr = val;
        s->regs[base >> 2] = val;
        break;
    case R580_CP_RB_CNTL:
    {
        /* RB_BUFSZ = log2 of the ring size in qwords */
        uint32_t l2qw = val & R580_RB_BUFSZ_MASK;

        s->pm4_buffer_cntl = val;
        s->regs[base >> 2] = val;
        s->pm4_ring_dwords = l2qw ? 2u << l2qw : 0;
        /* reconfiguring the ring aborts any half-parsed packet */
        s->pm4_ring.remaining = 0;
        trace_ati_r580_cp_rb_cntl(val, s->pm4_ring_dwords,
                                  !!(val & R580_RB_NO_UPDATE),
                                  (val & R580_BUF_SWAP_MASK) >> 16);
        break;
    }
    case R580_CP_RB_RPTR:
        /* read-only */
        trace_ati_r580_cp_rptr_ignored("CP_RB_RPTR", val,
                                       s->regs[R580_CP_RB_CNTL >> 2]);
        break;
    case R580_CP_RB_RPTR_WR:
        /* moved into CP_RB_RPTR by a CP_RB_WPTR write, if RB_RPTR_WR_ENA */
        s->regs[base >> 2] = val & 0x7fffff;
        if (!(s->regs[R580_CP_RB_CNTL >> 2] & R580_RB_RPTR_WR_ENA)) {
            trace_ati_r580_cp_rptr_ignored("CP_RB_RPTR_WR", val,
                                           s->regs[R580_CP_RB_CNTL >> 2]);
        }
        break;
    case R580_CP_RB_WPTR:
        ati_r580_cp_rptr_load(s);
        s->pm4_wptr = s->pm4_ring_dwords ?
                      val & (s->pm4_ring_dwords - 1) : val;
        s->regs[base >> 2] = val;
        ati_r580_pm4_run(s);
        break;
    case R580_CP_IB_BASE:
        s->regs[base >> 2] = val;
        break;
    case R580_CP_IB_BUFSZ:
        s->regs[base >> 2] = val;
        if (s->pm4_in_ib) {
            /* the indirect stream cannot restart itself */
            trace_ati_r580_pm4_ib_nested(s->regs[R580_CP_IB_BASE >> 2],
                                         val & 0x7fffff);
            qemu_log_mask(LOG_GUEST_ERROR,
                          "ati-r580: CP_IB_BUFSZ written from an indirect "
                          "buffer, ignored\n");
            break;
        }
        s->pm4_in_ib = true;
        ati_r580_pm4_indirect(s, s->regs[R580_CP_IB_BASE >> 2] & ~3u,
                              val & 0x7fffff);
        s->pm4_in_ib = false;
        break;
    case R580_PM4_MICROCODE_ADDR:
        s->pm4_ucode_waddr = val & (R580_PM4_MICROCODE_WORDS - 1);
        break;
    case R580_PM4_MICROCODE_RADDR:
        s->pm4_ucode_raddr = val & (R580_PM4_MICROCODE_WORDS - 1);
        break;
    case R580_PM4_MICROCODE_DATAH:
        s->pm4_microcode[s->pm4_ucode_waddr][0] = val;
        break;
    case R580_PM4_MICROCODE_DATAL:
        s->pm4_microcode[s->pm4_ucode_waddr][1] = val;
        s->pm4_ucode_waddr = (s->pm4_ucode_waddr + 1) &
                             (R580_PM4_MICROCODE_WORDS - 1);
        break;
    case R580_CP_CSQ_CNTL:
    case R580_CP_CSQ_MODE:
    case R580_CP_ME_CNTL:
    case R580_CP_RB_WPTR_DELAY:
    case R580_CP_RB_RPTR_ADDR:
    case R580_SCRATCH_UMSK:
    case R580_SCRATCH_ADDR:
        s->regs[base >> 2] = val;
        trace_ati_r580_cp_reg(ati_r580_reg_name(base), val);
        break;
    case R580_SCRATCH_REG_BASE ... R580_SCRATCH_REG_LAST:
        s->engine_scratch++;
        if (ati_r580_on_engine()) {
            /* a fence: what was drawn before it is in VRAM first */
            ati_r580_gl_release(s, R580_GLR_FENCE);
            ati_r580_zpass_settle(s);
        }
        qatomic_store_release(&s->regs[base >> 2], val);
        ati_r580_scratch_writeback(s, (base - R580_SCRATCH_REG_BASE) >> 2);
        break;
    case R300_ZB_ZPASS_DATA:
        s->regs[base >> 2] = val;
        ati_r580_zpass_reset(s, val);
        break;
    case R300_ZB_ZPASS_ADDR:
        s->regs[base >> 2] = val;
        ati_r580_zpass_dump(s, val & ~3u);
        break;
    case R300_VAP_PVS_UPLOAD_ADDRESS:
        s->regs[base >> 2] = val;
        s->pvs_upload_addr = val;
        s->pvs_upload_cnt = 0;
        break;
    case R300_VAP_PVS_VECTOR_DATA:
    case R300_VAP_PVS_UPLOAD_DATA:
        s->regs[base >> 2] = val;
        if (s->pvs_upload_addr >= R300_PVS_UCP_START) {
            unsigned v = s->pvs_upload_addr - R300_PVS_UCP_START;

            if (v < R300_PVS_CLIP_VECS) {
                v += s->pvs_upload_cnt / 4;
                if (v >= R300_PVS_CLIP_WRAP &&
                    s->pvs_upload_addr - R300_PVS_UCP_START <
                    R300_PVS_CLIP_WRAP) {
                    v %= R300_PVS_CLIP_WRAP;
                }
                if (v < R300_PVS_CLIP_VECS) {
                    s->pvs_clip[v * 4 + s->pvs_upload_cnt % 4] = val;
                }
            }
        } else if (s->pvs_upload_addr >= R300_PVS_CONST_START) {
            /*
             * The constants are a register file addressed by vector, four
             * dwords to a vector, and an upload writes into it starting
             * wherever UPLOAD_ADDRESS pointed. Taking the data only when
             * that address was exactly the base dropped every partial
             * update on the floor -- a guest rewriting one constant left
             * the whole matrix at its previous contents, silently, and
             * the next draw was transformed by a stale one.
             */
            unsigned idx = (s->pvs_upload_addr - R300_PVS_CONST_START) * 4 +
                           s->pvs_upload_cnt;

            if (idx < ARRAY_SIZE(s->pvs_const)) {
                s->pvs_const[idx] = val;
                if (idx + 1 > s->pvs_const_dwords) {
                    s->pvs_const_dwords = idx + 1;
                }
            }
        } else {
            /*
             * Program instructions, four dwords to a slot. A slot counts
             * as the guest's only once its last dword has arrived: a
             * half-written one holds a mixture of two programs, and the
             * interpreter has to be able to refuse to run it.
             */
            unsigned idx = s->pvs_upload_addr * 4 + s->pvs_upload_cnt;

            if (idx < ARRAY_SIZE(s->pvs_code)) {
                s->pvs_code[idx] = val;
                if ((idx & 3) == 3) {
                    s->pvs_code_slot_valid[(idx / 4) / 32] |=
                        1u << ((idx / 4) % 32);
                }
            }
            s->pvs_code_dwords++;
        }
        s->pvs_upload_cnt++;
        break;
    case R580_SURFACE_CNTL:
    case R580_SURFACE0_LOWER_BOUND ... R580_SURFACE7_INFO:
        s->regs[base >> 2] = val;
        trace_ati_r580_surface(ati_r580_reg_name(base), val);
        break;
    case R580_PM4_FIFO_DATA_EVEN ... R580_PM4_FIFO_APER_END:
        /*
         * Any dword in the CCE FIFO aperture is a push. Mac OS's driver
         * writes packets as bursts across 0x1000..0x101c; taking only
         * EVEN/ODD lost the tail of every burst -- typically the
         * DST_Y_X/DST_HEIGHT_WIDTH pair that arms a host-data blit, so
         * the HOST_DATA stream that followed (menu bar restore, window
         * icons, drag save-behind) was thrown away.
         */
        ati_r580_pm4_fifo_push(s, val);
        break;
    case R580_DST_OFFSET:
        s->dst_offset_reg = val & 0xfffffff0;
        ati_r580_resolve_gui_context(s);
        break;
    case R580_DST_PITCH:
        s->dst_pitch_reg = val & 0x3fff;
        s->dst_tile_reg = (val >> 16) & 1;
        s->dst_pitch_bytes_reg = true;
        ati_r580_resolve_gui_context(s);
        break;
    case R580_DST_WIDTH:
        s->dst_width = val & 0x3fff;
        ati_r580_2d_blt(s);
        break;
    case R580_DST_HEIGHT:
        s->dst_height = val & 0x3fff;
        break;
    case R580_SRC_X:
        s->src_x = val & 0x3fff;
        break;
    case R580_SRC_Y:
        s->src_y = val & 0x3fff;
        break;
    case R580_DST_X:
        s->dst_x = val & 0x3fff;
        break;
    case R580_DST_Y:
        s->dst_y = val & 0x3fff;
        break;
    case R580_SRC_PITCH_OFFSET:
        /*
         * Radeon packed layout (R5xx accel guide / DRM): offset in
         * [21:0] as multiples of 1KB, pitch in [29:22] as multiples
         * of 64 bytes, tiling flags on top. (The Rage 128 packed
         * fields this device started from were offset/32 and pitch in
         * 8-pixel units -- live captures of OS X's scroll copies,
         * e.g. DEFAULT_OFFSET 0x4b002658 for the 704px surface at
         * 0x996000, decode only under the Radeon rule.)
         */
        s->src_offset_reg = (val & 0x3fffff) << 10;
        s->src_pitch_reg = ((val >> 22) & 0xff) * 64;
        s->src_pitch_bytes_reg = true;
        s->src_tile_reg = val >> 30;
        ati_r580_resolve_gui_context(s);
        break;
    case R580_DST_PITCH_OFFSET:
    case R580_DST_PITCH_OFFSET_C:
        /* Radeon packed layout, as for SRC_PITCH_OFFSET above */
        s->dst_offset_reg = (val & 0x3fffff) << 10;
        s->dst_pitch_reg = ((val >> 22) & 0xff) * 64;
        s->dst_pitch_bytes_reg = true;
        s->dst_tile_reg = val >> 30;
        ati_r580_resolve_gui_context(s);
        break;
    case R580_SRC_Y_X:
        s->src_x = val & 0x3fff;
        s->src_y = (val >> 16) & 0x3fff;
        break;
    case R580_DST_Y_X:
        s->dst_x = val & 0x3fff;
        s->dst_y = (val >> 16) & 0x3fff;
        break;
    case R580_DST_HEIGHT_WIDTH:
        s->dst_width = val & 0x3fff;
        s->dst_height = (val >> 16) & 0x3fff;
        ati_r580_2d_blt(s);
        break;
    case R580_SCALE_DST_HEIGHT_WIDTH:
        /* the register-programmed scaler's kick: parameters were stored
         * by the writes before it */
        s->regs[base >> 2] = val;
        ati_r580_2d_scale_regs(s);
        break;
    case R580_DP_GUI_MASTER_CNTL:
    case R580_DP_GUI_MASTER_CNTL_C:
        s->dp_gui_master_cntl = val & 0xf800000f;
        /*
         * Only the aliased fields come from GUI_MASTER_CNTL. Rebuilding
         * the whole of DP_DATATYPE here wiped HOST_BIG_ENDIAN_EN, which
         * the driver programs separately (and, on this card, through a
         * PM4 type-0 packet rather than an MMIO write) -- and since every
         * PAINT/PAINT_MULTI/HOSTDATA_BLT packet carries its own GMC
         * dword, the bit was destroyed again before every single blit.
         * Host-supplied pixels then went into VRAM byte-reversed: on a
         * 32bpp desktop the unused byte landed in the blue slot, so
         * anything white came out yellow and red/green swapped places.
         */
        s->dp_datatype = (s->dp_datatype & ~R580_DP_DATATYPE_GMC_ALIAS) |
                         (val & 0x0f00) >> 8 | (val & 0x30f0) << 4 |
                         (val & 0x4000) << 16;
        s->dp_mix = (val & R580_GMC_ROP3_MASK) | (val & 0x7000000) >> 16;
        ati_r580_resolve_gui_context(s);
        break;
    case R580_DST_WIDTH_X:
        s->dst_x = val & 0x3fff;
        s->dst_width = (val >> 16) & 0x3fff;
        ati_r580_2d_blt(s);
        break;
    case R580_SRC_X_Y:
        s->src_y = val & 0x3fff;
        s->src_x = (val >> 16) & 0x3fff;
        break;
    case R580_DST_X_Y:
        s->dst_y = val & 0x3fff;
        s->dst_x = (val >> 16) & 0x3fff;
        break;
    case R580_DST_WIDTH_HEIGHT:
        s->dst_height = val & 0x3fff;
        s->dst_width = (val >> 16) & 0x3fff;
        ati_r580_2d_blt(s);
        break;
    case R580_DST_HEIGHT_Y:
        s->dst_y = val & 0x3fff;
        s->dst_height = (val >> 16) & 0x3fff;
        break;
    case R580_SRC_OFFSET:
        s->src_offset_reg = val & 0xfffffff0;
        ati_r580_resolve_gui_context(s);
        break;
    case R580_SRC_PITCH:
        s->src_pitch_reg = val & 0x3fff;
        s->src_tile_reg = (val >> 16) & 1;
        s->src_pitch_bytes_reg = true;
        ati_r580_resolve_gui_context(s);
        break;
    case R580_DP_BRUSH_BKGD_CLR:
        s->dp_brush_bkgd_clr = val;
        break;
    case R580_DP_BRUSH_FRGD_CLR:
    case R580_CONSTANT_COLOR_C:
        s->dp_brush_frgd_clr = val;
        trace_ati_r580_brush_clr(base, val);
        break;
    case R580_DP_CNTL:
        s->dp_cntl = val;
        break;
    case R580_DP_SRC_FRGD_CLR:
        s->dp_src_frgd_clr = val;
        break;
    case R580_DP_SRC_BKGD_CLR:
        s->dp_src_bkgd_clr = val;
        break;
    case R580_DP_DATATYPE:
        s->dp_datatype = val & 0xe0070f0f;
        trace_ati_r580_datatype(val, !!(val & R580_HOST_BIG_ENDIAN_EN));
        break;
    case R580_DP_MIX:
        s->dp_mix = val & 0x00ff0700;
        break;
    case R580_DP_WRITE_MASK:
    case R580_PLANE_3D_MASK_C:
        s->dp_write_mask = val;
        break;
    case R580_DEFAULT_OFFSET:
        /*
         * On Radeon this is DEFAULT_PITCH_OFFSET, packed like
         * SRC/DST_PITCH_OFFSET: OS X's scroll copies run with the GMC
         * defaults selected and only this register armed.
         */
        s->default_offset = (val & 0x3fffff) << 10;
        s->default_pitch = ((val >> 22) & 0xff) * 64;
        ati_r580_resolve_gui_context(s);
        break;
    case R580_DEFAULT_PITCH:
        s->default_pitch = val & 0x3fff;
        ati_r580_resolve_gui_context(s);
        break;
    case R580_DEFAULT_SC_BOTTOM_RIGHT:
        s->default_sc_right = sextract32(val, 0, 14);
        s->default_sc_bottom = sextract32(val, 16, 14);
        ati_r580_resolve_gui_context(s);
        break;
    case R580_SC_TOP_LEFT:
    case R580_SC_TOP_LEFT_C:
        s->sc_left_reg = sextract32(val, 0, 14);
        s->sc_top_reg = sextract32(val, 16, 14);
        ati_r580_resolve_gui_context(s);
        break;
    case R580_SC_LEFT:
        s->sc_left_reg = sextract32(val, 0, 14);
        ati_r580_resolve_gui_context(s);
        break;
    case R580_SC_TOP:
        s->sc_top_reg = sextract32(val, 0, 14);
        ati_r580_resolve_gui_context(s);
        break;
    case R580_SC_BOTTOM_RIGHT:
    case R580_SC_BOTTOM_RIGHT_C:
        s->sc_right_reg = sextract32(val, 0, 14);
        s->sc_bottom_reg = sextract32(val, 16, 14);
        ati_r580_resolve_gui_context(s);
        break;
    case R580_SC_RIGHT:
        s->sc_right_reg = sextract32(val, 0, 14);
        ati_r580_resolve_gui_context(s);
        break;
    case R580_SC_BOTTOM:
        s->sc_bottom_reg = sextract32(val, 0, 14);
        ati_r580_resolve_gui_context(s);
        break;
    case R580_SRC_SC_BOTTOM_RIGHT:
        s->src_sc_right_reg = sextract32(val, 0, 14);
        s->src_sc_bottom_reg = sextract32(val, 16, 14);
        ati_r580_resolve_gui_context(s);
        break;
    case R580_SRC_SC_RIGHT:
        s->src_sc_right_reg = sextract32(val, 0, 14);
        ati_r580_resolve_gui_context(s);
        break;
    case R580_SRC_SC_BOTTOM:
        s->src_sc_bottom_reg = sextract32(val, 0, 14);
        ati_r580_resolve_gui_context(s);
        break;
    case R580_HOST_DATA0:
    case R580_HOST_DATA1:
    case R580_HOST_DATA2:
    case R580_HOST_DATA3:
    case R580_HOST_DATA4:
    case R580_HOST_DATA5:
    case R580_HOST_DATA6:
    case R580_HOST_DATA7:
    case R580_HOST_DATA_LAST:
        if (!s->host_data_active) {
            break;
        }
        trace_ati_r580_hostdata_dw(base, val, s->host_data_next);
        s->host_data_acc[s->host_data_next++] = val;
        if (base == R580_HOST_DATA_LAST) {
            ati_r580_host_data_flush(s);
            s->host_data_active = false;
            s->host_data_next = 0;
        } else if (s->host_data_next >= 4) {
            ati_r580_host_data_flush(s);
            s->host_data_next = 0;
        }
        break;
    case R580_CFG_MIRROR_BASE ... R580_CFG_MIRROR_END:
        /* read-only mirror of PCI config space */
        break;
    default:
        /*
         * The fragment decode is cached on the control words; a new
         * instruction, slot range or rasterizer route has to drop it.
         */
        if (s->regs[base >> 2] != val &&
            ((base >= R300_RS_INST_COUNT && base <= R300_RS_INST_0 + 0x3c) ||
             (base >= R300_US_CODE_ADDR_0 &&
              base <= R300_US_TEX_INST_0 + 0x7c) ||
             (base >= R300_US_ALU_RGB_ADDR_0 &&
              base <= R300_US_ALU_ALPHA_INST_0 + 0xfc))) {
            s->us_sig = 0;
        }
        s->regs[base >> 2] = val;
        break;
    }
}

/*
 * GUI bus master: walk the descriptor table at the given guest-physical
 * address. Each 12-byte, little-endian entry is {dest register offset,
 * source system-memory address, control}; control's low 16 bits are a
 * byte count and bit 31 is END_OF_LIST (both inferred from the smoke
 * test -- see the comment on R580_BM_GUI_TABLE in ati_r580_regs.h).
 * The transfer completes synchronously, one dword at a time through the
 * normal register-write path so a descriptor can target any register
 * exactly as if the driver had written it directly via MM_INDEX/DATA.
 */
static void ati_r580_bm_gui_run(ATIR580State *s, uint32_t table)
{
    PCIDevice *pci = PCI_DEVICE(s);
    dma_addr_t desc = table;
    int entry;

    for (entry = 0; entry < 4096; entry++) {
        uint32_t d[3];
        uint32_t reg_off, sysaddr, ctrl, count;

        if (pci_dma_read(pci, desc, d, sizeof(d)) != MEMTX_OK) {
            break;
        }
        reg_off = le32_to_cpu(d[0]);
        sysaddr = le32_to_cpu(d[1]);
        ctrl = le32_to_cpu(d[2]);
        count = ctrl & 0xffff;
        trace_ati_r580_bm_desc(reg_off, sysaddr, ctrl);

        while (count >= 4) {
            uint32_t word;

            if (pci_dma_read(pci, sysaddr, &word, 4) != MEMTX_OK) {
                break;
            }
            ati_r580_reg_write32(s, reg_off & 0xfffc, le32_to_cpu(word));
            sysaddr += 4;
            reg_off += 4;
            count -= 4;
        }

        if (ctrl & (1u << 31)) {
            break;
        }
        desc += 12;
    }

    s->regs[R580_GEN_INT_STATUS >> 2] |= R580_BUSMASTER_EOL_INT;
    ati_r580_update_irq(s);
}

/*
 * The engine classification of an access: MM_DATA by the register MM_INDEX
 * points at, MM_INDEX itself as a plain BQL register.
 */
static bool ati_r580_mmio_engine(ATIR580State *s, uint32_t base)
{
    if (base == R580_MM_INDEX) {
        return false;
    }
    if (base == R580_MM_DATA) {
        uint32_t idx = qatomic_read(&s->regs[R580_MM_INDEX >> 2]);

        if (!(idx & R580_MM_APER) && (idx & 0xfffc) > R580_MM_DATA) {
            return ati_r580_engine_reg(idx & 0xfffc);
        }
    }
    return ati_r580_engine_reg(base);
}

static uint32_t ati_r580_read_lockless(ATIR580State *s, uint32_t base)
{
    bool engine = ati_r580_mmio_engine(s, base);
    bool bql;
    uint32_t val;

    if (ati_r580_engine_status_reg(base) || base == R580_RBBM_STATUS) {
        return ati_r580_reg_read32(s, base);
    }
    if (engine) {
        ati_r580_engine_claim(s);
        bql = ati_r580_engine_bql_reg(base) && !bql_locked();
    } else {
        bql = !bql_locked();
    }
    if (bql) {
        bql_lock();
    }
    val = ati_r580_reg_read32(s, base);
    if (bql) {
        bql_unlock();
    }
    if (engine) {
        ati_r580_engine_unclaim(s, false);
    }
    return val;
}

static uint64_t ati_r580_mmio_read(void *opaque, hwaddr addr,
                                      unsigned size)
{
    ATIR580State *s = opaque;
    uint32_t base = addr & 0xfffc;
    uint32_t val;

    if (s->lockless) {
        val = ati_r580_read_lockless(s, base);
    } else {
        if (s->engine_on && qatomic_read(&s->engine_busy) &&
            ati_r580_mmio_engine(s, base) &&
            !ati_r580_engine_status_reg(base)) {
            ati_r580_engine_wait(s);
        }
        val = ati_r580_reg_read32(s, base);
    }
    val = extract32(val, (addr & 3) * 8, size * 8);
    if (trace_event_get_state_backends(TRACE_ATI_R580_UNK_READ) ||
        trace_event_get_state_backends(TRACE_ATI_R580_REG_READ)) {
        if (ati_r580_reg_name(base)[0] == '?') {
            trace_ati_r580_unk_read(size, addr, val);
        } else {
            trace_ati_r580_reg_read(size, addr, ati_r580_reg_name(base),
                                    val);
        }
    }
    return val;
}

/*
 * The value a 1/2/4-byte write at `addr` leaves in its register: the
 * written lanes merged onto the current value.
 */
static uint32_t ati_r580_mmio_merge(ATIR580State *s, hwaddr addr,
                                    uint64_t data, unsigned size)
{
    uint32_t base = addr & 0xfffc;
    uint32_t val = data;

    if (size != 4 || (addr & 3)) {
        /*
         * Merge the written lanes onto the register's current value.
         * For the MM_DATA and CLOCK_CNTL_DATA pass-throughs the
         * current value lives behind the indirection, not in the raw
         * regs[] slot; merging against the raw slot would clobber the
         * target's other lanes.
         */
        uint32_t mbase = base;
        uint32_t merged;

        if (mbase == R580_MM_DATA) {
            mbase = s->regs[R580_MM_INDEX >> 2] & 0xfffc;
        }
        if (mbase == R580_CLOCK_CNTL_DATA) {
            merged = s->plls[s->regs[R580_CLOCK_CNTL_INDEX >> 2] &
                             R5XX_PLL_ADDR_MASK];
        } else {
            /* raw slot; auto-increment/FIFO registers keep their own
             * state elsewhere, so this stays side-effect free */
            merged = s->regs[mbase >> 2];
        }
        merged = deposit32(merged, (addr & 3) * 8, size * 8, data);
        val = merged;
    }
    return val;
}

/* a register write with lockless register access; see above */
static void ati_r580_write_lockless(ATIR580State *s, hwaddr addr,
                                    uint64_t data, unsigned size)
{
    uint32_t base = addr & 0xfffc;
    bool engine = ati_r580_mmio_engine(s, base);
    bool bql;

    if (base == R580_CP_RB_WPTR && size == 4 && !(addr & 3)) {
        ati_r580_engine_wptr(s, data);
        return;
    }
    if (engine) {
        ati_r580_engine_claim(s);
        bql = ati_r580_engine_bql_reg(base) && !bql_locked();
    } else {
        bql = !bql_locked();
    }
    if (bql) {
        bql_lock();
    }
    ati_r580_reg_write32(s, base, ati_r580_mmio_merge(s, addr, data, size));
    if (bql) {
        bql_unlock();
    }
    if (engine) {
        ati_r580_engine_unclaim(s, false);
    }
}

static void ati_r580_mmio_write(void *opaque, hwaddr addr, uint64_t data,
                                   unsigned size)
{
    ATIR580State *s = opaque;
    uint32_t base = addr & 0xfffc;
    uint32_t val = data;

    if (trace_event_get_state_backends(TRACE_ATI_R580_UNK_WRITE) ||
        trace_event_get_state_backends(TRACE_ATI_R580_REG_WRITE)) {
        if (ati_r580_reg_name(base)[0] == '?') {
            trace_ati_r580_unk_write(size, addr, data);
        } else {
            trace_ati_r580_reg_write(size, addr, ati_r580_reg_name(base),
                                     data);
        }
    }
    if (s->lockless) {
        ati_r580_write_lockless(s, addr, data, size);
        return;
    }
    if (s->engine_on) {
        if (base == R580_CP_RB_WPTR && size == 4 && !(addr & 3)) {
            if (s->regs[R580_CP_RB_CNTL >> 2] & R580_RB_RPTR_WR_ENA) {
                ati_r580_engine_wait(s);
                ati_r580_cp_rptr_load(s);
            }
            s->regs[base >> 2] = val;
            qatomic_store_release(&s->pm4_wptr, s->pm4_ring_dwords ?
                                  val & (s->pm4_ring_dwords - 1) : val);
            ati_r580_engine_kick(s);
            return;
        }
        if (qatomic_read(&s->engine_busy) && ati_r580_mmio_engine(s, base)) {
            ati_r580_engine_wait(s);
        }
    }
    ati_r580_reg_write32(s, base, ati_r580_mmio_merge(s, addr, data, size));
}

static const MemoryRegionOps ati_r580_mmio_ops = {
    .read = ati_r580_mmio_read,
    .write = ati_r580_mmio_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {
        .min_access_size = 1,
        .max_access_size = 4,
    },
};

/*
 * Diagnostic window over part of the frame buffer aperture (see the
 * fillwatch fields); it behaves exactly like the RAM it replaces.
 */
static uint64_t ati_r580_fillwatch_read(void *opaque, hwaddr addr,
                                           unsigned size)
{
    ATIR580State *s = opaque;
    uint32_t base = s->fillwatch_off + (uint32_t)addr;
    uint64_t val = ati_r580_aper_ld(s, base, size,
                                    ati_r580_aper_xor(s, base));

    trace_ati_r580_fillwatch_rd(base, size, val);
    return val;
}

static void ati_r580_fillwatch_write(void *opaque, hwaddr addr,
                                        uint64_t data, unsigned size)
{
    ATIR580State *s = opaque;
    uint32_t base = s->fillwatch_off + (uint32_t)addr;

    ati_r580_aper_st(s, base, data, size, ati_r580_aper_xor(s, base));

    /*
     * Coalesce. A strided fill writes each row as one contiguous run and
     * then jumps, so the run length and the jump to the next run's start
     * are exactly the geometry we are trying to recover: the delta
     * between successive run starts IS the stride the CPU is filling at.
     */
    if (s->fw_active && base == s->fw_run_end) {
        s->fw_run_end = base + size;
        return;
    }
    if (s->fw_active) {
        trace_ati_r580_fillwatch(s->fw_run_start,
                                    s->fw_run_end - s->fw_run_start,
                                    base - s->fw_run_start);
    }
    s->fw_run_start = base;
    s->fw_run_end = base + size;
    s->fw_active = true;
}

static const MemoryRegionOps ati_r580_fillwatch_ops = {
    .read = ati_r580_fillwatch_read,
    .write = ati_r580_fillwatch_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {
        .min_access_size = 1,
        .max_access_size = 8,
        .unaligned = true,
    },
    .impl = {
        .min_access_size = 1,
        .max_access_size = 8,
        .unaligned = true,
    },
};

/*
 * The DEPTH GUARD: a window over aperture 0 laid over the pages of a
 * depth buffer the host GPU holds newer than VRAM (gl-depth-resident;
 * see "GL-OWNED DEPTH BUFFER" in ati_r580_3d.c). Every guest CPU access
 * there comes here instead of reaching RAM. One that touches the Z
 * bytes writes the GPU's copy back first and lifts the guard; the access
 * itself is then done on VRAM exactly as the RAM it replaces would, so
 * a byte lands where a store through aperture 0 puts it (see
 * ati_r580_fillwatch_read()).
 */
static void ati_r580_zguard_set(ATIR580State *s, uint32_t lo, uint32_t hi,
                                uint32_t zlo, uint32_t zhi)
{
    memory_region_transaction_begin();
    if (hi > lo) {
        memory_region_set_enabled(&s->zg_win, false);
        memory_region_set_alias_offset(&s->zg_win, lo);
        memory_region_set_size(&s->zg_win, hi - lo);
        memory_region_set_address(&s->zg_win, lo);
    }
    memory_region_set_enabled(&s->zg_win, hi > lo);
    memory_region_transaction_commit();
    s->zg_on = hi > lo;
    s->zg_lo = lo;
    s->zg_hi = hi;
    s->zg_zlo = zlo;
    s->zg_zhi = zhi;
}

bool ati_r580_zguard_arm(ATIR580State *s, uint32_t lo, uint32_t hi,
                         uint32_t zlo, uint32_t zhi)
{
    bool bql = !bql_locked();

    if (!s->zg_ready || hi <= lo || hi > ATI_R580_VRAM_SIZE) {
        return false;
    }
    if (bql) {
        bql_lock();
    }
    ati_r580_zguard_set(s, lo, hi, zlo, zhi);
    if (bql) {
        bql_unlock();
    }
    return true;
}

/* the BQL is held: this is an ordinary MMIO region */
static void ati_r580_zguard_hit(ATIR580State *s, uint32_t off, unsigned len)
{
    bool lift;

    if (!s->zg_on || off + len <= s->zg_zlo || off >= s->zg_zhi) {
        return;
    }
    if (s->lockless) {
        ati_r580_engine_claim(s);
    } else {
        ati_r580_engine_wait(s);
    }
    lift = ati_r580_gl_zguard(s, off, len);
    if (lift && s->zg_on) {
        ati_r580_zguard_set(s, 0, 0, 0, 0);
    }
    if (s->lockless) {
        ati_r580_engine_unclaim(s, false);
    }
}

static uint64_t ati_r580_zguard_read(void *opaque, hwaddr addr,
                                     unsigned size)
{
    ATIR580State *s = opaque;

    if (addr + size > ATI_R580_VRAM_SIZE) {
        return 0;
    }
    ati_r580_zguard_hit(s, addr, size);
    return ati_r580_aper_ld(s, addr, size, ati_r580_aper_xor(s, addr));
}

static void ati_r580_zguard_write(void *opaque, hwaddr addr,
                                  uint64_t data, unsigned size)
{
    ATIR580State *s = opaque;

    if (addr + size > ATI_R580_VRAM_SIZE) {
        return;
    }
    ati_r580_zguard_hit(s, addr, size);
    ati_r580_aper_st(s, addr, data, size, ati_r580_aper_xor(s, addr));
}

static const MemoryRegionOps ati_r580_zguard_ops = {
    .read = ati_r580_zguard_read,
    .write = ati_r580_zguard_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {
        .min_access_size = 1,
        .max_access_size = 8,
        .unaligned = true,
    },
    .impl = {
        .min_access_size = 1,
        .max_access_size = 8,
        .unaligned = true,
    },
};

/* ---------------------------------------------------------------- */

static void ati_r580_reset_hold(Object *obj, ResetType type)
{
    ATIR580State *s = ATI_R580(obj);

    if (s->lockless) {
        ati_r580_engine_claim(s);
    } else {
        ati_r580_engine_wait(s);
    }
    /* the surface descriptors are about to go: resolve the target first */
    ati_r580_gl_reset(s);
    ati_r580_zpass_drop(s);
    s->zclr_pend = false;
    if (s->zg_on) {
        ati_r580_zguard_set(s, 0, 0, 0, 0);
    }
    memset(s->regs, 0, sizeof(s->regs));
    memset(s->plls, 0, sizeof(s->plls));
    /*
     * Posted clocks, as the VBIOS ASIC_Init leaves them: SPLL and MPLL
     * feedback dividers in bits 5-12, reference divider in bits 2-4, from
     * the 27 MHz reference (SCLK 594 MHz, MCLK 405 MHz).
     */
    s->plls[R5XX_PLL_SPLL_FUNC_CNTL] = 43 << 5;
    s->plls[R5XX_PLL_MPLL_FUNC_CNTL] = 29 << 5;

    /* scanout bookkeeping back to what a cold boot starts from */
    memset(&s->mode, 0, sizeof(s->mode));
    qatomic_set(&s->scan_fb, 0);
    s->force_redraw = true;
    if (s->hw_cursor_on) {
        s->hw_cursor_on = false;
        if (s->con) {
            qemu_console_set_mouse(s->con, 0, 0, false);
        }
    }
    s->hw_cursor_sum = 0;

    /* command-stream framing and the host-data accumulator */
    memset(&s->pm4_fifo, 0, sizeof(s->pm4_fifo));
    memset(&s->pm4_ring, 0, sizeof(s->pm4_ring));
    s->r300_indx_vf = 0;
    s->pm4_ucode_waddr = 0;
    s->pm4_ucode_raddr = 0;
    s->host_data_active = false;
    memset(&s->hd, 0, sizeof(s->hd));

    /* memory controller: VRAM at card address 0, no AGP window */
    memset(s->mc_ind, 0, sizeof(s->mc_ind));
    memset(s->pcie_ind, 0, sizeof(s->pcie_ind));
    s->regs[R580_MC_FB_LOCATION >> 2] =
        ((ATI_R580_VRAM_SIZE - 1) >> 16) << 16;
    s->regs[R580_MC_AGP_LOCATION >> 2] = 0xffffffff;
    s->mc_ind[R5XX_MC_CNTL0] = R5XX_MC_CNTL0_256BIT;
    s->regs[R580_MEM_CNTL >> 2] = R580_MEM_NUM_CHANNELS_256;
    s->regs[R580_CP_RB_CNTL >> 2] = R580_RB_NO_UPDATE;
    s->regs[R300_SU_REG_DEST >> 2] = 0xf;
    s->pm4_buffer_cntl = R580_RB_NO_UPDATE;
    s->pm4_ring_dwords = 0;
    s->pm4_rptr = 0;
    s->pm4_wptr = 0;
    s->pm4_buffer_addr = 0;
    s->pm4_in_ib = false;
    s->pm4_in_ring = false;
    ati_r580_avivo_reset(s);
    ati_r580_update_irq(s);
    if (s->lockless) {
        ati_r580_engine_unclaim(s, false);
    }
}

static void ati_r580_cursor_timer(void *opaque)
{
    ati_r580_avivo_cursor_apply(opaque);
}

/*
 * A cursor move is a burst of register writes (position, hot spot,
 * surface, control). Apply them once, a moment after the first write,
 * so the burst is seen whole; later writes do not re-arm the timer.
 */
void ati_r580_cursor_update(ATIR580State *s)
{
    if (!s->cursor_timer) {
        ati_r580_avivo_cursor_apply(s);
        return;
    }
    if (!timer_pending(s->cursor_timer)) {
        timer_mod(s->cursor_timer,
                  qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) + SCALE_MS);
    }
}

static void ati_r580_realize(PCIDevice *dev, Error **errp)
{
    ATIR580State *s = ATI_R580(dev);
    Object *obj = OBJECT(dev);

    s->con = qemu_graphic_console_create(DEVICE(dev), 0,
                                         &ati_r580_gfx_ops, s);

    if (s->device_id > 0xffff || s->subsys_vendor_id > 0xffff ||
        s->subsys_id > 0xffff || s->revision > 0xff) {
        error_setg(errp, "device-id, subsys-vendor-id and subsys-id are "
                   "16 bit, revision 8 bit");
        return;
    }
    pci_set_word(dev->config + PCI_DEVICE_ID, s->device_id);
    pci_set_word(dev->config + PCI_SUBSYSTEM_VENDOR_ID, s->subsys_vendor_id);
    pci_set_word(dev->config + PCI_SUBSYSTEM_ID, s->subsys_id);
    pci_set_byte(dev->config + PCI_REVISION_ID, s->revision);

    /* BAR0 is all of VRAM, linear; diagnostic windows lie over it */
    memory_region_init(&s->aper, obj, "ati-r580-aper",
                       ATI_R580_VRAM_SIZE);
    memory_region_init_ram(&s->vram, obj, "ati-r580-vram",
                           ATI_R580_VRAM_SIZE, &error_fatal);
    memory_region_set_log(&s->vram, true, DIRTY_MEMORY_VGA);
    memory_region_add_subregion(&s->aper, 0, &s->vram);
    /*
     * Optional diagnostic overlay on aperture 0 (see the fillwatch fields
     * in the header). Higher priority than the RAM alias underneath, so
     * stores in this range come to us instead of landing silently.
     */
    if (s->fillwatch_size &&
        (uint64_t)s->fillwatch_off + s->fillwatch_size <=
        ATI_R580_VRAM_SIZE) {
        memory_region_init_io(&s->vram_watch, obj,
                              &ati_r580_fillwatch_ops, s,
                              "ati-r580-fillwatch", s->fillwatch_size);
        memory_region_add_subregion_overlap(&s->aper, s->fillwatch_off,
                                            &s->vram_watch, 3);
    }

    /*
     * Optional diagnostic 3D draw capture (see the cap_ fields in the
     * header and ati_r580_cap.h). The file is opened here and the draw
     * path tests nothing but the resulting FILE *, so a device realized
     * without the property is untouched by any of it.
     */
    if (s->cap_path && s->cap_path[0]) {
        R580CapFileHdr h = { 0 };

        s->cap_fp = fopen(s->cap_path, "wb");
        if (!s->cap_fp) {
            error_setg_errno(errp, errno, "cannot open draw-capture file %s",
                             s->cap_path);
            return;
        }
        memcpy(h.magic, R580_CAP_MAGIC, sizeof(h.magic));
        h.hdr_bytes = sizeof(h);
        h.rec_hdr_bytes = sizeof(R580CapRecHdr);
        h.state_bytes = ati_r580_cap_state_bytes();
        h.vtx_bytes = ati_r580_cap_vtx_bytes();
        h.vram_bytes = ATI_R580_VRAM_SIZE;
        h.us_bytes = sizeof(R300UsProgram);
        if (fwrite(&h, 1, sizeof(h), s->cap_fp) != sizeof(h)) {
            error_setg_errno(errp, errno, "cannot write draw-capture file %s",
                             s->cap_path);
            fclose(s->cap_fp);
            s->cap_fp = NULL;
            return;
        }
    }

    /*
     * Host-GPU offload (phase 2, milestone M2). `gl=off` -- the default
     * -- opens nothing, and the draw path then tests one NULL pointer
     * and rasterizes exactly as it always did. `on` renders what the
     * backend can through it and falls back per draw for the rest;
     * `verify` runs both paths and diffs them, with the software result
     * still the one that lands in VRAM.
     */
    if (s->gl_path && s->gl_path[0] && strcmp(s->gl_path, "off")) {
        const char *why = NULL;

        if (!strcmp(s->gl_path, "on")) {
            s->gl_mode = R580_GL_ON;
        } else if (!strcmp(s->gl_path, "verify")) {
            s->gl_mode = R580_GL_VERIFY;
        } else if (!strcmp(s->gl_path, "fast")) {
            /*
             * `on` plus the additive one-pass blend. Its own comment in
             * r300_gl_addblend() has the measurements; the short of it
             * is Flurry.saver at 22.93 fps instead of 13.19, with
             * gl=verify's VALUE class falling from 99.9998% at delta 0
             * to 98.83% on that workload. It is deliberately not the
             * default: `on` renders what it renders exactly or hands it
             * back, and this mode trades that for frame rate.
             */
            s->gl_mode = R580_GL_ON;
            s->gl_fast = true;
        } else {
            error_setg(errp,
                       "gl must be one of off, on, fast, verify (got \"%s\")",
                       s->gl_path);
            return;
        }
        if (s->gl_workers < 1 || s->gl_workers > R580_GL_MAXWORKERS) {
            error_setg(errp, "gl-compile-workers must be 1..%d",
                       R580_GL_MAXWORKERS);
            return;
        }
        if (!s->gl_api || !s->gl_api[0] || !strcmp(s->gl_api, "gl")) {
            s->gl_ctx = ati_r580_gl_open(&why,
                                         s->gl_async ? s->gl_workers : 0);
        } else if (!strcmp(s->gl_api, "metal")) {
#ifdef CONFIG_DARWIN
            s->gl_ctx = ati_r580_gl_open_metal(&why, s->gl_async
                                               ? s->gl_workers : 0);
#else
            error_setg(errp, "gl-api=metal is only available on macOS");
            return;
#endif
        } else {
            error_setg(errp, "gl-api must be gl or metal (got \"%s\")",
                       s->gl_api);
            return;
        }
        if (!s->gl_ctx) {
            error_setg(errp, "gl=%s: %s", s->gl_path, why);
            return;
        }
        trace_ati_r580_gl_open(ati_r580_gl_describe(s->gl_ctx));
    }
    s->gl_pgbits = qemu_target_page_bits();
    s->scan_dirty = bitmap_new(ATI_R580_VRAM_SIZE >> s->gl_pgbits);
    s->scan_work = bitmap_new(ATI_R580_VRAM_SIZE >> s->gl_pgbits);
    s->gl_texlife = R580_TEXLIFE_DIRTY;
    if (s->gl_texlife_path && s->gl_texlife_path[0]) {
        if (!strcmp(s->gl_texlife_path, "burst")) {
            s->gl_texlife = R580_TEXLIFE_BURST;
        } else if (!strcmp(s->gl_texlife_path, "never")) {
            s->gl_texlife = R580_TEXLIFE_NEVER;
            warn_report("ati-radeon-x1900: gl-texlife=never never invalidates "
                        "a decoded texture. It is a control for proving the "
                        "invalidation guard fires, not a usable setting.");
        } else if (strcmp(s->gl_texlife_path, "dirty")) {
            error_setg(errp, "gl-texlife must be one of dirty, burst, never "
                       "(got \"%s\")", s->gl_texlife_path);
            return;
        }
    }
    if (s->gl_tex_slots > R300_GL_TEXCACHE ||
        (s->gl_tex_slots && s->gl_tex_slots < R300_GL_TEXCACHE_OLD)) {
        error_setg(errp, "gl-texcache-slots must be 0 or %u to %u (got %u)",
                   R300_GL_TEXCACHE_OLD, R300_GL_TEXCACHE, s->gl_tex_slots);
        return;
    }
    s->gl_tex_n = s->gl_tex_slots ? s->gl_tex_slots : R300_GL_TEXCACHE_OLD;
    s->gl_tex_hashed = s->gl_tex_n > R300_GL_TEXCACHE_OLD;
    memset(s->gl_tex_bucket, 0xff, sizeof(s->gl_tex_bucket));
    for (unsigned k = 0; k < R300_GL_TEXCACHE; k++) {
        s->gl_tex[k].hb = R300_GL_TEXNIL;
        s->gl_tex[k].hnext = R300_GL_TEXNIL;
    }

    if (s->gl_ctx && s->gl_zlazy) {
        memory_region_init_io(&s->zg_io, obj, &ati_r580_zguard_ops, s,
                              "ati-r580-zguard", ATI_R580_VRAM_SIZE);
        memory_region_init_alias(&s->zg_win, obj, "ati-r580-zguard-win",
                                 &s->zg_io, 0, qemu_target_page_size());
        memory_region_set_enabled(&s->zg_win, false);
        memory_region_add_subregion_overlap(&s->aper, 0, &s->zg_win, 2);
        s->zg_ready = true;
    }

    memory_region_init_io(&s->mmio, obj, &ati_r580_mmio_ops, s,
                          "ati-r580-mmio", ATI_R580_MMIO_SIZE);
    /*
     * BAR1: the 256-byte I/O register window. Registers above 0xFF are
     * reachable through it via the MM_INDEX/MM_DATA pair at 0x00/0x04
     * -- which the shared mmio ops already implement, so the window is
     * simply the low 256 bytes of the same register file.
     */
    memory_region_init_io(&s->io, obj, &ati_r580_mmio_ops, s,
                          "ati-r580-io", ATI_R580_IO_SIZE);
    /*
     * A register access waiting for the command processor releases the
     * BQL; another vCPU may access the registers meanwhile.
     */
    s->mmio.disable_reentrancy_guard = true;
    s->io.disable_reentrancy_guard = true;

    /*
     * Real R5xx layout: BAR0 32-bit prefetchable VRAM, BAR1 I/O, BAR2
     * registers. A 32-bit BAR keeps VRAM below 4 GiB, where the 32-bit
     * kernels map it.
     */
    pci_register_bar(dev, 0, PCI_BASE_ADDRESS_SPACE_MEMORY |
                     PCI_BASE_ADDRESS_MEM_PREFETCH, &s->aper);
    pci_register_bar(dev, 1, PCI_BASE_ADDRESS_SPACE_IO, &s->io);
    pci_register_bar(dev, 2, PCI_BASE_ADDRESS_SPACE_MEMORY, &s->mmio);
    pci_config_set_interrupt_pin(dev->config, 1);

    if (pci_bus_is_express(pci_get_bus(dev))) {
        int ret = pcie_endpoint_cap_init(dev, 0x80);

        if (ret < 0) {
            error_setg(errp, "cannot add the PCI Express capability");
            return;
        }
    } else {
        dev->cap_present &= ~QEMU_PCI_CAP_EXPRESS;
    }


    s->cursor_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL,
                                   ati_r580_cursor_timer, s);
    ati_r580_avivo_init(s);
    ati_r580_raster_init(s);
    ati_r580_engine_init(s);
    ati_r580_disp_init(s);
    /* see "LOCKLESS REGISTER ACCESS"; BAR1 reaches only BQL registers */
    s->lockless = s->lockless_mmio && s->engine_on;
    if (s->lockless) {
        memory_region_enable_lockless_io(&s->mmio);
    }
}

static void ati_r580_exit(PCIDevice *dev)
{
    ATIR580State *s = ATI_R580(dev);
    unsigned i;

    ati_r580_disp_fini(s);
    ati_r580_engine_fini(s);
    ati_r580_raster_fini(s);
    ati_r580_avivo_fini(s);
    timer_free(s->cursor_timer);
    s->cursor_timer = NULL;
    qemu_graphic_console_close(s->con);
    if (s->cap_fp) {
        fclose(s->cap_fp);
        s->cap_fp = NULL;
    }
    ati_r580_gl_release(s, R580_GLR_RESET);
    for (i = 0; i < R300_GL_TEXCACHE; i++) {
        g_free(s->gl_tex[i].rgba);
    }
    ati_r580_gl_close(s->gl_ctx);
    s->gl_ctx = NULL;
    g_free(s->gl_before);
    g_free(s->gl_out);
    g_free(s->gl_sw);
    g_free(s->gl_texbuf);
    g_free(s->scan_dirty);
    g_free(s->scan_work);
    g_free(s->gl_verts);
    g_free(s->gl_zstage);
    g_free(s->pvs_cc);
    g_free(s->gl_zbefore);
    g_free(s->gl_zgpu);
    g_free(s->gl_tcx);
}

static int ati_r580_pre_save(void *opaque)
{
    ati_r580_engine_wait(opaque);
    return 0;
}

static int ati_r580_post_load(void *opaque, int version_id)
{
    ATIR580State *s = opaque;

    s->force_redraw = true;
    s->lut_gen++;
    ati_r580_update_irq(s);
    return 0;
}

static const VMStateDescription vmstate_ati_r580 = {
    .name = "ati-radeon-x1900",
    .version_id = 1,
    .minimum_version_id = 1,
    .pre_save = ati_r580_pre_save,
    .post_load = ati_r580_post_load,
    .fields = (const VMStateField[]) {
        VMSTATE_PCI_DEVICE(parent_obj, ATIR580State),
        VMSTATE_UINT32_ARRAY(regs, ATIR580State, ATI_R580_NUM_REGS),
        VMSTATE_UINT32_ARRAY(plls, ATIR580State, ATI_R580_NUM_PLLS),
        VMSTATE_UINT32_ARRAY(mc_ind, ATIR580State, 256),
        VMSTATE_UINT32_ARRAY(pcie_ind, ATIR580State, 256),
        VMSTATE_BUFFER_UNSAFE(lut, ATIR580State, 0, 2 * 256 * 3),
        VMSTATE_BOOL_ARRAY(vblank_occurred, ATIR580State, ATI_R580_NUM_CRTC),
        VMSTATE_BOOL(hpd1_pending, ATIR580State),
        VMSTATE_UINT32(frame_count, ATIR580State),
        VMSTATE_END_OF_LIST()
    }
};

static const Property ati_r580_properties[] = {
    /*
     * Diagnostic only: lay an instrumented window over part of aperture 0
     * so CPU fills of that VRAM range become traceable. Off by default.
     */
    DEFINE_PROP_UINT32("fillwatch", ATIR580State, fillwatch_off, 0),
    DEFINE_PROP_UINT32("fillwatch-size", ATIR580State, fillwatch_size, 0),
    DEFINE_PROP_UINT32("device-id", ATIR580State, device_id,
                       PCI_DEVICE_ID_ATI_R580),
    DEFINE_PROP_UINT32("subsys-vendor-id", ATIR580State, subsys_vendor_id,
                       ATI_R580_SUBSYS_VENDOR),
    DEFINE_PROP_UINT32("subsys-id", ATIR580State, subsys_id,
                       ATI_R580_SUBSYS_ID),
    DEFINE_PROP_UINT32("revision", ATIR580State, revision, 0),
    /*
     * Diagnostic only: name a file and every draw the 3D engine runs is
     * written to it as a self-contained replay record for
     * doc/radeon9800/gl-replay/. Unset -- the default -- the capture code
     * never runs.
     */
    DEFINE_PROP_STRING("draw-capture", ATIR580State, cap_path),
    DEFINE_PROP_UINT32("draw-capture-max", ATIR580State, cap_max, 256),
    DEFINE_PROP_UINT32("draw-capture-max-px", ATIR580State, cap_max_px,
                       256 * 256),
    /*
     * Host-GPU offload: "off" (the default, and a provable no-op),
     * "on", or "verify". See ati_r580_gl.h.
     */
    DEFINE_PROP_STRING("gl", ATIR580State, gl_path),
    DEFINE_PROP_STRING("gl-api", ATIR580State, gl_api),
    /*
     * Diagnostic only (milestone M4): translate each vertex program the
     * guest uploads to GLSL and count whether the translator could
     * express it, readable as the "pvs-stats" property. Off by default,
     * and with it off r300_pvs_translate() returns before doing
     * anything at all.
     */
    DEFINE_PROP_BOOL("pvs-glsl", ATIR580State, pvs_glsl, false),
    /* command processor on its own thread; auto is on except under qtest */
    DEFINE_PROP_ON_OFF_AUTO("async-engine", ATIR580State, engine_async,
                            ON_OFF_AUTO_AUTO),
    /* register BAR without the BQL; needs the command processor thread */
    DEFINE_PROP_BOOL("lockless-mmio", ATIR580State, lockless_mmio, true),
    /* refresh conversion on its own thread; auto is on except under qtest */
    DEFINE_PROP_ON_OFF_AUTO("async-display", ATIR580State, async_display,
                            ON_OFF_AUTO_AUTO),
    /*
     * How long a decoded texture may live: "dirty" (the default and the
     * rule the decode depends on), "burst" (the M3 lifetime, kept as
     * the A/B arm), or "never" -- which is DELIBERATELY WRONG and
     * exists only so the guard can be shown to bite. See the texture
     * cache lifetime block in ati_r580_3d.c.
     */
    DEFINE_PROP_STRING("gl-texlife", ATIR580State, gl_texlife_path),
    /*
     * Decoded textures kept: 0 or 32 is the linear 32-entry cache, up to
     * 256 a hashed one bounded by R300_GL_TEXCACHE_BYTES.
     */
    DEFINE_PROP_UINT32("gl-texcache-slots", ATIR580State, gl_tex_slots, 256),
    /*
     * Depth- and stencil-tested draws on the host GPU, against a copy of
     * the Z buffer kept beside the colour target ("GL-OWNED DEPTH
     * BUFFER" in ati_r580_3d.c). Off sends them to the software path.
     */
    DEFINE_PROP_BOOL("gl-depth", ATIR580State, gl_depth, true),
    /* occlusion-queried draws: counted on the GPU, or off = software */
    DEFINE_PROP_BOOL("gl-zpass", ATIR580State, gl_zpass, true),
    DEFINE_PROP_BOOL("gl-async-compile", ATIR580State, gl_async, true),
    DEFINE_PROP_UINT32("gl-compile-workers", ATIR580State, gl_workers, 2),
    /*
     * Keep the GPU's depth buffer across releases and write it back only
     * when something reads it; off writes it back at every release.
     */
    DEFINE_PROP_BOOL("gl-depth-resident", ATIR580State, gl_zlazy, true),
    /* threads a draw's rows are split across: 0 = auto, 1 = serial */
    DEFINE_PROP_UINT32("raster-threads", ATIR580State, raster_threads, 0),
    DEFINE_PROP_UINT64("x-raster-split", ATIR580State, raster_tri_split, 0),
    DEFINE_PROP_UINT64("x-raster-serial", ATIR580State, raster_tri_serial, 0),
    DEFINE_EDID_PROPERTIES(ATIR580State, edid_info),
};

static const char *const ati_r580_gap_names[R580_GAP_MAX] = {
    [R580_GAP_P3_OPCODE]    = "packet3 opcode",
    [R580_GAP_PRIM]         = "primitive type",
    [R580_GAP_VTX_WALK]     = "vertex walk mode",
    [R580_GAP_TEX_FORMAT]   = "texture format",
    [R580_GAP_BLEND_FACTOR] = "blend factor",
    [R580_GAP_VTX_PROGRAM]  = "vertex program",
    [R580_GAP_DEST_OFF_VRAM] = "destination outside VRAM",
    [R580_GAP_VS_VECTOR_OP] = "vertex vector opcode",
    [R580_GAP_VS_MATH_OP]   = "vertex math opcode",
    [R580_GAP_VS_DST_FILE]  = "vertex destination file",
    [R580_GAP_TEX_SWIZZLE]  = "texture component select",
    [R580_GAP_AOS_ARRAYS]   = "vertex arrays bound",
    [R580_GAP_VTX_DATA_TYPE] = "vertex element data type",
    [R580_GAP_FS_RGB_OP]    = "fragment colour opcode",
    [R580_GAP_FS_ALPHA_OP]  = "fragment alpha opcode",
    [R580_GAP_FS_TEX_OP]    = "fragment texture instruction",
    [R580_GAP_FS_INDIRECT]  = "fragment indirection level",
    [R580_GAP_FS_RS_ROUTE]  = "rasterizer attribute routing",
    [R580_GAP_FS_OUT_FMT]   = "fragment output format",
    [R580_GAP_ZB_FORMAT]    = "depth buffer format",
    [R580_GAP_CB_FORMAT]    = "colour buffer format",
    [R580_GAP_VS_ADDR_MODE] = "vertex operand address mode",
    [R580_GAP_UCP]          = "user clip planes",
    [R580_GAP_ZPASS]        = "occlusion counter use",
};

void ati_r580_note_gap(ATIR580State *s, ATIR580GapKind kind, unsigned idx)
{
    if (kind >= R580_GAP_MAX || idx >= R580_GAP_SLOTS) {
        return;
    }
    if (!s->gap_count[kind][idx]) {
        warn_report("ati-radeon-x1900: unimplemented %s 0x%x -- the guest "
                    "asked for it and did not get it",
                    ati_r580_gap_names[kind], idx);
    }
    if (s->gap_count[kind][idx] != UINT32_MAX) {
        s->gap_count[kind][idx]++;
    }
}

/*
 * `gaps` property: everything ati_r580_note_gap() has seen this run,
 * with use counts. Read it from the monitor with
 *   qom-get /machine/peripheral-anon/device[N] gaps
 * after exercising a guest, and anything the model quietly ignores is
 * named rather than left to be inferred from a wrong-looking screen.
 *
 * Registers are counted separately, by the `silent-regs` property
 * below: a register this model stores but never reads produces no gap
 * from inside any implemented path, so those writes are checked
 * against the generated ati_r580_audit.h bitmap instead. `gaps` stays
 * register-free deliberately -- the Regression protocol's acceptance
 * is the literal string "none", and silent registers are diagnostic
 * leads, not automatic defects.
 */
static char *ati_r580_get_gaps(Object *obj, Error **errp)
{
    ATIR580State *s = ATI_R580(obj);
    GString *out = g_string_new(NULL);
    unsigned k, i;

    for (k = 0; k < R580_GAP_MAX; k++) {
        for (i = 0; i < R580_GAP_SLOTS; i++) {
            if (s->gap_count[k][i]) {
                g_string_append_printf(out, "%s%s 0x%x: %u",
                                       out->len ? "\n" : "",
                                       ati_r580_gap_names[k], i,
                                       s->gap_count[k][i]);
            }
        }
    }
    if (!out->len) {
        g_string_append(out, "none");
    }
    return g_string_free(out, FALSE);
}

/*
 * `silent-regs` property: every register the guest wrote this run that
 * no model code reads, with write counts -- the RBBM_GUICNTL class,
 * which `gaps` cannot see. Read it from the monitor with
 *   qom-get /machine/peripheral-anon/device[N] silent-regs
 * A row here is a lead, not a verdict: WAIT_UNTIL or a cache-control
 * register is harmless to ignore, RE_SHADE_MODEL is not. The bitmap it
 * checks against is generated (ati_r580_audit.h); regenerate it when
 * the model learns a register, or that register keeps being reported.
 */
static char *ati_r580_get_silent_regs(Object *obj, Error **errp)
{
    ATIR580State *s = ATI_R580(obj);
    GString *out = g_string_new(NULL);
    unsigned i;

    for (i = 0; i < R580_SILENT_REG_WORDS; i++) {
        if (s->silent_reg_count[i]) {
            g_string_append_printf(out, "%s0x%04x %s: %u",
                                   out->len ? "\n" : "",
                                   i << 2,
                                   ati_r580_reg_name(i << 2),
                                   s->silent_reg_count[i]);
        }
    }
    if (!out->len) {
        g_string_append(out, "none");
    }
    return g_string_free(out, FALSE);
}

static const char *const ati_r580_gl_fb_names[R580_GLF_MAX] = {
    [R580_GLF_PRIM]     = "primitive",
    [R580_GLF_RESOLVE]  = "aa resolve",
    [R580_GLF_RECT]     = "destination rect",
    [R580_GLF_ALIGN]    = "destination alignment",
    [R580_GLF_VRAMEND]  = "rectangle past VRAM",
    [R580_GLF_XOR]      = "non-uniform swapper",
    [R580_GLF_CLIPRULE] = "cliprect truth table",
    [R580_GLF_TEXTURE]  = "texture size",
    [R580_GLF_SELFBLEND] = "self-overlapping blend",
    [R580_GLF_SURFACE]  = "render target size",
    [R580_GLF_BACKEND]  = "backend declined",
    [R580_GLF_FSPROG]   = "fragment program refused",
    [R580_GLF_CBFMT]    = "16bpp or GART colour buffer",
    [R580_GLF_ZTEST]    = "depth or stencil test",
    [R580_GLF_ZPASS]    = "occlusion query (gl-zpass=off)",
    [R580_GLF_PROGWAIT] = "program still being built",
};

const char *ati_r580_gl_fb_name(ATIR580GlFallback why)
{
    return why < R580_GLF_MAX ? ati_r580_gl_fb_names[why] : "?";
}

/*
 * `pvs-stats` property: how much of what the guest actually uploads the
 * PVS-to-GLSL translation can express (milestone M4). Read it from the
 * monitor with
 *   qom-get /machine/peripheral-anon/device[N] pvs-stats
 *
 * It counts PROGRAMS, not draws -- a program is translated once, when
 * the control registers first name its instruction range. A refusal is
 * also noted as an ordinary gap, so `gaps` names the opcode too.
 */
static char *ati_r580_get_pvs(Object *obj, Error **errp)
{
    ATIR580State *s = ATI_R580(obj);
    GString *out = g_string_new(NULL);

    if (!s->pvs_glsl) {
        g_string_append(out, "off");
        return g_string_free(out, FALSE);
    }
    g_string_append_printf(out, "programs %" PRIu64 " translated, %" PRIu64
                           " refused (vector op %" PRIu64 ", math op %"
                           PRIu64 ", dst file %" PRIu64 ", address mode %"
                           PRIu64 ")",
                           s->pvs_tr_ok, s->pvs_tr_refused,
                           s->pvs_tr_by_reason[0], s->pvs_tr_by_reason[1],
                           s->pvs_tr_by_reason[2], s->pvs_tr_by_reason[3]);
    g_string_append_printf(out, "\nlast: %u bytes of GLSL, %u constants,"
                           " in 0x%x, out 0x%x", s->pvs_tr_last_bytes,
                           s->pvs_tr_last_nconst, s->pvs_tr_last_in,
                           s->pvs_tr_last_out);
    return g_string_free(out, FALSE);
}

/*
 * `us-stats` property: what the fragment stage actually ran (milestone
 * M5). Read it from the monitor with
 *   qom-get /machine/peripheral-anon/device[N] us-stats
 *
 * Unlike `pvs-stats` this one is never off: every draw's colour comes
 * from the guest's own fragment program now, so how many of them the
 * interpreter could execute and how many the GLSL translator could
 * express is the coverage measurement for the whole pixel path. A
 * refusal is also noted as an ordinary gap, so `gaps` names the
 * construct.
 */
static char *ati_r580_get_us(Object *obj, Error **errp)
{
    ATIR580State *s = ATI_R580(obj);
    GString *out = g_string_new(NULL);
    const R300UsProgram *p = &s->us_prog;

    g_string_append_printf(out, "draws %" PRIu64 ", refused %" PRIu64
                           "\nprogram decodes %" PRIu64
                           " (translated %" PRIu64 ", refused %" PRIu64 ")",
                           s->us_draws, s->us_refused,
                           s->us_glsl_ok_n + s->us_glsl_refused_n,
                           s->us_glsl_ok_n, s->us_glsl_refused_n);
    g_string_append_printf(out, "\nlast: %u alu at %u, %u tex at %u, "
                           "%u regs, texdst %d, colours (%d,%d), "
                           "%u bytes of GLSL",
                           p->nalu, p->alu_first, p->ntex, p->tex_first,
                           p->nregs_used, p->tex_dst, p->rs.col_reg[0],
                           p->rs.col_reg[1],
                           (unsigned)strlen(s->us_glsl));
    return g_string_free(out, FALSE);
}

/*
 * `gl-stats` property: how much of this session the host-GPU offload actually
 * carried, and -- under gl=verify -- how far its pixels were from the
 * software rasterizer's. Read it from the monitor with
 *   qom-get /machine/peripheral-anon/device[N] gl-stats
 *
 * The fallback tally is the coverage measurement: a draw family the
 * backend cannot render is named with its primitive type and counted,
 * so "the offload is on" is a number rather than an impression. The
 * verify histogram is the live analogue of the offline M1 harness --
 * the acceptance there is a maximum channel delta of 1 with at least
 * 99.9% of pixels at delta 0.
 */
/*
 * `engine-stats`: whether the command processor has its own thread, how
 * often the guest woke it, and how often and how long a CPU register
 * access waited for it.
 */
static char *ati_r580_get_engine(Object *obj, Error **errp)
{
    ATIR580State *s = ATI_R580(obj);

    if (!s->engine_on) {
        return g_strdup("synchronous");
    }
    return g_strdup_printf("thread, %s\nkicks %" PRIu64 "\nwaits %" PRIu64
                           ", %" PRIu64 " us\nbql writes %" PRIu64
                           "\nindirect buffers %" PRIu64
                           "\nscratch writes %" PRIu64
                           "\nlockless mmio %s\nclaims %" PRIu64
                           ", waited %" PRIu64 ", %" PRIu64 " us"
                           ", found busy %" PRIu64,
                           qatomic_read(&s->engine_busy) ? "busy" : "idle",
                           s->engine_kicks, s->engine_waits,
                           s->engine_wait_us, s->engine_bql_writes,
                           s->engine_ibs, s->engine_scratch,
                           s->lockless ? "on" : "off", s->engine_claims,
                           s->engine_claim_waits, s->engine_claim_wait_us,
                           s->engine_claim_busy);
}

/* the occlusion counter's line of `gl-stats` */
static void ati_r580_zpass_stats(ATIR580State *s, GString *out)
{
    g_string_append_printf(out, "\nocclusion counter: %" PRIu64 " resets, %"
                           PRIu64 " dumps (%" PRIu64 " waited for the GPU), %"
                           PRIu64 " reads; counted %" PRIu64 " draws on the "
                           "GPU, %" PRIu64 " in software (%" PRIu64
                           " by gl-zpass=off)",
                           s->zq_resets, s->zq_dumps, s->zq_deferred,
                           s->zq_reads, s->zq_gldraws, s->zq_swdraws,
                           s->zq_offdraws);
    if (s->zq_v_draws) {
        g_string_append_printf(out, "\nverify occlusion %" PRIu64 " draws, %"
                               PRIu64 " counted differently; samples "
                               "software %" PRIu64 ", GPU %" PRIu64,
                               s->zq_v_draws, s->zq_v_bad, s->zq_v_sw,
                               s->zq_v_gl);
    }
}

static char *ati_r580_get_gl(Object *obj, Error **errp)
{
    ATIR580State *s = ATI_R580(obj);
    GString *out = g_string_new(NULL);
    uint64_t fb = 0;
    unsigned k, i;

    if (!s->gl_ctx) {
        g_string_append(out, "off");
        ati_r580_zpass_stats(s, out);
        return g_string_free(out, FALSE);
    }
    for (k = 0; k < R580_GLF_MAX; k++) {
        for (i = 0; i < R580_GAP_SLOTS; i++) {
            fb += s->gl_fb[k][i];
        }
    }
    g_string_append_printf(out, "mode %s\nbackend %s\ndrawn %" PRIu64
                           "\nfallback %" PRIu64 " (%.2f%% offloaded)",
                           s->gl_mode == R580_GL_VERIFY ? "verify"
                           : s->gl_fast ? "fast" : "on",
                           ati_r580_gl_describe(s->gl_ctx), s->gl_drawn, fb,
                           s->gl_drawn + fb
                           ? 100.0 * s->gl_drawn / (s->gl_drawn + fb) : 0.0);
    g_string_append_printf(out, "\nview volume: %" PRIu64 " draws clipped, %"
                           PRIu64 " triangles cut, %" PRIu64
                           " primitives dropped, %" PRIu64
                           " draws with user planes", s->clip_draws,
                           s->clip_tris, s->clip_drop, s->clip_ucp);
    if (s->gl_nowork) {
        /*
         * The share of the fallbacks that were proved to paint nothing
         * and so kept the resident target instead of tearing it back
         * off the GPU. Read beside "released by draw fell back" below:
         * on a workload made of empty quads those two used to be the
         * same number.
         */
        g_string_append_printf(out, "\n  of which %" PRIu64 " paint no "
                               "pixels (%.2f%%): release skipped",
                               s->gl_nowork,
                               fb ? 100.0 * s->gl_nowork / fb : 0.0);
    }
    /*
     * The coherency measurement the roadmap's E2.4 asks for: how often
     * the resident render target had to be handed back to VRAM, and how
     * many pixels crossed the bus each way. A flush count near the draw
     * count means the target is not staying resident and the batching
     * is not working.
     */
    g_string_append_printf(out, "\nresident target: %" PRIu64 " flushes, %"
                           PRIu64 " px out, %" PRIu64 " px in"
                           "\ntexture cache: %" PRIu64 " hits, %" PRIu64
                           " decodes (%.1f%%), life %s"
                           "\n  missed by: %" PRIu64 " not admitted (range "
                           "dirty), %" PRIu64 " stale, %" PRIu64 " written, %"
                           PRIu64 " drawn over, %" PRIu64 " evicted",
                           s->gl_flushes, s->gl_flush_px, s->gl_seed_px,
                           s->gl_tex_hit, s->gl_tex_miss,
                           s->gl_tex_hit + s->gl_tex_miss
                           ? 100.0 * s->gl_tex_hit /
                             (s->gl_tex_hit + s->gl_tex_miss) : 0.0,
                           s->gl_texlife == R580_TEXLIFE_BURST ? "burst"
                           : s->gl_texlife == R580_TEXLIFE_NEVER
                             ? "never (CONTROL: invalidation OFF)" : "dirty",
                           s->gl_tex_noadmit, s->gl_tex_stale,
                           s->gl_tex_wrote, s->gl_tex_over, s->gl_tex_evict);
    g_string_append_printf(out, "\n  slots %u %s, %zu KiB held, %" PRIu64
                           " dropped for space", s->gl_tex_n,
                           s->gl_tex_hashed ? "hashed" : "linear",
                           s->gl_tex_bytes >> 10, s->gl_tex_trim);
    /*
     * WHICH hook ended each residency. "the target is not staying
     * resident" is a symptom whose cure depends entirely on which rule
     * is ending it, and this is the only thing that says.
     */
    for (k = 0; k < R580_GLR_MAX; k++) {
        if (s->gl_rel[k]) {
            g_string_append_printf(out, "\n  released by %s: %" PRIu64
                                   ", %" PRIu64 " px",
                                   ati_r580_gl_rel_name(k), s->gl_rel[k],
                                   s->gl_rel_px[k]);
        }
    }
    /*
     * The 2D engine's three paths, counted apart: they are one cause
     * above but call for opposite fixes, and a path that names its
     * ranges instead of releasing outright is booked under "ranged
     * read", so the line above cannot answer "how much of this is the
     * 2D engine" on its own. See ATIR580Gl2dPath.
     */
    for (k = 0; k < R580_GL2D_MAX; k++) {
        if (s->gl_rel_2d[k]) {
            g_string_append_printf(out, "\n    2D %s: %" PRIu64
                                   ", %" PRIu64 " px",
                                   ati_r580_gl_2d_path_name(k),
                                   s->gl_rel_2d[k], s->gl_rel_2d_px[k]);
        }
    }
    {
        uint64_t ph, pl, pf;

        ati_r580_gl_prog_stats(s->gl_ctx, &ph, &pl, &pf);
        g_string_append_printf(out, "\nfragment shaders: %" PRIu64
                               " cache hits, %" PRIu64 " linked, %" PRIu64
                               " would not build", ph, pl, pf);
        if (s->gl_async) {
            uint64_t wl, ww, wq;
            unsigned wi;

            ati_r580_gl_worker_stats(s->gl_ctx, &ww, &wl, &wq, &wi);
            g_string_append_printf(out, "\nprogram worker: %" PRIu64
                                   " variants drawn once, %" PRIu64
                                   " would not link, %" PRIu64
                                   " draws not waited for, %u in flight",
                                   ww, wl, wq, wi);
        }
        if (ati_r580_gl_barriers(s->gl_ctx)) {
            uint64_t qu, qf, qw;

            ati_r580_gl_queue_stats(s->gl_ctx, &qu, &qf, &qw);
            g_string_append_printf(out, "\ntexture barriers: %" PRIu64
                                   "\ndraw queue: %" PRIu64 " passes in %"
                                   PRIu64 " flushes, %" PRIu64 " waves",
                                   ati_r580_gl_barriers(s->gl_ctx),
                                   qu, qf, qw);
        }
    }
    if (ati_r580_gl_depth(s->gl_ctx)) {
        g_string_append_printf(out, "\ndepth buffer %s: %" PRIu64
                               " draws tested on the GPU, %" PRIu64
                               " flushes, %" PRIu64 " px out, %" PRIu64
                               " px in, %" PRIu64 " clears"
                               "\n  kept across a release %" PRIu64
                               ", stale %" PRIu64 ", dropped %" PRIu64,
                               s->gl_depth ? "on" : "off",
                               s->gl_zdrawn, s->gl_zflushes, s->gl_zflush_px,
                               s->gl_zseed_px, s->gl_zclears, s->gl_zkept,
                               s->gl_zstale, s->gl_zdropped);
        g_string_append_printf(out, "\n  resident %s: kept unwritten across "
                               "%" PRIu64 " releases, guard laid %" PRIu64
                               ", CPU accesses trapped %" PRIu64 " (%" PRIu64
                               " wrote back), %" PRIu64 " px cleared on the "
                               "GPU only",
                               s->gl_zlazy ? "on" : "off", s->gl_zlazy_rel,
                               s->gl_zg_arms, s->gl_zg_traps, s->gl_zg_back,
                               s->gl_zclr_px);
        for (k = 0; k < R580_GLZD_MAX; k++) {
            static const char *const zd[R580_GLZD_MAX] = {
                [R580_GLZD_PARK] = "not kept at a release",
                [R580_GLZD_STALE] = "kept, then stale",
                [R580_GLZD_OTHER] = "another depth buffer or drawn into",
                [R580_GLZD_SOFT] = "software path",
                [R580_GLZD_WRITER] = "2D, CP or MM_DATA write",
                [R580_GLZD_GUARD] = "guest CPU access",
                [R580_GLZD_RESET] = "reset, backend or grown target",
                [R580_GLZD_VERIFY] = "gl=verify",
            };

            if (s->gl_zdrop_why[k]) {
                g_string_append_printf(out, "\n  dropped by %s: %" PRIu64,
                                       zd[k], s->gl_zdrop_why[k]);
            }
        }
    }
    if (s->gl_addblend) {
        /*
         * Blended draws GL's own blender rendered in a single pass
         * because their destination term was the destination itself.
         * These are the ones the ordered partition used to spread over
         * many passes or refuse; if this counter is high and the
         * ordered average below is low, the lever is doing its job.
         */
        g_string_append_printf(out, "\nadd-blend draws %" PRIu64,
                               s->gl_addblend);
    }
    if (s->gl_multipass) {
        /*
         * How much work the self-overlap ordering is doing. A blended
         * draw whose primitives cross itself is rendered in this many
         * ordered passes rather than falling back; if the average
         * climbs, the partition is the thing to look at.
         */
        g_string_append_printf(out, "\nordered draws %" PRIu64
                               " in %" PRIu64 " passes (%.2f average)",
                               s->gl_multipass, s->gl_passes,
                               (double)s->gl_passes / s->gl_multipass);
    }
    for (k = 0; k < R580_GLF_MAX; k++) {
        for (i = 0; i < R580_GAP_SLOTS; i++) {
            if (s->gl_fb[k][i]) {
                g_string_append_printf(out, "\n  %s prim %u: %" PRIu64,
                                       ati_r580_gl_fb_names[k], i,
                                       s->gl_fb[k][i]);
            }
        }
    }
    if (s->gl_v_draws) {
        uint64_t vpx = s->gl_v_hist[0] + s->gl_v_hist[1] +
                       s->gl_v_hist[2] + s->gl_v_hist[3];

        g_string_append_printf(out,
                               "\nverify %" PRIu64 " draws, %" PRIu64
                               " px, max delta %u, %" PRIu64
                               " draws with any delta > 1"
                               "\n  VALUE %" PRIu64 " px: d0 %" PRIu64
                               " d1 %" PRIu64 " d2-4 %" PRIu64
                               " d>4 %" PRIu64 " = %.4f%% at delta 0, "
                               "max delta %u"
                               "\n  COVER %" PRIu64 " px, %" PRIu64
                               " differing = %.4f%% of the compared area"
                               "\n  of the value-class disagreements above "
                               "1/255, %" PRIu64 " are in a multi-triangle "
                               "draw (a shared-edge tie the offline "
                               "classifier scores as EDGE)",
                               s->gl_v_draws, s->gl_v_px, s->gl_v_max,
                               s->gl_v_bad,
                               vpx, s->gl_v_hist[0], s->gl_v_hist[1],
                               s->gl_v_hist[2], s->gl_v_hist[3],
                               vpx ? 100.0 * s->gl_v_hist[0] / vpx : 0.0,
                               s->gl_v_vmax,
                               s->gl_v_cover_px, s->gl_v_cover,
                               100.0 * s->gl_v_cover / s->gl_v_px,
                               s->gl_v_mesh);
    }
    if (s->gl_vz_draws) {
        uint64_t vpx = s->gl_vz_px - s->gl_vz_cover_px;

        g_string_append_printf(out,
                               "\nverify depth %" PRIu64 " draws, %" PRIu64
                               " px\n  VALUE %" PRIu64 " px, %" PRIu64
                               " differing = %.6f%% exact"
                               "\n  COVER %" PRIu64 " px, %" PRIu64
                               " differing",
                               s->gl_vz_draws, s->gl_vz_px, vpx,
                               s->gl_vz_diff,
                               vpx ? 100.0 * (vpx - s->gl_vz_diff) / vpx
                                   : 100.0,
                               s->gl_vz_cover_px, s->gl_vz_cover);
    }
    ati_r580_zpass_stats(s, out);
    return g_string_free(out, FALSE);
}

/*
 * `scanout` property: what the last refresh drew and the D1 registers it
 * came from.
 */
static char *ati_r580_get_scanout(Object *obj, Error **errp)
{
    ATIR580State *s = ATI_R580(obj);
    GString *out = g_string_new(NULL);

    g_string_append_printf(out,
        "drawn: %ux%u bpp=%u pitch=%u fb_offset=0x%x\n"
        "D1CRTC_CONTROL=0x%08x D1GRPH_ENABLE=0x%x D1GRPH_CONTROL=0x%08x\n"
        "D1GRPH_PRIMARY_SURFACE_ADDRESS=0x%08x D1GRPH_PITCH=%u "
        "X %u..%u Y %u..%u\n"
        "MC_FB_LOCATION=0x%08x frame_period_ns=%" PRId64 " frames=%u\n"
        "D1CUR_CONTROL=0x%08x D1CUR_POSITION=0x%08x\n"
        "async display: %s, frames %" PRIu64 ", deferred ticks %" PRIu64,
        s->mode.width, s->mode.height, s->mode.bpp, s->mode.pitch,
        s->mode.fb_offset,
        s->regs[R5XX_D1CRTC_CONTROL >> 2], s->regs[R5XX_D1GRPH_ENABLE >> 2],
        s->regs[R5XX_D1GRPH_CONTROL >> 2],
        s->regs[R5XX_D1GRPH_PRIMARY_SURFACE_ADDRESS >> 2],
        s->regs[R5XX_D1GRPH_PITCH >> 2],
        s->regs[R5XX_D1GRPH_X_START >> 2], s->regs[R5XX_D1GRPH_X_END >> 2],
        s->regs[R5XX_D1GRPH_Y_START >> 2], s->regs[R5XX_D1GRPH_Y_END >> 2],
        s->regs[R580_MC_FB_LOCATION >> 2], s->frame_period_ns,
        s->frame_count,
        s->regs[R5XX_D1CUR_CONTROL >> 2], s->regs[R5XX_D1CUR_POSITION >> 2],
        s->disp_on ? "on" : "off", s->disp_frames, s->disp_deferred);
    return g_string_free(out, FALSE);
}

/* `edid` property: the EDID on DDC1, hex */
static char *ati_r580_get_edid(Object *obj, Error **errp)
{
    ATIR580State *s = ATI_R580(obj);
    GString *out = g_string_new(NULL);
    unsigned i;

    for (i = 0; s->ddc_slave[0] && i < ATI_R580_EDID_SIZE; i++) {
        g_string_append_printf(out, "%02x", s->ddc_slave[0]->blob[i]);
    }
    return g_string_free(out, FALSE);
}

/*
 * `palette` property: the 256-entry DAC LUT as the scanout is reading
 * it right now. In every depth this device drives, that RAM is a
 * per-channel GAMMA table rather than a colour map -- 32bpp indexes it
 * with each of the pixel's own three bytes -- so a screen whose colours
 * are wrong while VRAM is right is a question about these 768 numbers
 * and nothing else. Reading them beats inferring a curve from three
 * sampled pixels, which is how the 10.3 purple screen was first
 * described.
 *
 * Printed as "idx r g b" for the entries that are not the identity,
 * plus a count, so an identity ramp is one short line.
 */
static char *ati_r580_get_palette(Object *obj, Error **errp)
{
    ATIR580State *s = ATI_R580(obj);
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

/*
 * Whether the draw capture is recording. A settable property rather
 * than a -device one because it has to be turned on at a MOMENT: a boot
 * to the desktop spends a thousand records before the application under
 * study draws anything, so a corpus of one app is taken by arming this
 * from the monitor right before launching it --
 *   qom-set /machine/peripheral-anon/device[N] draw-capture-arm true
 * It defaults to on so that every capture recipe recorded before it
 * existed still means what it said; `draw-capture-arm=off` on the
 * -device line is what defers a capture to the monitor.
 */
static bool ati_r580_get_cap_arm(Object *obj, Error **errp)
{
    return ATI_R580(obj)->cap_arm;
}

static void ati_r580_set_cap_arm(Object *obj, bool value, Error **errp)
{
    ATI_R580(obj)->cap_arm = value;
}

static void ati_r580_instance_init(Object *obj)
{
    ATI_R580(obj)->cap_arm = true;
    PCI_DEVICE(obj)->cap_present |= QEMU_PCI_CAP_EXPRESS;
}

static void ati_r580_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    ResettableClass *rc = RESETTABLE_CLASS(klass);
    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);

    object_class_property_add_bool(klass, "draw-capture-arm",
                                   ati_r580_get_cap_arm,
                                   ati_r580_set_cap_arm);
    object_class_property_add_str(klass, "gaps", ati_r580_get_gaps, NULL);
    object_class_property_add_str(klass, "silent-regs",
                                  ati_r580_get_silent_regs, NULL);
    object_class_property_add_str(klass, "scanout", ati_r580_get_scanout,
                                  NULL);
    object_class_property_add_str(klass, "gl-stats", ati_r580_get_gl, NULL);
    object_class_property_add_str(klass, "engine-stats",
                                  ati_r580_get_engine, NULL);
    object_class_property_add_str(klass, "pvs-stats", ati_r580_get_pvs, NULL);
    object_class_property_add_str(klass, "us-stats", ati_r580_get_us, NULL);
    object_class_property_add_str(klass, "palette", ati_r580_get_palette,
                                  NULL);
    object_class_property_add_str(klass, "edid", ati_r580_get_edid, NULL);

    k->class_id  = PCI_CLASS_DISPLAY_VGA;
    k->vendor_id = PCI_VENDOR_ID_ATI;
    k->device_id = PCI_DEVICE_ID_ATI_R580;
    k->subsystem_vendor_id = ATI_R580_SUBSYS_VENDOR;
    k->subsystem_id = ATI_R580_SUBSYS_ID;
    k->revision  = 0x00;
    /* the synthetic AtomBIOS; romfile= replaces it, e.g. with a GOP added */
    k->romfile   = "ati-r580-atombios.rom";
    k->realize   = ati_r580_realize;
    k->exit      = ati_r580_exit;
    dc->desc     = "ATI Radeon X1900 XT (R580)";
    dc->vmsd     = &vmstate_ati_r580;
    device_class_set_props(dc, ati_r580_properties);
    rc->phases.hold = ati_r580_reset_hold;
    set_bit(DEVICE_CATEGORY_DISPLAY, dc->categories);
}

static const TypeInfo ati_r580_type_info = {
    .name           = TYPE_ATI_R580,
    .parent         = TYPE_PCI_DEVICE,
    .instance_size  = sizeof(ATIR580State),
    .instance_init  = ati_r580_instance_init,
    .class_init     = ati_r580_class_init,
    .interfaces     = (const InterfaceInfo[]) {
        { INTERFACE_PCIE_DEVICE },
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { },
    },
};

static void ati_r580_register_types(void)
{
    type_register_static(&ati_r580_type_info);
}

type_init(ati_r580_register_types)
