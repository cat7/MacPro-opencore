/*
 * ATI R580 3D engine, R500 pipeline on the host GPU: the device side.
 *
 * A draw the CPU vertex stage has turned into window-space triangles is
 * offered here. If its state is one the backend reproduces exactly, it
 * is rendered on the host GPU into a RESIDENT copy of its colour buffer;
 * otherwise it falls back to the CPU rasteriser, and the reason is
 * counted.
 *
 * SURFACES are the host copies of colour buffers: an image of the
 * buffer's pitch, a SHADOW of the bytes the image was last known to
 * agree with VRAM on, and the DRAWN rectangle where the image is newer
 * than VRAM. TEXTURES are host copies of sampled ranges, with a shadow
 * of their source bytes. A texture lying inside a surface with the same
 * pitch and texel size samples the surface's image itself.
 *
 * COHERENCY. Every host copy is checked against VRAM before its first
 * use in each EPOCH: the rows that differ from the shadow are uploaded
 * again. A new epoch starts whenever the guest may have written memory
 * a later command depends on: a new ring write pointer, and any command
 * submitted outside the command processor thread. Writers inside the
 * device (2D engine, CP write-backs, the CPU rasteriser) invalidate the
 * copies they overlap at once. The drawn rectangles go back to VRAM at
 * every point ati_r580_gl_release() names, before any reader inside the
 * device touches them (ati_r580_gl_sync()), and before a writer
 * overwrites them (ati_r580_gl_wrote()).
 *
 * VERIFY renders each offloaded draw on the GPU, reads the result back,
 * lets the CPU path render it into VRAM, compares the two and makes the
 * GPU copy equal to VRAM again. VRAM is then exactly what gl=off leaves.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include <math.h>
#include "qemu/error-report.h"
#include "qemu/log.h"
#include "qemu/host-utils.h"
#include "qemu/timer.h"
#include "hw/pci/pci_device.h"
#include "ati_r580_int.h"
#include "ati_r580_gl.h"
#include "ati_r580_r5us.h"
#include "ati_r580_r5tex.h"
#include "ati_r580_r5gpu.h"
#include "ati_r580_r5gpu_fs.h"
#include "ati_r580_r5xx.h"

#define R5G_NSURF       32
#define R5G_NTEX        96
#define R5G_NSCRATCH    4
#define R5G_MAX_TRIS    8192
#define R5G_MAX_PASSES  256
#define R5G_HIST        16
#define R5G_FS_VERSION  4

/* the shading the CPU path's compiler fused: v * m + 0.5, lerps */
#if defined(__aarch64__)
#define R5G_FMA_DEF "#define R5_FMA(a, b, c) fma(a, b, c)\n"
#else
#define R5G_FMA_DEF "#define R5_FMA(a, b, c) ((a) * (b) + (c))\n"
#endif

typedef enum {
    R5G_FB_PRIM,        /* lines or points */
    R5G_FB_CBUF,        /* several colour buffers, multiwrite */
    R5G_FB_CFMT,        /* colour buffer format */
    R5G_FB_CMEM,        /* colour buffer outside VRAM, or beyond its pitch */
    R5G_FB_ZS,          /* depth or stencil test */
    R5G_FB_FOG,
    R5G_FB_ROP,
    R5G_FB_AA,          /* AA resolve */
    R5G_FB_CLIP,        /* clip rectangles in effect */
    R5G_FB_POLY,        /* polygon mode */
    R5G_FB_PROG,        /* fragment program not translatable */
    R5G_FB_QUAD,        /* fragment program uses MDH/MDV */
    R5G_FB_TEX,         /* texture format or kind */
    R5G_FB_TMEM,        /* texture memory */
    R5G_FB_W,           /* a vertex with w <= 0 */
    R5G_FB_FLAT,        /* triangles with differing flat inputs */
    R5G_FB_PASS,        /* too many overlapping primitives */
    R5G_FB_SURF,        /* no host image for the colour buffer */
    R5G_FB_BACKEND,     /* the backend declined */
    R5G_FB_PENDING,     /* the program is still being compiled */
    R5G_FB_LINK,        /* the program would not build */
    R5G_FB_MAX
} R5GFallback;

static const char *const r5g_fb_names[R5G_FB_MAX] = {
    [R5G_FB_PRIM] = "lines/points",
    [R5G_FB_CBUF] = "multiple colour buffers",
    [R5G_FB_CFMT] = "colour format",
    [R5G_FB_CMEM] = "colour buffer memory",
    [R5G_FB_ZS] = "depth/stencil",
    [R5G_FB_FOG] = "fog",
    [R5G_FB_ROP] = "rop",
    [R5G_FB_AA] = "AA resolve",
    [R5G_FB_CLIP] = "clip rectangles",
    [R5G_FB_POLY] = "polygon mode",
    [R5G_FB_PROG] = "program (flow control/relative)",
    [R5G_FB_QUAD] = "program (MDH/MDV)",
    [R5G_FB_TEX] = "texture format",
    [R5G_FB_TMEM] = "texture memory",
    [R5G_FB_W] = "w <= 0",
    [R5G_FB_FLAT] = "mixed flat inputs",
    [R5G_FB_PASS] = "too many overlaps",
    [R5G_FB_SURF] = "no host surface",
    [R5G_FB_BACKEND] = "backend declined",
    [R5G_FB_PENDING] = "program compiling",
    [R5G_FB_LINK] = "program build failed",
};

typedef struct R5GSurf {
    bool live;
    uint32_t off, pitch, bpp;
    int w, h;
    R5GpuImage *img;
    uint8_t *shadow;
    int dx0, dy0, dx1, dy1;     /* newer on the host; end exclusive */
    uint64_t epoch, lru;
} R5GSurf;

typedef struct R5GTex {
    bool live;
    uint32_t addr, pitch, bpp;
    int w, h, levels;
    uint8_t endian;
    bool vram;
    uint32_t lo, hi;            /* VRAM bytes covered, when vram */
    uint32_t laddr[16], lpitch[16];
    int lw[16], lh[16];
    size_t loff[16];
    R5GpuImage *img;
    uint8_t *shadow;
    uint64_t epoch, lru;
} R5GTex;

typedef struct R5GScratch {
    R5GpuImage *img;
    int w, h, bpp;
} R5GScratch;

typedef struct ATIR580R5Gpu ATIR580R5Gpu;

struct ATIR580R5Gpu {
    ATIR580State *s;
    R580GlCtx *ctx;
    bool verify, dst_fetch;
    uint64_t epoch, tick;
    R5GSurf surf[R5G_NSURF];
    R5GTex tex[R5G_NTEX];
    R5GScratch scratch[R5G_NSCRATCH];
    GHashTable *fs;             /* key -> shader text */

    /* the draw being built */
    float *verts, *planes;
    size_t verts_cap, planes_cap;
    unsigned *order, *pass;
    int *pbox, *tbox;
    size_t tris_cap;

    /* verify: the GPU's pixels of the last draw */
    R5GSurf *vsf;
    int vx0, vy0, vx1, vy1;
    uint8_t *vbuf;
    size_t vbuf_cap;

    /* statistics */
    uint64_t offered, drawn, empty, verified, vdiff_draws, vdiff_px;
    uint64_t vdiff_px1, vmax;
    uint64_t fb[R5G_FB_MAX];
    uint64_t up_bytes, up_calls, rd_bytes, rd_calls, flushes, checks;
    uint64_t aliased, feedback, tex_hits, tex_new, surf_new, evicts;
    uint64_t passes, draw_ns, max_draw_ns, verts_n;
    unsigned dumps, dumps_small;
    uint8_t *vpre;
    R5GpuDraw vreq;
    struct {
        uint32_t addr, pitch;
        int mode, ox, oy;
    } vtex[R5G_UNITS];
    struct {
        uint32_t dst, cpitch, blend, outfmt, tx0;
        uint64_t key;
        int x0, y0, x1, y1;
        uint32_t ndiff, maxd, gpu, cpu;
        int fx, fy;
    } hist[R5G_HIST];
    unsigned hist_n;
};

static uint8_t *r5g_vram(ATIR580R5Gpu *g)
{
    return memory_region_get_ram_ptr(&g->s->vram);
}

/* ---- surfaces ---- */

static void r5g_surf_flush(ATIR580R5Gpu *g, R5GSurf *sf);
static void r5g_invalidate(ATIR580R5Gpu *g, uint32_t lo, uint32_t hi,
                           const R5GSurf *keep);

static void r5g_res_update(ATIR580R5Gpu *g)
{
    bool res = false, any = false;
    int i;

    for (i = 0; i < R5G_NSURF; i++) {
        if (g->surf[i].live) {
            any = true;
            res |= g->surf[i].dx1 > g->surf[i].dx0;
        }
    }
    for (i = 0; i < R5G_NTEX && !any; i++) {
        any = g->tex[i].live;
    }
    qatomic_set(&g->s->r5g_res, res);
    qatomic_set(&g->s->r5g_any, any);
}

static void r5g_surf_free(ATIR580R5Gpu *g, R5GSurf *sf)
{
    if (!sf->live) {
        return;
    }
    r5g_surf_flush(g, sf);
    ati_r580_gl_r5_image_free(g->ctx, sf->img);
    g_free(sf->shadow);
    memset(sf, 0, sizeof(*sf));
}

/* the drawn rectangle back into VRAM and the shadow */
static void r5g_surf_flush(ATIR580R5Gpu *g, R5GSurf *sf)
{
    int w = sf->dx1 - sf->dx0, h = sf->dy1 - sf->dy0, y;
    uint8_t *vram = r5g_vram(g);
    uint64_t lo, hi;

    if (w <= 0 || h <= 0) {
        return;
    }
    lo = sf->off + (uint64_t)sf->dy0 * sf->pitch + (uint64_t)sf->dx0 * sf->bpp;
    if (ati_r580_gl_r5_read(g->ctx, sf->img, sf->dx0, sf->dy0, w, h,
                            vram + lo, sf->pitch)) {
        for (y = sf->dy0; y < sf->dy1; y++) {
            size_t o = (size_t)y * sf->pitch + (size_t)sf->dx0 * sf->bpp;

            memcpy(sf->shadow + o, vram + sf->off + o, (size_t)w * sf->bpp);
        }
        g->rd_bytes += (uint64_t)w * h * sf->bpp;
        g->rd_calls++;
    } else {
        /* the copy is lost: VRAM keeps what it had, and is the truth */
        sf->epoch = 0;
    }
    hi = sf->off + (uint64_t)(sf->dy1 - 1) * sf->pitch +
         (uint64_t)sf->dx1 * sf->bpp;
    memory_region_set_dirty(&g->s->vram, lo, hi - lo);
    sf->dx0 = sf->dy0 = sf->dx1 = sf->dy1 = 0;
    g->flushes++;
    r5g_invalidate(g, lo, hi, sf);
}

/* upload rows [y0, y1) of [x0, x1) from VRAM and keep them as the shadow */
static void r5g_surf_put(ATIR580R5Gpu *g, R5GSurf *sf, int x0, int x1,
                         int y0, int y1)
{
    uint8_t *src = r5g_vram(g) + sf->off;
    int y;

    if (x1 <= x0 || y1 <= y0) {
        return;
    }
    ati_r580_gl_r5_upload(g->ctx, sf->img, 0, x0, y0, x1 - x0, y1 - y0,
                          src + (size_t)y0 * sf->pitch + (size_t)x0 * sf->bpp,
                          sf->pitch);
    for (y = y0; y < y1; y++) {
        size_t o = (size_t)y * sf->pitch + (size_t)x0 * sf->bpp;

        memcpy(sf->shadow + o, src + o, (size_t)(x1 - x0) * sf->bpp);
    }
    g->up_bytes += (uint64_t)(x1 - x0) * (y1 - y0) * sf->bpp;
    g->up_calls++;
}

/* differing bytes of one row segment, as a pixel range; false if none */
static bool r5g_row_diff(const uint8_t *a, const uint8_t *b, int x0, int x1,
                         int bpp, int *d0, int *d1)
{
    size_t n = (size_t)(x1 - x0) * bpp, i, j;

    a += (size_t)x0 * bpp;
    b += (size_t)x0 * bpp;
    if (!memcmp(a, b, n)) {
        return false;
    }
    for (i = 0; i < n && a[i] == b[i]; i++) {
    }
    for (j = n; j > i && a[j - 1] == b[j - 1]; j--) {
    }
    *d0 = x0 + (int)(i / bpp);
    *d1 = x0 + (int)((j + bpp - 1) / bpp);
    return true;
}

/* bring the image up to date with VRAM outside the drawn rectangle */
static void r5g_surf_check(ATIR580R5Gpu *g, R5GSurf *sf)
{
    const uint8_t *vram = r5g_vram(g) + sf->off;
    int y, ry0 = -1, rx0 = 0, rx1 = 0;

    if (sf->epoch == g->epoch) {
        return;
    }
    g->checks++;
    for (y = 0; y <= sf->h; y++) {
        int d0 = INT_MAX, d1 = -1, a, b;
        bool any = false;

        if (y < sf->h) {
            const uint8_t *v = vram + (size_t)y * sf->pitch;
            const uint8_t *s = sf->shadow + (size_t)y * sf->pitch;

            if (y >= sf->dy0 && y < sf->dy1) {
                if (r5g_row_diff(v, s, 0, sf->dx0, sf->bpp, &a, &b)) {
                    d0 = a; d1 = b; any = true;
                }
                if (r5g_row_diff(v, s, sf->dx1, sf->w, sf->bpp, &a, &b)) {
                    d0 = MIN(d0, a); d1 = MAX(d1, b); any = true;
                }
                if (any && d0 < sf->dx1 && d1 > sf->dx0) {
                    /* a run across the drawn rectangle: two uploads */
                    r5g_surf_put(g, sf, d0, MIN(d1, sf->dx0), y, y + 1);
                    r5g_surf_put(g, sf, MAX(d0, sf->dx1), d1, y, y + 1);
                    any = false;
                }
            } else {
                any = r5g_row_diff(v, s, 0, sf->w, sf->bpp, &d0, &d1);
            }
        }
        if (any) {
            if (ry0 < 0) {
                ry0 = y;
                rx0 = d0;
                rx1 = d1;
            } else {
                rx0 = MIN(rx0, d0);
                rx1 = MAX(rx1, d1);
            }
            continue;
        }
        if (ry0 >= 0) {
            r5g_surf_put(g, sf, rx0, rx1, ry0, y);
            ry0 = -1;
        }
    }
    sf->epoch = g->epoch;
}

static R5GSurf *r5g_surf_slot(ATIR580R5Gpu *g)
{
    R5GSurf *v = NULL;
    int i;

    for (i = 0; i < R5G_NSURF; i++) {
        if (!g->surf[i].live) {
            return &g->surf[i];
        }
        if (!v || g->surf[i].lru < v->lru) {
            v = &g->surf[i];
        }
    }
    g->evicts++;
    r5g_surf_free(g, v);
    return v;
}

static R5GSurf *r5g_surf_new(ATIR580R5Gpu *g, uint32_t off, uint32_t pitch,
                             uint32_t bpp, int h)
{
    R5GSurf *sf = r5g_surf_slot(g);
    int w = pitch / bpp;
    int i;

    /* anything else over this memory goes back and away first */
    for (i = 0; i < R5G_NSURF; i++) {
        R5GSurf *o = &g->surf[i];

        if (o->live && o != sf && off < o->off + (uint64_t)o->pitch * o->h &&
            o->off < off + (uint64_t)pitch * h) {
            r5g_surf_free(g, o);
        }
    }
    sf->img = ati_r580_gl_r5_image(g->ctx, w, h, 1, bpp);
    if (!sf->img) {
        return NULL;
    }
    sf->live = true;
    sf->off = off;
    sf->pitch = pitch;
    sf->bpp = bpp;
    sf->w = w;
    sf->h = h;
    sf->shadow = g_malloc((size_t)pitch * h);
    r5g_surf_put(g, sf, 0, w, 0, h);
    sf->epoch = g->epoch;
    g->surf_new++;
    qatomic_set(&g->s->r5g_any, true);
    return sf;
}

/* the surface holding [off, ...) with this layout, if any */
static R5GSurf *r5g_surf_find(ATIR580R5Gpu *g, uint32_t off, uint32_t pitch,
                              uint32_t bpp, int *ox, int *oy)
{
    int i;

    for (i = 0; i < R5G_NSURF; i++) {
        R5GSurf *sf = &g->surf[i];
        uint32_t d;

        if (!sf->live || sf->pitch != pitch || sf->bpp != bpp ||
            off < sf->off || off - sf->off >= (uint64_t)pitch * sf->h) {
            continue;
        }
        d = off - sf->off;
        if (d % bpp) {
            continue;
        }
        *oy = d / pitch;
        *ox = (d % pitch) / bpp;
        return sf;
    }
    return NULL;
}

/* the colour buffer's surface, `rows` rows from its origin */
static R5GSurf *r5g_target(ATIR580R5Gpu *g, uint32_t off, uint32_t pitch,
                           uint32_t bpp, int rows, int *ox, int *oy)
{
    R5GSurf *sf = r5g_surf_find(g, off, pitch, bpp, ox, oy);
    int h;

    if (sf && *oy + rows <= sf->h) {
        return sf;
    }
    if (sf) {
        /* grown from its own start: the old rows come with it */
        off = sf->off;
        rows += *oy;
        h = MAX(rows, sf->h);
        r5g_surf_free(g, sf);
    } else {
        *ox = *oy = 0;
        h = rows;
    }
    h = ROUND_UP(h, 64);
    if ((uint64_t)off + (uint64_t)pitch * h > ATI_R580_VRAM_SIZE) {
        h = (ATI_R580_VRAM_SIZE - off) / pitch;
    }
    if (h < rows) {
        return NULL;
    }
    sf = r5g_surf_new(g, off, pitch, bpp, h);
    if (sf && *oy == 0 && *ox == 0) {
        r5g_surf_find(g, off, pitch, bpp, ox, oy);
    }
    return sf;
}

/* ---- invalidation ---- */

static void r5g_invalidate(ATIR580R5Gpu *g, uint32_t lo, uint32_t hi,
                           const R5GSurf *keep)
{
    int i;

    for (i = 0; i < R5G_NTEX; i++) {
        R5GTex *t = &g->tex[i];

        if (t->live && (!t->vram || (lo < t->hi && t->lo < hi))) {
            t->epoch = 0;
        }
    }
    for (i = 0; i < R5G_NSURF; i++) {
        R5GSurf *sf = &g->surf[i];

        if (sf->live && sf != keep && lo < sf->off + (uint64_t)sf->pitch * sf->h &&
            sf->off < hi) {
            sf->epoch = 0;
        }
    }
}

/* flush every surface whose drawn bytes meet [lo, hi), except `keep` */
static void r5g_flush_range(ATIR580R5Gpu *g, uint64_t lo, uint64_t hi,
                            const R5GSurf *keep)
{
    int i;

    for (i = 0; i < R5G_NSURF; i++) {
        R5GSurf *sf = &g->surf[i];
        uint64_t a, b;

        if (!sf->live || sf == keep || sf->dx1 <= sf->dx0) {
            continue;
        }
        a = sf->off + (uint64_t)sf->dy0 * sf->pitch + (uint64_t)sf->dx0 * sf->bpp;
        b = sf->off + (uint64_t)(sf->dy1 - 1) * sf->pitch +
            (uint64_t)sf->dx1 * sf->bpp;
        if (lo < b && a < hi) {
            r5g_surf_flush(g, sf);
        }
    }
}

/* ---- textures ---- */

static void r5g_tex_free(ATIR580R5Gpu *g, R5GTex *t)
{
    if (t->live) {
        ati_r580_gl_r5_image_free(g->ctx, t->img);
        g_free(t->shadow);
    }
    memset(t, 0, sizeof(*t));
}

/* one row of texels from wherever the level lives, endian undone */
static const uint8_t *r5g_tex_row(ATIR580R5Gpu *g, const R5GTex *t, int l,
                                  int y, uint8_t *buf)
{
    uint32_t a = t->laddr[l] + (uint32_t)y * t->lpitch[l];
    size_t n = (size_t)t->lw[l] * t->bpp;
    uint32_t a0 = a & ~3u, k = a & 3, nw = (k + n + 3) / 4, i;

    if (t->vram && !t->endian) {
        return r5g_vram(g) + (a - t->addr) + (t->lo);
    }
    ati_r580_mc_read_block(g->s, a0, (uint32_t *)buf, nw);
    for (i = 0; i < nw; i++) {
        uint32_t w = cpu_to_le32(((uint32_t *)buf)[i]);

        if (t->endian) {
            w = r5xx_endian_swap(t->endian, w);
        }
        ((uint32_t *)buf)[i] = w;
    }
    return buf + k;
}

static void r5g_tex_check(ATIR580R5Gpu *g, R5GTex *t)
{
    g_autofree uint8_t *buf = NULL;
    int l, y;

    if (t->epoch == g->epoch) {
        return;
    }
    g->checks++;
    buf = g_malloc((size_t)t->lw[0] * t->bpp + 8);
    for (l = 0; l < t->levels; l++) {
        size_t rn = (size_t)t->lw[l] * t->bpp;
        int ry0 = -1;
        g_autofree uint8_t *run = g_malloc(rn * t->lh[l]);

        for (y = 0; y <= t->lh[l]; y++) {
            uint8_t *sh = t->shadow + t->loff[l] + rn * y;
            const uint8_t *src = y < t->lh[l] ? r5g_tex_row(g, t, l, y, buf)
                                              : NULL;

            if (src && memcmp(src, sh, rn)) {
                memcpy(sh, src, rn);
                if (ry0 < 0) {
                    ry0 = y;
                }
                continue;
            }
            if (ry0 >= 0) {
                ati_r580_gl_r5_upload(g->ctx, t->img, l, 0, ry0, t->lw[l],
                                      y - ry0, t->shadow + t->loff[l] +
                                      rn * ry0, rn);
                g->up_bytes += rn * (y - ry0);
                g->up_calls++;
                ry0 = -1;
            }
        }
    }
    t->epoch = g->epoch;
}

static R5GTex *r5g_tex_get(ATIR580R5Gpu *g, const R5xxTexSampler *smp)
{
    ATIR580State *s = g->s;
    R5GTex *t, *v = NULL;
    R5xxSurfaceLayout l0;
    uint32_t off;
    size_t len = 0;
    int i, l;

    r5xx_tex_level_layout(smp, 0, 0, &l0);
    for (i = 0; i < R5G_NTEX; i++) {
        t = &g->tex[i];
        if (t->live && t->addr == l0.gpu_addr && t->pitch == l0.pitch_bytes &&
            t->w == (int)smp->width && t->h == (int)smp->height &&
            t->bpp == smp->bpp && t->levels == (int)smp->num_levels &&
            t->endian == smp->endian) {
            t->lru = ++g->tick;
            g->tex_hits++;
            return t;
        }
        if (!t->live) {
            if (!v || v->live) {
                v = t;
            }
        } else if (!v || (v->live && t->lru < v->lru)) {
            v = t;
        }
    }
    r5g_tex_free(g, v);
    t = v;
    t->addr = l0.gpu_addr;
    t->pitch = l0.pitch_bytes;
    t->w = smp->width;
    t->h = smp->height;
    t->bpp = smp->bpp;
    t->levels = smp->num_levels;
    t->endian = smp->endian;
    t->vram = ati_r580_mc_to_vram(s, l0.gpu_addr, &off);
    t->lo = UINT32_MAX;
    t->hi = 0;
    for (l = 0; l < t->levels; l++) {
        R5xxSurfaceLayout ll;
        uint32_t lo;

        r5xx_tex_level_layout(smp, l, 0, &ll);
        t->laddr[l] = ll.gpu_addr;
        t->lpitch[l] = ll.pitch_bytes;
        t->lw[l] = ll.width;
        t->lh[l] = ll.height;
        t->loff[l] = len;
        len += (size_t)ll.width * t->bpp * ll.height;
        if (t->vram) {
            if (!ati_r580_mc_to_vram(s, ll.gpu_addr, &lo) ||
                (uint64_t)lo + (uint64_t)ll.pitch_bytes * (ll.height - 1) +
                (uint64_t)ll.width * t->bpp > ATI_R580_VRAM_SIZE ||
                lo - off != ll.gpu_addr - l0.gpu_addr) {
                return NULL;
            }
            t->lo = MIN(t->lo, lo);
            t->hi = MAX(t->hi, lo + ll.pitch_bytes * (ll.height - 1) +
                               ll.width * t->bpp);
        }
    }
    /* r5g_tex_row reads VRAM relative to the level-0 address */
    if (t->vram) {
        t->lo = off;
    }
    t->img = ati_r580_gl_r5_image(g->ctx, t->w, t->h, t->levels, t->bpp);
    if (!t->img) {
        memset(t, 0, sizeof(*t));
        return NULL;
    }
    t->shadow = g_malloc(len);
    /* differs from any source: the first check uploads everything */
    for (l = 0; l < t->levels; l++) {
        g_autofree uint8_t *buf = g_malloc((size_t)t->lw[l] * t->bpp + 8);
        size_t rn = (size_t)t->lw[l] * t->bpp;
        int y;

        for (y = 0; y < t->lh[l]; y++) {
            const uint8_t *src = r5g_tex_row(g, t, l, y, buf);
            uint8_t *sh = t->shadow + t->loff[l] + rn * y;

            memcpy(sh, src, rn);
            sh[0] ^= 0xff;
        }
    }
    t->live = true;
    t->epoch = 0;
    t->lru = ++g->tick;
    g->tex_new++;
    qatomic_set(&g->s->r5g_any, true);
    return t;
}

/* ---- scratch images for feedback ---- */

static R5GpuImage *r5g_scratch(ATIR580R5Gpu *g, int k, int w, int h, int bpp)
{
    R5GScratch *sc = &g->scratch[k];

    if (sc->img && (sc->w < w || sc->h < h || sc->bpp != bpp)) {
        w = MAX(w, sc->w);
        h = MAX(h, sc->h);
        ati_r580_gl_r5_image_free(g->ctx, sc->img);
        sc->img = NULL;
    }
    if (!sc->img) {
        sc->w = ROUND_UP(w, 64);
        sc->h = ROUND_UP(h, 64);
        sc->bpp = bpp;
        sc->img = ati_r580_gl_r5_image(g->ctx, sc->w, sc->h, 1, bpp);
    }
    return sc->img;
}

/* ---- the fragment shader ---- */

static uint64_t r5g_hash(uint64_t h, uint64_t v)
{
    int i;

    for (i = 0; i < 8; i++) {
        h = (h ^ ((v >> (8 * i)) & 0xff)) * 1099511628211ull;
    }
    return h;
}

static char *r5g_replace(char *src, const char *from, const char *to)
{
    gchar **parts = g_strsplit(src, from, -1);
    char *r = g_strjoinv(to, parts);

    g_strfreev(parts);
    g_free(src);
    return r;
}

/* NULL when the program cannot run here */
static const char *r5g_fs(ATIR580R5Gpu *g, const R500UsDecoded *dec,
                          unsigned ninputs, uint32_t flat, uint32_t bound,
                          int planes, const R5GpuDraw *rq, uint64_t *key,
                          R5GFallback *why)
{
    uint8_t kinds[16] = { 0 };
    uint32_t used = r500_us_tex_units_used(dec);
    uint64_t k = 1469598103934665603ull;
    char *prog, *text;
    GString *o;
    unsigned i;

    k = r5g_hash(k, r500_us_hash(dec));
    k = r5g_hash(k, ninputs | ((uint64_t)flat << 8) | ((uint64_t)planes << 32) |
                    ((uint64_t)R5G_FS_VERSION << 40));
    k = r5g_hash(k, bound | ((uint64_t)used << 16));
    for (i = 0; i < 4; i++) {
        k = r5g_hash(k, (uint32_t)rq->ki[i][0] | ((uint64_t)rq->ki[i][1] << 16));
        k = r5g_hash(k, (uint32_t)rq->ki[i][2] | ((uint64_t)rq->ki[i][3] << 16));
    }
    for (i = 0; i < R5G_UNITS; i++) {
        if (bound & (1u << i)) {
            int c;

            k = r5g_hash(k, rq->ti[i][0][3] | ((uint64_t)rq->ti[i][1][0] << 16) |
                            ((uint64_t)rq->ti[i][1][1] << 32));
            for (c = 0; c < 4; c++) {
                k = r5g_hash(k, rq->ti[i][2][c] | (rq->ti[i][3][c] << 8));
            }
        }
    }
    *key = k;
    text = g_hash_table_lookup(g->fs, &k);
    if (text) {
        return *text ? text : NULL;
    }
    prog = r500_us_to_glsl(dec, kinds);
    if (prog && strstr(prog, "subgroupQuadBroadcast")) {
        free(prog);
        prog = NULL;
        *why = R5G_FB_QUAD;
    } else if (!prog) {
        *why = R5G_FB_PROG;
    }
    if (!prog) {
        uint64_t *kp = g_new(uint64_t, 1);

        *kp = k;
        g_hash_table_insert(g->fs, kp, g_strdup(""));
        return NULL;
    }
    text = g_strdup(prog);
    free(prog);
    text = r5g_replace(text, "gl_HelperInvocation", "false");
    text = r5g_replace(text, "(1.0/0.0)", "us_inf()");
    text = r5g_replace(text, "1.0 / sqrt(A.w)", "tx_div(1.0, sqrt(A.w))");
    text = r5g_replace(text, "1.0 / A.w", "tx_div(1.0, A.w)");
    text = r5g_replace(text, "vec3 R; float AV;",
                       "precise vec3 R; precise float AV;");
    for (i = R500_US_MAX_INPUTS; i-- > 0;) {
        char from[24], to[24];

        snprintf(from, sizeof(from), "us_input(%u)", i);
        if (i < ninputs && planes && !((flat >> i) & 1)) {
            snprintf(to, sizeof(to), "us_plane(%u)", i);
        } else if (i < ninputs) {
            snprintf(to, sizeof(to), "R5_IN(%u)", i);
        } else {
            snprintf(to, sizeof(to), "vec4(0.0)");
        }
        text = r5g_replace(text, from, to);
    }
    for (i = R5G_UNITS; i-- > 0;) {
        char from[24], to[24];

        snprintf(from, sizeof(from), "us_tex(%u, ", i);
        snprintf(to, sizeof(to), "us_tex_%u(", i);
        text = r5g_replace(text, from, to);
    }

    o = g_string_new(R5G_FMA_DEF);
    g_string_append_printf(o, "#define S_NIN %u\n#define S_PERSP %d\n",
                           ninputs, planes == 2);
    g_string_append_printf(o, "#define S_CF %d\n#define S_CMASK %d\n"
        "#define S_FL %d\n#define S_CS ivec4(%d, %d, %d, %d)\n"
        "#define S_B0 ivec4(%d, %d, %d, %d)\n"
        "#define S_B1 ivec4(%d, %d, %d, %d)\n",
        rq->ki[0][1], rq->ki[0][2], rq->ki[0][3],
        rq->ki[1][0], rq->ki[1][1], rq->ki[1][2], rq->ki[1][3],
        rq->ki[2][0], rq->ki[2][1], rq->ki[2][2], rq->ki[2][3],
        rq->ki[3][0], rq->ki[3][1], rq->ki[3][2], rq->ki[3][3]);
    for (i = 0; i < R5G_UNITS; i++) {
        if (bound & (1u << i)) {
            g_string_append_printf(o, "#define S_TFL_%u %d\n"
                "#define S_WRS_%u %d\n#define S_WRT_%u %d\n"
                "#define S_BITS_%u ivec4(%d, %d, %d, %d)\n"
                "#define S_SEL_%u ivec4(%d, %d, %d, %d)\n",
                i, rq->ti[i][0][3], i, rq->ti[i][1][0], i, rq->ti[i][1][1],
                i, rq->ti[i][2][0], rq->ti[i][2][1], rq->ti[i][2][2],
                rq->ti[i][2][3], i, rq->ti[i][3][0], rq->ti[i][3][1],
                rq->ti[i][3][2], rq->ti[i][3][3]);
        }
    }
    g_string_append(o, "vec4 us_out[4];\nfloat us_depth;\n"
                       "bool us_depth_written;\nbool us_killed;\n"
                       "bool us_keep;\n");
    g_string_append(o, r5g_fs_common);
    for (i = 0; i < R5G_UNITS; i++) {
        char num[8], *u;

        if (!(used & (1u << i))) {
            continue;
        }
        snprintf(num, sizeof(num), "%u", i);
        u = g_strdup((bound & (1u << i)) ? r5g_fs_unit : r5g_fs_unbound);
        u = r5g_replace(u, "@U", num);
        g_string_append(o, u);
        g_free(u);
    }
    g_string_append(o, text);
    g_string_append(o, r5g_fs_main);
    g_free(text);
    text = g_string_free(o, false);
    {
        uint64_t *kp = g_new(uint64_t, 1);

        *kp = k;
        g_hash_table_insert(g->fs, kp, text);
    }
    return text;
}

/* ---- primitives ---- */

static bool r5g_tri_overlap(const float *a, const float *b)
{
    const float *t[2] = { a, b };
    int i, e, k;

    for (i = 0; i < 2; i++) {
        for (e = 0; e < 3; e++) {
            const float *p = t[i] + 2 * e, *q = t[i] + 2 * ((e + 1) % 3);
            float nx = -(q[1] - p[1]), ny = q[0] - p[0];
            float mn[2] = { INFINITY, INFINITY }, mx[2] = { -INFINITY, -INFINITY };

            for (k = 0; k < 6; k++) {
                const float *v = t[k / 3] + 2 * (k % 3);
                float d = nx * v[0] + ny * v[1];

                mn[k / 3] = MIN(mn[k / 3], d);
                mx[k / 3] = MAX(mx[k / 3], d);
            }
            if (mx[0] <= mn[1] || mx[1] <= mn[0]) {
                return false;
            }
        }
    }
    return true;
}

static void r5g_grow(ATIR580R5Gpu *g, size_t tris, unsigned floats)
{
    if (tris > g->tris_cap) {
        g->tris_cap = MAX(tris, g->tris_cap * 2);
        g->order = g_renew(unsigned, g->order, g->tris_cap);
        g->pass = g_renew(unsigned, g->pass, g->tris_cap + 1);
        g->pbox = g_renew(int, g->pbox, 4 * (g->tris_cap + 1));
        g->tbox = g_renew(int, g->tbox, 4 * g->tris_cap);
    }
    if ((size_t)tris * 3 * floats > g->verts_cap) {
        g->verts_cap = MAX((size_t)tris * 3 * floats, g->verts_cap * 2);
        g->verts = g_renew(float, g->verts, g->verts_cap);
    }
}

/*
 * The inputs of an affine draw as plane equations through the triangles'
 * snapped corners, as the CPU rasteriser interpolates them; NULL when a
 * triangle's corners differ in w.
 */
static const float *r5g_planes(ATIR580R5Gpu *g, unsigned ntri,
                               unsigned ninputs, unsigned floats,
                               bool *persp)
{
    unsigned t, n, c, per = ninputs * 3 + 1;
    size_t need = (size_t)ntri * per * 4;

    *persp = false;
    if (!ninputs) {
        return NULL;
    }
    for (t = 0; t < ntri; t++) {
        const float *v = g->verts + (size_t)t * 3 * floats;

        if (v[3] != v[floats + 3] || v[3] != v[2 * floats + 3]) {
            *persp = true;
        }
    }
    if (need > g->planes_cap) {
        g->planes_cap = MAX(need, g->planes_cap * 2);
        g->planes = g_renew(float, g->planes, g->planes_cap);
    }
    for (t = 0; t < ntri; t++) {
        const float *v[3];
        double x0, y0, dx1, dy1, dx2, dy2, det, iw[3];
        float *o = g->planes + (size_t)t * per * 4;
        int k;

        v[0] = g->verts + (size_t)t * 3 * floats;
        v[1] = v[0] + floats;
        v[2] = v[1] + floats;
        x0 = v[0][0];
        y0 = v[0][1];
        dx1 = v[1][0] - x0;
        dy1 = v[1][1] - y0;
        dx2 = v[2][0] - x0;
        dy2 = v[2][1] - y0;
        det = dx1 * dy2 - dx2 * dy1;
        for (k = 0; k < 3; k++) {
            float w = v[k][3];

            iw[k] = *persp ? 1.0 / (w > 1e-30f ? w : 1e-30f) : 1.0;
        }
        for (n = 0; n <= ninputs; n++) {
            for (c = 0; c < 4; c++) {
                double a[3], A, B;

                for (k = 0; k < 3; k++) {
                    a[k] = n < ninputs ? iw[k] * v[k][4 + 4 * n + c]
                                       : (c ? 0.0 : iw[k]);
                }
                A = det ? ((a[1] - a[0]) * dy2 - (a[2] - a[0]) * dy1) / det
                        : 0;
                B = det ? (dx1 * (a[2] - a[0]) - dx2 * (a[1] - a[0])) / det
                        : 0;
                if (n < ninputs) {
                    o[12 * n + c] = A;
                    o[12 * n + 4 + c] = B;
                    o[12 * n + 8 + c] = a[0] - A * x0 - B * y0;
                } else if (!c) {
                    o[12 * n] = A;
                    o[12 * n + 1] = B;
                    o[12 * n + 2] = a[0] - A * x0 - B * y0;
                    o[12 * n + 3] = 0;
                }
            }
        }
    }
    return g->planes;
}

/* ---- the draw ---- */

static int r5g_cf(R5xxColorFormat f, uint32_t *bpp)
{
    switch (f) {
    case R5XX_COLOR_ARGB8888:   *bpp = 4; return R5G_CF_8888;
    case R5XX_COLOR_RGB565:     *bpp = 2; return R5G_CF_565;
    case R5XX_COLOR_ARGB1555:   *bpp = 2; return R5G_CF_1555;
    case R5XX_COLOR_ARGB4444:   *bpp = 2; return R5G_CF_4444;
    case R5XX_COLOR_ARGB2101010: *bpp = 4; return R5G_CF_2101010;
    case R5XX_COLOR_I8:         *bpp = 1; return R5G_CF_I8;
    default:                    return -1;
    }
}

static bool r5g_unit_ok(const R5xxTexSampler *smp)
{
    uint32_t fmt, i, nb = 0;

    if (!r5xx_tex_gpu_format(smp, &fmt, NULL, 0) || (fmt & 15) != 0 ||
        smp->is_3d || smp->is_cube || smp->gamma || smp->signed_mask ||
        smp->yuv_to_rgb || smp->depth_compare || smp->compressed ||
        (smp->bpp != 1 && smp->bpp != 2 && smp->bpp != 4) ||
        !smp->num_levels || smp->num_levels > 16 ||
        smp->width > 16384 || smp->height > 16384) {
        return false;
    }
    for (i = 0; i < 4; i++) {
        uint32_t b = (fmt >> (8 + 5 * i)) & 31;

        if (b > 10) {
            return false;
        }
        nb += b;
    }
    return nb && nb <= smp->bpp * 8;
}

static int r5g_fail(ATIR580R5Gpu *g, R5GFallback why)
{
    g->fb[why]++;
    return R5G_CPU;
}

int ati_r580_r5gpu_draw(ATIR580State *s, const R5GpuIn *in)
{
    ATIR580R5Gpu *g = s->r5g;
    const R5xxRaster *r = in->raster;
    const struct R5xxRasterDerived *d = &r->d;
    R5GpuDraw *rq;
    R5GSurf *sf;
    R5GFallback why = R5G_FB_PROG;
    uint32_t coff, cpitch, cbpp, hw, cmask, flat = 0, bound = 0;
    unsigned ninputs = MIN(r->num_fs_inputs, R5G_INPUTS), floats, ntri = 0;
    unsigned i, k, c, npass = 1;
    int ready, cf, ox, oy, sx0, sy0, sx1, sy1, bx0, by0, bx1, by1;
    int64_t t0 = get_clock();
    bool need_dst, have_flat = false, persp = false;
    const float sub = d->subpix;
    static const uint8_t pos_bgra[4] = { 2, 1, 0, 3 };

    if (!g) {
        return R5G_CPU;
    }
    g->offered++;
    if (!r->rt.num_cbufs || !r->rt.cbuf_enabled[0]) {
        g->empty++;
        return R5G_DONE;
    }
    if (r->rt.num_cbufs != 1 || d->num_multiwrites != 1) {
        return r5g_fail(g, R5G_FB_CBUF);
    }
    cf = r5g_cf(r->rt.cbuf_format[0], &cbpp);
    if (cf < 0 || d->cbuf_float[0] || r->rt.cbuf[0].bpp != cbpp) {
        return r5g_fail(g, R5G_FB_CFMT);
    }
    if (r->rt.zbuf_enabled && (d->z_enable || d->stencil_enable)) {
        return r5g_fail(g, R5G_FB_ZS);
    }
    if (d->fog_enable) {
        return r5g_fail(g, R5G_FB_FOG);
    }
    if (d->rop_enable) {
        return r5g_fail(g, R5G_FB_ROP);
    }
    if (d->aa_resolve) {
        return r5g_fail(g, R5G_FB_AA);
    }
    if (d->poly_mode_dual &&
        (d->poly_front_type != 2 || d->poly_back_type != 2)) {
        return r5g_fail(g, R5G_FB_POLY);
    }
    hw = d->cmask[0];
    cmask = 0;
    for (c = 0; c < 4; c++) {
        if ((hw >> c) & 1) {
            cmask |= 1u << pos_bgra[c];
        }
    }
    if (!cmask || d->sc_x1 < d->sc_x0 || d->sc_y1 < d->sc_y0) {
        g->empty++;
        return R5G_DONE;
    }
    for (i = 0; i < in->nprims; i++) {
        if (in->prims[i].kind != R5_PRIM_TRI) {
            return r5g_fail(g, R5G_FB_PRIM);
        }
        if (i && in->prims[i].flat_inputs != in->prims[0].flat_inputs) {
            return r5g_fail(g, R5G_FB_FLAT);
        }
    }
    if (in->nprims) {
        flat = in->prims[0].flat_inputs & ((1u << ninputs) - 1);
    }

    /* colour buffer */
    cpitch = r->rt.cbuf[0].pitch_bytes;
    if (!ati_r580_mc_to_vram(s, r->rt.cbuf[0].gpu_addr, &coff) ||
        (uint32_t)d->sc_x1 >= cpitch / cbpp || cpitch % cbpp ||
        (uint64_t)coff + (uint64_t)cpitch * (d->sc_y1 + 1) >
        ATI_R580_VRAM_SIZE) {
        return r5g_fail(g, R5G_FB_CMEM);
    }

    /* textures */
    for (i = 0; i < R5G_UNITS; i++) {
        if ((in->bound >> i) & 1) {
            if (!r5g_unit_ok(&in->samp[i])) {
                return r5g_fail(g, R5G_FB_TEX);
            }
            bound |= 1u << i;
        }
    }
    if (ctpop32(bound) > R5G_MAX_BOUND) {
        return r5g_fail(g, R5G_FB_TEX);
    }

    rq = g_new0(R5GpuDraw, 1);

    /* the triangles: snapped, culled, offset into the image */
    sf = r5g_target(g, coff, cpitch, cbpp, d->sc_y1 + 1, &ox, &oy);
    if (!sf) {
        g_free(rq);
        return r5g_fail(g, R5G_FB_SURF);
    }
    sf->lru = ++g->tick;
    floats = 4 + 4 * ninputs;
    r5g_grow(g, in->nprims + 1, floats);
    sx0 = d->sc_x0 + ox;
    sy0 = d->sc_y0 + oy;
    sx1 = d->sc_x1 + ox + 1;
    sy1 = d->sc_y1 + oy + 1;
    bx0 = by0 = INT_MAX;
    bx1 = by1 = INT_MIN;
    for (i = 0; i < in->nprims; i++) {
        const R5Prim *p = &in->prims[i];
        double area = ((double)p->v[1].x - p->v[0].x) *
                      ((double)p->v[2].y - p->v[0].y) -
                      ((double)p->v[2].x - p->v[0].x) *
                      ((double)p->v[1].y - p->v[0].y);
        int64_t X[3], Y[3], sa;
        bool front;
        int prov = d->provoking == 3 || d->provoking > 2 ? 2 : d->provoking;
        int *tb = &g->tbox[4 * ntri];
        float *vo = g->verts + (size_t)ntri * 3 * floats;
        int64_t mnx, mxx, mny, mxy;

        if (area == 0) {
            area = 1;
        }
        front = (area < 0) == d->front_ccw;
        if ((front && p->cull_front) || (!front && p->cull_back)) {
            continue;
        }
        for (k = 0; k < 3; k++) {
            X[k] = llrintf(p->v[k].x * sub);
            Y[k] = llrintf(p->v[k].y * sub);
            if (!(p->v[k].w > 0.0f) || !isfinite(p->v[k].w)) {
                g_free(rq);
                return r5g_fail(g, R5G_FB_W);
            }
        }
        sa = (X[1] - X[0]) * (Y[2] - Y[0]) - (X[2] - X[0]) * (Y[1] - Y[0]);
        if (!sa) {
            continue;
        }
        mnx = MIN(X[0], MIN(X[1], X[2]));
        mxx = MAX(X[0], MAX(X[1], X[2]));
        mny = MIN(Y[0], MIN(Y[1], Y[2]));
        mxy = MAX(Y[0], MAX(Y[1], Y[2]));
        tb[0] = MAX((int)floor((mnx - sub / 2) / sub) + ox, sx0);
        tb[1] = MAX((int)floor((mny - sub / 2) / sub) + oy, sy0);
        tb[2] = MIN((int)floor((mxx - sub / 2) / sub) + 1 + ox + 1, sx1);
        tb[3] = MIN((int)floor((mxy - sub / 2) / sub) + 1 + oy + 1, sy1);
        if (tb[2] <= tb[0] || tb[3] <= tb[1]) {
            continue;
        }
        bx0 = MIN(bx0, tb[0]);
        by0 = MIN(by0, tb[1]);
        bx1 = MAX(bx1, tb[2]);
        by1 = MAX(by1, tb[3]);
        for (k = 0; k < 3; k++) {
            const R5xxVertex *v = &p->v[k];
            float *o = vo + k * floats;
            unsigned a;

            o[0] = (float)((double)X[k] / sub) + ox;
            o[1] = (float)((double)Y[k] / sub) + oy;
            o[2] = v->z;
            o[3] = v->w;
            for (a = 0; a < ninputs; a++) {
                const R5xxVertex *src = ((flat >> a) & 1) ? &p->v[prov] : v;

                memcpy(o + 4 + 4 * a, src->attr[a].f, 16);
            }
        }
        have_flat |= flat != 0;
        ntri++;
    }
    if (!ntri) {
        g_free(rq);
        g->empty++;
        return R5G_DONE;
    }
    if (bx1 > sf->w) {
        /* pixels past the pitch land on the next row */
        g_free(rq);
        return r5g_fail(g, R5G_FB_SURF);
    }
    sx1 = MIN(sx1, sf->w);

    need_dst = d->blend_enable || cmask != 0xf;
    rq->pass = g->pass;
    rq->pbox = g->pbox;
    if (need_dst && !g->dst_fetch && ntri > 1) {
        unsigned *pn = g->order, maxp = 0, at;
        float tv[2][6];
        g_autofree float *tmp = NULL;

        if (ntri > R5G_MAX_TRIS) {
            g_free(rq);
            return r5g_fail(g, R5G_FB_PASS);
        }
        for (i = 0; i < ntri; i++) {
            const int *a = &g->tbox[4 * i];
            unsigned pi = 0;

            for (k = 0; k < 3; k++) {
                tv[0][2 * k] = g->verts[((size_t)i * 3 + k) * floats];
                tv[0][2 * k + 1] = g->verts[((size_t)i * 3 + k) * floats + 1];
            }
            for (c = 0; c < i; c++) {
                const int *b = &g->tbox[4 * c];
                unsigned q;

                if (pn[c] < pi || a[0] >= b[2] || b[0] >= a[2] ||
                    a[1] >= b[3] || b[1] >= a[3]) {
                    continue;
                }
                for (q = 0; q < 3; q++) {
                    tv[1][2 * q] = g->verts[((size_t)c * 3 + q) * floats];
                    tv[1][2 * q + 1] = g->verts[((size_t)c * 3 + q) * floats + 1];
                }
                if (r5g_tri_overlap(tv[0], tv[1])) {
                    pi = pn[c] + 1;
                }
            }
            pn[i] = pi;
            maxp = MAX(maxp, pi);
        }
        npass = maxp + 1;
        if (npass > R5G_MAX_PASSES) {
            g_free(rq);
            return r5g_fail(g, R5G_FB_PASS);
        }
        if (npass > 1) {
            /* regroup the triangles by pass, keeping their order */
            tmp = g_new(float, (size_t)ntri * 3 * floats);
            at = 0;
            for (k = 0; k < npass; k++) {
                int *pb = &g->pbox[4 * k];

                g->pass[k] = at * 3;
                pb[0] = pb[1] = INT_MAX;
                pb[2] = pb[3] = INT_MIN;
                for (i = 0; i < ntri; i++) {
                    if (pn[i] != k) {
                        continue;
                    }
                    memcpy(tmp + (size_t)at * 3 * floats,
                           g->verts + (size_t)i * 3 * floats,
                           3 * floats * sizeof(float));
                    pb[0] = MIN(pb[0], g->tbox[4 * i]);
                    pb[1] = MIN(pb[1], g->tbox[4 * i + 1]);
                    pb[2] = MAX(pb[2], g->tbox[4 * i + 2]);
                    pb[3] = MAX(pb[3], g->tbox[4 * i + 3]);
                    at++;
                }
            }
            memcpy(g->verts, tmp, (size_t)ntri * 3 * floats * sizeof(float));
        }
    }
    if (npass == 1) {
        g->pass[0] = 0;
        g->pbox[0] = bx0;
        g->pbox[1] = by0;
        g->pbox[2] = bx1;
        g->pbox[3] = by1;
    }
    g->pass[npass] = ntri * 3;
    rq->npass = npass;
    rq->planes = r5g_planes(g, ntri, ninputs, floats, &persp);

    /* the destination and the textures, up to date */
    r5g_flush_range(g, sf->off, sf->off + (uint64_t)sf->pitch * sf->h, sf);
    r5g_surf_check(g, sf);
    for (i = 0; i < R5G_UNITS; i++) {
        const R5xxTexSampler *smp = &in->samp[i];
        R5xxSurfaceLayout l0;
        R5GSurf *as;
        uint32_t toff, fmt = 0;
        int tx = 0, ty = 0;

        if (!(bound & (1u << i))) {
            continue;
        }
        r5xx_tex_level_layout(smp, 0, 0, &l0);
        r5xx_tex_gpu_format(smp, &fmt, NULL, 0);
        as = NULL;
        if (!smp->endian && ati_r580_mc_to_vram(s, l0.gpu_addr, &toff) &&
            (smp->num_levels == 1 ||
             (smp->mip_filter == R5XX_FILTER_NONE && (int)smp->min_lod == 0))) {
            as = r5g_surf_find(g, toff, l0.pitch_bytes, smp->bpp, &tx, &ty);
            if (as && (tx + (int)smp->width > as->w ||
                       ty + (int)smp->height > as->h)) {
                as = NULL;
                tx = ty = 0;
            }
        }
        g->vtex[i].addr = l0.gpu_addr;
        g->vtex[i].pitch = l0.pitch_bytes;
        g->vtex[i].mode = as ? (as == sf ? 2 : 1) : 0;
        if (as) {
            r5g_flush_range(g, as->off, as->off + (uint64_t)as->pitch * as->h, as);
            r5g_surf_check(g, as);
            as->lru = ++g->tick;
            g->aliased++;
            if (as == sf) {
                R5GpuImage *sc = r5g_scratch(g, i % R5G_NSCRATCH, smp->width,
                                             smp->height, smp->bpp);

                if (!sc || !ati_r580_gl_r5_copy(g->ctx, sc, 0, 0, sf->img,
                                                tx, ty, smp->width,
                                                smp->height)) {
                    g_free(rq);
                    return r5g_fail(g, R5G_FB_BACKEND);
                }
                g->feedback++;
                rq->img[i] = sc;
                tx = ty = 0;
            } else {
                rq->img[i] = as->img;
            }
        } else {
            R5GTex *t;

            if (ati_r580_mc_to_vram(s, l0.gpu_addr, &toff)) {
                r5g_flush_range(g, toff, toff + (uint64_t)l0.pitch_bytes *
                                smp->height * 2, NULL);
            }
            t = r5g_tex_get(g, smp);
            if (!t) {
                g_free(rq);
                return r5g_fail(g, R5G_FB_TMEM);
            }
            r5g_tex_check(g, t);
            rq->img[i] = t->img;
        }
        rq->ti[i][0][0] = smp->width;
        rq->ti[i][0][1] = smp->height;
        rq->ti[i][0][2] = smp->num_levels - 1;
        rq->ti[i][0][3] = (smp->is_rect ? R5G_TF_UNSCALED : 0) |
            (smp->min_filter == R5XX_FILTER_LINEAR ? R5G_TF_MIN_LIN : 0) |
            (smp->mag_filter == R5XX_FILTER_LINEAR ? R5G_TF_MAG_LIN : 0) |
            (smp->num_levels > 1 && smp->mip_filter == R5XX_FILTER_NEAREST ?
             R5G_TF_MIP_NEAR : 0) |
            (smp->num_levels > 1 && smp->mip_filter == R5XX_FILTER_LINEAR ?
             R5G_TF_MIP_LIN : 0) |
            (smp->mag_filter == R5XX_FILTER_NONE ? R5G_TF_MAG_UNDEF : 0) |
            (smp->min_filter == R5XX_FILTER_NONE ? R5G_TF_MIN_UNDEF : 0);
        rq->ti[i][1][0] = smp->wrap_s;
        rq->ti[i][1][1] = smp->wrap_t;
        rq->ti[i][1][2] = tx;
        rq->ti[i][1][3] = ty;
        for (c = 0; c < 4; c++) {
            rq->ti[i][2][c] = (fmt >> (8 + 5 * c)) & 31;
            rq->ti[i][3][c] = smp->swizzle[c];
        }
        rq->tf[i][0][0] = smp->lod_bias;
        rq->tf[i][0][1] = smp->min_lod;
        rq->tf[i][0][2] = smp->max_lod;
        memcpy(rq->tf[i][1], smp->border_color.f, 16);
    }
    rq->units = bound;

    rq->target = sf->img;
    rq->sx0 = sx0;
    rq->sy0 = sy0;
    rq->sx1 = sx1;
    rq->sy1 = sy1;
    rq->verts = g->verts;
    rq->nvert = ntri * 3;
    rq->ninputs = ninputs;
    rq->flat = have_flat ? flat : 0;
    rq->consts = in->consts;
    rq->ki[0][0] = cbpp * 8;
    rq->ki[0][1] = cf;
    rq->ki[0][2] = cmask;
    rq->ki[0][3] = (d->blend_enable ? R5G_DF_BLEND : 0) |
                   (d->alpha_test ? R5G_DF_ATEST : 0) |
                   (need_dst ? R5G_DF_DST : 0) |
                   (d->blend_enable && d->discard_mode ? R5G_DF_DISCARD : 0);
    for (c = 0; c < 4; c++) {
        rq->ki[1][c] = d->chan_src[0][c];
    }
    rq->ki[2][0] = d->rgb_eq;
    rq->ki[2][1] = d->rgb_src;
    rq->ki[2][2] = d->rgb_dst;
    rq->ki[2][3] = d->a_eq;
    rq->ki[3][0] = d->a_src;
    rq->ki[3][1] = d->a_dst;
    rq->ki[3][2] = d->discard_mode;
    rq->ki[3][3] = d->alpha_func;
    rq->ki[4][0] = d->clip_rule & 0xffff;
    for (c = 0; c < 4; c++) {
        rq->ki[5 + c][0] = d->cl_x0[c];
        rq->ki[5 + c][1] = d->cl_y0[c];
        rq->ki[5 + c][2] = d->cl_x1[c];
        rq->ki[5 + c][3] = d->cl_y1[c];
    }
    rq->ki[9][0] = ox;
    rq->ki[9][1] = oy;
    memcpy(rq->kf[0], d->blend_const.f, 16);
    rq->kf[1][0] = d->alpha_ref;

    rq->fs = r5g_fs(g, in->us, ninputs, flat, bound,
                    rq->planes ? 1 + persp : 0, rq,
                    &rq->key, &why);
    if (!rq->fs) {
        g_free(rq);
        return r5g_fail(g, why);
    }
    ready = ati_r580_gl_r5_prog_ready(g->ctx, rq, cbpp);
    if (ready != 1) {
        g_free(rq);
        return r5g_fail(g, ready ? R5G_FB_LINK : R5G_FB_PENDING);
    }
    if (g->verify && s->r5g_dump) {
        size_t n = (size_t)(bx1 - bx0) * (by1 - by0) * sf->bpp;

        g_free(g->vpre);
        g->vpre = g_malloc(n);
        ati_r580_gl_r5_read(g->ctx, sf->img, bx0, by0, bx1 - bx0, by1 - by0,
                            g->vpre, (bx1 - bx0) * sf->bpp);
    }
    if (!ati_r580_gl_r5_draw(g->ctx, rq)) {
        g_free(rq);
        /* the target may hold part of the draw: VRAM decides again */
        r5g_surf_free(g, sf);
        r5g_res_update(g);
        return r5g_fail(g, R5G_FB_BACKEND);
    }
    if (g->verify) {
        g->vreq = *rq;
    }
    g_free(rq);
    g->drawn++;
    g->passes += npass;
    g->verts_n += ntri * 3;
    t0 = get_clock() - t0;
    g->draw_ns += t0;
    g->max_draw_ns = MAX(g->max_draw_ns, (uint64_t)t0);

    if (g->verify) {
        size_t n = (size_t)(bx1 - bx0) * (by1 - by0) * sf->bpp;

        if (n > g->vbuf_cap) {
            g->vbuf_cap = n;
            g->vbuf = g_realloc(g->vbuf, n);
        }
        g->vsf = sf;
        g->vx0 = bx0;
        g->vy0 = by0;
        g->vx1 = bx1;
        g->vy1 = by1;
        if (!ati_r580_gl_r5_read(g->ctx, sf->img, bx0, by0, bx1 - bx0,
                                 by1 - by0, g->vbuf, (bx1 - bx0) * sf->bpp)) {
            g->vsf = NULL;
        }
        {
            unsigned h = g->hist_n % R5G_HIST;

            g->hist[h].dst = r->rt.cbuf[0].gpu_addr;
            g->hist[h].cpitch = r->regs.rb3d_colorpitch[0];
            g->hist[h].blend = r->regs.rb3d_blendcntl;
            g->hist[h].outfmt = r->regs.us_out_fmt[0];
            g->hist[h].tx0 = bound ? in->samp[ctz32(bound)].raw[4] : 0;
            g->hist[h].x0 = bx0;
            g->hist[h].y0 = by0;
            g->hist[h].x1 = bx1;
            g->hist[h].y1 = by1;
        }
        return R5G_VERIFY;
    }
    sf->dx0 = sf->dx1 > sf->dx0 ? MIN(sf->dx0, bx0) : bx0;
    sf->dy0 = sf->dy1 > sf->dy0 ? MIN(sf->dy0, by0) : by0;
    sf->dx1 = MAX(sf->dx1, bx1);
    sf->dy1 = MAX(sf->dy1, by1);
    /* what was decoded from these bytes is stale now */
    r5g_invalidate(g, sf->off + (uint64_t)by0 * sf->pitch,
                   sf->off + (uint64_t)by1 * sf->pitch, sf);
    qatomic_set(&s->r5g_res, true);
    qatomic_set(&s->r5g_any, true);
    return R5G_DONE;
}

/* the two renderings of a differing draw, as PPM files */
static void r5g_dump(ATIR580R5Gpu *g, const R5GSurf *sf, const uint8_t *vram)
{
    int w = g->vx1 - g->vx0, h = g->vy1 - g->vy0, x, y, k;

    for (k = 0; k < 2; k++) {
        g_autofree char *fn = g_strdup_printf("%s/r5diff-%02u-%s.ppm",
                                              g->s->r5g_dump, g->dumps,
                                              k ? "cpu" : "gpu");
        FILE *f = fopen(fn, "wb");

        if (!f) {
            return;
        }
        fprintf(f, "P6\n%d %d\n255\n", w, h);
        for (y = 0; y < h; y++) {
            const uint8_t *row = k ? vram + (size_t)(g->vy0 + y) * sf->pitch +
                                     (size_t)g->vx0 * 4
                                   : g->vbuf + (size_t)y * w * 4;

            for (x = 0; x < w; x++) {
                fputc(row[4 * x + 2], f);
                fputc(row[4 * x + 1], f);
                fputc(row[4 * x], f);
            }
        }
        fclose(f);
    }
    if (g->vpre) {
        g_autofree char *fn = g_strdup_printf("%s/r5diff-%02u-dst.bin",
                                              g->s->r5g_dump, g->dumps);
        FILE *f = fopen(fn, "wb");

        if (f) {
            fwrite(g->vpre, 4, (size_t)w * h, f);
            fclose(f);
        }
    }
    {
        g_autofree char *fn = g_strdup_printf("%s/r5diff-%02u.txt",
                                              g->s->r5g_dump, g->dumps);
        const R5GpuDraw *q = &g->vreq;
        FILE *f = fopen(fn, "w");
        unsigned i, u, fl = 4 + 4 * q->ninputs;

        if (f) {
            fprintf(f, "box %d,%d-%d,%d scissor %d,%d-%d,%d nvert %u "
                    "ninputs %u flat %x units %x npass %u key %016" PRIx64
                    "\n", g->vx0, g->vy0, g->vx1, g->vy1, q->sx0, q->sy0,
                    q->sx1, q->sy1, q->nvert, q->ninputs, q->flat, q->units,
                    q->npass, q->key);
            for (i = 0; i < q->npass && i < 64; i++) {
                fprintf(f, "pass %u %u %d %d %d %d\n", q->pass[i],
                        q->pass[i + 1], q->pbox[4 * i], q->pbox[4 * i + 1],
                        q->pbox[4 * i + 2], q->pbox[4 * i + 3]);
            }
            for (i = 0; i < R5G_KI; i++) {
                fprintf(f, "ki%u %d %d %d %d\n", i, q->ki[i][0], q->ki[i][1],
                        q->ki[i][2], q->ki[i][3]);
            }
            fprintf(f, "kf0 %g %g %g %g kf1 %g\n", q->kf[0][0], q->kf[0][1],
                    q->kf[0][2], q->kf[0][3], q->kf[1][0]);
            for (u = 0; u < R5G_UNITS; u++) {
                if (q->units & (1u << u)) {
                    fprintf(f, "unit %u ti %d %d %d %d / %d %d %d %d / %d %d %d "
                            "%d / %d %d %d %d tf %g %g %g\n", u,
                            q->ti[u][0][0], q->ti[u][0][1], q->ti[u][0][2],
                            q->ti[u][0][3], q->ti[u][1][0], q->ti[u][1][1],
                            q->ti[u][1][2], q->ti[u][1][3], q->ti[u][2][0],
                            q->ti[u][2][1], q->ti[u][2][2], q->ti[u][2][3],
                            q->ti[u][3][0], q->ti[u][3][1], q->ti[u][3][2],
                            q->ti[u][3][3], q->tf[u][0][0], q->tf[u][0][1],
                            q->tf[u][0][2]);
                    {
                        uint32_t im[8] = { 0 }, src[8];
                        int ox = q->ti[u][1][2], oy = q->ti[u][1][3], j;
                        int nw = MIN(8, q->ti[u][0][0]);

                        ati_r580_gl_r5_read(g->ctx, q->img[u], ox, oy, nw, 1,
                                            im, 32);
                        for (j = 0; j < 8; j++) {
                            src[j] = ati_r580_mc_read32(g->s,
                                                        g->vtex[u].addr + 4 * j);
                        }
                        fprintf(f, "  addr %08x mode %d image", g->vtex[u].addr,
                                g->vtex[u].mode);
                        for (j = 0; j < nw; j++) {
                            fprintf(f, " %08x", im[j]);
                        }
                        fprintf(f, "\n  source");
                        for (j = 0; j < 8; j++) {
                            fprintf(f, " %08x", src[j]);
                        }
                        fprintf(f, "\n");
                    }
                    {
                        g_autofree char *tn = g_strdup_printf(
                            "%s/r5diff-%02u-t%u.bin", g->s->r5g_dump,
                            g->dumps, u);
                        FILE *tf = fopen(tn, "wb");
                        int w = q->ti[u][0][0], h = q->ti[u][0][1], y, x;
                        int bpp = (q->ti[u][2][0] + q->ti[u][2][1] +
                                   q->ti[u][2][2] + q->ti[u][2][3] + 7) / 8;

                        bpp = bpp > 2 ? 4 : bpp;
                        for (y = 0; tf && y < h; y++) {
                            for (x = 0; x < w; x++) {
                                uint32_t a = g->vtex[u].addr +
                                             y * g->vtex[u].pitch + bpp * x;
                                uint32_t v = ati_r580_mc_read32(g->s, a & ~3u);

                                v >>= 8 * (a & 3);
                                fwrite(&v, bpp, 1, tf);
                            }
                        }
                        if (tf) {
                            fclose(tf);
                        }
                    }
                }
            }
            for (i = 0; i < R5G_CONSTS; i++) {
                fprintf(f, "c%u %.9g %.9g %.9g %.9g\n", i, q->consts[4 * i],
                        q->consts[4 * i + 1], q->consts[4 * i + 2],
                        q->consts[4 * i + 3]);
            }
            for (i = 0; i < q->nvert && i < 600; i++) {
                unsigned k;

                fprintf(f, "v%u", i);
                for (k = 0; k < fl; k++) {
                    fprintf(f, " %.9g", q->verts[i * fl + k]);
                }
                fprintf(f, "\n");
            }
            fprintf(f, "%s", q->fs);
            fclose(f);
        }
    }
    g->dumps++;
}

/* the CPU path has drawn the same draw into VRAM: compare, then resync */
void ati_r580_r5gpu_verify_end(ATIR580State *s)
{
    ATIR580R5Gpu *g = s->r5g;
    R5GSurf *sf = g ? g->vsf : NULL;
    const uint8_t *vram;
    uint64_t ndiff = 0, n1 = 0, mx = 0;
    int x, y, w, fx = -1, fy = -1, b;
    uint32_t fg = 0, fc = 0;

    if (!sf) {
        return;
    }
    g->vsf = NULL;
    vram = r5g_vram(g) + sf->off;
    w = g->vx1 - g->vx0;
    for (y = g->vy0; y < g->vy1; y++) {
        const uint8_t *c = vram + (size_t)y * sf->pitch +
                           (size_t)g->vx0 * sf->bpp;
        const uint8_t *p = g->vbuf + (size_t)(y - g->vy0) * w * sf->bpp;

        if (!memcmp(c, p, (size_t)w * sf->bpp)) {
            continue;
        }
        for (x = 0; x < w; x++) {
            int md = 0;

            for (b = 0; b < (int)sf->bpp; b++) {
                int dd = abs((int)c[x * sf->bpp + b] - (int)p[x * sf->bpp + b]);

                md = MAX(md, dd);
            }
            if (md) {
                ndiff++;
                if (md > 1) {
                    n1++;
                }
                if ((uint64_t)md > mx) {
                    mx = md;
                }
                if (fx < 0) {
                    fx = g->vx0 + x;
                    fy = y;
                    memcpy(&fg, p + x * sf->bpp, sf->bpp);
                    memcpy(&fc, c + x * sf->bpp, sf->bpp);
                }
            }
        }
    }
    g->verified++;
    if (ndiff) {
        unsigned h = g->hist_n++ % R5G_HIST;

        g->vdiff_draws++;
        g->vdiff_px += ndiff;
        g->vdiff_px1 += n1;
        g->vmax = MAX(g->vmax, mx);
        g->hist[h].ndiff = ndiff;
        g->hist[h].maxd = mx;
        g->hist[h].fx = fx;
        g->hist[h].fy = fy;
        g->hist[h].gpu = fg;
        g->hist[h].cpu = fc;
        if (s->r5g_dump && sf->bpp == 4 && g->dumps < 10 &&
            (mx > 1 || g->dumps_small++ < 3)) {
            r5g_dump(g, sf, vram);
        }
    }
    /* the image becomes VRAM again over the compared rectangle */
    r5g_surf_put(g, sf, g->vx0, g->vx1, g->vy0, g->vy1);
}

/* the CPU rasteriser is about to read and write VRAM */
void ati_r580_r5gpu_cpu_begin(ATIR580State *s, const R5xxRaster *r)
{
    ATIR580R5Gpu *g = s->r5g;
    int n;

    if (!g) {
        return;
    }
    if (!g->verify) {
        ati_r580_r5gpu_release(s);
    }
    for (n = 0; n < R5XX_MAX_COLOR_BUFS; n++) {
        uint32_t off;

        if (r->rt.cbuf_enabled[n] &&
            ati_r580_mc_to_vram(s, r->rt.cbuf[n].gpu_addr, &off)) {
            r5g_invalidate(g, off, off + (uint64_t)r->rt.cbuf[n].pitch_bytes *
                           (r->d.sc_y1 + 2), g->verify ? g->vsf : NULL);
        }
    }
    if (r->rt.zbuf_enabled) {
        uint32_t off;

        if (ati_r580_mc_to_vram(s, r->rt.zbuf.gpu_addr, &off)) {
            r5g_invalidate(g, off, off + (uint64_t)r->rt.zbuf.pitch_bytes *
                           (r->d.sc_y1 + 2), NULL);
        }
    }
    qatomic_set(&s->r5g_quiet, true);
}

void ati_r580_r5gpu_cpu_end(ATIR580State *s)
{
    qatomic_set(&s->r5g_quiet, false);
}

void ati_r580_r5gpu_release(ATIR580State *s)
{
    ATIR580R5Gpu *g = s->r5g;
    int i;

    if (!g || !qatomic_read(&s->r5g_res)) {
        return;
    }
    for (i = 0; i < R5G_NSURF; i++) {
        if (g->surf[i].live) {
            r5g_surf_flush(g, &g->surf[i]);
        }
    }
    qatomic_set(&s->r5g_res, false);
}

void ati_r580_r5gpu_sync(ATIR580State *s, uint32_t off, uint32_t len)
{
    ATIR580R5Gpu *g = s->r5g;

    if (!g || !qatomic_read(&s->r5g_res)) {
        return;
    }
    r5g_flush_range(g, off, (uint64_t)off + len, NULL);
    r5g_res_update(g);
}

void ati_r580_r5gpu_wrote(ATIR580State *s, uint32_t off, uint32_t len)
{
    ATIR580R5Gpu *g = s->r5g;

    if (!g || qatomic_read(&s->r5g_quiet)) {
        return;
    }
    r5g_flush_range(g, off, (uint64_t)off + len, NULL);
    r5g_invalidate(g, off, off + len, NULL);
    r5g_res_update(g);
}

void ati_r580_r5gpu_epoch(ATIR580State *s)
{
    if (s->r5g) {
        s->r5g->epoch++;
    }
}

void ati_r580_r5gpu_reset(ATIR580State *s)
{
    ATIR580R5Gpu *g = s->r5g;
    int i;

    if (!g) {
        return;
    }
    for (i = 0; i < R5G_NSURF; i++) {
        r5g_surf_free(g, &g->surf[i]);
    }
    for (i = 0; i < R5G_NTEX; i++) {
        r5g_tex_free(g, &g->tex[i]);
    }
    r5g_res_update(g);
}

void ati_r580_r5gpu_init(ATIR580State *s)
{
    ATIR580R5Gpu *g;

    if (!s->gl_ctx || !s->r5) {
        return;
    }
    g = g_new0(ATIR580R5Gpu, 1);
    g->s = s;
    g->ctx = s->gl_ctx;
    g->verify = s->gl_mode == R580_GL_VERIFY;
    g->dst_fetch = ati_r580_gl_r5_dst_fetch(g->ctx);
    g->epoch = 1;
    g->fs = g_hash_table_new_full(g_int64_hash, g_int64_equal, g_free, g_free);
    s->r5g = g;
}

void ati_r580_r5gpu_fini(ATIR580State *s)
{
    ATIR580R5Gpu *g = s->r5g;
    int i;

    if (!g) {
        return;
    }
    ati_r580_r5gpu_reset(s);
    for (i = 0; i < R5G_NSCRATCH; i++) {
        ati_r580_gl_r5_image_free(g->ctx, g->scratch[i].img);
    }
    g_hash_table_destroy(g->fs);
    g_free(g->verts);
    g_free(g->planes);
    g_free(g->order);
    g_free(g->pass);
    g_free(g->pbox);
    g_free(g->tbox);
    g_free(g->vbuf);
    g_free(g->vpre);
    g_free(g);
    s->r5g = NULL;
}

char *ati_r580_r5gpu_stats(ATIR580State *s)
{
    ATIR580R5Gpu *g = s->r5g;
    uint64_t progs, pfail, bdraws, lus, lmax, fbt = 0;
    GString *o;
    unsigned i;
    int ns = 0, nt = 0;

    if (!g) {
        return g_strdup("gpu path off\n");
    }
    ati_r580_gl_r5_stats(g->ctx, &progs, &pfail, &bdraws, &lus, &lmax);
    for (i = 0; i < R5G_NSURF; i++) {
        ns += g->surf[i].live;
    }
    for (i = 0; i < R5G_NTEX; i++) {
        nt += g->tex[i].live;
    }
    for (i = 0; i < R5G_FB_MAX; i++) {
        fbt += g->fb[i];
    }
    o = g_string_new(NULL);
    g_string_append_printf(o,
        "gpu path %s, %s, %s\n"
        "offered %" PRIu64 " gpu %" PRIu64 " empty %" PRIu64
        " cpu %" PRIu64 "\n"
        "gpu draw ms total %.1f max %.2f, passes %" PRIu64 ", verts %" PRIu64
        "\nprograms linked %" PRIu64 " failed %" PRIu64 ", link ms total %.1f"
        " max %.1f\n"
        "surfaces %d (new %" PRIu64 ", evicted %" PRIu64 "), textures %d "
        "(new %" PRIu64 ", hits %" PRIu64 "), aliased %" PRIu64
        ", feedback copies %" PRIu64 "\n"
        "uploads %" PRIu64 " (%.1f MB), readbacks %" PRIu64 " (%.1f MB), "
        "flushes %" PRIu64 ", checks %" PRIu64 ", epoch %" PRIu64 "\n",
        g->verify ? "verify" : "on", ati_r580_gl_describe(g->ctx),
        g->dst_fetch ? "destination fetch" : "destination copies",
        g->offered, g->drawn, g->empty, fbt,
        g->draw_ns / 1e6, g->max_draw_ns / 1e6, g->passes, g->verts_n,
        progs, pfail, lus / 1e3, lmax / 1e3, ns, g->surf_new, g->evicts, nt, g->tex_new,
        g->tex_hits, g->aliased, g->feedback,
        g->up_calls, g->up_bytes / 1e6, g->rd_calls, g->rd_bytes / 1e6,
        g->flushes, g->checks, g->epoch);
    g_string_append(o, "cpu fallbacks:");
    for (i = 0; i < R5G_FB_MAX; i++) {
        if (g->fb[i]) {
            g_string_append_printf(o, " %s %" PRIu64 ";", r5g_fb_names[i],
                                   g->fb[i]);
        }
    }
    g_string_append(o, "\n");
    if (g->verify) {
        g_string_append_printf(o,
            "verify: draws %" PRIu64 ", differing %" PRIu64 ", pixels %"
            PRIu64 " (by more than 1: %" PRIu64 "), max delta %" PRIu64 "\n",
            g->verified, g->vdiff_draws, g->vdiff_px, g->vdiff_px1, g->vmax);
        for (i = 0; i < R5G_HIST && i < g->hist_n; i++) {
            unsigned h = (g->hist_n - 1 - i) % R5G_HIST;

            g_string_append_printf(o,
                "  dst %08x cp %08x blend %08x out %08x tx0f1 %08x "
                "box %d,%d-%d,%d diff %u max %u first %d,%d gpu %08x "
                "cpu %08x\n", g->hist[h].dst, g->hist[h].cpitch,
                g->hist[h].blend, g->hist[h].outfmt, g->hist[h].tx0,
                g->hist[h].x0, g->hist[h].y0, g->hist[h].x1, g->hist[h].y1,
                g->hist[h].ndiff, g->hist[h].maxd, g->hist[h].fx,
                g->hist[h].fy, g->hist[h].gpu, g->hist[h].cpu);
        }
    }
    return g_string_free(o, false);
}
