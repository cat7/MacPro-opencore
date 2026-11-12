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
 * than VRAM. A depth/stencil buffer is a surface too, its raw words
 * tested and updated by the fragment stage; such a draw is exact only
 * with the CPU rasteriser's own coverage, so its triangles are drawn as
 * their bounding boxes and the shader applies the CPU's edge functions.
 * Lines and points go to the backend as the rectangles of pixels the
 * CPU rasteriser covers. TEXTURES are host copies of sampled ranges,
 * with a shadow of their source bytes. A texture lying inside a surface
 * with the same pitch and texel size samples the surface's image itself.
 *
 * COHERENCY. Every host copy is checked against VRAM before its first
 * use in each EPOCH: the rows that differ from the shadow are uploaded
 * again. A new epoch starts whenever the guest may have written memory
 * a later command depends on: a new ring write pointer, and any command
 * submitted outside the command processor thread. Writers inside the
 * device (2D engine, CP write-backs, the CPU rasteriser) invalidate the
 * copies they overlap at once. The drawn rectangles go back to VRAM
 * before any reader inside the device touches them (ati_r580_gl_sync()),
 * before a writer overwrites them (ati_r580_gl_wrote()), and at the
 * points ati_r580_gl_release() names, as RESIDENCY says.
 *
 * RESIDENCY. At a release after which the guest CPU may run (a fence,
 * the end of a ring or indirect buffer, a FIFO push, the display
 * refresh) a drawn rectangle goes back only if it meets the scanned out
 * framebuffer or its surface has no COLOUR GUARD in force. A guard is a
 * window over the aperture laid on the surface's pages when it is drawn
 * (ati_r580_cguard_set()); it is in force once every vCPU has run the
 * work queued behind its own TLB flush, since until then a vCPU may
 * still reach those pages as RAM. Every guest CPU load or store under a
 * window traps, writes the surfaces under it back and lifts it before
 * the access is done on VRAM (ati_r580_r5gpu_guard()). Other releases
 * (a CPU-rasterised draw, yield, reset, migration) write everything
 * back. A surface whose window the guest keeps touching is not guarded
 * again.
 *
 * A surface replaced by another layout over the same memory (a new
 * pitch, more rows) hands its drawn pixels to the new image on the GPU,
 * placed at the bytes they occupy, instead of going back to VRAM.
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
#include "exec/target_page.h"
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
#define R5G_FS_VERSION  6
/* vec4 of records a request may hold: R5_REC travels as a float */
#define R5G_MAX_RECS    (1u << 24)

/* the shading the CPU path's compiler fused: v * m + 0.5, lerps */
#if defined(__aarch64__)
#define R5G_FMA_DEF "#define R5_FMA(a, b, c) fma(a, b, c)\n"
#else
#define R5G_FMA_DEF "#define R5_FMA(a, b, c) ((a) * (b) + (c))\n"
#endif

typedef enum {
    R5G_FB_PRIM,        /* triangles, lines and points mixed */
    R5G_FB_CBUF,        /* several colour buffers, multiwrite */
    R5G_FB_CFMT,        /* colour buffer format */
    R5G_FB_CMEM,        /* colour buffer outside VRAM, or beyond its pitch */
    R5G_FB_ZS,          /* depth/stencil buffer memory or layout */
    R5G_FB_ZOFF,        /* depth offset, polygon offset */
    R5G_FB_ZBIG,        /* depth-tested triangle far outside the buffer */
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
    R5G_FB_RECS,        /* records past R5G_MAX_RECS */
    R5G_FB_SURF,        /* no host image for the colour buffer */
    R5G_FB_BACKEND,     /* the backend declined */
    R5G_FB_PENDING,     /* the program is still being compiled */
    R5G_FB_LINK,        /* the program would not build */
    R5G_FB_MAX
} R5GFallback;

static const char *const r5g_fb_names[R5G_FB_MAX] = {
    [R5G_FB_PRIM] = "mixed primitives",
    [R5G_FB_CBUF] = "multiple colour buffers",
    [R5G_FB_CFMT] = "colour format",
    [R5G_FB_CMEM] = "colour buffer memory",
    [R5G_FB_ZS] = "depth/stencil buffer",
    [R5G_FB_ZOFF] = "depth/polygon offset",
    [R5G_FB_ZBIG] = "depth-tested triangle out of range",
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
    [R5G_FB_RECS] = "too many records",
    [R5G_FB_SURF] = "no host surface",
    [R5G_FB_BACKEND] = "backend declined",
    [R5G_FB_PENDING] = "program compiling",
    [R5G_FB_LINK] = "program build failed",
};

/* who a readback was for: the release reasons, then these */
enum {
    R5G_RB_WROTE = R580_GLR_MAX,    /* a writer over drawn bytes */
    R5G_RB_CPU,                     /* the CPU rasteriser */
    R5G_RB_FREE,                    /* eviction, overlap, growth */
    R5G_RB_GUARD,                   /* the guest CPU, through a guard */
    R5G_RB_MAX
};

static const char *const r5g_rb_names[R5G_RB_MAX] = {
    [R580_GLR_SCANOUT] = "scanout", [R580_GLR_RING] = "ring",
    [R580_GLR_IB] = "ib", [R580_GLR_FIFO] = "fifo",
    [R580_GLR_READ] = "read", [R580_GLR_2D] = "2d",
    [R580_GLR_TARGET] = "target", [R580_GLR_FALLBACK] = "fallback",
    [R580_GLR_BACKEND] = "backend", [R580_GLR_RESET] = "reset",
    [R580_GLR_FENCE] = "fence", [R580_GLR_YIELD] = "yield",
    [R5G_RB_WROTE] = "wrote", [R5G_RB_CPU] = "cpu draw",
    [R5G_RB_FREE] = "free", [R5G_RB_GUARD] = "guest cpu",
};

/* traps after which a surface is not guarded again */
#define R5G_GUARD_TRAPS 4

#define R5G_RBSURF 8

typedef struct R5GSurf {
    bool live;
    uint32_t off, pitch, bpp;
    int w, h;
    R5GpuImage *img;
    uint8_t *shadow;
    int dx0, dy0, dx1, dy1;     /* newer on the host; end exclusive */
    uint64_t epoch, lru;
    int guard;                  /* colour guard window + 1; 0: none */
    unsigned traps;
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

/*
 * A triangle of a depth-tested draw: its z plane, and the CPU
 * rasteriser's edge functions, edge i covering subpixel c when
 * a[i] * c.x + b[i] * c.y + c[i] >= 0 (the fill rule's bias in c).
 */
typedef struct R5GTriZ {
    R5xxZPlane zp;
    bool front;
    int32_t a[3], b[3];
    int64_t c[3];
} R5GTriZ;

/* vec4 of a record's depth part */
#define R5G_ZREC        8

/* subpixel coordinates the exact edge test takes */
#define R5G_EDGE_MAX    (1 << 29)

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
    struct R5GTriZ *tz;         /* per triangle, with a depth buffer */
    float *verts2;
    size_t verts2_cap;
    size_t tris_cap;

    /* verify: the GPU's pixels of the last draw */
    R5GSurf *vsf, *vzsf;
    int vx0, vy0, vx1, vy1;
    uint8_t *vbuf, *vzbuf;
    size_t vbuf_cap, vzbuf_cap;

    /* statistics */
    uint64_t offered, drawn, empty, verified, vdiff_draws, vdiff_px;
    uint64_t vdiff_px1, vmax;
    uint64_t zdrawn, ldrawn, pdrawn, zverified, zdiff_draws, zdiff_px;
    uint64_t zcleared, zown;
    struct {
        int x, y;
        uint32_t gpu, cpu;
    } zhist[R5G_HIST];
    unsigned zhist_n;
    uint64_t fb[R5G_FB_MAX];
    uint64_t up_bytes, up_calls, rd_bytes, rd_calls, flushes, checks;
    uint64_t aliased, feedback, tex_hits, tex_new, surf_new, evicts;
    uint64_t passes, draw_ns, max_draw_ns, verts_n;
    int rb_why;
    uint64_t rb_n[R5G_RB_MAX], rb_bytes[R5G_RB_MAX], rb_ns[R5G_RB_MAX];
    uint64_t rb_read_ns;
    int gsurf[ATI_R580_CGUARDS];    /* surface + 1 under window k; 0: none */
    uint64_t g_arms, g_kept, g_traps, g_diff, g_noslot, g_off, adopted;
    struct {
        uint32_t off, pitch;
        uint64_t n, bytes;
    } rbs[R5G_RBSURF];
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
static void r5g_guard_arm(ATIR580R5Gpu *g, R5GSurf *sf);
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
    int why = g->rb_why;

    if (!sf->live) {
        return;
    }
    g->rb_why = R5G_RB_FREE;
    r5g_surf_flush(g, sf);
    g->rb_why = why;
    if (sf->guard) {
        /* the window stays until a trap or a new owner lifts it */
        g->gsurf[sf->guard - 1] = 0;
    }
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
    int64_t t0, t1;
    bool ok;

    if (w <= 0 || h <= 0) {
        return;
    }
    t0 = get_clock();
    lo = sf->off + (uint64_t)sf->dy0 * sf->pitch + (uint64_t)sf->dx0 * sf->bpp;
    if (g->rb_why == R5G_RB_GUARD) {
        /* what the guest would have read without the guard */
        size_t row = (size_t)w * sf->bpp, i;
        g_autofree uint8_t *buf = g_malloc(row * h);

        ok = ati_r580_gl_r5_read(g->ctx, sf->img, sf->dx0, sf->dy0, w, h,
                                 buf, row);
        for (y = 0; ok && y < h; y++) {
            uint8_t *v = vram + lo + (size_t)y * sf->pitch;

            for (i = 0; i < row; i++) {
                g->g_diff += v[i] != buf[row * y + i];
            }
            memcpy(v, buf + row * y, row);
        }
    } else {
        ok = ati_r580_gl_r5_read(g->ctx, sf->img, sf->dx0, sf->dy0, w, h,
                                 vram + lo, sf->pitch);
    }
    t1 = get_clock();
    g->rb_read_ns += t1 - t0;
    if (ok) {
        unsigned k;

        for (k = 0; k < R5G_RBSURF - 1; k++) {
            if (!g->rbs[k].n || (g->rbs[k].off == sf->off &&
                                 g->rbs[k].pitch == sf->pitch)) {
                break;
            }
        }
        g->rbs[k].off = sf->off;
        g->rbs[k].pitch = sf->pitch;
        g->rbs[k].n++;
        g->rbs[k].bytes += (uint64_t)w * h * sf->bpp;
        g->rb_n[g->rb_why]++;
        g->rb_bytes[g->rb_why] += (uint64_t)w * h * sf->bpp;
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
    g->rb_ns[g->rb_why] += get_clock() - t0;
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

/*
 * The drawn pixels of `o` into `n`, at the bytes they occupy under n's
 * layout, on the GPU; n's drawn rectangle grows to cover them. False
 * when they do not all lie in n, and o must go back to VRAM instead.
 */
static bool r5g_surf_adopt(ATIR580R5Gpu *g, R5GSurf *n, const R5GSurf *o)
{
    int w = o->dx1 - o->dx0, y, x0 = INT_MAX, y0 = INT_MAX, x1 = 0, y1 = 0;
    int64_t end = (int64_t)n->pitch * n->h;
    size_t nr = 0, cap;
    g_autofree int *runs = NULL;

    if (w <= 0 || o->dy1 <= o->dy0) {
        return true;
    }
    if (o->bpp != n->bpp) {
        return false;
    }
    cap = (size_t)(o->dy1 - o->dy0) *
          ((size_t)w * o->bpp / n->pitch + 2);
    runs = g_new(int, cap * 6);
    for (y = o->dy0; y < o->dy1; y++) {
        int64_t r = (int64_t)o->off + (int64_t)y * o->pitch +
                    (int64_t)o->dx0 * o->bpp - n->off;
        int sx = o->dx0, left = w;

        if (r < 0 || r + (int64_t)w * o->bpp > end || r % n->bpp) {
            return false;
        }
        while (left > 0) {
            int ny = r / n->pitch, nx = (r % n->pitch) / n->bpp;
            int c = MIN(left, n->w - nx);
            int *q = runs + nr * 6;

            if (nr && q[-6] == sx && q[-2] == c && q[-4] == nx &&
                q[-5] + q[-1] == y && q[-3] + q[-1] == ny) {
                q[-1]++;
            } else {
                if (nr == cap) {
                    return false;
                }
                q[0] = sx;
                q[1] = y;
                q[2] = nx;
                q[3] = ny;
                q[4] = c;
                q[5] = 1;
                nr++;
            }
            x0 = MIN(x0, nx);
            y0 = MIN(y0, ny);
            x1 = MAX(x1, nx + c);
            y1 = MAX(y1, ny + 1);
            sx += c;
            left -= c;
            r += (int64_t)c * n->bpp;
        }
    }
    if (!ati_r580_gl_r5_copy_runs(g->ctx, n->img, o->img, runs, nr)) {
        return false;
    }
    if (n->dx1 > n->dx0) {
        x0 = MIN(x0, n->dx0);
        y0 = MIN(y0, n->dy0);
        x1 = MAX(x1, n->dx1);
        y1 = MAX(y1, n->dy1);
    }
    n->dx0 = x0;
    n->dy0 = y0;
    n->dx1 = x1;
    n->dy1 = y1;
    g->adopted++;
    return true;
}

static R5GSurf *r5g_surf_new(ATIR580R5Gpu *g, uint32_t off, uint32_t pitch,
                             uint32_t bpp, int h)
{
    R5GSurf n = { 0 }, *sf;
    int w = pitch / bpp;
    int i;

    n.img = ati_r580_gl_r5_image(g->ctx, w, h, 1, bpp);
    if (!n.img) {
        return NULL;
    }
    n.off = off;
    n.pitch = pitch;
    n.bpp = bpp;
    n.w = w;
    n.h = h;
    n.shadow = g_malloc((size_t)pitch * h);
    r5g_surf_put(g, &n, 0, w, 0, h);
    n.epoch = g->epoch;
    /*
     * Anything else over this memory goes away: its drawn pixels move
     * into the new image on the GPU, or back to VRAM first.
     */
    for (i = 0; i < R5G_NSURF; i++) {
        R5GSurf *o = &g->surf[i];

        if (!o->live || off >= o->off + (uint64_t)o->pitch * o->h ||
            o->off >= off + (uint64_t)pitch * h) {
            continue;
        }
        if (r5g_surf_adopt(g, &n, o)) {
            o->dx0 = o->dy0 = o->dx1 = o->dy1 = 0;
            n.traps = MAX(n.traps, o->traps);
            if (o->guard && !n.guard) {
                /* its window watches the same pages */
                n.guard = o->guard;
                o->guard = 0;
            }
        } else {
            /* written back after the upload: the next check takes it */
            n.epoch = 0;
        }
        r5g_surf_free(g, o);
    }
    sf = r5g_surf_slot(g);
    *sf = n;
    sf->live = true;
    sf->lru = ++g->tick;
    if (sf->guard) {
        g->gsurf[sf->guard - 1] = sf - g->surf + 1;
    }
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
        /* grown from its own start: the old one is adopted */
        off = sf->off;
        rows += *oy;
        h = MAX(rows, sf->h);
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
                          int planes, int zbpp, const R5GpuDraw *rq,
                          uint64_t *key, R5GFallback *why)
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
    k = r5g_hash(k, bound | ((uint64_t)used << 16) | ((uint64_t)zbpp << 40));
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
    g_string_append_printf(o, "#define S_NIN %u\n#define S_PERSP %d\n"
                           "#define S_ZS %d\n", ninputs, planes == 2, zbpp);
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
                       "bool us_keep;\nuint om_zs;\n");
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
        g->tz = g_renew(R5GTriZ, g->tz, g->tris_cap);
    }
    if ((size_t)tris * 3 * floats > g->verts_cap) {
        g->verts_cap = MAX((size_t)tris * 3 * floats, g->verts_cap * 2);
        g->verts = g_renew(float, g->verts, g->verts_cap);
    }
}

/* room for n more vec4 of records; false past R5G_MAX_RECS */
static bool r5g_recs_room(ATIR580R5Gpu *g, size_t have, size_t n)
{
    size_t need = (have + n) * 4;

    if (have + n > R5G_MAX_RECS) {
        return false;
    }
    if (need > g->planes_cap) {
        g->planes_cap = MAX(need, g->planes_cap * 2);
        g->planes = g_renew(float, g->planes, g->planes_cap);
    }
    return true;
}

/*
 * The depth part at the end of a record: the z plane, then the edges
 * (all zero: every pixel drawn is covered)
 */
static void r5g_zrec(float *o, const R5xxZPlane *zp, bool front,
                     const R5GTriZ *tz)
{
    int i, k;

    o[0] = zp->a;
    o[1] = zp->b;
    o[2] = zp->z0;
    o[3] = front ? 1.0f : 0.0f;
    o[4] = zp->x0;
    o[5] = zp->y0;
    o[6] = o[7] = 0.0f;
    /* 32-bit halves of a, b and c as 16-bit floats, exact */
    for (i = 0; i < 3; i++) {
        uint32_t w[4] = {
            tz ? (uint32_t)tz->a[i] : 0, tz ? (uint32_t)tz->b[i] : 0,
            tz ? (uint32_t)((uint64_t)tz->c[i] >> 32) : 0,
            tz ? (uint32_t)tz->c[i] : 0,
        };

        for (k = 0; k < 4; k++) {
            o[8 + 8 * i + 2 * k] = (int16_t)(w[k] >> 16);
            o[8 + 8 * i + 2 * k + 1] = w[k] & 0xffff;
        }
    }
}

/*
 * The records of a triangle draw, triangle t's at t * rl: its inputs as
 * plane equations through its snapped corners, as the CPU rasteriser
 * interpolates them, then its depth plane when `zs`.
 */
static void r5g_tri_recs(ATIR580R5Gpu *g, unsigned ntri, unsigned ninputs,
                         unsigned floats, unsigned rl, bool zs, bool *persp)
{
    unsigned t, n, c;

    *persp = false;
    for (t = 0; t < ntri && ninputs; t++) {
        const float *v = g->verts + (size_t)t * 3 * floats;

        if (v[3] != v[floats + 3] || v[3] != v[2 * floats + 3]) {
            *persp = true;
        }
    }
    for (t = 0; t < ntri; t++) {
        const float *v[3];
        double x0, y0, dx1, dy1, dx2, dy2, det, iw[3];
        float *o = g->planes + (size_t)t * rl * 4;
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
        if (zs) {
            r5g_zrec(o + 12 * ninputs + 4, &g->tz[t].zp, g->tz[t].front,
                     &g->tz[t]);
        }
    }
}

/*
 * Lines and points as rectangles of pixels, two triangles each, sharing
 * their primitive's record.
 */
typedef struct R5GBuild {
    ATIR580R5Gpu *g;
    unsigned ninputs, floats, rl, ntri;
    size_t nrec;
    uint32_t flat;
    bool zs, persp, fail;
    int ox, oy, sub;
    const R5xxVertex *pv;       /* provoking vertex */
    size_t rec;                 /* the primitive's record */
    bool run, xm;               /* a line's run waiting to merge */
    int maj0, maj1, n0, n1;
} R5GBuild;

/*
 * A rectangle of the primitive with record `rec`; with a depth buffer
 * the vertex w carries the index of its depth part, `zrec`.
 */
static void r5g_rect(R5GBuild *b, int x0, int y0, int x1, int y1, size_t rec,
                     size_t zrec)
{
    ATIR580R5Gpu *g = b->g;
    static const int cx[6] = { 0, 1, 0, 1, 1, 0 }, cy[6] = { 0, 0, 1, 0, 1, 1 };
    float *vo;
    int k, *tb;
    unsigned a;

    r5g_grow(g, b->ntri + 2, b->floats);
    vo = g->verts + (size_t)b->ntri * 3 * b->floats;
    for (k = 0; k < 6; k++) {
        float *o = vo + k * b->floats;

        o[0] = (cx[k] ? x1 : x0) + b->ox;
        o[1] = (cy[k] ? y1 : y0) + b->oy;
        o[2] = rec;
        o[3] = b->zs ? zrec : 1.0f;
        for (a = 0; a < b->ninputs; a++) {
            if ((b->flat >> a) & 1) {
                memcpy(o + 4 + 4 * a, b->pv->attr[a].f, 16);
            } else {
                memset(o + 4 + 4 * a, 0, 16);
            }
        }
    }
    for (k = 0; k < 2; k++) {
        tb = &g->tbox[4 * (b->ntri + k)];
        tb[0] = x0 + b->ox;
        tb[1] = y0 + b->oy;
        tb[2] = x1 + b->ox;
        tb[3] = y1 + b->oy;
    }
    b->ntri += 2;
}

/* a new record, a copy of `from` when there is one; its index */
static size_t r5g_rec_new(R5GBuild *b, const size_t *from)
{
    size_t at = b->nrec;

    if (!b->rl || !r5g_recs_room(b->g, b->nrec, b->rl)) {
        b->fail |= b->rl != 0;
        return 0;
    }
    if (from) {
        memcpy(b->g->planes + at * 4, b->g->planes + *from * 4,
               (size_t)b->rl * 16);
    } else {
        memset(b->g->planes + at * 4, 0, (size_t)b->rl * 16);
    }
    b->nrec += b->rl;
    return at;
}

static void r5g_run_flush(R5GBuild *b)
{
    if (!b->run) {
        return;
    }
    b->run = false;
    if (b->xm) {
        r5g_rect(b, b->maj0, b->n0, b->maj1, b->n1, b->rec, 0);
    } else {
        r5g_rect(b, b->n0, b->maj0, b->n1, b->maj1, b->rec, 0);
    }
}

static void r5g_line_run(void *opaque, bool xmajor, int major, int minor0,
                         int minor1, float z)
{
    R5GBuild *b = opaque;

    if (b->zs) {
        /* z changes along the line: a depth part per step */
        size_t zrec = b->nrec;
        R5xxZPlane zp = { .z0 = z };

        if (!r5g_recs_room(b->g, b->nrec, R5G_ZREC)) {
            b->fail = true;
            return;
        }
        b->nrec += R5G_ZREC;
        r5g_zrec(b->g->planes + zrec * 4, &zp, true, NULL);
        if (xmajor) {
            r5g_rect(b, major, minor0, major + 1, minor1, b->rec, zrec);
        } else {
            r5g_rect(b, minor0, major, minor1, major + 1, b->rec, zrec);
        }
        return;
    }
    if (b->run && b->xm == xmajor && b->maj1 == major && b->n0 == minor0 &&
        b->n1 == minor1) {
        b->maj1++;
        return;
    }
    r5g_run_flush(b);
    b->run = true;
    b->xm = xmajor;
    b->maj0 = major;
    b->maj1 = major + 1;
    b->n0 = minor0;
    b->n1 = minor1;
}

/* the record of a line: its inputs along the major axis, as the CPU interpolates */
static void r5g_line_rec(R5GBuild *b, const R5xxVertex *v0, const R5xxVertex *v1,
                         float *o)
{
    float sub = b->sub;
    float x0 = (float)llrintf(v0->x * sub) / sub;
    float y0 = (float)llrintf(v0->y * sub) / sub;
    float x1 = (float)llrintf(v1->x * sub) / sub;
    float y1 = (float)llrintf(v1->y * sub) / sub;
    float dx = x1 - x0, dy = y1 - y0;
    bool xm = fabsf(dx) >= fabsf(dy);
    double d = xm ? dx : dy, iw[2];
    double m0 = xm ? (double)x0 + b->ox : (double)y0 + b->oy;
    unsigned n, c, k;

    for (k = 0; k < 2; k++) {
        float w = (k ? v1 : v0)->w;

        iw[k] = b->persp ? 1.0 / (w > 1e-30f ? w : 1e-30f) : 1.0;
    }
    for (n = 0; n <= b->ninputs; n++) {
        for (c = 0; c < 4; c++) {
            double a0, a1, A;

            if (n < b->ninputs) {
                a0 = iw[0] * v0->attr[n].f[c];
                a1 = iw[1] * v1->attr[n].f[c];
            } else if (!c) {
                a0 = iw[0];
                a1 = iw[1];
            } else {
                continue;
            }
            A = d ? (a1 - a0) / d : 0;
            if (n < b->ninputs) {
                o[12 * n + c] = xm ? A : 0;
                o[12 * n + 4 + c] = xm ? 0 : A;
                o[12 * n + 8 + c] = a0 - A * m0;
            } else {
                o[12 * n] = xm ? A : 0;
                o[12 * n + 1] = xm ? 0 : A;
                o[12 * n + 2] = a0 - A * m0;
                o[12 * n + 3] = 0;
            }
        }
    }
}

/* the record of a point: its inputs constant */
static void r5g_point_rec(R5GBuild *b, const R5xxVertex *v, float z, float *o)
{
    unsigned n;

    memset(o, 0, (size_t)b->rl * 16);
    for (n = 0; n < b->ninputs; n++) {
        memcpy(o + 12 * n + 8, v->attr[n].f, 16);
    }
    o[12 * b->ninputs + 2] = 1.0f;
    if (b->zs) {
        R5xxZPlane zp = { .z0 = z };

        r5g_zrec(o + 12 * b->ninputs + 4, &zp, true, NULL);
    }
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
    R5GSurf *sf, *zsf = NULL;
    R5GFallback why = R5G_FB_PROG;
    uint32_t coff, cpitch, cbpp, hw, cmask, flat = 0, bound = 0;
    uint32_t zoff = 0, zpitch = 0, zbpp = 0;
    unsigned ninputs = MIN(r->num_fs_inputs, R5G_INPUTS), floats, ntri = 0;
    unsigned i, k, c, npass = 1, rl;
    int ready, cf, ox, oy, sx0, sy0, sx1, sy1, bx0, by0, bx1, by1, kind;
    int64_t t0 = get_clock();
    bool need_dst, have_flat = false, persp = false, zs;
    const float sub = d->subpix;
    static const uint8_t pos_bgra[4] = { 2, 1, 0, 3 };
    size_t nrec = 0;

    if (!g) {
        return R5G_CPU;
    }
    g->offered++;
    zs = r->rt.zbuf_enabled &&
         (d->z_enable || (d->stencil_enable && r->rt.has_stencil));
    if (!r->rt.num_cbufs || !r->rt.cbuf_enabled[0]) {
        if (zs) {
            return r5g_fail(g, R5G_FB_CBUF);
        }
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
    if (zs) {
        zpitch = r->rt.zbuf.pitch_bytes;
        zbpp = r->rt.zbuf.bpp;
        if (d->depth_offset != 0.0f) {
            return r5g_fail(g, R5G_FB_ZOFF);
        }
        if (!ati_r580_mc_to_vram(s, r->rt.zbuf.gpu_addr, &zoff) ||
            (zbpp != 2 && zbpp != 4) || zpitch % zbpp ||
            (uint32_t)d->sc_x1 >= zpitch / zbpp ||
            (uint64_t)zoff + (uint64_t)zpitch * (d->sc_y1 + 1) >
            ATI_R580_VRAM_SIZE) {
            return r5g_fail(g, R5G_FB_ZS);
        }
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
    if ((!cmask && !zs) || d->sc_x1 < d->sc_x0 || d->sc_y1 < d->sc_y0) {
        g->empty++;
        return R5G_DONE;
    }
    kind = in->nprims ? in->prims[0].kind : R5_PRIM_TRI;
    kind = kind == R5_PRIM_POINT_SIZED ? R5_PRIM_POINT : kind;
    for (i = 0; i < in->nprims; i++) {
        int pk = in->prims[i].kind;

        if ((pk == R5_PRIM_POINT_SIZED ? R5_PRIM_POINT : pk) != kind) {
            return r5g_fail(g, R5G_FB_PRIM);
        }
        if (i && in->prims[i].flat_inputs != in->prims[0].flat_inputs) {
            return r5g_fail(g, R5G_FB_FLAT);
        }
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
    /* the depth buffer's copy takes a unit in the OpenGL backend */
    if (ctpop32(bound) > R5G_MAX_BOUND - (zs ? 1 : 0)) {
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
    if (zs) {
        R5GpuImage *cimg = sf->img;
        int zox, zoy;

        zsf = r5g_target(g, zoff, zpitch, zbpp, d->sc_y1 + 1, &zox, &zoy);
        if (zsf && zsf != sf && (zox != ox || zoy != oy) && !ox && !oy) {
            /* it lies inside a larger surface: give it one of its own */
            int h = ROUND_UP(d->sc_y1 + 1, 64);

            if ((uint64_t)zoff + (uint64_t)zpitch * h > ATI_R580_VRAM_SIZE) {
                h = (ATI_R580_VRAM_SIZE - zoff) / zpitch;
            }
            zsf = r5g_surf_new(g, zoff, zpitch, zbpp, h);
            zox = zoy = 0;
            g->zown++;
        }
        if (!zsf || zsf == sf || !sf->live || sf->img != cimg ||
            zox != ox || zoy != oy) {
            g_free(rq);
            return r5g_fail(g, R5G_FB_ZS);
        }
        zsf->lru = ++g->tick;
    }
    floats = 4 + 4 * ninputs;
    rl = ninputs || zs ? 3 * ninputs + 1 + (zs ? R5G_ZREC : 0) : 0;
    r5g_grow(g, in->nprims + 1, floats);
    sx0 = d->sc_x0 + ox;
    sy0 = d->sc_y0 + oy;
    sx1 = d->sc_x1 + ox + 1;
    sy1 = d->sc_y1 + oy + 1;
    bx0 = by0 = INT_MAX;
    bx1 = by1 = INT_MIN;
    if (in->nprims) {
        flat = in->prims[0].flat_inputs & ((1u << ninputs) - 1);
    }
    for (i = 0; i < in->nprims && kind == R5_PRIM_TRI; i++) {
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
        if (zs && (front ? d->po_front : d->po_back) &&
            r5xx_raster_poly_offset(r, &p->v[0], &p->v[1], &p->v[2],
                front ? d->po_front_scale : d->po_back_scale,
                front ? d->po_front_offset : d->po_back_offset) != 0) {
            g_free(rq);
            return r5g_fail(g, R5G_FB_ZOFF);
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
            o[2] = ntri * rl;
            o[3] = v->w;
            for (a = 0; a < ninputs; a++) {
                const R5xxVertex *src = ((flat >> a) & 1) ? &p->v[prov] : v;

                memcpy(o + 4 + 4 * a, src->attr[a].f, 16);
            }
        }
        if (zs) {
            const R5xxVertex *pv[3] = { &p->v[0], &p->v[1], &p->v[2] };
            R5GTriZ *tz = &g->tz[ntri];
            int64_t sg = sa < 0 ? -1 : 1;

            for (k = 0; k < 3; k++) {
                int a = (k + 1) % 3, b = (k + 2) % 3;
                int64_t ex = (X[b] - X[a]) * sg, ey = (Y[b] - Y[a]) * sg;

                if (X[k] <= -R5G_EDGE_MAX || X[k] >= R5G_EDGE_MAX ||
                    Y[k] <= -R5G_EDGE_MAX || Y[k] >= R5G_EDGE_MAX) {
                    g_free(rq);
                    return r5g_fail(g, R5G_FB_ZBIG);
                }
                tz->a[k] = -ey;
                tz->b[k] = ex;
                tz->c[k] = ey * X[a] - ex * Y[a] +
                           (((ey == 0 && ex > 0) || ey < 0) ? 0 : -1);
            }
            r5xx_raster_zplane(X, Y, pv, &tz->zp);
            tz->front = front;
        }
        have_flat |= flat != 0;
        ntri++;
    }
    if (kind == R5_PRIM_TRI && ntri && rl) {
        if (!r5g_recs_room(g, 0, (size_t)ntri * rl)) {
            g_free(rq);
            return r5g_fail(g, R5G_FB_RECS);
        }
        r5g_tri_recs(g, ntri, ninputs, floats, rl, zs, &persp);
        nrec = (size_t)ntri * rl;
    }
    if (kind == R5_PRIM_TRI && ntri && zs) {
        /*
         * The host rasteriser's coverage is not the CPU's: each triangle
         * is drawn as its bounding box, the shader keeping the pixels
         * the CPU's edge functions cover.
         */
        size_t need = (size_t)ntri * 6 * floats;
        static const int cx[6] = { 0, 1, 0, 1, 1, 0 };
        static const int cy[6] = { 0, 0, 1, 0, 1, 1 };

        if (need > g->verts2_cap) {
            g->verts2_cap = MAX(need, g->verts2_cap * 2);
            g->verts2 = g_renew(float, g->verts2, g->verts2_cap);
        }
        for (i = 0; i < ntri; i++) {
            const float *src = g->verts + (size_t)i * 3 * floats;
            const int *tb = &g->tbox[4 * i];

            for (k = 0; k < 6; k++) {
                float *o = g->verts2 + ((size_t)i * 6 + k) * floats;

                memcpy(o, src, floats * sizeof(float));
                o[0] = cx[k] ? tb[2] : tb[0];
                o[1] = cy[k] ? tb[3] : tb[1];
                o[3] = (size_t)i * rl + 3 * ninputs + 1;
            }
        }
        r5g_grow(g, (size_t)ntri * 2 + 1, floats);
        memcpy(g->verts, g->verts2, need * sizeof(float));
        for (i = ntri; i-- > 0;) {
            memcpy(&g->tbox[4 * (2 * i + 1)], &g->tbox[4 * i], 4 * sizeof(int));
            memcpy(&g->tbox[4 * (2 * i)], &g->tbox[4 * i], 4 * sizeof(int));
        }
        ntri *= 2;
    }
    if (kind != R5_PRIM_TRI) {
        R5GBuild b = {
            .g = g, .ninputs = ninputs, .floats = floats, .rl = rl,
            .flat = flat, .zs = zs, .ox = ox, .oy = oy, .sub = d->subpix,
        };
        int prov = d->provoking == 3 || d->provoking > 1 ? 1 : d->provoking;

        for (i = 0; i < in->nprims && kind == R5_PRIM_LINE; i++) {
            const R5Prim *p = &in->prims[i];

            if (!(p->v[0].w > 0.0f) || !isfinite(p->v[0].w) ||
                !(p->v[1].w > 0.0f) || !isfinite(p->v[1].w)) {
                g_free(rq);
                return r5g_fail(g, R5G_FB_W);
            }
            b.persp |= ninputs && p->v[0].w != p->v[1].w;
        }
        for (i = 0; i < in->nprims && !b.fail; i++) {
            const R5Prim *p = &in->prims[i];

            b.pv = &p->v[kind == R5_PRIM_LINE ? prov : 0];
            b.rec = r5g_rec_new(&b, NULL);
            if (b.fail) {
                break;
            }
            if (kind == R5_PRIM_LINE) {
                if (b.rl) {
                    r5g_line_rec(&b, &p->v[0], &p->v[1],
                                 g->planes + b.rec * 4);
                }
                r5xx_raster_line_runs(r, &p->v[0], &p->v[1], r5g_line_run,
                                      &b);
                r5g_run_flush(&b);
            } else {
                int box[4];
                float z;

                if (!(p->v[0].w > 0.0f) || !isfinite(p->v[0].w)) {
                    g_free(rq);
                    return r5g_fail(g, R5G_FB_W);
                }
                if (!r5xx_raster_point_box(r, &p->v[0],
                                           p->kind == R5_PRIM_POINT_SIZED,
                                           p->size, box, &z)) {
                    continue;
                }
                if (b.rl) {
                    r5g_point_rec(&b, &p->v[0], z, g->planes + b.rec * 4);
                }
                r5g_rect(&b, box[0], box[1], box[2], box[3], b.rec,
                         b.rec + 3 * ninputs + 1);
            }
        }
        if (b.fail) {
            g_free(rq);
            return r5g_fail(g, R5G_FB_RECS);
        }
        ntri = b.ntri;
        nrec = b.nrec;
        persp = b.persp;
        have_flat = flat != 0;
        for (i = 0; i < ntri; i++) {
            const int *tb = &g->tbox[4 * i];

            bx0 = MIN(bx0, tb[0]);
            by0 = MIN(by0, tb[1]);
            bx1 = MAX(bx1, tb[2]);
            by1 = MAX(by1, tb[3]);
        }
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
    if (zsf && bx1 > zsf->w) {
        g_free(rq);
        return r5g_fail(g, R5G_FB_ZS);
    }
    sx1 = MIN(sx1, sf->w);
    if (zsf) {
        sx1 = MIN(sx1, zsf->w);
    }

    /* a pixel that fails the depth test, or is discarded, keeps its colour */
    need_dst = d->blend_enable || cmask != 0xf || zs;
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
    rq->planes = nrec ? g->planes : NULL;
    rq->nplanes = nrec;

    /* the destination and the textures, up to date */
    r5g_flush_range(g, sf->off, sf->off + (uint64_t)sf->pitch * sf->h, sf);
    r5g_surf_check(g, sf);
    if (zsf) {
        r5g_flush_range(g, zsf->off,
                        zsf->off + (uint64_t)zsf->pitch * zsf->h, zsf);
        r5g_surf_check(g, zsf);
    }
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
        g->vtex[i].mode = as ? (as == sf || as == zsf ? 2 : 1) : 0;
        if (as) {
            r5g_flush_range(g, as->off, as->off + (uint64_t)as->pitch * as->h, as);
            r5g_surf_check(g, as);
            as->lru = ++g->tick;
            g->aliased++;
            if (as == sf || as == zsf) {
                R5GpuImage *sc = r5g_scratch(g, i % R5G_NSCRATCH, smp->width,
                                             smp->height, smp->bpp);

                if (!sc || !ati_r580_gl_r5_copy(g->ctx, sc, 0, 0, as->img,
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
    rq->ztarget = zsf ? zsf->img : NULL;
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
    if (zs) {
        rq->ki[10][0] = (d->z_enable ? R5G_ZF_TEST : 0) |
                        (d->z_enable && d->z_write ? R5G_ZF_WRITE : 0) |
                        (d->stencil_enable && r->rt.has_stencil ?
                         R5G_ZF_STENCIL : 0) |
                        (r->rt.z_is_16 ? R5G_ZF_16 : 0);
        rq->ki[10][1] = d->z_func;
        rq->ki[10][2] = d->subpix;
        for (k = 0; k < 2; k++) {
            rq->ki[11 + k][0] = d->s_func[k];
            rq->ki[11 + k][1] = d->s_fail[k];
            rq->ki[11 + k][2] = d->s_zfail[k];
            rq->ki[11 + k][3] = d->s_zpass[k];
            rq->ki[13][k] = d->s_ref[k] | (d->s_mask[k] << 8) |
                            (d->s_wmask[k] << 16);
        }
        rq->kf[2][0] = d->depth_scale;
    }

    rq->fs = r5g_fs(g, in->us, ninputs, flat, bound,
                    rq->planes ? 1 + persp : 0, zs ? zbpp : 0, rq,
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
        if (zsf) {
            r5g_surf_free(g, zsf);
        }
        r5g_res_update(g);
        return r5g_fail(g, R5G_FB_BACKEND);
    }
    if (g->verify) {
        g->vreq = *rq;
    }
    g_free(rq);
    g->drawn++;
    g->zdrawn += zs;
    g->ldrawn += kind == R5_PRIM_LINE;
    g->pdrawn += kind == R5_PRIM_POINT;
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
        g->vzsf = NULL;
        if (zsf && g->vsf) {
            n = (size_t)(bx1 - bx0) * (by1 - by0) * zsf->bpp;
            if (n > g->vzbuf_cap) {
                g->vzbuf_cap = n;
                g->vzbuf = g_realloc(g->vzbuf, n);
            }
            if (ati_r580_gl_r5_read(g->ctx, zsf->img, bx0, by0, bx1 - bx0,
                                    by1 - by0, g->vzbuf,
                                    (bx1 - bx0) * zsf->bpp)) {
                g->vzsf = zsf;
            }
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
    /* laid now, so that it is in force by the next release */
    r5g_guard_arm(g, sf);
    /* what was decoded from these bytes is stale now */
    r5g_invalidate(g, sf->off + (uint64_t)by0 * sf->pitch,
                   sf->off + (uint64_t)by1 * sf->pitch, sf);
    if (zsf) {
        zsf->dx0 = zsf->dx1 > zsf->dx0 ? MIN(zsf->dx0, bx0) : bx0;
        zsf->dy0 = zsf->dy1 > zsf->dy0 ? MIN(zsf->dy0, by0) : by0;
        zsf->dx1 = MAX(zsf->dx1, bx1);
        zsf->dy1 = MAX(zsf->dy1, by1);
        r5g_guard_arm(g, zsf);
        r5g_invalidate(g, zsf->off + (uint64_t)by0 * zsf->pitch,
                       zsf->off + (uint64_t)by1 * zsf->pitch, zsf);
    }
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
                        int tw = q->ti[u][0][0], th = q->ti[u][0][1], ty, tx;
                        int bpp = (q->ti[u][2][0] + q->ti[u][2][1] +
                                   q->ti[u][2][2] + q->ti[u][2][3] + 7) / 8;

                        bpp = bpp > 2 ? 4 : bpp;
                        for (ty = 0; tf && ty < th; ty++) {
                            for (tx = 0; tx < tw; tx++) {
                                uint32_t a = g->vtex[u].addr +
                                             ty * g->vtex[u].pitch + bpp * tx;
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
                unsigned c;

                fprintf(f, "v%u", i);
                for (c = 0; c < fl; c++) {
                    fprintf(f, " %.9g", q->verts[i * fl + c]);
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
    if (g->vzsf) {
        R5GSurf *zf = g->vzsf;
        size_t row = (size_t)w * zf->bpp;

        g->vzsf = NULL;
        ndiff = 0;
        for (y = g->vy0; y < g->vy1; y++) {
            const uint8_t *c = r5g_vram(g) + zf->off + (size_t)y * zf->pitch +
                               (size_t)g->vx0 * zf->bpp;
            const uint8_t *p = g->vzbuf + (size_t)(y - g->vy0) * row;

            if (!memcmp(c, p, row)) {
                continue;
            }
            for (x = 0; x < w; x++) {
                if (memcmp(c + x * zf->bpp, p + x * zf->bpp, zf->bpp)) {
                    if (!ndiff++) {
                        unsigned h = g->zhist_n++ % R5G_HIST;

                        g->zhist[h].x = g->vx0 + x;
                        g->zhist[h].y = y;
                        g->zhist[h].gpu = 0;
                        g->zhist[h].cpu = 0;
                        memcpy(&g->zhist[h].gpu, p + x * zf->bpp, zf->bpp);
                        memcpy(&g->zhist[h].cpu, c + x * zf->bpp, zf->bpp);
                    }
                }
            }
        }
        g->zverified++;
        g->zdiff_draws += ndiff != 0;
        g->zdiff_px += ndiff;
        r5g_surf_put(g, zf, g->vx0, g->vx1, g->vy0, g->vy1);
    }
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
        ati_r580_r5gpu_release(s, R5G_RB_CPU);
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

/* a release after which the guest CPU may run */
static bool r5g_rel_guest(int why)
{
    return why == R580_GLR_SCANOUT || why == R580_GLR_RING ||
           why == R580_GLR_IB || why == R580_GLR_FIFO ||
           why == R580_GLR_FENCE;
}

/* the drawn bytes, as a VRAM range */
static void r5g_drawn_span(const R5GSurf *sf, uint64_t *a, uint64_t *b)
{
    *a = sf->off + (uint64_t)sf->dy0 * sf->pitch + (uint64_t)sf->dx0 * sf->bpp;
    *b = sf->off + (uint64_t)(sf->dy1 - 1) * sf->pitch +
         (uint64_t)sf->dx1 * sf->bpp;
}

/* the drawn bytes meet the scanned out framebuffer, now or at the last refresh */
static bool r5g_scanned(ATIR580R5Gpu *g, const R5GSurf *sf)
{
    uint64_t fb = qatomic_read(&g->s->scan_fb), a, b;
    ATIR580Mode m;

    r5g_drawn_span(sf, &a, &b);
    if ((uint32_t)fb && a < (fb >> 32) + (uint32_t)fb && (fb >> 32) < b) {
        return true;
    }
    return ati_r580_avivo_scanout(g->s, &m) &&
           a < m.fb_offset + (uint64_t)m.pitch * m.height && m.fb_offset < b;
}

/* a guard over the surface's pages, grown to them if it has a smaller one */
static void r5g_guard_arm(ATIR580R5Gpu *g, R5GSurf *sf)
{
    ATIR580State *s = g->s;
    uint64_t pg = qemu_target_page_size();
    uint64_t lo = sf->off & ~(pg - 1);
    uint64_t hi = MIN(ROUND_UP(sf->off + (uint64_t)sf->pitch * sf->h, pg),
                      (uint64_t)ATI_R580_VRAM_SIZE);
    int k, stale = -1;

    if (!s->cg_ready || g->verify || sf->traps >= R5G_GUARD_TRAPS) {
        return;
    }
    if (sf->guard) {
        k = sf->guard - 1;
        if (s->cg_lo[k] <= lo && hi <= s->cg_hi[k]) {
            return;
        }
        if (s->cg_hi[k] > s->cg_lo[k]) {
            lo = MIN(lo, s->cg_lo[k]);
            hi = MAX(hi, s->cg_hi[k]);
        }
    } else {
        for (k = 0; k < ATI_R580_CGUARDS; k++) {
            if (g->gsurf[k]) {
                continue;
            }
            if (s->cg_hi[k] <= s->cg_lo[k]) {
                break;
            }
            if (stale < 0) {
                stale = k;
            }
        }
        if (k == ATI_R580_CGUARDS) {
            k = stale;
        }
        if (k < 0) {
            g->g_noslot++;
            return;
        }
    }
    if (!ati_r580_cguard_set(s, k, lo, hi)) {
        return;
    }
    g->gsurf[k] = sf - g->surf + 1;
    sf->guard = k + 1;
    g->g_arms++;
}

void ati_r580_r5gpu_release(ATIR580State *s, int why)
{
    ATIR580R5Gpu *g = s->r5g;
    bool keep;
    int i;

    if (!g || !qatomic_read(&s->r5g_res)) {
        return;
    }
    g->rb_why = why;
    keep = !g->verify && s->cg_ready && r5g_rel_guest(why);
    for (i = 0; i < R5G_NSURF; i++) {
        R5GSurf *sf = &g->surf[i];
        uint64_t a, b;

        if (!sf->live || sf->dx1 <= sf->dx0 || sf->dy1 <= sf->dy0) {
            continue;
        }
        r5g_drawn_span(sf, &a, &b);
        if (!keep || r5g_scanned(g, sf)) {
            r5g_surf_flush(g, sf);
            continue;
        }
        if (sf->guard && ati_r580_cguard_live(s, sf->guard - 1) &&
            a >= s->cg_lo[sf->guard - 1] && b <= s->cg_hi[sf->guard - 1]) {
            g->g_kept++;
            continue;
        }
        r5g_surf_flush(g, sf);
        r5g_guard_arm(g, sf);
    }
    r5g_res_update(g);
}

/*
 * The guest CPU is about to access [off, off+len) through a colour
 * guard, with the engine held: the surfaces under every window there go
 * back to VRAM and give their windows up. Returns the windows to lift.
 */
unsigned ati_r580_r5gpu_guard(ATIR580State *s, uint32_t off, unsigned len)
{
    ATIR580R5Gpu *g = s->r5g;
    unsigned lift = 0;
    int k;

    for (k = 0; k < ATI_R580_CGUARDS; k++) {
        if (s->cg_hi[k] <= s->cg_lo[k] || off >= s->cg_hi[k] ||
            off + len <= s->cg_lo[k]) {
            continue;
        }
        lift |= 1u << k;
        if (g && g->gsurf[k]) {
            R5GSurf *sf = &g->surf[g->gsurf[k] - 1];

            g->rb_why = R5G_RB_GUARD;
            g->g_traps++;
            if (++sf->traps == R5G_GUARD_TRAPS) {
                g->g_off++;
            }
            r5g_surf_flush(g, sf);
            sf->guard = 0;
            g->gsurf[k] = 0;
        }
    }
    if (g) {
        r5g_res_update(g);
    }
    return lift;
}

void ati_r580_r5gpu_sync(ATIR580State *s, uint32_t off, uint32_t len)
{
    ATIR580R5Gpu *g = s->r5g;

    if (!g || !qatomic_read(&s->r5g_res)) {
        return;
    }
    g->rb_why = R580_GLR_READ;
    r5g_flush_range(g, off, (uint64_t)off + len, NULL);
    r5g_res_update(g);
}

void ati_r580_r5gpu_wrote(ATIR580State *s, uint32_t off, uint32_t len)
{
    ATIR580R5Gpu *g = s->r5g;

    if (!g || qatomic_read(&s->r5g_quiet)) {
        return;
    }
    g->rb_why = R5G_RB_WROTE;
    r5g_flush_range(g, off, (uint64_t)off + len, NULL);
    r5g_invalidate(g, off, off + len, NULL);
    r5g_res_update(g);
}

/*
 * A depth fast clear is about to fill [0, w) x [0, h) of the buffer at
 * `off` in VRAM: drawn pixels it covers need not go back first.
 */
void ati_r580_r5gpu_zclear(ATIR580State *s, uint32_t off, uint32_t pitch,
                           uint32_t bpp, int w, int h)
{
    ATIR580R5Gpu *g = s->r5g;
    uint64_t len;
    int i;

    if (!g || w <= 0 || h <= 0 || off >= ATI_R580_VRAM_SIZE) {
        return;
    }
    len = (uint64_t)(h - 1) * pitch + (uint64_t)w * bpp;
    len = MIN(len, (uint64_t)ATI_R580_VRAM_SIZE - off);
    for (i = 0; i < R5G_NSURF && !qatomic_read(&s->r5g_quiet); i++) {
        R5GSurf *sf = &g->surf[i];
        uint32_t d;
        int ox, oy;

        if (!sf->live || sf->dx1 <= sf->dx0 || sf->pitch != pitch ||
            sf->bpp != bpp || off < sf->off) {
            continue;
        }
        d = off - sf->off;
        if (d % bpp || d / pitch >= (uint32_t)sf->h) {
            continue;
        }
        oy = d / pitch;
        ox = (d % pitch) / bpp;
        if (sf->dx0 >= ox && sf->dx1 <= ox + w &&
            sf->dy0 >= oy && sf->dy1 <= oy + h) {
            sf->dx0 = sf->dy0 = sf->dx1 = sf->dy1 = 0;
            g->zcleared++;
        }
    }
    ati_r580_r5gpu_wrote(s, off, len);
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
    g_free(g->vzbuf);
    g_free(g->tz);
    g_free(g->verts2);
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
        "gpu draws with depth/stencil %" PRIu64 ", of lines %" PRIu64
        ", of points %" PRIu64 "; drawn depth dropped by fast clears %"
        PRIu64 ", depth surfaces split off %" PRIu64 "\n"
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
        g->zdrawn, g->ldrawn, g->pdrawn, g->zcleared, g->zown,
        g->draw_ns / 1e6, g->max_draw_ns / 1e6, g->passes, g->verts_n,
        progs, pfail, lus / 1e3, lmax / 1e3, ns, g->surf_new, g->evicts, nt, g->tex_new,
        g->tex_hits, g->aliased, g->feedback,
        g->up_calls, g->up_bytes / 1e6, g->rd_calls, g->rd_bytes / 1e6,
        g->flushes, g->checks, g->epoch);
    g_string_append_printf(o, "readback backend ms %.1f, by:",
                           g->rb_read_ns / 1e6);
    {
        uint64_t tot = 0;

        for (i = 0; i < R5G_RB_MAX; i++) {
            tot += g->rb_ns[i];
            if (g->rb_n[i]) {
                g_string_append_printf(o, " %s %" PRIu64 " %.1f MB %.1f ms;",
                                       r5g_rb_names[i], g->rb_n[i],
                                       g->rb_bytes[i] / 1e6,
                                       g->rb_ns[i] / 1e6);
            }
        }
        g_string_append_printf(o, " total %.1f ms\nreadback surfaces:",
                               tot / 1e6);
    }
    for (i = 0; i < R5G_RBSURF && g->rbs[i].n; i++) {
        g_string_append_printf(o, " %08x/%u %" PRIu64 " %.1f MB;",
                               g->rbs[i].off, g->rbs[i].pitch, g->rbs[i].n,
                               g->rbs[i].bytes / 1e6);
    }
    g_string_append_printf(o, "\ncolour guards: laid %" PRIu64 ", kept %"
                           PRIu64 ", traps %" PRIu64 " (bytes newer than "
                           "VRAM %" PRIu64 "), no window %" PRIu64
                           ", given up %" PRIu64 "; surfaces adopted on "
                           "the GPU %" PRIu64 "\n", g->g_arms,
                           g->g_kept, g->g_traps, g->g_diff, g->g_noslot,
                           g->g_off, g->adopted);
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
        g_string_append_printf(o,
            "verify depth/stencil: draws %" PRIu64 ", differing %" PRIu64
            ", pixels %" PRIu64 "\n", g->zverified, g->zdiff_draws,
            g->zdiff_px);
        for (i = 0; i < R5G_HIST && i < g->zhist_n; i++) {
            unsigned h = (g->zhist_n - 1 - i) % R5G_HIST;

            g_string_append_printf(o, "  depth first %d,%d gpu %08x cpu %08x\n",
                                   g->zhist[h].x, g->zhist[h].y,
                                   g->zhist[h].gpu, g->zhist[h].cpu);
        }
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
