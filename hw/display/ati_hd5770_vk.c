/*
 * ATI Radeon HD 5770 (Juniper) 3D engine on the Vulkan rasterisation
 * pipeline (vk_raster.h): the card's side. It turns a draw of
 * ati_hd5770_gpu.h into the renderer's terms: guest surfaces as images
 * kept coherent with guest memory, pixel and vertex programs as GLSL
 * (ati_hd5770_glsl.c), registers as state.
 *
 * COHERENCE. An image is the copy of one surface (a texture with levels:
 * of its levels). Before a draw uses it, pages of its memory the device
 * names as written are compared with what was last seen there, and the
 * image is loaded again when they differ. An image drawn to is newer
 * than memory until it is written back: at a sync, at a fence when it
 * lies over the scanout or has not been drawn to for a second, when
 * another image over the same memory is used, or when the CPU has written
 * other bytes under it.
 *
 * FENCES. A fence with images to write back queues their read-out behind
 * the draws and returns a ticket; the thread vk_retire() waits for the
 * GPU, puts the pixels in memory and retires the ticket, tickets in
 * order. Until then the image stands for its memory: it is not compared
 * with it, another image over the same bytes waits for the ticket, and
 * so does a sync. The engine's thread owns everything else here; the
 * two share tickets, issued and retired under lock.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include <math.h>
#include "qemu/bswap.h"
#include "qemu/host-utils.h"
#include "qemu/timer.h"
#include "qemu/thread.h"
#include "qemu/rcu.h"
#include "ati_hd5770_gpu.h"
#include "vk_raster.h"

#define DC_STENCIL      (1u << 0)
#define DC_Z            (1u << 1)
#define DC_Z_WRITE      (1u << 2)

#define VK_HASH         4096
#define VK_LAZY_NS      1000000000ll
/* draws queued before a fence hands them to the GPU */
#define VK_UNSENT       128

enum {
    VD_TAKEN, VD_EMPTY, VD_RESOLVE, VD_PROG_FORM, VD_PROG_VTX,
    VD_PROG_INPUTS, VD_PROG_EMIT, VD_PROG_BUILD, VD_STENCIL, VD_RT_FMT,
    VD_RT_RAM, VD_BLEND_ORDER, VD_TEX_KIND, VD_TEX_FMT, VD_TEX_SURF,
    VD_TEX_TARGET, VD_DB_SURF, VD_W, VD_VS, VD_ROOM, VD_N,
};

static const char *const vd_name[VD_N] = {
    "taken", "empty", "resolve", "program form", "pixel vertex fetch", "inputs", "not expressible", "build failed",
    "stencil", "target format", "target in RAM", "blend order",
    "texture kind", "texture format", "texture surface", "texture is target",
    "depth surface", "w", "vertex stage", "room",
};

/* why an image went back to memory */
enum { VB_SYNC, VB_RANGE, VB_SCANOUT, VB_IDLE, VB_ALIAS, VB_N };

static const char *const vb_name[VB_N] = {
    "sync", "range sync", "scanout", "idle", "other image",
};

/* the parts of a draw, timed */
enum { VP_STATE, VP_IMAGES, VP_STAGES, VP_READY, VP_WORDS, VP_QUEUE, VP_N };

static const char *const vp_name[VP_N] = {
    "state", "images", "stages", "coherence", "words", "renderer",
};

typedef struct VkSurf {
    HD5770GpuSurf lv[HD5770_GPU_LEVELS];    /* lv[0]: the surface */
    uint32_t lw[HD5770_GPU_LEVELS], lh[HD5770_GPU_LEVELS];  /* elements */
    unsigned nlv, slices;
    uint64_t sb[HD5770_GPU_LEVELS];     /* a slice's bytes */
    bool mip;                   /* a texture's levels, never a target */
    bool vol;                   /* slices: the depth, halving by level */
    unsigned hf, zfmt, blk;
    uint32_t *pages;
    uint64_t lo, hi;            /* VRAM bytes of lv[0] */
    VkrImage *img;
    uint8_t *shadow[HD5770_GPU_LEVELS];
    uint64_t cov[HD5770_GPU_LEVELS];
    bool loaded, dirty;
    int r[4];                   /* drawn and not written back */
    uint64_t gen, epoch, used;
    uint64_t settled;           /* HD5770Vk.drawn when it last settled */
    uint64_t flight;            /* the ticket that writes it back */
    bool flying;                /* in HD5770Vk.flying */
    /*
     * PARTS. A target that is one level's slice of a texture's image is
     * copied into it on the GPU, not written back for it. serial names
     * an image as it is since its last load from memory; into: the image
     * this one was last copied into; copied: nothing was drawn to it
     * since.
     */
    uint64_t serial, into;
    bool copied;
    int64_t tdraw;
    struct VkSurf *hnext;       /* the next that starts at the same place */
} VkSurf;

/* a vertex buffer's bytes kept in the words */
typedef struct VkVb {
    uint64_t voff, gen, epoch;
    uint32_t len, off;
    uint8_t *seen;              /* the bytes as copied */
} VkVb;

/* a vertex stage's triangles in the words, of ring gen */
typedef struct VkTix {
    uint64_t id;
    uint32_t off, gen;
} VkTix;

/* buffers one thread works in */
typedef struct VkScratch {
    uint8_t *tmp;
    size_t tmp_len;
    uint64_t *offs;
    size_t noffs;
} VkScratch;

/* an image's pixels on their way to memory */
typedef struct VkBack {
    VkSurf *e;
    int r[4];
    const uint8_t *px;
} VkBack;

typedef struct VkTicket {
    uint64_t id;
    VkrMark *mark;
    GArray *backs;
} VkTicket;

/* what a draw's stages depend on, and the stages */
typedef struct VkPair {
    struct {
        uint64_t ps, vs;        /* the programs' hashes */
        uint32_t ps_n, vs_n, padded, has_vs;
        HD5770NatKey k;
        int32_t src[HD5770_GPU_INPUTS];
    } key;
    VkrShader *fs, *vsh;
    uint32_t flat;
} VkPair;

typedef struct VkNote {
    unsigned stage;
    char *src;
} VkNote;

/* a pipeline to build ahead: its stages by key */
typedef struct VkJob {
    char *fs, *vs;
    VkrPipeDesc d;
} VkJob;

typedef struct HD5770Vk {
    HD5770Gpu pub;
    VkRaster *r;
    uint8_t *vram, *ram;
    uint64_t vram_size, ram_size;
    GPtrArray *surfs, *dirty;
    GPtrArray *flying;          /* images a ticket may still write back */
    GArray *backs;              /* read-outs queued, not yet with the GPU */
    QemuThread thread;
    QemuMutex lock;
    QemuCond work, done;
    GQueue tickets;
    uint64_t issued, retired;
    bool quit;
    VkScratch sc, rsc;          /* the engine's, the retiring thread's */
    unsigned unsent;            /* draws queued since the last hand-over */
    VkSurf *hb[VK_HASH];        /* the images by where they start */
    uint64_t limit;             /* device memory the images may take */
    GHashTable *progs;          /* key -> VkrShader, or vk_fail */
    GHashTable *pairs;          /* VkPair by its key */
    uint64_t drawn;             /* counts the images that became dirty */
    uint64_t serial;
    /* build ahead: stages to compile (key -> VkNote), pipelines to build */
    GHashTable *noted;
    GArray *jobs;
    bool dry;
    uint64_t *pgen, gen, epoch, seq, failed;
    uint64_t *pulled;           /* per 64 pages: the epoch of the last pull */
    char *fail_log;
    char last_surf[160];        /* the last surface without an image */
    GHashTable *vbs;            /* VRAM offset -> VkVb */
    VkTix tix[4096];
    /* the vertex stages' constant file in the words, of ring cf_gen */
    uint32_t cf_seen[1024], cf_off, cf_gen;
    uint64_t nd[VD_N], dims[8], rtf[64], txf[64];
    uint64_t uploads, upload_bytes, nback[VB_N], back_bytes[VB_N];
    uint64_t fences, syncs, ranges, parts;
    uint64_t vs_draws, vs_tris, tris, vb_bytes, evicted;
    int64_t draw_ns, sync_ns, wait_ns;
    int64_t part_ns[VP_N];
    /* the retiring thread's: waiting for the GPU, copying to memory */
    int64_t rgpu_ns, rcopy_ns;
} HD5770Vk;

static int vk_fail;

static bool vk_decline(HD5770Vk *m, unsigned r, unsigned why)
{
    if (m->dry) {
        return false;
    }
    m->nd[r]++;
    m->pub.why = why;
    return false;
}

static void *vk_tmp(VkScratch *c, size_t n)
{
    if (c->tmp_len < n) {
        g_free(c->tmp);
        c->tmp = g_malloc(n);
        c->tmp_len = n;
    }
    return c->tmp;
}

/* surface bytes from o on in memory: *n of them are contiguous there */
static inline uint8_t *vk_at(HD5770Vk *m, const HD5770GpuSurf *s, uint64_t o,
                             size_t *n)
{
    if (s->space) {
        uint64_t q = o + s->pg0;

        *n = 4096 - (q & 4095);
        return m->ram + s->pages[q >> 12] + (q & 4095);
    }
    *n = SIZE_MAX;
    return m->vram + s->voff + o;
}

/* w x h elements at (x0, y0) of a surface from or to rows of px */
static void vk_rect_io(HD5770Vk *m, VkScratch *c, const HD5770GpuSurf *s,
                       uint32_t x0, uint32_t y0, uint32_t w, uint32_t h,
                       uint8_t *px, bool put)
{
    uint32_t bpe = s->s.bpe, x, y;
    size_t n;

    if (c->noffs < w) {
        g_free(c->offs);
        c->offs = g_new(uint64_t, w);
        c->noffs = w;
    }
    for (y = 0; y < h; y++) {
        uint8_t *row = px + (size_t)y * w * bpe;

        if (!s->space && s->s.fast &&
            s->s.mode < HD5770_ARRAY_1D_TILED_THIN1) {
            uint8_t *p = m->vram + s->voff +
                         (uint64_t)(y0 + y) * s->s.row_bytes +
                         (uint64_t)x0 * bpe;

            if (put) {
                memcpy(p, row, (size_t)w * bpe);
            } else {
                memcpy(row, p, (size_t)w * bpe);
            }
            continue;
        }
        hd5770_surf_row(&s->s, x0, y0 + y, w, c->offs);
        for (x = 0; x < w; x++) {
            uint8_t *p = vk_at(m, s, c->offs[x], &n);

            if (put) {
                memcpy(p, row + (size_t)x * bpe, bpe);
            } else {
                memcpy(row + (size_t)x * bpe, p, bpe);
            }
        }
    }
}

/* the bytes of a surface in memory, to the end of row y1 - 1 */
static uint64_t vk_cover(const HD5770GpuSurf *s)
{
    return hd5770_surf_size(&s->s);
}

static bool vk_in_vram(const HD5770Vk *m, const HD5770GpuSurf *s, uint64_t n)
{

    return s->space || (s->voff <= m->vram_size && n <= m->vram_size - s->voff);
}

/* depth planes: memory's elements to floats and back */
static void vk_z_in(unsigned zfmt, uint8_t *px, uint8_t *out, size_t n)
{
    size_t i;

    for (i = 0; i < n; i++) {
        float z;

        if (zfmt == 1) {
            z = lduw_le_p(px + i * 2) / 65535.0f;
        } else if (zfmt == 2) {
            z = (float)((ldl_le_p(px + i * 4) & 0xffffff) / 16777215.0);
        } else {
            memcpy(&z, px + i * 4, 4);
        }
        memcpy(out + i * 4, &z, 4);
    }
}

static void vk_z_out(unsigned zfmt, const uint8_t *in, uint8_t *px, size_t n)
{
    size_t i;

    for (i = 0; i < n; i++) {
        float z;
        double zd;

        memcpy(&z, in + i * 4, 4);
        zd = z > 0.0f ? MIN((double)z, 1.0) : 0.0;
        if (zfmt == 1) {
            stw_le_p(px + i * 2, (uint16_t)lrint(zd * 65535.0));
        } else if (zfmt == 2) {
            stl_le_p(px + i * 4, (ldl_le_p(px + i * 4) & 0xff000000u) |
                     (uint32_t)lrint(zd * 16777215.0));
        } else {
            memcpy(px + i * 4, &z, 4);
        }
    }
}

/* the device's log of writes to VRAM [lo, hi) into the page generations */
static void vk_pull(HD5770Vk *m, uint64_t lo, uint64_t hi)
{
    uint64_t c, i, bits;

    if (!m->pub.written) {
        return;
    }
    for (c = lo & ~0x3ffffull; c < hi; c += 0x40000) {
        /* the log is asked once an epoch */
        if (m->pulled[c >> 18] == m->epoch) {
            continue;
        }
        m->pulled[c >> 18] = m->epoch;
        bits = m->pub.written(m->pub.opaque, c);
        if (!bits) {
            continue;
        }
        m->gen++;
        for (i = 0; i < 64; i++) {
            if ((bits >> i) & 1 && (c >> 12) + i < (m->vram_size >> 12)) {
                m->pgen[(c >> 12) + i] = m->gen;
            }
        }
    }
}

static void vk_named(HD5770Vk *m, uint64_t lo, uint64_t hi)
{
    uint64_t p;

    m->gen++;
    for (p = lo >> 12; p < MIN((hi + 4095) >> 12, m->vram_size >> 12); p++) {
        m->pgen[p] = m->gen;
    }
}

static void vk_load(HD5770Vk *m, VkSurf *e)
{
    unsigned l, sl;

    /* PARTS copied into it are gone */
    e->serial = ++m->serial;
    for (l = 0; l < e->nlv; l++) {
        size_t n = (size_t)e->lw[l] * e->lh[l] * e->lv[l].s.bpe;
        uint8_t *px = vk_tmp(&m->sc, n + (size_t)e->lw[l] * e->lh[l] * 4);
        uint8_t *src = px;

        for (sl = 0; sl < (e->vol ? MAX(e->slices >> l, 1u) : e->slices);
             sl++) {
            HD5770GpuSurf s = e->lv[l];

            s.s.base += sl * e->sb[l];
            s.voff += sl * e->sb[l];
            vk_rect_io(m, &m->sc, &s, 0, 0, e->lw[l], e->lh[l], px, false);
            if (s.space) {
                if (e->shadow[l] && !memcmp(e->shadow[l], px, n) &&
                    e->loaded) {
                    continue;
                }
                if (!e->shadow[l]) {
                    e->shadow[l] = g_malloc(n);
                }
                memcpy(e->shadow[l], px, n);
            }
            if (e->zfmt) {
                src = px + n;
                vk_z_in(e->zfmt, px, src, (size_t)e->lw[l] * e->lh[l]);
            }
            vkr_image_write(m->r, e->img, l, sl, 0, 0, e->lw[l] * e->blk,
                            e->lh[l] * e->blk, src);
            m->uploads++;
            m->upload_bytes += n;
        }
        if (!e->lv[l].space) {
            if (!e->shadow[l]) {
                e->shadow[l] = g_malloc(e->cov[l]);
            }
            memcpy(e->shadow[l], m->vram + e->lv[l].voff, e->cov[l]);
        }
    }
    e->loaded = true;
}

/* the read-out of the pixels drawn is queued: the image is not dirty */
static void vk_back_queue(HD5770Vk *m, VkSurf *e, unsigned why)
{
    VkBack b = { .e = e, .r = { e->r[0], e->r[1], e->r[2], e->r[3] } };
    uint32_t w = e->r[2] - e->r[0], h = e->r[3] - e->r[1];

    e->dirty = false;
    g_ptr_array_remove_fast(m->dirty, e);
    b.px = vkr_image_read_later(m->r, e->img, e->r[0], e->r[1], w, h);
    g_array_append_val(m->backs, b);
    m->nback[why]++;
    m->back_bytes[why] += (uint64_t)w * h * e->lv[0].s.bpe;
}

/*
 * pixels read out go to memory, the bytes written are reported. own: on
 * the engine's thread, which also notes memory as seen; the retiring
 * thread leaves that to the write log.
 */
static void vk_back_apply(HD5770Vk *m, const VkBack *b, VkScratch *c,
                          bool own)
{
    VkSurf *e = b->e;
    const HD5770GpuSurf *s = &e->lv[0];
    uint32_t w = b->r[2] - b->r[0], h = b->r[3] - b->r[1];
    size_t n = (size_t)w * h;
    uint8_t *dst = (uint8_t *)b->px;
    uint64_t lo, hi;

    if (e->zfmt) {
        dst = vk_tmp(c, n * s->s.bpe);
        if (e->zfmt == 2) {
            /* the bits above the depth stay */
            vk_rect_io(m, c, s, b->r[0], b->r[1], w, h, dst, false);
        }
        vk_z_out(e->zfmt, b->px, dst, n);
    }
    vk_rect_io(m, c, s, b->r[0], b->r[1], w, h, dst, true);
    hd5770_surf_rows(&s->s, b->r[1], b->r[3], &lo, &hi);
    lo = MIN(lo - s->s.base, e->cov[0]);
    hi = MIN(hi - s->s.base, e->cov[0]);
    if (e->shadow[0]) {
        memcpy(e->shadow[0] + lo, m->vram + s->voff + lo, hi - lo);
    }
    if (own) {
        vk_named(m, s->voff + lo, s->voff + hi);
        e->gen = m->gen;
    }
    if (m->pub.done) {
        m->pub.done(m->pub.opaque, s->voff + lo, s->voff + hi);
    }
}

/* ticket id, and so every one before it, is retired */
static void vk_wait(HD5770Vk *m, uint64_t id)
{
    int64_t t0;

    if (qatomic_read(&m->retired) >= id) {
        return;
    }
    t0 = get_clock();
    qemu_mutex_lock(&m->lock);
    while (m->retired < id) {
        qemu_cond_wait(&m->done, &m->lock);
    }
    qemu_mutex_unlock(&m->lock);
    m->wait_ns += get_clock() - t0;
}

/* the read-outs queued are in memory */
static void vk_backs_now(HD5770Vk *m)
{
    VkrMark *k;
    unsigned i;

    if (!m->backs->len) {
        return;
    }
    k = vkr_mark(m->r);
    vkr_flush(m->r, true);
    m->unsent = 0;
    for (i = 0; i < m->backs->len; i++) {
        vk_back_apply(m, &g_array_index(m->backs, VkBack, i), &m->sc, true);
    }
    g_array_set_size(m->backs, 0);
    vkr_mark_free(m->r, k);
}

/*
 * images newer than memory over VRAM [lo, hi), but e and its parts, are
 * written back, and none over it is on its way there
 */
/* o's surface is slice sl of level l of the texture image e */
static bool vk_part(const VkSurf *e, const VkSurf *o, unsigned *l,
                    unsigned *sl)
{
    const HD5770Surf *b = &o->lv[0].s;

    if (!e->mip || o->mip || e->hf != o->hf || e->zfmt || o->zfmt ||
        e->blk != 1 || e->lv[0].space || o->lv[0].space) {
        return false;
    }
    for (*l = 0; *l < e->nlv; (*l)++) {
        const HD5770Surf *a = &e->lv[*l].s;
        uint64_t at = o->lv[0].voff - e->lv[*l].voff;
        unsigned n = e->vol ? MAX(e->slices >> *l, 1u) : e->slices;

        if (o->lv[0].voff < e->lv[*l].voff || !e->sb[*l] ||
            at % e->sb[*l] || at / e->sb[*l] >= n) {
            continue;
        }
        *sl = at / e->sb[*l];
        return a->pitch == b->pitch && a->mode == b->mode &&
               a->bpe == b->bpe && a->fast == b->fast &&
               a->row_bytes == b->row_bytes && a->banks == b->banks &&
               a->bankw == b->bankw && a->bankh == b->bankh &&
               a->mtilea == b->mtilea && a->tile_split == b->tile_split &&
               a->pipes == b->pipes &&
               a->pipe_interleave == b->pipe_interleave &&
               a->nondisp == b->nondisp && o->lh[0] >= e->lh[*l];
    }
    return false;
}

/* PARTS: the parts of e drawn since they were copied to it are copied */
static void vk_parts(HD5770Vk *m, VkSurf *e)
{
    GPtrArray *from[2] = { m->dirty, m->flying };
    unsigned i, k, l, sl;

    for (k = 0; k < 2 && e->mip; k++) {
        for (i = 0; i < from[k]->len; i++) {
            VkSurf *o = g_ptr_array_index(from[k], i);

            if (o == e || o->lo >= e->hi || e->lo >= o->hi ||
                (o->copied && o->into == e->serial) ||
                !vk_part(e, o, &l, &sl)) {
                continue;
            }
            vkr_image_copy(m->r, o->img, e->img, l, sl, e->lw[l], e->lh[l]);
            o->into = e->serial;
            o->copied = true;
            m->parts++;
        }
    }
}

static void vk_settle(HD5770Vk *m, const VkSurf *e, uint64_t lo, uint64_t hi,
                      unsigned why)
{
    uint64_t retired = qatomic_read(&m->retired), wait = 0;
    unsigned i, l, sl;

    for (i = 0; i < m->flying->len; i++) {
        VkSurf *o = g_ptr_array_index(m->flying, i);

        if (o->flight <= retired) {
            o->flying = false;
            g_ptr_array_remove_index_fast(m->flying, i--);
        } else if (o != e && o->lo < hi && lo < o->hi &&
                   !(e && vk_part(e, o, &l, &sl))) {
            wait = MAX(wait, o->flight);
        }
    }
    for (i = 0; i < m->dirty->len; i++) {
        VkSurf *o = g_ptr_array_index(m->dirty, i);

        if (o != e && o->lo < hi && lo < o->hi &&
            !(e && vk_part(e, o, &l, &sl))) {
            wait = MAX(wait, o->flight);
            vk_back_queue(m, o, why);
            i--;
        }
    }
    vk_wait(m, wait);
    vk_backs_now(m);
}

/* the image is what memory holds, or newer */
static void vk_fresh(HD5770Vk *m, VkSurf *e)
{
    unsigned l;
    bool diff = !e->loaded;

    if (e->dirty || e->flight > qatomic_read(&m->retired) ||
        (e->epoch == m->epoch && e->loaded)) {
        return;
    }
    e->epoch = m->epoch;
    if (e->lv[0].space) {
        vk_load(m, e);
        return;
    }
    for (l = 0; l < e->nlv && !diff; l++) {
        uint64_t a = e->lv[l].voff, z = a + e->cov[l], p;

        vk_pull(m, a, z);
        for (p = a >> 12; p <= (z - 1) >> 12 && !diff; p++) {
            uint64_t b0 = MAX(p << 12, a), b1 = MIN((p + 1) << 12, z);

            if (m->pub.written && m->pgen[p] <= e->gen) {
                continue;
            }
            diff = memcmp(e->shadow[l] + (b0 - a), m->vram + b0, b1 - b0);
        }
    }
    e->gen = m->gen;
    if (diff) {
        vk_load(m, e);
    }
}

/* the image is ready for a draw: none over it is newer than it */
static void vk_ready(HD5770Vk *m, VkSurf *e)
{
    /* nothing over it was drawn to since it settled */
    bool settled = e->settled == m->drawn;
    uint64_t serial = e->serial;
    unsigned l;

    e->used = m->seq;
    for (l = 0; l < e->nlv && !e->lv[l].space && !settled; l++) {
        vk_settle(m, e, e->lv[l].voff, e->lv[l].voff + e->cov[l], VB_ALIAS);
    }
    vk_fresh(m, e);
    /* a load from memory lost the parts copied before */
    if (!settled || e->serial != serial) {
        vk_parts(m, e);
    }
    e->settled = m->drawn;
}

static void vk_surf_free(gpointer p)
{
    VkSurf *e = p;
    unsigned l;

    for (l = 0; l < HD5770_GPU_LEVELS; l++) {
        g_free(e->shadow[l]);
    }
    g_free(e->pages);
    g_free(e);
}

static unsigned vk_hash(const HD5770GpuSurf *s)
{
    uint64_t k = !s->space ? s->voff :
                 ((s->npages ? s->pages[0] : 0) + (uint64_t)s->pg0) |
                 1ull << 63;

    return (k * 0x9e3779b97f4a7c15ull) >> 52;
}

static gint vk_older(gconstpointer a, gconstpointer b)
{
    const VkSurf *x = *(VkSurf *const *)a, *y = *(VkSurf *const *)b;

    return x->used < y->used ? -1 : x->used > y->used;
}

/* the quarter of the images used longest ago goes */
static void vk_evict(HD5770Vk *m)
{
    unsigned i, n = m->surfs->len / 4;

    g_ptr_array_sort(m->surfs, vk_older);
    for (i = 0; i < m->surfs->len && n; i++) {
        VkSurf *e = g_ptr_array_index(m->surfs, i), **h;

        if (e->dirty || e->used == m->seq ||
            e->flight > qatomic_read(&m->retired)) {
            continue;
        }
        if (e->flying) {
            g_ptr_array_remove_fast(m->flying, e);
        }
        for (h = &m->hb[vk_hash(&e->lv[0])]; *h != e; h = &(*h)->hnext) {
            /* search */
        }
        *h = e->hnext;
        vkr_image_free(m->r, e->img);
        g_ptr_array_remove_index(m->surfs, i);
        m->evicted++;
        i--;
        n--;
    }
    vkr_flush(m->r, true);
}

static bool vk_same(const VkSurf *e, const HD5770GpuSurf *s)
{
    const HD5770Surf *a = &e->lv[0].s, *b = &s->s;

    if (e->lv[0].space != s->space || a->pitch != b->pitch ||
        a->mode != b->mode || a->bpe != b->bpe || a->fast != b->fast) {
        return false;
    }
    if (a->fast && a->mode < HD5770_ARRAY_1D_TILED_THIN1) {
        if (a->row_bytes != b->row_bytes) {
            return false;
        }
    } else if (a->banks != b->banks || a->bankw != b->bankw ||
               a->bankh != b->bankh || a->mtilea != b->mtilea ||
               a->tile_split != b->tile_split || a->pipes != b->pipes ||
               a->pipe_interleave != b->pipe_interleave ||
               a->nondisp != b->nondisp) {
        return false;
    }
    if (!s->space) {
        return e->lv[0].voff == s->voff;
    }
    return e->lv[0].npages >= s->npages && e->lv[0].pg0 == s->pg0 &&
           !memcmp(e->pages, s->pages, s->npages * sizeof(uint32_t));
}

/* no image for the surface: why, for the counters */
static VkSurf *vk_no_surf(HD5770Vk *m, VkSurf *e, const char *why)
{
    const HD5770GpuSurf *s = &e->lv[0];

    snprintf(m->last_surf, sizeof(m->last_surf),
             "%s: format %u, %u x %u of %u x %u, %u bytes, mode %u, space %u,"
             " levels %u, slices %u%s", why, e->hf, e->lw[0], e->lh[0],
             s->s.pitch, s->s.height, s->s.bpe, s->s.mode, s->space, e->nlv,
             e->slices, e->vol ? " deep" : "");
    vk_surf_free(e);
    return NULL;
}

static VkSurf *vk_find(HD5770Vk *m, const HD5770GpuSurf *s, unsigned hf,
                       unsigned zfmt)
{
    VkSurf *e;

    for (e = m->hb[vk_hash(s)]; e; e = e->hnext) {
        if (!e->mip && e->hf == hf && e->zfmt == zfmt &&
            e->lv[0].s.height >= s->s.height &&
            (e->lv[0].s.fast || e->lv[0].s.height == s->s.height) &&
            vk_same(e, s)) {
            return e;
        }
    }
    return NULL;
}

/* the image of a texture's levels 0..nlv-1 */
static VkSurf *vk_find_tex(HD5770Vk *m, const HD5770GpuSurf *s, unsigned hf,
                           unsigned blk, const HD5770GpuTex *t, unsigned nlv)
{
    VkSurf *o;
    unsigned l;

    for (o = m->hb[vk_hash(s)]; o; o = o->hnext) {
        if (!o->mip || o->hf != hf || o->nlv != nlv ||
            o->slices != MAX(t->slices, 1u) || o->vol != (t->dim == 2) ||
            !vk_same(o, s) || o->lv[0].s.height != s->s.height) {
            continue;
        }
        for (l = 1; l < nlv && o->lv[l].voff == t->lv[l].voff &&
                    o->lw[l] * blk >= t->lw[l] &&
                    o->lh[l] * blk >= t->lh[l]; l++) {
            /* the levels */
        }
        if (l == nlv) {
            return o;
        }
    }
    return NULL;
}

/* the image of a surface; t: of a texture's levels 0..nlv-1 */
static VkSurf *vk_surf(HD5770Vk *m, const HD5770GpuSurf *s, unsigned hf,
                       unsigned zfmt, unsigned blk, const HD5770GpuTex *t,
                       unsigned nlv)
{
    VkSurf *e;
    unsigned i, l;

    e = t ? vk_find_tex(m, s, hf, blk, t, nlv) : vk_find(m, s, hf, zfmt);
    if (e) {
        /* this draw's: not dropped for a later image of it */
        e->used = m->seq;
        return e;
    }
    if (vkr_image_bytes(m->r) > m->limit) {
        vk_evict(m);
    }
    e = g_new0(VkSurf, 1);
    e->serial = ++m->serial;
    e->hf = hf;
    e->zfmt = zfmt;
    e->blk = blk;
    e->nlv = t ? nlv : 1;
    e->mip = t != NULL;
    e->slices = t ? MAX(t->slices, 1u) : 1;
    e->vol = t && t->dim == 2;
    e->lv[0] = *s;
    for (l = 0; l < e->nlv; l++) {
        e->lv[l] = l ? t->lv[l] : *s;
        e->sb[l] = t ? t->slice_bytes[l] : 0;
        e->cov[l] = ((e->vol ? MAX(e->slices >> l, 1u) : e->slices) - 1) *
                    e->sb[l] + vk_cover(&e->lv[l]);
        /* a texture's levels: their texels; else the whole surface */
        e->lw[l] = t ? DIV_ROUND_UP(l ? t->lw[l] : t->w, blk)
                     : e->lv[l].s.pitch;
        e->lh[l] = t ? DIV_ROUND_UP(l ? t->lh[l] : t->h, blk)
                     : e->lv[l].s.height;
        if ((e->lv[l].space && (l || e->slices > 1)) ||
            !vk_in_vram(m, &e->lv[l], e->cov[l]) ||
            !hd5770_surf_mode_ok(e->lv[l].s.mode)) {
            return vk_no_surf(m, e, "memory or mode");
        }
        if (e->lw[l] > e->lv[l].s.pitch || e->lh[l] > e->lv[l].s.height) {
            return vk_no_surf(m, e, "level larger than its surface");
        }
    }
    if (s->space) {
        e->pages = g_memdup2(s->pages, s->npages * sizeof(uint32_t));
        e->lv[0].pages = e->pages;
        if (!m->ram) {
            return vk_no_surf(m, e, "no guest RAM");
        }
        for (i = 0; i < s->npages; i++) {
            if (e->pages[i] > m->ram_size - 4096) {
                return vk_no_surf(m, e, "page outside guest RAM");
            }
        }
    } else {
        e->lo = s->voff;
        e->hi = s->voff + e->cov[0];
    }
    for (i = 0; i < 2 && !e->img; i++) {
        if (i) {
            vk_evict(m);
        }
        e->img = vkr_image_new(m->r, hf, e->lw[0] * blk, e->lh[0] * blk,
                               e->nlv, e->slices, e->vol,
                               !t && vkr_format_ok(m->r, hf, true));
    }
    if (!e->img) {
        return vk_no_surf(m, e, "no host image");
    }
    e->used = m->seq;
    e->hnext = m->hb[vk_hash(&e->lv[0])];
    m->hb[vk_hash(&e->lv[0])] = e;
    g_ptr_array_add(m->surfs, e);
    return e;
}

static void vk_drew(HD5770Vk *m, VkSurf *e, int x0, int y0, int x1, int y1)
{
    if (e->dirty && e->copied) {
        m->drawn++;
    }
    e->copied = false;
    if (!e->dirty) {
        e->dirty = true;
        m->drawn++;
        e->r[0] = x0;
        e->r[1] = y0;
        e->r[2] = x1;
        e->r[3] = y1;
        g_ptr_array_add(m->dirty, e);
    } else {
        e->r[0] = MIN(e->r[0], x0);
        e->r[1] = MIN(e->r[1], y0);
        e->r[2] = MAX(e->r[2], x1);
        e->r[3] = MAX(e->r[3], y1);
    }
    e->tdraw = get_clock();
}

static unsigned vk_tex_fmt(const HD5770GpuTex *t)
{
    if (t->bc) {
        unsigned sg = t->sgn & (t->bc == 5 ? 3 : 1);

        if (t->bc == 5 && (sg == 1 || sg == 2)) {
            return VKR_FMT_N;
        }
        return t->bc == 1 ? VKR_FMT_BC1 : t->bc == 2 ? VKR_FMT_BC2 :
               t->bc == 3 ? VKR_FMT_BC3 :
               t->bc == 4 ? (sg ? VKR_FMT_BC4S : VKR_FMT_BC4) :
               sg ? VKR_FMT_BC5S : VKR_FMT_BC5;
    }
    if (t->unorm8) {
        return VKR_FMT_RGBA8;
    }
    if (t->fmt == HD5770_FMT_16_16_16_16_FLOAT && t->is_float &&
        t->comps == 4) {
        return VKR_FMT_RGBA16F;
    }
    if (t->fmt == HD5770_FMT_32_FLOAT && t->is_float && t->comps == 1) {
        return VKR_FMT_R32F;
    }
    if (t->fmt == HD5770_FMT_32_32_FLOAT && t->is_float && t->comps == 2) {
        return VKR_FMT_RG32F;
    }
    if (t->comps == 4 &&
        ((t->fmt == HD5770_FMT_32_32_32_32_FLOAT && t->is_float) ||
         (t->fmt == HD5770_FMT_32_32_32_32 && t->num == HD5770_NUM_INT))) {
        /* floats, or their bits as integers */
        return VKR_FMT_RGBA32F;
    }
    if (t->fmt == HD5770_FMT_16_16_16_16 && t->num == HD5770_NUM_NORM &&
        !t->sgn && !t->is_float && t->comps == 4) {
        return VKR_FMT_RGBA16;
    }
    if (t->num == HD5770_NUM_NORM && !t->sgn && t->comps == 2 &&
        t->fmt == HD5770_FMT_8_8) {
        return VKR_FMT_RG8;
    }
    if (t->num == HD5770_NUM_NORM && !t->sgn && !t->is_float &&
        t->comps == 4 && t->fmt == HD5770_FMT_4_4_4_4) {
        /* the host's channels are the components in reverse */
        return VKR_FMT_ABGR4;
    }
    if (t->num == HD5770_NUM_NORM && !t->sgn && !t->is_float &&
        t->comps == 3 && t->fmt == HD5770_FMT_5_6_5) {
        /* the host's channels are the components in reverse */
        return VKR_FMT_RGB565;
    }
    if (t->num != HD5770_NUM_NORM || t->sgn || t->comps != 1) {
        return VKR_FMT_N;
    }
    return t->fmt == HD5770_FMT_8 ? VKR_FMT_R8 :
           t->fmt == HD5770_FMT_16 ? VKR_FMT_R16 : VKR_FMT_N;
}

/* a colour buffer: its image format, output class and write mask */
static unsigned vk_rt(const HD5770Surf *s, uint32_t fmt, uint32_t num,
                      uint32_t comps, bool is_float, bool unorm8,
                      uint32_t *src, uint32_t wmask, uint32_t *blend,
                      unsigned *rhf, unsigned *out, unsigned *mask)
{
    unsigned i, perm = 0;

    *out = 0;
    *mask = 0;
    if (num == 4 && comps == 1 &&
        (fmt == HD5770_FMT_8 || fmt == HD5770_FMT_32)) {
        *out = 1;
        *blend = 0;
        *mask = (wmask >> src[0]) & 1 ? 0xf : 0;
        *rhf = fmt == HD5770_FMT_8 ? VKR_FMT_R8 : VKR_FMT_RGBA8;
        return VD_TAKEN;
    }
    if (num == 0 && !is_float && comps == 1 && fmt == HD5770_FMT_8) {
        *mask = (wmask >> src[0]) & 1 ? 0xf : 0;
        *rhf = VKR_FMT_R8;
        return *blend ? VD_BLEND_ORDER : VD_TAKEN;
    }
    if (num == 0 && !is_float && comps == 2 && fmt == HD5770_FMT_8_8 &&
        src[0] < 4 && src[1] < 4) {
        *mask = ((wmask >> src[0]) & 1) | ((wmask >> src[1]) & 1) << 1;
        *rhf = VKR_FMT_RG8;
        return *blend ? VD_BLEND_ORDER : VD_TAKEN;
    }
    if (is_float && comps == 2 && fmt == HD5770_FMT_32_32_FLOAT &&
        src[0] < 4 && src[1] < 4) {
        *out = 2;
        *mask = ((wmask >> src[0]) & 1) | ((wmask >> src[1]) & 1) << 1;
        *rhf = VKR_FMT_RG32F;
        return *blend ? VD_BLEND_ORDER : VD_TAKEN;
    }
    if (num == 0 && !is_float && comps == 3 && fmt == HD5770_FMT_5_6_5 &&
        src[0] < 4 && src[1] < 4 && src[2] < 4) {
        uint32_t t = src[0];

        /* no alpha kept: the fourth channel is the program's, for blends */
        src[0] = src[2];
        src[2] = t;
        src[3] = 3;
        for (i = 0; i < 3; i++) {
            *mask |= ((wmask >> src[i]) & 1) << i;
        }
        *rhf = VKR_FMT_RGB565;
        return VD_TAKEN;
    }
    for (i = 0; i < 4; i++) {
        perm |= src[i] < 4 ? 1u << src[i] : 0x10;
    }
    if (num == 4 && comps == 4 && fmt == HD5770_FMT_32_32_32_32 &&
        perm == 0xf) {
        /* integers: the channels' bits, kept as floats */
        *out = 2;
        *blend = 0;
        *rhf = VKR_FMT_RGBA32F;
        for (i = 0; i < 4; i++) {
            *mask |= ((wmask >> src[i]) & 1) << i;
        }
        return VD_TAKEN;
    }
    *rhf = VKR_FMT_RGBA8;
    if ((fmt == HD5770_FMT_16_16_16_16_FLOAT ||
         fmt == HD5770_FMT_32_32_32_32 ||
         fmt == HD5770_FMT_32_32_32_32_FLOAT) && is_float && comps == 4 &&
        perm == 0xf) {
        *out = 2;
        *rhf = s->bpe == 16 ? VKR_FMT_RGBA32F : VKR_FMT_RGBA16F;
    } else if (fmt == HD5770_FMT_16_16_16_16 && num == 0 && !is_float &&
               comps == 4 && perm == 0xf) {
        *rhf = VKR_FMT_RGBA16;
    } else if (fmt == HD5770_FMT_4_4_4_4 && num == 0 && !is_float &&
               comps == 4 && perm == 0xf) {
        uint32_t t;

        *rhf = VKR_FMT_ABGR4;
        t = src[0], src[0] = src[3], src[3] = t;
        t = src[1], src[1] = src[2], src[2] = t;
    } else if (!unorm8 || comps != 4 || is_float || perm != 0xf) {
        return VD_RT_FMT;
    }
    if (*blend && src[3] != 3) {
        return VD_BLEND_ORDER;
    }
    for (i = 0; i < 4; i++) {
        *mask |= ((wmask >> src[i]) & 1) << i;
    }
    return VD_TAKEN;
}

static unsigned vk_factor(unsigned f)
{
    switch (f) {
    case 1: return VKR_BF_ONE;
    case 2: return VKR_BF_SRC_COLOR;
    case 3: return VKR_BF_INV_SRC_COLOR;
    case 4: return VKR_BF_SRC_ALPHA;
    case 5: return VKR_BF_INV_SRC_ALPHA;
    case 6: return VKR_BF_DST_ALPHA;
    case 7: return VKR_BF_INV_DST_ALPHA;
    case 8: return VKR_BF_DST_COLOR;
    case 9: return VKR_BF_INV_DST_COLOR;
    case 10: return VKR_BF_SRC_ALPHA_SAT;
    case 13: return VKR_BF_CONST_COLOR;
    case 14: return VKR_BF_INV_CONST_COLOR;
    case 19: return VKR_BF_CONST_ALPHA;
    case 20: return VKR_BF_INV_CONST_ALPHA;
    default: return VKR_BF_ZERO;
    }
}

/* combine(): function fn of CB_BLEND_CONTROL with its factors */
static void vk_combine(unsigned fn, unsigned sf, unsigned df, uint8_t *op,
                       uint8_t *s, uint8_t *d)
{
    static const uint8_t ops[5] = {
        VKR_BO_ADD, VKR_BO_SUB, VKR_BO_MIN, VKR_BO_MAX, VKR_BO_RSUB,
    };

    *s = vk_factor(sf);
    *d = vk_factor(df);
    if (fn < 5) {
        *op = ops[fn];
    } else {
        *op = VKR_BO_ADD;
        *s = VKR_BF_ONE;
        *d = VKR_BF_ZERO;
    }
}

static void vk_blend(uint32_t blend, unsigned mask, VkrBlend *b)
{
    unsigned csrc = blend & 0x1f, cfn = (blend >> 5) & 7;
    unsigned cdst = (blend >> 8) & 0x1f;
    unsigned asrc = csrc, afn = cfn, adst = cdst;

    memset(b, 0, sizeof(*b));
    b->mask = mask;
    if (!(blend & (1u << 30))) {
        return;
    }
    if ((blend >> 29) & 1) {
        asrc = (blend >> 16) & 0x1f;
        afn = (blend >> 21) & 7;
        adst = (blend >> 24) & 0x1f;
    }
    b->on = 1;
    vk_combine(cfn, csrc, cdst, &b->op, &b->src, &b->dst);
    vk_combine(afn, asrc, adst, &b->aop, &b->asrc, &b->adst);
}

/* SQ_TEX_SAMPLER word 0 and the texture's levels as a sampler */
static void vk_sampler(const HD5770GpuTex *t, bool mip, unsigned last,
                       VkrSampler *s)
{
    static const uint8_t wrap[8] = {
        VKR_WRAP_REPEAT, VKR_WRAP_MIRROR, VKR_WRAP_CLAMP,
        VKR_WRAP_MIRROR_ONCE, VKR_WRAP_BORDER, VKR_WRAP_MIRROR_ONCE,
        VKR_WRAP_BORDER, VKR_WRAP_MIRROR_ONCE,
    };
    uint32_t s0 = t->samp[0];

    memset(s, 0, sizeof(*s));
    s->wrap_u = wrap[s0 & 7];
    s->wrap_v = wrap[(s0 >> 3) & 7];
    s->wrap_w = wrap[(s0 >> 6) & 7];
    s->border = t->border[3] == 0.0f ? VKR_BORDER_ZERO :
                t->border[0] == 0.0f ? VKR_BORDER_BLACK : VKR_BORDER_WHITE;
    s->mag_linear = (s0 >> 9) & 1;
    s->min_linear = (s0 >> 11) & 1;
    if (mip) {
        s->mip_linear = t->mip_filter != 1;
        s->lod_bias = t->lod_bias;
        s->min_lod = MAX(t->min_lod, 0.0f);
        s->max_lod = MIN(t->max_lod, (float)last);
    } else {
        s->mag_linear |= s->min_linear;
        s->min_linear = s->mag_linear;
    }
}

static guint vk_pair_hash(gconstpointer p)
{
    const uint8_t *b = p;
    guint h = 2166136261u;
    size_t i;

    for (i = 0; i < sizeof(((VkPair *)0)->key); i++) {
        h = (h ^ b[i]) * 16777619u;
    }
    return h;
}

static gboolean vk_pair_equal(gconstpointer a, gconstpointer b)
{
    return !memcmp(a, b, sizeof(((VkPair *)0)->key));
}

/* build ahead: the stage of the key is known, or noted */
static bool vk_noted(HD5770Vk *m, const char *key)
{
    return g_hash_table_contains(m->progs, key) ||
           g_hash_table_contains(m->noted, key);
}

static void vk_note(HD5770Vk *m, const char *key, unsigned stage, char *src)
{
    VkNote *n;

    if (!src) {
        return;
    }
    n = g_new(VkNote, 1);
    n->stage = stage;
    n->src = src;
    g_hash_table_insert(m->noted, g_strdup(key), n);
}

static VkrShader *vk_prog(HD5770Vk *m, const char *key, unsigned stage,
                          char *src)
{
    gpointer v;
    VkrShader *s = NULL;
    char *log = NULL;

    if (g_hash_table_lookup_extended(m->progs, key, NULL, &v)) {
        g_free(src);
        return v == &vk_fail ? NULL : v;
    }
    if (src) {
        s = vkr_shader_new(m->r, stage, src, &log);
        if (!s) {
            m->failed++;
            if (!m->fail_log) {
                m->fail_log = g_strdup_printf("%s: %.400s", key, log);
            }
        }
        g_free(log);
        g_free(src);
        m->pub.built = true;
    }
    g_hash_table_insert(m->progs, g_strdup(key), s ? (gpointer)s : &vk_fail);
    return s;
}

static void vk_vb_free(gpointer p)
{
    VkVb *c = p;

    g_free(c->seen);
    g_free(c);
}

/* no page of VRAM [lo, hi) was written after generation gen */
static bool vk_unwritten(HD5770Vk *m, uint64_t lo, uint64_t hi, uint64_t gen)
{
    uint64_t p;

    if (!m->pub.written) {
        return false;
    }
    vk_pull(m, lo, hi);
    for (p = lo >> 12; p <= (hi - 1) >> 12; p++) {
        if (m->pgen[p] > gen) {
            return false;
        }
    }
    return true;
}

/*
 * a vertex buffer's bytes in the words; their byte offset there. Those
 * of VRAM are kept while the write log leaves their pages alone.
 */
static uint32_t vk_vb(HD5770Vk *m, const HD5770GpuSurf *b, uint32_t len)
{
    uint32_t off;
    uint8_t *p;

    if (!b->space) {
        VkVb *c = g_hash_table_lookup(m->vbs, &b->voff);

        if (b->voff >= m->vram_size) {
            len = 0;
        } else {
            len = MIN(len, m->vram_size - b->voff);
        }
        if (c && c->len >= len &&
            (c->epoch == m->epoch || !len ||
             vk_unwritten(m, c->voff, c->voff + c->len, c->gen) ||
             !memcmp(c->seen, m->vram + c->voff, c->len))) {
            /* written or not, the bytes are the ones copied */
            c->gen = c->epoch == m->epoch ? c->gen : m->gen;
            c->epoch = m->epoch;
            return c->off;
        }
        p = vkr_keep(m->r, len + 16, &off);
        if (!p) {
            vkr_keep_reset(m->r);
            g_hash_table_remove_all(m->vbs);
            c = NULL;
            p = vkr_keep(m->r, len + 16, &off);
        }
        if (!c) {
            c = g_new0(VkVb, 1);
            c->voff = b->voff;
            g_hash_table_insert(m->vbs, &c->voff, c);
        }
        if (len) {
            vk_pull(m, b->voff, b->voff + len);
        }
        c->gen = m->gen;
        c->epoch = m->epoch;
        c->len = len;
        c->off = off * 4;
        c->seen = g_realloc(c->seen, len);
        memcpy(c->seen, m->vram + b->voff, len);
        memcpy(p, c->seen, len);
        m->vb_bytes += len;
        return c->off;
    }
    p = vkr_data(m->r, len + 16, &off);
    m->vb_bytes += len;
    {
        uint64_t o = 0;

        while (o < len) {
            uint64_t q = o + b->pg0;
            size_t k = MIN(4096 - (q & 4095), len - o);

            if ((q >> 12) >= b->npages ||
                b->pages[q >> 12] > m->ram_size - 4096) {
                memset(p + o, 0, len - o);
                break;
            }
            memcpy(p + o, m->ram + b->pages[q >> 12] + (q & 4095), k);
            o += k;
        }
    }
    return off * 4;
}

static bool vk_draw_locked(HD5770Vk *m, const HD5770GpuDraw *d)
{
    const HD5770IR *ir = d->ir;
    const HD5770GpuVs *v = d->vs;
    HD5770NatKey k;
    VkrDraw q = { 0 };
    VkPair *pr, sk;
    struct {
        const HD5770GpuSurf *surf;
        VkSurf *e;
        unsigned hf, out, mask, cb, comps;
        uint32_t src[4], blend;
    } tg[8];
    VkSurf *zs = NULL, *ts[HD5770_GPU_TEX], *at[9];
    unsigned hf[HD5770_GPU_TEX], lvs[HD5770_GPU_TEX];
    unsigned nrt, nt = 0, nat = 0, i, j, c, mask = 0;
    bool hasz = d->db && d->zfmt;
    int64_t tp = get_clock(), tn;
#define VK_PART(i) (tn = get_clock(), m->part_ns[i] += tn - tp, tp = tn)
    uint32_t vw = UINT32_MAX, vh = UINT32_MAX, padded = 0, flat, *u, uo, vo;
    int sc[4] = { d->sc[0], d->sc[1], d->sc[2], d->sc[3] };
    int x0, y0, x1, y1, bx0 = INT_MAX, by0 = INT_MAX, bx1 = -1, by1 = -1;
    bool color, zclear, zwrite;
    GString *key;
    size_t need;

    m->seq++;
    memset(&k, 0, sizeof(k));
    if (d->resolve) {
        return vk_decline(m, VD_RESOLVE, HD5770_WHY_BK_NATIVE_STATE);
    }
    if (!ir || ir->vs) {
        return vk_decline(m, VD_PROG_FORM, HD5770_WHY_BK_PROG);
    }
    if (d->db && d->has_s) {
        return vk_decline(m, VD_STENCIL, HD5770_WHY_BK_NATIVE_STATE);
    }
    if (d->ninputs > 32 || ir->nres != d->ntex) {
        return vk_decline(m, VD_PROG_INPUTS, HD5770_WHY_BK_PROG);
    }
    /* vertex fetches: buffers of 8 and 32 bit elements */
    for (i = 0; i < ir->nfetches; i++) {
        const HD5770IRFetch *f = &ir->fetches[i];
        unsigned bid = (f->w[0] >> 8) & 0xff, fmt;

        if (f->kind != HD5770_IRF_VTX) {
            continue;
        }
        for (j = 0; j < d->nvb && d->vb[j].id != bid; j++) {
            /* search */
        }
        if (j == d->nvb || ir->cf) {
            return vk_decline(m, VD_PROG_INPUTS, HD5770_WHY_BK_PROG);
        }
        fmt = (f->w[1] >> 21) & 1 ? (d->vb[j].res[2] >> 20) & 0x3f
                                  : (f->w[1] >> 22) & 0x3f;
        if (fmt != HD5770_FMT_8 && fmt != HD5770_FMT_32) {
            return vk_decline(m, VD_PROG_VTX, HD5770_WHY_BK_PROG);
        }
    }
    k.nvb = d->nvb;
    for (i = 0; i < d->nvb; i++) {
        k.vb[i].id = d->vb[i].id;
        k.vb[i].fmt = (d->vb[i].res[2] >> 20) & 0x3f;
        k.vb[i].num = (d->vb[i].res[2] >> 26) & 3;
        k.vb[i].sgn = (d->vb[i].res[2] >> 28) & 1;
    }
    color = d->nmrt || !(d->db && !d->wmask);
    nrt = d->nmrt ? d->nmrt : 1;
    for (i = 0; i < nrt && color; i++) {
        const HD5770GpuRt *r = &d->mrt[i];
        unsigned why;

        if (d->nmrt) {
            tg[nt].surf = &r->surf;
            tg[nt].cb = r->cb;
            memcpy(tg[nt].src, r->src, sizeof(tg[nt].src));
            tg[nt].comps = r->comps;
            tg[nt].blend = (r->blend & (1u << 30)) && r->blend_ok ? r->blend
                                                                  : 0;
            why = vk_rt(&r->surf.s, r->fmt, r->num, r->comps, r->is_float,
                        r->unorm8, tg[nt].src,
                        (ir->cmask >> r->cb) & 1 ? r->wmask : 0,
                        &tg[nt].blend, &tg[nt].hf, &tg[nt].out, &tg[nt].mask);
        } else {
            tg[nt].surf = &d->rt;
            tg[nt].cb = 0;
            memcpy(tg[nt].src, d->rt_src, sizeof(tg[nt].src));
            tg[nt].comps = d->rt_comps;
            tg[nt].blend = (d->blend & (1u << 30)) && d->rt_blend ? d->blend
                                                                  : 0;
            why = vk_rt(&d->rt.s, d->rt_fmt, d->rt_num, d->rt_comps,
                        d->rt_float, d->rt_unorm8, tg[nt].src, d->wmask,
                        &tg[nt].blend, &tg[nt].hf, &tg[nt].out, &tg[nt].mask);
        }
        if (why != VD_TAKEN) {
            m->rtf[(d->nmrt ? r->fmt : d->rt_fmt) & 63] += why == VD_RT_FMT &&
                                                           !m->dry;
            return vk_decline(m, why, why == VD_BLEND_ORDER
                                      ? HD5770_WHY_BK_NATIVE_STATE
                                      : HD5770_WHY_BK_SURF);
        }
        if (!vkr_format_ok(m->r, tg[nt].hf, true)) {
            m->rtf[(d->nmrt ? r->fmt : d->rt_fmt) & 63] += !m->dry;
            return vk_decline(m, VD_RT_FMT, HD5770_WHY_BK_SURF);
        }
        if (tg[nt].surf->space) {
            return vk_decline(m, VD_RT_RAM, HD5770_WHY_BK_SURF);
        }
        mask |= tg[nt].mask;
        nt += tg[nt].mask != 0;
    }
    if (!mask && d->db) {
        nt = 0;
    }
    if (!nt) {
        memset(&tg[0], 0, sizeof(tg[0]));
        tg[0].comps = 4;
        for (i = 0; i < 4; i++) {
            tg[0].src[i] = i;
        }
    }
    k.out = tg[0].out;
    k.rt_comps = tg[0].comps;
    k.ncol = d->nmrt ? nt : 0;
    for (i = 0; i < k.ncol; i++) {
        k.col[i].cb = tg[i].cb;
        k.col[i].out = tg[i].out;
        k.col[i].comps = tg[i].comps;
        for (j = 0; j < 4; j++) {
            k.col[i].src[j] = tg[i].src[j];
        }
    }
    if (d->cliprule != 0xffff) {
        static const uint16_t one[4] = { 0xaaaa, 0xcccc, 0xf0f0, 0xff00 };

        for (i = 0; i < 4 && d->cliprule != one[i]; i++) {
            /* search */
        }
        if (i < 4) {
            sc[0] = MAX(sc[0], d->cr[i][0]);
            sc[1] = MAX(sc[1], d->cr[i][1]);
            sc[2] = MIN(sc[2], d->cr[i][2]);
            sc[3] = MIN(sc[3], d->cr[i][3]);
        } else {
            k.clip = true;
        }
    }
    for (i = 0; i < d->ntex; i++) {
        const HD5770GpuTex *t = &d->tex[i];

        k.dim[i] = t->dim;
        k.nsl[i] = t->dim == 1 ? 1 : MAX(t->slices, 1u);
        if (t->dim < 1 || t->dim == 4 || t->dim > 5 || t->slices > 256 ||
            (t->dim == 1 && t->slices > 1) ||
            ((k.nsl[i] > 1 || t->dim == 2) && t->surf.space)) {
            /* 1D */
            m->dims[MIN(t->dim, 7)] += !m->dry;
            return vk_decline(m, VD_TEX_KIND, HD5770_WHY_BK_SURF);
        }
        hf[i] = vk_tex_fmt(t);
        if (hf[i] == VKR_FMT_N) {
            m->txf[t->fmt & 63] += !m->dry;
            return vk_decline(m, VD_TEX_FMT, HD5770_WHY_BK_SURF);
        }
        lvs[i] = t->mip && !t->surf.space
                 ? MIN(t->last_lv, HD5770_GPU_LEVELS - 1) + 1 : 1;
        /* no more levels than the size has */
        lvs[i] = MIN(lvs[i], 32u - clz32(MAX(MAX(t->w, t->h), 1u)));
    }
    for (i = 0; i < d->ntri; i++) {
        const HD5770GpuTri *t = &d->tri[i];

        for (j = 0; j < 3 && d->persp; j++) {
            if (!(t->w[j] > 0.0f) || isinf(t->w[j])) {
                return vk_decline(m, VD_W, HD5770_WHY_BK_NATIVE_STATE);
            }
        }
        bx0 = MIN(bx0, t->x0);
        by0 = MIN(by0, t->y0);
        bx1 = MAX(bx1, t->x1 + 1);
        by1 = MAX(by1, t->y1 + 1);
    }
    if (v) {
        bx0 = sc[0];
        by0 = sc[1];
        bx1 = sc[2];
        by1 = sc[3];
        if (d->rctl & 1 && d->db) {
            return vk_decline(m, VD_VS, HD5770_WHY_BK_NATIVE_STATE);
        }
    }
    VK_PART(VP_STATE);
    for (i = 0; i < nt && !m->dry; i++) {
        tg[i].e = vk_surf(m, tg[i].surf, tg[i].hf, 0, 1, NULL, 1);
        if (!tg[i].e) {
            return vk_decline(m, VD_TEX_SURF, HD5770_WHY_BK_SURF);
        }
        at[nat++] = tg[i].e;
    }
    if (hasz && d->zs.space) {
        return vk_decline(m, VD_DB_SURF, HD5770_WHY_BK_SURF);
    }
    if (hasz && !m->dry) {
        zs = vk_surf(m, &d->zs, VKR_FMT_D32F, d->zfmt, 1, NULL, 1);
        if (!zs) {
            return vk_decline(m, VD_DB_SURF, HD5770_WHY_BK_SURF);
        }
        at[nat++] = zs;
    }
    for (i = 0; i < nat; i++) {
        vw = MIN(vw, at[i]->lv[0].s.pitch);
        vh = MIN(vh, at[i]->lv[0].s.height);
        for (j = 0; j < i; j++) {
            if (at[i] == at[j] ||
                (at[i]->lo < at[j]->hi && at[j]->lo < at[i]->hi)) {
                return vk_decline(m, VD_DB_SURF, HD5770_WHY_BK_SURF);
            }
        }
    }
    for (i = 0; i < nt; i++) {
        vh = MIN(vh, tg[i].surf->s.height);
    }
    if (zs) {
        vh = MIN(vh, d->zs.s.height);
    }
    x0 = MAX(MAX(sc[0], bx0), 0);
    y0 = MAX(MAX(sc[1], by0), 0);
    x1 = MIN(MIN(sc[2], bx1), (int)vw);
    y1 = MIN(MIN(sc[3], by1), (int)vh);
    if (!m->dry &&
        ((!d->ntri && !(v && v->ntidx && v->cull != 3)) || !nat || x0 >= x1 ||
         y0 >= y1 || (!mask && !d->db))) {
        m->nd[VD_EMPTY]++;
        return true;
    }
    for (i = 0; i < d->ntex; i++) {
        const HD5770GpuTex *t = &d->tex[i];
        unsigned bk = hf[i] >= VKR_FMT_BC1 && hf[i] <= VKR_FMT_BC5S ? 4 : 1;
        VkSurf *e = NULL;

        if (m->dry) {
            /* the image there is, or the one that would be made */
            bool lv = lvs[i] > 1 || k.nsl[i] > 1 || t->dim == 2;
            uint32_t iw = lv ? ROUND_UP(t->w, bk) : t->surf.s.pitch * bk;
            uint32_t ih = lv ? ROUND_UP(t->h, bk) : t->surf.s.height * bk;

            if (lv) {
                e = vk_find_tex(m, &t->surf, hf[i], bk, t, lvs[i]);
            } else {
                if (hf[i] == VKR_FMT_R32F && !t->surf.space) {
                    e = vk_find(m, &t->surf, VKR_FMT_D32F, 3);
                }
                e = e ? e : vk_find(m, &t->surf, hf[i], 0);
            }
            if (e) {
                iw = e->lw[0] * bk;
                ih = e->lh[0] * bk;
            }
            if (t->w != iw || t->h != ih) {
                padded |= 1u << i;
            }
            continue;
        }
        if (lvs[i] > 1 || k.nsl[i] > 1 || t->dim == 2) {
            e = vk_surf(m, &t->surf, hf[i], 0, bk, t, lvs[i]);
        } else {
            if (hf[i] == VKR_FMT_R32F && !t->surf.space) {
                /* a depth plane of floats: its own image is read */
                e = vk_find(m, &t->surf, VKR_FMT_D32F, 3);
                if (e) {
                    e->used = m->seq;
                }
            }
            if (!e) {
                e = vk_surf(m, &t->surf, hf[i], 0, bk, NULL, 1);
            }
        }
        if (!e || t->w > e->lw[0] * bk || t->h > e->lh[0] * bk) {
            return vk_decline(m, VD_TEX_SURF, HD5770_WHY_BK_SURF);
        }
        for (j = 0; j < nat; j++) {
            if (e == at[j] || (!e->lv[0].space && e->lo < at[j]->hi &&
                               at[j]->lo < e->hi)) {
                return vk_decline(m, VD_TEX_TARGET, HD5770_WHY_BK_SURF);
            }
        }
        ts[i] = e;
        if (t->w != e->lw[0] * bk || t->h != e->lh[0] * bk) {
            padded |= 1u << i;
        }
    }
    zclear = hasz && (d->rctl & 1);
    zwrite = hasz && (d->dctl & DC_Z) && (d->dctl & DC_Z_WRITE);
    k.zexp = hasz && !zclear && d->zexport && ir->zexport;
    k.zout = hasz && !zclear && (k.zexp ||
                               !(d->zmin <= 0.0f && d->zmax >= 1.0f));
    k.atest = d->alpha_test;
    k.ninputs = d->ninputs;
    k.pos_gpr = d->pos_gpr >= 0 && d->pos_gpr < 127 ? d->pos_gpr : -1;
    k.flat = d->flat;
    for (i = 0; i < 4; i++) {
        k.rt_src[i] = tg[0].src[i];
    }
    VK_PART(VP_IMAGES);
    memset(&sk, 0, sizeof(sk));
    sk.key.ps = ir->hash;
    sk.key.ps_n = ir->ninsts;
    sk.key.padded = padded;
    memcpy(&sk.key.k, &k, sizeof(k));
    if (v) {
        sk.key.has_vs = 1;
        sk.key.vs = v->ir->hash;
        sk.key.vs_n = v->ir->ninsts;
        memcpy(sk.key.src, v->src, k.ninputs * sizeof(int32_t));
    }
    pr = m->dry ? NULL : g_hash_table_lookup(m->pairs, &sk.key);
    if (pr) {
        q.fs = pr->fs;
        q.vs = pr->vsh;
        flat = pr->flat;
        goto staged;
    }
    flat = hd5770_glsl_native_flat(ir, &k);
    key = g_string_new(NULL);
    g_string_printf(key, "f%016" PRIx64 "/%u/%u/%d/%08x/%u%u%u%u/%u%u%u/%u%u%u/%x",
                    ir->hash, ir->ninsts, k.ninputs, k.pos_gpr, k.flat,
                    k.rt_src[0], k.rt_src[1], k.rt_src[2], k.rt_src[3], k.out,
                    k.rt_comps, k.clip, k.zout, k.zexp, k.atest, padded);
    for (i = 0; i < d->ntex; i++) {
        g_string_append_printf(key, ".%u.%u", k.dim[i], k.nsl[i]);
    }
    for (i = 0; i < k.nvb; i++) {
        g_string_append_printf(key, "/%u.%u.%u.%u", k.vb[i].id, k.vb[i].fmt,
                               k.vb[i].num, k.vb[i].sgn);
    }
    for (i = 0; i < k.ncol; i++) {
        g_string_append_printf(key, "/%u.%u.%u.%u%u%u%u", k.col[i].cb,
                               k.col[i].out, k.col[i].comps, k.col[i].src[0],
                               k.col[i].src[1], k.col[i].src[2],
                               k.col[i].src[3]);
    }
    if (m->dry) {
        VkJob job = { .fs = g_strdup(key->str) };

        vk_note(m, key->str, 1,
                vk_noted(m, key->str) ? NULL :
                hd5770_glsl_native_fs(ir, &k, padded, vkr_barycentric(m->r)));
        if (v) {
            g_string_printf(key, "v%016" PRIx64 "/%u/%u/%08x", v->ir->hash,
                            v->ir->ninsts, k.ninputs, flat);
            for (i = 0; i < k.ninputs; i++) {
                g_string_append_printf(key, ".%d", v->src[i]);
            }
            vk_note(m, key->str, 0,
                    vk_noted(m, key->str) ? NULL :
                    hd5770_glsl_native_vs(v->ir, k.ninputs, flat, v->src));
        } else {
            g_string_printf(key, "p%u/%08x", k.ninputs, flat);
            vk_note(m, key->str, 0,
                    vk_noted(m, key->str) ? NULL :
                    hd5770_glsl_native_pvs(k.ninputs, flat));
        }
        job.vs = g_string_free(key, false);
        job.d.ncolor = nt;
        for (i = 0; i < nt; i++) {
            job.d.fmt[i] = tg[i].hf;
            vk_blend(tg[i].blend, tg[i].mask, &job.d.blend[i]);
        }
        job.d.depth = hasz;
        g_array_append_val(m->jobs, job);
        return true;
    }
    q.fs = vk_prog(m, key->str, 1,
                   g_hash_table_contains(m->progs, key->str) ? NULL :
                   hd5770_glsl_native_fs(ir, &k, padded,
                                         vkr_barycentric(m->r)));
    if (!q.fs) {
        g_string_free(key, true);
        return vk_decline(m, VD_PROG_EMIT, HD5770_WHY_BK_PROG);
    }
    if (v) {
        g_string_printf(key, "v%016" PRIx64 "/%u/%u/%08x", v->ir->hash,
                        v->ir->ninsts, k.ninputs, flat);
        for (i = 0; i < k.ninputs; i++) {
            g_string_append_printf(key, ".%d", v->src[i]);
        }
        q.vs = vk_prog(m, key->str, 0,
                       g_hash_table_contains(m->progs, key->str) ? NULL :
                       hd5770_glsl_native_vs(v->ir, k.ninputs, flat, v->src));
    } else {
        g_string_printf(key, "p%u/%08x", k.ninputs, flat);
        q.vs = vk_prog(m, key->str, 0,
                       g_hash_table_contains(m->progs, key->str) ? NULL :
                       hd5770_glsl_native_pvs(k.ninputs, flat));
    }
    g_string_free(key, true);
    if (!q.vs) {
        return vk_decline(m, v ? VD_VS : VD_PROG_BUILD, HD5770_WHY_BK_PROG);
    }
    pr = g_memdup2(&sk, sizeof(sk));
    pr->fs = q.fs;
    pr->vsh = q.vs;
    pr->flat = flat;
    g_hash_table_insert(m->pairs, &pr->key, pr);
staged:
    VK_PART(VP_STAGES);
    for (i = 0; i < d->ntex; i++) {
        vk_ready(m, ts[i]);
    }
    for (i = 0; i < nat; i++) {
        vk_ready(m, at[i]);
    }
    need = (HD5770_NU_WORDS + HD5770_NV_WORDS + ir->nconsts + 1024 + 64) * 4 +
           (size_t)d->ntri * 3 * (4 + 4 * d->ninputs) * 4 + 4096;
    for (i = 0; i < d->nvb; i++) {
        need += d->vb[i].len + 64;
    }
    if (v) {
        need += (v->ir->nconsts + (size_t)v->ntidx) * 4 + 4096;
        for (i = 0; i < v->nvb; i++) {
            need += (size_t)v->vb[i].res[1] + 64;
        }
    }
    if (need > (64u << 20)) {
        return vk_decline(m, VD_ROOM, HD5770_WHY_BK_ROOM);
    }
    VK_PART(VP_READY);
    vkr_data_room(m->r, need);
    u = vkr_data(m->r, (HD5770_NU_WORDS + ir->nconsts + 1) * 4, &uo);
    memset(u, 0, HD5770_NU_WORDS * 4);
#define UF(i, f) do { float f_ = (f); memcpy(&u[i], &f_, 4); } while (0)
    UF(HD5770_NU_AREF, d->alpha_ref);
    u[HD5770_NU_AFUNC] = d->alpha_test ? d->alpha_func : 7;
    UF(HD5770_NU_PC, d->pix_center);
    u[HD5770_NU_PERSP] = d->persp;
    u[HD5770_NU_CLIPRULE] = d->cliprule;
    UF(HD5770_NU_VPX, 2.0f / vw);
    UF(HD5770_NU_VPY, 2.0f / vh);
    UF(HD5770_NU_ZMIN, d->zmin);
    UF(HD5770_NU_ZMAX, d->zmax);
    u[HD5770_NU_KV] = uo + HD5770_NU_WORDS;
    memcpy(&u[HD5770_NU_CR], d->cr, 16 * 4);
    u[HD5770_NU_BCONST] = d->bool_const;
    memcpy(&u[HD5770_NU_LCONST], d->loop_const, 32 * 4);
    if (ir->nconsts) {
        memcpy(u + HD5770_NU_WORDS, d->kv, ir->nconsts * 4);
    }
    for (i = 0; i < d->ntex; i++) {
        const HD5770GpuTex *t = &d->tex[i];
        uint32_t *tu = u + HD5770_NU_TEX + HD5770_NU_TEX_WORDS * i;
        unsigned bk = ts[i]->blk;

        UF(HD5770_NU_TEX + HD5770_NU_TEX_WORDS * i, (float)t->w);
        UF(HD5770_NU_TEX + HD5770_NU_TEX_WORDS * i + 1, (float)t->h);
        for (c = 0; c < 4; c++) {
            tu[2 + c] = hf[i] == VKR_FMT_ABGR4 && t->dst_sel[c] < 4
                        ? 3 - t->dst_sel[c]
                        : hf[i] == VKR_FMT_RGB565 && t->dst_sel[c] < 3
                        ? 2 - t->dst_sel[c] : t->dst_sel[c];
        }
        tu[6] = t->base_lv;
        tu[7] = t->levels;
        tu[8] = t->depth;
        tu[9] = (t->samp[0] >> 22) & 7;
        tu[10] = !t->is_float && t->num == HD5770_NUM_NORM &&
                 !(t->sgn & 1) && !t->bc && t->bits[0] && t->bits[0] < 32
                 ? (1u << t->bits[0]) - 1 : 0;
        UF(HD5770_NU_TEX + HD5770_NU_TEX_WORDS * i + 11,
           (float)t->w / (ts[i]->lw[0] * bk));
        UF(HD5770_NU_TEX + HD5770_NU_TEX_WORDS * i + 12,
           (float)t->h / (ts[i]->lh[0] * bk));
        tu[13] = t->samp[0] & 7;
        tu[14] = (t->samp[0] >> 3) & 7;
        q.tex[i].img = ts[i]->img;
        vk_sampler(t, lvs[i] > 1, lvs[i] - 1, &q.tex[i].s);
    }
    for (i = 0; i < d->nvb; i++) {
        uint32_t off;
        uint8_t *p = vkr_data(m->r, d->vb[i].len + 8, &off);

        memcpy(p, d->vb[i].data, d->vb[i].len);
        memset(p + d->vb[i].len, 0, 8);
        u[HD5770_NU_VB + 3 * i] = (d->vb[i].res[2] >> 8) & 0x7ff;
        u[HD5770_NU_VB + 3 * i + 1] = d->vb[i].len;
        u[HD5770_NU_VB + 3 * i + 2] = off * 4;
    }
    q.ntex = d->ntex;
    q.pc[0] = uo;
    if (v) {
        uint32_t *n, *ix;
        VkTix *ti;
        float f;

        /* the constant file stays where it is while it is the same */
        if (m->cf_gen != vkr_data_gen(m->r) ||
            memcmp(m->cf_seen, v->cfile, sizeof(m->cf_seen))) {
            memcpy(vkr_data(m->r, sizeof(m->cf_seen), &m->cf_off), v->cfile,
                   sizeof(m->cf_seen));
            memcpy(m->cf_seen, v->cfile, sizeof(m->cf_seen));
            m->cf_gen = vkr_data_gen(m->r);
        }
        n = vkr_data(m->r, (HD5770_NV_WORDS + v->ir->nconsts) * 4, &vo);
        memset(n, 0, HD5770_NV_WORDS * 4);
        n[0] = v->nslot;
        n[1] = v->ninst;
        n[2] = v->np;
        n[4] = vo + HD5770_NV_WORDS;
        memcpy(n + HD5770_NV_WORDS, v->kv, v->ir->nconsts * 4);
        n[5] = m->cf_off;
        n[6] = v->nvb;
        n[7] = v->base_vtx;
        n[8] = v->start_inst;
        for (i = 0; i < v->nvb; i++) {
            uint32_t *b = n + 24 + 16 * i;

            b[0] = v->vb[i].id;
            memcpy(b + 4, v->vb[i].res, sizeof(v->vb[i].res));
            b[12] = vk_vb(m, &v->vb[i].buf, v->vb[i].res[1] + 1);
        }
        n[HD5770_NV_X] = v->vte;
        n[HD5770_NV_X + 1] = v->sc_mode;
        n[HD5770_NV_X + 2] = v->window_offset;
        n[HD5770_NV_X + 3] = (v->clip_on ? 1 : 0) | (v->clip_dx ? 2 : 0) |
                             (v->clip_near ? 4 : 0) | (v->clip_far ? 8 : 0);
        memcpy(n + HD5770_NV_X + 4, v->vport, 6 * 4);
        f = 2.0f / vw;
        memcpy(n + HD5770_NV_X + 10, &f, 4);
        f = 2.0f / vh;
        memcpy(n + HD5770_NV_X + 11, &f, 4);
        memcpy(n + HD5770_NV_X + 12, &v->pix_center, 4);
        memcpy(n + HD5770_NV_X + 13, &v->zoff_const, 4);
        memcpy(n + HD5770_NV_DEF, v->def, 128 * 4);
        /* triangles with a name stay where they are in the words */
        ti = &m->tix[v->tidx_id % ARRAY_SIZE(m->tix)];
        if (v->tidx_id && ti->id == v->tidx_id &&
            ti->gen == vkr_data_gen(m->r)) {
            q.index = ti->off;
        } else {
            ix = vkr_data(m->r, (size_t)v->ntidx * 4, &q.index);
            memcpy(ix, v->tidx, (size_t)v->ntidx * 4);
            ti->id = v->tidx_id;
            ti->off = q.index;
            ti->gen = vkr_data_gen(m->r);
        }
        q.nindex = v->ntidx;
        q.ninst = v->ninst;
        q.pc[1] = vo;
        q.cull = v->cull;
        q.front_cw = v->face_cw;
        q.zbias_slope = v->zoff_slope;
        m->vs_draws++;
        m->vs_tris += (uint64_t)v->ntidx / 3 * v->ninst;
    } else {
        size_t vstride = 4 + 4 * (size_t)d->ninputs;
        float *vb = vkr_data(m->r, (size_t)d->ntri * 3 * vstride * 4,
                             &u[HD5770_NU_VTX]);

        for (i = 0; i < d->ntri; i++) {
            const HD5770GpuTri *t = &d->tri[i];
            const float *prm = d->prm + (size_t)i * d->ninputs * 12;

            for (j = 0; j < 3; j++) {
                float *p = vb + ((size_t)i * 3 + j) * vstride;

                p[0] = t->x[j];
                p[1] = t->y[j];
                p[2] = !zs ? 0.0f :
                       zclear ? fminf(fmaxf(d->zclear, d->zmin), d->zmax) :
                       t->z[j] == t->z[j] ? t->z[j] : 0.0f;
                p[3] = t->w[j];
                for (c = 0; c < d->ninputs; c++) {
                    memcpy(p + 4 + 4 * c,
                           prm + c * 12 + ((flat >> c) & 1 ? 0 : j * 4),
                           4 * sizeof(float));
                }
            }
        }
        q.nvert = d->ntri * 3;
        q.ninst = 1;
        q.front_cw = true;
        m->tris += d->ntri;
    }
#undef UF
    q.ncolor = nt;
    for (i = 0; i < nt; i++) {
        q.color[i] = tg[i].e->img;
        vk_blend(tg[i].blend, tg[i].mask, &q.blend[i]);
    }
    for (i = 0; i < 4; i++) {
        q.blend_const[i] = d->blend_const[tg[0].src[i] & 3];
    }
    q.depth = zs ? zs->img : NULL;
    q.ztest = zs && (d->dctl & DC_Z);
    q.zwrite = zwrite;
    q.zfunc = (d->dctl >> 4) & 7;
    q.width = vw;
    q.height = vh;
    q.scissor[0] = x0;
    q.scissor[1] = y0;
    q.scissor[2] = x1;
    q.scissor[3] = y1;
    VK_PART(VP_WORDS);
    vkr_draw(m->r, &q);
    VK_PART(VP_QUEUE);
#undef VK_PART
    m->unsent++;
    for (i = 0; i < nt; i++) {
        vk_drew(m, tg[i].e, x0, y0, x1, y1);
    }
    if (zwrite) {
        vk_drew(m, zs, x0, y0, x1, y1);
    }
    m->nd[VD_TAKEN]++;
    return true;
}

static bool vk_draw(void *opaque, const HD5770GpuDraw *d)
{
    HD5770Vk *m = opaque;
    int64_t t0 = get_clock();
    bool r = vk_draw_locked(m, d);

    m->draw_ns += get_clock() - t0;
    return r;
}

static void vk_job_clear(gpointer p)
{
    VkJob *j = p;

    g_free(j->fs);
    g_free(j->vs);
}

static void vk_note_free(gpointer p)
{
    VkNote *n = p;

    g_free(n->src);
    g_free(n);
}

/* the stages and pipelines noted are built, several at a time */
static void vk_build(HD5770Vk *m)
{
    unsigned n = g_hash_table_size(m->noted), i = 0, np = 0;
    g_autofree unsigned *stage = g_new(unsigned, n);
    g_autofree char **src = g_new(char *, n), **key = g_new(char *, n);
    g_autofree VkrShader **out = g_new0(VkrShader *, n);
    g_autofree VkrPipeDesc *pd = g_new(VkrPipeDesc, m->jobs->len);
    GHashTableIter it;
    gpointer kp, vp;

    g_hash_table_iter_init(&it, m->noted);
    while (g_hash_table_iter_next(&it, &kp, &vp)) {
        key[i] = kp;
        stage[i] = ((VkNote *)vp)->stage;
        src[i++] = ((VkNote *)vp)->src;
    }
    vkr_shaders_new(m->r, n, stage, src, out);
    for (i = 0; i < n; i++) {
        m->failed += !out[i];
        g_hash_table_insert(m->progs, g_strdup(key[i]),
                            out[i] ? (gpointer)out[i] : &vk_fail);
    }
    g_hash_table_remove_all(m->noted);
    for (i = 0; i < m->jobs->len; i++) {
        VkJob *j = &g_array_index(m->jobs, VkJob, i);
        gpointer fs = g_hash_table_lookup(m->progs, j->fs);
        gpointer vs = g_hash_table_lookup(m->progs, j->vs);

        if (fs && vs && fs != &vk_fail && vs != &vk_fail) {
            pd[np] = j->d;
            pd[np].fs = fs;
            pd[np++].vs = vs;
        }
    }
    vkr_pipes_ahead(m->r, pd, np);
    g_array_set_size(m->jobs, 0);
}

static void vk_prep(void *opaque, const HD5770GpuDraw *d)
{
    HD5770Vk *m = opaque;

    if (!d) {
        vk_build(m);
        return;
    }
    m->dry = true;
    vk_draw_locked(m, d);
    m->dry = false;
}

static void vk_epoch(HD5770Vk *m)
{
    m->epoch++;
    m->pub.kept = m->dirty->len != 0;
}

static void vk_sync(void *opaque)
{
    HD5770Vk *m = opaque;
    int64_t t0 = get_clock();

    vk_wait(m, m->issued);
    while (m->dirty->len) {
        vk_back_queue(m, g_ptr_array_index(m->dirty, 0), VB_SYNC);
    }
    vk_backs_now(m);
    vk_epoch(m);
    m->syncs++;
    m->sync_ns += get_clock() - t0;
}

static void vk_sync_range(void *opaque, uint64_t off, uint64_t len)
{
    HD5770Vk *m = opaque;
    int64_t t0 = get_clock();
    uint64_t seen = m->nback[VB_RANGE];

    vk_settle(m, NULL, off, off + len, VB_RANGE);
    if (seen != m->nback[VB_RANGE]) {
        vk_epoch(m);
    }
    m->ranges++;
    m->sync_ns += get_clock() - t0;
}

/* the retiring thread: FENCES above */
static void *vk_retire(void *opaque)
{
    HD5770Vk *m = opaque;
    VkTicket *t;
    int64_t t0, t1;
    unsigned i;

    rcu_register_thread();
    for (;;) {
        qemu_mutex_lock(&m->lock);
        while (!m->quit && g_queue_is_empty(&m->tickets)) {
            qemu_cond_wait(&m->work, &m->lock);
        }
        t = g_queue_pop_head(&m->tickets);
        qemu_mutex_unlock(&m->lock);
        if (!t) {
            break;
        }
        t0 = get_clock();
        vkr_mark_wait(m->r, t->mark);
        t1 = get_clock();
        for (i = 0; i < t->backs->len; i++) {
            vk_back_apply(m, &g_array_index(t->backs, VkBack, i), &m->rsc,
                          false);
        }
        vkr_mark_free(m->r, t->mark);
        qatomic_set(&m->rgpu_ns, m->rgpu_ns + t1 - t0);
        qatomic_set(&m->rcopy_ns, m->rcopy_ns + get_clock() - t1);
        qemu_mutex_lock(&m->lock);
        qatomic_set(&m->retired, t->id);
        qemu_cond_broadcast(&m->done);
        qemu_mutex_unlock(&m->lock);
        if (m->pub.retired) {
            m->pub.retired(m->pub.opaque, t->id);
        }
        g_array_free(t->backs, true);
        g_free(t);
    }
    rcu_unregister_thread();
    return NULL;
}

static uint64_t vk_fence(void *opaque)
{
    HD5770Vk *m = opaque;
    int64_t t0 = get_clock();
    uint64_t lo = qatomic_read(&m->pub.scan_lo);
    uint64_t hi = qatomic_read(&m->pub.scan_hi);
    unsigned i;

    for (i = 0; i < m->dirty->len; i++) {
        VkSurf *e = g_ptr_array_index(m->dirty, i);

        if ((e->lo < hi && lo < e->hi) || t0 - e->tdraw >= VK_LAZY_NS) {
            vk_back_queue(m, e, e->lo < hi && lo < e->hi ? VB_SCANOUT
                                                         : VB_IDLE);
            i--;
        }
    }
    if (m->backs->len) {
        VkTicket *t = g_new0(VkTicket, 1);

        t->id = m->issued + 1;
        for (i = 0; i < m->backs->len; i++) {
            VkSurf *e = g_array_index(m->backs, VkBack, i).e;

            e->flight = t->id;
            if (!e->flying) {
                e->flying = true;
                g_ptr_array_add(m->flying, e);
            }
        }
        t->backs = m->backs;
        m->backs = g_array_new(false, false, sizeof(VkBack));
        t->mark = vkr_mark(m->r);
        m->unsent = 0;
        qemu_mutex_lock(&m->lock);
        m->issued = t->id;
        g_queue_push_tail(&m->tickets, t);
        qemu_cond_signal(&m->work);
        qemu_mutex_unlock(&m->lock);
    } else if (m->unsent >= VK_UNSENT) {
        vkr_flush(m->r, false);
        m->unsent = 0;
    }
    vk_epoch(m);
    m->fences++;
    m->sync_ns += get_clock() - t0;
    return qatomic_read(&m->retired) < m->issued ? m->issued : 0;
}

static void vk_wrote(void *opaque, uint64_t off, uint64_t len)
{
    HD5770Vk *m = opaque;

    if (len) {
        /* an image drawn to elsewhere over these bytes is loaded again */
        vk_settle(m, NULL, off, off + len, VB_RANGE);
        vk_named(m, off, off + len);
    }
    m->epoch++;
}

static bool vk_ram(void *opaque, uint8_t *host, uint64_t size)
{
    HD5770Vk *m = opaque;

    m->ram = host;
    m->ram_size = size;
    return true;
}

static void vk_stats(void *opaque, GString *out)
{
    HD5770Vk *m = opaque;
    unsigned i;

    vkr_stats(m->r, out);
    g_string_append_printf(out,
        "vk: vertex stage draws %" PRIu64 " (%" PRIu64 " triangles), others %"
        PRIu64 " triangles; vertex buffer MB %" PRIu64 "; images %u, %" PRIu64
        " MB (%u newer than memory, %" PRIu64 " dropped)\n"
        "vk: loads %" PRIu64 " (%" PRIu64 " MB) fences %" PRIu64 " syncs %"
        PRIu64 " range syncs %" PRIu64 "; ms in draws %" PRId64
        " in syncs and fences %" PRId64 " (waiting for tickets %" PRId64
        "); tickets %" PRIu64 ", their thread ms for the GPU %" PRId64
        " copying %" PRId64 "\nvk: write-backs",
        m->vs_draws, m->vs_tris, m->tris, m->vb_bytes >> 20, m->surfs->len,
        vkr_image_bytes(m->r) >> 20, m->dirty->len, m->evicted, m->uploads, m->upload_bytes >> 20,
        m->fences, m->syncs, m->ranges, m->draw_ns / 1000000,
        m->sync_ns / 1000000, m->wait_ns / 1000000, m->issued,
        qatomic_read(&m->rgpu_ns) / 1000000,
        qatomic_read(&m->rcopy_ns) / 1000000);
    for (i = 0; i < VB_N; i++) {
        g_string_append_printf(out, " %s %" PRIu64 " (%" PRIu64 " MB);",
                               vb_name[i], m->nback[i],
                               m->back_bytes[i] >> 20);
    }
    g_string_append_printf(out, " parts copied %" PRIu64 ";", m->parts);
    g_string_append(out, "\nvk: ms in a draw's parts:");
    for (i = 0; i < VP_N; i++) {
        g_string_append_printf(out, " %s %" PRId64 ";", vp_name[i],
                               m->part_ns[i] / 1000000);
    }
    g_string_append(out, "\nvk: draws");
    for (i = 0; i < VD_N; i++) {
        if (m->nd[i]) {
            g_string_append_printf(out, " %s %" PRIu64 ";", vd_name[i],
                                   m->nd[i]);
        }
    }
    g_string_append_printf(out, "\nvk: texture kinds declined: 1D %" PRIu64
                           " 2D with slices %" PRIu64 " 3D %" PRIu64 " cube %"
                           PRIu64 " arrays %" PRIu64 "\n", m->dims[0],
                           m->dims[1], m->dims[2], m->dims[3],
                           m->dims[4] + m->dims[5]);
    if (m->last_surf[0]) {
        g_string_append_printf(out, "vk: last surface without an image: %s\n",
                               m->last_surf);
    }
    if (m->failed) {
        g_string_append_printf(out, "vk: stages not compiled %" PRIu64
                               ", the first: %s\n", m->failed, m->fail_log);
    }
    for (i = 0; i < 64; i++) {
        if (m->rtf[i] || m->txf[i]) {
            g_string_append_printf(out, "vk: format 0x%02x declined: target %"
                                   PRIu64 " texture %" PRIu64 "\n", i,
                                   m->rtf[i], m->txf[i]);
        }
    }
}

static void vk_close(void *opaque)
{
    HD5770Vk *m = opaque;
    unsigned i;

    vk_sync(m);
    qemu_mutex_lock(&m->lock);
    m->quit = true;
    qemu_cond_signal(&m->work);
    qemu_mutex_unlock(&m->lock);
    qemu_thread_join(&m->thread);
    qemu_mutex_destroy(&m->lock);
    qemu_cond_destroy(&m->work);
    qemu_cond_destroy(&m->done);
    for (i = 0; i < m->surfs->len; i++) {
        vkr_image_free(m->r, ((VkSurf *)g_ptr_array_index(m->surfs, i))->img);
    }
    g_ptr_array_free(m->surfs, true);
    g_ptr_array_free(m->dirty, true);
    g_ptr_array_free(m->flying, true);
    g_array_free(m->backs, true);
    g_hash_table_destroy(m->progs);
    g_hash_table_destroy(m->noted);
    g_hash_table_destroy(m->pairs);
    g_array_free(m->jobs, true);
    g_hash_table_destroy(m->vbs);
    vkr_close(m->r);
    g_free(m->pgen);
    g_free(m->pulled);
    g_free(m->fail_log);
    g_free(m->sc.tmp);
    g_free(m->sc.offs);
    g_free(m->rsc.tmp);
    g_free(m->rsc.offs);
    g_free(m);
}

static const HD5770GpuOps vk_ops = {
    .name = "vulkan",
    .min_pixels = 1024,
    .draw = vk_draw,
    .sync = vk_sync,
    .stats = vk_stats,
    .close = vk_close,
    .ram = vk_ram,
    .wrote = vk_wrote,
    .fence = vk_fence,
    .sync_range = vk_sync_range,
    .prep = vk_prep,
    .depth = true,
    .mipmaps = true,
    .resinfo = true,
    .full = true,
    .native = true,
};

HD5770Gpu *hd5770_vk_open(uint8_t *vram, uint64_t vram_size, const char **err)
{
    const char *base = g_getenv("LOCALAPPDATA");
    g_autofree char *dir = NULL;
    g_autofree uint32_t *tab = NULL;
    VkRaster *r;
    HD5770Vk *m;
    size_t n;

    if (base) {
        dir = g_build_filename(base, "qemu-x86-opencore", "hd5770-vk", NULL);
        g_mkdir_with_parents(dir, 0700);
    }
    r = vkr_open(dir, err);
    if (!r) {
        return NULL;
    }
    n = hd5770_glsl_tables(NULL, 0);
    tab = g_new0(uint32_t, n);
    hd5770_glsl_tables(tab, n);
    vkr_tables(r, tab, n);
    m = g_new0(HD5770Vk, 1);
    m->r = r;
    m->vram = vram;
    m->vram_size = vram_size;
    m->surfs = g_ptr_array_new_with_free_func(vk_surf_free);
    m->dirty = g_ptr_array_new();
    m->flying = g_ptr_array_new();
    m->backs = g_array_new(false, false, sizeof(VkBack));
    g_queue_init(&m->tickets);
    qemu_mutex_init(&m->lock);
    qemu_cond_init(&m->work);
    qemu_cond_init(&m->done);
    m->progs = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);
    m->noted = g_hash_table_new_full(g_str_hash, g_str_equal, g_free,
                                     vk_note_free);
    m->pairs = g_hash_table_new_full(vk_pair_hash, vk_pair_equal, NULL,
                                     g_free);
    m->drawn = 1;
    m->jobs = g_array_new(false, false, sizeof(VkJob));
    g_array_set_clear_func(m->jobs, vk_job_clear);
    m->vbs = g_hash_table_new_full(g_int64_hash, g_int64_equal, NULL,
                                   vk_vb_free);
    m->pulled = g_new0(uint64_t, (vram_size >> 18) + 1);
    m->pgen = g_new0(uint64_t, (vram_size >> 12) + 1);
    m->epoch = 1;
    /* twice the card's memory: surfaces are held in several forms */
    m->limit = MAX(vram_size * 2, 512ull << 20);
    m->pub.ops = &vk_ops;
    m->pub.b = m;
    m->pub.vs_draw = true;
    m->pub.ram_copied = true;
    m->cf_gen = UINT32_MAX;
    qemu_thread_create(&m->thread, "hd5770-vk", vk_retire, m,
                       QEMU_THREAD_JOINABLE);
    return &m->pub;
}
