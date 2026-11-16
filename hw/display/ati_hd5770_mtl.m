/*
 * ATI Radeon HD 5770 (Juniper) 3D engine on the host GPU through Metal.
 *
 * VRAM is the GPU's memory: on an Apple GPU the host memory of the VRAM
 * region is wrapped as one shared buffer, the kernels read textures and
 * read-modify-write render targets in it at their tiled addresses, and
 * nothing is copied back. A draw is one compute dispatch per triangle,
 * each thread one pixel of the triangle's box running the program
 * translated by ati_hd5770_msl.c. Dispatches run concurrently; a memory
 * barrier goes before one that writes pixels an earlier one wrote (the
 * two triangles of one primitive never do), or bytes an earlier one read,
 * or that reads bytes an earlier one wrote. Per-draw data goes to a
 * staging buffer that is reused once sync() has waited for the GPU.
 * Programs are compiled on first use and kept by the hash of their
 * words.
 *
 * COMPLETION. Each committed command buffer is a record holding the
 * ranges its draws wrote. Its completed handler retires only a
 * contiguous prefix of completed records (the order handlers run in is
 * not the order the GPU finished in): it reports their ranges with
 * done() and then retires the fences (ati_hd5770_gpu.h) whose command
 * buffers are all retired. sync() waits for every record and reports
 * the ranges itself. rlock keeps reports and retirements in order; flock
 * guards the record and fence queues. Neither is held while waiting for
 * the GPU.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/timer.h"
#include "qemu/atomic.h"
#import <Metal/Metal.h>
#include "ati_hd5770_gpu.h"

#define MTL_STAGING     (64 * 1024 * 1024)
#define MTL_ALIGN       256
#define MTL_COMMIT_AT   256     /* draws per command buffer */
#define MTL_TILE        64      /* threads per tile */
#define MTL_ACC         128     /* accesses tracked between barriers */

/* the layouts of hd5770_msl_lib */
typedef struct GSurf {
    uint32_t voff, mode, bpe, row_bytes, mt_per_row, tile_bytes, mt_bytes;
    uint32_t pipes, banks, bankw, bankh, pib;
    uint32_t mt_w_sh, mt_h_sh, pipes_sh, bankw_sh, bankh_sh, pib_sh, hi_sh;
    uint32_t fast;
    uint32_t space, pt_off, pg0, split_sh, split_stride;
    uint32_t eoff[64];
} GSurf;

typedef struct GTex {
    GSurf s;
    uint32_t fmt, num, w, h, dim, sgn, comps, is_float, unorm8, samp0;
    uint32_t dst_sel[4];
    uint32_t bits[4];
    float border[4];
    uint32_t mip, last_lv, mip_filter, lev_off;
    float min_lod, max_lod, lod_bias;
    uint32_t lw[16], lh[16];
    uint32_t base_lv, levels, depth, pad;
    uint32_t bc, slices;
    uint32_t sb[16];
} GTex;

typedef struct GVb {
    uint32_t id, off, len, pad;
    uint32_t res[8];
} GVb;

typedef struct GDraw {
    GSurf rt;
    uint32_t rt_fmt, rt_num, rt_comps, rt_unorm8, rt_float, rt_blend;
    uint32_t wmask, blend;
    uint32_t rt_bits[4];
    uint32_t rt_src[4];
    float blend_const[4];
    uint32_t alpha_test, alpha_func;
    float alpha_ref;
    uint32_t cliprule;
    int32_t cr[16];
    float pix_center;
    uint32_t persp;
    int32_t pos_gpr;
    uint32_t ninputs, flat, cfile_off, ntex, nvb;
    GVb vb[8];
    GSurf zs, ss;
    uint32_t db, zfmt, has_s, dctl, rctl, srm0, srm1, sclear;
    float zclear, zmin, zmax;
    uint32_t nmrt, mrt_off, zexp, sexp, bconst, cb;
    uint32_t lconst[32];
    uint32_t resolve, rs_same, rs_fmt, rs_num, rs_comps;
    uint32_t rs_src[4];
    GSurf rs;
    uint32_t tri_off, prm_off, prm_st, pad2;
} GDraw;

typedef struct GTri {
    float x[3], y[3], w[3];
    float area;
    int32_t x0, y0, x1, y1;
    float z[3];
    uint32_t front;
} GTri;

#define MTL_TRI_SLOT    128     /* staging bytes per GTri */

QEMU_BUILD_BUG_ON(sizeof(GSurf) != 4 * 89);
QEMU_BUILD_BUG_ON(sizeof(GTex) != 4 * 172);
QEMU_BUILD_BUG_ON(sizeof(GDraw) != 4 * 562);
QEMU_BUILD_BUG_ON(sizeof(GTri) != 4 * 18);

/* a committed command buffer */
typedef struct MtlCb {
    uint64_t seq;
    id<MTLCommandBuffer> cb;
    GArray *ranges, *ram_ranges;
    bool done;                  /* its completed handler ran */
} MtlCb;

/* a fence: retired once command buffers up to cbseq are */
typedef struct MtlFence {
    uint64_t ticket, cbseq;
} MtlFence;

/* a pixel box one dispatch wrote, or bytes it read */
typedef struct MtlAcc {
    uint64_t lo, hi;            /* VRAM bytes */
    uint64_t voff;
    uint32_t mode, row_bytes, bpe;
    int32_t x0, y0, x1, y1;
    uint64_t draw;
    uint32_t prim;
} MtlAcc;

typedef struct MtlProg {
    id<MTLComputePipelineState> pso;    /* valid once ready */
    int state;                          /* atomic: MTL_P_* */
} MtlProg;

enum { MTL_P_BUILDING, MTL_P_READY, MTL_P_FAILED };

enum {
    MTL_NO_PROG, MTL_NO_ROOM, MTL_NO_SURF, MTL_BUILDING, MTL_NO_REASONS,
};

typedef struct HD5770Mtl {
    HD5770Gpu pub;
    id<MTLDevice> dev;
    id<MTLCommandQueue> q;
    id<MTLBuffer> vram, st, ram;
    uint8_t *vram_host;
    uint64_t vram_size;
    id<MTLCommandBuffer> cb, last;
    id<MTLComputeCommandEncoder> enc;
    size_t st_used;
    unsigned pending;
    GHashTable *progs;
    GArray *ranges;             /* uint64_t pairs, VRAM written */
    GArray *ram_ranges;         /* the same in guest RAM */
    MtlAcc wr[MTL_ACC], rd[MTL_ACC];
    unsigned nwr, nrd;
    uint64_t barriers, dispatches, threads;
    /* the batch: tiles of consecutive triangles run by one dispatch */
    id<MTLComputePipelineState> bpso;
    GArray *tiles;              /* uint32_t x 8 per tile */
    uint64_t draws, tris, declined[MTL_NO_REASONS], syncs, commits;
    uint64_t compiles, compile_ns, wait_ns, encode_ns;
    uint64_t gpu_ns;            /* atomic: command buffers on the GPU */
    int building;               /* atomic: programs being compiled */
    /* COMPLETION above */
    GMutex flock, rlock;
    GQueue cbs;                 /* MtlCb, oldest first, not yet retired */
    GQueue fences;              /* MtlFence, oldest first */
    uint64_t cb_seq;            /* last committed */
    uint64_t cb_retired;        /* every record up to it retired */
    uint64_t tickets, fences_retired;
    int handlers;               /* atomic: completed handlers to come */
} HD5770Mtl;

static void mtl_exact_math(MTLCompileOptions *o)
{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#if defined(MAC_OS_VERSION_15_0) && \
    MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_VERSION_15_0
    if (@available(macOS 15.0, *)) {
        o.mathMode = MTLMathModeSafe;
        o.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    } else {
        o.fastMathEnabled = NO;
    }
#else
    o.fastMathEnabled = NO;
#endif
#pragma clang diagnostic pop
}

/*
 * The program of an IR, compiled in the background on first use: until
 * it is ready the device draws on the CPU.
 */
static MtlProg *mtl_prog(HD5770Mtl *m, const HD5770IR *ir, uint32_t mipmask)
{
    /* a program with mipmapped units is its own variant */
    uint64_t hash = ir->hash ^ (mipmask * 0x9e3779b97f4a7c15ull);
    MtlProg *p = g_hash_table_lookup(m->progs, &hash);
    char *src;

    if (p) {
        return p;
    }
    p = g_new0(MtlProg, 1);
    g_hash_table_insert(m->progs, g_memdup2(&hash, sizeof(hash)), p);
    src = hd5770_msl_program_mip(ir, mipmask);
    if (!src) {
        p->state = MTL_P_FAILED;
        return p;
    }
    m->compiles++;
    qatomic_inc(&m->building);
    @autoreleasepool {
        MTLCompileOptions *opt = [[MTLCompileOptions alloc] init];
        id<MTLDevice> dev = m->dev;
        int64_t t0 = get_clock();

        mtl_exact_math(opt);
        [dev newLibraryWithSource:[NSString stringWithUTF8String:src]
                          options:opt
                completionHandler:^(id<MTLLibrary> lib, NSError *e) {
            id<MTLFunction> fn = [lib newFunctionWithName:@"hd_main"];
            void (^done)(id<MTLComputePipelineState>, NSError *) =
                ^(id<MTLComputePipelineState> pso, NSError *pe) {
                NSError *r = pe ? pe : e;

                if (pso) {
                    p->pso = [pso retain];
                } else {
                    fprintf(stderr, "ati-hd5770: Metal program %016" PRIx64
                            " not built: %s\n", hash,
                            r ? [[r localizedDescription] UTF8String] : "?");
                }
                qatomic_add(&m->compile_ns, get_clock() - t0);
                qatomic_store_release(&p->state,
                                      pso ? MTL_P_READY : MTL_P_FAILED);
                qatomic_dec(&m->building);
            };

            if (fn) {
                /* the pipeline too is built off this (the compiler's) queue */
                [dev newComputePipelineStateWithFunction:fn
                                       completionHandler:done];
                [fn release];
            } else {
                done(nil, nil);
            }
        }];
        [opt release];
    }
    g_free(src);
    return p;
}

static size_t mtl_take(HD5770Mtl *m, size_t len)
{
    size_t o = ROUND_UP(m->st_used, MTL_ALIGN);

    m->st_used = o + len;
    return o;
}

/* the batch as one dispatch */
static void mtl_flush(HD5770Mtl *m)
{
    size_t o, len = m->tiles->len * sizeof(uint32_t);

    if (!len) {
        return;
    }
    o = mtl_take(m, len);
    memcpy((uint8_t *)m->st.contents + o, m->tiles->data, len);
    [m->enc setComputePipelineState:m->bpso];
    [m->enc setBuffer:m->vram offset:0 atIndex:0];
    [m->enc setBuffer:m->st offset:0 atIndex:1];
    [m->enc setBuffer:m->st offset:o atIndex:2];
    [m->enc setBuffer:m->ram ? m->ram : m->st offset:0 atIndex:3];
    [m->enc dispatchThreadgroups:MTLSizeMake(m->tiles->len / 8, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(MTL_TILE, 1, 1)];
    m->threads += (uint64_t)m->tiles->len / 8 * MTL_TILE;
    g_array_set_size(m->tiles, 0);
    m->dispatches++;
}

/* the ranges a retired record's draws wrote, reported; the record freed */
static void mtl_report(HD5770Mtl *m, MtlCb *r)
{
    guint i;

    for (i = 0; i + 1 < r->ranges->len; i += 2) {
        m->pub.done(m->pub.opaque, g_array_index(r->ranges, uint64_t, i),
                    g_array_index(r->ranges, uint64_t, i + 1));
    }
    for (i = 0; i + 1 < r->ram_ranges->len && m->pub.done_ram; i += 2) {
        m->pub.done_ram(m->pub.opaque,
                        g_array_index(r->ram_ranges, uint64_t, i),
                        g_array_index(r->ram_ranges, uint64_t, i + 1));
    }
    g_array_free(r->ranges, true);
    g_array_free(r->ram_ranges, true);
    [r->cb release];
    g_free(r);
}

/*
 * The completed prefix of the records, out of the queue (flock held):
 * a record counts as completed when its handler ran, or with `status`
 * when the command buffer says so.
 */
static GSList *mtl_take_done(HD5770Mtl *m, bool status)
{
    GSList *l = NULL;
    MtlCb *r;

    while ((r = g_queue_peek_head(&m->cbs)) &&
           (r->done || (status && r->cb.status >= MTLCommandBufferStatusCompleted))) {
        g_queue_pop_head(&m->cbs);
        m->cb_retired = r->seq;
        l = g_slist_prepend(l, r);
    }
    return g_slist_reverse(l);
}

/* a command buffer's completed handler: COMPLETION above */
static void mtl_retire(HD5770Mtl *m, uint64_t seq)
{
    GSList *done, *l, *tickets = NULL;
    MtlFence *f;
    GList *e;

    g_mutex_lock(&m->rlock);
    g_mutex_lock(&m->flock);
    for (e = m->cbs.head; e; e = e->next) {
        MtlCb *r = e->data;

        if (r->seq == seq) {
            r->done = true;
            break;
        }
    }
    if (!m->pub.retired) {
        /* no fences: sync() reports */
        g_mutex_unlock(&m->flock);
        g_mutex_unlock(&m->rlock);
        return;
    }
    done = mtl_take_done(m, false);
    while ((f = g_queue_peek_head(&m->fences)) && f->cbseq <= m->cb_retired) {
        g_queue_pop_head(&m->fences);
        tickets = g_slist_prepend(tickets, f);
    }
    tickets = g_slist_reverse(tickets);
    g_mutex_unlock(&m->flock);
    if (done || tickets) {
        if (m->pub.thread) {
            m->pub.thread(m->pub.opaque, true);
        }
        for (l = done; l; l = l->next) {
            mtl_report(m, l->data);
        }
        for (l = tickets; l; l = l->next) {
            f = l->data;
            m->pub.retired(m->pub.opaque, f->ticket);
            qatomic_set(&m->fences_retired, f->ticket);
            g_free(f);
        }
        if (m->pub.thread) {
            m->pub.thread(m->pub.opaque, false);
        }
    }
    g_slist_free(done);
    g_slist_free(tickets);
    g_mutex_unlock(&m->rlock);
}

static void mtl_end(HD5770Mtl *m)
{
    MtlCb *r;
    uint64_t seq;

    if (m->enc) {
        mtl_flush(m);
        [m->enc endEncoding];
        [m->enc release];
        m->enc = nil;
    }
    if (m->cb) {
        r = g_new0(MtlCb, 1);
        r->cb = [m->cb retain];
        r->ranges = m->ranges;
        r->ram_ranges = m->ram_ranges;
        m->ranges = g_array_new(false, false, sizeof(uint64_t));
        m->ram_ranges = g_array_new(false, false, sizeof(uint64_t));
        g_mutex_lock(&m->flock);
        r->seq = seq = ++m->cb_seq;
        g_queue_push_tail(&m->cbs, r);
        g_mutex_unlock(&m->flock);
        qatomic_inc(&m->handlers);
        [m->cb addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            qatomic_add(&m->gpu_ns,
                        (uint64_t)((cb.GPUEndTime - cb.GPUStartTime) * 1e9));
            mtl_retire(m, seq);
            qatomic_dec(&m->handlers);
        }];
        [m->cb commit];
        [m->last release];
        m->last = m->cb;
        m->cb = nil;
        m->commits++;
    }
    m->pending = 0;
}

static void mtl_sync(void *opaque)
{
    HD5770Mtl *m = opaque;
    GSList *done, *l;
    GList *e;
    int64_t t0;

    @autoreleasepool {
        mtl_end(m);
        t0 = get_clock();
        /* every record still queued, without the locks */
        g_mutex_lock(&m->flock);
        e = g_queue_peek_tail_link(&m->cbs);
        l = NULL;
        for (; e; e = e->prev) {
            l = g_slist_prepend(l, [((MtlCb *)e->data)->cb retain]);
        }
        g_mutex_unlock(&m->flock);
        for (done = l; done; done = done->next) {
            id<MTLCommandBuffer> c = done->data;

            [c waitUntilCompleted];
            [c release];
        }
        g_slist_free(l);
        m->wait_ns += get_clock() - t0;
        [m->last release];
        m->last = nil;
    }
    m->st_used = 0;
    m->syncs++;
    g_mutex_lock(&m->rlock);
    g_mutex_lock(&m->flock);
    done = mtl_take_done(m, true);
    g_mutex_unlock(&m->flock);
    for (l = done; l; l = l->next) {
        mtl_report(m, l->data);
    }
    g_slist_free(done);
    g_mutex_unlock(&m->rlock);
}

/* FENCES in ati_hd5770_gpu.h */
static uint64_t mtl_fence(void *opaque)
{
    HD5770Mtl *m = opaque;
    MtlFence *f;
    uint64_t t = 0;

    @autoreleasepool {
        mtl_end(m);
    }
    g_mutex_lock(&m->flock);
    if (m->cb_retired < m->cb_seq || !g_queue_is_empty(&m->fences)) {
        f = g_new(MtlFence, 1);
        f->ticket = t = ++m->tickets;
        f->cbseq = m->cb_seq;
        g_queue_push_tail(&m->fences, f);
    }
    g_mutex_unlock(&m->flock);
    return t;
}

static void mtl_surf(HD5770Mtl *m, GSurf *g, const HD5770GpuSurf *s)
{
    const HD5770Surf *f = &s->s;

    memset(g, 0, sizeof(*g));
    g->voff = s->voff;
    if (s->space) {
        size_t o = mtl_take(m, s->npages * 4);

        memcpy((uint8_t *)m->st.contents + o, s->pages, s->npages * 4);
        g->space = 1;
        g->pt_off = o;
        g->pg0 = s->pg0;
    }
    g->mode = f->mode;
    g->bpe = f->bpe;
    g->row_bytes = f->row_bytes;
    g->mt_per_row = f->mt_per_row;
    g->tile_bytes = f->tile_bytes;
    g->mt_bytes = f->mt_bytes;
    g->pipes = f->pipes;
    g->banks = f->banks;
    g->bankw = f->bankw;
    g->bankh = f->bankh;
    g->pib = f->pipe_interleave;
    g->mt_w_sh = f->mt_w_sh;
    g->mt_h_sh = f->mt_h_sh;
    g->pipes_sh = f->pipes_sh;
    g->bankw_sh = f->bankw_sh;
    g->bankh_sh = f->bankh_sh;
    g->pib_sh = f->pib_sh;
    g->hi_sh = f->hi_sh;
    g->fast = f->fast;
    if (f->split) {
        g->split_sh = f->split_sh;
        g->split_stride = f->split_stride;
    }
    if (f->mode >= HD5770_ARRAY_1D_TILED_THIN1) {
        memcpy(g->eoff, f->eoff, sizeof(g->eoff));
    }
}

static void mtl_range(GArray *a, uint64_t lo, uint64_t hi)
{
    guint n = a->len;

    if (n && g_array_index(a, uint64_t, n - 2) <= hi &&
        lo <= g_array_index(a, uint64_t, n - 1)) {
        g_array_index(a, uint64_t, n - 2) =
            MIN(lo, g_array_index(a, uint64_t, n - 2));
        g_array_index(a, uint64_t, n - 1) =
            MAX(hi, g_array_index(a, uint64_t, n - 1));
        return;
    }
    g_array_append_val(a, lo);
    g_array_append_val(a, hi);
}

static bool mtl_overlap(uint64_t a0, uint64_t a1, uint64_t b0, uint64_t b1)
{
    return a0 < b1 && b0 < a1;
}

/*
 * Hazard ranges: VRAM bytes; all of guest RAM as one range above them
 * (surfaces there are scattered pages), and a key per surface.
 */
#define MTL_RAM_RANGE   (1ull << 40)

static uint64_t mtl_key(const HD5770GpuSurf *s)
{
    return s->space ? MTL_RAM_RANGE + s->pages[0] : s->voff;
}

static uint64_t mtl_lo(const HD5770GpuSurf *s)
{
    return s->space ? MTL_RAM_RANGE : s->voff;
}

static uint64_t mtl_hi(const HD5770GpuSurf *s)
{
    return s->space ? MTL_RAM_RANGE + 1 : s->voff + hd5770_surf_size(&s->s);
}

/* the slices of level l of a texture */
static uint64_t mtl_tex_lvbytes(const HD5770GpuTex *t, unsigned l)
{
    uint32_t ns = t->dim == 2 ? MAX(t->slices >> l, 1u) : MAX(t->slices, 1u);

    return MAX(t->slice_bytes[l] * ns,
               hd5770_surf_size(l ? &t->lv[l].s : &t->surf.s));
}

/* the VRAM bytes a texture reads: level 0, and the mip chain after it */
static uint64_t mtl_tex_lo(const HD5770GpuTex *t)
{
    uint64_t lo = mtl_lo(&t->surf);
    unsigned l;

    for (l = 1; t->mip && l <= t->last_lv; l++) {
        lo = MIN(lo, mtl_lo(&t->lv[l]));
    }
    return lo;
}

static uint64_t mtl_tex_hi(const HD5770GpuTex *t)
{
    uint64_t hi = t->surf.space ? mtl_hi(&t->surf)
                                : t->surf.voff + mtl_tex_lvbytes(t, 0);
    unsigned l;

    for (l = 1; t->mip && l <= t->last_lv; l++) {
        hi = MAX(hi, t->lv[l].voff + mtl_tex_lvbytes(t, l));
    }
    return hi;
}

/* the surfaces a request writes: targets, depth plane, stencil plane */
static unsigned mtl_writes(const HD5770GpuDraw *d, const HD5770GpuSurf **ws)
{
    unsigned n = 0, i;

    ws[n++] = &d->rt;
    if (d->db && d->zfmt) {
        ws[n++] = &d->zs;
    }
    if (d->db && d->has_s) {
        ws[n++] = &d->ss;
    }
    for (i = 1; i < d->nmrt; i++) {
        ws[n++] = &d->mrt[i].surf;
    }
    return n;
}

/* the bytes rows y0..y1 of a written surface take */
static void mtl_rows(const HD5770GpuSurf *s, int32_t y0, int32_t y1,
                     uint64_t *lo, uint64_t *hi)
{
    if (s->space) {
        *lo = MTL_RAM_RANGE;
        *hi = MTL_RAM_RANGE + 1;
        return;
    }
    hd5770_surf_rows(&s->s, y0, y1 + 1, lo, hi);
    *lo = *lo - s->s.base + s->voff;
    *hi = *hi - s->s.base + s->voff;
}

/*
 * A barrier before a dispatch writing the pixel box of triangle t in the
 * request's written surfaces, if an earlier one since the last barrier
 * conflicts. reads: the draw's textures, checked once per draw (first).
 */
static void mtl_hazard(HD5770Mtl *m, const HD5770GpuDraw *d,
                       const HD5770GpuTri *t, bool first)
{
    const HD5770GpuSurf *ws[11];
    unsigned nw = mtl_writes(d, ws), i, j, k;
    bool bar = m->nwr + nw > MTL_ACC || m->nrd + d->ntex + 1 > MTL_ACC;
    uint64_t lo[11], hi[11];
    /*
     * The pixels the triangle can cover: centres within its vertices'
     * extent, widened by far more than the edge functions' rounding.
     */
    float pc = d->pix_center, mg = 1.0f / 16;
    int32_t x0 = ceilf(MIN(t->x[0], MIN(t->x[1], t->x[2])) - pc - mg);
    int32_t x1 = floorf(MAX(t->x[0], MAX(t->x[1], t->x[2])) - pc + mg);
    int32_t y0 = ceilf(MIN(t->y[0], MIN(t->y[1], t->y[2])) - pc - mg);
    int32_t y1 = floorf(MAX(t->y[0], MAX(t->y[1], t->y[2])) - pc + mg);

    for (j = 0; j < nw; j++) {
        mtl_rows(ws[j], t->y0, t->y1, &lo[j], &hi[j]);
    }
    for (i = 0; !bar && i < m->nwr; i++) {
        const MtlAcc *a = &m->wr[i];

        for (j = 0; !bar && j < nw; j++) {
            const HD5770GpuSurf *w = ws[j];

            if (a->voff == mtl_key(w) && a->mode == w->s.mode &&
                a->row_bytes == w->s.row_bytes && a->bpe == w->s.bpe) {
                bar = a->x0 <= x1 && x0 <= a->x1 && a->y0 <= y1 &&
                      y0 <= a->y1 &&
                      !(a->draw == m->draws && a->prim == t->prim);
            } else {
                bar = mtl_overlap(a->lo, a->hi, lo[j], hi[j]);
            }
        }
        for (k = 0; first && !bar && k < d->ntex; k++) {
            bar = mtl_overlap(a->lo, a->hi, mtl_tex_lo(&d->tex[k]),
                              mtl_tex_hi(&d->tex[k]));
        }
        if (first && !bar && d->resolve) {
            bar = mtl_overlap(a->lo, a->hi, mtl_lo(&d->rs), mtl_hi(&d->rs));
        }
    }
    for (i = 0; !bar && i < m->nrd; i++) {
        for (j = 0; !bar && j < nw; j++) {
            bar = mtl_overlap(m->rd[i].lo, m->rd[i].hi, lo[j], hi[j]);
        }
    }
    if (bar) {
        mtl_flush(m);
        [m->enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        m->nwr = m->nrd = 0;
        m->barriers++;
    }
    if (first || bar) {
        for (k = 0; k < d->ntex; k++) {
            m->rd[m->nrd].lo = mtl_tex_lo(&d->tex[k]);
            m->rd[m->nrd].hi = mtl_tex_hi(&d->tex[k]);
            m->nrd++;
        }
        if (d->resolve) {
            m->rd[m->nrd].lo = mtl_lo(&d->rs);
            m->rd[m->nrd].hi = mtl_hi(&d->rs);
            m->nrd++;
        }
    }
    for (j = 0; j < nw; j++) {
        m->wr[m->nwr] = (MtlAcc) {
            .lo = lo[j], .hi = hi[j], .voff = mtl_key(ws[j]),
            .mode = ws[j]->s.mode, .row_bytes = ws[j]->s.row_bytes,
            .bpe = ws[j]->s.bpe, .x0 = x0, .y0 = y0, .x1 = x1, .y1 = y1,
            .draw = m->draws, .prim = t->prim,
        };
        m->nwr++;
    }
}

/*
 * A draw of more than this many triangles runs by tiles of the screen:
 * the thread of a pixel draws the triangles over its tile in order, so
 * no barrier is needed inside the draw.
 */
#define MTL_BIN_MIN     2

static void mtl_binned(HD5770Mtl *m, const HD5770GpuDraw *d, GDraw *u,
                       size_t o_draw, size_t o_tex, size_t o_kv,
                       int *miny, int *maxy)
{
    size_t prm_tri = (size_t)d->ninputs * 12 * sizeof(float);
    size_t ps = ROUND_UP(MAX(prm_tri, 16), 16);
    uint8_t *st = m->st.contents;
    size_t o_tri = mtl_take(m, MTL_TRI_SLOT * d->ntri);
    size_t o_prm = mtl_take(m, ps * d->ntri), o_bin;
    int bx0 = INT_MAX, by0 = INT_MAX, bx1 = -1, by1 = -1, tx, ty, ntx, nty;
    uint32_t *cnt, *at, *bin, total = 0;
    HD5770GpuTri box;
    unsigned i;

    u->tri_off = o_tri;
    u->prm_off = o_prm;
    u->prm_st = ps;
    for (i = 0; i < d->ntri; i++) {
        const HD5770GpuTri *t = &d->tri[i];
        GTri *gt = (GTri *)(st + o_tri + MTL_TRI_SLOT * i);

        memcpy(gt->x, t->x, sizeof(gt->x));
        memcpy(gt->y, t->y, sizeof(gt->y));
        memcpy(gt->w, t->w, sizeof(gt->w));
        gt->area = t->area;
        gt->x0 = t->x0;
        gt->y0 = t->y0;
        gt->x1 = t->x1;
        gt->y1 = t->y1;
        memcpy(gt->z, t->z, sizeof(gt->z));
        gt->front = t->front;
        memcpy(st + o_prm + ps * i, d->prm + (size_t)i * d->ninputs * 12,
               prm_tri);
        if (t->x1 >= t->x0 && t->y1 >= t->y0) {
            bx0 = MIN(bx0, t->x0);
            by0 = MIN(by0, t->y0);
            bx1 = MAX(bx1, t->x1);
            by1 = MAX(by1, t->y1);
        }
    }
    m->tris += d->ntri;
    if (bx1 < bx0) {
        return;
    }
    /* the whole draw as one box for the hazards between dispatches */
    memset(&box, 0, sizeof(box));
    box.x[0] = box.x[2] = bx0 + d->pix_center;
    box.x[1] = bx1 + d->pix_center;
    box.y[0] = box.y[1] = by0 + d->pix_center;
    box.y[2] = by1 + d->pix_center;
    box.x0 = bx0;
    box.y0 = by0;
    box.x1 = bx1;
    box.y1 = by1;
    box.prim = UINT32_MAX;
    mtl_hazard(m, d, &box, true);
    /* the triangles of each 8 x 8 tile of the box, in order */
    bx0 &= ~7;
    by0 &= ~7;
    ntx = (bx1 - bx0) / 8 + 1;
    nty = (by1 - by0) / 8 + 1;
    cnt = g_new0(uint32_t, (size_t)ntx * nty);
    for (i = 0; i < d->ntri; i++) {
        const HD5770GpuTri *t = &d->tri[i];

        if (t->x1 < t->x0 || t->y1 < t->y0) {
            continue;
        }
        for (ty = (t->y0 - by0) / 8; ty <= (t->y1 - by0) / 8; ty++) {
            for (tx = (t->x0 - bx0) / 8; tx <= (t->x1 - bx0) / 8; tx++) {
                cnt[ty * ntx + tx]++;
                total++;
            }
        }
    }
    o_bin = mtl_take(m, (size_t)total * 4 + 4);
    bin = (uint32_t *)(st + o_bin);
    at = g_new(uint32_t, (size_t)ntx * nty);
    total = 0;
    for (ty = 0; ty < nty; ty++) {
        for (tx = 0; tx < ntx; tx++) {
            uint32_t c = cnt[ty * ntx + tx];

            at[ty * ntx + tx] = total;
            if (c) {
                uint32_t te[8] = {
                    o_draw, o_bin + total * 4, c, o_tex, o_kv,
                    bx0 + tx * 8, by0 + ty * 8, 8 | 0x200,
                };

                g_array_append_vals(m->tiles, te, 8);
            }
            total += c;
        }
    }
    for (i = 0; i < d->ntri; i++) {
        const HD5770GpuTri *t = &d->tri[i];

        if (t->x1 < t->x0 || t->y1 < t->y0) {
            continue;
        }
        for (ty = (t->y0 - by0) / 8; ty <= (t->y1 - by0) / 8; ty++) {
            for (tx = (t->x0 - bx0) / 8; tx <= (t->x1 - bx0) / 8; tx++) {
                bin[at[ty * ntx + tx]++] = i;
            }
        }
    }
    g_free(cnt);
    g_free(at);
    *miny = MIN(*miny, by0);
    *maxy = MAX(*maxy, by1);
}

static bool mtl_draw(void *opaque, const HD5770GpuDraw *d)
{
    HD5770Mtl *m = opaque;
    MtlProg *p;
    size_t need, o_draw, o_tex, o_tri, o_prm, o_kv, o_cf, o_vb[HD5770_GPU_VB];
    size_t prm_tri = (size_t)d->ninputs * 12 * sizeof(float);
    uint8_t *st = m->st.contents;
    GDraw *u;
    int64_t t0;
    int miny = INT_MAX, maxy = -1;
    unsigned i, k;
    bool first = true;
    uint32_t mipmask = 0;

    for (i = 0; i < d->ntex; i++) {
        mipmask |= (uint32_t)d->tex[i].mip << i;
    }
    p = mtl_prog(m, d->ir, mipmask);
    switch (qatomic_load_acquire(&p->state)) {
    case MTL_P_BUILDING:
        m->declined[MTL_BUILDING]++;
        m->pub.building = true;
        return false;
    case MTL_P_FAILED:
        m->declined[MTL_NO_PROG]++;
        return false;
    default:
        break;
    }
    for (i = 0; i < MAX(d->nmrt, 1u); i++) {
        const HD5770GpuSurf *rs = d->nmrt ? &d->mrt[i].surf : &d->rt;

        if ((!rs->space &&
             rs->voff + hd5770_surf_size(&rs->s) > m->vram_size) ||
            (rs->space && !m->ram) || !(rs->s.fast || rs->s.split)) {
            m->declined[MTL_NO_SURF]++;
            return false;
        }
    }
    for (i = 0; i < d->ntex; i++) {
        const HD5770GpuTex *t = &d->tex[i];

        for (k = 0; k <= (t->mip ? t->last_lv : 0); k++) {
            if (t->slice_bytes[k] > UINT32_MAX ||
                (!t->surf.space && (k ? t->lv[k].voff : t->surf.voff) +
                 mtl_tex_lvbytes(t, k) > m->vram_size)) {
                m->declined[MTL_NO_SURF]++;
                return false;
            }
        }
    }
    if (d->db &&
        ((d->zfmt && (d->zs.space || !(d->zs.s.fast || d->zs.s.split) ||
                      d->zs.voff + hd5770_surf_size(&d->zs.s) > m->vram_size)) ||
         (d->has_s && (d->ss.space || !(d->ss.s.fast || d->ss.s.split) ||
                       d->ss.voff + hd5770_surf_size(&d->ss.s) > m->vram_size)))) {
        m->declined[MTL_NO_SURF]++;
        return false;
    }
    need = sizeof(GDraw) + sizeof(GTex) * HD5770_GPU_TEX + 4096 +
           d->ntri * (MTL_TRI_SLOT + prm_tri + 16 + 2 * MTL_ALIGN) +
           (d->ir->nconsts + 1) * 4 + 8 * MTL_ALIGN;
    {
        int bx0 = INT_MAX, by0 = INT_MAX, bx1 = -1, by1 = -1;

        for (i = 0; i < d->ntri; i++) {
            const HD5770GpuTri *t = &d->tri[i];

            if (t->x1 >= t->x0 && t->y1 >= t->y0) {
                /*
                 * binned: an entry per 8x8 screen tile it touches; else
                 * tiles of at least 8 pixels each way, or a row of 64
                 */
                need += d->ntri > MTL_BIN_MIN ?
                        4 * ((size_t)(t->x1 - t->x0) / 8 + 2) *
                        ((size_t)(t->y1 - t->y0) / 8 + 2) :
                        32 * ((size_t)(t->x1 - t->x0) / 8 + 2) *
                        ((size_t)(t->y1 - t->y0) / 1 + 2);
                bx0 = MIN(bx0, t->x0);
                by0 = MIN(by0, t->y0);
                bx1 = MAX(bx1, t->x1);
                by1 = MAX(by1, t->y1);
            }
        }
        if (bx1 >= bx0) {
            /* the screen tiles of a binned draw */
            need += 32 * ((size_t)(bx1 - bx0) / 8 + 2) *
                    ((size_t)(by1 - by0) / 8 + 2) + MTL_ALIGN;
        }
    }
    /* the batch's tiles go to staging when it is dispatched */
    need += m->tiles->len * sizeof(uint32_t);
    for (i = 0; i < d->nvb; i++) {
        need += d->vb[i].len + MTL_ALIGN;
    }
    need += (d->rt.npages + 1) * 4 + MTL_ALIGN;
    need += (d->rs.npages + 1) * 4 + MTL_ALIGN;
    if (d->resolve && ((d->rs.space && !m->ram) || !(d->rs.s.fast || d->rs.s.split) ||
                       (!d->rs.space && d->rs.voff +
                        hd5770_surf_size(&d->rs.s) > m->vram_size))) {
        m->declined[MTL_NO_SURF]++;
        return false;
    }
    for (i = 0; i < d->nmrt; i++) {
        need += sizeof(GDraw) + (d->mrt[i].surf.npages + 1) * 4 + 2 * MTL_ALIGN;
    }
    for (i = 0; i < d->ntex; i++) {
        if (d->tex[i].surf.space && !m->ram) {
            m->declined[MTL_NO_SURF]++;
            return false;
        }
        need += (d->tex[i].surf.npages + 1) * 4 + MTL_ALIGN;
        if (d->tex[i].mip) {
            need += (d->tex[i].last_lv + 1) * sizeof(GSurf) + MTL_ALIGN;
        }
    }
    if (need > MTL_STAGING) {
        m->declined[MTL_NO_ROOM]++;
        return false;
    }
    if (m->st_used + need > MTL_STAGING) {
        mtl_sync(m);
    }
    t0 = get_clock();
    o_draw = mtl_take(m, sizeof(GDraw));
    o_tex = mtl_take(m, sizeof(GTex) * MAX(d->ntex, 1));
    o_kv = mtl_take(m, (d->ir->nconsts + 1) * 4);
    o_cf = mtl_take(m, 4096);
    memcpy(st + o_cf, d->cfile, 4096);
    for (i = 0; i < d->nvb; i++) {
        o_vb[i] = mtl_take(m, d->vb[i].len);
        memcpy(st + o_vb[i], d->vb[i].data, d->vb[i].len);
    }
    for (i = 0; i < d->ir->nconsts; i++) {
        memcpy(st + o_kv + 4 * i, &d->kv[i], 4);
    }

    u = (GDraw *)(st + o_draw);
    memset(u, 0, sizeof(*u));
    mtl_surf(m, &u->rt, &d->rt);
    u->rt_fmt = d->rt_fmt;
    u->rt_num = d->rt_num;
    u->rt_comps = d->rt_comps;
    u->rt_unorm8 = d->rt_unorm8;
    u->rt_float = d->rt_float;
    u->rt_blend = d->rt_blend;
    u->wmask = d->wmask;
    u->blend = d->blend;
    memcpy(u->rt_bits, d->rt_bits, sizeof(u->rt_bits));
    memcpy(u->rt_src, d->rt_src, sizeof(u->rt_src));
    memcpy(u->blend_const, d->blend_const, sizeof(u->blend_const));
    u->alpha_test = d->alpha_test;
    u->alpha_func = d->alpha_func;
    u->alpha_ref = d->alpha_ref;
    u->cliprule = d->cliprule;
    memcpy(u->cr, d->cr, sizeof(u->cr));
    u->pix_center = d->pix_center;
    u->persp = d->persp;
    u->pos_gpr = d->pos_gpr >= 0 && d->ir->map[d->pos_gpr] != 0xff
                 ? d->ir->map[d->pos_gpr] : -1;
    u->ninputs = d->ninputs;
    u->flat = d->flat;
    u->cfile_off = o_cf;
    u->ntex = d->ntex;
    u->nvb = d->nvb;
    u->zexp = d->zexport;
    u->sexp = d->sexport;
    if (d->resolve) {
        mtl_surf(m, &u->rs, &d->rs);
        u->resolve = 1;
        u->rs_same = d->rs_same;
        u->rs_fmt = d->rs_fmt;
        u->rs_num = d->rs_num;
        u->rs_comps = d->rs_comps;
        memcpy(u->rs_src, d->rs_src, sizeof(u->rs_src));
    }
    u->bconst = d->bool_const;
    memcpy(u->lconst, d->loop_const, sizeof(u->lconst));
    if (d->db) {
        if (d->zfmt) {
            mtl_surf(m, &u->zs, &d->zs);
        }
        if (d->has_s) {
            mtl_surf(m, &u->ss, &d->ss);
        }
        u->db = 1;
        u->zfmt = d->zfmt;
        u->has_s = d->has_s;
        u->dctl = d->dctl;
        u->rctl = d->rctl;
        u->srm0 = d->srm[0];
        u->srm1 = d->srm[1];
        u->sclear = d->sclear;
        u->zclear = d->zclear;
        u->zmin = d->zmin;
        u->zmax = d->zmax;
    }
    for (i = 0; i < d->nvb; i++) {
        u->vb[i].id = d->vb[i].id;
        u->vb[i].off = o_vb[i];
        u->vb[i].len = d->vb[i].len;
        memcpy(u->vb[i].res, d->vb[i].res, sizeof(u->vb[i].res));
    }
    if (d->nmrt) {
        /* each target: a copy of the draw with its own target fields */
        size_t o_mrt = mtl_take(m, sizeof(GDraw) * d->nmrt);

        u->nmrt = d->nmrt;
        u->mrt_off = o_mrt;
        for (i = 0; i < d->nmrt; i++) {
            const HD5770GpuRt *r = &d->mrt[i];
            GDraw *c = (GDraw *)(st + o_mrt) + i;

            memcpy(c, u, sizeof(*c));
            c->nmrt = 0;
            mtl_surf(m, &c->rt, &r->surf);
            c->rt_fmt = r->fmt;
            c->rt_num = r->num;
            c->rt_comps = r->comps;
            c->rt_unorm8 = r->unorm8;
            c->rt_float = r->is_float;
            c->rt_blend = r->blend_ok;
            c->wmask = r->wmask;
            c->blend = r->blend;
            memcpy(c->rt_bits, r->bits, sizeof(c->rt_bits));
            memcpy(c->rt_src, r->src, sizeof(c->rt_src));
            c->cb = r->cb;
        }
    }
    for (i = 0; i < d->ntex; i++) {
        const HD5770GpuTex *t = &d->tex[i];
        GTex *g = (GTex *)(st + o_tex) + i;

        memset(g, 0, sizeof(*g));
        mtl_surf(m, &g->s, &t->surf);
        g->fmt = t->fmt;
        g->num = t->num;
        g->w = t->w;
        g->h = t->h;
        g->dim = t->dim;
        g->sgn = t->sgn;
        g->comps = t->comps;
        g->is_float = t->is_float;
        g->unorm8 = t->unorm8;
        g->samp0 = t->samp[0];
        g->base_lv = t->base_lv;
        g->levels = t->levels;
        g->depth = t->depth;
        for (k = 0; k < 4; k++) {
            g->dst_sel[k] = t->dst_sel[k];
            g->bits[k] = t->bits[k];
            g->border[k] = t->border[k];
        }
        g->lw[0] = t->w;
        g->lh[0] = t->h;
        g->bc = t->bc;
        g->slices = MAX(t->slices, 1u);
        for (k = 0; k < 16; k++) {
            g->sb[k] = k < HD5770_GPU_LEVELS ? (uint32_t)t->slice_bytes[k] : 0;
        }
        if (t->mip) {
            size_t lo = mtl_take(m, (t->last_lv + 1) * sizeof(GSurf));
            GSurf *ls = (GSurf *)(st + lo);

            for (k = 1; k <= t->last_lv; k++) {
                mtl_surf(m, &ls[k], &t->lv[k]);
                g->lw[k] = t->lw[k];
                g->lh[k] = t->lh[k];
            }
            g->mip = 1;
            g->last_lv = t->last_lv;
            g->mip_filter = t->mip_filter;
            g->lev_off = lo;
            g->min_lod = t->min_lod;
            g->max_lod = t->max_lod;
            g->lod_bias = t->lod_bias;
        }
    }

    @autoreleasepool {
        if (!m->cb) {
            m->cb = [[m->q commandBuffer] retain];
        }
        if (!m->enc) {
            m->enc = [[m->cb computeCommandEncoderWithDispatchType:
                       MTLDispatchTypeConcurrent] retain];
            m->nwr = m->nrd = 0;
        }
        if (m->bpso != p->pso) {
            mtl_flush(m);
            m->bpso = p->pso;
        }
        if (d->ntri > MTL_BIN_MIN) {
            mtl_binned(m, d, u, o_draw, o_tex, o_kv, &miny, &maxy);
            /* a dispatch of its own */
            mtl_flush(m);
        }
        for (i = 0; i < d->ntri && d->ntri <= MTL_BIN_MIN; i++) {
            const HD5770GpuTri *t = &d->tri[i], *pt[2] = { t, NULL };
            size_t ps = ROUND_UP(MAX(prm_tri, 16), 16);
            unsigned tw, th, n = 1, q;
            int x, y, bx0, by0, bx1, by1;

            if (t->x1 < t->x0 || t->y1 < t->y0) {
                continue;
            }
            /* the two halves of a rectangle or quad share one set of tiles */
            if (i + 1 < d->ntri && d->tri[i + 1].prim == t->prim &&
                d->tri[i + 1].x1 >= d->tri[i + 1].x0 &&
                d->tri[i + 1].y1 >= d->tri[i + 1].y0) {
                pt[1] = &d->tri[i + 1];
                n = 2;
            }
            o_tri = mtl_take(m, MTL_TRI_SLOT * n);
            o_prm = mtl_take(m, ps * n);
            bx0 = t->x0;
            by0 = t->y0;
            bx1 = t->x1;
            by1 = t->y1;
            for (q = 0; q < n; q++) {
                const HD5770GpuTri *u = pt[q];
                GTri *gt = (GTri *)(st + o_tri + MTL_TRI_SLOT * q);

                mtl_hazard(m, d, u, first);
                first = false;
                memcpy(gt->x, u->x, sizeof(gt->x));
                memcpy(gt->y, u->y, sizeof(gt->y));
                memcpy(gt->w, u->w, sizeof(gt->w));
                gt->area = u->area;
                gt->x0 = u->x0;
                gt->y0 = u->y0;
                gt->x1 = u->x1;
                gt->y1 = u->y1;
                memcpy(gt->z, u->z, sizeof(gt->z));
                gt->front = u->front;
                memcpy(st + o_prm + ps * q,
                       d->prm + (size_t)(i + q) * d->ninputs * 12, prm_tri);
                bx0 = MIN(bx0, u->x0);
                by0 = MIN(by0, u->y0);
                bx1 = MAX(bx1, u->x1);
                by1 = MAX(by1, u->y1);
                m->tris++;
            }
            /* tiles of 64 x 1 for thin boxes, up to 8 x 8 */
            th = by1 - by0 + 1;
            th = th >= 8 ? 8 : th >= 4 ? 4 : th >= 2 ? 2 : 1;
            tw = MTL_TILE / th;
            for (y = by0; y <= by1; y += th) {
                for (x = bx0; x <= bx1; x += tw) {
                    uint32_t te[8] = {
                        o_draw, o_tri, o_prm, o_tex, o_kv, x, y,
                        tw | (n - 1) << 8 | (uint32_t)(ps / 16) << 16,
                    };

                    g_array_append_vals(m->tiles, te, 8);
                }
            }
            miny = MIN(miny, by0);
            maxy = MAX(maxy, by1);
            i += n - 1;
        }
    }
    if (maxy >= miny) {
        const HD5770GpuSurf *ws[11];
        unsigned nw = mtl_writes(d, ws);

        for (i = 0; i < nw; i++) {
            uint64_t lo, hi;

            if (ws[i]->space) {
                for (k = 0; k < ws[i]->npages; k++) {
                    mtl_range(m->ram_ranges, ws[i]->pages[k],
                              ws[i]->pages[k] + 4096);
                }
                continue;
            }
            mtl_rows(ws[i], miny, maxy, &lo, &hi);
            mtl_range(m->ranges, lo, MIN(hi, m->vram_size));
        }
    }
    m->draws++;
    m->encode_ns += get_clock() - t0;
    if (++m->pending >= MTL_COMMIT_AT) {
        @autoreleasepool {
            mtl_end(m);
        }
    }
    return true;
}

static void mtl_stats(void *opaque, GString *out)
{
    HD5770Mtl *m = opaque;

    g_string_append_printf(out,
        "metal: %s draws %" PRIu64 " triangles %" PRIu64 " declined: program %"
        PRIu64 " compiling %" PRIu64 " room %" PRIu64 " surface %" PRIu64
        "\n", [[m->dev name] UTF8String], m->draws, m->tris,
        m->declined[MTL_NO_PROG], m->declined[MTL_BUILDING],
        m->declined[MTL_NO_ROOM], m->declined[MTL_NO_SURF]);
    g_string_append_printf(out,
        "metal: syncs %" PRIu64 " commits %" PRIu64 " dispatches %" PRIu64
        " threads %" PRIu64 " barriers %" PRIu64 " gpu ms %" PRIu64 " wait ms %" PRIu64
        " encode ms %" PRIu64 " programs %u compiled %" PRIu64 " in ms %"
        PRIu64 "\n", m->syncs, m->commits, m->dispatches, m->threads,
        m->barriers,
        qatomic_read(&m->gpu_ns) / 1000000, m->wait_ns / 1000000,
        m->encode_ns / 1000000, g_hash_table_size(m->progs), m->compiles,
        qatomic_read(&m->compile_ns) / 1000000);
    g_string_append_printf(out, "metal: fences %" PRIu64 " retired %" PRIu64
                           "\n", m->tickets, qatomic_read(&m->fences_retired));
}

static void mtl_prog_free(gpointer p)
{
    MtlProg *mp = p;

    [mp->pso release];
    g_free(mp);
}

static void mtl_close(void *opaque)
{
    HD5770Mtl *m = opaque;

    mtl_sync(m);
    while (qatomic_read(&m->building) || qatomic_read(&m->handlers)) {
        g_usleep(1000);
    }
    @autoreleasepool {
        g_hash_table_destroy(m->progs);
        {
            MtlFence *f;

            while ((f = g_queue_pop_head(&m->fences))) {
                g_free(f);
            }
        }
        g_mutex_clear(&m->flock);
        g_mutex_clear(&m->rlock);
        g_array_free(m->ranges, true);
        g_array_free(m->tiles, true);
        g_array_free(m->ram_ranges, true);
        [m->ram release];
        [m->st release];
        [m->vram release];
        [m->q release];
        [m->dev release];
    }
    g_free(m);
}

static bool mtl_ram(void *opaque, uint8_t *host, uint64_t size)
{
    HD5770Mtl *m = opaque;
    size_t page = getpagesize();

    if (m->ram || (uintptr_t)host % page || size % page ||
        size > m->dev.maxBufferLength) {
        return false;
    }
    @autoreleasepool {
        m->ram = [m->dev newBufferWithBytesNoCopy:host length:size
                                          options:MTLResourceStorageModeShared
                                      deallocator:nil];
    }
    return m->ram != nil;
}

static const HD5770GpuOps mtl_ops = {
    .name = "metal",
    /* the CPU finishes these sooner than a round trip */
    .min_pixels = 1024,
    .depth = true,
    .mipmaps = true,
    .resinfo = true,
    .full = true,
    .draw = mtl_draw,
    .sync = mtl_sync,
    .stats = mtl_stats,
    .close = mtl_close,
    .ram = mtl_ram,
    .fence = mtl_fence,
};

HD5770Gpu *hd5770_mtl_open(uint8_t *vram, uint64_t vram_size,
                           const char **err)
{
    HD5770Mtl *m;
    size_t page = getpagesize();

    *err = NULL;
    if ((uintptr_t)vram % page || vram_size % page) {
        *err = "VRAM is not page aligned";
        return NULL;
    }
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();

        if (!dev) {
            *err = "no Metal device";
            return NULL;
        }
        if (!dev.hasUnifiedMemory) {
            [dev release];
            *err = "the Metal backend needs unified memory";
            return NULL;
        }
        m = g_new0(HD5770Mtl, 1);
        m->dev = dev;
        m->vram_host = vram;
        m->vram_size = vram_size;
        m->vram = [dev newBufferWithBytesNoCopy:vram length:vram_size
                                        options:MTLResourceStorageModeShared
                                    deallocator:nil];
        m->st = [dev newBufferWithLength:MTL_STAGING
                                 options:MTLResourceStorageModeShared];
        if (!m->vram || !m->st) {
            [m->vram release];
            [m->st release];
            [dev release];
            g_free(m);
            *err = "cannot map VRAM for the GPU";
            return NULL;
        }
        m->q = [dev newCommandQueue];
    }
    m->progs = g_hash_table_new_full(g_int64_hash, g_int64_equal, g_free,
                                     mtl_prog_free);
    m->ranges = g_array_new(false, false, sizeof(uint64_t));
    m->tiles = g_array_new(false, false, sizeof(uint32_t));
    m->ram_ranges = g_array_new(false, false, sizeof(uint64_t));
    g_mutex_init(&m->flock);
    g_mutex_init(&m->rlock);
    g_queue_init(&m->cbs);
    g_queue_init(&m->fences);
    m->pub.ops = &mtl_ops;
    m->pub.b = m;
    return &m->pub;
}
