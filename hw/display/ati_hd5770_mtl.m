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
    uint32_t eoff[64];
} GSurf;

typedef struct GTex {
    GSurf s;
    uint32_t fmt, num, w, h, dim, sgn, comps, is_float, unorm8, samp0;
    uint32_t dst_sel[4];
    uint32_t bits[4];
    float border[4];
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
} GDraw;

typedef struct GTri {
    float x[3], y[3], w[3];
    float area;
    int32_t x0, y0, x1, y1;
} GTri;

QEMU_BUILD_BUG_ON(sizeof(GSurf) != 4 * 84);
QEMU_BUILD_BUG_ON(sizeof(GTex) != 4 * 106);
QEMU_BUILD_BUG_ON(sizeof(GDraw) != 4 * 228);
QEMU_BUILD_BUG_ON(sizeof(GTri) != 4 * 14);

/* a pixel box one dispatch wrote, or bytes it read */
typedef struct MtlAcc {
    uint64_t lo, hi;            /* VRAM bytes */
    uint32_t voff, mode, row_bytes, bpe;
    int32_t x0, y0, x1, y1;
    uint64_t draw;
    uint32_t prim;
} MtlAcc;

typedef struct MtlProg {
    id<MTLComputePipelineState> pso;    /* nil: not translatable */
} MtlProg;

enum {
    MTL_NO_PROG, MTL_NO_ROOM, MTL_NO_SURF, MTL_NO_REASONS,
};

typedef struct HD5770Mtl {
    HD5770Gpu pub;
    id<MTLDevice> dev;
    id<MTLCommandQueue> q;
    id<MTLBuffer> vram, st;
    uint8_t *vram_host;
    uint64_t vram_size;
    id<MTLCommandBuffer> cb, last;
    id<MTLComputeCommandEncoder> enc;
    size_t st_used;
    unsigned pending;
    GHashTable *progs;
    GArray *ranges;             /* uint64_t pairs, VRAM written */
    MtlAcc wr[MTL_ACC], rd[MTL_ACC];
    unsigned nwr, nrd;
    uint64_t barriers, dispatches;
    /* the batch: tiles of consecutive triangles run by one dispatch */
    id<MTLComputePipelineState> bpso;
    GArray *tiles;              /* uint32_t x 8 per tile */
    uint64_t draws, tris, declined[MTL_NO_REASONS], syncs, commits;
    uint64_t compiles, compile_ns, wait_ns, encode_ns;
    uint64_t gpu_ns;            /* atomic: command buffers on the GPU */
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

static MtlProg *mtl_prog(HD5770Mtl *m, const HD5770IR *ir)
{
    MtlProg *p = g_hash_table_lookup(m->progs, &ir->hash);
    int64_t t0;
    char *src;

    if (p) {
        return p;
    }
    p = g_new0(MtlProg, 1);
    g_hash_table_insert(m->progs, g_memdup2(&ir->hash, sizeof(ir->hash)), p);
    src = hd5770_msl_program(ir);
    if (!src) {
        return p;
    }
    t0 = get_clock();
    @autoreleasepool {
        MTLCompileOptions *opt = [[MTLCompileOptions alloc] init];
        NSError *e = nil;
        id<MTLLibrary> lib;

        mtl_exact_math(opt);
        lib = [m->dev newLibraryWithSource:[NSString stringWithUTF8String:src]
                                   options:opt error:&e];
        [opt release];
        if (lib) {
            id<MTLFunction> fn = [lib newFunctionWithName:@"hd_main"];

            if (fn) {
                p->pso = [m->dev newComputePipelineStateWithFunction:fn
                                                               error:&e];
                [fn release];
            }
            [lib release];
        }
        if (!p->pso) {
            fprintf(stderr, "ati-hd5770: Metal program %016" PRIx64
                    " not built: %s\n", ir->hash,
                    e ? [[e localizedDescription] UTF8String] : "?");
        }
    }
    g_free(src);
    m->compiles++;
    m->compile_ns += get_clock() - t0;
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
    [m->enc dispatchThreadgroups:MTLSizeMake(m->tiles->len / 8, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(MTL_TILE, 1, 1)];
    g_array_set_size(m->tiles, 0);
    m->dispatches++;
}

static void mtl_end(HD5770Mtl *m)
{
    if (m->enc) {
        mtl_flush(m);
        [m->enc endEncoding];
        [m->enc release];
        m->enc = nil;
    }
    if (m->cb) {
        [m->cb addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            qatomic_add(&m->gpu_ns,
                        (uint64_t)((cb.GPUEndTime - cb.GPUStartTime) * 1e9));
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
    int64_t t0;
    guint i;

    @autoreleasepool {
        mtl_end(m);
        if (m->last) {
            t0 = get_clock();
            [m->last waitUntilCompleted];
            m->wait_ns += get_clock() - t0;
            [m->last release];
            m->last = nil;
        }
    }
    m->st_used = 0;
    m->syncs++;
    for (i = 0; i + 1 < m->ranges->len; i += 2) {
        m->pub.done(m->pub.opaque, g_array_index(m->ranges, uint64_t, i),
                    g_array_index(m->ranges, uint64_t, i + 1));
    }
    g_array_set_size(m->ranges, 0);
}

static void mtl_surf(GSurf *g, const HD5770GpuSurf *s)
{
    const HD5770Surf *f = &s->s;

    memset(g, 0, sizeof(*g));
    g->voff = s->voff;
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
    if (f->mode >= HD5770_ARRAY_1D_TILED_THIN1) {
        memcpy(g->eoff, f->eoff, sizeof(g->eoff));
    }
}

static void mtl_range(HD5770Mtl *m, uint64_t lo, uint64_t hi)
{
    guint n = m->ranges->len;

    if (n && g_array_index(m->ranges, uint64_t, n - 2) <= hi &&
        lo <= g_array_index(m->ranges, uint64_t, n - 1)) {
        g_array_index(m->ranges, uint64_t, n - 2) =
            MIN(lo, g_array_index(m->ranges, uint64_t, n - 2));
        g_array_index(m->ranges, uint64_t, n - 1) =
            MAX(hi, g_array_index(m->ranges, uint64_t, n - 1));
        return;
    }
    g_array_append_val(m->ranges, lo);
    g_array_append_val(m->ranges, hi);
}

static bool mtl_overlap(uint64_t a0, uint64_t a1, uint64_t b0, uint64_t b1)
{
    return a0 < b1 && b0 < a1;
}

/*
 * A barrier before a dispatch writing box (x0, y0)-(x1, y1) of the target,
 * bytes [lo, hi), if an earlier one since the last barrier conflicts.
 * reads: the draw's textures, checked once per draw (first).
 */
static void mtl_hazard(HD5770Mtl *m, const HD5770GpuDraw *d,
                       const HD5770GpuTri *t, uint64_t lo, uint64_t hi,
                       bool first)
{
    const HD5770GpuSurf *rt = &d->rt;
    bool bar = m->nwr >= MTL_ACC || m->nrd + d->ntex > MTL_ACC;
    unsigned i, k;
    /*
     * The pixels the triangle can cover: centres within its vertices'
     * extent, widened by far more than the edge functions' rounding.
     */
    float pc = d->pix_center, mg = 1.0f / 16;
    int32_t x0 = ceilf(MIN(t->x[0], MIN(t->x[1], t->x[2])) - pc - mg);
    int32_t x1 = floorf(MAX(t->x[0], MAX(t->x[1], t->x[2])) - pc + mg);
    int32_t y0 = ceilf(MIN(t->y[0], MIN(t->y[1], t->y[2])) - pc - mg);
    int32_t y1 = floorf(MAX(t->y[0], MAX(t->y[1], t->y[2])) - pc + mg);

    for (i = 0; !bar && i < m->nwr; i++) {
        const MtlAcc *a = &m->wr[i];

        if (a->voff == rt->voff && a->mode == rt->s.mode &&
            a->row_bytes == rt->s.row_bytes && a->bpe == rt->s.bpe) {
            bar = a->x0 <= x1 && x0 <= a->x1 && a->y0 <= y1 &&
                  y0 <= a->y1 && !(a->draw == m->draws && a->prim == t->prim);
        } else {
            bar = mtl_overlap(a->lo, a->hi, lo, hi);
        }
        for (k = 0; first && !bar && k < d->ntex; k++) {
            const HD5770GpuSurf *ts = &d->tex[k].surf;

            bar = mtl_overlap(a->lo, a->hi, ts->voff,
                              ts->voff + hd5770_surf_size(&ts->s));
        }
    }
    for (i = 0; !bar && i < m->nrd; i++) {
        bar = mtl_overlap(m->rd[i].lo, m->rd[i].hi, lo, hi);
    }
    if (bar) {
        mtl_flush(m);
        [m->enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        m->nwr = m->nrd = 0;
        m->barriers++;
    }
    if (first || bar) {
        for (k = 0; k < d->ntex; k++) {
            const HD5770GpuSurf *ts = &d->tex[k].surf;

            m->rd[m->nrd].lo = ts->voff;
            m->rd[m->nrd].hi = ts->voff + hd5770_surf_size(&ts->s);
            m->nrd++;
        }
    }
    m->wr[m->nwr] = (MtlAcc) {
        .lo = lo, .hi = hi, .voff = rt->voff, .mode = rt->s.mode,
        .row_bytes = rt->s.row_bytes, .bpe = rt->s.bpe,
        .x0 = x0, .y0 = y0, .x1 = x1, .y1 = y1,
        .draw = m->draws, .prim = t->prim,
    };
    m->nwr++;
}

static bool mtl_draw(void *opaque, const HD5770GpuDraw *d)
{
    HD5770Mtl *m = opaque;
    MtlProg *p = mtl_prog(m, d->ir);
    size_t need, o_draw, o_tex, o_tri, o_prm, o_kv, o_cf, o_vb[HD5770_GPU_VB];
    size_t prm_tri = (size_t)d->ninputs * 12 * sizeof(float);
    uint8_t *st = m->st.contents;
    GDraw *u;
    int64_t t0;
    int miny = INT_MAX, maxy = -1;
    unsigned i, k;
    bool first = true;

    if (!p->pso) {
        m->declined[MTL_NO_PROG]++;
        return false;
    }
    if (d->rt.voff + hd5770_surf_size(&d->rt.s) > m->vram_size ||
        !d->rt.s.fast) {
        m->declined[MTL_NO_SURF]++;
        return false;
    }
    need = sizeof(GDraw) + sizeof(GTex) * HD5770_GPU_TEX + 4096 +
           d->ntri * (sizeof(GTri) + prm_tri + 2 * MTL_ALIGN) +
           (d->ir->nconsts + 1) * 4 + 8 * MTL_ALIGN;
    for (i = 0; i < d->ntri; i++) {
        const HD5770GpuTri *t = &d->tri[i];

        if (t->x1 >= t->x0 && t->y1 >= t->y0) {
            /* tiles of at least 8 pixels each way, or a row of 64 */
            need += 32 * ((size_t)(t->x1 - t->x0) / 8 + 2) *
                    ((size_t)(t->y1 - t->y0) / 1 + 2);
        }
    }
    /* the batch's tiles go to staging when it is dispatched */
    need += m->tiles->len * sizeof(uint32_t);
    for (i = 0; i < d->nvb; i++) {
        need += d->vb[i].len + MTL_ALIGN;
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
    mtl_surf(&u->rt, &d->rt);
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
    for (i = 0; i < d->nvb; i++) {
        u->vb[i].id = d->vb[i].id;
        u->vb[i].off = o_vb[i];
        u->vb[i].len = d->vb[i].len;
        memcpy(u->vb[i].res, d->vb[i].res, sizeof(u->vb[i].res));
    }
    for (i = 0; i < d->ntex; i++) {
        const HD5770GpuTex *t = &d->tex[i];
        GTex *g = (GTex *)(st + o_tex) + i;

        mtl_surf(&g->s, &t->surf);
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
        for (k = 0; k < 4; k++) {
            g->dst_sel[k] = t->dst_sel[k];
            g->bits[k] = t->bits[k];
            g->border[k] = t->border[k];
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
        for (i = 0; i < d->ntri; i++) {
            const HD5770GpuTri *t = &d->tri[i];
            GTri *gt;
            unsigned tw, th;
            int x, y;

            if (t->x1 < t->x0 || t->y1 < t->y0) {
                continue;
            }
            {
                uint64_t lo, hi;

                hd5770_surf_rows(&d->rt.s, t->y0, t->y1 + 1, &lo, &hi);
                mtl_hazard(m, d, t, lo - d->rt.s.base + d->rt.voff,
                           hi - d->rt.s.base + d->rt.voff, first);
                first = false;
            }
            o_tri = mtl_take(m, sizeof(GTri));
            gt = (GTri *)(st + o_tri);
            memcpy(gt->x, t->x, sizeof(gt->x));
            memcpy(gt->y, t->y, sizeof(gt->y));
            memcpy(gt->w, t->w, sizeof(gt->w));
            gt->area = t->area;
            gt->x0 = t->x0;
            gt->y0 = t->y0;
            gt->x1 = t->x1;
            gt->y1 = t->y1;
            o_prm = mtl_take(m, MAX(prm_tri, 16));
            memcpy(st + o_prm, d->prm + (size_t)i * d->ninputs * 12, prm_tri);
            /* tiles of 64 x 1 for thin boxes, up to 8 x 8 */
            th = t->y1 - t->y0 + 1;
            th = th >= 8 ? 8 : th >= 4 ? 4 : th >= 2 ? 2 : 1;
            tw = MTL_TILE / th;
            for (y = t->y0; y <= t->y1; y += th) {
                for (x = t->x0; x <= t->x1; x += tw) {
                    uint32_t te[8] = {
                        o_draw, o_tri, o_prm, o_tex, o_kv, x, y, tw,
                    };

                    g_array_append_vals(m->tiles, te, 8);
                }
            }
            miny = MIN(miny, t->y0);
            maxy = MAX(maxy, t->y1);
            m->tris++;
        }
    }
    if (maxy >= miny) {
        uint64_t lo, hi;

        hd5770_surf_rows(&d->rt.s, miny, maxy + 1, &lo, &hi);
        lo = lo - d->rt.s.base + d->rt.voff;
        hi = hi - d->rt.s.base + d->rt.voff;
        mtl_range(m, lo, MIN(hi, m->vram_size));
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
        PRIu64 " room %" PRIu64 " surface %" PRIu64 "\n",
        [[m->dev name] UTF8String], m->draws, m->tris,
        m->declined[MTL_NO_PROG], m->declined[MTL_NO_ROOM],
        m->declined[MTL_NO_SURF]);
    g_string_append_printf(out,
        "metal: syncs %" PRIu64 " commits %" PRIu64 " dispatches %" PRIu64
        " barriers %" PRIu64 " gpu ms %" PRIu64 " wait ms %" PRIu64
        " encode ms %" PRIu64 " programs %u compiled %" PRIu64 " in ms %"
        PRIu64 "\n", m->syncs, m->commits, m->dispatches, m->barriers,
        qatomic_read(&m->gpu_ns) / 1000000, m->wait_ns / 1000000,
        m->encode_ns / 1000000, g_hash_table_size(m->progs), m->compiles,
        m->compile_ns / 1000000);
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
    @autoreleasepool {
        g_hash_table_destroy(m->progs);
        g_array_free(m->ranges, true);
        g_array_free(m->tiles, true);
        [m->st release];
        [m->vram release];
        [m->q release];
        [m->dev release];
    }
    g_free(m);
}

static const HD5770GpuOps mtl_ops = {
    .name = "metal",
    .draw = mtl_draw,
    .sync = mtl_sync,
    .stats = mtl_stats,
    .close = mtl_close,
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
    m->pub.ops = &mtl_ops;
    m->pub.b = m;
    return &m->pub;
}
