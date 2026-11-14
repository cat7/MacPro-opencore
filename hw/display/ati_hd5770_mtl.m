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
    uint32_t space, pt_off, pg0;
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
} GDraw;

typedef struct GTri {
    float x[3], y[3], w[3];
    float area;
    int32_t x0, y0, x1, y1;
    float z[3];
    uint32_t front;
} GTri;

#define MTL_TRI_SLOT    128     /* staging bytes per GTri */

QEMU_BUILD_BUG_ON(sizeof(GSurf) != 4 * 87);
QEMU_BUILD_BUG_ON(sizeof(GTex) != 4 * 148);
QEMU_BUILD_BUG_ON(sizeof(GDraw) != 4 * 416);
QEMU_BUILD_BUG_ON(sizeof(GTri) != 4 * 18);

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
    for (i = 0; i + 1 < m->ram_ranges->len; i += 2) {
        if (m->pub.done_ram) {
            m->pub.done_ram(m->pub.opaque,
                            g_array_index(m->ram_ranges, uint64_t, i),
                            g_array_index(m->ram_ranges, uint64_t, i + 1));
        }
    }
    g_array_set_size(m->ram_ranges, 0);
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
    uint64_t hi = mtl_hi(&t->surf);
    unsigned l;

    for (l = 1; t->mip && l <= t->last_lv; l++) {
        hi = MAX(hi, mtl_hi(&t->lv[l]));
    }
    return hi;
}

/* the surfaces a request writes: target, depth plane, stencil plane */
static unsigned mtl_writes(const HD5770GpuDraw *d, const HD5770GpuSurf **ws)
{
    unsigned n = 0;

    ws[n++] = &d->rt;
    if (d->db && d->zfmt) {
        ws[n++] = &d->zs;
    }
    if (d->db && d->has_s) {
        ws[n++] = &d->ss;
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
    const HD5770GpuSurf *ws[3];
    unsigned nw = mtl_writes(d, ws), i, j, k;
    bool bar = m->nwr + nw > MTL_ACC || m->nrd + d->ntex > MTL_ACC;
    uint64_t lo[3], hi[3];
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
        return false;
    case MTL_P_FAILED:
        m->declined[MTL_NO_PROG]++;
        return false;
    default:
        break;
    }
    if ((!d->rt.space &&
         d->rt.voff + hd5770_surf_size(&d->rt.s) > m->vram_size) ||
        (d->rt.space && !m->ram) || !d->rt.s.fast) {
        m->declined[MTL_NO_SURF]++;
        return false;
    }
    if (d->db &&
        ((d->zfmt && (d->zs.space || !d->zs.s.fast ||
                      d->zs.voff + hd5770_surf_size(&d->zs.s) > m->vram_size)) ||
         (d->has_s && (d->ss.space || !d->ss.s.fast ||
                       d->ss.voff + hd5770_surf_size(&d->ss.s) > m->vram_size)))) {
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
    need += (d->rt.npages + 1) * 4 + MTL_ALIGN;
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
    for (i = 0; i < d->ntex; i++) {
        const HD5770GpuTex *t = &d->tex[i];
        GTex *g = (GTex *)(st + o_tex) + i;

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
        for (k = 0; k < 4; k++) {
            g->dst_sel[k] = t->dst_sel[k];
            g->bits[k] = t->bits[k];
            g->border[k] = t->border[k];
        }
        g->lw[0] = t->w;
        g->lh[0] = t->h;
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
        for (i = 0; i < d->ntri; i++) {
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
    if (maxy >= miny && d->rt.space) {
        for (i = 0; i < d->rt.npages; i++) {
            mtl_range(m->ram_ranges, d->rt.pages[i], d->rt.pages[i] + 4096);
        }
    } else if (maxy >= miny) {
        uint64_t lo, hi;

        mtl_rows(&d->rt, miny, maxy, &lo, &hi);
        mtl_range(m->ranges, lo, MIN(hi, m->vram_size));
    }
    if (maxy >= miny && d->db) {
        const HD5770GpuSurf *ws[3];
        unsigned nw = mtl_writes(d, ws);

        for (i = 1; i < nw; i++) {
            uint64_t lo, hi;

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
    while (qatomic_read(&m->building)) {
        g_usleep(1000);
    }
    @autoreleasepool {
        g_hash_table_destroy(m->progs);
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
    .draw = mtl_draw,
    .sync = mtl_sync,
    .stats = mtl_stats,
    .close = mtl_close,
    .ram = mtl_ram,
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
    m->pub.ops = &mtl_ops;
    m->pub.b = m;
    return &m->pub;
}
