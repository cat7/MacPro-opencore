/*
 * ATI Radeon HD 5770 (Juniper) 3D engine on the host GPU through Metal.
 *
 * VRAM is the GPU's memory: on an Apple GPU the host memory of the VRAM
 * region is wrapped as one shared buffer, the kernels read textures and
 * read-modify-write render targets in it at their tiled addresses, and
 * nothing is copied back. A draw is one compute dispatch per triangle,
 * in order, each thread one pixel of the triangle's box running the
 * program translated by ati_hd5770_msl.c. Per-draw data goes to a
 * staging buffer that is reused once sync() has waited for the GPU.
 * Programs are compiled on first use and kept by the hash of their
 * words.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/timer.h"
#import <Metal/Metal.h>
#include "ati_hd5770_gpu.h"

#define MTL_STAGING     (64 * 1024 * 1024)
#define MTL_ALIGN       256
#define MTL_COMMIT_AT   64      /* draws per command buffer */

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
    uint64_t draws, tris, declined[MTL_NO_REASONS], syncs, commits;
    uint64_t compiles, compile_ns, wait_ns, encode_ns;
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

static void mtl_end(HD5770Mtl *m)
{
    if (m->enc) {
        [m->enc endEncoding];
        [m->enc release];
        m->enc = nil;
    }
    if (m->cb) {
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
            m->enc = [[m->cb computeCommandEncoder] retain];
        }
        [m->enc setComputePipelineState:p->pso];
        [m->enc setBuffer:m->vram offset:0 atIndex:0];
        [m->enc setBuffer:m->st offset:o_draw atIndex:1];
        [m->enc setBuffer:m->st offset:o_kv atIndex:3];
        [m->enc setBuffer:m->st offset:0 atIndex:4];
        [m->enc setBuffer:m->st offset:o_tex atIndex:5];
        for (i = 0; i < d->ntri; i++) {
            const HD5770GpuTri *t = &d->tri[i];
            GTri *gt;
            unsigned w = t->x1 - t->x0 + 1, h = t->y1 - t->y0 + 1;
            NSUInteger tw, th, mx = p->pso.maxTotalThreadsPerThreadgroup;

            if (t->x1 < t->x0 || t->y1 < t->y0) {
                continue;
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
            tw = MIN(w, h == 1 ? 256u : 16u);
            th = MIN(h, MAX(1u, (unsigned)(mx / tw)));
            th = MIN(th, 16u);
            [m->enc setBuffer:m->st offset:o_tri atIndex:2];
            [m->enc setBuffer:m->st offset:o_prm atIndex:6];
            [m->enc dispatchThreads:MTLSizeMake(w, h, 1)
              threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
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
        PRIu64 " room %" PRIu64 " surface %" PRIu64 "\n"
        "metal: syncs %" PRIu64 " commits %" PRIu64 " wait ms %" PRIu64
        " encode ms %" PRIu64 " programs %u compiled %" PRIu64 " in ms %"
        PRIu64 "\n", [[m->dev name] UTF8String], m->draws, m->tris,
        m->declined[MTL_NO_PROG], m->declined[MTL_NO_ROOM],
        m->declined[MTL_NO_SURF], m->syncs, m->commits, m->wait_ns / 1000000,
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
    m->pub.ops = &mtl_ops;
    m->pub.b = m;
    return &m->pub;
}
