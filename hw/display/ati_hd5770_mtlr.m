/*
 * ATI Radeon HD 5770 (Juniper) 3D engine on the host's own pipeline
 * (Metal render passes), gpu=native.
 *
 * A draw's triangles go through the host rasteriser with the pixel
 * program as a fragment function (hd5770_msl_native_fs()); the vertex
 * stage stays on the CPU. Results are close to the CPU path's, not bit
 * for bit. Whatever is not expressible is declined and drawn on the CPU.
 *
 * SURFACES. Targets and textures are host textures holding the bytes of
 * the guest surface (RGBA8, byte i of an element in channel i). An entry
 * is checked against memory at its first use after each sync (a hash of
 * its bytes; uploaded when it differs), drawn into, and written back to
 * VRAM at the next sync, or before another entry over the same bytes is
 * used. At most one of the entries over a byte is newer than memory.
 *
 * Accepted: one 8_8_8_8 UNORM colour target in VRAM, no depth, programs
 * in the straight form, 2D 8_8_8_8 UNORM textures without mipmaps.
 * Builds are synchronous.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/timer.h"
#import <Metal/Metal.h>
#include "ati_hd5770_gpu.h"

#define NAT_RING        (48u << 20)
#define NAT_CACHE_MAX   (512ull << 20)

enum {
    ND_TAKEN, ND_EMPTY, ND_RT_SPACE, ND_RT_FMT, ND_RT_SURF, ND_STATE_DB,
    ND_CLIPRECT, ND_BLEND_ORDER, ND_PROG_FORM, ND_PROG_EXPORTS,
    ND_PROG_INPUTS, ND_PROG_EMIT, ND_PROG_BUILD,
    ND_TEX_KIND, ND_TEX_FMT, ND_TEX_SURF, ND_TEX_TARGET, ND_W, ND_ROOM,
    ND_N
};

static const char *const nd_names[ND_N] = {
    "taken", "nothing to draw", "target in guest RAM", "target format",
    "target surface", "depth/MRT/resolve", "clip rectangles",
    "blend with alpha not last", "program form", "program exports",
    "program inputs", "program not expressible",
    "program build failed", "texture kind", "texture format",
    "texture surface", "texture is the target", "1/w", "room",
};

/* the uniforms of hd5770_msl_native_fs() */
typedef struct NatU {
    float tsize[16][4];
    uint32_t tsel[16][4];
    uint32_t twrap[16][4];
    float tborder[16][4];
    float vp[4];
    float alpha_ref;
    uint32_t alpha_func;
    float pc;
    uint32_t persp;
} NatU;

typedef struct NatSurf {
    HD5770Surf s;
    uint8_t space;
    uint64_t voff;
    uint32_t *pages;
    uint32_t npages, pg0;
    uint64_t lo, hi;            /* its bytes in VRAM (space 0) */
    id<MTLTexture> tex;
    uint64_t hash;
    bool have;                  /* tex holds the bytes hashed */
    bool checked;               /* against memory since the last sync */
    bool dirty;                 /* tex is newer than memory */
    int dx0, dy0, dx1, dy1;
    uint64_t used;
} NatSurf;

typedef struct NatProg {
    id<MTLFunction> vfn, ffn;
} NatProg;

typedef struct HD5770Mtlr {
    HD5770Gpu pub;
    id<MTLDevice> dev;
    id<MTLCommandQueue> q;
    uint8_t *vram, *ram;
    uint64_t vram_size, ram_size;
    id<MTLBuffer> ring;
    size_t ring_at;
    id<MTLSamplerState> samp[2];
    GHashTable *progs, *psos;
    GPtrArray *surfs;
    GArray *ranges;
    id<MTLCommandBuffer> last;
    id<MTLCommandBuffer> cb;
    id<MTLRenderCommandEncoder> enc;
    NatSurf *cur;
    uint64_t epoch;
    uint8_t *tmp;
    size_t tmp_len;
    uint64_t *offs;
    size_t offs_len;
    /* statistics */
    uint64_t nd[ND_N], tris, passes, waits, uploads, upload_bytes;
    uint64_t wbacks, wback_bytes, hashes, hash_bytes, syncs;
    uint64_t builds, pso_builds, build_ns, draw_build_ns;
    int64_t draw_ns, sync_ns;
    GArray *build_ms;
} HD5770Mtlr;

static void nat_exact_math(MTLCompileOptions *o)
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
    /*
     * the lowest version with barycentric_coord: later ones spend 50 ms
     * per library on the texture types
     */
    o.languageVersion = MTLLanguageVersion2_2;
}

static bool nat_decline(HD5770Mtlr *m, unsigned r, unsigned why)
{
    m->nd[r]++;
    m->pub.why = why;
    return false;
}

static uint64_t nat_hash_bytes(uint64_t h, const uint8_t *p, size_t n)
{
    uint64_t w;

    for (; n >= 8; n -= 8, p += 8) {
        memcpy(&w, p, 8);
        h = (h ^ w) * 0x9e3779b97f4a7c15ull;
        h ^= h >> 29;
    }
    if (n) {
        w = 0;
        memcpy(&w, p, n);
        h = (h ^ w) * 0x9e3779b97f4a7c15ull;
        h ^= h >> 29;
    }
    return h;
}

static uint64_t nat_hash(HD5770Mtlr *m, const NatSurf *e)
{
    uint64_t h = 0x243f6a8885a308d3ull, sz, at;

    m->hashes++;
    if (!e->space) {
        m->hash_bytes += e->hi - e->lo;
        return nat_hash_bytes(h, m->vram + e->lo, e->hi - e->lo);
    }
    sz = hd5770_surf_size(&e->s);
    for (at = 0; at < sz; ) {
        uint64_t q = at + e->pg0;
        size_t n = MIN(sz - at, 4096 - (q & 4095));

        h = nat_hash_bytes(h, m->ram + e->pages[q >> 12] + (q & 4095), n);
        at += n;
    }
    m->hash_bytes += sz;
    return h;
}

static void nat_end_pass(HD5770Mtlr *m)
{
    if (m->enc) {
        [m->enc endEncoding];
        [m->enc release];
        m->enc = nil;
        [m->cb commit];
        [m->last release];
        m->last = m->cb;
        m->cb = nil;
    }
    m->cur = NULL;
}

/* everything encoded is done */
static void nat_wait(HD5770Mtlr *m)
{
    nat_end_pass(m);
    if (m->last) {
        [m->last waitUntilCompleted];
        [m->last release];
        m->last = nil;
        m->waits++;
    }
    m->ring_at = 0;
}

static void *nat_tmp(HD5770Mtlr *m, size_t n)
{
    if (m->tmp_len < n) {
        m->tmp = g_realloc(m->tmp, n);
        m->tmp_len = n;
    }
    return m->tmp;
}

static uint64_t *nat_offs(HD5770Mtlr *m, size_t n)
{
    if (m->offs_len < n) {
        m->offs = g_realloc(m->offs, n * sizeof(uint64_t));
        m->offs_len = n;
    }
    return m->offs;
}

static inline uint8_t *nat_at(HD5770Mtlr *m, const NatSurf *e, uint64_t o)
{
    if (e->space) {
        uint64_t q = o + e->pg0;

        return m->ram + e->pages[q >> 12] + (q & 4095);
    }
    return m->vram + e->voff + o;
}

/* memory to the host texture */
static void nat_upload(HD5770Mtlr *m, NatSurf *e)
{
    uint32_t w = e->s.pitch, h = e->s.height, x, y;
    uint32_t *px;
    uint64_t *off;

    nat_wait(m);
    if (!e->space && e->s.mode < HD5770_ARRAY_1D_TILED_THIN1) {
        [e->tex replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0
                    withBytes:m->vram + e->voff bytesPerRow:e->s.row_bytes];
    } else {
        px = nat_tmp(m, (size_t)w * h * 4);
        off = nat_offs(m, w);
        for (y = 0; y < h; y++) {
            hd5770_surf_row(&e->s, 0, y, w, off);
            for (x = 0; x < w; x++) {
                memcpy(&px[(size_t)y * w + x], nat_at(m, e, off[x]), 4);
            }
        }
        [e->tex replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0
                    withBytes:px bytesPerRow:w * 4];
    }
    m->uploads++;
    m->upload_bytes += (uint64_t)w * h * 4;
}

static bool nat_overlap(const NatSurf *a, const NatSurf *b)
{
    return !a->space && !b->space && a->lo < b->hi && b->lo < a->hi;
}

/* the host texture's newer pixels to VRAM */
static void nat_writeback(HD5770Mtlr *m, NatSurf *e)
{
    uint32_t w, h, x, y;
    uint32_t *px;
    uint64_t *off, lo, hi, r[2];
    unsigned i;

    if (!e->dirty) {
        return;
    }
    nat_wait(m);
    w = e->dx1 - e->dx0;
    h = e->dy1 - e->dy0;
    px = nat_tmp(m, (size_t)w * h * 4);
    off = nat_offs(m, w);
    [e->tex getBytes:px bytesPerRow:w * 4
          fromRegion:MTLRegionMake2D(e->dx0, e->dy0, w, h) mipmapLevel:0];
    for (y = 0; y < h; y++) {
        hd5770_surf_row(&e->s, e->dx0, e->dy0 + y, w, off);
        for (x = 0; x < w; x++) {
            memcpy(m->vram + e->voff + off[x], &px[(size_t)y * w + x], 4);
        }
    }
    hd5770_surf_rows(&e->s, e->dy0, e->dy1, &lo, &hi);
    r[0] = lo - e->s.base + e->voff;
    r[1] = MIN(hi - e->s.base + e->voff, m->vram_size);
    g_array_append_vals(m->ranges, r, 2);
    e->dirty = false;
    e->hash = nat_hash(m, e);
    m->wbacks++;
    m->wback_bytes += (uint64_t)w * h * 4;
    for (i = 0; i < m->surfs->len; i++) {
        NatSurf *f = g_ptr_array_index(m->surfs, i);

        if (f != e && nat_overlap(f, e)) {
            f->checked = false;
        }
    }
}

static void nat_surf_free(gpointer p)
{
    NatSurf *e = p;

    [e->tex release];
    g_free(e->pages);
    g_free(e);
}

static bool nat_same(const NatSurf *e, const HD5770GpuSurf *s)
{
    const HD5770Surf *a = &e->s, *b = &s->s;

    if (e->space != s->space || a->pitch != b->pitch ||
        a->height != b->height || a->mode != b->mode || a->bpe != b->bpe ||
        a->banks != b->banks || a->bankw != b->bankw ||
        a->bankh != b->bankh || a->mtilea != b->mtilea ||
        a->tile_split != b->tile_split || a->pipes != b->pipes ||
        a->pipe_interleave != b->pipe_interleave ||
        a->nondisp != b->nondisp) {
        return false;
    }
    if (!s->space) {
        return e->voff == s->voff;
    }
    return e->npages == s->npages && e->pg0 == s->pg0 &&
           !memcmp(e->pages, s->pages, s->npages * sizeof(uint32_t));
}

/* a surface the pipeline can hold: its entry, else NULL */
static NatSurf *nat_surf(HD5770Mtlr *m, const HD5770GpuSurf *s)
{
    MTLTextureDescriptor *td;
    NatSurf *e;
    uint64_t sz = hd5770_surf_size(&s->s);
    unsigned i;

    if (s->s.bpe != 4 || !s->s.fast || !s->s.pitch || !s->s.height ||
        s->s.pitch > 16384 || s->s.height > 16384 ||
        !hd5770_surf_mode_ok(s->s.mode)) {
        return NULL;
    }
    if (s->space) {
        if (!m->ram || (s->pg0 & 3) ||
            (sz + s->pg0 + 4095) / 4096 > s->npages) {
            return NULL;
        }
        for (i = 0; i < s->npages; i++) {
            if ((uint64_t)s->pages[i] + 4096 > m->ram_size) {
                return NULL;
            }
        }
    } else if (s->voff > m->vram_size || sz > m->vram_size - s->voff) {
        return NULL;
    }
    for (i = 0; i < m->surfs->len; i++) {
        e = g_ptr_array_index(m->surfs, i);
        if (nat_same(e, s)) {
            e->used = m->epoch;
            return e;
        }
    }
    e = g_new0(NatSurf, 1);
    e->s = s->s;
    e->space = s->space;
    e->voff = s->voff;
    e->lo = s->voff;
    e->hi = s->voff + sz;
    if (s->space) {
        e->pages = g_memdup2(s->pages, s->npages * sizeof(uint32_t));
        e->npages = s->npages;
        e->pg0 = s->pg0;
    }
    td = [MTLTextureDescriptor
          texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                       width:s->s.pitch height:s->s.height
                                   mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    e->tex = [m->dev newTextureWithDescriptor:td];
    if (!e->tex) {
        g_free(e->pages);
        g_free(e);
        return NULL;
    }
    e->used = m->epoch;
    g_ptr_array_add(m->surfs, e);
    return e;
}

/* the entry's texture holds what memory and the draws queued make of it */
static void nat_ready(HD5770Mtlr *m, NatSurf *e)
{
    unsigned i;

    for (i = 0; i < m->surfs->len; i++) {
        NatSurf *f = g_ptr_array_index(m->surfs, i);

        if (f != e && f->dirty && nat_overlap(f, e)) {
            nat_writeback(m, f);
        }
    }
    if (!e->checked && !e->dirty) {
        uint64_t h = nat_hash(m, e);

        if (!e->have || h != e->hash) {
            nat_upload(m, e);
            e->hash = h;
            e->have = true;
        }
    }
    e->checked = true;
}

static void nat_prog_free(gpointer p)
{
    NatProg *pr = p;

    if (pr) {
        [pr->vfn release];
        [pr->ffn release];
        g_free(pr);
    }
}

static void nat_built(HD5770Mtlr *m, int64_t t0)
{
    int64_t ns = get_clock() - t0;

    m->build_ns += ns;
    m->draw_build_ns += ns;
}

static NatProg *nat_prog(HD5770Mtlr *m, const HD5770IR *ir,
                         const HD5770NatKey *k, const char *key)
{
    NatProg *pr = NULL;
    gpointer v;
    char *src;
    int64_t t0;

    if (g_hash_table_lookup_extended(m->progs, key, NULL, &v)) {
        return v;
    }
    src = hd5770_msl_native_fs(ir, k);
    if (src) {
        MTLCompileOptions *o = [[MTLCompileOptions alloc] init];
        NSError *e = nil;
        id<MTLLibrary> lib;

        nat_exact_math(o);
        t0 = get_clock();
        lib = [m->dev newLibraryWithSource:[NSString stringWithUTF8String:src]
                                   options:o error:&e];
        nat_built(m, t0);
        m->builds++;
        [o release];
        if (lib) {
            pr = g_new0(NatProg, 1);
            pr->vfn = [lib newFunctionWithName:@"hd_nvs"];
            pr->ffn = [lib newFunctionWithName:@"hd_nfs"];
            [lib release];
        } else {
            fprintf(stderr, "ati-hd5770: native program %016" PRIx64
                    " not built: %s\n", ir->hash,
                    e.localizedDescription.UTF8String);
        }
        g_free(src);
    }
    g_hash_table_insert(m->progs, g_strdup(key), pr);
    return pr;
}

static MTLBlendFactor nat_factor(unsigned f)
{
    switch (f) {
    case 1: return MTLBlendFactorOne;
    case 2: return MTLBlendFactorSourceColor;
    case 3: return MTLBlendFactorOneMinusSourceColor;
    case 4: return MTLBlendFactorSourceAlpha;
    case 5: return MTLBlendFactorOneMinusSourceAlpha;
    case 6: return MTLBlendFactorDestinationAlpha;
    case 7: return MTLBlendFactorOneMinusDestinationAlpha;
    case 8: return MTLBlendFactorDestinationColor;
    case 9: return MTLBlendFactorOneMinusDestinationColor;
    case 10: return MTLBlendFactorSourceAlphaSaturated;
    case 13: return MTLBlendFactorBlendColor;
    case 14: return MTLBlendFactorOneMinusBlendColor;
    case 19: return MTLBlendFactorBlendAlpha;
    case 20: return MTLBlendFactorOneMinusBlendAlpha;
    default: return MTLBlendFactorZero;
    }
}

/* combine(): function fn of CB_BLEND_CONTROL with its factors */
static void nat_blend(unsigned fn, unsigned sf, unsigned df,
                      MTLBlendOperation *op, MTLBlendFactor *s,
                      MTLBlendFactor *d)
{
    *s = nat_factor(sf);
    *d = nat_factor(df);
    switch (fn) {
    case 0: *op = MTLBlendOperationAdd; break;
    case 1: *op = MTLBlendOperationSubtract; break;
    case 2: *op = MTLBlendOperationMin; break;
    case 3: *op = MTLBlendOperationMax; break;
    case 4: *op = MTLBlendOperationReverseSubtract; break;
    default:
        *op = MTLBlendOperationAdd;
        *s = MTLBlendFactorOne;
        *d = MTLBlendFactorZero;
        break;
    }
}

static id<MTLRenderPipelineState> nat_pso(HD5770Mtlr *m, NatProg *pr,
                                          const char *pkey, uint32_t blend,
                                          unsigned mask)
{
    MTLRenderPipelineDescriptor *d;
    MTLRenderPipelineColorAttachmentDescriptor *c;
    id<MTLRenderPipelineState> p;
    NSError *e = nil;
    char key[160];
    gpointer v;
    int64_t t0;

    snprintf(key, sizeof(key), "%s/%08x/%x", pkey, blend, mask);
    if (g_hash_table_lookup_extended(m->psos, key, NULL, &v)) {
        return v;
    }
    d = [[MTLRenderPipelineDescriptor alloc] init];
    d.vertexFunction = pr->vfn;
    d.fragmentFunction = pr->ffn;
    c = d.colorAttachments[0];
    c.pixelFormat = MTLPixelFormatRGBA8Unorm;
    c.writeMask = (mask & 1 ? MTLColorWriteMaskRed : 0) |
                  (mask & 2 ? MTLColorWriteMaskGreen : 0) |
                  (mask & 4 ? MTLColorWriteMaskBlue : 0) |
                  (mask & 8 ? MTLColorWriteMaskAlpha : 0);
    if (blend & (1u << 30)) {
        unsigned csrc = blend & 0x1f, cfn = (blend >> 5) & 7;
        unsigned cdst = (blend >> 8) & 0x1f;
        unsigned asrc = csrc, afn = cfn, adst = cdst;
        MTLBlendOperation op;
        MTLBlendFactor sf, df;

        if ((blend >> 29) & 1) {
            asrc = (blend >> 16) & 0x1f;
            afn = (blend >> 21) & 7;
            adst = (blend >> 24) & 0x1f;
        }
        c.blendingEnabled = YES;
        nat_blend(cfn, csrc, cdst, &op, &sf, &df);
        c.rgbBlendOperation = op;
        c.sourceRGBBlendFactor = sf;
        c.destinationRGBBlendFactor = df;
        nat_blend(afn, asrc, adst, &op, &sf, &df);
        c.alphaBlendOperation = op;
        c.sourceAlphaBlendFactor = sf;
        c.destinationAlphaBlendFactor = df;
    }
    t0 = get_clock();
    p = [m->dev newRenderPipelineStateWithDescriptor:d error:&e];
    nat_built(m, t0);
    m->pso_builds++;
    [d release];
    if (!p) {
        fprintf(stderr, "ati-hd5770: native pipeline not built: %s\n",
                e.localizedDescription.UTF8String);
    }
    g_hash_table_insert(m->psos, g_strdup(key), p);
    return p;
}

static void nat_pso_free(gpointer p)
{
    [(id)p release];
}

/* n bytes of the ring, aligned; NULL when it has no room left */
static void *nat_ring(HD5770Mtlr *m, size_t n, size_t *at)
{
    size_t a = ROUND_UP(m->ring_at, 256);

    if (a + n > NAT_RING) {
        return NULL;
    }
    *at = a;
    m->ring_at = a + n;
    return (uint8_t *)m->ring.contents + a;
}

static bool nat_draw_locked(HD5770Mtlr *m, const HD5770GpuDraw *d)
{
    const HD5770IR *ir = d->ir;
    HD5770NatKey k = { 0 };
    NatSurf *rt, *ts[HD5770_GPU_TEX];
    NatProg *pr;
    id<MTLRenderPipelineState> pso;
    NatU *u;
    float *vb;
    uint32_t *kv;
    size_t o_u, o_v, o_k, vstride = 4 + 4 * (size_t)d->ninputs, need;
    uint32_t blend, perm = 0;
    unsigned i, j, c, mask = 0;
    int x0, y0, x1, y1, bx0 = INT_MAX, by0 = INT_MAX, bx1 = -1, by1 = -1;
    char pkey[96];

    if (d->db || d->nmrt || d->resolve || d->zexport || d->sexport) {
        return nat_decline(m, ND_STATE_DB, HD5770_WHY_BK_NATIVE_STATE);
    }
    if (!ir || ir->cf || ir->vs) {
        return nat_decline(m, ND_PROG_FORM, HD5770_WHY_BK_PROG);
    }
    if (ir->zexport || ir->cmask != 1) {
        return nat_decline(m, ND_PROG_EXPORTS, HD5770_WHY_BK_PROG);
    }
    if (d->nvb || d->ninputs > 32 || ir->nres != d->ntex) {
        return nat_decline(m, ND_PROG_INPUTS, HD5770_WHY_BK_PROG);
    }
    if (d->rt.space) {
        return nat_decline(m, ND_RT_SPACE, HD5770_WHY_BK_SURF);
    }
    for (i = 0; i < 4; i++) {
        perm |= d->rt_src[i] < 4 ? 1u << d->rt_src[i] : 0x10;
    }
    if (!d->rt_unorm8 || d->rt_comps != 4 || d->rt_float || perm != 0xf) {
        return nat_decline(m, ND_RT_FMT, HD5770_WHY_BK_SURF);
    }
    if (d->cliprule != 0xffff) {
        return nat_decline(m, ND_CLIPRECT, HD5770_WHY_BK_NATIVE_STATE);
    }
    blend = (d->blend & (1u << 30)) && d->rt_blend ? d->blend : 0;
    if (blend && d->rt_src[3] != 3) {
        return nat_decline(m, ND_BLEND_ORDER, HD5770_WHY_BK_NATIVE_STATE);
    }
    for (i = 0; i < 4; i++) {
        mask |= ((d->wmask >> d->rt_src[i]) & 1) << i;
    }
    for (i = 0; i < d->ntex; i++) {
        const HD5770GpuTex *t = &d->tex[i];

        if (t->dim != 1 || t->mip || t->bc || t->slices > 1) {
            return nat_decline(m, ND_TEX_KIND, HD5770_WHY_BK_SURF);
        }
        if (!t->unorm8) {
            return nat_decline(m, ND_TEX_FMT, HD5770_WHY_BK_SURF);
        }
    }
    for (i = 0; i < d->ntri; i++) {
        const HD5770GpuTri *t = &d->tri[i];

        for (j = 0; j < 3 && d->persp; j++) {
            if (!(t->w[j] > 0.0f) || isinf(t->w[j])) {
                return nat_decline(m, ND_W, HD5770_WHY_BK_NATIVE_STATE);
            }
        }
        bx0 = MIN(bx0, t->x0);
        by0 = MIN(by0, t->y0);
        bx1 = MAX(bx1, t->x1 + 1);
        by1 = MAX(by1, t->y1 + 1);
    }
    rt = nat_surf(m, &d->rt);
    if (!rt) {
        return nat_decline(m, ND_RT_SURF, HD5770_WHY_BK_SURF);
    }
    x0 = MAX(MAX(d->sc[0], bx0), 0);
    y0 = MAX(MAX(d->sc[1], by0), 0);
    x1 = MIN(MIN(d->sc[2], bx1), (int)rt->s.pitch);
    y1 = MIN(MIN(d->sc[3], by1), (int)rt->s.height);
    if (!d->ntri || x0 >= x1 || y0 >= y1 || !mask) {
        m->nd[ND_EMPTY]++;
        return true;
    }
    for (i = 0; i < d->ntex; i++) {
        ts[i] = nat_surf(m, &d->tex[i].surf);
        if (!ts[i] || d->tex[i].w > ts[i]->s.pitch ||
            d->tex[i].h > ts[i]->s.height) {
            return nat_decline(m, ND_TEX_SURF, HD5770_WHY_BK_SURF);
        }
        if (ts[i] == rt || nat_overlap(ts[i], rt)) {
            return nat_decline(m, ND_TEX_TARGET, HD5770_WHY_BK_SURF);
        }
    }
    k.ninputs = d->ninputs;
    k.pos_gpr = d->pos_gpr >= 0 && d->pos_gpr < 127 ? d->pos_gpr : -1;
    k.flat = d->flat;
    for (i = 0; i < 4; i++) {
        k.rt_src[i] = d->rt_src[i];
    }
    snprintf(pkey, sizeof(pkey), "%016" PRIx64 "/%u/%d/%08x/%u%u%u%u/%u",
             ir->hash, k.ninputs, k.pos_gpr, k.flat, k.rt_src[0],
             k.rt_src[1], k.rt_src[2], k.rt_src[3], ir->ninsts);
    pr = nat_prog(m, ir, &k, pkey);
    if (!pr) {
        return nat_decline(m, ND_PROG_EMIT, HD5770_WHY_BK_PROG);
    }
    pso = nat_pso(m, pr, pkey, blend, mask);
    if (!pso) {
        return nat_decline(m, ND_PROG_BUILD, HD5770_WHY_BK_PROG);
    }
    need = sizeof(NatU) + (size_t)d->ntri * 3 * vstride * sizeof(float) +
           MAX(ir->nconsts, 1u) * sizeof(uint32_t) + 3 * 256;
    if (need > NAT_RING) {
        return nat_decline(m, ND_ROOM, HD5770_WHY_BK_ROOM);
    }
    /* uploads and write-backs wait for the GPU and empty the ring */
    for (i = 0; i < d->ntex; i++) {
        nat_ready(m, ts[i]);
    }
    nat_ready(m, rt);
    if (m->ring_at + need > NAT_RING) {
        nat_wait(m);
    }
    u = nat_ring(m, sizeof(NatU), &o_u);
    vb = nat_ring(m, (size_t)d->ntri * 3 * vstride * sizeof(float), &o_v);
    kv = nat_ring(m, MAX(ir->nconsts, 1u) * sizeof(uint32_t), &o_k);
    memset(u, 0, sizeof(*u));
    for (i = 0; i < d->ntex; i++) {
        const HD5770GpuTex *t = &d->tex[i];

        u->tsize[i][0] = t->w;
        u->tsize[i][1] = t->h;
        u->tsize[i][2] = ts[i]->s.pitch;
        u->tsize[i][3] = ts[i]->s.height;
        u->twrap[i][0] = t->samp[0] & 7;
        u->twrap[i][1] = (t->samp[0] >> 3) & 7;
        for (c = 0; c < 4; c++) {
            u->tsel[i][c] = t->dst_sel[c];
            u->tborder[i][c] = t->border[c];
        }
    }
    u->vp[0] = 2.0f / rt->s.pitch;
    u->vp[1] = 2.0f / rt->s.height;
    u->alpha_ref = d->alpha_ref;
    u->alpha_func = d->alpha_test ? d->alpha_func : 7;
    u->pc = d->pix_center;
    u->persp = d->persp;
    for (i = 0; i < d->ntri; i++) {
        const HD5770GpuTri *t = &d->tri[i];
        const float *prm = d->prm + (size_t)i * d->ninputs * 12;

        for (j = 0; j < 3; j++) {
            float *v = vb + ((size_t)i * 3 + j) * vstride;

            v[0] = t->x[j];
            v[1] = t->y[j];
            v[2] = t->z[j];
            v[3] = t->w[j];
            for (c = 0; c < d->ninputs; c++) {
                memcpy(v + 4 + 4 * c,
                       prm + c * 12 + ((d->flat >> c) & 1 ? 0 : j * 4),
                       4 * sizeof(float));
            }
        }
    }
    if (ir->nconsts) {
        memcpy(kv, d->kv, ir->nconsts * sizeof(uint32_t));
    }
    if (m->cur != rt) {
        MTLRenderPassDescriptor *rp;

        nat_end_pass(m);
        rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = rt->tex;
        rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        m->cb = [[m->q commandBuffer] retain];
        m->enc = [[m->cb renderCommandEncoderWithDescriptor:rp] retain];
        [m->enc setCullMode:MTLCullModeNone];
        m->cur = rt;
        m->passes++;
    }
    [m->enc setRenderPipelineState:pso];
    [m->enc setScissorRect:(MTLScissorRect){ x0, y0, x1 - x0, y1 - y0 }];
    [m->enc setBlendColorRed:d->blend_const[d->rt_src[0]]
                       green:d->blend_const[d->rt_src[1]]
                        blue:d->blend_const[d->rt_src[2]]
                       alpha:d->blend_const[d->rt_src[3]]];
    [m->enc setVertexBuffer:m->ring offset:o_v atIndex:0];
    [m->enc setVertexBuffer:m->ring offset:o_u atIndex:1];
    [m->enc setFragmentBuffer:m->ring offset:o_u atIndex:0];
    [m->enc setFragmentBuffer:m->ring offset:o_k atIndex:1];
    for (i = 0; i < d->ntex; i++) {
        unsigned s0 = d->tex[i].samp[0];

        [m->enc setFragmentTexture:ts[i]->tex atIndex:i];
        [m->enc setFragmentSamplerState:
                    m->samp[((s0 >> 9) & 1) || ((s0 >> 11) & 1)] atIndex:i];
    }
    [m->enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0
               vertexCount:d->ntri * 3];
    if (!rt->dirty) {
        rt->dirty = true;
        rt->dx0 = x0;
        rt->dy0 = y0;
        rt->dx1 = x1;
        rt->dy1 = y1;
    } else {
        rt->dx0 = MIN(rt->dx0, x0);
        rt->dy0 = MIN(rt->dy0, y0);
        rt->dx1 = MAX(rt->dx1, x1);
        rt->dy1 = MAX(rt->dy1, y1);
    }
    m->nd[ND_TAKEN]++;
    m->tris += d->ntri;
    return true;
}

static bool nat_draw(void *opaque, const HD5770GpuDraw *d)
{
    HD5770Mtlr *m = opaque;
    int64_t t0 = get_clock();
    bool ok;

    m->draw_build_ns = 0;
    @autoreleasepool {
        ok = nat_draw_locked(m, d);
    }
    if (m->draw_build_ns) {
        /* what the draw waited for its program and pipeline */
        float ms = m->draw_build_ns / 1e6f;

        g_array_append_val(m->build_ms, ms);
    }
    m->draw_ns += get_clock() - t0;
    return ok;
}

static void nat_sync(void *opaque)
{
    HD5770Mtlr *m = opaque;
    int64_t t0 = get_clock();
    uint64_t bytes = 0;
    unsigned i;

    @autoreleasepool {
        nat_wait(m);
        for (i = 0; i < m->surfs->len; i++) {
            nat_writeback(m, g_ptr_array_index(m->surfs, i));
        }
    }
    for (i = 0; i + 1 < m->ranges->len; i += 2) {
        if (m->pub.done) {
            m->pub.done(m->pub.opaque, g_array_index(m->ranges, uint64_t, i),
                        g_array_index(m->ranges, uint64_t, i + 1));
        }
    }
    g_array_set_size(m->ranges, 0);
    for (i = 0; i < m->surfs->len; i++) {
        NatSurf *e = g_ptr_array_index(m->surfs, i);

        e->checked = false;
        bytes += (uint64_t)e->s.pitch * e->s.height * 4;
    }
    if (bytes > NAT_CACHE_MAX) {
        for (i = m->surfs->len; i-- > 0; ) {
            NatSurf *e = g_ptr_array_index(m->surfs, i);

            if (e->used != m->epoch) {
                g_ptr_array_remove_index_fast(m->surfs, i);
            }
        }
    }
    m->epoch++;
    m->syncs++;
    m->sync_ns += get_clock() - t0;
}

/* the CPU wrote memory: after a sync, so no entry is newer than it */
static void nat_wrote(void *opaque, uint64_t off, uint64_t len)
{
    HD5770Mtlr *m = opaque;
    unsigned i;

    for (i = 0; i < m->surfs->len; i++) {
        NatSurf *e = g_ptr_array_index(m->surfs, i);

        if (len ? !e->space && e->lo < off + len && off < e->hi
                : e->space) {
            e->checked = false;
        }
    }
}

static bool nat_ram(void *opaque, uint8_t *host, uint64_t size)
{
    HD5770Mtlr *m = opaque;

    m->ram = host;
    m->ram_size = size;
    return true;
}

static int nat_cmp_float(gconstpointer a, gconstpointer b)
{
    float x = *(const float *)a, y = *(const float *)b;

    return x < y ? -1 : x > y;
}

static void nat_stats(void *opaque, GString *out)
{
    HD5770Mtlr *m = opaque;
    GArray *s = g_array_copy(m->build_ms);
    unsigned i;

    g_array_sort(s, nat_cmp_float);
    g_string_append_printf(out, "native: %s draws %" PRIu64 " triangles %"
                           PRIu64 " declined:", m->dev.name.UTF8String,
                           m->nd[ND_TAKEN], m->tris);
    for (i = ND_EMPTY; i < ND_N; i++) {
        if (m->nd[i]) {
            g_string_append_printf(out, " %s %" PRIu64 ",", nd_names[i],
                                   m->nd[i]);
        }
    }
    g_string_append_printf(out, "\nnative: syncs %" PRIu64 " passes %" PRIu64
                           " waits %" PRIu64 " uploads %" PRIu64 " (%" PRIu64
                           " MB) write-backs %" PRIu64 " (%" PRIu64
                           " MB) hashes %" PRIu64 " (%" PRIu64 " MB) surfaces"
                           " %u draw ms %" PRId64 " sync ms %" PRId64 "\n",
                           m->syncs, m->passes, m->waits, m->uploads,
                           m->upload_bytes >> 20, m->wbacks,
                           m->wback_bytes >> 20, m->hashes,
                           m->hash_bytes >> 20, m->surfs->len,
                           m->draw_ns / 1000000, m->sync_ns / 1000000);
    g_string_append_printf(out, "native: programs %" PRIu64 " pipelines %"
                           PRIu64 " draws that built %u, ms: total %" PRIu64
                           " median %.1f worst %.1f\n", m->builds,
                           m->pso_builds, s->len,
                           m->build_ns / 1000000,
                           s->len ? g_array_index(s, float, s->len / 2) : 0.0f,
                           s->len ? g_array_index(s, float, s->len - 1) : 0.0f);
    g_array_free(s, true);
}

static void nat_close(void *opaque)
{
    HD5770Mtlr *m = opaque;

    @autoreleasepool {
        nat_sync(m);
    }
    g_ptr_array_free(m->surfs, true);
    g_hash_table_destroy(m->psos);
    g_hash_table_destroy(m->progs);
    g_array_free(m->ranges, true);
    g_array_free(m->build_ms, true);
    g_free(m->tmp);
    g_free(m->offs);
    [m->samp[0] release];
    [m->samp[1] release];
    [m->ring release];
    [m->q release];
    [m->dev release];
    g_free(m);
}

static const HD5770GpuOps nat_ops = {
    .name = "metal-native",
    .min_pixels = 1024,
    .draw = nat_draw,
    .sync = nat_sync,
    .stats = nat_stats,
    .close = nat_close,
    .ram = nat_ram,
    .wrote = nat_wrote,
};

HD5770Gpu *hd5770_mtlr_open(uint8_t *vram, uint64_t vram_size,
                            const char **err)
{
    HD5770Mtlr *m;
    unsigned i;

    *err = NULL;
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        MTLCompileOptions *o;
        id<MTLLibrary> lib;

        if (!dev) {
            *err = "no Metal device";
            return NULL;
        }
        m = g_new0(HD5770Mtlr, 1);
        m->dev = dev;
        m->vram = vram;
        m->vram_size = vram_size;
        m->q = [dev newCommandQueue];
        m->ring = [dev newBufferWithLength:NAT_RING
                                   options:MTLResourceStorageModeShared];
        for (i = 0; i < 2; i++) {
            MTLSamplerDescriptor *sd = [[MTLSamplerDescriptor alloc] init];

            sd.minFilter = i ? MTLSamplerMinMagFilterLinear
                             : MTLSamplerMinMagFilterNearest;
            sd.magFilter = sd.minFilter;
            sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
            sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
            m->samp[i] = [dev newSamplerStateWithDescriptor:sd];
            [sd release];
        }
        if (!m->q || !m->ring || !m->samp[0] || !m->samp[1]) {
            [m->samp[0] release];
            [m->samp[1] release];
            [m->ring release];
            [m->q release];
            [dev release];
            g_free(m);
            *err = "cannot set up the Metal pipeline";
            return NULL;
        }
        /* the compiler's first library takes 100 ms and more */
        o = [[MTLCompileOptions alloc] init];
        nat_exact_math(o);
        lib = [dev newLibraryWithSource:@"#include <metal_stdlib>\n"
               "using namespace metal;\n"
               "fragment float4 f(float4 p [[position]], texture2d<float> t,\n"
               "                  sampler s) { return t.sample(s, p.xy); }\n"
                                options:o error:NULL];
        [lib release];
        [o release];
    }
    m->progs = g_hash_table_new_full(g_str_hash, g_str_equal, g_free,
                                     nat_prog_free);
    m->psos = g_hash_table_new_full(g_str_hash, g_str_equal, g_free,
                                    nat_pso_free);
    m->surfs = g_ptr_array_new_with_free_func(nat_surf_free);
    m->ranges = g_array_new(false, false, sizeof(uint64_t));
    m->build_ms = g_array_new(false, false, sizeof(float));
    m->pub.ops = &nat_ops;
    m->pub.b = m;
    return &m->pub;
}
