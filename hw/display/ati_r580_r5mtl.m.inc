/*
 * ATI R580 3D engine, R500 pipeline on the host GPU: the Metal backend
 * (ati_r580_r5gpu.h). Included by ati_r580_mtl.m, whose command buffers
 * and staging arenas it shares.
 *
 * Images are R8Uint / R16Uint / R32Uint textures. The fragment stage
 * reads its destination as colour attachment 0 and the depth/stencil
 * buffer as colour attachment 1 (Apple GPUs), so a draw is one pass
 * however its primitives overlap. The shader text is the common dialect
 * wrapped in a struct, whose members stand for its globals.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "ati_r580_r5gpu.h"

#define R5MTL_PROGS     128
#define R5MTL_LUT_W     1024
#define R5MTL_LUT_H     10

struct R5GpuImage {
    id<MTLTexture> tex;
    int w, h, levels, bpp;
};

enum { R5MTL_EMPTY, R5MTL_PENDING, R5MTL_READY, R5MTL_FAILED };

typedef struct R5MtlProg {
    uint64_t key;
    int bpp;
    int state;                  /* under R5Mtl.lock */
    id<MTLRenderPipelineState> pso;
    uint64_t lru;
} R5MtlProg;

typedef struct R5MtlBlock {
    float kc[R5G_CONSTS][4];
    int32_t ki[R5G_KI][4];
    float kf[R5G_KF][4];
    int32_t ti[R5G_UNITS * R5G_TI][4];
    float tf[R5G_UNITS * R5G_TF][4];
} R5MtlBlock;

typedef struct R5MtlVa {
    float vp[4];
    uint32_t stride, pad[3];
} R5MtlVa;

typedef struct R5Mtl {
    R5MtlProg prog[R5MTL_PROGS];
    uint64_t tick;
    id<MTLTexture> lut;
    R5GpuImage *enc_target;     /* g->enc renders into it when set */
    R5GpuImage *enc_ztarget;    /* and into this as attachment 1 */
    QemuMutex lock;
    int inflight;               /* compiles not finished */
    uint64_t links, fails, draws, link_us, link_max;
    int reports;
} R5Mtl;

static const char r5mtl_head[] =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"#if __METAL_VERSION__ >= 320\n"
"#pragma METAL fp math_mode(safe)\n"
"#pragma METAL fp contract(off)\n"
"#endif\n"
"typedef float2 vec2;\n"
"typedef float3 vec3;\n"
"typedef float4 vec4;\n"
"typedef int2 ivec2;\n"
"typedef int3 ivec3;\n"
"typedef int4 ivec4;\n"
"typedef uint2 uvec2;\n"
"typedef uint3 uvec3;\n"
"typedef uint4 uvec4;\n"
"#define precise\n"
"#define floatBitsToUint(x) as_type<uint>(x)\n"
"#define uintBitsToFloat(x) as_type<float>(x)\n"
"#define R5_KC(i) K->kc[i]\n"
"#define R5_KI(i) K->ki[i]\n"
"#define R5_KF(i) K->kf[i]\n"
"#define R5_IN(n) in.fi##n\n"
"#define R5_LUT(nb, v) lut.read(uint2(uint(v), uint((nb) - 1))).r\n"
"#define R5_DFDX(e) dfdx(e)\n"
"#define R5_DFDY(e) dfdy(e)\n"
"#define R5_DST dst\n"
"#define R5_ZDST zdst\n"
"#define R5_FRAG in.pos.xy\n"
"#define R5_REC int(in.rec)\n"
"#define R5_ZREC int(in.zrec)\n"
"#define R5_PLANE(i) pl[i]\n"
"#define R5_THIS\n";

static void mtl_r5_close(R580MtlCtx *g)
{
    R5Mtl *r = g->r5;
    unsigned k;

    if (!r) {
        return;
    }
    while (qatomic_read(&r->inflight)) {
        g_usleep(1000);
    }
    qemu_mutex_destroy(&r->lock);
    for (k = 0; k < R5MTL_PROGS; k++) {
        [r->prog[k].pso release];
    }
    [r->lut release];
    g_free(r);
    g->r5 = NULL;
}

static R5Mtl *r5mtl_get(R580MtlCtx *g)
{
    R5Mtl *r = g->r5;
    MTLTextureDescriptor *td;
    float *lut;
    int nb, v;

    if (r) {
        return r;
    }
    r = g->r5 = g_new0(R5Mtl, 1);
    qemu_mutex_init(&r->lock);
    td = [MTLTextureDescriptor
          texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
          width:R5MTL_LUT_W height:R5MTL_LUT_H mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    r->lut = [g->dev newTextureWithDescriptor:td];
    lut = g_new0(float, R5MTL_LUT_W * R5MTL_LUT_H);
    for (nb = 1; nb <= R5MTL_LUT_H; nb++) {
        for (v = 0; v < (1 << nb) && v < R5MTL_LUT_W; v++) {
            lut[(nb - 1) * R5MTL_LUT_W + v] =
                (float)v / (float)((1u << nb) - 1);
        }
    }
    [r->lut replaceRegion:MTLRegionMake2D(0, 0, R5MTL_LUT_W, R5MTL_LUT_H)
              mipmapLevel:0 withBytes:lut
              bytesPerRow:R5MTL_LUT_W * sizeof(float)];
    g_free(lut);
    return r;
}

static MTLPixelFormat r5mtl_fmt(int bpp)
{
    return bpp == 1 ? MTLPixelFormatR8Uint :
           bpp == 2 ? MTLPixelFormatR16Uint : MTLPixelFormatR32Uint;
}

struct R5GpuImage *ati_r580_gl_r5_image(R580MtlCtx *g, int w, int h,
                                        int levels, int bpp)
{
    R5GpuImage *im;

    if (!g || w <= 0 || h <= 0 || w > 16384 || h > 16384 || levels < 1 ||
        levels > 16 || (bpp != 1 && bpp != 2 && bpp != 4)) {
        return NULL;
    }
    @autoreleasepool {
        MTLTextureDescriptor *td = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:r5mtl_fmt(bpp)
            width:w height:h mipmapped:NO];

        r5mtl_get(g);
        td.mipmapLevelCount = levels;
        td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        td.storageMode = MTLStorageModePrivate;
        im = g_new0(R5GpuImage, 1);
        im->tex = [g->dev newTextureWithDescriptor:td];
        if (!im->tex) {
            g_free(im);
            return NULL;
        }
    }
    im->w = w;
    im->h = h;
    im->levels = levels;
    im->bpp = bpp;
    return im;
}

void ati_r580_gl_r5_image_free(R580MtlCtx *g, struct R5GpuImage *im)
{
    if (!g || !im) {
        return;
    }
    if (g->r5 && (g->r5->enc_target == im || g->r5->enc_ztarget == im)) {
        mtl_end_enc(g);
        g->r5->enc_target = NULL;
        g->r5->enc_ztarget = NULL;
    }
    /* command buffers in flight hold their own references */
    [im->tex release];
    g_free(im);
}

bool ati_r580_gl_r5_upload(R580MtlCtx *g, struct R5GpuImage *im, int level,
                           int x, int y, int w, int h, const void *data,
                           unsigned stride)
{
    if (!g || !im || level < 0 || level >= im->levels || w <= 0 || h <= 0 ||
        x < 0 || y < 0 || x + w > MAX(im->w >> level, 1) ||
        y + h > MAX(im->h >> level, 1)) {
        return false;
    }
    @autoreleasepool {
        size_t row = (size_t)w * im->bpp, off;
        id<MTLBuffer> b = mtl_alloc(g, row * h, &off);
        uint8_t *st = (uint8_t *)b.contents + off;
        id<MTLBlitCommandEncoder> bl;
        int k;

        for (k = 0; k < h; k++) {
            memcpy(st + row * k, (const uint8_t *)data + (size_t)stride * k,
                   row);
        }
        bl = mtl_blit(g);
        [bl copyFromBuffer:b sourceOffset:off sourceBytesPerRow:row
            sourceBytesPerImage:row * h sourceSize:MTLSizeMake(w, h, 1)
            toTexture:im->tex destinationSlice:0 destinationLevel:level
            destinationOrigin:MTLOriginMake(x, y, 0)];
        [bl endEncoding];
    }
    return true;
}

bool ati_r580_gl_r5_read(R580MtlCtx *g, struct R5GpuImage *im, int x, int y,
                         int w, int h, void *data, unsigned stride)
{
    if (!g || !im || w <= 0 || h <= 0 || x < 0 || y < 0 ||
        x + w > im->w || y + h > im->h) {
        return false;
    }
    @autoreleasepool {
        size_t row = (size_t)w * im->bpp;
        id<MTLBuffer> b = [[g->dev newBufferWithLength:row * h
                            options:MTLResourceStorageModeShared] autorelease];
        id<MTLBlitCommandEncoder> bl = mtl_blit(g);
        int k;

        [bl copyFromTexture:im->tex sourceSlice:0 sourceLevel:0
               sourceOrigin:MTLOriginMake(x, y, 0)
                 sourceSize:MTLSizeMake(w, h, 1)
                   toBuffer:b destinationOffset:0
      destinationBytesPerRow:row destinationBytesPerImage:row * h];
        [bl endEncoding];
        mtl_finish(g);
        for (k = 0; k < h; k++) {
            memcpy((uint8_t *)data + (size_t)stride * k,
                   (const uint8_t *)b.contents + row * k, row);
        }
    }
    return !g->gpu_failed;
}

bool ati_r580_gl_r5_copy(R580MtlCtx *g, struct R5GpuImage *dst, int dx, int dy,
                         struct R5GpuImage *src, int sx, int sy, int w, int h)
{
    if (!g || !dst || !src || src->bpp != dst->bpp || w <= 0 || h <= 0 ||
        sx < 0 || sy < 0 || sx + w > src->w || sy + h > src->h ||
        dx < 0 || dy < 0 || dx + w > dst->w || dy + h > dst->h) {
        return false;
    }
    @autoreleasepool {
        id<MTLBlitCommandEncoder> bl = mtl_blit(g);

        [bl copyFromTexture:src->tex sourceSlice:0 sourceLevel:0
               sourceOrigin:MTLOriginMake(sx, sy, 0)
                 sourceSize:MTLSizeMake(w, h, 1)
                  toTexture:dst->tex destinationSlice:0 destinationLevel:0
          destinationOrigin:MTLOriginMake(dx, dy, 0)];
        [bl endEncoding];
    }
    return true;
}

bool ati_r580_gl_r5_copy_runs(R580MtlCtx *g, struct R5GpuImage *dst,
                              struct R5GpuImage *src, const int *runs,
                              unsigned n)
{
    const int *r;
    unsigned k;

    if (!g || !dst || !src || src->bpp != dst->bpp) {
        return false;
    }
    for (k = 0, r = runs; k < n; k++, r += 6) {
        if (r[4] <= 0 || r[5] <= 0 || r[0] < 0 || r[1] < 0 ||
            r[0] + r[4] > src->w || r[1] + r[5] > src->h || r[2] < 0 ||
            r[3] < 0 || r[2] + r[4] > dst->w || r[3] + r[5] > dst->h) {
            return false;
        }
    }
    @autoreleasepool {
        id<MTLBlitCommandEncoder> bl = mtl_blit(g);

        for (k = 0, r = runs; k < n; k++, r += 6) {
            [bl copyFromTexture:src->tex sourceSlice:0 sourceLevel:0
                   sourceOrigin:MTLOriginMake(r[0], r[1], 0)
                     sourceSize:MTLSizeMake(r[4], r[5], 1)
                      toTexture:dst->tex destinationSlice:0
               destinationLevel:0
              destinationOrigin:MTLOriginMake(r[2], r[3], 0)];
        }
        [bl endEncoding];
    }
    return true;
}

bool ati_r580_gl_r5_dst_fetch(R580MtlCtx *g)
{
    return true;
}

void ati_r580_gl_r5_stats(R580MtlCtx *g, uint64_t *progs, uint64_t *fails,
                          uint64_t *draws, uint64_t *link_us,
                          uint64_t *link_max)
{
    R5Mtl *r = g ? g->r5 : NULL;

    *progs = r ? r->links : 0;
    *fails = r ? r->fails : 0;
    *draws = r ? r->draws : 0;
    *link_us = r ? r->link_us : 0;
    *link_max = r ? r->link_max : 0;
}

static char *r5mtl_source(const R5GpuDraw *d)
{
    GString *o = g_string_new(r5mtl_head);
    unsigned n, u, k;

    g_string_append_printf(o,
        "struct R5K {\n"
        "    float4 kc[%d]; int4 ki[%d]; float4 kf[%d];\n"
        "    int4 ti[%d]; float4 tf[%d];\n"
        "};\n"
        "#define R5_TI(u, i) K->ti[(u) * %d + (i)]\n"
        "#define R5_TF(u, i) K->tf[(u) * %d + (i)]\n",
        R5G_CONSTS, R5G_KI, R5G_KF, R5G_UNITS * R5G_TI, R5G_UNITS * R5G_TF,
        R5G_TI, R5G_TF);
    g_string_append(o, "struct VOut {\n    float4 pos [[position]];\n"
                       "    float rec [[flat]];\n    float zrec [[flat]];\n");
    for (n = 0; n < d->ninputs; n++) {
        g_string_append_printf(o, "    float4 fi%u%s;\n", n,
                               (d->flat >> n) & 1 ? " [[flat]]" : "");
    }
    g_string_append_printf(o, "};\n"
        "struct VA { float4 vp; uint stride; };\n"
        "vertex VOut r5_vs(uint vid [[vertex_id]],\n"
        "                  const device float *vb [[buffer(0)]],\n"
        "                  constant VA &a [[buffer(1)]])\n"
        "{\n"
        "    const device float *v = vb + vid * a.stride;\n"
        "    VOut o;\n"
        "    float w = %s;\n"
        "    o.pos = float4((v[0] * a.vp.x - 1.0) * w, (1.0 - v[1] * a.vp.y) * w,"
        " 0.0, w);\n"
        "    o.rec = v[2];\n"
        "    o.zrec = v[3];\n", d->ztarget ? "1.0" : "v[3]");
    for (n = 0; n < d->ninputs; n++) {
        g_string_append_printf(o, "    o.fi%u = float4(v[%u], v[%u], v[%u], "
                               "v[%u]);\n", n, 4 + 4 * n, 5 + 4 * n,
                               6 + 4 * n, 7 + 4 * n);
    }
    g_string_append(o, "    return o;\n}\n");
    for (u = 0; u < R5G_UNITS; u++) {
        if (d->units & (1u << u)) {
            g_string_append_printf(o, "#define R5_FETCH%u(x, y, l) "
                "t%u.read(uint2(uint(x), uint(y)), uint(l)).r\n", u, u);
        }
    }
    g_string_append(o, "struct R5F {\n"
                       "    constant R5K *K;\n"
                       "    VOut in;\n"
                       "    uint dst;\n"
                       "    uint zdst;\n"
                       "    const device float4 *pl;\n"
                       "    texture2d<float> lut;\n");
    for (u = 0; u < R5G_UNITS; u++) {
        if (d->units & (1u << u)) {
            g_string_append_printf(o, "    texture2d<uint> t%u;\n", u);
        }
    }
    g_string_append(o, d->fs);
    g_string_append_printf(o, "};\n"
        "struct FOut { uint c [[color(0)]];%s };\n"
        "fragment FOut r5_fs(VOut in [[stage_in]],\n"
        "                    constant R5K &k [[buffer(0)]],\n"
        "                    const device float4 *pl [[buffer(1)]],\n"
        "%s"
        "                    uint dst [[color(0)]],\n",
        d->ztarget ? " uint z [[color(1)]];" : "",
        d->ztarget ? "                    uint zdst [[color(1)]],\n" : "");
    for (u = 0, k = 0; u < R5G_UNITS; u++) {
        if (d->units & (1u << u)) {
            g_string_append_printf(o, "                    texture2d<uint> t%u "
                                   "[[texture(%u)]],\n", u, k++);
        }
    }
    g_string_append_printf(o,
        "                    texture2d<float> lut [[texture(%d)]])\n"
        "{\n"
        "    R5F f;\n"
        "    f.K = &k;\n"
        "    f.in = in;\n"
        "    f.dst = dst;\n"
        "%s"
        "    f.pl = pl;\n"
        "    f.lut = lut;\n", R5G_MAX_BOUND,
        d->ztarget ? "    f.zdst = zdst;\n" : "    f.zdst = 0u;\n");
    for (u = 0; u < R5G_UNITS; u++) {
        if (d->units & (1u << u)) {
            g_string_append_printf(o, "    f.t%u = t%u;\n", u, u);
        }
    }
    g_string_append(o,
        "    FOut o;\n"
        "    o.c = f.om_pixel();\n"
        "    if (!f.us_keep) discard_fragment();\n");
    if (d->ztarget) {
        g_string_append(o, "    o.z = f.om_zs;\n");
    }
    g_string_append(o,
        "    return o;\n"
        "}\n");
    return g_string_free(o, false);
}

static void r5mtl_done(R5Mtl *r, R5MtlProg *pr, uint64_t key,
                       id<MTLRenderPipelineState> pso, NSError *e,
                       int64_t t0)
{
    qemu_mutex_lock(&r->lock);
    if (pr->state == R5MTL_PENDING && pr->key == key) {
        pr->pso = [pso retain];
        pr->state = pso ? R5MTL_READY : R5MTL_FAILED;
        t0 = g_get_monotonic_time() - t0;
        r->link_us += t0;
        r->link_max = MAX(r->link_max, (uint64_t)t0);
        if (pso) {
            r->links++;
        } else {
            r->fails++;
            if (r->reports++ < 4) {
                error_report("ati-radeon-x1900: R500 Metal program: %s",
                             e ? e.localizedDescription.UTF8String : "?");
            }
        }
    }
    qemu_mutex_unlock(&r->lock);
    qatomic_dec(&r->inflight);
}

/*
 * The pipeline for a request: 1 built, 0 being built on Metal's own
 * threads (the caller draws on the CPU meanwhile), -1 it will not build.
 */
static int r5mtl_prog(R580MtlCtx *g, R5Mtl *r, const R5GpuDraw *d, int bpp,
                      id<MTLRenderPipelineState> *out)
{
    R5MtlProg *pr, *victim = NULL;
    uint64_t key = d->key;
    unsigned k;
    int st, zbpp;

    *out = nil;
    qemu_mutex_lock(&r->lock);
    for (k = 0; k < R5MTL_PROGS; k++) {
        pr = &r->prog[k];
        if (pr->state != R5MTL_EMPTY && pr->key == key && pr->bpp == bpp) {
            pr->lru = ++r->tick;
            st = pr->state;
            *out = pr->pso;
            qemu_mutex_unlock(&r->lock);
            return st == R5MTL_READY ? 1 : st == R5MTL_PENDING ? 0 : -1;
        }
        if (pr->state != R5MTL_PENDING &&
            (!victim || pr->lru < victim->lru)) {
            victim = pr;
        }
    }
    if (!victim) {
        qemu_mutex_unlock(&r->lock);
        return 0;
    }
    pr = victim;
    [pr->pso release];
    memset(pr, 0, sizeof(*pr));
    pr->key = key;
    pr->bpp = bpp;
    pr->lru = ++r->tick;
    pr->state = R5MTL_PENDING;
    zbpp = d->ztarget ? d->ztarget->bpp : 0;
    qatomic_inc(&r->inflight);
    qemu_mutex_unlock(&r->lock);
    @autoreleasepool {
        int64_t t0 = g_get_monotonic_time();
        char *src = r5mtl_source(d);
        MTLCompileOptions *opt = [[MTLCompileOptions alloc] init];
        id<MTLDevice> dev = g->dev;

        mtl_exact_math(opt);
        [dev newLibraryWithSource:@(src) options:opt
                completionHandler:^(id<MTLLibrary> lib, NSError *e) {
            MTLRenderPipelineDescriptor *pd;
            id<MTLFunction> vf, ff;

            if (!lib) {
                r5mtl_done(r, pr, key, nil, e, t0);
                return;
            }
            pd = [[MTLRenderPipelineDescriptor alloc] init];
            vf = [lib newFunctionWithName:@"r5_vs"];
            ff = [lib newFunctionWithName:@"r5_fs"];
            pd.vertexFunction = vf;
            pd.fragmentFunction = ff;
            pd.colorAttachments[0].pixelFormat = r5mtl_fmt(bpp);
            if (zbpp) {
                pd.colorAttachments[1].pixelFormat = r5mtl_fmt(zbpp);
            }
            [dev newRenderPipelineStateWithDescriptor:pd
                completionHandler:^(id<MTLRenderPipelineState> pso,
                                    NSError *e2) {
                r5mtl_done(r, pr, key, pso, e2, t0);
            }];
            [vf release];
            [ff release];
            [pd release];
        }];
        [opt release];
        g_free(src);
    }
    return 0;
}

int ati_r580_gl_r5_prog_ready(R580MtlCtx *g, const struct R5GpuDraw *d,
                              int bpp)
{
    id<MTLRenderPipelineState> pso;

    if (!g) {
        return -1;
    }
    return r5mtl_prog(g, r5mtl_get(g), d, bpp, &pso);
}

static id<MTLRenderCommandEncoder> r5mtl_enc(R580MtlCtx *g, R5Mtl *r,
                                             R5GpuImage *t, R5GpuImage *zt)
{
    MTLRenderPassDescriptor *rp;

    if (g->enc && r->enc_target == t && r->enc_ztarget == zt) {
        return g->enc;
    }
    mtl_end_enc(g);
    mtl_cb(g);
    rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = t->tex;
    rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    if (zt) {
        rp.colorAttachments[1].texture = zt->tex;
        rp.colorAttachments[1].loadAction = MTLLoadActionLoad;
        rp.colorAttachments[1].storeAction = MTLStoreActionStore;
    }
    g->enc = [[g->cb renderCommandEncoderWithDescriptor:rp] retain];
    [g->enc setCullMode:MTLCullModeNone];
    g->enc_pso = nil;
    g->enc_dss = nil;
    g->enc_sref = -1;
    g->enc_vw = g->enc_vh = 0;
    r->enc_target = t;
    r->enc_ztarget = zt;
    g->passes++;
    return g->enc;
}

bool ati_r580_gl_r5_draw(R580MtlCtx *g, const struct R5GpuDraw *d)
{
    R5GpuImage *t = d ? d->target : NULL;
    unsigned u, k;
    R5Mtl *r;

    if (!g || !t || !d->nvert || d->sx1 <= d->sx0 || d->sy1 <= d->sy0 ||
        d->sx1 > t->w || d->sy1 > t->h) {
        return false;
    }
    @autoreleasepool {
        id<MTLRenderPipelineState> pso;
        id<MTLRenderCommandEncoder> enc;
        id<MTLBuffer> vb, ub;
        size_t voff, uoff, vlen;
        R5MtlBlock *blk;
        R5MtlVa va;
        MTLScissorRect sc;
        MTLViewport vp;

        r = r5mtl_get(g);
        if (r5mtl_prog(g, r, d, t->bpp, &pso) != 1) {
            return false;
        }
        vlen = (size_t)d->nvert * (4 + 4 * d->ninputs) * sizeof(float);
        vb = mtl_alloc(g, vlen, &voff);
        memcpy((uint8_t *)vb.contents + voff, d->verts, vlen);
        ub = mtl_alloc(g, sizeof(R5MtlBlock), &uoff);
        blk = (R5MtlBlock *)((uint8_t *)ub.contents + uoff);
        memcpy(blk->kc, d->consts, sizeof(blk->kc));
        memcpy(blk->ki, d->ki, sizeof(blk->ki));
        memcpy(blk->kf, d->kf, sizeof(blk->kf));
        memcpy(blk->ti, d->ti, sizeof(blk->ti));
        memcpy(blk->tf, d->tf, sizeof(blk->tf));
        memset(&va, 0, sizeof(va));
        va.vp[0] = 2.0f / t->w;
        va.vp[1] = 2.0f / t->h;
        va.stride = 4 + 4 * d->ninputs;

        enc = r5mtl_enc(g, r, t, d->ztarget);
        [enc setRenderPipelineState:pso];
        [enc setVertexBuffer:vb offset:voff atIndex:0];
        [enc setVertexBytes:&va length:sizeof(va) atIndex:1];
        [enc setFragmentBuffer:ub offset:uoff atIndex:0];
        if (d->planes) {
            size_t plen = (size_t)d->nplanes * 16;
            size_t poff;
            id<MTLBuffer> pb = mtl_alloc(g, plen, &poff);

            memcpy((uint8_t *)pb.contents + poff, d->planes, plen);
            [enc setFragmentBuffer:pb offset:poff atIndex:1];
        } else {
            [enc setFragmentBuffer:ub offset:uoff atIndex:1];
        }
        for (u = 0, k = 0; u < R5G_UNITS; u++) {
            if (d->units & (1u << u)) {
                if (k == R5G_MAX_BOUND) {
                    return false;
                }
                [enc setFragmentTexture:d->img[u]->tex atIndex:k++];
            }
        }
        [enc setFragmentTexture:r->lut atIndex:R5G_MAX_BOUND];
        vp.originX = 0;
        vp.originY = 0;
        vp.width = t->w;
        vp.height = t->h;
        vp.znear = 0;
        vp.zfar = 1;
        [enc setViewport:vp];
        sc.x = d->sx0;
        sc.y = d->sy0;
        sc.width = d->sx1 - d->sx0;
        sc.height = d->sy1 - d->sy0;
        [enc setScissorRect:sc];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0
                vertexCount:d->nvert];
        r->draws++;
        if (++g->cb_draws >= MTL_CB_DRAWS) {
            mtl_commit(g);
        }
    }
    return !g->gpu_failed;
}
