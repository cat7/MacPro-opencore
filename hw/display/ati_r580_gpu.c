/*
 * ATI R300/R350 -- host GPU backend dispatch on macOS.
 *
 * Based on lyons88/qemu-g5 934422f4f9.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "ati_r580_gpu.h"

struct R580GlCtx {
    bool mtl;
    void *impl;
};

#define OGL(g) ((R580OglCtx *)(g)->impl)
#define MTL(g) ((R580MtlCtx *)(g)->impl)
#define FWD(g, fn, ...) \
    ((g)->mtl ? r580_mtl_##fn(MTL(g), ##__VA_ARGS__) \
              : r580_ogl_##fn(OGL(g), ##__VA_ARGS__))

static R580GlCtx *gpu_wrap(bool mtl, void *impl)
{
    R580GlCtx *g;

    if (!impl) {
        return NULL;
    }
    g = g_new0(R580GlCtx, 1);
    g->mtl = mtl;
    g->impl = impl;
    return g;
}

R580GlCtx *ati_r580_gl_open(const char **err, unsigned workers)
{
    return gpu_wrap(false, r580_ogl_open(err, workers));
}

R580GlCtx *ati_r580_gl_open_metal(const char **err, unsigned workers)
{
    return gpu_wrap(true, r580_mtl_open(err, workers));
}

void ati_r580_gl_close(R580GlCtx *g)
{
    if (g) {
        if (g->mtl) {
            r580_mtl_close(MTL(g));
        } else {
            r580_ogl_close(OGL(g));
        }
        g_free(g);
    }
}

int ati_r580_gl_prog_ready(R580GlCtx *g, uint64_t key, bool add,
                           const char *glsl, uint32_t wmask)
{
    return g ? FWD(g, prog_ready, key, add, glsl, wmask) : -1;
}

void ati_r580_gl_worker_stats(R580GlCtx *g, uint64_t *warms, uint64_t *failed,
                              uint64_t *waits, unsigned *inflight)
{
    if (g && g->mtl) {
        r580_mtl_worker_stats(MTL(g), warms, failed, waits, inflight);
    } else {
        r580_ogl_worker_stats(g ? OGL(g) : NULL, warms, failed, waits,
                              inflight);
    }
}

bool ati_r580_gl_target(R580GlCtx *g, int w, int h, bool *lost)
{
    if (!g) {
        *lost = false;
        return false;
    }
    return FWD(g, target, w, h, lost);
}

bool ati_r580_gl_seed(R580GlCtx *g, int x0, int y0, int w, int h,
                      const uint8_t *base, unsigned pitch, unsigned xr)
{
    return g && FWD(g, seed, x0, y0, w, h, base, pitch, xr);
}

bool ati_r580_gl_fetch(R580GlCtx *g, int x0, int y0, int w, int h,
                       uint8_t *base, unsigned pitch, unsigned xr)
{
    return g && FWD(g, fetch, x0, y0, w, h, base, pitch, xr);
}

bool ati_r580_gl_depth(R580GlCtx *g)
{
    return g && FWD(g, depth);
}

bool ati_r580_gl_zseed(R580GlCtx *g, int x0, int y0, int w, int h,
                       const uint32_t *z, int mode)
{
    return g && FWD(g, zseed, x0, y0, w, h, z, mode);
}

bool ati_r580_gl_zfetch(R580GlCtx *g, int x0, int y0, int w, int h,
                        uint32_t *z, int mode)
{
    return g && FWD(g, zfetch, x0, y0, w, h, z, mode);
}

bool ati_r580_gl_zclear(R580GlCtx *g, int x0, int y0, int w, int h,
                        uint32_t z, int mode)
{
    return g && FWD(g, zclear, x0, y0, w, h, z, mode);
}

bool ati_r580_gl_draw(R580GlCtx *g, const R580GlReq *req)
{
    return g && FWD(g, draw, req);
}

uint64_t ati_r580_gl_zq_mark(R580GlCtx *g)
{
    return g ? FWD(g, zq_mark) : 0;
}

bool ati_r580_gl_zq_sum(R580GlCtx *g, uint64_t ticket, bool wait,
                        uint64_t *sum)
{
    return g && FWD(g, zq_sum, ticket, wait, sum);
}

void ati_r580_gl_tex_forget(R580GlCtx *g, unsigned slot)
{
    if (g) {
        FWD(g, tex_forget, slot);
    }
}

const char *ati_r580_gl_describe(R580GlCtx *g)
{
    return g ? FWD(g, describe) : "none";
}

void ati_r580_gl_prog_stats(R580GlCtx *g, uint64_t *hits, uint64_t *links,
                            uint64_t *failed)
{
    if (g) {
        FWD(g, prog_stats, hits, links, failed);
    } else {
        *hits = *links = *failed = 0;
    }
}

uint64_t ati_r580_gl_barriers(R580GlCtx *g)
{
    return g ? FWD(g, barriers) : 0;
}

void ati_r580_gl_queue_stats(R580GlCtx *g, uint64_t *units, uint64_t *flushes,
                             uint64_t *waves)
{
    if (g) {
        FWD(g, queue_stats, units, flushes, waves);
    } else {
        *units = *flushes = *waves = 0;
    }
}

void ati_r580_gl_hold(bool hold)
{
    r580_ogl_hold(hold);
}

void ati_r580_gl_unbind(R580GlCtx *g)
{
    if (g) {
        FWD(g, unbind);
    }
}

struct R5GpuImage *ati_r580_gl_r5_image(R580GlCtx *g, int w, int h,
                                        int levels, int bpp)
{
    return g ? FWD(g, r5_image, w, h, levels, bpp) : NULL;
}

void ati_r580_gl_r5_image_free(R580GlCtx *g, struct R5GpuImage *im)
{
    if (g) {
        FWD(g, r5_image_free, im);
    }
}

bool ati_r580_gl_r5_upload(R580GlCtx *g, struct R5GpuImage *im, int level,
                           int x, int y, int w, int h, const void *data,
                           unsigned stride)
{
    return g && FWD(g, r5_upload, im, level, x, y, w, h, data, stride);
}

bool ati_r580_gl_r5_read(R580GlCtx *g, struct R5GpuImage *im, int x, int y,
                         int w, int h, void *data, unsigned stride)
{
    return g && FWD(g, r5_read, im, x, y, w, h, data, stride);
}

bool ati_r580_gl_r5_copy(R580GlCtx *g, struct R5GpuImage *dst, int dx, int dy,
                         struct R5GpuImage *src, int sx, int sy, int w, int h)
{
    return g && FWD(g, r5_copy, dst, dx, dy, src, sx, sy, w, h);
}

bool ati_r580_gl_r5_copy_runs(R580GlCtx *g, struct R5GpuImage *dst,
                              struct R5GpuImage *src, const int *runs,
                              unsigned n)
{
    return g && FWD(g, r5_copy_runs, dst, src, runs, n);
}

bool ati_r580_gl_r5_draw(R580GlCtx *g, const struct R5GpuDraw *d)
{
    return g && FWD(g, r5_draw, d);
}

int ati_r580_gl_r5_prog_ready(R580GlCtx *g, const struct R5GpuDraw *d,
                              int bpp)
{
    return g ? FWD(g, r5_prog_ready, d, bpp) : -1;
}

bool ati_r580_gl_r5_dst_fetch(R580GlCtx *g)
{
    return g && FWD(g, r5_dst_fetch);
}

void ati_r580_gl_r5_stats(R580GlCtx *g, uint64_t *progs, uint64_t *fails,
                          uint64_t *draws, uint64_t *link_us,
                          uint64_t *link_max)
{
    if (g) {
        FWD(g, r5_stats, progs, fails, draws, link_us, link_max);
    } else {
        *progs = *fails = *draws = *link_us = *link_max = 0;
    }
}
