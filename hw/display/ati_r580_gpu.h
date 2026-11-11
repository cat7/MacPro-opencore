/*
 * ATI R300/R350 -- host GPU backend selection on macOS.
 *
 * The device calls the interface in ati_r580_gl.h. On macOS two
 * implementations of it are built: OpenGL (ati_r580_gl.c) and Metal
 * (ati_r580_mtl.m). Each defines R580_GPU_IMPL_OGL or R580_GPU_IMPL_MTL
 * before including this header, which renames the interface to its own
 * prefix; ati_r580_gpu.c then provides the ati_r580_gl_* entry points
 * and forwards each call to the backend the context was opened with.
 * Other hosts build ati_r580_gl.c under the interface names directly.
 *
 * Based on lyons88/qemu-g5 934422f4f9.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef ATI_R580_GPU_H
#define ATI_R580_GPU_H

#if defined(R580_GPU_IMPL_OGL)
#define R580GlCtx                   R580OglCtx
#define R580_GPU_P(n)               r580_ogl_##n
#elif defined(R580_GPU_IMPL_MTL)
#define R580GlCtx                   R580MtlCtx
#define R580_GPU_P(n)               r580_mtl_##n
#endif

#ifdef R580_GPU_P
#define ati_r580_gl_open            R580_GPU_P(open)
#define ati_r580_gl_close           R580_GPU_P(close)
#define ati_r580_gl_prog_ready      R580_GPU_P(prog_ready)
#define ati_r580_gl_worker_stats    R580_GPU_P(worker_stats)
#define ati_r580_gl_target          R580_GPU_P(target)
#define ati_r580_gl_seed            R580_GPU_P(seed)
#define ati_r580_gl_fetch           R580_GPU_P(fetch)
#define ati_r580_gl_depth           R580_GPU_P(depth)
#define ati_r580_gl_zseed           R580_GPU_P(zseed)
#define ati_r580_gl_zfetch          R580_GPU_P(zfetch)
#define ati_r580_gl_zclear          R580_GPU_P(zclear)
#define ati_r580_gl_draw            R580_GPU_P(draw)
#define ati_r580_gl_zq_mark         R580_GPU_P(zq_mark)
#define ati_r580_gl_zq_sum          R580_GPU_P(zq_sum)
#define ati_r580_gl_tex_forget      R580_GPU_P(tex_forget)
#define ati_r580_gl_describe        R580_GPU_P(describe)
#define ati_r580_gl_prog_stats      R580_GPU_P(prog_stats)
#define ati_r580_gl_barriers        R580_GPU_P(barriers)
#define ati_r580_gl_queue_stats     R580_GPU_P(queue_stats)
#define ati_r580_gl_hold            R580_GPU_P(hold)
#define ati_r580_gl_unbind          R580_GPU_P(unbind)
#define ati_r580_gl_r5_image        R580_GPU_P(r5_image)
#define ati_r580_gl_r5_image_free   R580_GPU_P(r5_image_free)
#define ati_r580_gl_r5_upload       R580_GPU_P(r5_upload)
#define ati_r580_gl_r5_read         R580_GPU_P(r5_read)
#define ati_r580_gl_r5_copy         R580_GPU_P(r5_copy)
#define ati_r580_gl_r5_draw         R580_GPU_P(r5_draw)
#define ati_r580_gl_r5_prog_ready   R580_GPU_P(r5_prog_ready)
#define ati_r580_gl_r5_dst_fetch    R580_GPU_P(r5_dst_fetch)
#define ati_r580_gl_r5_stats        R580_GPU_P(r5_stats)
#endif

#include "ati_r580_gl.h"

#ifndef R580_GPU_P
/* the dispatcher's view of both implementations */
#define R580_GPU_DECLARE(T, P)                                              \
    typedef struct T T;                                                     \
    T *P##_open(const char **err, unsigned workers);                        \
    void P##_close(T *g);                                                   \
    int P##_prog_ready(T *g, uint64_t key, bool add, const char *glsl,      \
                       uint32_t wmask);                                     \
    void P##_worker_stats(T *g, uint64_t *warms, uint64_t *failed,          \
                          uint64_t *waits, unsigned *inflight);             \
    bool P##_target(T *g, int w, int h, bool *lost);                        \
    bool P##_seed(T *g, int x0, int y0, int w, int h,                       \
                  const uint8_t *base, unsigned pitch, unsigned xr);        \
    bool P##_fetch(T *g, int x0, int y0, int w, int h,                      \
                   uint8_t *base, unsigned pitch, unsigned xr);             \
    bool P##_depth(T *g);                                                   \
    bool P##_zseed(T *g, int x0, int y0, int w, int h, const uint32_t *z,   \
                   int mode);                                               \
    bool P##_zfetch(T *g, int x0, int y0, int w, int h, uint32_t *z,        \
                    int mode);                                              \
    bool P##_zclear(T *g, int x0, int y0, int w, int h, uint32_t z,         \
                    int mode);                                              \
    bool P##_draw(T *g, const R580GlReq *req);                              \
    uint64_t P##_zq_mark(T *g);                                             \
    bool P##_zq_sum(T *g, uint64_t ticket, bool wait, uint64_t *sum);       \
    void P##_tex_forget(T *g, unsigned slot);                               \
    const char *P##_describe(T *g);                                         \
    void P##_prog_stats(T *g, uint64_t *hits, uint64_t *links,              \
                        uint64_t *failed);                                  \
    uint64_t P##_barriers(T *g);                                            \
    void P##_queue_stats(T *g, uint64_t *units, uint64_t *flushes,          \
                         uint64_t *waves);                                  \
    void P##_hold(bool hold);                                               \
    void P##_unbind(T *g);                                                  \
    struct R5GpuImage *P##_r5_image(T *g, int w, int h, int levels,         \
                                    int bpp);                               \
    void P##_r5_image_free(T *g, struct R5GpuImage *im);                    \
    bool P##_r5_upload(T *g, struct R5GpuImage *im, int level, int x,       \
                       int y, int w, int h, const void *data,               \
                       unsigned stride);                                    \
    bool P##_r5_read(T *g, struct R5GpuImage *im, int x, int y, int w,      \
                     int h, void *data, unsigned stride);                   \
    bool P##_r5_copy(T *g, struct R5GpuImage *dst, int dx, int dy,          \
                     struct R5GpuImage *src, int sx, int sy, int w, int h); \
    bool P##_r5_draw(T *g, const struct R5GpuDraw *d);                      \
    int P##_r5_prog_ready(T *g, const struct R5GpuDraw *d, int bpp);        \
    bool P##_r5_dst_fetch(T *g);                                            \
    void P##_r5_stats(T *g, uint64_t *progs, uint64_t *fails,               \
                      uint64_t *draws, uint64_t *link_us,                  \
                      uint64_t *link_max);

R580_GPU_DECLARE(R580OglCtx, r580_ogl)
R580_GPU_DECLARE(R580MtlCtx, r580_mtl)
#endif

#endif /* ATI_R580_GPU_H */
