/*
 * ATI R580 3D engine, R500 pipeline on the host GPU: the interface
 * between the device and the GL / Metal backends.
 *
 * The vertex stage stays on the CPU (ati_r580_r5xx.c); a request carries
 * window-space triangles with their fragment-program inputs. The backend
 * rasterises them, runs the R500 fragment program translated to the
 * common shading dialect below, samples textures from raw texel images,
 * and blends and packs into a raw colour image, all as the CPU path
 * (ati_r580_r5raster.c, ati_r580_r5us.c, ati_r580_r5tex.c) does.
 *
 * IMAGES hold texels exactly as they lie in VRAM: one unsigned integer
 * per pixel of 1, 2 or 4 bytes. A colour buffer is an image the backend
 * renders into; a texture is an image it samples, or a rectangle of a
 * colour buffer's image when the texture and the buffer share memory.
 * Format decoding, channel selects, filtering, blending and packing are
 * all done in the shader.
 *
 * SHADING DIALECT. The fragment shader text built by
 * ati_r580_r5gpu_fs_text() is valid GLSL 4.10 and, behind the Metal
 * backend's prelude, valid MSL. It names the following, which each
 * backend defines:
 *   vec2..vec4, ivec2..ivec4, uvec2..uvec4   vector types
 *   precise                                 no contraction
 *   floatBitsToUint, uintBitsToFloat, isnan
 *   R5_IN(n)        fragment-program input n (vec4)
 *   R5_KC(i)        constant vector i (vec4)
 *   R5_KI(i)        draw integer parameter vector i (ivec4)
 *   R5_KF(i)        draw float parameter vector i (vec4)
 *   R5_TI(u, i)     unit u integer parameter vector i (ivec4)
 *   R5_TF(u, i)     unit u float parameter vector i (vec4)
 *   R5_FETCHu(x, y, l)  raw texel of unit u's image (uint)
 *   R5_LUT(nb, v)   float(v) / float(2^nb - 1), correctly rounded
 *   R5_DFDX(e), R5_DFDY(e)
 *   R5_DST          the destination's raw pixel (uint)
 *   R5_FRAG         the fragment's window position in the image (vec2)
 *   R5_PRIM         the triangle's index in the request (int)
 *   R5_PLANE(i)     vec4 i of the request's plane equations
 *   R5_THIS         member access prefix (empty in GLSL)
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef ATI_R580_R5GPU_H
#define ATI_R580_R5GPU_H

#include "ati_r580_r5raster.h"

/* a window-space primitive, as the band workers and the GPU path take it */
enum { R5_PRIM_TRI, R5_PRIM_POINT, R5_PRIM_POINT_SIZED, R5_PRIM_LINE };

typedef struct R5Prim {
    uint8_t kind;
    bool cull_front, cull_back;
    uint32_t flat_inputs;
    float size;
    R5xxVertex v[3];
} R5Prim;

#define R5G_INPUTS      R5XX_MAX_FS_INPUTS
#define R5G_UNITS       16
/* units one draw may bind; the backends keep two more for themselves */
#define R5G_MAX_BOUND   13
#define R5G_CONSTS      256
/* integer and float parameter vectors per unit and per draw */
#define R5G_TI          4
#define R5G_TF          2
#define R5G_KI          10
#define R5G_KF          4

/*
 * Draw integer parameters, R5_KI(i).c:
 *   KI0  fmt: x bits per pixel 8/16/32, y channel layout (R5G_CF_*),
 *        z write mask over canonical RGBA, w flags (R5G_DF_*)
 *   KI1  chan_src[0..3]: the shader channel feeding canonical R, G, B, A
 *   KI2  blend: rgb eq, rgb src, rgb dst, alpha eq
 *   KI3  blend: alpha src, alpha dst, discard mode, alpha func
 *   KI4  x clip rule (SC_CLIP_RULE), y first triangle of the pass,
 *        zw destination snapshot origin
 *   KI5..KI8  clip rectangles x0, y0, x1, y1, inclusive
 *   KI9  xy the colour buffer's origin inside the image
 * Draw float parameters, R5_KF(i):
 *   KF0  blend constant
 *   KF1  x alpha reference
 */
#define R5G_CF_8888     0       /* A8 R8 G8 B8, B at bit 0 */
#define R5G_CF_565      1
#define R5G_CF_1555     2
#define R5G_CF_4444     3
#define R5G_CF_2101010  4
#define R5G_CF_I8       5

#define R5G_DF_BLEND    1
#define R5G_DF_ATEST    2
#define R5G_DF_DST      4       /* the destination is read */
#define R5G_DF_DISCARD  8       /* blend discard mode is in effect */

/*
 * Unit integer parameters, R5_TI(u, i):
 *   TI0  level-0 width, height, last level, flags (R5G_TF_*)
 *   TI1  wrap s, wrap t (R5xxWrap), image origin x, y
 *   TI2  bits of channel X, Y, Z, W (0: absent)
 *   TI3  swizzle sel R, G, B, A (R300_TX_FORMAT_SEL_*)
 * Unit float parameters, R5_TF(u, i):
 *   TF0  lod bias, min lod, max lod
 *   TF1  border colour
 */
#define R5G_TF_UNSCALED 1
#define R5G_TF_MIN_LIN  2
#define R5G_TF_MAG_LIN  4
#define R5G_TF_MIP_NEAR 8
#define R5G_TF_MIP_LIN  16
#define R5G_TF_MAG_UNDEF 32     /* FILTER_NONE as magnification */
#define R5G_TF_MIN_UNDEF 64

typedef struct R5GpuImage R5GpuImage;

typedef struct R5GpuDraw {
    R5GpuImage *target;
    /* scissor in image coordinates, end exclusive */
    int sx0, sy0, sx1, sy1;
    /*
     * Vertices, three per triangle: window x, y relative to the target
     * origin, z, clip w, then `ninputs` vec4 inputs. Flat inputs (bit n
     * of `flat`) carry the provoking vertex's value at all three.
     */
    const float *verts;
    unsigned nvert, ninputs;
    uint32_t flat;
    /*
     * Passes for a draw that reads its destination: triangles in one
     * pass do not overlap. pass[k] is the first vertex of pass k, and
     * pass[npass] == nvert. Each pass's bounding box in image
     * coordinates is pbox[4k..4k+3] (x0, y0, x1, y1, end exclusive).
     */
    /*
     * Plane equations, or NULL: triangle t's record is 3 * ninputs + 1
     * vec4 long; for input n it holds vec4 A, B, C at 3n, the input
     * being A * x + B * y + C at image position (x, y); its last vec4
     * holds the plane of 1/w, by which a perspective draw's inputs are
     * divided.
     */
    const float *planes;
    const unsigned *pass;
    const int *pbox;
    unsigned npass;
    /* fragment shader text and the key it is cached under */
    const char *fs;
    uint64_t key;
    const float *consts;            /* R5G_CONSTS vec4 */
    R5GpuImage *img[R5G_UNITS];
    uint32_t units;
    int32_t ti[R5G_UNITS][R5G_TI][4];
    float tf[R5G_UNITS][R5G_TF][4];
    int32_t ki[R5G_KI][4];
    float kf[R5G_KF][4];
} R5GpuDraw;

#endif /* ATI_R580_R5GPU_H */
