/*
 * ATI Radeon HD 5770 (Juniper) 3D engine on a host GPU: the interface
 * between the draw path (ati_hd5770_draw.c) and a backend.
 *
 * A request is one draw after the vertex stage: window-space triangles
 * in rasterisation order, each with the pixel shader inputs at its three
 * vertices, the pixel shader as IR (ati_hd5770_ir.h), its constants, and
 * the render target, textures and vertex buffers as the CPU path sees
 * them. The backend produces in VRAM what the CPU path produces:
 * coverage and barycentrics by the CPU rasteriser's edge functions,
 * interpolation, the shader, the alpha test, blending and packing into
 * the target's format through its tiling, pixel by pixel; triangles of
 * one request in order where they overlap.
 *
 * MEMORY. Surfaces are addressed by their byte offset in VRAM
 * (HD5770GpuSurf.voff), through the layout of HD5770Surf. The backend
 * gets the host memory of VRAM when it is opened and may keep its own
 * copies (then it must keep them coherent: every byte a request reads is
 * as VRAM holds it when the request is made, given that requests queued
 * before it have been applied). Vertex buffer bytes come with the
 * request (they may lie in system memory).
 *
 * ORDER. draw() queues; sync() returns once everything queued is in
 * VRAM. The device calls sync() before anything else reads or writes
 * memory a draw may touch: a CPU-path draw, a CP packet that accesses
 * memory, a fence, the DMA engine, the end of a ring run, reset,
 * migration. Between two syncs only queued draws touch the surfaces.
 * done(lo, hi) is called by sync() for each VRAM byte range written
 * since the last sync, so the device can mark it dirty.
 *
 * A backend that cannot reproduce a request returns false from draw()
 * without queueing anything; the device then syncs and draws it on the
 * CPU.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef ATI_HD5770_GPU_H
#define ATI_HD5770_GPU_H

#include "ati_hd5770_gfx.h"
#include "ati_hd5770_ir.h"

#define HD5770_GPU_TEX          16
#define HD5770_GPU_VB           8
#define HD5770_GPU_INPUTS       32

/* a surface in VRAM */
typedef struct HD5770GpuSurf {
    HD5770Surf s;               /* prepared, s.fast set */
    uint64_t voff;              /* VRAM offset of s.base */
} HD5770GpuSurf;

/* a texture: the CPU path's Tex */
typedef struct HD5770GpuTex {
    HD5770GpuSurf surf;
    uint32_t fmt, num, w, h, dim, sgn, comps;
    uint32_t dst_sel[4];
    uint32_t bits[4];
    uint32_t samp[3];
    float border[4];
    bool is_float, unorm8;
} HD5770GpuTex;

/* a vertex buffer resource the pixel shader fetches from */
typedef struct HD5770GpuVb {
    uint8_t id;                 /* resource, as in the fetch words */
    uint32_t res[8];            /* SQ_RESOURCE words */
    const uint8_t *data;        /* bytes 0..len-1 of the buffer */
    uint32_t len;
} HD5770GpuVb;

/* a triangle: vertices in the CPU rasteriser's order (positive area) */
typedef struct HD5770GpuTri {
    float x[3], y[3], w[3];     /* window position, 1/w */
    float area;
    int32_t x0, y0, x1, y1;     /* pixels to visit, inclusive */
} HD5770GpuTri;

typedef struct HD5770GpuDraw {
    const HD5770IR *ir;
    const uint32_t *kv;         /* ir->nconsts bound constants, raw */
    const uint32_t *cfile;      /* PS DX9 constant file, 256 x 4 */

    /* render target, colour buffer 0 */
    HD5770GpuSurf rt;
    uint32_t rt_fmt, rt_num, rt_comps;
    uint32_t rt_bits[4], rt_src[4];
    bool rt_unorm8, rt_float, rt_blend;
    uint32_t wmask, blend;      /* CB_TARGET_MASK & CB_SHADER_MASK, BLEND0 */
    float blend_const[4];
    bool alpha_test;
    uint32_t alpha_func;
    float alpha_ref;
    uint32_t cliprule;          /* 0xffff: no clip rectangles */
    int32_t cr[4][4];           /* clip rectangles x0, y0, x1, y1 */
    float pix_center;
    bool persp;
    int pos_gpr;                /* GPR of the position, -1 none */

    /* textures, in ir->res order */
    unsigned ntex;
    HD5770GpuTex tex[HD5770_GPU_TEX];
    unsigned nvb;
    HD5770GpuVb vb[HD5770_GPU_VB];

    /*
     * Triangles, and for each the inputs: prm[(t * ninputs + p) * 12 +
     * v * 4 + c] is input p, channel c at vertex v of triangle t. Input p
     * is flat (the vertex 0 value everywhere) when bit p of flat is set.
     */
    unsigned ntri, ninputs;
    const HD5770GpuTri *tri;
    const float *prm;
    uint32_t flat;
} HD5770GpuDraw;

typedef struct HD5770GpuOps {
    const char *name;
    bool (*draw)(void *b, const HD5770GpuDraw *d);
    void (*sync)(void *b);
    void (*stats)(void *b, GString *out);
    void (*close)(void *b);
} HD5770GpuOps;

typedef struct HD5770Gpu {
    const HD5770GpuOps *ops;
    void *b;
    /* the device's: VRAM bytes [lo, hi) were written */
    void (*done)(void *opaque, uint64_t lo, uint64_t hi);
    void *opaque;
} HD5770Gpu;

/* the Metal backend (macOS); NULL and *err set if there is none */
HD5770Gpu *hd5770_mtl_open(uint8_t *vram, uint64_t vram_size,
                           const char **err);

/*
 * The shading language text of a request's program, for backends that
 * compile it: MSL (ati_hd5770_msl.c). The kernel's interface is
 * described there.
 */
char *hd5770_msl_program(const HD5770IR *ir);
extern const char hd5770_msl_lib[];

#endif
