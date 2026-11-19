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
 * (HD5770GpuSurf.voff), or page by page in guest RAM, through the layout
 * of HD5770Surf. The backend
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
 * FENCES (added 2026-11-15). A backend with fence() completes the
 * device's fences itself, so the device does not wait for the GPU before
 * a fence: fence() commits what is queued and returns a ticket, or 0
 * when everything queued before it is in memory with its ranges reported
 * and every earlier ticket retired. The backend then calls retired(opaque, ticket) once
 * every request queued before that fence() is in memory, done() and
 * done_ram() having been called for the ranges those requests wrote:
 * one call at a time, tickets in increasing order, on a thread of the
 * backend's own (between thread(opaque, true) and thread(opaque, false)
 * when it is not a thread QEMU created). Those calls must not block on
 * the device's locks or the BQL. sync() waits for the GPU, not for any
 * retired() call. Without fence() the device syncs before each fence and
 * completes it itself.
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
#define HD5770_GPU_LEVELS       15

/*
 * A surface in VRAM, or (space 1) in guest RAM: pages[i] is the offset in
 * the RAM given to ops->ram() of the 4 KiB page holding surface bytes
 * from i * 4096 - pg0 on, pg0 being the base's offset in its page.
 */
typedef struct HD5770GpuSurf {
    HD5770Surf s;               /* prepared, s.fast set */
    uint64_t voff;              /* VRAM offset of s.base */
    uint8_t space;
    const uint32_t *pages;
    uint32_t npages, pg0;
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
    /*
     * Added 2026-11-14: a mipmapped 2D texture (normalised coordinates):
     * lv[l] is level l (lv[0] = surf), lw/lh its size; the level of
     * detail comes from the coordinates' change to the pixel on the
     * right and below, as tfetch_stage() in ati_hd5770_draw.c computes
     * it. Only for backends that set HD5770GpuOps.mipmaps.
     */
    bool mip;
    uint32_t last_lv, mip_filter;
    float min_lod, max_lod, lod_bias;
    const HD5770GpuSurf *lv;
    uint32_t lw[HD5770_GPU_LEVELS], lh[HD5770_GPU_LEVELS];
    /*
     * Added 2026-11-14 (M4b): GET_TEXTURE_RESINFO (fetch opcode 4)
     * returns, as integers, the width and height of level base_lv + x
     * (x the integer in the source's first component, each at least 1;
     * height 1 for a 1D texture), depth and levels. Only for backends
     * that set HD5770GpuOps.resinfo.
     */
    uint32_t base_lv, levels, depth;
    /*
     * Added 2026-11-17 (HD5770GpuOps.full): bc 1..5 for BC1..BC5 (the
     * surface's elements are 4x4 blocks, as texel() in
     * ati_hd5770_draw.c decodes them); slices of a 3D (dim 2), cube
     * (dim 3, six faces) or array (dim 4, 5) texture follow each other in
     * a level, slice_bytes[l] apart (a 3D texture's depth halves with the
     * levels); surf and lv[] cover all slices. tex_slice() picks the
     * slice.
     */
    uint32_t bc, slices;
    uint64_t slice_bytes[HD5770_GPU_LEVELS];
} HD5770GpuTex;

/*
 * Added 2026-11-17 (HD5770GpuOps.full): a colour buffer of a draw with
 * more render targets: export cb goes to it, with its own format, write
 * mask and CB_BLENDn_CONTROL, as put_pixel() writes it.
 */
typedef struct HD5770GpuRt {
    HD5770GpuSurf surf;
    uint32_t cb;
    uint32_t fmt, num, comps;
    uint32_t bits[4], src[4];
    bool unorm8, is_float, blend_ok;
    uint32_t wmask, blend;
} HD5770GpuRt;

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
    uint32_t prim;              /* triangles of one primitive share no pixel */
    /* added 2026-11-14: depth and stencil */
    float z[3];                 /* window z of the vertices */
    uint32_t front;             /* the triangle faces front */
} HD5770GpuTri;

struct HD5770GpuVs;
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

    /*
     * Added 2026-11-14: depth and stencil, as db_test() in
     * ati_hd5770_draw.c. With db set, each covered pixel that survives
     * the kill and the alpha test is tested against the depth plane zs
     * (zfmt 1 Z_16, 2 Z_24, 3 Z_32_FLOAT, 0 none) and the stencil plane
     * ss (has_s), with DB_DEPTH_CONTROL dctl, DB_RENDER_CONTROL rctl
     * (clear enables), DB_STENCILREFMASK front/back srm, the clear
     * values, the viewport depth range; the planes are updated, and the
     * colour is written only if both tests pass. The fragment depth is
     * z[0] * b0 + z[1] * b1 + z[2] * b2 of the triangle.
     */
    bool db;
    HD5770GpuSurf zs, ss;
    uint32_t zfmt;
    bool has_s;
    uint32_t dctl, rctl, srm[2], sclear;
    float zclear, zmin, zmax;

    /*
     * Added 2026-11-17, only for backends that set HD5770GpuOps.full.
     * nmrt > 0: the draw writes the colour buffers mrt[0..nmrt-1] (the
     * rt fields above are mrt[0]'s), each from its own export; as
     * shade() in ati_hd5770_draw.c, the alpha test reads export 0.
     * zexport/sexport: the fragment depth (the stencil reference) comes
     * from the program's depth export when it made one. bool_const and
     * loop_const: the pixel shader's SQ_BOOL_CONST and SQ_LOOP_CONST for
     * a program in the IR's CF form.
     */
    unsigned nmrt;
    HD5770GpuRt mrt[8];
    bool zexport, sexport;
    uint32_t bool_const;
    uint32_t loop_const[32];
    /*
     * resolve (full): no program runs (ir has no ops); each covered
     * pixel of rt gets the pixel of rs as resolve_pixel() copies it: its
     * bytes when rs_same, else its rs_comps components of format rs_fmt
     * (rs_num 7 float, 4 integer, else UNORM; component i is channel
     * rs_src[i]) through the output stage.
     */
    bool resolve, rs_same;
    HD5770GpuSurf rs;
    uint32_t rs_fmt, rs_num, rs_comps, rs_src[4];
    /* added 2026-11-17: the scissor, x0, y0, x1, y1 (x1, y1 exclusive) */
    int32_t sc[4];
    /*
     * added 2026-11-18, only for backends with HD5770Gpu.vs_draw: the
     * draw before its vertex stage. ntri is 0; the backend runs vertex
     * stage vs (below) itself and rasterises its triangles vs->tidx.
     */
    const struct HD5770GpuVs *vs;
} HD5770GpuDraw;

/*
 * The vertex stage of a draw. Pass A runs vertex shader ir once per
 * index slot: for instance i < ninst and slot s < nslot the index is
 *   idx_kind 0 (auto):  s + ioff
 *   idx_kind 1, 2:      raw = (1) element s of buffer ib (ib_i32 ? 32
 *                       : 16 bits, byte order ib_swap as draw_indices()
 *                       reads it; 0 from element ib_max on) or (2)
 *                       imm[s]; raw == reset with restart set marks a
 *                       restart slot, else MIN(MAX(raw + ioff, lo), hi)
 * and the shader runs with R0 = (index, 0, 0, i), its vertex fetches as
 * vfetch_vs() does them, reading buffer vb[k].buf in place (vb[k].id:
 * the fetch's resource id, plus 160 for the fetch shader's), DX9
 * constant file cfile, constants kv (bound as hd5770_ir_bind() binds
 * them). It writes record i * nslot + s of HD5770_VS_REC(np) words: the
 * clip space position, parameters 0..np-1, the mask of the parameters
 * exported, flags HD5770_VSF_* (channels not exported are 0).
 *
 * The rest is for the triangle set-up after it: slot[s] is UINT32_MAX
 * at a restart slot; the slots are assembled as primitives of type prim
 * (VGT_PRIMITIVE_TYPE) and set up as assemble() and gpu_tri() set them
 * up, with the registers below. Pixel inputs: input p of the pixel
 * shader is parameter src[p] of the vertex shader (-1: def[p]), flat if
 * bit p of flat is set (the provoking vertex's).
 */
#define HD5770_VS_REC(np)       (4 + 4 * (np) + 2)
#define HD5770_VSF_RESTART      1u
#define HD5770_VSF_POS          2u

typedef struct HD5770GpuVsVb {
    uint32_t id;
    uint32_t res[8];            /* SQ_RESOURCE words */
    HD5770GpuSurf buf;          /* its bytes 0..res[1] */
} HD5770GpuVsVb;

typedef struct HD5770GpuVs {
    const HD5770IR *ir;
    const uint32_t *kv;
    const uint32_t *cfile;      /* VS DX9 constant file, 256 x 4 */
    unsigned nslot, ninst, np, prim;
    unsigned idx_kind;
    HD5770GpuSurf ib;
    bool ib_i32, restart;
    uint32_t ib_swap, ib_max, ioff, lo, hi, reset;
    const uint32_t *imm;
    uint32_t base_vtx, start_inst;  /* the fetch types' offsets */
    unsigned nvb;
    HD5770GpuVsVb vb[HD5770_GPU_VB];
    /* set-up */
    const uint32_t *slot;
    unsigned ninputs;           /* as HD5770GpuDraw's */
    uint32_t flat;
    float pix_center;
    uint32_t vte, sc_mode, window_offset;
    float vport[6];
    bool clip_on, clip_dx, clip_near, clip_far;
    unsigned cull;              /* bit 0 front, bit 1 back */
    bool face_cw, prov_last;
    int32_t x0, y0, x1, y1;     /* scissor, x1 and y1 exclusive */
    float poly_scale[2], poly_units[2], poly_clamp;  /* front, back */
    int8_t poly_db_bits;
    uint32_t zfmt;
    int32_t src[32];
    float def[32][4];
    /*
     * added 2026-11-18 (HD5770GpuDraw.vs): the triangles of one instance
     * as vertex indices (the index the shader runs with), three each,
     * the vertex whose flat inputs count first
     */
    const uint32_t *tidx;
    unsigned ntidx;
    /*
     * the polygon offset of the sides not culled (alike, not clamped):
     * zoff_const + zoff_slope * the steeper of the depth's slopes
     */
    float zoff_const, zoff_slope;
} HD5770GpuVs;

/* the parameters a vertex shader's records hold (HD5770GpuVs.np) */
unsigned hd5770_vs_np(const HD5770IR *ir);

/*
 * The emits of a vertex stage, as assemble() makes them: entries of five
 * words (a, b, c, provoking vertex, kind), records i * nslot + s;
 * kind 1: the second half of a rectangle, its third vertex a + b - c.
 * out NULL: only the count.
 */
unsigned hd5770_vs_emits(const HD5770GpuVs *v, uint32_t *out);

typedef struct HD5770GpuOps {
    const char *name;
    bool (*draw)(void *b, const HD5770GpuDraw *d);
    void (*sync)(void *b);
    void (*stats)(void *b, GString *out);
    void (*close)(void *b);
    /* host memory of guest RAM for surfaces of space 1; false: none */
    bool (*ram)(void *b, uint8_t *host, uint64_t size);
    /* draws covering fewer pixels stay on the CPU; 0: all are offered */
    unsigned min_pixels;
    /* added 2026-11-14: takes requests with db set */
    bool depth;
    /* added 2026-11-14: takes requests with mipmapped textures */
    bool mipmaps;
    /* added 2026-11-14 (M4b): takes programs with GET_TEXTURE_RESINFO */
    bool resinfo;
    /* added 2026-11-15: FENCES above */
    uint64_t (*fence)(void *b);
    /*
     * added 2026-11-17: takes BC, 3D, cube and array textures, SAMPLE_C,
     * CUBE, programs in the CF form, depth exports and more render
     * targets (the fields marked HD5770GpuOps.full)
     */
    bool full;
    /*
     * added 2026-11-17: the CPU wrote VRAM [off, off + len), or guest RAM
     * when len is 0; NULL for a backend that reads memory in place
     */
    void (*wrote)(void *b, uint64_t off, uint64_t len);
    /*
     * added 2026-11-18: as sync() for the surfaces over VRAM bytes
     * [off, off + len) only: the work queued is done and those surfaces
     * are in memory; others drawn may stay newer than memory (kept is
     * set then). NULL: sync() is used.
     */
    void (*sync_range)(void *b, uint64_t off, uint64_t len);
    /*
     * added 2026-11-17: vertex stage v alone, after the work before it:
     * its records to rec (HD5770GpuVs); with tri set, also its triangles
     * (*ntri of them, at most hd5770_vs_emits() * 3) and their inputs
     * (prm as HD5770GpuDraw.prm). false: not run.
     */
    bool (*vs_records)(void *b, const HD5770GpuVs *v, uint32_t *rec,
                       HD5770GpuTri *tri, float *prm, unsigned *ntri);
    /*
     * added 2026-11-18: build ahead. prep(b, d): the programs and
     * pipelines draw() would build for d are noted; nothing is drawn and
     * no surface is touched. prep(b, NULL): the ones noted are built,
     * several at a time. NULL: draw() builds as it goes.
     */
    void (*prep)(void *b, const HD5770GpuDraw *d);
    /*
     * added 2026-11-18: the native pipeline (below). Its surfaces may
     * stay newer than memory after a fence and its depth differs from
     * the CPU's in the last place: for it alone the device records the
     * scanout's flips, keeps one rasteriser per depth plane and reads
     * register 0x8e30 busy while the engine has work.
     */
    bool native;
} HD5770GpuOps;

typedef struct HD5770Gpu {
    const HD5770GpuOps *ops;
    void *b;
    /* the device's: VRAM bytes [lo, hi) were written */
    void (*done)(void *opaque, uint64_t lo, uint64_t hi);
    void *opaque;
    /* the same for guest RAM bytes, as offsets in the RAM of ops->ram() */
    void (*done_ram)(void *opaque, uint64_t lo, uint64_t hi);
    /* added 2026-11-15: FENCES above; NULL while the device uses none */
    void (*retired)(void *opaque, uint64_t ticket);
    void (*thread)(void *opaque, bool enter);
    /*
     * added 2026-11-17: set by draw() when it declines because the
     * program is still being built; the draw is then left out
     */
    bool building;
    /*
     * added 2026-11-17: set by draw() when it declines, HD5770_WHY_BK_*;
     * cleared before each call; 0 counts as HD5770_WHY_BK_OTHER
     */
    unsigned why;
    /*
     * added 2026-11-17: the device's log of VRAM writes, for a backend
     * that keeps copies. Bit i of the result is set when page
     * (off >> 12) + i (i < 64, off a multiple of 64 pages) may have been
     * written, by the guest, the device or a done() range, since the
     * call before for it; a write is reported at least once. NULL: no
     * log, every page may have been written.
     */
    uint64_t (*written)(void *opaque, uint64_t off);
    /* added 2026-11-18: takes draws with HD5770GpuDraw.vs */
    bool vs_draw;
    /*
     * added 2026-11-18: set by fence(), cleared by sync(): surfaces drawn
     * are still newer than memory although the fence left nothing
     * outstanding; the ranges queued stay pending
     */
    bool kept;
    /*
     * added 2026-11-18: the VRAM bytes [scan_lo, scan_hi) the display
     * reads: surfaces over them are in memory after every fence
     */
    uint64_t scan_lo, scan_hi;
    /* added 2026-11-18: set by a draw() that built a program or pipeline */
    bool built;
    /*
     * added 2026-11-19: draw() copies what the request reads of guest
     * RAM; a later write to guest RAM does not wait for queued draws
     * that only read it
     */
    bool ram_copied;
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
/* the same for draws whose texture units in mipmask are mipmapped */
char *hd5770_msl_program_mip(const HD5770IR *ir, uint32_t mipmask);
/*
 * A vertex shader's IR (hd5770_vs_ir()) as MSL kernel hd_vs, interface
 * in ati_hd5770_msl.c
 */
char *hd5770_msl_vs_program(const HD5770IR *ir);
/* the triangle set-up and box drawing of the vertex stage */
extern const char hd5770_msl_setup[];
extern const char hd5770_msl_lib[];

/*
 * The functions hd_nb() and hd_run() of a program in the CF form, as
 * text common to MSL and GLSL (ati_hd5770_emit.c, which lists the
 * macros the backend defines first)
 */
/* a program without loops or calls becomes straight code */
void hd5770_emit_cf(GString *o, const HD5770IR *ir, uint32_t mipmask);

/*
 * The native pipeline (macOS, ati_hd5770_mtlr.m): draws go through the
 * host's rasteriser into host textures kept coherent with VRAM, close to
 * but not bit for bit what the CPU path produces. A pixel program is a
 * fragment function hd_nfs, with the pass-through vertex function
 * hd_nvs, for the draws of key k: ninputs inputs (those in flat by the
 * provoking vertex), the position in GPR pos_gpr (-1 none). The target
 * is a host texture of rt_bytes (1 or 4) 8-bit channels: with out 0,
 * channel i is component rt_src[i] of rt_comps (1 or 4) as UNORM; with
 * out 1 the channels are the bytes of component rt_src[0] as an integer;
 * with out 2 the texture's channels are floats, channel i component
 * rt_src[i].
 * clip: the clip rectangle rule applies. vb[0..nvb-1]: the buffers of
 * its vertex fetches, as HD5770GpuDraw.vb, with the format, number type
 * and sign of the resource. mip[i]: texture unit i is mipmapped, with
 * that many levels after the first, each a texture of its own; the
 * levels of all units count from texture index 16 on, in unit order.
 * zout: the function writes the depth, the rasteriser's clamped to the
 * depth range. dim[i], nsl[i]: unit i's SQ_TEX_DIM and its slices (cube
 * faces, array layers, the depth of a 3D texture, which halves with the
 * levels), each slice of each level a texture of its own: after the
 * units' first textures, from index 16 on, come unit by unit, level by
 * level, the slices in order. Programs in the CF form, fetches with a
 * comparison and units with slices go through a function per unit that
 * reads the fetch words at run time. NULL: not expressible.
 */
#define HD5770_NAT_LEVELS       112     /* textures after the units' first */
typedef struct HD5770NatKey {
    uint8_t ninputs;
    int8_t pos_gpr;
    uint8_t rt_src[4];
    uint32_t flat;
    uint8_t out, rt_comps, rt_bytes;
    bool clip;
    uint8_t nvb;
    struct {
        uint8_t id, fmt, num, sgn;
    } vb[HD5770_GPU_VB];
    uint8_t mip[HD5770_GPU_TEX];
    bool zout;
    uint8_t dim[HD5770_GPU_TEX], nsl[HD5770_GPU_TEX];
    bool zexp;                  /* the depth written is the program's */
    /*
     * ncol > 0: more render targets; output i is export col[i].cb, of the
     * class out with comps and src as rt_comps and rt_src
     */
    uint8_t ncol;
    struct {
        uint8_t cb, out, comps, src[4];
    } col[8];
    /*
     * a pixel leaves the buffers whose export it did not make as they
     * are: the function reads them (Apple GPUs)
     */
    bool keep;
    /* the alpha test is on (GLSL: a function without it discards less) */
    bool atest;
} HD5770NatKey;
const char *hd5770_msl_alu(void);
char *hd5770_msl_native_fs(const HD5770IR *ir, const HD5770NatKey *k);
/* the inputs of hd_nfs that are not interpolated */
uint32_t hd5770_msl_native_flat(const HD5770IR *ir, const HD5770NatKey *k);
/*
 * A vertex shader's IR as vertex function hd_bvs for the fragment
 * functions above (HD5770GpuDraw.vs): the vertex index is the vertex id,
 * the instance the instance id; buffers 0 VRAM, 1 the draw (NatVs of
 * ati_hd5770_mtlr.m, offsets from its start), 2 guest RAM. Input p of
 * ninputs is parameter src[p] (-1: the draw's default), as bits when bit
 * p of flat is set. NULL: not expressible.
 */
char *hd5770_msl_native_vs(const HD5770IR *ir, unsigned ninputs,
                           uint32_t flat, const int32_t *src);
HD5770Gpu *hd5770_mtlr_open(uint8_t *vram, uint64_t vram_size,
                            const char **err);

/*
 * The rasterisation pipeline on Vulkan (Windows, ati_hd5770_vk.c with the
 * card-neutral renderer vk_raster.c): the stages of a draw as Vulkan
 * GLSL (ati_hd5770_glsl.c). The fragment stage of a pixel program for
 * key k (2D units, cube and array units as layers of one image, 3D
 * units as volumes: k->dim, k->nsl; a unit in padded lies in a larger
 * image and is wrapped by the stage; bary: the host gives barycentrics;
 * NULL: not expressible), the vertex
 * stage that passes window-space triangles through, and a vertex
 * shader's IR as the vertex stage (HD5770GpuDraw.vs). The draw's block
 * of words is HD5770_NU_*, the vertex stage's set-up HD5770_NV_*.
 */
enum {
    HD5770_NU_AREF, HD5770_NU_AFUNC, HD5770_NU_PC, HD5770_NU_PERSP,
    HD5770_NU_CLIPRULE, HD5770_NU_VPX, HD5770_NU_VPY, HD5770_NU_ZMIN,
    HD5770_NU_ZMAX, HD5770_NU_KV, HD5770_NU_VTX, HD5770_NU_ZERO,
    HD5770_NU_CR = 12,          /* 16: the clip rectangles */
    /*
     * per unit: w, h, dst_sel[4], base level, levels, depth, cmp, cmp
     * max, the part of the image the texture takes (x, y), wrap x, y
     */
    HD5770_NU_TEX = 28, HD5770_NU_TEX_WORDS = 16,
    HD5770_NU_BCONST = HD5770_NU_TEX + HD5770_NU_TEX_WORDS * HD5770_GPU_TEX,
    HD5770_NU_LCONST,           /* 32 */
    /* per buffer of the program's vertex fetches: stride, bytes, at byte */
    HD5770_NU_VB = HD5770_NU_LCONST + 32,
    HD5770_NU_WORDS = HD5770_NU_VB + 3 * HD5770_GPU_VB,
    /* vte, sc_mode, window offset, clip, vport[6], 2/w, 2/h, centre, zoff */
    HD5770_NV_X = 152, HD5770_NV_DEF = 168,
    HD5770_NV_WORDS = HD5770_NV_DEF + 128,
};
char *hd5770_glsl_native_fs(const HD5770IR *ir, const HD5770NatKey *k,
                            uint32_t padded, bool bary);
uint32_t hd5770_glsl_native_flat(const HD5770IR *ir, const HD5770NatKey *k);
char *hd5770_glsl_native_pvs(unsigned ninputs, uint32_t flat);
char *hd5770_glsl_native_vs(const HD5770IR *ir, unsigned ninputs,
                            uint32_t flat, const int32_t *src);
HD5770Gpu *hd5770_vk_open(uint8_t *vram, uint64_t vram_size,
                          const char **err);

/* the words of the tables buffer; the number needed if n is too small */
size_t hd5770_glsl_tables(uint32_t *out, size_t n);

#endif
