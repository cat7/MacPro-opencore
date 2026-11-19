/*
 * ATI Radeon HD 5770 (Juniper) draws on the CPU.
 *
 * A draw packet runs the vertex shader (and the fetch shader it calls)
 * once per vertex, assembles rectangles, quads or triangles, applies the
 * viewport and the scissors, rasterises with pixel centres at the
 * position PA_SU_VTX_CNTL names, runs the pixel shader per covered
 * pixel with its inputs interpolated from the vertex shader's parameter
 * exports, and blends the result into colour buffer 0 through the
 * Evergreen surface layout. Textures and vertex buffers are read through
 * the resource descriptors, constants from the DX9 constant file or the
 * constant buffers.
 *
 * Draws using state the model does not implement are skipped and
 * counted by reason; shaders are counted in a census.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include <math.h>
#include <float.h>
#include "qemu/bswap.h"
#include "qemu/timer.h"
#include "ati_hd5770_gfx.h"
#include "ati_hd5770_sh.h"
#include "ati_hd5770_gpu.h"

#define CTX(g, off)     ((g)->ctx[((off) - 0x28000) >> 2])
#define CFG(g, off)     ((g)->cfg[(off) >> 2])

/* context and config registers */
#define VGT_PRIMITIVE_TYPE              0x8958
#define DB_RENDER_CONTROL               0x28000
#define DB_STENCIL_CLEAR                0x28028
#define DB_DEPTH_CLEAR                  0x2802c
#define DB_Z_INFO                       0x28040
#define DB_STENCIL_INFO                 0x28044
#define DB_Z_WRITE_BASE                 0x28050
#define DB_STENCIL_WRITE_BASE           0x28054
#define DB_DEPTH_SIZE                   0x28058
#define GB_ADDR_CONFIG                  0x98f8
#define PA_SC_SCREEN_SCISSOR_TL         0x28030
#define PA_SC_SCREEN_SCISSOR_BR         0x28034
#define SQ_ALU_CONST_BUFFER_SIZE_PS_0   0x28140
#define SQ_ALU_CONST_BUFFER_SIZE_VS_0   0x28180
#define PA_SC_WINDOW_OFFSET             0x28200
#define PA_SC_WINDOW_SCISSOR_TL         0x28204
#define PA_SC_WINDOW_SCISSOR_BR         0x28208
#define PA_SC_CLIPRECT_RULE             0x2820c
#define CB_TARGET_MASK                  0x28238
#define CB_SHADER_MASK                  0x2823c
#define PA_SC_GENERIC_SCISSOR_TL        0x28240
#define PA_SC_GENERIC_SCISSOR_BR        0x28244
#define PA_SC_VPORT_SCISSOR_0_TL        0x28250
#define PA_SC_VPORT_SCISSOR_0_BR        0x28254
#define PA_SC_VPORT_ZMIN_0              0x282d0
#define PA_SC_VPORT_ZMAX_0              0x282d4
#define VGT_MAX_VTX_INDX                0x28400
#define VGT_MIN_VTX_INDX                0x28404
#define VGT_INDX_OFFSET                 0x28408
#define VGT_MULTI_PRIM_IB_RESET_INDX    0x2840c
#define VGT_MULTI_PRIM_IB_RESET_EN      0x28a94
#define SX_ALPHA_TEST_CONTROL           0x28410
#define CB_BLEND_RED                    0x28414
#define DB_STENCILREFMASK               0x28430
#define DB_STENCILREFMASK_BF            0x28434
#define SX_ALPHA_REF                    0x28438
#define PA_CL_VPORT_XSCALE_0            0x2843c
#define SPI_VS_OUT_ID_0                 0x2861c
#define SPI_PS_INPUT_CNTL_0             0x28644
#define SPI_VS_OUT_CONFIG               0x286c4
#define SPI_PS_IN_CONTROL_0             0x286cc
#define SPI_PS_IN_CONTROL_1             0x286d0
#define SPI_INTERP_CONTROL_0            0x286d4
#define SPI_BARYC_CNTL                  0x286e0
#define CB_BLEND0_CONTROL               0x28780
#define DB_DEPTH_CONTROL                0x28800
#define CB_COLOR_CONTROL                0x28808
#define DB_SHADER_CONTROL               0x2880c
#define PA_CL_CLIP_CNTL                 0x28810
#define PA_SU_POLY_OFFSET_DB_FMT_CNTL   0x28b78
#define PA_SU_POLY_OFFSET_CLAMP         0x28b7c
#define PA_SU_POLY_OFFSET_FRONT_SCALE   0x28b80
#define PA_SU_SC_MODE_CNTL              0x28814
#define PA_CL_VTE_CNTL                  0x28818
#define SQ_PGM_START_PS                 0x28840
#define SQ_PGM_START_VS                 0x2885c
#define SQ_PGM_START_FS                 0x288a4
#define SQ_ALU_CONST_CACHE_PS_0         0x28940
#define SQ_ALU_CONST_CACHE_VS_0         0x28980
#define PA_SC_MODE_CNTL_0               0x28a48
#define PA_SC_AA_CONFIG                 0x28be0
#define PA_SU_VTX_CNTL                  0x28c08
#define CB_COLOR0_BASE                  0x28c60
#define CB_COLOR_STRIDE                 0x3c
#define SQ_RESOURCE_0                   0x30000
#define SQ_LOOP_CONST_0                 0x3a200
#define SQ_BOOL_CONST_0                 0x3a500
#define SQ_SAMPLER_0                    0x3c000
#define SQ_VTX_BASE_VTX_LOC             0x3cff0
#define SQ_VTX_START_INST_LOC           0x3cff4

#define RES_VS                          176
#define RES_FS                          992
#define SAMPLER_VS                      18
#define MAX_VERTS                       (1u << 18)
#define MAX_PROG_DWORDS                 8192

enum {
    PRIM_POINTLIST = 1, PRIM_LINELIST = 2, PRIM_LINESTRIP = 3,
    PRIM_TRILIST = 4, PRIM_TRIFAN = 5, PRIM_TRISTRIP = 6,
    PRIM_RECTLIST = 0x11, PRIM_QUADLIST = 0x13, PRIM_QUADSTRIP = 0x14,
    PRIM_POLYGON = 0x15,
};

/* CB_COLOR_CONTROL.MODE */
enum { CB_DISABLE = 0, CB_NORMAL = 1, CB_RESOLVE = 3 };

/* DB_DEPTH_CONTROL */
#define DC_STENCIL      (1u << 0)
#define DC_Z            (1u << 1)
#define DC_Z_WRITE      (1u << 2)
#define DC_BACKFACE     (1u << 7)
/* DB_RENDER_CONTROL */
#define RC_DEPTH_CLEAR  (1u << 0)
#define RC_STENCIL_CLEAR (1u << 1)
#define RC_COPY         (3u << 2)
#define RC_COLOR_DISABLE (1u << 12)

/*
 * PA_SU_SC_MODE_CNTL.FACE = 0: counter-clockwise triangles face front.
 * Window y grows downwards; a positive edge() area is clockwise there.
 */
#define FACE_CW_IS_AREA_POS     1

typedef struct ShEntry {
    uint64_t addr;
    uint32_t hash;
    uint32_t ndw;
    uint8_t *text;              /* its ndw words as they lie in memory */
    HD5770Prog *prog;
    struct ShEntry *next;
} ShEntry;

#define SH_BUCKETS 256

/*
 * The index work of a draw from an index buffer, kept while the
 * buffer's bytes stay as they were read (raw): the indices, the CPU
 * path's slots and distinct vertices, the triangles for the host's
 * vertex stage
 */
typedef struct IdxEntry {
    uint64_t base;
    uint32_t max, nv, isz, swap, ioff, lo, hi, restart, reset, prim;
    uint8_t *raw;
    uint32_t *idx, *slot, *vid, *tidx;
    unsigned nu, ntidx;
    uint64_t tidx_id;           /* names tidx's content */
    uint64_t used, bytes;
    struct IdxEntry *next;
} IdxEntry;

#define IDX_BUCKETS     1024
#define IDX_CACHE_BYTES (128u << 20)
#define IDX_CACHE_AGE   65536   /* draws */

#define HD5770_TEX_LEVELS 15

typedef struct Tex {
    HD5770Surf s;
    unsigned fmt, num, w, h, dim;
    uint32_t sgn;               /* signed per component */
    unsigned dst_sel[4];
    uint32_t samp[3];
    float border[4];
    unsigned comps;
    bool is_float;
    bool unorm8;                /* 8_8_8_8 UNORM */
    unsigned base_lv, levels, depth;    /* BASE_LEVEL, levels from it, depth */
    /*
     * Slices (3D depth, array layers, cube faces) follow each other in
     * a level, slice_bytes apart; z is the slice texel() reads.
     */
    unsigned slices, z;
    float zn;                   /* 3D: z in [0, 1) */
    bool ld;                    /* LD: no filtering between slices */
    /* SAMPLE_C*: DEPTH_COMPARE_FUNCTION + 1 (0 none), reference */
    unsigned cmp;
    float ref;
    unsigned rbits;             /* bits of component 0 */
    uint64_t slice_bytes[HD5770_TEX_LEVELS];
    /* BC1..BC5: 4x4 blocks are the surface's elements; last one decoded */
    unsigned bc;
    uint64_t bc_addr;
    uint32_t bc_texel[16][4];
    /* the micro tile last addressed */
    uint32_t mtx, mty;
    uint64_t mbase;
    /* mipmaps: levels 1..last_lv from MIP_ADDRESS, sampled by LOD */
    bool mip;
    unsigned last_lv, mip_filter;
    float min_lod, max_lod, lod_bias;
    HD5770Surf lv[HD5770_TEX_LEVELS];
    uint32_t lw[HD5770_TEX_LEVELS], lh[HD5770_TEX_LEVELS];
} Tex;

#define TEX_CACHE 8

/* textures set up, by their resource and sampler words */
typedef struct TexMemo {
    uint32_t key[16];
    bool ok;
    Tex t;
} TexMemo;

#define TEX_MEMO        128

#define SH_LAST 64

struct HD5770ShaderCache {
    ShEntry *b[SH_BUCKETS];
    ShEntry *last[4][SH_LAST];  /* by stage and address: the last program */
    IdxEntry *ib[IDX_BUCKETS];
    uint64_t idx_bytes, idx_tick, tidx_ids;
    TexMemo tm[TEX_MEMO];
};

/* the fields of one render target in Draw */
#define MRT_FIRST   offsetof(Draw, rt)
#define MRT_BYTES   (offsetof(Draw, color) - offsetof(Draw, rt))

/* a vertex after the vertex shader and the viewport */
typedef struct Vtx {
    float x, y, z, w;           /* window coordinates, w = 1/clip w */
    float cp[4];                /* clip space position */
    bool clip;                  /* clipped against the near and far planes */
    HD5770Val param[HD5770_PARAMS][4];
} Vtx;

/* the state one draw needs, gathered once */
typedef struct Draw {
    HD5770Gfx *g;
    HD5770GfxStats *st;
    HD5770Span span_tex, span_vb, span_cb;
    /*
     * Render target: colour buffer 0, or the one put_pixel() writes.
     * The fields from rt up to color are one target's (MRT_BYTES).
     */
    HD5770Surf rt;
    unsigned rt_fmt, rt_num, rt_swap, rt_bpe, rt_comps;
    bool rt_blend, rt_float;
    unsigned rt_bits[4], rt_src[4];
    bool rt_unorm8;             /* 8_8_8_8 UNORM */
    uint32_t rt_mtx, rt_mty;    /* the micro tile last addressed */
    uint64_t rt_mbase;
    HD5770Span span_rt;
    uint32_t wmask;
    uint32_t blend;
    bool color;                 /* colour buffer 0 is written */
    /* more render targets: nmrt targets' fields, export cb[k] to target k */
    unsigned nmrt, mrt_cb[8];
    uint8_t *mrt;
    float blend_const[4];
    bool alpha_test;
    unsigned alpha_func;
    float alpha_ref;
    /* clip box, inclusive-exclusive */
    int x0, y0, x1, y1;
    uint32_t cliprule;
    int cr[4][4];               /* clip rects x0, y0, x1, y1 */
    float pix_center;
    /* depth and stencil buffers */
    bool db;                    /* tested or written */
    HD5770Surf zs, ss;
    unsigned zfmt;              /* 1 Z_16, 2 Z_24, 3 Z_32_FLOAT, 0 none */
    bool has_s;
    uint32_t dctl, rctl, srm[2];
    float zclear, zmin, zmax;
    uint32_t sclear;
    bool zexport, sexport;      /* the pixel shader exports depth, stencil */
    HD5770Span span_z, span_s;
    uint32_t z_mtx, z_mty, s_mtx, s_mty;
    uint64_t z_mbase, s_mbase;
    float zoff;                 /* polygon offset of the current triangle */
    /* clipping against the near and far planes (PA_CL_CLIP_CNTL) */
    bool clip_on, clip_dx, clip_near, clip_far;
    /* culling and facing */
    unsigned cull;              /* bit 0 front, bit 1 back */
    bool face_cw, prov_last, front;
    bool mips;                  /* a PS texture is mipmapped */
    /* resolve: colour buffer 0 is copied to colour buffer 1 */
    bool resolve;
    HD5770Surf rs;
    unsigned rs_fmt, rs_num, rs_swap;
    uint32_t rs_mtx, rs_mty;
    uint64_t rs_mbase;
    HD5770Span span_rs;
    /* PS input p comes from VS parameter ps_src[p], or a default */
    int ps_src[HD5770_PARAMS];
    float ps_default[HD5770_PARAMS][4];
    bool ps_flat[HD5770_PARAMS];
    bool persp;
    unsigned pos_gpr;
    bool pos_ena;
    /* current primitive, for the pixel shader's interpolation */
    const Vtx *tv[3];
    const Vtx *tvp;             /* provoking vertex: flat inputs */
    float area;
    int px, py;                 /* current pixel */
    int wy0, wy1;               /* rows the CPU rasteriser visited */
    HD5770Thread *nt;           /* scratch thread of neighbour runs */
    float b0, b1, b2;           /* barycentrics of the current pixel */
    float pw;                   /* 1/sum, perspective */
    HD5770ShEnv vs_env, ps_env;
    const HD5770Prog *vs, *ps, *fs;
    bool ok;
    /* pixel shader lanes */
    HD5770Lanes *lanes, *nh, *nv;   /* nh, nv: neighbours for gradients */
    unsigned nlanes;            /* 1, 3 with neighbours; made when rasterised */
    bool gpu_prog;              /* the program may go to the host GPU */
    unsigned gfb;               /* why the last draw stayed on the CPU */
    /* programs first seen by a draw looked at ahead: not kept */
    HD5770Prog *ahead_prog[3];
    unsigned ahead_progs;
    /* HD5770_WHY_*: of the last part, and the draw's first CPU reason */
    unsigned cwhy, why;
    HD5770Val *kv;
    int lx[HD5770_LANES], ly[HD5770_LANES];
    /* the triangles of a draw for the host GPU */
    GArray *gtri, *gprm;
    const HD5770GpuVs *gvs;     /* the request goes with its vertex stage */
    unsigned gnin;
    /* textures set up so far in this draw, by resource and sampler */
    Tex tex[TEX_CACHE];
    uint32_t tex_key[TEX_CACHE];
    unsigned n_tex;
    /* vertex shader constant buffers read so far in this instance */
    HD5770Val (*kc[16])[4];
    uint64_t *kc_ok[16];
} Draw;

#define KC_MAX          4096

/* c / 255.0f, as comp_value() computes it for an 8-bit UNORM component */
static float unorm8[256];

/* GB_ADDR_CONFIG reads as the Juniper value until written */
static uint32_t addr_config(HD5770Gfx *g)
{
    return CFG(g, GB_ADDR_CONFIG) ? CFG(g, GB_ADDR_CONFIG)
                                  : HD5770_GB_ADDR_CONFIG_JUNIPER;
}

/* ---------------------------------------------------------------- */
/* Host GPU: what is queued, and waiting for it                     */

/* the VRAM offset of [mc, mc + len), if it lies in VRAM */
static bool vram_off(HD5770Gfx *g, uint64_t mc, uint64_t len, uint64_t *off)
{
    uint64_t n;
    uint8_t *p;

    if (!g->vram) {
        return false;
    }
    p = g->mem.map(g->mem.opaque, mc, &n, false);
    /* contiguous VRAM only: not GART pages that happen to be VRAM */
    if (!p || p < g->vram || p >= g->vram + g->vram_size || n < len) {
        return false;
    }
    *off = p - g->vram;
    return *off + len <= g->vram_size;
}

static void pend_add(HD5770Gfx *g, uint64_t lo, uint64_t hi)
{
    unsigned i;

    for (i = 0; i < g->npend; i++) {
        if (lo <= g->pend[i][1] && g->pend[i][0] <= hi) {
            g->pend[i][0] = MIN(g->pend[i][0], lo);
            g->pend[i][1] = MAX(g->pend[i][1], hi);
            return;
        }
    }
    if (g->npend == HD5770_PENDING) {
        /* one range covering the last one and this */
        g->pend[g->npend - 1][0] = MIN(g->pend[g->npend - 1][0], lo);
        g->pend[g->npend - 1][1] = MAX(g->pend[g->npend - 1][1], hi);
        return;
    }
    g->pend[g->npend][0] = lo;
    g->pend[g->npend][1] = hi;
    g->npend++;
}

/* a plane with split tiles: the bytes from its base its elements reach */
static uint64_t split_cov(HD5770Gfx *g, const HD5770Surf *s)
{
    uint64_t cov = 0;
    uint32_t x, y;
    unsigned i;

    for (i = 0; i < ARRAY_SIZE(g->scov); i++) {
        const HD5770Surf *c = &g->scov[i].s;

        if (g->scov[i].cov && c->pitch == s->pitch &&
            c->height == s->height && c->mode == s->mode &&
            c->bpe == s->bpe && c->banks == s->banks &&
            c->bankw == s->bankw && c->bankh == s->bankh &&
            c->mtilea == s->mtilea && c->tile_split == s->tile_split &&
            c->pipes == s->pipes &&
            c->pipe_interleave == s->pipe_interleave &&
            c->nondisp == s->nondisp) {
            return g->scov[i].cov;
        }
    }
    for (y = 0; y < s->height; y++) {
        for (x = 0; x < s->pitch; x++) {
            cov = MAX(cov, hd5770_surf_offset(s, x, y) + s->bpe);
        }
    }
    i = g->scov_next++ % ARRAY_SIZE(g->scov);
    g->scov[i].s = *s;
    g->scov[i].cov = cov;
    return cov;
}

/* the same for a range written; without room it joins the nearest */
static void wpend_add(HD5770Gfx *g, uint64_t lo, uint64_t hi)
{
    uint64_t gap, best = UINT64_MAX;
    unsigned i, at = 0;

    pend_add(g, lo, hi);
    for (i = 0; i < g->nwpend; i++) {
        if (lo <= g->wpend[i][1] && g->wpend[i][0] <= hi) {
            gap = 0;
        } else {
            gap = lo > g->wpend[i][1] ? lo - g->wpend[i][1]
                                      : g->wpend[i][0] - hi;
        }
        if (gap < best) {
            best = gap;
            at = i;
        }
    }
    if (best && g->nwpend < HD5770_WPENDING) {
        g->wpend[g->nwpend][0] = lo;
        g->wpend[g->nwpend][1] = hi;
        g->nwpend++;
        return;
    }
    g->wpend[at][0] = MIN(g->wpend[at][0], lo);
    g->wpend[at][1] = MAX(g->wpend[at][1], hi);
}

static void gpu_lock(HD5770Gfx *g, bool on)
{
    if (g->lock) {
        g->lock(g, on);
    }
}

bool hd5770_gfx_built(HD5770Gfx *g)
{
    bool r = g->gpu && g->gpu->built;

    if (r) {
        g->gpu->built = false;
    }
    return r;
}

struct HD5770GfxAhead {
    HD5770Gfx g;
    HD5770GfxStats st;
};

HD5770GfxAhead *hd5770_gfx_ahead_begin(HD5770Gfx *g, const uint32_t *ctx,
                                       const uint32_t *cfg,
                                       const uint32_t *alu)
{
    HD5770GfxAhead *a;

    if (!g->gpu || g->gpu_mode != HD5770_GPU_ON || !g->gpu->ops->prep) {
        return NULL;
    }
    gpu_lock(g, true);
    a = g_new(HD5770GfxAhead, 1);
    a->g = *g;
    a->st = *g->st;
    g->ctx = ctx;
    g->cfg = cfg;
    g->alu = alu;
    g->dry = true;
    return a;
}

void hd5770_gfx_ahead_end(HD5770Gfx *g, HD5770GfxAhead *a)
{
    uint8_t *ram = g->ram;
    uint64_t ram_size = g->ram_size;

    *g->st = a->st;
    *g = a->g;
    /* guest RAM attached meanwhile stays */
    g->ram = ram;
    g->ram_size = ram_size;
    g_free(a);
    gpu_lock(g, false);
    /* the builds: other threads may use the backend meanwhile */
    g->gpu->ops->prep(g->gpu->b, NULL);
}

void hd5770_gfx_sync(HD5770Gfx *g)
{
    if (!g->gpu || g->dry) {
        return;
    }
    gpu_lock(g, true);
    if (g->npend || g->ram_pend || g->nvpend || g->ram_vpend) {
        g->gpu->ops->sync(g->gpu->b);
        g->npend = 0;
        g->nwpend = 0;
        g->nvpend = 0;
        g->ram_vpend = false;
        g->ram_pend = false;
        g->ram_wpend = false;
        g->gst.syncs++;
    }
    gpu_lock(g, false);
}

bool hd5770_gfx_busy(HD5770Gfx *g)
{
    return g->gpu && (g->npend || g->ram_pend || g->nvpend || g->ram_vpend);
}

uint64_t hd5770_gfx_fence(HD5770Gfx *g)
{
    uint64_t t;

    gpu_lock(g, true);
    t = g->gpu->ops->fence(g->gpu->b);
    if (!t) {
        /* everything queued is in memory, or kept by the backend */
        if (!g->gpu->kept) {
            g->npend = 0;
            g->nwpend = 0;
        }
        g->nvpend = 0;
        g->ram_vpend = false;
        g->ram_pend = false;
        g->ram_wpend = false;
    }
    gpu_lock(g, false);
    return t;
}

/*
 * A read minds only what queued draws write: with the native pipeline.
 * The other backends are waited for by any use of the bytes, as they
 * expect (their copies and pools turn over at a sync).
 */
static bool rd_written(const HD5770Gfx *g)
{
    return g->gpu && g->gpu->ops->native;
}

/*
 * queued draws use VRAM [off, off + len); vb: or read it in place;
 * rd: write it
 */
static bool pend_hit(const HD5770Gfx *g, uint64_t off, uint64_t len, bool vb,
                     bool rd)
{
    unsigned i;

    rd &= rd_written(g);
    for (i = 0; rd && i < g->nwpend; i++) {
        if (off < g->wpend[i][1] && g->wpend[i][0] < off + len) {
            return true;
        }
    }
    for (i = 0; !rd && i < g->npend; i++) {
        if (off < g->pend[i][1] && g->pend[i][0] < off + len) {
            return true;
        }
    }
    for (i = 0; vb && i < g->nvpend; i++) {
        if (off < g->vpend[i][1] && g->vpend[i][0] < off + len) {
            return true;
        }
    }
    return false;
}

bool hd5770_gfx_pending(HD5770Gfx *g, uint64_t off, uint64_t len, bool rd)
{
    bool hit = false;

    if (!g->gpu) {
        return false;
    }
    gpu_lock(g, true);
    hit = pend_hit(g, off, len, !rd, rd);
    g->gst.touch_syncs += hit;
    gpu_lock(g, false);
    return hit;
}

/* bytes [lo, hi) leave the ranges written: nothing queued writes them */
static void wpend_cut(HD5770Gfx *g, uint64_t lo, uint64_t hi)
{
    unsigned i;

    for (i = 0; i < g->nwpend; i++) {
        uint64_t a = g->wpend[i][0], b = g->wpend[i][1];

        if (lo >= b || a >= hi) {
            continue;
        }
        if (lo <= a && b <= hi) {
            g->wpend[i][0] = g->wpend[g->nwpend - 1][0];
            g->wpend[i][1] = g->wpend[g->nwpend - 1][1];
            g->nwpend--;
            i--;
        } else if (lo <= a) {
            g->wpend[i][0] = hi;
        } else if (b <= hi) {
            g->wpend[i][1] = lo;
        } else if (g->nwpend < HD5770_WPENDING) {
            g->wpend[i][1] = lo;
            g->wpend[g->nwpend][0] = hi;
            g->wpend[g->nwpend][1] = b;
            g->nwpend++;
        }
    }
}

/* the backend brings VRAM [off, off + len) up to date, and no more */
static void pend_sync_range(HD5770Gfx *g, uint64_t off, uint64_t len)
{
    g->gpu->ops->sync_range(g->gpu->b, off, len);
    g->gst.range_syncs++;
    /* the work queued is done: nothing reads memory in place */
    g->nvpend = 0;
    g->ram_vpend = false;
    if (!g->gpu->kept) {
        g->npend = 0;
        g->nwpend = 0;
        g->ram_pend = false;
        g->ram_wpend = false;
    } else {
        unsigned i;

        wpend_cut(g, off, off + len);
        /* what was read is read: the ranges kept are the ones pending */
        g->npend = 0;
        for (i = 0; i < g->nwpend; i++) {
            pend_add(g, g->wpend[i][0], g->wpend[i][1]);
        }
        g->ram_pend = g->ram_wpend;
        if (pend_hit(g, off, len, false, true)) {
            /* no room to cut the range out */
            hd5770_gfx_sync(g);
        }
    }
}

/*
 * vb false: before a read, which the vertex buffers queued and the
 * ranges only read do not mind
 */
static void pend_touch(HD5770Gfx *g, uint64_t off, uint64_t len, bool vb)
{
    if (!g->gpu || g->dry) {
        return;
    }
    gpu_lock(g, true);
    if (pend_hit(g, off, len, vb, !vb)) {
        g->gst.touch_syncs++;
        if (g->gpu->ops->sync_range) {
            pend_sync_range(g, off, len);
        } else {
            hd5770_gfx_sync(g);
        }
    }
    gpu_lock(g, false);
}

void hd5770_gfx_touch(HD5770Gfx *g, uint64_t off, uint64_t len)
{
    pend_touch(g, off, len, true);
}

/*
 * The CPU writes surface s: after the queued draws that touch it. With
 * the native pipeline a surface not in one piece of VRAM is taken page
 * by page: runs of VRAM pages by range, and guest RAM minds only the
 * draws that use guest RAM.
 */
static void pend_write(HD5770Gfx *g, const HD5770Surf *s)
{
    uint64_t size = hd5770_surf_size(s), off, at, n, lo = 0, hi = 0;
    bool all = !rd_written(g);

    if (vram_off(g, s->base, size, &off)) {
        hd5770_gfx_touch(g, off, size);
        return;
    }
    if (!g->ram_pend && !g->npend) {
        return;
    }
    for (at = s->base & ~0xfffull; !all && at < s->base + size; at += 4096) {
        uint8_t *p = g->mem.map(g->mem.opaque, at, &n, false);

        if (p && p >= g->vram && p < g->vram + g->vram_size) {
            off = p - g->vram;
            if (off != hi) {
                hd5770_gfx_touch(g, lo, hi - lo);
                lo = off;
            }
            hi = off + 4096;
        } else if (p) {
            all = g->ram_pend || g->ram_vpend;
        }
    }
    if (all) {
        hd5770_gfx_sync(g);
    } else {
        hd5770_gfx_touch(g, lo, hi - lo);
    }
}

/* the CPU wrote memory at mc: a backend with a copy of it checks it again */
static void cpu_wrote(HD5770Gfx *g, uint64_t mc, uint64_t len)
{
    uint64_t off;

    if (!g->gpu || !g->gpu->ops->wrote || !len) {
        return;
    }
    if (vram_off(g, mc, len, &off)) {
        g->gpu->ops->wrote(g->gpu->b, off, len);
    } else {
        g->gpu->ops->wrote(g->gpu->b, 0, 0);
    }
}

/* the CPU reads memory at mc: queued draws first, if they may write it */
static void pend_check(HD5770Gfx *g, uint64_t mc, uint64_t len)
{
    uint64_t off, n;
    uint8_t *p;
    bool ram = rd_written(g) ? g->ram_wpend : g->ram_pend;

    if (!(rd_written(g) ? g->nwpend : g->npend) && !ram) {
        return;
    }
    if (vram_off(g, mc, len, &off)) {
        pend_touch(g, off, len, false);
        return;
    }
    p = g->vram ? g->mem.map(g->mem.opaque, mc, &n, false) : NULL;
    if (p && p >= g->vram && p < g->vram + g->vram_size && n >= len) {
        /* VRAM through a GART page */
        pend_touch(g, p - g->vram, len, false);
    } else if (ram || !p ||
               (p >= g->vram && p < g->vram + g->vram_size)) {
        hd5770_gfx_sync(g);
    }
}

/*
 * The host GPU will read memory at mc in place (a vertex buffer): queued
 * draws first, if they write it
 */
static void pend_check_vb(HD5770Gfx *g, uint64_t mc, uint64_t len)
{
    uint64_t off, n;
    uint8_t *p;

    if (!g->nwpend && !g->ram_wpend) {
        return;
    }
    if (vram_off(g, mc, len, &off)) {
        pend_touch(g, off, len, false);
        return;
    }
    p = g->vram ? g->mem.map(g->mem.opaque, mc, &n, false) : NULL;
    if (p && p >= g->vram && p < g->vram + g->vram_size && n >= len) {
        pend_touch(g, p - g->vram, len, false);
    } else if (g->ram_wpend || !p ||
               (p >= g->vram && p < g->vram + g->vram_size)) {
        hd5770_gfx_sync(g);
    }
}

/*
 * A surface for the host GPU: in VRAM, or page by page in guest RAM;
 * size bytes from its base (0: the surface's own size).
 */
static bool gpu_surf_n(HD5770Gfx *g, const HD5770Surf *s, uint64_t size,
                       bool write, HD5770GpuSurf *o)
{
    uint64_t first = s->base & ~0xfffull, n, i;
    uint32_t *pages;

    if (!size) {
        size = hd5770_surf_size(s);
    }

    memset(o, 0, sizeof(*o));
    o->s = *s;
    if (!s->fast && !(s->split && g->gpu && g->gpu->ops->full)) {
        return false;
    }
    if (vram_off(g, s->base, size, &o->voff)) {
        return true;
    }
    n = DIV_ROUND_UP((s->base & 0xfff) + size, 4096);
    if (!g->ram_for || n > 16384) {
        return false;
    }
    pages = g_new(uint32_t, n);
    for (i = 0; i < n; i++) {
        uint64_t len;
        uint8_t *p = g->mem.map(g->mem.opaque, first + i * 4096, &len, write);

        if (!p || len < 4096 || !g->ram_for(g, p) || p < g->ram ||
            p + 4096 > g->ram + g->ram_size || p - g->ram > UINT32_MAX - 4096) {
            g_free(pages);
            return false;
        }
        pages[i] = p - g->ram;
    }
    o->space = 1;
    o->pages = pages;
    o->npages = n;
    o->pg0 = s->base & 0xfff;
    return true;
}

static bool gpu_surf(HD5770Gfx *g, const HD5770Surf *s, bool write,
                     HD5770GpuSurf *o)
{
    return gpu_surf_n(g, s, 0, write, o);
}

/* ---------------------------------------------------------------- */
/* Shader cache and census                                          */

static uint32_t hash32(const uint32_t *p, uint32_t n)
{
    uint32_t h = 2166136261u, i;

    for (i = 0; i < n; i++) {
        h = (h ^ p[i]) * 16777619u;
    }
    return h;
}

static void census(HD5770GfxStats *st, uint64_t addr, uint32_t hash,
                   unsigned stage, const HD5770Prog *p)
{
    uint32_t i;

    for (i = 0; i < st->n_sh; i++) {
        if (st->sh[i].hash == hash && st->sh[i].stage == stage) {
            st->sh[i].uses++;
            return;
        }
    }
    if (st->n_sh < HD5770_SHADER_SLOTS) {
        HD5770ShaderInfo *si = &st->sh[st->n_sh++];

        memset(si, 0, sizeof(*si));
        si->addr = addr;
        si->hash = hash;
        si->stage = stage;
        si->uses = 1;
        hd5770_prog_census(p, si);
    } else {
        st->sh_other++;
    }
}

static const HD5770Prog *get_prog(Draw *d, uint64_t addr, unsigned stage)
{
    HD5770Gfx *g = d->g;
    uint32_t *buf, ndw, hash;
    ShEntry *e, **last;
    uint8_t *text;
    unsigned b;

    if (!addr) {
        return NULL;
    }
    last = &g->cache->last[stage & 3][(addr >> 8) % SH_LAST];
    e = *last;
    if (e && e->addr == addr) {
        /* the program last found here, if its words are still there */
        uint64_t n = 0;
        uint8_t *p = g->mem.map(g->mem.opaque, addr, &n, false);

        if (p && e->text && n >= (uint64_t)e->ndw * 4 &&
            !memcmp(p, e->text, (size_t)e->ndw * 4)) {
            census(d->st, addr, e->hash, stage, e->prog);
            return e->prog;
        }
    }
    buf = g_new(uint32_t, MAX_PROG_DWORDS);
    hd5770_mem_read(&g->mem, addr, buf, 4096);
    ndw = hd5770_prog_size(buf, 1024);
    if (ndw > 1024 - 4) {
        hd5770_mem_read(&g->mem, addr, buf, MAX_PROG_DWORDS * 4);
        ndw = hd5770_prog_size(buf, MAX_PROG_DWORDS);
    }
    text = g_memdup2(buf, (size_t)ndw * 4);
    for (b = 0; b < ndw; b++) {
        buf[b] = le32_to_cpu(buf[b]);
    }
    hash = hash32(buf, ndw) ^ stage;
    b = hash % SH_BUCKETS;
    for (e = g->cache->b[b]; e; e = e->next) {
        if (e->hash == hash && e->ndw == ndw && e->addr == addr) {
            break;
        }
    }
    if (!e && g->dry) {
        /* parsed again, and counted, by the draw itself */
        HD5770Prog *p = hd5770_prog_new(buf, ndw, stage, d->st);

        g_free(text);
        g_free(buf);
        if (d->ahead_progs == ARRAY_SIZE(d->ahead_prog)) {
            hd5770_prog_free(p);
            return NULL;
        }
        d->ahead_prog[d->ahead_progs++] = p;
        return p;
    }
    if (!e) {
        e = g_new0(ShEntry, 1);
        e->addr = addr;
        e->hash = hash;
        e->ndw = ndw;
        e->prog = hd5770_prog_new(buf, ndw, stage, d->st);
        e->text = text;
        text = NULL;
        e->next = g->cache->b[b];
        g->cache->b[b] = e;
    }
    *last = e;
    g_free(text);
    g_free(buf);
    census(d->st, addr, hash, stage, e->prog);
    return e->prog;
}

/* ---------------------------------------------------------------- */
/* Element conversion                                               */

static float half_to_float(uint16_t h)
{
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff, f;
    HD5770Val v;

    if (e == 0) {
        if (!m) {
            f = s << 31;
        } else {
            e = 127 - 15 + 1;
            while (!(m & 0x400)) {
                m <<= 1;
                e--;
            }
            m &= 0x3ff;
            f = (s << 31) | (e << 23) | (m << 13);
        }
    } else if (e == 31) {
        f = (s << 31) | 0x7f800000 | (m << 13);
    } else {
        f = (s << 31) | ((e + 127 - 15) << 23) | (m << 13);
    }
    v.u = f;
    return v.f;
}

/* a float as a half, rounded to nearest even */
static uint16_t float_to_half(float f)
{
    HD5770Val v;
    uint32_t x, s, m;
    int e;

    v.f = f;
    x = v.u;
    s = (x >> 16) & 0x8000;
    e = (int)((x >> 23) & 0xff) - 127 + 15;
    m = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff) {
        return s | 0x7c00 | (m ? 0x200 | (m >> 13) : 0);
    }
    if (e >= 31) {
        return s | 0x7c00;
    }
    if (e <= 0) {
        uint32_t sh, r, h;

        if (e < -10) {
            return s;
        }
        m |= 0x800000;
        sh = 14 - e;
        h = m >> sh;
        r = m & ((1u << sh) - 1);
        if (r > (1u << (sh - 1)) || (r == (1u << (sh - 1)) && (h & 1))) {
            h++;
        }
        return s | h;
    }
    {
        uint32_t h = s | (e << 10) | (m >> 13), r = m & 0x1fff;

        if (r > 0x1000 || (r == 0x1000 && (h & 1))) {
            h++;
        }
        return h;
    }
}

/* one component of an element to a shader value */
static HD5770Val comp_value(uint32_t c, unsigned bits, bool is_float,
                            bool is_signed, unsigned num)
{
    HD5770Val v;

    if (is_float) {
        if (bits == 16) {
            v.f = half_to_float(c);
        } else {
            v.u = c;
        }
        return v;
    }
    if (is_signed && bits < 32 && (c >> (bits - 1)) & 1) {
        c |= ~((1u << bits) - 1);
    }
    switch (num) {
    case HD5770_NUM_NORM:
        if (bits == 32) {
            v.f = is_signed ? (int32_t)c / 2147483647.0f : c / 4294967295.0f;
        } else if (is_signed) {
            v.f = MAX((int32_t)c / (float)((1u << (bits - 1)) - 1), -1.0f);
        } else {
            v.f = c / (float)((1u << bits) - 1);
        }
        break;
    case HD5770_NUM_INT:
        v.u = c;
        break;
    default:
        v.f = is_signed ? (float)(int32_t)c : (float)c;
        break;
    }
    return v;
}

/* ---------------------------------------------------------------- */
/* Vertex fetch                                                     */

/*
 * Constant buffer 0 of a stage is the DX9 constant file the CP keeps
 * from SET_ALU_CONST: fetch buffer 128 reads it as float4 elements.
 */
static void vfetch_stage(Draw *d, const uint32_t *w, uint32_t index,
                         HD5770Val v[4], unsigned base_r, const uint32_t *cfile)
{
    HD5770Gfx *g = d->g;
    unsigned bid = (w[0] >> 8) & 0xff, id = bid + base_r;
    const uint32_t *r = &g->ctx[(SQ_RESOURCE_0 - 0x28000) / 4 + id * 8];
    uint64_t base = r[0] | ((uint64_t)(r[2] & 0xff) << 32);
    uint32_t size = r[1] + 1, stride = (r[2] >> 8) & 0x7ff;
    unsigned fmt, num, sgn, bits[4], n, k;
    uint32_t off = w[2] & 0xffff, c[4];
    uint64_t at;
    uint8_t buf[16];

    if (bid == 128) {
        uint32_t e = index + off / 16;

        for (k = 0; k < 4; k++) {
            v[k].u = e < 256 ? cfile[e * 4 + k] : 0;
        }
        d->st->vtx_fetch++;
        return;
    }
    v[0].f = v[1].f = v[2].f = 0.0f;
    v[3].f = 1.0f;
    if ((w[1] >> 21) & 1) {
        fmt = (r[2] >> 20) & 0x3f;
        num = (r[2] >> 26) & 3;
        sgn = (r[2] >> 28) & 1;
    } else {
        fmt = (w[1] >> 22) & 0x3f;
        num = (w[1] >> 28) & 3;
        sgn = (w[1] >> 30) & 1;
    }
    n = hd5770_fmt_bytes(fmt);
    if (!n) {
        d->st->gap_vtx_fmt[fmt]++;
        d->ok = false;
        return;
    }
    if (((w[2] >> 16) & 3) || ((r[2] >> 30) & 3)) {
        d->st->gap_misc[HD5770_GAP_ROP]++;
    }
    at = (uint64_t)index * stride + off;
    if (at + n > size) {
        v[0].u = v[1].u = v[2].u = 0;
        return;
    }
    pend_check(g, base + at, n);
    hd5770_mem_read(&g->mem, base + at, buf, n);
    hd5770_fmt_unpack(fmt, buf, c, bits);
    for (k = 0; k < 4 && bits[k]; k++) {
        v[k] = comp_value(c[k], bits[k], hd5770_fmt_is_float(fmt), sgn, num);
    }
    d->st->vtx_fetch++;
}

static void vfetch_vs(void *opaque, const uint32_t *w, uint32_t index,
                      unsigned stage, HD5770Val v[4])
{
    Draw *d = opaque;

    /* FETCH_TYPE: base vertex, start instance or no offset */
    switch ((w[0] >> 5) & 3) {
    case 0:
        index += d->g->ind_draw ? d->g->ind_vtx : CTX(d->g, SQ_VTX_BASE_VTX_LOC);
        break;
    case 1:
        index += d->g->ind_draw ? d->g->ind_inst
                                : CTX(d->g, SQ_VTX_START_INST_LOC);
        break;
    }
    vfetch_stage(d, w, index, v, stage == HD5770_STAGE_FS ? RES_FS : RES_VS,
                 d->g->alu + 1024);
}

static void vfetch_ps(void *opaque, const uint32_t *w, uint32_t index,
                      unsigned stage, HD5770Val v[4])
{
    Draw *d = opaque;

    vfetch_stage(d, w, index, v, 0, d->g->alu);
}

/* ---------------------------------------------------------------- */
/* Textures                                                       */

static void tex_levels(Tex *t, uint64_t mip_addr);
static unsigned bc_kind(unsigned fmt);

static bool tex_setup_regs(Draw *d, unsigned id, unsigned sid, Tex *t);

/*
 * the texture of resource id with sampler sid: as set up before with
 * these registers, or set up in tmp; valid until the next one is asked for
 */
static const Tex *tex_peek(Draw *d, unsigned id, unsigned sid, Tex *tmp)
{
    HD5770Gfx *g = d->g;
    const uint32_t *r = &g->ctx[(SQ_RESOURCE_0 - 0x28000) / 4 + id * 8];
    const uint32_t *sm = &g->ctx[(SQ_SAMPLER_0 - 0x28000) / 4 + sid * 3];
    uint64_t gaps = d->st->gap_misc[HD5770_GAP_MSAA] +
                    d->st->gap_misc[HD5770_GAP_MIP];
    uint32_t key[16] = { 0 }, h;
    TexMemo *e;

    memcpy(key, r, 8 * 4);
    memcpy(key + 8, sm, 3 * 4);
    if (((sm[0] >> 20) & 3) == 3) {
        unsigned st = sid < SAMPLER_VS ? 0 : sid < 2 * SAMPLER_VS ? 1 : 2;

        memcpy(key + 11, g->border[st][(sid - st * SAMPLER_VS) & 31], 16);
    }
    key[15] = addr_config(g);
    h = hash32(key, 16);
    e = &g->cache->tm[(h ^ h >> 16) % TEX_MEMO];
    if (e->ok && !memcmp(e->key, key, 16 * 4)) {
        return &e->t;
    }
    if (!tex_setup_regs(d, id, sid, tmp)) {
        return NULL;
    }
    if (gaps == d->st->gap_misc[HD5770_GAP_MSAA] +
                d->st->gap_misc[HD5770_GAP_MIP]) {
        memcpy(e->key, key, 16 * 4);
        e->t = *tmp;
        e->ok = true;
    }
    return tmp;
}

static bool tex_setup(Draw *d, unsigned id, unsigned sid, Tex *t)
{
    const Tex *p = tex_peek(d, id, sid, t);

    if (p && p != t) {
        *t = *p;
    }
    return p != NULL;
}

static bool tex_setup_regs(Draw *d, unsigned id, unsigned sid, Tex *t)
{
    HD5770Gfx *g = d->g;
    const uint32_t *r = &g->ctx[(SQ_RESOURCE_0 - 0x28000) / 4 + id * 8];
    const uint32_t *sm = &g->ctx[(SQ_SAMPLER_0 - 0x28000) / 4 + sid * 3];
    unsigned k;

    memset(t, 0, sizeof(*t));
    if (((r[7] >> 30) & 3) != 2) {
        return false;
    }
    t->dim = r[0] & 7;
    t->w = ((r[0] >> 18) & 0x3fff) + 1;
    t->h = ((r[1]) & 0x3fff) + 1;
    t->fmt = r[7] & 0x3f;
    t->num = (r[4] >> 8) & 3;
    for (k = 0; k < 4; k++) {
        t->sgn |= ((r[4] >> (2 * k)) & 3) == 1 ? 1u << k : 0;
        t->dst_sel[k] = (r[4] >> (16 + 3 * k)) & 7;
    }
    t->base_lv = r[4] >> 28;
    t->levels = (r[5] & 0xf) >= t->base_lv ? (r[5] & 0xf) - t->base_lv + 1 : 1;
    t->depth = t->dim == 2 || t->dim >= 4 ? ((r[1] >> 14) & 0x1fff) + 1 : 1;
    t->s.base = (uint64_t)r[2] << 8;
    t->s.pitch = (((r[0] >> 6) & 0xfff) + 1) * 8;
    t->s.height = t->h;
    t->s.mode = (r[1] >> 28) & 0xf;
    t->s.nondisp = (r[0] >> 5) & 1;
    t->s.bpe = hd5770_fmt_bytes(t->fmt);
    t->bc = bc_kind(t->fmt);
    if (t->bc) {
        /* the elements are 4x4 blocks; PITCH counts texels */
        t->s.bpe = t->bc == 1 || t->bc == 4 ? 8 : 16;
        t->s.pitch = MAX(t->s.pitch / 4, 1u);
        t->s.height = DIV_ROUND_UP(t->h, 4);
    }
    t->s.mtilea = 1u << ((r[7] >> 6) & 3);
    t->s.bankw = 1u << ((r[7] >> 8) & 3);
    t->s.bankh = 1u << ((r[7] >> 10) & 3);
    t->s.banks = 2u << ((r[7] >> 16) & 3);
    t->s.tile_split = 64u << ((r[6] >> 29) & 7);
    hd5770_surf_set_addr_config(&t->s, addr_config(g));
    hd5770_surf_prep(&t->s);
    memcpy(t->samp, sm, sizeof(t->samp));
    switch ((sm[0] >> 20) & 3) {
    case 1:
        t->border[3] = 1.0f;
        break;
    case 2:
        t->border[0] = t->border[1] = t->border[2] = t->border[3] = 1.0f;
        break;
    case 3: {
        /* the colour the sampler's border registers hold */
        unsigned st = sid < SAMPLER_VS ? 0 : sid < 2 * SAMPLER_VS ? 1 : 2;

        memcpy(t->border, g->border[st][(sid - st * SAMPLER_VS) & 31],
               sizeof(t->border));
        break;
    }
    default:
        break;
    }
    if (!t->s.bpe) {
        d->st->gap_tex_fmt[t->fmt]++;
        return false;
    }
    if (!hd5770_surf_mode_ok(t->s.mode)) {
        d->st->gap_mode[t->s.mode]++;
        return false;
    }
    if (t->dim == 6) {
        /*
         * 2D_MSAA: sample 0 of a 2D texture. Apple's depth copies give
         * single-sample depth buffers four samples here.
         */
        if (r[5] & 0xf) {
            d->st->gap_misc[HD5770_GAP_MSAA]++;
        }
        t->dim = 1;
    }
    if (t->dim == 7) {
        d->st->gap_misc[HD5770_GAP_TEX_DIM]++;
        return false;
    }
    /* 3D: depth slices; CUBE: six faces +X -X +Y -Y +Z -Z; arrays: layers */
    t->slices = t->dim == 3 ? 6 : t->depth;
    t->slice_bytes[0] = hd5770_surf_size(&t->s);
    if (t->dim != 0 && (r[5] & 0xf) > (r[4] >> 28) && ((sm[0] >> 15) & 3)) {
        if (r[4] >> 28) {
            /* a base level other than 0: level 0 is sampled */
            d->st->gap_misc[HD5770_GAP_MIP]++;
        } else {
            uint32_t b = sm[2] & 0x3fff;

            t->mip = true;
            t->last_lv = MIN(r[5] & 0xf, HD5770_TEX_LEVELS - 1);
            t->mip_filter = (sm[0] >> 15) & 3;
            t->min_lod = (sm[1] & 0xfff) / 256.0f;
            t->max_lod = ((sm[1] >> 12) & 0xfff) / 256.0f;
            t->lod_bias = (int32_t)(b << 18) / (float)(1 << 26);
        }
    }
    if (t->dim == 0 || t->dim == 4) {
        t->h = 1;
    }
    t->comps = t->bc ? (t->bc == 4 ? 1 : t->bc == 5 ? 2 : 4)
                     : hd5770_fmt_comps(t->fmt);
    t->is_float = hd5770_fmt_is_float(t->fmt);
    t->unorm8 = t->fmt == HD5770_FMT_8_8_8_8 && t->num == HD5770_NUM_NORM &&
                !t->sgn;
    {
        uint8_t zero[16] = { 0 };
        uint32_t c[4];
        unsigned b[4];

        hd5770_fmt_unpack(t->fmt, zero, c, b);
        t->rbits = b[0];
    }
    t->mtx = t->mty = UINT32_MAX;
    if (t->mip) {
        tex_levels(t, (uint64_t)r[3] << 8);
    }
    return true;
}

/* offset of element (x, y) of a prepared surface, through a micro tile memo */
static inline uint64_t surf_off(const HD5770Surf *s, uint32_t *mtx,
                                uint32_t *mty, uint64_t *mbase,
                                uint32_t x, uint32_t y)
{
    if (!s->fast || s->mode < HD5770_ARRAY_1D_TILED_THIN1) {
        return hd5770_surf_offset_fast(s, x, y);
    }
    if ((x >> 3) != *mtx || (y >> 3) != *mty) {
        *mtx = x >> 3;
        *mty = y >> 3;
        *mbase = hd5770_surf_tile(s, x >> 3, y >> 3);
    }
    return *mbase + s->eoff[(y & 7) * 8 + (x & 7)];
}

/* slices of level l: a 3D texture's depth halves with the levels */
static unsigned tex_slices(const Tex *t, unsigned l)
{
    return t->dim == 2 ? MAX(t->slices >> l, 1u) : t->slices;
}

/* BC1..BC5 (FMT 0x31..0x35) */
static unsigned bc_kind(unsigned fmt)
{
    return fmt >= 0x31 && fmt <= 0x35 ? fmt - 0x30 : 0;
}

static void bc_565(uint32_t c, uint32_t rgb[3])
{
    uint32_t r = (c >> 11) & 31, g = (c >> 5) & 63, b = c & 31;

    rgb[0] = (r << 3) | (r >> 2);
    rgb[1] = (g << 2) | (g >> 4);
    rgb[2] = (b << 3) | (b >> 2);
}

/*
 * The 8-entry interpolated channel of a BC3 alpha / BC4 / BC5 block;
 * signed (SNORM BC4/BC5): endpoints and results are int8, the extremes
 * of the 6-value mode -127 and 127.
 */
static void bc_chan(const uint8_t *b, uint32_t out[16][4], unsigned c,
                    bool sgn)
{
    int32_t a0 = sgn ? (int8_t)b[0] : b[0], a1 = sgn ? (int8_t)b[1] : b[1];
    int32_t pal[8];
    uint64_t bits = 0;
    unsigned k;

    for (k = 0; k < 6; k++) {
        bits |= (uint64_t)b[2 + k] << (8 * k);
    }
    pal[0] = a0;
    pal[1] = a1;
    for (k = 2; k < 8; k++) {
        if (a0 > a1) {
            pal[k] = ((8 - k) * a0 + (k - 1) * a1) / 7;
        } else if (k < 6) {
            pal[k] = ((6 - k) * a0 + (k - 1) * a1) / 5;
        } else if (sgn) {
            pal[k] = k == 6 ? -127 : 127;
        } else {
            pal[k] = k == 6 ? 0 : 255;
        }
    }
    for (k = 0; k < 16; k++) {
        out[k][c] = (uint32_t)pal[(bits >> (3 * k)) & 7];
    }
}

/*
 * One 4x4 block, texels in row order, components 0..3 = R, G, B, A
 * (BC4: R; BC5: R, G) as 8-bit values.
 */
static void bc_decode(unsigned bc, const uint8_t *b, uint32_t out[16][4],
                      uint32_t sgn)
{
    unsigned k, t;

    memset(out, 0, 16 * sizeof(out[0]));
    if (bc >= 4) {
        bc_chan(b, out, 0, sgn & 1);
        if (bc == 5) {
            bc_chan(b + 8, out, 1, (sgn >> 1) & 1);
        }
        return;
    }
    {
        const uint8_t *cb = bc == 1 ? b : b + 8;
        uint32_t c0 = lduw_le_p(cb), c1 = lduw_le_p(cb + 2);
        uint32_t idx = ldl_le_p(cb + 4), pal[4][3];
        bool four = bc != 1 || c0 > c1;

        bc_565(c0, pal[0]);
        bc_565(c1, pal[1]);
        for (k = 0; k < 3; k++) {
            if (four) {
                pal[2][k] = (2 * pal[0][k] + pal[1][k]) / 3;
                pal[3][k] = (pal[0][k] + 2 * pal[1][k]) / 3;
            } else {
                pal[2][k] = (pal[0][k] + pal[1][k]) / 2;
                pal[3][k] = 0;
            }
        }
        for (t = 0; t < 16; t++) {
            unsigned i = (idx >> (2 * t)) & 3;

            for (k = 0; k < 3; k++) {
                out[t][k] = pal[i][k];
            }
            out[t][3] = bc == 1 && !four && i == 3 ? 0 : 255;
            if (bc == 2) {
                out[t][3] = ((b[t >> 1] >> ((t & 1) * 4)) & 15) * 17;
            }
        }
        if (bc == 3) {
            bc_chan(b, out, 3, false);
        }
    }
}

/* texel (x, y) of slice z of level lv as raw components, x and y inside it */
static void texel_z(Draw *d, Tex *t, unsigned lv, unsigned z, int x, int y,
                    HD5770Val out[4])
{
    const HD5770Surf *ls = lv ? &t->lv[lv] : &t->s;
    int ex = t->bc ? x >> 2 : x, ey = t->bc ? y >> 2 : y;
    uint64_t off;
    uint8_t *p;
    uint32_t c[4];
    unsigned bits[4], k, n;
    HD5770Val v[6];

    off = lv || z ? hd5770_surf_offset_fast(ls, ex, ey)
                  : surf_off(&t->s, &t->mtx, &t->mty, &t->mbase, ex, ey);
    off += z * t->slice_bytes[lv];
    p = hd5770_span_ptr(&d->g->mem, &d->span_tex, ls->base + off, t->s.bpe,
                        false);
    if (!p) {
        out[0].f = out[1].f = out[2].f = out[3].f = 0.0f;
        return;
    }
    if (t->bc) {
        unsigned i = (y & 3) * 4 + (x & 3);

        if (t->bc_addr != ls->base + off + 1) {
            bc_decode(t->bc, p, t->bc_texel, t->bc >= 4 ? t->sgn : 0);
            t->bc_addr = ls->base + off + 1;
        }
        for (k = 0; k < 4; k++) {
            int32_t e = (int32_t)t->bc_texel[i][k];

            /* SNORM: e / 127, -128 reads as -1 */
            v[k].f = t->bc >= 4 && ((t->sgn >> k) & 1)
                     ? MAX(e / 127.0f, -1.0f) : unorm8[e & 0xff];
        }
        if (t->bc >= 4) {
            v[1].f = t->bc == 5 ? v[1].f : 0.0f;
            v[2].f = 0.0f;
            v[3].f = 1.0f;
        }
    } else if (t->unorm8) {
        uint32_t e = ldl_le_p(p);

        for (k = 0; k < 4; k++) {
            v[k].f = unorm8[(e >> (8 * k)) & 0xff];
        }
    } else {
        hd5770_fmt_unpack(t->fmt, p, c, bits);
        n = t->comps;
        for (k = 0; k < 4; k++) {
            if (k < n) {
                v[k] = comp_value(c[k], bits[k], t->is_float,
                                  (t->sgn >> k) & 1, t->num);
            } else {
                v[k].f = k == 3 ? 1.0f : 0.0f;
            }
        }
    }
    if (t->cmp) {
        /*
         * 1 when the reference passes against the fetched value; the
         * reference is converted to the format first (UNORM: rounded to
         * its steps)
         */
        float f = v[0].f, r = t->ref;
        bool pass;

        if (!t->is_float && t->num == HD5770_NUM_NORM && !(t->sgn & 1) &&
            !t->bc && t->rbits && t->rbits < 32) {
            uint32_t mx = (1u << t->rbits) - 1;
            float q = fminf(fmaxf(r, 0.0f), 1.0f) * mx;

            q += 0.5f;
            r = (uint32_t)q / (float)mx;
        }

        switch (t->cmp - 1) {
        case 1: pass = r < f; break;
        case 2: pass = r == f; break;
        case 3: pass = r <= f; break;
        case 4: pass = r > f; break;
        case 5: pass = r != f; break;
        case 6: pass = r >= f; break;
        case 7: pass = true; break;
        default: pass = false; break;
        }
        v[0].f = v[1].f = v[2].f = v[3].f = pass ? 1.0f : 0.0f;
    }
    v[4].f = 0.0f;
    v[5].f = 1.0f;
    for (k = 0; k < 4; k++) {
        unsigned s = t->dst_sel[k];

        out[k] = s < 6 ? v[s] : v[4];
    }
}

/*
 * Texel (x, y) of level lv: of slice t->z, or of a 3D texture at depth
 * t->zn of the level's slices, nearest, or (Z_FILTER linear) between
 * the two nearest slice centres, the slice index wrapping or clamped as
 * CLAMP_Z wraps or clamps the coordinate.
 */
static void texel(Draw *d, Tex *t, unsigned lv, int x, int y,
                  HD5770Val out[4])
{
    unsigned ns, k;

    if (t->dim != 2) {
        texel_z(d, t, lv, t->z, x, y, out);
        return;
    }
    ns = tex_slices(t, lv);
    if (((t->samp[0] >> 13) & 3) == 2 && ns > 1 && !t->ld) {
        float zf = t->zn * ns, a;
        int z0, z1;
        HD5770Val v1[4];

        zf -= 0.5f;
        z0 = (int)floorf(zf);
        z1 = z0 + 1;
        a = zf - z0;
        if (((t->samp[0] >> 6) & 7) == 0) {
            z0 = (z0 + (int)ns) % (int)ns;
            z1 = z1 % (int)ns;
        } else {
            z0 = MAX(z0, 0);
            z1 = MIN(z1, (int)ns - 1);
        }
        texel_z(d, t, lv, z0, x, y, out);
        texel_z(d, t, lv, z1, x, y, v1);
        for (k = 0; k < 4; k++) {
            out[k].f = out[k].f + (v1[k].f - out[k].f) * a;
        }
        return;
    }
    /* the nearest slice of the level */
    texel_z(d, t, lv, MIN((unsigned)(t->zn * ns), ns - 1), x, y, out);
}

/* apply the address mode; false if the coordinate reads the border */
static bool wrap(int *c, int size, unsigned mode)
{
    int v = *c;

    switch (mode) {
    case 0:                     /* wrap */
        v %= size;
        if (v < 0) {
            v += size;
        }
        break;
    case 1: {                   /* mirror */
        int p = v % (2 * size);

        if (p < 0) {
            p += 2 * size;
        }
        v = p < size ? p : 2 * size - 1 - p;
        break;
    }
    case 2:                     /* clamp to the last texel */
    case 3:
    case 5:
    case 7:
        if (mode != 2) {
            v = v < 0 ? -v - 1 : v;
        }
        v = MAX(0, MIN(v, size - 1));
        break;
    default:                    /* border */
        if (v < 0 || v >= size) {
            return false;
        }
        break;
    }
    *c = v;
    return true;
}

static void sample_point(Draw *d, Tex *t, unsigned lv, int x, int y,
                         HD5770Val out[4])
{
    if (!wrap(&x, lv ? t->lw[lv] : t->w, t->samp[0] & 7) ||
        !wrap(&y, lv ? t->lh[lv] : t->h, (t->samp[0] >> 3) & 7)) {
        unsigned k;

        for (k = 0; k < 4; k++) {
            out[k].f = t->border[k];
        }
        return;
    }
    texel(d, t, lv, x, y, out);
}

/*
 * The mip chain of a 2D texture: level l (1..last_lv) follows level
 * l - 1 from MIP_ADDRESS on, each as large as its own size padded to its
 * tiling; a 2D-tiled level smaller than a macro tile and all after it
 * are 1D tiled. The six faces of a cube map's level are followed by the
 * room of two more.
 */
static void tex_levels(Tex *t, uint64_t mip_addr)
{
    const HD5770Surf *s0 = &t->s;
    uint32_t mtw = 8 * s0->bankw * s0->pipes * s0->mtilea;
    uint32_t mth = 8 * s0->bankh * s0->banks / s0->mtilea;
    uint32_t bpe = s0->bpe, grp = s0->pipe_interleave;
    uint64_t off = mip_addr;
    unsigned mode = s0->mode, l;

    for (l = 1; l <= t->last_lv; l++) {
        HD5770Surf *ls = &t->lv[l];
        uint32_t w = MAX(t->w >> l, 1u), h = MAX(t->h >> l, 1u), pw, ph;

        t->lw[l] = w;
        t->lh[l] = h;
        if (t->bc) {
            /* in blocks */
            w = DIV_ROUND_UP(w, 4);
            h = DIV_ROUND_UP(h, 4);
        }

        if (mode == HD5770_ARRAY_2D_TILED_THIN1 && (w < mtw || h < mth)) {
            mode = HD5770_ARRAY_1D_TILED_THIN1;
        }
        switch (mode) {
        case HD5770_ARRAY_2D_TILED_THIN1:
            pw = ROUND_UP(w, mtw);
            ph = ROUND_UP(h, mth);
            break;
        case HD5770_ARRAY_1D_TILED_THIN1:
            pw = ROUND_UP(w, MAX(8u, grp / (8 * bpe)));
            ph = ROUND_UP(h, 8);
            break;
        case HD5770_ARRAY_LINEAR_ALIGNED:
            pw = ROUND_UP(w, MAX(64u, grp / bpe));
            ph = h;
            break;
        default:
            pw = w;
            ph = h;
            break;
        }
        *ls = *s0;
        ls->base = off;
        ls->mode = mode;
        ls->pitch = pw;
        ls->height = ph;
        hd5770_surf_prep(ls);
        t->slice_bytes[l] = (uint64_t)pw * ph * bpe;
        /* a cube map's level takes the room of 8 faces */
        off += t->slice_bytes[l] * (t->dim == 3 ? 8 : tex_slices(t, l));
    }
}

/* log2 by the exponent and a linear mantissa: the same on every host */
static float lod_log2(float x)
{
    HD5770Val v;

    if (!(x > 0.0f)) {
        return -128.0f;
    }
    v.f = x;
    return (float)((int)((v.u >> 23) & 0xff) - 127) +
           (float)(v.u & 0x7fffff) * (1.0f / 8388608.0f);
}

/* the texture of resource id and sampler sid, set up once per draw */
static Tex *tex_get(Draw *d, unsigned id, unsigned sid)
{
    uint32_t key = id | sid << 16;
    unsigned i;

    for (i = 0; i < d->n_tex; i++) {
        if (d->tex_key[i] == key) {
            return &d->tex[i];
        }
    }
    i = d->n_tex < TEX_CACHE ? d->n_tex : TEX_CACHE - 1;
    if (!tex_setup(d, id, sid, &d->tex[i])) {
        return NULL;
    }
    pend_check(d->g, d->tex[i].s.base,
               d->tex[i].slice_bytes[0] * d->tex[i].slices);
    if (d->tex[i].mip) {
        unsigned l;

        for (l = 1; l <= d->tex[i].last_lv; l++) {
            pend_check(d->g, d->tex[i].lv[l].base,
                       d->tex[i].slice_bytes[l] * tex_slices(&d->tex[i], l));
        }
    }
    d->tex_key[i] = key;
    d->n_tex = MAX(d->n_tex, i + 1);
    return &d->tex[i];
}

/*
 * The slice a fetch reads: the cube face (z; s and t come in [1, 2] as
 * the CUBE instruction's tc / |ma| + 1.5 and are made [0, 1]), the array
 * layer (y of a 1D array, z of a 2D array), the depth of a 3D texture.
 */
static void tex_slice(Tex *t, const uint32_t *w, HD5770Val c[4], bool ld)
{
    int layer;
    float zn;

    switch (t->dim) {
    case 3:
        layer = c[2].f > 0.0f ? (int)c[2].f : 0;
        t->z = MIN(layer, 5);
        c[0].f -= 1.0f;
        c[1].f -= 1.0f;
        break;
    case 4:
    case 5:
        layer = ld ? c[t->dim - 3].i : (int)rintf(c[t->dim - 3].f);
        t->z = MAX(0, MIN(layer, (int)t->slices - 1));
        break;
    case 2:
        if (ld) {
            zn = (c[2].i + 0.5f) / t->slices;
        } else {
            zn = (w[1] >> 30) & 1 ? c[2].f : c[2].f / t->slices;
        }
        if (((t->samp[0] >> 6) & 7) == 0) {
            zn -= floorf(zn);
        }
        t->zn = zn > 0.0f ? MIN(zn, 1.0f) : 0.0f;
        break;
    }
}

/* a normalised coordinate in texels of a level, with the half-texel offset */
static float lv_coord(float c, uint32_t size, int o)
{
    float r = c * size;

    r += o * 0.5f;
    return r;
}

/* point or bilinear sample of level lv at (u, vv), in its texels */
static void sample_lv(Draw *d, Tex *t, unsigned lv, float u, float vv,
                      bool lin, HD5770Val v[4])
{
    unsigned k;

    if (!lin) {
        sample_point(d, t, lv, (int)floorf(u), (int)floorf(vv), v);
        return;
    }
    {
        /* bilinear */
        float fx = u - 0.5f, fy = vv - 0.5f;
        int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
        float ax = fx - x0, ay = fy - y0;
        HD5770Val t00[4], t10[4], t01[4], t11[4];

        sample_point(d, t, lv, x0, y0, t00);
        sample_point(d, t, lv, x0 + 1, y0, t10);
        sample_point(d, t, lv, x0, y0 + 1, t01);
        sample_point(d, t, lv, x0 + 1, y0 + 1, t11);
        for (k = 0; k < 4; k++) {
            float top = t00[k].f + (t10[k].f - t00[k].f) * ax;
            float bot = t01[k].f + (t11[k].f - t01[k].f) * ax;

            v[k].f = top + (bot - top) * ay;
        }
    }
}

/*
 * TEX. dv: the derivatives of the coordinates across x (0..3) and y
 * (4..7), NULL if not known; they choose the level of a mipmapped
 * texture.
 */
static void tfetch_stage(Draw *d, const uint32_t *w, const HD5770Val c0[4],
                         HD5770Val v[4], unsigned base_r, unsigned base_s,
                         const float *dv)
{
    unsigned op = w[0] & 0x1f, id, sid, k;
    float u, vv;
    Tex *t;
    HD5770Val c[4];
    int ox = ((int32_t)(w[2] << 27)) >> 27, oy = ((int32_t)(w[2] << 22)) >> 27;
    bool nx = (w[1] >> 28) & 1, ny = (w[1] >> 29) & 1;

    id = ((w[0] >> 8) & 0xff) + base_r;
    sid = ((w[2] >> 15) & 0x1f) + base_s;
    v[0].f = v[1].f = v[2].f = 0.0f;
    v[3].f = 0.0f;
    t = tex_get(d, id, sid);
    if (!t) {
        d->st->gap_misc[HD5770_GAP_TEX_FETCH]++;
        d->ok = false;
        return;
    }
    d->st->tex_fetch++;
    memcpy(c, c0, sizeof(c));
    t->z = 0;
    t->cmp = 0;
    if (op >= 0x18 && op <= 0x1b) {
        /*
         * SAMPLE_C*: as SAMPLE*, each texel compared with the reference:
         * w, or z when w holds the LOD or its bias (_L, _LB)
         */
        t->cmp = ((t->samp[0] >> 22) & 7) + 1;
        t->ref = op == 0x19 || op == 0x1a ? c[2].f : c[3].f;
        op -= 8;
    }
    t->ld = op == 0x03;
    if (op != 0x04) {
        tex_slice(t, w, c, op == 0x03);
    }
    if (op == 0x04) {
        /* GET_TEXTURE_RESINFO: the size of level x, the number of levels */
        unsigned lv = MIN(c[0].u + t->base_lv, 31u);

        v[0].u = MAX(t->w >> lv, 1u);
        v[1].u = t->dim == 0 ? 1 : MAX(t->h >> lv, 1u);
        v[2].u = t->depth;
        v[3].u = t->levels;
        return;
    }
    if (op == 0x03) {           /* LD: integer texel coordinates */
        int x = c[0].i + ox / 2, y = c[1].i + oy / 2;

        if (x >= 0 && y >= 0 && x < (int)t->w && y < (int)t->h) {
            texel(d, t, 0, x, y, v);
        }
        return;
    }
    if (t->mip && nx && ny) {
        /* the level of detail, relative to level 0 */
        float lod = 0.0f, a, b, l0, f;
        unsigned lv;
        bool lin;
        HD5770Val v1[4];

        if (op == 0x11) {               /* SAMPLE_L */
            lod = c[3].f;
        } else if (op != 0x13) {        /* not SAMPLE_LZ */
            float ax = dv ? dv[0] * t->w : 0.0f, ay = dv ? dv[1] * t->h : 0.0f;
            float bx = dv ? dv[4] * t->w : 0.0f, by = dv ? dv[5] * t->h : 0.0f;

            a = ax * ax + ay * ay;
            b = bx * bx + by * by;
            lod = 0.5f * lod_log2(fmaxf(a, b));
            if (op == 0x12) {           /* SAMPLE_LB */
                lod += c[3].f;
            }
            if ((w[1] >> 21) & 0x7f) {
                /* the instruction's LOD_BIAS, S3.4 */
                lod += ((int32_t)(w[1] << 4) >> 25) / 16.0f;
            }
        }
        lod = fminf(fmaxf(lod + t->lod_bias, t->min_lod), t->max_lod);
        d->st->lod_fetches++;
        if (!(lod > 0.0f)) {
            /* magnification */
            sample_lv(d, t, 0, lv_coord(c[0].f, t->w, ox),
                      lv_coord(c[1].f, t->h, oy), (t->samp[0] >> 9) & 1, v);
            return;
        }
        lin = (t->samp[0] >> 11) & 1;
        if (t->mip_filter == 1) {
            lv = MIN((unsigned)floorf(lod + 0.5f), t->last_lv);
            sample_lv(d, t, lv, lv_coord(c[0].f, lv ? t->lw[lv] : t->w, ox),
                      lv_coord(c[1].f, lv ? t->lh[lv] : t->h, oy), lin, v);
            return;
        }
        l0 = floorf(lod);
        f = lod - l0;
        lv = MIN((unsigned)l0, t->last_lv);
        sample_lv(d, t, lv, lv_coord(c[0].f, lv ? t->lw[lv] : t->w, ox),
                  lv_coord(c[1].f, lv ? t->lh[lv] : t->h, oy), lin, v);
        if (lv < t->last_lv) {
            lv++;
            sample_lv(d, t, lv, lv_coord(c[0].f, t->lw[lv], ox),
                      lv_coord(c[1].f, t->lh[lv], oy), lin, v1);
            for (k = 0; k < 4; k++) {
                v[k].f = v[k].f + (v1[k].f - v[k].f) * f;
            }
        }
        return;
    }
    u = c[0].f;
    vv = c[1].f;
    if (nx) {
        u *= t->w;
    }
    if (ny) {
        vv *= t->h;
    }
    u += ox * 0.5f;
    vv += oy * 0.5f;
    if (t->dim == 0) {
        vv = 0.5f;
    }
    sample_lv(d, t, 0, u, vv, ((t->samp[0] >> 9) & 1) || ((t->samp[0] >> 11) & 1),
              v);
}

static void tfetch_ps(void *opaque, const uint32_t *w, const HD5770Val c[4],
                      HD5770Thread *th, HD5770Val v[4])
{
    tfetch_stage(opaque, w, c, v, 0, 0, NULL);
}

static void tfetch_ps_d(void *opaque, const uint32_t *w, const HD5770Val c[4],
                        const float dv[8], HD5770Val v[4])
{
    tfetch_stage(opaque, w, c, v, 0, 0, dv);
}

static void tfetch_vs(void *opaque, const uint32_t *w, const HD5770Val c[4],
                      HD5770Thread *th, HD5770Val v[4])
{
    tfetch_stage(opaque, w, c, v, RES_VS, SAMPLER_VS, NULL);
}

/* ---------------------------------------------------------------- */
/* Constants and interpolation                                      */

static void kcache_ps(void *opaque, unsigned buf, unsigned idx,
                      HD5770Val v[4])
{
    Draw *d = opaque;
    uint64_t base;

    if (!buf) {
        memcpy(v, d->g->alu + (idx & 255) * 4, 16);
        return;
    }
    base = (uint64_t)CTX(d->g, SQ_ALU_CONST_CACHE_PS_0 + 4 * buf)
                    << 8;

    pend_check(d->g, base + idx * 16ull, 16);
    hd5770_mem_read(&d->g->mem, base + idx * 16ull, v, 16);
}

static void kcache_vs(void *opaque, unsigned buf, unsigned idx,
                      HD5770Val v[4])
{
    Draw *d = opaque;
    uint64_t base;

    if (!buf) {
        memcpy(v, d->g->alu + 1024 + (idx & 255) * 4, 16);
        return;
    }
    if (buf < 16 && idx < KC_MAX && d->kc_ok[buf] &&
        (d->kc_ok[buf][idx / 64] >> (idx % 64) & 1)) {
        memcpy(v, d->kc[buf][idx], 16);
        return;
    }
    base = (uint64_t)CTX(d->g, SQ_ALU_CONST_CACHE_VS_0 + 4 * buf)
                    << 8;

    pend_check(d->g, base + idx * 16ull, 16);
    hd5770_mem_read(&d->g->mem, base + idx * 16ull, v, 16);
    if (buf < 16 && idx < KC_MAX) {
        if (!d->kc_ok[buf]) {
            d->kc_ok[buf] = g_new0(uint64_t, KC_MAX / 64);
            d->kc[buf] = g_malloc(KC_MAX * sizeof(*d->kc[buf]));
        }
        memcpy(d->kc[buf][idx], v, 16);
        d->kc_ok[buf][idx / 64] |= 1ull << (idx % 64);
    }
}

/* constant buffers may change between instances (the draw writes memory) */
static void kcache_drop(Draw *d)
{
    unsigned k;

    for (k = 0; k < 16; k++) {
        g_free(d->kc_ok[k]);
        g_free(d->kc[k]);
        d->kc_ok[k] = NULL;
        d->kc[k] = NULL;
    }
}

static float interp(void *opaque, unsigned p, unsigned chan, bool flat)
{
    Draw *d = opaque;
    int src;
    float a, b, c;

    if (p >= HD5770_PARAMS) {
        return 0.0f;
    }
    src = d->ps_src[p];
    if (src < 0) {
        return d->ps_default[p][chan];
    }
    if (flat || d->ps_flat[p]) {
        return d->tvp->param[src][chan].f;
    }
    a = d->tv[0]->param[src][chan].f;
    b = d->tv[1]->param[src][chan].f;
    c = d->tv[2]->param[src][chan].f;
    if (d->persp) {
        return (a * d->tv[0]->w * d->b0 + b * d->tv[1]->w * d->b1 +
                c * d->tv[2]->w * d->b2) * d->pw;
    }
    return a * d->b0 + b * d->b1 + c * d->b2;
}

/* interp() for the pixels of the lanes */
static void interp_n(void *opaque, const HD5770Lanes *l, unsigned p,
                     unsigned chan, bool flat, HD5770Val *out)
{
    Draw *d = opaque;
    unsigned x, n = l->n;
    int src;
    float a, b, c;

    if (p >= HD5770_PARAMS) {
        for (x = 0; x < n; x++) {
            out[x].f = 0.0f;
        }
        return;
    }
    src = d->ps_src[p];
    if (src < 0) {
        for (x = 0; x < n; x++) {
            out[x].f = d->ps_default[p][chan];
        }
        return;
    }
    if (flat || d->ps_flat[p]) {
        a = d->tvp->param[src][chan].f;
        for (x = 0; x < n; x++) {
            out[x].f = a;
        }
        return;
    }
    a = d->tv[0]->param[src][chan].f;
    b = d->tv[1]->param[src][chan].f;
    c = d->tv[2]->param[src][chan].f;
    if (d->persp) {
        for (x = 0; x < n; x++) {
            out[x].f = (a * d->tv[0]->w * l->b0[x] +
                        b * d->tv[1]->w * l->b1[x] +
                        c * d->tv[2]->w * l->b2[x]) * l->pw[x];
        }
        return;
    }
    for (x = 0; x < n; x++) {
        out[x].f = a * l->b0[x] + b * l->b1[x] + c * l->b2[x];
    }
}

/* ---------------------------------------------------------------- */
/* Render target and blending                                       */

/* which shader channel feeds memory component i */
static unsigned swap_src(unsigned swap, unsigned n, unsigned i)
{
    static const uint8_t m[4][4][4] = {
        { { 0 }, { 1 }, { 2 }, { 3 } },
        { { 0, 1 }, { 0, 3 }, { 1, 0 }, { 3, 0 } },
        { { 0, 1, 2 }, { 0, 1, 3 }, { 2, 1, 0 }, { 3, 1, 0 } },
        { { 0, 1, 2, 3 }, { 2, 1, 0, 3 }, { 3, 2, 1, 0 }, { 3, 0, 1, 2 } },
    };

    return m[n - 1][swap][i];
}

/* colour buffer cb as the draw's render target */
static bool rt_setup(Draw *d, unsigned cb)
{
    HD5770Gfx *g = d->g;
    uint32_t base = CB_COLOR0_BASE + cb * CB_COLOR_STRIDE;
    uint32_t info = CTX(g, base + 0x10), attrib = CTX(g, base + 0x14);
    uint32_t pitch = CTX(g, base + 0x4), slice = CTX(g, base + 0x8);
    unsigned i;

    d->rt_fmt = (info >> 2) & 0x3f;
    if (!d->rt_fmt) {
        return false;
    }
    d->rt_num = (info >> 12) & 7;
    d->rt_swap = (info >> 15) & 3;
    d->rt_bpe = hd5770_fmt_bytes(d->rt_fmt);
    d->rt_comps = hd5770_fmt_comps(d->rt_fmt);
    d->rt_float = d->rt_num == 7;
    d->rt_blend = !((info >> 20) & 1);
    d->st->rt_fmt[d->rt_fmt]++;
    if (!d->rt_bpe || (d->rt_bpe > 8 && d->rt_num != 4) ||
        !(d->rt_num == 0 || d->rt_num == 4 || d->rt_num == 7) ||
        (d->rt_float && !hd5770_fmt_is_float(d->rt_fmt) &&
         d->rt_fmt != HD5770_FMT_32_FLOAT)) {
        d->st->gap_fmt[d->rt_fmt]++;
        return false;
    }
    d->rt.base = (uint64_t)CTX(g, base) << 8;
    if (((info >> 8) & 0xf) == HD5770_ARRAY_LINEAR_GENERAL) {
        /* CB_COLOR0_VIEW.SLICE_START[7:0]: the address's low byte */
        d->rt.base |= CTX(g, base + 0xc) & 0xff;
    }
    d->rt.pitch = ((pitch & 0x7ff) + 1) * 8;
    d->rt.height = ((slice & 0x3fffff) + 1) * 64 / d->rt.pitch;
    d->rt.bpe = d->rt_bpe;
    d->rt.mode = (info >> 8) & 0xf;
    d->rt.nondisp = (attrib >> 4) & 1;
    d->rt.tile_split = 64u << ((attrib >> 5) & 0xf);
    d->rt.banks = 2u << ((attrib >> 10) & 3);
    d->rt.bankw = 1u << ((attrib >> 13) & 3);
    d->rt.bankh = 1u << ((attrib >> 16) & 3);
    d->rt.mtilea = 1u << ((attrib >> 19) & 3);
    hd5770_surf_set_addr_config(&d->rt, addr_config(g));
    hd5770_surf_prep(&d->rt);
    {
        uint8_t zero[16] = { 0 };
        uint32_t c[4];

        hd5770_fmt_unpack(d->rt_fmt, zero, c, d->rt_bits);
        for (i = 0; i < d->rt_comps; i++) {
            d->rt_src[i] = swap_src(d->rt_swap, d->rt_comps, i);
        }
    }
    d->rt_unorm8 = d->rt_fmt == HD5770_FMT_8_8_8_8 && d->rt_num == 0;
    d->rt_mtx = d->rt_mty = UINT32_MAX;
    if (!hd5770_surf_mode_ok(d->rt.mode)) {
        d->st->gap_mode[d->rt.mode]++;
        return false;
    }
    if (((info >> 17) & 1) ||
        (((info >> 18) & 1) && (CTX(g, PA_SC_AA_CONFIG) & 7))) {
        /* fast clear metadata, or compressed multisampled data */
        d->st->gap_misc[HD5770_GAP_ROP]++;
    }
    d->x1 = MIN(d->x1, (int)d->rt.pitch);
    d->y1 = MIN(d->y1, (int)d->rt.height);
    return true;
}

/*
 * Colour buffers 1..7 with a format and a write mask: the draw writes
 * every such target, export n to colour buffer n, each with its own
 * write mask and CB_BLENDn_CONTROL.
 */
static void mrt_setup(Draw *d)
{
    HD5770Gfx *g = d->g;
    uint32_t mask = CTX(g, CB_TARGET_MASK) & CTX(g, CB_SHADER_MASK);
    uint8_t *cur = (uint8_t *)d + MRT_FIRST;
    unsigned i, n = 0;

    for (i = 1; i < 8; i++) {
        if ((CTX(g, CB_COLOR0_BASE + i * CB_COLOR_STRIDE + 0x10) >> 2) & 0x3f &&
            (mask >> (4 * i)) & 0xf) {
            break;
        }
    }
    if (i == 8) {
        return;
    }
    d->mrt = g_malloc(8 * MRT_BYTES);
    if (d->color) {
        memcpy(d->mrt, cur, MRT_BYTES);
        d->mrt_cb[n++] = 0;
    }
    for (i = 1; i < 8; i++) {
        uint32_t m = (mask >> (4 * i)) & 0xf;

        if (!m || !((CTX(g, CB_COLOR0_BASE + i * CB_COLOR_STRIDE + 0x10) >> 2)
                    & 0x3f) || !rt_setup(d, i)) {
            continue;
        }
        d->wmask = m;
        d->blend = CTX(g, CB_BLEND0_CONTROL + 4 * i);
        memcpy(d->mrt + n * MRT_BYTES, cur, MRT_BYTES);
        d->mrt_cb[n++] = i;
    }
    d->nmrt = n;
    if (n) {
        memcpy(cur, d->mrt, MRT_BYTES);
        d->color = true;
    }
}

/* render target k of an MRT draw */
static HD5770Surf *mrt_surf(Draw *d, unsigned k)
{
    return (HD5770Surf *)(d->mrt + k * MRT_BYTES + offsetof(Draw, rt) -
                          MRT_FIRST);
}

static float blend_factor(unsigned f, const float *src, const float *dst,
                          const float *cst, unsigned ch, bool *ok)
{
    switch (f) {
    case 0:
        return 0.0f;
    case 1:
        return 1.0f;
    case 2:
        return src[ch];
    case 3:
        return 1.0f - src[ch];
    case 4:
        return src[3];
    case 5:
        return 1.0f - src[3];
    case 6:
        return dst[3];
    case 7:
        return 1.0f - dst[3];
    case 8:
        return dst[ch];
    case 9:
        return 1.0f - dst[ch];
    case 10:
        return ch == 3 ? 1.0f : MIN(src[3], 1.0f - dst[3]);
    case 13:
        return cst[ch];
    case 14:
        return 1.0f - cst[ch];
    case 19:
        return cst[3];
    case 20:
        return 1.0f - cst[3];
    default:
        *ok = false;
        return 0.0f;
    }
}

static float combine(unsigned fn, float s, float sf, float dv, float df,
                     bool *ok)
{
    switch (fn) {
    case 0:
        return s * sf + dv * df;
    case 1:
        return s * sf - dv * df;
    case 2:
        return MIN(s, dv);
    case 3:
        return MAX(s, dv);
    case 4:
        return dv * df - s * sf;
    default:
        *ok = false;
        return s;
    }
}

static void put_pixel(Draw *d, int x, int y, const HD5770Val *col)
{
    uint64_t off = surf_off(&d->rt, &d->rt_mtx, &d->rt_mty, &d->rt_mbase,
                            x, y);
    uint8_t *p = hd5770_span_ptr(&d->g->mem, &d->span_rt, d->rt.base + off,
                                 d->rt_bpe, true);
    uint32_t c[4] = { 0 }, old[4];
    const unsigned *bits = d->rt_bits;
    unsigned i, n = d->rt_comps;
    float src[4], dst[4];
    bool ok = true;

    if (!p) {
        return;
    }
    if (d->rt_unorm8) {
        uint32_t e = ldl_le_p(p);

        for (i = 0; i < 4; i++) {
            old[i] = (e >> (8 * i)) & 0xff;
        }
    } else {
        unsigned b[4];

        hd5770_fmt_unpack(d->rt_fmt, p, old, b);
    }
    if (d->rt_num == 4) {
        /* integer targets: no blending */
        for (i = 0; i < n; i++) {
            unsigned s = d->rt_src[i];

            c[i] = (d->wmask >> s) & 1 ? col[s].u : old[i];
        }
        hd5770_fmt_pack(d->rt_fmt, p, c);
        return;
    }
    if (d->rt_float) {
        /* float targets: blending without clamps */
        HD5770Val out[4];

        for (i = 0; i < 4; i++) {
            out[i] = col[i];
            src[i] = col[i].f;
            dst[i] = i == 3 ? 1.0f : 0.0f;
        }
        for (i = 0; i < n; i++) {
            HD5770Val o;

            o.u = old[i];
            dst[d->rt_src[i]] = bits[i] == 16 ? half_to_float(old[i]) : o.f;
        }
        if (d->blend & (1u << 30) && d->rt_blend) {
            unsigned csrc = d->blend & 0x1f, cfn = (d->blend >> 5) & 7;
            unsigned cdst = (d->blend >> 8) & 0x1f;
            unsigned asrc = csrc, afn = cfn, adst = cdst;

            if ((d->blend >> 29) & 1) {
                asrc = (d->blend >> 16) & 0x1f;
                afn = (d->blend >> 21) & 7;
                adst = (d->blend >> 24) & 0x1f;
            }
            for (i = 0; i < 4; i++) {
                unsigned fs = i == 3 ? asrc : csrc, fd = i == 3 ? adst : cdst;
                unsigned fn = i == 3 ? afn : cfn;

                out[i].f = combine(fn, src[i],
                                   blend_factor(fs, src, dst, d->blend_const,
                                                i, &ok),
                                   dst[i],
                                   blend_factor(fd, src, dst, d->blend_const,
                                                i, &ok), &ok);
            }
            d->st->blend_on++;
            if (!ok) {
                d->st->gap_misc[HD5770_GAP_BLEND_FUNC]++;
            }
        }
        for (i = 0; i < n; i++) {
            unsigned s = d->rt_src[i];

            c[i] = !((d->wmask >> s) & 1) ? old[i] :
                   bits[i] == 16 ? float_to_half(out[s].f) : out[s].u;
        }
        hd5770_fmt_pack(d->rt_fmt, p, c);
        return;
    }
    for (i = 0; i < 4; i++) {
        src[i] = col[i].f;
        if (src[i] != src[i]) {
            src[i] = 0.0f;
        }
        dst[i] = i == 3 ? 1.0f : 0.0f;
    }
    for (i = 0; i < n; i++) {
        unsigned s = d->rt_src[i];

        dst[s] = d->rt_unorm8 ? unorm8[old[i]]
                              : old[i] / (float)((1u << bits[i]) - 1);
    }
    if (d->blend & (1u << 30) && d->rt_blend) {
        unsigned csrc = d->blend & 0x1f, cfn = (d->blend >> 5) & 7;
        unsigned cdst = (d->blend >> 8) & 0x1f;
        unsigned asrc = csrc, afn = cfn, adst = cdst;
        float out[4];

        if ((d->blend >> 29) & 1) {
            asrc = (d->blend >> 16) & 0x1f;
            afn = (d->blend >> 21) & 7;
            adst = (d->blend >> 24) & 0x1f;
        }
        for (i = 0; i < 4; i++) {
            float s = fminf(fmaxf(src[i], 0.0f), 1.0f);
            float sc[4] = {
                fminf(fmaxf(src[0], 0.0f), 1.0f),
                fminf(fmaxf(src[1], 0.0f), 1.0f),
                fminf(fmaxf(src[2], 0.0f), 1.0f),
                fminf(fmaxf(src[3], 0.0f), 1.0f),
            };
            unsigned fs = i == 3 ? asrc : csrc, fd = i == 3 ? adst : cdst;
            unsigned fn = i == 3 ? afn : cfn;

            out[i] = combine(fn, s,
                             blend_factor(fs, sc, dst, d->blend_const, i, &ok),
                             dst[i],
                             blend_factor(fd, sc, dst, d->blend_const, i, &ok),
                             &ok);
        }
        memcpy(src, out, sizeof(out));
        d->st->blend_on++;
        if (!ok) {
            d->st->gap_misc[HD5770_GAP_BLEND_FUNC]++;
        }
    }
    for (i = 0; i < n; i++) {
        unsigned s = d->rt_src[i];
        uint32_t max = (1u << bits[i]) - 1;
        float v = fminf(fmaxf(src[s], 0.0f), 1.0f);

        c[i] = (d->wmask >> s) & 1 ? (uint32_t)(v * max + 0.5f) : old[i];
    }
    if (d->rt_unorm8) {
        stl_le_p(p, (c[0] & 0xff) | (c[1] & 0xff) << 8 | (c[2] & 0xff) << 16 |
                    (c[3] & 0xff) << 24);
    } else {
        hd5770_fmt_pack(d->rt_fmt, p, c);
    }
}

/* ---------------------------------------------------------------- */
/* Depth and stencil                                                */

static bool db_cmp(unsigned fn, uint32_t a, uint32_t b)
{
    switch (fn) {
    case 0:
        return false;
    case 1:
        return a < b;
    case 2:
        return a == b;
    case 3:
        return a <= b;
    case 4:
        return a > b;
    case 5:
        return a != b;
    case 6:
        return a >= b;
    default:
        return true;
    }
}

static bool db_cmpf(unsigned fn, float a, float b)
{
    switch (fn) {
    case 0:
        return false;
    case 1:
        return a < b;
    case 2:
        return a == b;
    case 3:
        return a <= b;
    case 4:
        return a > b;
    case 5:
        return a != b;
    case 6:
        return a >= b;
    default:
        return true;
    }
}

static uint32_t stencil_op(unsigned op, uint32_t v, uint32_t ref)
{
    switch (op) {
    case 0:
        return v;
    case 1:
        return 0;
    case 2:
        return ref;
    case 3:
        return v < 255 ? v + 1 : 255;
    case 4:
        return v ? v - 1 : 0;
    case 5:
        return ~v & 0xff;
    case 6:
        return (v + 1) & 0xff;
    default:
        return (v - 1) & 0xff;
    }
}

/* the depth of a fragment as the buffer stores it */
static uint32_t db_zbits(unsigned zfmt, float z)
{
    HD5770Val v;

    switch (zfmt) {
    case 1:
        return (uint32_t)rintf(z * 65535.0f);
    case 2:
        return (uint32_t)rintf(z * 16777215.0f);
    default:
        v.f = z;
        return v.u;
    }
}

/*
 * The stencil and depth tests of the fragment at (x, y) with depth z;
 * the buffers are updated as the tests say. sref is the stencil
 * reference the shader exported, or -1. True: the colour is written.
 */
static bool db_test(Draw *d, int x, int y, float z, int sref)
{
    uint32_t dctl = d->dctl;
    bool front = d->front || !(dctl & DC_BACKFACE);
    uint32_t srm = d->srm[front ? 0 : 1];
    uint8_t *zp = NULL, *sp = NULL;
    uint32_t znew = 0, zold = 0;
    bool zpass = true, spass = true;

    d->st->db_tests++;
    if (d->zfmt) {
        uint64_t off = surf_off(&d->zs, &d->z_mtx, &d->z_mty, &d->z_mbase,
                                x, y);

        zp = hd5770_span_ptr(&d->g->mem, &d->span_z, d->zs.base + off,
                             d->zs.bpe, true);
    }
    if (zp) {
        float zf = d->rctl & RC_DEPTH_CLEAR ? d->zclear : z;
        HD5770Val o;

        zf = fminf(fmaxf(zf, d->zmin), d->zmax);
        znew = db_zbits(d->zfmt, zf);
        if (d->zfmt == 1) {
            zold = lduw_le_p(zp);
        } else {
            zold = ldl_le_p(zp);
        }
        if (d->zfmt == 2) {
            zold &= 0xffffff;
        }
        o.u = zold;
        if (dctl & DC_Z) {
            zpass = d->zfmt == 3 ? db_cmpf((dctl >> 4) & 7, zf, o.f)
                                 : db_cmp((dctl >> 4) & 7, znew, zold);
        }
    }
    if (d->has_s) {
        uint64_t off = surf_off(&d->ss, &d->s_mtx, &d->s_mty, &d->s_mbase,
                                x, y);

        sp = hd5770_span_ptr(&d->g->mem, &d->span_s, d->ss.base + off, 1,
                             true);
    }
    if (sp) {
        uint32_t sold = *sp, snew, ref, wm = (srm >> 16) & 0xff;

        ref = d->rctl & RC_STENCIL_CLEAR ? d->sclear
              : sref >= 0 ? (uint32_t)sref & 0xff : srm & 0xff;
        if (dctl & DC_STENCIL) {
            uint32_t m = (srm >> 8) & 0xff;
            unsigned op;

            spass = db_cmp(front ? (dctl >> 8) & 7 : (dctl >> 20) & 7,
                           ref & m, sold & m);
            op = !spass ? (front ? dctl >> 11 : dctl >> 23) :
                 !zpass ? (front ? dctl >> 17 : dctl >> 29) :
                          (front ? dctl >> 14 : dctl >> 26);
            snew = stencil_op(op & 7, sold, ref);
        } else {
            snew = ref;
        }
        *sp = (sold & ~wm) | (snew & wm);
    }
    /* Z_WRITE_ENABLE writes only with Z_ENABLE */
    if (zp && zpass && spass && (dctl & DC_Z) && (dctl & DC_Z_WRITE)) {
        if (d->zfmt == 1) {
            stw_le_p(zp, znew);
        } else if (d->zfmt == 2) {
            stl_le_p(zp, znew | (ldl_le_p(zp) & 0xff000000u));
        } else {
            stl_le_p(zp, znew);
        }
    }
    if (!(zpass && spass)) {
        d->st->db_fails++;
        return false;
    }
    return true;
}

/* the depth buffer and stencil buffer of the draw, if it uses them */
static void db_setup(Draw *d)
{
    HD5770Gfx *g = d->g;
    uint32_t zinfo = CTX(g, DB_Z_INFO), sinfo = CTX(g, DB_STENCIL_INFO);
    uint32_t size = CTX(g, DB_DEPTH_SIZE);
    HD5770Surf s;

    d->dctl = CTX(g, DB_DEPTH_CONTROL);
    d->rctl = CTX(g, DB_RENDER_CONTROL);
    d->zfmt = zinfo & 3;
    d->has_s = sinfo & 1;
    if (!(d->dctl & (DC_Z | DC_Z_WRITE)) && !(d->rctl & RC_DEPTH_CLEAR)) {
        d->zfmt = 0;
    }
    if (!(d->dctl & DC_STENCIL) && !(d->rctl & RC_STENCIL_CLEAR)) {
        d->has_s = false;
    }
    if (d->rctl & RC_COPY) {
        /* depth/stencil copy to colour buffer 0 */
        d->st->gap_misc[HD5770_GAP_DEPTH]++;
    }
    if (!d->zfmt && !d->has_s) {
        return;
    }
    memset(&s, 0, sizeof(s));
    s.pitch = ((size & 0x7ff) + 1) * 8;
    s.height = (((size >> 11) & 0x7ff) + 1) * 8;
    s.mode = (zinfo >> 4) & 0xf;
    s.banks = 2u << ((zinfo >> 12) & 3);
    s.bankw = 1u << ((zinfo >> 16) & 3);
    s.bankh = 1u << ((zinfo >> 20) & 3);
    s.mtilea = 1u << ((zinfo >> 24) & 3);
    s.nondisp = true;
    hd5770_surf_set_addr_config(&s, addr_config(g));
    if (!hd5770_surf_mode_ok(s.mode)) {
        d->st->gap_mode[s.mode]++;
        d->zfmt = 0;
        d->has_s = false;
        return;
    }
    d->zs = s;
    d->zs.base = (uint64_t)CTX(g, DB_Z_WRITE_BASE) << 8;
    d->zs.bpe = d->zfmt == 1 ? 2 : 4;
    d->zs.tile_split = 64u << ((zinfo >> 8) & 7);
    hd5770_surf_prep(&d->zs);
    d->ss = s;
    d->ss.base = (uint64_t)CTX(g, DB_STENCIL_WRITE_BASE) << 8;
    d->ss.bpe = 1;
    d->ss.tile_split = 64u << ((sinfo >> 8) & 7);
    hd5770_surf_prep(&d->ss);
    d->srm[0] = CTX(g, DB_STENCILREFMASK);
    d->srm[1] = CTX(g, DB_STENCILREFMASK_BF);
    memcpy(&d->zclear, &g->ctx[(DB_DEPTH_CLEAR - 0x28000) / 4], 4);
    d->sclear = CTX(g, DB_STENCIL_CLEAR) & 0xff;
    memcpy(&d->zmin, &g->ctx[(PA_SC_VPORT_ZMIN_0 - 0x28000) / 4], 4);
    memcpy(&d->zmax, &g->ctx[(PA_SC_VPORT_ZMAX_0 - 0x28000) / 4], 4);
    d->zmin = fmaxf(d->zmin, 0.0f);
    d->zmax = fminf(d->zmax, 1.0f);
    if (!(d->zmin <= d->zmax)) {
        d->zmin = 0.0f;
        d->zmax = 1.0f;
    }
    d->zexport = CTX(g, DB_SHADER_CONTROL) & 1;
    d->sexport = (CTX(g, DB_SHADER_CONTROL) >> 1) & 1;
    d->z_mtx = d->z_mty = d->s_mtx = d->s_mty = UINT32_MAX;
    d->x1 = MIN(d->x1, (int)s.pitch);
    d->y1 = MIN(d->y1, (int)s.height);
    d->db = true;
}

/* resolve: colour buffer 1 gets colour buffer 0 (one sample) */
static void resolve_pixel(Draw *d, int x, int y)
{
    uint64_t so = surf_off(&d->rs, &d->rs_mtx, &d->rs_mty, &d->rs_mbase,
                           x, y);
    uint8_t *sp = hd5770_span_ptr(&d->g->mem, &d->span_rs, d->rs.base + so,
                                  d->rs.bpe, false);
    HD5770Val col[4];
    uint32_t c[4];
    unsigned bits[4], i, n;

    if (!sp) {
        return;
    }
    if (d->rs_fmt == d->rt_fmt && d->rs_num == d->rt_num &&
        d->rs_swap == d->rt_swap) {
        uint64_t off = surf_off(&d->rt, &d->rt_mtx, &d->rt_mty, &d->rt_mbase,
                                x, y);
        uint8_t *p = hd5770_span_ptr(&d->g->mem, &d->span_rt,
                                     d->rt.base + off, d->rt_bpe, true);

        if (p) {
            memcpy(p, sp, d->rt_bpe);
            d->st->pixels++;
        }
        return;
    }
    hd5770_fmt_unpack(d->rs_fmt, sp, c, bits);
    n = hd5770_fmt_comps(d->rs_fmt);
    col[0].f = col[1].f = col[2].f = 0.0f;
    col[3].f = 1.0f;
    for (i = 0; i < n; i++) {
        unsigned s = swap_src(d->rs_swap, n, i);

        col[s] = comp_value(c[i], bits[i], d->rs_num == 7, false,
                            d->rs_num == 4 ? HD5770_NUM_INT : HD5770_NUM_NORM);
    }
    put_pixel(d, x, y, col);
    d->st->pixels++;
}

/* ---------------------------------------------------------------- */
/* Rasteriser                                                       */

static bool alpha_pass(Draw *d, float a)
{
    float r = d->alpha_ref;

    switch (d->alpha_func) {
    case 0:
        return false;
    case 1:
        return a < r;
    case 2:
        return a == r;
    case 3:
        return a <= r;
    case 4:
        return a > r;
    case 5:
        return a != r;
    case 6:
        return a >= r;
    default:
        return true;
    }
}

static float edge(float ax, float ay, float bx, float by, float px, float py);

/* barycentrics of pixel (x, y) in the current triangle */
static void bary_at(Draw *d, int x, int y, float *b0, float *b1, float *b2,
                    float *pw)
{
    const Vtx *v0 = d->tv[0], *v1 = d->tv[1], *v2 = d->tv[2];
    float px = x + d->pix_center, py = y + d->pix_center;

    *b0 = edge(v1->x, v1->y, v2->x, v2->y, px, py) / d->area;
    *b1 = edge(v2->x, v2->y, v0->x, v0->y, px, py) / d->area;
    *b2 = edge(v0->x, v0->y, v1->x, v1->y, px, py) / d->area;
    if (d->persp) {
        float s = *b0 * v0->w + *b1 * v1->w + *b2 * v2->w;

        *pw = s != 0.0f ? 1.0f / s : 0.0f;
    }
}

static void bary(Draw *d, int x, int y)
{
    bary_at(d, x, y, &d->b0, &d->b1, &d->b2, &d->pw);
}

static void ps_setup(Draw *d, int x, int y, HD5770Thread *t)
{
    memset(t->color, 0, sizeof(t->color));
    t->color_mask = 0;
    t->killed = false;
    t->depth_written = false;
    if (d->pos_ena) {
        t->r[d->pos_gpr][0].f = x + d->pix_center;
        t->r[d->pos_gpr][1].f = y + d->pix_center;
        t->r[d->pos_gpr][2].f = 0.0f;
        t->r[d->pos_gpr][3].f = 1.0f;
    }
    t->r[0][0].f = d->b1;
    t->r[0][1].f = d->b2;
}

static bool neighbour(void *opaque, int dx, int dy, uint32_t cf,
                      uint32_t inst, unsigned gpr, HD5770Val out[4])
{
    Draw *d = opaque;
    float b0 = d->b0, b1 = d->b1, b2 = d->b2, pw = d->pw;
    bool ok;

    if (!d->nt) {
        d->nt = g_new0(HD5770Thread, 1);
    }
    bary(d, d->px + dx, d->py + dy);
    ps_setup(d, d->px + dx, d->py + dy, d->nt);
    d->ps_env.neighbour = NULL;
    ok = hd5770_prog_run_until(d->ps, &d->ps_env, d->nt, cf, inst, gpr, out);
    d->ps_env.neighbour = neighbour;
    d->b0 = b0;
    d->b1 = b1;
    d->b2 = b2;
    d->pw = pw;
    return ok;
}

static bool clip_pass(Draw *d, int x, int y)
{
    unsigned i, idx = 0;

    for (i = 0; i < 4; i++) {
        if (x >= d->cr[i][0] && x < d->cr[i][2] &&
            y >= d->cr[i][1] && y < d->cr[i][3]) {
            idx |= 1u << i;
        }
    }
    return (d->cliprule >> idx) & 1;
}

/* the interpolated depth at barycentrics b0, b1, b2 */
static inline float frag_z(const Draw *d, float b0, float b1, float b2)
{
    return (d->tv[0]->z + d->zoff) * b0 + (d->tv[1]->z + d->zoff) * b1 +
           (d->tv[2]->z + d->zoff) * b2;
}

static void shade(Draw *d, int x, int y, HD5770Thread *t)
{
    if (d->cliprule != 0xffff && !clip_pass(d, x, y)) {
        return;
    }
    if (d->resolve) {
        resolve_pixel(d, x, y);
        return;
    }
    d->px = x;
    d->py = y;
    ps_setup(d, x, y, t);
    if (!hd5770_prog_run(d->ps, &d->ps_env, t)) {
        d->ok = false;
        return;
    }
    if (t->killed) {
        d->st->kills++;
        return;
    }
    if (d->alpha_test && (t->color_mask & 1) &&
        !alpha_pass(d, t->color[0][3].f)) {
        return;
    }
    if (d->db) {
        bool exp = t->depth_written;
        float z = exp && d->zexport ? t->depth.f
                                    : frag_z(d, d->b0, d->b1, d->b2);

        if (!db_test(d, x, y, z, exp && d->sexport ? (int)(t->stencil.u & 0xff)
                                                   : -1)) {
            return;
        }
    }
    if (d->mrt) {
        uint8_t *cur = (uint8_t *)d + MRT_FIRST;
        unsigned k;

        for (k = 0; k < d->nmrt; k++) {
            unsigned cb = d->mrt_cb[k];

            if (t->color_mask & (1u << cb)) {
                memcpy(cur, d->mrt + k * MRT_BYTES, MRT_BYTES);
                put_pixel(d, x, y, t->color[cb]);
                memcpy(d->mrt + k * MRT_BYTES, cur, MRT_BYTES);
            }
        }
        memcpy(cur, d->mrt, MRT_BYTES);
        d->st->pixels++;
        return;
    }
    if (!d->color || !(t->color_mask & 1)) {
        return;
    }
    put_pixel(d, x, y, t->color[0]);
    d->st->pixels++;
}

/*
 * Edge function, evaluated with the endpoints in a fixed order so that
 * a shared edge gives exactly opposite values to its two triangles.
 */
static float edge(float ax, float ay, float bx, float by, float px, float py)
{
    if (ax > bx || (ax == bx && ay > by)) {
        return -((ax - bx) * (py - by) - (ay - by) * (px - bx));
    }
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

/* run the pixel shader on the collected pixels and write them in order */
static void lanes_flush(Draw *d)
{
    HD5770Lanes *l = d->lanes;
    unsigned x, k;

    if (!l->n) {
        return;
    }
    if (d->nh) {
        d->nh->n = d->nv->n = l->n;
    }
    hd5770_prog_run_lanes(d->ps, &d->ps_env, d->kv, l, d->nh, d->nv);
    if (!d->ok) {
        /* a failed fetch stops the draw after the first pixel, as serially */
        l->n = 1;
    }
    for (x = 0; x < l->n; x++) {
        HD5770Val col[4];

        if (l->killed[x]) {
            d->st->kills++;
            continue;
        }
        for (k = 0; k < 4; k++) {
            col[k] = l->color[k][x];
        }
        if (l->color0 && d->alpha_test && !alpha_pass(d, col[3].f)) {
            continue;
        }
        if (d->db) {
            bool exp = l->depth_written;
            float z = exp && d->zexport ? l->depth[x].f
                                        : frag_z(d, l->b0[x], l->b1[x], l->b2[x]);

            if (!db_test(d, d->lx[x], d->ly[x], z,
                         exp && d->sexport ? (int)(l->stencil[x].u & 0xff)
                                           : -1)) {
                continue;
            }
        }
        if (d->mrt) {
            uint8_t *cur = (uint8_t *)d + MRT_FIRST;
            unsigned m;

            for (m = 0; m < d->nmrt; m++) {
                unsigned cb = d->mrt_cb[m];

                if (l->color_mask & (1u << cb)) {
                    HD5770Val mc[4];

                    for (k = 0; k < 4; k++) {
                        mc[k] = cb ? l->mcolor[cb][k][x] : col[k];
                    }
                    memcpy(cur, d->mrt + m * MRT_BYTES, MRT_BYTES);
                    put_pixel(d, d->lx[x], d->ly[x], mc);
                    memcpy(d->mrt + m * MRT_BYTES, cur, MRT_BYTES);
                }
            }
            memcpy(cur, d->mrt, MRT_BYTES);
            d->st->pixels++;
            continue;
        }
        if (!d->color || !l->color0) {
            continue;
        }
        put_pixel(d, d->lx[x], d->ly[x], col);
        d->st->pixels++;
    }
    l->n = 0;
}

static void lanes_add(Draw *d, int x, int y)
{
    HD5770Lanes *l = d->lanes;
    unsigned k;

    if (d->cliprule != 0xffff && !clip_pass(d, x, y)) {
        return;
    }
    k = l->n++;
    d->lx[k] = x;
    d->ly[k] = y;
    l->b0[k] = d->b0;
    l->b1[k] = d->b1;
    l->b2[k] = d->b2;
    l->pw[k] = d->pw;
    l->px[k] = x + d->pix_center;
    l->py[k] = y + d->pix_center;
    if (d->nh) {
        HD5770Lanes *h = d->nh, *v = d->nv;

        bary_at(d, x + 1, y, &h->b0[k], &h->b1[k], &h->b2[k], &h->pw[k]);
        h->px[k] = x + 1 + d->pix_center;
        h->py[k] = y + d->pix_center;
        bary_at(d, x, y + 1, &v->b0[k], &v->b1[k], &v->b2[k], &v->pw[k]);
        v->px[k] = x + d->pix_center;
        v->py[k] = y + 1 + d->pix_center;
    }
    if (l->n == HD5770_LANES) {
        lanes_flush(d);
    }
}

/* top-left rule: an edge owns its pixels if it is a top or left edge */
static bool owns(float ax, float ay, float bx, float by)
{
    return (ay == by && bx < ax) || by < ay;
}

static void raster_tri(Draw *d, const Vtx *a, const Vtx *b, const Vtx *c,
                       HD5770Thread *t)
{
    float area = edge(a->x, a->y, b->x, b->y, c->x, c->y);
    int minx, maxx, miny, maxy, x, y;
    const Vtx *v0 = a, *v1 = b, *v2 = c;
    float pc = d->pix_center;

    if (area == 0.0f || area != area) {
        return;
    }
    if (d->nlanes && !d->lanes) {
        d->lanes = g_new0(HD5770Lanes, d->nlanes);
        d->lanes->pos_gpr = d->pos_ena ? (int)d->pos_gpr : -1;
        if (d->nlanes == 3) {
            d->nh = &d->lanes[1];
            d->nv = &d->lanes[2];
            d->nh->pos_gpr = d->nv->pos_gpr = d->lanes->pos_gpr;
        }
    }
    if (area < 0) {
        v1 = c;
        v2 = b;
        area = -area;
    }
    minx = MAX(d->x0, (int)floorf(MIN(a->x, MIN(b->x, c->x)) - pc));
    maxx = MIN(d->x1 - 1, (int)ceilf(MAX(a->x, MAX(b->x, c->x)) - pc));
    miny = MAX(d->y0, (int)floorf(MIN(a->y, MIN(b->y, c->y)) - pc));
    maxy = MIN(d->y1 - 1, (int)ceilf(MAX(a->y, MAX(b->y, c->y)) - pc));
    if (minx <= maxx && miny <= maxy) {
        d->wy0 = MIN(d->wy0, miny);
        d->wy1 = MAX(d->wy1, maxy);
    }
    d->tv[0] = v0;
    d->tv[1] = v1;
    d->tv[2] = v2;
    d->area = area;
    for (y = miny; y <= maxy && d->ok; y++) {
        float py = y + pc;

        for (x = minx; x <= maxx && d->ok; x++) {
            float px = x + pc;
            float w0 = edge(v1->x, v1->y, v2->x, v2->y, px, py);
            float w1 = edge(v2->x, v2->y, v0->x, v0->y, px, py);
            float w2 = edge(v0->x, v0->y, v1->x, v1->y, px, py);

            if (w0 < 0 || w1 < 0 || w2 < 0) {
                continue;
            }
            if ((w0 == 0 && !owns(v1->x, v1->y, v2->x, v2->y)) ||
                (w1 == 0 && !owns(v2->x, v2->y, v0->x, v0->y)) ||
                (w2 == 0 && !owns(v0->x, v0->y, v1->x, v1->y))) {
                continue;
            }
            d->b0 = w0 / area;
            d->b1 = w1 / area;
            d->b2 = w2 / area;
            if (d->persp) {
                float s = d->b0 * v0->w + d->b1 * v1->w + d->b2 * v2->w;

                d->pw = s != 0.0f ? 1.0f / s : 0.0f;
            }
            if (d->lanes) {
                lanes_add(d, x, y);
            } else {
                shade(d, x, y, t);
            }
        }
    }
    if (d->lanes) {
        lanes_flush(d);
    }
}

/* ---------------------------------------------------------------- */
/* The draw                                                         */

static void scissor(Draw *d, uint32_t tl, uint32_t br, bool woff, int wx,
                    int wy)
{
    int x0 = tl & 0x7fff, y0 = (tl >> 16) & 0x7fff;
    int x1 = br & 0x7fff, y1 = (br >> 16) & 0x7fff;

    if (woff && !(tl >> 31)) {
        x0 += wx;
        x1 += wx;
        y0 += wy;
        y1 += wy;
    }
    d->x0 = MAX(d->x0, x0);
    d->y0 = MAX(d->y0, y0);
    d->x1 = MIN(d->x1, x1);
    d->y1 = MIN(d->y1, y1);
}

/* R0 on entry: x vertex index, w instance index */
/* window coordinates of clip space position x, y, z, w */
static void vtx_window(Draw *d, Vtx *v, float x, float y, float z, float w)
{
    HD5770Gfx *g = d->g;
    const float *vp = (const float *)&g->ctx[(PA_CL_VPORT_XSCALE_0 - 0x28000)
                                             / 4];
    uint32_t vte = CTX(g, PA_CL_VTE_CNTL);

    if (!(vte & (1u << 8))) {
        if (w == 0.0f) {
            w = 1.0f;
        }
        x /= w;
        y /= w;
        z /= w;
        w = 1.0f / w;
    } else if (!(vte & (1u << 10))) {
        w = w != 0.0f ? 1.0f / w : 1.0f;
    }
    if (vte & 1) {
        x *= vp[0];
    }
    if (vte & 2) {
        x += vp[1];
    }
    if (vte & 4) {
        y *= vp[2];
    }
    if (vte & 8) {
        y += vp[3];
    }
    if (vte & 16) {
        z *= vp[4];
    }
    if (vte & 32) {
        z += vp[5];
    }
    if (CTX(g, PA_SU_SC_MODE_CNTL) & (1u << 16)) {
        uint32_t wo = CTX(g, PA_SC_WINDOW_OFFSET);

        x += (int16_t)(wo & 0xffff);
        y += (int16_t)(wo >> 16);
    }
    v->x = x;
    v->y = y;
    v->z = z;
    v->w = w;
}

/*
 * Distances of a clip space position to the near and far planes
 * (PA_CL_CLIP_CNTL: DX_CLIP_SPACE_DEF puts near at z = 0, else z = -w);
 * negative = outside.
 */
static void clip_dist(const Draw *d, const float *cp, float *dn, float *df)
{
    *dn = d->clip_dx ? cp[2] : cp[2] + cp[3];
    *df = cp[3] - cp[2];
    if (!d->clip_near) {
        *dn = 0.0f;
    }
    if (!d->clip_far) {
        *df = 0.0f;
    }
}

/* R0 on entry: x vertex index, w instance index */
static bool vs_run(Draw *d, uint32_t index, uint32_t inst, Vtx *v,
                   HD5770Thread *t)
{
    float dn, df;
    unsigned k;

    memset(t->r, 0, sizeof(t->r[0]) * 4);
    t->r[0][0].u = index;
    t->r[0][3].u = inst;
    t->param_mask = 0;
    t->pos_written = false;
    if (!hd5770_prog_run(d->vs, &d->vs_env, t) || !d->ok) {
        return false;
    }
    for (k = 0; k < 4; k++) {
        v->cp[k] = t->pos[k].f;
    }
    if (!(CTX(d->g, PA_CL_VTE_CNTL) & (1u << 8)) && !(v->cp[3] > 0.0f)) {
        /* behind the eye */
        d->st->gap_misc[HD5770_GAP_CLIP]++;
    }
    clip_dist(d, v->cp, &dn, &df);
    v->clip = d->clip_on && (dn < 0.0f || df < 0.0f);
    vtx_window(d, v, v->cp[0], v->cp[1], v->cp[2], v->cp[3]);
    for (k = 0; k < HD5770_PARAMS; k++) {
        if (t->param_mask & (1u << k)) {
            memcpy(v->param[k], t->param[k], sizeof(v->param[k]));
        }
    }
    d->st->verts++;
    return true;
}

static void ps_inputs(Draw *d)
{
    HD5770Gfx *g = d->g;
    unsigned nin = CTX(g, SPI_PS_IN_CONTROL_0) & 0x3f, p, k;
    unsigned nout = ((CTX(g, SPI_VS_OUT_CONFIG) >> 1) & 0x1f) + 1;
    /* FLAT_SHADE of an input counts only with the global FLAT_SHADE_ENA */
    bool flat_ena = CTX(g, SPI_INTERP_CONTROL_0) & 1;

    for (p = 0; p < HD5770_PARAMS; p++) {
        uint32_t c = CTX(g, SPI_PS_INPUT_CNTL_0 + 4 * p);
        unsigned sem = c & 0xff, def = (c >> 8) & 3;

        d->ps_src[p] = -1;
        d->ps_flat[p] = flat_ena && ((c >> 10) & 1);
        d->ps_default[p][0] = d->ps_default[p][1] = d->ps_default[p][2] =
            def >= 2 ? 1.0f : 0.0f;
        d->ps_default[p][3] = def & 1 ? 1.0f : 0.0f;
        if (p >= nin) {
            continue;
        }
        for (k = 0; k < nout && k < HD5770_PARAMS; k++) {
            uint32_t id = CTX(g, SPI_VS_OUT_ID_0 + 4 * (k / 4));

            if (((id >> (8 * (k % 4))) & 0xff) == sem) {
                d->ps_src[p] = k;
                break;
            }
        }
    }
    d->pos_ena = (CTX(g, SPI_PS_IN_CONTROL_0) >> 8) & 1;
    d->pos_gpr = (CTX(g, SPI_PS_IN_CONTROL_0) >> 10) & 0x1f;
    d->persp = (CTX(g, SPI_BARYC_CNTL) & 0x33) != 0;
}

typedef void TriFn(Draw *d, const Vtx *a, const Vtx *b, const Vtx *c,
                   HD5770Thread *t);

/*
 * A triangle of the draw: its facing, culling, the provoking vertex pv
 * (flat inputs; with PROVOKING_VTX_LAST, else the first vertex), then
 * tri().
 */
/*
 * POLY_OFFSET_FRONT/BACK: z + SCALE / 16 * max(|dz/dx|, |dz/dy|) +
 * OFFSET * 2^POLY_OFFSET_NEG_NUM_DB_BITS, clamped by PA_SU_POLY_OFFSET_CLAMP
 * when that is not 0; the same for every pixel of the triangle.
 */
static void poly_offset(Draw *d, const Vtx *a, const Vtx *b, const Vtx *c)
{
    HD5770Gfx *g = d->g;
    unsigned side = d->front ? 0 : 8;
    float scale, units, clamp, area, dzx, dzy, o;
    int8_t nb = CTX(g, PA_SU_POLY_OFFSET_DB_FMT_CNTL) & 0xff;

    memcpy(&scale, &g->ctx[(PA_SU_POLY_OFFSET_FRONT_SCALE + side - 0x28000) / 4],
           4);
    memcpy(&units, &g->ctx[(PA_SU_POLY_OFFSET_FRONT_SCALE + side + 4 - 0x28000)
                           / 4], 4);
    memcpy(&clamp, &g->ctx[(PA_SU_POLY_OFFSET_CLAMP - 0x28000) / 4], 4);
    area = (b->x - a->x) * (c->y - a->y) - (c->x - a->x) * (b->y - a->y);
    dzx = dzy = 0.0f;
    if (area != 0.0f) {
        dzx = ((b->z - a->z) * (c->y - a->y) - (c->z - a->z) * (b->y - a->y)) /
              area;
        dzy = ((c->z - a->z) * (b->x - a->x) - (b->z - a->z) * (c->x - a->x)) /
              area;
    }
    /*
     * One unit is half (Z_24) or a quarter (Z_16) of 2^NEG_NUM_DB_BITS:
     * Mesa's r600 driver programs GL's units times 2 and 4 for those
     * formats (evergreen_emit_polygon_offset())
     */
    o = scale / 16.0f * fmaxf(fabsf(dzx), fabsf(dzy)) +
        units * ldexpf(d->zfmt == 1 ? 0.25f : d->zfmt == 2 ? 0.5f : 1.0f, nb);
    if (clamp > 0.0f) {
        o = fminf(o, clamp);
    } else if (clamp < 0.0f) {
        o = fmaxf(o, clamp);
    }
    d->zoff = o == o ? o : 0.0f;
}

static void emit_tri(Draw *d, const Vtx *a, const Vtx *b, const Vtx *c,
                     const Vtx *tvp, HD5770Thread *t, TriFn *tri)
{
    float area = edge(a->x, a->y, b->x, b->y, c->x, c->y);
    bool cw = FACE_CW_IS_AREA_POS ? area > 0 : area < 0;

    d->front = d->face_cw ? cw : !cw;
    if (d->cull & (d->front ? 1 : 2)) {
        d->st->culled++;
        return;
    }
    d->tvp = tvp;
    d->zoff = 0.0f;
    if (CTX(d->g, PA_SU_SC_MODE_CNTL) & (d->front ? 1u << 11 : 1u << 12)) {
        poly_offset(d, a, b, c);
    }
    tri(d, a, b, c, t);
}

/* the point at f of the way from a to b, in clip space */
static void clip_lerp(Draw *d, const Vtx *a, const Vtx *b, float f, Vtx *o)
{
    unsigned p, k;

    for (k = 0; k < 4; k++) {
        o->cp[k] = a->cp[k] + (b->cp[k] - a->cp[k]) * f;
    }
    for (p = 0; p < HD5770_PARAMS; p++) {
        for (k = 0; k < 4; k++) {
            o->param[p][k].f = a->param[p][k].f +
                               (b->param[p][k].f - a->param[p][k].f) * f;
        }
    }
    o->clip = false;
    vtx_window(d, o, o->cp[0], o->cp[1], o->cp[2], o->cp[3]);
}

/*
 * A triangle with a vertex beyond the near or far plane: the part inside
 * (clip space, attributes linear there) as a fan; flat inputs keep the
 * provoking vertex of the whole triangle.
 */
static void emit(Draw *d, const Vtx *a, const Vtx *b, const Vtx *c,
                 const Vtx *pv, HD5770Thread *t, TriFn *tri)
{
    const Vtx *tvp = d->prov_last ? pv : a;
    Vtx *buf, *in[9], *out[9], **src = in, **dst = out;
    unsigned n = 3, m, plane, i, nb = 0;

    if (!a->clip && !b->clip && !c->clip) {
        emit_tri(d, a, b, c, tvp, t, tri);
        return;
    }
    d->st->clipped++;
    buf = g_new(Vtx, 12);
    in[0] = (Vtx *)a;
    in[1] = (Vtx *)b;
    in[2] = (Vtx *)c;
    for (plane = 0; plane < 2 && n >= 3; plane++) {
        float dist[9], dn, df;

        for (i = 0; i < n; i++) {
            clip_dist(d, src[i]->cp, &dn, &df);
            dist[i] = plane ? df : dn;
        }
        m = 0;
        for (i = 0; i < n; i++) {
            unsigned j = (i + 1) % n;

            if (dist[i] >= 0.0f) {
                dst[m++] = src[i];
            }
            if ((dist[i] >= 0.0f) != (dist[j] >= 0.0f) && nb < 12 && m < 9) {
                clip_lerp(d, src[i], src[j], dist[i] / (dist[i] - dist[j]),
                          &buf[nb]);
                dst[m++] = &buf[nb++];
            }
        }
        n = m;
        src = dst;
        dst = src == in ? out : in;
    }
    for (i = 1; i + 1 < n && d->ok; i++) {
        emit_tri(d, src[0], src[i], src[i + 1], tvp, t, tri);
    }
    g_free(buf);
}

/* the primitives of a draw as triangles, in order */
static void assemble(Draw *d, Vtx *const *vb, unsigned nv, unsigned prim,
                     HD5770Thread *t, TriFn *tri)
{
    HD5770GfxStats *st = d->st;
    unsigned i;

    switch (prim) {
    case PRIM_RECTLIST:
        /* three corners, v1 adjacent to both others: v3 = v0 + v2 - v1 */
        for (i = 0; i + 3 <= nv && d->ok; i += 3) {
            Vtx v3 = *vb[i + 2];
            unsigned p, k;

            v3.x = vb[i]->x + vb[i + 2]->x - vb[i + 1]->x;
            v3.y = vb[i]->y + vb[i + 2]->y - vb[i + 1]->y;
            v3.z = vb[i]->z + vb[i + 2]->z - vb[i + 1]->z;
            for (p = 0; p < HD5770_PARAMS; p++) {
                for (k = 0; k < 4; k++) {
                    v3.param[p][k].f = vb[i]->param[p][k].f +
                                       vb[i + 2]->param[p][k].f -
                                       vb[i + 1]->param[p][k].f;
                }
            }
            emit(d, vb[i], vb[i + 1], vb[i + 2], vb[i + 2], t, tri);
            emit(d, vb[i], vb[i + 2], &v3, vb[i + 2], t, tri);
            st->prims++;
        }
        break;
    case PRIM_QUADLIST:
        for (i = 0; i + 4 <= nv && d->ok; i += 4) {
            emit(d, vb[i], vb[i + 1], vb[i + 2], vb[i + 3], t, tri);
            emit(d, vb[i], vb[i + 2], vb[i + 3], vb[i + 3], t, tri);
            st->prims++;
        }
        break;
    case PRIM_QUADSTRIP:
        /* quad i: vertices 2i, 2i + 1, 2i + 3, 2i + 2 */
        for (i = 0; i + 4 <= nv && d->ok; i += 2) {
            emit(d, vb[i], vb[i + 1], vb[i + 3], vb[i + 3], t, tri);
            emit(d, vb[i], vb[i + 3], vb[i + 2], vb[i + 3], t, tri);
            st->prims++;
        }
        break;
    case PRIM_TRILIST:
        for (i = 0; i + 3 <= nv && d->ok; i += 3) {
            emit(d, vb[i], vb[i + 1], vb[i + 2], vb[i + 2], t, tri);
            st->prims++;
        }
        break;
    case PRIM_TRISTRIP:
        for (i = 2; i < nv && d->ok; i++) {
            if (i & 1) {
                emit(d, vb[i - 1], vb[i - 2], vb[i], vb[i], t, tri);
            } else {
                emit(d, vb[i - 2], vb[i - 1], vb[i], vb[i], t, tri);
            }
            st->prims++;
        }
        break;
    case PRIM_TRIFAN:
    case PRIM_POLYGON:
        for (i = 2; i < nv && d->ok; i++) {
            emit(d, vb[0], vb[i - 1], vb[i], vb[i], t, tri);
            st->prims++;
        }
        break;
    }
}

/* a triangle for the host GPU: raster_tri()'s set-up */
static void gpu_tri(Draw *d, const Vtx *a, const Vtx *b, const Vtx *c,
                    HD5770Thread *t)
{
    float area = edge(a->x, a->y, b->x, b->y, c->x, c->y);
    const Vtx *v[3] = { a, b, c };
    float pc = d->pix_center;
    HD5770GpuTri gt;
    unsigned p, k, q;

    if (area == 0.0f || area != area) {
        return;
    }
    if (area < 0) {
        v[1] = c;
        v[2] = b;
        area = -area;
    }
    gt.x0 = MAX(d->x0, (int)floorf(MIN(a->x, MIN(b->x, c->x)) - pc));
    gt.x1 = MIN(d->x1 - 1, (int)ceilf(MAX(a->x, MAX(b->x, c->x)) - pc));
    gt.y0 = MAX(d->y0, (int)floorf(MIN(a->y, MIN(b->y, c->y)) - pc));
    gt.y1 = MIN(d->y1 - 1, (int)ceilf(MAX(a->y, MAX(b->y, c->y)) - pc));
    if (gt.x0 > gt.x1 || gt.y0 > gt.y1) {
        return;
    }
    for (q = 0; q < 3; q++) {
        gt.x[q] = v[q]->x;
        gt.y[q] = v[q]->y;
        gt.w[q] = v[q]->w;
    }
    gt.area = area;
    gt.prim = d->st->prims;
    for (q = 0; q < 3; q++) {
        gt.z[q] = v[q]->z + d->zoff;
    }
    gt.front = d->front;
    g_array_append_val(d->gtri, gt);
    for (p = 0; p < d->gnin; p++) {
        int src = d->ps_src[p];

        for (q = 0; q < 3; q++) {
            const Vtx *vq = d->ps_flat[p] ? d->tvp : v[q];

            for (k = 0; k < 4; k++) {
                float f = src < 0 ? d->ps_default[p][k]
                                  : vq->param[src][k].f;

                g_array_append_val(d->gprm, f);
            }
        }
    }
}

enum {
    GFB_PROG, GFB_SMALL, GFB_RT, GFB_TEX, GFB_VB, GFB_BACKEND, GFB_DB,
    GFB_NONE,
};

/* a draw falls back to the CPU for reason r */
#define GPU_FB(d, r, w) ((d)->gfb = (r), (d)->cwhy = (w), \
                         (d)->g->gst.fallback[(r)]++)

static const char *const why_names[HD5770_WHY_N] = {
    [HD5770_WHY_GPU] = "host GPU",
    [HD5770_WHY_BUILDING] = "left out while building",
    [HD5770_WHY_EMPTY] = "nothing to draw",
    [HD5770_WHY_SKIPPED] = "skipped",
    [HD5770_WHY_SMALL] = "small",
    [HD5770_WHY_RESOLVE] = "resolve",
    [HD5770_WHY_NO_TARGET] = "no colour, no depth",
    [HD5770_WHY_DEPTH] = "depth/stencil",
    [HD5770_WHY_ZEXPORT] = "depth export",
    [HD5770_WHY_RT_WIDE] = "target over 8 bytes",
    [HD5770_WHY_MRT] = "more render targets",
    [HD5770_WHY_RT_SURF] = "target surface",
    [HD5770_WHY_RT_RAM_VERIFY] = "target in RAM (verify)",
    [HD5770_WHY_MIPS] = "mipmaps",
    [HD5770_WHY_FEEDBACK] = "samples its target",
    [HD5770_WHY_IR_NONE] = "no IR",
    [HD5770_WHY_IR_FS_CF] = "no IR: fetch shader",
    [HD5770_WHY_IR_CLAUSE] = "no IR: ALU clause kind",
    [HD5770_WHY_IR_ALU_OP] = "no IR: ALU op",
    [HD5770_WHY_IR_REL_CONST] = "no IR: indexed constant",
    [HD5770_WHY_IR_GPRS] = "no IR: registers",
    [HD5770_WHY_IR_TEX_UNITS] = "no IR: texture units",
    [HD5770_WHY_IR_FLOW] = "no IR: flow control",
    [HD5770_WHY_IR_CF_OP] = "no IR: CF op",
    [HD5770_WHY_CF_FORM] = "CF form",
    [HD5770_WHY_EXPORTS] = "colour/depth exports",
    [HD5770_WHY_RESINFO] = "RESINFO",
    [HD5770_WHY_FETCH_OP] = "SAMPLE_C",
    [HD5770_WHY_CUBE] = "CUBE",
    [HD5770_WHY_FLAT_LAST] = "flat input, last vertex",
    [HD5770_WHY_TEX_SETUP] = "texture not modelled",
    [HD5770_WHY_TEX_SAMPLERS] = "texture with two samplers",
    [HD5770_WHY_TEX_KIND] = "BC/3D/cube/array texture",
    [HD5770_WHY_TEX_SURF] = "texture surface",
    [HD5770_WHY_TEX_MIP_SURF] = "mip level surface",
    [HD5770_WHY_VB_FMT] = "vertex format",
    [HD5770_WHY_VB_COUNT] = "vertex buffers",
    [HD5770_WHY_VB_SIZE] = "vertex buffer size",
    [HD5770_WHY_DB_SURF] = "depth surface",
    [HD5770_WHY_RS_SURF] = "resolve surface",
    [HD5770_WHY_BK_PROG] = "backend: program",
    [HD5770_WHY_BK_SURF] = "backend: surface",
    [HD5770_WHY_BK_ROOM] = "backend: room",
    [HD5770_WHY_BK_DEAD] = "backend: lost",
    [HD5770_WHY_BK_OTHER] = "backend: other",
    [HD5770_WHY_BK_BUILD_SLOW] = "backend: build time",
    [HD5770_WHY_BK_NATIVE_STATE] = "backend: state",
    [HD5770_WHY_MIXED_DEPTH] = "mixed depth",
};

const char *hd5770_why_name(unsigned why)
{
    return why < HD5770_WHY_N && why_names[why] ? why_names[why] : "?";
}

/* the program of a resolve: none */
static HD5770IR resolve_ir = { .hash = 0x7265736f6c766500ull };

/*
 * texture unit u of a request: the CPU path's Tex, in VRAM;
 * HD5770_WHY_GPU, else why not
 */
static unsigned gpu_tex(Draw *d, const HD5770IR *ir, unsigned u,
                        HD5770GpuTex *gt)
{
    unsigned id = ir->res[u], sid = 0, i, k;
    bool seen = false;
    Tex tmp;
    const Tex *tp;

    for (i = 0; i < ir->nfetches; i++) {
        const HD5770IRFetch *f = &ir->fetches[i];
        unsigned s = (f->w[2] >> 15) & 0x1f;

        if (f->kind == HD5770_IRF_TEX && f->res == id) {
            if (seen && s != sid) {
                return HD5770_WHY_TEX_SAMPLERS;
            }
            sid = s;
            seen = true;
        }
    }
    memset(gt, 0, sizeof(*gt));
    tp = tex_peek(d, id, sid, &tmp);
    if (!tp) {
        return HD5770_WHY_TEX_SETUP;
    }
    if (!d->g->gpu->ops->full && (tp->bc || (tp->dim != 0 && tp->dim != 1))) {
        /* compressed, 3D, cube and array textures: backends with full */
        return HD5770_WHY_TEX_KIND;
    }
    if (!gpu_surf_n(d->g, &tp->s, MAX(tp->slice_bytes[0] * MAX(tp->slices, 1u),
                                    hd5770_surf_size(&tp->s)), false,
                    &gt->surf)) {
        return HD5770_WHY_TEX_SURF;
    }
    gt->bc = tp->bc;
    gt->slices = MAX(tp->slices, 1u);
    memcpy(gt->slice_bytes, tp->slice_bytes, sizeof(gt->slice_bytes));
    gt->fmt = tp->fmt;
    gt->num = tp->num;
    gt->w = tp->w;
    gt->h = tp->h;
    gt->dim = tp->dim;
    gt->sgn = tp->sgn;
    gt->comps = tp->comps;
    gt->is_float = tp->is_float;
    gt->unorm8 = tp->unorm8;
    gt->base_lv = tp->base_lv;
    gt->levels = tp->levels;
    gt->depth = tp->depth;
    memcpy(gt->samp, tp->samp, sizeof(gt->samp));
    memcpy(gt->border, tp->border, sizeof(gt->border));
    {
        uint8_t zero[16] = { 0 };
        uint32_t c[4];

        hd5770_fmt_unpack(tp->fmt, zero, c, gt->bits);
    }
    for (k = 0; k < 4; k++) {
        gt->dst_sel[k] = tp->dst_sel[k];
    }
    if (tp->mip) {
        HD5770GpuSurf *lv = g_new0(HD5770GpuSurf, tp->last_lv + 1);

        gt->lv = lv;
        lv[0] = gt->surf;
        for (k = 1; k <= tp->last_lv; k++) {
            if (!gpu_surf_n(d->g, &tp->lv[k],
                            MAX(tp->slice_bytes[k] * tex_slices(tp, k),
                                hd5770_surf_size(&tp->lv[k])),
                            false, &lv[k]) || lv[k].space) {
                return HD5770_WHY_TEX_MIP_SURF;
            }
            gt->lw[k] = tp->lw[k];
            gt->lh[k] = tp->lh[k];
        }
        gt->lw[0] = tp->w;
        gt->lh[0] = tp->h;
        gt->mip = true;
        gt->last_lv = tp->last_lv;
        gt->mip_filter = tp->mip_filter;
        gt->min_lod = tp->min_lod;
        gt->max_lod = tp->max_lod;
        gt->lod_bias = tp->lod_bias;
    }
    return HD5770_WHY_GPU;
}

/*
 * the vertex buffers the pixel shader fetches from, copied;
 * HD5770_WHY_GPU, else why not
 */
static unsigned gpu_vbs(Draw *d, const HD5770IR *ir, HD5770GpuDraw *q)
{
    HD5770Gfx *g = d->g;
    unsigned i, k;

    for (i = 0; i < ir->nfetches; i++) {
        const HD5770IRFetch *f = &ir->fetches[i];
        unsigned bid = (f->w[0] >> 8) & 0xff, fmt;
        const uint32_t *r;
        uint64_t base;
        uint32_t size;

        if (f->kind != HD5770_IRF_VTX || bid == 128) {
            continue;
        }
        r = &g->ctx[(SQ_RESOURCE_0 - 0x28000) / 4 + bid * 8];
        fmt = (f->w[1] >> 21) & 1 ? (r[2] >> 20) & 0x3f : (f->w[1] >> 22) & 0x3f;
        if (!hd5770_fmt_bytes(fmt)) {
            return HD5770_WHY_VB_FMT;
        }
        for (k = 0; k < q->nvb && q->vb[k].id != bid; k++) {
            /* search */
        }
        if (k < q->nvb) {
            continue;
        }
        if (q->nvb == HD5770_GPU_VB) {
            return HD5770_WHY_VB_COUNT;
        }
        base = r[0] | ((uint64_t)(r[2] & 0xff) << 32);
        size = r[1] + 1;
        if (size > 4 * 1024 * 1024) {
            return HD5770_WHY_VB_SIZE;
        }
        pend_check(g, base, size);
        q->vb[k].id = bid;
        memcpy(q->vb[k].res, r, sizeof(q->vb[k].res));
        q->vb[k].len = size;
        q->vb[k].data = g_malloc(size);
        hd5770_mem_read(&g->mem, base, (uint8_t *)q->vb[k].data, size);
        q->nvb++;
    }
    return HD5770_WHY_GPU;
}

static void gpu_req_free(HD5770GpuDraw *q)
{
    unsigned i;

    for (i = 0; i < q->nvb; i++) {
        g_free((void *)q->vb[i].data);
    }
    for (i = 0; i < HD5770_GPU_TEX; i++) {
        g_free((void *)q->tex[i].surf.pages);
        g_free((void *)q->tex[i].lv);
    }
    for (i = 0; i < q->nmrt; i++) {
        g_free((void *)q->mrt[i].surf.pages);
    }
    g_free((void *)q->rt.pages);
    g_free((void *)q->zs.pages);
    g_free((void *)q->ss.pages);
    g_free((void *)q->rs.pages);
    g_free(q);
}

/*
 * Draws with depth or stencil cover at least this many pixels to go to
 * the host GPU: smaller ones (pieces of a model, a few hundred pixels
 * each) cost more in round trips and barriers than on the CPU.
 */
#define HD5770_DB_MIN_PIXELS    16384
/*
 * a draw without depth whose program is being built goes to the CPU up
 * to this size, and is left out above it
 */
#define HD5770_BUILD_CPU_PIXELS (4u << 20)

/*
 * The CPU's work per pixel of a program, in units of the desktop's
 * programs (a few instructions and one fetch, on lanes): the size
 * thresholds above are for those. Programs on the interpreter cost
 * more, and mipmapped or gradient fetches run the neighbours too.
 */
static uint64_t ps_cost(const Draw *d, const HD5770IR *ir)
{
    uint64_t c = ir->ninsts + 8 * ir->nfetches;

    if (!d->nlanes) {
        c *= 4;
    }
    if (d->mips || ir->grad) {
        c *= 3;
    }
    return MAX(c / 24, 1);
}

/*
 * Rows y0..y1 of the draw's target, depth or stencil plane are in VRAM
 * ranges queued draws use: drawing them on the CPU would wait for the GPU.
 */
static bool gpu_queued(HD5770Gfx *g, Draw *d, int y0, int y1)
{
    const HD5770Surf *ws[3] = { d->color ? &d->rt : NULL,
                                d->db && d->zfmt ? &d->zs : NULL,
                                d->db && d->has_s ? &d->ss : NULL };
    unsigned i, k;

    for (i = 0; i < 3 && y1 >= y0; i++) {
        uint64_t lo, hi, off;

        if (!ws[i]) {
            continue;
        }
        hd5770_surf_rows(ws[i], y0, y1 + 1, &lo, &hi);
        if (!vram_off(g, lo, hi - lo, &off)) {
            continue;
        }
        for (k = 0; k < g->npend; k++) {
            if (off < g->pend[k][1] && g->pend[k][0] < off + (hi - lo)) {
                return true;
            }
        }
    }
    return false;
}

/*
 * Where a draw's indices come from: none (kind 0), the buffer of max
 * elements at MC address base (1), the packet (2)
 */
typedef struct IdxSrc {
    unsigned kind;
    uint64_t base;
    uint32_t max;
    unsigned isz, swap;
} IdxSrc;

/* the inputs the pixel shader interpolates (its IR's), 0 without one */
static unsigned ps_ninputs(Draw *d)
{
    const HD5770IR *ir = hd5770_prog_ir(d->ps);
    unsigned i, n = 0;

    for (i = 0; ir && i < ir->ninsts; i++) {
        if (ir->insts[i].interp) {
            n = MAX(n, ir->insts[i].prm + 1u);
        }
    }
    return n;
}

/* a byte buffer of len bytes at MC address base, in place */
static bool gpu_buf(HD5770Gfx *g, uint64_t base, uint64_t len,
                    HD5770GpuSurf *o)
{
    HD5770Surf s;

    memset(&s, 0, sizeof(s));
    s.base = base;
    s.pitch = MAX(len, 1);
    s.height = 1;
    s.bpe = 1;
    s.mode = HD5770_ARRAY_LINEAR_GENERAL;
    s.fast = true;
    return gpu_surf_n(g, &s, MAX(len, 1), false, o);
}

static void gpu_vs_free(HD5770GpuVs *v)
{
    unsigned k;

    g_free((void *)v->ib.pages);
    for (k = 0; k < v->nvb; k++) {
        g_free((void *)v->vb[k].buf.pages);
    }
}

/*
 * The vertex stage of a draw for the host GPU (HD5770GpuVs): nv index
 * slots from is (idx: the indices draw_indices() read), slot[] the CPU
 * path's (UINT32_MAX: restart), ninst instances
 */
static bool gpu_vs(Draw *d, HD5770GpuVs *v, const IdxSrc *is,
                   const uint32_t *idx, const uint32_t *slot, unsigned nv,
                   unsigned prim, unsigned ninst, HD5770Val **kvp)
{
    HD5770Gfx *g = d->g;
    const HD5770IR *ir = hd5770_vs_ir((HD5770Prog *)d->vs, d->fs);
    unsigned i, k, side;
    HD5770Val *kv;

    memset(v, 0, sizeof(*v));
    *kvp = NULL;
    if (!ir) {
        return false;
    }
    v->ir = ir;
    v->cfile = g->alu + 1024;
    v->nslot = nv;
    v->ninst = ninst;
    v->np = hd5770_vs_np(ir);
    v->prim = prim;
    v->slot = slot;
    v->idx_kind = is->kind;
    v->ioff = CTX(g, VGT_INDX_OFFSET);
    v->lo = CTX(g, VGT_MIN_VTX_INDX);
    v->hi = CTX(g, VGT_MAX_VTX_INDX);
    v->restart = CTX(g, VGT_MULTI_PRIM_IB_RESET_EN) & 1;
    v->reset = CTX(g, VGT_MULTI_PRIM_IB_RESET_INDX);
    v->ib_i32 = g->index_type & 1;
    v->ib_swap = (g->index_type >> 2) & 3;
    v->imm = idx;
    if (is->kind == 1) {
        v->ib_max = MIN(is->max, nv);
        if (v->ib_max) {
            uint64_t len = (uint64_t)v->ib_max << (v->ib_i32 ? 2 : 1);

            pend_check(g, is->base, len);
            if (!gpu_buf(g, is->base, len, &v->ib)) {
                return false;
            }
        }
    }
    v->base_vtx = g->ind_draw ? g->ind_vtx : CTX(g, SQ_VTX_BASE_VTX_LOC);
    v->start_inst = g->ind_draw ? g->ind_inst : CTX(g, SQ_VTX_START_INST_LOC);
    for (i = 0; i < ir->nfetches; i++) {
        const HD5770IRFetch *f = &ir->fetches[i];
        unsigned bid = (f->w[0] >> 8) & 0xff, id, fmt;
        const uint32_t *r;
        uint64_t base;
        uint32_t size;

        if (f->kind != HD5770_IRF_VTX) {
            return false;
        }
        if (bid == 128) {
            continue;
        }
        id = bid + (f->fs ? 160 : 0);
        r = &g->ctx[(SQ_RESOURCE_0 - 0x28000) / 4 +
                    (bid + (f->fs ? RES_FS : RES_VS)) * 8];
        fmt = (f->w[1] >> 21) & 1 ? (r[2] >> 20) & 0x3f
                                  : (f->w[1] >> 22) & 0x3f;
        if (!hd5770_fmt_bytes(fmt)) {
            return false;
        }
        for (k = 0; k < v->nvb && v->vb[k].id != id; k++) {
            /* search */
        }
        if (k < v->nvb) {
            continue;
        }
        if (v->nvb == HD5770_GPU_VB) {
            return false;
        }
        base = r[0] | ((uint64_t)(r[2] & 0xff) << 32);
        size = r[1] + 1;
        (g->gpu->vs_draw ? pend_check_vb : pend_check)(g, base, size);
        if (!gpu_buf(g, base, size, &v->vb[k].buf)) {
            return false;
        }
        v->vb[k].id = id;
        memcpy(v->vb[k].res, r, sizeof(v->vb[k].res));
        v->nvb++;
    }
    kv = g_new0(HD5770Val, ir->nconsts + 1);
    hd5770_ir_bind(ir, &d->vs_env, kv);
    *kvp = kv;
    v->kv = (const uint32_t *)kv;
    v->vte = CTX(g, PA_CL_VTE_CNTL);
    v->sc_mode = CTX(g, PA_SU_SC_MODE_CNTL);
    v->window_offset = CTX(g, PA_SC_WINDOW_OFFSET);
    memcpy(v->vport, &g->ctx[(PA_CL_VPORT_XSCALE_0 - 0x28000) / 4],
           sizeof(v->vport));
    v->clip_on = d->clip_on;
    v->clip_dx = d->clip_dx;
    v->clip_near = d->clip_near;
    v->clip_far = d->clip_far;
    v->cull = d->cull;
    v->face_cw = d->face_cw;
    v->prov_last = d->prov_last;
    v->x0 = d->x0;
    v->y0 = d->y0;
    v->x1 = d->x1;
    v->y1 = d->y1;
    for (side = 0; side < 2; side++) {
        memcpy(&v->poly_scale[side],
               &g->ctx[(PA_SU_POLY_OFFSET_FRONT_SCALE + side * 8 - 0x28000) / 4],
               4);
        memcpy(&v->poly_units[side],
               &g->ctx[(PA_SU_POLY_OFFSET_FRONT_SCALE + side * 8 + 4 - 0x28000)
                       / 4], 4);
    }
    memcpy(&v->poly_clamp, &g->ctx[(PA_SU_POLY_OFFSET_CLAMP - 0x28000) / 4], 4);
    v->poly_db_bits = CTX(g, PA_SU_POLY_OFFSET_DB_FMT_CNTL) & 0xff;
    v->zfmt = d->zfmt;
    for (i = 0; i < HD5770_PARAMS && i < 32; i++) {
        v->src[i] = d->ps_src[i];
        memcpy(v->def[i], d->ps_default[i], sizeof(v->def[i]));
    }
    v->ninputs = ps_ninputs(d);
    for (i = 0; i < v->ninputs; i++) {
        if (d->ps_src[i] < 0 || d->ps_flat[i]) {
            v->flat |= 1u << i;
        }
    }
    v->pix_center = d->pix_center;
    return true;
}

static void vs_emit(uint32_t *out, unsigned *n, uint32_t a, uint32_t b,
                    uint32_t c, uint32_t pv, uint32_t kind)
{
    if (out) {
        uint32_t *e = out + *n * 5;

        e[0] = a;
        e[1] = b;
        e[2] = c;
        e[3] = pv;
        e[4] = kind;
    }
    (*n)++;
}

unsigned hd5770_vs_emits(const HD5770GpuVs *v, uint32_t *out)
{
    unsigned n = 0, inst, i, s0, k;
    uint32_t *vp = g_new(uint32_t, MAX(v->nslot, 1u));

    for (inst = 0; inst < v->ninst; inst++) {
        uint32_t b = inst * v->nslot;

        /* runs of vertices between restart indices, as hd5770_gfx_draw() */
        for (i = 0, s0 = 0, k = 0; i <= v->nslot; i++) {
            const uint32_t *q;
            unsigned nv, j;

            if (i < v->nslot && v->slot[i] != UINT32_MAX) {
                vp[k++] = b + i;
                continue;
            }
            q = vp + s0;
            nv = k - s0;
            s0 = k;
            /* assemble() */
            switch (v->prim) {
            case PRIM_RECTLIST:
                for (j = 0; j + 3 <= nv; j += 3) {
                    vs_emit(out, &n, q[j], q[j + 1], q[j + 2], q[j + 2], 0);
                    vs_emit(out, &n, q[j], q[j + 2], q[j + 1], q[j + 2], 1);
                }
                break;
            case PRIM_QUADLIST:
                for (j = 0; j + 4 <= nv; j += 4) {
                    vs_emit(out, &n, q[j], q[j + 1], q[j + 2], q[j + 3], 0);
                    vs_emit(out, &n, q[j], q[j + 2], q[j + 3], q[j + 3], 0);
                }
                break;
            case PRIM_QUADSTRIP:
                for (j = 0; j + 4 <= nv; j += 2) {
                    vs_emit(out, &n, q[j], q[j + 1], q[j + 3], q[j + 3], 0);
                    vs_emit(out, &n, q[j], q[j + 3], q[j + 2], q[j + 3], 0);
                }
                break;
            case PRIM_TRILIST:
                for (j = 0; j + 3 <= nv; j += 3) {
                    vs_emit(out, &n, q[j], q[j + 1], q[j + 2], q[j + 2], 0);
                }
                break;
            case PRIM_TRISTRIP:
                for (j = 2; j < nv; j++) {
                    if (j & 1) {
                        vs_emit(out, &n, q[j - 1], q[j - 2], q[j], q[j], 0);
                    } else {
                        vs_emit(out, &n, q[j - 2], q[j - 1], q[j], q[j], 0);
                    }
                }
                break;
            case PRIM_TRIFAN:
            case PRIM_POLYGON:
                for (j = 2; j < nv; j++) {
                    vs_emit(out, &n, q[0], q[j - 1], q[j], q[j], 0);
                }
                break;
            }
        }
    }
    g_free(vp);
    return n;
}

/*
 * Verify mode: pass A of the draw's vertex stage on the host GPU, its
 * records (nv slots per instance), or NULL
 */
static uint32_t vs_ulp(uint32_t a, uint32_t b)
{
    int64_t x = (int32_t)a < 0 ? (int64_t)INT32_MIN - (int32_t)a : (int32_t)a;
    int64_t y = (int32_t)b < 0 ? (int64_t)INT32_MIN - (int32_t)b : (int32_t)b;

    return MIN(llabs(x - y), UINT32_MAX);
}

typedef struct VsCheck {
    uint32_t *rec;
    unsigned np, ninputs;
    /* the host GPU's triangles and inputs, and the CPU's (gpu_tri()) */
    HD5770GpuTri *tri;
    float *prm;
    unsigned ntri;
    GArray *ctri, *cprm;
} VsCheck;

static void vs_check_free(VsCheck *c)
{
    g_free(c->rec);
    g_free(c->tri);
    g_free(c->prm);
    if (c->ctri) {
        g_array_free(c->ctri, true);
        g_array_free(c->cprm, true);
    }
}

/*
 * Verify mode: the draw's vertex stage on the host GPU, its records (nv
 * slots per instance) and, if the pixel shader has an IR, triangles
 */
static bool vs_check_run(Draw *d, const IdxSrc *is, const uint32_t *idx,
                         const uint32_t *slot, unsigned nv, unsigned prim,
                         unsigned ninst, VsCheck *c)
{
    HD5770Gfx *g = d->g;
    HD5770GpuVs v;
    HD5770Val *kv = NULL;
    uint64_t words, ntri;

    memset(c, 0, sizeof(*c));
    if (gpu_vs(d, &v, is, idx, slot, nv, prim, ninst, &kv)) {
        words = (uint64_t)nv * ninst * HD5770_VS_REC(v.np);
        ntri = (uint64_t)hd5770_vs_emits(&v, NULL) * 3;
        if (words && words < 64 * 1024 * 1024 && ntri < 4 * 1024 * 1024) {
            c->rec = g_new(uint32_t, words);
            c->np = v.np;
            c->ninputs = v.ninputs;
            if (hd5770_prog_ir(d->ps)) {
                c->tri = g_new(HD5770GpuTri, MAX(ntri, 1));
                c->prm = g_new(float, MAX(ntri, 1) * MAX(v.ninputs, 1) * 12);
                c->ctri = g_array_new(false, false, sizeof(HD5770GpuTri));
                c->cprm = g_array_new(false, false, sizeof(float));
            }
            gpu_lock(g, true);
            if (!g->gpu->ops->vs_records(g->gpu->b, &v, c->rec, c->tri,
                                         c->prm, &c->ntri)) {
                vs_check_free(c);
                memset(c, 0, sizeof(*c));
            }
            gpu_lock(g, false);
        }
    }
    gpu_vs_free(&v);
    g_free(kv);
    return c->rec != NULL;
}

/*
 * The CPU's triangles of instance vb as gpu_tri() sets them up, after
 * the host GPU's (slots slot[0..nv-1] in vp, as hd5770_gfx_draw() runs
 * them); the draw's counts stay as they were
 */
static void vs_check_tris(Draw *d, VsCheck *c, Vtx *vb, Vtx **vp,
                          const uint32_t *slot, unsigned nv, unsigned prim,
                          HD5770Thread *t)
{
    HD5770GfxStats *st = d->st;
    uint64_t prims = st->prims, culled = st->culled, clipped = st->clipped;
    GArray *gt = d->gtri, *gp = d->gprm;
    unsigned gnin = d->gnin, i, s0 = 0, k = 0;

    d->gtri = c->ctri;
    d->gprm = c->cprm;
    d->gnin = c->ninputs;
    for (i = 0; i <= nv && d->ok; i++) {
        if (i < nv && slot[i] != UINT32_MAX) {
            vp[k++] = &vb[slot[i]];
            continue;
        }
        if (k > s0) {
            assemble(d, vp + s0, k - s0, prim, t, gpu_tri);
        }
        s0 = k;
    }
    d->gtri = gt;
    d->gprm = gp;
    d->gnin = gnin;
    st->prims = prims;
    st->culled = culled;
    st->clipped = clipped;
}

/* the host GPU's triangles against the CPU's */
static void vs_check_cmp(Draw *d, VsCheck *c)
{
    HD5770GpuStats *t = &d->g->gst;
    unsigned n = MIN(c->ntri, c->ctri->len), i, k, np = c->ninputs * 12;

    t->vs_tri_draws++;
    t->vs_tris += c->ctri->len;
    if (c->ntri != c->ctri->len) {
        t->vs_tri_count++;
    }
    for (i = 0; i < n; i++) {
        HD5770GpuTri *a = &g_array_index(c->ctri, HD5770GpuTri, i);
        HD5770GpuTri *b = &c->tri[i];
        const float *pa = &g_array_index(c->cprm, float, (size_t)i * np);
        const float *pb = c->prm + (size_t)i * np;
        unsigned bad = 0;
        uint32_t ua, ub;

        a->prim = b->prim = 0;
        for (k = 0; k < offsetof(HD5770GpuTri, z) / 4 + 4; k++) {
            memcpy(&ua, (uint32_t *)a + k, 4);
            memcpy(&ub, (uint32_t *)b + k, 4);
            if (ua != ub) {
                bad++;
                t->vs_tri_ulp = MAX(t->vs_tri_ulp, vs_ulp(ua, ub));
            }
        }
        for (k = 0; k < np; k++) {
            memcpy(&ua, &pa[k], 4);
            memcpy(&ub, &pb[k], 4);
            if (ua != ub) {
                bad++;
                t->vs_tri_ulp = MAX(t->vs_tri_ulp, vs_ulp(ua, ub));
            }
        }
        t->vs_tri_bad += bad != 0;
    }
}

/*
 * Instance inst's records against the CPU's vertices vb (vm: each one's
 * exported parameters, bit 32 the position written): the channels the
 * program exports
 */
static void vs_check(Draw *d, const uint32_t *rec, unsigned np,
                     const uint32_t *slot, unsigned nv, unsigned inst,
                     const Vtx *vb, const uint64_t *vm)
{
    HD5770GpuStats *t = &d->g->gst;
    const HD5770IR *ir = hd5770_vs_ir((HD5770Prog *)d->vs, d->fs);
    unsigned w = HD5770_VS_REC(np), i, p, k, cpos = 0, cprm[HD5770_PARAMS];

    memset(cprm, 0, sizeof(cprm));
    for (i = 0; ir && i < ir->nexports; i++) {
        const HD5770IRExport *e = &ir->exports[i];
        unsigned m = 0;

        for (k = 0; k < 4; k++) {
            m |= e->sel[k] < 6 ? 1u << k : 0;
        }
        if (e->target == HD5770_IR_POS) {
            cpos |= m;
        } else if (e->target - HD5770_IR_PARAM0 < HD5770_PARAMS) {
            cprm[e->target - HD5770_IR_PARAM0] |= m;
        }
    }
    for (i = 0; i < nv; i++) {
        const uint32_t *r = rec + ((size_t)inst * nv + i) * w;
        unsigned bad = 0;

        t->vs_recs++;
        if (slot[i] == UINT32_MAX) {
            bad = !(r[w - 1] & HD5770_VSF_RESTART);
        } else {
            const Vtx *x = &vb[slot[i]];
            uint64_t m = vm[slot[i]];

            bad += r[w - 1] != (m >> 32 ? HD5770_VSF_POS : 0);
            bad += r[w - 2] != (uint32_t)m;
            for (k = 0; k < 4; k++) {
                uint32_t c;

                memcpy(&c, &x->cp[k], 4);
                if ((cpos >> k) & 1 && c != r[k]) {
                    bad++;
                    t->vs_ulp = MAX(t->vs_ulp, vs_ulp(c, r[k]));
                }
            }
            for (p = 0; p < np; p++) {
                for (k = 0; (m >> p) & 1 && k < 4; k++) {
                    uint32_t c = x->param[p][k].u, gv = r[4 + 4 * p + k];

                    if ((cprm[p] >> k) & 1 && c != gv) {
                        bad++;
                        t->vs_ulp = MAX(t->vs_ulp, vs_ulp(c, gv));
                    }
                }
            }
        }
        t->vs_bad += bad != 0;
        t->vs_bad_words += bad;
    }
}

/*
 * One rasteriser per depth plane: the host GPU's interpolated depth and
 * the CPU's differ in the last place, so a LEQUAL or EQUAL pass of one
 * over depth the other wrote loses pixels. Kept for the native pipeline;
 * the other backends draw as the CPU does.
 */
static bool zown_on(const HD5770Gfx *g, const Draw *d)
{
    return g->gpu && g->gpu_mode == HD5770_GPU_ON && g->gpu->ops->native &&
           d->db && d->zfmt;
}

/* the entry of the draw's depth plane; made if it has none */
static int zown_find(HD5770Gfx *g, const Draw *d, bool make)
{
    unsigned i;

    for (i = 0; i < HD5770_ZOWN; i++) {
        if (g->zown[i].used && g->zown[i].base == d->zs.base) {
            return i;
        }
    }
    if (!make) {
        return -1;
    }
    i = g->zown_next++ % HD5770_ZOWN;
    memset(&g->zown[i], 0, sizeof(g->zown[i]));
    g->zown[i].used = true;
    g->zown[i].base = d->zs.base;
    g->zown[i].ext[0] = g->zown[i].ext[1] = INT_MAX;
    g->zown[i].ext[2] = g->zown[i].ext[3] = INT_MIN;
    return i;
}

/* the plane holds depth the host GPU wrote since its last clear */
static bool zown_gpu(HD5770Gfx *g, const Draw *d)
{
    int i = zown_on(g, d) ? zown_find(g, d, false) : -1;

    return i >= 0 && g->zown[i].gpu;
}

/* the draw has drawn, on the host GPU or the CPU */
static void zown_drawn(HD5770Gfx *g, const Draw *d, bool gpu)
{
    int i;

    if (!zown_on(g, d)) {
        return;
    }
    i = zown_find(g, d, true);
    if (d->rctl & RC_DEPTH_CLEAR) {
        /* a constant: the same from both; over all drawn since the last */
        if (d->x0 <= g->zown[i].ext[0] && d->y0 <= g->zown[i].ext[1] &&
            d->x1 >= g->zown[i].ext[2] && d->y1 >= g->zown[i].ext[3]) {
            g->zown[i].gpu = g->zown[i].cpu = false;
            g->zown[i].ext[0] = g->zown[i].ext[1] = INT_MAX;
            g->zown[i].ext[2] = g->zown[i].ext[3] = INT_MIN;
        }
        return;
    }
    if (gpu && g->zown[i].cpu) {
        g->gst.mixed_gpu++;
    }
    if ((d->dctl & DC_Z) && (d->dctl & DC_Z_WRITE)) {
        if (gpu) {
            g->zown[i].gpu = true;
        } else {
            g->zown[i].cpu = true;
        }
        g->zown[i].ext[0] = MIN(g->zown[i].ext[0], d->x0);
        g->zown[i].ext[1] = MIN(g->zown[i].ext[1], d->y0);
        g->zown[i].ext[2] = MAX(g->zown[i].ext[2], d->x1);
        g->zown[i].ext[3] = MAX(g->zown[i].ext[3], d->y1);
    }
}

/*
 * The draw on the host GPU, if the backend takes it. With gpu=verify the
 * CPU path draws it again over the same pixels, the two are compared,
 * and VRAM keeps the CPU's result.
 */
static bool gpu_draw(Draw *d, Vtx *const *vb, unsigned nv, unsigned prim,
                     HD5770Thread *t)
{
    HD5770Gfx *g = d->g;
    const HD5770IR *ir;
    HD5770GpuDraw *q;
    uint64_t area = 0, rng[11][2];
    unsigned why;
    uint8_t *pre[11] = { NULL };
    const HD5770GpuSurf *ws[11];
    unsigned i, nr = 0, nws;
    int y0 = INT_MAX, y1 = -1;
    bool ok = false, full;

    if (!g->gpu || g->gpu_mode == HD5770_GPU_OFF) {
        return false;
    }
    g->gst.offered++;
    full = g->gpu->ops->full;
    if ((d->resolve && !full) || (!d->color && !d->db) ||
        (d->db && !g->gpu->ops->depth) || (d->db && d->zexport && !full)) {
        GPU_FB(d, GFB_DB, d->resolve ? HD5770_WHY_RESOLVE :
                          !d->color && !d->db ? HD5770_WHY_NO_TARGET :
                          d->db && !g->gpu->ops->depth ? HD5770_WHY_DEPTH :
                          HD5770_WHY_ZEXPORT);
        return false;
    }
    if ((d->rt_bpe > 8 || d->mrt) && !full) {
        GPU_FB(d, GFB_RT, d->mrt ? HD5770_WHY_MRT : HD5770_WHY_RT_WIDE);
        return false;
    }
    if (d->mips && !g->gpu->ops->mipmaps) {
        GPU_FB(d, GFB_TEX, HD5770_WHY_MIPS);
        return false;
    }
    if (!d->gpu_prog && !d->resolve) {
        why = hd5770_prog_ir_why(d->ps);
        GPU_FB(d, GFB_PROG, why == HD5770_WHY_GPU ? HD5770_WHY_FEEDBACK : why);
        return false;
    }
    ir = d->resolve ? &resolve_ir : hd5770_prog_ir(d->ps);
    if (!full && (ir->cf || ir->cmask > 1 || ir->zexport)) {
        GPU_FB(d, GFB_PROG, ir->cf ? HD5770_WHY_CF_FORM : HD5770_WHY_EXPORTS);
        return false;
    }
    for (i = 0; i < ir->nfetches; i++) {
        unsigned fop = ir->fetches[i].w[0] & 0x1f;

        if (ir->fetches[i].kind == HD5770_IRF_TEX &&
            ((fop == 0x04 && !g->gpu->ops->resinfo) ||
             (fop >= 0x18 && !full))) {
            GPU_FB(d, GFB_PROG, fop == 0x04 ? HD5770_WHY_RESINFO
                                            : HD5770_WHY_FETCH_OP);
            return false;
        }
    }
    for (i = 0; i < ir->ninsts && !full; i++) {
        if (ir->insts[i].cube) {
            GPU_FB(d, GFB_PROG, HD5770_WHY_CUBE);
            return false;
        }
    }
    if (d->prov_last) {
        for (i = 0; i < ir->ninsts; i++) {
            if (ir->insts[i].interp && ir->insts[i].flat) {
                GPU_FB(d, GFB_PROG, HD5770_WHY_FLAT_LAST);
                return false;
            }
        }
    }
    q = g_new0(HD5770GpuDraw, 1);
    if (d->color && !gpu_surf(g, &d->rt, true, &q->rt)) {
        GPU_FB(d, GFB_RT, HD5770_WHY_RT_SURF);
        goto out;
    }
    if (d->mrt) {
        uint8_t *cur = (uint8_t *)d + MRT_FIRST;

        for (i = 0; i < d->nmrt; i++) {
            HD5770GpuRt *r = &q->mrt[i];
            bool sok;

            memcpy(cur, d->mrt + i * MRT_BYTES, MRT_BYTES);
            sok = gpu_surf(g, &d->rt, true, &r->surf);
            r->cb = d->mrt_cb[i];
            r->fmt = d->rt_fmt;
            r->num = d->rt_num;
            r->comps = d->rt_comps;
            memcpy(r->bits, d->rt_bits, sizeof(r->bits));
            memcpy(r->src, d->rt_src, sizeof(r->src));
            r->unorm8 = d->rt_unorm8;
            r->is_float = d->rt_float;
            r->blend_ok = d->rt_blend;
            r->wmask = d->wmask;
            r->blend = d->blend;
            q->nmrt++;
            if (!sok) {
                memcpy(cur, d->mrt, MRT_BYTES);
                GPU_FB(d, GFB_RT, HD5770_WHY_RT_SURF);
                goto out;
            }
        }
        memcpy(cur, d->mrt, MRT_BYTES);
        g_free((void *)q->rt.pages);
        q->rt = q->mrt[0].surf;
        q->rt.pages = q->rt.npages ? g_memdup2(q->rt.pages, q->rt.npages * 4)
                                   : NULL;
    }
    if (d->resolve) {
        uint64_t off;

        if (!gpu_surf(g, &d->rs, false, &q->rs) ||
            (!q->rs.space && !vram_off(g, d->rs.base, 1, &off))) {
            GPU_FB(d, GFB_DB, HD5770_WHY_RS_SURF);
            goto out;
        }
        q->resolve = true;
        q->rs_same = d->rs_fmt == d->rt_fmt && d->rs_num == d->rt_num &&
                     d->rs_swap == d->rt_swap;
        q->rs_fmt = d->rs_fmt;
        q->rs_num = d->rs_num;
        q->rs_comps = hd5770_fmt_comps(d->rs_fmt);
        for (i = 0; i < q->rs_comps && i < 4; i++) {
            q->rs_src[i] = swap_src(d->rs_swap, q->rs_comps, i);
        }
    }
    q->zexport = d->zexport;
    q->sexport = d->sexport;
    q->bool_const = g->ctx[(SQ_BOOL_CONST_0 - 0x28000) / 4];
    memcpy(q->loop_const, &g->ctx[(SQ_LOOP_CONST_0 - 0x28000) / 4],
           sizeof(q->loop_const));
    if (d->db) {
        /* depth and stencil planes in VRAM */
        if ((d->zfmt && (!gpu_surf(g, &d->zs, true, &q->zs) || q->zs.space)) ||
            (d->has_s && (!gpu_surf(g, &d->ss, true, &q->ss) ||
                          q->ss.space))) {
            GPU_FB(d, GFB_DB, HD5770_WHY_DB_SURF);
            goto out;
        }
        q->db = true;
        q->zfmt = d->zfmt;
        q->has_s = d->has_s;
        q->dctl = d->dctl;
        q->rctl = d->rctl;
        q->srm[0] = d->srm[0];
        q->srm[1] = d->srm[1];
        q->sclear = d->sclear;
        q->zclear = d->zclear;
        q->zmin = d->zmin;
        q->zmax = d->zmax;
        if (!d->color) {
            /* no colour: wmask 0, the depth (or stencil) plane stands in */
            q->rt = d->zfmt ? q->zs : q->ss;
        }
    }
    q->ir = ir;
    q->kv = (const uint32_t *)d->kv;
    q->cfile = g->alu;
    q->rt_fmt = d->rt_fmt;
    q->rt_num = d->rt_num;
    q->rt_comps = d->rt_comps;
    memcpy(q->rt_bits, d->rt_bits, sizeof(q->rt_bits));
    memcpy(q->rt_src, d->rt_src, sizeof(q->rt_src));
    q->rt_unorm8 = d->rt_unorm8;
    q->rt_float = d->rt_float;
    q->rt_blend = d->rt_blend;
    q->wmask = d->color ? d->wmask : 0;
    q->blend = d->blend;
    memcpy(q->blend_const, d->blend_const, sizeof(q->blend_const));
    q->alpha_test = d->alpha_test;
    q->alpha_func = d->alpha_func;
    q->alpha_ref = d->alpha_ref;
    q->cliprule = d->cliprule;
    memcpy(q->cr, d->cr, sizeof(q->cr));
    q->pix_center = d->pix_center;
    q->sc[0] = d->x0;
    q->sc[1] = d->y0;
    q->sc[2] = d->x1;
    q->sc[3] = d->y1;
    q->persp = d->persp;
    q->pos_gpr = d->pos_ena ? (int)d->pos_gpr : -1;
    for (i = 0; i < ir->nres; i++) {
        why = gpu_tex(d, ir, i, &q->tex[i]);
        if (why != HD5770_WHY_GPU) {
            GPU_FB(d, GFB_TEX, why);
            goto out;
        }
    }
    q->ntex = ir->nres;
    why = gpu_vbs(d, ir, q);
    if (why != HD5770_WHY_GPU) {
        GPU_FB(d, GFB_VB, why);
        goto out;
    }
    d->gnin = 0;
    for (i = 0; i < ir->ninsts; i++) {
        if (ir->insts[i].interp) {
            d->gnin = MAX(d->gnin, ir->insts[i].prm + 1u);
        }
    }
    q->ninputs = d->gnin;
    for (i = 0; i < d->gnin; i++) {
        if (d->ps_src[i] < 0 || d->ps_flat[i]) {
            q->flat |= 1u << i;
        }
    }
    if (d->gvs) {
        /* the backend runs the vertex stage: anywhere in the scissor */
        q->vs = d->gvs;
        y0 = d->y0;
        y1 = d->y1 - 1;
    } else {
        d->gtri = g_array_new(false, false, sizeof(HD5770GpuTri));
        d->gprm = g_array_new(false, false, sizeof(float));
        assemble(d, vb, nv, prim, t, gpu_tri);
        q->ntri = d->gtri->len;
        q->tri = (const HD5770GpuTri *)d->gtri->data;
        q->prm = (const float *)d->gprm->data;
    }
    for (i = 0; i < q->ntri; i++) {
        y0 = MIN(y0, q->tri[i].y0);
        y1 = MAX(y1, q->tri[i].y1);
        area += (uint64_t)(q->tri[i].x1 - q->tri[i].x0 + 1) *
                (q->tri[i].y1 - q->tri[i].y0 + 1);
    }
    /*
     * A small draw stays on the CPU; not while a backend that draws into a
     * copy of memory has draws queued: the CPU would wait for them and
     * the copy be read back first; nor a draw on a depth plane the host
     * GPU has written
     */
    if (g->gpu_mode == HD5770_GPU_ON && !d->gvs &&
        !g->dry &&
        !(g->gpu->ops->wrote && (g->npend || g->ram_pend)) &&
        (d->db ? area * ps_cost(d, ir) < HD5770_DB_MIN_PIXELS
               : area * ps_cost(d, ir) < g->gpu->ops->min_pixels &&
                 !gpu_queued(g, d, y0, y1))) {
        if (!zown_gpu(g, d)) {
            GPU_FB(d, GFB_SMALL, HD5770_WHY_SMALL);
            goto out;
        }
        g->gst.mixed_kept++;
    }
    /* the VRAM rows the draw writes: targets, depth, stencil */
    ws[0] = q->rt.space || !d->color ? NULL : &q->rt;
    ws[1] = q->db && q->zfmt ? &q->zs : NULL;
    ws[2] = q->db && q->has_s ? &q->ss : NULL;
    nws = 3;
    for (i = 1; i < q->nmrt; i++) {
        ws[nws++] = q->mrt[i].surf.space ? NULL : &q->mrt[i].surf;
    }
    for (i = 0; i < nws && y1 >= y0; i++) {
        uint64_t lo, hi;

        if (!ws[i]) {
            continue;
        }
        hd5770_surf_rows(&ws[i]->s, y0,
                         d->gvs ? MIN(y1 + 1, (int)ws[i]->s.height) : y1 + 1,
                         &lo, &hi);
        if (!ws[i]->s.fast && ws[i]->s.split) {
            hi = MIN(hi, ws[i]->s.base + split_cov(g, &ws[i]->s));
        }
        rng[nr][0] = lo - ws[i]->s.base + ws[i]->voff;
        rng[nr][1] = MIN(hi - ws[i]->s.base + ws[i]->voff, g->vram_size);
        if (rng[nr][1] > rng[nr][0]) {
            nr++;
        }
    }
    for (i = 1; i < q->nmrt; i++) {
        g->ram_pend |= q->mrt[i].surf.space;
        g->ram_wpend |= q->mrt[i].surf.space;
        if (q->mrt[i].surf.space && g->gpu_mode == HD5770_GPU_VERIFY) {
            GPU_FB(d, GFB_RT, HD5770_WHY_RT_RAM_VERIFY);
            goto out;
        }
    }
    if (q->rt.space && g->gpu_mode == HD5770_GPU_VERIFY) {
        /* verified draws are the ones in VRAM */
        GPU_FB(d, GFB_RT, HD5770_WHY_RT_RAM_VERIFY);
        goto out;
    }
    if (g->gpu_mode == HD5770_GPU_VERIFY) {
        hd5770_gfx_sync(g);
        for (i = 0; i < nr; i++) {
            pre[i] = g_memdup2(g->vram + rng[i][0], rng[i][1] - rng[i][0]);
        }
    }
    if (g->dry) {
        if (q->ntri || q->vs) {
            g->gpu->ops->prep(g->gpu->b, q);
        }
        goto out;
    }
    g->gpu->building = false;
    g->gpu->why = 0;
    if ((q->ntri || q->vs) && !g->gpu->ops->draw(g->gpu->b, q)) {
        for (i = 0; i < nr; i++) {
            g_free(pre[i]);
        }
        if (g->gpu->building && g->gpu_mode == HD5770_GPU_ON &&
            !d->gvs && !d->db &&
            area * ps_cost(d, ir) < HD5770_BUILD_CPU_PIXELS) {
            /* the desktop's draws: nothing draws them again */
            GPU_FB(d, GFB_BACKEND, HD5770_WHY_BK_BUILD_SLOW);
            goto out;
        }
        if (g->gpu->building && g->gpu_mode == HD5770_GPU_ON) {
            /*
             * left out rather than drawn by the CPU, which can take
             * seconds: the frame is incomplete until the program is built
             */
            g->gst.build_skips++;
            d->cwhy = HD5770_WHY_BUILDING;
            ok = true;
            goto out;
        }
        GPU_FB(d, GFB_BACKEND, g->gpu->why ? g->gpu->why : HD5770_WHY_BK_OTHER);
        goto out;
    }
    ok = true;
    d->cwhy = HD5770_WHY_GPU;
    if (q->ntri || q->vs) {
        zown_drawn(g, d, true);
    }
    g->gst.taken++;
    g->gst.taken_db += q->db;
    for (i = 0; i < nr; i++) {
        wpend_add(g, rng[i][0], rng[i][1]);
    }
    g->ram_pend |= q->rt.space;
    g->ram_wpend |= q->rt.space;
    if (q->resolve) {
        g->ram_pend |= q->rs.space;
        g->ram_wpend |= q->rs.space;
        if (!q->rs.space) {
            wpend_add(g, q->rs.voff, q->rs.voff + hd5770_surf_size(&q->rs.s));
        }
    }
    for (i = 0; q->vs && i < q->vs->nvb; i++) {
        /* the vertex buffers are read in place */
        const HD5770GpuSurf *b = &q->vs->vb[i].buf;

        g->ram_vpend |= b->space && !g->gpu->ram_copied;
        if (!b->space) {
            uint64_t lo = b->voff;
            uint64_t hi = MIN(b->voff + b->s.pitch, g->vram_size);
            unsigned j;

            for (j = 0; j < g->nvpend; j++) {
                if (lo <= g->vpend[j][1] && g->vpend[j][0] <= hi) {
                    break;
                }
            }
            if (j == ARRAY_SIZE(g->vpend)) {
                j--;
            }
            if (j == g->nvpend) {
                g->vpend[j][0] = lo;
                g->vpend[j][1] = hi;
                g->nvpend++;
            } else {
                g->vpend[j][0] = MIN(g->vpend[j][0], lo);
                g->vpend[j][1] = MAX(g->vpend[j][1], hi);
            }
        }
    }
    for (i = 0; i < q->ntex; i++) {
        /* guest RAM the GPU reads is not tracked by range */
        g->ram_pend |= q->tex[i].surf.space && !g->gpu->ram_copied;
        if (!q->tex[i].surf.space) {
            pend_add(g, q->tex[i].surf.voff, q->tex[i].surf.voff +
                     MAX(q->tex[i].slice_bytes[0] * q->tex[i].slices,
                         hd5770_surf_size(&q->tex[i].surf.s)));
        }
        if (q->tex[i].mip) {
            unsigned l;

            for (l = 1; l <= q->tex[i].last_lv; l++) {
                uint32_t ns = q->tex[i].dim == 2
                              ? MAX(q->tex[i].slices >> l, 1u)
                              : q->tex[i].slices;

                pend_add(g, q->tex[i].lv[l].voff, q->tex[i].lv[l].voff +
                         MAX(q->tex[i].slice_bytes[l] * ns,
                             hd5770_surf_size(&q->tex[i].lv[l].s)));
            }
        }
    }
    if (g->gpu_mode == HD5770_GPU_VERIFY && nr) {
        uint8_t *gpu[3];
        uint64_t k, nd = 0, big = 0;
        unsigned mx = 0;

        hd5770_gfx_sync(g);
        for (i = 0; i < nr; i++) {
            uint64_t n = rng[i][1] - rng[i][0];

            gpu[i] = g_memdup2(g->vram + rng[i][0], n);
            memcpy(g->vram + rng[i][0], pre[i], n);
        }
        d->ok = true;
        if (d->mrt) {
            unsigned m;

            for (m = 0; m < d->nmrt; m++) {
                memset(d->mrt + m * MRT_BYTES + offsetof(Draw, span_rt) -
                       MRT_FIRST, 0, sizeof(HD5770Span));
            }
        }
        d->span_rt.p = d->span_z.p = d->span_s.p = NULL;
        d->span_rt.lo = d->span_rt.hi = 0;
        d->span_z.lo = d->span_z.hi = d->span_s.lo = d->span_s.hi = 0;
        assemble(d, vb, nv, prim, t, raster_tri);
        for (i = 0; i < nr; i++) {
            for (k = 0; k < rng[i][1] - rng[i][0]; k++) {
                unsigned df = abs((int)gpu[i][k] -
                                  (int)g->vram[rng[i][0] + k]);

                nd += df != 0;
                big += df > 1;
                mx = MAX(mx, df);
            }
            g_free(gpu[i]);
            g_free(pre[i]);
        }
        g->gst.verified++;
        g->gst.vdiff_draws += nd != 0;
        g->gst.vdiff_bytes += nd;
        g->gst.vdiff_big += big;
        g->gst.vmax = MAX(g->gst.vmax, mx);
    }
out:
    if (d->gtri) {
        g_array_free(d->gtri, true);
        g_array_free(d->gprm, true);
        d->gtri = d->gprm = NULL;
    }
    gpu_req_free(q);
    return ok;
}

/*
 * poly_offset() for a host rasteriser: the constant and the slope factor
 * when the sides drawn have the same offset and no clamp
 */
static bool vsb_offset(Draw *d, float *zc, float *zsl)
{
    HD5770Gfx *g = d->g;
    uint32_t sc = CTX(g, PA_SU_SC_MODE_CNTL);
    unsigned en = (sc >> 11) & 3, vis = ~d->cull & 3, side;
    float scale[2], units[2], clamp, r;

    if (!(en & vis)) {
        return true;
    }
    if ((en & vis) != vis) {
        return false;
    }
    for (side = 0; side < 2; side++) {
        memcpy(&scale[side],
               &g->ctx[(PA_SU_POLY_OFFSET_FRONT_SCALE + side * 8 - 0x28000) / 4],
               4);
        memcpy(&units[side],
               &g->ctx[(PA_SU_POLY_OFFSET_FRONT_SCALE + side * 8 + 4 - 0x28000)
                       / 4], 4);
    }
    memcpy(&clamp, &g->ctx[(PA_SU_POLY_OFFSET_CLAMP - 0x28000) / 4], 4);
    if (clamp != 0.0f ||
        (vis == 3 && (scale[0] != scale[1] || units[0] != units[1]))) {
        return false;
    }
    side = vis == 2 ? 1 : 0;
    r = d->zfmt == 1 ? 0.25f : d->zfmt == 2 ? 0.5f : 1.0f;
    *zsl = scale[side] / 16.0f;
    *zc = units[side] *
          ldexpf(r, (int8_t)(CTX(g, PA_SU_POLY_OFFSET_DB_FMT_CNTL) & 0xff));
    return *zc == *zc && *zsl == *zsl;
}

/*
 * The draw with its vertex stage on the host GPU (HD5770Gpu.vs_draw):
 * all instances and runs in one request. false: not taken, the stage
 * runs here.
 */
static bool vsb_draw(Draw *d, const IdxSrc *is, IdxEntry *ic,
                     const uint32_t *idx,
                     const uint32_t *slot, const uint32_t *vid, unsigned nv,
                     unsigned nu, unsigned count, unsigned prim,
                     unsigned ninst, HD5770Thread *t)
{
    HD5770Gfx *g = d->g;
    HD5770GpuVs v, v1;
    HD5770Val *kv = NULL;
    uint32_t *em = NULL, *tidx = NULL;
    unsigned why = HD5770_VSB_N, n, i, j;
    float zc = 0.0f, zsl = 0.0f;
    bool taken = false;

    if (!g->gpu || g->gpu_mode != HD5770_GPU_ON) {
        return false;
    }
    if (!g->gpu->vs_draw) {
        why = HD5770_VSB_OFF;
    } else if (!hd5770_vs_ir((HD5770Prog *)d->vs, d->fs)) {
        why = HD5770_VSB_NO_IR;
    } else if (prim == PRIM_RECTLIST) {
        why = HD5770_VSB_PRIM;
    } else if (d->resolve || (d->db && (d->rctl & 1))) {
        why = HD5770_VSB_CLEAR;
    } else if (!d->persp) {
        why = HD5770_VSB_INTERP;
    } else if (CTX(g, PA_CL_VTE_CNTL) & (1u << 8)) {
        why = HD5770_VSB_VTE;
    } else if (!vsb_offset(d, &zc, &zsl)) {
        why = HD5770_VSB_OFFSET;
    } else if (!d->db && (uint64_t)nu * ninst < 256) {
        why = HD5770_VSB_FEW;
    } else if (count > MAX_VERTS) {
        why = HD5770_VSB_COUNT;
    } else if (!gpu_vs(d, &v, is, idx, slot, nv, prim, ninst, &kv)) {
        gpu_vs_free(&v);
        why = HD5770_VSB_BUFFER;
    }
    if (why != HD5770_VSB_N) {
        g->gst.vsb_why[why]++;
        return false;
    }
    v.zoff_const = zc;
    v.zoff_slope = zsl;
    /* the triangles of an instance, by vertex index */
    if (ic && ic->tidx) {
        tidx = ic->tidx;
        n = ic->ntidx / 3;
    } else {
        v1 = v;
        v1.ninst = 1;
        n = hd5770_vs_emits(&v1, NULL);
        em = g_new(uint32_t, MAX(n, 1u) * 5);
        tidx = g_new(uint32_t, MAX(n, 1u) * 3);
        hd5770_vs_emits(&v1, em);
        for (i = 0; i < n; i++) {
            for (j = 0; j < 3; j++) {
                tidx[i * 3 + j] = vid[slot[em[i * 5 + j]]];
            }
        }
        if (ic) {
            ic->tidx = tidx;
            ic->ntidx = n * 3;
            ic->tidx_id = ++d->g->cache->tidx_ids;
            ic->bytes += 12ull * MAX(n, 1u);
            d->g->cache->idx_bytes += 12ull * MAX(n, 1u);
        }
    }
    v.tidx = tidx;
    v.ntidx = n * 3;
    v.tidx_id = ic ? ic->tidx_id : 0;
    d->gvs = &v;
    d->gfb = GFB_NONE;
    d->cwhy = HD5770_WHY_N;
    gpu_lock(g, true);
    taken = gpu_draw(d, NULL, 0, prim, t);
    gpu_lock(g, false);
    taken |= g->dry;
    d->gvs = NULL;
    if (taken) {
        g->gst.vsb_taken++;
        d->why = HD5770_WHY_GPU;
    } else {
        g->gst.vsb_why[HD5770_VSB_DRAW]++;
        g->gst.offered--;
        if (d->gfb < GFB_NONE) {
            g->gst.fallback[d->gfb]--;
        }
    }
    gpu_vs_free(&v);
    g_free(kv);
    g_free(em);
    if (!ic) {
        g_free(tidx);
    }
    return taken;
}

/* the pixel shader samples a mipmapped texture: it needs derivatives */
static void ps_mips(Draw *d)
{
    uint8_t ids[16];
    unsigned nres, i;

    hd5770_prog_resources(d->ps, ids, &nres, ARRAY_SIZE(ids));
    for (i = 0; i < nres; i++) {
        const uint32_t *r = &d->g->ctx[(SQ_RESOURCE_0 - 0x28000) / 4 + ids[i] * 8];

        if (((r[7] >> 30) & 3) == 2 && (r[0] & 7) == 1 &&
            (r[5] & 0xf) > (r[4] >> 28)) {
            d->mips = true;
            d->ps_env.tfetch_d = tfetch_ps_d;
            return;
        }
    }
}

/* a texture of the pixel shader overlaps a target the draw writes */
static bool ps_feedback(Draw *d, bool all)
{
    uint8_t ids[16];
    unsigned nres, i, k, nt = d->mrt ? d->nmrt : 1;

    hd5770_prog_resources(d->ps, ids, &nres, ARRAY_SIZE(ids));
    for (i = 0; i < nres; i++) {
        Tex tmp;
        const Tex *t = tex_peek(d, ids[i], 0, &tmp);
        uint64_t lo, hi;

        if (!t) {
            return true;
        }
        lo = t->s.base;
        hi = lo + (all ? t->slice_bytes[0] * MAX(t->slices, 1u)
                       : hd5770_surf_size(&t->s));
        if (all && t->mip) {
            for (k = 1; k <= t->last_lv; k++) {
                lo = MIN(lo, t->lv[k].base);
                hi = MAX(hi, t->lv[k].base + t->slice_bytes[k] * tex_slices(t, k));
            }
        }
        for (k = 0; k < nt; k++) {
            const HD5770Surf *rs = d->mrt ? mrt_surf(d, k) : &d->rt;
            uint64_t rlo = rs->base, rhi = rs->base + hd5770_surf_size(rs);

            if (lo < rhi && rlo < hi) {
                return true;
            }
        }
    }
    return false;
}

/*
 * The pixel shader runs on lanes unless it samples the render target:
 * then each pixel must see the ones drawn before it. A program with an
 * IR may go to the host GPU under the same condition (gpu_prog).
 */
static void lanes_setup(Draw *d)
{
    unsigned nconst, i;
    bool grad;
    const HD5770IR *ir = hd5770_prog_ir(d->ps);

    if (d->resolve || !ir) {
        return;
    }
    if (d->g->gpu && d->g->gpu_mode != HD5770_GPU_OFF && !ps_feedback(d, true)) {
        d->gpu_prog = true;
    }
    if (!hd5770_prog_lanes(d->ps, &nconst, &grad) || ps_feedback(d, false)) {
        if (d->gpu_prog) {
            d->kv = g_new(HD5770Val, MAX(ir->nconsts, 1));
            hd5770_prog_bind(d->ps, &d->ps_env, d->kv);
        }
        return;
    }
    grad |= d->mips;
    d->kv = g_new(HD5770Val, MAX(nconst, 1));
    hd5770_prog_bind(d->ps, &d->ps_env, d->kv);
    d->nlanes = grad ? 3 : 1;
    (void)i;
}

bool hd5770_gfx_state_packet(HD5770Gfx *g, const uint32_t *b, uint32_t n)
{
    switch ((b[0] >> 8) & 0xff) {
    case 0x13:                  /* INDEX_BUFFER_SIZE */
        if (n >= 2) {
            g->index_size = b[1];
        }
        return true;
    case 0x26:                  /* INDEX_BASE */
        if (n >= 3) {
            g->index_base = (b[1] & ~1u) | ((uint64_t)(b[2] & 0xff) << 32);
        }
        return true;
    case 0x2a:                  /* INDEX_TYPE */
        if (n >= 2) {
            g->index_type = b[1];
        }
        return true;
    case 0x2f:                  /* NUM_INSTANCES */
        if (n >= 2) {
            g->num_instances = b[1];
        }
        return true;
    case 0x11:                  /* SET_BASE */
        if (n >= 4 && (b[1] & 0xf) == 1) {
            /* the draw-indirect data */
            g->ind_base = (b[2] & ~7u) | ((uint64_t)(b[3] & 0xff) << 32);
        }
        return true;
    }
    return false;
}

/*
 * The indices of a draw packet: *idx = NULL for auto-generated ones
 * (0..count-1), else count indices as the VGT fetches them (out of the
 * bounds of the buffer: 0). False for a packet not modelled.
 */
static bool draw_indices(HD5770Gfx *g, const uint32_t *b, uint32_t n,
                         uint32_t *count, uint32_t **idx, IdxSrc *src)
{
    unsigned op = (b[0] >> 8) & 0xff;
    bool i32 = g->index_type & 1;
    unsigned isz = i32 ? 4 : 2, swap = (g->index_type >> 2) & 3;
    uint64_t base, max;
    uint32_t *v, i;

    *idx = NULL;
    memset(src, 0, sizeof(*src));
    g->ind_draw = false;
    switch (op) {
    case 0x24:                  /* DRAW_INDIRECT */
    case 0x25: {                /* DRAW_INDEX_INDIRECT */
        /*
         * The arguments in memory at SET_BASE + offset: vertex count,
         * instances, first vertex (index), [base vertex,] first
         * instance; the first vertex and instance as the vertex fetch's
         * SQ_VTX_BASE_VTX_LOC and SQ_VTX_START_INST_LOC
         */
        uint32_t a[5] = { 0 };
        unsigned na = op == 0x24 ? 4 : 5;

        if (n < 3) {
            return false;
        }
        pend_check(g, g->ind_base + b[1], na * 4);
        hd5770_mem_read(&g->mem, g->ind_base + b[1], a, na * 4);
        for (i = 0; i < na; i++) {
            a[i] = le32_to_cpu(a[i]);
        }
        g->ind_draw = true;
        g->ind_ninst = a[1];
        *count = a[0];
        if (op == 0x24) {
            g->ind_vtx = a[2];
            g->ind_inst = a[3];
            return true;
        }
        g->ind_vtx = a[3];
        g->ind_inst = a[4];
        base = g->index_base + ((uint64_t)a[2] << (i32 ? 2 : 1));
        max = g->index_size > a[2] ? g->index_size - a[2] : 0;
        break;
    }
    case 0x2d:                  /* DRAW_INDEX_AUTO */
        if (n < 3) {
            return false;
        }
        *count = b[1];
        return true;
    case 0x2b:                  /* DRAW_INDEX */
        if (n < 5) {
            return false;
        }
        base = b[1] | ((uint64_t)(b[2] & 0xff) << 32);
        *count = b[3];
        max = b[3];
        break;
    case 0x27:                  /* DRAW_INDEX_2 */
        if (n < 6) {
            return false;
        }
        max = b[1];
        base = b[2] | ((uint64_t)(b[3] & 0xff) << 32);
        *count = b[4];
        break;
    case 0x29:                  /* DRAW_INDEX_OFFSET */
        if (n < 4) {
            return false;
        }
        base = g->index_base + ((uint64_t)b[1] << (i32 ? 2 : 1));
        *count = b[2];
        max = b[2];
        break;
    case 0x35:                  /* DRAW_INDEX_OFFSET_2 */
        if (n < 5) {
            return false;
        }
        max = b[1];
        base = g->index_base + ((uint64_t)b[2] << (i32 ? 2 : 1));
        *count = b[3];
        break;
    case 0x2e:                  /* DRAW_INDEX_IMMD: indices in the packet */
        if (n < 3) {
            return false;
        }
        *count = MIN(b[1], (n - 3) * (i32 ? 1 : 2));
        v = g_new(uint32_t, MAX(*count, 1));
        for (i = 0; i < *count; i++) {
            v[i] = i32 ? b[3 + i] : (b[3 + i / 2] >> (16 * (i & 1))) & 0xffff;
        }
        *idx = v;
        src->kind = 2;
        return true;
    default:
        return false;
    }
    *count = MIN(*count, MAX_VERTS);
    max = MIN(max, *count);
    src->kind = 1;
    src->base = base;
    src->max = max;
    src->isz = isz;
    src->swap = swap;
    pend_check(g, base, max * isz);
    return true;
}

/* the indices of a draw from an index buffer; *rawp: the bytes read */
static uint32_t *idx_read(HD5770Gfx *g, const IdxSrc *src, uint32_t count,
                          uint8_t **rawp)
{
    bool i32 = src->isz == 4;
    unsigned isz = src->isz, swap = src->swap;
    uint64_t base = src->base, max = src->max;
    uint32_t *v, i, k;
    uint8_t *raw;

    v = g_new0(uint32_t, MAX(count, 1));
    raw = g_malloc(max * isz + 1);
    hd5770_mem_read(&g->mem, base, raw, max * isz);
    for (i = 0; i < max; i++) {
        uint32_t x = i32 ? ldl_le_p(raw + 4 * i) : lduw_le_p(raw + 2 * i);

        if (swap == 1) {
            /* bytes swapped in each 16-bit half */
            x = i32 ? ((x & 0x00ff00ffu) << 8) | ((x >> 8) & 0x00ff00ffu)
                    : bswap16(x);
        } else if (swap == 2 && i32) {
            x = bswap32(x);
        } else if (swap == 3 && i32) {
            x = (x << 16) | (x >> 16);
        }
        v[i] = x;
    }
    for (k = max; k < count; k++) {
        v[k] = 0;
    }
    *rawp = raw;
    return v;
}

static void idx_entry_free(HD5770ShaderCache *c, IdxEntry *e)
{
    c->idx_bytes -= e->bytes;
    g_free(e->raw);
    g_free(e->idx);
    g_free(e->slot);
    g_free(e->vid);
    g_free(e->tidx);
    g_free(e);
}

static unsigned idx_bucket(const IdxSrc *src, uint32_t nv)
{
    return (uint32_t)((src->base >> 1) * 2654435761u + nv * 40503u) %
           IDX_BUCKETS;
}

/* the entry of this draw's index work if the buffer still holds its bytes */
static IdxEntry *idx_cache_get(HD5770Gfx *g, const IdxSrc *src, uint32_t nv,
                               unsigned prim)
{
    HD5770ShaderCache *c = g->cache;
    IdxEntry **pe, *e;
    uint64_t len = (uint64_t)src->max * src->isz, n = 0;
    bool restart = CTX(g, VGT_MULTI_PRIM_IB_RESET_EN) & 1, same;
    uint8_t *p;

    c->idx_tick++;
    for (pe = &c->ib[idx_bucket(src, nv)]; (e = *pe); pe = &e->next) {
        if (e->base == src->base && e->max == src->max && e->nv == nv &&
            e->isz == src->isz && e->swap == src->swap && e->prim == prim &&
            e->ioff == CTX(g, VGT_INDX_OFFSET) &&
            e->lo == CTX(g, VGT_MIN_VTX_INDX) &&
            e->hi == CTX(g, VGT_MAX_VTX_INDX) && e->restart == restart &&
            e->reset == CTX(g, VGT_MULTI_PRIM_IB_RESET_INDX)) {
            break;
        }
    }
    if (!e) {
        return NULL;
    }
    p = len ? g->mem.map(g->mem.opaque, src->base, &n, false) : NULL;
    if (!len) {
        same = true;
    } else if (p && n >= len) {
        same = !memcmp(p, e->raw, len);
    } else {
        uint8_t *tmp = g_malloc(len);

        hd5770_mem_read(&g->mem, src->base, tmp, len);
        same = !memcmp(tmp, e->raw, len);
        g_free(tmp);
    }
    if (!same) {
        *pe = e->next;
        idx_entry_free(c, e);
        return NULL;
    }
    e->used = c->idx_tick;
    return e;
}

/* keeps a draw's index work; the arrays are the entry's from here on */
static IdxEntry *idx_cache_put(HD5770Gfx *g, const IdxSrc *src, uint32_t nv,
                               unsigned prim, uint8_t *raw, uint32_t *idx,
                               uint32_t *slot, uint32_t *vid, unsigned nu)
{
    HD5770ShaderCache *c = g->cache;
    IdxEntry *e = g_new0(IdxEntry, 1);
    unsigned b;

    if (c->idx_bytes > IDX_CACHE_BYTES) {
        /* the entries not used lately go, or all of them */
        uint64_t old = c->idx_tick > IDX_CACHE_AGE
                       ? c->idx_tick - IDX_CACHE_AGE : 0;
        unsigned pass;

        for (pass = 0; pass < 2 && c->idx_bytes > IDX_CACHE_BYTES / 2;
             pass++) {
            for (b = 0; b < IDX_BUCKETS; b++) {
                IdxEntry **pe = &c->ib[b], *o;

                while ((o = *pe)) {
                    if (pass || o->used < old) {
                        *pe = o->next;
                        idx_entry_free(c, o);
                    } else {
                        pe = &o->next;
                    }
                }
            }
        }
    }
    e->base = src->base;
    e->max = src->max;
    e->nv = nv;
    e->isz = src->isz;
    e->swap = src->swap;
    e->prim = prim;
    e->ioff = CTX(g, VGT_INDX_OFFSET);
    e->lo = CTX(g, VGT_MIN_VTX_INDX);
    e->hi = CTX(g, VGT_MAX_VTX_INDX);
    e->restart = CTX(g, VGT_MULTI_PRIM_IB_RESET_EN) & 1;
    e->reset = CTX(g, VGT_MULTI_PRIM_IB_RESET_INDX);
    e->raw = raw;
    e->idx = idx;
    e->slot = slot;
    e->vid = vid;
    e->nu = nu;
    e->used = c->idx_tick;
    e->bytes = sizeof(*e) + (uint64_t)src->max * src->isz + 1 +
               12ull * MAX(nv, 1);
    c->idx_bytes += e->bytes;
    b = idx_bucket(src, nv);
    e->next = c->ib[b];
    c->ib[b] = e;
    return e;
}

/* the primitives of vertices vp[0..nv-1]: on the host GPU or the CPU */
static void draw_prims(Draw *d, Vtx *const *vp, unsigned nv, unsigned prim,
                       HD5770Thread *t)
{
    HD5770Gfx *g = d->g;
    bool taken;
    unsigned i;

    memset(t, 0, sizeof(*t));
    d->gfb = GFB_NONE;
    d->cwhy = HD5770_WHY_N;
    gpu_lock(g, true);
    taken = gpu_draw(d, vp, nv, prim, t);
    gpu_lock(g, false);
    taken |= g->dry;
    if (!taken && d->cwhy < HD5770_WHY_N && zown_gpu(g, d)) {
        /* the CPU draws over depth the host GPU wrote */
        g->gst.mixed_why[d->cwhy]++;
        d->cwhy = HD5770_WHY_MIXED_DEPTH;
    }
    if (d->cwhy < HD5770_WHY_N &&
        (d->why >= HD5770_WHY_N ||
         (d->why <= HD5770_WHY_BUILDING && d->cwhy > HD5770_WHY_BUILDING))) {
        d->why = d->cwhy;
    }
    if (!taken) {
        const HD5770Surf *ws[10] = { d->color ? &d->rt : NULL,
                                     d->db && d->zfmt ? &d->zs : NULL,
                                     d->db && d->has_s ? &d->ss : NULL };
        unsigned nws = 3;

        if (d->mrt) {
            /* the targets of an MRT draw: target 0 is d->rt now */
            for (i = 1; i < d->nmrt; i++) {
                ws[nws++] = mrt_surf(d, i);
            }
        }
        /* the CPU writes the targets: after queued draws that touch them */
        for (i = 0; i < nws; i++) {
            if (ws[i]) {
                pend_write(g, ws[i]);
            }
        }
        if (d->resolve) {
            pend_write(g, &d->rs);
        }
        int64_t t0 = get_clock();
        uint64_t px0 = d->st->pixels;

        d->wy0 = INT_MAX;
        d->wy1 = -1;
        assemble(d, vp, nv, prim, t, raster_tri);
        g->gst.fb_ns[MIN(d->gfb, GFB_NONE)] += get_clock() - t0;
        g->gst.fb_px[MIN(d->gfb, GFB_NONE)] += d->st->pixels - px0;
        if (d->cwhy < HD5770_WHY_N) {
            g->gst.why_ns[d->cwhy] += get_clock() - t0;
            g->gst.why_px[d->cwhy] += d->st->pixels - px0;
        }
        for (i = 0; i < nws && d->ok && d->wy1 >= d->wy0; i++) {
            uint64_t lo, hi;

            if (ws[i]) {
                /* the rows drawn */
                hd5770_surf_rows(ws[i], d->wy0, d->wy1 + 1, &lo, &hi);
                if (d->g->mem.dirty) {
                    d->g->mem.dirty(d->g->mem.opaque, lo, hi - lo);
                }
                cpu_wrote(g, lo, hi - lo);
            }
        }
        if (d->resolve && d->ok) {
            cpu_wrote(g, d->rs.base, hd5770_surf_size(&d->rs));
        }
        if (d->wy1 >= d->wy0) {
            zown_drawn(g, d, false);
        }
    }
}

void hd5770_gfx_draw(HD5770Gfx *g, const uint32_t *b, uint32_t n)
{
    HD5770GfxStats *st = g->st;
    unsigned prim, i, nv, nu, mode;
    unsigned reason = HD5770_SKIP_MAX;
    uint32_t count, cctl, inst, ninst, *idx = NULL, *vid = NULL, *slot = NULL;
    IdxSrc isrc;
    IdxEntry *ic = NULL;
    uint8_t *raw = NULL;
    VsCheck vc = { 0 };
    uint64_t *vm = NULL;
    int64_t t0 = get_clock();
    Draw *d;
    Vtx *vb = NULL, **vp = NULL;
    HD5770Thread *t = NULL;
    int wx, wy;

    st->draws++;
    if (!draw_indices(g, b, n, &count, &idx, &isrc)) {
        st->skip_reason[HD5770_SKIP_INDEX]++;
        st->skipped++;
        g->gst.why_draws[HD5770_WHY_SKIPPED]++;
        return;
    }
    /* the textures are set up when first used */
    d = g_malloc(sizeof(Draw));
    memset(d, 0, offsetof(Draw, tex));
    memset((uint8_t *)d + offsetof(Draw, tex_key), 0,
           sizeof(Draw) - offsetof(Draw, tex_key));
    d->why = HD5770_WHY_N;
    d->g = g;
    d->st = st;
    d->ok = true;
    prim = CFG(g, VGT_PRIMITIVE_TYPE) & 0x3f;
    cctl = CTX(g, CB_COLOR_CONTROL);
    mode = (cctl >> 4) & 7;
    if (mode != CB_DISABLE && mode != CB_NORMAL && mode != CB_RESOLVE) {
        reason = HD5770_SKIP_DISABLED;
        goto out;
    }
    if (((cctl >> 16) & 0xff) != 0xcc) {
        st->gap_misc[HD5770_GAP_ROP]++;
    }
    d->cliprule = CTX(g, PA_SC_CLIPRECT_RULE) & 0xffff;
    if (prim != PRIM_RECTLIST && prim != PRIM_QUADLIST &&
        prim != PRIM_TRILIST && prim != PRIM_TRISTRIP &&
        prim != PRIM_TRIFAN && prim != PRIM_QUADSTRIP &&
        prim != PRIM_POLYGON) {
        st->gap_prim[prim]++;
        reason = HD5770_SKIP_PRIM;
        goto out;
    }
    d->x0 = 0;
    d->y0 = 0;
    d->x1 = 16384;
    d->y1 = 16384;
    wx = (int16_t)(CTX(g, PA_SC_WINDOW_OFFSET) & 0xffff);
    wy = (int16_t)(CTX(g, PA_SC_WINDOW_OFFSET) >> 16);
    scissor(d, CTX(g, PA_SC_SCREEN_SCISSOR_TL), CTX(g, PA_SC_SCREEN_SCISSOR_BR),
            false, 0, 0);
    scissor(d, CTX(g, PA_SC_WINDOW_SCISSOR_TL), CTX(g, PA_SC_WINDOW_SCISSOR_BR),
            true, wx, wy);
    scissor(d, CTX(g, PA_SC_GENERIC_SCISSOR_TL),
            CTX(g, PA_SC_GENERIC_SCISSOR_BR), true, wx, wy);
    if (CTX(g, PA_SC_MODE_CNTL_0) & 2) {
        scissor(d, CTX(g, PA_SC_VPORT_SCISSOR_0_TL),
                CTX(g, PA_SC_VPORT_SCISSOR_0_BR), true, wx, wy);
    }
    for (i = 0; i < 4; i++) {
        uint32_t tl = CTX(g, PA_SC_CLIPRECT_RULE + 4 + 8 * i);
        uint32_t br = CTX(g, PA_SC_CLIPRECT_RULE + 8 + 8 * i);

        d->cr[i][0] = (tl & 0x7fff) + wx;
        d->cr[i][1] = ((tl >> 16) & 0x7fff) + wy;
        d->cr[i][2] = (br & 0x7fff) + wx;
        d->cr[i][3] = ((br >> 16) & 0x7fff) + wy;
    }
    db_setup(d);
    if (mode == CB_RESOLVE) {
        /* colour buffer 0 to colour buffer 1, one sample */
        if (!rt_setup(d, 0)) {
            reason = HD5770_SKIP_RT;
            goto out;
        }
        d->rs = d->rt;
        d->rs_fmt = d->rt_fmt;
        d->rs_num = d->rt_num;
        d->rs_swap = d->rt_swap;
        d->rs_mtx = d->rs_mty = UINT32_MAX;
        if (CTX(g, PA_SC_AA_CONFIG) & 7 || !rt_setup(d, 1)) {
            reason = HD5770_SKIP_RT;
            goto out;
        }
        d->resolve = true;
        d->color = true;
        d->db = false;
        st->resolves++;
    } else if (mode == CB_NORMAL && !(d->rctl & RC_COLOR_DISABLE)) {
        d->color = rt_setup(d, 0);
        if (!d->color && !d->db) {
            reason = HD5770_SKIP_RT;
            goto out;
        }
    }
    if (!d->color && !d->db) {
        if (mode == CB_DISABLE && !(d->rctl & RC_COPY)) {
            /* nothing to write, e.g. an HTILE decompress pass */
            st->drawn++;
        } else {
            reason = HD5770_SKIP_NOCOLOR;
        }
        goto out;
    }
    if (CTX(g, PA_SC_AA_CONFIG) & 7) {
        st->gap_misc[HD5770_GAP_MSAA]++;
    }
    {
        uint32_t sc = CTX(g, PA_SU_SC_MODE_CNTL);

        uint32_t cc = CTX(g, PA_CL_CLIP_CNTL);

        d->clip_near = !((cc >> 26) & 1);
        d->clip_far = !((cc >> 27) & 1);
        d->clip_dx = (cc >> 19) & 1;
        d->clip_on = !((cc >> 16) & 1) && prim != PRIM_RECTLIST &&
                     !(CTX(g, PA_CL_VTE_CNTL) & (1u << 8)) &&
                     (d->clip_near || d->clip_far);
        d->cull = prim == PRIM_RECTLIST ? 0 : sc & 3;
        d->face_cw = (sc >> 2) & 1;
        d->prov_last = (sc >> 19) & 1;
        if (sc & (1u << 13)) {
            /* offset for points and lines */
            st->gap_misc[HD5770_GAP_POLY_OFFSET]++;
        }
        if (CTX(g, SPI_PS_IN_CONTROL_1) & (1u << 8)) {
            st->gap_misc[HD5770_GAP_FACE]++;
        }
    }
    d->wmask = CTX(g, CB_TARGET_MASK) & CTX(g, CB_SHADER_MASK) & 0xf;
    if (d->resolve) {
        d->wmask = 0xf;
    }
    if (!d->wmask) {
        d->color = false;
    }
    d->blend = CTX(g, CB_BLEND0_CONTROL);
    memcpy(d->blend_const, &g->ctx[(CB_BLEND_RED - 0x28000) / 4],
           sizeof(d->blend_const));
    d->alpha_test = (CTX(g, SX_ALPHA_TEST_CONTROL) >> 3) & 1;
    d->alpha_func = CTX(g, SX_ALPHA_TEST_CONTROL) & 7;
    memcpy(&d->alpha_ref, &g->ctx[(SX_ALPHA_REF - 0x28000) / 4], 4);
    d->pix_center = (CTX(g, PA_SU_VTX_CNTL) & 1) ? 0.5f : 0.0f;
    if (d->resolve) {
        d->blend = 0;
        d->alpha_test = false;
    } else if (mode == CB_NORMAL && !(d->rctl & RC_COLOR_DISABLE)) {
        mrt_setup(d);
    }
    if (d->x0 >= d->x1 || d->y0 >= d->y1 || (!d->color && !d->db)) {
        reason = HD5770_SKIP_MAX;
        st->drawn++;
        goto out;
    }

    d->vs = get_prog(d, (uint64_t)CTX(g, SQ_PGM_START_VS) << 8,
                     HD5770_STAGE_VS);
    d->fs = get_prog(d, (uint64_t)CTX(g, SQ_PGM_START_FS) << 8,
                     HD5770_STAGE_FS);
    d->ps = get_prog(d, (uint64_t)CTX(g, SQ_PGM_START_PS) << 8,
                     HD5770_STAGE_PS);
    if (!d->vs) {
        reason = HD5770_SKIP_NO_VS;
        goto out;
    }
    if (!d->ps) {
        reason = HD5770_SKIP_NO_PS;
        goto out;
    }
    if (hd5770_prog_bad(d->vs) || hd5770_prog_bad(d->ps) ||
        (d->fs && hd5770_prog_bad(d->fs))) {
        reason = HD5770_SKIP_SHADER;
        goto out;
    }
    if (g->gpu && g->gpu_mode != HD5770_GPU_OFF &&
        !hd5770_vs_ir((HD5770Prog *)d->vs, d->fs)) {
        g->gst.vs_why[hd5770_vs_ir_why(d->vs)]++;
    }
    d->vs_env.opaque = d;
    d->vs_env.stage = HD5770_STAGE_VS;
    d->vs_env.cfile = g->alu + 1024;
    d->vs_env.kcache = kcache_vs;
    d->vs_env.vfetch = vfetch_vs;
    d->vs_env.tfetch = tfetch_vs;
    d->vs_env.fetch = d->fs;
    d->vs_env.loop_const = &g->ctx[(SQ_LOOP_CONST_0 - 0x28000) / 4 + 32];
    d->vs_env.bool_const = &g->ctx[(SQ_BOOL_CONST_0 - 0x28000) / 4 + 1];
    d->vs_env.st = st;
    d->ps_env = d->vs_env;
    d->ps_env.stage = HD5770_STAGE_PS;
    d->ps_env.cfile = g->alu;
    d->ps_env.kcache = kcache_ps;
    d->ps_env.fetch = NULL;
    d->ps_env.interp = interp;
    d->ps_env.tfetch = tfetch_ps;
    d->ps_env.vfetch = vfetch_ps;
    d->ps_env.neighbour = neighbour;
    d->ps_env.loop_const = &g->ctx[(SQ_LOOP_CONST_0 - 0x28000) / 4];
    d->ps_env.bool_const = &g->ctx[(SQ_BOOL_CONST_0 - 0x28000) / 4];
    ps_inputs(d);
    d->ps_env.interp_n = interp_n;
    ps_mips(d);
    lanes_setup(d);

    nv = MIN(count, MAX_VERTS);
    if (count > MAX_VERTS) {
        st->gap_misc[HD5770_GAP_LOOP]++;
    }
    /*
     * Vertex indices: index + VGT_INDX_OFFSET (fetched indices clamped
     * to [VGT_MIN_VTX_INDX, VGT_MAX_VTX_INDX]); each distinct vertex is
     * shaded once per instance. A fetched index equal to
     * VGT_MULTI_PRIM_IB_RESET_INDX (when enabled) starts a new primitive.
     */
    if (isrc.kind == 1) {
        ic = idx_cache_get(g, &isrc, nv, prim);
        if (!ic) {
            idx = idx_read(g, &isrc, count, &raw);
        }
    }
    if (ic) {
        idx = ic->idx;
        slot = ic->slot;
        vid = ic->vid;
        nu = ic->nu;
        goto have_idx;
    }
    slot = g_new(uint32_t, MAX(nv, 1));
    vid = g_new(uint32_t, MAX(nv, 1));
    if (!idx) {
        for (i = 0; i < nv; i++) {
            vid[i] = i + CTX(g, VGT_INDX_OFFSET);
            slot[i] = i;
        }
        nu = nv;
    } else {
        bool restart = CTX(g, VGT_MULTI_PRIM_IB_RESET_EN) & 1;
        uint32_t lo = CTX(g, VGT_MIN_VTX_INDX), hi = CTX(g, VGT_MAX_VTX_INDX);
        uint32_t xmin = UINT32_MAX, xmax = 0, *map = NULL;
        GHashTable *seen = NULL;

        /* the vertex of each index: a table over the index range if small */
        for (i = 0; i < nv; i++) {
            uint32_t x = idx[i];

            if (restart && x == CTX(g, VGT_MULTI_PRIM_IB_RESET_INDX)) {
                slot[i] = UINT32_MAX;
                continue;
            }
            x = MIN(MAX(x + CTX(g, VGT_INDX_OFFSET), lo), hi);
            slot[i] = x;
            xmin = MIN(xmin, x);
            xmax = MAX(xmax, x);
        }
        if (xmin <= xmax && (uint64_t)xmax - xmin < 4ull * nv + 65536) {
            map = g_new0(uint32_t, (size_t)(xmax - xmin) + 1);
        } else {
            seen = g_hash_table_new(NULL, NULL);
        }
        nu = 0;
        for (i = 0; i < nv; i++) {
            uint32_t x = slot[i], u;

            if (restart && idx[i] == CTX(g, VGT_MULTI_PRIM_IB_RESET_INDX)) {
                continue;
            }
            if (map) {
                u = map[x - xmin];
                if (!u) {
                    vid[nu] = x;
                    u = map[x - xmin] = ++nu;
                }
                u--;
            } else {
                gpointer pu = g_hash_table_lookup(seen, GUINT_TO_POINTER(x + 1));

                if (pu) {
                    u = GPOINTER_TO_UINT(pu) - 1;
                } else {
                    u = nu++;
                    vid[u] = x;
                    g_hash_table_insert(seen, GUINT_TO_POINTER(x + 1),
                                        GUINT_TO_POINTER(u + 1));
                }
            }
            slot[i] = u;
        }
        g_free(map);
        if (seen) {
            g_hash_table_destroy(seen);
        }
    }
    if (isrc.kind == 1) {
        ic = idx_cache_put(g, &isrc, nv, prim, raw, idx, slot, vid, nu);
    }
have_idx:
    vb = g_new(Vtx, MAX(nu, 1) + 1);
    vp = g_new(Vtx *, MAX(nv, 1));
    t = g_new0(HD5770Thread, 1);
    ninst = MAX(g->ind_draw ? g->ind_ninst : g->num_instances, 1);
    if (g->gpu && g->gpu_mode == HD5770_GPU_VERIFY &&
        g->gpu->ops->vs_records &&
        vs_check_run(d, &isrc, idx, slot, nv, prim, ninst, &vc)) {
        vm = g_new(uint64_t, MAX(nu, 1));
        g->gst.vs_draws++;
    }
    if (vsb_draw(d, &isrc, ic, idx, slot, vid, nv, nu, count, prim, ninst,
                 t)) {
        st->drawn++;
        goto out;
    }
    if (g->dry && (uint64_t)nu * ninst > 1024) {
        /* not worth the CPU's vertex stage */
        goto out;
    }
    for (inst = 0; inst < ninst && d->ok; inst++) {
        unsigned s0 = 0, k = 0;
        int64_t tv = get_clock();

        kcache_drop(d);
        for (i = 0; i < nu; i++) {
            if (!vs_run(d, vid[i], inst, &vb[i], t)) {
                reason = HD5770_SKIP_SHADER;
                goto out;
            }
            if (vm) {
                vm[i] = t->param_mask | (uint64_t)t->pos_written << 32;
            }
        }
        g->gst.vs_ns += get_clock() - tv;
        if (vc.rec) {
            vs_check(d, vc.rec, vc.np, slot, nv, inst, vb, vm);
            if (vc.ctri) {
                vs_check_tris(d, &vc, vb, vp, slot, nv, prim, t);
            }
        }
        /* runs of vertices between restart indices */
        for (i = 0; i <= nv && d->ok; i++) {
            if (i < nv && slot[i] != UINT32_MAX) {
                vp[k++] = &vb[slot[i]];
                continue;
            }
            if (k) {
                draw_prims(d, vp + s0, k - s0, prim, t);
            }
            s0 = k;
        }
    }
    if (d->ok) {
        st->drawn++;
        if (vc.ctri) {
            vs_check_cmp(d, &vc);
        }
    } else {
        reason = HD5770_SKIP_SHADER;
    }
out:
    if (reason != HD5770_SKIP_MAX) {
        st->skip_reason[reason]++;
        st->skipped++;
    }
    g->gst.why_draws[reason != HD5770_SKIP_MAX ? HD5770_WHY_SKIPPED :
                     d->why < HD5770_WHY_N ? d->why : HD5770_WHY_EMPTY]++;
    vs_check_free(&vc);
    g_free(vm);
    g_free(vb);
    g_free(vp);
    if (!ic) {
        g_free(vid);
        g_free(slot);
        g_free(idx);
    }
    g_free(t);
    g_free(d->mrt);
    g_free(d->nt);
    g_free(d->lanes);
    for (i = 0; i < d->ahead_progs; i++) {
        hd5770_prog_free(d->ahead_prog[i]);
    }
    g_free(d->kv);
    kcache_drop(d);
    g_free(d);
    st->ns += get_clock() - t0;
}

/* ---------------------------------------------------------------- */

void hd5770_gfx_border(HD5770Gfx *g, const uint32_t *regs, uint32_t off,
                       uint32_t val)
{
    unsigned st = (off - 0xa400) / 0x14, k = (off - 0xa400) % 0x14 / 4;

    if (off < 0xa400 || st > 2 || !k) {
        return;
    }
    g->border[st][regs[(0xa400 + st * 0x14) >> 2] & 31][k - 1] = val;
}

void hd5770_gfx_init(HD5770Gfx *g)
{
    unsigned i;

    for (i = 0; i < 256; i++) {
        unorm8[i] = i / (float)((1u << 8) - 1);
    }
    g->cache = g_new0(HD5770ShaderCache, 1);
}

void hd5770_gfx_fini(HD5770Gfx *g)
{
    unsigned b;

    if (!g->cache) {
        return;
    }
    for (b = 0; b < SH_BUCKETS; b++) {
        ShEntry *e = g->cache->b[b];

        while (e) {
            ShEntry *n = e->next;

            hd5770_prog_free(e->prog);
            g_free(e->text);
            g_free(e);
            e = n;
        }
    }
    for (b = 0; b < IDX_BUCKETS; b++) {
        while (g->cache->ib[b]) {
            IdxEntry *e = g->cache->ib[b];

            g->cache->ib[b] = e->next;
            idx_entry_free(g->cache, e);
        }
    }
    g_free(g->cache);
    g->cache = NULL;
}

static const char *const skip_names[HD5770_SKIP_MAX] = {
    "no pixel shader", "no vertex shader", "primitive type",
    "render target", "shader", "colour writes off", "CB mode",
    "draw opcode",
};

static const char *const misc_names[HD5770_GAP_MISC_MAX] = {
    "depth/stencil copy to colour", "more render targets", "clip rects",
    "ROP/endian/CB metadata", "blend function", "kcache mode",
    "vertex count", "relative index mode", "texture dimension",
    "predicated ALU", "program layout (clause range, ALU slots)",
    "multisampling", "vertices with w <= 0", "mipmaps (base level drawn)",
    "front face input", "polygon offset",
    "program stopped: fetch from a resource without a texture",
    "program stopped: stack overflow", "program stopped: pop of empty stack",
    "program stopped: loop nesting",
    "program stopped: step limit", "program stopped: CF instruction",
    "program stopped: derivative",
};

static void gap_list(GString *out, const char *name, const uint64_t *v,
                     unsigned n)
{
    unsigned i;

    for (i = 0; i < n; i++) {
        if (v[i]) {
            g_string_append_printf(out, "%s%s 0x%02x: %" PRIu64,
                                   out->len ? "\n" : "", name, i, v[i]);
        }
    }
}

void hd5770_gfx_gaps(HD5770Gfx *g, GString *out)
{
    HD5770GfxStats *st = g->st;
    unsigned i;

    gap_list(out, "ALU op2", st->gap_alu_op2, 256);
    gap_list(out, "ALU op3", st->gap_alu_op3, 32);
    gap_list(out, "CF inst", st->gap_cf, 256);
    gap_list(out, "TEX inst", st->gap_tex, 32);
    gap_list(out, "VTX inst", st->gap_vtx, 32);
    gap_list(out, "export type", st->gap_export, 8);
    gap_list(out, "colour format", st->gap_fmt, 64);
    gap_list(out, "texture format", st->gap_tex_fmt, 64);
    gap_list(out, "vertex format", st->gap_vtx_fmt, 64);
    gap_list(out, "array mode", st->gap_mode, 16);
    gap_list(out, "primitive", st->gap_prim, 64);
    if (st->gap_src_sel) {
        g_string_append_printf(out, "%sALU operand select: %" PRIu64,
                               out->len ? "\n" : "", st->gap_src_sel);
    }
    for (i = 0; i < HD5770_GAP_MISC_MAX; i++) {
        if (st->gap_misc[i]) {
            g_string_append_printf(out, "%s%s: %" PRIu64, out->len ? "\n" : "",
                                   misc_names[i], st->gap_misc[i]);
        }
    }
    for (i = 0; i < HD5770_SKIP_MAX; i++) {
        if (st->skip_reason[i]) {
            g_string_append_printf(out, "%sdraws skipped, %s: %" PRIu64,
                                   out->len ? "\n" : "", skip_names[i],
                                   st->skip_reason[i]);
        }
    }
}

void hd5770_gfx_stats(HD5770Gfx *g, GString *out)
{
    HD5770GfxStats *st = g->st;
    uint32_t i, k;

    g_string_append_printf(out,
        "3D: draws %" PRIu64 " drawn %" PRIu64 " skipped %" PRIu64
        " prims %" PRIu64 " verts %" PRIu64 " pixels %" PRIu64 "\n"
        "3D: blended %" PRIu64 " tex fetches %" PRIu64 " vtx fetches %" PRIu64
        " kills %" PRIu64 " ms in draws %" PRIu64 "\n"
        "3D: culled %" PRIu64 " depth/stencil tests %" PRIu64 " failed %"
        PRIu64 " resolves %" PRIu64 " mipmapped fetches %" PRIu64
        " clipped %" PRIu64 "\n",
        st->draws, st->drawn, st->skipped, st->prims, st->verts, st->pixels,
        st->blend_on, st->tex_fetch, st->vtx_fetch, st->kills,
        st->ns / 1000000, st->culled, st->db_tests, st->db_fails,
        st->resolves, st->lod_fetches, st->clipped);
    g_string_append(out, "colour formats drawn:");
    for (i = 0; i < 64; i++) {
        if (st->rt_fmt[i]) {
            g_string_append_printf(out, " 0x%02x:%" PRIu64, i, st->rt_fmt[i]);
        }
    }
    g_string_append_printf(out, "\nshaders %u (+%" PRIu64 " not listed):\n",
                           st->n_sh, st->sh_other);
    for (i = 0; i < st->n_sh; i++) {
        HD5770ShaderInfo *si = &st->sh[i];

        g_string_append_printf(out, " %s 0x%010" PRIx64 " hash %08x %u dw"
                               " uses %" PRIu64 " tex %u vtx %u%s cf",
                               si->stage == 0 ? "VS" :
                               si->stage == 1 ? "PS" : "FS",
                               si->addr, si->hash, si->dwords, si->uses,
                               si->tex_ops, si->vtx_ops,
                               si->bad ? " NOT-MODELLED" : "");
        for (k = 0; k < si->n_cf_ops; k++) {
            g_string_append_printf(out, " %02x", si->cf_ops[k]);
        }
        g_string_append(out, " alu");
        for (k = 0; k < si->n_alu_ops; k++) {
            g_string_append_printf(out, " %03x", si->alu_ops[k]);
        }
        g_string_append_c(out, '\n');
    }
}

/*
 * Draws by where they ran, one line per reason: "why NAME: draws D
 * cpu-ms M cpu-kpixels P" (M and P also from draws partly on the GPU),
 * then "vs-why NAME: draws D" for vertex shaders without host IR.
 */
void hd5770_gfx_why_stats(HD5770Gfx *g, GString *out)
{
    const HD5770GpuStats *t = &g->gst;
    unsigned i;

    for (i = 0; i < HD5770_WHY_N; i++) {
        if (t->why_draws[i] || t->why_ns[i] || t->why_px[i]) {
            g_string_append_printf(out, "why %s: draws %" PRIu64 " cpu-ms %"
                                   PRIu64 " cpu-kpixels %" PRIu64 "\n",
                                   hd5770_why_name(i), t->why_draws[i],
                                   t->why_ns[i] / 1000000,
                                   t->why_px[i] / 1000);
        }
    }
    for (i = 0; i < HD5770_WHY_N; i++) {
        if (t->mixed_why[i]) {
            g_string_append_printf(out, "mixed depth, %s: draws %" PRIu64 "\n",
                                   hd5770_why_name(i), t->mixed_why[i]);
        }
    }
    if (t->mixed_kept || t->mixed_gpu) {
        g_string_append_printf(out, "mixed depth: small draws kept on the host"
                               " GPU %" PRIu64 ", host GPU draws over depth"
                               " the CPU drew %" PRIu64 "\n", t->mixed_kept,
                               t->mixed_gpu);
    }
    for (i = 0; i < HD5770_WHY_N; i++) {
        if (t->vs_why[i]) {
            g_string_append_printf(out, "vs-why %s: draws %" PRIu64 "\n",
                                   hd5770_why_name(i), t->vs_why[i]);
        }
    }
}

void hd5770_gfx_gpu_stats(HD5770Gfx *g, GString *out)
{
    HD5770GpuStats *t = &g->gst;

    if (!g->gpu) {
        g_string_append(out, "gpu: off\n");
        return;
    }
    g_string_append_printf(out,
        "gpu: %s offered %" PRIu64 " taken %" PRIu64 " (depth/stencil %" PRIu64
        ") fallback: program %"
        PRIu64 " small %" PRIu64 " target %" PRIu64 " texture %" PRIu64
        " vertex buffer %" PRIu64 " backend %" PRIu64 " depth/resolve %"
        PRIu64 " left out while building %" PRIu64 "\n"
        "gpu: syncs %" PRIu64 " (on access %" PRIu64 ") verified %" PRIu64
        " differing draws %" PRIu64 " bytes %" PRIu64 " (>1: %" PRIu64
        ") max %" PRIu64 "\n",
        g->gpu_mode == HD5770_GPU_VERIFY ? "verify" : "on", t->offered,
        t->taken, t->taken_db, t->fallback[GFB_PROG], t->fallback[GFB_SMALL],
        t->fallback[GFB_RT], t->fallback[GFB_TEX], t->fallback[GFB_VB],
        t->fallback[GFB_BACKEND], t->fallback[GFB_DB], t->build_skips, t->syncs, t->touch_syncs, t->verified,
        t->vdiff_draws, t->vdiff_bytes, t->vdiff_big, t->vmax);
    if (t->vs_draws) {
        g_string_append_printf(out,
            "gpu: vertex records checked: draws %" PRIu64 " records %" PRIu64
            " differing %" PRIu64 " (words %" PRIu64 ") max ulp %" PRIu64 "\n",
            t->vs_draws, t->vs_recs, t->vs_bad, t->vs_bad_words, t->vs_ulp);
        g_string_append_printf(out,
            "gpu: triangles checked: draws %" PRIu64 " triangles %" PRIu64
            " differing %" PRIu64 " (max ulp %" PRIu64 ") draws with another"
            " count %" PRIu64 "\n",
            t->vs_tri_draws, t->vs_tris, t->vs_tri_bad, t->vs_tri_ulp,
            t->vs_tri_count);
    }
    g_string_append_printf(out,
        "gpu: on the CPU ms/kpixels: program %" PRIu64 "/%" PRIu64
        " small %" PRIu64 "/%" PRIu64 " target %" PRIu64 "/%" PRIu64
        " texture %" PRIu64 "/%" PRIu64 " vertex buffer %" PRIu64 "/%" PRIu64
        " backend %" PRIu64 "/%" PRIu64 " depth/resolve %" PRIu64 "/%" PRIu64
        "\n",
        t->fb_ns[GFB_PROG] / 1000000, t->fb_px[GFB_PROG] / 1000,
        t->fb_ns[GFB_SMALL] / 1000000, t->fb_px[GFB_SMALL] / 1000,
        t->fb_ns[GFB_RT] / 1000000, t->fb_px[GFB_RT] / 1000,
        t->fb_ns[GFB_TEX] / 1000000, t->fb_px[GFB_TEX] / 1000,
        t->fb_ns[GFB_VB] / 1000000, t->fb_px[GFB_VB] / 1000,
        t->fb_ns[GFB_BACKEND] / 1000000, t->fb_px[GFB_BACKEND] / 1000,
        t->fb_ns[GFB_DB] / 1000000, t->fb_px[GFB_DB] / 1000);
    g_string_append_printf(out, "gpu: vertex stage: on the host GPU %" PRIu64
                           " draws; on the CPU ms %" PRIu64 ", draws:",
                           t->vsb_taken, t->vs_ns / 1000000);
    for (unsigned i = 0; i < HD5770_VSB_N; i++) {
        static const char *const nm[HD5770_VSB_N] = {
            "backend takes none", "no IR", "rectangle list", "clear",
            "inputs not perspective", "positions not in clip space",
            "polygon offset", "few vertices without depth",
            "vertex count", "vertex buffer", "draw not taken",
        };

        if (t->vsb_why[i]) {
            g_string_append_printf(out, " %s %" PRIu64 ",", nm[i],
                                   t->vsb_why[i]);
        }
    }
    g_string_append(out, "\n");
    hd5770_gfx_why_stats(g, out);
    g->gpu->ops->stats(g->gpu->b, out);
}

/* the programs seen so far, disassembled */
void hd5770_gfx_dump_shaders(HD5770Gfx *g, GString *out)
{
    unsigned b;

    for (b = 0; g->cache && b < SH_BUCKETS; b++) {
        ShEntry *e;

        for (e = g->cache->b[b]; e; e = e->next) {
            g_string_append_printf(out, "program 0x%010" PRIx64 " hash %08x"
                                   " %u dwords%s\n", e->addr, e->hash,
                                   e->ndw, hd5770_prog_bad(e->prog)
                                   ? " (not modelled)" : "");
            hd5770_prog_dump(e->prog, out);
        }
    }
}
