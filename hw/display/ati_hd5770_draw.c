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
#define VGT_INDX_OFFSET                 0x28408
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

#define RES_VS                          176
#define RES_FS                          992
#define SAMPLER_VS                      18
#define MAX_VERTS                       8192
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
    HD5770Prog *prog;
    struct ShEntry *next;
} ShEntry;

#define SH_BUCKETS 256

struct HD5770ShaderCache {
    ShEntry *b[SH_BUCKETS];
};

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

/* a vertex after the vertex shader and the viewport */
typedef struct Vtx {
    float x, y, z, w;           /* window coordinates, w = 1/clip w */
    HD5770Val param[HD5770_PARAMS][4];
} Vtx;

/* the state one draw needs, gathered once */
typedef struct Draw {
    HD5770Gfx *g;
    HD5770GfxStats *st;
    HD5770Span span_tex, span_rt, span_vb, span_cb;
    /* render target */
    HD5770Surf rt;
    unsigned rt_fmt, rt_num, rt_swap, rt_bpe, rt_comps;
    bool rt_blend, rt_float;
    unsigned rt_bits[4], rt_src[4];
    bool rt_unorm8;             /* 8_8_8_8 UNORM */
    uint32_t rt_mtx, rt_mty;    /* the micro tile last addressed */
    uint64_t rt_mbase;
    uint32_t wmask;
    bool color;                 /* colour buffer 0 is written */
    uint32_t blend;
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
    HD5770Val *kv;
    int lx[HD5770_LANES], ly[HD5770_LANES];
    /* the triangles of a draw for the host GPU */
    GArray *gtri, *gprm;
    unsigned gnin;
    /* textures set up so far in this draw, by resource and sampler */
    Tex tex[TEX_CACHE];
    uint32_t tex_key[TEX_CACHE];
    unsigned n_tex;
} Draw;

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

static void gpu_lock(HD5770Gfx *g, bool on)
{
    if (g->lock) {
        g->lock(g, on);
    }
}

void hd5770_gfx_sync(HD5770Gfx *g)
{
    if (!g->gpu) {
        return;
    }
    gpu_lock(g, true);
    if (g->npend || g->ram_pend) {
        g->gpu->ops->sync(g->gpu->b);
        g->npend = 0;
        g->ram_pend = false;
        g->gst.syncs++;
    }
    gpu_lock(g, false);
}

void hd5770_gfx_touch(HD5770Gfx *g, uint64_t off, uint64_t len)
{
    unsigned i;

    if (!g->gpu) {
        return;
    }
    gpu_lock(g, true);
    for (i = 0; i < g->npend; i++) {
        if (off < g->pend[i][1] && g->pend[i][0] < off + len) {
            g->gst.touch_syncs++;
            hd5770_gfx_sync(g);
            break;
        }
    }
    gpu_lock(g, false);
}

/* the CPU reads memory at mc: queued draws first, if they may write it */
static void pend_check(HD5770Gfx *g, uint64_t mc, uint64_t len)
{
    uint64_t off;

    if (g->npend && vram_off(g, mc, len, &off)) {
        hd5770_gfx_touch(g, off, len);
    } else if (g->ram_pend && !vram_off(g, mc, len, &off)) {
        hd5770_gfx_sync(g);
    }
}

/* a surface for the host GPU: in VRAM, or page by page in guest RAM */
static bool gpu_surf(HD5770Gfx *g, const HD5770Surf *s, bool write,
                     HD5770GpuSurf *o)
{
    uint64_t size = hd5770_surf_size(s), first = s->base & ~0xfffull, n, i;
    uint32_t *pages;

    memset(o, 0, sizeof(*o));
    o->s = *s;
    if (!s->fast) {
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
    ShEntry *e;
    unsigned b;

    if (!addr) {
        return NULL;
    }
    buf = g_new(uint32_t, MAX_PROG_DWORDS);
    hd5770_mem_read(&g->mem, addr, buf, 4096);
    ndw = hd5770_prog_size(buf, 1024);
    if (ndw > 1024 - 4) {
        hd5770_mem_read(&g->mem, addr, buf, MAX_PROG_DWORDS * 4);
        ndw = hd5770_prog_size(buf, MAX_PROG_DWORDS);
    }
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
    if (!e) {
        e = g_new0(ShEntry, 1);
        e->addr = addr;
        e->hash = hash;
        e->ndw = ndw;
        e->prog = hd5770_prog_new(buf, ndw, stage, d->st);
        e->next = g->cache->b[b];
        g->cache->b[b] = e;
    }
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

static bool tex_setup(Draw *d, unsigned id, unsigned sid, Tex *t)
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
    if (t->dim != 1 && t->dim != 0) {
        d->st->gap_misc[HD5770_GAP_TEX_DIM]++;
        return false;
    }
    if (t->dim == 1 && (r[5] & 0xf) > (r[4] >> 28) && ((sm[0] >> 15) & 3)) {
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
    if (t->dim == 0) {
        t->h = 1;
    }
    t->comps = hd5770_fmt_comps(t->fmt);
    t->is_float = hd5770_fmt_is_float(t->fmt);
    t->unorm8 = t->fmt == HD5770_FMT_8_8_8_8 && t->num == HD5770_NUM_NORM &&
                !t->sgn;
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

/* texel (x, y) of level lv as raw components, x and y inside it */
static void texel(Draw *d, Tex *t, unsigned lv, int x, int y,
                  HD5770Val out[4])
{
    const HD5770Surf *ls = lv ? &t->lv[lv] : &t->s;
    uint64_t off = lv ? hd5770_surf_offset_fast(ls, x, y)
                      : surf_off(&t->s, &t->mtx, &t->mty, &t->mbase, x, y);
    uint8_t *p = hd5770_span_ptr(&d->g->mem, &d->span_tex, ls->base + off,
                                 t->s.bpe, false);
    uint32_t c[4];
    unsigned bits[4], k, n;
    HD5770Val v[6];

    if (!p) {
        out[0].f = out[1].f = out[2].f = out[3].f = 0.0f;
        return;
    }
    if (t->unorm8) {
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
    v[4].f = 0.0f;
    v[5].f = 1.0f;
    for (k = 0; k < 4; k++) {
        unsigned s = t->dst_sel[k];

        out[k] = s < 6 ? v[s] : v[4];
    }
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
 * are 1D tiled.
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
        t->lw[l] = w;
        t->lh[l] = h;
        off += (uint64_t)pw * ph * bpe;
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
    pend_check(d->g, d->tex[i].s.base, hd5770_surf_size(&d->tex[i].s));
    if (d->tex[i].mip) {
        unsigned l;

        for (l = 1; l <= d->tex[i].last_lv; l++) {
            pend_check(d->g, d->tex[i].lv[l].base,
                       hd5770_surf_size(&d->tex[i].lv[l]));
        }
    }
    d->tex_key[i] = key;
    d->n_tex = MAX(d->n_tex, i + 1);
    return &d->tex[i];
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
static void tfetch_stage(Draw *d, const uint32_t *w, const HD5770Val c[4],
                         HD5770Val v[4], unsigned base_r, unsigned base_s,
                         const float *dv)
{
    unsigned op = w[0] & 0x1f, id, sid, k;
    float u, vv;
    Tex *t;
    int ox = ((int32_t)(w[2] << 27)) >> 27, oy = ((int32_t)(w[2] << 22)) >> 27;
    bool nx = (w[1] >> 28) & 1, ny = (w[1] >> 29) & 1;

    id = ((w[0] >> 8) & 0xff) + base_r;
    sid = ((w[2] >> 15) & 0x1f) + base_s;
    v[0].f = v[1].f = v[2].f = 0.0f;
    v[3].f = 0.0f;
    t = tex_get(d, id, sid);
    if (!t) {
        d->ok = false;
        return;
    }
    d->st->tex_fetch++;
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
    base = (uint64_t)CTX(d->g, SQ_ALU_CONST_CACHE_VS_0 + 4 * buf)
                    << 8;

    pend_check(d->g, base + idx * 16ull, 16);
    hd5770_mem_read(&d->g->mem, base + idx * 16ull, v, 16);
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
    for (i = 1; i < 8 && !cb; i++) {
        if ((CTX(g, base + i * CB_COLOR_STRIDE + 0x10) >> 2) & 0x3f &&
            (CTX(g, CB_TARGET_MASK) >> (4 * i)) & 0xf) {
            d->st->gap_misc[HD5770_GAP_MRT]++;
            break;
        }
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
    return d->tv[0]->z * b0 + d->tv[1]->z * b1 + d->tv[2]->z * b2;
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
        if (d->db && !db_test(d, d->lx[x], d->ly[x],
                              frag_z(d, l->b0[x], l->b1[x], l->b2[x]), -1)) {
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

static bool vs_run(Draw *d, uint32_t index, Vtx *v, HD5770Thread *t)
{
    HD5770Gfx *g = d->g;
    const float *vp = (const float *)&g->ctx[(PA_CL_VPORT_XSCALE_0 - 0x28000)
                                             / 4];
    uint32_t vte = CTX(g, PA_CL_VTE_CNTL);
    float x, y, z, w;
    unsigned k;

    memset(t->r, 0, sizeof(t->r[0]) * 4);
    t->r[0][0].u = index;
    t->param_mask = 0;
    t->pos_written = false;
    if (!hd5770_prog_run(d->vs, &d->vs_env, t) || !d->ok) {
        return false;
    }
    x = t->pos[0].f;
    y = t->pos[1].f;
    z = t->pos[2].f;
    w = t->pos[3].f;
    if (!(vte & (1u << 8))) {
        if (!(w > 0.0f)) {
            /* behind the eye: not clipped */
            d->st->gap_misc[HD5770_GAP_CLIP]++;
        }
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
static void emit(Draw *d, const Vtx *a, const Vtx *b, const Vtx *c,
                 const Vtx *pv, HD5770Thread *t, TriFn *tri)
{
    float area = edge(a->x, a->y, b->x, b->y, c->x, c->y);
    bool cw = FACE_CW_IS_AREA_POS ? area > 0 : area < 0;

    d->front = d->face_cw ? cw : !cw;
    if (d->cull & (d->front ? 1 : 2)) {
        d->st->culled++;
        return;
    }
    d->tvp = d->prov_last ? pv : a;
    tri(d, a, b, c, t);
}

/* the primitives of a draw as triangles, in order */
static void assemble(Draw *d, Vtx *vb, unsigned nv, unsigned prim,
                     HD5770Thread *t, TriFn *tri)
{
    HD5770GfxStats *st = d->st;
    unsigned i;

    switch (prim) {
    case PRIM_RECTLIST:
        /* three corners, v1 adjacent to both others: v3 = v0 + v2 - v1 */
        for (i = 0; i + 3 <= nv && d->ok; i += 3) {
            Vtx v3 = vb[i + 2];
            unsigned p, k;

            v3.x = vb[i].x + vb[i + 2].x - vb[i + 1].x;
            v3.y = vb[i].y + vb[i + 2].y - vb[i + 1].y;
            v3.z = vb[i].z + vb[i + 2].z - vb[i + 1].z;
            for (p = 0; p < HD5770_PARAMS; p++) {
                for (k = 0; k < 4; k++) {
                    v3.param[p][k].f = vb[i].param[p][k].f +
                                       vb[i + 2].param[p][k].f -
                                       vb[i + 1].param[p][k].f;
                }
            }
            emit(d, &vb[i], &vb[i + 1], &vb[i + 2], &vb[i + 2], t, tri);
            emit(d, &vb[i], &vb[i + 2], &v3, &vb[i + 2], t, tri);
            st->prims++;
        }
        break;
    case PRIM_QUADLIST:
        for (i = 0; i + 4 <= nv && d->ok; i += 4) {
            emit(d, &vb[i], &vb[i + 1], &vb[i + 2], &vb[i + 3], t, tri);
            emit(d, &vb[i], &vb[i + 2], &vb[i + 3], &vb[i + 3], t, tri);
            st->prims++;
        }
        break;
    case PRIM_QUADSTRIP:
        /* quad i: vertices 2i, 2i + 1, 2i + 3, 2i + 2 */
        for (i = 0; i + 4 <= nv && d->ok; i += 2) {
            emit(d, &vb[i], &vb[i + 1], &vb[i + 3], &vb[i + 3], t, tri);
            emit(d, &vb[i], &vb[i + 3], &vb[i + 2], &vb[i + 3], t, tri);
            st->prims++;
        }
        break;
    case PRIM_TRILIST:
        for (i = 0; i + 3 <= nv && d->ok; i += 3) {
            emit(d, &vb[i], &vb[i + 1], &vb[i + 2], &vb[i + 2], t, tri);
            st->prims++;
        }
        break;
    case PRIM_TRISTRIP:
        for (i = 2; i < nv && d->ok; i++) {
            if (i & 1) {
                emit(d, &vb[i - 1], &vb[i - 2], &vb[i], &vb[i], t, tri);
            } else {
                emit(d, &vb[i - 2], &vb[i - 1], &vb[i], &vb[i], t, tri);
            }
            st->prims++;
        }
        break;
    case PRIM_TRIFAN:
    case PRIM_POLYGON:
        for (i = 2; i < nv && d->ok; i++) {
            emit(d, &vb[0], &vb[i - 1], &vb[i], &vb[i], t, tri);
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
        gt.z[q] = v[q]->z;
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
};

/* texture unit u of a request: the CPU path's Tex, in VRAM */
static bool gpu_tex(Draw *d, const HD5770IR *ir, unsigned u,
                    HD5770GpuTex *gt)
{
    unsigned id = ir->res[u], sid = 0, i, k;
    bool seen = false;
    Tex t;

    for (i = 0; i < ir->nfetches; i++) {
        const HD5770IRFetch *f = &ir->fetches[i];
        unsigned s = (f->w[2] >> 15) & 0x1f;

        if (f->kind == HD5770_IRF_TEX && f->res == id) {
            if (seen && s != sid) {
                return false;
            }
            sid = s;
            seen = true;
        }
    }
    memset(gt, 0, sizeof(*gt));
    if (!tex_setup(d, id, sid, &t) || !gpu_surf(d->g, &t.s, false, &gt->surf)) {
        return false;
    }
    gt->fmt = t.fmt;
    gt->num = t.num;
    gt->w = t.w;
    gt->h = t.h;
    gt->dim = t.dim;
    gt->sgn = t.sgn;
    gt->comps = t.comps;
    gt->is_float = t.is_float;
    gt->unorm8 = t.unorm8;
    gt->base_lv = t.base_lv;
    gt->levels = t.levels;
    gt->depth = t.depth;
    memcpy(gt->samp, t.samp, sizeof(gt->samp));
    memcpy(gt->border, t.border, sizeof(gt->border));
    {
        uint8_t zero[16] = { 0 };
        uint32_t c[4];

        hd5770_fmt_unpack(t.fmt, zero, c, gt->bits);
    }
    for (k = 0; k < 4; k++) {
        gt->dst_sel[k] = t.dst_sel[k];
    }
    if (t.mip) {
        HD5770GpuSurf *lv = g_new0(HD5770GpuSurf, t.last_lv + 1);

        gt->lv = lv;
        lv[0] = gt->surf;
        for (k = 1; k <= t.last_lv; k++) {
            if (!gpu_surf(d->g, &t.lv[k], false, &lv[k]) || lv[k].space) {
                return false;
            }
            gt->lw[k] = t.lw[k];
            gt->lh[k] = t.lh[k];
        }
        gt->lw[0] = t.w;
        gt->lh[0] = t.h;
        gt->mip = true;
        gt->last_lv = t.last_lv;
        gt->mip_filter = t.mip_filter;
        gt->min_lod = t.min_lod;
        gt->max_lod = t.max_lod;
        gt->lod_bias = t.lod_bias;
    }
    return true;
}

/* the vertex buffers the pixel shader fetches from, copied */
static bool gpu_vbs(Draw *d, const HD5770IR *ir, HD5770GpuDraw *q)
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
            return false;
        }
        for (k = 0; k < q->nvb && q->vb[k].id != bid; k++) {
            /* search */
        }
        if (k < q->nvb) {
            continue;
        }
        if (q->nvb == HD5770_GPU_VB) {
            return false;
        }
        base = r[0] | ((uint64_t)(r[2] & 0xff) << 32);
        size = r[1] + 1;
        if (size > 4 * 1024 * 1024) {
            return false;
        }
        pend_check(g, base, size);
        q->vb[k].id = bid;
        memcpy(q->vb[k].res, r, sizeof(q->vb[k].res));
        q->vb[k].len = size;
        q->vb[k].data = g_malloc(size);
        hd5770_mem_read(&g->mem, base, (uint8_t *)q->vb[k].data, size);
        q->nvb++;
    }
    return true;
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
    g_free((void *)q->rt.pages);
    g_free((void *)q->zs.pages);
    g_free((void *)q->ss.pages);
    g_free(q);
}

/*
 * Draws with depth or stencil cover at least this many pixels to go to
 * the host GPU: smaller ones (pieces of a model, a few hundred pixels
 * each) cost more in round trips and barriers than on the CPU.
 */
#define HD5770_DB_MIN_PIXELS    16384

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
 * The draw on the host GPU, if the backend takes it. With gpu=verify the
 * CPU path draws it again over the same pixels, the two are compared,
 * and VRAM keeps the CPU's result.
 */
static bool gpu_draw(Draw *d, Vtx *vb, unsigned nv, unsigned prim,
                     HD5770Thread *t)
{
    HD5770Gfx *g = d->g;
    const HD5770IR *ir;
    HD5770GpuDraw *q;
    uint64_t area = 0, rng[3][2];
    uint8_t *pre[3] = { NULL };
    const HD5770GpuSurf *ws[3];
    unsigned i, nr = 0;
    int y0 = INT_MAX, y1 = -1;
    bool ok = false;

    if (!g->gpu || g->gpu_mode == HD5770_GPU_OFF) {
        return false;
    }
    g->gst.offered++;
    if (d->resolve || (!d->color && !d->db) ||
        (d->db && !g->gpu->ops->depth) || (d->db && d->zexport)) {
        g->gst.fallback[GFB_DB]++;
        return false;
    }
    if (d->rt_bpe > 8) {
        g->gst.fallback[GFB_RT]++;
        return false;
    }
    if (d->mips && !g->gpu->ops->mipmaps) {
        g->gst.fallback[GFB_TEX]++;
        return false;
    }
    if (!d->lanes) {
        g->gst.fallback[GFB_PROG]++;
        return false;
    }
    ir = hd5770_prog_ir(d->ps);
    for (i = 0; i < ir->nfetches && !g->gpu->ops->resinfo; i++) {
        if (ir->fetches[i].kind == HD5770_IRF_TEX &&
            (ir->fetches[i].w[0] & 0x1f) == 0x04) {
            g->gst.fallback[GFB_PROG]++;
            return false;
        }
    }
    if (d->prov_last) {
        for (i = 0; i < ir->ninsts; i++) {
            if (ir->insts[i].interp && ir->insts[i].flat) {
                g->gst.fallback[GFB_PROG]++;
                return false;
            }
        }
    }
    q = g_new0(HD5770GpuDraw, 1);
    if (d->color && !gpu_surf(g, &d->rt, true, &q->rt)) {
        g->gst.fallback[GFB_RT]++;
        goto out;
    }
    if (d->db) {
        /* depth and stencil planes in VRAM */
        if ((d->zfmt && (!gpu_surf(g, &d->zs, true, &q->zs) || q->zs.space)) ||
            (d->has_s && (!gpu_surf(g, &d->ss, true, &q->ss) ||
                          q->ss.space))) {
            g->gst.fallback[GFB_DB]++;
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
    q->persp = d->persp;
    q->pos_gpr = d->pos_ena ? (int)d->pos_gpr : -1;
    for (i = 0; i < ir->nres; i++) {
        if (!gpu_tex(d, ir, i, &q->tex[i])) {
            g->gst.fallback[GFB_TEX]++;
            goto out;
        }
    }
    q->ntex = ir->nres;
    if (!gpu_vbs(d, ir, q)) {
        g->gst.fallback[GFB_VB]++;
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
    d->gtri = g_array_new(false, false, sizeof(HD5770GpuTri));
    d->gprm = g_array_new(false, false, sizeof(float));
    assemble(d, vb, nv, prim, t, gpu_tri);
    q->ntri = d->gtri->len;
    q->tri = (const HD5770GpuTri *)d->gtri->data;
    q->prm = (const float *)d->gprm->data;
    for (i = 0; i < q->ntri; i++) {
        y0 = MIN(y0, q->tri[i].y0);
        y1 = MAX(y1, q->tri[i].y1);
        area += (uint64_t)(q->tri[i].x1 - q->tri[i].x0 + 1) *
                (q->tri[i].y1 - q->tri[i].y0 + 1);
    }
    if (g->gpu_mode == HD5770_GPU_ON &&
        (d->db ? area < HD5770_DB_MIN_PIXELS
               : area < g->gpu->ops->min_pixels && !gpu_queued(g, d, y0, y1))) {
        g->gst.fallback[GFB_SMALL]++;
        goto out;
    }
    /* the VRAM rows the draw writes: target, depth, stencil */
    ws[0] = q->rt.space || !d->color ? NULL : &q->rt;
    ws[1] = q->db && q->zfmt ? &q->zs : NULL;
    ws[2] = q->db && q->has_s ? &q->ss : NULL;
    for (i = 0; i < 3 && y1 >= y0; i++) {
        uint64_t lo, hi;

        if (!ws[i]) {
            continue;
        }
        hd5770_surf_rows(&ws[i]->s, y0, y1 + 1, &lo, &hi);
        rng[nr][0] = lo - ws[i]->s.base + ws[i]->voff;
        rng[nr][1] = MIN(hi - ws[i]->s.base + ws[i]->voff, g->vram_size);
        if (rng[nr][1] > rng[nr][0]) {
            nr++;
        }
    }
    if (q->rt.space && g->gpu_mode == HD5770_GPU_VERIFY) {
        /* verified draws are the ones in VRAM */
        g->gst.fallback[GFB_RT]++;
        goto out;
    }
    if (g->gpu_mode == HD5770_GPU_VERIFY) {
        hd5770_gfx_sync(g);
        for (i = 0; i < nr; i++) {
            pre[i] = g_memdup2(g->vram + rng[i][0], rng[i][1] - rng[i][0]);
        }
    }
    if (q->ntri && !g->gpu->ops->draw(g->gpu->b, q)) {
        g->gst.fallback[GFB_BACKEND]++;
        for (i = 0; i < nr; i++) {
            g_free(pre[i]);
        }
        goto out;
    }
    ok = true;
    g->gst.taken++;
    g->gst.taken_db += q->db;
    for (i = 0; i < nr; i++) {
        pend_add(g, rng[i][0], rng[i][1]);
    }
    g->ram_pend |= q->rt.space;
    for (i = 0; i < q->ntex; i++) {
        /* guest RAM the GPU reads is not tracked by range */
        g->ram_pend |= q->tex[i].surf.space;
        if (!q->tex[i].surf.space) {
            pend_add(g, q->tex[i].surf.voff,
                     q->tex[i].surf.voff + hd5770_surf_size(&q->tex[i].surf.s));
        }
        if (q->tex[i].mip) {
            unsigned l;

            for (l = 1; l <= q->tex[i].last_lv; l++) {
                pend_add(g, q->tex[i].lv[l].voff, q->tex[i].lv[l].voff +
                         hd5770_surf_size(&q->tex[i].lv[l].s));
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

/*
 * The pixel shader runs on lanes unless it samples the render target:
 * then each pixel must see the ones drawn before it.
 */
static void lanes_setup(Draw *d)
{
    uint64_t rlo = d->rt.base, rhi = d->rt.base + hd5770_surf_size(&d->rt);
    uint8_t ids[16];
    unsigned nconst, nres, i;
    bool grad;

    if (d->resolve || !hd5770_prog_lanes(d->ps, &nconst, &grad)) {
        return;
    }
    hd5770_prog_resources(d->ps, ids, &nres, ARRAY_SIZE(ids));
    for (i = 0; i < nres; i++) {
        Tex t;
        uint64_t lo, hi;

        if (!tex_setup(d, ids[i], 0, &t)) {
            return;
        }
        lo = t.s.base;
        hi = lo + hd5770_surf_size(&t.s);
        if (lo < rhi && rlo < hi) {
            return;
        }
    }
    grad |= d->mips;
    d->kv = g_new(HD5770Val, MAX(nconst, 1));
    hd5770_prog_bind(d->ps, &d->ps_env, d->kv);
    d->lanes = g_new0(HD5770Lanes, grad ? 3 : 1);
    d->lanes->pos_gpr = d->pos_ena ? (int)d->pos_gpr : -1;
    if (grad) {
        d->nh = &d->lanes[1];
        d->nv = &d->lanes[2];
        d->nh->pos_gpr = d->nv->pos_gpr = d->lanes->pos_gpr;
    }
}

void hd5770_gfx_draw(HD5770Gfx *g, const uint32_t *b, uint32_t n)
{
    HD5770GfxStats *st = g->st;
    unsigned op = (b[0] >> 8) & 0xff, prim, i, nv, mode;
    unsigned reason = HD5770_SKIP_MAX;
    uint32_t count, cctl;
    int64_t t0 = get_clock();
    Draw *d;
    Vtx *vb = NULL;
    HD5770Thread *t = NULL;
    int wx, wy;
    bool taken;

    st->draws++;
    if (op != 0x2d || n < 3) {
        st->skip_reason[HD5770_SKIP_INDEX]++;
        st->skipped++;
        return;
    }
    d = g_new0(Draw, 1);
    d->g = g;
    d->st = st;
    d->ok = true;
    count = b[1];
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

        d->cull = prim == PRIM_RECTLIST ? 0 : sc & 3;
        d->face_cw = (sc >> 2) & 1;
        d->prov_last = (sc >> 19) & 1;
        if (sc & (7u << 11)) {
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
    vb = g_new(Vtx, MAX(nv, 1) + 1);
    t = g_new0(HD5770Thread, 1);
    for (i = 0; i < nv; i++) {
        if (!vs_run(d, i + CTX(g, VGT_INDX_OFFSET), &vb[i], t)) {
            reason = HD5770_SKIP_SHADER;
            goto out;
        }
    }
    memset(t, 0, sizeof(*t));
    gpu_lock(g, true);
    taken = gpu_draw(d, vb, nv, prim, t);
    gpu_lock(g, false);
    if (!taken) {
        const HD5770Surf *ws[3] = { d->color ? &d->rt : NULL,
                                    d->db && d->zfmt ? &d->zs : NULL,
                                    d->db && d->has_s ? &d->ss : NULL };

        /* the CPU writes the targets: after queued draws that touch them */
        for (i = 0; i < 3; i++) {
            uint64_t off;

            if (!ws[i]) {
                continue;
            }
            if (vram_off(g, ws[i]->base, hd5770_surf_size(ws[i]), &off)) {
                hd5770_gfx_touch(g, off, hd5770_surf_size(ws[i]));
            } else if (g->ram_pend) {
                hd5770_gfx_sync(g);
            }
        }
        if (d->resolve) {
            pend_check(g, d->rs.base, hd5770_surf_size(&d->rs));
        }
        d->wy0 = INT_MAX;
        d->wy1 = -1;
        assemble(d, vb, nv, prim, t, raster_tri);
        for (i = 0; i < 3 && d->ok && d->g->mem.dirty && d->wy1 >= d->wy0;
             i++) {
            uint64_t lo, hi;

            if (ws[i]) {
                /* the rows drawn */
                hd5770_surf_rows(ws[i], d->wy0, d->wy1 + 1, &lo, &hi);
                d->g->mem.dirty(d->g->mem.opaque, lo, hi - lo);
            }
        }
    }
    if (d->ok) {
        st->drawn++;
    } else {
        reason = HD5770_SKIP_SHADER;
    }
out:
    if (reason != HD5770_SKIP_MAX) {
        st->skip_reason[reason]++;
        st->skipped++;
    }
    g_free(vb);
    g_free(t);
    g_free(d->nt);
    g_free(d->lanes);
    g_free(d->kv);
    g_free(d);
    st->ns += get_clock() - t0;
}

/* ---------------------------------------------------------------- */

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
            g_free(e);
            e = n;
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
        PRIu64 " resolves %" PRIu64 " mipmapped fetches %" PRIu64 "\n",
        st->draws, st->drawn, st->skipped, st->prims, st->verts, st->pixels,
        st->blend_on, st->tex_fetch, st->vtx_fetch, st->kills,
        st->ns / 1000000, st->culled, st->db_tests, st->db_fails,
        st->resolves, st->lod_fetches);
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
        PRIu64 "\n"
        "gpu: syncs %" PRIu64 " (on access %" PRIu64 ") verified %" PRIu64
        " differing draws %" PRIu64 " bytes %" PRIu64 " (>1: %" PRIu64
        ") max %" PRIu64 "\n",
        g->gpu_mode == HD5770_GPU_VERIFY ? "verify" : "on", t->offered,
        t->taken, t->taken_db, t->fallback[GFB_PROG], t->fallback[GFB_SMALL],
        t->fallback[GFB_RT], t->fallback[GFB_TEX], t->fallback[GFB_VB],
        t->fallback[GFB_BACKEND], t->fallback[GFB_DB], t->syncs, t->touch_syncs, t->verified,
        t->vdiff_draws, t->vdiff_bytes, t->vdiff_big, t->vmax);
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
