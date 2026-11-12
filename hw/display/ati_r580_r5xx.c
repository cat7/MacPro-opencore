/*
 * ATI R580 3D engine, R500 pipeline (CPU path).
 *
 * Register latch for the R500 program memories (PVS through
 * VAP_PVS_VECTOR_INDX/DATA, US through GA_US_VECTOR_INDEX/DATA), vertex
 * fetch (VAP_PROG_STREAM_CNTL[_EXT], VAP_VTX_AOS_*, immediate data),
 * vertex program, clipping, viewport, RS_IP/RS_INST routing, primitive
 * assembly, and the per-quad rasteriser running the R500 fragment
 * program.
 *
 * Vertex fetch, clipping and RS routing adapted from rv530-qemu 0b06c9f
 * (hw/display/rv530/r5xx_3d.c).
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include <math.h>
#include "qemu/log.h"
#include "qemu/thread.h"
#include "qemu/rcu.h"
#include "qemu/timer.h"
#include "hw/pci/pci_device.h"
#include "ati_r580_int.h"
#include "ati_r580_r5pvs.h"
#include "ati_r580_r5us.h"
#include "ati_r580_r5tex.h"
#include "ati_r580_r5raster.h"
#include "ati_r580_r5gpu.h"
#include "ati_r580_r5xx.h"
#include "trace.h"

#define SE_VPORT_XSCALE           0x1d98
#define SE_VPORT_XOFFSET          0x1d9c
#define SE_VPORT_YSCALE           0x1da0
#define SE_VPORT_YOFFSET          0x1da4
#define SE_VPORT_ZSCALE           0x1da8
#define SE_VPORT_ZOFFSET          0x1dac
#define VAP_CNTL                  0x2080
#define VAP_ALT_NUM_VERTICES      0x2088
#define VAP_INDEX_OFFSET          0x208c
#define VAP_OUTPUT_VTX_FMT_0      0x2090
#define VAP_OUTPUT_VTX_FMT_1      0x2094
#define VAP_VTE_CNTL              0x20b0
#define VAP_VTX_SIZE              0x20b4
#define VAP_VTX_NUM_ARRAYS        0x20c0
#define VAP_VF_MAX_VTX_INDX       0x2134
#define VAP_VF_MIN_VTX_INDX       0x2138
#define VAP_CNTL_STATUS           0x2140
#define VAP_PROG_STREAM_CNTL_0    0x2150
#define VAP_PROG_STREAM_CNTL_EXT_0 0x21e0
#define VAP_PVS_VECTOR_INDX       0x2200
#define VAP_PVS_VECTOR_DATA       0x2204
#define VAP_PVS_UPLOAD_DATA       0x2208
#define VAP_CLIP_CNTL             0x221c
#define VAP_PVS_FLOW_CNTL_ADDRS_0 0x2230
#define VAP_PVS_FLOW_CNTL_LOOP_INDEX_0 0x2290
#define VAP_PVS_CODE_CNTL_0       0x22d0
#define VAP_PVS_CONST_CNTL        0x22d4
#define VAP_PVS_CODE_CNTL_1       0x22d8
#define VAP_PVS_FLOW_CNTL_OPC     0x22dc
#define VAP_PVS_FLOW_CNTL_ADDRS_LW_0 0x2500
#define VAP_PVS_FLOW_CNTL_ADDRS_UW_0 0x2504
#define GB_ENABLE                 0x4008
#define GB_TILE_CONFIG            0x4018
#define GB_SELECT                 0x401c
#define GB_AA_CONFIG              0x4020
#define R500_RS_IP_0              0x4074
#define TX_ENABLE                 0x4104
#define GB_PS3_ENABLE             0x4118
#define PS3_MODE                  (1u << 0)
#define GA_POINT_S0               0x4200
#define GA_POINT_T0               0x4204
#define GA_POINT_S1               0x4208
#define GA_POINT_T1               0x420c
#define GA_POINT_SIZE             0x421c
#define GA_POINT_MINMAX           0x4230
#define GA_LINE_CNTL              0x4234
#define GA_US_VECTOR_INDEX        0x4250
#define GA_US_VECTOR_DATA         0x4254
#define GA_COLOR_CONTROL_PS3      0x4258
#define GA_COLOR_CONTROL          0x4278
#define GA_POLY_MODE              0x4288
#define GA_ROUND_MODE             0x428c
#define GA_OFFSET                 0x4290
#define GA_FOG_SCALE              0x4294
#define GA_FOG_OFFSET             0x4298
#define SU_POLY_OFFSET_FRONT_SCALE 0x42a4
#define SU_POLY_OFFSET_FRONT_OFFSET 0x42a8
#define SU_POLY_OFFSET_BACK_SCALE 0x42ac
#define SU_POLY_OFFSET_BACK_OFFSET 0x42b0
#define SU_POLY_OFFSET_ENABLE     0x42b4
#define SU_CULL_MODE              0x42b8
#define SU_DEPTH_SCALE            0x42c0
#define SU_DEPTH_OFFSET           0x42c4
#define RS_INST_COUNT             0x4304
#define R500_RS_INST_0            0x4320
#define SC_EDGERULE               0x43a8
#define SC_CLIP_0_A               0x43b0
#define SC_CLIP_0_B               0x43b4
#define SC_CLIP_RULE              0x43d0
#define SC_SCISSORS_TL            0x43e0
#define SC_SCISSORS_BR            0x43e4
#define TX_FILTER0_0              0x4400
#define TX_FILTER1_0              0x4440
#define TX_FORMAT0_0              0x4480
#define TX_FORMAT1_0              0x44c0
#define TX_FORMAT2_0              0x4500
#define TX_OFFSET_0               0x4540
#define TX_CHROMA_KEY_0           0x4580
#define TX_BORDER_COLOR_0         0x45c0
#define US_CONFIG                 0x4600
#define US_PIXSIZE                0x4604
#define US_FC_BOOL_CONST          0x4620
#define US_FC_CTRL                0x4624
#define US_CODE_ADDR              0x4630
#define US_CODE_RANGE             0x4634
#define US_CODE_OFFSET            0x4638
#define US_OUT_FMT_0              0x46a4
#define US_W_FMT                  0x46b4
#define FG_FOG_BLEND              0x4bc0
#define FG_FOG_FACTOR             0x4bc4
#define FG_FOG_COLOR_R            0x4bc8
#define FG_FOG_COLOR_G            0x4bcc
#define FG_FOG_COLOR_B            0x4bd0
#define FG_ALPHA_FUNC             0x4bd4
#define FG_DEPTH_SRC              0x4bd8
#define FG_ALPHA_VALUE            0x4be0
#define US_FC_INT_CONST_0         0x4c00
#define RB3D_CCTL                 0x4e00
#define RB3D_BLENDCNTL            0x4e04
#define RB3D_ABLENDCNTL           0x4e08
#define RB3D_COLOR_CHANNEL_MASK   0x4e0c
#define RB3D_BLEND_COLOR          0x4e10
#define RB3D_ROPCNTL              0x4e18
#define RB3D_CLRCMP_FLIPE         0x4e1c
#define RB3D_CLRCMP_CLR           0x4e20
#define RB3D_CLRCMP_MSK           0x4e24
#define RB3D_COLOROFFSET0         0x4e28
#define RB3D_COLORPITCH0          0x4e38
#define RB3D_DITHER_CTL           0x4e50
#define RB3D_AARESOLVE_OFFSET     0x4e80
#define RB3D_AARESOLVE_PITCH      0x4e84
#define RB3D_AARESOLVE_CTL        0x4e88
#define RB3D_CONSTANT_COLOR_AR    0x4ef8
#define RB3D_CONSTANT_COLOR_GB    0x4efc
#define ZB_CNTL                   0x4f00
#define ZB_ZSTENCILCNTL           0x4f04
#define ZB_STENCILREFMASK         0x4f08
#define ZB_FORMAT                 0x4f10
#define ZB_ZTOP                   0x4f14
#define ZB_DEPTHOFFSET            0x4f20
#define ZB_DEPTHPITCH             0x4f24
#define ZB_DEPTHCLEARVALUE        0x4f28
#define ZB_STENCILREFMASK_BF      0x4fd4
#define US_FC_INST_0              0x9800
#define US_FC_ADDR_0              0xa000

#define VF_PRIM(v)                ((v) & 0xf)
#define VF_WALK(v)                (((v) >> 4) & 3)
#define VF_INDEX_32               (1u << 11)
#define VF_USE_ALT_NUM            (1u << 14)
#define VF_NUM(v)                 ((v) >> 16)
#define WALK_INDICES              1
#define WALK_EMBEDDED             3

#define VTE_X_SCALE_ENA           (1u << 0)
#define VTE_X_OFFSET_ENA          (1u << 1)
#define VTE_Y_SCALE_ENA           (1u << 2)
#define VTE_Y_OFFSET_ENA          (1u << 3)
#define VTE_Z_SCALE_ENA           (1u << 4)
#define VTE_Z_OFFSET_ENA          (1u << 5)
#define VTE_XY_FMT                (1u << 8)
#define VTE_Z_FMT                 (1u << 9)
#define VTE_W0_FMT                (1u << 10)

#define VAP_TCL_BYPASS            (1u << 8)
#define VAP_DX_CLIP_SPACE         (1u << 22)
#define CLIP_DISABLE              (1u << 16)
#define GB_POINT_STUFF_ENABLE     (1u << 0)

#define PVS_MEM_VECS              2048
#define PVS_CONST_START           1024
#define PVS_UCP_START             1536
#define NUM_SLOTS                 14
#define SLOT_POS                  0
#define SLOT_PSIZE                1
#define SLOT_COL0                 2
#define SLOT_TEX0                 6
#define MAX_VBUF                  16
#define VCACHE_SIZE               256
#define MAX_CLIP_VERTS            24
#define R5_MAX_THREADS            8

#define REG(s, off) ((s)->regs[(off) >> 2])

typedef struct GfxVert {
    r5xx_vec4 slot[NUM_SLOTS];
} GfxVert;

typedef struct ClipCtx {
    int nplanes;
    float plane[8][4];
} ClipCtx;

typedef struct RsCtx {
    uint32_t ip[16], inst[16];
    int ninst;
    int tex_comp_base[8];
    int tex_comp_n[8];
    bool face_used;
    int face_addr;
} RsCtx;

typedef struct VtxSource {
    const uint32_t *immd;
    uint32_t immd_dwords;
    uint32_t vtx_size;
} VtxSource;

#define PRIM_TRI            R5_PRIM_TRI
#define PRIM_POINT          R5_PRIM_POINT
#define PRIM_POINT_SIZED    R5_PRIM_POINT_SIZED
#define PRIM_LINE           R5_PRIM_LINE

/* per-thread memory interface: dirty extent of the stores it made */
typedef struct R5Mem {
    R5xxMem mem;
    ATIR580State *s;
    uint64_t dlo, dhi;
} R5Mem;

struct ATIR580R5 {
    ATIR580State *s;
    R5Mem m[R5_MAX_THREADS + 1];

    union {
        uint32_t u[PVS_MEM_VECS * 4];
        float f[PVS_MEM_VECS * 4];
    } pvs;
    uint32_t pvs_index;
    bool pvs_dirty;
    R5xxPvsProgram pvs_prog;
    R5xxPvsDecoded *pvs_dec;

    uint32_t us_code[R500_US_MAX_INSTR * 6];
    union {
        uint32_t u[R500_US_MAX_CONST * 4];
        float f[R500_US_MAX_CONST * 4];
    } us_const;
    uint32_t us_index;
    bool us_is_const;
    bool us_clamp;
    bool us_dirty;
    R500UsProgram us_prog;
    R500UsDecoded *us_dec;

    struct {
        uint32_t addr, stride;
    } vbuf[MAX_VBUF];
    int num_vbuf;
    uint32_t indx_vf;

    R5xxTexSampler samplers[R5XX_NUM_TEX_UNITS];
    R5xxRaster raster;
    uint32_t vcache_tag[VCACHE_SIZE];
    GfxVert vcache[VCACHE_SIZE];

    /* band workers */
    int nthreads;
    bool collect, quit;
    R5Prim *prims;
    uint32_t nprims, cap;
    QemuThread th[R5_MAX_THREADS];
    QemuSemaphore go[R5_MAX_THREADS], done;
    R5xxRaster wr[R5_MAX_THREADS];

    /* statistics */
    uint64_t draws, prims_n, verts, skipped;
    uint64_t us_decodes, us_fail, us_fc, pvs_decodes, pvs_fail;
    uint64_t setup_fail, px_in, px_out, quads, zclears, draw_ns;
    uint64_t max_draw_ns, gpu_draws;
    uint64_t tiled_cb, tiled_tx, wide_cb;
    uint64_t tex_bad, tex_unbound;
    /* the last draws, for diagnosis */
    struct {
        uint32_t vf, n, dst, cpitch, blend, ablend, out_fmt, txen;
        uint32_t tx[5];
        uint32_t us_addr, us0, us3;
        int x0, y0, x1, y1;
        uint64_t px;
    } hist[32];
    unsigned hist_pos;
    uint32_t last_tex_regs[6];
    uint32_t last_dst, last_tex0, last_us_n, last_vf;
    char last_err[160];
};

static float r5_f32(uint32_t v)
{
    float f;

    memcpy(&f, &v, 4);
    return f;
}

static void r5_err(ATIR580R5 *g, const char *fmt, ...) G_GNUC_PRINTF(2, 3);
static void r5_err(ATIR580R5 *g, const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(g->last_err, sizeof(g->last_err), fmt, ap);
    va_end(ap);
    qemu_log_mask(LOG_UNIMP, "ati-r580: r500: %s\n", g->last_err);
}

/* ---- card address space ---- */

static uint8_t *r5_mem_map(void *opaque, uint32_t addr, uint32_t len,
                           bool write)
{
    R5Mem *m = opaque;
    ATIR580State *s = m->s;
    uint32_t off;

    if (!ati_r580_mc_to_vram(s, addr, &off) ||
        (uint64_t)off + len > ATI_R580_VRAM_SIZE) {
        return NULL;
    }
    if (write) {
        ati_r580_gl_dirty(s, off, len);
    } else {
        ati_r580_gl_touch(s, off, len);
    }
    return (uint8_t *)memory_region_get_ram_ptr(&s->vram) + off;
}

static void r5_mem_written(void *opaque, uint8_t *host, uint32_t len)
{
    R5Mem *m = opaque;
    uint64_t off = host - (uint8_t *)memory_region_get_ram_ptr(&m->s->vram);

    if (off < m->dlo) {
        m->dlo = off;
    }
    if (off + len > m->dhi) {
        m->dhi = off + len;
    }
}

static void r5_mem_read(void *opaque, uint32_t addr, void *buf, uint32_t len)
{
    R5Mem *m = opaque;
    uint8_t *p = r5_mem_map(opaque, addr, len, false);
    uint8_t *d = buf;

    if (p) {
        memcpy(buf, p, len);
        return;
    }
    while (len) {
        uint32_t a = addr & ~3u, k = addr & 3, n = MIN(4 - k, len);
        uint32_t w = cpu_to_le32(ati_r580_mc_read32(m->s, a));

        memcpy(d, (uint8_t *)&w + k, n);
        d += n;
        addr += n;
        len -= n;
    }
}

static void r5_mem_write(void *opaque, uint32_t addr, const void *buf,
                         uint32_t len)
{
    R5Mem *m = opaque;
    uint8_t *p = r5_mem_map(opaque, addr, len, true);
    const uint8_t *d = buf;

    if (p) {
        memcpy(p, buf, len);
        r5_mem_written(opaque, p, len);
        return;
    }
    while (len) {
        uint32_t a = addr & ~3u, k = addr & 3, n = MIN(4 - k, len);
        uint32_t w = 0;

        if (n != 4) {
            w = cpu_to_le32(ati_r580_mc_read32(m->s, a));
        }
        memcpy((uint8_t *)&w + k, d, n);
        ati_r580_mc_write32(m->s, a, le32_to_cpu(w));
        d += n;
        addr += n;
        len -= n;
    }
}

static void R5XX_PRINTF(3, 0)
r5_mem_log(void *opaque, int level, const char *fmt, va_list ap)
{
    if (level <= 1) {
        char buf[256];

        vsnprintf(buf, sizeof(buf), fmt, ap);
        qemu_log_mask(LOG_GUEST_ERROR, "ati-r580: r500: %s\n", buf);
    }
}

/* VRAM is kept linear: the CPU aperture, 2D engine and scanout read it so */
static bool r5_linear(void *opaque, uint32_t gpu_addr)
{
    return true;
}

static void r5_mem_flush(ATIR580R5 *g, R5Mem *m)
{
    if (m->dhi > m->dlo) {
        memory_region_set_dirty(&g->s->vram, m->dlo, m->dhi - m->dlo);
    }
    m->dlo = UINT64_MAX;
    m->dhi = 0;
}

/* ---- register latch ---- */

void ati_r580_r5xx_reg_write(ATIR580State *s, uint32_t off, uint32_t val)
{
    ATIR580R5 *g = s->r5;

    if (!g) {
        return;
    }
    switch (off) {
    case VAP_PVS_VECTOR_INDX:
        g->pvs_index = (val & 0x7ff) * 4;
        break;
    case VAP_PVS_VECTOR_DATA:
    case VAP_PVS_UPLOAD_DATA:
        if (g->pvs_index < PVS_MEM_VECS * 4) {
            if (g->pvs_index < PVS_CONST_START * 4) {
                g->pvs_dirty = true;
            }
            g->pvs.u[g->pvs_index] = val;
        }
        g->pvs_index++;
        break;
    case VAP_PVS_CODE_CNTL_0:
    case VAP_PVS_CODE_CNTL_1:
    case VAP_PVS_CONST_CNTL:
    case VAP_PVS_FLOW_CNTL_OPC:
    case VAP_CNTL:
        g->pvs_dirty = true;
        break;
    case GA_US_VECTOR_INDEX:
        g->us_is_const = (val >> 16) & 1;
        g->us_clamp = (val >> 17) & 1;
        g->us_index = g->us_is_const ? (val & 0xff) * 4 : (val & 0x1ff) * 6;
        break;
    case GA_US_VECTOR_DATA:
        /* the index self-increments and wraps at 512 instructions */
        if (g->us_is_const) {
            if (g->us_clamp) {
                float f = r5_f32(val);

                f = f > 1.0f ? 1.0f : (f < -1.0f ? -1.0f : f);
                memcpy(&val, &f, 4);
            }
            g->us_const.u[g->us_index % (R500_US_MAX_CONST * 4)] = val;
        } else {
            g->us_code[g->us_index % (R500_US_MAX_INSTR * 6)] = val;
            g->us_dirty = true;
        }
        g->us_index++;
        break;
    default:
        if ((off >= VAP_PVS_FLOW_CNTL_ADDRS_0 && off < 0x22c0) ||
            (off >= VAP_PVS_FLOW_CNTL_ADDRS_LW_0 && off < 0x2580)) {
            g->pvs_dirty = true;
        }
        if ((off >= US_CONFIG && off < 0x4700) ||
            (off >= US_FC_INST_0 && off < US_FC_INST_0 + 0x40) ||
            (off >= US_FC_ADDR_0 && off < US_FC_ADDR_0 + 0x40) ||
            off == US_FC_BOOL_CONST ||
            (off >= US_FC_INT_CONST_0 && off < US_FC_INT_CONST_0 + 0x80)) {
            g->us_dirty = true;
        }
        break;
    }
}

/* ---- programs ---- */

static bool r5_update_pvs(ATIR580State *s, ATIR580R5 *g)
{
    R5xxPvsProgram *p = &g->pvs_prog;
    char err[160];
    int i;

    if (!g->pvs_dirty && g->pvs_dec) {
        return true;
    }
    memcpy(p->code, g->pvs.u, sizeof(p->code));
    p->code_cntl0 = REG(s, VAP_PVS_CODE_CNTL_0);
    p->code_cntl1 = REG(s, VAP_PVS_CODE_CNTL_1);
    p->const_cntl = REG(s, VAP_PVS_CONST_CNTL);
    p->vap_cntl = REG(s, VAP_CNTL);
    p->fc_opc = REG(s, VAP_PVS_FLOW_CNTL_OPC);
    for (i = 0; i < R5XX_PVS_MAX_FC; i++) {
        p->fc_addrs[i] = REG(s, VAP_PVS_FLOW_CNTL_ADDRS_0 + 4 * i);
        p->fc_loop_index[i] = REG(s, VAP_PVS_FLOW_CNTL_LOOP_INDEX_0 + 4 * i);
        p->fc_addrs_lw[i] = REG(s, VAP_PVS_FLOW_CNTL_ADDRS_LW_0 + 8 * i);
        p->fc_addrs_uw[i] = REG(s, VAP_PVS_FLOW_CNTL_ADDRS_UW_0 + 8 * i);
    }
    p->is_r500 = true;
    if (g->pvs_dec) {
        r5xx_pvs_free(g->pvs_dec);
    }
    g->pvs_dec = r5xx_pvs_decode(p, err, sizeof(err));
    if (!g->pvs_dec) {
        g->pvs_fail++;
        r5_err(g, "vertex program: %s", err);
        return false;
    }
    g->pvs_decodes++;
    if (trace_event_get_state_backends(TRACE_ATI_R580_R5_DRAW)) {
        g_autofree char *buf = g_malloc(65536);

        r5xx_pvs_disasm(p, buf, 65536);
        trace_ati_r580_r5_draw(buf);
    }
    g->pvs_dirty = false;
    return true;
}

static bool r5_update_us(ATIR580State *s, ATIR580R5 *g)
{
    R500UsProgram *p = &g->us_prog;
    char err[160];
    int i;

    if (!g->us_dirty && g->us_dec) {
        return true;
    }
    memcpy(p->code, g->us_code, sizeof(p->code));
    p->config = REG(s, US_CONFIG);
    p->pixsize = REG(s, US_PIXSIZE);
    p->code_addr = REG(s, US_CODE_ADDR);
    p->code_range = REG(s, US_CODE_RANGE);
    p->code_offset = REG(s, US_CODE_OFFSET);
    p->fc_ctrl = REG(s, US_FC_CTRL);
    for (i = 0; i < R500_US_MAX_FC; i++) {
        p->fc_inst[i] = REG(s, US_FC_INST_0 + 4 * i);
        p->fc_addr[i] = REG(s, US_FC_ADDR_0 + 4 * i);
    }
    for (i = 0; i < 4; i++) {
        p->out_fmt[i] = REG(s, US_OUT_FMT_0 + 4 * i);
    }
    p->w_fmt = REG(s, US_W_FMT);
    for (i = 0; i < 32; i++) {
        p->fc_int_const[i] = REG(s, US_FC_INT_CONST_0 + 4 * i);
    }
    p->fc_bool_const = REG(s, US_FC_BOOL_CONST);
    if (g->us_dec) {
        r500_us_free(g->us_dec);
    }
    g->us_dec = r500_us_decode(p, err, sizeof(err));
    if (!g->us_dec) {
        g->us_fail++;
        r5_err(g, "fragment program: %s", err);
        return false;
    }
    g->us_decodes++;
    if (r500_us_uses_flow_control(g->us_dec)) {
        g->us_fc++;
    }
    g->last_us_n = r500_us_num_instructions(g->us_dec);
    if (trace_event_get_state_backends(TRACE_ATI_R580_R5_DRAW)) {
        g_autofree char *buf = g_malloc(65536);

        r500_us_disasm(p, buf, 65536);
        trace_ati_r580_r5_draw(buf);
    }
    g->us_dirty = false;
    return true;
}

static void r5_tex_cb(void *opaque, const R5xxTexRequest *req,
                      r5xx_vec4 out[4])
{
    R5Mem *m = opaque;
    ATIR580R5 *g = m->s->r5;
    int i;

    if (req->unit < 0 || req->unit >= R5XX_NUM_TEX_UNITS ||
        !(REG(m->s, TX_ENABLE) & (1u << req->unit)) ||
        !g->samplers[req->unit].valid) {
        g->tex_unbound++;
        for (i = 0; i < 4; i++) {
            out[i] = r5xx_vec4_make(0, 0, 0, 1);
        }
        return;
    }
    r5xx_tex_sample_quad(&g->samplers[req->unit], &m->mem, req, out);
}

static void r5_update_textures(ATIR580State *s, ATIR580R5 *g)
{
    uint32_t used = r500_us_tex_units_used(g->us_dec) & REG(s, TX_ENABLE);
    int n;

    for (n = 0; n < R5XX_NUM_TEX_UNITS; n++) {
        R5xxTexRegs tr;
        char err[128];
        uint32_t off;

        g->samplers[n].valid = false;
        if (!(used & (1u << n))) {
            continue;
        }
        tr.filter0 = REG(s, TX_FILTER0_0 + 4 * n);
        tr.filter1 = REG(s, TX_FILTER1_0 + 4 * n);
        tr.format0 = REG(s, TX_FORMAT0_0 + 4 * n);
        tr.format1 = REG(s, TX_FORMAT1_0 + 4 * n);
        tr.format2 = REG(s, TX_FORMAT2_0 + 4 * n);
        tr.offset = REG(s, TX_OFFSET_0 + 4 * n);
        tr.chroma_key = REG(s, TX_CHROMA_KEY_0 + 4 * n);
        tr.border_color = REG(s, TX_BORDER_COLOR_0 + 4 * n);
        if (!r5xx_tex_decode(&tr, true, &g->samplers[n], err, sizeof(err))) {
            g->tex_bad++;
            g->last_tex_regs[0] = tr.format0;
            g->last_tex_regs[1] = tr.format1;
            g->last_tex_regs[2] = tr.format2;
            g->last_tex_regs[3] = tr.offset;
            g->last_tex_regs[4] = tr.filter0;
            g->last_tex_regs[5] = n;
            r5_err(g, "texture unit %d: %s", n, err);
        }
        /* TXO_ENDIAN applies to texels fetched over the bus only */
        if (ati_r580_mc_to_vram(s, g->samplers[n].layout.gpu_addr, &off)) {
            g->samplers[n].endian = 0;
        }
    }
}

static void r5_raster_regs(ATIR580State *s, R5xxRasterRegs *r)
{
    int i;

    memset(r, 0, sizeof(*r));
    r->rb3d_cctl = REG(s, RB3D_CCTL);
    r->rb3d_blendcntl = REG(s, RB3D_BLENDCNTL);
    r->rb3d_ablendcntl = REG(s, RB3D_ABLENDCNTL);
    r->rb3d_color_channel_mask = REG(s, RB3D_COLOR_CHANNEL_MASK);
    r->rb3d_blend_color = REG(s, RB3D_BLEND_COLOR);
    r->rb3d_ropcntl = REG(s, RB3D_ROPCNTL);
    r->rb3d_clrcmp_flipe = REG(s, RB3D_CLRCMP_FLIPE);
    r->rb3d_clrcmp_clr = REG(s, RB3D_CLRCMP_CLR);
    r->rb3d_clrcmp_msk = REG(s, RB3D_CLRCMP_MSK);
    for (i = 0; i < 4; i++) {
        r->rb3d_coloroffset[i] = REG(s, RB3D_COLOROFFSET0 + 4 * i);
        r->rb3d_colorpitch[i] = REG(s, RB3D_COLORPITCH0 + 4 * i);
        r->us_out_fmt[i] = REG(s, US_OUT_FMT_0 + 4 * i);
        r->sc_clip_a[i] = REG(s, SC_CLIP_0_A + 8 * i);
        r->sc_clip_b[i] = REG(s, SC_CLIP_0_B + 8 * i);
    }
    r->rb3d_dither_ctl = REG(s, RB3D_DITHER_CTL);
    r->rb3d_constant_color_ar = REG(s, RB3D_CONSTANT_COLOR_AR);
    r->rb3d_constant_color_gb = REG(s, RB3D_CONSTANT_COLOR_GB);
    r->zb_cntl = REG(s, ZB_CNTL);
    r->zb_zstencilcntl = REG(s, ZB_ZSTENCILCNTL);
    r->zb_stencilrefmask = REG(s, ZB_STENCILREFMASK);
    r->zb_format = REG(s, ZB_FORMAT);
    r->zb_ztop = REG(s, ZB_ZTOP);
    r->zb_depthoffset = REG(s, ZB_DEPTHOFFSET);
    r->zb_depthpitch = REG(s, ZB_DEPTHPITCH);
    r->zb_depthclearvalue = REG(s, ZB_DEPTHCLEARVALUE);
    r->zb_stencilrefmask_bf = REG(s, ZB_STENCILREFMASK_BF);
    r->fg_fog_blend = REG(s, FG_FOG_BLEND);
    r->fg_fog_factor = REG(s, FG_FOG_FACTOR);
    r->fg_fog_color_r = REG(s, FG_FOG_COLOR_R);
    r->fg_fog_color_g = REG(s, FG_FOG_COLOR_G);
    r->fg_fog_color_b = REG(s, FG_FOG_COLOR_B);
    r->fg_alpha_func = REG(s, FG_ALPHA_FUNC);
    r->fg_depth_src = REG(s, FG_DEPTH_SRC);
    r->fg_alpha_value = REG(s, FG_ALPHA_VALUE);
    r->sc_scissors_tl = REG(s, SC_SCISSORS_TL);
    r->sc_scissors_br = REG(s, SC_SCISSORS_BR);
    r->sc_clip_rule = REG(s, SC_CLIP_RULE);
    r->sc_edgerule = REG(s, SC_EDGERULE);
    r->su_cull_mode = REG(s, SU_CULL_MODE);
    r->su_poly_offset_front_scale = REG(s, SU_POLY_OFFSET_FRONT_SCALE);
    r->su_poly_offset_front_offset = REG(s, SU_POLY_OFFSET_FRONT_OFFSET);
    r->su_poly_offset_back_scale = REG(s, SU_POLY_OFFSET_BACK_SCALE);
    r->su_poly_offset_back_offset = REG(s, SU_POLY_OFFSET_BACK_OFFSET);
    r->su_poly_offset_enable = REG(s, SU_POLY_OFFSET_ENABLE);
    r->su_depth_scale = REG(s, SU_DEPTH_SCALE);
    r->su_depth_offset = REG(s, SU_DEPTH_OFFSET);
    r->ga_poly_mode = REG(s, GA_POLY_MODE);
    r->ga_color_control = REG(s, GA_COLOR_CONTROL);
    r->ga_point_size = REG(s, GA_POINT_SIZE);
    r->ga_point_minmax = REG(s, GA_POINT_MINMAX);
    r->ga_line_cntl = REG(s, GA_LINE_CNTL);
    r->ga_round_mode = REG(s, GA_ROUND_MODE);
    r->ga_offset = REG(s, GA_OFFSET);
    r->ga_fog_scale = REG(s, GA_FOG_SCALE);
    r->ga_fog_offset = REG(s, GA_FOG_OFFSET);
    r->gb_tile_config = REG(s, GB_TILE_CONFIG);
    r->gb_aa_config = REG(s, GB_AA_CONFIG);
    r->rb3d_aaresolve_offset = REG(s, RB3D_AARESOLVE_OFFSET);
    r->rb3d_aaresolve_pitch = REG(s, RB3D_AARESOLVE_PITCH);
    r->rb3d_aaresolve_ctl = REG(s, RB3D_AARESOLVE_CTL);
    r->gb_select = REG(s, GB_SELECT);
    r->is_r500 = true;
}

/* ---- vertex fetch ---- */

static uint32_t psc_type_dwords(uint32_t type)
{
    switch (type) {
    case 0: return 1;
    case 1: return 2;
    case 2: return 3;
    case 3: return 4;
    case 7: return 2;
    case 10: return 8;
    case 12: return 2;
    default: return 1;
    }
}

static void psc_decode(uint32_t type, bool sign, bool norm, const uint32_t *d,
                       float out[4])
{
    int i;

    out[0] = out[1] = out[2] = 0.0f;
    out[3] = 1.0f;
    switch (type) {
    case 0: case 1: case 2: case 3:
        for (i = 0; i <= (int)type; i++) {
            out[i] = r5_f32(le32_to_cpu(d[i]));
        }
        break;
    case 10:
        for (i = 0; i < 4; i++) {
            out[i] = r5_f32(le32_to_cpu(d[i]));
        }
        break;
    case 4: case 5: {
        uint32_t v = le32_to_cpu(d[0]);

        for (i = 0; i < 4; i++) {
            uint32_t b = (v >> (8 * i)) & 0xff;
            float f = sign ? (float)(int8_t)b : (float)b;

            if (norm || type == 5) {
                f = sign ? MAX(f / 127.0f, -1.0f) : f / 255.0f;
            }
            out[i] = f;
        }
        if (type == 5) {
            /* D3DCOLOR: bytes B, G, R, A */
            float t = out[0];

            out[0] = out[2];
            out[2] = t;
        }
        break;
    }
    case 6: case 7: {
        int n = type == 6 ? 2 : 4;

        for (i = 0; i < n; i++) {
            uint32_t w = (le32_to_cpu(d[i / 2]) >> (16 * (i & 1))) & 0xffff;
            float f = sign ? (float)(int16_t)w : (float)w;

            if (norm) {
                f = sign ? MAX(f / 32767.0f, -1.0f) : f / 65535.0f;
            }
            out[i] = f;
        }
        break;
    }
    case 11: case 12: {
        int n = type == 11 ? 2 : 4;

        for (i = 0; i < n; i++) {
            uint32_t w = (le32_to_cpu(d[i / 2]) >> (16 * (i & 1))) & 0xffff;

            out[i] = r5xx_half_to_float(w);
        }
        break;
    }
    case 8: case 9: {
        /* VECTOR_3_TTT 10:10:10, VECTOR_3_EET 11:11:10 */
        static const unsigned bits[2][3] = { { 10, 10, 10 }, { 11, 11, 10 } };
        uint32_t v = le32_to_cpu(d[0]);
        unsigned sh = 0;

        for (i = 0; i < 3; i++) {
            unsigned n = bits[type - 8][i];
            uint32_t f = (v >> sh) & ((1u << n) - 1);
            float x;

            sh += n;
            if (sign) {
                x = (float)((int32_t)(f << (32 - n)) >> (32 - n));
                if (norm) {
                    x = MAX(x / (float)((1u << (n - 1)) - 1), -1.0f);
                }
            } else {
                x = (float)f;
                if (norm) {
                    x /= (float)((1u << n) - 1);
                }
            }
            out[i] = x;
        }
        break;
    }
    default:
        break;
    }
}

/*
 * VAP_VTX_NUM_ARRAYS, then per pair of arrays VAP_VTX_AOS_ATTR (count and
 * stride of both, in dwords) and the two VAP_VTX_AOS_ADDR.
 */
static void r5_load_aos(ATIR580State *s, ATIR580R5 *g)
{
    uint32_t n = REG(s, VAP_VTX_NUM_ARRAYS) & 0x1f;
    uint32_t i;

    g->num_vbuf = MIN(n, MAX_VBUF);
    for (i = 0; i < (uint32_t)g->num_vbuf; i++) {
        uint32_t pair = VAP_VTX_NUM_ARRAYS + 4 + 12 * (i / 2);
        uint32_t attr = REG(s, pair) >> (16 * (i & 1));

        g->vbuf[i].stride = ((attr >> 8) & 0xff) * 4;
        g->vbuf[i].addr = REG(s, pair + 4 + 4 * (i & 1));
    }
}

static void r5_fetch(ATIR580State *s, ATIR580R5 *g, const VtxSource *src,
                     uint32_t index, r5xx_vec4 in[R5XX_PVS_MAX_INPUTS])
{
    uint32_t off = 0;
    int e;

    for (e = 0; e < 16; e++) {
        uint32_t psc = (REG(s, VAP_PROG_STREAM_CNTL_0 + 4 * (e / 2)) >>
                        (16 * (e & 1))) & 0xffff;
        uint32_t ext = (REG(s, VAP_PROG_STREAM_CNTL_EXT_0 + 4 * (e / 2)) >>
                        (16 * (e & 1))) & 0xffff;
        uint32_t type = psc & 0xf, skip = (psc >> 4) & 0xf;
        uint32_t dst = (psc >> 8) & 0x1f;
        bool last = psc & (1u << 13);
        bool sign = psc & (1u << 14), norm = psc & (1u << 15);
        uint32_t nd = psc_type_dwords(type);
        uint32_t raw[8] = { 0 };
        float v[4], sel[6];
        unsigned h;
        int c;

        if (src->immd) {
            uint32_t base = index * src->vtx_size + off + skip;

            if (base + nd <= src->immd_dwords) {
                memcpy(raw, src->immd + base, nd * 4);
            }
            off += skip + nd;
        } else {
            int a = e < g->num_vbuf ? e : g->num_vbuf - 1;

            if (a >= 0) {
                uint32_t addr = g->vbuf[a].addr + index * g->vbuf[a].stride +
                                skip * 4;
                unsigned k;

                if (!(addr & 3)) {
                    ati_r580_mc_read_block(s, addr, raw, nd);
                    for (k = 0; k < nd; k++) {
                        raw[k] = cpu_to_le32(raw[k]);
                    }
                } else {
                    r5_mem_read(&g->m[0], addr, raw, nd * 4);
                }
            }
        }
        /* FLOAT_8 is two FLOAT_4 into consecutive destinations */
        for (h = 0; h < (type == 10 ? 2u : 1u); h++) {
            psc_decode(type, sign, norm, raw + 4 * h, v);
            sel[0] = v[0]; sel[1] = v[1]; sel[2] = v[2]; sel[3] = v[3];
            sel[4] = 0.0f; sel[5] = 1.0f;
            if (dst + h < R5XX_PVS_MAX_INPUTS) {
                for (c = 0; c < 4; c++) {
                    uint32_t sw = (ext >> (3 * c)) & 7;

                    if ((ext >> (12 + c)) & 1) {
                        in[dst + h].f[c] = sel[sw < 6 ? sw : 4];
                    }
                }
            }
        }
        if (last) {
            break;
        }
    }
}

static void r5_vertex(ATIR580State *s, ATIR580R5 *g, const VtxSource *src,
                      uint32_t index, GfxVert *out)
{
    r5xx_vec4 in[R5XX_PVS_MAX_INPUTS];
    r5xx_vec4 res[R5XX_PVS_MAX_OUTPUTS];
    uint32_t fmt0 = REG(s, VAP_OUTPUT_VTX_FMT_0);
    uint32_t fmt1 = REG(s, VAP_OUTPUT_VTX_FMT_1);
    bool bypass = REG(s, VAP_CNTL_STATUS) & VAP_TCL_BYPASS;
    int i, k = 0;

    for (i = 0; i < R5XX_PVS_MAX_INPUTS; i++) {
        in[i] = r5xx_vec4_make(0, 0, 0, 1);
    }
    r5_fetch(s, g, src, index, in);
    for (i = 0; i < NUM_SLOTS; i++) {
        out->slot[i] = r5xx_vec4_make(0, 0, 0, 1);
    }
    if (bypass) {
        /* input vector n feeds output slot n */
        for (i = 0; i < NUM_SLOTS && i < R5XX_PVS_MAX_INPUTS; i++) {
            out->slot[i] = in[i];
        }
        return;
    }
    for (i = 0; i < R5XX_PVS_MAX_OUTPUTS; i++) {
        res[i] = r5xx_vec4_make(0, 0, 0, 1);
    }
    r5xx_pvs_run(g->pvs_dec, (const r5xx_vec4 *)&g->pvs.f[PVS_CONST_START * 4],
                 in, res);
    /* results packed in VAP_OUTPUT_VTX_FMT slot order */
    out->slot[SLOT_POS] = res[k++];
    if (fmt0 & (1u << 16)) {
        out->slot[SLOT_PSIZE] = res[k++];
    }
    for (i = 0; i < 4; i++) {
        if (fmt0 & (2u << i)) {
            out->slot[SLOT_COL0 + i] = res[k++];
        }
    }
    for (i = 0; i < 8 && k < R5XX_PVS_MAX_OUTPUTS; i++) {
        if ((fmt1 >> (3 * i)) & 7) {
            out->slot[SLOT_TEX0 + i] = res[k++];
        }
    }
}

/* ---- clipping, viewport, RS routing ---- */

static void r5_clip_setup(ATIR580State *s, ATIR580R5 *g, ClipCtx *cc)
{
    uint32_t cntl = REG(s, VAP_CLIP_CNTL);
    bool bypass = REG(s, VAP_CNTL_STATUS) & VAP_TCL_BYPASS;
    int i;

    cc->nplanes = 0;
    if (bypass || (REG(s, VAP_VTE_CNTL) & VTE_XY_FMT)) {
        return;
    }
    /* w > 0 */
    cc->plane[0][0] = 0;
    cc->plane[0][1] = 0;
    cc->plane[0][2] = 0;
    cc->plane[0][3] = 1;
    cc->nplanes = 1;
    if (!(cntl & CLIP_DISABLE)) {
        bool dx = REG(s, VAP_CNTL) & VAP_DX_CLIP_SPACE;

        cc->plane[1][0] = 0;
        cc->plane[1][1] = 0;
        cc->plane[1][2] = 1;
        cc->plane[1][3] = dx ? 0 : 1;
        cc->nplanes = 2;
        for (i = 0; i < 6; i++) {
            if (cntl & (1u << i)) {
                memcpy(cc->plane[cc->nplanes],
                       &g->pvs.f[(PVS_UCP_START + i) * 4], 4 * sizeof(float));
                cc->nplanes++;
            }
        }
    }
}

static inline float plane_dist(const float p[4], const r5xx_vec4 *v)
{
    return p[0] * v->f[0] + p[1] * v->f[1] + p[2] * v->f[2] + p[3] * v->f[3];
}

static void vert_lerp(GfxVert *o, const GfxVert *a, const GfxVert *b, float t)
{
    int i, c;

    for (i = 0; i < NUM_SLOTS; i++) {
        for (c = 0; c < 4; c++) {
            o->slot[i].f[c] = a->slot[i].f[c] +
                              (b->slot[i].f[c] - a->slot[i].f[c]) * t;
        }
    }
}

static int r5_clip_poly(const ClipCtx *cc, GfxVert *in, int n, GfxVert *tmp)
{
    GfxVert *a = in, *b = tmp;
    int p;

    for (p = 0; p < cc->nplanes && n > 0; p++) {
        const float *pl = cc->plane[p];
        float eps = (p == 0) ? 1e-6f : 0.0f;
        bool all_in = true;
        int i, m = 0;

        for (i = 0; i < n; i++) {
            if (plane_dist(pl, &a[i].slot[SLOT_POS]) < eps) {
                all_in = false;
                break;
            }
        }
        if (all_in) {
            continue;
        }
        for (i = 0; i < n; i++) {
            const GfxVert *cur = &a[i], *nxt = &a[(i + 1) % n];
            float dc = plane_dist(pl, &cur->slot[SLOT_POS]) - eps;
            float dn = plane_dist(pl, &nxt->slot[SLOT_POS]) - eps;

            if (dc >= 0 && m < MAX_CLIP_VERTS) {
                b[m++] = *cur;
            }
            if ((dc >= 0) != (dn >= 0) && m < MAX_CLIP_VERTS) {
                vert_lerp(&b[m++], cur, nxt, dc / (dc - dn));
            }
        }
        n = m;
        {
            GfxVert *t = a;

            a = b;
            b = t;
        }
    }
    if (a != in) {
        memcpy(in, a, n * sizeof(GfxVert));
    }
    return n;
}

/*
 * GB_ENABLE texture stuffing: set i whose TEXi_SOURCE is ST or STR is
 * generated by the GA for the primitive classes enabled in bits 0-2.
 */
static uint32_t r5_stuffed_sets(ATIR580State *s, uint32_t prim)
{
    uint32_t gbe = REG(s, GB_ENABLE);
    uint32_t cls = prim == 1 ? 1 :
                   (prim == 2 || prim == 3 || prim == 12) ? 2 : 4;
    uint32_t mask = 0;
    int i;

    if (!(gbe & cls)) {
        return 0;
    }
    for (i = 0; i < 8; i++) {
        uint32_t src = (gbe >> (16 + 2 * i)) & 3;

        if (src == 1 || src == 2) {
            mask |= 1u << i;
        }
    }
    return mask;
}

static void r5_rs_setup(ATIR580State *s, RsCtx *rs, R5xxRaster *r,
                        uint32_t prim)
{
    uint32_t fmt1 = REG(s, VAP_OUTPUT_VTX_FMT_1);
    uint32_t cc = REG(s, GA_COLOR_CONTROL);
    /* GA_COLOR_CONTROL_PS3 shading applies in PS3 mode only */
    uint32_t ps3 = (REG(s, GB_PS3_ENABLE) & PS3_MODE) ?
                   REG(s, GA_COLOR_CONTROL_PS3) : 0;
    uint32_t gbe = REG(s, GB_ENABLE);
    uint32_t stuffed = r5_stuffed_sets(s, prim);
    int i, comp = 0, max_in = 0;

    memset(rs, 0, sizeof(*rs));
    for (i = 0; i < 8; i++) {
        int n = (fmt1 >> (3 * i)) & 7;

        if (stuffed & (1u << i)) {
            n = ((gbe >> (16 + 2 * i)) & 3) == 2 ? 3 : 2;
        }
        n = n > 4 ? 4 : n;
        rs->tex_comp_base[i] = comp;
        rs->tex_comp_n[i] = n;
        comp += n;
    }
    rs->ninst = (REG(s, RS_INST_COUNT) & 0xf) + 1;
    r->flat_inputs = 0;
    for (i = 0; i < 16; i++) {
        rs->ip[i] = REG(s, R500_RS_IP_0 + 4 * i);
        rs->inst[i] = REG(s, R500_RS_INST_0 + 4 * i);
    }
    for (i = 0; i < rs->ninst; i++) {
        uint32_t inst = rs->inst[i];

        if (inst & (1u << 4)) {
            int addr = (inst >> 5) & 0x7f;
            int id = inst & 0xf;

            if (addr < R5XX_MAX_FS_INPUTS) {
                max_in = MAX(max_in, addr + 1);
                if (((ps3 >> (2 * id)) & 3) == 1) {
                    r->flat_inputs |= 1u << addr;
                }
            }
        }
        if ((inst >> 16) & 3) {
            int addr = (inst >> 18) & 0x7f;

            if (addr < R5XX_MAX_FS_INPUTS) {
                max_in = MAX(max_in, addr + 1);
                if (((inst >> 16) & 3) == 3) {
                    rs->face_used = true;
                    rs->face_addr = addr;
                    r->flat_inputs |= 1u << addr;
                } else if ((cc & 3) == 1) {
                    r->flat_inputs |= 1u << addr;
                }
            }
        }
    }
    r->num_fs_inputs = max_in;
}

static float rs_tex_comp(const RsCtx *rs, const GfxVert *v, int ptr)
{
    int t;

    if (ptr == 62) {
        return 0.0f;
    }
    if (ptr == 63) {
        return 1.0f;
    }
    for (t = 0; t < 8; t++) {
        int c = ptr - rs->tex_comp_base[t];

        if (c >= 0 && c < rs->tex_comp_n[t]) {
            return v->slot[SLOT_TEX0 + t].f[c];
        }
    }
    return 0.0f;
}

static r5xx_vec4 rs_color(const GfxVert *v, uint32_t ip)
{
    int ptr = (ip >> 24) & 7;
    int fmt = (ip >> 27) & 0xf;
    r5xx_vec4 c = ptr < 4 ? v->slot[SLOT_COL0 + ptr]
                          : r5xx_vec4_make(0, 0, 0, 1);

    switch (fmt) {
    case 1: c.f[3] = 0; break;
    case 2: c.f[3] = 1; break;
    case 4: c.f[0] = c.f[1] = c.f[2] = 0; break;
    case 5: c = r5xx_vec4_make(0, 0, 0, 0); break;
    case 6: c = r5xx_vec4_make(0, 0, 0, 1); break;
    case 8: c.f[0] = c.f[1] = c.f[2] = 1; break;
    case 9: c = r5xx_vec4_make(1, 1, 1, 0); break;
    case 10: c = r5xx_vec4_make(1, 1, 1, 1); break;
    default: break;
    }
    return c;
}

static inline float regf(ATIR580State *s, uint32_t off)
{
    return r5_f32(REG(s, off));
}

static inline float vport(ATIR580State *s, uint32_t vte, uint32_t off,
                          uint32_t ena, float def)
{
    return (vte & ena) ? regf(s, off) : def;
}

static void r5_to_window(ATIR580State *s, const RsCtx *rs, const GfxVert *v,
                         R5xxVertex *o)
{
    uint32_t vte = REG(s, VAP_VTE_CNTL);
    float x = v->slot[SLOT_POS].f[0], y = v->slot[SLOT_POS].f[1];
    float z = v->slot[SLOT_POS].f[2], w = v->slot[SLOT_POS].f[3];
    float iw;
    int i;

    if (vte & VTE_XY_FMT) {
        iw = 1.0f;
        o->w = (vte & VTE_W0_FMT) ? (w != 0.0f ? 1.0f / w : 1.0f) : 1.0f;
    } else {
        iw = (w != 0.0f) ? 1.0f / w : 0.0f;
        o->w = w;
        x *= iw;
        y *= iw;
    }
    if (!(vte & VTE_Z_FMT)) {
        z *= iw;
    }
    o->x = x * vport(s, vte, SE_VPORT_XSCALE, VTE_X_SCALE_ENA, 1.0f) +
           vport(s, vte, SE_VPORT_XOFFSET, VTE_X_OFFSET_ENA, 0.0f);
    o->y = y * vport(s, vte, SE_VPORT_YSCALE, VTE_Y_SCALE_ENA, 1.0f) +
           vport(s, vte, SE_VPORT_YOFFSET, VTE_Y_OFFSET_ENA, 0.0f);
    o->z = z * vport(s, vte, SE_VPORT_ZSCALE, VTE_Z_SCALE_ENA, 1.0f) +
           vport(s, vte, SE_VPORT_ZOFFSET, VTE_Z_OFFSET_ENA, 0.0f);

    for (i = 0; i < R5XX_MAX_FS_INPUTS; i++) {
        o->attr[i] = r5xx_vec4_make(0, 0, 0, 1);
    }
    for (i = 0; i < rs->ninst; i++) {
        uint32_t inst = rs->inst[i];

        if (inst & (1u << 4)) {
            uint32_t ip = rs->ip[inst & 0xf];
            int addr = (inst >> 5) & 0x7f;

            if (addr < R5XX_MAX_FS_INPUTS) {
                o->attr[addr] = r5xx_vec4_make(
                    rs_tex_comp(rs, v, ip & 0x3f),
                    rs_tex_comp(rs, v, (ip >> 6) & 0x3f),
                    rs_tex_comp(rs, v, (ip >> 12) & 0x3f),
                    rs_tex_comp(rs, v, (ip >> 18) & 0x3f));
            }
        }
        if (((inst >> 16) & 3) == 1) {
            uint32_t ip = rs->ip[(inst >> 12) & 0xf];
            int addr = (inst >> 18) & 0x7f;

            if (addr < R5XX_MAX_FS_INPUTS) {
                o->attr[addr] = rs_color(v, ip);
            }
        } else if (((inst >> 16) & 3) == 2) {
            /* COL_CN 2: window position, XY00 */
            int addr = (inst >> 18) & 0x7f;

            if (addr < R5XX_MAX_FS_INPUTS) {
                o->attr[addr] = r5xx_vec4_make(o->x, o->y, 0, 0);
            }
        }
    }
}

/* ---- primitive sinks ---- */

typedef struct DrawCtx {
    ATIR580State *s;
    ATIR580R5 *g;
    ClipCtx clip;
    RsCtx rs;
    bool psize;
} DrawCtx;

static R5Prim *r5_queue(ATIR580R5 *g, uint8_t kind)
{
    R5Prim *p;

    if (g->nprims == g->cap) {
        g->cap = g->cap ? g->cap * 2 : 1024;
        g->prims = g_renew(R5Prim, g->prims, g->cap);
    }
    p = &g->prims[g->nprims++];
    p->kind = kind;
    p->flat_inputs = g->raster.flat_inputs;
    p->cull_front = g->raster.d.cull_front;
    p->cull_back = g->raster.d.cull_back;
    return p;
}

static void r5_tri(ATIR580R5 *g, const R5xxVertex *a, const R5xxVertex *b,
                   const R5xxVertex *c)
{
    if (g->collect) {
        R5Prim *p = r5_queue(g, PRIM_TRI);

        p->v[0] = *a;
        p->v[1] = *b;
        p->v[2] = *c;
        return;
    }
    r5xx_raster_triangle(&g->raster, a, b, c);
}

static void r5_point(ATIR580R5 *g, const R5xxVertex *v, bool sized,
                     float size)
{
    if (g->collect) {
        R5Prim *p = r5_queue(g, sized ? PRIM_POINT_SIZED : PRIM_POINT);

        p->v[0] = *v;
        p->size = size;
        return;
    }
    if (sized) {
        r5xx_raster_point_sized(&g->raster, v, size);
    } else {
        r5xx_raster_point(&g->raster, v);
    }
}

static void r5_line(ATIR580R5 *g, const R5xxVertex *a, const R5xxVertex *b)
{
    if (g->collect) {
        R5Prim *p = r5_queue(g, PRIM_LINE);

        p->v[0] = *a;
        p->v[1] = *b;
        return;
    }
    r5xx_raster_line(&g->raster, a, b);
}

static void r5_emit_tri(DrawCtx *dc, const GfxVert *a, const GfxVert *b,
                        const GfxVert *c)
{
    ATIR580R5 *g = dc->g;
    GfxVert poly[MAX_CLIP_VERTS], tmp[MAX_CLIP_VERTS];
    R5xxVertex w[MAX_CLIP_VERTS];
    int n, i;

    poly[0] = *a;
    poly[1] = *b;
    poly[2] = *c;
    n = r5_clip_poly(&dc->clip, poly, 3, tmp);
    if (n < 3) {
        return;
    }
    for (i = 0; i < n; i++) {
        r5_to_window(dc->s, &dc->rs, &poly[i], &w[i]);
    }
    if (dc->rs.face_used) {
        float area = (w[1].x - w[0].x) * (w[2].y - w[0].y) -
                     (w[2].x - w[0].x) * (w[1].y - w[0].y);
        bool ccw = area < 0;
        float f = (ccw == g->raster.d.front_ccw) ? 1.0f : 0.0f;

        for (i = 0; i < n; i++) {
            w[i].attr[dc->rs.face_addr] = r5xx_vec4_make(f, 0, 0, 0);
        }
    }
    for (i = 1; i + 1 < n; i++) {
        r5_tri(g, &w[0], &w[i], &w[i + 1]);
        g->prims_n++;
    }
}

static void r5_emit_line(DrawCtx *dc, const GfxVert *a, const GfxVert *b)
{
    GfxVert v[2] = { *a, *b };
    R5xxVertex w[2];
    int p;

    for (p = 0; p < dc->clip.nplanes; p++) {
        float da = plane_dist(dc->clip.plane[p], &v[0].slot[SLOT_POS]);
        float db = plane_dist(dc->clip.plane[p], &v[1].slot[SLOT_POS]);

        if (da < 0 && db < 0) {
            return;
        }
        if (da < 0) {
            vert_lerp(&v[0], &v[0], &v[1], da / (da - db));
        } else if (db < 0) {
            vert_lerp(&v[1], &v[0], &v[1], da / (da - db));
        }
    }
    r5_to_window(dc->s, &dc->rs, &v[0], &w[0]);
    r5_to_window(dc->s, &dc->rs, &v[1], &w[1]);
    r5_line(dc->g, &w[0], &w[1]);
    dc->g->prims_n++;
}

/*
 * Point sprite with GB_ENABLE.POINT_STUFF_ENABLE: stuffed sets run from
 * GA_POINT_S0 (left) to S1 (right) and T0 (bottom) to T1 (top). Drawn
 * as two triangles.
 */
static bool r5_emit_stuffed_point(DrawCtx *dc, const GfxVert *a,
                                  const R5xxVertex *c)
{
    ATIR580State *s = dc->s;
    ATIR580R5 *g = dc->g;
    uint32_t gbe = REG(s, GB_ENABLE);
    float s0 = regf(s, GA_POINT_S0), t0 = regf(s, GA_POINT_T0);
    float s1 = regf(s, GA_POINT_S1), t1 = regf(s, GA_POINT_T1);
    float hw, hh;
    R5xxVertex w[4];
    uint32_t flat;
    bool cf, cb;
    int k, i;

    if (!(gbe & GB_POINT_STUFF_ENABLE) || !(gbe >> 16)) {
        return false;
    }
    if (dc->psize) {
        hw = hh = a->slot[SLOT_PSIZE].f[0] * 0.5f;
    } else {
        hw = g->raster.d.point_half_w;
        hh = g->raster.d.point_half_h;
    }
    for (k = 0; k < 4; k++) {
        GfxVert v = *a;
        bool right = k == 1 || k == 2, bottom = k >= 2;

        for (i = 0; i < 8; i++) {
            uint32_t src = (gbe >> (16 + 2 * i)) & 3;

            if (src == 1 || src == 2) {
                v.slot[SLOT_TEX0 + i] = r5xx_vec4_make(right ? s1 : s0,
                                                       bottom ? t0 : t1,
                                                       0.0f, 1.0f);
            }
        }
        r5_to_window(s, &dc->rs, &v, &w[k]);
        w[k].x = c->x + (right ? hw : -hw);
        w[k].y = c->y + (bottom ? hh : -hh);
    }
    flat = g->raster.flat_inputs;
    g->raster.flat_inputs = 0;
    cf = g->raster.d.cull_front;
    cb = g->raster.d.cull_back;
    g->raster.d.cull_front = g->raster.d.cull_back = false;
    r5_tri(g, &w[0], &w[1], &w[2]);
    r5_tri(g, &w[0], &w[2], &w[3]);
    g->raster.d.cull_front = cf;
    g->raster.d.cull_back = cb;
    g->raster.flat_inputs = flat;
    return true;
}

static void r5_emit_point(DrawCtx *dc, const GfxVert *a)
{
    R5xxVertex w;
    int p;

    for (p = 0; p < dc->clip.nplanes; p++) {
        if (plane_dist(dc->clip.plane[p], &a->slot[SLOT_POS]) < 0) {
            return;
        }
    }
    r5_to_window(dc->s, &dc->rs, a, &w);
    dc->g->prims_n++;
    if (r5_emit_stuffed_point(dc, a, &w)) {
        return;
    }
    r5_point(dc->g, &w, dc->psize, dc->psize ? a->slot[SLOT_PSIZE].f[0] : 0);
}

static const GfxVert *r5_get_vert(ATIR580State *s, ATIR580R5 *g,
                                  const VtxSource *src, uint32_t index)
{
    uint32_t slot = index % VCACHE_SIZE;

    if (g->vcache_tag[slot] != index + 1) {
        r5_vertex(s, g, src, index, &g->vcache[slot]);
        g->vcache_tag[slot] = index + 1;
        g->verts++;
    }
    return &g->vcache[slot];
}

static void r5_run_prims(DrawCtx *dc, uint32_t prim, const VtxSource *src,
                         const uint32_t *idx, uint32_t n)
{
    ATIR580State *s = dc->s;
    ATIR580R5 *g = dc->g;
    GfxVert v[4];
    uint32_t i;

    memset(g->vcache_tag, 0, sizeof(g->vcache_tag));
#define V(k) r5_get_vert(s, g, src, idx ? idx[k] : (k))
    switch (prim) {
    case 1:
        for (i = 0; i < n; i++) {
            r5_emit_point(dc, V(i));
        }
        break;
    case 2:
        for (i = 0; i + 1 < n; i += 2) {
            v[0] = *V(i);
            r5_emit_line(dc, &v[0], V(i + 1));
        }
        break;
    case 3:
    case 12:
        for (i = 0; i + 1 < n; i++) {
            v[0] = *V(i);
            r5_emit_line(dc, &v[0], V(i + 1));
        }
        if (prim == 12 && n > 2) {
            v[0] = *V(n - 1);
            r5_emit_line(dc, &v[0], V(0));
        }
        break;
    case 4:
        for (i = 0; i + 2 < n; i += 3) {
            v[0] = *V(i);
            v[1] = *V(i + 1);
            r5_emit_tri(dc, &v[0], &v[1], V(i + 2));
        }
        break;
    case 5:
    case 15:
        for (i = 1; i + 1 < n; i++) {
            v[0] = *V(0);
            v[1] = *V(i);
            r5_emit_tri(dc, &v[0], &v[1], V(i + 1));
        }
        break;
    case 6:
        for (i = 0; i + 2 < n; i++) {
            if (i & 1) {
                v[0] = *V(i + 1);
                v[1] = *V(i);
            } else {
                v[0] = *V(i);
                v[1] = *V(i + 1);
            }
            r5_emit_tri(dc, &v[0], &v[1], V(i + 2));
        }
        break;
    case 13:
        for (i = 0; i + 3 < n; i += 4) {
            v[0] = *V(i);
            v[1] = *V(i + 1);
            v[2] = *V(i + 2);
            v[3] = *V(i + 3);
            r5_emit_tri(dc, &v[0], &v[1], &v[2]);
            r5_emit_tri(dc, &v[0], &v[2], &v[3]);
        }
        break;
    case 14:
        for (i = 0; i + 3 < n; i += 2) {
            v[0] = *V(i);
            v[1] = *V(i + 1);
            v[2] = *V(i + 3);
            v[3] = *V(i + 2);
            r5_emit_tri(dc, &v[0], &v[1], &v[2]);
            r5_emit_tri(dc, &v[0], &v[2], &v[3]);
        }
        break;
    default:
        r5_err(g, "primitive type %u", prim);
        break;
    }
#undef V
}

/* ---- band workers ---- */

static void r5_replay(ATIR580R5 *g, R5xxRaster *r)
{
    uint32_t i;

    for (i = 0; i < g->nprims; i++) {
        const R5Prim *p = &g->prims[i];

        r->flat_inputs = p->flat_inputs;
        r->d.cull_front = p->cull_front;
        r->d.cull_back = p->cull_back;
        switch (p->kind) {
        case PRIM_TRI:
            r5xx_raster_triangle(r, &p->v[0], &p->v[1], &p->v[2]);
            break;
        case PRIM_POINT:
            r5xx_raster_point(r, &p->v[0]);
            break;
        case PRIM_POINT_SIZED:
            r5xx_raster_point_sized(r, &p->v[0], p->size);
            break;
        case PRIM_LINE:
            r5xx_raster_line(r, &p->v[0], &p->v[1]);
            break;
        }
    }
}

typedef struct R5Worker {
    ATIR580R5 *g;
    int i;
} R5Worker;

static R5Worker r5_workers[R5_MAX_THREADS];

static void *r5_worker(void *arg)
{
    R5Worker *w = arg;
    ATIR580R5 *g = w->g;

    rcu_register_thread();
    for (;;) {
        qemu_sem_wait(&g->go[w->i]);
        if (qatomic_read(&g->quit)) {
            break;
        }
        r5_replay(g, &g->wr[w->i]);
        qemu_sem_post(&g->done);
    }
    rcu_unregister_thread();
    return NULL;
}

/* the vertex stage: window-space primitives into g->prims */
static void r5_collect(DrawCtx *dc, uint32_t prim, const VtxSource *src,
                       const uint32_t *idx, uint32_t n)
{
    ATIR580R5 *g = dc->g;

    g->nprims = 0;
    g->collect = true;
    r5_run_prims(dc, prim, src, idx, n);
    g->collect = false;
}

/* the collected primitives through the CPU rasteriser */
static void r5_draw_cpu(ATIR580R5 *g)
{
    int y0 = g->raster.d.sc_y0, y1 = g->raster.d.sc_y1, h = y1 - y0 + 1;
    int t, nt, band;

    if (!g->nprims) {
        return;
    }
    if (g->nthreads < 2 || h < 64) {
        g->raster.mem = &g->m[0].mem;
        g->raster.us.tex_opaque = &g->m[0];
        r5_replay(g, &g->raster);
        return;
    }
    nt = MIN(g->nthreads, h / 32);
    /* even band height: a 2x2 quad never straddles two bands */
    band = ((h + nt - 1) / nt + 1) & ~1;
    for (t = 0; t < nt; t++) {
        R5xxRaster *r = &g->wr[t];

        *r = g->raster;
        r->mem = &g->m[t + 1].mem;
        r->us.tex_opaque = &g->m[t + 1];
        r->pixels_in = r->pixels_out = r->quads = r->zpass_count = 0;
        r->d.sc_y0 = y0 + t * band;
        r->d.sc_y1 = MIN(y0 + (t + 1) * band - 1, y1);
        if (r->d.sc_y0 > y1) {
            r->d.sc_y0 = y1 + 1;
        }
        qemu_sem_post(&g->go[t]);
    }
    for (t = 0; t < nt; t++) {
        qemu_sem_wait(&g->done);
    }
    for (t = 0; t < nt; t++) {
        g->raster.pixels_in += g->wr[t].pixels_in;
        g->raster.pixels_out += g->wr[t].pixels_out;
        g->raster.quads += g->wr[t].quads;
        g->raster.zpass_count += g->wr[t].zpass_count;
    }
}

/* ---- draws ---- */

static void r5_draw(ATIR580State *s, uint32_t vf, const VtxSource *src,
                    const uint32_t *idx, uint32_t n)
{
    ATIR580R5 *g = s->r5;
    uint32_t prim = VF_PRIM(vf);
    bool bypass = REG(s, VAP_CNTL_STATUS) & VAP_TCL_BYPASS;
    int64_t t0 = get_clock();
    char err[160];
    DrawCtx dc;
    int i, mode;

    g->draws++;
    g->last_vf = vf;
    if (!n || !prim) {
        return;
    }
    if ((!bypass && !r5_update_pvs(s, g)) || !r5_update_us(s, g)) {
        g->skipped++;
        return;
    }
    r5_load_aos(s, g);
    r5_update_textures(s, g);

    memset(&dc, 0, sizeof(dc));
    dc.s = s;
    dc.g = g;
    dc.psize = REG(s, VAP_OUTPUT_VTX_FMT_0) & (1u << 16);
    r5_clip_setup(s, g, &dc.clip);

    r5_raster_regs(s, &g->raster.regs);
    g->raster.mem = &g->m[0].mem;
    g->raster.us.prog = g->us_dec;
    g->raster.us.consts = (const r5xx_vec4 *)g->us_const.f;
    g->raster.us.tex = r5_tex_cb;
    g->raster.us.tex_opaque = &g->m[0];
    r5_rs_setup(s, &dc.rs, &g->raster, prim);
    if (!r5xx_raster_setup(&g->raster, err, sizeof(err))) {
        g->setup_fail++;
        r5_err(g, "raster setup: %s", err);
        return;
    }
    /* COLORENDIAN, like TXO_ENDIAN, swaps only what crosses the bus */
    for (i = 0; i < R5XX_MAX_COLOR_BUFS; i++) {
        uint32_t off;

        if (g->raster.rt.cbuf_enabled[i] &&
            ati_r580_mc_to_vram(s, g->raster.rt.cbuf[i].gpu_addr, &off)) {
            g->raster.rt.cbuf[i].endian = 0;
        }
    }
    if (REG(s, RB3D_COLORPITCH0) & (7u << 16)) {
        g->tiled_cb++;
    }
    if (g->raster.rt.cbuf[0].bpp > 4) {
        g->wide_cb++;
    }
    for (i = 0; i < R5XX_NUM_TEX_UNITS; i++) {
        if (g->samplers[i].valid &&
            g->samplers[i].layout.tiling != R5XX_TILE_LINEAR) {
            g->tiled_tx++;
            break;
        }
    }
    g->last_dst = REG(s, RB3D_COLOROFFSET0);
    g->last_tex0 = REG(s, TX_OFFSET_0);
    g->raster.pixels_in = g->raster.pixels_out = g->raster.quads = 0;
    r5_collect(&dc, prim, src, idx, n);
    mode = R5G_CPU;
    if (s->r5g) {
        R5GpuIn gi = {
            .raster = &g->raster,
            .us = g->us_dec,
            .consts = g->us_const.f,
            .samp = g->samplers,
            .used = r500_us_tex_units_used(g->us_dec),
            .prims = g->prims,
            .nprims = g->nprims,
        };

        for (i = 0; i < R5XX_NUM_TEX_UNITS; i++) {
            if ((gi.used & REG(s, TX_ENABLE) & (1u << i)) &&
                g->samplers[i].valid) {
                gi.bound |= 1u << i;
            }
        }
        mode = ati_r580_r5gpu_draw(s, &gi);
        if (mode == R5G_DONE) {
            g->gpu_draws++;
        }
    }
    if (mode != R5G_DONE) {
        ati_r580_r5gpu_cpu_begin(s, &g->raster);
        r5_draw_cpu(g);
        ati_r580_r5gpu_cpu_end(s);
        if (mode == R5G_VERIFY) {
            ati_r580_r5gpu_verify_end(s);
        }
    }
    {
        unsigned h = g->hist_pos++ % ARRAY_SIZE(g->hist);
        uint32_t a0 = (REG(s, US_CODE_ADDR) & 0x1ff) +
                      (REG(s, US_CODE_OFFSET) & 0x1ff);

        g->hist[h].vf = vf;
        g->hist[h].n = n;
        g->hist[h].dst = REG(s, RB3D_COLOROFFSET0);
        g->hist[h].cpitch = REG(s, RB3D_COLORPITCH0);
        g->hist[h].blend = REG(s, RB3D_BLENDCNTL);
        g->hist[h].ablend = REG(s, RB3D_ABLENDCNTL);
        g->hist[h].out_fmt = REG(s, US_OUT_FMT_0);
        g->hist[h].txen = REG(s, TX_ENABLE);
        g->hist[h].tx[0] = REG(s, TX_FORMAT0_0);
        g->hist[h].tx[1] = REG(s, TX_FORMAT1_0);
        g->hist[h].tx[2] = REG(s, TX_FORMAT2_0);
        g->hist[h].tx[3] = REG(s, TX_OFFSET_0);
        g->hist[h].tx[4] = REG(s, TX_FILTER0_0);
        g->hist[h].us_addr = REG(s, US_CODE_ADDR);
        g->hist[h].us0 = g->us_code[(a0 % R500_US_MAX_INSTR) * 6];
        g->hist[h].us3 = g->us_code[(a0 % R500_US_MAX_INSTR) * 6 + 3];
        g->hist[h].x0 = g->raster.d.sc_x0;
        g->hist[h].y0 = g->raster.d.sc_y0;
        g->hist[h].x1 = g->raster.d.sc_x1;
        g->hist[h].y1 = g->raster.d.sc_y1;
        g->hist[h].px = g->raster.pixels_out;
    }
    if (trace_event_get_state_backends(TRACE_ATI_R580_R5_DRAW)) {
        GString *h = g_string_new(NULL);
        int k;

        g_string_append_printf(h, "vf 0x%08x n %u dst 0x%08x cp 0x%08x "
            "blend 0x%08x/0x%08x px %" PRIu64 " sc %d,%d-%d,%d vte 0x%x "
            "byp %d fmt0 0x%x fmt1 0x%x psc 0x%08x/0x%08x ext 0x%08x/0x%08x "
            "aos %u 0x%08x 0x%08x idxoff 0x%x vport %g %g %g %g "
            "rs %u ip0 0x%08x inst0 0x%08x inst1 0x%08x tx_en 0x%x "
            "outfmt 0x%08x cctl 0x%08x cmask 0x%x ps3 0x%x/0x%08x cc 0x%x",
            vf, n, REG(s, RB3D_COLOROFFSET0), REG(s, RB3D_COLORPITCH0),
            REG(s, RB3D_BLENDCNTL), REG(s, RB3D_ABLENDCNTL),
            g->raster.pixels_out, g->raster.d.sc_x0, g->raster.d.sc_y0,
            g->raster.d.sc_x1, g->raster.d.sc_y1, REG(s, VAP_VTE_CNTL),
            bypass, REG(s, VAP_OUTPUT_VTX_FMT_0), REG(s, VAP_OUTPUT_VTX_FMT_1),
            REG(s, VAP_PROG_STREAM_CNTL_0), REG(s, VAP_PROG_STREAM_CNTL_0 + 4),
            REG(s, VAP_PROG_STREAM_CNTL_EXT_0),
            REG(s, VAP_PROG_STREAM_CNTL_EXT_0 + 4),
            REG(s, VAP_VTX_NUM_ARRAYS), REG(s, VAP_VTX_NUM_ARRAYS + 4),
            REG(s, VAP_VTX_NUM_ARRAYS + 8), REG(s, VAP_INDEX_OFFSET),
            regf(s, SE_VPORT_XSCALE), regf(s, SE_VPORT_XOFFSET),
            regf(s, SE_VPORT_YSCALE), regf(s, SE_VPORT_YOFFSET),
            REG(s, RS_INST_COUNT), REG(s, R500_RS_IP_0), REG(s, R500_RS_INST_0),
            REG(s, R500_RS_INST_0 + 4), REG(s, TX_ENABLE),
            REG(s, US_OUT_FMT_0), REG(s, RB3D_CCTL),
            REG(s, RB3D_COLOR_CHANNEL_MASK), REG(s, GB_PS3_ENABLE),
            REG(s, GA_COLOR_CONTROL_PS3), REG(s, GA_COLOR_CONTROL));
        for (k = 0; k < 2; k++) {
            if (REG(s, TX_ENABLE) & (1u << k)) {
                g_string_append_printf(h, " tx%d %08x/%08x/%08x/%08x/%08x",
                    k, REG(s, TX_FORMAT0_0 + 4 * k), REG(s, TX_FORMAT1_0 + 4 * k),
                    REG(s, TX_FORMAT2_0 + 4 * k), REG(s, TX_OFFSET_0 + 4 * k),
                    REG(s, TX_FILTER0_0 + 4 * k));
            }
        }
        for (k = 0; k < g->num_vbuf && k < 4; k++) {
            g_string_append_printf(h, " vb%d 0x%08x/%u", k, g->vbuf[k].addr,
                                   g->vbuf[k].stride);
        }
        if (idx) {
            g_string_append_printf(h, " idx %u %u %u %u", idx[0],
                                   n > 1 ? idx[1] : 0, n > 2 ? idx[2] : 0,
                                   n > 3 ? idx[3] : 0);
        }
        for (k = 0; k < 8 && (uint32_t)k < n; k++) {
            uint32_t ix = idx ? idx[k] : (uint32_t)k;
            const GfxVert *v = &g->vcache[ix % VCACHE_SIZE];

            if (g->vcache_tag[ix % VCACHE_SIZE] != ix + 1) {
                continue;
            }
            g_string_append_printf(h, " v%u pos %g %g %g %g c0 %g %g %g %g "
                                   "t0 %g %g %g %g", ix,
                                   v->slot[SLOT_POS].f[0], v->slot[SLOT_POS].f[1],
                                   v->slot[SLOT_POS].f[2], v->slot[SLOT_POS].f[3],
                                   v->slot[SLOT_COL0].f[0], v->slot[SLOT_COL0].f[1],
                                   v->slot[SLOT_COL0].f[2], v->slot[SLOT_COL0].f[3],
                                   v->slot[SLOT_TEX0].f[0], v->slot[SLOT_TEX0].f[1],
                                   v->slot[SLOT_TEX0].f[2], v->slot[SLOT_TEX0].f[3]);
        }
        trace_ati_r580_r5_draw(h->str);
        g_string_free(h, true);
    }
    g->px_in += g->raster.pixels_in;
    g->px_out += g->raster.pixels_out;
    g->quads += g->raster.quads;
    for (i = 0; i <= R5_MAX_THREADS; i++) {
        r5_mem_flush(g, &g->m[i]);
    }
    t0 = get_clock() - t0;
    g->draw_ns += t0;
    g->max_draw_ns = MAX(g->max_draw_ns, (uint64_t)t0);
}

static uint32_t r5_num_verts(ATIR580State *s, uint32_t vf)
{
    if (vf & VF_USE_ALT_NUM) {
        return REG(s, VAP_ALT_NUM_VERTICES) & 0xffffff;
    }
    return VF_NUM(vf);
}

static void r5_draw_indexed(ATIR580State *s, uint32_t vf,
                            const uint32_t *words, uint32_t nwords)
{
    uint32_t n = r5_num_verts(s, vf);
    bool i32 = vf & VF_INDEX_32;
    int32_t bias = ((int32_t)(REG(s, VAP_INDEX_OFFSET) << 7)) >> 7;
    uint32_t vmin = REG(s, VAP_VF_MIN_VTX_INDX) & 0xffffff;
    uint32_t vmax = REG(s, VAP_VF_MAX_VTX_INDX) & 0xffffff;
    g_autofree uint32_t *idx = NULL;
    VtxSource src = { 0 };
    uint32_t i;

    n = MIN(n, nwords * (i32 ? 1 : 2));
    idx = g_new(uint32_t, n ? n : 1);
    for (i = 0; i < n; i++) {
        uint32_t v = i32 ? words[i] : (words[i / 2] >> (16 * (i & 1))) & 0xffff;
        int64_t b = (int64_t)v + bias;

        if (vmax && (b < vmin || b > vmax)) {
            b = b < vmin ? vmin : vmax;
        }
        idx[i] = (uint32_t)b;
    }
    r5_draw(s, vf, &src, idx, n);
}

void ati_r580_r5xx_draw_immd(ATIR580State *s, const uint32_t *dw, unsigned n)
{
    VtxSource src = { 0 };
    uint32_t vf = dw[0];

    src.immd = dw + 1;
    src.immd_dwords = n - 1;
    src.vtx_size = REG(s, VAP_VTX_SIZE) & 0x7f;
    if (!src.vtx_size) {
        s->r5->skipped++;
        r5_err(s->r5, "immediate draw with VAP_VTX_SIZE 0");
        return;
    }
    r5_draw(s, vf, &src, NULL,
            MIN(r5_num_verts(s, vf), src.immd_dwords / src.vtx_size));
}

void ati_r580_r5xx_draw_vbuf(ATIR580State *s, uint32_t vf)
{
    VtxSource src = { 0 };

    if (VF_WALK(vf) == WALK_INDICES) {
        s->r5->indx_vf = vf;
        return;
    }
    /* VAP_INDEX_OFFSET biases the generated indices of a vertex list */
    if (REG(s, VAP_INDEX_OFFSET) & 0x1ffffff) {
        int32_t bias = ((int32_t)(REG(s, VAP_INDEX_OFFSET) << 7)) >> 7;
        uint32_t n = r5_num_verts(s, vf), i;
        g_autofree uint32_t *idx = g_new(uint32_t, n ? n : 1);

        for (i = 0; i < n; i++) {
            idx[i] = (uint32_t)((int64_t)i + bias);
        }
        r5_draw(s, vf, &src, idx, n);
        return;
    }
    r5_draw(s, vf, &src, NULL, r5_num_verts(s, vf));
}

void ati_r580_r5xx_draw_indx(ATIR580State *s, const uint32_t *dw, unsigned n)
{
    if (n == 1) {
        s->r5->indx_vf = dw[0];
        return;
    }
    r5_draw_indexed(s, dw[0], dw + 1, n - 1);
}

void ati_r580_r5xx_indx_buffer(ATIR580State *s, uint32_t port, uint32_t addr,
                               uint32_t n)
{
    ATIR580R5 *g = s->r5;
    uint32_t vf = g->indx_vf;
    g_autofree uint32_t *buf = NULL;

    g->indx_vf = 0;
    if (!vf || !n || n > (1u << 22)) {
        g->skipped++;
        return;
    }
    buf = g_new(uint32_t, n);
    ati_r580_mc_read_block(s, addr & ~3u, buf, n);
    r5_draw_indexed(s, vf, buf, n);
}

/* 3D_CLEAR_ZMASK: depth/stencil fast clear to ZB_DEPTHCLEARVALUE */
void ati_r580_r5xx_clear_zmask(ATIR580State *s, uint32_t first, uint32_t n,
                               uint32_t val)
{
    ATIR580R5 *g = s->r5;
    g_autofree R5xxRaster *r = g_new0(R5xxRaster, 1);

    R5xxSurfaceLayout l;
    uint32_t off;
    int x1, y1;

    r5_raster_regs(s, &r->regs);
    r->mem = &g->m[0].mem;
    if (r5xx_raster_zmask_area(r, n, &l, &x1, &y1) &&
        ati_r580_mc_to_vram(s, l.gpu_addr, &off)) {
        ati_r580_r5gpu_zclear(s, off, l.pitch_bytes, l.bpp, x1, y1);
    } else {
        ati_r580_r5gpu_wrote(s, 0, ATI_R580_VRAM_SIZE);
    }
    qatomic_set(&s->r5g_quiet, true);
    r5xx_raster_zmask_clear(r, n, REG(s, ZB_DEPTHCLEARVALUE));
    qatomic_set(&s->r5g_quiet, false);
    r5_mem_flush(g, &g->m[0]);
    g->zclears++;
}

static char *r5_stats_main(ATIR580R5 *g);

char *ati_r580_r5xx_stats(ATIR580State *s)
{
    ATIR580R5 *g = s->r5;
    g_autofree char *m = NULL;
    GString *o;
    unsigned i;

    if (!g) {
        return g_strdup("r500 pipeline off\n");
    }
    m = r5_stats_main(g);
    o = g_string_new(m);
    {
        g_autofree char *gs = ati_r580_r5gpu_stats(s);

        g_string_append(o, gs);
    }
    g_string_append(o, "last draws (vf n dst cpitch blend ablend outfmt "
                       "txen tx0:f0/f1/f2/off/filt0 us_addr us[0] us[3] "
                       "scissor pixels):\n");
    for (i = 0; i < ARRAY_SIZE(g->hist) && i < g->hist_pos; i++) {
        unsigned h = (g->hist_pos - 1 - i) % ARRAY_SIZE(g->hist);

        g_string_append_printf(o, "%08x %u %08x %08x %08x %08x %08x %x "
                               "%08x/%08x/%08x/%08x/%08x %08x %08x %08x "
                               "%d,%d-%d,%d %" PRIu64 "\n",
                               g->hist[h].vf, g->hist[h].n, g->hist[h].dst,
                               g->hist[h].cpitch, g->hist[h].blend,
                               g->hist[h].ablend, g->hist[h].out_fmt,
                               g->hist[h].txen, g->hist[h].tx[0],
                               g->hist[h].tx[1], g->hist[h].tx[2],
                               g->hist[h].tx[3], g->hist[h].tx[4],
                               g->hist[h].us_addr, g->hist[h].us0,
                               g->hist[h].us3, g->hist[h].x0, g->hist[h].y0,
                               g->hist[h].x1, g->hist[h].y1, g->hist[h].px);
    }
    return g_string_free(o, false);
}

static char *r5_stats_main(ATIR580R5 *g)
{
    return g_strdup_printf(
        "r500 draws %" PRIu64 " skipped %" PRIu64 " setup-fail %" PRIu64
        " prims %" PRIu64 " verts %" PRIu64 "\n"
        "us decodes %" PRIu64 " fail %" PRIu64 " flow-control %" PRIu64
        " last-ninst %u\n"
        "pvs decodes %" PRIu64 " fail %" PRIu64 "\n"
        "pixels in %" PRIu64 " out %" PRIu64 " quads %" PRIu64
        " zclears %" PRIu64 "\n"
        "draw ms total %.1f max %.2f threads %d, on the gpu %" PRIu64 "\n"
        "tiled colour buffers %" PRIu64 " tiled textures %" PRIu64
        " wide (64/128-bit) colour buffers %" PRIu64 "\n"
        "last vf 0x%08x dst 0x%08x tex0 0x%08x\n"
        "texture decode failures %" PRIu64 " (unit %u f0 0x%08x f1 0x%08x"
        " f2 0x%08x off 0x%08x filt0 0x%08x), samples of unbound units %"
        PRIu64 "\n"
        "last error: %s\n",
        g->draws, g->skipped, g->setup_fail, g->prims_n, g->verts,
        g->us_decodes, g->us_fail, g->us_fc, g->last_us_n,
        g->pvs_decodes, g->pvs_fail, g->px_in, g->px_out, g->quads,
        g->zclears, g->draw_ns / 1e6, g->max_draw_ns / 1e6, g->nthreads,
        g->gpu_draws,
        g->tiled_cb, g->tiled_tx, g->wide_cb,
        g->last_vf, g->last_dst, g->last_tex0,
        g->tex_bad, g->last_tex_regs[5], g->last_tex_regs[0],
        g->last_tex_regs[1], g->last_tex_regs[2], g->last_tex_regs[3],
        g->last_tex_regs[4], g->tex_unbound, g->last_err);
}

void ati_r580_r5xx_init(ATIR580State *s, unsigned threads)
{
    ATIR580R5 *g = g_new0(ATIR580R5, 1);
    int i;

    g->s = s;
    for (i = 0; i <= R5_MAX_THREADS; i++) {
        g->m[i].s = s;
        g->m[i].dlo = UINT64_MAX;
        g->m[i].mem.opaque = &g->m[i];
        g->m[i].mem.map = r5_mem_map;
        g->m[i].mem.read = r5_mem_read;
        g->m[i].mem.write = r5_mem_write;
        g->m[i].mem.written = r5_mem_written;
        g->m[i].mem.log = r5_mem_log;
    }
    r5xx_linear_hook = r5_linear;
    r5xx_linear_hook_opaque = s;
    s->r5 = g;

    g->nthreads = MIN(threads, R5_MAX_THREADS);
    if (g->nthreads < 2) {
        g->nthreads = 0;
        return;
    }
    qemu_sem_init(&g->done, 0);
    for (i = 0; i < g->nthreads; i++) {
        r5_workers[i].g = g;
        r5_workers[i].i = i;
        qemu_sem_init(&g->go[i], 0);
        qemu_thread_create(&g->th[i], "ati-r580-r5", r5_worker,
                           &r5_workers[i], QEMU_THREAD_JOINABLE);
    }
}

void ati_r580_r5xx_reset(ATIR580State *s)
{
    ATIR580R5 *g = s->r5;

    if (!g) {
        return;
    }
    g->pvs_dirty = g->us_dirty = true;
    g->indx_vf = 0;
}

void ati_r580_r5xx_fini(ATIR580State *s)
{
    ATIR580R5 *g = s->r5;
    int i;

    if (!g) {
        return;
    }
    qatomic_set(&g->quit, true);
    for (i = 0; i < g->nthreads; i++) {
        qemu_sem_post(&g->go[i]);
    }
    for (i = 0; i < g->nthreads; i++) {
        qemu_thread_join(&g->th[i]);
        qemu_sem_destroy(&g->go[i]);
    }
    if (g->nthreads) {
        qemu_sem_destroy(&g->done);
    }
    if (g->pvs_dec) {
        r5xx_pvs_free(g->pvs_dec);
    }
    if (g->us_dec) {
        r500_us_free(g->us_dec);
    }
    g_free(g->prims);
    g_free(g);
    s->r5 = NULL;
}
