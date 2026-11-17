/*
 * ATI Radeon HD 5770 (Juniper) 3D engine model: surfaces, shaders, draws.
 *
 * Nothing here names a device type: the engine sees registers as arrays
 * and memory through HD5770Mem, so the same code runs in the device and
 * on a recorded command stream.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef ATI_HD5770_GFX_H
#define ATI_HD5770_GFX_H

/* ---------------------------------------------------------------- */
/* Memory as the engine sees it (MC addresses)                      */

typedef struct HD5770Mem {
    void *opaque;
    /*
     * Host pointer for MC address mc and the number of bytes valid from
     * it in *len; NULL when nothing is mapped there.
     */
    uint8_t *(*map)(void *opaque, uint64_t mc, uint64_t *len, bool write);
    /* bytes at mc..mc+len-1 were written through a map() pointer */
    void (*dirty)(void *opaque, uint64_t mc, uint64_t len);
} HD5770Mem;

/* one mapped span, the cache of the accessors below */
typedef struct HD5770Span {
    uint64_t lo, hi;            /* MC range [lo, hi) */
    uint8_t *p;                 /* host pointer of lo */
    bool write;
} HD5770Span;

uint8_t *hd5770_span_ptr(HD5770Mem *m, HD5770Span *c, uint64_t mc,
                         unsigned n, bool write);
void hd5770_mem_read(HD5770Mem *m, uint64_t mc, void *buf, uint64_t len);

/* ---------------------------------------------------------------- */
/* Surfaces                                                         */

enum {
    HD5770_ARRAY_LINEAR_GENERAL = 0,
    HD5770_ARRAY_LINEAR_ALIGNED = 1,
    HD5770_ARRAY_1D_TILED_THIN1 = 2,
    HD5770_ARRAY_2D_TILED_THIN1 = 4,
};

typedef struct HD5770Surf {
    uint64_t base;              /* MC address */
    uint32_t pitch;             /* elements per row */
    uint32_t height;            /* rows per slice */
    uint32_t bpe;               /* bytes per element: 1, 2, 4, 8, 16 */
    uint32_t mode;              /* array mode */
    uint32_t banks, bankw, bankh, mtilea;
    uint32_t tile_split;        /* bytes */
    uint32_t pipes;             /* from GB_ADDR_CONFIG */
    uint32_t pipe_interleave;   /* bytes */
    bool nondisp;               /* non-displayable micro tile order */
    /* derived by hd5770_surf_prep() */
    bool fast;                  /* offsets from the tables below */
    uint8_t mt_w_sh, mt_h_sh, pipes_sh, banks_sh, bankw_sh, bankh_sh;
    uint8_t pib_sh, hi_sh;
    uint32_t mt_per_row, tile_bytes, row_bytes;
    uint64_t mt_bytes;
    uint32_t eoff[64];          /* offset of element (x & 7, y & 7) */
    /*
     * 2D tiles larger than the tile split (fast false): split_sh is
     * log2 of the split, the fields above are those of split-sized
     * tiles, eoff the element offsets in the whole tile, and each part
     * of a tile lies split_stride bytes after the previous one
     */
    bool split;
    uint8_t split_sh;
    uint64_t split_stride;
} HD5770Surf;

#define HD5770_GB_ADDR_CONFIG_JUNIPER   0x02010002

bool hd5770_surf_mode_ok(uint32_t mode);
uint64_t hd5770_surf_offset(const HD5770Surf *s, uint32_t x, uint32_t y);
uint64_t hd5770_surf_size(const HD5770Surf *s);
void hd5770_surf_set_addr_config(HD5770Surf *s, uint32_t gb_addr_config);
/* precompute the address tables; after every change of the fields above */
void hd5770_surf_prep(HD5770Surf *s);
/* offset of micro tile (tx, ty) of a prepared 1D/2D surface */
uint64_t hd5770_surf_tile(const HD5770Surf *s, uint32_t tx, uint32_t ty);
/* MC range [*lo, *hi) holding rows y0..y1-1 */
void hd5770_surf_rows(const HD5770Surf *s, uint32_t y0, uint32_t y1,
                      uint64_t *lo, uint64_t *hi);

/* hd5770_surf_offset() of a prepared surface */
static inline uint64_t hd5770_surf_offset_fast(const HD5770Surf *s,
                                               uint32_t x, uint32_t y)
{
    if (!s->fast) {
        return hd5770_surf_offset(s, x, y);
    }
    if (s->mode < HD5770_ARRAY_1D_TILED_THIN1) {
        return (uint64_t)y * s->row_bytes + (uint64_t)x * s->bpe;
    }
    return hd5770_surf_tile(s, x >> 3, y >> 3) + s->eoff[(y & 7) * 8 + (x & 7)];
}

/*
 * Offsets of elements x0..x0+n-1 of row y of a prepared surface; one
 * micro tile address per eight elements.
 */
void hd5770_surf_row(const HD5770Surf *s, uint32_t x0, uint32_t y, uint32_t n,
                     uint64_t *off);

/* ---------------------------------------------------------------- */
/* Formats (FMT_* of resources and CB_COLORn_INFO)                  */

enum {
    HD5770_FMT_8 = 0x01, HD5770_FMT_4_4 = 0x02, HD5770_FMT_3_3_2 = 0x03,
    HD5770_FMT_16 = 0x05, HD5770_FMT_16_FLOAT = 0x06, HD5770_FMT_8_8 = 0x07,
    HD5770_FMT_5_6_5 = 0x08, HD5770_FMT_6_5_5 = 0x09,
    HD5770_FMT_1_5_5_5 = 0x0a, HD5770_FMT_4_4_4_4 = 0x0b,
    HD5770_FMT_5_5_5_1 = 0x0c, HD5770_FMT_32 = 0x0d,
    HD5770_FMT_32_FLOAT = 0x0e, HD5770_FMT_16_16 = 0x0f,
    HD5770_FMT_16_16_FLOAT = 0x10, HD5770_FMT_8_24 = 0x11,
    HD5770_FMT_24_8 = 0x13, HD5770_FMT_10_11_11 = 0x15,
    HD5770_FMT_11_11_10 = 0x17, HD5770_FMT_2_10_10_10 = 0x19,
    HD5770_FMT_8_8_8_8 = 0x1a, HD5770_FMT_10_10_10_2 = 0x1b,
    HD5770_FMT_32_32 = 0x1d, HD5770_FMT_32_32_FLOAT = 0x1e,
    HD5770_FMT_16_16_16_16 = 0x1f, HD5770_FMT_16_16_16_16_FLOAT = 0x20,
    HD5770_FMT_32_32_32_32 = 0x22, HD5770_FMT_32_32_32_32_FLOAT = 0x23,
    HD5770_FMT_32_32_32_FLOAT = 0x30,
};

enum {
    HD5770_NUM_NORM = 0, HD5770_NUM_INT = 1, HD5770_NUM_SCALED = 2,
};

/* bytes per element of a format, 0 if not modelled */
unsigned hd5770_fmt_bytes(unsigned fmt);
/* number of components of a colour format */
unsigned hd5770_fmt_comps(unsigned fmt);
/*
 * Unpack one element into components 0..3 (memory order) as raw
 * integers; *bits gets the width of each component.
 */
void hd5770_fmt_unpack(unsigned fmt, const uint8_t *p, uint32_t c[4],
                       unsigned bits[4]);
void hd5770_fmt_pack(unsigned fmt, uint8_t *p, const uint32_t c[4]);
bool hd5770_fmt_is_float(unsigned fmt);

/* ---------------------------------------------------------------- */
/* Statistics and gaps of the 3D engine                             */

#define HD5770_SHADER_SLOTS     96

typedef struct HD5770ShaderInfo {
    uint64_t addr;              /* MC address of the program */
    uint32_t hash;
    uint32_t dwords;
    uint8_t stage;              /* 0 VS, 1 PS, 2 FS */
    uint64_t uses;
    uint16_t alu_ops[64];       /* distinct ALU opcodes, 0x800 | OP3 */
    uint8_t n_alu_ops;
    uint8_t cf_ops[16];
    uint8_t n_cf_ops;
    uint8_t tex_ops, vtx_ops;
    bool bad;                   /* holds something not modelled */
} HD5770ShaderInfo;

typedef struct HD5770GfxStats {
    uint64_t draws, drawn, skipped, prims, pixels, verts;
    uint64_t skip_reason[16];
    uint64_t gap_alu_op2[256], gap_alu_op3[32], gap_cf[256];
    uint64_t gap_tex[32], gap_vtx[32], gap_fmt[64], gap_tex_fmt[64];
    uint64_t gap_vtx_fmt[64], gap_mode[16], gap_prim[64], gap_misc[32];
    uint64_t gap_export[8], gap_src_sel;
    uint64_t blend_on, tex_fetch, vtx_fetch, kills;
    uint64_t culled, db_tests, db_fails, resolves, lod_fetches, clipped;
    uint64_t rt_fmt[64];
    uint64_t ns;                /* host time in draws */
    HD5770ShaderInfo sh[HD5770_SHADER_SLOTS];
    uint32_t n_sh;
    uint64_t sh_other;
} HD5770GfxStats;

enum {
    HD5770_SKIP_NO_PS, HD5770_SKIP_NO_VS, HD5770_SKIP_PRIM,
    HD5770_SKIP_RT, HD5770_SKIP_SHADER, HD5770_SKIP_NOCOLOR,
    HD5770_SKIP_DISABLED, HD5770_SKIP_INDEX, HD5770_SKIP_MAX,
};

enum {
    HD5770_GAP_DEPTH, HD5770_GAP_MRT, HD5770_GAP_SCISSOR_RECTS,
    HD5770_GAP_ROP, HD5770_GAP_BLEND_FUNC, HD5770_GAP_KCACHE,
    HD5770_GAP_LOOP, HD5770_GAP_REL, HD5770_GAP_TEX_DIM,
    HD5770_GAP_PRED, HD5770_GAP_DECODE, HD5770_GAP_MSAA, HD5770_GAP_CLIP,
    HD5770_GAP_MIP, HD5770_GAP_FACE, HD5770_GAP_POLY_OFFSET,
    HD5770_GAP_MISC_MAX,
};

/* ---------------------------------------------------------------- */
/* The engine                                                       */

#define HD5770_ALU_CONST_DWORDS 4096

typedef struct HD5770ShaderCache HD5770ShaderCache;
typedef struct HD5770Gpu HD5770Gpu;

enum {
    HD5770_GPU_OFF,             /* every draw on the CPU */
    HD5770_GPU_ON,              /* draws the backend takes on the host GPU */
    HD5770_GPU_VERIFY,          /* both, compared; VRAM as the CPU leaves it */
};

#define HD5770_PENDING          16

typedef struct HD5770GpuStats {
    uint64_t offered, taken, fallback[8], taken_db;
    uint64_t verified, vdiff_draws, vdiff_bytes, vdiff_big, vmax;
    uint64_t syncs, touch_syncs;
    uint64_t build_skips;       /* left out while their program builds */
    /*
     * verify mode: the host GPU's vertex records (HD5770GpuVs) against
     * the CPU's vertices: draws, records, records and words differing,
     * the largest difference in units in the last place
     */
    uint64_t vs_draws, vs_recs, vs_bad, vs_bad_words, vs_ulp;
    /*
     * the same for the triangles set up from them, and the draws whose
     * triangle count differs
     */
    uint64_t vs_tri_draws, vs_tris, vs_tri_bad, vs_tri_ulp, vs_tri_count;
    /* the CPU's time and pixels for the draws it took, by reason */
    uint64_t fb_ns[8], fb_px[8];
} HD5770GpuStats;

typedef struct HD5770Gfx {
    const uint32_t *ctx;        /* register file from 0x28000 */
    const uint32_t *cfg;        /* MMIO register file from 0 */
    const uint32_t *alu;        /* SET_ALU_CONST file, HD5770_ALU_CONST_DWORDS */
    HD5770Mem mem;
    HD5770GfxStats *st;
    HD5770ShaderCache *cache;
    /* host GPU backend (ati_hd5770_gpu.h), NULL for none */
    HD5770Gpu *gpu;
    int gpu_mode;
    uint8_t *vram;              /* host memory of VRAM */
    uint64_t vram_size;
    /* guest RAM for surfaces outside VRAM: ram_for() attaches its block */
    uint8_t *ram;
    uint64_t ram_size;
    bool (*ram_for)(struct HD5770Gfx *g, uint8_t *host);
    bool ram_pend;              /* queued draws use guest RAM */
    bool ram_wpend;             /* queued draws write guest RAM */
    /* VRAM ranges queued draws read or write, [lo, hi) */
    uint64_t pend[HD5770_PENDING][2];
    unsigned npend;
    HD5770GpuStats gst;
    /* serialises the backend between threads; recursive, may be NULL */
    void (*lock)(struct HD5770Gfx *g, bool on);
    void *lock_opaque;
    /* draw state the CP takes from packets (INDEX_TYPE, INDEX_BASE, ...) */
    uint32_t index_type;        /* bit 0 32-bit indices, bits 3:2 swap */
    uint64_t index_base;        /* MC address */
    uint32_t index_size;        /* INDEX_BUFFER_SIZE, indices */
    uint32_t num_instances;     /* 0 counts as 1 */
    /* DRAW_INDIRECT / DRAW_INDEX_INDIRECT: SET_BASE, and the draw's data */
    uint64_t ind_base;
    bool ind_draw;
    uint32_t ind_ninst, ind_vtx, ind_inst;
    /*
     * Border colours of samplers with BORDER_COLOR_TYPE register, per
     * stage (PS, VS, GS) and sampler, as TD_*_SAMPLER0_BORDER_INDEX
     * selects them
     */
    uint32_t border[3][32][4];
} HD5770Gfx;

/* a write to a TD_*_SAMPLER0_BORDER_* register (0xa400..0xa43c) */
void hd5770_gfx_border(HD5770Gfx *g, const uint32_t *regs, uint32_t off,
                       uint32_t val);

void hd5770_gfx_init(HD5770Gfx *g);
void hd5770_gfx_fini(HD5770Gfx *g);
/* everything the backend has queued is in VRAM */
void hd5770_gfx_sync(HD5770Gfx *g);
/* draws were queued since the last sync */
bool hd5770_gfx_busy(HD5770Gfx *g);
/* a fence the backend completes (FENCES in ati_hd5770_gpu.h): its ticket */
uint64_t hd5770_gfx_fence(HD5770Gfx *g);
/* about to access VRAM bytes [off, off + len) on the CPU */
void hd5770_gfx_touch(HD5770Gfx *g, uint64_t off, uint64_t len);
/* whether queued draws may write VRAM bytes [off, off + len) */
bool hd5770_gfx_pending(HD5770Gfx *g, uint64_t off, uint64_t len);
void hd5770_gfx_gpu_stats(HD5770Gfx *g, GString *out);
/* a draw packet, b[0] its header, n dwords */
void hd5770_gfx_draw(HD5770Gfx *g, const uint32_t *b, uint32_t n);
/* a packet that sets draw state; false when op is not one of them */
bool hd5770_gfx_state_packet(HD5770Gfx *g, const uint32_t *b, uint32_t n);
void hd5770_gfx_stats(HD5770Gfx *g, GString *out);
void hd5770_gfx_gaps(HD5770Gfx *g, GString *out);
void hd5770_gfx_dump_shaders(HD5770Gfx *g, GString *out);

#endif
