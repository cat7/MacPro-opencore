/*
 * ATI Radeon HD 5770 (Juniper) 3D engine on the host's own pipeline
 * (Metal render passes), gpu=native.
 *
 * A draw's triangles go through the host rasteriser with the pixel
 * program as a fragment function (hd5770_msl_native_fs()); the vertex
 * stage stays on the CPU. Results are close to the CPU path's, not bit
 * for bit. Whatever is not expressible is declined and drawn on the CPU.
 *
 * SURFACES. Targets and textures are host textures holding the bytes of
 * the guest surface (RGBA8, R8 or R16 by the element size: byte i of an
 * element in channel i, a 16-bit element in one; RG8 for 8_8,
 * RGBA16Float for 16_16_16_16_FLOAT), in bands of rows: a
 * band is held once a draw needs it (a texture's rows, the rows a draw
 * covers of its target), with a shadow of its bytes. At a band's first
 * use after each sync, and after wrote() or a write-back over its
 * memory, the pages of VRAM that the device's write log
 * (HD5770Gpu.written), wrote() or a write-back named since the band was
 * last compared are compared with the shadow, and the band is uploaded
 * if one differs; without a log, and in guest RAM, every page is
 * compared. The rectangles drawn are written back to memory and the
 * shadow at the next sync, or before another entry over the same bytes
 * is used. At most one of the entries over a byte is newer than memory.
 *
 * FENCES (ati_hd5770_gpu.h). fence() commits the passes with a copy of
 * the rectangles drawn to a read buffer of its own and returns; the
 * command buffer's completed handler copies them to memory and the
 * shadows, reports the ranges and retires the ticket. Until then the
 * bands under the rectangles are in flight: memory and shadow both hold
 * what they held before the draws, so a band in flight is not compared
 * for upload; if its pages differ all the same (a guest write), or
 * another entry over its bytes is used, the fences are completed first.
 * sync() completes every fence, in order, before its own write-back.
 * lk guards the entries, shadows and fence queue against the handlers
 * and is released while waiting for the GPU; rlock keeps the retired()
 * calls in order.
 *
 * DEPTH AND STENCIL. The depth plane and the stencil plane are entries
 * of their own, host depth and stencil textures (Z_16: Depth16Unorm;
 * Z_24 and Z_32_FLOAT: Depth32Float; stencil: Stencil8) attached to
 * the pass separately (both in one draw: Apple GPUs only, else
 * declined), kept like the colour entries: the shadow holds
 * the guest's bytes; Z_24 is converted when a band is uploaded and when
 * it is written back, the byte above the 24 bits taken from memory.
 * The host tests the fragment's depth unrounded against a plane that
 * holds unrounded depths where it drew.
 *
 * Accepted: one colour target (8_8_8_8, 8_8 or 8 UNORM,
 * 16_16_16_16_FLOAT, 8 or 32 integer) in
 * VRAM or guest RAM, or none; depth and stencil planes in VRAM with
 * tests, writes, two-sided stencil and the clears; programs in the
 * straight form with texture fetches, RESINFO and vertex fetches of 8
 * and 32 bit elements; 2D 8_8_8_8, 8_8, 8 and 16 UNORM and
 * 16_16_16_16_FLOAT textures, mipmapped
 * ones with each level an entry; clip rectangles. Builds are
 * synchronous.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/timer.h"
#include "qemu/bswap.h"
#ifdef __aarch64__
#include <arm_neon.h>
#endif
#import <Metal/Metal.h>
#include "ati_hd5770_gpu.h"

#define NAT_RING        (48u << 20)
#define NAT_RINGS       3
#define NAT_READ        (32u << 20)     /* write-backs of one wait */
/* textures and shadows: this and twice the VRAM */
#define NAT_CACHE_BASE  (512ull << 20)
#define NAT_IDLE        256             /* syncs an unused entry stays */

enum {
    ND_TAKEN, ND_EMPTY, ND_RT_FMT, ND_RT_SURF, ND_STATE_DB, ND_DB_SURF,
    ND_BLEND_ORDER, ND_PROG_FORM, ND_PROG_EXPORTS,
    ND_PROG_INPUTS, ND_PROG_VTX, ND_PROG_EMIT, ND_PROG_BUILD,
    ND_TEX_KIND, ND_TEX_FMT, ND_TEX_SURF, ND_TEX_TARGET, ND_W, ND_ROOM,
    ND_MRT, ND_RESOLVE, ND_ZEXPORT, ND_RT_WIDE, ND_FETCH_OP, ND_TEX_BC,
    ND_TEX_3D, ND_TEX_CUBE, ND_TEX_ARRAY, ND_TEX_MANY, ND_TEX_UNITS, ND_VS,
    ND_N
};

static const char *const nd_names[ND_N] = {
    "taken", "nothing to draw", "target format",
    "target surface", "stencil export", "depth surface",
    "blend not on 8_8_8_8 alpha last", "program form", "program exports",
    "program inputs", "vertex fetch format", "program not expressible",
    "program build failed", "texture kind", "texture format",
    "texture surface", "texture is the target", "1/w", "room",
    "more render targets", "resolve", "depth export",
    "target over 8 bytes", "fetch operation", "BC texture", "3D texture",
    "cube texture", "array texture", "texture levels and slices",
    "texture units with levels",
    "vertex stage",
};

/* the layouts of hd5770_msl_native_vs() */
typedef struct GBuf {
    uint32_t voff, space, pt_off, pg0;
} GBuf;

typedef struct GVsVb {
    uint32_t id, pad0, pad1, pad2;
    uint32_t res[8];
    GBuf b;
} GVsVb;

typedef struct GVsDraw {
    uint32_t nslot, ninst, rec_words, np;
    uint32_t kv_off, cfile_off, nvb, base_vtx;
    uint32_t start_inst, idx_kind, ib_i32, ib_swap;
    uint32_t ib_max, ioff, lo, hi;
    uint32_t restart, reset, imm_off, pad;
    GBuf ib;
    GVsVb vb[8];
} GVsDraw;

typedef struct NatVs {
    GVsDraw v;
    uint32_t vte, sc_mode, wofs, clip;
    float vport[6];
    float vpx, vpy;
    float pc, zoff, pad[2];
    float def[32][4];
} NatVs;

/* the uniforms of hd5770_msl_native_fs() */
typedef struct NatU {
    float tsize[16][4];
    uint32_t tsel[16][4];
    uint32_t twrap[16][4];
    float tborder[16][4];
    float vp[4];
    uint32_t vbi[HD5770_GPU_VB][4];
    int32_t cr[4][4];
    float alpha_ref;
    uint32_t alpha_func;
    float pc;
    uint32_t persp;
    uint32_t cliprule;
    uint32_t pad[3];
    float tlod[16][4];          /* bias, lowest, highest */
    uint32_t tmip[16][4];       /* mip filter, last level, mag, min linear */
    uint32_t tres[16][4];       /* base level, levels, depth */
    float lsize[HD5770_NAT_LEVELS][4];
    float zr[4];                /* the depth range */
    uint32_t tcmp[16][4];       /* comparison + 1, UNORM steps, Z linear, Z clamp */
    uint32_t lconst[32];
    uint32_t bconst, zero, pad3[2];
} NatU;

/* host formats: the element's bytes */
enum {
    NF_RGBA8, NF_R8, NF_R16, NF_D16, NF_D24, NF_D32, NF_S8, NF_RGBA16F, NF_RG8,
    NF_RGBA32F, NF_R32F, NF_RGBA16, NF_ABGR4,
    NF_BC1, NF_BC2, NF_BC3, NF_BC4, NF_BC4S, NF_BC5, NF_BC5S,
    NF_N
};

static const MTLPixelFormat nf_fmt[NF_N] = {
    MTLPixelFormatRGBA8Unorm, MTLPixelFormatR8Unorm, MTLPixelFormatR16Unorm,
    MTLPixelFormatDepth16Unorm, MTLPixelFormatDepth32Float,
    MTLPixelFormatDepth32Float, MTLPixelFormatStencil8,
    MTLPixelFormatRGBA16Float, MTLPixelFormatRG8Unorm,
    MTLPixelFormatRGBA32Float, MTLPixelFormatR32Float,
    MTLPixelFormatRGBA16Unorm, MTLPixelFormatABGR4Unorm,
    MTLPixelFormatBC1_RGBA, MTLPixelFormatBC2_RGBA, MTLPixelFormatBC3_RGBA,
    MTLPixelFormatBC4_RUnorm, MTLPixelFormatBC4_RSnorm,
    MTLPixelFormatBC5_RGUnorm, MTLPixelFormatBC5_RGSnorm,
};
static const uint8_t nf_bpe[NF_N] = {
    4, 1, 2, 2, 4, 4, 1, 8, 2, 16, 4, 8, 2, 8, 16, 16, 8, 8, 16, 16,
};
/* texels along an element: the BC formats' elements are 4x4 blocks */
#define NF_BLK(hf)      ((hf) >= NF_BC1 ? 4u : 1u)

#define NAT_BAND        32      /* rows compared and uploaded together */
#define NAT_HASH        4096
#define NAT_RECTS       4

typedef struct NatSurf {
    HD5770Surf s;
    uint8_t space, hf;
    uint64_t voff;
    uint32_t *pages;
    uint32_t npages, pg0;
    uint32_t pmin, pmax;        /* its lowest and highest page (space 1) */
    uint64_t lo, hi;            /* its bytes in VRAM (space 0) */
    uint64_t cov;               /* bytes from the base its rows lie in */
    id<MTLTexture> tex;
    uint8_t *shadow;            /* the bytes tex holds, in the bands held */
    uint32_t bh, nb;            /* rows of a band, bands */
    /*
     * per band: tex and shadow hold it; the write generation and the
     * stamp it was last compared with memory at
     */
    uint8_t *held;
    uint64_t *bgen, *bstamp;
    uint16_t *fly;              /* per band: fences in flight over it */
    uint64_t *fstamp;           /* the stamp it was compared at in flight */
    uint32_t nfly;              /* rectangles in flight */
    uint64_t hstamp;            /* the stamp the write log was read at */
    uint8_t *amiss;             /* per band: the audit reported it */
    uint64_t pin;               /* the draw that uses it */
    uint64_t pass;              /* the last pass that uses it */
    uint8_t run;                /* elements of a micro tile row in a run */
    /*
     * split tiles in VRAM: np parts; element i of a tile lies spg[i]
     * after the first byte of part spp[i], which is element spj[] of it;
     * spf: the parts' offsets + 1 per tile, once asked for
     */
    uint8_t np, spp[64], spj[32];
    uint32_t spg[64];
    uint64_t *spf;
    /* a depth plane: vertex stages of its draws since its clear */
    uint8_t vsm;                /* bit 0 the CPU's, bit 1 the host's */
    bool dirty;                 /* tex is newer than memory */
    bool recent;                /* drawn since the last fence */
    bool kept;                  /* stays out of the fence being made */
    int64_t tdraw;              /* when it was last drawn */
    int nr, r[NAT_RECTS][4];    /* where: x0, y0, x1, y1 */
    size_t ro[NAT_RECTS];       /* their pixels in the read buffer */
    uint64_t used;
    struct NatSurf *hnext;      /* the next entry of its hash chain */
    bool act;                   /* listed in HD5770Mtlr.act */
} NatSurf;

/* a rectangle a fence writes back */
typedef struct NatRect {
    NatSurf *e;
    int r[4];
    size_t ro;                  /* its pixels in the fence's buffer */
} NatRect;

/* FENCES above */
typedef struct NatFence {
    uint64_t ticket;
    id<MTLCommandBuffer> cb;    /* nil: nothing to wait for */
    id<MTLBuffer> buf;
    bool gpu_done, copied;
    uint64_t passes;            /* the passes up to it */
    unsigned n;
    NatRect *rc;
} NatFence;

typedef struct NatProg {
    id<MTLFunction> vfn, ffn;
} NatProg;

/* what a wait for the GPU is for */
enum {
    NW_UPLOAD,                  /* a pass in flight uses the texture */
    NW_RING,
    NW_SYNC,                    /* write-back of everything */
    NW_ALIAS,                   /* write-back for another view of the bytes */
    NW_EVICT,
    NW_RANGE,
    NW_N,
};

static const char *const nw_names[NW_N] = {
    "upload", "ring", "sync", "alias", "evict", "range",
};

typedef struct HD5770Mtlr {
    HD5770Gpu pub;
    id<MTLDevice> dev;
    id<MTLCommandQueue> q;
    uint8_t *vram, *ram;
    uint64_t vram_size, ram_size;
    id<MTLBuffer> ring, rbuf;
    /* ring is rings[ring_i]; ring i is read by passes up to ring_pass[i] */
    id<MTLBuffer> rings[NAT_RINGS];
    uint64_t ring_pass[NAT_RINGS];
    unsigned ring_i;
    id<MTLBuffer> bvram, bram;  /* memory in place, for vertex stages */
    size_t ring_at;
    GHashTable *vprogs;
    uint64_t vs_draws, vs_tris, vs_builds, vs_build_ns;
    uint64_t vs_mixed[2];       /* CPU stage after host's, host's after CPU */
    id<MTLSamplerState> samp[2];
    GHashTable *progs, *psos, *dss;
    id<MTLDepthStencilState> ds_off;
    bool sep_ds;                /* depth and stencil attach separately */
    bool bc_ok;                 /* BC formats are textures */
    bool pk_ok;                 /* 4_4_4_4 is a format */
    bool zlazy;                 /* planes drawn stay out of a fence */
    bool clazy;                 /* colour targets drawn, but the scanout's */
    bool ztex;                  /* a plane of floats is read as it is */
    int64_t lazy_ns;            /* a surface kept stays out this long */
    int64_t now;                /* the draw's or the fence's time */
    GPtrArray *surfs;
    NatSurf *hb[NAT_HASH];      /* the entries by where they start */
    GPtrArray *act;             /* the entries newer than memory or in flight */
    GArray *ranges;
    id<MTLCommandBuffer> last;
    id<MTLCommandBuffer> cb;
    id<MTLRenderCommandEncoder> enc;
    NatSurf *cur, *cur_z, *cur_s;   /* the pass's attachments */
    NatSurf *curm[7];           /* its colour buffers after the first */
    unsigned ncurm;
    bool pass_on;
    uint32_t vw, vh;            /* its size */
    uint64_t epoch;
    /* per VRAM page: the write generation it was last named at */
    uint64_t *pgen, gen;
    /* changes when memory may have: bands are compared once per stamp */
    uint64_t stamp[2];          /* VRAM, guest RAM */
    uint64_t bytes, seq;        /* of the entries; the draw's number */
    uint64_t cache_max, bytes_most, roomed;
    uint64_t lv_cut;            /* levels left out: over the host's count */
    uint64_t pass_done;         /* passes up to this one are complete */
    uint64_t cb_passes;         /* the same, from the completed handlers */
    GMutex lk, rlock;
    GQueue fences;              /* NatFence, oldest first, not retired */
    GPtrArray *bufs;            /* read buffers not in use */
    uint64_t tickets;
    int cbs_out;                /* command buffers committed, not complete */
    uint8_t *tmp, *flags;
    size_t tmp_len, flags_len;
    /* statistics */
    uint64_t nd[ND_N], tris, passes, waits, uploads, upload_bytes;
    uint64_t wbacks, wback_bytes, checks, cmp_bytes, syncs, first_bytes;
    uint64_t builds, pso_builds, build_ns, draw_build_ns;
    uint64_t nwait[NW_N], ring_swaps, up_after;
    int64_t wait_by_ns[NW_N];
    int64_t draw_ns, sync_ns, wait_ns, check_ns, upload_ns, wback_ns;
    uint64_t nfences, fence_empty, fence_now, drains, fly_cmp, holds;
    uint64_t fence_wait;        /* fences that only await passes */
    uint64_t fence_own;         /* of them with their own command buffer */
    uint64_t kept_rects, rsyncs;
    int64_t fence_ns, drain_ns, fcopy_ns;
    GArray *build_ms;
    /*
     * BUILD AHEAD (ati_hd5770_gfx.h): dry is set while draws are only
     * looked at; jl, jp: the libraries and pipelines noted (NatJob), jk
     * their keys; dp: the pass as it will be then
     */
    bool dry, dp_set;
    struct {
        bool on;
        uint64_t rt, z, s;
        unsigned zf, sf, nt;
    } dp;
    GPtrArray *jl, *jp;
    GHashTable *jk;
    int job_next, job_end;
    GPtrArray *job_run;
    uint64_t ahead, ahead_draws, ahead_libs, ahead_psos;
    int64_t ahead_ns, ahead_lib_ns;
    /* HD5770_NAT_AUDIT */
    bool audit;
    uint64_t au_bands, au_bytes, au_named, au_ram, au_missed, au_pages;
    uint64_t au_events, au_lines, au_db;
    int64_t au_ns;
} HD5770Mtlr;

static void nat_exact_math(MTLCompileOptions *o)
{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#if defined(MAC_OS_VERSION_15_0) && \
    MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_VERSION_15_0
    if (@available(macOS 15.0, *)) {
        o.mathMode = MTLMathModeSafe;
        o.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    } else {
        o.fastMathEnabled = NO;
    }
#else
    o.fastMathEnabled = NO;
#endif
#pragma clang diagnostic pop
    /*
     * the lowest version with barycentric_coord: later ones spend 50 ms
     * per library on the texture types
     */
    o.languageVersion = MTLLanguageVersion2_2;
}

static bool nat_decline(HD5770Mtlr *m, unsigned r, unsigned why)
{
    if (m->dry) {
        return false;
    }
    m->nd[r]++;
    m->pub.why = why;
    return false;
}

#define NAT_CBS_MAX     256             /* the queue's command buffers */
#define NAT_CBS_OUT     200

/*
 * Committed and counted until complete. Metal's completed handlers run
 * one after another and a queue gives out NAT_CBS_MAX command buffers at
 * a time: a handler waiting for lk while its holder waits for a command
 * buffer would not end. nat_hold() keeps the count low before lk is taken.
 */
static void nat_commit(HD5770Mtlr *m, id<MTLCommandBuffer> cb)
{
    uint64_t passes = m->passes;

    qatomic_inc(&m->cbs_out);
    [cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
        /* the handlers run one after another, in the order committed */
        qatomic_set(&m->cb_passes, passes);
        qatomic_dec(&m->cbs_out);
    }];
    [cb commit];
}

static void nat_hold(HD5770Mtlr *m)
{
    if (qatomic_read(&m->cbs_out) > NAT_CBS_OUT) {
        m->holds++;
        while (qatomic_read(&m->cbs_out) > NAT_CBS_OUT) {
            g_usleep(20);
        }
    }
}

static void nat_end_pass(HD5770Mtlr *m)
{
    if (m->enc) {
        [m->enc endEncoding];
        [m->enc release];
        m->enc = nil;
        nat_commit(m, m->cb);
        [m->last release];
        m->last = m->cb;
        m->cb = nil;
    }
    m->cur = m->cur_z = m->cur_s = NULL;
    m->ncurm = 0;
    m->pass_on = false;
}

/* everything encoded is done; lk is released meanwhile */
static void nat_wait(HD5770Mtlr *m, unsigned why)
{
    nat_end_pass(m);
    if (m->last) {
        int64_t t0 = get_clock();
        id<MTLCommandBuffer> last = m->last;

        m->last = nil;
        g_mutex_unlock(&m->lk);
        [last waitUntilCompleted];
        g_mutex_lock(&m->lk);
        [last release];
        m->waits++;
        m->wait_ns += get_clock() - t0;
        m->nwait[why]++;
        m->wait_by_ns[why] += get_clock() - t0;
    }
    m->pass_done = m->passes;
    m->ring_at = 0;
}

static void *nat_tmp(HD5770Mtlr *m, size_t n)
{
    if (m->tmp_len < n) {
        m->tmp = g_realloc(m->tmp, n);
        m->tmp_len = n;
    }
    return m->tmp;
}

/* surface bytes from o on in memory: *n of them are contiguous there */
static inline uint8_t *nat_at(HD5770Mtlr *m, const NatSurf *e, uint64_t o,
                              size_t *n)
{
    if (e->space) {
        uint64_t q = o + e->pg0;

        *n = 4096 - (q & 4095);
        return m->ram + e->pages[q >> 12] + (q & 4095);
    }
    *n = e->cov - o;
    return m->vram + e->voff + o;
}

static void nat_get(HD5770Mtlr *m, const NatSurf *e, uint64_t o, uint8_t *dst,
                    size_t n)
{
    while (n) {
        size_t k;
        uint8_t *p = nat_at(m, e, o, &k);

        k = MIN(k, n);
        memcpy(dst, p, k);
        dst += k;
        o += k;
        n -= k;
    }
}

/* to memory and to the shadow */
static void nat_put(HD5770Mtlr *m, const NatSurf *e, uint64_t o,
                    const uint8_t *src, size_t n)
{
    memcpy(e->shadow + o, src, n);
    while (n) {
        size_t k;
        uint8_t *p = nat_at(m, e, o, &k);

        k = MIN(k, n);
        memcpy(p, src, k);
        src += k;
        o += k;
        n -= k;
    }
}

static inline void nat_cp(uint8_t *d, const uint8_t *s, size_t n)
{
    switch (n) {
    case 1: *d = *s; break;
    case 2: memcpy(d, s, 2); break;
    case 4: memcpy(d, s, 4); break;
    case 8: memcpy(d, s, 8); break;
    case 16: memcpy(d, s, 16); break;
    case 32: memcpy(d, s, 32); break;
    default: memcpy(d, s, n); break;
    }
}

/*
 * elements x0..x0+n-1 of row y from memory (put false) or to memory and
 * the shadow
 */
static void nat_row(HD5770Mtlr *m, const NatSurf *e, uint32_t x0, uint32_t y,
                    uint32_t n, uint8_t *buf, bool put)
{
    const HD5770Surf *s = &e->s;
    uint32_t x = x0, end = x0 + n, bpe = s->bpe;
    uint8_t *mem = m->vram + e->voff, *sh = e->shadow;

    if (!s->fast) {
        /* tiles in split pieces: element by element */
        for (; x < end; x++, buf += bpe) {
            uint64_t o = hd5770_surf_offset(s, x, y);

            if (put) {
                nat_put(m, e, o, buf, bpe);
            } else {
                nat_get(m, e, o, buf, bpe);
            }
        }
        return;
    }
    if (s->mode < HD5770_ARRAY_1D_TILED_THIN1) {
        uint64_t o = (uint64_t)y * s->row_bytes + (uint64_t)x * bpe;

        if (put) {
            nat_put(m, e, o, buf, (size_t)n * bpe);
        } else {
            nat_get(m, e, o, buf, (size_t)n * bpe);
        }
        return;
    }
    while (x < end) {
        uint64_t t = hd5770_surf_tile(s, x >> 3, y >> 3);
        const uint32_t *eo = &s->eoff[(y & 7) * 8];
        uint32_t stop = MIN(end, (x | 7) + 1), k;

        for (; x < stop; x += k, buf += k * bpe) {
            /* to the end of the run, or of the row */
            k = MIN(e->run - (x & (e->run - 1)), stop - x);
            if (e->space) {
                if (put) {
                    nat_put(m, e, t + eo[x & 7], buf, (size_t)k * bpe);
                } else {
                    nat_get(m, e, t + eo[x & 7], buf, (size_t)k * bpe);
                }
            } else if (put) {
                nat_cp(mem + t + eo[x & 7], buf, (size_t)k * bpe);
                nat_cp(sh + t + eo[x & 7], buf, (size_t)k * bpe);
            } else {
                nat_cp(buf, mem + t + eo[x & 7], (size_t)k * bpe);
            }
        }
    }
}

enum { NIO_GET, NIO_PUT, NIO_PUT_Z24 };

/*
 * Rectangle x0, y0, w, h of an entry in micro tiles in VRAM, tile by
 * tile, runs of rb bytes: from memory to px, or from px to memory and
 * the shadow; Z24: the byte above the 24 bits stays as memory has it
 */
static inline __attribute__((always_inline)) void
nat_tiles(const NatSurf *e, uint8_t *mem, uint8_t *sh, uint32_t x0,
          uint32_t y0, uint32_t w, uint32_t h, uint8_t *px, const size_t rb,
          const int op)
{
    const HD5770Surf *s = &e->s;
    const uint32_t x1 = x0 + w, y1 = y0 + h, bpe = s->bpe, re = rb / bpe;
    uint32_t tx, ty, x, y;

    for (ty = y0 >> 3; ty <= (y1 - 1) >> 3; ty++) {
        uint32_t ya = MAX(ty << 3, y0), yb = MIN((ty + 1) << 3, y1);

        for (tx = x0 >> 3; tx <= (x1 - 1) >> 3; tx++) {
            uint64_t t = hd5770_surf_tile(s, tx, ty);
            uint32_t xa = MAX(tx << 3, x0), xb = MIN((tx + 1) << 3, x1);
            uint8_t *tm = mem + t, *ts = sh + t;

            for (y = ya; y < yb; y++) {
                const uint32_t *eo = &s->eoff[(y & 7) * 8];
                uint8_t *p = px + ((size_t)(y - y0) * w + (xa - x0)) * bpe;

                if (op == NIO_PUT_Z24) {
                    for (x = xa; x < xb; x++, p += 4) {
                        uint32_t o = eo[x & 7];
                        uint32_t v = ldl_le_p(p) |
                                     (ldl_le_p(tm + o) & 0xff000000u);

                        stl_le_p(tm + o, v);
                        stl_le_p(ts + o, v);
                    }
                } else if (xb - xa == 8) {
                    for (x = 0; x < 8; x += re, p += rb) {
                        if (op == NIO_GET) {
                            memcpy(p, tm + eo[x], rb);
                        } else {
                            memcpy(tm + eo[x], p, rb);
                            memcpy(ts + eo[x], p, rb);
                        }
                    }
                } else {
                    for (x = xa; x < xb; x++, p += bpe) {
                        if (op == NIO_GET) {
                            memcpy(p, tm + eo[x & 7], bpe);
                        } else {
                            memcpy(tm + eo[x & 7], p, bpe);
                            memcpy(ts + eo[x & 7], p, bpe);
                        }
                    }
                }
            }
        }
    }
}

/* the layout of split tiles, checked against hd5770_surf_offset() */
static void nat_split_prep(NatSurf *e)
{
    const HD5770Surf *s = &e->s;
    uint32_t np = (64 * s->bpe) >> s->split_sh, tw = s->pitch / 8;
    uint32_t th = s->height / 8, i, p, c;
    uint64_t f[32];

    if (np < 2 || np > 32 || !tw || !th) {
        return;
    }
    memset(e->spj, 0xff, sizeof(e->spj));
    for (i = 0; i < 64; i++) {
        e->spp[i] = s->eoff[i] >> s->split_sh;
        if (e->spp[i] >= np) {
            return;
        }
        if (!(s->eoff[i] & ((1u << s->split_sh) - 1))) {
            e->spj[e->spp[i]] = i;
        }
    }
    for (c = 0; c < 5; c++) {
        uint32_t tx = c == 1 || c == 3 ? tw - 1 : c == 4 ? tw / 2 : 0;
        uint32_t ty = c == 2 || c == 3 ? th - 1 : c == 4 ? th / 2 : 0;

        for (p = 0; p < np; p++) {
            if (e->spj[p] == 0xff) {
                return;
            }
            f[p] = hd5770_surf_offset(s, tx * 8 + (e->spj[p] & 7),
                                      ty * 8 + (e->spj[p] >> 3));
        }
        for (i = 0; i < 64; i++) {
            uint64_t o = hd5770_surf_offset(s, tx * 8 + (i & 7),
                                            ty * 8 + (i >> 3));

            if (o < f[e->spp[i]] || o - f[e->spp[i]] > UINT32_MAX ||
                (c && e->spg[i] != o - f[e->spp[i]])) {
                return;
            }
            e->spg[i] = o - f[e->spp[i]];
        }
    }
    e->spf = g_try_new0(uint64_t, (size_t)tw * th * np);
    e->np = e->spf ? np : 0;
}

static inline __attribute__((always_inline)) void
nat_tiles_split(NatSurf *e, uint8_t *mem, uint8_t *sh, uint32_t x0,
                uint32_t y0, uint32_t w, uint32_t h, uint8_t *px,
                const size_t bpe, const int op)
{
    const HD5770Surf *s = &e->s;
    const uint32_t x1 = x0 + w, y1 = y0 + h, tw = s->pitch / 8, np = e->np;
    uint32_t tx, ty, x, y, p;

    for (ty = y0 >> 3; ty <= (y1 - 1) >> 3; ty++) {
        uint32_t ya = MAX(ty << 3, y0), yb = MIN((ty + 1) << 3, y1);

        for (tx = x0 >> 3; tx <= (x1 - 1) >> 3; tx++) {
            uint64_t *f = &e->spf[((size_t)ty * tw + tx) * np];
            uint32_t xa = MAX(tx << 3, x0), xb = MIN((tx + 1) << 3, x1);

            if (!f[0]) {
                for (p = 0; p < np; p++) {
                    f[p] = hd5770_surf_offset(s, tx * 8 + (e->spj[p] & 7),
                                              ty * 8 + (e->spj[p] >> 3)) + 1;
                }
            }
            for (y = ya; y < yb; y++) {
                uint8_t *q = px + ((size_t)(y - y0) * w + (xa - x0)) * bpe;

                for (x = xa; x < xb; x++, q += bpe) {
                    uint32_t i = (y & 7) * 8 + (x & 7);
                    uint64_t o = f[e->spp[i]] - 1 + e->spg[i];

                    if (op == NIO_GET) {
                        memcpy(q, mem + o, bpe);
                    } else if (op == NIO_PUT) {
                        memcpy(mem + o, q, bpe);
                        memcpy(sh + o, q, bpe);
                    } else {
                        uint32_t v = ldl_le_p(q) |
                                     (ldl_le_p(mem + o) & 0xff000000u);

                        stl_le_p(mem + o, v);
                        stl_le_p(sh + o, v);
                    }
                }
            }
        }
    }
}

/* a rectangle of the entry, its elements px, as nat_row() does a row */
static void nat_rect_io(HD5770Mtlr *m, NatSurf *e, uint32_t x0,
                        uint32_t y0, uint32_t w, uint32_t h, uint8_t *px,
                        int op)
{
    const HD5770Surf *s = &e->s;
    uint8_t *mem = m->vram + e->voff, *sh = e->shadow;
    uint32_t bpe = s->bpe, x, y;

    if (!w || !h) {
        return;
    }
    if (e->np && x0 + w <= (s->pitch & ~7u) && y0 + h <= (s->height & ~7u)) {
#define NAT_SPLIT(n)                                                    \
    case n:                                                             \
        if (op == NIO_GET) {                                            \
            nat_tiles_split(e, mem, sh, x0, y0, w, h, px, n, NIO_GET);  \
        } else if (op == NIO_PUT) {                                     \
            nat_tiles_split(e, mem, sh, x0, y0, w, h, px, n, NIO_PUT);  \
        } else {                                                        \
            nat_tiles_split(e, mem, sh, x0, y0, w, h, px, n,            \
                            NIO_PUT_Z24);                               \
        }                                                               \
        return;
        switch (bpe) {
        NAT_SPLIT(1)
        NAT_SPLIT(2)
        NAT_SPLIT(4)
        NAT_SPLIT(8)
        NAT_SPLIT(16)
        }
#undef NAT_SPLIT
    }
    if (e->space || !s->fast || s->mode < HD5770_ARRAY_1D_TILED_THIN1) {
        uint8_t *old = op == NIO_PUT_Z24 ? g_malloc((size_t)w * 4) : NULL;

        for (y = 0; y < h; y++) {
            uint8_t *row = px + (size_t)y * w * bpe;

            if (old) {
                nat_row(m, e, x0, y0 + y, w, old, false);
                for (x = 0; x < w; x++) {
                    stl_le_p(row + x * 4, ldl_le_p(row + x * 4) |
                             (ldl_le_p(old + x * 4) & 0xff000000u));
                }
            }
            nat_row(m, e, x0, y0 + y, w, row, op != NIO_GET);
        }
        g_free(old);
        return;
    }
    if (op == NIO_PUT_Z24) {
        nat_tiles(e, mem, sh, x0, y0, w, h, px, 4, NIO_PUT_Z24);
        return;
    }
#define NAT_TILES(n)                                                    \
    case n:                                                             \
        if (op == NIO_GET) {                                            \
            nat_tiles(e, mem, sh, x0, y0, w, h, px, n, NIO_GET);        \
        } else {                                                        \
            nat_tiles(e, mem, sh, x0, y0, w, h, px, n, NIO_PUT);        \
        }                                                               \
        break;
    switch (e->run * bpe) {
    NAT_TILES(1)
    NAT_TILES(2)
    NAT_TILES(4)
    NAT_TILES(8)
    NAT_TILES(16)
    NAT_TILES(32)
    NAT_TILES(64)
    NAT_TILES(128)
    default:
        g_assert_not_reached();
    }
#undef NAT_TILES
}

/* n depths as floats to 24 bits, in place */
static void nat_z24(uint8_t *px, size_t n)
{
    size_t i = 0;

#ifdef __aarch64__
    const float64x2_t k = vdupq_n_f64(16777215.0);
    const float64x2_t one = vdupq_n_f64(1.0), zero = vdupq_n_f64(0.0);

    for (; i + 4 <= n; i += 4) {
        float32x4_t z = vreinterpretq_f32_u8(vld1q_u8(px + i * 4));
        float64x2_t a = vcvt_f64_f32(vget_low_f32(z));
        float64x2_t b = vcvt_high_f64_f32(z);

        a = vmulq_f64(vminnmq_f64(vmaxnmq_f64(a, zero), one), k);
        b = vmulq_f64(vminnmq_f64(vmaxnmq_f64(b, zero), one), k);
        vst1q_u8(px + i * 4, vreinterpretq_u8_u32(
                     vcombine_u32(vmovn_u64(vcvtnq_u64_f64(a)),
                                  vmovn_u64(vcvtnq_u64_f64(b)))));
    }
#endif
    for (; i < n; i++) {
        float z;
        double zd;

        memcpy(&z, px + i * 4, 4);
        zd = z > 0.0f ? MIN((double)z, 1.0) : 0.0;
        stl_le_p(px + i * 4, (uint32_t)lrint(zd * 16777215.0));
    }
}

/* the surface bytes [*a, *z) band b's rows lie in */
static void nat_band(const NatSurf *e, uint32_t b, uint64_t *a, uint64_t *z)
{
    uint64_t lo, hi;

    hd5770_surf_rows(&e->s, b * e->bh, MIN((b + 1) * e->bh, e->s.height),
                     &lo, &hi);
    *a = MIN(lo - e->s.base, e->cov);
    *z = MIN(hi - e->s.base, e->cov);
}

/* rows y0..y1-1 of memory to the host texture */
static void *nat_ring(HD5770Mtlr *m, size_t n, size_t *at);

/*
 * rows y0..y1-1 of a texture a pass in flight uses, n bytes at src: a
 * copy from the ring after that pass, counted as a pass itself (the
 * ring and the texture are in use until it is done)
 */
static void nat_upload_after(HD5770Mtlr *m, NatSurf *e, uint32_t y0,
                             uint32_t y1, const uint8_t *src, size_t row)
{
    uint32_t bk = NF_BLK(e->hf);
    size_t n = row * (y1 - y0), at;
    uint8_t *p = nat_ring(m, n, &at);
    id<MTLCommandBuffer> cb;
    id<MTLBlitCommandEncoder> bl;

    memcpy(p, src, n);
    nat_end_pass(m);
    cb = [[m->q commandBuffer] retain];
    bl = [cb blitCommandEncoder];
    [bl copyFromBuffer:m->ring sourceOffset:at sourceBytesPerRow:row
   sourceBytesPerImage:n
            sourceSize:MTLSizeMake(e->s.pitch * bk, (y1 - y0) * bk, 1)
             toTexture:e->tex destinationSlice:0 destinationLevel:0
     destinationOrigin:MTLOriginMake(0, y0 * bk, 0)];
    [bl endEncoding];
    m->passes++;
    e->pass = m->passes;
    nat_commit(m, cb);
    [m->last release];
    m->last = cb;
    m->up_after++;
}

static void nat_upload(HD5770Mtlr *m, NatSurf *e, uint32_t y0, uint32_t y1)
{
    uint32_t w = e->s.pitch, bpe = e->s.bpe, bk = NF_BLK(e->hf);
    bool lin = !e->space && e->s.fast &&
               e->s.mode < HD5770_ARRAY_1D_TILED_THIN1 && e->hf != NF_D24;
    size_t row = lin ? e->s.row_bytes : (size_t)w * bpe;
    bool fly;
    int64_t t0;

    if (e->pass > m->pass_done) {
        m->pass_done = MAX(m->pass_done, qatomic_read(&m->cb_passes));
    }
    /* a pass not yet complete uses the texture */
    fly = e->pass > m->pass_done;
    if (fly && ((e->hf >= NF_D16 && e->hf <= NF_S8) ||
                ROUND_UP(m->ring_at, 256) + row * (y1 - y0) > NAT_RING)) {
        nat_wait(m, NW_UPLOAD);
        fly = false;
    }
    t0 = get_clock();
    if (lin) {
        const uint8_t *src = m->vram + e->voff + (uint64_t)y0 * row;

        if (fly) {
            nat_upload_after(m, e, y0, y1, src, row);
        } else {
            [e->tex replaceRegion:MTLRegionMake2D(0, y0 * bk, w * bk,
                                                  (y1 - y0) * bk)
                      mipmapLevel:0 withBytes:src bytesPerRow:row];
        }
    } else {
        uint8_t *px = nat_tmp(m, (size_t)w * (y1 - y0) * bpe);

        nat_rect_io(m, e, 0, y0, w, y1 - y0, px, NIO_GET);
        if (e->hf == NF_D24) {
            size_t i, n = (size_t)w * (y1 - y0);

            for (i = 0; i < n; i++) {
                float z = (ldl_le_p(px + i * 4) & 0xffffff) / 16777215.0f;

                memcpy(px + i * 4, &z, 4);
            }
        }
        if (fly) {
            nat_upload_after(m, e, y0, y1, px, row);
        } else {
            [e->tex replaceRegion:MTLRegionMake2D(0, y0 * bk, w * bk,
                                                  (y1 - y0) * bk)
                      mipmapLevel:0 withBytes:px bytesPerRow:row];
        }
    }
    m->uploads++;
    m->upload_bytes += (uint64_t)w * (y1 - y0) * bpe;
    m->upload_ns += get_clock() - t0;
}

/* VRAM bytes [lo, hi) were written: their pages get a new generation */
static void nat_named(HD5770Mtlr *m, uint64_t lo, uint64_t hi)
{
    uint64_t p;

    if (hi <= lo) {
        return;
    }
    m->gen++;
    for (p = lo >> 12; p <= (hi - 1) >> 12 && p < m->vram_size >> 12; p++) {
        m->pgen[p] = m->gen;
    }
}

/* the device's log for the entry's pages, 64 pages at a time */
static void nat_log(HD5770Mtlr *m, const NatSurf *e)
{
    uint64_t p;

    for (p = (e->lo >> 12) & ~63ull; p <= (e->hi - 1) >> 12; p += 64) {
        uint64_t bits = m->pub.written
                        ? m->pub.written(m->pub.opaque, p << 12) : ~0ull;
        unsigned i;

        if (!bits) {
            continue;
        }
        m->gen++;
        for (i = 0; i < 64 && p + i < m->vram_size >> 12; i++) {
            if ((bits >> i) & 1) {
                m->pgen[p + i] = m->gen;
            }
        }
    }
}

/* band b has rows drawn and not written back */
static bool nat_band_dirty(const NatSurf *e, uint32_t b)
{
    int k;

    for (k = 0; e->dirty && k < e->nr; k++) {
        if (e->r[k][1] < (int)((b + 1) * e->bh) &&
            e->r[k][3] > (int)(b * e->bh)) {
            return true;
        }
    }
    return false;
}

/* a page of band b named since its last comparison differs */
static bool nat_band_diff(HD5770Mtlr *m, const NatSurf *e, uint32_t b)
{
    uint64_t a, z, o;
    bool diff = false;

    nat_band(e, b, &a, &z);
    for (o = a; o < z && !diff; ) {
        size_t n;
        uint8_t *p8 = nat_at(m, e, o, &n);

        n = MIN(n, z - o);
        if (!e->space) {
            /* up to the end of the page */
            n = MIN(n, 4096 - ((e->voff + o) & 4095));
            if (m->pgen[(e->voff + o) >> 12] <= e->bgen[b]) {
                o += n;
                continue;
            }
        }
        m->cmp_bytes += n;
        diff = memcmp(p8, e->shadow + o, n);
        o += n;
    }
    return diff;
}

static void nat_drain(HD5770Mtlr *m);

/*
 * Rows y0..y1-1 of memory against the shadow, once per stamp: the bands
 * with a page that differs, and the ones not held, are uploaded and
 * copied to the shadow. In VRAM only the pages named since the band's
 * last comparison are compared.
 */
static void nat_check(HD5770Mtlr *m, NatSurf *e, uint32_t y0, uint32_t y1)
{
    int64_t t0 = get_clock(), up0 = m->upload_ns;
    int64_t w0 = m->wait_ns + m->drain_ns + m->fcopy_ns;
    uint32_t b, r0, b0 = y0 / e->bh, b1 = (y1 - 1) / e->bh;
    uint64_t a, z;
    uint64_t stamp = m->stamp[e->space];
    uint8_t *fl;

    for (b = b0; b <= b1 && (e->bstamp[b] == stamp ||
                             (e->fly[b] && e->fstamp[b] == stamp)); b++) {
        /* compared already */
    }
    if (b > b1) {
        return;
    }
    if (m->flags_len < e->nb + 1) {
        m->flags = g_realloc(m->flags, e->nb + 1);
        m->flags_len = e->nb + 1;
    }
    fl = m->flags;
    if (!e->space && e->hstamp != stamp) {
        nat_log(m, e);
        e->hstamp = stamp;
    }
    for (b = b0; b <= b1 && e->nfly; b++) {
        /* in flight: left alone unless memory changed under it */
        if (e->fly[b] && e->bstamp[b] != stamp && e->fstamp[b] != stamp &&
            !nat_band_dirty(e, b)) {
            m->fly_cmp++;
            if (!e->held[b] || nat_band_diff(m, e, b)) {
                nat_drain(m);
                break;
            }
            e->fstamp[b] = stamp;
        }
    }
    for (b = b0; b <= b1; b++) {
        bool diff = !e->held[b];

        fl[b] = 0;
        if (e->bstamp[b] == stamp || nat_band_dirty(e, b) || e->fly[b]) {
            continue;
        }
        diff = diff || nat_band_diff(m, e, b);
        fl[b] = diff;
        e->bgen[b] = m->gen;
        e->bstamp[b] = stamp;
    }
    fl[b1 + 1] = 0;
    for (b = b0, r0 = UINT32_MAX; b <= b1 + 1; b++) {
        if (fl[b] && r0 == UINT32_MAX) {
            r0 = b;
        } else if (!fl[b] && r0 != UINT32_MAX) {
            nat_upload(m, e, r0 * e->bh, MIN(b * e->bh, e->s.height));
            r0 = UINT32_MAX;
        }
    }
    for (b = b0; b <= b1; b++) {
        if (fl[b]) {
            if (!e->held[b]) {
                m->first_bytes += (uint64_t)e->s.pitch * e->s.bpe *
                                  (MIN((b + 1) * e->bh, e->s.height) -
                                   b * e->bh);
            }
            nat_band(e, b, &a, &z);
            nat_get(m, e, a, e->shadow + a, z - a);
            e->held[b] = 1;
        }
    }
    m->checks++;
    m->check_ns += get_clock() - t0 - (m->upload_ns - up0) -
                   (m->wait_ns + m->drain_ns + m->fcopy_ns - w0);
}

/* both over some byte */
static bool nat_alias(const NatSurf *a, const NatSurf *b)
{
    uint32_t i, j;

    if (a->space != b->space) {
        return false;
    }
    if (!a->space) {
        return a->lo < b->hi && b->lo < a->hi;
    }
    if (a->pmin > b->pmax || b->pmin > a->pmax) {
        return false;
    }
    for (i = 0; i < a->npages; i++) {
        if (a->pages[i] < b->pmin || a->pages[i] > b->pmax) {
            continue;
        }
        for (j = 0; j < b->npages; j++) {
            if (a->pages[i] == b->pages[j]) {
                return true;
            }
        }
    }
    return false;
}

/* the device is told the ranges written */
static void nat_report(HD5770Mtlr *m)
{
    unsigned i;

    for (i = 0; i + 2 < m->ranges->len; i += 3) {
        uint64_t *r = &g_array_index(m->ranges, uint64_t, i);

        if (r[0] && m->pub.done_ram) {
            m->pub.done_ram(m->pub.opaque, r[1], r[2]);
        } else if (!r[0] && m->pub.done) {
            m->pub.done(m->pub.opaque, r[1], r[2]);
        }
    }
    g_array_set_size(m->ranges, 0);
}

/* rectangle rc of the entry, its pixels px, to memory and the shadow */
static void nat_rect_out(HD5770Mtlr *m, NatSurf *e, const int *rc,
                         uint8_t *px)
{
    uint32_t x0 = rc[0], y0 = rc[1], w = rc[2] - x0, h = rc[3] - y0;
    uint32_t bpe = e->s.bpe;
    uint64_t lo, hi, r[3], o;

    if (e->hf == NF_D24) {
        nat_z24(px, (size_t)w * h);
        nat_rect_io(m, e, x0, y0, w, h, px, NIO_PUT_Z24);
    } else {
        nat_rect_io(m, e, x0, y0, w, h, px, NIO_PUT);
    }
    hd5770_surf_rows(&e->s, y0, y0 + h, &lo, &hi);
    lo = MIN(lo - e->s.base, e->cov);
    hi = MIN(hi - e->s.base, e->cov);
    if (!e->space) {
        r[0] = 0;
        r[1] = e->voff + lo;
        r[2] = e->voff + hi;
        g_array_append_vals(m->ranges, r, 3);
        nat_named(m, r[1], r[2]);
    } else {
        for (o = lo; o < hi; o = ((o + e->pg0) | 4095) + 1 - e->pg0) {
            r[0] = 1;
            r[1] = e->pages[(o + e->pg0) >> 12];
            r[2] = r[1] + 4096;
            g_array_append_vals(m->ranges, r, 3);
        }
    }
    m->wbacks++;
    m->wback_bytes += (uint64_t)w * h * bpe;
}

/* the entries over the bytes of e, written back, compare them again */
static void nat_aliased(HD5770Mtlr *m, const NatSurf *e)
{
    unsigned j;

    for (j = 0; j < m->surfs->len; j++) {
        NatSurf *f = g_ptr_array_index(m->surfs, j);

        if (f != e && nat_alias(f, e)) {
            m->stamp[e->space]++;
            break;
        }
    }
}

static void nat_fly(NatSurf *e, const int *rc, int by)
{
    uint32_t b;

    for (b = rc[1] / e->bh; b <= (uint32_t)(rc[3] - 1) / e->bh; b++) {
        e->fly[b] += by;
    }
    e->nfly += by;
}

static inline unsigned nat_hash(unsigned space, uint64_t at)
{
    return ((at >> 8) * 0x9e3779b97f4a7c15ull + space) >> 52;
}

static inline unsigned nat_hash_of(const NatSurf *e)
{
    return nat_hash(e->space, e->space ? e->pages[0] : e->voff);
}

/*
 * The fences whose command buffers are complete, oldest first: their
 * rectangles to memory, the ranges reported (lk held)
 */
static void nat_fence_copy(HD5770Mtlr *m)
{
    GList *l;
    unsigned i;

    for (l = m->fences.head; l; l = l->next) {
        NatFence *f = l->data;
        int64_t t0;

        if (f->copied) {
            continue;
        }
        if (f->cb && !f->gpu_done &&
            f->cb.status < MTLCommandBufferStatusCompleted) {
            break;
        }
        t0 = get_clock();
        for (i = 0; i < f->n; i++) {
            NatRect *c = &f->rc[i];

            nat_rect_out(m, c->e, c->r,
                         (uint8_t *)f->buf.contents + c->ro);
            nat_fly(c->e, c->r, -1);
            nat_aliased(m, c->e);
        }
        g_free(f->rc);
        f->rc = NULL;
        if (f->cb) {
            m->pass_done = MAX(m->pass_done, f->passes);
            if (m->last == f->cb && !m->enc) {
                /* nothing committed after it: the ring is free */
                [m->last release];
                m->last = nil;
                m->ring_at = 0;
            }
            [f->cb release];
            f->cb = nil;
        }
        if (f->buf) {
            if (m->bufs->len < 8) {
                g_ptr_array_add(m->bufs, f->buf);
            } else {
                [f->buf release];
            }
            f->buf = nil;
        }
        f->copied = true;
        m->fcopy_ns += get_clock() - t0;
    }
    nat_report(m);
}

/* every fence's rectangles are in memory; lk is released meanwhile */
static void nat_drain(HD5770Mtlr *m)
{
    bool waited = false;
    int64_t t0 = get_clock(), c0 = m->fcopy_ns;
    GList *l;

    for (;;) {
        NatFence *f = NULL;

        nat_fence_copy(m);
        for (l = m->fences.head; l; l = l->next) {
            f = l->data;
            if (!f->copied) {
                break;
            }
        }
        if (!l) {
            break;
        }
        if (f->cb) {
            id<MTLCommandBuffer> cb = [f->cb retain];

            g_mutex_unlock(&m->lk);
            [cb waitUntilCompleted];
            g_mutex_lock(&m->lk);
            [cb release];
            waited = true;
        }
    }
    if (waited) {
        m->drains++;
        m->drain_ns += get_clock() - t0 - (m->fcopy_ns - c0);
    }
}

/* a fence's command buffer is complete: on a thread of Metal's */
static void nat_retire(HD5770Mtlr *m, uint64_t ticket)
{
    GSList *done = NULL, *i;
    NatFence *f;
    GList *l;

    if (m->pub.thread) {
        m->pub.thread(m->pub.opaque, true);
    }
    g_mutex_lock(&m->rlock);
    g_mutex_lock(&m->lk);
    for (l = m->fences.head; l; l = l->next) {
        f = l->data;
        if (f->ticket == ticket) {
            f->gpu_done = true;
            break;
        }
    }
    nat_fence_copy(m);
    while ((f = g_queue_peek_head(&m->fences)) && f->copied) {
        g_queue_pop_head(&m->fences);
        done = g_slist_prepend(done, f);
    }
    g_mutex_unlock(&m->lk);
    done = g_slist_reverse(done);
    for (i = done; i; i = i->next) {
        f = i->data;
        if (m->pub.retired) {
            m->pub.retired(m->pub.opaque, f->ticket);
        }
        g_free(f);
    }
    g_slist_free(done);
    g_mutex_unlock(&m->rlock);
    if (m->pub.thread) {
        m->pub.thread(m->pub.opaque, false);
    }
}

/*
 * The newer pixels of the host textures to memory: of entry only, or of
 * all, after the fences'. The GPU copies the rectangles drawn to the
 * read buffer behind the draws; one wait.
 */
static void nat_writeback(HD5770Mtlr *m, NatSurf *only, unsigned why)
{
    id<MTLCommandBuffer> cb = nil;
    id<MTLBlitCommandEncoder> bl = nil;
    size_t at = 0;
    unsigned i;
    int64_t t0;
    int k;

    nat_drain(m);
    for (i = 0; i < m->act->len; i++) {
        NatSurf *e = g_ptr_array_index(m->act, i);

        if (!e->dirty || (only && e != only)) {
            continue;
        }
        for (k = 0; k < e->nr; k++) {
            uint32_t w = e->r[k][2] - e->r[k][0], h = e->r[k][3] - e->r[k][1];
            size_t n = (size_t)w * h * e->s.bpe;

            e->ro[k] = SIZE_MAX;
            if (at + n > NAT_READ) {
                continue;
            }
            if (!bl) {
                if (m->enc) {
                    /* in the command buffer of the pass */
                    [m->enc endEncoding];
                    [m->enc release];
                    m->enc = nil;
                    m->cur = m->cur_z = m->cur_s = NULL;
                    m->ncurm = 0;
                    m->pass_on = false;
                    cb = m->cb;
                    m->cb = nil;
                } else {
                    cb = [[m->q commandBuffer] retain];
                }
                bl = [cb blitCommandEncoder];
            }
            [bl copyFromTexture:e->tex sourceSlice:0 sourceLevel:0
                   sourceOrigin:MTLOriginMake(e->r[k][0], e->r[k][1], 0)
                     sourceSize:MTLSizeMake(w, h, 1) toBuffer:m->rbuf
              destinationOffset:at destinationBytesPerRow:w * e->s.bpe
       destinationBytesPerImage:n];
            e->ro[k] = at;
            at += ROUND_UP(n, 16);
        }
    }
    if (bl) {
        [bl endEncoding];
        nat_commit(m, cb);
        [m->last release];
        m->last = cb;
    }
    nat_wait(m, why);
    t0 = get_clock();
    for (i = 0; i < m->act->len; i++) {
        NatSurf *e = g_ptr_array_index(m->act, i);
        uint32_t bpe = e->s.bpe;

        if (!e->dirty || (only && e != only)) {
            continue;
        }
        for (k = 0; k < e->nr; k++) {
            uint32_t x0 = e->r[k][0], y0 = e->r[k][1];
            uint32_t w = e->r[k][2] - x0, h = e->r[k][3] - y0;
            uint8_t *px;

            if (e->ro[k] != SIZE_MAX) {
                px = (uint8_t *)m->rbuf.contents + e->ro[k];
            } else {
                px = nat_tmp(m, (size_t)w * h * bpe);
                [e->tex getBytes:px bytesPerRow:w * bpe
                      fromRegion:MTLRegionMake2D(x0, y0, w, h) mipmapLevel:0];
            }
            nat_rect_out(m, e, e->r[k], px);
        }
        e->dirty = false;
        e->nr = 0;
        nat_aliased(m, e);
    }
    m->wback_ns += get_clock() - t0;
}

/*
 * a depth or stencil plane drawn since the last fence, or within
 * lazy_ns, stays out of this one; so does a colour target in VRAM that
 * is not scanned out
 */
static inline bool nat_kept(const HD5770Mtlr *m, const NatSurf *e)
{
    if (!e->recent && m->now - e->tdraw >= m->lazy_ns) {
        return false;
    }
    if (e->hf == NF_D16 || e->hf == NF_D24 || e->hf == NF_D32 ||
        e->hf == NF_S8) {
        return m->zlazy;
    }
    return m->clazy && !e->space &&
           !(e->lo < qatomic_read(&m->pub.scan_hi) &&
             qatomic_read(&m->pub.scan_lo) < e->hi);
}

/* the fence is made: what stays newer than memory, for the device */
static void nat_fenced(HD5770Mtlr *m)
{
    unsigned i;

    m->pub.kept = false;
    for (i = 0; i < m->act->len; i++) {
        NatSurf *e = g_ptr_array_index(m->act, i);

        m->pub.kept |= e->dirty;
        e->recent = false;
    }
}

/* a read buffer of n bytes or more */
static id<MTLBuffer> nat_buf(HD5770Mtlr *m, size_t n)
{
    unsigned i;

    for (i = 0; i < m->bufs->len; i++) {
        id<MTLBuffer> b = g_ptr_array_index(m->bufs, i);

        if (b.length >= n) {
            g_ptr_array_remove_index_fast(m->bufs, i);
            return b;
        }
    }
    return [m->dev newBufferWithLength:ROUND_UP(n, 1u << 20)
                               options:MTLResourceStorageModeShared];
}

/*
 * FENCES above: the passes and the copy of the rectangles drawn are
 * committed; the handler does the rest
 */
static uint64_t nat_fence_locked(HD5770Mtlr *m, bool *drew)
{
    id<MTLCommandBuffer> cb, pcb = nil;
    id<MTLBlitCommandEncoder> bl;
    id<MTLBuffer> buf = nil;
    NatFence *f;
    size_t at = 0, need = 0;
    uint64_t ticket;
    unsigned i, n = 0;
    bool busy = false;
    int k;

    for (i = 0; i < m->act->len; i++) {
        NatSurf *e = g_ptr_array_index(m->act, i);

        /* the scanout moves meanwhile: asked once */
        e->kept = nat_kept(m, e);
        for (k = 0; e->dirty && !e->kept && k < e->nr; k++) {
            need += ROUND_UP((size_t)(e->r[k][2] - e->r[k][0]) *
                             (e->r[k][3] - e->r[k][1]) * e->s.bpe, 16);
            n++;
        }
    }
    if (n) {
        buf = nat_buf(m, need);
        if (!buf) {
            nat_writeback(m, NULL, NW_SYNC);
            n = 0;
        }
    }
    *drew = n;
    if (!n) {
        if (m->enc) {
            /* the open pass's command buffer carries the fence */
            [m->enc endEncoding];
            [m->enc release];
            m->enc = nil;
            pcb = m->cb;
            m->cb = nil;
        }
        m->cur = m->cur_z = m->cur_s = NULL;
        m->ncurm = 0;
        m->pass_on = false;
        nat_report(m);
        /* passes in flight read vertex buffers in place */
        m->pass_done = MAX(m->pass_done, qatomic_read(&m->cb_passes));
        busy = pcb || m->passes > m->pass_done;
        if (g_queue_is_empty(&m->fences) && !busy) {
            m->fence_now++;
            return 0;
        }
        m->fence_empty++;
    }
    f = g_new0(NatFence, 1);
    f->ticket = ticket = ++m->tickets;
    if (n) {
        if (m->enc) {
            /* in the command buffer of the pass */
            [m->enc endEncoding];
            [m->enc release];
            m->enc = nil;
            m->cur = m->cur_z = m->cur_s = NULL;
            m->pass_on = false;
            cb = m->cb;
            m->cb = nil;
        } else {
            cb = [[m->q commandBuffer] retain];
        }
        f->rc = g_new(NatRect, n);
        f->buf = buf;
        bl = [cb blitCommandEncoder];
        for (i = 0; i < m->act->len; i++) {
            NatSurf *e = g_ptr_array_index(m->act, i);

            if (e->kept) {
                m->kept_rects += e->dirty ? e->nr : 0;
                continue;
            }
            for (k = 0; e->dirty && k < e->nr; k++) {
                NatRect *c = &f->rc[f->n++];
                uint32_t w = e->r[k][2] - e->r[k][0];
                uint32_t h = e->r[k][3] - e->r[k][1];
                size_t len = (size_t)w * h * e->s.bpe;

                c->e = e;
                memcpy(c->r, e->r[k], sizeof(c->r));
                c->ro = at;
                [bl copyFromTexture:e->tex sourceSlice:0 sourceLevel:0
                       sourceOrigin:MTLOriginMake(c->r[0], c->r[1], 0)
                         sourceSize:MTLSizeMake(w, h, 1) toBuffer:buf
                  destinationOffset:at destinationBytesPerRow:w * e->s.bpe
           destinationBytesPerImage:len];
                at += ROUND_UP(len, 16);
                nat_fly(e, c->r, 1);
            }
            e->dirty = false;
            e->nr = 0;
        }
        [bl endEncoding];
        f->passes = m->passes;
        f->cb = [cb retain];
        [cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
            @autoreleasepool {
                nat_retire(m, ticket);
            }
        }];
        g_queue_push_tail(&m->fences, f);
        nat_commit(m, cb);
        [m->last release];
        m->last = cb;
    } else if (busy) {
        /* done when the passes before it are */
        cb = pcb ? pcb : [[m->q commandBuffer] retain];
        m->fence_own += !pcb;
        f->passes = m->passes;
        f->cb = [cb retain];
        [cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
            @autoreleasepool {
                nat_retire(m, ticket);
            }
        }];
        g_queue_push_tail(&m->fences, f);
        nat_commit(m, cb);
        [m->last release];
        m->last = cb;
        m->fence_wait++;
    } else {
        g_queue_push_tail(&m->fences, f);
    }
    m->nfences++;
    return ticket;
}

/* the rectangle joins the ones drawn */
static void nat_drew(HD5770Mtlr *m, NatSurf *e, int x0, int y0, int x1,
                     int y1)
{
    int k, n = 0;

    e->dirty = true;
    e->recent = true;
    e->tdraw = m->now;
    if (!e->act) {
        e->act = true;
        g_ptr_array_add(m->act, e);
    }
    for (k = 0; k < e->nr; k++) {
        int *r = e->r[k];

        if ((x0 <= r[2] && r[0] <= x1 && y0 <= r[3] && r[1] <= y1) ||
            e->nr == NAT_RECTS) {
            /* touching, or no room: one rectangle of both */
            x0 = MIN(x0, r[0]);
            y0 = MIN(y0, r[1]);
            x1 = MAX(x1, r[2]);
            y1 = MAX(y1, r[3]);
        } else {
            memmove(e->r[n++], r, sizeof(e->r[0]));
        }
    }
    if (n < e->nr) {
        /* the larger rectangle may touch others */
        e->nr = n;
        nat_drew(m, e, x0, y0, x1, y1);
        return;
    }
    e->r[n][0] = x0;
    e->r[n][1] = y0;
    e->r[n][2] = x1;
    e->r[n][3] = y1;
    e->nr = n + 1;
}

static void nat_surf_free(gpointer p)
{
    NatSurf *e = p;

    [e->tex release];
    g_free(e->shadow);
    g_free(e->held);
    g_free(e->bgen);
    g_free(e->bstamp);
    g_free(e->amiss);
    g_free(e->fly);
    g_free(e->fstamp);
    g_free(e->pages);
    g_free(e->spf);
    g_free(e);
}

static uint64_t nat_bytes(const NatSurf *e)
{
    return (uint64_t)e->s.pitch * e->s.height * e->s.bpe + e->cov;
}

static void nat_drop(HD5770Mtlr *m, NatSurf *e)
{
    NatSurf **h;

    if (e->dirty) {
        nat_writeback(m, e, NW_EVICT);
    }
    if (m->cur == e || m->cur_z == e || m->cur_s == e || m->ncurm) {
        nat_end_pass(m);
    }
    m->bytes -= nat_bytes(e);
    for (h = &m->hb[nat_hash_of(e)]; *h != e; h = &(*h)->hnext) {
        /* search */
    }
    *h = e->hnext;
    if (e->act) {
        g_ptr_array_remove_fast(m->act, e);
    }
    g_ptr_array_remove_fast(m->surfs, e);
}

/* drops the entries last used before sync number before, but this draw's */
static void nat_evict(HD5770Mtlr *m, uint64_t before)
{
    unsigned i;

    for (i = m->surfs->len; i-- > 0; ) {
        NatSurf *e = g_ptr_array_index(m->surfs, i);

        if (e->used < before && e->pin != m->seq && !e->nfly) {
            nat_drop(m, e);
        }
    }
}

static int nat_cmp_pin(const void *a, const void *b)
{
    const NatSurf *x = *(NatSurf *const *)a, *y = *(NatSurf *const *)b;

    return x->pin < y->pin ? -1 : x->pin > y->pin;
}

/*
 * room for an entry of n bytes: the entries no draw has used for longest
 * go, down to three quarters of the limit
 */
static void nat_room(HD5770Mtlr *m, uint64_t n)
{
    uint64_t goal = m->cache_max - m->cache_max / 4;
    unsigned i, len, pass;
    NatSurf **o;

    for (pass = 0; pass < 2 && m->bytes + n > m->cache_max; pass++) {
        if (pass) {
            /* the rest is in flight */
            nat_drain(m);
        }
        len = m->surfs->len;
        o = g_memdup2(m->surfs->pdata, len * sizeof(*o));
        qsort(o, len, sizeof(*o), nat_cmp_pin);
        for (i = 0; i < len && m->bytes + n > goal; i++) {
            if (o[i]->pin != m->seq && !o[i]->nfly) {
                nat_drop(m, o[i]);
                m->roomed++;
            }
        }
        g_free(o);
    }
}

/* the entry holds the surface: the same rows, as many or more */
static bool nat_same(const NatSurf *e, const HD5770GpuSurf *s, unsigned hf)
{
    const HD5770Surf *a = &e->s, *b = &s->s;

    if (e->space != s->space || e->hf != hf || a->pitch != b->pitch ||
        a->height < b->height || a->mode != b->mode || a->bpe != b->bpe ||
        a->fast != b->fast || (!a->fast && a->height != b->height)) {
        return false;
    }
    if (a->fast && a->mode < HD5770_ARRAY_1D_TILED_THIN1) {
        /* rows one after another: the tile parameters do not matter */
        if (a->row_bytes != b->row_bytes) {
            return false;
        }
    } else if (a->banks != b->banks || a->bankw != b->bankw ||
               a->bankh != b->bankh || a->mtilea != b->mtilea ||
               a->tile_split != b->tile_split || a->pipes != b->pipes ||
               a->pipe_interleave != b->pipe_interleave ||
               a->nondisp != b->nondisp) {
        return false;
    }
    if (!s->space) {
        return e->voff == s->voff;
    }
    return e->npages >= s->npages && e->pg0 == s->pg0 &&
           !memcmp(e->pages, s->pages, s->npages * sizeof(uint32_t));
}

/* the entry that holds the surface in host format hf, if there is one */
static NatSurf *nat_find(HD5770Mtlr *m, const HD5770GpuSurf *s, unsigned hf)
{
    NatSurf *e;

    if (s->space && !s->npages) {
        return NULL;
    }
    for (e = m->hb[nat_hash(s->space, s->space ? s->pages[0] : s->voff)]; e;
         e = e->hnext) {
        if (nat_same(e, s, hf)) {
            e->used = m->epoch;
            e->pin = m->seq;
            return e;
        }
    }
    return NULL;
}

/* a surface the pipeline can hold in host format hf: its entry, else NULL */
static NatSurf *nat_surf(HD5770Mtlr *m, const HD5770GpuSurf *s, unsigned hf)
{
    MTLTextureDescriptor *td;
    NatSurf *e;
    uint64_t cov, lo, hi;
    unsigned i;

    if (s->s.bpe != nf_bpe[hf] || !(s->s.fast || s->s.split) || !s->s.pitch ||
        !s->s.height || s->s.pitch > 16384 || s->s.height > 16384 ||
        !hd5770_surf_mode_ok(s->s.mode)) {
        return NULL;
    }
    e = nat_find(m, s, hf);
    if (e || (s->space && !s->npages)) {
        return e;
    }
    hd5770_surf_rows(&s->s, 0, s->s.height, &lo, &hi);
    cov = MAX(hd5770_surf_size(&s->s), hi - s->s.base);
    if (!s->s.fast) {
        /* tiles in split pieces: the bytes its elements reach */
        uint32_t x, y;

        cov = 0;
        for (y = 0; y < s->s.height; y++) {
            for (x = 0; x < s->s.pitch; x++) {
                cov = MAX(cov, hd5770_surf_offset(&s->s, x, y) + s->s.bpe);
            }
        }
    }
    if (s->space) {
        if (!m->ram || (s->pg0 & (s->s.bpe - 1)) ||
            (cov + s->pg0 + 4095) / 4096 > s->npages) {
            return NULL;
        }
        for (i = 0; i < s->npages; i++) {
            if ((uint64_t)s->pages[i] + 4096 > m->ram_size) {
                return NULL;
            }
        }
    } else if (s->voff > m->vram_size || cov > m->vram_size - s->voff) {
        return NULL;
    }
    if (m->bytes + (uint64_t)s->s.pitch * s->s.height * s->s.bpe + cov >
        m->cache_max) {
        nat_room(m, (uint64_t)s->s.pitch * s->s.height * s->s.bpe + cov);
    }
    e = g_new0(NatSurf, 1);
    e->s = s->s;
    e->space = s->space;
    e->hf = hf;
    e->voff = s->voff;
    e->cov = cov;
    e->lo = s->voff;
    e->hi = s->voff + cov;
    if (s->space) {
        e->pages = g_memdup2(s->pages, s->npages * sizeof(uint32_t));
        e->npages = s->npages;
        e->pg0 = s->pg0;
        e->pmin = UINT32_MAX;
        for (i = 0; i < s->npages; i++) {
            e->pmin = MIN(e->pmin, s->pages[i]);
            e->pmax = MAX(e->pmax, s->pages[i]);
        }
    }
    e->bh = NAT_BAND;
    if (!s->s.fast) {
        e->bh = s->s.height;
    } else if (s->s.mode == HD5770_ARRAY_2D_TILED_THIN1) {
        e->bh = ROUND_UP(NAT_BAND, 1u << s->s.mt_h_sh);
    }
    e->nb = DIV_ROUND_UP(s->s.height, e->bh);
    for (i = 1; i < e->nb; i++) {
        uint64_t a0, z0, a1, z1;

        nat_band(e, i - 1, &a0, &z0);
        nat_band(e, i, &a1, &z1);
        if (a1 < z0) {
            /* bands that share bytes: one band */
            e->bh = s->s.height;
            e->nb = 1;
        }
    }
    e->held = g_new0(uint8_t, e->nb);
    e->bgen = g_new0(uint64_t, e->nb);
    e->bstamp = g_new0(uint64_t, e->nb);
    e->amiss = g_new0(uint8_t, e->nb);
    e->fly = g_new0(uint16_t, e->nb);
    e->fstamp = g_new0(uint64_t, e->nb);
    for (e->run = s->s.fast ? 8 : 1; e->run > 1; e->run >>= 1) {
        for (i = 0; i < 64; i++) {
            if ((i & (e->run - 1)) &&
                s->s.eoff[i] != s->s.eoff[i - 1] + s->s.bpe) {
                break;
            }
        }
        if (i == 64) {
            break;
        }
    }
    if (!s->s.fast && s->s.split && !s->space) {
        nat_split_prep(e);
    }
    td = [MTLTextureDescriptor
          texture2DDescriptorWithPixelFormat:nf_fmt[hf]
                                       width:s->s.pitch * NF_BLK(hf)
                                      height:s->s.height * NF_BLK(hf)
                                   mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead |
               (hf >= NF_BC1 ? 0 : MTLTextureUsageRenderTarget);
    td.storageMode = MTLStorageModeShared;
    e->tex = [m->dev newTextureWithDescriptor:td];
    e->shadow = g_try_malloc(cov);
    if (!e->tex || !e->shadow) {
        nat_surf_free(e);
        return NULL;
    }
    e->used = m->epoch;
    e->pin = m->seq;
    m->bytes += nat_bytes(e);
    m->bytes_most = MAX(m->bytes_most, m->bytes);
    g_ptr_array_add(m->surfs, e);
    i = nat_hash_of(e);
    e->hnext = m->hb[i];
    m->hb[i] = e;
    return e;
}

/*
 * rows y0..y1-1 of the entry's texture hold what memory and the draws
 * queued make of them
 */
static void nat_ready(HD5770Mtlr *m, NatSurf *e, uint32_t y0, uint32_t y1)
{
    unsigned i;

    for (i = 0; i < m->act->len; i++) {
        NatSurf *f = g_ptr_array_index(m->act, i);

        if (!f->dirty && !f->nfly) {
            f->act = false;
            g_ptr_array_remove_index_fast(m->act, i--);
            continue;
        }
        if (f != e && nat_alias(f, e)) {
            if (f->dirty) {
                nat_writeback(m, f, NW_ALIAS);
            } else {
                nat_drain(m);
            }
        }
    }
    nat_check(m, e, y0, y1);
}

static void nat_prog_free(gpointer p)
{
    NatProg *pr = p;

    if (pr) {
        [pr->vfn release];
        [pr->ffn release];
        g_free(pr);
    }
}

static void nat_built(HD5770Mtlr *m, int64_t t0)
{
    int64_t ns = get_clock() - t0;

    m->build_ns += ns;
    m->draw_build_ns += ns;
    m->pub.built = true;
}

/* the builds touch no state of the backend: any thread, without lk */
static NatProg *nat_prog_build(HD5770Mtlr *m, const char *src, bool keep,
                               uint64_t hash)
{
    MTLCompileOptions *o = [[MTLCompileOptions alloc] init];
    NatProg *pr = NULL;
    NSError *e = nil;
    id<MTLLibrary> lib;

    nat_exact_math(o);
    if (keep) {
        /* buffers as inputs of the function */
        o.languageVersion = MTLLanguageVersion2_3;
    }
    lib = [m->dev newLibraryWithSource:[NSString stringWithUTF8String:src]
                               options:o error:&e];
    [o release];
    if (lib) {
        pr = g_new0(NatProg, 1);
        pr->vfn = [lib newFunctionWithName:@"hd_nvs"];
        pr->ffn = [lib newFunctionWithName:@"hd_nfs"];
        [lib release];
    } else {
        fprintf(stderr, "ati-hd5770: native program %016" PRIx64
                " not built: %s\n", hash, e.localizedDescription.UTF8String);
    }
    return pr;
}

static id<MTLFunction> nat_vprog_build(HD5770Mtlr *m, const char *src,
                                       uint64_t hash)
{
    MTLCompileOptions *o = [[MTLCompileOptions alloc] init];
    id<MTLFunction> fn = nil;
    NSError *e = nil;
    id<MTLLibrary> lib;

    nat_exact_math(o);
    o.preserveInvariance = YES;
    lib = [m->dev newLibraryWithSource:[NSString stringWithUTF8String:src]
                               options:o error:&e];
    [o release];
    if (lib) {
        fn = [lib newFunctionWithName:@"hd_bvs"];
        [lib release];
    } else {
        fprintf(stderr, "ati-hd5770: native vertex program %016" PRIx64
                " not built: %s\n", hash, e.localizedDescription.UTF8String);
    }
    return fn;
}

static NatProg *nat_prog(HD5770Mtlr *m, const HD5770IR *ir,
                         const HD5770NatKey *k, const char *key)
{
    NatProg *pr = NULL;
    gpointer v;
    char *src;
    int64_t t0;

    if (g_hash_table_lookup_extended(m->progs, key, NULL, &v)) {
        return v;
    }
    src = hd5770_msl_native_fs(ir, k);
    if (src) {
        t0 = get_clock();
        pr = nat_prog_build(m, src, k->keep, ir->hash);
        nat_built(m, t0);
        m->builds++;
        g_free(src);
    }
    g_hash_table_insert(m->progs, g_strdup(key), pr);
    return pr;
}

static MTLBlendFactor nat_factor(unsigned f)
{
    switch (f) {
    case 1: return MTLBlendFactorOne;
    case 2: return MTLBlendFactorSourceColor;
    case 3: return MTLBlendFactorOneMinusSourceColor;
    case 4: return MTLBlendFactorSourceAlpha;
    case 5: return MTLBlendFactorOneMinusSourceAlpha;
    case 6: return MTLBlendFactorDestinationAlpha;
    case 7: return MTLBlendFactorOneMinusDestinationAlpha;
    case 8: return MTLBlendFactorDestinationColor;
    case 9: return MTLBlendFactorOneMinusDestinationColor;
    case 10: return MTLBlendFactorSourceAlphaSaturated;
    case 13: return MTLBlendFactorBlendColor;
    case 14: return MTLBlendFactorOneMinusBlendColor;
    case 19: return MTLBlendFactorBlendAlpha;
    case 20: return MTLBlendFactorOneMinusBlendAlpha;
    default: return MTLBlendFactorZero;
    }
}

/* combine(): function fn of CB_BLEND_CONTROL with its factors */
static void nat_blend(unsigned fn, unsigned sf, unsigned df,
                      MTLBlendOperation *op, MTLBlendFactor *s,
                      MTLBlendFactor *d)
{
    *s = nat_factor(sf);
    *d = nat_factor(df);
    switch (fn) {
    case 0: *op = MTLBlendOperationAdd; break;
    case 1: *op = MTLBlendOperationSubtract; break;
    case 2: *op = MTLBlendOperationMin; break;
    case 3: *op = MTLBlendOperationMax; break;
    case 4: *op = MTLBlendOperationReverseSubtract; break;
    default:
        *op = MTLBlendOperationAdd;
        *s = MTLBlendFactorOne;
        *d = MTLBlendFactorZero;
        break;
    }
}

/* hf, zf, sf: the formats of the pass's attachments, NF_N: none */
/* a colour buffer of a draw */
/* the vertex function of vertex stage v for a program's inputs */
static id<MTLFunction> nat_vprog(HD5770Mtlr *m, const HD5770GpuVs *v,
                                 unsigned ninputs, uint32_t flat,
                                 const char *key)
{
    id<MTLFunction> fn = nil;
    gpointer p;
    char *src;
    int64_t t0;

    if (g_hash_table_lookup_extended(m->vprogs, key, NULL, &p)) {
        return p;
    }
    src = hd5770_msl_native_vs(v->ir, ninputs, flat, v->src);
    if (src) {
        t0 = get_clock();
        fn = nat_vprog_build(m, src, v->ir->hash);
        m->vs_build_ns += get_clock() - t0;
        nat_built(m, t0);
        m->vs_builds++;
        g_free(src);
    }
    g_hash_table_insert(m->vprogs, g_strdup(key), fn);
    return fn;
}

typedef struct NatTg {
    const HD5770GpuSurf *surf;
    NatSurf *e;
    unsigned hf, out, mask, cb, comps;
    uint32_t blend, src[4];
} NatTg;

/* the key of a pipeline: its programs' and its attachments */
static char *nat_pso_key(const char *pkey, const NatTg *tg, unsigned nt,
                         unsigned zf, unsigned sf)
{
    GString *key = g_string_new(pkey);
    unsigned i;

    for (i = 0; i < nt; i++) {
        g_string_append_printf(key, "/%u/%08x/%x", tg[i].hf, tg[i].blend,
                               tg[i].mask);
    }
    g_string_append_printf(key, "/%u/%u/%u", nt, zf, sf);
    return g_string_free(key, false);
}

/* a pipeline's descriptor, without its functions */
static MTLRenderPipelineDescriptor *nat_pso_desc(const NatTg *tg, unsigned nt,
                                                 unsigned zf, unsigned sf)
{
    MTLRenderPipelineDescriptor *d;
    MTLRenderPipelineColorAttachmentDescriptor *c;
    unsigned i;

    d = [[MTLRenderPipelineDescriptor alloc] init];
    if (zf < NF_N) {
        d.depthAttachmentPixelFormat = nf_fmt[zf];
    }
    if (sf < NF_N) {
        d.stencilAttachmentPixelFormat = nf_fmt[sf];
    }
    for (i = 0; i < nt; i++) {
        uint32_t blend = tg[i].blend;
        unsigned mask = tg[i].mask;

        c = d.colorAttachments[i];
        c.pixelFormat = nf_fmt[tg[i].hf];
        c.writeMask = (mask & 1 ? MTLColorWriteMaskRed : 0) |
                      (mask & 2 ? MTLColorWriteMaskGreen : 0) |
                      (mask & 4 ? MTLColorWriteMaskBlue : 0) |
                      (mask & 8 ? MTLColorWriteMaskAlpha : 0);
        if (!(blend & (1u << 30))) {
            continue;
        }
        {
        unsigned csrc = blend & 0x1f, cfn = (blend >> 5) & 7;
        unsigned cdst = (blend >> 8) & 0x1f;
        unsigned asrc = csrc, afn = cfn, adst = cdst;
        MTLBlendOperation op;
        MTLBlendFactor sf, df;

        if ((blend >> 29) & 1) {
            asrc = (blend >> 16) & 0x1f;
            afn = (blend >> 21) & 7;
            adst = (blend >> 24) & 0x1f;
        }
        c.blendingEnabled = YES;
        nat_blend(cfn, csrc, cdst, &op, &sf, &df);
        c.rgbBlendOperation = op;
        c.sourceRGBBlendFactor = sf;
        c.destinationRGBBlendFactor = df;
        nat_blend(afn, asrc, adst, &op, &sf, &df);
        c.alphaBlendOperation = op;
        c.sourceAlphaBlendFactor = sf;
        c.destinationAlphaBlendFactor = df;
        }
    }
    return d;
}

/* any thread, without lk; d is released */
static id<MTLRenderPipelineState> nat_pso_build(HD5770Mtlr *m,
                                                MTLRenderPipelineDescriptor *d,
                                                id<MTLFunction> vfn,
                                                id<MTLFunction> ffn)
{
    id<MTLRenderPipelineState> p;
    NSError *e = nil;

    d.vertexFunction = vfn;
    d.fragmentFunction = ffn;
    p = [m->dev newRenderPipelineStateWithDescriptor:d error:&e];
    [d release];
    if (!p) {
        fprintf(stderr, "ati-hd5770: native pipeline not built: %s\n",
                e.localizedDescription.UTF8String);
    }
    return p;
}

static id<MTLRenderPipelineState> nat_pso(HD5770Mtlr *m, NatProg *pr,
                                          id<MTLFunction> vfn,
                                          const char *pkey, const NatTg *tg,
                                          unsigned nt, unsigned zf,
                                          unsigned sf)
{
    id<MTLRenderPipelineState> p;
    char *key = nat_pso_key(pkey, tg, nt, zf, sf);
    gpointer v;
    int64_t t0;

    if (g_hash_table_lookup_extended(m->psos, key, NULL, &v)) {
        g_free(key);
        return v;
    }
    t0 = get_clock();
    p = nat_pso_build(m, nat_pso_desc(tg, nt, zf, sf), vfn ? vfn : pr->vfn,
                      pr->ffn);
    nat_built(m, t0);
    m->pso_builds++;
    g_hash_table_insert(m->psos, key, p);
    return p;
}

/*
 * BUILD AHEAD: a library (kind 0 a program's, 1 a vertex stage's) or a
 * pipeline (kind 2; fkey, vkey: its libraries) to build
 */
typedef struct NatJob {
    int kind;
    char *key, *src, *fkey, *vkey;
    bool keep;
    uint64_t hash;
    MTLRenderPipelineDescriptor *desc;
    id<MTLFunction> vfn, ffn;
    void *out;
} NatJob;

#define NAT_BUILDERS    4

static void nat_job_free(gpointer p)
{
    NatJob *j = p;

    [j->desc release];
    g_free(j->key);
    g_free(j->src);
    g_free(j->fkey);
    g_free(j->vkey);
    g_free(j);
}

static void *nat_job_thread(void *opaque)
{
    HD5770Mtlr *m = opaque;

    for (;;) {
        int i = qatomic_fetch_inc(&m->job_next);
        NatJob *j;

        if (i >= m->job_end) {
            break;
        }
        j = g_ptr_array_index(m->job_run, i);
        @autoreleasepool {
            if (j->kind == 0) {
                j->out = nat_prog_build(m, j->src, j->keep, j->hash);
            } else if (j->kind == 1) {
                j->out = nat_vprog_build(m, j->src, j->hash);
            } else if (j->ffn) {
                MTLRenderPipelineDescriptor *d = j->desc;

                j->desc = nil;
                j->out = nat_pso_build(m, d, j->vfn, j->ffn);
            }
        }
    }
    return NULL;
}

/* the jobs of a, NAT_BUILDERS at a time */
static void nat_jobs_run(HD5770Mtlr *m, GPtrArray *a)
{
    GThread *th[NAT_BUILDERS - 1];
    unsigned i, n = MIN(a->len, NAT_BUILDERS);

    if (!n) {
        return;
    }
    m->job_run = a;
    m->job_next = 0;
    m->job_end = a->len;
    for (i = 0; i + 1 < n; i++) {
        th[i] = g_thread_new("hd5770-build", nat_job_thread, m);
    }
    nat_job_thread(m);
    for (i = 0; i + 1 < n; i++) {
        g_thread_join(th[i]);
    }
}

/*
 * What was noted is built and entered under its keys. Only the thread
 * that draws notes and builds; lk is not held while the builders run,
 * so syncs and fences of other threads go on.
 */
static void nat_build_ahead(HD5770Mtlr *m)
{
    int64_t t0 = get_clock();
    unsigned i;

    if (!m->jl->len && !m->jp->len) {
        return;
    }
    nat_jobs_run(m, m->jl);
    g_mutex_lock(&m->lk);
    m->ahead_lib_ns += get_clock() - t0;
    for (i = 0; i < m->jl->len; i++) {
        NatJob *j = g_ptr_array_index(m->jl, i);

        if (j->kind == 0) {
            g_hash_table_insert(m->progs, g_strdup(j->key), j->out);
            m->builds++;
        } else {
            g_hash_table_insert(m->vprogs, g_strdup(j->key), j->out);
            m->vs_builds++;
        }
    }
    for (i = 0; i < m->jp->len; i++) {
        NatJob *j = g_ptr_array_index(m->jp, i);
        NatProg *pr = g_hash_table_lookup(m->progs, j->fkey);
        id<MTLFunction> vfn = j->vkey ? g_hash_table_lookup(m->vprogs, j->vkey)
                                      : nil;

        if (pr && (!j->vkey || vfn)) {
            j->vfn = vfn ? vfn : pr->vfn;
            j->ffn = pr->ffn;
        }
    }
    g_mutex_unlock(&m->lk);
    nat_jobs_run(m, m->jp);
    g_mutex_lock(&m->lk);
    for (i = 0; i < m->jp->len; i++) {
        NatJob *j = g_ptr_array_index(m->jp, i);

        if (j->ffn) {
            g_hash_table_insert(m->psos, g_strdup(j->key), j->out);
            m->pso_builds++;
            m->ahead_psos++;
        }
    }
    m->ahead++;
    m->ahead_libs += m->jl->len;
    m->ahead_ns += get_clock() - t0;
    m->build_ns += get_clock() - t0;
    g_ptr_array_set_size(m->jl, 0);
    g_ptr_array_set_size(m->jp, 0);
    g_hash_table_remove_all(m->jk);
    g_mutex_unlock(&m->lk);
}

static void nat_note_pso(HD5770Mtlr *m, const char *pkey, const char *fkey,
                         const char *vkey, const NatTg *tg, unsigned nt,
                         unsigned zf, unsigned sf)
{
    char *key = nat_pso_key(pkey, tg, nt, zf, sf);
    NatJob *j;

    if (g_hash_table_contains(m->psos, key) ||
        g_hash_table_contains(m->jk, key)) {
        g_free(key);
        return;
    }
    j = g_new0(NatJob, 1);
    j->kind = 2;
    j->key = key;
    j->fkey = g_strdup(fkey);
    j->vkey = g_strdup(vkey);
    j->desc = nat_pso_desc(tg, nt, zf, sf);
    g_ptr_array_add(m->jp, j);
    g_hash_table_add(m->jk, g_strdup(key));
}

/*
 * A draw looked at ahead: its program fkey of key k, the vertex stage's
 * and the pipeline over them (pkey: fkey so far) are noted unless built
 * or noted. The pipeline's depth and stencil formats are the pass's if
 * the draw would go on in the pass before it; both forms are noted then.
 */
static void nat_note(HD5770Mtlr *m, const HD5770GpuDraw *d,
                     const HD5770NatKey *k, GString *pkey, const NatTg *tg,
                     unsigned nt, bool hz, bool hs)
{
    const HD5770IR *ir = d->ir;
    const HD5770GpuVs *v = d->vs;
    char *fkey = g_strdup(pkey->str), *vkey = NULL, *src;
    unsigned zf = !hz ? NF_N : d->zfmt == 1 ? NF_D16 :
                  d->zfmt == 2 ? NF_D24 : NF_D32;
    unsigned sf = hs ? NF_S8 : NF_N, i;
    uint64_t rt = nt ? tg[0].surf->s.base + 1 +
                       ((uint64_t)tg[0].surf->space << 63) : 0;
    uint64_t z = hz ? d->zs.s.base + 1 : 0, s = hs ? d->ss.s.base + 1 : 0;
    NatJob *j;
    bool reuse;

    m->ahead_draws++;
    if (!g_hash_table_contains(m->progs, fkey) &&
        !g_hash_table_contains(m->jk, fkey)) {
        char *fs = hd5770_msl_native_fs(ir, k);

        if (!fs) {
            g_free(fkey);
            return;
        }
        j = g_new0(NatJob, 1);
        j->key = g_strdup(fkey);
        j->src = fs;
        j->keep = k->keep;
        j->hash = ir->hash;
        g_ptr_array_add(m->jl, j);
        g_hash_table_add(m->jk, g_strdup(fkey));
    }
    if (v) {
        uint32_t fl = hd5770_msl_native_flat(ir, k);
        GString *vk = g_string_new(NULL);

        g_string_printf(vk, "%016" PRIx64 "/%u/%u/%08x", v->ir->hash,
                        v->ir->ninsts, k->ninputs, fl);
        for (i = 0; i < k->ninputs; i++) {
            g_string_append_printf(vk, ".%d", v->src[i]);
        }
        g_string_append_printf(pkey, "|%s", vk->str);
        vkey = g_string_free(vk, false);
        if (!g_hash_table_contains(m->vprogs, vkey) &&
            !g_hash_table_contains(m->jk, vkey)) {
            src = hd5770_msl_native_vs(v->ir, k->ninputs, fl, v->src);
            if (!src) {
                g_free(fkey);
                g_free(vkey);
                return;
            }
            j = g_new0(NatJob, 1);
            j->kind = 1;
            j->key = g_strdup(vkey);
            j->src = src;
            j->hash = v->ir->hash;
            g_ptr_array_add(m->jl, j);
            g_hash_table_add(m->jk, g_strdup(vkey));
        }
    }
    if (!m->dp_set) {
        m->dp_set = true;
        m->dp.on = m->pass_on;
        m->dp.rt = m->cur ? m->cur->s.base + 1 +
                            ((uint64_t)m->cur->space << 63) : 0;
        m->dp.z = m->cur_z ? m->cur_z->s.base + 1 : 0;
        m->dp.s = m->cur_s ? m->cur_s->s.base + 1 : 0;
        m->dp.zf = m->cur_z ? m->cur_z->hf : NF_N;
        m->dp.sf = m->cur_s ? m->cur_s->hf : NF_N;
        m->dp.nt = m->ncurm + 1;
    }
    reuse = m->dp.on && m->dp.rt == rt && (!z || m->dp.z == z) &&
            (!s || m->dp.s == s) && m->dp.nt == MAX(nt, 1);
    if (reuse && (m->dp.zf != zf || m->dp.sf != sf)) {
        nat_note_pso(m, pkey->str, fkey, vkey, tg, nt, m->dp.zf, m->dp.sf);
    } else if (!reuse) {
        m->dp.on = true;
        m->dp.rt = rt;
        m->dp.z = z;
        m->dp.s = s;
        m->dp.zf = zf;
        m->dp.sf = sf;
        m->dp.nt = MAX(nt, 1);
    }
    nat_note_pso(m, pkey->str, fkey, vkey, tg, nt, zf, sf);
    g_free(fkey);
    g_free(vkey);
}

static void nat_pso_free(gpointer p)
{
    [(id)p release];
}

/* DB_DEPTH_CONTROL */
#define DC_STENCIL      (1u << 0)
#define DC_Z            (1u << 1)
#define DC_Z_WRITE      (1u << 2)
#define DC_BACKFACE     (1u << 7)

/* one face's stencil test as db_test() in ati_hd5770_draw.c makes it */
static MTLStencilDescriptor *nat_stencil(uint32_t dctl, uint32_t srm,
                                         bool back)
{
    MTLStencilDescriptor *sd = [[[MTLStencilDescriptor alloc] init]
                                autorelease];
    unsigned sh = back ? 12 : 0;

    sd.writeMask = (srm >> 16) & 0xff;
    if (dctl & DC_STENCIL) {
        /* the operations and the functions are numbered as Metal's */
        sd.readMask = (srm >> 8) & 0xff;
        sd.stencilCompareFunction = (dctl >> (8 + sh)) & 7;
        sd.stencilFailureOperation = (dctl >> (11 + sh)) & 7;
        sd.depthStencilPassOperation = (dctl >> (14 + sh)) & 7;
        sd.depthFailureOperation = (dctl >> (17 + sh)) & 7;
    } else {
        /* a clear: the reference */
        sd.stencilCompareFunction = MTLCompareFunctionAlways;
        sd.stencilFailureOperation = MTLStencilOperationReplace;
        sd.depthStencilPassOperation = MTLStencilOperationReplace;
        sd.depthFailureOperation = MTLStencilOperationReplace;
    }
    return sd;
}

/* the tests of a draw with depth plane z and stencil plane st */
static id<MTLDepthStencilState> nat_ds(HD5770Mtlr *m, const HD5770GpuDraw *d,
                                       bool z, bool st)
{
    MTLDepthStencilDescriptor *dd;
    id<MTLDepthStencilState> ds;
    bool two = d->dctl & DC_BACKFACE;
    unsigned zfn = (d->dctl >> 4) & 7;
    char key[80];
    gpointer v;

    snprintf(key, sizeof(key), "%d%d/%08x/%04x/%04x", z, st,
             d->dctl & (st ? ~0u : 0x7e), st ? (d->srm[0] >> 8) & 0xffff : 0,
             st && two ? (d->srm[1] >> 8) & 0xffff : 0);
    if (g_hash_table_lookup_extended(m->dss, key, NULL, &v)) {
        return v;
    }
    dd = [[MTLDepthStencilDescriptor alloc] init];
    if (z && (d->dctl & DC_Z)) {
        dd.depthCompareFunction = zfn;
        dd.depthWriteEnabled = (d->dctl & DC_Z_WRITE) != 0;
    }
    if (st) {
        dd.frontFaceStencil = nat_stencil(d->dctl, d->srm[0], false);
        dd.backFaceStencil = nat_stencil(d->dctl, d->srm[two], two);
    }
    ds = [m->dev newDepthStencilStateWithDescriptor:dd];
    [dd release];
    g_hash_table_insert(m->dss, g_strdup(key), ds);
    return ds;
}

/* n bytes of the ring, aligned; NULL when it has no room left */
static void *nat_ring(HD5770Mtlr *m, size_t n, size_t *at)
{
    size_t a = ROUND_UP(m->ring_at, 256);

    if (a + n > NAT_RING) {
        return NULL;
    }
    *at = a;
    m->ring_at = a + n;
    return (uint8_t *)m->ring.contents + a;
}

/*
 * the ring is full: another one that no pass in flight reads, made if
 * there is none yet; false: all are read, the caller waits
 */
static bool nat_ring_next(HD5770Mtlr *m)
{
    uint64_t done = qatomic_read(&m->cb_passes);
    unsigned i;

    m->ring_pass[m->ring_i] = m->passes;
    for (i = 0; i < NAT_RINGS; i++) {
        if (i == m->ring_i) {
            continue;
        }
        if (!m->rings[i]) {
            m->rings[i] = [m->dev newBufferWithLength:NAT_RING
                                  options:MTLResourceStorageModeShared];
            if (!m->rings[i]) {
                continue;
            }
            break;
        }
        if (m->ring_pass[i] <= done) {
            break;
        }
    }
    if (i == NAT_RINGS) {
        return false;
    }
    m->ring_i = i;
    m->ring = m->rings[i];
    m->ring_at = 0;
    m->ring_swaps++;
    return true;
}

/* the host format of a texture, NF_N: none */
static unsigned nat_tex_fmt(const HD5770Mtlr *m, const HD5770GpuTex *t)
{
    if (t->bc) {
        /* the host decodes the blocks */
        unsigned sg = t->sgn & (t->bc == 5 ? 3 : 1);

        if (!m->bc_ok || (t->bc == 5 && (sg == 1 || sg == 2))) {
            return NF_N;
        }
        return t->bc == 1 ? NF_BC1 : t->bc == 2 ? NF_BC2 :
               t->bc == 3 ? NF_BC3 :
               t->bc == 4 ? (sg ? NF_BC4S : NF_BC4) :
               sg ? NF_BC5S : NF_BC5;
    }
    if (t->unorm8) {
        return NF_RGBA8;
    }
    if (t->fmt == HD5770_FMT_16_16_16_16_FLOAT && t->is_float &&
        t->comps == 4) {
        return NF_RGBA16F;
    }
    if (t->fmt == HD5770_FMT_32_FLOAT && t->is_float && t->comps == 1) {
        return NF_R32F;
    }
    if (t->comps == 4 &&
        ((t->fmt == HD5770_FMT_32_32_32_32_FLOAT && t->is_float) ||
         (t->fmt == HD5770_FMT_32_32_32_32 && t->num == HD5770_NUM_INT))) {
        /* floats, or their bits as integers */
        return NF_RGBA32F;
    }
    if (t->fmt == HD5770_FMT_16_16_16_16 && t->num == HD5770_NUM_NORM &&
        !t->sgn && !t->is_float && t->comps == 4) {
        return NF_RGBA16;
    }
    if (t->num == HD5770_NUM_NORM && !t->sgn && t->comps == 2 &&
        t->fmt == HD5770_FMT_8_8) {
        return NF_RG8;
    }
    if (t->num == HD5770_NUM_NORM && !t->sgn && !t->is_float &&
        t->comps == 4 && t->fmt == HD5770_FMT_4_4_4_4 && m->pk_ok) {
        /* the host's channels are the components in reverse */
        return NF_ABGR4;
    }
    if (t->num != HD5770_NUM_NORM || t->sgn || t->comps != 1) {
        return NF_N;
    }
    return t->fmt == HD5770_FMT_8 ? NF_R8 :
           t->fmt == HD5770_FMT_16 ? NF_R16 : NF_N;
}

/*
 * A colour buffer: its host format, the class of the function's output
 * for it (HD5770NatKey), its write mask in the host's channels; the
 * blend is dropped where it cannot apply. ND_TAKEN, else why not.
 */
static unsigned nat_rt(const HD5770Mtlr *m, const HD5770Surf *s,
                       uint32_t fmt, uint32_t num,
                       uint32_t comps, bool is_float, bool unorm8,
                       uint32_t *src, uint32_t wmask, uint32_t *blend,
                       unsigned *rhf, unsigned *out, unsigned *mask)
{
    unsigned i, perm = 0;

    *out = 0;
    *mask = 0;
    if (num == 4 && comps == 1 &&
        (fmt == HD5770_FMT_8 || fmt == HD5770_FMT_32)) {
        /* an integer: the bits of its channel */
        *out = 1;
        *blend = 0;
        *mask = (wmask >> src[0]) & 1 ? 0xf : 0;
        *rhf = fmt == HD5770_FMT_8 ? NF_R8 : NF_RGBA8;
        return ND_TAKEN;
    }
    if (num == 0 && !is_float && comps == 1 && fmt == HD5770_FMT_8) {
        *mask = (wmask >> src[0]) & 1 ? 0xf : 0;
        *rhf = NF_R8;
        return *blend ? ND_BLEND_ORDER : ND_TAKEN;
    }
    if (num == 0 && !is_float && comps == 2 && fmt == HD5770_FMT_8_8 &&
        src[0] < 4 && src[1] < 4) {
        /* channel i is component src[i] */
        *mask = ((wmask >> src[0]) & 1) | ((wmask >> src[1]) & 1) << 1;
        *rhf = NF_RG8;
        return *blend ? ND_BLEND_ORDER : ND_TAKEN;
    }
    for (i = 0; i < 4; i++) {
        perm |= src[i] < 4 ? 1u << src[i] : 0x10;
    }
    *rhf = NF_RGBA8;
    if ((fmt == HD5770_FMT_16_16_16_16_FLOAT ||
         fmt == HD5770_FMT_32_32_32_32 ||
         fmt == HD5770_FMT_32_32_32_32_FLOAT) && is_float && comps == 4 &&
        perm == 0xf) {
        /* floats as they are */
        *out = 2;
        *rhf = s->bpe == 16 ? NF_RGBA32F : NF_RGBA16F;
    } else if (fmt == HD5770_FMT_16_16_16_16 && num == 0 && !is_float &&
               comps == 4 && perm == 0xf) {
        *rhf = NF_RGBA16;
    } else if (fmt == HD5770_FMT_4_4_4_4 && num == 0 && !is_float &&
               comps == 4 && perm == 0xf && m->pk_ok) {
        /* the host's channels are the components in reverse */
        uint32_t t;

        *rhf = NF_ABGR4;
        t = src[0], src[0] = src[3], src[3] = t;
        t = src[1], src[1] = src[2], src[2] = t;
    } else if (!unorm8 || comps != 4 || is_float || perm != 0xf) {
        return s->bpe > 8 ? ND_RT_WIDE : ND_RT_FMT;
    }
    if (*blend && src[3] != 3) {
        return ND_BLEND_ORDER;
    }
    for (i = 0; i < 4; i++) {
        *mask |= ((wmask >> src[i]) & 1) << i;
    }
    return ND_TAKEN;
}

static bool nat_draw_locked(HD5770Mtlr *m, const HD5770GpuDraw *d)
{
    const HD5770IR *ir = d->ir;
    HD5770NatKey k = { 0 };
    NatSurf *rt = NULL, *zs = NULL, *ss = NULL, *ts[HD5770_GPU_TEX];
    NatSurf *lv[HD5770_NAT_LEVELS], *ex[HD5770_NAT_LEVELS], *at[10], *pz, *ps;
    NatTg tg[8];
    unsigned nrt, nt = 0;
    unsigned nlv = 0, nex = 0, nat = 0, vw, vh, hh;
    uint32_t exr[HD5770_NAT_LEVELS];
    uint8_t tn[HD5770_GPU_TEX];
    bool ufn = false;
    bool color, reuse, zclear, zwrite;
    id<MTLDepthStencilState> ds = nil;
    NatProg *pr;
    id<MTLRenderPipelineState> pso;
    NatU *u;
    float *vb;
    uint32_t *kv;
    uint8_t *vd[HD5770_GPU_VB];
    size_t o_u, o_v, o_k, o_d[HD5770_GPU_VB];
    size_t vstride = 4 + 4 * (size_t)d->ninputs, need;
    unsigned i, j, c, mask = 0, hf[HD5770_GPU_TEX];
    int x0, y0, x1, y1, bx0 = INT_MAX, by0 = INT_MAX, bx1 = -1, by1 = -1;
    int sc[4] = { d->sc[0], d->sc[1], d->sc[2], d->sc[3] };
    GString *pkey;
    const HD5770GpuVs *v = d->vs;
    id<MTLFunction> vfn = nil;
    NatVs *nv = NULL;
    size_t o_n = 0, o_i = 0, vneed = 0;
    bool hz = d->db && d->zfmt;

    m->seq++;
    if (d->resolve) {
        return nat_decline(m, ND_RESOLVE, HD5770_WHY_BK_NATIVE_STATE);
    }
    /* no colour is written: the target is not needed */
    color = d->nmrt || !(d->db && !d->wmask);
    if (!ir || ir->vs) {
        return nat_decline(m, ND_PROG_FORM, HD5770_WHY_BK_PROG);
    }
    ufn = ir->cf;
    if (ir->zexport && d->sexport && d->db && d->has_s) {
        /* the stencil reference from the program */
        return nat_decline(m, ND_STATE_DB, HD5770_WHY_BK_NATIVE_STATE);
    }
    if (d->ninputs > 32 || ir->nres != d->ntex) {
        return nat_decline(m, ND_PROG_INPUTS, HD5770_WHY_BK_PROG);
    }
    /* vertex fetches: buffers of 8 and 32 bit elements */
    for (i = 0; i < ir->nfetches; i++) {
        const HD5770IRFetch *f = &ir->fetches[i];
        unsigned bid = (f->w[0] >> 8) & 0xff, fmt;

        if (f->kind == HD5770_IRF_TEX && (f->w[0] & 0x1f) >= 0x1c) {
            return nat_decline(m, ND_FETCH_OP, HD5770_WHY_BK_PROG);
        }
        ufn |= f->kind == HD5770_IRF_TEX && (f->w[0] & 0x1f) >= 0x18;
        if (ir->cf && f->kind != HD5770_IRF_TEX) {
            /* the CF form: texture fetches only */
            return nat_decline(m, ND_PROG_FORM, HD5770_WHY_BK_PROG);
        }
        if (f->kind != HD5770_IRF_VTX) {
            continue;
        }
        for (j = 0; j < d->nvb && d->vb[j].id != bid; j++) {
            /* search */
        }
        if (j == d->nvb) {
            /* the constant file */
            return nat_decline(m, ND_PROG_INPUTS, HD5770_WHY_BK_PROG);
        }
        fmt = (f->w[1] >> 21) & 1 ? (d->vb[j].res[2] >> 20) & 0x3f
                                  : (f->w[1] >> 22) & 0x3f;
        if (fmt != HD5770_FMT_8 && fmt != HD5770_FMT_32) {
            return nat_decline(m, ND_PROG_VTX, HD5770_WHY_BK_PROG);
        }
    }
    k.nvb = d->nvb;
    for (i = 0; i < d->nvb; i++) {
        k.vb[i].id = d->vb[i].id;
        k.vb[i].fmt = (d->vb[i].res[2] >> 20) & 0x3f;
        k.vb[i].num = (d->vb[i].res[2] >> 26) & 3;
        k.vb[i].sgn = (d->vb[i].res[2] >> 28) & 1;
    }
    nrt = d->nmrt ? d->nmrt : 1;
    for (i = 0; i < nrt && color; i++) {
        const HD5770GpuRt *r = &d->mrt[i];
        unsigned why;

        if (d->nmrt) {
            tg[nt].surf = &r->surf;
            tg[nt].cb = r->cb;
            memcpy(tg[nt].src, r->src, sizeof(tg[nt].src));
            tg[nt].comps = r->comps;
            tg[nt].blend = (r->blend & (1u << 30)) && r->blend_ok ? r->blend
                                                                  : 0;
            why = nat_rt(m, &r->surf.s, r->fmt, r->num, r->comps,
                         r->is_float, r->unorm8, tg[nt].src,
                         (ir->cmask >> r->cb) & 1 ? r->wmask : 0,
                         &tg[nt].blend, &tg[nt].hf, &tg[nt].out,
                         &tg[nt].mask);
        } else {
            tg[nt].surf = &d->rt;
            tg[nt].cb = 0;
            memcpy(tg[nt].src, d->rt_src, sizeof(tg[nt].src));
            tg[nt].comps = d->rt_comps;
            tg[nt].blend = (d->blend & (1u << 30)) && d->rt_blend ? d->blend
                                                                  : 0;
            why = nat_rt(m, &d->rt.s, d->rt_fmt, d->rt_num, d->rt_comps,
                         d->rt_float, d->rt_unorm8, tg[nt].src, d->wmask,
                         &tg[nt].blend, &tg[nt].hf, &tg[nt].out,
                         &tg[nt].mask);
        }
        if (why != ND_TAKEN) {
            return nat_decline(m, why, why == ND_BLEND_ORDER
                                       ? HD5770_WHY_BK_NATIVE_STATE
                                       : HD5770_WHY_BK_SURF);
        }
        mask |= tg[nt].mask;
        /* a buffer nothing is written to is left out */
        nt += tg[nt].mask != 0;
    }
    if (!mask && d->db) {
        color = false;
        nt = 0;
    }
    if (!nt) {
        tg[0].hf = NF_RGBA8;
        tg[0].out = 0;
        tg[0].comps = 4;
        tg[0].blend = 0;
        tg[0].cb = 0;
        for (i = 0; i < 4; i++) {
            tg[0].src[i] = i;
        }
    }
    k.out = tg[0].out;
    k.rt_comps = tg[0].comps;
    k.rt_bytes = nf_bpe[tg[0].hf];
    k.ncol = d->nmrt ? nt : 0;
    /*
     * only flow control can leave an export out; with blending the buffer
     * read would be blended again
     */
    k.keep = ir->cf && k.ncol > 1 && m->sep_ds;
    for (i = 0; i < k.ncol; i++) {
        k.keep = k.keep && !tg[i].blend;
    }
    for (i = 0; i < k.ncol; i++) {
        k.col[i].cb = tg[i].cb;
        k.col[i].out = tg[i].out;
        k.col[i].comps = tg[i].comps;
        for (j = 0; j < 4; j++) {
            k.col[i].src[j] = tg[i].src[j];
        }
    }
    if (d->cliprule != 0xffff) {
        static const uint16_t one[4] = { 0xaaaa, 0xcccc, 0xf0f0, 0xff00 };

        for (i = 0; i < 4 && d->cliprule != one[i]; i++) {
            /* search */
        }
        if (i < 4) {
            /* inside one rectangle: the scissor */
            sc[0] = MAX(sc[0], d->cr[i][0]);
            sc[1] = MAX(sc[1], d->cr[i][1]);
            sc[2] = MIN(sc[2], d->cr[i][2]);
            sc[3] = MIN(sc[3], d->cr[i][3]);
        } else {
            k.clip = true;
        }
    }
    for (i = 0; i < d->ntex; i++) {
        const HD5770GpuTex *t = &d->tex[i];

        if (t->dim == 0 || t->dim == 4 || t->dim > 5) {
            /* 1D */
            return nat_decline(m, ND_TEX_KIND, HD5770_WHY_BK_SURF);
        }
        k.dim[i] = t->dim;
        k.nsl[i] = 1;
        if (t->dim != 1 && t->slices > 1) {
            /* each slice a texture, in VRAM */
            if (t->slices > 64 || t->surf.space) {
                return nat_decline(m, t->dim == 2 ? ND_TEX_3D :
                                      t->dim == 3 ? ND_TEX_CUBE : ND_TEX_ARRAY,
                                   HD5770_WHY_BK_SURF);
            }
            k.nsl[i] = t->slices;
            ufn = true;
        }
        k.mip[i] = t->mip ? t->last_lv : 0;
        nlv += k.mip[i];
        /* a unit with levels or slices: all its textures, in an array */
        tn[i] = 0;
        for (c = 0; c <= k.mip[i]; c++) {
            tn[i] += t->dim == 2 ? MAX(k.nsl[i] >> c, 1u) : k.nsl[i];
        }
        nex += tn[i] > 1 ? tn[i] : 0;
        hf[i] = nat_tex_fmt(m, t);
        if (hf[i] == NF_N) {
            return nat_decline(m, t->bc ? ND_TEX_BC : ND_TEX_FMT,
                               HD5770_WHY_BK_SURF);
        }
    }
    if (d->ntex > 14 && (nlv || ufn)) {
        return nat_decline(m, ND_TEX_UNITS, HD5770_WHY_BK_SURF);
    }
    while (nex > HD5770_NAT_LEVELS) {
        /*
         * more textures than a function takes: the smallest level of the
         * unit with most levels is left out, the one before it is read
         */
        for (i = 1, j = 0; i < d->ntex; i++) {
            if (k.mip[i] > k.mip[j]) {
                j = i;
            }
        }
        if (!k.mip[j]) {
            return nat_decline(m, ND_TEX_MANY, HD5770_WHY_BK_SURF);
        }
        c = d->tex[j].dim == 2 ? MAX(k.nsl[j] >> k.mip[j], 1u) : k.nsl[j];
        k.mip[j]--;
        nlv--;
        nex -= tn[j] - c > 1 ? c : tn[j];
        tn[j] -= c;
        m->lv_cut++;
    }
    for (i = 0; i < d->ntri; i++) {
        const HD5770GpuTri *t = &d->tri[i];

        for (j = 0; j < 3 && d->persp; j++) {
            if (!(t->w[j] > 0.0f) || isinf(t->w[j])) {
                return nat_decline(m, ND_W, HD5770_WHY_BK_NATIVE_STATE);
            }
        }
        bx0 = MIN(bx0, t->x0);
        by0 = MIN(by0, t->y0);
        bx1 = MAX(bx1, t->x1 + 1);
        by1 = MAX(by1, t->y1 + 1);
    }
    if (v) {
        /* no triangles here: the scissor */
        bx0 = sc[0];
        by0 = sc[1];
        bx1 = sc[2];
        by1 = sc[3];
        if (!m->bvram || (d->rctl & 1 && d->db)) {
            return nat_decline(m, ND_VS, HD5770_WHY_BK_NATIVE_STATE);
        }
        for (i = 0; i < v->nvb; i++) {
            if (v->vb[i].buf.space && !m->bram) {
                return nat_decline(m, ND_VS, HD5770_WHY_BK_NATIVE_STATE);
            }
        }
    }
    if (m->dry) {
        /* no surface is looked up */
        zclear = hz && (d->rctl & 1);
        k.zexp = hz && !zclear && d->zexport && ir->zexport;
        k.zout = hz && !zclear && (k.zexp ||
                                   !(d->zmin <= 0.0f && d->zmax >= 1.0f));
        goto keys;
    }
    vw = vh = hh = UINT32_MAX;
    for (i = 0; i < nt; i++) {
        tg[i].e = tg[i].surf->space && !m->pub.done_ram ? NULL :
                  nat_surf(m, tg[i].surf, tg[i].hf);
        if (!tg[i].e) {
            return nat_decline(m, ND_RT_SURF, HD5770_WHY_BK_SURF);
        }
        at[nat++] = tg[i].e;
        hh = MIN(hh, tg[i].surf->s.height);
    }
    rt = nt ? tg[0].e : NULL;
    if (d->db && d->zfmt) {
        zs = d->zs.space ? NULL :
             nat_surf(m, &d->zs, d->zfmt == 1 ? NF_D16 :
                                 d->zfmt == 2 ? NF_D24 : NF_D32);
        if (!zs) {
            return nat_decline(m, ND_DB_SURF, HD5770_WHY_BK_SURF);
        }
        at[nat++] = zs;
        hh = MIN(hh, d->zs.s.height);
    }
    if (d->db && d->has_s) {
        if (zs && !m->sep_ds) {
            return nat_decline(m, ND_DB_SURF, HD5770_WHY_BK_SURF);
        }
        ss = d->ss.space ? NULL : nat_surf(m, &d->ss, NF_S8);
        if (!ss) {
            return nat_decline(m, ND_DB_SURF, HD5770_WHY_BK_SURF);
        }
        at[nat++] = ss;
        hh = MIN(hh, d->ss.s.height);
    }
    for (i = 0; i < nat; i++) {
        vw = MIN(vw, at[i]->s.pitch);
        vh = MIN(vh, at[i]->s.height);
        for (j = 0; j < i; j++) {
            if (nat_alias(at[i], at[j])) {
                return nat_decline(m, ND_DB_SURF, HD5770_WHY_BK_SURF);
            }
        }
    }
    hh = MIN(hh, vh);
    x0 = MAX(MAX(sc[0], bx0), 0);
    y0 = MAX(MAX(sc[1], by0), 0);
    x1 = MIN(MIN(sc[2], bx1), (int)vw);
    y1 = MIN(MIN(sc[3], by1), (int)hh);
    if ((!d->ntri && !(v && v->ntidx && v->cull != 3)) || !nat || x0 >= x1 ||
        y0 >= y1 || (!mask && !d->db)) {
        m->nd[ND_EMPTY]++;
        return true;
    }
    for (i = 0, nlv = 0, nex = 0; i < d->ntex; i++) {
        const HD5770GpuTex *t = &d->tex[i];
        unsigned bk = NF_BLK(hf[i]), sl, ns;

        for (c = 0; c <= k.mip[i]; c++) {
            const HD5770GpuSurf *ls = c ? &t->lv[c] : &t->surf;
            uint32_t lw = c ? t->lw[c] : t->w, lh = c ? t->lh[c] : t->h;

            ns = t->dim == 2 ? MAX(k.nsl[i] >> c, 1u) : k.nsl[i];
            for (sl = 0; sl < ns; sl++) {
                HD5770GpuSurf sv = *ls;
                NatSurf *e;

                sv.s.base += sl * t->slice_bytes[c];
                sv.voff += sl * t->slice_bytes[c];
                e = NULL;
                if (m->ztex && hf[i] == NF_R32F && tn[i] <= 1 && !sv.space) {
                    /* a depth plane of floats: its own texture is read */
                    e = nat_find(m, &sv, NF_D32);
                }
                if (!e) {
                    e = (c || sl) && ls->space ? NULL
                                               : nat_surf(m, &sv, hf[i]);
                }
                if (!e || lw > e->s.pitch * bk || lh > e->s.height * bk) {
                    return nat_decline(m, ND_TEX_SURF, HD5770_WHY_BK_SURF);
                }
                for (j = 0; j < nat; j++) {
                    if (e == at[j] || nat_alias(e, at[j])) {
                        return nat_decline(m, ND_TEX_TARGET,
                                           HD5770_WHY_BK_SURF);
                    }
                }
                if (!c && !sl) {
                    ts[i] = e;
                }
                if (tn[i] > 1) {
                    exr[nex] = ls->s.height;
                    ex[nex++] = e;
                }
                if (c && !sl) {
                    lv[nlv++] = e;
                }
            }
        }
    }
    zclear = zs && (d->rctl & 1);
    zwrite = zs && (d->dctl & DC_Z) && (d->dctl & DC_Z_WRITE);
    /* a depth range other than 0..1: clamped in the function */
    k.zexp = zs && !zclear && d->zexport && ir->zexport;
    k.zout = zs && !zclear && (k.zexp ||
                               !(d->zmin <= 0.0f && d->zmax >= 1.0f));
keys:
    k.ninputs = d->ninputs;
    k.pos_gpr = d->pos_gpr >= 0 && d->pos_gpr < 127 ? d->pos_gpr : -1;
    k.flat = d->flat;
    for (i = 0; i < 4; i++) {
        k.rt_src[i] = tg[0].src[i];
    }
    pkey = g_string_new(NULL);
    g_string_printf(pkey, "%016" PRIx64 "/%u/%d/%08x/%u%u%u%u/%u/%u%u%u%u",
                    ir->hash, k.ninputs, k.pos_gpr, k.flat, k.rt_src[0],
                    k.rt_src[1], k.rt_src[2], k.rt_src[3], ir->ninsts,
                    k.out, k.rt_comps, k.rt_bytes, k.clip);
    for (i = 0; i < k.nvb; i++) {
        g_string_append_printf(pkey, "/%u.%u.%u.%u", k.vb[i].id, k.vb[i].fmt,
                               k.vb[i].num, k.vb[i].sgn);
    }
    g_string_append_printf(pkey, "/%u%u%u", k.zout, k.zexp, k.keep);
    for (i = 0; i < k.ncol; i++) {
        g_string_append_printf(pkey, "/%u.%u.%u.%u%u%u%u", k.col[i].cb,
                               k.col[i].out, k.col[i].comps, k.col[i].src[0],
                               k.col[i].src[1], k.col[i].src[2],
                               k.col[i].src[3]);
    }
    for (i = 0; i < d->ntex && nlv; i++) {
        g_string_append_printf(pkey, ".%u", k.mip[i]);
    }
    for (i = 0; i < d->ntex && ufn; i++) {
        g_string_append_printf(pkey, "/%u.%u", k.dim[i], k.nsl[i]);
    }
    if (m->dry) {
        nat_note(m, d, &k, pkey, tg, nt, hz, d->db && d->has_s);
        g_string_free(pkey, true);
        return true;
    }
    pr = nat_prog(m, ir, &k, pkey->str);
    if (!pr) {
        g_string_free(pkey, true);
        return nat_decline(m, ND_PROG_EMIT, HD5770_WHY_BK_PROG);
    }
    if (v) {
        uint32_t fl = hd5770_msl_native_flat(ir, &k);
        GString *vk = g_string_new(NULL);

        g_string_printf(vk, "%016" PRIx64 "/%u/%u/%08x", v->ir->hash,
                        v->ir->ninsts, k.ninputs, fl);
        for (i = 0; i < k.ninputs; i++) {
            g_string_append_printf(vk, ".%d", v->src[i]);
        }
        vfn = nat_vprog(m, v, k.ninputs, fl, vk->str);
        g_string_append_printf(pkey, "|%s", vk->str);
        g_string_free(vk, true);
        if (!vfn) {
            g_string_free(pkey, true);
            return nat_decline(m, ND_VS, HD5770_WHY_BK_PROG);
        }
        vneed = ROUND_UP(sizeof(NatVs), 16) +
                ROUND_UP((v->ir->nconsts + 1) * 4, 16) + 4096;
        for (i = 0; i < v->nvb; i++) {
            vneed += v->vb[i].buf.npages * 4;
        }
    }
    need = sizeof(NatU) + (size_t)d->ntri * 3 * vstride * sizeof(float) +
           MAX(ir->nconsts, 1u) * sizeof(uint32_t) + 3 * 256 + vneed + 256 +
           (v ? (size_t)v->ntidx * 4 + 256 : 0);
    for (i = 0; i < d->nvb; i++) {
        need += d->vb[i].len + 4 + 256;
    }
    if (need > NAT_RING) {
        g_string_free(pkey, true);
        return nat_decline(m, ND_ROOM, HD5770_WHY_BK_ROOM);
    }
    /* uploads and write-backs wait for the GPU and empty the ring */
    for (i = 0; i < d->ntex; i++) {
        if (tn[i] <= 1) {
            nat_ready(m, ts[i], 0, d->tex[i].surf.s.height);
        }
    }
    for (i = 0; i < nex; i++) {
        nat_ready(m, ex[i], 0, exr[i]);
    }
    for (i = 0; i < nat; i++) {
        nat_ready(m, at[i], y0, y1);
    }
    if (m->ring_at + need > NAT_RING && !nat_ring_next(m)) {
        nat_wait(m, NW_RING);
    }
    /* the pass goes on if it has what the draw needs, at the same size */
    reuse = m->pass_on && m->cur == rt && (!zs || m->cur_z == zs) &&
            (!ss || m->cur_s == ss) && m->vw == vw && m->vh == vh &&
            m->ncurm == (nt ? nt - 1 : 0);
    for (i = 1; i < nt && reuse; i++) {
        reuse = m->curm[i - 1] == tg[i].e;
    }
    pz = reuse ? m->cur_z : zs;
    ps = reuse ? m->cur_s : ss;
    pso = nat_pso(m, pr, vfn, pkey->str, tg, nt, pz ? pz->hf : NF_N,
                  ps ? ps->hf : NF_N);
    g_string_free(pkey, true);
    if (!pso) {
        return nat_decline(m, ND_PROG_BUILD, HD5770_WHY_BK_PROG);
    }
    if (pz || ps) {
        ds = zs || ss ? nat_ds(m, d, zs != NULL, ss != NULL) : m->ds_off;
        if (!ds) {
            return nat_decline(m, ND_PROG_BUILD, HD5770_WHY_BK_PROG);
        }
    }
    u = nat_ring(m, sizeof(NatU), &o_u);
    vb = nat_ring(m, (size_t)d->ntri * 3 * vstride * sizeof(float), &o_v);
    kv = nat_ring(m, MAX(ir->nconsts, 1u) * sizeof(uint32_t), &o_k);
    memset(u, 0, sizeof(*u));
    if (v) {
        uint8_t *st = nat_ring(m, vneed, &o_n);
        uint32_t *ix = nat_ring(m, (size_t)v->ntidx * 4, &o_i);
        size_t o = ROUND_UP(sizeof(NatVs), 16);

        nv = (NatVs *)st;
        memset(nv, 0, sizeof(*nv));
        memcpy(ix, v->tidx, (size_t)v->ntidx * 4);
        nv->v.nslot = v->nslot;
        nv->v.ninst = v->ninst;
        nv->v.np = v->np;
        nv->v.kv_off = o;
        memcpy(st + o, v->kv, v->ir->nconsts * 4);
        o += ROUND_UP((v->ir->nconsts + 1) * 4, 16);
        nv->v.cfile_off = o;
        memcpy(st + o, v->cfile, 4096);
        o += 4096;
        nv->v.nvb = v->nvb;
        nv->v.base_vtx = v->base_vtx;
        nv->v.start_inst = v->start_inst;
        for (i = 0; i < v->nvb; i++) {
            const HD5770GpuSurf *b = &v->vb[i].buf;

            nv->v.vb[i].id = v->vb[i].id;
            memcpy(nv->v.vb[i].res, v->vb[i].res, sizeof(nv->v.vb[i].res));
            nv->v.vb[i].b.voff = b->voff;
            if (b->space) {
                nv->v.vb[i].b.space = 1;
                nv->v.vb[i].b.pt_off = o;
                nv->v.vb[i].b.pg0 = b->pg0;
                memcpy(st + o, b->pages, b->npages * 4);
                o += b->npages * 4;
            }
        }
        nv->vte = v->vte;
        nv->sc_mode = v->sc_mode;
        nv->wofs = v->window_offset;
        nv->clip = (v->clip_on ? 1 : 0) | (v->clip_dx ? 2 : 0) |
                   (v->clip_near ? 4 : 0) | (v->clip_far ? 8 : 0);
        memcpy(nv->vport, v->vport, sizeof(nv->vport));
        nv->vpx = 2.0f / vw;
        nv->vpy = 2.0f / vh;
        nv->pc = v->pix_center;
        nv->zoff = v->zoff_const;
        memcpy(nv->def, v->def, sizeof(nv->def));
    }
    for (i = 0; i < d->nvb; i++) {
        vd[i] = nat_ring(m, d->vb[i].len + 4, &o_d[i]);
        memcpy(vd[i], d->vb[i].data, d->vb[i].len);
        u->vbi[i][0] = (d->vb[i].res[2] >> 8) & 0x7ff;
        u->vbi[i][1] = d->vb[i].len;
    }
    for (i = 0, nlv = 0; i < d->ntex; i++) {
        const HD5770GpuTex *t = &d->tex[i];

        u->tsize[i][0] = t->w;
        u->tsize[i][1] = t->h;
        u->tsize[i][2] = ts[i]->s.pitch * NF_BLK(hf[i]);
        u->tsize[i][3] = ts[i]->s.height * NF_BLK(hf[i]);
        u->twrap[i][0] = t->samp[0] & 7;
        u->twrap[i][1] = (t->samp[0] >> 3) & 7;
        u->twrap[i][2] = ((t->samp[0] >> 9) & 1) || ((t->samp[0] >> 11) & 1);
        u->tcmp[i][0] = ((t->samp[0] >> 22) & 7) + 1;
        u->tcmp[i][1] = !t->is_float && t->num == HD5770_NUM_NORM &&
                        !(t->sgn & 1) && !t->bc && t->bits[0] &&
                        t->bits[0] < 32 ? (1u << t->bits[0]) - 1 : 0;
        u->tcmp[i][2] = ((t->samp[0] >> 13) & 3) == 2;
        u->tcmp[i][3] = (t->samp[0] >> 6) & 7;
        for (c = 0; c < 4; c++) {
            u->tsel[i][c] = hf[i] == NF_ABGR4 && t->dst_sel[c] < 4 ?
                            3 - t->dst_sel[c] : t->dst_sel[c];
            u->tborder[i][c] = t->border[c];
        }
        u->tres[i][0] = t->base_lv;
        u->tres[i][1] = t->levels;
        u->tres[i][2] = t->depth;
        if (k.mip[i]) {
            u->tlod[i][0] = t->lod_bias;
            u->tlod[i][1] = t->min_lod;
            u->tlod[i][2] = t->max_lod;
            u->tmip[i][0] = t->mip_filter;
            u->tmip[i][1] = MIN(t->last_lv, k.mip[i]);
            u->tmip[i][2] = (t->samp[0] >> 9) & 1;
            u->tmip[i][3] = (t->samp[0] >> 11) & 1;
        }
        for (c = 1; c <= k.mip[i]; c++, nlv++) {
            u->lsize[nlv][0] = t->lw[c];
            u->lsize[nlv][1] = t->lh[c];
            u->lsize[nlv][2] = lv[nlv]->s.pitch * NF_BLK(hf[i]);
            u->lsize[nlv][3] = lv[nlv]->s.height * NF_BLK(hf[i]);
        }
    }
    u->vp[0] = 2.0f / vw;
    u->vp[1] = 2.0f / vh;
    u->zr[0] = d->zmin;
    u->zr[1] = d->zmax;
    u->alpha_ref = d->alpha_ref;
    u->alpha_func = d->alpha_test ? d->alpha_func : 7;
    u->pc = d->pix_center;
    u->persp = d->persp;
    u->cliprule = d->cliprule;
    memcpy(u->cr, d->cr, sizeof(u->cr));
    u->bconst = d->bool_const;
    memcpy(u->lconst, d->loop_const, sizeof(u->lconst));
    for (i = 0; i < d->ntri; i++) {
        const HD5770GpuTri *t = &d->tri[i];
        const float *prm = d->prm + (size_t)i * d->ninputs * 12;

        for (j = 0; j < 3; j++) {
            float *v = vb + ((size_t)i * 3 + j) * vstride;

            v[0] = t->x[j];
            v[1] = t->y[j];
            v[2] = !zs ? 0.0f :
                   zclear ? fminf(fmaxf(d->zclear, d->zmin), d->zmax) :
                   t->z[j] == t->z[j] ? t->z[j] : 0.0f;
            v[3] = t->w[j];
            for (c = 0; c < d->ninputs; c++) {
                memcpy(v + 4 + 4 * c,
                       prm + c * 12 + ((d->flat >> c) & 1 ? 0 : j * 4),
                       4 * sizeof(float));
            }
        }
    }
    if (ir->nconsts) {
        memcpy(kv, d->kv, ir->nconsts * sizeof(uint32_t));
    }
    if (!reuse) {
        MTLRenderPassDescriptor *rp;

        nat_end_pass(m);
        rp = [MTLRenderPassDescriptor renderPassDescriptor];
        for (i = 0; i < nt; i++) {
            rp.colorAttachments[i].texture = tg[i].e->tex;
            rp.colorAttachments[i].loadAction = MTLLoadActionLoad;
            rp.colorAttachments[i].storeAction = MTLStoreActionStore;
        }
        if (zs) {
            rp.depthAttachment.texture = zs->tex;
            rp.depthAttachment.loadAction = MTLLoadActionLoad;
            rp.depthAttachment.storeAction = MTLStoreActionStore;
        }
        if (ss) {
            rp.stencilAttachment.texture = ss->tex;
            rp.stencilAttachment.loadAction = MTLLoadActionLoad;
            rp.stencilAttachment.storeAction = MTLStoreActionStore;
        }
        m->cb = [[m->q commandBuffer] retain];
        m->enc = [[m->cb renderCommandEncoderWithDescriptor:rp] retain];
        [m->enc setCullMode:MTLCullModeNone];
        [m->enc setDepthClipMode:MTLDepthClipModeClamp];
        [m->enc setViewport:(MTLViewport){ 0.0, 0.0, vw, vh, 0.0, 1.0 }];
        m->cur = rt;
        m->ncurm = nt ? nt - 1 : 0;
        for (i = 1; i < nt; i++) {
            m->curm[i - 1] = tg[i].e;
        }
        m->cur_z = zs;
        m->cur_s = ss;
        m->pass_on = true;
        m->vw = vw;
        m->vh = vh;
        m->passes++;
    }
    if (m->cur) {
        m->cur->pass = m->passes;
    }
    for (i = 0; i < m->ncurm; i++) {
        m->curm[i]->pass = m->passes;
    }
    if (m->cur_z) {
        m->cur_z->pass = m->passes;
    }
    if (m->cur_s) {
        m->cur_s->pass = m->passes;
    }
    for (i = 0; i < d->ntex; i++) {
        ts[i]->pass = m->passes;
    }
    for (i = 0; i < nex; i++) {
        ex[i]->pass = m->passes;
    }
    [m->enc setRenderPipelineState:pso];
    if (ds) {
        [m->enc setDepthStencilState:ds];
    }
    if (ss) {
        bool clr = (d->rctl >> 1) & 1;
        bool two = d->dctl & DC_BACKFACE;

        [m->enc setStencilFrontReferenceValue:clr ? d->sclear & 0xff
                                                  : d->srm[0] & 0xff
                           backReferenceValue:clr ? d->sclear & 0xff
                                                  : d->srm[two] & 0xff];
    }
    [m->enc setScissorRect:(MTLScissorRect){ x0, y0, x1 - x0, y1 - y0 }];
    [m->enc setBlendColorRed:d->blend_const[tg[0].src[0] & 3]
                       green:d->blend_const[tg[0].src[1] & 3]
                        blue:d->blend_const[tg[0].src[2] & 3]
                       alpha:d->blend_const[tg[0].src[3] & 3]];
    [m->enc setVertexBuffer:m->ring offset:o_v atIndex:0];
    [m->enc setVertexBuffer:m->ring offset:o_u atIndex:1];
    [m->enc setFragmentBuffer:m->ring offset:o_u atIndex:0];
    [m->enc setFragmentBuffer:m->ring offset:o_k atIndex:1];
    for (i = 0; i < d->nvb; i++) {
        [m->enc setFragmentBuffer:m->ring offset:o_d[i] atIndex:2 + i];
    }
    for (i = 0; i < d->ntex; i++) {
        unsigned s0 = d->tex[i].samp[0];

        [m->enc setFragmentTexture:ts[i]->tex atIndex:i];
        [m->enc setFragmentSamplerState:
                    m->samp[((s0 >> 9) & 1) || ((s0 >> 11) & 1)] atIndex:i];
    }
    for (i = 0; i < nex; i++) {
        [m->enc setFragmentTexture:ex[i]->tex atIndex:16 + i];
    }
    if (!v && ss && (d->dctl & DC_BACKFACE)) {
        /*
         * two-sided stencil: the triangles come clockwise; the ones that
         * face back are drawn with the winding turned
         */
        for (i = 0; i < d->ntri; i = j) {
            for (j = i + 1; j < d->ntri &&
                            !d->tri[j].front == !d->tri[i].front; j++) {
                /* a run */
            }
            [m->enc setFrontFacingWinding:d->tri[i].front
                                          ? MTLWindingClockwise
                                          : MTLWindingCounterClockwise];
            [m->enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:i * 3
                       vertexCount:(j - i) * 3];
        }
        [m->enc setFrontFacingWinding:MTLWindingClockwise];
    } else if (v) {
        [m->enc setVertexBuffer:m->bvram offset:0 atIndex:0];
        [m->enc setVertexBuffer:m->ring offset:o_n atIndex:1];
        [m->enc setVertexBuffer:m->bram ? m->bram : m->bvram offset:0
                        atIndex:2];
        [m->enc setCullMode:v->cull == 1 ? MTLCullModeFront :
                            v->cull == 2 ? MTLCullModeBack : MTLCullModeNone];
        [m->enc setFrontFacingWinding:v->face_cw ? MTLWindingClockwise
                                                 : MTLWindingCounterClockwise];
        [m->enc setDepthBias:0.0f slopeScale:v->zoff_slope clamp:0.0f];
        [m->enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                           indexCount:v->ntidx
                            indexType:MTLIndexTypeUInt32
                          indexBuffer:m->ring
                    indexBufferOffset:o_i
                        instanceCount:v->ninst];
        [m->enc setDepthBias:0.0f slopeScale:0.0f clamp:0.0f];
        [m->enc setCullMode:MTLCullModeNone];
        [m->enc setFrontFacingWinding:MTLWindingClockwise];
        m->vs_draws++;
        m->vs_tris += (uint64_t)v->ntidx / 3 * v->ninst;
    } else {
        [m->enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0
                   vertexCount:d->ntri * 3];
    }
    if (zs && zclear) {
        zs->vsm = 0;
    } else if (zs && (d->dctl & DC_Z)) {
        unsigned bit = v ? 2 : 1;

        m->vs_mixed[v != NULL] += (zs->vsm & ~bit) != 0;
        zs->vsm |= bit;
    }
    for (i = 0; i < nt; i++) {
        nat_drew(m, tg[i].e, x0, y0, x1, y1);
    }
    if (zwrite) {
        nat_drew(m, zs, x0, y0, x1, y1);
    }
    if (ss) {
        nat_drew(m, ss, x0, y0, x1, y1);
    }
    m->nd[ND_TAKEN]++;
    m->tris += d->ntri;
    return true;
}

static bool nat_draw(void *opaque, const HD5770GpuDraw *d)
{
    HD5770Mtlr *m = opaque;
    int64_t t0 = get_clock();
    bool ok;

    m->draw_build_ns = 0;
    nat_hold(m);
    g_mutex_lock(&m->lk);
    m->now = t0;
    @autoreleasepool {
        ok = nat_draw_locked(m, d);
    }
    g_mutex_unlock(&m->lk);
    if (m->draw_build_ns) {
        /* what the draw waited for its program and pipeline */
        float ms = m->draw_build_ns / 1e6f;

        g_array_append_val(m->build_ms, ms);
    }
    m->draw_ns += get_clock() - t0;
    return ok;
}

static void nat_prep(void *opaque, const HD5770GpuDraw *d)
{
    HD5770Mtlr *m = opaque;

    @autoreleasepool {
        if (d) {
            g_mutex_lock(&m->lk);
            m->dry = true;
            nat_draw_locked(m, d);
            m->dry = false;
            g_mutex_unlock(&m->lk);
        } else {
            m->dp_set = false;
            nat_build_ahead(m);
        }
    }
}

/*
 * HD5770_NAT_AUDIT: every band held against memory, in full, after a
 * sync's write-back. A VRAM page that differs from the shadow although
 * nothing named it since the band's last comparison is a missed write:
 * no later comparison would look at it. Memory is compared before the
 * log is read, so a write that lands in between is named.
 */
static void nat_audit(HD5770Mtlr *m)
{
    int64_t t0 = get_clock();
    unsigned i;
    uint32_t b;

    for (i = 0; i < m->surfs->len; i++) {
        NatSurf *e = g_ptr_array_index(m->surfs, i);
        bool logged = false;

        for (b = 0; b < e->nb; b++) {
            uint64_t a, z, o, first = 0;
            uint32_t named = 0, missed = 0;

            if (!e->held[b] || nat_band_dirty(e, b)) {
                continue;
            }
            nat_band(e, b, &a, &z);
            m->au_bands++;
            m->au_db += e->hf >= NF_D16 && e->hf <= NF_S8;
            m->au_bytes += z - a;
            for (o = a; o < z; ) {
                size_t n;
                uint8_t *p8 = nat_at(m, e, o, &n);

                n = MIN(n, z - o);
                if (!e->space) {
                    n = MIN(n, 4096 - ((e->voff + o) & 4095));
                }
                if (memcmp(p8, e->shadow + o, n)) {
                    if (e->space) {
                        named++;
                    } else {
                        if (!logged) {
                            nat_log(m, e);
                            logged = true;
                        }
                        if (m->pgen[(e->voff + o) >> 12] > e->bgen[b]) {
                            named++;
                        } else if (!missed++) {
                            first = o;
                        }
                    }
                }
                o += n;
            }
            if (e->space) {
                m->au_ram += named != 0;
                continue;
            }
            m->au_named += named != 0;
            if (!missed) {
                e->amiss[b] = 0;
                continue;
            }
            m->au_missed++;
            m->au_pages += missed;
            if (e->amiss[b]) {
                continue;
            }
            e->amiss[b] = 1;
            m->au_events++;
            if (m->au_lines++ < 500) {
                const uint8_t *q = m->vram + e->voff + first;
                const uint8_t *h = e->shadow + first;
                size_t k, n = MIN(4096 - ((e->voff + first) & 4095),
                                  z - first);

                for (k = 0; k + 1 < n && q[k] == h[k]; k++) {
                }
                k = MIN(k, n < 4 ? 0 : n - 4);
                fprintf(stderr, "nat-audit: missed write: sync %" PRIu64
                        " draw %" PRIu64 " surface vram 0x%" PRIx64 " %ux%u"
                        " bpe %u mode %u format %u rows %u-%u: %u pages,"
                        " first at 0x%" PRIx64 " memory %02x%02x%02x%02x"
                        " held %02x%02x%02x%02x\n", m->syncs, m->seq,
                        e->voff, e->s.pitch, e->s.height, e->s.bpe,
                        e->s.mode, e->hf, b * e->bh,
                        MIN((b + 1) * e->bh, e->s.height) - 1, missed,
                        e->voff + first + k, q[k & ~3], q[(k & ~3) + 1],
                        q[(k & ~3) + 2], q[(k & ~3) + 3], h[k & ~3],
                        h[(k & ~3) + 1], h[(k & ~3) + 2], h[(k & ~3) + 3]);
            }
        }
    }
    m->au_ns += get_clock() - t0;
}

/*
 * memory may change from here on: bands are compared again; with idle,
 * the entries not used for long go
 */
static void nat_epoch(HD5770Mtlr *m, bool idle)
{
    m->stamp[0]++;
    m->stamp[1]++;
    m->seq++;
    if (!idle) {
        return;
    }
    if (m->bytes > m->cache_max || !(m->epoch & 63)) {
        /* over the limit: all but this sync's; else the long unused */
        @autoreleasepool {
            nat_evict(m, m->bytes > m->cache_max ? m->epoch :
                         m->epoch > NAT_IDLE ? m->epoch - NAT_IDLE : 0);
        }
    }
    m->epoch++;
}

static void nat_sync_locked(HD5770Mtlr *m)
{
    int64_t t0 = get_clock();

    @autoreleasepool {
        nat_writeback(m, NULL, NW_SYNC);
    }
    nat_report(m);
    if (m->audit) {
        nat_audit(m);
    }
    m->pub.kept = false;
    nat_epoch(m, true);
    m->syncs++;
    m->sync_ns += get_clock() - t0;
}

static void nat_sync(void *opaque)
{
    HD5770Mtlr *m = opaque;

    nat_hold(m);
    g_mutex_lock(&m->lk);
    nat_sync_locked(m);
    g_mutex_unlock(&m->lk);
}

static void nat_sync_range(void *opaque, uint64_t off, uint64_t len)
{
    HD5770Mtlr *m = opaque;
    int64_t t0 = get_clock();
    unsigned i;

    nat_hold(m);
    g_mutex_lock(&m->lk);
    @autoreleasepool {
        nat_drain(m);
        for (i = 0; i < m->act->len; i++) {
            NatSurf *e = g_ptr_array_index(m->act, i);

            if (e->dirty && !e->space && e->lo < off + len && off < e->hi) {
                nat_writeback(m, e, NW_RANGE);
            }
        }
        /* draws that read memory in place */
        nat_wait(m, NW_RANGE);
    }
    nat_report(m);
    m->pub.kept = false;
    for (i = 0; i < m->act->len; i++) {
        NatSurf *e = g_ptr_array_index(m->act, i);

        m->pub.kept |= e->dirty;
    }
    nat_epoch(m, false);
    m->rsyncs++;
    m->sync_ns += get_clock() - t0;
    g_mutex_unlock(&m->lk);
}

static uint64_t nat_fence(void *opaque)
{
    HD5770Mtlr *m = opaque;
    int64_t t0 = get_clock();
    uint64_t t;
    bool drew;

    nat_hold(m);
    g_mutex_lock(&m->lk);
    m->now = t0;
    @autoreleasepool {
        t = nat_fence_locked(m, &drew);
    }
    nat_fenced(m);
    nat_epoch(m, drew);
    m->fence_ns += get_clock() - t0;
    g_mutex_unlock(&m->lk);
    return t;
}

/* the CPU wrote memory: after a sync, so no entry is newer than it */
static void nat_wrote(void *opaque, uint64_t off, uint64_t len)
{
    HD5770Mtlr *m = opaque;

    g_mutex_lock(&m->lk);
    nat_named(m, off, off + len);
    m->stamp[!len]++;
    g_mutex_unlock(&m->lk);
}

static bool nat_ram(void *opaque, uint8_t *host, uint64_t size)
{
    HD5770Mtlr *m = opaque;

    m->ram = host;
    m->ram_size = size;
    if (m->bvram && !m->bram &&
        !((uintptr_t)host % qemu_real_host_page_size()) &&
        !(size % qemu_real_host_page_size()) &&
        size <= m->dev.maxBufferLength) {
        @autoreleasepool {
            m->bram = [m->dev newBufferWithBytesNoCopy:host length:size
                                  options:MTLResourceStorageModeShared
                                  deallocator:nil];
        }
    }
    return true;
}

static int nat_cmp_float(gconstpointer a, gconstpointer b)
{
    float x = *(const float *)a, y = *(const float *)b;

    return x < y ? -1 : x > y;
}

static void nat_stats(void *opaque, GString *out)
{
    HD5770Mtlr *m = opaque;
    GArray *s;
    unsigned i;

    g_mutex_lock(&m->lk);
    s = g_array_copy(m->build_ms);

    g_array_sort(s, nat_cmp_float);
    g_string_append_printf(out, "native: %s draws %" PRIu64 " triangles %"
                           PRIu64 " declined:", m->dev.name.UTF8String,
                           m->nd[ND_TAKEN], m->tris);
    for (i = ND_EMPTY; i < ND_N; i++) {
        if (m->nd[i]) {
            g_string_append_printf(out, " %s %" PRIu64 ",", nd_names[i],
                                   m->nd[i]);
        }
    }
    g_string_append_printf(out, "\nnative: syncs %" PRIu64 " passes %" PRIu64
                           " waits %" PRIu64 " uploads %" PRIu64 " (%" PRIu64
                           " KB, %" PRIu64 " KB of bands not held before)"
                           " write-backs %" PRIu64 " (%" PRIu64
                           " KB) checks %" PRIu64 " (compared %" PRIu64
                           " KB) surfaces %u\n",
                           m->syncs, m->passes, m->waits, m->uploads,
                           m->upload_bytes >> 10, m->first_bytes >> 10,
                           m->wbacks,
                           m->wback_bytes >> 10, m->checks,
                           m->cmp_bytes >> 10, m->surfs->len);
    g_string_append_printf(out, "native: entries hold %" PRIu64 " MB, at most %"
                           PRIu64 " of %" PRIu64 " MB, dropped for room %"
                           PRIu64 ", texture levels left out %" PRIu64 "\n",
                           m->bytes >> 20, m->bytes_most >> 20,
                           m->cache_max >> 20, m->roomed, m->lv_cut);
    g_string_append_printf(out, "native: ms: draw %" PRId64 " sync %" PRId64
                           ", of them wait %" PRId64 " check %" PRId64
                           " upload %" PRId64 " write-back %" PRId64 "\n",
                           m->draw_ns / 1000000, m->sync_ns / 1000000,
                           m->wait_ns / 1000000, m->check_ns / 1000000,
                           m->upload_ns / 1000000, m->wback_ns / 1000000);
    g_string_append_printf(out, "native: fences %" PRIu64 " (nothing"
                           " outstanding %" PRIu64 ", nothing drawn %" PRIu64
                           ") completed early %" PRIu64 " bands compared in"
                           " flight %" PRIu64 " holds %" PRIu64 ", ms: fence %"
                           PRId64 " early wait %" PRId64 " copy %" PRId64
                           ", plane rectangles kept %" PRIu64
                           ", syncs for a range %" PRIu64
                           ", fences for passes in flight %" PRIu64
                           " (own command buffer %" PRIu64 ")"
                           "\n", m->nfences + m->fence_now, m->fence_now,
                           m->fence_empty, m->drains, m->fly_cmp, m->holds,
                           m->fence_ns / 1000000, m->drain_ns / 1000000,
                           m->fcopy_ns / 1000000, m->kept_rects, m->rsyncs,
                           m->fence_wait, m->fence_own);
    g_string_append_printf(out, "native: waits for:");
    for (i = 0; i < NW_N; i++) {
        g_string_append_printf(out, " %s %" PRIu64 " (%" PRId64 " ms)",
                               nw_names[i], m->nwait[i],
                               m->wait_by_ns[i] / 1000000);
    }
    g_string_append_printf(out, ", ring changes %" PRIu64
                           ", uploads after a pass in flight %" PRIu64 "\n",
                           m->ring_swaps, m->up_after);
    g_string_append_printf(out, "native: programs %" PRIu64 " pipelines %"
                           PRIu64 " draws that built %u, ms: total %" PRIu64
                           " median %.1f worst %.1f\n", m->builds,
                           m->pso_builds, s->len,
                           m->build_ns / 1000000,
                           s->len ? g_array_index(s, float, s->len / 2) : 0.0f,
                           s->len ? g_array_index(s, float, s->len - 1) : 0.0f);
    g_string_append_printf(out, "native: built ahead: %" PRIu64 " times, %"
                           PRIu64 " draws looked at, %" PRIu64
                           " libraries %" PRIu64 " pipelines in %" PRId64
                           " ms (libraries %" PRId64 ")\n", m->ahead,
                           m->ahead_draws, m->ahead_libs, m->ahead_psos,
                           m->ahead_ns / 1000000,
                           m->ahead_lib_ns / 1000000);
    g_string_append_printf(out, "native: vertex stage on the host: draws %"
                           PRIu64 " triangles %" PRIu64 " programs %" PRIu64
                           " (ms %" PRIu64 "); mixed vertex stage on a depth"
                           " plane since its clear: CPU stage after host's %"
                           PRIu64 ", host's after CPU %" PRIu64 "\n",
                           m->vs_draws, m->vs_tris, m->vs_builds,
                           m->vs_build_ns / 1000000, m->vs_mixed[0],
                           m->vs_mixed[1]);
    if (m->audit) {
        g_string_append_printf(out, "native: audit: bands compared %" PRIu64
                               " (%" PRIu64 " MB; depth and stencil %" PRIu64
                               "), differing and named %"
                               PRIu64 ", differing in guest RAM %" PRIu64
                               ", missed writes: %" PRIu64 " bands (%" PRIu64
                               " band-syncs, %" PRIu64 " pages), ms %" PRId64
                               "\n", m->au_bands, m->au_bytes >> 20,
                               m->au_db, m->au_named, m->au_ram, m->au_events,
                               m->au_missed, m->au_pages,
                               m->au_ns / 1000000);
    }
    g_array_free(s, true);
    g_mutex_unlock(&m->lk);
}

static void nat_close(void *opaque)
{
    HD5770Mtlr *m = opaque;
    unsigned i;

    g_mutex_lock(&m->lk);
    @autoreleasepool {
        nat_sync_locked(m);
    }
    g_mutex_unlock(&m->lk);
    while (qatomic_read(&m->cbs_out)) {
        g_usleep(100);
    }
    for (i = 0; i < m->bufs->len; i++) {
        [(id<MTLBuffer>)g_ptr_array_index(m->bufs, i) release];
    }
    g_ptr_array_free(m->bufs, true);
    g_mutex_clear(&m->lk);
    g_mutex_clear(&m->rlock);
    g_ptr_array_free(m->surfs, true);
    g_ptr_array_free(m->act, true);
    g_hash_table_destroy(m->dss);
    [m->ds_off release];
    g_hash_table_destroy(m->psos);
    g_hash_table_destroy(m->progs);
    g_hash_table_destroy(m->vprogs);
    [m->bvram release];
    [m->bram release];
    g_array_free(m->ranges, true);
    g_array_free(m->build_ms, true);
    g_free(m->tmp);
    g_free(m->flags);
    g_free(m->pgen);
    [m->samp[0] release];
    [m->samp[1] release];
    for (i = 0; i < NAT_RINGS; i++) {
        [m->rings[i] release];
    }
    [m->rbuf release];
    [m->q release];
    [m->dev release];
    g_free(m);
}

static const HD5770GpuOps nat_ops = {
    .name = "metal-native",
    .min_pixels = 1024,
    .draw = nat_draw,
    .sync = nat_sync,
    .stats = nat_stats,
    .close = nat_close,
    .ram = nat_ram,
    .wrote = nat_wrote,
    .fence = nat_fence,
    .sync_range = nat_sync_range,
    .prep = nat_prep,
    .depth = true,
    .mipmaps = true,
    .resinfo = true,
    .full = true,
    .native = true,
};

HD5770Gpu *hd5770_mtlr_open(uint8_t *vram, uint64_t vram_size,
                            const char **err)
{
    HD5770Mtlr *m;
    unsigned i;

    *err = NULL;
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        MTLCompileOptions *o;
        id<MTLLibrary> lib;

        if (!dev) {
            *err = "no Metal device";
            return NULL;
        }
        m = g_new0(HD5770Mtlr, 1);
        m->dev = dev;
        m->vram = vram;
        m->vram_size = vram_size;
        m->cache_max = NAT_CACHE_BASE + 2 * vram_size;
        m->q = [dev newCommandQueueWithMaxCommandBufferCount:NAT_CBS_MAX];
        m->ring = [dev newBufferWithLength:NAT_RING
                                   options:MTLResourceStorageModeShared];
        m->rings[0] = m->ring;
        m->rbuf = [dev newBufferWithLength:NAT_READ
                                   options:MTLResourceStorageModeShared];
        for (i = 0; i < 2; i++) {
            MTLSamplerDescriptor *sd = [[MTLSamplerDescriptor alloc] init];

            sd.minFilter = i ? MTLSamplerMinMagFilterLinear
                             : MTLSamplerMinMagFilterNearest;
            sd.magFilter = sd.minFilter;
            sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
            sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
            m->samp[i] = [dev newSamplerStateWithDescriptor:sd];
            [sd release];
        }
        if (!m->q || !m->ring || !m->rbuf || !m->samp[0] || !m->samp[1]) {
            [m->samp[0] release];
            [m->samp[1] release];
            [m->ring release];
            [m->rbuf release];
            [m->q release];
            [dev release];
            g_free(m);
            *err = "cannot set up the Metal pipeline";
            return NULL;
        }
        /* the compiler's first library takes 100 ms and more */
        o = [[MTLCompileOptions alloc] init];
        nat_exact_math(o);
        lib = [dev newLibraryWithSource:@"#include <metal_stdlib>\n"
               "using namespace metal;\n"
               "fragment float4 f(float4 p [[position]], texture2d<float> t,\n"
               "                  sampler s) { return t.sample(s, p.xy); }\n"
                                options:o error:NULL];
        [lib release];
        [o release];
    }
    m->progs = g_hash_table_new_full(g_str_hash, g_str_equal, g_free,
                                     nat_prog_free);
    m->vprogs = g_hash_table_new_full(g_str_hash, g_str_equal, g_free,
                                      nat_pso_free);
    m->psos = g_hash_table_new_full(g_str_hash, g_str_equal, g_free,
                                    nat_pso_free);
    m->dss = g_hash_table_new_full(g_str_hash, g_str_equal, g_free,
                                   nat_pso_free);
    @autoreleasepool {
        MTLDepthStencilDescriptor *dd = [[MTLDepthStencilDescriptor alloc]
                                         init];

        m->ds_off = [m->dev newDepthStencilStateWithDescriptor:dd];
        [dd release];
        /* Apple GPUs; others want one packed texture for both */
        m->sep_ds = [m->dev supportsFamily:MTLGPUFamilyApple1];
        m->bc_ok = m->dev.supportsBCTextureCompression;
        m->pk_ok = m->sep_ds;
        m->zlazy = !g_strcmp0(g_getenv("HD5770_NAT_ZLAZY") ?: "1", "1");
        m->clazy = !g_strcmp0(g_getenv("HD5770_NAT_CLAZY") ?: "1", "1");
        m->ztex = !g_strcmp0(g_getenv("HD5770_NAT_ZTEX") ?: "1", "1");
        m->lazy_ns = 1000000ll * atoi(g_getenv("HD5770_NAT_LAZY_MS") ?: "300");
    }
    m->surfs = g_ptr_array_new_with_free_func(nat_surf_free);
    m->act = g_ptr_array_new();
    m->ranges = g_array_new(false, false, sizeof(uint64_t));
    m->bufs = g_ptr_array_new();
    g_queue_init(&m->fences);
    g_mutex_init(&m->lk);
    g_mutex_init(&m->rlock);
    m->pgen = g_new0(uint64_t, (vram_size >> 12) + 1);
    m->stamp[0] = m->stamp[1] = 1;
    m->build_ms = g_array_new(false, false, sizeof(float));
    m->jl = g_ptr_array_new_with_free_func(nat_job_free);
    m->jp = g_ptr_array_new_with_free_func(nat_job_free);
    m->jk = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);
    m->audit = getenv("HD5770_NAT_AUDIT") != NULL;
    if (!getenv("HD5770_NAT_VSA") &&
        !((uintptr_t)vram % qemu_real_host_page_size()) &&
        !(vram_size % qemu_real_host_page_size()) &&
        vram_size <= m->dev.maxBufferLength) {
        /* HD5770_NAT_VSA: every vertex stage on the CPU */
        @autoreleasepool {
            m->bvram = [m->dev newBufferWithBytesNoCopy:vram length:vram_size
                                   options:MTLResourceStorageModeShared
                                   deallocator:nil];
        }
        m->pub.vs_draw = m->bvram != nil;
    }
    m->pub.ops = &nat_ops;
    m->pub.b = m;
    return &m->pub;
}
