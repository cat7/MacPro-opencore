/*
 * QEMU ATI Radeon X1900 XT (R580) emulation
 *
 * A PCI Express R5xx card: AVIVO display controller (ati_r580_avivo.c),
 * synthetic AtomBIOS with a register mailbox (ati_r580_atom.c), and the
 * command processor, 2D and 3D engines carried over from the R300 model
 * (ati_r580_cp.c, ati_r580_2d.c, ati_r580_3d.c).
 *
 * This work is licensed under the GNU GPL license version 2 or later.
 */

#ifndef ATI_R580_INT_H
#define ATI_R580_INT_H

#include "hw/pci/pci_device.h"
#include "hw/display/edid.h"
#include "qemu/bitmap.h"
#include "hw/i2c/bitbang_i2c.h"
#include "qemu/timer.h"
#include "qemu/thread.h"
#include "qapi/qapi-types-common.h"
#include "qom/object.h"
#include "ati_r580_pvs.h"
#include "ati_r580_us.h"
#include "ati_r580_cap.h"
#include "ati_r580_gl.h"
#include "ati_r580_avivo.h"

#define PCI_VENDOR_ID_ATI              0x1002

#define TYPE_ATI_R580 "ati-radeon-x1900"
OBJECT_DECLARE_SIMPLE_TYPE(ATIR580State, ATI_R580)

/*
 * BAR0: 256MB prefetchable frame buffer aperture, all of VRAM
 * BAR1: 256 bytes I/O, the first registers (MM_INDEX/MM_DATA)
 * BAR2: 64KB registers
 */
#define ATI_R580_APER_SIZE   (256 * 1024 * 1024)
#define ATI_R580_VRAM_SIZE   (256 * 1024 * 1024)
#define ATI_R580_CGUARDS     32
#define ATI_R580_MMIO_SIZE   (64 * 1024)
#define ATI_R580_IO_SIZE     256
#define ATI_R580_NUM_REGS    (ATI_R580_MMIO_SIZE / 4)
#define ATI_R580_NUM_PLLS    64
/* granule of the texture cache's scan of the VRAM dirty bitmap */
#define ATI_R580_FB_SCAN_BLOCK (64 * 1024)

/*
 * Kinds of "the hardware does this, we do not" gap the engines report
 * through ati_r580_note_gap(). Each kind indexes its counters by the
 * field value itself (an opcode, a format code), so the tally names
 * exactly what a guest asked for and never got.
 */
typedef enum ATIR580GapKind {
    R580_GAP_P3_OPCODE,      /* packet3 opcode the parser discards */
    R580_GAP_PRIM,           /* VAP_VF_CNTL primitive type */
    R580_GAP_VTX_WALK,       /* VAP_VF_CNTL vertex walk mode */
    R580_GAP_TEX_FORMAT,     /* TX_FORMAT1 texel format code */
    R580_GAP_BLEND_FACTOR,   /* RB3D_BLENDCNTL src/dst factor code */
    R580_GAP_VTX_PROGRAM,    /* draw needs a vertex program we don't run */
    R580_GAP_DEST_OFF_VRAM,  /* work discarded: destination outside VRAM */
    R580_GAP_VS_VECTOR_OP,   /* PVS vector-engine opcode not implemented */
    R580_GAP_VS_MATH_OP,     /* PVS math-engine opcode not implemented */
    R580_GAP_VS_DST_FILE,    /* PVS destination register file not modelled */
    R580_GAP_TEX_SWIZZLE,    /* TX_FORMAT1 component select not modelled */
    R580_GAP_AOS_ARRAYS,     /* more vertex arrays bound than we fetch */
    /*
     * VAP_PROG_STREAM_CNTL element format not unpacked. Indexed by the
     * DATA_TYPE code itself for a type this model cannot fetch, and by
     * 0x10 | code for a VAP_PROG_STREAM_CNTL_EXT swizzle select it does
     * not know. Either way the element falls back to the raw-float read
     * that predates the format decoder, so the tally is the only place
     * the guest's request is visible.
     */
    R580_GAP_VTX_DATA_TYPE,
    R580_GAP_FS_RGB_OP,      /* US colour-side opcode not implemented */
    R580_GAP_FS_ALPHA_OP,    /* US alpha-side opcode not implemented */
    R580_GAP_FS_TEX_OP,      /* US texture instruction not implemented */
    R580_GAP_FS_INDIRECT,    /* US_CONFIG names an indirection level */
    R580_GAP_FS_RS_ROUTE,    /* rasterizer routes something we do not emit */
    R580_GAP_FS_OUT_FMT,     /* US_OUT_FMT_0 pixel format not modelled */
    R580_GAP_ZB_FORMAT,      /* ZB_FORMAT depth format not modelled */
    R580_GAP_CB_FORMAT,      /* COLORFORMAT the rasteriser cannot store */
    R580_GAP_VS_ADDR_MODE,   /* PVS operand addressing mode not modelled */
    R580_GAP_UCP,            /* VAP_CLIP_CNTL user clip planes enabled */
    R580_GAP_ZPASS,          /* occlusion counter use not modelled */
    R580_GAP_ZB_LAYOUT,      /* depth layout the aperture does not map */
    R580_GAP_MAX
} ATIR580GapKind;

#define R580_GAP_SLOTS 256

/* dumps of the occlusion counter waiting for the GPU's count */
#define R580_ZQ_PEND 16

/*
 * How the "gl" property is set: whether the host-GPU backend renders a
 * draw, and whether the software rasterizer runs alongside it as a
 * check. `off` is the default and is a provable no-op -- the draw path
 * tests one NULL pointer and calls exactly what it called before.
 */
typedef enum ATIR580GlMode {
    R580_GL_OFF = 0,
    R580_GL_ON,             /* GL renders what it can; the rest falls back */
    R580_GL_VERIFY          /* both run, they are diffed, software wins */
} ATIR580GlMode;

/*
 * Why a draw the GL backend was offered went to the software rasterizer
 * instead. Counted per primitive type, the same shape as the gap
 * tracker, so `qom-get <device> gl` says how much of a session the
 * offload actually covered rather than leaving it to be inferred.
 */
typedef enum ATIR580GlFallback {
    R580_GLF_PRIM,          /* point sprites, lines: not assembled here */
    R580_GLF_RESOLVE,       /* AA resolve: the colour buffer is the source */
    R580_GLF_RECT,          /* no drawable destination rectangle */
    R580_GLF_ALIGN,         /* destination offset or pitch not dword-aligned */
    R580_GLF_VRAMEND,       /* rectangle runs off the end of VRAM */
    R580_GLF_XOR,           /* aperture swapper not uniform over the rect */
    R580_GLF_CLIPRULE,      /* a genuine 4-rect cliprect truth table */
    R580_GLF_TEXTURE,       /* texture too large to decode per draw */
    R580_GLF_SELFBLEND,     /* self-overlapping blend needing too many passes */
    R580_GLF_SURFACE,       /* target too large, or it would not resize */
    R580_GLF_BACKEND,       /* the backend itself declined the request */
    R580_GLF_FSPROG,        /* fragment program the translator refused */
    R580_GLF_CBFMT,         /* 16bpp or GART colour buffer */
    R580_GLF_ZTEST,         /* depth or stencil test */
    R580_GLF_ZPASS,         /* counted by an occlusion query, gl-zpass=off */
    R580_GLF_PROGWAIT,      /* the program worker still has the program */
    R580_GLF_MAX
} ATIR580GlFallback;

/*
 * Which coherency hook gave the resident render target back. The rules
 * are stated at "GL-OWNED RENDER TARGET" in ati_r580_3d.c; this counts
 * how often each of them actually fires and how many pixels it moved,
 * because "the target is not staying resident" is a symptom whose cure
 * depends entirely on WHICH rule is ending it. Read the tally from
 * `gl-stats`.
 */
typedef enum ATIR580GlRel {
    R580_GLR_SCANOUT,       /* the display refresh, and the cursor's timer */
    R580_GLR_RING,          /* the end of a command-processor ring run */
    R580_GLR_IB,            /* the end of an indirect buffer */
    R580_GLR_FIFO,          /* a CP FIFO push through the register aperture */
    R580_GLR_READ,          /* a named-range reader, ati_r580_gl_sync() */
    R580_GLR_2D,            /* the 2D scaler, and host-data pushes */
    R580_GLR_TARGET,        /* a draw bound a different colour buffer */
    R580_GLR_FALLBACK,      /* a draw the offload handed back */
    R580_GLR_BACKEND,       /* the backend declined, mid-draw */
    R580_GLR_RESET,         /* reset, unrealize */
    R580_GLR_FENCE,         /* a scratch or read pointer write, threaded CP */
    R580_GLR_YIELD,         /* the CP let a claimant in mid-buffer */
    R580_GLR_MAX
} ATIR580GlRel;

/* why the GPU's copy of the depth buffer was given up */
typedef enum ATIR580GlZDrop {
    R580_GLZD_PARK,         /* could not be kept across a release */
    R580_GLZD_STALE,        /* kept, then VRAM under it changed */
    R580_GLZD_OTHER,        /* another depth buffer, or drawn into */
    R580_GLZD_SOFT,         /* the software path wrote Z */
    R580_GLZD_WRITER,       /* 2D engine, CP or MM_DATA wrote its range */
    R580_GLZD_GUARD,        /* the guest CPU touched it */
    R580_GLZD_RESET,        /* reset, backend failure, target grown */
    R580_GLZD_VERIFY,       /* gl=verify */
    R580_GLZD_MAX
} ATIR580GlZDrop;

/*
 * WHICH 2D path ended the residency. R580_GLR_2D above lumps three
 * unrelated operations together, and they call for opposite fixes: a
 * blit's ranges usually miss the render target entirely, a host-data
 * push never can (its destination is an 8bpp glyph atlas), and the
 * scaler is a video path 10.5 does not use at all. The cause enum
 * itself cannot carry the split -- the offline gl-replay harness in
 * doc/radeon9800 compiles ati_r580_3d.c against its own copy of this
 * header -- so the three paths keep their own tally beside it, filled
 * by ati_r580_2d_gl_note() and reported by `gl-stats`.
 *
 * A path is credited whatever cause the release was booked under, so
 * the tally stays honest when a range-gated path releases through
 * ati_r580_gl_sync() as R580_GLR_READ rather than as R580_GLR_2D.
 */
typedef enum ATIR580Gl2dPath {
    R580_GL2D_BLIT,         /* ati_r580_2d_blt(), via its named ranges */
    R580_GL2D_HOST,         /* ati_r580_host_data_flush(), CPU-pushed */
    R580_GL2D_SCALE,        /* ati_r580_2d_scale_run() */
    R580_GL2D_MAX
} ATIR580Gl2dPath;

/*
 * How long a DECODED TEXTURE may live. The default states the rule the
 * decode actually depends on -- see r300_gl_tex_current() -- and the
 * other two exist to measure it:
 *
 *   burst   the M3 lifetime: every release drops the whole cache. Safe,
 *           and on Flurry it left a 0.6% hit rate. Kept as the A/B arm.
 *   dirty   the default: an entry lives while the VRAM it was decoded
 *           from is unwritten, which is what correctness requires.
 *   never   deliberately WRONG -- entries are never invalidated at all.
 *           It exists so the guard can be shown to bite: a gl=verify
 *           run in this mode must FAIL. Never a shipping configuration.
 */
typedef enum ATIR580GlTexLife {
    R580_TEXLIFE_DIRTY,
    R580_TEXLIFE_BURST,
    R580_TEXLIFE_NEVER,
} ATIR580GlTexLife;

/*
 * A blended draw whose own primitives overlap is rendered in several
 * ordered passes rather than handed back to the software rasterizer
 * (milestone M3; see r300_gl_passes()). These bound the partition: a
 * draw needing more triangles or more passes than this falls back and
 * is counted, so the limits are visible in `gl-stats` rather than
 * silently shaping what the offload covers.
 */
#define R300_GL_TRI_MAX     512
#define R300_GL_PASS_MAX    128

/*
 * The decoded-texture cache: at most R300_GL_TEXCACHE entries (the
 * gl-texcache-slots property picks how many are used), each at most
 * R300_GL_TEXCACHE_MAX texels per face. With more than the classic 32
 * slots, lookups go through a hash on the decode key and the decoded
 * bytes are bounded by R300_GL_TEXCACHE_BYTES, least recently used
 * first, so a guest cannot make the device allocate without limit.
 */
#define R300_GL_TEXCACHE       R580_GL_TEXSLOTS
#define R300_GL_TEXCACHE_OLD   32
#define R300_GL_TEXCACHE_MAX   (256 * 1024)
#define R300_GL_TEXCACHE_BYTES (128u << 20)
#define R300_GL_TEXHASH        512
#define R300_GL_TEXNIL         0xffff

/*
 * The guard on a cached entry (see r300_gl_tex_current()) walks the
 * host pages its VRAM range covers. A texture spanning more pages than
 * this is decoded into the scratch buffer and not cached, so that the
 * walk stays a bounded cost per draw -- 4 MB at a 4 KB page, where the
 * cache's own texel limit is a quarter of that.
 */
#define R300_GL_DIRTY_PAGES  1024

typedef struct ATIR580PM4Parser {
    uint32_t remaining;      /* data dwords still expected */
    uint32_t type;           /* packet type of the in-flight packet */
    uint32_t reg;            /* running register offset, packet0 */
    bool one_reg;
    uint32_t p1_reg1;        /* packet1's two register offsets */
    uint32_t p1_reg2;
    uint32_t p3_opcode;      /* packet3 2D-draw sub-state */
    uint32_t p3_params[8];
    uint32_t p3_scale[16];      /* R580_SCALE_PKT_DWORDS */
    uint32_t p3_param_idx;
    uint32_t p3_total;       /* payload dwords the packet3 declared */
} ATIR580PM4Parser;

typedef struct ATIR580Mode {
    uint32_t width;
    uint32_t height;
    uint32_t pitch;      /* bytes per scanline */
    uint32_t bpp;        /* bytes per pixel */
    uint32_t fb_offset;  /* byte offset into VRAM */
    uint32_t pix_width;  /* raw CRTC_PIX_WIDTH field, for draw dispatch */
} ATIR580Mode;

/*
 * One frame for the display worker to convert, filled on the main loop
 * while the worker is idle; see "ASYNC DISPLAY" in ati_r580.c.
 */
typedef struct ATIR580DispJob {
    ATIR580Mode mode;
    struct DisplaySurface *ds;
    const uint8_t *vram;
    const uint8_t (*pal)[3];    /* palette, or the device's own */
    uint8_t palette[256][3];
} ATIR580DispJob;

enum {
    ATI_R580_DISP_IDLE,         /* job is the main loop's */
    ATI_R580_DISP_QUEUED,       /* job is the worker's from here ... */
    ATI_R580_DISP_RUNNING,
    ATI_R580_DISP_DONE,         /* ... until the hand-over takes it back */
};

typedef struct ATIR580R5 ATIR580R5;

struct ATIR580State {
    PCIDevice parent_obj;

    MemoryRegion aper;        /* BAR0: all of VRAM, linear */
    MemoryRegion vram;
    MemoryRegion mmio;        /* BAR2: 64KB register file */
    MemoryRegion io;          /* BAR1: 256-byte I/O register window */
    AddressSpace agp_as;      /* AGP transactions, via the host's GART */
    bool agp_as_valid;
    QemuConsole *con;

    uint32_t regs[ATI_R580_NUM_REGS];
    uint32_t plls[ATI_R580_NUM_PLLS];
    /*
     * Command processor thread. A CPU write of CP_RB_WPTR records the
     * pointer and wakes it; see "COMMAND PROCESSOR THREAD" in ati_r580.c.
     */
    OnOffAuto engine_async;
    bool engine_on;
    QemuThread engine_thread;
    QemuMutex engine_lock;
    QemuCond engine_cond;
    QemuEvent engine_idle;
    VMChangeStateEntry *engine_vmse;
    bool engine_busy;           /* atomic */
    bool engine_kick, engine_quit;
    uint64_t engine_kicks, engine_waits, engine_wait_us, engine_bql_writes;
    uint64_t engine_ibs, engine_scratch;
    unsigned engine_rptr_wb;
    /*
     * Register accesses without the BQL; see "LOCKLESS REGISTER ACCESS"
     * in ati_r580.c. engine_claimed and the claim counters are under
     * engine_lock.
     */
    bool lockless_mmio;         /* property */
    bool lockless;              /* in effect: needs the engine thread */
    bool engine_claimed;
    QemuCond engine_claim_cond;
    uint64_t engine_claims, engine_claim_waits, engine_claim_wait_us;
    uint64_t engine_claim_wait_max_us;
    uint32_t engine_claim_wanted;
    uint64_t engine_yields, engine_ib_yields;
    uint64_t engine_runs, engine_run_ns;
    bool ib_yield;              /* property: yield inside indirect buffers */
    uint64_t engine_claim_waits_50ms;
    uint32_t engine_claim_wait_max_reg, engine_claim_reg;
    uint64_t engine_claim_busy;
    /* the decoded-texture cache, held by the draw path */
    QemuRecMutex gl_tex_lock;
    /*
     * What the display's refresh hands the draw path: the pages its
     * dirty-bitmap clear found written, not yet applied to the texture
     * cache, and the framebuffer range it shows (offset << 32 | length).
     * scan_lock covers only the hand-over, so the refresh never waits
     * for a draw; scan_seq is odd while a clear is under way.
     */
    QemuMutex scan_lock;
    unsigned long *scan_dirty;
    unsigned long *scan_work;
    bool scan_pending;          /* atomic */
    unsigned scan_seq;          /* atomic */
    uint64_t scan_fb;           /* atomic */
    /*
     * Display worker; see "ASYNC DISPLAY" in ati_r580.c. disp_state and
     * disp_quit are under disp_lock; disp_job belongs to whoever
     * disp_state says; disp_tick_missed and the counters to the BQL.
     */
    OnOffAuto async_display;    /* property */
    bool disp_on;
    QemuThread disp_thread;
    QemuMutex disp_lock;
    QemuCond disp_cond;
    QEMUBH *disp_bh;
    unsigned disp_state;
    bool disp_quit;
    bool disp_tick_missed;
    ATIR580DispJob disp_job;
    uint64_t disp_frames, disp_deferred;
    /* PCI identity (properties) */
    uint32_t device_id;
    uint32_t subsys_vendor_id;
    uint32_t subsys_id;
    uint32_t revision;
    /* MC and PCIE indirect register files (MC_IND_*, PCIE_INDEX/DATA) */
    uint32_t mc_ind[256];
    uint32_t pcie_ind[256];

    /* scanout palette: the D1 LUT the last refresh drew with */
    uint8_t palette[256][3];
    ATIR580Mode mode;      /* what the last refresh actually drew */
    /* redraw the whole surface next pass regardless of dirty state */
    bool force_redraw;

    /* AVIVO: frame timeline, vblank and hot plug interrupt state */
    QEMUTimer *vblank_timer;
    int64_t frame_start_ns;
    int64_t frame_period_ns;
    uint32_t frame_count;
    bool vblank_occurred[ATI_R580_NUM_CRTC];
    bool hpd1_pending;
    /* DC_LUTA/DC_LUTB, picked per head by DxGRPH_LUT_SEL */
    uint8_t lut[2][256][3];
    unsigned lut_seq;        /* next component of a DC_LUT_SEQ_COLOR write */
    unsigned lut_gen;        /* bumped on every LUT change */
    unsigned lut_drawn;      /* lut_gen of the last refresh */
    /* DDC1-3 on the DC GPIO pads; the monitor's EDID on DDC1 */
    bitbang_i2c_interface ddc_i2c[ATI_R580_NUM_DDC];
    struct ATIR580EdidState *ddc_slave[ATI_R580_NUM_DDC];
    uint32_t ddc_y[ATI_R580_NUM_DDC];
    qemu_edid_info edid_info;
    uint32_t xres, yres;     /* the mode the card resets to */
    /* coalesces a burst of cursor-register writes into one host update */
    QEMUTimer *cursor_timer;

    /*
     * PM4/CCE GUI command-FIFO ring buffer -- the mechanism the real
     * Mac driver actually uses for register/2D submission on this
     * card (distinct from, and used after, the older one-shot
     * BM_GUI_TABLE descriptor engine). Register offsets and the ring
     * living in VRAM (this PCI, non-AGP variant has no GART/system-
     * memory command access) are cross-verified between a live trace
     * of the real driver and independent reference source; see the
     * comment on R580_PM4_BUFFER_OFFSET in ati_r580_regs.h.
     * Consumed synchronously on each PM4_BUFFER_DL_WPTR write, same
     * style as the BM engine.
     */
    bool bm_busy;              /* a BM_GUI_TABLE walk is in progress */
    uint32_t pm4_rptr;         /* ring read pointer, in dwords */
    uint32_t pm4_wptr;         /* ring write pointer, in dwords */
    /* ring base: VRAM offset, or GART offset when AGP_OFFSET_FLAG set */
    uint32_t pm4_buffer_addr;
    uint32_t pm4_buffer_cntl;
    uint32_t pm4_ring_dwords;  /* ring size decoded from CNTL, 0 = no ring */
    bool pm4_in_ib;            /* fetching from the indirect buffer */
    bool pm4_in_ring;          /* fetching from the ring */

    /*
     * CCE microcode store (PM4_MICROCODE_ADDR/RADDR/DATAH/DATAL):
     * 256 x 64-bit words, streamed as high/low pairs with the address
     * auto-incrementing after each DATAL access -- kept so a driver
     * that reads its upload back to verify sees exactly what it wrote.
     */
    uint32_t pm4_microcode[256][2];
    uint16_t pm4_ucode_waddr;
    uint16_t pm4_ucode_raddr;

    /*
     * PIO alternative to the ring: the real Mac driver actually
     * submits its command stream by writing raw dwords straight to
     * PM4_FIFO_DATA_EVEN/ODD (0x1000/0x1004, undocumented in the OEM
     * manual but confirmed live: consecutive writes there decode as
     * an ordinary PM4 packet stream, ending in a packet0 write of the
     * exact 64-bit fence value the driver then polls for at
     * GUI_SCRATCH_REG0/1). Both addresses behave identically -- they
     * just feed the next dword into this same parser state machine.
     */
    /*
     * One packet-parser state per independent command stream: the PIO
     * FIFO stream gets a persistent one (writes arrive one dword at a
     * time across many MMIO accesses), while each indirect-buffer
     * dispatch runs a fresh private instance -- an IB dispatch is
     * triggered from *inside* a FIFO packet (packet0 hitting
     * IW_INDOFF/INDSIZE), so sharing the parser state would corrupt
     * the framing of both streams (a real bug: it desynced the FIFO
     * stream after the first IB and wedged the guest driver).
     */
    ATIR580PM4Parser pm4_fifo;
    ATIR580PM4Parser pm4_ring;

    /* unimplemented-command tally -- see ati_r580_note_gap() */
    uint32_t gap_count[R580_GAP_MAX][R580_GAP_SLOTS];

    /*
     * Silent-register tally -- see ati_r580_audit_reg_write(). One
     * counter per register in the audited window, indexed by offset>>2:
     * a first-seen slot list saturated on Mac OS X 10.5, which writes
     * over 200 distinct silent registers in one session (measured
     * 2026-08-30: all 160 slots used, 75960 writes dropped). The word
     * count mirrors R580_REG_AUDIT_LIMIT in the generated
     * ati_r580_audit.h; ati_r580_dbg.c build-checks the pairing.
     */
#define R580_SILENT_REG_WORDS 6144
    uint32_t silent_reg_count[R580_SILENT_REG_WORDS];

    /*
     * 2D GUI (destination datapath) engine state -- ported from the
     * real upstream `ati-vga` device (hw/display/ati.c/ati_2d.c), not
     * generic/free-standing; register semantics are documented on the
     * offsets themselves in ati_r580_regs.h. These mirror upstream's
     * ATIVGARegs 2D fields one-to-one so the ported blt logic needs no
     * renaming.
     */
    uint32_t dst_offset, dst_pitch, dst_tile, dst_width, dst_height;
    uint32_t dst_x, dst_y;
    uint32_t src_offset, src_pitch, src_tile, src_x, src_y;
    uint32_t dp_gui_master_cntl;
    uint32_t dp_brush_bkgd_clr, dp_brush_frgd_clr;
    uint32_t dp_src_frgd_clr, dp_src_bkgd_clr;
    /*
     * Scissors are SIGNED 14-bit fields -- the register guide gives the
     * range as -8192..8191 ("Destination left scissor", RAGE 128 VR/GL
     * Register Reference Manual 7.6). Holding them unsigned turned a
     * legitimately negative left/top edge -- what the driver programs
     * for a window hanging off the left or top of the screen -- into a
     * huge positive one, which clips the whole drawing away.
     */
    int32_t sc_top, sc_left, sc_bottom, sc_right;
    int32_t src_sc_bottom, src_sc_right;
    uint32_t dp_cntl, dp_datatype, dp_mix, dp_write_mask;
    uint32_t default_offset, default_pitch;
    int32_t default_sc_bottom, default_sc_right;

    /*
     * The pitch/offset and scissor fields above are the EFFECTIVE values
     * the drawing code uses; these are the register values they are
     * derived from. DP_GUI_MASTER_CNTL's SRC/DST_PITCH_OFFSET_CNTL and
     * SRC/DST_CLIPPING bits select, per operation, whether the effective
     * value comes from the source/destination registers or from the
     * DEFAULT_* ones -- a selection, not a transfer. Folding it straight
     * into the effective field on every GMC write (which is what this
     * used to do) destroys the register behind it, so the result depends
     * on the order the driver happens to write things in. Mac OS X hits
     * exactly that: it programs the separate SRC_PITCH/SRC_OFFSET
     * registers rather than the packed SRC_PITCH_OFFSET, and every
     * PAINT/BITBLT packet carries its own GMC dword, so a blit could run
     * with a pitch left over from an earlier, differently sized surface.
     * One 8-pixel pitch unit of staleness shears the copy by 8 pixels per
     * row -- the diagonally streaked window contents on the Rage.
     */
    uint32_t src_offset_reg, src_pitch_reg, src_tile_reg;
    uint32_t dst_offset_reg, dst_pitch_reg, dst_tile_reg;
    /*
     * Radeon's standalone SRC_PITCH/DST_PITCH registers count BYTES;
     * the packed *_PITCH_OFFSET registers keep the Rage 128 8-pixel
     * unit. Track which flavour each side's live value came from so
     * the engine applies the right stride (OS X pages textures in
     * through byte-pitch blits; treating those as 8-pixel units
     * shredded every window into 32x-spaced slices).
     */
    bool src_pitch_bytes_reg, dst_pitch_bytes_reg;
    bool src_pitch_bytes, dst_pitch_bytes;
    int32_t sc_top_reg, sc_left_reg, sc_bottom_reg, sc_right_reg;
    int32_t src_sc_bottom_reg, src_sc_right_reg;

    /*
     * Diagnostic: CPU stores into the frame buffer go through aperture 0,
     * which is a plain RAM alias, so nothing in this device can see them
     * -- and that is exactly the path that fills the driver's offscreen
     * staging surface. Setting the "fillwatch"/"fillwatch-size" properties
     * lays an instrumented IO window over that range of aperture 0 so the
     * fills become visible. Writes are coalesced into contiguous runs and
     * traced one line per run, so a full surface fill costs a line per row
     * rather than one per store.
     */
    MemoryRegion vram_watch;
    uint32_t fillwatch_off, fillwatch_size;
    uint32_t fw_run_start, fw_run_end;
    bool fw_active;

    /*
     * Diagnostic: 3D draw capture. When the "draw-capture" property names
     * a file, every draw the rasterizer runs is also written there as a
     * self-contained record -- the resolved draw state, the transformed
     * vertices, the texture, and the destination rectangle before and
     * after the draw. `cap_fp` is NULL unless the property was given, and
     * that pointer is the only thing the draw path tests, so an unarmed
     * capture costs one predictable branch per draw and changes nothing.
     * See ati_r580_cap.h for the record format.
     */
    char *cap_path;
    FILE *cap_fp;
    bool cap_arm;               /* record at all: a settable QOM property */
    uint32_t cap_max;           /* records to take before closing the file */
    uint32_t cap_max_px;        /* skip a draw whose rectangle exceeds this */
    uint32_t cap_index;         /* records written so far */
    uint32_t cap_skipped;
    struct {
        uint32_t off, len, xr, hash, rec;
    } cap_tex[R580_CAP_TEX_CACHE];
    unsigned cap_tex_n;

    /*
     * Host-GPU offload (phase 2, milestone M2). `gl_path` is the "gl"
     * property as given; `gl_mode` is it parsed, and `gl_ctx` is NULL
     * unless a backend was actually opened -- that pointer is the only
     * thing the draw path tests, so a device left at the default is
     * untouched by any of this. See ati_r580_gl.h for the backend
     * interface and ati_r580_gl.c for the GL implementation of it.
     */
    char *gl_path;
    char *gl_api;               /* "gl-api": gl (default) or metal */
    ATIR580GlMode gl_mode;
    struct R580GlCtx *gl_ctx;
    uint64_t gl_drawn;          /* draws the backend rendered */
    uint64_t gl_fb[R580_GLF_MAX][R580_GAP_SLOTS];   /* fallbacks, by prim */
    /*
     * Of those fallbacks, the ones that were PROVED to paint no pixels
     * and therefore did not give the resident render target back. They
     * are counted in gl_fb[] as well -- this is a sub-count of it, not
     * a family beside it, and it exists so that `gl-stats` shows the
     * skipped release rather than leaving it to be inferred from a
     * flush count that went down.
     */
    uint64_t gl_nowork;
    /* view-volume clipping: draws cut, triangles cut, primitives dropped */
    uint64_t clip_draws, clip_tris, clip_drop, clip_ucp;
    /* gl=verify: the per-pixel agreement between the two paths */
    uint64_t gl_v_px, gl_v_hist[4];     /* delta 0, 1, 2-4, above 4 */
    uint64_t gl_v_draws, gl_v_bad;      /* draws compared / with delta > 1 */
    uint64_t gl_v_cover_px, gl_v_cover;  /* coverage-class pixels / differing */
    uint64_t gl_v_mesh;         /* ... of those, from a multi-triangle draw */
    unsigned gl_v_max, gl_v_vmax;
    /*
     * Scratch for one request, grown on demand and never shrunk: a draw
     * needs the destination rectangle twice over in RGBA, the decoded
     * texture, and the expanded triangle list. Milestone M3 hands these
     * to a queue instead, which is why they are sized per request rather
     * than allocated inside the backend.
     */
    uint8_t *gl_before, *gl_out, *gl_sw, *gl_texbuf;
    size_t gl_rect_sz, gl_texbuf_sz;
    float *gl_verts;
    size_t gl_verts_sz;
    float *gl_tcx;              /* the general form's coordinate block */
    size_t gl_tcx_sz;
    /*
     * The self-overlap partition's working set, sized once rather than
     * allocated per draw: which pass each triangle landed in, the
     * triangles in pass order, where each pass begins in the vertex
     * array, and the bounding boxes the pairwise test screens with.
     */
    uint8_t gl_pass[R300_GL_TRI_MAX];
    unsigned gl_order[R300_GL_TRI_MAX];
    unsigned gl_pass_first[R300_GL_PASS_MAX + 1];
    float gl_bbox[R300_GL_TRI_MAX][4];
    bool gl_fast;               /* gl=fast: allow the additive one-pass blend */
    uint64_t gl_addblend;       /* draws GL's own blender ordered */
    uint64_t gl_multipass;      /* draws that needed more than one pass */
    uint64_t gl_passes;         /* ... and how many passes in total */
    /*
     * The GL-owned render target (milestone M3). The rules that keep it
     * coherent with emulated VRAM are stated in full at "GL-OWNED RENDER
     * TARGET" in ati_r580_3d.c; these are the four facts they need.
     * `gl_res` says a target is resident at all and is the ONLY thing
     * the hot hooks test.
     */
    bool gl_res;
    uint32_t gl_res_off, gl_res_pitch;
    unsigned gl_res_xr;
    int gl_tex_w, gl_tex_h;             /* the backend texture's extent */
    int gl_vx0, gl_vy0, gl_vx1, gl_vy1; /* seeded: the GPU copy is good */
    int gl_dx0, gl_dy0, gl_dx1, gl_dy1; /* drawn: the GPU copy is NEWER */
    uint64_t gl_flushes;                /* fetches back into VRAM */
    uint64_t gl_flush_px, gl_seed_px;   /* ... and pixels moved each way */
    uint64_t gl_rel[R580_GLR_MAX];      /* which hook ended a residency */
    uint64_t gl_rel_px[R580_GLR_MAX];   /* ... and what it cost to */
    /* the same question one level finer for the 2D engine's three paths */
    uint64_t gl_rel_2d[R580_GL2D_MAX];
    uint64_t gl_rel_2d_px[R580_GL2D_MAX];
    /*
     * The GL-owned DEPTH buffer. The rules are at "GL-OWNED DEPTH BUFFER"
     * in ati_r580_3d.c. `gl_zres` says the GPU holds a copy of the Z
     * buffer described by gl_z_*, good inside the seeded rectangle and
     * newer than VRAM inside the drawn one; `gl_zpark` that the copy
     * outlived a release and must be checked against the dirty bitmap
     * before it is used again; `gl_ztaint` that VRAM under it was written
     * while it was resident, so it may not be kept.
     */
    bool gl_depth;              /* "gl-depth": depth-tested draws on the GPU */
    bool gl_zres, gl_zpark, gl_ztaint;
    uint64_t gl_zepoch;
    uint32_t gl_z_off, gl_z_pitch;
    bool gl_z_macro, gl_z_micro, gl_z_aa, gl_z_z16;
    int gl_zvx0, gl_zvy0, gl_zvx1, gl_zvy1;     /* seeded */
    int gl_zdx0, gl_zdy0, gl_zdx1, gl_zdy1;     /* drawn: GPU is NEWER */
    uint32_t *gl_zstage, *gl_zbefore, *gl_zgpu;
    size_t gl_zstage_n, gl_zv_n;
    uint64_t gl_zdrawn, gl_zflushes, gl_zflush_px, gl_zseed_px;
    uint64_t gl_zclears, gl_zkept, gl_zstale, gl_zdropped;
    uint64_t gl_zdrop_why[R580_GLZD_MAX];
    /*
     * "gl-depth-resident": the copy stays on the GPU across releases and
     * goes back to VRAM only when something reads it. The guest CPU's
     * accesses are trapped by the DEPTH GUARD, a window over the
     * aperture at [zg_lo, zg_hi) laid over the Z buffer's pages; zg_zlo
     * and zg_zhi are the Z bytes inside it. Changed with the BQL and the
     * engine both held. See "GL-OWNED DEPTH BUFFER".
     */
    bool gl_zlazy;
    bool zg_ready, zg_on;
    MemoryRegion zg_io, zg_win;
    uint32_t zg_lo, zg_hi, zg_zlo, zg_zhi;
    uint64_t gl_zlazy_rel, gl_zg_arms, gl_zg_traps, gl_zg_back;
    /*
     * "gl-colour-resident": COLOUR GUARDS, windows over the aperture
     * laid over the pages of colour buffers the host GPU keeps newer
     * than VRAM across releases (see "RESIDENCY" in ati_r580_r5gpu.c).
     * Window k covers [cg_lo[k], cg_hi[k]), empty when lifted. Changed
     * with the BQL and the engine both held.
     */
    bool gl_cres;
    bool cg_ready;
    MemoryRegion cg_io, cg_win[ATI_R580_CGUARDS];
    uint32_t cg_lo[ATI_R580_CGUARDS], cg_hi[ATI_R580_CGUARDS];
    /*
     * A window traps every vCPU once each has run, after it was laid,
     * the work its own TLB flush was queued before: generation cg_gen
     * is in force when cg_acks reaches cg_need, and cg_live says so.
     */
    uint32_t cg_gen[ATI_R580_CGUARDS], cg_live[ATI_R580_CGUARDS];
    unsigned cg_acks[ATI_R580_CGUARDS], cg_need[ATI_R580_CGUARDS];
    /*
     * DEPTH WINDOW: a window over aperture 0 at [zw_lo, zw_hi), the real
     * extent of a depth buffer whose layout is not the linear one VRAM
     * keeps, laid while a SURFACEn range starts at the buffer. Guest
     * CPU accesses there reach the model's word for the same pixel (see
     * ati_r580_zwin_update()). Derived from the depth buffers drawn
     * lately, most recent first, not from the live registers, which
     * another context may have changed by the time the guest reads;
     * changed with the BQL and the engine both held.
     */
    struct {
        uint32_t off, zp, aa;
        bool z16;
    } zw_buf[4];
    bool zw_on;
    MemoryRegion zw_io, zw_win;
    uint32_t zw_surf;           /* SURFACEn with a range: bit n */
    uint32_t zw_lo, zw_hi, zw_pitch;
    unsigned zw_smp;            /* log2 of the samples a pixel */
    unsigned zw_miss;           /* untranslated layout last noted, + 1 */
    uint64_t zw_arms, zw_hits, zw_untr;
    uint64_t gl_zclr_px;        /* cleared on the GPU only */
    /* gl=verify over the depth buffer, classed as the colour is */
    uint64_t gl_vz_draws, gl_vz_px, gl_vz_diff, gl_vz_cover_px, gl_vz_cover;
    /*
     * The occlusion counter (ZB_ZPASS_DATA, ZB_ZPASS_ADDR); the rules
     * are at "ZPASS COUNTER" in ati_r580_3d.c. Engine state, except
     * zq_draw, which the raster threads add to.
     */
    bool gl_zpass;              /* "gl-zpass": count queried draws on the GPU */
    bool gl_async;              /* "gl-async-compile": worker threads */
    uint32_t gl_workers;        /* "gl-compile-workers": 1..8 */
    bool zq_on;                 /* reset since the last dump: counting */
    bool zq_late;               /* drawn uncounted since the last dump */
    bool zq_gl;                 /* GL counted a draw since the reset */
    /* a 3D_CLEAR_ZMASK held for the first draw on its buffer */
    bool zclr_pend;
    uint32_t zclr_first, zclr_n, zclr_zoff, zclr_zp, zclr_clr;
    uint32_t zclr_zfmt, zclr_bw, zclr_smp;
    uint32_t zq_base;           /* the value ZB_ZPASS_DATA was given */
    uint64_t zq_sw;             /* software samples since the reset */
    uint64_t zq_draw;           /* the current draw's software samples */
    uint64_t zq_t0;             /* backend ticket at the reset */
    bool zq_cv;                 /* zq_cp is the backend's sum at zq_ct */
    uint64_t zq_ct, zq_cp;
    struct {
        uint32_t addr, part;    /* where, and base + software samples */
        unsigned swap;          /* ZB_DEPTHPITCH DEPTHENDIAN */
        uint64_t t0, t;         /* backend tickets: reset, dump */
    } zq_pend[R580_ZQ_PEND];
    unsigned zq_npend;
    uint64_t zq_resets, zq_dumps, zq_deferred, zq_gldraws, zq_swdraws;
    uint64_t zq_offdraws, zq_reads;
    /* gl=verify: each draw counted both ways */
    uint64_t zq_v_draws, zq_v_bad, zq_v_sw, zq_v_gl;
    /*
     * Decoded textures, keyed on everything the decode depends on.
     *
     * An entry is valid while the VRAM range it was decoded from is
     * UNWRITTEN -- that, and not the render target's residency, is what
     * the decode depends on. `epoch` is the dirty-bitmap generation the
     * entry was last known clean in; see r300_gl_tex_current() for the
     * rule and its two enforcement points.
     */
    struct {
        uint32_t off, len, pitch;
        unsigned bpp, code, xr;
        unsigned yuv;
        unsigned sel[4];
        int w, h;
        unsigned nlev;              /* mip levels decoded, one after another */
        bool cube;                  /* each level six faces tall */
        uint32_t lay;               /* offset of the last of them */
        uint8_t *rgba;
        size_t sz;
        uint64_t used;
        uint64_t epoch;             /* the bitmap generation it is clean in */
        unsigned npg;               /* host pages the range spans */
        bool live;                  /* the decoded bytes are current */
        bool up;                    /* ... and the backend has them too */
        uint16_t hb;                /* hash bucket, or R300_GL_TEXNIL */
        uint16_t hnext;             /* next entry in that bucket */
    } gl_tex[R300_GL_TEXCACHE];
    uint32_t gl_tex_slots;      /* "gl-texcache-slots" */
    unsigned gl_tex_n;          /* entries in use */
    bool gl_tex_hashed;         /* hash lookup and byte bound */
    uint16_t gl_tex_bucket[R300_GL_TEXHASH];
    size_t gl_tex_bytes;        /* decoded bytes held */
    uint64_t gl_tex_pin;        /* entries used after this are this draw's */
    uint64_t gl_tex_trim;       /* live entries dropped for the byte bound */
    uint64_t gl_tex_seq, gl_tex_hit, gl_tex_miss;
    uint64_t gl_tex_stale;      /* entries the dirty guard killed */
    /*
     * Why the other lookups missed. A hit rate is only actionable with
     * these beside it: an entry refused admission, one a writer killed,
     * one a draw rendered over and one the LRU evicted are four
     * different problems.
     */
    uint64_t gl_tex_noadmit;    /* range was dirty: cannot be guarded */
    uint64_t gl_tex_wrote;      /* a writer hook killed it */
    uint64_t gl_tex_over;       /* a draw rendered into its range */
    uint64_t gl_tex_evict;      /* the LRU gave its slot away */
    uint64_t gl_epoch;          /* bumped whenever the VGA bitmap is cleared */
    unsigned gl_pgbits;         /* qemu_target_page_bits(), resolved once */
    ATIR580GlTexLife gl_texlife;
    char *gl_texlife_path;
    bool gl_tex_any;            /* any entry live: the hot hooks test this */

    /*
     * D1 hardware cursor as published to the console. hw_cursor_sum is
     * a checksum of the image, so a shape change made by writing VRAM
     * alone is picked up without re-uploading an unchanged cursor.
     */
    bool hw_cursor_on;
    uint32_t hw_cursor_sum;
    /* last position published to the console */
    int hw_cursor_x, hw_cursor_y;

    /* HOST_DATA0-7/LAST accumulator, same protocol as upstream */
    bool host_data_active;
    uint32_t host_data_row, host_data_col, host_data_next;
    uint32_t host_data_acc[4];
    /*
     * The drawing context the transfer STARTED with. A host-data transfer
     * spans many register writes, and its final partial accumulator is
     * only flushed when the NEXT blit arrives -- by which time the packet
     * that carries it has already installed its own destination rectangle,
     * datatype, pitch and scissors. Reading the live registers there wrote
     * one glyph's tail into the next glyph's cell, and where the stale
     * column index ran past the new width the unsigned `dst_width - col`
     * underflowed and painted sixteen pixels across its neighbours.
     * A transfer owns its context until it ends.
     */
    struct {
        uint32_t dst_x, dst_y, dst_width, dst_height;
        uint32_t dst_offset, dst_pitch;
        bool dst_pitch_bytes;
        uint32_t datatype;
        uint32_t src_frgd_clr, src_bkgd_clr;
        int sc_left, sc_top, sc_right, sc_bottom;
        unsigned host_swap_xor;   /* RBBM_GUICNTL.HOST_DATA_SWAP as a XOR */
    } hd;

    /*
     * Vertex-program RAM as uploaded through VAP_PVS_UPLOAD_ADDRESS/DATA.
     * It is one flat vector-indexed address space: vectors below
     * R300_PVS_CONST_START are program instructions, four dwords each,
     * and from there up they are the constant file. Neither is ever
     * cleared, so `pvs_code_slot_valid` remembers which instruction slots
     * the guest has actually written -- the control registers alone would
     * happily point at whatever a previous program left behind.
     */
    uint32_t pvs_upload_addr;
    uint32_t pvs_upload_cnt;
    uint32_t pvs_const[R300_PVS_CONST_SLOTS * 4];
    uint32_t pvs_const_dwords;
    /* user clip planes and point sprite state, R300_PVS_UCP_START on */
    uint32_t pvs_clip[R300_PVS_CLIP_VECS * 4];
    uint32_t pvs_code[R300_PVS_CODE_SLOTS * 4];
    R300PvsCompiled *pvs_cc;    /* the draw in hand's program, decoded */
    uint32_t pvs_code_slot_valid[R300_PVS_CODE_SLOTS / 32];
    /* dwords ever uploaded to the code region: is there a program at all */
    uint32_t pvs_code_dwords;

    /*
     * The Z buffer of the draw in hand, decoded from the ZB_* registers
     * by r300_setup_draw(). Not in R300DrawState: that structure's size
     * is the draw-capture format's version number.
     */
    struct {
        bool z_en;          /* ZB_CNTL Z_ENABLE or STENCIL_ENABLE */
        bool z_test;        /* ZB_CNTL Z_ENABLE */
        bool z_wr;          /* ZB_CNTL ZWRITEENABLE */
        bool s_en;          /* ZB_CNTL STENCIL_ENABLE */
        bool s_fb;          /* ZB_CNTL STENCIL_FRONT_BACK */
        uint32_t zsc;       /* ZB_ZSTENCILCNTL */
        uint8_t s_ref, s_mask, s_wmask;     /* ZB_STENCILREFMASK */
        bool vte_zs, vte_zo;    /* VAP_VTE_CNTL z scale/offset enables */
        uint32_t vte_fmt;   /* VAP_VTE_CNTL VTX_XY/Z/W0_FMT bits */
        unsigned zfunc;     /* ZB_ZSTENCILCNTL ZFUNC */
        uint32_t off;       /* VRAM byte offset, 0x20-aligned */
        uint32_t pitch;     /* pixels */
        bool macro, micro;  /* ZB_DEPTHPITCH tiling */
        bool aa;            /* GB_AA_CONFIG: two samples per pixel */
        bool z16;           /* ZB_FORMAT 16-bit Z, no stencil */
    } zb;

    /*
     * Parallel rasterisation, ati_r580_3d.c. `raster_threads` counts the
     * threads that draw, the submitting one included: 0 for half the
     * host's cores, 1 to draw serially. NULL `raster` draws serially.
     */
    struct R300Raster *raster;
    uint32_t raster_threads;
    uint64_t raster_tri_split;      /* triangles drawn across threads */
    uint64_t raster_tri_serial;     /* ... and on the submitting thread */

    /*
     * Phase 2, milestone M4: how much of what real guests upload the
     * GLSL translation in ati_r580_pvs_glsl.c can express. Armed by the
     * "pvs-glsl" property and otherwise never entered, because the
     * translation is not on any pixel's path -- the offline three-way
     * harness measures what it COMPUTES, and this measures what it
     * COVERS, which no corpus of seven programs can answer.
     *
     * A program is translated once, when the control registers first
     * name it: `pvs_tr_sig` is the range and constant base the last
     * attempt was made for, so a thousand draws of one program cost one
     * translation.
     */
    bool pvs_glsl;
    uint64_t pvs_tr_sig;
    uint64_t pvs_tr_ok, pvs_tr_refused;
    uint64_t pvs_tr_by_reason[4];   /* vector op, math op, dst file, addr */
    uint32_t pvs_tr_last_bytes, pvs_tr_last_nconst;
    uint32_t pvs_tr_last_in, pvs_tr_last_out;

    /*
     * Phase 2, milestone M5: the fragment program in force, decoded once
     * per draw out of the US register banks. It lives here rather than in
     * the per-draw state because it is three kilobytes of decoded
     * instruction and because the GL backend keys its shader cache on it.
     * `us_sig` is the signature of the control words and instruction
     * dwords the decode was made from, so a thousand draws of one program
     * cost one decode.
     */
    R300UsProgram us_prog;
    uint64_t us_sig;
    uint32_t us_out_fmt;        /* US_OUT_FMT_0 the decode was made with */
    uint32_t us_vtx_fmt1;       /* VAP_OUTPUT_VTX_FMT_1 likewise */
    uint64_t us_draws, us_refused;
    /*
     * The same program as GLSL, for the host-GPU backend, translated
     * whenever the decode is. `us_glsl_key` is what the backend caches
     * the linked shader under and is never zero for a usable program.
     * The constants are flattened per draw because they change without
     * the program changing.
     */
    char us_glsl[64 * 1024];
    bool us_glsl_ok;
    uint64_t us_glsl_key;
    uint64_t us_glsl_ok_n, us_glsl_refused_n;
    /*
     * A `gl_simple` program in the general form, for a draw that needs
     * its fetch inside the shader (a cube map); made on demand for the
     * program `us_glsl_gen_for` names, keyed by `us_glsl_gen_key`.
     */
    char us_glsl_gen[64 * 1024];
    uint64_t us_glsl_gen_for, us_glsl_gen_key;
    float us_konst_flat[R300_US_CONSTS * 4];

    /*
     * Staging buffer for an in-flight R300 3D_DRAW_IMMD_2 or
     * 3D_DRAW_INDX_2 payload (VAP_VF_CNTL + inline vertices or
     * indices). Scratch state only: a packet split across a migration
     * is lost, like the 2D host-data accumulator above.
     */
    uint32_t r300_immd[16384];
    /* VAP_VF_CNTL of a 3D_DRAW_INDX_2 waiting for its INDX_BUFFER */
    uint32_t r300_indx_vf;

    /* R500 pipeline (ati_r580_r5xx.c); NULL runs the R300 one */
    ATIR580R5 *r5;
    /* its host-GPU path (ati_r580_r5gpu.c); NULL without a backend */
    struct ATIR580R5Gpu *r5g;
    bool r5g_res;               /* a host copy is newer than VRAM */
    bool r5g_any;               /* host copies exist */
    bool r5g_quiet;             /* the CPU rasteriser runs; it invalidated */
    uint32_t r5g_wptr;
    char *r5g_dump;             /* gl=verify writes differing draws here */
    bool r500;
    uint32_t r500_threads;
};


/* ati_r580_dbg.c */
const char *ati_r580_reg_name(uint32_t base);

/*
 * Report a command the hardware understands and this model does not.
 * Warns on the first sight of each distinct value and counts every
 * one, so a silently dropped packet type shows up in the log the
 * first time a guest uses it instead of years later as a rendering
 * mystery. Packet3 opcode 0x1b -- the rectangle-only blit that moves
 * a window's body during a drag -- went unnoticed exactly that way:
 * its trace event existed but was never armed in any capture.
 * Read the running tally with `qom-get <device> gaps`.
 */
void ati_r580_note_gap(ATIR580State *s, ATIR580GapKind kind, unsigned idx);

/*
 * The register-level half of the same coverage question: tally a write
 * to a register no model code reads (the RBBM_GUICNTL class -- stored
 * into s->regs[] and consulted by no logic, so no gap ever fires).
 * Called from the ati_r580_reg_write32() funnel, which both the MMIO
 * path and PM4 type-0/1 packets go through. Read the tally with
 * `qom-get <device> silent-regs`.
 */
void ati_r580_audit_reg_write(ATIR580State *s, uint32_t base);

/* sizes of the private draw-capture payload structs (ati_r580_cap.h) */
uint32_t ati_r580_cap_state_bytes(void);
uint32_t ati_r580_cap_vtx_bytes(void);

/* why a draw went to the software rasterizer instead of the GL backend */
const char *ati_r580_gl_fb_name(ATIR580GlFallback why);

/*
 * GL-OWNED RENDER TARGET -- the coherency hooks. The rules are stated
 * in full where they are implemented, at "GL-OWNED RENDER TARGET" in
 * ati_r580_3d.c; what follows is how the rest of the device obeys them.
 *
 * ati_r580_gl_release() gives the target back: anything the host GPU
 * holds is fetched into VRAM first, and the GPU copy stops being
 * trusted. Call it before ANY code that reads or writes VRAM outside
 * the 3D draw path, and at every boundary where the guest CPU might run
 * -- which in this device means the end of a command-processor run,
 * because a whole ring or indirect buffer is drained inside one guest
 * store and nothing else can interleave with it.
 *
 * ati_r580_gl_touch() is the same thing for code that names a range and
 * is hot enough to care: it costs one predictable branch when no target
 * is resident, which is always the case with the default gl=off.
 */
void ati_r580_gl_release(ATIR580State *s, ATIR580GlRel why);
/* reset: give the target back and drop the decoded textures */
void ati_r580_gl_reset(ATIR580State *s);
void ati_r580_gl_sync(ATIR580State *s, uint32_t off, uint32_t len);
void ati_r580_gl_wrote(ATIR580State *s, uint32_t off, uint32_t len);
const char *ati_r580_gl_rel_name(ATIR580GlRel why);
/* the 2D sub-tally's row label -- see ATIR580Gl2dPath */
const char *ati_r580_gl_2d_path_name(ATIR580Gl2dPath path);

/*
 * The occlusion counter; see "ZPASS COUNTER" in ati_r580_3d.c. Called
 * by the thread that has the engine: the ZB_ZPASS_DATA and ZB_ZPASS_ADDR
 * writes, a ZB_ZPASS_DATA read, the guest-visible points where pending
 * dumps must have landed (settle), and reset.
 */
void ati_r580_zpass_reset(ATIR580State *s, uint32_t val);
void ati_r580_zpass_dump(ATIR580State *s, uint32_t addr);
uint32_t ati_r580_zpass_read(ATIR580State *s);
void ati_r580_zpass_settle(ATIR580State *s);
void ati_r580_zpass_drop(ATIR580State *s);

/*
 * The display refresh has just snapshotted and CLEARED the VGA dirty
 * bitmap, which is the one thing the decoded-texture cache reads to
 * know whether the guest CPU wrote over a texture. Hand the pages it
 * marks over before the snapshot is discarded (with scan_lock held,
 * the same hold as the clear); ati_r580_gl_epoch_apply() then kills
 * the entries over them and carries the rest into the new generation,
 * on the draw path, with gl_tex_lock held.
 */
void ati_r580_gl_epoch(ATIR580State *s, DirtyBitmapSnapshot *snap);
void ati_r580_gl_epoch_apply(ATIR580State *s);

/*
 * Take responsibility for the VGA dirty bits over a VRAM range the
 * decoded-texture cache wants to guard, feeding the scanout's own
 * accumulator first so nothing it uses is lost. False when the range
 * may not be claimed. Implemented in ati_r580.c, beside the scanout
 * code whose bits these are.
 */
bool ati_r580_gl_admit(ATIR580State *s, uint32_t off, uint32_t len);

/*
 * The depth guard (see "GL-OWNED DEPTH BUFFER" in ati_r580_3d.c).
 * ati_r580_zguard_arm() lays it over [lo, hi), page aligned, taking the
 * BQL if the caller does not hold it; the engine must be this thread's.
 * ati_r580_gl_zguard() is the trap's half in the draw code: the guest
 * CPU is about to access [off, off+len), the engine is held, and it
 * returns true when the guard has nothing left to watch.
 */
bool ati_r580_zguard_arm(ATIR580State *s, uint32_t lo, uint32_t hi,
                         uint32_t zlo, uint32_t zhi);
bool ati_r580_gl_zguard(ATIR580State *s, uint32_t off, unsigned len);

/*
 * Colour guard window k over [lo, hi), page aligned; lo == hi lifts it.
 * Takes the BQL if the caller does not hold it; the engine must be this
 * thread's. False when no window can be laid.
 */
bool ati_r580_cguard_set(ATIR580State *s, int k, uint32_t lo, uint32_t hi);

/*
 * The depth window again after a write to SURFACEn or to the depth
 * buffer's offset, pitch, format or GB_AA_CONFIG. Takes the BQL if the
 * caller does not hold it; the engine must be this thread's or idle.
 */
void ati_r580_zwin_update(ATIR580State *s);
void ati_r580_zwin_note(ATIR580State *s);
/* window k traps every guest CPU access to [cg_lo[k], cg_hi[k]) */
static inline bool ati_r580_cguard_live(ATIR580State *s, int k)
{
    return s->cg_hi[k] > s->cg_lo[k] &&
           qatomic_read(&s->cg_live[k]) == s->cg_gen[k];
}

bool ati_r580_on_engine(void);
void ati_r580_engine_wait(ATIR580State *s);

/*
 * May this thread use the GL target and the command processor's state
 * right now: always on the engine thread or the thread that claimed the
 * engine, elsewhere only while it is idle. Without lockless register
 * access it cannot become busy under the BQL, which every other caller
 * holds; with it, a caller here takes the engine if it is idle, and
 * ati_r580_gl_leave() gives it back when *claimed says so.
 */
bool ati_r580_gl_enter(ATIR580State *s, bool *claimed);
void ati_r580_gl_leave(ATIR580State *s, bool claimed);

/* something is about to READ this range of VRAM */
static inline void ati_r580_gl_touch(ATIR580State *s, uint32_t off,
                                     uint32_t len)
{
    if (unlikely(s->gl_res || s->gl_zres || s->r5g_res)) {
        ati_r580_gl_sync(s, off, len);
    }
}

/* ... and about to WRITE it, which also stales anything decoded from it */
static inline void ati_r580_gl_dirty(ATIR580State *s, uint32_t off,
                                     uint32_t len)
{
    if (unlikely(s->gl_res || s->gl_tex_any || s->gl_zres ||
                 (s->r5g_any && !s->r5g_quiet))) {
        ati_r580_gl_wrote(s, off, len);
    }
}

/* ati_r580_3d.c */
void ati_r580_r300_draw_immd(ATIR580State *s, const uint32_t *dw, unsigned n);
void ati_r580_r300_draw_vbuf(ATIR580State *s, uint32_t vf);
void ati_r580_r300_draw_indx(ATIR580State *s, const uint32_t *dw, unsigned n);
void ati_r580_r300_indx_buffer(ATIR580State *s, uint32_t port, uint32_t addr,
                               uint32_t n);
void ati_r580_r300_clear_zmask(ATIR580State *s, uint32_t first, uint32_t n,
                               uint32_t val);
void ati_r580_raster_init(ATIR580State *s);
void ati_r580_raster_fini(ATIR580State *s);

/* ati_r580.c MC-window translation, shared with the engines */
bool ati_r580_mc_to_vram(ATIR580State *s, uint32_t addr, uint32_t *off);
uint32_t ati_r580_mc_read32(ATIR580State *s, uint32_t addr);
void ati_r580_mc_write32(ATIR580State *s, uint32_t addr, uint32_t val);
void ati_r580_mc_read_block(ATIR580State *s, uint32_t addr, uint32_t *dst,
                            unsigned n);
#define R580_MC_MAP_REGS 11
void ati_r580_mc_map_save(ATIR580State *s, uint32_t *map);
bool ati_r580_mc_map_same(ATIR580State *s, const uint32_t *map);
/*
 * Trace helper: name the window a card address resolves through and
 * return the address it resolves to ("vram" -> a VRAM byte offset,
 * "agp" -> a bus address after AGP_BASE, "bus" -> passed through
 * untranslated). Read-only; it exists so a trace can record where a
 * fetch really went instead of leaving it to be inferred from
 * post-hoc register reads.
 */
const char *ati_r580_mc_describe(ATIR580State *s, uint32_t addr,
                                 uint64_t *target);

/* ati_r580_2d.c */
void ati_r580_2d_blt(ATIR580State *s);
void ati_r580_2d_scale(ATIR580State *s, const uint32_t *pkt);
void ati_r580_2d_scale_regs(ATIR580State *s);
bool ati_r580_host_data_flush(ATIR580State *s);

/* the frame buffer aperture maps VRAM linearly: always 0 */
unsigned ati_r580_aper_xor(ATIR580State *s, uint32_t off);
/* a chip-native little-endian VRAM dword */
uint32_t ati_r580_vram_ld32(ATIR580State *s, uint32_t off);

/* ati_r580_cp.c: PM4 command processor and engine thread */
void ati_r580_pm4_run(ATIR580State *s);
void ati_r580_pm4_run_ring(ATIR580State *s);
void ati_r580_pm4_parse(ATIR580State *s, ATIR580PM4Parser *p, uint32_t val);
void ati_r580_pm4_fifo_push(ATIR580State *s, uint32_t val);
void ati_r580_pm4_indirect(ATIR580State *s, uint32_t offset, uint32_t dwords);
void ati_r580_cp_rptr_load(ATIR580State *s);
void ati_r580_scratch_writeback(ATIR580State *s, unsigned n);
bool ati_r580_engine_reg(uint32_t base);
bool ati_r580_engine_status_reg(uint32_t base);
bool ati_r580_engine_private_reg(uint32_t base);
bool ati_r580_engine_bql_reg(uint32_t base);
void ati_r580_engine_kick(ATIR580State *s);
void ati_r580_engine_claim(ATIR580State *s);
void ati_r580_engine_unclaim(ATIR580State *s, bool kick);
void ati_r580_engine_wptr(ATIR580State *s, uint32_t val);
void ati_r580_engine_init(ATIR580State *s);
void ati_r580_engine_fini(ATIR580State *s);

/* ati_r580.c */
void ati_r580_reg_write32(ATIR580State *s, uint32_t base, uint32_t val);
void ati_r580_update_irq(ATIR580State *s);
void ati_r580_cursor_update(ATIR580State *s);

/* ati_r580_avivo.c: AVIVO display controller, on the BQL */
void ati_r580_avivo_init(ATIR580State *s);
void ati_r580_avivo_fini(ATIR580State *s);
void ati_r580_avivo_reset(ATIR580State *s);
bool ati_r580_avivo_read(ATIR580State *s, uint32_t off, uint32_t *val);
bool ati_r580_avivo_write(ATIR580State *s, uint32_t off, uint32_t val);
bool ati_r580_avivo_irq_pending(ATIR580State *s);
bool ati_r580_avivo_scanout(ATIR580State *s, ATIR580Mode *m);
const uint8_t (*ati_r580_avivo_lut(ATIR580State *s))[3];
void ati_r580_avivo_cursor_apply(ATIR580State *s);

/* ati_r580_atom.c: AtomBIOS mailbox */
bool ati_r580_atom_write(ATIR580State *s, uint32_t off, uint32_t val);

#endif /* ATI_R580_INT_H */
