/*
 * ATI Radeon HD 5770 (Juniper) shader programs: R800 (Evergreen) bytecode
 * decoder and interpreter.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef ATI_HD5770_SH_H
#define ATI_HD5770_SH_H

#include "ati_hd5770_gfx.h"
#include "ati_hd5770_ir.h"

#define HD5770_GPRS             128
#define HD5770_PARAMS           32
#define HD5770_MRTS             8

typedef union HD5770Val {
    float f;
    uint32_t u;
    int32_t i;
} HD5770Val;

enum {
    HD5770_STAGE_VS = 0,
    HD5770_STAGE_PS = 1,
    HD5770_STAGE_FS = 2,
};

typedef struct HD5770Prog HD5770Prog;
typedef struct HD5770Thread HD5770Thread;
typedef struct HD5770Lanes HD5770Lanes;

/* what a program reads from outside its registers */
typedef struct HD5770ShEnv {
    void *opaque;
    unsigned stage;             /* HD5770_STAGE_VS or _PS */
    const uint32_t *cfile;      /* DX9 constant file of the stage, 256 x 4 */
    /* constant buffer n of the stage, element idx */
    void (*kcache)(void *opaque, unsigned buf, unsigned idx, HD5770Val v[4]);
    /*
     * VTX fetch: w = the instruction's three words, index from the GPR,
     * stage of the program that fetches (a fetch shader has its own
     * resources)
     */
    void (*vfetch)(void *opaque, const uint32_t *w, uint32_t index,
                   unsigned stage, HD5770Val v[4]);
    /* TEX: w = the instruction's three words, coordinates c */
    void (*tfetch)(void *opaque, const uint32_t *w, const HD5770Val c[4],
                   HD5770Thread *t, HD5770Val v[4]);
    /* PS: interpolated parameter p, or its value at the first vertex */
    float (*interp)(void *opaque, unsigned p, unsigned chan, bool flat);
    /*
     * PS: register gpr at the neighbouring pixel (dx, dy), as it is when
     * that pixel reaches fetch instruction inst of CF instruction cf
     */
    bool (*neighbour)(void *opaque, int dx, int dy, uint32_t cf,
                      uint32_t inst, unsigned gpr, HD5770Val out[4]);
    /* PS lanes: parameter p of lanes 0..l->n-1 */
    void (*interp_n)(void *opaque, const struct HD5770Lanes *l, unsigned p,
                     unsigned chan, bool flat, HD5770Val *out);
    /*
     * PS, mipmapped textures: TEX with the derivatives of the
     * coordinates, d[0..3] across x and d[4..7] across y; NULL: tfetch()
     * for every TEX
     */
    void (*tfetch_d)(void *opaque, const uint32_t *w, const HD5770Val c[4],
                     const float d[8], HD5770Val v[4]);
    const HD5770Prog *fetch;    /* fetch shader a VS calls */
    const uint32_t *loop_const; /* SQ_LOOP_CONST of the stage, 32 */
    const uint32_t *bool_const; /* SQ_BOOL_CONST of the stage */
    HD5770GfxStats *st;
} HD5770ShEnv;

struct HD5770Thread {
    HD5770Val r[HD5770_GPRS][4];
    HD5770Val pos[4];
    HD5770Val param[HD5770_PARAMS][4];
    HD5770Val color[HD5770_MRTS][4];
    uint32_t param_mask, color_mask;
    bool pos_written, killed;
    HD5770Val depth, stencil;   /* the DB export: depth, stencil reference */
    bool depth_written;
};

HD5770Prog *hd5770_prog_new(const uint32_t *code, uint32_t ndw,
                            unsigned stage, HD5770GfxStats *st);
void hd5770_prog_free(HD5770Prog *p);
bool hd5770_prog_bad(const HD5770Prog *p);
/* run to the end of the program; false if it holds something not modelled */
bool hd5770_prog_run(const HD5770Prog *p, const HD5770ShEnv *env,
                     HD5770Thread *t);
/* run up to fetch instruction inst of CF instruction cf, return gpr */
bool hd5770_prog_run_until(const HD5770Prog *p, const HD5770ShEnv *env,
                           HD5770Thread *t, uint32_t cf, uint32_t inst,
                           unsigned gpr, HD5770Val out[4]);
/*
 * Pixel shaders without flow control run on up to HD5770_LANES pixels
 * at a time, each instruction decoded once for all of them: constants
 * are resolved once per draw (hd5770_prog_bind()), operands are lane
 * arrays. Every lane computes what hd5770_prog_run() computes for its
 * pixel, operation for operation.
 */
#define HD5770_LANES            16
#define HD5770_LANE_GPRS        HD5770_IR_GPRS

typedef struct HD5770Lanes {
    unsigned n;
    /*
     * inputs: barycentrics (b1, b2 go to r0.xy), 1/w for perspective,
     * the position in GPR pos_gpr if >= 0
     */
    float b0[HD5770_LANES], b1[HD5770_LANES], b2[HD5770_LANES];
    float pw[HD5770_LANES];
    float px[HD5770_LANES], py[HD5770_LANES];
    int pos_gpr;
    /* outputs */
    HD5770Val color[4][HD5770_LANES];   /* colour export 0 */
    bool color0;                        /* colour 0 exported */
    bool killed[HD5770_LANES];
    /* state */
    HD5770Val r[HD5770_LANE_GPRS][4][HD5770_LANES];
    HD5770Val pv[5][HD5770_LANES];
} HD5770Lanes;

/*
 * the program runs on lanes; *nconst constants for hd5770_prog_bind();
 * *grad: it takes gradients, and hd5770_prog_run_lanes() needs the
 * pixels to the right and below each lane in nh and nv
 */
bool hd5770_prog_lanes(const HD5770Prog *p, unsigned *nconst, bool *grad);
/* the IR of a program that runs on lanes, else NULL */
const HD5770IR *hd5770_prog_ir(const HD5770Prog *p);
/* resources [0..*n) the program's fetch instructions read */
void hd5770_prog_resources(const HD5770Prog *p, uint8_t *ids, unsigned *n,
                           unsigned max);
void hd5770_prog_bind(const HD5770Prog *p, const HD5770ShEnv *env,
                      HD5770Val *kv);
void hd5770_prog_run_lanes(const HD5770Prog *p, const HD5770ShEnv *env,
                           const HD5770Val *kv, HD5770Lanes *l,
                           HD5770Lanes *nh, HD5770Lanes *nv);

/* the decoded size in dwords, from the end-of-program CF instruction */
uint32_t hd5770_prog_size(const uint32_t *code, uint32_t max);
/* census entry of a program */
void hd5770_prog_census(const HD5770Prog *p, HD5770ShaderInfo *si);
/* disassembly, for reports */
void hd5770_prog_dump(const HD5770Prog *p, GString *out);

#endif
