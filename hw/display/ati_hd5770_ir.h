/*
 * ATI Radeon HD 5770 (Juniper) shader IR: a pixel shader as
 * straight-line operations on a compact register file, or (CF form) as
 * its control flow program. The CPU lane executor (ati_hd5770_sh.c)
 * runs the straight-line form, host GPU backends translate both.
 *
 * REGISTERS. GPR n of the program is compact register map[n] (ngpr in
 * all, at most HD5770_IR_GPRS), four 32-bit channels each, untyped.
 * Compact register map[0] starts with the barycentrics: .x = b1,
 * .y = b2; the position GPR, if the draw has one, with (x + c, y + c,
 * 0, 1), c the pixel centre.
 *
 * OPS run in order:
 *  ALU    groups[first .. first + n - 1]. A group is one VLIW bundle:
 *         every source of every instruction is read before any result
 *         is written. Results then go to PV (slot 0-3) or PS (slot 4),
 *         and, if wr, to register dst channel dst_chan. An instruction
 *         computes what eval() and post() in ati_hd5770_sh.c compute
 *         for its opcode (OP2 0x000-0x0ff, OP3 0x800 | op). Three kinds
 *         are group operations: DOT4/DOT4_IEEE (all of them in a group
 *         get the sum over the group's dot instructions, in order),
 *         MAX4 (the maximum), INTERP_* (parameter prm, channel dst_chan,
 *         the provoking vertex's value if flat). KILL* opcodes kill the
 *         pixel when true. needed: the result is written or read later;
 *         an instruction that is not needed may be skipped.
 *  FETCH  fetches[first .. first + n - 1], each in order: TEX (SAMPLE*
 *         or LD of resource res, words w as in the R800 ISA), VTX (vertex
 *         fetch of buffer resource w[0] >> 8 & 0xff at the index in
 *         src channel w[0] >> 24 & 3), GRAD_H / GRAD_V (the source at the
 *         pixel to the right / below minus at this pixel, as it is when
 *         that pixel reaches this fetch). Destination selects from w[1].
 *  EXPORT exports[first .. first + n - 1]: to target (colour buffer
 *         0-7, or 61: the depth in x and the stencil reference in y)
 *         from register gpr, channel selects sel[4] (0-3 channel, 4 zero,
 *         5 one, other: 0u).
 *
 * Group operations also include CUBE (the instructions of a group with
 * cube set compute one cube map face selection together, as cube() in
 * ati_hd5770_sh.c). TEX fetches include SAMPLE_C* (opcodes 0x18-0x1b).
 *
 * CF FORM (cf set). A program with flow control has one op per CF
 * instruction, ops[i] being CF instruction i, and runs as
 * hd5770_prog_run() runs it: ALU ops carry their CF instruction in cf
 * (0x80 | ALU CF opcode: push before, pop after, else after, break,
 * continue), the others in cf too (kind HD5770_IRO_CF: PUSH, POP, ELSE,
 * JUMP, LOOP_*, CALL, RETURN, KILL, END_PROGRAM, NOP). addr is the
 * target op, popn, cond and cst (boolean or loop constant) as in the CF
 * word, eop the end-of-program bit. Instructions may then update the
 * execution mask and the predicate (update_exec, update_pred), be
 * predicated (pred_sel 2: when the predicate is set, 1: when clear),
 * set the address register (mova) and address registers relatively
 * (rel, dst_rel; index_mode 0-3 AR channel, 4 loop index); ngpr is then
 * up to 128 and map[] the identity when rel is set.
 *
 * SOURCES: GPR (idx, chan), PV (chan 0-3, or 4 for PS), CONST (slot k
 * of the constants bound per draw, see consts[]), IMM (imm bits). neg
 * and abs are applied as eval() applies them (not to integer opcodes).
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef ATI_HD5770_IR_H
#define ATI_HD5770_IR_H

#define HD5770_IR_GPRS          64

enum {
    HD5770_IRS_GPR, HD5770_IRS_PV, HD5770_IRS_CONST, HD5770_IRS_IMM,
};

typedef struct HD5770IRSrc {
    uint8_t kind, idx, chan;
    bool neg, abs;
    uint16_t k;
    uint32_t imm;
    bool rel;                   /* CF form: GPR idx + the index register */
} HD5770IRSrc;

typedef struct HD5770IRInst {
    uint16_t op;                /* OP2, or 0x800 | OP3 */
    uint8_t slot, dst, dst_chan, nsrc, omod, prm;
    bool wr, clamp, needed, interp, flat, dot, max4, kill;
    HD5770IRSrc src[3];
    const void *priv;           /* the interpreter's decoded instruction */
    bool cube;
    /* CF form */
    uint8_t pred_sel, index_mode;
    bool update_exec, update_pred, dst_rel, mova;
} HD5770IRInst;

typedef struct HD5770IRGroup {
    uint16_t first, n;
    bool dot, max4, cube;
} HD5770IRGroup;

enum {
    HD5770_IRF_TEX, HD5770_IRF_VTX, HD5770_IRF_GRAD_H, HD5770_IRF_GRAD_V,
};

typedef struct HD5770IRFetch {
    uint8_t kind, src, dst;
    uint8_t res;                /* TEX: resource id (PS resources 0..) */
    bool fs;                    /* VTX of the fetch shader (its resources) */
    uint32_t w[3];
} HD5770IRFetch;

typedef struct HD5770IRExport {
    uint8_t gpr;
    uint8_t sel[4];
    uint8_t target;
} HD5770IRExport;

#define HD5770_IR_DEPTH         61
/* vertex shaders: the position, and parameter n at HD5770_IR_PARAM0 + n */
#define HD5770_IR_POS           60
#define HD5770_IR_PARAM0        64

enum { HD5770_IRO_ALU, HD5770_IRO_FETCH, HD5770_IRO_EXPORT, HD5770_IRO_CF };

typedef struct HD5770IROp {
    uint8_t kind;
    uint16_t first, n;
    /* CF form */
    uint8_t cf, popn, cond, cst;
    uint16_t addr;
    bool eop;
} HD5770IROp;

/* constant slot k: kcache bank/element idx channel, or DX9 file entry */
typedef struct HD5770IRConst {
    bool cfile;
    uint8_t bank, chan;
    uint32_t idx;
} HD5770IRConst;

typedef struct HD5770IR {
    uint8_t map[128];           /* GPR to compact register, 0xff none */
    unsigned ngpr;
    HD5770IROp *ops;
    HD5770IRGroup *groups;
    HD5770IRInst *insts;
    HD5770IRFetch *fetches;
    HD5770IRExport *exports;
    HD5770IRConst *consts;
    unsigned nops, ngroups, ninsts, nfetches, nexports, nconsts;
    uint8_t res[16];            /* distinct TEX resources, in first use order */
    unsigned nres;
    bool grad;                  /* GRAD_H/GRAD_V present */
    uint64_t hash;              /* of the program words */
    uint32_t cmask;             /* colour buffers exported */
    bool zexport;               /* the depth/stencil export */
    bool cf;                    /* CF FORM above */
    bool vs;                    /* a vertex shader (and its fetch shader) */
    bool rel;                   /* relative register addressing */
} HD5770IR;

#endif
