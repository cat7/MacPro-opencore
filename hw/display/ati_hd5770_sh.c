/*
 * ATI Radeon HD 5770 (Juniper) shader programs: R800 (Evergreen) bytecode
 * decoder and interpreter.
 *
 * A program is a list of 64-bit control flow (CF) instructions; ALU
 * clauses hold VLIW groups of up to five instructions (slots x, y, z, w
 * and the transcendental t) followed by their literals, fetch clauses
 * hold 128-bit TEX or VTX instructions. One invocation runs at a time:
 * the execution mask is one bit, the CF stack holds its saved values.
 * Opcodes, CF instructions, operand selects and fetch modes the
 * interpreter does not implement are counted per opcode in the stats and
 * make the program "bad": the draw that uses it is skipped, not guessed.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include <math.h>
#include <float.h>
#include "qemu/bswap.h"
#include "ati_hd5770_sh.h"

#define MAX_CF          1024
#define STACK_DEPTH     32
#define LOOP_DEPTH      8
#define CALL_DEPTH      8
#define STEP_LIMIT      (1u << 20)

/* CF instructions (CF_WORD1[29:22]) */
enum {
    CF_NOP = 0x00, CF_TEX = 0x01, CF_VTX = 0x02, CF_LOOP_START = 0x04,
    CF_LOOP_END = 0x05, CF_LOOP_START_DX10 = 0x06,
    CF_LOOP_START_NO_AL = 0x07, CF_LOOP_CONTINUE = 0x08,
    CF_LOOP_BREAK = 0x09, CF_JUMP = 0x0a, CF_PUSH = 0x0b, CF_ELSE = 0x0d,
    CF_POP = 0x0e, CF_CALL = 0x12, CF_CALL_FS = 0x13, CF_RETURN = 0x14,
    CF_KILL = 0x18, CF_END_PROGRAM = 0x19, CF_WAIT_ACK = 0x1a,
    CF_TEX_ACK = 0x1b, CF_VTX_ACK = 0x1c, CF_EXPORT = 0x53,
    CF_EXPORT_DONE = 0x54,
};

/* ALU CF instructions (CF_ALU_WORD1[29:26]) */
enum {
    CF_ALU = 0x8, CF_ALU_PUSH_BEFORE = 0x9, CF_ALU_POP_AFTER = 0xa,
    CF_ALU_POP2_AFTER = 0xb, CF_ALU_EXTENDED = 0xc, CF_ALU_CONTINUE = 0xd,
    CF_ALU_BREAK = 0xe, CF_ALU_ELSE_AFTER = 0xf,
};

/* OP2 ALU instructions */
enum {
    OP2_ADD = 0x00, OP2_MUL = 0x01, OP2_MUL_IEEE = 0x02, OP2_MAX = 0x03,
    OP2_MIN = 0x04, OP2_MAX_DX10 = 0x05, OP2_MIN_DX10 = 0x06,
    OP2_SETE = 0x08, OP2_SETGT = 0x09, OP2_SETGE = 0x0a, OP2_SETNE = 0x0b,
    OP2_SETE_DX10 = 0x0c, OP2_SETGT_DX10 = 0x0d, OP2_SETGE_DX10 = 0x0e,
    OP2_SETNE_DX10 = 0x0f, OP2_FRACT = 0x10, OP2_TRUNC = 0x11,
    OP2_CEIL = 0x12, OP2_RNDNE = 0x13, OP2_FLOOR = 0x14,
    OP2_ASHR_INT = 0x15, OP2_LSHR_INT = 0x16, OP2_LSHL_INT = 0x17,
    OP2_MOV = 0x19, OP2_NOP = 0x1a, OP2_PRED_SETGT_UINT = 0x1e,
    OP2_PRED_SETGE_UINT = 0x1f, OP2_PRED_SETE = 0x20,
    OP2_PRED_SETGT = 0x21, OP2_PRED_SETGE = 0x22, OP2_PRED_SETNE = 0x23,
    OP2_KILLE = 0x2c, OP2_KILLGT = 0x2d, OP2_KILLGE = 0x2e,
    OP2_KILLNE = 0x2f, OP2_AND_INT = 0x30, OP2_OR_INT = 0x31,
    OP2_XOR_INT = 0x32, OP2_NOT_INT = 0x33, OP2_ADD_INT = 0x34,
    OP2_SUB_INT = 0x35, OP2_MAX_INT = 0x36, OP2_MIN_INT = 0x37,
    OP2_MAX_UINT = 0x38, OP2_MIN_UINT = 0x39, OP2_SETE_INT = 0x3a,
    OP2_SETGT_INT = 0x3b, OP2_SETGE_INT = 0x3c, OP2_SETNE_INT = 0x3d,
    OP2_SETGT_UINT = 0x3e, OP2_SETGE_UINT = 0x3f,
    OP2_KILLGT_UINT = 0x40, OP2_KILLGE_UINT = 0x41,
    OP2_PRED_SETE_INT = 0x42, OP2_PRED_SETGT_INT = 0x43,
    OP2_PRED_SETGE_INT = 0x44, OP2_PRED_SETNE_INT = 0x45,
    OP2_KILLE_INT = 0x46, OP2_KILLGT_INT = 0x47, OP2_KILLGE_INT = 0x48,
    OP2_KILLNE_INT = 0x49, OP2_FLT_TO_INT = 0x50,
    OP2_EXP_IEEE = 0x81, OP2_LOG_CLAMPED = 0x82, OP2_LOG_IEEE = 0x83,
    OP2_RECIP_CLAMPED = 0x84, OP2_RECIP_FF = 0x85, OP2_RECIP_IEEE = 0x86,
    OP2_RECIPSQRT_CLAMPED = 0x87, OP2_RECIPSQRT_FF = 0x88,
    OP2_RECIPSQRT_IEEE = 0x89, OP2_SQRT_IEEE = 0x8a, OP2_SIN = 0x8d,
    OP2_COS = 0x8e, OP2_MULLO_INT = 0x8f, OP2_MULHI_INT = 0x90,
    OP2_MULLO_UINT = 0x91, OP2_MULHI_UINT = 0x92, OP2_FLT_TO_UINT = 0x9a,
    OP2_INT_TO_FLT = 0x9b, OP2_UINT_TO_FLT = 0x9c,
    OP2_FLT_TO_INT_RPI = 0xb0, OP2_FLT_TO_INT_FLOOR = 0xb1, OP2_DOT4 = 0xbe,
    OP2_DOT4_IEEE = 0xbf, OP2_MAX4 = 0xc1, OP2_MOVA_INT = 0xcc,
    OP2_INTERP_XY = 0xd6, OP2_INTERP_ZW = 0xd7, OP2_INTERP_X = 0xd8,
    OP2_INTERP_Z = 0xd9, OP2_INTERP_LOAD_P0 = 0xe0,
};

/* OP3 ALU instructions */
enum {
    OP3_BFE_UINT = 0x04, OP3_BFE_INT = 0x05, OP3_BFI_INT = 0x06,
    OP3_FMA = 0x07, OP3_MULADD = 0x14, OP3_MULADD_M2 = 0x15,
    OP3_MULADD_M4 = 0x16, OP3_MULADD_D2 = 0x17, OP3_MULADD_IEEE = 0x18,
    OP3_CNDE = 0x19, OP3_CNDGT = 0x1a, OP3_CNDGE = 0x1b,
    OP3_CNDE_INT = 0x1c, OP3_CNDGT_INT = 0x1d, OP3_CNDGE_INT = 0x1e,
};

/* operand selects */
enum {
    SEL_KCACHE0 = 128, SEL_KCACHE1 = 160, SEL_0 = 248, SEL_1 = 249,
    SEL_1_INT = 250, SEL_M_1_INT = 251, SEL_0_5 = 252, SEL_LITERAL = 253,
    SEL_PV = 254, SEL_PS = 255, SEL_CFILE = 256, SEL_PARAM = 448,
};

/* one decoded ALU instruction */
typedef struct AluInst {
    uint16_t op;                /* OP2 opcode, or 0x800 | OP3 opcode */
    uint8_t slot;               /* 0-3 x-w, 4 t */
    uint8_t dst_gpr, dst_chan;
    bool dst_rel, write, clamp;
    uint8_t omod, index_mode, pred_sel;
    bool update_exec, update_pred;
    uint16_t sel[3];
    uint8_t chan[3];
    bool neg[3], abs_[3], rel[3];
    uint8_t nsrc;
} AluInst;

typedef struct AluGroup {
    AluInst inst[5];
    uint8_t n;
    uint32_t lit[4];
} AluGroup;

typedef struct CfInst {
    uint32_t w0, w1;
    uint8_t op;                 /* CF instruction, ALU ones as 0x80 | op */
    uint16_t group0, ngroups;   /* ALU: groups[] range */
} CfInst;

struct HD5770Prog {
    uint32_t *code;
    uint32_t ndw;
    unsigned stage;
    CfInst cf[MAX_CF];
    uint32_t ncf;
    AluGroup *groups;
    uint32_t ngroups;
    bool bad;
    uint32_t hash;
};

/* ---------------------------------------------------------------- */
/* Decoder                                                          */

static bool op2_trans_only(unsigned op)
{
    return (op >= 0x81 && op <= 0x9c) || op == 0x50 || op == 0x9a ||
           op == 0x9b || op == 0x9c;
}

static bool op2_known(unsigned op)
{
    switch (op) {
    case OP2_ADD: case OP2_MUL: case OP2_MUL_IEEE: case OP2_MAX:
    case OP2_MIN: case OP2_MAX_DX10: case OP2_MIN_DX10: case OP2_SETE:
    case OP2_SETGT: case OP2_SETGE: case OP2_SETNE: case OP2_SETE_DX10:
    case OP2_SETGT_DX10: case OP2_SETGE_DX10: case OP2_SETNE_DX10:
    case OP2_FRACT: case OP2_TRUNC: case OP2_CEIL: case OP2_RNDNE:
    case OP2_FLOOR: case OP2_ASHR_INT: case OP2_LSHR_INT:
    case OP2_LSHL_INT: case OP2_MOV: case OP2_NOP:
    case OP2_PRED_SETGT_UINT: case OP2_PRED_SETGE_UINT:
    case OP2_PRED_SETE: case OP2_PRED_SETGT: case OP2_PRED_SETGE:
    case OP2_PRED_SETNE: case OP2_KILLE: case OP2_KILLGT: case OP2_KILLGE:
    case OP2_KILLNE: case OP2_AND_INT: case OP2_OR_INT: case OP2_XOR_INT:
    case OP2_NOT_INT: case OP2_ADD_INT: case OP2_SUB_INT:
    case OP2_MAX_INT: case OP2_MIN_INT: case OP2_MAX_UINT:
    case OP2_MIN_UINT: case OP2_SETE_INT: case OP2_SETGT_INT:
    case OP2_SETGE_INT: case OP2_SETNE_INT: case OP2_SETGT_UINT:
    case OP2_SETGE_UINT: case OP2_KILLGT_UINT: case OP2_KILLGE_UINT:
    case OP2_PRED_SETE_INT: case OP2_PRED_SETGT_INT:
    case OP2_PRED_SETGE_INT: case OP2_PRED_SETNE_INT:
    case OP2_KILLE_INT: case OP2_KILLGT_INT: case OP2_KILLGE_INT:
    case OP2_KILLNE_INT: case OP2_FLT_TO_INT: case OP2_EXP_IEEE:
    case OP2_LOG_CLAMPED: case OP2_LOG_IEEE: case OP2_RECIP_CLAMPED:
    case OP2_RECIP_FF: case OP2_RECIP_IEEE: case OP2_RECIPSQRT_CLAMPED:
    case OP2_RECIPSQRT_FF: case OP2_RECIPSQRT_IEEE: case OP2_SQRT_IEEE:
    case OP2_SIN: case OP2_COS: case OP2_MULLO_INT: case OP2_MULHI_INT:
    case OP2_MULLO_UINT: case OP2_MULHI_UINT: case OP2_FLT_TO_UINT:
    case OP2_INT_TO_FLT: case OP2_UINT_TO_FLT: case OP2_DOT4:
    case OP2_FLT_TO_INT_RPI: case OP2_FLT_TO_INT_FLOOR: case OP2_DOT4_IEEE: case OP2_MAX4: case OP2_MOVA_INT:
    case OP2_INTERP_XY: case OP2_INTERP_ZW: case OP2_INTERP_X:
    case OP2_INTERP_Z: case OP2_INTERP_LOAD_P0:
        return true;
    default:
        return false;
    }
}

static bool op3_known(unsigned op)
{
    switch (op) {
    case OP3_BFE_UINT: case OP3_BFE_INT: case OP3_BFI_INT: case OP3_FMA:
    case OP3_MULADD: case OP3_MULADD_M2: case OP3_MULADD_M4:
    case OP3_MULADD_D2: case OP3_MULADD_IEEE: case OP3_CNDE:
    case OP3_CNDGT: case OP3_CNDGE: case OP3_CNDE_INT: case OP3_CNDGT_INT:
    case OP3_CNDGE_INT:
        return true;
    default:
        return false;
    }
}

static unsigned op2_nsrc(unsigned op)
{
    switch (op) {
    case OP2_FRACT: case OP2_TRUNC: case OP2_CEIL: case OP2_RNDNE:
    case OP2_FLOOR: case OP2_MOV: case OP2_NOT_INT: case OP2_FLT_TO_INT:
    case OP2_EXP_IEEE: case OP2_LOG_CLAMPED: case OP2_LOG_IEEE:
    case OP2_RECIP_CLAMPED: case OP2_RECIP_FF: case OP2_RECIP_IEEE:
    case OP2_RECIPSQRT_CLAMPED: case OP2_RECIPSQRT_FF:
    case OP2_RECIPSQRT_IEEE: case OP2_SQRT_IEEE: case OP2_SIN: case OP2_COS:
    case OP2_FLT_TO_UINT: case OP2_INT_TO_FLT: case OP2_UINT_TO_FLT:
    case OP2_MOVA_INT: case OP2_MAX4: case OP2_INTERP_LOAD_P0:
    case OP2_FLT_TO_INT_RPI: case OP2_FLT_TO_INT_FLOOR:
        return 1;
    case OP2_NOP:
        return 0;
    default:
        return 2;
    }
}

static bool sel_ok(unsigned sel)
{
    return sel < 128 || (sel >= SEL_KCACHE0 && sel < 192) ||
           (sel >= SEL_0 && sel <= SEL_PS) ||
           (sel >= SEL_CFILE && sel < 512);
}

static void gap_mark(HD5770Prog *p)
{
    p->bad = true;
}

static bool decode_alu(HD5770Prog *p, CfInst *cf, HD5770GfxStats *st,
                       uint32_t *cap)
{
    uint32_t addr = (cf->w0 & 0x3fffff) * 2;
    uint32_t count = ((cf->w1 >> 18) & 0x7f) + 1;
    uint32_t end = addr + count * 2, pos = addr;

    if (end > p->ndw) {
        st->gap_misc[HD5770_GAP_DECODE]++;
        gap_mark(p);
        return false;
    }
    cf->group0 = p->ngroups;
    while (pos < end) {
        AluGroup g;
        bool last = false, slot_used[5] = { false };
        unsigned maxlit = 0, i;
        bool uses_lit = false;

        memset(&g, 0, sizeof(g));
        while (!last && pos + 2 <= end && g.n < 5) {
            uint32_t a = p->code[pos], b = p->code[pos + 1];
            AluInst *in = &g.inst[g.n++];
            unsigned s;

            pos += 2;
            last = a >> 31;
            in->sel[0] = a & 0x1ff;
            in->rel[0] = (a >> 9) & 1;
            in->chan[0] = (a >> 10) & 3;
            in->neg[0] = (a >> 12) & 1;
            in->sel[1] = (a >> 13) & 0x1ff;
            in->rel[1] = (a >> 22) & 1;
            in->chan[1] = (a >> 23) & 3;
            in->neg[1] = (a >> 25) & 1;
            in->index_mode = (a >> 26) & 7;
            in->pred_sel = (a >> 29) & 3;
            in->dst_gpr = (b >> 21) & 0x7f;
            in->dst_rel = (b >> 28) & 1;
            in->dst_chan = (b >> 29) & 3;
            in->clamp = b >> 31;
            if ((b >> 15) & 7) {
                unsigned op = (b >> 13) & 0x1f;

                in->op = 0x800 | op;
                in->sel[2] = b & 0x1ff;
                in->rel[2] = (b >> 9) & 1;
                in->chan[2] = (b >> 10) & 3;
                in->neg[2] = (b >> 12) & 1;
                in->write = true;
                in->nsrc = 3;
                if (!op3_known(op)) {
                    st->gap_alu_op3[op]++;
                    gap_mark(p);
                }
            } else {
                unsigned op = (b >> 7) & 0x7ff;

                in->op = op;
                in->abs_[0] = b & 1;
                in->abs_[1] = (b >> 1) & 1;
                in->update_exec = (b >> 2) & 1;
                in->update_pred = (b >> 3) & 1;
                in->write = (b >> 4) & 1;
                in->omod = (b >> 5) & 3;
                in->nsrc = op < 0x100 ? op2_nsrc(op) : 2;
                if (op >= 0x100 || !op2_known(op)) {
                    st->gap_alu_op2[op & 0xff]++;
                    gap_mark(p);
                }
            }
            for (s = 0; s < in->nsrc; s++) {
                unsigned sel = in->sel[s];

                if (sel == SEL_LITERAL) {
                    uses_lit = true;
                    maxlit = MAX(maxlit, in->chan[s] + 1u);
                } else if (!sel_ok(sel) &&
                           !(sel >= SEL_PARAM && sel < SEL_PARAM + 32)) {
                    st->gap_src_sel++;
                    gap_mark(p);
                }
                if (in->rel[s] && in->index_mode > 4) {
                    st->gap_misc[HD5770_GAP_REL]++;
                    gap_mark(p);
                }
            }
            if (in->dst_rel && in->index_mode > 4) {
                st->gap_misc[HD5770_GAP_REL]++;
                gap_mark(p);
            }
            if (in->pred_sel) {
                st->gap_misc[HD5770_GAP_PRED]++;
            }
            /* slot: trans-only ops and a taken vector slot go to t */
            if ((!(in->op & 0x800) && op2_trans_only(in->op)) ||
                slot_used[in->dst_chan]) {
                in->slot = 4;
            } else {
                in->slot = in->dst_chan;
            }
            if (slot_used[in->slot]) {
                st->gap_misc[HD5770_GAP_DECODE]++;
                gap_mark(p);
            }
            slot_used[in->slot] = true;
        }
        if (uses_lit) {
            unsigned nl = maxlit > 2 ? 4 : 2;

            for (i = 0; i < nl && pos < end; i++) {
                g.lit[i] = p->code[pos++];
            }
        }
        if (p->ngroups >= *cap) {
            *cap *= 2;
            p->groups = g_renew(AluGroup, p->groups, *cap);
        }
        p->groups[p->ngroups++] = g;
    }
    cf->ngroups = p->ngroups - cf->group0;
    return true;
}

static bool cf_known(unsigned op)
{
    switch (op) {
    case CF_NOP: case CF_TEX: case CF_VTX: case CF_LOOP_START:
    case CF_LOOP_END: case CF_LOOP_START_DX10: case CF_LOOP_START_NO_AL:
    case CF_LOOP_CONTINUE: case CF_LOOP_BREAK: case CF_JUMP: case CF_PUSH:
    case CF_ELSE: case CF_POP: case CF_CALL: case CF_CALL_FS:
    case CF_RETURN: case CF_KILL: case CF_END_PROGRAM: case CF_WAIT_ACK:
    case CF_TEX_ACK: case CF_VTX_ACK: case CF_EXPORT: case CF_EXPORT_DONE:
        return true;
    default:
        return false;
    }
}

static void decode_fetch(HD5770Prog *p, CfInst *cf, HD5770GfxStats *st)
{
    uint32_t addr = (cf->w0 & 0xffffff) * 2;
    uint32_t count = ((cf->w1 >> 10) & 0x3f) + 1, i;

    if (addr + count * 4 > p->ndw) {
        st->gap_misc[HD5770_GAP_DECODE]++;
        gap_mark(p);
        return;
    }
    for (i = 0; i < count; i++) {
        const uint32_t *w = &p->code[addr + i * 4];
        unsigned op = w[0] & 0x1f;

        if (cf->op == CF_VTX || op == 0) {
            /* vertex fetch, in a VC or a TC clause */
            if (op != 0 || ((w[0] >> 5) & 3) == 3) {
                st->gap_vtx[op]++;
                gap_mark(p);
            }
        } else {
            switch (op) {
            case 0x07:              /* GET_GRADIENTS_H */
            case 0x08:              /* GET_GRADIENTS_V */
            case 0x03:              /* LD */
            case 0x10:              /* SAMPLE */
            case 0x11:              /* SAMPLE_L */
            case 0x12:              /* SAMPLE_LB */
            case 0x13:              /* SAMPLE_LZ */
                break;
            default:
                st->gap_tex[op]++;
                gap_mark(p);
                break;
            }
        }
    }
}

uint32_t hd5770_prog_size(const uint32_t *code, uint32_t max)
{
    uint32_t i, end = 0, first = max;

    /* CF instructions run up to end of program or the first clause */
    for (i = 0; i + 1 < max && i < first && i < MAX_CF * 2; i += 2) {
        uint32_t w0 = code[i], w1 = code[i + 1];
        uint32_t cend = i + 2;

        if ((w1 >> 29) & 1) {
            uint32_t a = (w0 & 0x3fffff) * 2;

            cend = a + (((w1 >> 18) & 0x7f) + 1) * 2;
            first = MIN(first, a);
        } else {
            unsigned op = (w1 >> 22) & 0xff;

            if (op == CF_TEX || op == CF_VTX) {
                uint32_t a = (w0 & 0xffffff) * 2;

                cend = a + (((w1 >> 10) & 0x3f) + 1) * 4;
                first = MIN(first, a);
            }
        }
        end = MAX(end, cend);
        if (!((w1 >> 29) & 1) && ((w1 >> 21) & 1)) {
            break;
        }
    }
    return MIN(end, max);
}

HD5770Prog *hd5770_prog_new(const uint32_t *code, uint32_t ndw,
                            unsigned stage, HD5770GfxStats *st)
{
    HD5770Prog *p = g_new0(HD5770Prog, 1);
    uint32_t i, cap = 16, first = ndw;

    p->code = g_memdup2(code, ndw * 4);
    p->ndw = ndw;
    p->stage = stage;
    p->groups = g_new(AluGroup, cap);
    for (i = 0; i + 1 < ndw && p->ncf < MAX_CF && i < first; i += 2) {
        CfInst *cf = &p->cf[p->ncf++];
        bool eop;

        cf->w0 = code[i];
        cf->w1 = code[i + 1];
        if ((cf->w1 >> 29) & 1) {
            first = MIN(first, (cf->w0 & 0x3fffff) * 2);
        } else if (((cf->w1 >> 22) & 0xff) == CF_TEX ||
                   ((cf->w1 >> 22) & 0xff) == CF_VTX) {
            first = MIN(first, (cf->w0 & 0xffffff) * 2);
        }
        if ((cf->w1 >> 29) & 1) {
            unsigned op = (cf->w1 >> 26) & 0xf;

            cf->op = 0x80 | op;
            if (op == CF_ALU_EXTENDED || op == CF_ALU_CONTINUE ||
                op == CF_ALU_BREAK) {
                st->gap_cf[cf->op]++;
                gap_mark(p);
            }
            if (((cf->w0 >> 30) & 3) > 2 || ((cf->w1 & 3) > 2)) {
                st->gap_misc[HD5770_GAP_KCACHE]++;
                gap_mark(p);
            }
            decode_alu(p, cf, st, &cap);
            continue;
        }
        cf->op = (cf->w1 >> 22) & 0xff;
        eop = (cf->w1 >> 21) & 1;
        if (!cf_known(cf->op)) {
            st->gap_cf[cf->op]++;
            gap_mark(p);
        } else if (cf->op == CF_TEX || cf->op == CF_VTX) {
            decode_fetch(p, cf, st);
        } else if (cf->op == CF_EXPORT || cf->op == CF_EXPORT_DONE) {
            unsigned type = (cf->w0 >> 13) & 3;

            if (type == 3 || ((cf->w0 >> 30) & 3)) {
                st->gap_export[type]++;
                gap_mark(p);
            }
        }
        if (eop) {
            break;
        }
        if (cf->op == CF_RETURN && stage == HD5770_STAGE_FS) {
            break;
        }
    }
    return p;
}

void hd5770_prog_free(HD5770Prog *p)
{
    if (p) {
        g_free(p->code);
        g_free(p->groups);
        g_free(p);
    }
}

bool hd5770_prog_bad(const HD5770Prog *p)
{
    return p->bad;
}

/* ---------------------------------------------------------------- */
/* Interpreter                                                      */

typedef struct Run {
    const HD5770Prog *p;
    const HD5770ShEnv *env;
    HD5770Thread *t;
    HD5770Val pv[4], ps;
    int32_t ar[4];
    int32_t loop_idx;
    bool active, pred;
    bool stack[STACK_DEPTH];
    unsigned sp;
    /* kcache windows of the current ALU clause */
    unsigned kc_bank[2], kc_addr[2], kc_mode[2];
    bool ok;
    /* stop point of a neighbour run */
    bool stop, stopped;
    uint32_t stop_cf, stop_inst;
    unsigned stop_gpr;
    HD5770Val *stop_out;
    uint32_t pc;
} Run;

static float fclamp01(float v)
{
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : (v != v ? 0.0f : v));
}

static HD5770Val fval(float f)
{
    HD5770Val v;

    v.f = f;
    return v;
}

static HD5770Val uval(uint32_t u)
{
    HD5770Val v;

    v.u = u;
    return v;
}

static HD5770Val read_src(Run *r, const AluInst *in, unsigned s,
                          const AluGroup *g)
{
    unsigned sel = in->sel[s], chan = in->chan[s];
    HD5770Val v;
    int32_t idx;

    if (sel < 128) {
        idx = sel;
        if (in->rel[s]) {
            idx += in->index_mode == 4 ? r->loop_idx : r->ar[in->index_mode & 3];
        }
        v = r->t->r[idx & (HD5770_GPRS - 1)][chan];
    } else if (sel < 192) {
        unsigned k = sel >= SEL_KCACHE1, i = sel - (k ? SEL_KCACHE1 : SEL_KCACHE0);
        HD5770Val c[4];

        idx = r->kc_addr[k] * 16 + i;
        if (in->rel[s]) {
            idx += in->index_mode == 4 ? r->loop_idx : r->ar[in->index_mode & 3];
        }
        r->env->kcache(r->env->opaque, r->kc_bank[k], idx, c);
        v = c[chan];
    } else if (sel >= SEL_CFILE && sel < 512) {
        idx = sel - SEL_CFILE;
        if (in->rel[s]) {
            idx += in->index_mode == 4 ? r->loop_idx : r->ar[in->index_mode & 3];
        }
        v.u = r->env->cfile ? r->env->cfile[(idx & 255) * 4 + chan] : 0;
    } else {
        switch (sel) {
        case SEL_0:
            v.f = 0.0f;
            break;
        case SEL_1:
            v.f = 1.0f;
            break;
        case SEL_1_INT:
            v.i = 1;
            break;
        case SEL_M_1_INT:
            v.i = -1;
            break;
        case SEL_0_5:
            v.f = 0.5f;
            break;
        case SEL_LITERAL:
            v.u = g->lit[chan];
            break;
        case SEL_PV:
            v = r->pv[chan];
            break;
        case SEL_PS:
            v = r->ps;
            break;
        default:
            v.u = 0;
            break;
        }
    }
    return v;
}

static float fmod_src(const AluInst *in, unsigned s, HD5770Val v)
{
    float f = v.f;

    if (in->abs_[s]) {
        f = fabsf(f);
    }
    if (in->neg[s]) {
        f = -f;
    }
    return f;
}

/* DX9 multiply: 0 times anything is 0 */
static float mul_dx9(float a, float b)
{
    if (a == 0.0f || b == 0.0f) {
        return 0.0f;
    }
    return a * b;
}

static float max_dx9(float a, float b)
{
    return a >= b ? a : (b != b ? a : b);
}

static float min_dx9(float a, float b)
{
    return a < b ? a : (b != b ? a : b);
}

static float rndne(float f)
{
    return rintf(f);
}

/*
 * Evaluate one instruction. dot/interp group operations are finished
 * by the caller; *pred_out is the predicate an instruction sets, *kill
 * true when it kills the pixel.
 */
static HD5770Val eval(Run *r, const AluInst *in, const AluGroup *g,
                      HD5770Val s[3], bool *pred_out, bool *has_pred,
                      bool *kill)
{
    float a = 0, b = 0, c = 0;
    HD5770Val v;
    unsigned op = in->op;

    v.u = 0;
    if (op & 0x800) {
        a = s[0].f;
        b = s[1].f;
        c = s[2].f;
        if (in->neg[0]) {
            a = -a;
        }
        if (in->neg[1]) {
            b = -b;
        }
        if (in->neg[2]) {
            c = -c;
        }
        switch (op & 0x1f) {
        case OP3_MULADD:
            return fval(mul_dx9(a, b) + c);
        case OP3_MULADD_M2:
            return fval((mul_dx9(a, b) + c) * 2.0f);
        case OP3_MULADD_M4:
            return fval((mul_dx9(a, b) + c) * 4.0f);
        case OP3_MULADD_D2:
            return fval((mul_dx9(a, b) + c) * 0.5f);
        case OP3_MULADD_IEEE:
            return fval(a * b + c);
        case OP3_FMA:
            return fval(fmaf(a, b, c));
        case OP3_CNDE:
            return a == 0.0f ? (in->neg[1] ? fval(b) : s[1])
                             : (in->neg[2] ? fval(c) : s[2]);
        case OP3_CNDGT:
            return a > 0.0f ? (in->neg[1] ? fval(b) : s[1])
                            : (in->neg[2] ? fval(c) : s[2]);
        case OP3_CNDGE:
            return a >= 0.0f ? (in->neg[1] ? fval(b) : s[1])
                             : (in->neg[2] ? fval(c) : s[2]);
        case OP3_CNDE_INT:
            return s[0].i == 0 ? s[1] : s[2];
        case OP3_CNDGT_INT:
            return s[0].i > 0 ? s[1] : s[2];
        case OP3_CNDGE_INT:
            return s[0].i >= 0 ? s[1] : s[2];
        case OP3_BFE_UINT: {
            unsigned off = s[1].u & 31, w = s[2].u & 31;

            return uval(w ? (s[0].u >> off) & ((1u << w) - 1) : 0);
        }
        case OP3_BFE_INT: {
            unsigned off = s[1].u & 31, w = s[2].u & 31;

            if (!w) {
                return uval(0);
            }
            v.i = (int32_t)(s[0].u << (32 - off - w)) >> (32 - w);
            return v;
        }
        case OP3_BFI_INT:
            return uval((s[0].u & s[1].u) | (~s[0].u & s[2].u));
        default:
            return v;
        }
    }
    if (in->nsrc >= 1) {
        a = fmod_src(in, 0, s[0]);
    }
    if (in->nsrc >= 2) {
        b = fmod_src(in, 1, s[1]);
    }
    switch (op) {
    case OP2_ADD:
        return fval(a + b);
    case OP2_MUL:
        return fval(mul_dx9(a, b));
    case OP2_MUL_IEEE:
        return fval(a * b);
    case OP2_MAX:
        return fval(max_dx9(a, b));
    case OP2_MIN:
        return fval(min_dx9(a, b));
    case OP2_MAX_DX10:
        return fval(fmaxf(a, b));
    case OP2_MIN_DX10:
        return fval(fminf(a, b));
    case OP2_SETE:
        return fval(a == b ? 1.0f : 0.0f);
    case OP2_SETGT:
        return fval(a > b ? 1.0f : 0.0f);
    case OP2_SETGE:
        return fval(a >= b ? 1.0f : 0.0f);
    case OP2_SETNE:
        return fval(a != b ? 1.0f : 0.0f);
    case OP2_SETE_DX10:
        return uval(a == b ? ~0u : 0);
    case OP2_SETGT_DX10:
        return uval(a > b ? ~0u : 0);
    case OP2_SETGE_DX10:
        return uval(a >= b ? ~0u : 0);
    case OP2_SETNE_DX10:
        return uval(a != b ? ~0u : 0);
    case OP2_FRACT:
        return fval(a - floorf(a));
    case OP2_TRUNC:
        return fval(truncf(a));
    case OP2_CEIL:
        return fval(ceilf(a));
    case OP2_RNDNE:
        return fval(rndne(a));
    case OP2_FLOOR:
        return fval(floorf(a));
    case OP2_ASHR_INT:
        v.i = s[0].i >> (s[1].u & 31);
        return v;
    case OP2_LSHR_INT:
        return uval(s[0].u >> (s[1].u & 31));
    case OP2_LSHL_INT:
        return uval(s[0].u << (s[1].u & 31));
    case OP2_MOV:
        return in->abs_[0] || in->neg[0] ? fval(a) : s[0];
    case OP2_NOP:
        return v;
    case OP2_PRED_SETE:
        *has_pred = true;
        *pred_out = a == b;
        return fval(*pred_out ? 0.0f : 1.0f);
    case OP2_PRED_SETGT:
        *has_pred = true;
        *pred_out = a > b;
        return fval(*pred_out ? 0.0f : 1.0f);
    case OP2_PRED_SETGE:
        *has_pred = true;
        *pred_out = a >= b;
        return fval(*pred_out ? 0.0f : 1.0f);
    case OP2_PRED_SETNE:
        *has_pred = true;
        *pred_out = a != b;
        return fval(*pred_out ? 0.0f : 1.0f);
    case OP2_PRED_SETGT_UINT:
        *has_pred = true;
        *pred_out = s[0].u > s[1].u;
        return fval(*pred_out ? 0.0f : 1.0f);
    case OP2_PRED_SETGE_UINT:
        *has_pred = true;
        *pred_out = s[0].u >= s[1].u;
        return fval(*pred_out ? 0.0f : 1.0f);
    case OP2_PRED_SETE_INT:
        *has_pred = true;
        *pred_out = s[0].i == s[1].i;
        return uval(*pred_out ? 0 : 1);
    case OP2_PRED_SETGT_INT:
        *has_pred = true;
        *pred_out = s[0].i > s[1].i;
        return uval(*pred_out ? 0 : 1);
    case OP2_PRED_SETGE_INT:
        *has_pred = true;
        *pred_out = s[0].i >= s[1].i;
        return uval(*pred_out ? 0 : 1);
    case OP2_PRED_SETNE_INT:
        *has_pred = true;
        *pred_out = s[0].i != s[1].i;
        return uval(*pred_out ? 0 : 1);
    case OP2_KILLE:
        *kill = a == b;
        return fval(*kill ? 1.0f : 0.0f);
    case OP2_KILLGT:
        *kill = a > b;
        return fval(*kill ? 1.0f : 0.0f);
    case OP2_KILLGE:
        *kill = a >= b;
        return fval(*kill ? 1.0f : 0.0f);
    case OP2_KILLNE:
        *kill = a != b;
        return fval(*kill ? 1.0f : 0.0f);
    case OP2_KILLGT_UINT:
        *kill = s[0].u > s[1].u;
        return uval(*kill);
    case OP2_KILLGE_UINT:
        *kill = s[0].u >= s[1].u;
        return uval(*kill);
    case OP2_KILLE_INT:
        *kill = s[0].i == s[1].i;
        return uval(*kill);
    case OP2_KILLGT_INT:
        *kill = s[0].i > s[1].i;
        return uval(*kill);
    case OP2_KILLGE_INT:
        *kill = s[0].i >= s[1].i;
        return uval(*kill);
    case OP2_KILLNE_INT:
        *kill = s[0].i != s[1].i;
        return uval(*kill);
    case OP2_AND_INT:
        return uval(s[0].u & s[1].u);
    case OP2_OR_INT:
        return uval(s[0].u | s[1].u);
    case OP2_XOR_INT:
        return uval(s[0].u ^ s[1].u);
    case OP2_NOT_INT:
        return uval(~s[0].u);
    case OP2_ADD_INT:
        return uval(s[0].u + s[1].u);
    case OP2_SUB_INT:
        return uval(s[0].u - s[1].u);
    case OP2_MAX_INT:
        return s[0].i > s[1].i ? s[0] : s[1];
    case OP2_MIN_INT:
        return s[0].i < s[1].i ? s[0] : s[1];
    case OP2_MAX_UINT:
        return s[0].u > s[1].u ? s[0] : s[1];
    case OP2_MIN_UINT:
        return s[0].u < s[1].u ? s[0] : s[1];
    case OP2_SETE_INT:
        return uval(s[0].i == s[1].i ? ~0u : 0);
    case OP2_SETGT_INT:
        return uval(s[0].i > s[1].i ? ~0u : 0);
    case OP2_SETGE_INT:
        return uval(s[0].i >= s[1].i ? ~0u : 0);
    case OP2_SETNE_INT:
        return uval(s[0].i != s[1].i ? ~0u : 0);
    case OP2_SETGT_UINT:
        return uval(s[0].u > s[1].u ? ~0u : 0);
    case OP2_SETGE_UINT:
        return uval(s[0].u >= s[1].u ? ~0u : 0);
    case OP2_FLT_TO_INT:
        v.i = a != a ? 0 : (a >= 2147483647.0f ? INT32_MAX :
              (a <= -2147483648.0f ? INT32_MIN : (int32_t)truncf(a)));
        return v;
    case OP2_FLT_TO_INT_FLOOR:
        a = floorf(a);
        v.i = a != a ? 0 : (a >= 2147483647.0f ? INT32_MAX :
              (a <= -2147483648.0f ? INT32_MIN : (int32_t)a));
        return v;
    case OP2_FLT_TO_INT_RPI:
        a = floorf(a + 0.5f);
        v.i = a != a ? 0 : (a >= 2147483647.0f ? INT32_MAX :
              (a <= -2147483648.0f ? INT32_MIN : (int32_t)a));
        return v;
    case OP2_FLT_TO_UINT:
        return uval(a != a || a <= 0.0f ? 0 :
                    (a >= 4294967295.0f ? UINT32_MAX : (uint32_t)truncf(a)));
    case OP2_INT_TO_FLT:
        return fval((float)s[0].i);
    case OP2_UINT_TO_FLT:
        return fval((float)s[0].u);
    case OP2_EXP_IEEE:
        return fval(exp2f(a));
    case OP2_LOG_CLAMPED:
        return fval(a == 0.0f ? -FLT_MAX : log2f(a));
    case OP2_LOG_IEEE:
        return fval(log2f(a));
    case OP2_RECIP_CLAMPED:
        return fval(a == 0.0f ? FLT_MAX : 1.0f / a);
    case OP2_RECIP_FF:
        return fval(a == 0.0f ? 0.0f : 1.0f / a);
    case OP2_RECIP_IEEE:
        return fval(1.0f / a);
    case OP2_RECIPSQRT_CLAMPED:
        return fval(a == 0.0f ? FLT_MAX : 1.0f / sqrtf(a));
    case OP2_RECIPSQRT_FF:
        return fval(a == 0.0f ? 0.0f : 1.0f / sqrtf(a));
    case OP2_RECIPSQRT_IEEE:
        return fval(1.0f / sqrtf(a));
    case OP2_SQRT_IEEE:
        return fval(sqrtf(a));
    case OP2_SIN:
        return fval(sinf(a * 2.0f * 3.14159265358979f));
    case OP2_COS:
        return fval(cosf(a * 2.0f * 3.14159265358979f));
    case OP2_MULLO_INT:
    case OP2_MULLO_UINT:
        return uval(s[0].u * s[1].u);
    case OP2_MULHI_INT:
        return uval(((int64_t)s[0].i * s[1].i) >> 32);
    case OP2_MULHI_UINT:
        return uval(((uint64_t)s[0].u * s[1].u) >> 32);
    case OP2_MOVA_INT:
        return s[0];
    default:
        return v;
    }
}

static HD5770Val post(const AluInst *in, HD5770Val v)
{
    unsigned op = in->op;
    bool is_float = (op & 0x800) ? (op & 0x1f) >= OP3_FMA &&
                                   (op & 0x1f) <= OP3_CNDGE
                                 : true;

    if (!(op & 0x800)) {
        switch (op) {
        case OP2_ASHR_INT: case OP2_LSHR_INT: case OP2_LSHL_INT:
        case OP2_AND_INT: case OP2_OR_INT: case OP2_XOR_INT:
        case OP2_NOT_INT: case OP2_ADD_INT: case OP2_SUB_INT:
        case OP2_MAX_INT: case OP2_MIN_INT: case OP2_MAX_UINT:
        case OP2_MIN_UINT: case OP2_SETE_INT: case OP2_SETGT_INT:
        case OP2_SETGE_INT: case OP2_SETNE_INT: case OP2_SETGT_UINT:
        case OP2_SETGE_UINT: case OP2_FLT_TO_INT: case OP2_FLT_TO_UINT:
        case OP2_FLT_TO_INT_RPI: case OP2_FLT_TO_INT_FLOOR:
        case OP2_MULLO_INT: case OP2_MULHI_INT: case OP2_MULLO_UINT:
        case OP2_MULHI_UINT: case OP2_MOVA_INT: case OP2_SETE_DX10:
        case OP2_SETGT_DX10: case OP2_SETGE_DX10: case OP2_SETNE_DX10:
        case OP2_PRED_SETE_INT: case OP2_PRED_SETGT_INT:
        case OP2_PRED_SETGE_INT: case OP2_PRED_SETNE_INT:
        case OP2_KILLGT_UINT: case OP2_KILLGE_UINT: case OP2_KILLE_INT:
        case OP2_KILLGT_INT: case OP2_KILLGE_INT: case OP2_KILLNE_INT:
            is_float = false;
            break;
        case OP2_MOV:
            is_float = in->omod || in->clamp || in->abs_[0] || in->neg[0];
            break;
        default:
            break;
        }
    }
    if (!is_float) {
        return v;
    }
    switch (in->omod) {
    case 1:
        v.f *= 2.0f;
        break;
    case 2:
        v.f *= 4.0f;
        break;
    case 3:
        v.f *= 0.5f;
        break;
    default:
        break;
    }
    if (in->clamp) {
        v.f = fclamp01(v.f);
    }
    return v;
}

static void run_group(Run *r, const AluGroup *g)
{
    HD5770Val res[5], src[5][3];
    bool wr[5] = { false }, any_pred = false, new_pred = false;
    float dot = 0.0f;
    bool has_dot = false;
    unsigned i, k;

    for (i = 0; i < g->n; i++) {
        const AluInst *in = &g->inst[i];

        for (k = 0; k < in->nsrc; k++) {
            src[i][k] = read_src(r, in, k, g);
        }
    }
    for (i = 0; i < g->n; i++) {
        const AluInst *in = &g->inst[i];
        bool hp = false, pr = false, kill = false;
        unsigned op = in->op;

        if (op == OP2_DOT4 || op == OP2_DOT4_IEEE) {
            float a = fmod_src(in, 0, src[i][0]), b = fmod_src(in, 1, src[i][1]);

            dot += op == OP2_DOT4 ? mul_dx9(a, b) : a * b;
            has_dot = true;
            res[i].u = 0;
            continue;
        }
        if (op == OP2_INTERP_XY || op == OP2_INTERP_ZW ||
            op == OP2_INTERP_X || op == OP2_INTERP_Z ||
            op == OP2_INTERP_LOAD_P0) {
            unsigned prm = 0;

            for (k = 0; k < in->nsrc; k++) {
                if (in->sel[k] >= SEL_PARAM && in->sel[k] < SEL_PARAM + 32) {
                    prm = in->sel[k] - SEL_PARAM;
                }
            }
            res[i].f = r->env->interp
                ? r->env->interp(r->env->opaque, prm, in->dst_chan,
                                 op == OP2_INTERP_LOAD_P0)
                : 0.0f;
            continue;
        }
        if (op == OP2_MAX4) {
            res[i] = src[i][0];
            continue;
        }
        res[i] = eval(r, in, g, src[i], &pr, &hp, &kill);
        if (kill && r->active) {
            r->t->killed = true;
        }
        if (hp) {
            any_pred = true;
            new_pred = pr;
            if (in->update_exec && r->active) {
                r->active = pr;
            }
        }
        if (op == OP2_MOVA_INT && r->active) {
            r->ar[in->dst_chan] = src[i][0].i;
        }
    }
    if (has_dot) {
        for (i = 0; i < g->n; i++) {
            if (g->inst[i].op == OP2_DOT4 || g->inst[i].op == OP2_DOT4_IEEE) {
                res[i].f = dot;
            }
        }
    }
    {
        float m = -FLT_MAX;
        bool has_max4 = false;

        for (i = 0; i < g->n; i++) {
            if (g->inst[i].op == OP2_MAX4) {
                m = max_dx9(m, fmod_src(&g->inst[i], 0, src[i][0]));
                has_max4 = true;
            }
        }
        if (has_max4) {
            for (i = 0; i < g->n; i++) {
                if (g->inst[i].op == OP2_MAX4) {
                    res[i].f = m;
                }
            }
        }
    }
    for (i = 0; i < g->n; i++) {
        res[i] = post(&g->inst[i], res[i]);
        wr[i] = g->inst[i].write && g->inst[i].op != OP2_NOP;
    }
    /* results reach PV/PS and the registers together */
    for (i = 0; i < g->n; i++) {
        const AluInst *in = &g->inst[i];

        if (in->slot == 4) {
            r->ps = res[i];
        } else {
            r->pv[in->slot] = res[i];
        }
    }
    if (r->active) {
        for (i = 0; i < g->n; i++) {
            const AluInst *in = &g->inst[i];
            int32_t idx = in->dst_gpr;

            if (!wr[i]) {
                continue;
            }
            if (in->pred_sel == 2 && !r->pred) {
                continue;
            }
            if (in->pred_sel == 1 && r->pred) {
                continue;
            }
            if (in->dst_rel) {
                idx += in->index_mode == 4 ? r->loop_idx
                                           : r->ar[in->index_mode & 3];
            }
            r->t->r[idx & (HD5770_GPRS - 1)][in->dst_chan] = res[i];
        }
        if (any_pred) {
            r->pred = new_pred;
        }
    }
}

static void run_fetch(Run *r, const CfInst *cf)
{
    const HD5770Prog *p = r->p;
    uint32_t addr = (cf->w0 & 0xffffff) * 2;
    uint32_t count = ((cf->w1 >> 10) & 0x3f) + 1, i, k;

    if (!r->active) {
        return;
    }
    for (i = 0; i < count && addr + i * 4 + 3 <= p->ndw; i++) {
        const uint32_t *w = &p->code[addr + i * 4];
        HD5770Val out[4], src[4];
        unsigned dst = w[1] & 0x7f;
        unsigned sg = (w[0] >> 16) & 0x7f;
        unsigned op = w[0] & 0x1f;

        if (r->stop && r->pc == r->stop_cf && i == r->stop_inst) {
            memcpy(r->stop_out, r->t->r[r->stop_gpr], 4 * sizeof(HD5770Val));
            r->stopped = true;
            return;
        }
        if (cf->op == CF_TEX && (op == 0x07 || op == 0x08)) {
            /* derivative across the quad: neighbour minus this pixel */
            HD5770Val nb[4];

            if (r->stop) {
                /* a neighbour run on its way to its stop point */
                memcpy(nb, r->t->r[sg], sizeof(nb));
            } else if (!r->env->neighbour ||
                       !r->env->neighbour(r->env->opaque, op == 0x07,
                                          op == 0x08, r->pc, i, sg, nb)) {
                r->ok = false;
                return;
            }
            for (k = 0; k < 4; k++) {
                unsigned sel = (w[2] >> (20 + 3 * k)) & 7;

                out[k].f = sel < 4 ? nb[sel].f - r->t->r[sg][sel].f : 0.0f;
            }
            for (k = 0; k < 4; k++) {
                unsigned sel = (w[1] >> (9 + 3 * k)) & 7;

                if (sel < 4) {
                    r->t->r[dst][k] = out[sel];
                } else if (sel == 4) {
                    r->t->r[dst][k].f = 0.0f;
                } else if (sel == 5) {
                    r->t->r[dst][k].f = 1.0f;
                }
            }
            continue;
        }

        if (cf->op == CF_VTX || (w[0] & 0x1f) == 0) {
            unsigned ssel = (w[0] >> 24) & 3;
            uint32_t index = r->t->r[sg][ssel].u;

            r->env->vfetch(r->env->opaque, w, index, p->stage, out);
            for (k = 0; k < 4; k++) {
                unsigned sel = (w[1] >> (9 + 3 * k)) & 7;

                if (sel < 4) {
                    r->t->r[dst][k] = out[sel];
                } else if (sel == 4) {
                    r->t->r[dst][k].f = 0.0f;
                } else if (sel == 5) {
                    r->t->r[dst][k].f = 1.0f;
                }
            }
        } else {
            for (k = 0; k < 4; k++) {
                unsigned sel = (w[2] >> (20 + 3 * k)) & 7;

                src[k] = sel < 4 ? r->t->r[sg][sel]
                                 : fval(sel == 5 ? 1.0f : 0.0f);
            }
            r->env->tfetch(r->env->opaque, w, src, r->t, out);
            for (k = 0; k < 4; k++) {
                unsigned sel = (w[1] >> (9 + 3 * k)) & 7;

                if (sel < 4) {
                    r->t->r[dst][k] = out[sel];
                } else if (sel == 4) {
                    r->t->r[dst][k].f = 0.0f;
                } else if (sel == 5) {
                    r->t->r[dst][k].f = 1.0f;
                }
            }
        }
    }
}

static void run_export(Run *r, const CfInst *cf)
{
    unsigned type = (cf->w0 >> 13) & 3, base = cf->w0 & 0x1fff;
    unsigned gpr = (cf->w0 >> 15) & 0x7f, burst = (cf->w1 >> 16) & 0xf;
    unsigned i, k;

    if (!r->active) {
        return;
    }
    for (i = 0; i <= burst; i++) {
        HD5770Val v[4];
        uint32_t mask = 0;

        for (k = 0; k < 4; k++) {
            unsigned sel = (cf->w1 >> (3 * k)) & 7;

            if (sel < 4) {
                v[k] = r->t->r[(gpr + i) & 127][sel];
            } else if (sel == 4) {
                v[k].f = 0.0f;
            } else if (sel == 5) {
                v[k].f = 1.0f;
            } else {
                v[k].u = 0;
                continue;
            }
            mask |= 1u << k;
        }
        if (r->p->stage == HD5770_STAGE_PS) {
            if (type == 0 && base + i < HD5770_MRTS) {
                memcpy(r->t->color[base + i], v, sizeof(v));
                r->t->color_mask |= 1u << (base + i);
            } else if (type == 0 && base + i == 61) {
                r->t->depth = v[0];
                r->t->depth_written = true;
            }
        } else if (type == 1) {
            if (base + i == 60) {
                for (k = 0; k < 4; k++) {
                    if (mask & (1u << k)) {
                        r->t->pos[k] = v[k];
                    }
                }
                r->t->pos_written = true;
            }
        } else if (type == 2 && base + i < HD5770_PARAMS) {
            for (k = 0; k < 4; k++) {
                if (mask & (1u << k)) {
                    r->t->param[base + i][k] = v[k];
                }
            }
            r->t->param_mask |= 1u << (base + i);
        }
    }
}

static void push(Run *r)
{
    if (r->sp < STACK_DEPTH) {
        r->stack[r->sp++] = r->active;
    } else {
        r->ok = false;
    }
}

static void pop(Run *r, unsigned n)
{
    while (n--) {
        if (!r->sp) {
            r->ok = false;
            return;
        }
        r->active = r->stack[--r->sp];
    }
}

static bool cond_true(Run *r, uint32_t w1)
{
    unsigned cond = (w1 >> 8) & 3, cst = (w1 >> 3) & 0x1f;
    bool b = r->env->bool_const &&
             ((r->env->bool_const[cst / 32] >> (cst % 32)) & 1);

    switch (cond) {
    case 0:
        return r->active;
    case 1:
        return false;
    case 2:
        return r->active && b;
    default:
        return r->active && !b;
    }
}

static bool run_prog(Run *r)
{
    const HD5770Prog *p = r->p;
    uint32_t pc = 0, steps = 0, ret[CALL_DEPTH];
    unsigned rsp = 0;
    struct {
        uint32_t start, end, count;
        int32_t idx, inc;
        bool dx10;
        unsigned sp;
    } loops[LOOP_DEPTH];
    unsigned lsp = 0;

    while (pc < p->ncf && steps++ < STEP_LIMIT && r->ok && !r->stopped) {
        const CfInst *cf = &p->cf[pc];
        bool eop = !(cf->op & 0x80) && ((cf->w1 >> 21) & 1);
        uint32_t addr = cf->w0 & 0xffffff;
        unsigned popn = cf->w1 & 7;

        pc++;
        if (cf->op & 0x80) {
            unsigned op = cf->op & 0xf, g;

            r->kc_bank[0] = (cf->w0 >> 22) & 0xf;
            r->kc_bank[1] = (cf->w0 >> 26) & 0xf;
            r->kc_mode[0] = (cf->w0 >> 30) & 3;
            r->kc_mode[1] = cf->w1 & 3;
            r->kc_addr[0] = (cf->w1 >> 2) & 0xff;
            r->kc_addr[1] = (cf->w1 >> 10) & 0xff;
            if (op == CF_ALU_PUSH_BEFORE) {
                push(r);
            }
            for (g = 0; g < cf->ngroups; g++) {
                run_group(r, &p->groups[cf->group0 + g]);
            }
            if (op == CF_ALU_POP_AFTER) {
                pop(r, 1);
            } else if (op == CF_ALU_POP2_AFTER) {
                pop(r, 2);
            } else if (op == CF_ALU_ELSE_AFTER) {
                r->active = (r->sp ? r->stack[r->sp - 1] : true) && !r->active;
            }
            continue;
        }
        switch (cf->op) {
        case CF_NOP:
        case CF_WAIT_ACK:
        case CF_TEX_ACK:
        case CF_VTX_ACK:
            break;
        case CF_TEX:
        case CF_VTX:
            r->pc = pc - 1;
            run_fetch(r, cf);
            break;
        case CF_EXPORT:
        case CF_EXPORT_DONE:
            run_export(r, cf);
            break;
        case CF_PUSH:
            push(r);
            if (!cond_true(r, cf->w1)) {
                /* nothing would run: pop and jump */
                pop(r, popn ? popn : 1);
                pc = addr;
            }
            break;
        case CF_JUMP:
            if (!cond_true(r, cf->w1)) {
                pop(r, popn);
                pc = addr;
            }
            break;
        case CF_ELSE: {
            bool before = r->sp ? r->stack[r->sp - 1] : true;

            if (popn) {
                r->ok = false;
            }
            r->active = before && !r->active;
            if (!r->active) {
                pc = addr;
            }
            break;
        }
        case CF_POP:
            pop(r, popn);
            break;
        case CF_CALL:
            if (rsp < CALL_DEPTH && cond_true(r, cf->w1)) {
                ret[rsp++] = pc;
                pc = addr;
            }
            break;
        case CF_RETURN:
            if (!rsp) {
                return r->ok;
            }
            pc = ret[--rsp];
            break;
        case CF_CALL_FS:
            if (r->env->fetch) {
                Run sub = *r;

                sub.p = r->env->fetch;
                sub.sp = 0;
                run_prog(&sub);
                r->ok = sub.ok;
            }
            break;
        case CF_LOOP_START:
        case CF_LOOP_START_DX10:
        case CF_LOOP_START_NO_AL: {
            unsigned cst = (cf->w1 >> 3) & 0x1f;
            uint32_t lc = r->env->loop_const ? r->env->loop_const[cst] : 0;

            if (lsp >= LOOP_DEPTH) {
                r->ok = false;
                break;
            }
            loops[lsp].start = pc;
            loops[lsp].end = addr;
            loops[lsp].dx10 = cf->op == CF_LOOP_START_DX10;
            loops[lsp].count = lc & 0xfff;
            loops[lsp].idx = (lc >> 12) & 0xfff;
            loops[lsp].inc = (int8_t)(lc >> 24);
            loops[lsp].sp = r->sp;
            push(r);
            if (!loops[lsp].dx10 && !loops[lsp].count) {
                pop(r, 1);
                pc = addr;
                break;
            }
            if (cf->op != CF_LOOP_START_NO_AL) {
                r->loop_idx = loops[lsp].idx;
            }
            lsp++;
            break;
        }
        case CF_LOOP_END:
            if (!lsp) {
                r->ok = false;
                break;
            }
            {
                typeof(loops[0]) *l = &loops[lsp - 1];

                if (r->active && (l->dx10 || --l->count)) {
                    l->idx += l->inc;
                    r->loop_idx = l->idx;
                    pc = l->start;
                } else {
                    r->sp = l->sp + 1;
                    pop(r, 1);
                    lsp--;
                }
            }
            break;
        case CF_LOOP_BREAK:
            if (lsp && cond_true(r, cf->w1)) {
                r->active = false;
            }
            break;
        case CF_LOOP_CONTINUE:
            if (lsp && cond_true(r, cf->w1)) {
                pc = loops[lsp - 1].end - 1;
            }
            break;
        case CF_KILL:
            if (cond_true(r, cf->w1)) {
                r->t->killed = true;
            }
            break;
        case CF_END_PROGRAM:
            return r->ok;
        default:
            r->ok = false;
            break;
        }
        if (eop) {
            break;
        }
        if (cf->op == CF_RETURN && p->stage == HD5770_STAGE_FS) {
            break;
        }
    }
    if (steps >= STEP_LIMIT) {
        r->ok = false;
    }
    return r->ok;
}

bool hd5770_prog_run(const HD5770Prog *p, const HD5770ShEnv *env,
                     HD5770Thread *t)
{
    Run r;

    memset(&r, 0, sizeof(r));
    r.p = p;
    r.env = env;
    r.t = t;
    r.active = true;
    r.ok = !p->bad;
    if (!r.ok) {
        return false;
    }
    return run_prog(&r);
}

bool hd5770_prog_run_until(const HD5770Prog *p, const HD5770ShEnv *env,
                           HD5770Thread *t, uint32_t cf, uint32_t inst,
                           unsigned gpr, HD5770Val out[4])
{
    Run r;

    memset(&r, 0, sizeof(r));
    r.p = p;
    r.env = env;
    r.t = t;
    r.active = true;
    r.ok = !p->bad;
    r.stop = true;
    r.stop_cf = cf;
    r.stop_inst = inst;
    r.stop_gpr = gpr;
    r.stop_out = out;
    if (!r.ok) {
        return false;
    }
    run_prog(&r);
    return r.stopped;
}

/* ---------------------------------------------------------------- */
/* Census and listing                                               */

void hd5770_prog_census(const HD5770Prog *p, HD5770ShaderInfo *si)
{
    uint32_t i, g, k;

    si->dwords = p->ndw;
    si->bad = p->bad;
    si->n_alu_ops = si->n_cf_ops = 0;
    si->tex_ops = si->vtx_ops = 0;
    for (i = 0; i < p->ncf; i++) {
        const CfInst *cf = &p->cf[i];

        for (k = 0; k < si->n_cf_ops && si->cf_ops[k] != cf->op; k++) {
            /* search */
        }
        if (k == si->n_cf_ops && k < ARRAY_SIZE(si->cf_ops)) {
            si->cf_ops[si->n_cf_ops++] = cf->op;
        }
        if (cf->op == CF_TEX) {
            si->tex_ops += ((cf->w1 >> 10) & 0x3f) + 1;
        } else if (cf->op == CF_VTX) {
            si->vtx_ops += ((cf->w1 >> 10) & 0x3f) + 1;
        }
        if (!(cf->op & 0x80)) {
            continue;
        }
        for (g = 0; g < cf->ngroups; g++) {
            const AluGroup *gr = &p->groups[cf->group0 + g];
            unsigned n;

            for (n = 0; n < gr->n; n++) {
                uint16_t op = gr->inst[n].op;

                for (k = 0; k < si->n_alu_ops && si->alu_ops[k] != op; k++) {
                    /* search */
                }
                if (k == si->n_alu_ops && k < ARRAY_SIZE(si->alu_ops)) {
                    si->alu_ops[si->n_alu_ops++] = op;
                }
            }
        }
    }
}

void hd5770_prog_dump(const HD5770Prog *p, GString *out)
{
    uint32_t i, g, n;

    for (i = 0; i < p->ncf; i++) {
        const CfInst *cf = &p->cf[i];

        g_string_append_printf(out, "  %02u %08x %08x cf 0x%02x", i,
                               cf->w0, cf->w1, cf->op);
        if (cf->op & 0x80) {
            g_string_append_printf(out, " kc0 %u@%u kc1 %u@%u\n",
                                   (cf->w0 >> 22) & 0xf, (cf->w1 >> 2) & 0xff,
                                   (cf->w0 >> 26) & 0xf,
                                   (cf->w1 >> 10) & 0xff);
            for (g = 0; g < cf->ngroups; g++) {
                const AluGroup *gr = &p->groups[cf->group0 + g];

                for (n = 0; n < gr->n; n++) {
                    const AluInst *in = &gr->inst[n];

                    g_string_append_printf(out,
                        "     %c %s%02x r%u.%c%s <- %u.%c%s %u.%c%s %u.%c%s%s\n",
                        "xyzwt"[in->slot], in->op & 0x800 ? "op3:" : "op2:",
                        in->op & 0x7ff, in->dst_gpr, "xyzw"[in->dst_chan],
                        in->write ? "" : "(nw)",
                        in->sel[0], "xyzw"[in->chan[0]],
                        in->neg[0] ? "-" : (in->abs_[0] ? "|" : ""),
                        in->sel[1], "xyzw"[in->chan[1]],
                        in->neg[1] ? "-" : (in->abs_[1] ? "|" : ""),
                        in->sel[2], "xyzw"[in->chan[2]],
                        in->neg[2] ? "-" : "", in->clamp ? " sat" : "");
                }
                if (gr->lit[0] || gr->lit[1] || gr->lit[2] || gr->lit[3]) {
                    g_string_append_printf(out, "       lit %08x %08x %08x %08x\n",
                                           gr->lit[0], gr->lit[1], gr->lit[2],
                                           gr->lit[3]);
                }
            }
        } else {
            g_string_append_c(out, '\n');
            if (cf->op == CF_TEX || cf->op == CF_VTX) {
                uint32_t a = (cf->w0 & 0xffffff) * 2;
                uint32_t c = ((cf->w1 >> 10) & 0x3f) + 1, k;

                for (k = 0; k < c && a + k * 4 + 3 <= p->ndw; k++) {
                    g_string_append_printf(out, "     %08x %08x %08x\n",
                                           p->code[a + k * 4],
                                           p->code[a + k * 4 + 1],
                                           p->code[a + k * 4 + 2]);
                }
            }
        }
    }
}
