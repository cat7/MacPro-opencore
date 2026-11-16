/*
 * ATI Radeon HD 5770 (Juniper) shader programs: R800 (Evergreen) bytecode
 * decoder and interpreter.
 *
 * A program is a list of 64-bit control flow (CF) instructions; ALU
 * clauses hold VLIW groups of up to five instructions (slots x, y, z, w
 * and the transcendental t) followed by their literals, fetch clauses
 * hold 128-bit TEX or VTX instructions. The interpreter runs one
 * invocation at a time: the execution mask is one bit, the CF stack holds
 * its saved values. Pixel shaders without flow control also have a lane
 * form that runs up to HD5770_LANES pixels per instruction.
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
    OP2_DOT4_IEEE = 0xbf, OP2_CUBE = 0xc0, OP2_MAX4 = 0xc1, OP2_MOVA_INT = 0xcc,
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

typedef struct LaneProg LaneProg;

struct HD5770Prog {
    LaneProg *lane;             /* lane form, or NULL */
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
    case OP2_CUBE:
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
            case 0x04:              /* GET_TEXTURE_RESINFO */
            case 0x10:              /* SAMPLE */
            case 0x11:              /* SAMPLE_L */
            case 0x12:              /* SAMPLE_LB */
            case 0x13:              /* SAMPLE_LZ */
            case 0x18:              /* SAMPLE_C */
            case 0x19:              /* SAMPLE_C_L */
            case 0x1a:              /* SAMPLE_C_LB */
            case 0x1b:              /* SAMPLE_C_LZ */
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

        if ((w1 >> 29) & 1 && ((w1 >> 26) & 0xf) == CF_ALU_EXTENDED) {
            /* the extension words of the next ALU instruction */
        } else if ((w1 >> 29) & 1) {
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

static void lane_free(LaneProg *lp);
static LaneProg *lane_compile(const HD5770Prog *p, bool cfm);

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
        if ((cf->w1 >> 29) & 1 && ((cf->w1 >> 26) & 0xf) == CF_ALU_EXTENDED) {
            /* kcache sets 2 and 3 of the ALU instruction that follows */
            cf->op = 0x80 | CF_ALU_EXTENDED;
            if ((cf->w0 >> 4) & 0xff) {
                /* indexed constant buffers */
                st->gap_cf[cf->op]++;
                gap_mark(p);
            }
            continue;
        }
        if ((cf->w1 >> 29) & 1) {
            first = MIN(first, (cf->w0 & 0x3fffff) * 2);
        } else if (((cf->w1 >> 22) & 0xff) == CF_TEX ||
                   ((cf->w1 >> 22) & 0xff) == CF_VTX) {
            first = MIN(first, (cf->w0 & 0xffffff) * 2);
        }
        if ((cf->w1 >> 29) & 1) {
            unsigned op = (cf->w1 >> 26) & 0xf;

            cf->op = 0x80 | op;
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
    if (stage == HD5770_STAGE_PS && !p->bad) {
        p->lane = lane_compile(p, false);
        if (!p->lane) {
            p->lane = lane_compile(p, true);
        }
    }
    return p;
}

void hd5770_prog_free(HD5770Prog *p)
{
    if (p) {
        lane_free(p->lane);
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
    unsigned kc_bank[4], kc_addr[4], kc_mode[4];
    bool kc_ext, kc_ext_next;   /* the clause uses kcache sets 2 and 3 */
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
    } else if (r->kc_ext && sel >= 256 && sel < 320) {
        /* kcache sets 2 and 3 of an ALU_EXTENDED clause */
        unsigned k = sel >= 288 ? 3 : 2, i = sel - (k == 3 ? 288 : 256);
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
        /* clamped to [-256, 255], to AR and the GPR */
        v.i = MIN(MAX(s[0].i, -256), 255);
        return v;
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

/*
 * CUBE in slots x..w, sources Rn.zzxy and Rn.yxzz: x = T, y = S,
 * z = 2 * major axis, w = face (+X -X +Y -Y +Z -Z), as in OpenGL's cube
 * map face selection (sc, tc, ma).
 */
/* the results of CUBE in slots x, y, z, w for the vector v */
static void cube_sel(const float v[3], float out[4])
{
    float ax = fabsf(v[0]), ay = fabsf(v[1]), az = fabsf(v[2]);
    float sc, tc, ma, face;

    if (ax >= ay && ax >= az) {
        ma = v[0];
        face = v[0] >= 0.0f ? 0.0f : 1.0f;
        sc = v[0] >= 0.0f ? -v[2] : v[2];
        tc = -v[1];
    } else if (ay >= az) {
        ma = v[1];
        face = v[1] >= 0.0f ? 2.0f : 3.0f;
        sc = v[0];
        tc = v[1] >= 0.0f ? v[2] : -v[2];
    } else {
        ma = v[2];
        face = v[2] >= 0.0f ? 4.0f : 5.0f;
        sc = v[2] >= 0.0f ? v[0] : -v[0];
        tc = -v[1];
    }
    out[0] = tc;
    out[1] = sc;
    out[2] = 2.0f * ma;
    out[3] = face;
}

static void cube(const AluGroup *g, HD5770Val src[5][3], HD5770Val res[5])
{
    float v[3] = { 0.0f, 0.0f, 0.0f }, out[4];
    bool any = false;
    unsigned i;

    for (i = 0; i < g->n; i++) {
        const AluInst *in = &g->inst[i];

        if (in->op != OP2_CUBE) {
            continue;
        }
        any = true;
        if (in->slot == 2) {
            v[0] = fmod_src(in, 0, src[i][0]);
        } else if (in->slot == 3) {
            v[1] = fmod_src(in, 0, src[i][0]);
        } else if (in->slot == 0) {
            v[2] = fmod_src(in, 0, src[i][0]);
        }
    }
    if (!any) {
        return;
    }
    cube_sel(v, out);
    for (i = 0; i < g->n; i++) {
        if (g->inst[i].op == OP2_CUBE) {
            res[i].f = out[MIN(g->inst[i].slot, 3)];
        }
    }
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
        if (op == OP2_MAX4 || op == OP2_CUBE) {
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
            r->ar[in->dst_chan] = res[i].i;
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
    cube(g, src, res);
    for (i = 0; i < g->n; i++) {
        if (g->inst[i].op != OP2_CUBE || g->inst[i].slot != 3) {
            /* output modifiers do not touch CUBE's face id */
            res[i] = post(&g->inst[i], res[i]);
        }
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
            if (r->env->tfetch_d && op != 0x03) {
                /* derivatives from the neighbours to the right and below */
                HD5770Val nx[4], ny[4];
                float dv[8] = { 0 };

                if (!r->stop && r->env->neighbour &&
                    r->env->neighbour(r->env->opaque, 1, 0, r->pc, i, sg, nx) &&
                    r->env->neighbour(r->env->opaque, 0, 1, r->pc, i, sg, ny)) {
                    for (k = 0; k < 4; k++) {
                        unsigned sel = (w[2] >> (20 + 3 * k)) & 7;

                        if (sel < 4) {
                            dv[k] = nx[sel].f - src[k].f;
                            dv[4 + k] = ny[sel].f - src[k].f;
                        }
                    }
                }
                r->env->tfetch_d(r->env->opaque, w, src, dv, out);
            } else {
                r->env->tfetch(r->env->opaque, w, src, r->t, out);
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
                r->t->stencil = v[1];
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
        if (cf->op == (0x80 | CF_ALU_EXTENDED)) {
            r->kc_bank[2] = (cf->w0 >> 22) & 0xf;
            r->kc_bank[3] = (cf->w0 >> 26) & 0xf;
            r->kc_mode[2] = (cf->w0 >> 30) & 3;
            r->kc_mode[3] = cf->w1 & 3;
            r->kc_addr[2] = (cf->w1 >> 2) & 0xff;
            r->kc_addr[3] = (cf->w1 >> 10) & 0xff;
            r->kc_ext_next = true;
            continue;
        }
        if (cf->op & 0x80) {
            unsigned op = cf->op & 0xf, g;

            r->kc_ext = r->kc_ext_next;
            r->kc_ext_next = false;
            r->kc_bank[0] = (cf->w0 >> 22) & 0xf;
            r->kc_bank[1] = (cf->w0 >> 26) & 0xf;
            r->kc_mode[0] = (cf->w0 >> 30) & 3;
            r->kc_mode[1] = cf->w1 & 3;
            r->kc_addr[0] = (cf->w1 >> 2) & 0xff;
            r->kc_addr[1] = (cf->w1 >> 10) & 0xff;
            if (op == CF_ALU_PUSH_BEFORE || op == CF_ALU_BREAK ||
                op == CF_ALU_CONTINUE) {
                push(r);
            }
            for (g = 0; g < cf->ngroups; g++) {
                run_group(r, &p->groups[cf->group0 + g]);
            }
            if (op == CF_ALU_BREAK || op == CF_ALU_CONTINUE) {
                /*
                 * PUSH, ALU, ELSE, BREAK (CONTINUE), POP: a pixel the
                 * clause's PRED_SET left inactive leaves the loop (this
                 * iteration): straight to its LOOP_END
                 */
                bool before = r->sp ? r->stack[r->sp - 1] : true;
                bool hit = before && !r->active;

                pop(r, 1);
                if (hit && lsp) {
                    r->sp = loops[lsp - 1].sp + 1;
                    r->active = op == CF_ALU_CONTINUE;
                    pc = loops[lsp - 1].end - 1;
                }
                continue;
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
/* Lanes                                                            */

/* the IR of a program, built at decode time */
struct LaneProg {
    HD5770IR ir;
};

static void lane_free(LaneProg *lp)
{
    if (lp) {
        g_free(lp->ir.ops);
        g_free(lp->ir.groups);
        g_free(lp->ir.insts);
        g_free(lp->ir.fetches);
        g_free(lp->ir.exports);
        g_free(lp->ir.consts);
        g_free(lp);
    }
}

static bool lane_gpr(HD5770IR *ir, unsigned gpr, uint8_t *out)
{
    gpr &= HD5770_GPRS - 1;
    if (ir->map[gpr] == 0xff) {
        if (ir->ngpr >= (ir->cf ? HD5770_GPRS : HD5770_IR_GPRS)) {
            return false;
        }
        ir->map[gpr] = ir->ngpr++;
    }
    *out = ir->map[gpr];
    return true;
}

static bool op_lane_ok(const AluInst *in)
{
    if (in->pred_sel || in->update_exec || in->update_pred ||
        in->dst_rel || in->rel[0] || in->rel[1] || in->rel[2]) {
        return false;
    }
    if (in->op & 0x800) {
        return true;
    }
    switch (in->op) {
    case OP2_MOVA_INT:
    case OP2_PRED_SETE: case OP2_PRED_SETGT: case OP2_PRED_SETGE:
    case OP2_PRED_SETNE: case OP2_PRED_SETGT_UINT: case OP2_PRED_SETGE_UINT:
    case OP2_PRED_SETE_INT: case OP2_PRED_SETGT_INT:
    case OP2_PRED_SETGE_INT: case OP2_PRED_SETNE_INT:
        return false;
    default:
        return true;
    }
}

static bool op_kills(unsigned op)
{
    return (op >= OP2_KILLE && op <= OP2_KILLNE) ||
           op == OP2_KILLGT_UINT || op == OP2_KILLGE_UINT ||
           (op >= OP2_KILLE_INT && op <= OP2_KILLNE_INT);
}

static bool op_is_interp(unsigned op)
{
    return op == OP2_INTERP_XY || op == OP2_INTERP_ZW ||
           op == OP2_INTERP_X || op == OP2_INTERP_Z ||
           op == OP2_INTERP_LOAD_P0;
}

/*
 * Operand k of an instruction of the ALU clause cf; ext: the clause's
 * ALU_EXTENDED word pair (kcache sets 2 and 3), or NULL.
 */
static bool lane_src(HD5770IR *ir, const CfInst *cf, const CfInst *ext,
                     const AluInst *in, const AluGroup *g, unsigned k,
                     HD5770IRSrc *ls, unsigned *cap)
{
    unsigned sel = in->sel[k], chan = in->chan[k];

    memset(ls, 0, sizeof(*ls));
    ls->chan = chan;
    ls->neg = in->neg[k];
    ls->abs = in->abs_[k];
    if (sel < 128) {
        ls->kind = HD5770_IRS_GPR;
        ls->rel = in->rel[k];
        return lane_gpr(ir, sel, &ls->idx);
    }
    if ((sel >= SEL_KCACHE0 && sel < 192) || (sel >= SEL_CFILE && sel < 512)) {
        HD5770IRConst *c;

        if (in->rel[k]) {
            /* indexed constants are read on the CPU */
            return false;
        }
        if (ir->nconsts >= *cap) {
            *cap *= 2;
            ir->consts = g_renew(HD5770IRConst, ir->consts, *cap);
        }
        c = &ir->consts[ir->nconsts];
        memset(c, 0, sizeof(*c));
        c->chan = chan;
        if (sel < 192) {
            unsigned kk = sel >= SEL_KCACHE1;

            c->bank = kk ? (cf->w0 >> 26) & 0xf : (cf->w0 >> 22) & 0xf;
            c->idx = (kk ? (cf->w1 >> 10) & 0xff : (cf->w1 >> 2) & 0xff) * 16 +
                     sel - (kk ? SEL_KCACHE1 : SEL_KCACHE0);
        } else if (ext && sel < 320) {
            /* kcache sets 2 (256..287) and 3 (288..319) */
            unsigned kk = sel >= 288;

            c->bank = kk ? (ext->w0 >> 26) & 0xf : (ext->w0 >> 22) & 0xf;
            c->idx = (kk ? (ext->w1 >> 10) & 0xff : (ext->w1 >> 2) & 0xff) *
                     16 + sel - (kk ? 288 : 256);
        } else {
            c->cfile = true;
            c->idx = sel - SEL_CFILE;
        }
        ls->kind = HD5770_IRS_CONST;
        ls->k = ir->nconsts++;
        return true;
    }
    ls->kind = HD5770_IRS_IMM;
    switch (sel) {
    case SEL_0:
        ls->imm = fval(0.0f).u;
        break;
    case SEL_1:
        ls->imm = fval(1.0f).u;
        break;
    case SEL_1_INT:
        ls->imm = 1;
        break;
    case SEL_M_1_INT:
        ls->imm = (uint32_t)-1;
        break;
    case SEL_0_5:
        ls->imm = fval(0.5f).u;
        break;
    case SEL_LITERAL:
        ls->imm = g->lit[chan];
        break;
    case SEL_PV:
        ls->kind = HD5770_IRS_PV;
        break;
    case SEL_PS:
        ls->kind = HD5770_IRS_PV;
        ls->chan = 4;
        break;
    default:
        ls->imm = 0;
        break;
    }
    return true;
}

/* the PV/PS value of slot s of group gi may be read by a later group */
static bool lane_pv_read(const HD5770IR *ir, unsigned gi, unsigned s)
{
    unsigned h, i, k;

    for (h = gi + 1; h < ir->ngroups; h++) {
        const HD5770IRGroup *g = &ir->groups[h];
        bool has = false;

        for (i = 0; i < g->n; i++) {
            const HD5770IRInst *li = &ir->insts[g->first + i];

            for (k = 0; k < li->nsrc; k++) {
                if (li->src[k].kind == HD5770_IRS_PV && li->src[k].chan == s) {
                    return true;
                }
            }
            has |= li->slot == s;
        }
        if (has) {
            return false;
        }
    }
    return false;
}

/* a GPR the program addresses relatively */
static bool prog_rel(const HD5770Prog *p)
{
    unsigned g, i;

    for (g = 0; g < p->ngroups; g++) {
        for (i = 0; i < p->groups[g].n; i++) {
            const AluInst *in = &p->groups[g].inst[i];

            if (in->dst_rel || (in->rel[0] && in->sel[0] < 128) ||
                (in->rel[1] && in->sel[1] < 128) ||
                (in->rel[2] && in->sel[2] < 128 && in->nsrc > 2)) {
                return true;
            }
        }
    }
    return false;
}

/*
 * The IR of a pixel shader: straight-line (cfm false; fails on flow
 * control, predicates, relative addressing, more than HD5770_IR_GPRS
 * registers), or the CF form.
 */
static LaneProg *lane_compile(const HD5770Prog *p, bool cfm)
{
    LaneProg *lp = g_new0(LaneProg, 1);
    HD5770IR *ir = &lp->ir;
    unsigned ccap = 16, pc, gi, i, k, nfetch = 0, nexp = 0;
    uint64_t h = 0xcbf29ce484222325ull;
    const CfInst *ext = NULL;
    uint8_t dummy;

    memset(ir->map, 0xff, sizeof(ir->map));
    ir->cf = cfm;
    for (pc = 0; pc < p->ncf; pc++) {
        if (p->cf[pc].op == CF_TEX || p->cf[pc].op == CF_VTX) {
            nfetch += ((p->cf[pc].w1 >> 10) & 0x3f) + 1;
        }
        if (p->cf[pc].op == CF_EXPORT || p->cf[pc].op == CF_EXPORT_DONE) {
            nexp += ((p->cf[pc].w1 >> 16) & 0xf) + 1;
        }
    }
    ir->ops = g_new0(HD5770IROp, p->ncf + 1);
    ir->groups = g_new0(HD5770IRGroup, p->ngroups + 1);
    ir->insts = g_new0(HD5770IRInst, p->ngroups * 5 + 1);
    ir->fetches = g_new0(HD5770IRFetch, nfetch + 1);
    ir->exports = g_new0(HD5770IRExport, nexp + 1);
    ir->consts = g_new0(HD5770IRConst, ccap);
    for (i = 0; i < p->ndw; i++) {
        h = (h ^ p->code[i]) * 0x100000001b3ull;
    }
    ir->hash = h;
    if (cfm && prog_rel(p)) {
        /* relative addresses reach any register */
        ir->rel = true;
        for (i = 0; i < HD5770_GPRS; i++) {
            ir->map[i] = i;
        }
        ir->ngpr = HD5770_GPRS;
    }
    /* r0 holds the barycentrics */
    lane_gpr(ir, 0, &dummy);
    for (pc = 0; pc < p->ncf; pc++) {
        const CfInst *cf = &p->cf[pc];
        HD5770IROp *o = &ir->ops[ir->nops];

        if (cfm) {
            o->kind = HD5770_IRO_CF;
            o->cf = cf->op;
            o->addr = MIN(cf->w0 & 0xffffff, 0xffffu);
            o->popn = cf->w1 & 7;
            o->cond = (cf->w1 >> 8) & 3;
            o->cst = (cf->w1 >> 3) & 0x1f;
            o->eop = !(cf->op & 0x80) && ((cf->w1 >> 21) & 1);
        }
        if (cf->op == (0x80 | CF_ALU_EXTENDED)) {
            /* the kcache sets 2 and 3 of the clause that follows */
            ext = cf;
            if (cfm) {
                ir->nops++;
            }
            continue;
        }
        if (cf->op & 0x80) {
            unsigned g;

            if (!cfm && (cf->op & 0xf) != CF_ALU) {
                goto fail;
            }
            o->kind = HD5770_IRO_ALU;
            o->first = ir->ngroups;
            o->n = cf->ngroups;
            for (g = 0; g < cf->ngroups; g++) {
                const AluGroup *ag = &p->groups[cf->group0 + g];
                HD5770IRGroup *lg = &ir->groups[ir->ngroups++];

                lg->first = ir->ninsts;
                lg->n = ag->n;
                for (i = 0; i < ag->n; i++) {
                    const AluInst *in = &ag->inst[i];
                    HD5770IRInst *li = &ir->insts[ir->ninsts++];

                    if (!cfm && !op_lane_ok(in)) {
                        goto fail;
                    }
                    li->priv = in;
                    li->op = in->op;
                    li->slot = in->slot;
                    li->dst_chan = in->dst_chan;
                    li->nsrc = in->nsrc;
                    li->omod = in->omod;
                    li->clamp = in->clamp;
                    li->interp = !(in->op & 0x800) && op_is_interp(in->op);
                    li->flat = in->op == OP2_INTERP_LOAD_P0;
                    li->dot = in->op == OP2_DOT4 || in->op == OP2_DOT4_IEEE;
                    li->max4 = in->op == OP2_MAX4;
                    li->cube = in->op == OP2_CUBE;
                    li->kill = !(in->op & 0x800) && op_kills(in->op);
                    li->pred_sel = in->pred_sel;
                    li->index_mode = in->index_mode;
                    li->update_exec = in->update_exec;
                    li->update_pred = in->update_pred;
                    li->dst_rel = in->dst_rel;
                    li->mova = in->op == OP2_MOVA_INT;
                    lg->dot |= li->dot;
                    lg->max4 |= li->max4;
                    lg->cube |= li->cube;
                    for (k = 0; k < in->nsrc; k++) {
                        if (li->interp && in->sel[k] >= SEL_PARAM &&
                            in->sel[k] < SEL_PARAM + 32) {
                            li->prm = in->sel[k] - SEL_PARAM;
                        }
                        if (!lane_src(ir, cf, ext, in, ag, k, &li->src[k],
                                      &ccap)) {
                            goto fail;
                        }
                    }
                    li->wr = in->write && in->op != OP2_NOP;
                    if (li->wr && !lane_gpr(ir, in->dst_gpr, &li->dst)) {
                        goto fail;
                    }
                }
            }
            ext = NULL;
            ir->nops++;
            continue;
        }
        switch (cf->op) {
        case CF_NOP:
        case CF_WAIT_ACK:
        case CF_TEX_ACK:
        case CF_VTX_ACK:
            if (cfm) {
                ir->nops++;
            }
            break;
        case CF_TEX:
        case CF_VTX: {
            uint32_t addr = (cf->w0 & 0xffffff) * 2;
            uint32_t count = ((cf->w1 >> 10) & 0x3f) + 1;

            o->kind = HD5770_IRO_FETCH;
            o->first = ir->nfetches;
            for (i = 0; i < count && addr + i * 4 + 3 <= p->ndw; i++) {
                const uint32_t *w = &p->code[addr + i * 4];
                HD5770IRFetch *f = &ir->fetches[ir->nfetches++];
                unsigned op = w[0] & 0x1f;

                memcpy(f->w, w, sizeof(f->w));
                if (!lane_gpr(ir, (w[0] >> 16) & 0x7f, &f->src) ||
                    !lane_gpr(ir, w[1] & 0x7f, &f->dst)) {
                    goto fail;
                }
                if (cf->op == CF_VTX || op == 0) {
                    f->kind = HD5770_IRF_VTX;
                } else if (op == 0x07 || op == 0x08) {
                    f->kind = op == 0x07 ? HD5770_IRF_GRAD_H
                                         : HD5770_IRF_GRAD_V;
                    ir->grad = true;
                } else {
                    f->kind = HD5770_IRF_TEX;
                    f->res = (w[0] >> 8) & 0xff;
                    for (k = 0; k < ir->nres && ir->res[k] != f->res; k++) {
                        /* search */
                    }
                    if (k == ir->nres) {
                        if (ir->nres == ARRAY_SIZE(ir->res)) {
                            goto fail;
                        }
                        ir->res[ir->nres++] = f->res;
                    }
                }
            }
            o->n = ir->nfetches - o->first;
            ir->nops++;
            break;
        }
        case CF_EXPORT:
        case CF_EXPORT_DONE: {
            unsigned type = (cf->w0 >> 13) & 3, base = cf->w0 & 0x1fff;
            unsigned gpr = (cf->w0 >> 15) & 0x7f;
            unsigned burst = (cf->w1 >> 16) & 0xf, n;

            o->kind = HD5770_IRO_EXPORT;
            o->first = ir->nexports;
            for (n = 0; n <= burst; n++) {
                HD5770IRExport *e;

                /* colour buffers 0-7, and depth/stencil */
                if (type != 0 ||
                    (base + n >= HD5770_MRTS && base + n != HD5770_IR_DEPTH)) {
                    continue;
                }
                e = &ir->exports[ir->nexports++];
                e->target = base + n;
                if (!lane_gpr(ir, (gpr + n) & 127, &e->gpr)) {
                    goto fail;
                }
                for (k = 0; k < 4; k++) {
                    e->sel[k] = (cf->w1 >> (3 * k)) & 7;
                }
                if (e->target == HD5770_IR_DEPTH) {
                    ir->zexport = true;
                } else {
                    ir->cmask |= 1u << e->target;
                }
            }
            o->n = ir->nexports - o->first;
            if (o->n || cfm) {
                ir->nops++;
            }
            break;
        }
        case CF_END_PROGRAM:
            if (cfm) {
                ir->nops++;
                break;
            }
            pc = p->ncf;
            continue;
        case CF_PUSH:
        case CF_POP:
        case CF_ELSE:
        case CF_JUMP:
        case CF_LOOP_START:
        case CF_LOOP_START_DX10:
        case CF_LOOP_START_NO_AL:
        case CF_LOOP_END:
        case CF_LOOP_BREAK:
        case CF_LOOP_CONTINUE:
        case CF_CALL:
        case CF_RETURN:
        case CF_KILL:
            if (!cfm) {
                goto fail;
            }
            ir->nops++;
            break;
        default:
            goto fail;
        }
        if (!cfm && ((cf->w1 >> 21) & 1)) {
            break;
        }
    }
    for (gi = 0; gi < ir->ngroups; gi++) {
        const HD5770IRGroup *g = &ir->groups[gi];

        for (i = 0; i < g->n; i++) {
            HD5770IRInst *li = &ir->insts[g->first + i];

            /* a CUBE group reads the sources of all its CUBEs */
            li->needed = cfm || li->wr || li->dot || li->max4 || li->kill ||
                         li->cube || lane_pv_read(ir, gi, li->slot);
        }
    }
    return lp;
fail:
    lane_free(lp);
    return NULL;
}

const HD5770IR *hd5770_prog_ir(const HD5770Prog *p)
{
    return p->lane ? &p->lane->ir : NULL;
}

bool hd5770_prog_lanes(const HD5770Prog *p, unsigned *nconst, bool *grad)
{
    if (!p->lane || p->lane->ir.cf) {
        return false;
    }
    *nconst = p->lane->ir.nconsts;
    *grad = p->lane->ir.grad;
    return true;
}

void hd5770_prog_resources(const HD5770Prog *p, uint8_t *ids, unsigned *n,
                           unsigned max)
{
    unsigned i;

    *n = 0;
    for (i = 0; p->lane && i < p->lane->ir.nres && i < max; i++) {
        ids[(*n)++] = p->lane->ir.res[i];
    }
}

void hd5770_prog_bind(const HD5770Prog *p, const HD5770ShEnv *env,
                      HD5770Val *kv)
{
    const HD5770IR *ir = &p->lane->ir;
    unsigned i;

    for (i = 0; i < ir->nconsts; i++) {
        const HD5770IRConst *c = &ir->consts[i];

        if (c->cfile) {
            kv[i].u = env->cfile ? env->cfile[(c->idx & 255) * 4 + c->chan]
                                 : 0;
        } else {
            HD5770Val v[4];

            env->kcache(env->opaque, c->bank, c->idx, v);
            kv[i] = v[c->chan];
        }
    }
}

static void lane_group(const HD5770IR *ir, const HD5770IRGroup *g,
                       const HD5770ShEnv *env, const HD5770Val *kv,
                       HD5770Lanes *l)
{
    HD5770Val res[5][HD5770_LANES], tmp[5][3][HD5770_LANES];
    const HD5770Val *src[5][3];
    float dot[HD5770_LANES];
    unsigned n = l->n, i, k, x;

    for (i = 0; i < g->n; i++) {
        const HD5770IRInst *li = &ir->insts[g->first + i];

        if (!li->needed) {
            continue;
        }
        for (k = 0; k < li->nsrc; k++) {
            const HD5770IRSrc *ls = &li->src[k];

            switch (ls->kind) {
            case HD5770_IRS_GPR:
                src[i][k] = l->r[ls->idx][ls->chan];
                break;
            case HD5770_IRS_PV:
                src[i][k] = l->pv[ls->chan];
                break;
            default:
                for (x = 0; x < n; x++) {
                    if (ls->kind == HD5770_IRS_CONST) {
                        tmp[i][k][x] = kv[ls->k];
                    } else {
                        tmp[i][k][x].u = ls->imm;
                    }
                }
                src[i][k] = tmp[i][k];
                break;
            }
        }
    }
    if (g->dot) {
        for (x = 0; x < n; x++) {
            dot[x] = 0.0f;
        }
    }
    for (i = 0; i < g->n; i++) {
        const HD5770IRInst *li = &ir->insts[g->first + i];
        const AluInst *in = li->priv;
        unsigned op = in->op;

        if (!li->needed) {
            continue;
        }
        if (li->dot) {
            for (x = 0; x < n; x++) {
                float a = fmod_src(in, 0, src[i][0][x]);
                float b = fmod_src(in, 1, src[i][1][x]);

                dot[x] += op == OP2_DOT4 ? mul_dx9(a, b) : a * b;
            }
            continue;
        }
        if (li->interp) {
            env->interp_n(env->opaque, l, li->prm, li->dst_chan, li->flat,
                          res[i]);
            continue;
        }
        if (li->max4 || li->cube) {
            continue;
        }
        for (x = 0; x < n; x++) {
            HD5770Val sv[3];
            bool hp = false, pr = false, kill = false;

            for (k = 0; k < in->nsrc; k++) {
                sv[k] = src[i][k][x];
            }
            res[i][x] = eval(NULL, in, NULL, sv, &pr, &hp, &kill);
            if (kill) {
                l->killed[x] = true;
            }
        }
    }
    if (g->dot) {
        for (i = 0; i < g->n; i++) {
            if (ir->insts[g->first + i].dot) {
                for (x = 0; x < n; x++) {
                    res[i][x].f = dot[x];
                }
            }
        }
    }
    if (g->max4) {
        for (x = 0; x < n; x++) {
            float m = -FLT_MAX;

            for (i = 0; i < g->n; i++) {
                const HD5770IRInst *li = &ir->insts[g->first + i];

                if (li->max4) {
                    m = max_dx9(m, fmod_src(li->priv, 0, src[i][0][x]));
                }
            }
            for (i = 0; i < g->n; i++) {
                if (ir->insts[g->first + i].max4) {
                    res[i][x].f = m;
                }
            }
        }
    }
    if (g->cube) {
        for (x = 0; x < n; x++) {
            float v[3] = { 0.0f, 0.0f, 0.0f }, out[4];

            for (i = 0; i < g->n; i++) {
                const HD5770IRInst *li = &ir->insts[g->first + i];

                if (li->cube && (li->slot == 2 || li->slot == 3 ||
                                 li->slot == 0)) {
                    v[li->slot == 0 ? 2 : li->slot - 2] =
                        fmod_src(li->priv, 0, src[i][0][x]);
                }
            }
            cube_sel(v, out);
            for (i = 0; i < g->n; i++) {
                const HD5770IRInst *li = &ir->insts[g->first + i];

                if (li->cube) {
                    res[i][x].f = out[MIN(li->slot, 3)];
                }
            }
        }
    }
    for (i = 0; i < g->n; i++) {
        const HD5770IRInst *li = &ir->insts[g->first + i];

        if (!li->needed) {
            continue;
        }
        if ((li->omod || li->clamp) && !(li->cube && li->slot == 3)) {
            /* output modifiers do not touch CUBE's face id */
            for (x = 0; x < n; x++) {
                res[i][x] = post(li->priv, res[i][x]);
            }
        }
        memcpy(l->pv[li->slot], res[i], n * sizeof(HD5770Val));
    }
    for (i = 0; i < g->n; i++) {
        const HD5770IRInst *li = &ir->insts[g->first + i];

        if (li->wr) {
            memcpy(l->r[li->dst][li->dst_chan], res[i],
                   n * sizeof(HD5770Val));
        }
    }
}

static void lane_dst(HD5770Val (*r)[HD5770_LANES], const uint32_t *w,
                     unsigned x, const HD5770Val out[4])
{
    unsigned k;

    for (k = 0; k < 4; k++) {
        unsigned sel = (w[1] >> (9 + 3 * k)) & 7;

        if (sel < 4) {
            r[k][x] = out[sel];
        } else if (sel == 4) {
            r[k][x].f = 0.0f;
        } else if (sel == 5) {
            r[k][x].f = 1.0f;
        }
    }
}

/*
 * Fetches on the lane sets l[0..nl-1]: l[0] the pixels, l[1] and l[2]
 * (if nl is 3) their neighbours to the right and below. Each fetch runs
 * on the pixels first: a gradient reads a neighbour as it is when it
 * reaches the instruction. In a neighbour, as in a neighbour run of the
 * interpreter, a gradient reads the pixel itself.
 */
static void lane_fetch(const HD5770Prog *p, const HD5770IROp *o,
                       const HD5770ShEnv *env, HD5770Lanes **l, unsigned nl)
{
    const HD5770IR *ir = &p->lane->ir;
    unsigned i, k, x, q;
    float dv[HD5770_LANES][8];

    for (i = 0; i < o->n; i++) {
        const HD5770IRFetch *f = &ir->fetches[o->first + i];
        const uint32_t *w = f->w;

        for (q = 0; q < nl; q++) {
            HD5770Val (*sg)[HD5770_LANES] = l[q]->r[f->src];
            HD5770Val (*dst)[HD5770_LANES] = l[q]->r[f->dst];
            HD5770Val out[4], src[4];

            if (f->kind == HD5770_IRF_GRAD_H || f->kind == HD5770_IRF_GRAD_V) {
                HD5770Lanes *nb = q ? l[q]
                                    : l[f->kind == HD5770_IRF_GRAD_H ? 1 : 2];
                HD5770Val (*ng)[HD5770_LANES] = nb->r[f->src];

                for (x = 0; x < l[q]->n; x++) {
                    for (k = 0; k < 4; k++) {
                        unsigned sel = (w[2] >> (20 + 3 * k)) & 7;

                        out[k].f = sel < 4 ? ng[sel][x].f - sg[sel][x].f : 0.0f;
                    }
                    lane_dst(dst, w, x, out);
                }
                continue;
            }
            if (f->kind == HD5770_IRF_VTX) {
                unsigned ssel = (w[0] >> 24) & 3;

                for (x = 0; x < l[q]->n; x++) {
                    env->vfetch(env->opaque, w, sg[ssel][x].u, p->stage, out);
                    lane_dst(dst, w, x, out);
                }
                continue;
            }
            if (!q && env->tfetch_d && nl == 3 && (w[0] & 0x1f) != 0x03) {
                /*
                 * derivatives across the quad, the same for the pixel and
                 * its neighbours; read before any set's result lands
                 */
                for (x = 0; x < l[0]->n; x++) {
                    for (k = 0; k < 4; k++) {
                        unsigned sel = (w[2] >> (20 + 3 * k)) & 7;

                        dv[x][k] = dv[x][4 + k] = 0.0f;
                        if (sel < 4) {
                            dv[x][k] = l[1]->r[f->src][sel][x].f - sg[sel][x].f;
                            dv[x][4 + k] = l[2]->r[f->src][sel][x].f -
                                           sg[sel][x].f;
                        }
                    }
                }
            }
            for (x = 0; x < l[q]->n; x++) {
                for (k = 0; k < 4; k++) {
                    unsigned sel = (w[2] >> (20 + 3 * k)) & 7;

                    src[k] = sel < 4 ? sg[sel][x]
                                     : fval(sel == 5 ? 1.0f : 0.0f);
                }
                if (env->tfetch_d && nl == 3 && (w[0] & 0x1f) != 0x03) {
                    env->tfetch_d(env->opaque, w, src, dv[x], out);
                } else {
                    env->tfetch(env->opaque, w, src, NULL, out);
                }
                lane_dst(dst, w, x, out);
            }
        }
    }
}

static void lane_export(const HD5770IR *ir, const HD5770IRExport *e,
                        HD5770Lanes *l)
{
    HD5770Val (*r)[HD5770_LANES] = l->r[e->gpr];
    HD5770Val (*c)[HD5770_LANES];
    unsigned k, x;

    if (e->target == HD5770_IR_DEPTH) {
        for (x = 0; x < l->n; x++) {
            HD5770Val v[2];

            for (k = 0; k < 2; k++) {
                unsigned sel = e->sel[k];

                v[k].u = sel < 4 ? r[sel][x].u : sel == 5 ? fval(1.0f).u : 0;
            }
            l->depth[x] = v[0];
            l->stencil[x] = v[1];
        }
        l->depth_written = true;
        return;
    }
    c = e->target ? l->mcolor[e->target] : l->color;
    for (k = 0; k < 4; k++) {
        unsigned sel = e->sel[k];

        for (x = 0; x < l->n; x++) {
            if (sel < 4) {
                c[k][x] = r[sel][x];
            } else if (sel == 4) {
                c[k][x].f = 0.0f;
            } else if (sel == 5) {
                c[k][x].f = 1.0f;
            } else {
                c[k][x].u = 0;
            }
        }
    }
    l->color_mask |= 1u << e->target;
    l->color0 |= !e->target;
}

static void lane_inputs(const HD5770IR *ir, HD5770Lanes *l)
{
    unsigned x, r0 = ir->map[0];

    memset(l->killed, 0, sizeof(l->killed));
    if (l->pos_gpr >= 0 && ir->map[l->pos_gpr] != 0xff) {
        unsigned q = ir->map[l->pos_gpr];

        for (x = 0; x < l->n; x++) {
            l->r[q][0][x].f = l->px[x];
            l->r[q][1][x].f = l->py[x];
            l->r[q][2][x].f = 0.0f;
            l->r[q][3][x].f = 1.0f;
        }
    }
    for (x = 0; x < l->n; x++) {
        l->r[r0][0][x].f = l->b1[x];
        l->r[r0][1][x].f = l->b2[x];
    }
}

void hd5770_prog_run_lanes(const HD5770Prog *p, const HD5770ShEnv *env,
                           const HD5770Val *kv, HD5770Lanes *l,
                           HD5770Lanes *nh, HD5770Lanes *nv)
{
    const HD5770IR *ir = &p->lane->ir;
    HD5770Lanes *set[3] = { l, nh, nv };
    unsigned i, g, q, nl = ir->grad || nh ? 3 : 1;

    l->color0 = false;
    l->color_mask = 0;
    l->depth_written = false;
    for (q = 0; q < nl; q++) {
        lane_inputs(ir, set[q]);
    }
    for (i = 0; i < ir->nops; i++) {
        const HD5770IROp *o = &ir->ops[i];

        switch (o->kind) {
        case HD5770_IRO_ALU:
            for (g = 0; g < o->n; g++) {
                for (q = 0; q < nl; q++) {
                    lane_group(ir, &ir->groups[o->first + g], env, kv, set[q]);
                }
            }
            break;
        case HD5770_IRO_FETCH:
            lane_fetch(p, o, env, set, nl);
            break;
        default:
            for (g = 0; g < o->n; g++) {
                lane_export(ir, &ir->exports[o->first + g], l);
            }
            break;
        }
    }
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
