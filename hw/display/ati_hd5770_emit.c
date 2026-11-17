/*
 * ATI Radeon HD 5770 (Juniper) pixel shader IR in its CF form
 * (ati_hd5770_ir.h) as shading language text common to MSL and GLSL.
 *
 * The program becomes two functions that run it as hd5770_prog_run()
 * does, one CF instruction at a time through a switch on the program
 * counter, with the execution mask, its stack, the predicate, the loop
 * and call stacks and the address registers as variables:
 *
 *   void hd_nb(...)   a neighbour run: the whole program, recording at
 *                     each derivative point (a gradient, or a fetch of
 *                     a mipmapped unit) the source register the first
 *                     time it gets there (rec, seen) -- what
 *                     hd5770_prog_run_until() returns for that point,
 *                     as the run is the same up to there. A gradient
 *                     reads the pixel itself, a mipmapped fetch has no
 *                     derivatives.
 *   void hd_run(...)  the pixel: kills, the colour exports (C, mask cm),
 *                     the depth export (zd, sd, zw). The neighbours to
 *                     the right and below run first (hd_nb); a
 *                     derivative point reads their recorded sources.
 *
 * The text uses only syntax both languages share and these macros,
 * which the backend defines first:
 *   HD_CTX_PARAMS, HD_CTX_ARGS   the per-draw context as parameters/args
 *   HD_K(k)                      bound constant k (uint)
 *   HD_BCONST, HD_LCONST(i)      SQ_BOOL_CONST, SQ_LOOP_CONST i
 *   HD_POSGPR                    compact register of the position, or -1
 *   HD_TEX(u, w0, w1, w2, c, dv, o)  hd_tfetch_d() of texture unit u
 *   HD_ZERO                      an int 0 the compiler cannot see through
 *   HD_VTX(w0, w1, w2, index, o)     hd_vfetch()
 *   HD_INTERP(p, ch, flat, b0, b1, b2, pw)  hd_interp()
 *   HD_OUTB(n), HD_OUTU(n), HD_OUTC(n), HD_OUTA(n, a, b), HD_OUTBA(n, a)
 *                                output parameters: bool, uint,
 *                                uint[8][4], uint[a][b], bool[a]
 * and the library functions hd_eval, hd_pred, hd_post, hd_fmod, hd_mul,
 * hd_mulieee, hd_max9, hd_cube, hd_fadd, hd_fsub, F, U, I.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "ati_hd5770_gpu.h"

/* CF opcodes, as ati_hd5770_sh.c numbers them */
enum {
    E_NOP = 0x00, E_LOOP_START = 0x04, E_LOOP_END = 0x05,
    E_LOOP_START_DX10 = 0x06, E_LOOP_START_NO_AL = 0x07,
    E_LOOP_CONTINUE = 0x08, E_LOOP_BREAK = 0x09, E_JUMP = 0x0a,
    E_PUSH = 0x0b, E_ELSE = 0x0d, E_POP = 0x0e, E_CALL = 0x12,
    E_RETURN = 0x14, E_KILL = 0x18, E_END_PROGRAM = 0x19,
    E_ALU_PUSH_BEFORE = 0x9, E_ALU_POP_AFTER = 0xa, E_ALU_POP2_AFTER = 0xb,
    E_ALU_CONTINUE = 0xd, E_ALU_BREAK = 0xe, E_ALU_ELSE_AFTER = 0xf,
};

#define E_STACK         32
#define E_LOOPS         8
#define E_CALLS         8
#define E_STEPS         (1u << 20)

static void e_cond(GString *o, const HD5770IROp *op)
{
    switch (op->cond) {
    case 0:
        g_string_append(o, "act_");
        break;
    case 1:
        g_string_append(o, "false");
        break;
    default:
        g_string_append_printf(o, "(act_ && %s((HD_BCONST >> %uu) & 1u) != 0u)",
                               op->cond == 3 ? "!" : "", op->cst);
        break;
    }
}

static void e_pop(GString *o, const char *n)
{
    g_string_append_printf(o,
        "        for (int k_ = 0; k_ < %s; k_++) {\n"
        "            if (sp > 0) {\n                sp--;\n                act_ = stk[sp];\n"
        "            }\n        }\n", n);
}

static void e_push(GString *o)
{
    g_string_append_printf(o,
        "        if (sp < %d) {\n            stk[sp] = act_;\n            sp++;\n"
        "        }\n", E_STACK);
}

/* the register index of a source or destination */
static void e_reg(GString *o, unsigned idx, bool rel, unsigned mode)
{
    if (!rel) {
        g_string_append_printf(o, "%u", idx);
    } else if (mode == 4) {
        g_string_append_printf(o, "uint(%u + lidx) & 127u", idx);
    } else {
        g_string_append_printf(o, "uint(%u + ar[%u]) & 127u", idx, mode & 3);
    }
}

static unsigned e_mods(const HD5770IRInst *in)
{
    unsigned m = 0, k;

    for (k = 0; k < in->nsrc; k++) {
        m |= (in->src[k].neg ? 1u : 0) << k;
        m |= (in->src[k].abs ? 1u : 0) << (4 + k);
    }
    return m;
}

static bool e_is_pred(unsigned op)
{
    return (op >= 0x1e && op <= 0x23) || (op >= 0x42 && op <= 0x45);
}

static void e_group(GString *o, const HD5770IR *ir, const HD5770IRGroup *g,
                    bool main)
{
    unsigned i, k;

    g_string_append(o, "        {\n");
    for (i = 0; i < g->n; i++) {
        const HD5770IRInst *in = &ir->insts[g->first + i];

        for (k = 0; k < in->nsrc; k++) {
            const HD5770IRSrc *s = &in->src[k];

            g_string_append_printf(o, "            uint s%u_%u = ", i, k);
            switch (s->kind) {
            case HD5770_IRS_GPR:
                g_string_append(o, "R[");
                e_reg(o, s->idx, s->rel, in->index_mode);
                g_string_append_printf(o, "][%u];\n", s->chan);
                break;
            case HD5770_IRS_PV:
                g_string_append_printf(o, "P[%u];\n", s->chan);
                break;
            case HD5770_IRS_CONST:
                g_string_append_printf(o, "HD_K(%uu);\n", s->k);
                break;
            default:
                g_string_append_printf(o, "0x%08xu;\n", s->imm);
                break;
            }
        }
        g_string_append_printf(o, "            uint q%u = 0u;\n", i);
    }
    g_string_append(o, "            float dt = 0.0f;\n"
                       "            bool anyp = false, newp = false;\n");
    for (i = 0; i < g->n; i++) {
        const HD5770IRInst *in = &ir->insts[g->first + i];
        unsigned m = e_mods(in);

        if (in->dot) {
            g_string_append_printf(o, "            dt = hd_fadd(dt, %s(hd_fmod(%uu, 0u, s%u_0), "
                                   "hd_fmod(%uu, 1u, s%u_1)));\n",
                                   in->op == 0xbe ? "hd_mul" : "hd_mulieee",
                                   m, i, m, i);
        } else if (in->interp) {
            g_string_append_printf(o, "            q%u = U(HD_INTERP(%uu, %uu, %s, HB0, HB1, "
                                   "HB2, HPW));\n", i, in->prm, in->dst_chan,
                                   in->flat ? "true" : "false");
        } else if (in->max4 || in->cube) {
            g_string_append_printf(o, "            q%u = s%u_0;\n", i, i);
        } else {
            g_string_append_printf(o, "            {\n                bool k_ = false, pr_ = false;\n");
            if (!(in->op & 0x800) && e_is_pred(in->op)) {
                g_string_append_printf(o, "                q%u = hd_pred(0x%xu, %uu, s%u_0, s%u_1, pr_);\n"
                                       "                anyp = true;\n"
                                       "                newp = pr_;\n", i, in->op, m,
                                       i, i);
                if (in->update_exec) {
                    g_string_append(o, "                if (act_) {\n"
                                       "                    act_ = pr_;\n"
                                       "                }\n");
                }
            } else {
                g_string_append_printf(o, "                q%u = hd_eval(0x%xu, %uu", i,
                                       in->op, m);
                for (k = 0; k < 3; k++) {
                    if (k < in->nsrc) {
                        g_string_append_printf(o, ", s%u_%u", i, k);
                    } else {
                        g_string_append(o, ", 0u");
                    }
                }
                g_string_append(o, ", k_);\n");
                if (main && in->kill) {
                    g_string_append(o, "                if (k_ && act_) {\n"
                                       "                    killed = true;\n"
                                       "                }\n");
                }
            }
            if (in->mova) {
                g_string_append_printf(o, "                if (act_) {\n"
                                       "                    ar[%u] = I(q%u);\n"
                                       "                }\n", in->dst_chan, i);
            }
            g_string_append(o, "            }\n");
        }
    }
    if (g->dot) {
        for (i = 0; i < g->n; i++) {
            if (ir->insts[g->first + i].dot) {
                g_string_append_printf(o, "            q%u = U(dt);\n", i);
            }
        }
    }
    if (g->max4) {
        g_string_append(o, "            {\n                float m_ = -FLT_MAX;\n");
        for (i = 0; i < g->n; i++) {
            const HD5770IRInst *in = &ir->insts[g->first + i];

            if (in->max4) {
                g_string_append_printf(o, "                m_ = hd_max9(m_, hd_fmod(%uu, 0u, s%u_0));\n",
                                       e_mods(in), i);
            }
        }
        for (i = 0; i < g->n; i++) {
            if (ir->insts[g->first + i].max4) {
                g_string_append_printf(o, "                q%u = U(m_);\n", i);
            }
        }
        g_string_append(o, "            }\n");
    }
    if (g->cube) {
        const char *v[3] = { "0.0f", "0.0f", "0.0f" };
        char b[3][48];

        for (i = 0; i < g->n; i++) {
            const HD5770IRInst *in = &ir->insts[g->first + i];
            int at = in->slot == 2 ? 0 : in->slot == 3 ? 1 : in->slot == 0 ? 2 : -1;

            if (in->cube && at >= 0) {
                snprintf(b[at], sizeof(b[at]), "hd_fmod(%uu, 0u, s%u_0)",
                         e_mods(in), i);
                v[at] = b[at];
            }
        }
        g_string_append_printf(o, "            {\n                uint co[4];\n"
                               "                hd_cube(%s, %s, %s, co);\n",
                               v[0], v[1], v[2]);
        for (i = 0; i < g->n; i++) {
            const HD5770IRInst *in = &ir->insts[g->first + i];

            if (in->cube) {
                g_string_append_printf(o, "                q%u = co[%u];\n", i,
                                       MIN(in->slot, 3));
            }
        }
        g_string_append(o, "            }\n");
    }
    for (i = 0; i < g->n; i++) {
        const HD5770IRInst *in = &ir->insts[g->first + i];

        if ((in->omod || in->clamp) && !(in->cube && in->slot == 3)) {
            g_string_append_printf(o, "            q%u = hd_post(0x%xu, %uu, %s, %uu, q%u);\n",
                                   i, in->op, e_mods(in),
                                   in->clamp ? "true" : "false", in->omod, i);
        }
    }
    for (i = 0; i < g->n; i++) {
        const HD5770IRInst *in = &ir->insts[g->first + i];

        g_string_append_printf(o, "            P[%u] = q%u;\n", in->slot, i);
    }
    g_string_append(o, "            if (act_) {\n");
    for (i = 0; i < g->n; i++) {
        const HD5770IRInst *in = &ir->insts[g->first + i];

        if (!in->wr) {
            continue;
        }
        g_string_append(o, "                ");
        if (in->pred_sel == 2) {
            g_string_append(o, "if (pred) ");
        } else if (in->pred_sel == 1) {
            g_string_append(o, "if (!pred) ");
        }
        g_string_append(o, "R[");
        e_reg(o, in->dst, in->dst_rel, in->index_mode);
        g_string_append_printf(o, "][%u] = q%u;\n", in->dst_chan, i);
    }
    g_string_append(o, "                if (anyp) {\n                    pred = newp;\n"
                       "                }\n            }\n        }\n");
}

/* the unit of a TEX fetch, and whether it is mipmapped */
static unsigned e_unit(const HD5770IR *ir, const HD5770IRFetch *f)
{
    unsigned u;

    for (u = 0; u < ir->nres && ir->res[u] != f->res; u++) {
        /* search */
    }
    return u;
}

/*
 * The derivative point of fetch f (a gradient, or a sample of a
 * mipmapped unit), or -1
 */
static int e_point(const HD5770IR *ir, const HD5770IRFetch *f, uint32_t mipmask,
                   const int *pidx, unsigned fi)
{
    return pidx[fi];
}

static void e_fetch_wb(GString *o, const HD5770IRFetch *f);

/*
 * Fetch f before its texture sample: the source, the derivatives and the
 * parameters of the shared site (TEX), or the whole fetch (the others)
 */
static void e_fetch(GString *o, const HD5770IR *ir, const HD5770IRFetch *f,
                    unsigned fi, uint32_t mipmask, bool main, const int *pidx)
{
    int p = e_point(ir, f, mipmask, pidx, fi);
    unsigned k;

    g_string_append(o, "                o[0] = 0u;\n                o[1] = 0u;\n"
                       "                o[2] = 0u;\n                o[3] = 0u;\n");
    if (!main && p >= 0) {
        unsigned np = 0, i;

        for (i = 0; i < ir->nfetches; i++) {
            np += pidx[i] >= 0;
        }
        /*
         * the source as a neighbour run that stops here returns it; the
         * run ends once it has every point
         */
        g_string_append_printf(o, "                if (!seen[%d]) {\n"
                               "                    seen[%d] = true;\n"
                               "                    nseen++;\n"
                               "                    if (nseen >= %uu) {\n"
                               "                        run = false;\n"
                               "                    }\n", p, p, np);
        for (k = 0; k < 4; k++) {
            g_string_append_printf(o, "                    rec[%d][%u] = R[%u][%u];\n", p,
                                   k, f->src, k);
        }
        g_string_append(o, "                }\n");
    }
    if (f->kind == HD5770_IRF_GRAD_H || f->kind == HD5770_IRF_GRAD_V) {
        g_string_append(o, "                uint nb[4];\n");
        for (k = 0; k < 4; k++) {
            g_string_append_printf(o, "                nb[%u] = R[%u][%u];\n", k, f->src, k);
        }
        if (main) {
            /* the neighbour as it reaches this fetch; itself in a neighbour */
            unsigned n = f->kind == HD5770_IRF_GRAD_H ? 1 : 2;

            g_string_append_printf(o, "                if (seen%u[%d]) {\n", n, p);
            for (k = 0; k < 4; k++) {
                g_string_append_printf(o, "                    nb[%u] = rec%u[%d][%u];\n", k,
                                       n, p, k);
            }
            g_string_append(o, "                }\n");
        }
        for (k = 0; k < 4; k++) {
            unsigned sel = (f->w[2] >> (20 + 3 * k)) & 7;

            if (sel < 4) {
                g_string_append_printf(o, "                o[%u] = U(hd_fsub(F(nb[%u]), "
                                       "F(R[%u][%u])));\n", k, sel, f->src, sel);
            } else {
                g_string_append_printf(o, "                o[%u] = U(0.0f);\n", k);
            }
        }
    } else if (f->kind == HD5770_IRF_VTX) {
        g_string_append_printf(o, "                HD_VTX(0x%08xu, 0x%08xu, 0x%08xu, R[%u][%u], o);\n",
                               f->w[0], f->w[1], f->w[2], f->src,
                               (f->w[0] >> 24) & 3);
    } else {
        unsigned u = e_unit(ir, f);

        for (k = 0; k < 4; k++) {
            unsigned sel = (f->w[2] >> (20 + 3 * k)) & 7;

            if (sel < 4) {
                g_string_append_printf(o, "                c[%u] = R[%u][%u];\n", k,
                                       f->src, sel);
            } else {
                g_string_append_printf(o, "                c[%u] = U(%s);\n", k,
                                       sel == 5 ? "1.0f" : "0.0f");
            }
        }
        for (k = 0; k < 8; k++) {
            g_string_append_printf(o, "                dv[%u] = 0.0f;\n", k);
        }
        if (p >= 0 && main) {
            /* derivatives: the neighbours as they reach this fetch */
            g_string_append_printf(o, "                if (seen1[%d] && seen2[%d]) {\n", p, p);
            for (k = 0; k < 4; k++) {
                unsigned sel = (f->w[2] >> (20 + 3 * k)) & 7;

                if (sel < 4) {
                    g_string_append_printf(o, "                    dv[%u] = hd_fsub(F(rec1[%d][%u]), F(c[%u]));\n"
                                           "                    dv[%u] = hd_fsub(F(rec2[%d][%u]), F(c[%u]));\n",
                                           k, p, sel, k, 4 + k, p, sel, k);
                }
            }
            g_string_append(o, "                }\n");
        }
        /* the fetch itself at the one site the program has (e_body) */
        g_string_append_printf(o, "                fu = %uu;\n                fw0 = 0x%08xu;\n"
                               "                fw1 = 0x%08xu;\n                fw2 = 0x%08xu;\n"
                               "                dotex = true;\n",
                               u, f->w[0], f->w[1], f->w[2]);
        return;
    }
    e_fetch_wb(o, f);
}

/* the destination of a fetch from its result o */
static void e_fetch_wb(GString *o, const HD5770IRFetch *f)
{
    unsigned k;

    for (k = 0; k < 4; k++) {
        unsigned sel = (f->w[1] >> (9 + 3 * k)) & 7;

        if (sel < 4) {
            g_string_append_printf(o, "                R[%u][%u] = o[%u];\n", f->dst, k, sel);
        } else if (sel == 4 || sel == 5) {
            g_string_append_printf(o, "                R[%u][%u] = U(%s);\n", f->dst, k,
                                   sel == 5 ? "1.0f" : "0.0f");
        }
    }
}

static void e_export(GString *o, const HD5770IRExport *e)
{
    unsigned k;
    const char *val[4];
    char b[4][32];

    for (k = 0; k < 4; k++) {
        unsigned sel = e->sel[k];

        if (sel < 4) {
            snprintf(b[k], sizeof(b[k]), "R[%u][%u]", e->gpr, sel);
            val[k] = b[k];
        } else {
            val[k] = sel == 4 ? "U(0.0f)" : sel == 5 ? "U(1.0f)" : "0u";
        }
    }
    if (e->target == HD5770_IR_DEPTH) {
        g_string_append_printf(o, "            zd = %s;\n            sd = %s;\n"
                               "            zw = true;\n", val[0], val[1]);
        return;
    }
    for (k = 0; k < 4; k++) {
        g_string_append_printf(o, "            C[%u][%u] = %s;\n", e->target, k, val[k]);
    }
    g_string_append_printf(o, "            cm |= %uu;\n", 1u << e->target);
}

/*
 * Fetches first..last - 1 by the index in fq (a variable of that name):
 * the source, the parameters and one texture sample site, the result
 */
static void e_dispatch(GString *o, const HD5770IR *ir, unsigned first,
                       unsigned last, uint32_t mipmask, bool main,
                       const int *pidx, const char *fq)
{
    unsigned i;

    g_string_append_printf(o, "            uint o[4];\n            uint c[4];\n"
                           "            float dv[8];\n"
                           "            uint fu = 0u, fw0 = 0u, fw1 = 0u, fw2 = 0u;\n"
                           "            bool dotex = false;\n"
                           "            switch (%s) {\n", fq);
    for (i = first; i < last; i++) {
        g_string_append_printf(o, "            case %u: {\n", i);
        e_fetch(o, ir, &ir->fetches[i], i, mipmask, main, pidx);
        g_string_append(o, "                break;\n            }\n");
    }
    g_string_append_printf(o, "            default:\n                break;\n            }\n"
                           "            if (dotex) {\n"
                           "                HD_TEX(fu, fw0, fw1, fw2, c, dv, o);\n"
                           "                switch (%s) {\n", fq);
    for (i = first; i < last; i++) {
        if (ir->fetches[i].kind == HD5770_IRF_TEX) {
            g_string_append_printf(o, "                case %u: {\n", i);
            e_fetch_wb(o, &ir->fetches[i]);
            g_string_append(o, "                    break;\n                }\n");
        }
    }
    g_string_append(o, "                default:\n                    break;\n"
                       "                }\n            }\n");
}

/*
 * Whether the program runs straight through: no loops, calls or backward
 * jumps; a forward jump skips the ops before its target
 */
static bool e_linear(const HD5770IR *ir)
{
    unsigned i;

    for (i = 0; i < ir->nops; i++) {
        const HD5770IROp *op = &ir->ops[i];
        unsigned t = op->cf & 0xf;

        switch (op->kind) {
        case HD5770_IRO_ALU:
            if (t == E_ALU_BREAK || t == E_ALU_CONTINUE) {
                return false;
            }
            continue;
        case HD5770_IRO_FETCH:
        case HD5770_IRO_EXPORT:
            continue;
        default:
            break;
        }
        switch (op->cf) {
        case E_PUSH:
        case E_JUMP:
        case E_ELSE:
            if (op->addr <= i) {
                return false;
            }
            break;
        case E_POP:
        case E_KILL:
        case E_END_PROGRAM:
        case E_NOP:
            break;
        default:
            return false;
        }
    }
    return true;
}

/* the state machine of one function */
static void e_body(GString *o, const HD5770IR *ir, uint32_t mipmask, bool main,
                   const int *pidx, bool straight)
{
    unsigned i, j, ng = MAX(ir->ngpr, 1u);
    GString *fh = g_string_new(NULL), *ob = g_string_new(NULL);
    bool lin = straight && e_linear(ir);

    g_string_append(fh, "        if (fq >= 0) {\n");

    e_dispatch(fh, ir, 0, ir->nfetches, mipmask, main, pidx, "fq");
    g_string_append(fh, "            fq = fq < fend ? fq + 1 : -1;\n"
                        "            continue;\n        }\n");

    g_string_append_printf(o,
        "    uint R[%u][4];\n"
        "    uint P[5];\n"
        "    for (uint i = 0u; i < %uu; i++) {\n"
        "        R[i][0] = 0u;\n        R[i][1] = 0u;\n        R[i][2] = 0u;\n"
        "        R[i][3] = 0u;\n    }\n"
        "    for (uint i = 0u; i < 5u; i++) {\n        P[i] = 0u;\n    }\n",
        ng, ng);
    /* constant indices only, so the registers can stay registers */
    for (i = 0; i < ng; i++) {
        g_string_append_printf(o,
            "    if (HD_POSGPR == %u) {\n"
            "        R[%u][0] = U(HPX);\n        R[%u][1] = U(HPY);\n"
            "        R[%u][2] = U(0.0f);\n        R[%u][3] = U(1.0f);\n"
            "    }\n", i, i, i, i, i);
    }
    g_string_append_printf(o,
        "    R[%u][0] = U(HB1);\n    R[%u][1] = U(HB2);\n",
        ir->map[0], ir->map[0]);
    g_string_append_printf(o,
        "    int ar[4];\n    ar[0] = 0;\n    ar[1] = 0;\n    ar[2] = 0;\n    ar[3] = 0;\n"
        "    int lidx = 0, sp = 0, lsp = 0, rsp = 0, pc = 0;\n"
        "    bool act_ = true, pred = false, run = true;\n"
        "    bool stk[%d];\n"
        "    int ls_start[%d], ls_end[%d], ls_cnt[%d], ls_idx[%d], ls_inc[%d], ls_sp[%d];\n"
        "    bool ls_dx10[%d];\n"
        "    int rets[%d];\n"
        "    uint steps = 0u, nseen = 0u;\n"
        "    int fq = -1, fend = -1, skp = 0;\n",
        E_STACK, E_LOOPS, E_LOOPS, E_LOOPS, E_LOOPS, E_LOOPS, E_LOOPS, E_LOOPS,
        E_CALLS);
    if (!lin) {
        g_string_append_printf(o,
            "    while ((run && pc < %u && steps < %uu) || fq >= 0) {\n"
            "%s"
            "        int cur = pc;\n"
            "        steps++;\n"
            "        pc++;\n"
            "        switch (cur) {\n", ir->nops, E_STEPS, fh->str);
    }
    g_string_free(fh, true);
    for (i = 0; i < ir->nops; i++) {
        const HD5770IROp *op = &ir->ops[i];
        unsigned t = op->cf & 0xf;

        g_string_set_size(ob, 0);
        switch (op->kind) {
        case HD5770_IRO_ALU:
            if (t == E_ALU_PUSH_BEFORE || t == E_ALU_BREAK ||
                t == E_ALU_CONTINUE) {
                e_push(ob);
            }
            for (j = 0; j < op->n; j++) {
                e_group(ob, ir, &ir->groups[op->first + j], main);
            }
            if (t == E_ALU_BREAK || t == E_ALU_CONTINUE) {
                /* PUSH, ALU, ELSE, BREAK (CONTINUE), POP */
                g_string_append(ob, "        {\n"
                                   "            bool hit = (sp > 0 ? stk[sp - 1] : true) && !act_;\n");
                e_pop(ob, "1");
                g_string_append_printf(ob,
                    "            if (hit && lsp > 0) {\n"
                    "                sp = ls_sp[lsp - 1] + 1;\n"
                    "                act_ = %s;\n"
                    "                pc = ls_end[lsp - 1] - 1;\n"
                    "            }\n        }\n",
                    t == E_ALU_CONTINUE ? "true" : "false");
            } else if (t == E_ALU_POP_AFTER) {
                e_pop(ob, "1");
            } else if (t == E_ALU_POP2_AFTER) {
                e_pop(ob, "2");
            } else if (t == E_ALU_ELSE_AFTER) {
                g_string_append(ob, "        act_ = (sp > 0 ? stk[sp - 1] : true) && !act_;\n");
            }
            break;
        case HD5770_IRO_FETCH:
            if (op->n && lin) {
                /* one sample site for the clause, in a loop kept a loop */
                g_string_append_printf(ob, "        if (act_) {\n"
                                       "        for (int fl = %u; fl < %u + HD_ZERO; fl++) {\n",
                                       op->first, op->first + op->n);
                e_dispatch(ob, ir, op->first, op->first + op->n, mipmask, main,
                           pidx, "fl");
                g_string_append(ob, "        }\n        }\n");
            } else if (op->n) {
                g_string_append_printf(ob, "        if (act_) {\n            fq = %u;\n"
                                       "            fend = %u;\n        }\n",
                                       op->first, op->first + op->n - 1);
            }
            break;
        case HD5770_IRO_EXPORT:
            if (main && op->n) {
                g_string_append(ob, "        if (act_) {\n");
                for (j = 0; j < op->n; j++) {
                    e_export(ob, &ir->exports[op->first + j]);
                }
                g_string_append(ob, "        }\n");
            }
            break;
        default:
            switch (op->cf) {
            case E_PUSH:
                e_push(ob);
                g_string_append(ob, "        if (!");
                e_cond(ob, op);
                g_string_append(ob, ") {\n");
                e_pop(ob, op->popn ? (op->popn == 1 ? "1" : op->popn == 2 ? "2" :
                                     op->popn == 3 ? "3" : op->popn == 4 ? "4" :
                                     op->popn == 5 ? "5" : op->popn == 6 ? "6" : "7")
                                  : "1");
                g_string_append_printf(ob, "            pc = %u;\n        }\n", op->addr);
                break;
            case E_JUMP: {
                char n[4];

                snprintf(n, sizeof(n), "%u", op->popn);
                g_string_append(ob, "        if (!");
                e_cond(ob, op);
                g_string_append(ob, ") {\n");
                e_pop(ob, n);
                g_string_append_printf(ob, "            pc = %u;\n        }\n", op->addr);
                break;
            }
            case E_ELSE: {
                char n[4];

                snprintf(n, sizeof(n), "%u", op->popn);
                g_string_append(ob,
                    "        act_ = (sp > 0 ? stk[sp - 1] : true) && !act_;\n"
                    "        if (!act_) {\n");
                e_pop(ob, n);
                g_string_append_printf(ob, "            pc = %u;\n        }\n", op->addr);
                break;
            }
            case E_POP: {
                char n[4];

                snprintf(n, sizeof(n), "%u", op->popn);
                e_pop(ob, n);
                break;
            }
            case E_CALL:
                g_string_append_printf(ob, "        if (rsp < %d && ", E_CALLS);
                e_cond(ob, op);
                g_string_append_printf(ob, ") {\n            rets[rsp] = pc;\n"
                                       "            rsp++;\n            pc = %u;\n"
                                       "        }\n", op->addr);
                break;
            case E_RETURN:
                g_string_append(ob, "        if (rsp == 0) {\n            run = false;\n"
                                   "        } else {\n            rsp--;\n"
                                   "            pc = rets[rsp];\n        }\n");
                break;
            case E_LOOP_START:
            case E_LOOP_START_DX10:
            case E_LOOP_START_NO_AL:
                g_string_append_printf(ob,
                    "        if (lsp >= %d) {\n            run = false;\n        } else {\n"
                    "            uint lc = HD_LCONST(%uu);\n"
                    "            ls_start[lsp] = pc;\n"
                    "            ls_end[lsp] = %u;\n"
                    "            ls_dx10[lsp] = %s;\n"
                    "            ls_cnt[lsp] = int(lc & 0xfffu);\n"
                    "            ls_idx[lsp] = int((lc >> 12) & 0xfffu);\n"
                    "            ls_inc[lsp] = I(lc) >> 24;\n"
                    "            ls_sp[lsp] = sp;\n",
                    E_LOOPS, op->cst, op->addr,
                    op->cf == E_LOOP_START_DX10 ? "true" : "false");
                e_push(ob);
                g_string_append_printf(ob,
                    "            if (!ls_dx10[lsp] && ls_cnt[lsp] == 0) {\n");
                e_pop(ob, "1");
                g_string_append_printf(ob,
                    "                pc = %u;\n            } else {\n"
                    "%s"
                    "                lsp++;\n            }\n        }\n", op->addr,
                    op->cf != E_LOOP_START_NO_AL
                    ? "                lidx = ls_idx[lsp];\n" : "");
                break;
            case E_LOOP_END:
                g_string_append(ob,
                    "        if (lsp == 0) {\n            run = false;\n        } else {\n"
                    "            int l_ = lsp - 1;\n"
                    "            bool again = false;\n"
                    "            if (act_) {\n"
                    "                if (ls_dx10[l_]) {\n                    again = true;\n"
                    "                } else {\n                    ls_cnt[l_]--;\n"
                    "                    again = ls_cnt[l_] != 0;\n                }\n"
                    "            }\n"
                    "            if (again) {\n"
                    "                ls_idx[l_] += ls_inc[l_];\n"
                    "                lidx = ls_idx[l_];\n"
                    "                pc = ls_start[l_];\n"
                    "            } else {\n"
                    "                sp = ls_sp[l_] + 1;\n");
                e_pop(ob, "1");
                g_string_append(ob, "                lsp--;\n            }\n        }\n");
                break;
            case E_LOOP_BREAK:
                g_string_append(ob, "        if (lsp > 0 && ");
                e_cond(ob, op);
                g_string_append(ob, ") {\n            act_ = false;\n        }\n");
                break;
            case E_LOOP_CONTINUE:
                g_string_append(ob, "        if (lsp > 0 && ");
                e_cond(ob, op);
                g_string_append(ob, ") {\n            pc = ls_end[lsp - 1] - 1;\n        }\n");
                break;
            case E_KILL:
                if (main) {
                    g_string_append(ob, "        if (");
                    e_cond(ob, op);
                    g_string_append(ob, ") {\n            killed = true;\n        }\n");
                }
                break;
            case E_END_PROGRAM:
                g_string_append(ob, "        run = false;\n");
                break;
            default:
                break;
            }
            break;
        }
        if (op->eop) {
            g_string_append(ob, "        run = false;\n");
        }
        if (lin) {
            /* a jump taken skips the ops before its target */
            char *b = g_strjoinv("skp = ", g_strsplit(ob->str, "pc = ", -1));

            g_string_append_printf(o, "    if (run && skp <= %u) {\n%s    }\n", i, b);
            g_free(b);
        } else {
            g_string_append_printf(o, "        case %u: {\n%s        break;\n        }\n",
                                   i, ob->str);
        }
    }
    if (!lin) {
        g_string_append(o, "        default:\n            run = false;\n            break;\n"
                           "        }\n    }\n");
    }
    g_string_free(ob, true);
}

void hd5770_emit_cf(GString *o, const HD5770IR *ir, uint32_t mipmask,
                    bool straight)
{
    int *pidx = g_new(int, ir->nfetches + 1);
    unsigned i, np = 0;

    for (i = 0; i < ir->nfetches; i++) {
        const HD5770IRFetch *f = &ir->fetches[i];
        unsigned op = f->w[0] & 0x1f;

        pidx[i] = -1;
        if (f->kind == HD5770_IRF_GRAD_H || f->kind == HD5770_IRF_GRAD_V ||
            (f->kind == HD5770_IRF_TEX && op != 0x03 && op != 0x04 &&
             ((mipmask >> e_unit(ir, f)) & 1))) {
            pidx[i] = np++;
        }
    }
    if (np) {
        g_string_append_printf(o,
            "void hd_nb(HD_CTX_PARAMS, float HB0, float HB1, float HB2, float HPW,\n"
            "           float HPX, float HPY, HD_OUTA(rec, %u, 4), HD_OUTBA(seen, %u))\n{\n"
            "    for (uint i = 0u; i < %uu; i++) {\n"
            "        seen[i] = false;\n"
            "        rec[i][0] = 0u;\n        rec[i][1] = 0u;\n        rec[i][2] = 0u;\n"
            "        rec[i][3] = 0u;\n    }\n", np, np, np);
        e_body(o, ir, mipmask, false, pidx, straight);
        g_string_append(o, "}\n\n");
    }
    g_string_append(o,
        "void hd_run(HD_CTX_PARAMS, float HB0, float HB1, float HB2, float HPW,\n"
        "            float HPX, float HPY,\n"
        "            float N1B0, float N1B1, float N1B2, float N1PW, float N1PX, float N1PY,\n"
        "            float N2B0, float N2B1, float N2B2, float N2PW, float N2PX, float N2PY,\n"
        "            HD_OUTB(killed), HD_OUTC(C), HD_OUTU(cm), HD_OUTU(zd), HD_OUTU(sd),\n"
        "            HD_OUTB(zw))\n{\n"
        "    killed = false;\n    cm = 0u;\n    zd = 0u;\n    sd = 0u;\n    zw = false;\n"
        "    for (uint i = 0u; i < 8u; i++) {\n"
        "        C[i][0] = 0u;\n        C[i][1] = 0u;\n        C[i][2] = 0u;\n"
        "        C[i][3] = 0u;\n    }\n");
    if (np) {
        /*
         * the neighbours to the right and below, each run once: the
         * sources of the derivative points as they first reach them
         */
        g_string_append_printf(o,
            "    uint rec1[%u][4], rec2[%u][4];\n"
            "    bool seen1[%u], seen2[%u];\n"
            "    hd_nb(HD_CTX_ARGS, N1B0, N1B1, N1B2, N1PW, N1PX, N1PY, rec1, seen1);\n"
            "    hd_nb(HD_CTX_ARGS, N2B0, N2B1, N2B2, N2PW, N2PX, N2PY, rec2, seen2);\n",
            np, np, np, np);
    }
    e_body(o, ir, mipmask, true, pidx, straight);
    g_string_append(o, "}\n\n");
    g_free(pidx);
}

unsigned hd5770_vs_np(const HD5770IR *ir)
{
    unsigned i, np = 0;

    for (i = 0; i < ir->nexports; i++) {
        unsigned t = ir->exports[i].target;

        if (t >= HD5770_IR_PARAM0 && t - HD5770_IR_PARAM0 + 1 > np) {
            np = t - HD5770_IR_PARAM0 + 1;
        }
    }
    return np;
}
