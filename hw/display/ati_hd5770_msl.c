/*
 * ATI Radeon HD 5770 (Juniper) pixel shader IR to Metal Shading Language.
 *
 * hd5770_msl_lib is the shared part: the layouts of the per-draw data
 * (GDraw, GTex, GTri, the same in ati_hd5770_mtl.m) and, as functions,
 * what the CPU path computes per pixel -- ALU opcodes (eval(), post()),
 * texture and vertex fetches, edge functions and interpolation, the
 * alpha test, blending and packing through the tiled address. Fast math
 * and contraction are off; where the CPU path's compiler fuses a
 * multiply and an add on AArch64 (a * b + c, a * b - c * d), the text
 * names the fma().
 *
 * hd5770_msl_program() adds one compute kernel per program, hd_main:
 * threadgroup g of 64 threads covers tile g of the dispatch, a block of
 * the box of a triangle, or of the two halves of a rectangle or quad
 * (they share no pixel); thread l is its pixel (x + l % w, y + l / w). The
 * thread runs the coverage test, the clip rectangles, the program and
 * the output stage. Buffers: 0 VRAM, 1 staging bytes, 3 guest RAM (for
 * surfaces outside VRAM, through their page tables), 2 tiles: eight
 * words each, the staging offsets of the GDraw, the GTri, the
 * triangle's inputs (ninputs x 3 vertices x 4 floats), the GTex array
 * and the bound constants, then x, y, and w | second half << 8 | the
 * inputs' stride in vec4 << 16; a second half's GTri is 128 bytes on.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "ati_hd5770_gpu.h"

const char hd5770_msl_lib[] =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"#pragma METAL fp math_mode(safe)\n"
"#pragma METAL fp contract(off)\n"
"\n"
"/* layouts shared with ati_hd5770_mtl.m */\n"
"struct GSurf {\n"
"    uint voff, mode, bpe, row_bytes, mt_per_row, tile_bytes, mt_bytes;\n"
"    uint pipes, banks, bankw, bankh, pib;\n"
"    uint mt_w_sh, mt_h_sh, pipes_sh, bankw_sh, bankh_sh, pib_sh, hi_sh, fast;\n"
"    uint space, pt_off, pg0, split_sh, split_stride;\n"
"    uint eoff[64];\n"
"};\n"
"struct GTex {\n"
"    GSurf s;\n"
"    uint fmt, num, w, h, dim, sgn, comps, is_float, unorm8, samp0;\n"
"    uint dst_sel[4];\n"
"    uint bits[4];\n"
"    float border[4];\n"
"    uint mip, last_lv, mip_filter, lev_off;\n"
"    float min_lod, max_lod, lod_bias;\n"
"    uint lw[16], lh[16];\n"
"    uint base_lv, levels, depth, pad;\n"
"    uint bc, slices;\n"
"    uint sb[16];\n"
"};\n"
"struct GVb {\n"
"    uint id, off, len, pad;\n"
"    uint res[8];\n"
"};\n"
"struct GDraw {\n"
"    GSurf rt;\n"
"    uint rt_fmt, rt_num, rt_comps, rt_unorm8, rt_float, rt_blend, wmask, blend;\n"
"    uint rt_bits[4];\n"
"    uint rt_src[4];\n"
"    float blend_const[4];\n"
"    uint alpha_test, alpha_func;\n"
"    float alpha_ref;\n"
"    uint cliprule;\n"
"    int cr[16];\n"
"    float pix_center;\n"
"    uint persp;\n"
"    int pos_gpr;\n"
"    uint ninputs, flat, cfile_off, ntex, nvb;\n"
"    GVb vb[8];\n"
"    GSurf zs, ss;\n"
"    uint db, zfmt, has_s, dctl, rctl, srm0, srm1, sclear;\n"
"    float zclear, zmin, zmax;\n"
"    uint nmrt, mrt_off, zexp, sexp, bconst, cb;\n"
"    uint lconst[32];\n"
"    uint resolve, rs_same, rs_fmt, rs_num, rs_comps;\n"
"    uint rs_src[4];\n"
"    GSurf rs;\n"
"    uint tri_off, prm_off, prm_st, pad2;\n"
"};\n"
"struct GTri {\n"
"    float x[3], y[3], w[3];\n"
"    float area;\n"
"    int x0, y0, x1, y1;\n"
"    float z[3];\n"
"    uint front;\n"
"};\n"
"\n"
"/* c / 255.0f for c = 0..255, as the CPU path computes it */\n"
"HD_U8_TABLE\n"
"\n"
"/* ---- formats ---- */\n"
"\n"
"constant uchar hd_fmt_bits[64][4] = {\n"
"    {0,0,0,0}, {8,0,0,0}, {4,4,0,0}, {2,3,3,0}, {0,0,0,0}, {16,0,0,0},\n"
"    {16,0,0,0}, {8,8,0,0}, {5,6,5,0}, {5,5,6,0}, {5,5,5,1}, {4,4,4,4},\n"
"    {1,5,5,5}, {32,0,0,0}, {32,0,0,0}, {16,16,0,0}, {16,16,0,0},\n"
"    {24,8,0,0}, {0,0,0,0}, {8,24,0,0}, {0,0,0,0}, {11,11,10,0},\n"
"    {0,0,0,0}, {10,11,11,0}, {0,0,0,0}, {10,10,10,2}, {8,8,8,8},\n"
"    {2,10,10,10}, {0,0,0,0}, {32,32,0,0}, {32,32,0,0}, {16,16,16,16},\n"
"    {16,16,16,16}, {0,0,0,0}, {32,32,32,32}, {32,32,32,32},\n"
"    {0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0},\n"
"    {0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0},\n"
"    {32,32,32,0},\n"
"};\n"
"\n"
"static __attribute__((always_inline)) uint hd_fmt_bytes(uint fmt)\n"
"{\n"
"    switch (fmt) {\n"
"    case 0x01: case 0x02: case 0x03: return 1;\n"
"    case 0x05: case 0x06: case 0x07: case 0x08: case 0x09: case 0x0a:\n"
"    case 0x0b: case 0x0c: return 2;\n"
"    case 0x0d: case 0x0e: case 0x0f: case 0x10: case 0x11: case 0x13:\n"
"    case 0x15: case 0x17: case 0x19: case 0x1a: case 0x1b: return 4;\n"
"    case 0x1d: case 0x1e: case 0x1f: case 0x20: return 8;\n"
"    case 0x30: return 12;\n"
"    case 0x22: case 0x23: return 16;\n"
"    default: return 0;\n"
"    }\n"
"}\n"
"\n"
"static __attribute__((always_inline)) bool hd_fmt_float(uint fmt)\n"
"{\n"
"    return fmt == 0x06 || fmt == 0x0e || fmt == 0x10 || fmt == 0x1e ||\n"
"           fmt == 0x20 || fmt == 0x23 || fmt == 0x30;\n"
"}\n"
"\n"
"static __attribute__((always_inline)) uint hd_rd(device const uchar *p, uint off, uint n)\n"
"{\n"
"    uint v = 0;\n"
"\n"
"    if (n == 4 && !(off & 3)) {\n"
"        return *(device const uint *)(p + off);\n"
"    }\n"
"    for (uint i = 0; i < n; i++) {\n"
"        v |= (uint)p[off + i] << (8 * i);\n"
"    }\n"
"    return v;\n"
"}\n"
"\n"
"static __attribute__((always_inline)) void hd_unpack(uint fmt, device const uchar *p, uint off, thread uint *c)\n"
"{\n"
"    uint n = hd_fmt_bytes(fmt), sh = 0;\n"
"    ulong v;\n"
"\n"
"    c[0] = c[1] = c[2] = c[3] = 0;\n"
"    if (n > 8) {\n"
"        for (uint i = 0; i < 4 && fmt < 64 && hd_fmt_bits[fmt][i]; i++) {\n"
"            c[i] = hd_rd(p, off + 4 * i, 4);\n"
"        }\n"
"        return;\n"
"    }\n"
"    if (n == 8) {\n"
"        v = (ulong)hd_rd(p, off, 4) | (ulong)hd_rd(p, off + 4, 4) << 32;\n"
"    } else {\n"
"        v = hd_rd(p, off, n);\n"
"    }\n"
"    for (uint i = 0; i < 4 && fmt < 64 && hd_fmt_bits[fmt][i]; i++) {\n"
"        uint b = hd_fmt_bits[fmt][i];\n"
"\n"
"        c[i] = (uint)((v >> sh) & (b == 32 ? 0xffffffffull : (1ull << b) - 1));\n"
"        sh += b;\n"
"    }\n"
"}\n"
"\n"
"static __attribute__((always_inline)) float hd_half(uint h)\n"
"{\n"
"    return float(as_type<half>((ushort)h));\n"
"}\n"
"\n"
"/* float_to_half(): round to nearest even */\n"
"static __attribute__((always_inline)) uint hd_f2h(float f)\n"
"{\n"
"    uint x = as_type<uint>(f);\n"
"    uint s = (x >> 16) & 0x8000, m = x & 0x7fffff;\n"
"    int e = (int)((x >> 23) & 0xff) - 127 + 15;\n"
"\n"
"    if (((x >> 23) & 0xff) == 0xff) {\n"
"        return s | 0x7c00 | (m ? 0x200 | (m >> 13) : 0);\n"
"    }\n"
"    if (e >= 31) {\n"
"        return s | 0x7c00;\n"
"    }\n"
"    if (e <= 0) {\n"
"        if (e < -10) {\n"
"            return s;\n"
"        }\n"
"        m |= 0x800000;\n"
"        uint sh = (uint)(14 - e), h = m >> sh, r = m & ((1u << sh) - 1);\n"
"        if (r > (1u << (sh - 1)) || (r == (1u << (sh - 1)) && (h & 1))) {\n"
"            h++;\n"
"        }\n"
"        return s | h;\n"
"    }\n"
"    uint h = s | ((uint)e << 10) | (m >> 13), r = m & 0x1fff;\n"
"    if (r > 0x1000 || (r == 0x1000 && (h & 1))) {\n"
"        h++;\n"
"    }\n"
"    return h;\n"
"}\n"
"\n"
"/* comp_value() */\n"
"static __attribute__((always_inline)) uint hd_comp(uint c, uint bits, bool is_float, bool is_signed, uint num)\n"
"{\n"
"    if (is_float) {\n"
"        return bits == 16 ? as_type<uint>(hd_half(c)) : c;\n"
"    }\n"
"    if (is_signed && bits < 32 && ((c >> (bits - 1)) & 1)) {\n"
"        c |= ~((1u << bits) - 1);\n"
"    }\n"
"    switch (num) {\n"
"    case 0:\n"
"        if (bits == 32) {\n"
"            return as_type<uint>(is_signed ? (float)(int)c / 2147483647.0f\n"
"                                           : (float)c / 4294967295.0f);\n"
"        } else if (is_signed) {\n"
"            return as_type<uint>(max((float)(int)c /\n"
"                                     (float)((1u << (bits - 1)) - 1), -1.0f));\n"
"        }\n"
"        return as_type<uint>((float)c / (float)((1u << bits) - 1));\n"
"    case 1:\n"
"        return c;\n"
"    default:\n"
"        return as_type<uint>(is_signed ? (float)(int)c : (float)c);\n"
"    }\n"
"}\n"
"\n"
"/* ---- surfaces ---- */\n"
"\n"
"static __attribute__((always_inline)) uint hd_pipe(uint tx, uint ty, uint pipes)\n"
"{\n"
"    switch (pipes) {\n"
"    case 2:\n"
"        return (tx & 1) ^ (ty & 1);\n"
"    case 4:\n"
"        return ((tx & 1) ^ ((ty >> 1) & 1)) | (((tx >> 1) & 1) ^ (ty & 1)) << 1;\n"
"    case 8:\n"
"        return ((tx & 1) ^ ((ty >> 2) & 1)) |\n"
"               (((tx >> 1) & 1) ^ ((ty >> 1) & 1) ^ ((ty >> 2) & 1)) << 1 |\n"
"               (((tx >> 2) & 1) ^ (ty & 1)) << 2;\n"
"    default:\n"
"        return 0;\n"
"    }\n"
"}\n"
"\n"
"static __attribute__((always_inline)) uint hd_bank(uint tx, uint ty, uint banks)\n"
"{\n"
"    uint x3 = tx & 1, x4 = (tx >> 1) & 1, x5 = (tx >> 2) & 1, x6 = (tx >> 3) & 1;\n"
"    uint y3 = ty & 1, y4 = (ty >> 1) & 1, y5 = (ty >> 2) & 1, y6 = (ty >> 3) & 1;\n"
"\n"
"    switch (banks) {\n"
"    case 16:\n"
"        return (x3 ^ y6) | (x4 ^ y5 ^ y6) << 1 | (x5 ^ y4) << 2 | (x6 ^ y3) << 3;\n"
"    case 8:\n"
"        return (x3 ^ y5) | (x4 ^ y4 ^ y5) << 1 | (x5 ^ y3) << 2;\n"
"    case 4:\n"
"        return (x3 ^ y4) | (x4 ^ y3) << 1;\n"
"    case 2:\n"
"        return x3 ^ y3;\n"
"    default:\n"
"        return 0;\n"
"    }\n"
"}\n"
"\n"
"/* hd5770_surf_offset_fast(): offset of element (x, y) */\n"
"static __attribute__((always_inline)) uint hd_addr(device const GSurf &s, uint x, uint y)\n"
"{\n"
"    uint tx = x >> 3, ty = y >> 3, total, pipe, bank, t;\n"
"\n"
"    if (s.mode < 2) {\n"
"        return y * s.row_bytes + x * s.bpe;\n"
"    }\n"
"    if (s.mode == 2) {\n"
"        t = (ty * s.mt_per_row + tx) * s.tile_bytes;\n"
"    } else if (s.split_sh) {\n"
"        /* hd5770_surf_offset(): tiles split in parts split_stride apart */\n"
"        uint e = s.eoff[(y & 7) * 8 + (x & 7)], sp = e >> s.split_sh;\n"
"\n"
"        total = ((ty >> (s.mt_h_sh - 3)) * s.mt_per_row + (tx >> (s.mt_w_sh - 3))) *\n"
"                s.mt_bytes + (e & ((1u << s.split_sh) - 1)) +\n"
"                (((ty & (s.bankh - 1)) << s.bankw_sh) +\n"
"                 ((tx >> s.pipes_sh) & (s.bankw - 1))) * s.tile_bytes +\n"
"                sp * s.split_stride;\n"
"        pipe = hd_pipe(tx, ty, s.pipes);\n"
"        bank = (hd_bank(tx >> (s.bankw_sh + s.pipes_sh), ty >> s.bankh_sh, s.banks) ^\n"
"                ((s.banks / 2 + 1) * sp)) & (s.banks - 1);\n"
"        return (total & (s.pib - 1)) | pipe << s.pib_sh |\n"
"               bank << (s.pib_sh + s.pipes_sh) | (total >> s.pib_sh) << s.hi_sh;\n"
"    } else {\n"
"        total = ((ty >> (s.mt_h_sh - 3)) * s.mt_per_row + (tx >> (s.mt_w_sh - 3))) *\n"
"                s.mt_bytes +\n"
"                (((ty & (s.bankh - 1)) << s.bankw_sh) +\n"
"                 ((tx >> s.pipes_sh) & (s.bankw - 1))) * s.tile_bytes;\n"
"        pipe = hd_pipe(tx, ty, s.pipes);\n"
"        bank = hd_bank(tx >> (s.bankw_sh + s.pipes_sh), ty >> s.bankh_sh,\n"
"                       s.banks) & (s.banks - 1);\n"
"        t = (total & (s.pib - 1)) | pipe << s.pib_sh |\n"
"            bank << (s.pib_sh + s.pipes_sh) | (total >> s.pib_sh) << s.hi_sh;\n"
"    }\n"
"    return t + s.eoff[(y & 7) * 8 + (x & 7)];\n"
"}\n"
"\n"
"/* the byte at surface offset o: in VRAM, or through the page table in RAM */\n"
"static __attribute__((always_inline)) device uchar *hd_at(device const GSurf &s, uint o,\n"
"                                                         device uchar *vram, device uchar *ram,\n"
"                                                         device const uchar *st)\n"
"{\n"
"    if (s.space) {\n"
"        uint q = s.pg0 + o;\n"
"        device const uint *pt = (device const uint *)(st + s.pt_off);\n"
"\n"
"        return ram + pt[q >> 12] + (q & 4095);\n"
"    }\n"
"    return vram + s.voff + o;\n"
"}\n"
"\n"
"/* exp2f() rounded as the host's: 2^n * e^(f ln 2) in float pairs */\n"
"static float2 hd_df_norm(float s, float e)\n"
"{\n"
"    float h = s + e;\n"
"\n"
"    return float2(h, e - (h - s));\n"
"}\n"
"static float2 hd_df_add(float2 x, float2 y)\n"
"{\n"
"    float s = x.x + y.x, bb = s - x.x;\n"
"    float e = (x.x - (s - bb)) + (y.x - bb);\n"
"\n"
"    return hd_df_norm(s, e + x.y + y.y);\n"
"}\n"
"static float2 hd_df_mul(float2 x, float2 y)\n"
"{\n"
"    float p = x.x * y.x, e = fma(x.x, y.x, -p);\n"
"\n"
"    return hd_df_norm(p, e + (x.x * y.y + x.y * y.x));\n"
"}\n"
"static float hd_exp2(float a)\n"
"{\n"
"    const float2 c[7] = {\n"
"        float2(1.0f, 0.0f), float2(1.0f, 0.0f), float2(5.000000000e-01f, 0.0f),\n"
"        float2(1.666666716e-01f, -4.967053879e-09f), float2(4.166666791e-02f, -1.241763470e-09f),\n"
"        float2(8.333333768e-03f, -4.346172033e-10f), float2(1.388888923e-03f, -3.363109444e-11f),\n"
"    };\n"
"\n"
"    if (isnan(a) || a < -151.0f || a >= 128.0f) {\n"
"        return exp2(a);\n"
"    }\n"
"    float n = rint(a), f = a - n;\n"
"    float th = f * 6.931471825e-01f;\n"
"    float2 t = hd_df_norm(th, fma(f, 6.931471825e-01f, -th) + f * -1.904654212e-09f);\n"
"    float q = fma(fma(fma(fma(2.505210794e-08f, t.x, 2.755731998e-07f), t.x, 2.755731884e-06f),\n"
"                      t.x, 2.480158764e-05f), t.x, 1.984127011e-04f);\n"
"    float2 p = float2(q, 0.0f);\n"
"\n"
"    for (int k = 6; k >= 0; k--) {\n"
"        p = hd_df_add(hd_df_mul(p, t), c[k]);\n"
"    }\n"
"    int e = (int)n;\n"
"    if (e >= -126) {\n"
"        return ldexp(p.x + p.y, e);\n"
"    }\n"
"    return ldexp(ldexp(p.x + p.y, e + 64), -64);\n"
"}\n"
"\n"
"/* log2f() rounded as the host's: e + 2 atanh((m - 1) / (m + 1)) / ln 2 in float pairs */\n"
"static float hd_log2(float a)\n"
"{\n"
"    if (isnan(a) || !(a > 0.0f) || isinf(a) || a < FLT_MIN) {\n"
"        return log2(a);\n"
"    }\n"
"    int e;\n"
"    float m = frexp(a, e);\n"
"\n"
"    if (m < 7.071067691e-01f) {\n"
"        m *= 2.0f;\n"
"        e--;\n"
"    }\n"
"    float num = m - 1.0f;\n"
"    float dh = m + 1.0f;\n"
"    float bb = dh - m, dl = (m - (dh - bb)) + (1.0f - bb);\n"
"    float sh = num / dh;\n"
"    float sl = (fma(-sh, dh, num) - sh * dl) / dh;\n"
"    float2 sv = hd_df_norm(sh, sl);\n"
"    float2 s2 = hd_df_mul(sv, sv);\n"
"    float x = s2.x;\n"
"    float q = fma(fma(fma(fma(fma(6.666667014e-02f, x, 7.692307979e-02f), x, 9.090909362e-02f),\n"
"                          x, 1.111111119e-01f), x, 1.428571492e-01f), x, 0.0f);\n"
"    float2 p = hd_df_add(float2(q, 0.0f), float2(2.000000030e-01f, -2.980232283e-09f));\n"
"\n"
"    p = hd_df_add(hd_df_mul(p, s2), float2(3.333333433e-01f, -9.934107759e-09f));\n"
"    p = hd_df_add(hd_df_mul(p, s2), float2(1.0f, 0.0f));\n"
"    p = hd_df_mul(p, float2(sv.x * 2.0f, sv.y * 2.0f));\n"
"    p = hd_df_mul(p, float2(1.442695022e+00f, 1.925963034e-08f));\n"
"    p = hd_df_add(float2((float)e, 0.0f), p);\n"
"    return p.x + p.y;\n"
"}\n"
"\n"
"/* sinf(t) (c 0) and cosf(t) (c 1) rounded as the host's: t - k pi/2 and\n"
" * the series in float pairs */\n"
"static float hd_sincos(float t, uint c)\n"
"{\n"
"    if (!(fabs(t) <= 1.0e4f)) {\n"
"        return c ? cos(t) : sin(t);\n"
"    }\n"
"    if (t == 0.0f) {\n"
"        return c ? 1.0f : t;\n"
"    }\n"
"    float k = rint(t * 6.366197467e-01f);\n"
"    float h1 = k * 1.570796371e+00f, e1 = fma(k, 1.570796371e+00f, -h1);\n"
"    float h2 = k * -4.371138829e-08f, e2 = fma(k, -4.371138829e-08f, -h2);\n"
"    float2 r = float2(t - h1, 0.0f);\n"
"\n"
"    r = hd_df_add(r, float2(-e1, 0.0f));\n"
"    r = hd_df_add(r, float2(-h2, 0.0f));\n"
"    r = hd_df_add(r, float2(-e2, 0.0f));\n"
"    r = hd_df_add(r, float2(k * 1.715124510e-15f, 0.0f));\n"
"    float2 r2 = hd_df_mul(r, r), p, s;\n"
"    float x = r2.x;\n"
"    uint q = ((uint)(int)k + c) & 3u;\n"
"\n"
"    if (q & 1u) {\n"
"        p = float2(fma(fma(fma(fma(-1.561920681e-16f, x, 4.779477256e-14f), x, -1.147074536e-11f),\n"
"                           x, 2.087675588e-09f), x, -2.755731998e-07f), 0.0f);\n"
"        p = hd_df_add(hd_df_mul(p, r2), float2(2.480158764e-05f, -3.406996094e-13f));\n"
"        p = hd_df_add(hd_df_mul(p, r2), float2(-1.388888923e-03f, 3.363109444e-11f));\n"
"        p = hd_df_add(hd_df_mul(p, r2), float2(4.166666791e-02f, -1.241763470e-09f));\n"
"        p = hd_df_add(hd_df_mul(p, r2), float2(-5.000000000e-01f, 0.0f));\n"
"        s = hd_df_add(float2(1.0f, 0.0f), hd_df_mul(p, r2));\n"
"    } else {\n"
"        p = float2(fma(fma(fma(fma(2.811457359e-15f, x, -7.647163610e-13f), x, 1.605904437e-10f),\n"
"                           x, -2.505210794e-08f), x, 2.755731884e-06f), 0.0f);\n"
"        p = hd_df_add(hd_df_mul(p, r2), float2(-1.984127011e-04f, 2.725596875e-12f));\n"
"        p = hd_df_add(hd_df_mul(p, r2), float2(8.333333768e-03f, -4.346172033e-10f));\n"
"        p = hd_df_add(hd_df_mul(p, r2), float2(-1.666666716e-01f, 4.967053879e-09f));\n"
"        s = hd_df_add(r, hd_df_mul(hd_df_mul(p, r2), r));\n"
"    }\n"
"    float v = s.x + s.y;\n"
"\n"
"    return q >= 2u ? -v : v;\n"
"}\n"
"\n"
"/* ---- ALU: eval() and post() ---- */\n"
"\n"
"static __attribute__((always_inline)) float hd_mul(float a, float b)\n"
"{\n"
"    if (a == 0.0f || b == 0.0f) {\n"
"        return 0.0f;\n"
"    }\n"
"    return a * b;\n"
"}\n"
"\n"
"static __attribute__((always_inline)) float hd_max9(float a, float b)\n"
"{\n"
"    return a >= b ? a : (isnan(b) ? a : b);\n"
"}\n"
"\n"
"static __attribute__((always_inline)) float hd_min9(float a, float b)\n"
"{\n"
"    return a < b ? a : (isnan(b) ? a : b);\n"
"}\n"
"\n"
"static __attribute__((always_inline)) float hd_cl01(float v)\n"
"{\n"
"    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : (isnan(v) ? 0.0f : v));\n"
"}\n"
"\n"
"#define F(x) as_type<float>(x)\n"
"#define U(x) as_type<uint>(x)\n"
"#define I(x) as_type<int>(x)\n"
"\n"
"/* mods: bit k neg of source k, bit 4 + k abs */\n"
"static __attribute__((always_inline)) float hd_fmod(uint mods, uint k, uint v)\n"
"{\n"
"    float f = F(v);\n"
"\n"
"    if ((mods >> (4 + k)) & 1) {\n"
"        f = fabs(f);\n"
"    }\n"
"    if ((mods >> k) & 1) {\n"
"        f = -f;\n"
"    }\n"
"    return f;\n"
"}\n"
"\n"
"static __attribute__((always_inline)) int hd_f2i(float a)\n"
"{\n"
"    return isnan(a) ? 0 : (a >= 2147483647.0f ? 2147483647 :\n"
"           (a <= -2147483648.0f ? (int)0x80000000 : (int)a));\n"
"}\n"
"\n"
"static __attribute__((always_inline)) uint hd_eval(uint op, uint mods, uint s0, uint s1, uint s2,\n"
"                    thread bool &kill)\n"
"{\n"
"    float a = 0, b = 0, c = 0;\n"
"\n"
"    if (op & 0x800) {\n"
"        a = F(s0);\n"
"        b = F(s1);\n"
"        c = F(s2);\n"
"        if (mods & 1) {\n"
"            a = -a;\n"
"        }\n"
"        if (mods & 2) {\n"
"            b = -b;\n"
"        }\n"
"        if (mods & 4) {\n"
"            c = -c;\n"
"        }\n"
"        switch (op & 0x1f) {\n"
"        case 0x14: return U(hd_mul(a, b) + c);\n"
"        case 0x15: return U((hd_mul(a, b) + c) * 2.0f);\n"
"        case 0x16: return U((hd_mul(a, b) + c) * 4.0f);\n"
"        case 0x17: return U((hd_mul(a, b) + c) * 0.5f);\n"
"        case 0x18: return U(fma(a, b, c));\n"
"        case 0x07: return U(fma(a, b, c));\n"
"        case 0x19: return a == 0.0f ? ((mods & 2) ? U(b) : s1) : ((mods & 4) ? U(c) : s2);\n"
"        case 0x1a: return a > 0.0f ? ((mods & 2) ? U(b) : s1) : ((mods & 4) ? U(c) : s2);\n"
"        case 0x1b: return a >= 0.0f ? ((mods & 2) ? U(b) : s1) : ((mods & 4) ? U(c) : s2);\n"
"        case 0x1c: return I(s0) == 0 ? s1 : s2;\n"
"        case 0x1d: return I(s0) > 0 ? s1 : s2;\n"
"        case 0x1e: return I(s0) >= 0 ? s1 : s2;\n"
"        case 0x04: {\n"
"            uint off = s1 & 31, w = s2 & 31;\n"
"            return w ? (s0 >> off) & ((1u << w) - 1) : 0;\n"
"        }\n"
"        case 0x05: {\n"
"            uint off = s1 & 31, w = s2 & 31;\n"
"            return w ? U((int)(s0 << (32 - off - w)) >> (32 - w)) : 0;\n"
"        }\n"
"        case 0x06: return (s0 & s1) | (~s0 & s2);\n"
"        default: return 0;\n"
"        }\n"
"    }\n"
"    a = hd_fmod(mods, 0, s0);\n"
"    b = hd_fmod(mods, 1, s1);\n"
"    switch (op) {\n"
"    case 0x00: return U(a + b);\n"
"    case 0x01: return U(hd_mul(a, b));\n"
"    case 0x02: return U(a * b);\n"
"    case 0x03: return U(hd_max9(a, b));\n"
"    case 0x04: return U(hd_min9(a, b));\n"
"    case 0x05: return U(fmax(a, b));\n"
"    case 0x06: return U(fmin(a, b));\n"
"    case 0x08: return U(a == b ? 1.0f : 0.0f);\n"
"    case 0x09: return U(a > b ? 1.0f : 0.0f);\n"
"    case 0x0a: return U(a >= b ? 1.0f : 0.0f);\n"
"    case 0x0b: return U(a != b ? 1.0f : 0.0f);\n"
"    case 0x0c: return a == b ? ~0u : 0;\n"
"    case 0x0d: return a > b ? ~0u : 0;\n"
"    case 0x0e: return a >= b ? ~0u : 0;\n"
"    case 0x0f: return a != b ? ~0u : 0;\n"
"    case 0x10: return U(a - floor(a));\n"
"    case 0x11: return U(trunc(a));\n"
"    case 0x12: return U(ceil(a));\n"
"    case 0x13: return U(rint(a));\n"
"    case 0x14: return U(floor(a));\n"
"    case 0x15: return U(I(s0) >> (s1 & 31));\n"
"    case 0x16: return s0 >> (s1 & 31);\n"
"    case 0x17: return s0 << (s1 & 31);\n"
"    case 0x19: return (mods & 0x11) ? U(a) : s0;\n"
"    case 0x1a: return 0;\n"
"    case 0x2c: kill = a == b; return U(kill ? 1.0f : 0.0f);\n"
"    case 0x2d: kill = a > b; return U(kill ? 1.0f : 0.0f);\n"
"    case 0x2e: kill = a >= b; return U(kill ? 1.0f : 0.0f);\n"
"    case 0x2f: kill = a != b; return U(kill ? 1.0f : 0.0f);\n"
"    case 0x40: kill = s0 > s1; return kill;\n"
"    case 0x41: kill = s0 >= s1; return kill;\n"
"    case 0x46: kill = I(s0) == I(s1); return kill;\n"
"    case 0x47: kill = I(s0) > I(s1); return kill;\n"
"    case 0x48: kill = I(s0) >= I(s1); return kill;\n"
"    case 0x49: kill = I(s0) != I(s1); return kill;\n"
"    case 0x30: return s0 & s1;\n"
"    case 0x31: return s0 | s1;\n"
"    case 0x32: return s0 ^ s1;\n"
"    case 0x33: return ~s0;\n"
"    case 0x34: return s0 + s1;\n"
"    case 0x35: return s0 - s1;\n"
"    case 0x36: return I(s0) > I(s1) ? s0 : s1;\n"
"    case 0x37: return I(s0) < I(s1) ? s0 : s1;\n"
"    case 0x38: return s0 > s1 ? s0 : s1;\n"
"    case 0x39: return s0 < s1 ? s0 : s1;\n"
"    case 0x3a: return I(s0) == I(s1) ? ~0u : 0;\n"
"    case 0x3b: return I(s0) > I(s1) ? ~0u : 0;\n"
"    case 0x3c: return I(s0) >= I(s1) ? ~0u : 0;\n"
"    case 0x3d: return I(s0) != I(s1) ? ~0u : 0;\n"
"    case 0x3e: return s0 > s1 ? ~0u : 0;\n"
"    case 0x3f: return s0 >= s1 ? ~0u : 0;\n"
"    case 0x50: return U(hd_f2i(trunc(a)));\n"
"    case 0xb1: return U(hd_f2i(floor(a)));\n"
"    case 0xb0: return U(hd_f2i(floor(a + 0.5f)));\n"
"    case 0x9a: return isnan(a) || a <= 0.0f ? 0 :\n"
"                      (a >= 4294967295.0f ? 0xffffffffu : (uint)trunc(a));\n"
"    case 0x9b: return U((float)I(s0));\n"
"    case 0x9c: return U((float)s0);\n"
"    case 0x81: return U(hd_exp2(a));\n"
"    case 0x82: return U(a == 0.0f ? -FLT_MAX : hd_log2(a));\n"
"    case 0x83: return U(hd_log2(a));\n"
"    case 0x84: return U(a == 0.0f ? FLT_MAX : 1.0f / a);\n"
"    case 0x85: return U(a == 0.0f ? 0.0f : 1.0f / a);\n"
"    case 0x86: return U(1.0f / a);\n"
"    case 0x87: return U(a == 0.0f ? FLT_MAX : 1.0f / sqrt(a));\n"
"    case 0x88: return U(a == 0.0f ? 0.0f : 1.0f / sqrt(a));\n"
"    case 0x89: return U(1.0f / sqrt(a));\n"
"    case 0x8a: return U(sqrt(a));\n"
"    case 0x8d: return U(hd_sincos(a * 2.0f * 3.14159265358979f, 0u));\n"
"    case 0x8e: return U(hd_sincos(a * 2.0f * 3.14159265358979f, 1u));\n"
"    case 0x8f: case 0x91: return s0 * s1;\n"
"    case 0x90: return (uint)(((long)I(s0) * I(s1)) >> 32);\n"
"    case 0x92: return (uint)(((ulong)s0 * s1) >> 32);\n"
"    case 0xcc: return U(clamp(I(s0), -256, 255));\n"
"    default: return 0;\n"
"    }\n"
"}\n"
"\n"
"static __attribute__((always_inline)) bool hd_isfloat(uint op, uint mods, bool clamp, uint omod)\n"
"{\n"
"    if (op & 0x800) {\n"
"        return (op & 0x1f) >= 0x07 && (op & 0x1f) <= 0x1b;\n"
"    }\n"
"    switch (op) {\n"
"    case 0x15: case 0x16: case 0x17: case 0x30: case 0x31: case 0x32:\n"
"    case 0x33: case 0x34: case 0x35: case 0x36: case 0x37: case 0x38:\n"
"    case 0x39: case 0x3a: case 0x3b: case 0x3c: case 0x3d: case 0x3e:\n"
"    case 0x3f: case 0x50: case 0x9a: case 0xb0: case 0xb1: case 0x8f:\n"
"    case 0x90: case 0x91: case 0x92: case 0xcc: case 0x0c: case 0x0d:\n"
"    case 0x0e: case 0x0f: case 0x42: case 0x43: case 0x44: case 0x45:\n"
"    case 0x40: case 0x41: case 0x46: case 0x47: case 0x48: case 0x49:\n"
"        return false;\n"
"    case 0x19:\n"
"        return omod || clamp || (mods & 0x11);\n"
"    default:\n"
"        return true;\n"
"    }\n"
"}\n"
"\n"
"static __attribute__((always_inline)) uint hd_post(uint op, uint mods, bool clamp, uint omod, uint v)\n"
"{\n"
"    float f;\n"
"\n"
"    if (!hd_isfloat(op, mods, clamp, omod)) {\n"
"        return v;\n"
"    }\n"
"    f = F(v);\n"
"    if (omod == 1) {\n"
"        f *= 2.0f;\n"
"    } else if (omod == 2) {\n"
"        f *= 4.0f;\n"
"    } else if (omod == 3) {\n"
"        f *= 0.5f;\n"
"    }\n"
"    if (clamp) {\n"
"        f = hd_cl01(f);\n"
"    }\n"
"    return U(f);\n"
"}\n"
"\n"
"static __attribute__((always_inline)) float hd_fadd(float a, float b)\n"
"{\n"
"    return a + b;\n"
"}\n"
"\n"
"static __attribute__((always_inline)) float hd_fsub(float a, float b)\n"
"{\n"
"    return a - b;\n"
"}\n"
"\n"
"/* PRED_SET*: the value eval() returns, and the predicate */\n"
"static __attribute__((always_inline)) uint hd_pred(uint op, uint mods, uint s0, uint s1, thread bool &p)\n"
"{\n"
"    float a = hd_fmod(mods, 0, s0), b = hd_fmod(mods, 1, s1);\n"
"\n"
"    switch (op) {\n"
"    case 0x20: p = a == b; break;\n"
"    case 0x21: p = a > b; break;\n"
"    case 0x22: p = a >= b; break;\n"
"    case 0x23: p = a != b; break;\n"
"    case 0x1e: p = s0 > s1; break;\n"
"    case 0x1f: p = s0 >= s1; break;\n"
"    case 0x42: p = I(s0) == I(s1); return p ? 0u : 1u;\n"
"    case 0x43: p = I(s0) > I(s1); return p ? 0u : 1u;\n"
"    case 0x44: p = I(s0) >= I(s1); return p ? 0u : 1u;\n"
"    case 0x45: p = I(s0) != I(s1); return p ? 0u : 1u;\n"
"    default: p = false; return 0u;\n"
"    }\n"
"    return U(p ? 0.0f : 1.0f);\n"
"}\n"
"\n"
"/* cube(): CUBE's results in slots x, y, z, w for the vector (x, y, z) */\n"
"static __attribute__((always_inline)) void hd_cube(float x, float y, float z, thread uint *o)\n"
"{\n"
"    float ax = fabs(x), ay = fabs(y), az = fabs(z), sc, tc, ma, face;\n"
"\n"
"    if (ax >= ay && ax >= az) {\n"
"        ma = x;\n"
"        face = x >= 0.0f ? 0.0f : 1.0f;\n"
"        sc = x >= 0.0f ? -z : z;\n"
"        tc = -y;\n"
"    } else if (ay >= az) {\n"
"        ma = y;\n"
"        face = y >= 0.0f ? 2.0f : 3.0f;\n"
"        sc = x;\n"
"        tc = y >= 0.0f ? z : -z;\n"
"    } else {\n"
"        ma = z;\n"
"        face = z >= 0.0f ? 4.0f : 5.0f;\n"
"        sc = z >= 0.0f ? x : -x;\n"
"        tc = -y;\n"
"    }\n"
"    o[0] = U(tc);\n"
"    o[1] = U(sc);\n"
"    o[2] = U(2.0f * ma);\n"
"    o[3] = U(face);\n"
"}\n"
"\n"
"/* ---- fetches ---- */\n"
"\n"
"static __attribute__((always_inline)) bool hd_wrap(thread int &c, int size, uint mode)\n"
"{\n"
"    int v = c;\n"
"\n"
"    switch (mode) {\n"
"    case 0:\n"
"        v %= size;\n"
"        if (v < 0) {\n"
"            v += size;\n"
"        }\n"
"        break;\n"
"    case 1: {\n"
"        int p = v % (2 * size);\n"
"\n"
"        if (p < 0) {\n"
"            p += 2 * size;\n"
"        }\n"
"        v = p < size ? p : 2 * size - 1 - p;\n"
"        break;\n"
"    }\n"
"    case 2: case 3: case 5: case 7:\n"
"        if (mode != 2) {\n"
"            v = v < 0 ? -v - 1 : v;\n"
"        }\n"
"        v = max(0, min(v, size - 1));\n"
"        break;\n"
"    default:\n"
"        if (v < 0 || v >= size) {\n"
"            return false;\n"
"        }\n"
"        break;\n"
"    }\n"
"    c = v;\n"
"    return true;\n"
"}\n"
"\n"
"static __attribute__((always_inline)) int hd_floori(float f)\n"
"{\n"
"    return isnan(f) ? 0 : (int)clamp(floor(f), -2147483648.0f, 2147483520.0f);\n"
"}\n"
"\n"
"/* what a fetch adds to its coordinates: tex_slice() and SAMPLE_C */\n"
"struct HdF {\n"
"    uint z;                     /* slice */\n"
"    float zn;                   /* 3D: depth in [0, 1] */\n"
"    uint cmp;                   /* SAMPLE_C: DEPTH_COMPARE_FUNCTION + 1 */\n"
"    float ref;\n"
"    bool ld;\n"
"};\n"
"\n"
"/* bc_chan(): channel value of texel i of a BC3 alpha / BC4 / BC5 block */\n"
"static __attribute__((noinline)) int hd_bc_chan(device const uchar *b, uint i, bool sgn)\n"
"{\n"
"    int a0 = sgn ? (int)as_type<char>(b[0]) : (int)b[0];\n"
"    int a1 = sgn ? (int)as_type<char>(b[1]) : (int)b[1];\n"
"    ulong bits = 0;\n"
"    int k;\n"
"\n"
"    for (uint j = 0; j < 6; j++) {\n"
"        bits |= (ulong)b[2 + j] << (8 * j);\n"
"    }\n"
"    k = (int)((bits >> (3 * i)) & 7);\n"
"    if (k == 0) {\n"
"        return a0;\n"
"    }\n"
"    if (k == 1) {\n"
"        return a1;\n"
"    }\n"
"    if (a0 > a1) {\n"
"        return ((8 - k) * a0 + (k - 1) * a1) / 7;\n"
"    }\n"
"    if (k < 6) {\n"
"        return ((6 - k) * a0 + (k - 1) * a1) / 5;\n"
"    }\n"
"    return sgn ? (k == 6 ? -127 : 127) : (k == 6 ? 0 : 255);\n"
"}\n"
"\n"
"static __attribute__((always_inline)) uint3 hd_565(uint c)\n"
"{\n"
"    uint r = (c >> 11) & 31, g = (c >> 5) & 63, b = c & 31;\n"
"\n"
"    return uint3((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2));\n"
"}\n"
"\n"
"/* bc_decode(): texel i of a BC1..BC5 block as 8-bit (BC4/5 SNORM: int) values */\n"
"static __attribute__((noinline)) int4 hd_bc(device const uchar *b, uint bc, uint i, uint sgn)\n"
"{\n"
"    int4 o = int4(0);\n"
"\n"
"    if (bc >= 4) {\n"
"        o.x = hd_bc_chan(b, i, sgn & 1);\n"
"        if (bc == 5) {\n"
"            o.y = hd_bc_chan(b + 8, i, (sgn >> 1) & 1);\n"
"        }\n"
"        return o;\n"
"    }\n"
"    device const uchar *cb = bc == 1 ? b : b + 8;\n"
"    uint c0 = (uint)cb[0] | (uint)cb[1] << 8, c1 = (uint)cb[2] | (uint)cb[3] << 8;\n"
"    uint idx = (uint)cb[4] | (uint)cb[5] << 8 | (uint)cb[6] << 16 | (uint)cb[7] << 24;\n"
"    bool four = bc != 1 || c0 > c1;\n"
"    uint sel = (idx >> (2 * i)) & 3, a;\n"
"    uint3 p0 = hd_565(c0), p1 = hd_565(c1), p;\n"
"\n"
"    if (sel == 0) {\n"
"        p = p0;\n"
"    } else if (sel == 1) {\n"
"        p = p1;\n"
"    } else if (sel == 2) {\n"
"        p = four ? (2 * p0 + p1) / 3 : (p0 + p1) / 2;\n"
"    } else {\n"
"        p = four ? (p0 + 2 * p1) / 3 : uint3(0);\n"
"    }\n"
"    a = bc == 1 && !four && sel == 3 ? 0 : 255;\n"
"    if (bc == 2) {\n"
"        a = ((b[i >> 1] >> ((i & 1) * 4)) & 15) * 17;\n"
"    } else if (bc == 3) {\n"
"        a = (uint)hd_bc_chan(b, i, false);\n"
"    }\n"
"    return int4(int3(p), (int)a);\n"
"}\n"
"\n"
"/* texel_z(): texel (x, y) of slice z of level lv (surface ls) */\n"
"static __attribute__((noinline)) uint4 hd_texel_z(device uchar *vram, device uchar *ram, device const uchar *st,\n"
"                     device const GTex &t, device const GSurf &ls, uint lv, uint z,\n"
"                     int x, int y, HdF f)\n"
"{\n"
"    int ex = t.bc ? x >> 2 : x, ey = t.bc ? y >> 2 : y;\n"
"    device uchar *p = hd_at(ls, hd_addr(ls, ex, ey) + z * t.sb[lv], vram, ram, st);\n"
"    uint4 v;\n"
"    uint c[4];\n"
"\n"
"    if (t.bc) {\n"
"        int4 e = hd_bc(p, t.bc, (uint)((y & 3) * 4 + (x & 3)), t.bc >= 4 ? t.sgn : 0);\n"
"\n"
"        for (uint k = 0; k < 4; k++) {\n"
"            v[k] = t.bc >= 4 && ((t.sgn >> k) & 1) ? U(max((float)e[k] / 127.0f, -1.0f))\n"
"                                                     : U(hd_u8[e[k] & 0xff]);\n"
"        }\n"
"        if (t.bc >= 4) {\n"
"            if (t.bc != 5) {\n"
"                v[1] = U(0.0f);\n"
"            }\n"
"            v[2] = U(0.0f);\n"
"            v[3] = U(1.0f);\n"
"        }\n"
"    } else if (t.unorm8) {\n"
"        uint e = hd_rd(p, 0, 4);\n"
"\n"
"        v = uint4(U(hd_u8[e & 0xff]), U(hd_u8[(e >> 8) & 0xff]),\n"
"                  U(hd_u8[(e >> 16) & 0xff]), U(hd_u8[e >> 24]));\n"
"    } else {\n"
"        hd_unpack(t.fmt, p, 0, c);\n"
"        for (uint k = 0; k < 4; k++) {\n"
"            if (k < t.comps) {\n"
"                v[k] = hd_comp(c[k], t.bits[k], t.is_float, (t.sgn >> k) & 1, t.num);\n"
"            } else {\n"
"                v[k] = U(k == 3 ? 1.0f : 0.0f);\n"
"            }\n"
"        }\n"
"    }\n"
"    if (f.cmp) {\n"
"        /* 1 when the reference passes against the fetched value */\n"
"        float fv = F(v[0]), r = f.ref;\n"
"        bool pass;\n"
"\n"
"        if (!t.is_float && t.num == 0 && !(t.sgn & 1) && !t.bc && t.bits[0] && t.bits[0] < 32) {\n"
"            /* the reference in the format's steps */\n"
"            uint mx = (1u << t.bits[0]) - 1;\n"
"            float q = min(max(r, 0.0f), 1.0f) * (float)mx;\n"
"\n"
"            q = q + 0.5f;\n"
"            r = (float)(uint)q / (float)mx;\n"
"        }\n"
"        switch (f.cmp - 1) {\n"
"        case 1: pass = r < fv; break;\n"
"        case 2: pass = r == fv; break;\n"
"        case 3: pass = r <= fv; break;\n"
"        case 4: pass = r > fv; break;\n"
"        case 5: pass = r != fv; break;\n"
"        case 6: pass = r >= fv; break;\n"
"        case 7: pass = true; break;\n"
"        default: pass = false; break;\n"
"        }\n"
"        v = uint4(U(pass ? 1.0f : 0.0f));\n"
"    }\n"
"    uint4 o;\n"
"    for (uint k = 0; k < 4; k++) {\n"
"        uint s = t.dst_sel[k];\n"
"\n"
"        o[k] = s < 4 ? v[s] : s == 5 ? U(1.0f) : U(0.0f);\n"
"    }\n"
"    return o;\n"
"}\n"
"\n"
"/* texel(): of slice f.z, or of a 3D texture at depth f.zn */\n"
"static __attribute__((noinline)) uint4 hd_texel_s(device uchar *vram, device uchar *ram, device const uchar *st,\n"
"                     device const GTex &t, device const GSurf &ls, uint lv, int x, int y,\n"
"                     HdF f)\n"
"{\n"
"    uint ns;\n"
"\n"
"    if (t.dim != 2) {\n"
"        return hd_texel_z(vram, ram, st, t, ls, lv, f.z, x, y, f);\n"
"    }\n"
"    ns = max(t.slices >> lv, 1u);\n"
"    if (((t.samp0 >> 13) & 3) == 2 && ns > 1 && !f.ld) {\n"
"        float zf = f.zn * (float)ns, a;\n"
"        int z0, z1;\n"
"        uint4 v0, v1, o;\n"
"\n"
"        zf = zf - 0.5f;\n"
"        z0 = hd_floori(zf);\n"
"        z1 = z0 + 1;\n"
"        a = zf - (float)z0;\n"
"        if (((t.samp0 >> 6) & 7) == 0) {\n"
"            z0 = (z0 + (int)ns) % (int)ns;\n"
"            z1 = z1 % (int)ns;\n"
"        } else {\n"
"            z0 = max(z0, 0);\n"
"            z1 = min(z1, (int)ns - 1);\n"
"        }\n"
"        v0 = hd_texel_z(vram, ram, st, t, ls, lv, (uint)z0, x, y, f);\n"
"        v1 = hd_texel_z(vram, ram, st, t, ls, lv, (uint)z1, x, y, f);\n"
"        for (uint k = 0; k < 4; k++) {\n"
"            o[k] = U(fma(F(v1[k]) - F(v0[k]), a, F(v0[k])));\n"
"        }\n"
"        return o;\n"
"    }\n"
"    return hd_texel_z(vram, ram, st, t, ls, lv, min((uint)(f.zn * (float)ns), ns - 1), x, y, f);\n"
"}\n"
"\n"
"static __attribute__((always_inline)) uint4 hd_point_s(device uchar *vram, device uchar *ram, device const uchar *st,\n"
"                     device const GTex &t, device const GSurf &ls, uint lv, uint lw, uint lh,\n"
"                     int x, int y, HdF f)\n"
"{\n"
"    if (!hd_wrap(x, lw, t.samp0 & 7) || !hd_wrap(y, lh, (t.samp0 >> 3) & 7)) {\n"
"        return uint4(U(t.border[0]), U(t.border[1]), U(t.border[2]), U(t.border[3]));\n"
"    }\n"
"    return hd_texel_s(vram, ram, st, t, ls, lv, x, y, f);\n"
"}\n"
"\n"
"/* sample_lv(): point or bilinear sample of a level at (u, vv) in its texels */\n"
"static __attribute__((noinline)) uint4 hd_sample_lv(device uchar *vram, device uchar *ram, device const uchar *st,\n"
"                      device const GTex &t, uint l, float u, float vv, bool lin,\n"
"                      HdF f)\n"
"{\n"
"    device const GSurf &ls = l ? ((device const GSurf *)(st + t.lev_off))[l] : t.s;\n"
"    uint lw = t.lw[l], lh = t.lh[l];\n"
"\n"
"    if (!lin) {\n"
"        return hd_point_s(vram, ram, st, t, ls, l, lw, lh, hd_floori(u), hd_floori(vv), f);\n"
"    }\n"
"    float fx = u - 0.5f, fy = vv - 0.5f;\n"
"    int x0 = hd_floori(fx), y0 = hd_floori(fy);\n"
"    float ax = fx - x0, ay = fy - y0;\n"
"    uint4 t00, t10, t01, t11, v;\n"
"\n"
"    t00 = hd_point_s(vram, ram, st, t, ls, l, lw, lh, x0, y0, f);\n"
"    t10 = hd_point_s(vram, ram, st, t, ls, l, lw, lh, x0 + 1, y0, f);\n"
"    t01 = hd_point_s(vram, ram, st, t, ls, l, lw, lh, x0, y0 + 1, f);\n"
"    t11 = hd_point_s(vram, ram, st, t, ls, l, lw, lh, x0 + 1, y0 + 1, f);\n"
"    for (uint k = 0; k < 4; k++) {\n"
"        float top = fma(F(t10[k]) - F(t00[k]), ax, F(t00[k]));\n"
"        float bot = fma(F(t11[k]) - F(t01[k]), ax, F(t01[k]));\n"
"\n"
"        v[k] = U(fma(bot - top, ay, top));\n"
"    }\n"
"    return v;\n"
"}\n"
"\n"
"/* lod_log2() */\n"
"static __attribute__((always_inline)) float hd_lodlog2(float x)\n"
"{\n"
"    if (!(x > 0.0f)) {\n"
"        return -128.0f;\n"
"    }\n"
"    uint b = as_type<uint>(x);\n"
"    return fma((float)(b & 0x7fffff), 1.0f / 8388608.0f, (float)((int)((b >> 23) & 0xff) - 127));\n"
"}\n"
"\n"
"/* tex_slice(): the slice of a fetch; cube coordinates made [0, 1] */\n"
"static __attribute__((always_inline)) void hd_slice(device const GTex &t, uint w1, thread uint *c, bool ld,\n"
"                     thread HdF &f)\n"
"{\n"
"    switch (t.dim) {\n"
"    case 3: {\n"
"        int layer = F(c[2]) > 0.0f ? (int)F(c[2]) : 0;\n"
"\n"
"        f.z = (uint)min(layer, 5);\n"
"        c[0] = U(F(c[0]) - 1.0f);\n"
"        c[1] = U(F(c[1]) - 1.0f);\n"
"        break;\n"
"    }\n"
"    case 4:\n"
"    case 5: {\n"
"        int layer = ld ? I(c[t.dim - 3]) : (int)rint(F(c[t.dim - 3]));\n"
"\n"
"        f.z = (uint)max(0, min(layer, (int)t.slices - 1));\n"
"        break;\n"
"    }\n"
"    case 2: {\n"
"        float zn;\n"
"\n"
"        if (ld) {\n"
"            zn = ((float)I(c[2]) + 0.5f) / (float)t.slices;\n"
"        } else {\n"
"            zn = (w1 >> 30) & 1 ? F(c[2]) : F(c[2]) / (float)t.slices;\n"
"        }\n"
"        if (((t.samp0 >> 6) & 7) == 0) {\n"
"            zn -= floor(zn);\n"
"        }\n"
"        f.zn = zn > 0.0f ? min(zn, 1.0f) : 0.0f;\n"
"        break;\n"
"    }\n"
"    default:\n"
"        break;\n"
"    }\n"
"}\n"
"\n"
"/* tfetch_stage() for the pixel shader; dv: the derivatives (0 if none) */\n"
"static __attribute__((noinline)) uint4 hd_tfetch_v(device uchar *vram, device uchar *ram, device const uchar *st,\n"
"                      device const GTex &t, uint w0,\n"
"                      uint w1, uint w2, uint4 c0, float4 dx, float4 dy)\n"
"{\n"
"    float dv[8] = { dx.x, dx.y, dx.z, dx.w, dy.x, dy.y, dy.z, dy.w };\n"
"    uint4 v;\n"
"    uint op = w0 & 0x1f, lv;\n"
"    int ox = ((int)(w2 << 27)) >> 27, oy = ((int)(w2 << 22)) >> 27;\n"
"    float lod = 0.0f, l0, fr, u, vv;\n"
"    bool lin;\n"
"    uint4 v1;\n"
"    uint c[4] = { c0[0], c0[1], c0[2], c0[3] };\n"
"    HdF f = { 0, 0.0f, 0, 0.0f, false };\n"
"\n"
"    v = uint4(U(0.0f));\n"
"    if (op >= 0x18 && op <= 0x1b) {\n"
"        /* SAMPLE_C*: as SAMPLE*, each texel compared with w */\n"
"        f.cmp = ((t.samp0 >> 22) & 7) + 1;\n"
"        f.ref = op == 0x19 || op == 0x1a ? F(c[2]) : F(c[3]);\n"
"        op -= 8;\n"
"    }\n"
"    f.ld = op == 0x03;\n"
"    if (op != 0x04) {\n"
"        hd_slice(t, w1, c, op == 0x03, f);\n"
"    }\n"
"    if (op == 0x04) {\n"
"        uint l = min(c[0] + t.base_lv, 31u);\n"
"\n"
"        v[0] = max(t.w >> l, 1u);\n"
"        v[1] = t.dim == 0 ? 1u : max(t.h >> l, 1u);\n"
"        v[2] = t.depth;\n"
"        v[3] = t.levels;\n"
"        return v;\n"
"    }\n"
"    if (op == 0x03) {\n"
"        int x = I(c[0]) + ox / 2, y = I(c[1]) + oy / 2;\n"
"\n"
"        if (x >= 0 && y >= 0 && x < (int)t.w && y < (int)t.h) {\n"
"            v = hd_texel_s(vram, ram, st, t, t.s, 0, x, y, f);\n"
"        }\n"
"        return v;\n"
"    }\n"
"    if (t.mip && ((w1 >> 28) & 1) && ((w1 >> 29) & 1)) {\n"
"        if (op == 0x11) {\n"
"            lod = F(c[3]);\n"
"        } else if (op != 0x13) {\n"
"            float ax = dv[0] * t.w, ay = dv[1] * t.h, bx = dv[4] * t.w, by = dv[5] * t.h;\n"
"            float a = fma(ax, ax, ay * ay), b = fma(bx, bx, by * by);\n"
"\n"
"            lod = 0.5f * hd_lodlog2(max(a, b));\n"
"            if (op == 0x12) {\n"
"                lod += F(c[3]);\n"
"            }\n"
"            if ((w1 >> 21) & 0x7f) {\n"
"                /* the instruction's LOD_BIAS, S3.4 */\n"
"                lod += (float)((int)(w1 << 4) >> 25) / 16.0f;\n"
"            }\n"
"        }\n"
"        lod = min(max(lod + t.lod_bias, t.min_lod), t.max_lod);\n"
"        if (!(lod > 0.0f)) {\n"
"            v = hd_sample_lv(vram, ram, st, t, 0, fma((float)ox, 0.5f, F(c[0]) * t.w),\n"
"                         fma((float)oy, 0.5f, F(c[1]) * t.h), (t.samp0 >> 9) & 1, f);\n"
"            return v;\n"
"        }\n"
"        lin = (t.samp0 >> 11) & 1;\n"
"        if (t.mip_filter == 1) {\n"
"            lv = min((uint)floor(lod + 0.5f), t.last_lv);\n"
"            v = hd_sample_lv(vram, ram, st, t, lv, fma((float)ox, 0.5f, F(c[0]) * t.lw[lv]),\n"
"                         fma((float)oy, 0.5f, F(c[1]) * t.lh[lv]), lin, f);\n"
"            return v;\n"
"        }\n"
"        l0 = floor(lod);\n"
"        fr = lod - l0;\n"
"        lv = min((uint)l0, t.last_lv);\n"
"        v = hd_sample_lv(vram, ram, st, t, lv, fma((float)ox, 0.5f, F(c[0]) * t.lw[lv]),\n"
"                     fma((float)oy, 0.5f, F(c[1]) * t.lh[lv]), lin, f);\n"
"        if (lv < t.last_lv) {\n"
"            lv++;\n"
"            v1 = hd_sample_lv(vram, ram, st, t, lv, fma((float)ox, 0.5f, F(c[0]) * t.lw[lv]),\n"
"                         fma((float)oy, 0.5f, F(c[1]) * t.lh[lv]), lin, f);\n"
"            for (uint k = 0; k < 4; k++) {\n"
"                v[k] = U(fma(F(v1[k]) - F(v[k]), fr, F(v[k])));\n"
"            }\n"
"        }\n"
"        return v;\n"
"    }\n"
"    u = F(c[0]);\n"
"    vv = F(c[1]);\n"
"    if ((w1 >> 28) & 1) {\n"
"        u *= t.w;\n"
"    }\n"
"    if ((w1 >> 29) & 1) {\n"
"        vv *= t.h;\n"
"    }\n"
"    u = fma((float)ox, 0.5f, u);\n"
"    vv = fma((float)oy, 0.5f, vv);\n"
"    if (t.dim == 0) {\n"
"        vv = 0.5f;\n"
"    }\n"
"    v = hd_sample_lv(vram, ram, st, t, 0, u, vv, ((t.samp0 >> 9) & 1) || ((t.samp0 >> 11) & 1), f);\n"
"    return v;\n"
"}\n"
"\n"
"/* the same with the fetch's arrays */\n"
"static __attribute__((always_inline)) void hd_tfetch_d(device uchar *vram, device uchar *ram, device const uchar *st,\n"
"                      device const GTex &t, uint w0,\n"
"                      uint w1, uint w2, thread const uint *c, thread const float *dv,\n"
"                      thread uint *v)\n"
"{\n"
"    uint4 r = hd_tfetch_v(vram, ram, st, t, w0, w1, w2, uint4(c[0], c[1], c[2], c[3]),\n"
"                          float4(dv[0], dv[1], dv[2], dv[3]), float4(dv[4], dv[5], dv[6], dv[7]));\n"
"\n"
"    v[0] = r.x;\n"
"    v[1] = r.y;\n"
"    v[2] = r.z;\n"
"    v[3] = r.w;\n"
"}\n"
"\n"
"static __attribute__((always_inline)) void hd_tfetch(device uchar *vram, device uchar *ram, device const uchar *st,\n"
"                      device const GTex &t, uint w0,\n"
"                      uint w1, uint w2, thread const uint *c, thread uint *v)\n"
"{\n"
"    uint4 r = hd_tfetch_v(vram, ram, st, t, w0, w1, w2, uint4(c[0], c[1], c[2], c[3]),\n"
"                          float4(0.0f), float4(0.0f));\n"
"\n"
"    v[0] = r.x;\n"
"    v[1] = r.y;\n"
"    v[2] = r.z;\n"
"    v[3] = r.w;\n"
"}\n"
"\n"
"/* vfetch_stage() for the pixel shader */\n"
"static __attribute__((noinline)) void hd_vfetch(device const GDraw &u, device const uchar *st, uint w0,\n"
"                      uint w1, uint w2, uint index, thread uint *v)\n"
"{\n"
"    uint bid = (w0 >> 8) & 0xff, off = w2 & 0xffff, fmt, num, sgn, n, k;\n"
"    uint c[4];\n"
"\n"
"    if (bid == 128) {\n"
"        uint e = index + off / 16;\n"
"        device const uint *cf = (device const uint *)(st + u.cfile_off);\n"
"\n"
"        for (k = 0; k < 4; k++) {\n"
"            v[k] = e < 256 ? cf[e * 4 + k] : 0;\n"
"        }\n"
"        return;\n"
"    }\n"
"    v[0] = v[1] = v[2] = U(0.0f);\n"
"    v[3] = U(1.0f);\n"
"    for (k = 0; k < u.nvb && u.vb[k].id != bid; k++) {\n"
"    }\n"
"    if (k == u.nvb) {\n"
"        return;\n"
"    }\n"
"    device const GVb &b = u.vb[k];\n"
"    uint size = b.res[1] + 1, stride = (b.res[2] >> 8) & 0x7ff;\n"
"    if ((w1 >> 21) & 1) {\n"
"        fmt = (b.res[2] >> 20) & 0x3f;\n"
"        num = (b.res[2] >> 26) & 3;\n"
"        sgn = (b.res[2] >> 28) & 1;\n"
"    } else {\n"
"        fmt = (w1 >> 22) & 0x3f;\n"
"        num = (w1 >> 28) & 3;\n"
"        sgn = (w1 >> 30) & 1;\n"
"    }\n"
"    n = hd_fmt_bytes(fmt);\n"
"    ulong at = (ulong)index * stride + off;\n"
"    if (at + n > size || at + n > b.len) {\n"
"        v[0] = v[1] = v[2] = 0;\n"
"        return;\n"
"    }\n"
"    hd_unpack(fmt, st, b.off + (uint)at, c);\n"
"    for (k = 0; k < 4 && hd_fmt_bits[fmt][k]; k++) {\n"
"        v[k] = hd_comp(c[k], hd_fmt_bits[fmt][k], hd_fmt_float(fmt), sgn, num);\n"
"    }\n"
"}\n"
"\n"
"/* ---- rasterisation ---- */\n"
"\n"
"static __attribute__((always_inline)) float hd_edge(float ax, float ay, float bx, float by, float px, float py)\n"
"{\n"
"    if (ax > bx || (ax == bx && ay > by)) {\n"
"        return -fma(ax - bx, py - by, -((ay - by) * (px - bx)));\n"
"    }\n"
"    return fma(bx - ax, py - ay, -((by - ay) * (px - ax)));\n"
"}\n"
"\n"
"static __attribute__((always_inline)) bool hd_owns(float ax, float ay, float bx, float by)\n"
"{\n"
"    return (ay == by && bx < ax) || by < ay;\n"
"}\n"
"\n"
"/* interp() */\n"
"static __attribute__((always_inline)) float hd_interp(device const GDraw &u, device const GTri &t,\n"
"                       device const float *prm, uint p, uint chan, bool flat,\n"
"                       float b0, float b1, float b2, float pw)\n"
"{\n"
"    float a, b, c;\n"
"\n"
"    if (p >= u.ninputs) {\n"
"        return 0.0f;\n"
"    }\n"
"    a = prm[p * 12 + chan];\n"
"    if (flat || ((u.flat >> p) & 1)) {\n"
"        return a;\n"
"    }\n"
"    b = prm[p * 12 + 4 + chan];\n"
"    c = prm[p * 12 + 8 + chan];\n"
"    if (u.persp) {\n"
"        return fma(c * t.w[2], b2, fma(a * t.w[0], b0, (b * t.w[1]) * b1)) * pw;\n"
"    }\n"
"    return fma(c, b2, fma(a, b0, b * b1));\n"
"}\n"
"\n"
"/* ---- depth and stencil: db_test() ---- */\n"
"\n"
"static __attribute__((always_inline)) bool hd_dbcmp(uint fn, uint a, uint b)\n"
"{\n"
"    switch (fn) {\n"
"    case 0: return false;\n"
"    case 1: return a < b;\n"
"    case 2: return a == b;\n"
"    case 3: return a <= b;\n"
"    case 4: return a > b;\n"
"    case 5: return a != b;\n"
"    case 6: return a >= b;\n"
"    default: return true;\n"
"    }\n"
"}\n"
"\n"
"static __attribute__((always_inline)) bool hd_dbcmpf(uint fn, float a, float b)\n"
"{\n"
"    switch (fn) {\n"
"    case 0: return false;\n"
"    case 1: return a < b;\n"
"    case 2: return a == b;\n"
"    case 3: return a <= b;\n"
"    case 4: return a > b;\n"
"    case 5: return a != b;\n"
"    case 6: return a >= b;\n"
"    default: return true;\n"
"    }\n"
"}\n"
"\n"
"static __attribute__((always_inline)) uint hd_sop(uint op, uint v, uint ref)\n"
"{\n"
"    switch (op) {\n"
"    case 0: return v;\n"
"    case 1: return 0;\n"
"    case 2: return ref;\n"
"    case 3: return v < 255 ? v + 1 : 255;\n"
"    case 4: return v ? v - 1 : 0;\n"
"    case 5: return ~v & 0xff;\n"
"    case 6: return (v + 1) & 0xff;\n"
"    default: return (v - 1) & 0xff;\n"
"    }\n"
"}\n"
"\n"
"/* the depth test alone: whether db() will pass (no stencil, no clear) */\n"
"static __attribute__((noinline)) bool hd_ztest(device uchar *vram, device uchar *ram,\n"
"                  device const uchar *st, device const GDraw &u, int x, int y, float z)\n"
"{\n"
"    device uchar *zp = hd_at(u.zs, hd_addr(u.zs, x, y), vram, ram, st);\n"
"    float zf = min(max(z, u.zmin), u.zmax);\n"
"    uint znew = u.zfmt == 1 ? (uint)rint(zf * 65535.0f) :\n"
"                u.zfmt == 2 ? (uint)rint(zf * 16777215.0f) : as_type<uint>(zf);\n"
"    uint zold = u.zfmt == 1 ? (uint)*(device ushort *)zp : *(device uint *)zp;\n"
"\n"
"    if (u.zfmt == 2) {\n"
"        zold &= 0xffffff;\n"
"    }\n"
"    return u.zfmt == 3 ? hd_dbcmpf((u.dctl >> 4) & 7, zf, as_type<float>(zold))\n"
"                       : hd_dbcmp((u.dctl >> 4) & 7, znew, zold);\n"
"}\n"
"\n"
"static __attribute__((noinline)) bool hd_db(device uchar *vram, device uchar *ram,\n"
"                  device const uchar *st, device const GDraw &u,\n"
"                  int x, int y, float z, int sref, bool tfront)\n"
"{\n"
"    uint dctl = u.dctl;\n"
"    bool front = tfront || !(dctl & 0x80);\n"
"    uint srm = front ? u.srm0 : u.srm1;\n"
"    uint znew = 0, zold = 0;\n"
"    bool zpass = true, spass = true;\n"
"    device uchar *zp = 0;\n"
"\n"
"    if (u.zfmt) {\n"
"        float zf = (u.rctl & 1) ? u.zclear : z;\n"
"\n"
"        zp = hd_at(u.zs, hd_addr(u.zs, x, y), vram, ram, st);\n"
"        zf = min(max(zf, u.zmin), u.zmax);\n"
"        znew = u.zfmt == 1 ? (uint)rint(zf * 65535.0f) :\n"
"               u.zfmt == 2 ? (uint)rint(zf * 16777215.0f) : as_type<uint>(zf);\n"
"        zold = u.zfmt == 1 ? (uint)*(device ushort *)zp : *(device uint *)zp;\n"
"        if (u.zfmt == 2) {\n"
"            zold &= 0xffffff;\n"
"        }\n"
"        if (dctl & 2) {\n"
"            zpass = u.zfmt == 3 ? hd_dbcmpf((dctl >> 4) & 7, zf, as_type<float>(zold))\n"
"                                : hd_dbcmp((dctl >> 4) & 7, znew, zold);\n"
"        }\n"
"    }\n"
"    if (u.has_s) {\n"
"        device uchar *sp = hd_at(u.ss, hd_addr(u.ss, x, y), vram, ram, st);\n"
"        uint sold = *sp, snew, wm = (srm >> 16) & 0xff;\n"
"        uint ref = (u.rctl & 2) ? u.sclear : sref >= 0 ? (uint)sref & 0xff : srm & 0xff;\n"
"\n"
"        if (dctl & 1) {\n"
"            uint m = (srm >> 8) & 0xff, op;\n"
"\n"
"            spass = hd_dbcmp(front ? (dctl >> 8) & 7 : (dctl >> 20) & 7, ref & m, sold & m);\n"
"            op = !spass ? (front ? dctl >> 11 : dctl >> 23) :\n"
"                 !zpass ? (front ? dctl >> 17 : dctl >> 29) :\n"
"                          (front ? dctl >> 14 : dctl >> 26);\n"
"            snew = hd_sop(op & 7, sold, ref);\n"
"        } else {\n"
"            snew = ref;\n"
"        }\n"
"        *sp = (uchar)((sold & ~wm) | (snew & wm));\n"
"    }\n"
"    if (zp && zpass && spass && (dctl & 6) == 6) {\n"
"        if (u.zfmt == 1) {\n"
"            *(device ushort *)zp = (ushort)znew;\n"
"        } else if (u.zfmt == 2) {\n"
"            *(device uint *)zp = znew | (*(device uint *)zp & 0xff000000u);\n"
"        } else {\n"
"            *(device uint *)zp = znew;\n"
"        }\n"
"    }\n"
"    return zpass && spass;\n"
"}\n"
"\n"
"/* ---- output: alpha_pass(), put_pixel() ---- */\n"
"\n"
"static __attribute__((always_inline)) bool hd_alpha(device const GDraw &u, float a)\n"
"{\n"
"    float r = u.alpha_ref;\n"
"\n"
"    switch (u.alpha_func) {\n"
"    case 0: return false;\n"
"    case 1: return a < r;\n"
"    case 2: return a == r;\n"
"    case 3: return a <= r;\n"
"    case 4: return a > r;\n"
"    case 5: return a != r;\n"
"    case 6: return a >= r;\n"
"    default: return true;\n"
"    }\n"
"}\n"
"\n"
"static __attribute__((always_inline)) float hd_bfactor(uint f, float4 src, float4 dst,\n"
"                        device const float *cst, uint ch)\n"
"{\n"
"    switch (f) {\n"
"    case 0: return 0.0f;\n"
"    case 1: return 1.0f;\n"
"    case 2: return src[ch];\n"
"    case 3: return 1.0f - src[ch];\n"
"    case 4: return src[3];\n"
"    case 5: return 1.0f - src[3];\n"
"    case 6: return dst[3];\n"
"    case 7: return 1.0f - dst[3];\n"
"    case 8: return dst[ch];\n"
"    case 9: return 1.0f - dst[ch];\n"
"    case 10: return ch == 3 ? 1.0f : min(src[3], 1.0f - dst[3]);\n"
"    case 13: return cst[ch];\n"
"    case 14: return 1.0f - cst[ch];\n"
"    case 19: return cst[3];\n"
"    case 20: return 1.0f - cst[3];\n"
"    default: return 0.0f;\n"
"    }\n"
"}\n"
"\n"
"static __attribute__((always_inline)) float hd_combine(uint fn, float s, float sf, float dv, float df)\n"
"{\n"
"    switch (fn) {\n"
"    case 0: return fma(s, sf, dv * df);\n"
"    case 1: return fma(s, sf, -(dv * df));\n"
"    case 2: return min(s, dv);\n"
"    case 3: return max(s, dv);\n"
"    case 4: return fma(dv, df, -(s * sf));\n"
"    default: return s;\n"
"    }\n"
"}\n"
"\n"
"static __attribute__((always_inline)) void hd_pack_store(device uchar *vram, uint off, uint fmt, thread const uint *c)\n"
"{\n"
"    uint n = hd_fmt_bytes(fmt), sh = 0;\n"
"    ulong v = 0;\n"
"\n"
"    if (n > 8) {\n"
"        for (uint i = 0; i < 4 && hd_fmt_bits[fmt][i]; i++) {\n"
"            *(device uint *)(vram + off + 4 * i) = c[i];\n"
"        }\n"
"        return;\n"
"    }\n"
"    for (uint i = 0; i < 4 && hd_fmt_bits[fmt][i]; i++) {\n"
"        uint b = hd_fmt_bits[fmt][i];\n"
"        ulong m = b == 32 ? 0xffffffffull : (1ull << b) - 1;\n"
"\n"
"        v |= ((ulong)c[i] & m) << sh;\n"
"        sh += b;\n"
"    }\n"
"    switch (n) {\n"
"    case 1: vram[off] = (uchar)v; break;\n"
"    case 2: *(device ushort *)(vram + off) = (ushort)v; break;\n"
"    case 4: *(device uint *)(vram + off) = (uint)v; break;\n"
"    case 8:\n"
"        *(device uint *)(vram + off) = (uint)v;\n"
"        *(device uint *)(vram + off + 4) = (uint)(v >> 32);\n"
"        break;\n"
"    default: break;\n"
"    }\n"
"}\n"
"\n"
"static __attribute__((noinline)) void hd_put(device uchar *vram, device uchar *ram, device const uchar *st,\n"
"                   device const GDraw &u, int x, int y,\n"
"                   thread const uint *col)\n"
"{\n"
"    device uchar *p = hd_at(u.rt, hd_addr(u.rt, x, y), vram, ram, st);\n"
"    uint4 c = uint4(0), old;\n"
"    uint n = u.rt_comps;\n"
"    float4 src, dst;\n"
"\n"
"    if (u.rt_unorm8) {\n"
"        uint e = *(device const uint *)p;\n"
"\n"
"        old = uint4(e & 0xff, (e >> 8) & 0xff, (e >> 16) & 0xff, e >> 24);\n"
"    } else {\n"
"        uint o[4];\n"
"\n"
"        hd_unpack(u.rt_fmt, p, 0, o);\n"
"        old = uint4(o[0], o[1], o[2], o[3]);\n"
"    }\n"
"    if (u.rt_float) {\n"
"        /* float targets: blending without clamps */\n"
"        uint4 ob = uint4(col[0], col[1], col[2], col[3]);\n"
"        float4 fs = float4(F(col[0]), F(col[1]), F(col[2]), F(col[3]));\n"
"        float4 fd = float4(0.0f, 0.0f, 0.0f, 1.0f);\n"
"\n"
"        for (uint i = 0; i < n; i++) {\n"
"            fd[u.rt_src[i]] = u.rt_bits[i] == 16 ? hd_half(old[i]) : as_type<float>(old[i]);\n"
"        }\n"
"        if ((u.blend & (1u << 30)) && u.rt_blend) {\n"
"            uint csrc = u.blend & 0x1f, cfn = (u.blend >> 5) & 7;\n"
"            uint cdst = (u.blend >> 8) & 0x1f;\n"
"            uint asrc = csrc, afn = cfn, adst = cdst;\n"
"\n"
"            if ((u.blend >> 29) & 1) {\n"
"                asrc = (u.blend >> 16) & 0x1f;\n"
"                afn = (u.blend >> 21) & 7;\n"
"                adst = (u.blend >> 24) & 0x1f;\n"
"            }\n"
"            for (uint i = 0; i < 4; i++) {\n"
"                uint bs = i == 3 ? asrc : csrc, bd = i == 3 ? adst : cdst;\n"
"                uint fn = i == 3 ? afn : cfn;\n"
"\n"
"                ob[i] = as_type<uint>(hd_combine(fn, fs[i],\n"
"                                                 hd_bfactor(bs, fs, fd, u.blend_const, i), fd[i],\n"
"                                                 hd_bfactor(bd, fs, fd, u.blend_const, i)));\n"
"            }\n"
"        }\n"
"        for (uint i = 0; i < n; i++) {\n"
"            uint s = u.rt_src[i];\n"
"\n"
"            c[i] = !((u.wmask >> s) & 1) ? old[i] :\n"
"                   u.rt_bits[i] == 16 ? hd_f2h(as_type<float>(ob[s])) : ob[s];\n"
"        }\n"
"        {\n"
"            uint cc[4] = { c[0], c[1], c[2], c[3] };\n"
"\n"
"            hd_pack_store(p, 0, u.rt_fmt, cc);\n"
"        }\n"
"        return;\n"
"    }\n"
"    if (u.rt_num == 4) {\n"
"        for (uint i = 0; i < n; i++) {\n"
"            uint s = u.rt_src[i];\n"
"\n"
"            c[i] = (u.wmask >> s) & 1 ? col[s] : old[i];\n"
"        }\n"
"        {\n"
"            uint cc[4] = { c[0], c[1], c[2], c[3] };\n"
"\n"
"            hd_pack_store(p, 0, u.rt_fmt, cc);\n"
"        }\n"
"        return;\n"
"    }\n"
"    for (uint i = 0; i < 4; i++) {\n"
"        src[i] = F(col[i]);\n"
"        if (isnan(src[i])) {\n"
"            src[i] = 0.0f;\n"
"        }\n"
"        dst[i] = i == 3 ? 1.0f : 0.0f;\n"
"    }\n"
"    for (uint i = 0; i < n; i++) {\n"
"        dst[u.rt_src[i]] = u.rt_unorm8 ? hd_u8[old[i]]\n"
"                                       : (float)old[i] / (float)((1u << u.rt_bits[i]) - 1);\n"
"    }\n"
"    if ((u.blend & (1u << 30)) && u.rt_blend) {\n"
"        uint csrc = u.blend & 0x1f, cfn = (u.blend >> 5) & 7;\n"
"        uint cdst = (u.blend >> 8) & 0x1f;\n"
"        uint asrc = csrc, afn = cfn, adst = cdst;\n"
"        float4 out, sc;\n"
"\n"
"        if ((u.blend >> 29) & 1) {\n"
"            asrc = (u.blend >> 16) & 0x1f;\n"
"            afn = (u.blend >> 21) & 7;\n"
"            adst = (u.blend >> 24) & 0x1f;\n"
"        }\n"
"        for (uint i = 0; i < 4; i++) {\n"
"            sc[i] = min(max(src[i], 0.0f), 1.0f);\n"
"        }\n"
"        for (uint i = 0; i < 4; i++) {\n"
"            uint fs = i == 3 ? asrc : csrc, fd = i == 3 ? adst : cdst;\n"
"            uint fn = i == 3 ? afn : cfn;\n"
"\n"
"            out[i] = hd_combine(fn, sc[i],\n"
"                                hd_bfactor(fs, sc, dst, u.blend_const, i), dst[i],\n"
"                                hd_bfactor(fd, sc, dst, u.blend_const, i));\n"
"        }\n"
"        src = out;\n"
"    }\n"
"    for (uint i = 0; i < n; i++) {\n"
"        uint s = u.rt_src[i];\n"
"        uint mx = (1u << u.rt_bits[i]) - 1;\n"
"        float v = min(max(src[s], 0.0f), 1.0f);\n"
"\n"
"        c[i] = (u.wmask >> s) & 1 ? (uint)fma(v, (float)mx, 0.5f) : old[i];\n"
"    }\n"
"    if (u.rt_unorm8) {\n"
"        *(device uint *)p = (c[0] & 0xff) | (c[1] & 0xff) << 8 |\n"
"                                       (c[2] & 0xff) << 16 | (c[3] & 0xff) << 24;\n"
"        return;\n"
"    }\n"
"    {\n"
"        uint cc[4] = { c[0], c[1], c[2], c[3] };\n"
"\n"
"        hd_pack_store(p, 0, u.rt_fmt, cc);\n"
"    }\n"
"}\n"
"\n"
"static __attribute__((always_inline)) bool hd_clip(device const GDraw &u, int x, int y)\n"
"{\n"
"    uint idx = 0;\n"
"\n"
"    for (uint i = 0; i < 4; i++) {\n"
"        if (x >= u.cr[i * 4] && x < u.cr[i * 4 + 2] &&\n"
"            y >= u.cr[i * 4 + 1] && y < u.cr[i * 4 + 3]) {\n"
"            idx |= 1u << i;\n"
"        }\n"
"    }\n"
"    return (u.cliprule >> idx) & 1;\n"
"}\n";

static void src_expr(GString *o, const HD5770IRSrc *s, unsigned q)
{
    switch (s->kind) {
    case HD5770_IRS_GPR:
        g_string_append_printf(o, "R%u[%u][%u]", q, s->idx, s->chan);
        break;
    case HD5770_IRS_PV:
        g_string_append_printf(o, "P%u[%u]", q, s->chan);
        break;
    case HD5770_IRS_CONST:
        g_string_append_printf(o, "kv[%u]", s->k);
        break;
    default:
        g_string_append_printf(o, "0x%08xu", s->imm);
        break;
    }
}

static unsigned src_mods(const HD5770IRInst *in)
{
    unsigned m = 0, k;

    for (k = 0; k < in->nsrc; k++) {
        m |= (in->src[k].neg ? 1u : 0) << k;
        m |= (in->src[k].abs ? 1u : 0) << (4 + k);
    }
    return m;
}

static void emit_group(GString *o, const HD5770IR *ir, const HD5770IRGroup *g,
                       unsigned q)
{
    unsigned i, k;

    g_string_append(o, "    {\n");
    for (i = 0; i < g->n; i++) {
        const HD5770IRInst *in = &ir->insts[g->first + i];

        if (!in->needed) {
            continue;
        }
        for (k = 0; k < in->nsrc; k++) {
            g_string_append_printf(o, "        uint s%u_%u = ", i, k);
            src_expr(o, &in->src[k], q);
            g_string_append(o, ";\n");
        }
        g_string_append_printf(o, "        uint q%u = 0;\n", i);
    }
    if (g->dot) {
        g_string_append(o, "        float dot = 0.0f;\n");
    }
    for (i = 0; i < g->n; i++) {
        const HD5770IRInst *in = &ir->insts[g->first + i];
        unsigned m = src_mods(in);

        if (!in->needed) {
            continue;
        }
        if (in->dot) {
            g_string_append_printf(o, "        dot = dot + %s(hd_fmod(%uu, 0, s%u_0), "
                                   "hd_fmod(%uu, 1, s%u_1));\n",
                                   in->op == 0xbe ? "hd_mul" : "hd_mulieee",
                                   m, i, m, i);
        } else if (in->interp) {
            g_string_append_printf(o, "        q%u = U(hd_interp(u, t, prm, %u, %u, "
                                   "%s, B0_%u, B1_%u, B2_%u, PW_%u));\n", i,
                                   in->prm, in->dst_chan,
                                   in->flat ? "true" : "false", q, q, q, q);
        } else if (in->max4) {
            continue;
        } else if (in->cube) {
            g_string_append_printf(o, "        q%u = s%u_0;\n", i, i);
        } else {
            g_string_append_printf(o, "        { bool k = false; q%u = hd_eval(0x%xu, "
                                   "%uu", i, in->op, m);
            for (k = 0; k < 3; k++) {
                if (k < in->nsrc) {
                    g_string_append_printf(o, ", s%u_%u", i, k);
                } else {
                    g_string_append(o, ", 0u");
                }
            }
            g_string_append(o, q ? ", k); }\n"
                                 : ", k); if (k) { killed = true; } }\n");
        }
    }
    if (g->dot) {
        for (i = 0; i < g->n; i++) {
            if (ir->insts[g->first + i].dot && ir->insts[g->first + i].needed) {
                g_string_append_printf(o, "        q%u = U(dot);\n", i);
            }
        }
    }
    if (g->max4) {
        g_string_append(o, "        { float m = -FLT_MAX;\n");
        for (i = 0; i < g->n; i++) {
            const HD5770IRInst *in = &ir->insts[g->first + i];

            if (in->max4) {
                g_string_append_printf(o, "          m = hd_max9(m, hd_fmod(%uu, 0, s%u_0));\n",
                                       src_mods(in), i);
            }
        }
        for (i = 0; i < g->n; i++) {
            if (ir->insts[g->first + i].max4) {
                g_string_append_printf(o, "          q%u = U(m);\n", i);
            }
        }
        g_string_append(o, "        }\n");
    }
    if (g->cube) {
        const char *v[3] = { "0.0f", "0.0f", "0.0f" };
        char b[3][48];

        for (i = 0; i < g->n; i++) {
            const HD5770IRInst *in = &ir->insts[g->first + i];
            int at = in->slot == 2 ? 0 : in->slot == 3 ? 1 : in->slot == 0 ? 2 : -1;

            if (in->cube && at >= 0) {
                snprintf(b[at], sizeof(b[at]), "hd_fmod(%uu, 0, s%u_0)",
                         src_mods(in), i);
                v[at] = b[at];
            }
        }
        g_string_append_printf(o, "        { uint co[4]; hd_cube(%s, %s, %s, co);\n",
                               v[0], v[1], v[2]);
        for (i = 0; i < g->n; i++) {
            const HD5770IRInst *in = &ir->insts[g->first + i];

            if (in->cube) {
                g_string_append_printf(o, "          q%u = co[%u];\n", i,
                                       MIN(in->slot, 3));
            }
        }
        g_string_append(o, "        }\n");
    }
    for (i = 0; i < g->n; i++) {
        const HD5770IRInst *in = &ir->insts[g->first + i];

        if (!in->needed) {
            continue;
        }
        if ((in->omod || in->clamp) && !(in->cube && in->slot == 3)) {
            g_string_append_printf(o, "        q%u = hd_post(0x%xu, %uu, %s, %uu, q%u);\n",
                                   i, in->op, src_mods(in),
                                   in->clamp ? "true" : "false", in->omod, i);
        }
        g_string_append_printf(o, "        P%u[%u] = q%u;\n", q, in->slot, i);
    }
    for (i = 0; i < g->n; i++) {
        const HD5770IRInst *in = &ir->insts[g->first + i];

        if (in->wr) {
            g_string_append_printf(o, "        R%u[%u][%u] = q%u;\n", q, in->dst,
                                   in->dst_chan, i);
        }
    }
    g_string_append(o, "    }\n");
}

static void emit_fetch(GString *o, const HD5770IR *ir, const HD5770IRFetch *f,
                       unsigned q, uint32_t mipmask, unsigned fi)
{
    unsigned k, unit = 0;
    bool mip = false;

    if (f->kind == HD5770_IRF_TEX && (f->w[0] & 0x1f) != 0x03 &&
        (f->w[0] & 0x1f) != 0x04) {
        for (unit = 0; unit < ir->nres && ir->res[unit] != f->res; unit++) {
            /* search */
        }
        mip = (mipmask >> unit) & 1;
    }
    if (mip && !q) {
        /*
         * derivatives across the quad from the neighbour sets, read
         * before any set's result lands; the same for all three
         */
        g_string_append_printf(o, "    float DV%u[8] = {", fi);
        for (k = 0; k < 8; k++) {
            unsigned sel = (f->w[2] >> (20 + 3 * (k & 3))) & 7;

            if (sel < 4) {
                g_string_append_printf(o, " F(R%u[%u][%u]) - F(R0[%u][%u])",
                                       k < 4 ? 1 : 2, f->src, sel, f->src, sel);
            } else {
                g_string_append(o, " 0.0f");
            }
            g_string_append(o, k < 7 ? "," : " };\n");
        }
    }
    g_string_append(o, "    {\n        uint o[4] = { 0, 0, 0, 0 };\n");
    if (f->kind == HD5770_IRF_GRAD_H || f->kind == HD5770_IRF_GRAD_V) {
        /* the neighbour minus this pixel; in a neighbour, itself */
        unsigned nb = q ? q : f->kind == HD5770_IRF_GRAD_H ? 1 : 2;

        for (k = 0; k < 4; k++) {
            unsigned sel = (f->w[2] >> (20 + 3 * k)) & 7;

            if (sel < 4) {
                g_string_append_printf(o, "        o[%u] = U(F(R%u[%u][%u]) - "
                                       "F(R%u[%u][%u]));\n", k, nb, f->src, sel,
                                       q, f->src, sel);
            } else {
                g_string_append_printf(o, "        o[%u] = U(0.0f);\n", k);
            }
        }
    } else if (f->kind == HD5770_IRF_VTX) {
        g_string_append_printf(o, "        hd_vfetch(u, st, 0x%08xu, 0x%08xu, 0x%08xu, "
                               "R%u[%u][%u], o);\n", f->w[0], f->w[1], f->w[2],
                               q, f->src, (f->w[0] >> 24) & 3);
    } else {
        for (unit = 0; unit < ir->nres && ir->res[unit] != f->res; unit++) {
            /* search */
        }
        g_string_append(o, "        uint c[4] = {");
        for (k = 0; k < 4; k++) {
            unsigned sel = (f->w[2] >> (20 + 3 * k)) & 7;

            if (sel < 4) {
                g_string_append_printf(o, " R%u[%u][%u]", q, f->src, sel);
            } else {
                g_string_append(o, sel == 5 ? " U(1.0f)" : " U(0.0f)");
            }
            g_string_append(o, k < 3 ? "," : " };\n");
        }
        if (mip) {
            g_string_append_printf(o, "        hd_tfetch_d(vram, ram, st, tex[%u], 0x%08xu, 0x%08xu, "
                                   "0x%08xu, c, DV%u, o);\n", unit, f->w[0], f->w[1],
                                   f->w[2], fi);
        } else {
            g_string_append_printf(o, "        hd_tfetch(vram, ram, st, tex[%u], 0x%08xu, 0x%08xu, "
                                   "0x%08xu, c, o);\n", unit, f->w[0], f->w[1], f->w[2]);
        }
    }
    for (k = 0; k < 4; k++) {
        unsigned sel = (f->w[1] >> (9 + 3 * k)) & 7;

        if (sel < 4) {
            g_string_append_printf(o, "        R%u[%u][%u] = o[%u];\n", q, f->dst,
                                   k, sel);
        } else if (sel == 4 || sel == 5) {
            g_string_append_printf(o, "        R%u[%u][%u] = U(%s);\n", q, f->dst,
                                   k, sel == 5 ? "1.0f" : "0.0f");
        }
    }
    g_string_append(o, "    }\n");
}

/* the exports of an op of the straight-line form */
static void emit_export(GString *o, const HD5770IR *ir, const HD5770IROp *op)
{
    unsigned j, k;

    for (j = 0; j < op->n; j++) {
        const HD5770IRExport *e = &ir->exports[op->first + j];
        char v[4][32];

        for (k = 0; k < 4; k++) {
            unsigned sel = e->sel[k];

            if (sel < 4) {
                snprintf(v[k], sizeof(v[k]), "R0[%u][%u]", e->gpr, sel);
            } else {
                snprintf(v[k], sizeof(v[k]), "%s", sel == 4 ? "U(0.0f)" :
                         sel == 5 ? "U(1.0f)" : "0u");
            }
        }
        if (e->target == HD5770_IR_DEPTH) {
            g_string_append_printf(o, "    zd = %s;\n    sd = %s;\n    zw = true;\n",
                                   v[0], v[1]);
            continue;
        }
        for (k = 0; k < 4; k++) {
            g_string_append_printf(o, "    C[%u][%u] = %s;\n", e->target, k, v[k]);
        }
        g_string_append_printf(o, "    cm |= %uu;\n", 1u << e->target);
    }
}

char *hd5770_msl_program(const HD5770IR *ir)
{
    return hd5770_msl_program_mip(ir, 0);
}

/*
 * The vertex stage after the vertex shader, compiled once: kernel
 * hd_setup builds the triangles of a draw from its records as
 * assemble(), emit() and gpu_tri() in ati_hd5770_draw.c do, into fixed
 * slots (k per emit, in order; an empty slot has x1 < x0).
 */
const char hd5770_msl_setup[] =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct GTri {\n"
"    float x[3], y[3], w[3];\n"
"    float area;\n"
"    int x0, y0, x1, y1;\n"
"    float z[3];\n"
"    uint front;\n"
"};\n"
"struct GSetup {\n"
"    uint nemit, nvtx, rec_words, k;\n"
"    uint prims_off, rec_off, tri_off, prm_off;\n"
"    uint prm_st, ninputs, flat, vte;\n"
"    uint sc_mode, wofs, clip_on, clip_dx;\n"
"    uint clip_near, clip_far, cull, face_cw;\n"
"    uint prov_last, zfmt;\n"
"    int poly_bits;\n"
"    float pix_center;\n"
"    int x0, y0, x1, y1;\n"
"    float vport[6];\n"
"    float poly_scale[2], poly_units[2];\n"
"    float poly_clamp;\n"
"    uint np;\n"
"    int src[32];\n"
"    float def[32][4];\n"
"};\n"
"struct SV {\n"
"    float cp[4];\n"
"    float x, y, z, w;\n"
"    bool clip;\n"
"    float in[32][4];\n"
"};\n"
"\n"
"/* vtx_window() */\n"
"static void sv_window(device const GSetup &s, thread SV &v)\n"
"{\n"
"    float x = v.cp[0], y = v.cp[1], z = v.cp[2], w = v.cp[3];\n"
"\n"
"    if (!(s.vte & (1u << 8))) {\n"
"        if (w == 0.0f) {\n"
"            w = 1.0f;\n"
"        }\n"
"        x /= w;\n"
"        y /= w;\n"
"        z /= w;\n"
"        w = 1.0f / w;\n"
"    } else if (!(s.vte & (1u << 10))) {\n"
"        w = w != 0.0f ? 1.0f / w : 1.0f;\n"
"    }\n"
"    if (s.vte & 1) {\n"
"        x *= s.vport[0];\n"
"    }\n"
"    if (s.vte & 2) {\n"
"        x += s.vport[1];\n"
"    }\n"
"    if (s.vte & 4) {\n"
"        y *= s.vport[2];\n"
"    }\n"
"    if (s.vte & 8) {\n"
"        y += s.vport[3];\n"
"    }\n"
"    if (s.vte & 16) {\n"
"        z *= s.vport[4];\n"
"    }\n"
"    if (s.vte & 32) {\n"
"        z += s.vport[5];\n"
"    }\n"
"    if (s.sc_mode & (1u << 16)) {\n"
"        x += (float)(short)(s.wofs & 0xffff);\n"
"        y += (float)(short)(s.wofs >> 16);\n"
"    }\n"
"    v.x = x;\n"
"    v.y = y;\n"
"    v.z = z;\n"
"    v.w = w;\n"
"}\n"
"\n"
"/* clip_dist() */\n"
"static void sv_dist(device const GSetup &s, thread const SV &v, thread float &dn,\n"
"                    thread float &df)\n"
"{\n"
"    dn = s.clip_dx ? v.cp[2] : v.cp[2] + v.cp[3];\n"
"    df = v.cp[3] - v.cp[2];\n"
"    if (!s.clip_near) {\n"
"        dn = 0.0f;\n"
"    }\n"
"    if (!s.clip_far) {\n"
"        df = 0.0f;\n"
"    }\n"
"}\n"
"\n"
"/* the vertex shader's record u as the pixel inputs see it */\n"
"static void sv_load(device const GSetup &s, device const uchar *sc, uint u, thread SV &v)\n"
"{\n"
"    device const float *r = (device const float *)(sc + s.rec_off) + (ulong)u * s.rec_words;\n"
"    float dn, df;\n"
"\n"
"    for (uint k = 0; k < 4; k++) {\n"
"        v.cp[k] = r[k];\n"
"    }\n"
"    for (uint p = 0; p < s.ninputs; p++) {\n"
"        for (uint k = 0; k < 4; k++) {\n"
"            v.in[p][k] = s.src[p] < 0 ? s.def[p][k]\n"
"                         : s.src[p] < (int)s.np ? r[4 + s.src[p] * 4 + k] : 0.0f;\n"
"        }\n"
"    }\n"
"    sv_dist(s, v, dn, df);\n"
"    v.clip = s.clip_on && (dn < 0.0f || df < 0.0f);\n"
"    sv_window(s, v);\n"
"}\n"
"\n"
"static float sv_edge(float ax, float ay, float bx, float by, float px, float py)\n"
"{\n"
"    if (ax > bx || (ax == bx && ay > by)) {\n"
"        return -fma(ax - bx, py - by, -((ay - by) * (px - bx)));\n"
"    }\n"
"    return fma(bx - ax, py - ay, -((by - ay) * (px - ax)));\n"
"}\n"
"\n"
"/* clip_lerp() */\n"
"static void sv_lerp(device const GSetup &s, thread const SV &a, thread const SV &b,\n"
"                    float f, thread SV &o)\n"
"{\n"
"    for (uint k = 0; k < 4; k++) {\n"
"        o.cp[k] = fma(b.cp[k] - a.cp[k], f, a.cp[k]);\n"
"    }\n"
"    for (uint p = 0; p < s.ninputs; p++) {\n"
"        for (uint k = 0; k < 4; k++) {\n"
"            o.in[p][k] = fma(b.in[p][k] - a.in[p][k], f, a.in[p][k]);\n"
"        }\n"
"    }\n"
"    o.clip = false;\n"
"    sv_window(s, o);\n"
"}\n"
"\n"
"/* emit_tri() with poly_offset() and gpu_tri() into slot n */\n"
"static void sv_tri(device const GSetup &s, device uchar *sc, uint n,\n"
"                   thread const SV &a, thread const SV &b, thread const SV &c,\n"
"                   thread const SV &tvp)\n"
"{\n"
"    device GTri &t = *(device GTri *)(sc + s.tri_off + (ulong)n * 128);\n"
"    float area = sv_edge(a.x, a.y, b.x, b.y, c.x, c.y), zoff = 0.0f, pc = s.pix_center;\n"
"    bool cw = area > 0.0f;\n"
"    bool front = s.face_cw ? cw : !cw;\n"
"\n"
"    if (s.cull & (front ? 1u : 2u)) {\n"
"        return;\n"
"    }\n"
"    if (s.sc_mode & (front ? 1u << 11 : 1u << 12)) {\n"
"        uint side = front ? 0 : 1;\n"
"        float ar = fma(b.x - a.x, c.y - a.y, -((c.x - a.x) * (b.y - a.y)));\n"
"        float dzx = 0.0f, dzy = 0.0f, o;\n"
"\n"
"        if (ar != 0.0f) {\n"
"            dzx = fma(b.z - a.z, c.y - a.y, -((c.z - a.z) * (b.y - a.y))) / ar;\n"
"            dzy = fma(c.z - a.z, b.x - a.x, -((b.z - a.z) * (c.x - a.x))) / ar;\n"
"        }\n"
"        o = fma(s.poly_scale[side] / 16.0f, fmax(fabs(dzx), fabs(dzy)),\n"
"                s.poly_units[side] * ldexp(s.zfmt == 1 ? 0.25f : s.zfmt == 2 ? 0.5f : 1.0f,\n"
"                                           s.poly_bits));\n"
"        if (s.poly_clamp > 0.0f) {\n"
"            o = fmin(o, s.poly_clamp);\n"
"        } else if (s.poly_clamp < 0.0f) {\n"
"            o = fmax(o, s.poly_clamp);\n"
"        }\n"
"        zoff = o == o ? o : 0.0f;\n"
"    }\n"
"    /* gpu_tri() */\n"
"    if (area == 0.0f || area != area) {\n"
"        return;\n"
"    }\n"
"    thread const SV *v0 = &a, *v1 = &b, *v2 = &c;\n"
"    if (area < 0.0f) {\n"
"        v1 = &c;\n"
"        v2 = &b;\n"
"        area = -area;\n"
"    }\n"
"    int x0 = max(s.x0, (int)floor(min(a.x, min(b.x, c.x)) - pc));\n"
"    int x1 = min(s.x1 - 1, (int)ceil(max(a.x, max(b.x, c.x)) - pc));\n"
"    int y0 = max(s.y0, (int)floor(min(a.y, min(b.y, c.y)) - pc));\n"
"    int y1 = min(s.y1 - 1, (int)ceil(max(a.y, max(b.y, c.y)) - pc));\n"
"    if (x0 > x1 || y0 > y1) {\n"
"        return;\n"
"    }\n"
"    thread const SV *vv[3] = { v0, v1, v2 };\n"
"    for (uint q = 0; q < 3; q++) {\n"
"        t.x[q] = vv[q]->x;\n"
"        t.y[q] = vv[q]->y;\n"
"        t.w[q] = vv[q]->w;\n"
"        t.z[q] = vv[q]->z + zoff;\n"
"    }\n"
"    t.area = area;\n"
"    t.front = front;\n"
"    device float *prm = (device float *)(sc + s.prm_off + (ulong)n * s.prm_st);\n"
"    for (uint p = 0; p < s.ninputs; p++) {\n"
"        for (uint q = 0; q < 3; q++) {\n"
"            thread const SV *vq = ((s.flat >> p) & 1) ? &tvp : vv[q];\n"
"\n"
"            for (uint k = 0; k < 4; k++) {\n"
"                prm[(p * 3 + q) * 4 + k] = vq->in[p][k];\n"
"            }\n"
"        }\n"
"    }\n"
"    /* the box last: the slot counts once it is set */\n"
"    t.x0 = x0;\n"
"    t.y0 = y0;\n"
"    t.x1 = x1;\n"
"    t.y1 = y1;\n"
"}\n"
"\n"
"/*\n"
" * Emit e of the draw, as emit() runs it: entry (a, b, c, pv, kind) of\n"
" * the table; kind 1: the second half of a rectangle, (a, b, a + b - c)\n"
" */\n"
"kernel void hd_setup(device const uchar *st [[buffer(1)]],\n"
"                     device uchar *sc [[buffer(2)]],\n"
"                     device const GSetup &s [[buffer(4)]],\n"
"                     uint gid [[thread_position_in_grid]])\n"
"{\n"
"    if (gid >= s.nemit) {\n"
"        return;\n"
"    }\n"
"    device const uint *e = (device const uint *)(st + s.prims_off) + gid * 5;\n"
"    uint base = gid * s.k;\n"
"    SV a, b, c, pv;\n"
"\n"
"    for (uint i = 0; i < s.k; i++) {\n"
"        device GTri &t = *(device GTri *)(sc + s.tri_off + (ulong)(base + i) * 128);\n"
"        t.x0 = 1;\n"
"        t.x1 = 0;\n"
"    }\n"
"    sv_load(s, sc, e[0], a);\n"
"    sv_load(s, sc, e[1], b);\n"
"    sv_load(s, sc, e[2], c);\n"
"    sv_load(s, sc, e[3], pv);\n"
"    if (e[4] == 1) {\n"
"        /* v3 = v0 + v2 - v1, as a copy of v2 */\n"
"        SV v3 = b;\n"
"\n"
"        v3.x = a.x + b.x - c.x;\n"
"        v3.y = a.y + b.y - c.y;\n"
"        v3.z = a.z + b.z - c.z;\n"
"        for (uint p = 0; p < s.ninputs; p++) {\n"
"            for (uint k = 0; k < 4; k++) {\n"
"                v3.in[p][k] = a.in[p][k] + b.in[p][k] - c.in[p][k];\n"
"            }\n"
"        }\n"
"        c = v3;\n"
"    }\n"
"    thread const SV &tvp = s.prov_last ? pv : a;\n"
"    if (!a.clip && !b.clip && !c.clip) {\n"
"        sv_tri(s, sc, base, a, b, c, tvp);\n"
"        return;\n"
"    }\n"
"    SV buf[6];\n"
"    uint src[9], dst[9], n = 3, nb = 0;\n"
"    /* 0..2: a, b, c; 3..: buf */\n"
"    src[0] = 0;\n"
"    src[1] = 1;\n"
"    src[2] = 2;\n"
"    for (uint plane = 0; plane < 2 && n >= 3; plane++) {\n"
"        float dist[9], dn, df;\n"
"        uint m = 0;\n"
"\n"
"        for (uint i = 0; i < n; i++) {\n"
"            thread const SV &v = src[i] == 0 ? a : src[i] == 1 ? b : src[i] == 2 ? c : buf[src[i] - 3];\n"
"            sv_dist(s, v, dn, df);\n"
"            dist[i] = plane ? df : dn;\n"
"        }\n"
"        for (uint i = 0; i < n; i++) {\n"
"            uint j = (i + 1) % n;\n"
"\n"
"            if (dist[i] >= 0.0f) {\n"
"                dst[m++] = src[i];\n"
"            }\n"
"            if ((dist[i] >= 0.0f) != (dist[j] >= 0.0f) && nb < 6 && m < 9) {\n"
"                thread const SV &vi = src[i] == 0 ? a : src[i] == 1 ? b : src[i] == 2 ? c : buf[src[i] - 3];\n"
"                thread const SV &vj = src[j] == 0 ? a : src[j] == 1 ? b : src[j] == 2 ? c : buf[src[j] - 3];\n"
"                sv_lerp(s, vi, vj, dist[i] / (dist[i] - dist[j]), buf[nb]);\n"
"                dst[m++] = 3 + nb++;\n"
"            }\n"
"        }\n"
"        n = m;\n"
"        for (uint i = 0; i < n; i++) {\n"
"            src[i] = dst[i];\n"
"        }\n"
"    }\n"
"    for (uint i = 1; i + 1 < n && i <= s.k; i++) {\n"
"        thread const SV &v0 = src[0] == 0 ? a : src[0] == 1 ? b : src[0] == 2 ? c : buf[src[0] - 3];\n"
"        thread const SV &v1 = src[i] == 0 ? a : src[i] == 1 ? b : src[i] == 2 ? c : buf[src[i] - 3];\n"
"        thread const SV &v2 = src[i + 1] == 0 ? a : src[i + 1] == 1 ? b : src[i + 1] == 2 ? c : buf[src[i + 1] - 3];\n"
"        sv_tri(s, sc, base + i - 1, v0, v1, v2, tvp);\n"
"    }\n"
"}\n";

/* the library with its table filled in */
static GString *msl_head(void)
{
    GString *o = g_string_new(hd5770_msl_lib);
    GString *tab = g_string_new("constant float hd_u8[256] = {");
    const char *mark = strstr(o->str, "HD_U8_TABLE");
    unsigned i;

    for (i = 0; i < 256; i++) {
        float f = i / (float)((1u << 8) - 1);
        uint32_t b;

        memcpy(&b, &f, 4);
        g_string_append_printf(tab, "%sas_type<float>(0x%08xu)", i ? ", " : "", b);
    }
    g_string_append(tab, "};");
    if (mark) {
        size_t at = mark - o->str;

        g_string_erase(o, at, strlen("HD_U8_TABLE"));
        g_string_insert(o, at, tab->str);
    }
    g_string_free(tab, true);
    g_string_append(o,
        "static float hd_mulieee(float a, float b) { return a * b; }\n");
    return o;
}

/*
 * A vertex shader (hd5770_vs_ir()) as kernel hd_vs, pass A of
 * HD5770GpuVs: thread n runs slot n % nslot of instance n / nslot and
 * writes record n (out + n * rec_words). Buffers: 0 VRAM, 1 staging
 * (GVsDraw at buffer 4's offset, the constants, the constant file, the
 * page tables, immediate indices), 2 the records, 3 guest RAM.
 */
static const char msl_vs_lib[] =
"struct GBuf {\n"
"    uint voff, space, pt_off, pg0;\n"
"};\n"
"struct GVsVb {\n"
"    uint id, pad0, pad1, pad2;\n"
"    uint res[8];\n"
"    GBuf b;\n"
"};\n"
"struct GVsDraw {\n"
"    uint nslot, ninst, rec_words, np;\n"
"    uint kv_off, cfile_off, nvb, base_vtx;\n"
"    uint start_inst, idx_kind, ib_i32, ib_swap;\n"
"    uint ib_max, ioff, lo, hi;\n"
"    uint restart, reset, imm_off, pad;\n"
"    GBuf ib;\n"
"    GVsVb vb[8];\n"
"};\n"
"/* the byte at offset o of buffer b: in VRAM, or through its page table */\n"
"static __attribute__((always_inline)) device const uchar *hd_bat(device const GBuf &b, uint o,\n"
"    device const uchar *vram, device const uchar *ram, device const uchar *st)\n"
"{\n"
"    if (b.space) {\n"
"        uint q = b.pg0 + o;\n"
"        device const uint *pt = (device const uint *)(st + b.pt_off);\n"
"\n"
"        return ram + pt[q >> 12] + (q & 4095);\n"
"    }\n"
"    return vram + b.voff + o;\n"
"}\n"
"/* n <= 4 bytes little endian at offset o of buffer b */\n"
"static __attribute__((always_inline)) uint hd_brd(device const GBuf &b, uint o, uint n,\n"
"    device const uchar *vram, device const uchar *ram, device const uchar *st)\n"
"{\n"
"    uint v = 0;\n"
"\n"
"    if (n == 4 && !(o & 3)) {\n"
"        return *(device const uint *)hd_bat(b, o, vram, ram, st);\n"
"    }\n"
"    for (uint i = 0; i < n; i++) {\n"
"        v |= (uint)*hd_bat(b, o + i, vram, ram, st) << (8 * i);\n"
"    }\n"
"    return v;\n"
"}\n"
"/* vfetch_vs(): the fetch type's offset, then vfetch_stage() */\n"
"static __attribute__((noinline)) void hd_vsfetch(device const GVsDraw &v,\n"
"    device const uchar *vram, device const uchar *ram, device const uchar *st,\n"
"    uint w0, uint w1, uint w2, uint index, uint fsb, thread uint *o)\n"
"{\n"
"    uint bid = (w0 >> 8) & 0xff, off = w2 & 0xffff, fmt, num, sgn, n, k;\n"
"    uint c[4], w[4];\n"
"\n"
"    if (((w0 >> 5) & 3) == 0) {\n"
"        index += v.base_vtx;\n"
"    } else if (((w0 >> 5) & 3) == 1) {\n"
"        index += v.start_inst;\n"
"    }\n"
"    if (bid == 128) {\n"
"        uint e = index + off / 16;\n"
"        device const uint *cf = (device const uint *)(st + v.cfile_off);\n"
"\n"
"        for (k = 0; k < 4; k++) {\n"
"            o[k] = e < 256 ? cf[e * 4 + k] : 0;\n"
"        }\n"
"        return;\n"
"    }\n"
"    o[0] = o[1] = o[2] = U(0.0f);\n"
"    o[3] = U(1.0f);\n"
"    for (k = 0; k < v.nvb && v.vb[k].id != bid + fsb; k++) {\n"
"    }\n"
"    if (k == v.nvb) {\n"
"        return;\n"
"    }\n"
"    device const GVsVb &b = v.vb[k];\n"
"    uint size = b.res[1] + 1, stride = (b.res[2] >> 8) & 0x7ff;\n"
"    if ((w1 >> 21) & 1) {\n"
"        fmt = (b.res[2] >> 20) & 0x3f;\n"
"        num = (b.res[2] >> 26) & 3;\n"
"        sgn = (b.res[2] >> 28) & 1;\n"
"    } else {\n"
"        fmt = (w1 >> 22) & 0x3f;\n"
"        num = (w1 >> 28) & 3;\n"
"        sgn = (w1 >> 30) & 1;\n"
"    }\n"
"    n = hd_fmt_bytes(fmt);\n"
"    ulong at = (ulong)index * stride + off;\n"
"    if (at + n > size) {\n"
"        o[0] = o[1] = o[2] = 0;\n"
"        return;\n"
"    }\n"
"    for (k = 0; k < 4; k++) {\n"
"        w[k] = 4 * k < n ? hd_brd(b.b, (uint)at + 4 * k, min(n - 4 * k, 4u), vram, ram, st) : 0;\n"
"    }\n"
"    c[0] = c[1] = c[2] = c[3] = 0;\n"
"    if (n > 8) {\n"
"        for (k = 0; k < 4 && hd_fmt_bits[fmt][k]; k++) {\n"
"            c[k] = w[k];\n"
"        }\n"
"    } else {\n"
"        ulong x = (ulong)w[0] | (ulong)w[1] << 32;\n"
"        uint sh = 0;\n"
"\n"
"        for (k = 0; k < 4 && hd_fmt_bits[fmt][k]; k++) {\n"
"            uint bb = hd_fmt_bits[fmt][k];\n"
"\n"
"            c[k] = (uint)((x >> sh) & (bb == 32 ? 0xffffffffull : (1ull << bb) - 1));\n"
"            sh += bb;\n"
"        }\n"
"    }\n"
"    for (k = 0; k < 4 && hd_fmt_bits[fmt][k]; k++) {\n"
"        o[k] = hd_comp(c[k], hd_fmt_bits[fmt][k], hd_fmt_float(fmt), sgn, num);\n"
"    }\n"
"}\n"
"/* the vertex index of slot s, as hd5770_gfx_draw(); false: a restart */\n"
"static bool hd_vsindex(device const GVsDraw &v, device const uchar *vram,\n"
"    device const uchar *ram, device const uchar *st, uint s, thread uint &x)\n"
"{\n"
"    uint raw;\n"
"\n"
"    if (v.idx_kind == 0) {\n"
"        x = s + v.ioff;\n"
"        return true;\n"
"    }\n"
"    if (v.idx_kind == 2) {\n"
"        raw = ((device const uint *)(st + v.imm_off))[s];\n"
"    } else if (s >= v.ib_max) {\n"
"        raw = 0;\n"
"    } else if (v.ib_i32) {\n"
"        raw = hd_brd(v.ib, s * 4, 4, vram, ram, st);\n"
"        if (v.ib_swap == 1) {\n"
"            raw = ((raw & 0x00ff00ffu) << 8) | ((raw >> 8) & 0x00ff00ffu);\n"
"        } else if (v.ib_swap == 2) {\n"
"            raw = (raw << 24) | ((raw << 8) & 0xff0000u) | ((raw >> 8) & 0xff00u) | (raw >> 24);\n"
"        } else if (v.ib_swap == 3) {\n"
"            raw = (raw << 16) | (raw >> 16);\n"
"        }\n"
"    } else {\n"
"        raw = hd_brd(v.ib, s * 2, 2, vram, ram, st);\n"
"        if (v.ib_swap == 1) {\n"
"            raw = ((raw & 0xffu) << 8) | (raw >> 8);\n"
"        }\n"
"    }\n"
"    if (v.restart && raw == v.reset) {\n"
"        return false;\n"
"    }\n"
"    x = min(max(raw + v.ioff, v.lo), v.hi);\n"
"    return true;\n"
"}\n";

char *hd5770_msl_vs_program(const HD5770IR *ir)
{
    GString *o = msl_head();
    unsigned i, j, k, ng = MAX(ir->ngpr, 1u), np = hd5770_vs_np(ir);

    g_string_append(o, msl_vs_lib);
    g_string_append_printf(o,
        "kernel void hd_vs(device const uchar *vram [[buffer(0)]],\n"
        "    device const uchar *st [[buffer(1)]], device uint *out [[buffer(2)]],\n"
        "    device const uchar *ram [[buffer(3)]], device const GVsDraw &v [[buffer(4)]],\n"
        "    uint gid [[thread_position_in_grid]])\n"
        "{\n"
        "    if (gid >= v.nslot * v.ninst) {\n        return;\n    }\n"
        "    device const uint *kv = (device const uint *)(st + v.kv_off);\n"
        "    uint inst = gid / v.nslot, index = 0, pm = 0, fl = 0;\n"
        "    device uint *rec = out + (ulong)gid * %u;\n"
        "    for (uint i = 0; i < %u; i++) {\n        rec[i] = 0;\n    }\n"
        "    if (!hd_vsindex(v, vram, ram, st, gid %% v.nslot, index)) {\n"
        "        rec[%u] = %uu;\n        return;\n    }\n"
        "    bool killed = false;\n"
        "    uint R0[%u][4], P0[5] = { 0, 0, 0, 0, 0 };\n"
        "    for (uint i = 0; i < %u; i++) {\n"
        "        R0[i][0] = R0[i][1] = R0[i][2] = R0[i][3] = 0;\n"
        "    }\n"
        "    R0[%u][0] = index;\n    R0[%u][3] = inst;\n",
        HD5770_VS_REC(np), HD5770_VS_REC(np), HD5770_VS_REC(np) - 1,
        HD5770_VSF_RESTART, ng, ng, ir->map[0], ir->map[0]);
    for (i = 0; i < ir->nops; i++) {
        const HD5770IROp *op = &ir->ops[i];

        switch (op->kind) {
        case HD5770_IRO_ALU:
            for (j = 0; j < op->n; j++) {
                emit_group(o, ir, &ir->groups[op->first + j], 0);
            }
            break;
        case HD5770_IRO_FETCH:
            for (j = 0; j < op->n; j++) {
                const HD5770IRFetch *f = &ir->fetches[op->first + j];

                g_string_append_printf(o,
                    "    {\n        uint o[4] = { 0, 0, 0, 0 };\n"
                    "        hd_vsfetch(v, vram, ram, st, 0x%08xu, 0x%08xu, 0x%08xu, "
                    "R0[%u][%u], %uu, o);\n",
                    f->w[0], f->w[1], f->w[2], f->src, (f->w[0] >> 24) & 3,
                    f->fs ? 160 : 0);
                for (k = 0; k < 4; k++) {
                    unsigned sel = (f->w[1] >> (9 + 3 * k)) & 7;

                    if (sel < 4) {
                        g_string_append_printf(o, "        R0[%u][%u] = o[%u];\n",
                                               f->dst, k, sel);
                    } else if (sel == 4 || sel == 5) {
                        g_string_append_printf(o, "        R0[%u][%u] = U(%s);\n",
                                               f->dst, k, sel == 5 ? "1.0f" : "0.0f");
                    }
                }
                g_string_append(o, "    }\n");
            }
            break;
        default:
            for (j = 0; j < op->n; j++) {
                const HD5770IRExport *e = &ir->exports[op->first + j];
                bool pos = e->target == HD5770_IR_POS;
                unsigned base = pos ? 0 : 4 + 4 * (e->target - HD5770_IR_PARAM0);

                for (k = 0; k < 4; k++) {
                    unsigned sel = e->sel[k];

                    if (sel < 4) {
                        g_string_append_printf(o, "    rec[%u] = R0[%u][%u];\n",
                                               base + k, e->gpr, sel);
                    } else if (sel == 4 || sel == 5) {
                        g_string_append_printf(o, "    rec[%u] = U(%s);\n",
                                               base + k, sel == 5 ? "1.0f" : "0.0f");
                    }
                }
                if (pos) {
                    g_string_append_printf(o, "    fl = %uu;\n", HD5770_VSF_POS);
                } else {
                    g_string_append_printf(o, "    pm |= %uu;\n",
                                           1u << (e->target - HD5770_IR_PARAM0));
                }
            }
            break;
        }
    }
    g_string_append_printf(o, "    rec[%u] = pm;\n    rec[%u] = fl;\n}\n",
                           HD5770_VS_REC(np) - 2, HD5770_VS_REC(np) - 1);
    return g_string_free(o, false);
}

char *hd5770_msl_program_mip(const HD5770IR *ir, uint32_t mipmask)
{
    GString *o = msl_head();
    unsigned i, j, q, nq = ir->grad || mipmask ? 3 : 1;
    unsigned ng = MAX(ir->ngpr, 1u);

    g_string_append(o,
        "/* raster_tri(): pixel (x, y) of the triangle's box and covered */\n"
        "static bool hd_cover(device const GTri &t, int x, int y, float px, float py,\n"
        "                     thread float &w0, thread float &w1, thread float &w2)\n"
        "{\n"
        "    if (x < t.x0 || x > t.x1 || y < t.y0 || y > t.y1) {\n        return false;\n    }\n"
        "    w0 = hd_edge(t.x[1], t.y[1], t.x[2], t.y[2], px, py);\n"
        "    w1 = hd_edge(t.x[2], t.y[2], t.x[0], t.y[0], px, py);\n"
        "    w2 = hd_edge(t.x[0], t.y[0], t.x[1], t.y[1], px, py);\n"
        "    if (w0 < 0 || w1 < 0 || w2 < 0) {\n        return false;\n    }\n"
        "    return !((w0 == 0 && !hd_owns(t.x[1], t.y[1], t.x[2], t.y[2])) ||\n"
        "             (w1 == 0 && !hd_owns(t.x[2], t.y[2], t.x[0], t.y[0])) ||\n"
        "             (w2 == 0 && !hd_owns(t.x[0], t.y[0], t.x[1], t.y[1])));\n"
        "}\n");
    g_string_append(o,
        "/* resolve_pixel() */\n"
        "static void hd_resolve(device uchar *vram, device uchar *ram, device const uchar *st,\n"
        "                       device const GDraw &u, int x, int y)\n"
        "{\n"
        "    device uchar *sp = hd_at(u.rs, hd_addr(u.rs, x, y), vram, ram, st);\n"
        "    uint c[4], col[4] = { U(0.0f), U(0.0f), U(0.0f), U(1.0f) };\n"
        "\n"
        "    if (u.rs_same) {\n"
        "        device uchar *p = hd_at(u.rt, hd_addr(u.rt, x, y), vram, ram, st);\n"
        "\n"
        "        for (uint i = 0; i < u.rt.bpe; i++) {\n"
        "            p[i] = sp[i];\n"
        "        }\n"
        "        return;\n"
        "    }\n"
        "    hd_unpack(u.rs_fmt, sp, 0, c);\n"
        "    for (uint i = 0; i < u.rs_comps && i < 4; i++) {\n"
        "        col[u.rs_src[i]] = hd_comp(c[i], hd_fmt_bits[u.rs_fmt][i], u.rs_num == 7, false,\n"
        "                                   u.rs_num == 4 ? 1 : 0);\n"
        "    }\n"
        "    hd_put(vram, ram, st, u, x, y, col);\n"
        "}\n");
    if (ir->cf) {
        g_string_append(o,
            "#define HD_CTX_PARAMS device uchar *vram, device uchar *ram, device const uchar *st, \\\n"
            "    device const GDraw &u, device const GTri &t, device const float *prm, \\\n"
            "    device const GTex *tex, device const uint *kv\n"
            "#define HD_CTX_ARGS vram, ram, st, u, t, prm, tex, kv\n"
            "#define HD_K(k) kv[k]\n"
            "#define HD_BCONST u.bconst\n"
            "#define HD_LCONST(i) u.lconst[i]\n"
            "#define HD_POSGPR u.pos_gpr\n"
            "#define HD_TEX(n, w0, w1, w2, c, dv, o) hd_tfetch_d(vram, ram, st, tex[n], w0, w1, w2, c, dv, o)\n"
            "#define HD_ZERO 0\n"
            "#define HD_VTX(w0, w1, w2, i, o) hd_vfetch(u, st, w0, w1, w2, i, o)\n"
            "#define HD_INTERP(p, ch, fl, b0, b1, b2, pw) hd_interp(u, t, prm, p, ch, fl, b0, b1, b2, pw)\n"
            "#define HD_OUTA(n, a, b) thread uint (&n)[a][b]\n"
            "#define HD_OUTBA(n, a) thread bool (&n)[a]\n"
            "#define HD_OUTB(n) thread bool &n\n"
            "#define HD_OUTU(n) thread uint &n\n"
            "#define HD_OUTC(n) thread uint (&n)[8][4]\n");
        hd5770_emit_cf(o, ir, mipmask, true);
    }
    g_string_append(o,
        "/* the pixel (x, y) of triangle t: false if t does not cover it */\n"
        "static __attribute__((noinline)) bool hd_pixel(device uchar *vram, device uchar *ram, device const uchar *st,\n"
        "                     device const GDraw &u, device const GTri &t, device const float *prm,\n"
        "                     device const GTex *tex, device const uint *kv, int x, int y)\n"
        "{\n"
        "    float pc = u.pix_center, px = x + pc, py = y + pc;\n"
        "    float w0, w1, w2;\n"
        "    if (!hd_cover(t, x, y, px, py, w0, w1, w2)) {\n        return false;\n    }\n"
        "    if (u.cliprule != 0xffff && !hd_clip(u, x, y)) {\n        return true;\n    }\n"
        "    if (u.resolve) {\n        hd_resolve(vram, ram, st, u, x, y);\n        return true;\n    }\n"
        "    uint C[8][4];\n"
        "    for (uint i = 0; i < 8; i++) {\n"
        "        C[i][0] = C[i][1] = C[i][2] = C[i][3] = 0;\n"
        "    }\n"
        "    uint cm = 0, zd = 0, sd = 0;\n"
        "    bool killed = false, zw = false;\n");
    for (q = 0; q < nq; q++) {
        /* bary(): barycentrics at the set's pixel */
        g_string_append_printf(o,
            "    float PX_%u = (x + %d) + pc, PY_%u = (y + %d) + pc;\n"
            "    float B0_%u = hd_edge(t.x[1], t.y[1], t.x[2], t.y[2], PX_%u, PY_%u) / t.area;\n"
            "    float B1_%u = hd_edge(t.x[2], t.y[2], t.x[0], t.y[0], PX_%u, PY_%u) / t.area;\n"
            "    float B2_%u = hd_edge(t.x[0], t.y[0], t.x[1], t.y[1], PX_%u, PY_%u) / t.area;\n"
            "    float PW_%u = 0.0f;\n"
            "    if (u.persp) {\n"
            "        float s = fma(B2_%u, t.w[2], fma(B0_%u, t.w[0], B1_%u * t.w[1]));\n"
            "        PW_%u = s != 0.0f ? 1.0f / s : 0.0f;\n"
            "    }\n",
            q, q == 1, q, q == 2, q, q, q, q, q, q, q, q, q, q, q, q, q, q);
        if (ir->cf) {
            continue;
        }
        g_string_append_printf(o,
            "    uint R%u[%u][4], P%u[5] = { 0, 0, 0, 0, 0 };\n"
            "    for (uint i = 0; i < %u; i++) {\n"
            "        R%u[i][0] = R%u[i][1] = R%u[i][2] = R%u[i][3] = 0;\n"
            "    }\n", q, ng, q, ng, q, q, q, q);
        /* registers are indexed by constants only, so they stay registers */
        for (i = 0; i < ir->ngpr; i++) {
            g_string_append_printf(o,
                "    if (u.pos_gpr == %u) {\n"
                "        R%u[%u][0] = U(PX_%u);\n        R%u[%u][1] = U(PY_%u);\n"
                "        R%u[%u][2] = U(0.0f);\n        R%u[%u][3] = U(1.0f);\n"
                "    }\n", i, q, i, q, q, i, q, q, i, q, i);
        }
        g_string_append_printf(o, "    R%u[%u][0] = U(B1_%u);\n"
                               "    R%u[%u][1] = U(B2_%u);\n",
                               q, ir->map[0], q, q, ir->map[0], q);
    }
    /*
     * the depth test before the program when its answer cannot change: no
     * depth export, stencil or clear; the test after the program writes
     */
    if (!ir->zexport) {
        g_string_append(o,
            "    if (u.db && u.zfmt && !u.has_s && !(u.rctl & 1) && (u.dctl & 2) &&\n"
            "        !hd_ztest(vram, ram, st, u, x, y,\n"
            "                  fma(t.z[2], B2_0, fma(t.z[0], B0_0, t.z[1] * B1_0)))) {\n"
            "        return true;\n    }\n");
    }
    if (ir->cf) {
        const char *n1 = nq == 3 ? "B0_1, B1_1, B2_1, PW_1, PX_1, PY_1"
                                 : "0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f";
        const char *n2 = nq == 3 ? "B0_2, B1_2, B2_2, PW_2, PX_2, PY_2"
                                 : "0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f";

        g_string_append_printf(o,
            "    hd_run(HD_CTX_ARGS, B0_0, B1_0, B2_0, PW_0, PX_0, PY_0,\n"
            "           %s,\n           %s,\n"
            "           killed, C, cm, zd, sd, zw);\n", n1, n2);
    }
    for (i = 0; i < ir->nops && !ir->cf; i++) {
        const HD5770IROp *op = &ir->ops[i];

        switch (op->kind) {
        case HD5770_IRO_ALU:
            for (j = 0; j < op->n; j++) {
                for (q = 0; q < nq; q++) {
                    emit_group(o, ir, &ir->groups[op->first + j], q);
                }
            }
            break;
        case HD5770_IRO_FETCH:
            for (j = 0; j < op->n; j++) {
                for (q = 0; q < nq; q++) {
                    emit_fetch(o, ir, &ir->fetches[op->first + j], q, mipmask,
                               op->first + j);
                }
            }
            break;
        default:
            emit_export(o, ir, op);
            break;
        }
    }
    g_string_append(o,
        "    if (killed) {\n        return true;\n    }\n"
        "    if ((cm & 1) && u.alpha_test && !hd_alpha(u, F(C[0][3]))) {\n        return true;\n    }\n"
        "    if (u.db) {\n"
        "        float z = zw && u.zexp ? F(zd)\n"
        "                  : fma(t.z[2], B2_0, fma(t.z[0], B0_0, t.z[1] * B1_0));\n"
        "        if (!hd_db(vram, ram, st, u, x, y, z, zw && u.sexp ? (int)(sd & 0xff) : -1,\n"
        "                   t.front != 0)) {\n            return true;\n        }\n"
        "    }\n"
        "    if (u.nmrt) {\n"
        "        for (uint k = 0; k < u.nmrt; k++) {\n"
        "            device const GDraw &m = *(device const GDraw *)(st + u.mrt_off + k * sizeof(GDraw));\n"
        "            if ((cm >> m.cb) & 1) {\n"
        "                hd_put(vram, ram, st, m, x, y, C[m.cb]);\n"
        "            }\n"
        "        }\n"
        "        return true;\n"
        "    }\n"
        "    if (!(cm & 1) || !u.wmask) {\n        return true;\n    }\n"
        "    hd_put(vram, ram, st, u, x, y, C[0]);\n"
        "    return true;\n"
        "}\n"
        "kernel void hd_main(device uchar *vram [[buffer(0)]],\n"
        "    device const uchar *st [[buffer(1)]], device const uint *tiles [[buffer(2)]],\n"
        "    device uchar *ram [[buffer(3)]],\n"
        "    uint tg [[threadgroup_position_in_grid]],\n"
        "    uint l [[thread_index_in_threadgroup]])\n"
        "{\n"
        "    device const uint *te = tiles + tg * 8;\n"
        "    device const GDraw &u = *(device const GDraw *)(st + te[0]);\n"
        "    device const GTex *tex = (device const GTex *)(st + te[3]);\n"
        "    device const uint *kv = (device const uint *)(st + te[4]);\n"
        "    uint sw = te[7] & 0xff;\n"
        "    int x = (int)te[5] + (int)(l % sw), y = (int)te[6] + (int)(l / sw);\n"
        "    if (te[7] & 0x200) {\n"
        "        /* a tile of the screen: the triangles of the draw over it, in order */\n"
        "        device const uint *bin = (device const uint *)(st + te[1]);\n"
        "        for (uint k = 0; k < te[2]; k++) {\n"
        "            uint i = bin[k];\n"
        "            hd_pixel(vram, ram, st, u, *(device const GTri *)(st + u.tri_off + i * 128),\n"
        "                     (device const float *)(st + u.prm_off + i * u.prm_st), tex, kv, x, y);\n"
        "        }\n"
        "        return;\n"
        "    }\n"
        "    device const GTri *tp = (device const GTri *)(st + te[1]);\n"
        "    device const float *prm = (device const float *)(st + te[2]);\n"
        "    if (!hd_pixel(vram, ram, st, u, *tp, prm, tex, kv, x, y) && (te[7] & 0x100)) {\n"
        "        /* the other half of the primitive; they share no pixel */\n"
        "        hd_pixel(vram, ram, st, u, *(device const GTri *)(st + te[1] + 128),\n"
        "                 prm + (te[7] >> 16) * 4, tex, kv, x, y);\n"
        "    }\n"
        "}\n");
    return g_string_free(o, false);
}

/*
 * The native pipeline (ati_hd5770_mtlr.m). hd5770_msl_alu() is the part
 * of hd5770_msl_lib a native program needs: the prelude and the ALU
 * functions, without formats, surfaces, fetches, rasterisation and the
 * output stage.
 */
const char *hd5770_msl_alu(void)
{
    static char *alu;

    if (!alu) {
        const char *a = strstr(hd5770_msl_lib, "/* layouts shared");
        const char *b = strstr(hd5770_msl_lib, "/* exp2f() rounded");
        const char *c = strstr(hd5770_msl_lib, "/* ---- fetches ---- */");
        GString *o = g_string_new_len(hd5770_msl_lib, a - hd5770_msl_lib);

        g_string_append_len(o, b, c - b);
        g_string_append(o,
            "static float hd_mulieee(float a, float b) { return a * b; }\n");
        alu = g_string_free(o, false);
    }
    return alu;
}

/* slices of level l of a unit */
static unsigned nat_slices(const HD5770NatKey *k, unsigned unit, unsigned l)
{
    unsigned n = MAX(k->nsl[unit], 1u);

    return k->dim[unit] == 2 ? MAX(n >> l, 1u) : n;
}

/* the textures of a unit: all levels and slices; 1: its own alone */
static unsigned nat_ntex(const HD5770NatKey *k, unsigned unit)
{
    unsigned l, n = 0;

    for (l = 0; l <= k->mip[unit]; l++) {
        n += nat_slices(k, unit, l);
    }
    return n;
}

/* the textures of a unit as a parameter or as an argument */
static void nat_tlist(GString *o, const HD5770NatKey *k, unsigned unit,
                      bool decl)
{
    unsigned n = nat_ntex(k, unit);

    if (n > 1) {
        if (decl) {
            g_string_append_printf(o, ", array<texture2d<float>, %u> a%u", n,
                                   unit);
        } else {
            g_string_append_printf(o, ", a%u", unit);
        }
    } else {
        g_string_append_printf(o, ", %st%u", decl ? "texture2d<float> " : "",
                               unit);
    }
}

/* a sample of level lv at tc of a unit, in its function hd_nt */
static void nat_call(GString *o, const HD5770NatKey *k, unsigned unit,
                     const char *lv, const char *tc, const char *lin)
{
    g_string_append_printf(o, "%s%u(u", k->dim[unit] == 2 ? "hd_nz" : "hd_ns",
                           unit);
    nat_tlist(o, k, unit, false);
    g_string_append_printf(o, ", %s, %s, %s, %s, cmp, ref)", lv,
                           k->dim[unit] == 2 ? "zn, zl" : "sl", tc, lin);
}

/*
 * The functions of a unit: hd_ns, a level and slice sampled at a place
 * in its texels; hd_nt, a fetch by its words as tfetch_stage() in
 * ati_hd5770_draw.c does it. slot: the unit's first entry in u.lsize.
 */
static void nat_unit(GString *o, const HD5770NatKey *k, unsigned unit,
                     unsigned slot)
{
    unsigned l, ns, dim = k->dim[unit], n0 = MAX(k->nsl[unit], 1u);
    char tn[32], ls[40];

    g_string_append_printf(o, "static float4 hd_ns%u(constant NU &u", unit);
    nat_tlist(o, k, unit, true);
    g_string_append_printf(o,
        ", uint lv, uint sl, float2 tc, bool lin, uint cmp, float ref)\n"
        "{\n"
        "    float4 ls = u.tsize[%u];\n", unit);
    if (nat_ntex(k, unit) > 1) {
        g_string_append(o, "    uint ix = sl;\n");
        if (k->mip[unit]) {
            g_string_append_printf(o,
                "    lv = min(lv, %uu);\n"
                "    if (lv != 0u) {\n"
                "        ls = u.lsize[%u + lv - 1u];\n"
                "    }\n", k->mip[unit], slot);
            if (dim == 2) {
                g_string_append(o, "    switch (lv) {\n");
                for (l = 1, ns = 0; l <= k->mip[unit]; l++) {
                    ns += nat_slices(k, unit, l - 1);
                    g_string_append_printf(o, "    case %u: ix += %uu; break;\n",
                                           l, ns);
                }
                g_string_append(o, "    default: break;\n    }\n");
            } else {
                g_string_append_printf(o, "    ix += lv * %uu;\n", n0);
            }
        }
        g_string_append_printf(o, "    ix = min(ix, %uu);\n",
                               nat_ntex(k, unit) - 1);
        snprintf(tn, sizeof(tn), "a%u[ix]", unit);
    } else {
        snprintf(tn, sizeof(tn), "t%u", unit);
    }
    g_string_append_printf(o,
        "    return hd_nsamp(%s, ls, tc, u.twrap[%u], lin, u.tborder[%u],"
        " u.tsel[%u], cmp, ref, u.tcmp[%u].y);\n}\n", tn, unit, unit, unit,
        unit);
    if (dim == 2) {
        /* texel(): the nearest slice of the level, or between two */
        g_string_append_printf(o, "static float4 hd_nz%u(constant NU &u", unit);
        nat_tlist(o, k, unit, true);
        g_string_append_printf(o,
            ", uint lv, float zn, bool zl, float2 tc, bool lin, uint cmp, float ref)\n"
            "{\n"
            "    uint ns = max(%uu >> lv, 1u);\n"
            "    if (zl && ns > 1u) {\n"
            "        float zf = zn * float(ns) - 0.5f;\n"
            "        int z0 = int(floor(zf)), z1 = z0 + 1;\n"
            "        float a = zf - float(z0);\n"
            "        if (u.tcmp[%u].w == 0u) {\n"
            "            z0 = (z0 + int(ns)) %% int(ns);\n"
            "            z1 = z1 %% int(ns);\n"
            "        } else {\n"
            "            z0 = max(z0, 0);\n"
            "            z1 = min(z1, int(ns) - 1);\n"
            "        }\n", n0, unit);
        g_string_append_printf(o, "        float4 va = hd_ns%u(u", unit);
        nat_tlist(o, k, unit, false);
        g_string_append_printf(o, ", lv, uint(z0), tc, lin, cmp, ref);\n"
                                  "        float4 vb = hd_ns%u(u", unit);
        nat_tlist(o, k, unit, false);
        g_string_append_printf(o, ", lv, uint(z1), tc, lin, cmp, ref);\n"
                                  "        return va + (vb - va) * a;\n"
                                  "    }\n"
                                  "    return hd_ns%u(u", unit);
        nat_tlist(o, k, unit, false);
        g_string_append(o, ", lv, min(uint(zn * float(ns)), ns - 1u), tc, lin, cmp, ref);\n}\n");
    }
    g_string_append_printf(o, "static void hd_nt%u(constant NU &u", unit);
    nat_tlist(o, k, unit, true);
    g_string_append_printf(o,
        ", uint w0, uint w1, uint w2, thread uint (&c)[4], float4 dv,\n"
        "    thread uint (&o)[4])\n"
        "{\n"
        "    uint op = w0 & 0x1fu, cmp = 0u, sl = 0u;\n"
        "    int ox = int(w2 << 27u) >> 27, oy = int(w2 << 22u) >> 27;\n"
        "    float4 ts = u.tsize[%u], v = float4(0.0f);\n"
        "    float ref = 0.0f, c0 = F(c[0]), c1 = F(c[1]), zn = 0.0f;\n"
        "    bool zl = false;\n"
        "    o[0] = 0u;\n    o[1] = 0u;\n    o[2] = 0u;\n    o[3] = 0u;\n"
        "    if (op >= 0x18u && op <= 0x1bu) {\n"
        "        cmp = u.tcmp[%u].x;\n"
        "        ref = op == 0x19u || op == 0x1au ? F(c[2]) : F(c[3]);\n"
        "        op -= 8u;\n"
        "    }\n"
        "    if (op == 4u) {\n"
        "        uint lv = min(c[0] + u.tres[%u].x, 31u);\n"
        "        o[0] = max(uint(ts.x) >> lv, 1u);\n"
        "        o[1] = max(uint(ts.y) >> lv, 1u);\n"
        "        o[2] = u.tres[%u].z;\n"
        "        o[3] = u.tres[%u].y;\n"
        "        return;\n"
        "    }\n", unit, unit, unit, unit, unit);
    /* tex_slice() */
    if (dim == 3) {
        g_string_append(o,
            "    {\n        float f = F(c[2]);\n"
            "        sl = f > 0.0f ? min(uint(f), 5u) : 0u;\n"
            "        c0 -= 1.0f;\n        c1 -= 1.0f;\n    }\n");
    } else if (dim == 5) {
        g_string_append_printf(o,
            "    {\n        int ly = op == 3u ? I(c[2]) : int(rint(F(c[2])));\n"
            "        sl = uint(clamp(ly, 0, %d));\n    }\n", (int)n0 - 1);
    } else if (dim == 2) {
        g_string_append_printf(o,
            "    zn = op == 3u ? (float(I(c[2])) + 0.5f) / %u.0f :\n"
            "         ((w1 >> 30u) & 1u) != 0u ? F(c[2]) : F(c[2]) / %u.0f;\n"
            "    if (u.tcmp[%u].w == 0u) {\n        zn -= floor(zn);\n    }\n"
            "    zn = zn > 0.0f ? min(zn, 1.0f) : 0.0f;\n"
            "    zl = u.tcmp[%u].z != 0u && op != 3u;\n", n0, n0, unit, unit);
    }
    g_string_append(o,
        "    if (op == 3u) {\n"
        "        int2 xy = int2(I(c[0]) + ox / 2, I(c[1]) + oy / 2);\n"
        "        if (xy.x >= 0 && xy.y >= 0 && xy.x < int(ts.x) && xy.y < int(ts.y)) {\n"
        "            v = ");
    nat_call(o, k, unit, "0u", "float2(xy) + 0.5f", "false");
    g_string_append(o, ";\n        }\n    } else {\n"
        "        bool nx = ((w1 >> 28u) & 1u) != 0u, ny = ((w1 >> 29u) & 1u) != 0u;\n"
        "        float2 of = float2(float(ox), float(oy)) * 0.5f;\n");
    if (k->mip[unit]) {
        g_string_append_printf(o,
            "        if (nx && ny) {\n"
            "            float lod = 0.0f;\n"
            "            float2 cn = float2(c0, c1);\n"
            "            if (op == 0x11u) {\n"
            "                lod = F(c[3]);\n"
            "            } else if (op != 0x13u) {\n"
            "                float2 da = dv.xy * ts.xy, db = dv.zw * ts.xy;\n"
            "                lod = 0.5f * hd_nlog2(max(dot(da, da), dot(db, db)));\n"
            "                if (op == 0x12u) {\n"
            "                    lod += F(c[3]);\n"
            "                }\n"
            "                lod += float(int(w1 << 4u) >> 25) / 16.0f;\n"
            "            }\n"
            "            lod = min(max(lod + u.tlod[%u].x, u.tlod[%u].y), u.tlod[%u].z);\n"
            "            if (!(lod > 0.0f)) {\n"
            "                v = ", unit, unit, unit);
        snprintf(ls, sizeof(ls), "u.tmip[%u].z != 0u", unit);
        nat_call(o, k, unit, "0u", "cn * ts.xy + of", ls);
        g_string_append_printf(o, ";\n"
            "            } else {\n"
            "                bool lin = u.tmip[%u].w != 0u;\n"
            "                uint last = u.tmip[%u].y;\n"
            "                float l0 = u.tmip[%u].x == 1u ? floor(lod + 0.5f) : floor(lod);\n"
            "                uint lv = min(uint(l0), last);\n"
            "                float4 ls = ts;\n"
            "                if (lv != 0u) {\n"
            "                    ls = u.lsize[%u + lv - 1u];\n"
            "                }\n"
            "                v = ", unit, unit, unit, slot);
        nat_call(o, k, unit, "lv", "cn * ls.xy + of", "lin");
        g_string_append_printf(o, ";\n"
            "                if (u.tmip[%u].x != 1u && lv < last) {\n"
            "                    ls = u.lsize[%u + lv];\n"
            "                    float4 v1 = ", unit, slot);
        nat_call(o, k, unit, "lv + 1u", "cn * ls.xy + of", "lin");
        g_string_append(o, ";\n"
            "                    v = v + (v1 - v) * (lod - l0);\n"
            "                }\n"
            "            }\n"
            "        } else\n");
    }
    g_string_append(o,
        "        {\n"
        "            float2 tc = float2(nx ? c0 * ts.x : c0, ny ? c1 * ts.y : c1) + of;\n"
        "            v = ");
    snprintf(ls, sizeof(ls), "u.twrap[%u].z != 0u", unit);
    nat_call(o, k, unit, "0u", "tc", ls);
    g_string_append(o, ";\n        }\n    }\n"
        "    o[0] = U(v.x);\n    o[1] = U(v.y);\n    o[2] = U(v.z);\n    o[3] = U(v.w);\n}\n");
}

/* the TEX operations taken: LD, RESINFO, SAMPLE*, SAMPLE_C* */
static bool nat_tex_op(unsigned op)
{
    return op == 0x03 || op == 0x04 || (op >= 0x10 && op <= 0x13) ||
           (op >= 0x18 && op <= 0x1b);
}

/* the fetch goes through its unit's function */
static bool nat_unit_fn(const HD5770IR *ir, const HD5770NatKey *k,
                        const HD5770IRFetch *f)
{
    unsigned unit, op = f->w[0] & 0x1f;

    if (f->kind != HD5770_IRF_TEX) {
        return false;
    }
    for (unit = 0; unit < ir->nres && ir->res[unit] != f->res; unit++) {
        /* search */
    }
    return ir->cf || op >= 0x18 ||
           (unit < HD5770_GPU_TEX && k->nsl[unit] > 1);
}

/*
 * a TEX fetch as a sample of the unit's host texture; a VTX fetch as
 * vfetch_stage() in ati_hd5770_draw.c reads an 8 or 32 bit element
 */
static bool nat_fetch(GString *o, const HD5770IR *ir, const HD5770IRFetch *f,
                      const HD5770NatKey *key)
{
    unsigned k, unit, op = f->w[0] & 0x1f;
    int ox = ((int32_t)(f->w[2] << 27)) >> 27;
    int oy = ((int32_t)(f->w[2] << 22)) >> 27;

    g_string_append(o, "    {\n        uint o[4] = { 0, 0, 0, 0 };\n");
    if (f->kind == HD5770_IRF_GRAD_H || f->kind == HD5770_IRF_GRAD_V) {
        for (k = 0; k < 4; k++) {
            unsigned sel = (f->w[2] >> (20 + 3 * k)) & 7;

            if (sel < 4) {
                g_string_append_printf(o, "        o[%u] = U(%s(F(R0[%u][%u])));\n",
                                       k, f->kind == HD5770_IRF_GRAD_H ?
                                       "dfdx" : "dfdy", f->src, sel);
            } else {
                g_string_append_printf(o, "        o[%u] = U(0.0f);\n", k);
            }
        }
    } else if (f->kind == HD5770_IRF_VTX) {
        unsigned bid = (f->w[0] >> 8) & 0xff, fmt, num, sgn, n;

        for (k = 0; k < key->nvb && key->vb[k].id != bid; k++) {
            /* search */
        }
        if (k == key->nvb) {
            return false;
        }
        if ((f->w[1] >> 21) & 1) {
            fmt = key->vb[k].fmt;
            num = key->vb[k].num;
            sgn = key->vb[k].sgn;
        } else {
            fmt = (f->w[1] >> 22) & 0x3f;
            num = (f->w[1] >> 28) & 3;
            sgn = (f->w[1] >> 30) & 1;
        }
        if (fmt != 0x01 && fmt != 0x0d) {
            return false;
        }
        n = fmt == 0x01 ? 1 : 4;
        g_string_append_printf(o,
            "        uint ix = R0[%u][%u], st = u.vbi[%u].x, ln = u.vbi[%u].y;\n"
            "        o[3] = U(1.0f);\n"
            "        if ((st == 0u || ix <= ln / st) && ix * st + %uu <= ln) {\n"
            "            uint at = ix * st + %uu;\n",
            f->src, (f->w[0] >> 24) & 3, k, k, (f->w[2] & 0xffff) + n,
            f->w[2] & 0xffff);
        if (n == 1) {
            g_string_append_printf(o, "            uint c = vb%u[at];\n", k);
            if (sgn) {
                g_string_append(o, "            c = uint(int(c << 24) >> 24);\n");
            }
        } else {
            g_string_append_printf(o,
                "            uint c = uint(vb%u[at]) | uint(vb%u[at + 1u]) << 8 |\n"
                "                     uint(vb%u[at + 2u]) << 16 | uint(vb%u[at + 3u]) << 24;\n",
                k, k, k, k);
        }
        if (num == 1) {
            g_string_append(o, "            o[0] = c;\n");
        } else if (num != 0) {
            g_string_append_printf(o, "            o[0] = U(float(%sc));\n",
                                   sgn ? "(int)" : "");
        } else if (n == 4) {
            g_string_append_printf(o, "            o[0] = U(float(%sc) / %s);\n",
                                   sgn ? "(int)" : "",
                                   sgn ? "2147483647.0f" : "4294967295.0f");
        } else if (sgn) {
            g_string_append(o, "            o[0] = U(max(float((int)c) / 127.0f, -1.0f));\n");
        } else {
            g_string_append(o, "            o[0] = U(float(c) / 255.0f);\n");
        }
        g_string_append(o, "        }\n");
    } else if (f->kind != HD5770_IRF_TEX || !nat_tex_op(op)) {
        return false;
    } else if (nat_unit_fn(ir, key, f)) {
        for (unit = 0; unit < ir->nres && ir->res[unit] != f->res; unit++) {
            /* search */
        }
        g_string_append(o, "        uint c[4] = {");
        for (k = 0; k < 4; k++) {
            unsigned sel = (f->w[2] >> (20 + 3 * k)) & 7;

            if (sel < 4) {
                g_string_append_printf(o, " R0[%u][%u]", f->src, sel);
            } else {
                g_string_append(o, sel == 5 ? " U(1.0f)" : " U(0.0f)");
            }
            g_string_append(o, k < 3 ? "," : " };\n");
        }
        g_string_append_printf(o,
            "        float2 cn = float2(F(c[0]), F(c[1]));\n"
            "        hd_nt%u(u", unit);
        nat_tlist(o, key, unit, false);
        g_string_append_printf(o, ", 0x%08xu, 0x%08xu, 0x%08xu, c,"
                               " float4(dfdx(cn), dfdy(cn)), o);\n",
                               f->w[0], f->w[1], f->w[2]);
    } else if (op == 0x04) {
        /* GET_TEXTURE_RESINFO, as tfetch_stage() */
        unsigned sel = (f->w[2] >> 20) & 7;

        for (unit = 0; unit < ir->nres && ir->res[unit] != f->res; unit++) {
            /* search */
        }
        if (sel < 4) {
            g_string_append_printf(o, "        uint lv = min(R0[%u][%u] + u.tres[%u].x, 31u);\n",
                                   f->src, sel, unit);
        } else {
            g_string_append_printf(o, "        uint lv = min(%s + u.tres[%u].x, 31u);\n",
                                   sel == 5 ? "U(1.0f)" : "0u", unit);
        }
        g_string_append_printf(o,
            "        o[0] = max(uint(u.tsize[%u].x) >> lv, 1u);\n"
            "        o[1] = max(uint(u.tsize[%u].y) >> lv, 1u);\n"
            "        o[2] = u.tres[%u].z;\n"
            "        o[3] = u.tres[%u].y;\n", unit, unit, unit, unit);
    } else {
        unsigned slot = 0;

        for (unit = 0; unit < ir->nres && ir->res[unit] != f->res; unit++) {
            slot += key->mip[unit];
        }
        g_string_append(o, "        uint c[4] = {");
        for (k = 0; k < 4; k++) {
            unsigned sel = (f->w[2] >> (20 + 3 * k)) & 7;

            if (sel < 4) {
                g_string_append_printf(o, " R0[%u][%u]", f->src, sel);
            } else {
                g_string_append(o, sel == 5 ? " U(1.0f)" : " U(0.0f)");
            }
            g_string_append(o, k < 3 ? "," : " };\n");
        }
        g_string_append_printf(o, "        float4 ts = u.tsize[%u], v;\n", unit);
        if (op == 0x03) {
            g_string_append_printf(o,
                "        int2 xy = int2(I(c[0]) + %d, I(c[1]) + %d);\n"
                "        v = float4(0.0f);\n"
                "        if (xy.x >= 0 && xy.y >= 0 && xy.x < (int)ts.x && xy.y < (int)ts.y) {\n"
                "            v = hd_nsel(t%u.read(uint2(xy)), u.tsel[%u]);\n"
                "        }\n", ox / 2, oy / 2, unit, unit);
        } else if (key->mip[unit] && ((f->w[1] >> 28) & 3) == 3) {
            /* the level of detail and the levels, as tfetch_stage() */
            g_string_append(o, "        float2 cn = float2(F(c[0]), F(c[1]));\n"
                               "        float lod = 0.0f;\n");
            if (op == 0x11) {
                g_string_append(o, "        lod = F(c[3]);\n");
            } else if (op != 0x13) {
                g_string_append(o,
                    "        float2 da = dfdx(cn) * ts.xy, db = dfdy(cn) * ts.xy;\n"
                    "        lod = 0.5f * hd_nlog2(max(dot(da, da), dot(db, db)));\n");
                if (op == 0x12) {
                    g_string_append(o, "        lod += F(c[3]);\n");
                }
                if ((f->w[1] >> 21) & 0x7f) {
                    g_string_append_printf(o, "        lod += %d.0f / 16.0f;\n",
                                           (int32_t)(f->w[1] << 4) >> 25);
                }
            }
            g_string_append_printf(o,
                "        lod = min(max(lod + u.tlod[%u].x, u.tlod[%u].y), u.tlod[%u].z);\n"
                "        uint lv = 0, last = u.tmip[%u].y;\n"
                "        bool lin = u.tmip[%u].z != 0u, two = false;\n"
                "        float fr = 0.0f;\n"
                "        if (lod > 0.0f) {\n"
                "            lin = u.tmip[%u].w != 0u;\n"
                "            if (u.tmip[%u].x == 1u) {\n"
                "                lv = min(uint(floor(lod + 0.5f)), last);\n"
                "            } else {\n"
                "                float l0 = floor(lod);\n"
                "                fr = lod - l0;\n"
                "                lv = min(uint(l0), last);\n"
                "                two = lv < last;\n"
                "            }\n"
                "        }\n"
                "        float4 v1 = float4(0.0f);\n"
                "        float2 of = float2(%d.0f, %d.0f) * 0.5f;\n",
                unit, unit, unit, unit, unit, unit, unit, ox, oy);
            for (k = 0; k < 2; k++) {
                g_string_append_printf(o,
                    "        %s{\n"
                    "            uint l_ = min(lv%s, %uu);\n"
                    "            float4 ls_ = ts;\n"
                    "            if (l_ != 0u) {\n"
                    "                ls_ = u.lsize[%u + l_ - 1u];\n"
                    "            }\n"
                    "            %s = hd_nlv(a%u[l_], ls_, cn, of, u.twrap[%u], lin,"
                    " u.tborder[%u], u.tsel[%u]);\n"
                    "        }\n", k ? "if (two) " : "", k ? " + 1u" : "",
                    key->mip[unit], slot, k ? "v1" : "v", unit, unit, unit,
                    unit);
            }
            g_string_append(o, "        if (two) {\n            v = v + (v1 - v) * fr;\n        }\n");
        } else {
            g_string_append_printf(o,
                "        float2 tc = float2(F(c[0])%s, F(c[1])%s);\n"
                "        bool bo = false;\n"
                "        tc = tc + float2(%d.0f, %d.0f) * 0.5f;\n"
                "        tc.x = hd_nwrap(tc.x, ts.x, u.twrap[%u].x, bo);\n"
                "        tc.y = hd_nwrap(tc.y, ts.y, u.twrap[%u].y, bo);\n"
                "        v = t%u.sample(s%u, tc / ts.zw);\n"
                "        v = bo ? u.tborder[%u] : hd_nsel(v, u.tsel[%u]);\n",
                (f->w[1] >> 28) & 1 ? " * ts.x" : "",
                (f->w[1] >> 29) & 1 ? " * ts.y" : "", ox, oy, unit, unit, unit,
                unit, unit, unit);
        }
        g_string_append(o, "        o[0] = U(v.x);\n        o[1] = U(v.y);\n"
                           "        o[2] = U(v.z);\n        o[3] = U(v.w);\n");
    }
    for (k = 0; k < 4; k++) {
        unsigned sel = (f->w[1] >> (9 + 3 * k)) & 7;

        if (sel < 4) {
            g_string_append_printf(o, "        R0[%u][%u] = o[%u];\n", f->dst, k, sel);
        } else if (sel == 4 || sel == 5) {
            g_string_append_printf(o, "        R0[%u][%u] = U(%s);\n", f->dst, k,
                                   sel == 5 ? "1.0f" : "0.0f");
        }
    }
    g_string_append(o, "    }\n");
    return true;
}

char *hd5770_msl_native_fs(const HD5770IR *ir, const HD5770NatKey *k)
{
    GString *o;
    unsigned i, j, ng = MAX(ir->ngpr, 1u), nlv = 0, ntx = 0;
    unsigned slot[HD5770_GPU_TEX];
    uint32_t iflat = 0, ivar = 0, ufn = 0;
    bool mips = false;

    if (ir->vs || k->ninputs > 32 || k->nvb > HD5770_GPU_VB ||
        ir->nres > HD5770_GPU_TEX) {
        return NULL;
    }
    for (i = 0; i < ir->nres; i++) {
        slot[i] = nlv;
        nlv += k->mip[i];
        mips |= k->mip[i] != 0;
        ntx += nat_ntex(k, i) > 1 ? nat_ntex(k, i) : 0;
    }
    for (i = 0; i < ir->nfetches; i++) {
        const HD5770IRFetch *f = &ir->fetches[i];

        if (ir->cf && (f->kind != HD5770_IRF_TEX ||
                       !nat_tex_op(f->w[0] & 0x1f))) {
            /* the CF form: texture fetches only, no gradients */
            return NULL;
        }
        if (nat_unit_fn(ir, k, f)) {
            for (j = 0; j < ir->nres && ir->res[j] != f->res; j++) {
                /* search */
            }
            if (j == ir->nres) {
                return NULL;
            }
            ufn |= 1u << j;
        }
    }
    for (i = 0; i < ir->nres; i++) {
        if (k->nsl[i] > 1 && !((ufn >> i) & 1)) {
            /* slices are reached through the unit's function only */
            ufn |= 1u << i;
        }
    }
    if (nlv > HD5770_NAT_LEVELS || ntx > HD5770_NAT_LEVELS ||
        ((mips || ufn) && ir->nres > 14)) {
        return NULL;
    }
    for (i = 0; i < ir->ninsts; i++) {
        const HD5770IRInst *in = &ir->insts[i];

        if (in->interp && in->needed && in->prm < k->ninputs) {
            *(in->flat ? &iflat : &ivar) |= 1u << in->prm;
        }
    }
    if (iflat & ivar & ~k->flat) {
        return NULL;
    }
    o = g_string_new(hd5770_msl_alu());
    g_string_append_printf(o,
        "struct NU {\n"
        "    float4 tsize[16];\n"
        "    uint4 tsel[16];\n"
        "    uint4 twrap[16];\n"
        "    float4 tborder[16];\n"
        "    float4 vp;\n"
        "    uint4 vbi[8];\n"
        "    int4 cr[4];\n"
        "    float alpha_ref;\n"
        "    uint alpha_func;\n"
        "    float pc;\n"
        "    uint persp;\n"
        "    uint cliprule;\n"
        "    uint pad0, pad1, pad2;\n"
        "    float4 tlod[16];\n"
        "    uint4 tmip[16];\n"
        "    uint4 tres[16];\n"
        "    float4 lsize[%u];\n"
        "    float4 zr;\n"
        "    uint4 tcmp[16];\n"
        "    uint lconst[32];\n"
        "    uint bconst, zero, pad3, pad4;\n"
        "};\n"
        "struct NV {\n    float4 pos [[position]];\n", HD5770_NAT_LEVELS);
    for (i = 0; i < k->ninputs; i++) {
        bool fl = ((k->flat | iflat) >> i) & 1;

        g_string_append_printf(o, "    %s p%u [[user(p%u)%s]];\n",
                               fl ? "uint4" : "float4", i, i, fl ? ", flat" : "");
    }
    g_string_append(o, "};\nstruct NF {\n    float4 pos [[position]];\n");
    for (i = 0; i < k->ninputs; i++) {
        bool fl = ((k->flat | iflat) >> i) & 1;

        g_string_append_printf(o, "    %s p%u [[user(p%u)%s]];\n",
                               fl ? "uint4" : "float4", i, i, fl ? ", flat" : "");
    }
    g_string_append(o,
        "    float3 bc [[barycentric_coord, center_no_perspective]];\n};\n");
    /* the vertices: window x, y, z, 1/w, then the inputs */
    g_string_append_printf(o,
        "vertex NV hd_nvs(uint vid [[vertex_id]], const device float *vb [[buffer(0)]],\n"
        "                 constant NU &u [[buffer(1)]])\n"
        "{\n"
        "    const device float *v = vb + vid * %uu;\n"
        "    float cw = u.persp ? 1.0f / v[3] : 1.0f;\n"
        "    NV r;\n"
        "    r.pos = float4(((v[0] - u.pc + 0.5f) * u.vp.x - 1.0f) * cw,\n"
        "                   (1.0f - (v[1] - u.pc + 0.5f) * u.vp.y) * cw, v[2] * cw, cw);\n",
        4 + 4 * k->ninputs);
    for (i = 0; i < k->ninputs; i++) {
        bool fl = ((k->flat | iflat) >> i) & 1;

        g_string_append_printf(o, "    r.p%u = %sfloat4(v[%u], v[%u], v[%u], v[%u])%s;\n",
                               i, fl ? "as_type<uint4>(" : "", 4 + 4 * i,
                               5 + 4 * i, 6 + 4 * i, 7 + 4 * i, fl ? ")" : "");
    }
    g_string_append(o, "    return r;\n}\n");
    for (i = 0; i < 32; i++) {
        bool fl = ((k->flat | iflat) >> i) & 1;

        if (i >= k->ninputs) {
            g_string_append_printf(o, "#define HD_NI_%u(ch) 0.0f\n", i);
        } else {
            g_string_append_printf(o, "#define HD_NI_%u(ch) %s(in.p%u[ch])\n", i,
                                   fl ? "F" : "", i);
        }
    }
    g_string_append(o,
        "#define hd_interp(u, t, prm, p, ch, fl, b0, b1, b2, pw) HD_NI_##p(ch)\n"
        "static __attribute__((always_inline)) float4 hd_nsel(float4 v, uint4 s)\n"
        "{\n"
        "    float4 r;\n"
        "    for (uint k = 0; k < 4; k++) {\n"
        "        r[k] = s[k] < 4 ? v[s[k]] : s[k] == 5 ? 1.0f : 0.0f;\n"
        "    }\n"
        "    return r;\n"
        "}\n"
        "/* hd_wrap() on a coordinate in texels of a w wide texture */\n"
        "static __attribute__((always_inline)) float hd_nwrap(float t, float w, uint m, thread bool &out)\n"
        "{\n"
        "    switch (m) {\n"
        "    case 0: t = t - floor(t / w) * w; break;\n"
        "    case 1: {\n"
        "        float p = t - floor(t / (2.0f * w)) * 2.0f * w;\n"
        "        t = p < w ? p : 2.0f * w - p;\n"
        "        break;\n"
        "    }\n"
        "    case 2: break;\n"
        "    case 3: case 5: case 7: t = fabs(t); break;\n"
        "    default: if (!(t >= 0.0f && t < w)) { out = true; } break;\n"
        "    }\n"
        "    return clamp(t, 0.5f, w - 0.5f);\n"
        "}\n"
        "/* wrap() in ati_hd5770_draw.c */\n"
        "static __attribute__((always_inline)) int hd_niwrap(int v, int size, uint m, thread bool &out)\n"
        "{\n"
        "    switch (m) {\n"
        "    case 0:\n"
        "        v %= size;\n"
        "        if (v < 0) {\n            v += size;\n        }\n"
        "        break;\n"
        "    case 1: {\n"
        "        int p = v % (2 * size);\n"
        "        if (p < 0) {\n            p += 2 * size;\n        }\n"
        "        v = p < size ? p : 2 * size - 1 - p;\n"
        "        break;\n"
        "    }\n"
        "    case 2: v = clamp(v, 0, size - 1); break;\n"
        "    case 3: case 5: case 7: v = clamp(v < 0 ? -v - 1 : v, 0, size - 1); break;\n"
        "    default:\n"
        "        if (v < 0 || v >= size) {\n            out = true;\n            v = 0;\n        }\n"
        "        break;\n"
        "    }\n"
        "    return v;\n"
        "}\n"
        "static __attribute__((always_inline)) bool hd_nalpha(uint fn, float r, float a)\n"
        "{\n"
        "    switch (fn) {\n"
        "    case 0: return false;\n"
        "    case 1: return a < r;\n"
        "    case 2: return a == r;\n"
        "    case 3: return a <= r;\n"
        "    case 4: return a > r;\n"
        "    case 5: return a != r;\n"
        "    case 6: return a >= r;\n"
        "    default: return true;\n"
        "    }\n"
        "}\n"
        "/* lod_log2() in ati_hd5770_draw.c */\n"
        "static __attribute__((always_inline)) float hd_nlog2(float x)\n"
        "{\n"
        "    if (!(x > 0.0f)) {\n"
        "        return -128.0f;\n"
        "    }\n"
        "    uint b = as_type<uint>(x);\n"
        "    return float(int((b >> 23) & 0xffu) - 127) + float(b & 0x7fffffu) * (1.0f / 8388608.0f);\n"
        "}\n");
    if (mips || ufn) {
        g_string_append(o,
            "constexpr sampler hd_sn(coord::normalized, address::clamp_to_edge, filter::nearest);\n"
            "constexpr sampler hd_sl(coord::normalized, address::clamp_to_edge, filter::linear);\n"
            "/* a level: its size and its texture's in ls, cn normalised */\n"
            "static __attribute__((always_inline)) float4 hd_nlv(texture2d<float> t, float4 ls, float2 cn,\n"
            "    float2 of, uint4 wrap, bool lin, float4 border, uint4 sel)\n"
            "{\n"
            "    float2 tc = cn * ls.xy + of;\n"
            "    bool bo = false;\n"
            "    tc.x = hd_nwrap(tc.x, ls.x, wrap.x, bo);\n"
            "    tc.y = hd_nwrap(tc.y, ls.y, wrap.y, bo);\n"
            "    float4 v;\n"
            "    if (lin) {\n"
            "        v = t.sample(hd_sl, tc / ls.zw, level(0.0f));\n"
            "    } else {\n"
            "        v = t.sample(hd_sn, tc / ls.zw, level(0.0f));\n"
            "    }\n"
            "    return bo ? border : hd_nsel(v, sel);\n"
            "}\n");
    }
    if (ufn) {
        g_string_append(o,
            "/* a texel compared with the reference, as texel_z() */\n"
            "static float4 hd_ncmp1(texture2d<float> t, float4 ls, int2 p, uint4 wrap, float4 border,\n"
            "                       uint4 sel, uint cmp, float ref)\n"
            "{\n"
            "    bool bo = false, pass = false;\n"
            "    int x = hd_niwrap(p.x, int(ls.x), wrap.x, bo);\n"
            "    int y = hd_niwrap(p.y, int(ls.y), wrap.y, bo);\n"
            "    if (bo) {\n        return border;\n    }\n"
            "    float f = t.read(uint2(uint(x), uint(y))).x;\n"
            "    switch (cmp - 1u) {\n"
            "    case 1: pass = ref < f; break;\n"
            "    case 2: pass = ref == f; break;\n"
            "    case 3: pass = ref <= f; break;\n"
            "    case 4: pass = ref > f; break;\n"
            "    case 5: pass = ref != f; break;\n"
            "    case 6: pass = ref >= f; break;\n"
            "    case 7: pass = true; break;\n"
            "    default: break;\n"
            "    }\n"
            "    return hd_nsel(float4(pass ? 1.0f : 0.0f), sel);\n"
            "}\n"
            "/* sample_lv(): a level's texture at tc in its texels; ls its size and its texture's */\n"
            "static float4 hd_nsamp(texture2d<float> t, float4 ls, float2 tc, uint4 wrap, bool lin,\n"
            "                       float4 border, uint4 sel, uint cmp, float ref, uint mx)\n"
            "{\n"
            "    if (cmp == 0u) {\n"
            "        bool bo = false;\n"
            "        float4 v;\n"
            "        tc.x = hd_nwrap(tc.x, ls.x, wrap.x, bo);\n"
            "        tc.y = hd_nwrap(tc.y, ls.y, wrap.y, bo);\n"
            "        if (lin) {\n"
            "            v = t.sample(hd_sl, tc / ls.zw, level(0.0f));\n"
            "        } else {\n"
            "            v = t.sample(hd_sn, tc / ls.zw, level(0.0f));\n"
            "        }\n"
            "        return bo ? border : hd_nsel(v, sel);\n"
            "    }\n"
            "    if (mx != 0u) {\n"
            "        ref = floor(clamp(ref, 0.0f, 1.0f) * float(mx) + 0.5f) / float(mx);\n"
            "    }\n"
            "    if (!lin) {\n"
            "        return hd_ncmp1(t, ls, int2(floor(tc)), wrap, border, sel, cmp, ref);\n"
            "    }\n"
            "    float2 f = tc - 0.5f, fl = floor(f), a = f - fl;\n"
            "    int2 p = int2(fl);\n"
            "    float4 t00 = hd_ncmp1(t, ls, p, wrap, border, sel, cmp, ref);\n"
            "    float4 t10 = hd_ncmp1(t, ls, p + int2(1, 0), wrap, border, sel, cmp, ref);\n"
            "    float4 t01 = hd_ncmp1(t, ls, p + int2(0, 1), wrap, border, sel, cmp, ref);\n"
            "    float4 t11 = hd_ncmp1(t, ls, p + int2(1, 1), wrap, border, sel, cmp, ref);\n"
            "    float4 top = t00 + (t10 - t00) * a.x, bot = t01 + (t11 - t01) * a.x;\n"
            "    return top + (bot - top) * a.y;\n"
            "}\n");
        for (i = 0; i < ir->nres; i++) {
            if ((ufn >> i) & 1) {
                nat_unit(o, k, i, slot[i]);
            }
        }
    }
    if (ir->cf) {
        /* the program's functions (ati_hd5770_emit.c) */
        for (i = 0; i < 32; i++) {
            bool fl = ((k->flat | iflat) >> i) & 1;

            if (i >= k->ninputs) {
                g_string_append_printf(o, "#define HD_NI_%uu(ch) 0.0f\n", i);
            } else {
                g_string_append_printf(o, "#define HD_NI_%uu(ch) %s(in.p%u[ch])\n",
                                       i, fl ? "F" : "", i);
            }
        }
        g_string_append(o, "#define HD_CTX_PARAMS NF in, constant NU &u, constant uint *kv");
        for (i = 0; i < ir->nres; i++) {
            if ((ufn >> i) & 1) {
                nat_tlist(o, k, i, true);
            }
        }
        g_string_append(o, "\n#define HD_CTX_ARGS in, u, kv");
        for (i = 0; i < ir->nres; i++) {
            if ((ufn >> i) & 1) {
                nat_tlist(o, k, i, false);
            }
        }
        g_string_append(o,
            "\nstatic void hd_ntex(HD_CTX_PARAMS, uint fu, uint w0, uint w1, uint w2,\n"
            "                    thread uint (&c)[4], float4 dv, thread uint (&o)[4])\n"
            "{\n"
            "    switch (fu) {\n");
        for (i = 0; i < ir->nres; i++) {
            if ((ufn >> i) & 1) {
                g_string_append_printf(o, "    case %u: hd_nt%u(u", i, i);
                nat_tlist(o, k, i, false);
                g_string_append(o, ", w0, w1, w2, c, dv, o); break;\n");
            }
        }
        g_string_append_printf(o,
            "    default: o[0] = 0u; o[1] = 0u; o[2] = 0u; o[3] = 0u; break;\n"
            "    }\n"
            "}\n"
            "#define HD_K(k) kv[k]\n"
            "#define HD_BCONST u.bconst\n"
            "#define HD_LCONST(i) u.lconst[i]\n"
            "#define HD_POSGPR %d\n"
            "#define HD_TEX(fu, w0, w1, w2, c, dv, o) { \\\n"
            "    float2 cn_ = float2(F(c[0]), F(c[1])); \\\n"
            "    hd_ntex(HD_CTX_ARGS, fu, w0, w1, w2, c, float4(dfdx(cn_), dfdy(cn_)), o); }\n"
            "#define HD_ZERO int(u.zero)\n"
            "#define HD_VTX(w0, w1, w2, i, o)\n"
            "#define HD_INTERP(p, ch, fl, b0, b1, b2, pw) HD_NI_##p(ch)\n"
            "#define HD_OUTA(n, a, b) thread uint (&n)[a][b]\n"
            "#define HD_OUTBA(n, a) thread bool (&n)[a]\n"
            "#define HD_OUTB(n) thread bool &n\n"
            "#define HD_OUTU(n) thread uint &n\n"
            "#define HD_OUTC(n) thread uint (&n)[8][4]\n",
            k->pos_gpr >= 0 && (unsigned)k->pos_gpr < ir->ngpr ? k->pos_gpr : -1);
        hd5770_emit_cf(o, ir, 0, true);
    }
    if (k->zout || k->ncol) {
        g_string_append(o, "struct NO {\n");
        for (i = 0; i < MAX(k->ncol, 1u); i++) {
            g_string_append_printf(o, "    float4 c%u [[color(%u)]];\n", i, i);
        }
        if (k->zout) {
            g_string_append(o, "    float z [[depth(any)]];\n");
        }
        g_string_append(o, "};\n");
    }
    g_string_append_printf(o,
        "fragment %s hd_nfs(NF in [[stage_in]], constant NU &u [[buffer(0)]],\n"
        "                       constant uint *kv [[buffer(1)]]",
        k->zout || k->ncol ? "NO" : "float4");
    for (i = 0; i < k->nvb; i++) {
        g_string_append_printf(o, ",\n    const device uchar *vb%u [[buffer(%u)]]",
                               i, 2 + i);
    }
    for (i = 0; i < k->ncol && k->keep; i++) {
        g_string_append_printf(o, ",\n    float4 d%u [[color(%u)]]", i, i);
    }
    for (i = 0, ntx = 0; i < ir->nres; i++) {
        unsigned n = nat_ntex(k, i);

        g_string_append_printf(o, ",\n    texture2d<float> t%u [[texture(%u)]], "
                               "sampler s%u [[sampler(%u)]]", i, i, i, i);
        if (n > 1) {
            g_string_append_printf(o, ",\n    array<texture2d<float>, %u> a%u"
                                   " [[texture(%u)]]", n, i, 16 + ntx);
            ntx += n;
        }
    }
    g_string_append(o,
        ")\n{\n"
        "    uint C[8][4];\n"
        "    for (uint i = 0; i < 8; i++) {\n"
        "        C[i][0] = C[i][1] = C[i][2] = C[i][3] = 0;\n"
        "    }\n"
        "    uint cm = 0, zd = 0, sd = 0;\n"
        "    bool killed = false, zw = false;\n");
    if (ir->cf) {
        g_string_append(o,
            "    hd_run(HD_CTX_ARGS, in.bc.x, in.bc.y, in.bc.z, 0.0f,\n"
            "           in.pos.x - 0.5f + u.pc, in.pos.y - 0.5f + u.pc,\n"
            "           0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,\n"
            "           killed, C, cm, zd, sd, zw);\n");
    } else {
        g_string_append_printf(o,
            "    uint R0[%u][4], P0[5] = { 0, 0, 0, 0, 0 };\n"
            "    for (uint i = 0; i < %u; i++) {\n"
            "        R0[i][0] = R0[i][1] = R0[i][2] = R0[i][3] = 0;\n"
            "    }\n", ng, ng);
    }
    if (!ir->cf && k->pos_gpr >= 0 && (unsigned)k->pos_gpr < ir->ngpr) {
        g_string_append_printf(o,
            "    R0[%d][0] = U(in.pos.x - 0.5f + u.pc);\n"
            "    R0[%d][1] = U(in.pos.y - 0.5f + u.pc);\n"
            "    R0[%d][2] = U(0.0f);\n    R0[%d][3] = U(1.0f);\n",
            k->pos_gpr, k->pos_gpr, k->pos_gpr, k->pos_gpr);
    }
    if (!ir->cf) {
        g_string_append_printf(o, "    R0[%u][0] = U(in.bc.y);\n    R0[%u][1] = U(in.bc.z);\n",
                               ir->map[0], ir->map[0]);
    }
    for (i = 0; i < ir->nops && !ir->cf; i++) {
        const HD5770IROp *op = &ir->ops[i];

        switch (op->kind) {
        case HD5770_IRO_ALU:
            for (j = 0; j < op->n; j++) {
                emit_group(o, ir, &ir->groups[op->first + j], 0);
            }
            break;
        case HD5770_IRO_FETCH:
            for (j = 0; j < op->n; j++) {
                if (!nat_fetch(o, ir, &ir->fetches[op->first + j], k)) {
                    g_string_free(o, true);
                    return NULL;
                }
            }
            break;
        default:
            emit_export(o, ir, op);
            break;
        }
    }
    if (k->clip) {
        /* clip_pass() in ati_hd5770_draw.c */
        g_string_append(o,
            "    {\n"
            "        int2 p = int2(in.pos.xy);\n"
            "        uint idx = 0;\n"
            "        for (uint i = 0; i < 4; i++) {\n"
            "            if (p.x >= u.cr[i].x && p.x < u.cr[i].z && p.y >= u.cr[i].y &&\n"
            "                p.y < u.cr[i].w) {\n"
            "                idx |= 1u << i;\n"
            "            }\n"
            "        }\n"
            "        killed = killed || !((u.cliprule >> idx) & 1u);\n"
            "    }\n");
    }
    if (k->ncol) {
        /* shade(): the alpha test reads export 0 */
        g_string_append(o,
            "    if (killed || cm == 0u ||\n"
            "        ((cm & 1u) != 0u && !hd_nalpha(u.alpha_func, u.alpha_ref, F(C[0][3])))) {\n"
            "        discard_fragment();\n"
            "    }\n");
    } else {
        g_string_append(o,
            "    if (killed || !(cm & 1) || !hd_nalpha(u.alpha_func, u.alpha_ref, F(C[0][3]))) {\n"
            "        discard_fragment();\n"
            "    }\n");
    }
    for (i = 0; i < MAX(k->ncol, 1u); i++) {
        unsigned cb = k->ncol ? k->col[i].cb & 7 : 0;
        unsigned out = k->ncol ? k->col[i].out : k->out;
        unsigned comps = k->ncol ? k->col[i].comps : k->rt_comps;
        const uint8_t *src = k->ncol ? k->col[i].src : k->rt_src;

        if (out == 1) {
            /* an integer target: the bytes of the channel */
            g_string_append_printf(o,
                "    uint v%u = C[%u][%u];\n"
                "    float4 r%u = float4(float(v%u & 0xffu), float((v%u >> 8) & 0xffu),\n"
                "                      float((v%u >> 16) & 0xffu), float(v%u >> 24)) / 255.0f;\n",
                i, cb, src[0] & 3, i, i, i, i, i);
        } else if (comps == 1) {
            g_string_append_printf(o,
                "    float4 r%u = float4(F(C[%u][%u]), 0.0f, 0.0f, 0.0f);\n"
                "    r%u = select(r%u, float4(0.0f), isnan(r%u));\n", i, cb,
                src[0] & 3, i, i, i);
        } else {
            /* byte i of the target is channel src[i] */
            g_string_append_printf(o,
                "    float4 r%u = float4(F(C[%u][%u]), F(C[%u][%u]), F(C[%u][%u]), F(C[%u][%u]));\n",
                i, cb, src[0] & 3, cb, src[1] & 3, cb, src[2] & 3, cb,
                src[3] & 3);
            if (out != 2) {
                g_string_append_printf(o,
                    "    r%u = select(r%u, float4(0.0f), isnan(r%u));\n", i, i, i);
            }
        }
    }
    if (k->zout || k->ncol) {
        g_string_append(o, "    NO no;\n");
        for (i = 0; i < MAX(k->ncol, 1u); i++) {
            if (k->ncol && k->keep) {
                /* shade(): only the buffers of the exports made */
                g_string_append_printf(o,
                    "    no.c%u = ((cm >> %uu) & 1u) != 0u ? r%u : d%u;\n",
                    i, k->col[i].cb & 7, i, i);
            } else {
                g_string_append_printf(o, "    no.c%u = r%u;\n", i, i);
            }
        }
        if (k->zout) {
            g_string_append_printf(o,
                "    no.z = min(max(%s, u.zr.x), u.zr.y);\n",
                k->zexp ? "(zw ? F(zd) : in.pos.z)" : "in.pos.z");

        }
        g_string_append(o, "    return no;\n}\n");
    } else {
        g_string_append(o, "    return r0;\n}\n");
    }
    return g_string_free(o, false);
}
