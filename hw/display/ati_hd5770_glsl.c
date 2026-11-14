/*
 * ATI Radeon HD 5770 (Juniper) pixel shader IR to GLSL compute shaders,
 * for the OpenGL backend (ati_hd5770_gl.c).
 *
 * hd5770_glsl_lib is the port of hd5770_msl_lib (ati_hd5770_msl.c): the
 * layouts of the per-draw data (the same in ati_hd5770_gl.c) and, as
 * functions, what the CPU path computes per pixel. Every float operation
 * is precise and rounded as the CPU path rounds it: division and square
 * root are corrected to the IEEE result through doubles, and a * b + c
 * is fused only where the CPU path's compiler fuses it (HD_FUSED: on
 * AArch64 hosts, not on x86, whose build has no FMA).
 *
 * Storage buffer 0 holds VRAM at VRAM offsets, then 4 KiB slots that hold
 * the pages of surfaces in guest RAM.
 *
 * hd5770_glsl_program() adds main(): work group g of 64 invocations is
 * tile tile0 + g, a block of a triangle's box; invocation l is its pixel
 * (x + l % w, y + l / w). Storage buffers: 0 VRAM and RAM slots, 1
 * surfaces, 2 draws, 3 textures, 4 triangles, 5 words (inputs, constants,
 * the constant file, vertex buffer bytes, RAM slots), 6 tiles: eight words each,
 * the draw, the triangle, its inputs' word offset, its first texture,
 * the constants' word offset, then x, y, w. A program that fetches
 * gradients runs the pixels to the right and below alongside, in
 * register sets of their own (r_1, r_2), as the CPU path's lanes do.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "ati_hd5770_gpu.h"

/* the CPU path is compiled with a * b + c fused on AArch64 hosts only */
#if defined(__aarch64__)
#define HD5770_GLSL_FUSED 1
#else
#define HD5770_GLSL_FUSED 0
#endif

const char hd5770_glsl_lib[] =
"#pragma optionNV(fastmath off)\n"
"#pragma optionNV(fastprecision off)\n"
"layout(local_size_x = 64) in;\n"
"\n"
"struct GSurf {\n"
"    uint voff, mode, bpe, row_bytes, mt_per_row, tile_bytes, mt_bytes;\n"
"    uint pipes, banks, bankw, bankh, pib;\n"
"    uint mt_w_sh, mt_h_sh, pipes_sh, bankw_sh, bankh_sh, pib_sh, hi_sh, fast;\n"
"    uint space, pt, pg0;\n"
"    uint eoff[64];\n"
"};\n"
"struct GTex {\n"
"    uint s, fmt, num, w, h, dim, sgn, comps, is_float, unorm8, samp0;\n"
"    uint dst_sel[4];\n"
"    uint bits[4];\n"
"    float border[4];\n"
"};\n"
"struct GVb {\n"
"    uint id, off, len, pad;\n"
"    uint res[8];\n"
"};\n"
"struct GDraw {\n"
"    uint rt, rt_fmt, rt_num, rt_comps, rt_unorm8, rt_float, rt_blend, wmask, blend;\n"
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
"    uint ninputs, flat_, cfile, ntex, nvb;\n"
"    GVb vb[8];\n"
"};\n"
"struct GTri {\n"
"    float x[3], y[3], w[3];\n"
"    float area;\n"
"    int x0, y0, x1, y1;\n"
"};\n"
"\n"
"layout(std430, binding = 0) coherent buffer Vram { uint vram[]; };\n"
"layout(std430, binding = 1) readonly buffer Surfs { GSurf surfs[]; };\n"
"layout(std430, binding = 2) readonly buffer Draws { GDraw draws[]; };\n"
"layout(std430, binding = 3) readonly buffer Texs { GTex texs[]; };\n"
"layout(std430, binding = 4) readonly buffer Tris { GTri tris[]; };\n"
"layout(std430, binding = 5) readonly buffer Words { uint words[]; };\n"
"layout(std430, binding = 6) readonly buffer Tiles { uint tiles[]; };\n"
"/* c / 255.0f for c in 0..255, then the bits of the formats' components */\n"
"layout(std430, binding = 7) readonly buffer Tables { uint hd_u8[256]; uint hd_fmt_bits[]; };\n"
"uniform uint tile0;\n"
"\n"
"#define F(x) uintBitsToFloat(x)\n"
"#define U(x) floatBitsToUint(x)\n"
"#define I(x) int(x)\n"
"#define FLT_MAX uintBitsToFloat(0x7f7fffffu)\n"
"\n"
"/* ---- exact arithmetic ---- */\n"
"\n"
"bool hd_isnan(float f)\n"
"{\n"
"    return (U(f) & 0x7fffffffu) > 0x7f800000u;\n"
"}\n"
"\n"
"/* a * b + c, fused where the CPU path's compiler fuses it */\n"
"float hd_mad(float a, float b, float c)\n"
"{\n"
"#if HD_FUSED\n"
"    precise float r = fma(a, b, c);\n"
"#else\n"
"    precise float m = a * b;\n"
"    precise float r = m + c;\n"
"#endif\n"
"    return r;\n"
"}\n"
"\n"
"/* a * b - c * d */\n"
"float hd_msub(float a, float b, float c, float d)\n"
"{\n"
"    precise float n = c * d;\n"
"#if HD_FUSED\n"
"    precise float r = fma(a, b, -n);\n"
"#else\n"
"    precise float m = a * b;\n"
"    precise float r = m - n;\n"
"#endif\n"
"    return r;\n"
"}\n"
"\n"
"/* a * b + c * d */\n"
"float hd_sum2(float a, float b, float c, float d)\n"
"{\n"
"    precise float cd = c * d;\n"
"#if HD_FUSED\n"
"    precise float r = fma(a, b, cd);\n"
"#else\n"
"    precise float ab = a * b;\n"
"    precise float r = ab + cd;\n"
"#endif\n"
"    return r;\n"
"}\n"
"\n"
"/* a * b + c * d + e * f, in that order */\n"
"float hd_sum3(float a, float b, float c, float d, float e, float f)\n"
"{\n"
"    precise float cd = c * d;\n"
"#if HD_FUSED\n"
"    precise float r = fma(e, f, fma(a, b, cd));\n"
"#else\n"
"    precise float ab = a * b;\n"
"    precise float ef = e * f;\n"
"    precise float s = ab + cd;\n"
"    precise float r = s + ef;\n"
"#endif\n"
"    return r;\n"
"}\n"
"\n"
"float hd_fmul(float a, float b)\n"
"{\n"
"    precise float r = a * b;\n"
"    return r;\n"
"}\n"
"\n"
"float hd_fadd(float a, float b)\n"
"{\n"
"    precise float r = a + b;\n"
"    return r;\n"
"}\n"
"\n"
"float hd_fsub(float a, float b)\n"
"{\n"
"    precise float r = a - b;\n"
"    return r;\n"
"}\n"
"\n"
"float hd_next(float q, int d)\n"
"{\n"
"    return F(uint(int(U(q)) + d));\n"
"}\n"
"\n"
"/* a / b correctly rounded (the GPU's division is not) */\n"
"float hd_div(float a, float b)\n"
"{\n"
"    precise float q = a / b;\n"
"    float aq = abs(q);\n"
"    uint ea = (U(a) >> 23) & 0xffu, eb = (U(b) >> 23) & 0xffu;\n"
"\n"
"    if (ea == 0u || ea == 0xffu || eb == 0u || eb == 0xffu ||\n"
"        aq < uintBitsToFloat(0x00800000u) || aq > FLT_MAX) {\n"
"        if (hd_isnan(a)) {\n"
"            return F(U(a) | 0x400000u);\n"
"        }\n"
"        if (hd_isnan(b)) {\n"
"            return F(U(b) | 0x400000u);\n"
"        }\n"
"        if (eb == 0u && (U(b) & 0x7fffffu) == 0u) {\n"
"            return a == 0.0 ? F(0xffc00000u) :\n"
"                   F(((U(a) ^ U(b)) & 0x80000000u) | 0x7f800000u);\n"
"        }\n"
"        return hd_isnan(q) ? F(0xffc00000u) : q;\n"
"    }\n"
"    {\n"
"        double da = abs(double(a)), db = abs(double(b));\n"
"        float m = aq;\n"
"\n"
"        for (int k = 0; k < 3; k++) {\n"
"            float up = hd_next(m, 1), dn = hd_next(m, -1);\n"
"            precise double hu = (double(m) + double(up)) * 0.5LF;\n"
"            precise double hd = (double(m) + double(dn)) * 0.5LF;\n"
"            precise double pu = hu * db;\n"
"            precise double pd = hd * db;\n"
"\n"
"            if (pu < da) {\n"
"                m = up;\n"
"            } else if (pd > da) {\n"
"                m = dn;\n"
"            }\n"
"        }\n"
"        return ((U(a) ^ U(b)) & 0x80000000u) != 0u ? -m : m;\n"
"    }\n"
"}\n"
"\n"
"/* sqrt correctly rounded */\n"
"float hd_sqrt(float a)\n"
"{\n"
"    float s = sqrt(a);\n"
"\n"
"    if (!(a > 0.0) || a > FLT_MAX || ((U(a) >> 23) & 0xffu) == 0u) {\n"
"        return a == 0.0 ? a : (a > 0.0 ? s : (hd_isnan(a) ? a : F(0xffc00000u)));\n"
"    }\n"
"    {\n"
"        double da = double(a);\n"
"\n"
"        for (int k = 0; k < 3; k++) {\n"
"            float up = hd_next(s, 1), dn = hd_next(s, -1);\n"
"            precise double hu = (double(s) + double(up)) * 0.5LF;\n"
"            precise double hd = (double(s) + double(dn)) * 0.5LF;\n"
"            precise double pu = hu * hu;\n"
"            precise double pd = hd * hd;\n"
"\n"
"            if (pu < da) {\n"
"                s = up;\n"
"            } else if (pd > da) {\n"
"                s = dn;\n"
"            }\n"
"        }\n"
"    }\n"
"    return s;\n"
"}\n"
"\n"
"/* C99 fmaxf/fminf as the CPU path's libm has them */\n"
"float hd_fmax(float a, float b)\n"
"{\n"
"    return a >= b || hd_isnan(b) ? a : b;\n"
"}\n"
"\n"
"float hd_fmin(float a, float b)\n"
"{\n"
"    return a <= b || hd_isnan(b) ? a : b;\n"
"}\n"
"\n"
"/* ---- formats ---- */\n"
"\n"
"\n"
"uint hd_fbits(uint fmt, uint i)\n"
"{\n"
"    return fmt < 49u ? hd_fmt_bits[fmt * 4u + i] : 0u;\n"
"}\n"
"\n"
"uint hd_fmt_bytes(uint fmt)\n"
"{\n"
"    switch (fmt) {\n"
"    case 0x01u: case 0x02u: case 0x03u: return 1u;\n"
"    case 0x05u: case 0x06u: case 0x07u: case 0x08u: case 0x09u: case 0x0au:\n"
"    case 0x0bu: case 0x0cu: return 2u;\n"
"    case 0x0du: case 0x0eu: case 0x0fu: case 0x10u: case 0x11u: case 0x13u:\n"
"    case 0x15u: case 0x17u: case 0x19u: case 0x1au: case 0x1bu: return 4u;\n"
"    case 0x1du: case 0x1eu: case 0x1fu: case 0x20u: return 8u;\n"
"    case 0x30u: return 12u;\n"
"    case 0x22u: case 0x23u: return 16u;\n"
"    default: return 0u;\n"
"    }\n"
"}\n"
"\n"
"bool hd_fmt_float(uint fmt)\n"
"{\n"
"    return fmt == 0x06u || fmt == 0x0eu || fmt == 0x10u || fmt == 0x1eu ||\n"
"           fmt == 0x20u || fmt == 0x23u || fmt == 0x30u;\n"
"}\n"
"\n"
"/* n bytes little endian at byte off of VRAM (st false) or the words */\n"
"uint hd_rd(bool st, uint off, uint n)\n"
"{\n"
"    uint w = off >> 2u, sh = (off & 3u) * 8u, v;\n"
"\n"
"    v = st ? words[w] : vram[w];\n"
"    if (sh != 0u) {\n"
"        v >>= sh;\n"
"        if (sh + n * 8u > 32u) {\n"
"            v |= (st ? words[w + 1u] : vram[w + 1u]) << (32u - sh);\n"
"        }\n"
"    }\n"
"    return n >= 4u ? v : v & ((1u << (n * 8u)) - 1u);\n"
"}\n"
"\n"
"uint hd_ext(uint lo, uint hi, uint sh, uint b)\n"
"{\n"
"    uint v;\n"
"\n"
"    if (sh >= 32u) {\n"
"        v = hi >> (sh - 32u);\n"
"    } else if (sh == 0u) {\n"
"        v = lo;\n"
"    } else {\n"
"        v = (lo >> sh) | (hi << (32u - sh));\n"
"    }\n"
"    return b >= 32u ? v : v & ((1u << b) - 1u);\n"
"}\n"
"\n"
"void hd_unpack(uint fmt, bool st, uint off, out uint c[4])\n"
"{\n"
"    uint n = hd_fmt_bytes(fmt), sh = 0u, lo, hi = 0u;\n"
"\n"
"    c[0] = 0u; c[1] = 0u; c[2] = 0u; c[3] = 0u;\n"
"    if (n > 8u) {\n"
"        for (uint i = 0u; i < 4u && hd_fbits(fmt, i) != 0u; i++) {\n"
"            c[i] = hd_rd(st, off + 4u * i, 4u);\n"
"        }\n"
"        return;\n"
"    }\n"
"    if (n == 8u) {\n"
"        lo = hd_rd(st, off, 4u);\n"
"        hi = hd_rd(st, off + 4u, 4u);\n"
"    } else {\n"
"        lo = n == 0u ? 0u : hd_rd(st, off, n);\n"
"    }\n"
"    for (uint i = 0u; i < 4u && hd_fbits(fmt, i) != 0u; i++) {\n"
"        uint b = hd_fbits(fmt, i);\n"
"\n"
"        c[i] = hd_ext(lo, hi, sh, b);\n"
"        sh += b;\n"
"    }\n"
"}\n"
"\n"
"/* half_to_float() */\n"
"float hd_half(uint h)\n"
"{\n"
"    uint s = (h >> 15u) & 1u, e = (h >> 10u) & 0x1fu, m = h & 0x3ffu, f;\n"
"\n"
"    if (e == 0u) {\n"
"        if (m == 0u) {\n"
"            f = s << 31u;\n"
"        } else {\n"
"            e = 127u - 15u + 1u;\n"
"            while ((m & 0x400u) == 0u) {\n"
"                m <<= 1u;\n"
"                e--;\n"
"            }\n"
"            m &= 0x3ffu;\n"
"            f = (s << 31u) | (e << 23u) | (m << 13u);\n"
"        }\n"
"    } else if (e == 31u) {\n"
"        f = (s << 31u) | 0x7f800000u | (m << 13u);\n"
"    } else {\n"
"        f = (s << 31u) | ((e + 127u - 15u) << 23u) | (m << 13u);\n"
"    }\n"
"    return F(f);\n"
"}\n"
"\n"
"/* c / 255.0f */\n"
"float hd_unorm8(uint c)\n"
"{\n"
"    return F(hd_u8[c & 0xffu]);\n"
"}\n"
"\n"
"/* comp_value() */\n"
"uint hd_comp(uint c, uint bits, bool is_float, bool is_signed, uint num)\n"
"{\n"
"    if (is_float) {\n"
"        return bits == 16u ? U(hd_half(c)) : c;\n"
"    }\n"
"    if (is_signed && bits < 32u && ((c >> (bits - 1u)) & 1u) != 0u) {\n"
"        c |= ~((1u << bits) - 1u);\n"
"    }\n"
"    switch (num) {\n"
"    case 0u:\n"
"        if (bits == 32u) {\n"
"            return U(is_signed ? hd_div(float(int(c)), 2147483647.0)\n"
"                               : hd_div(float(c), 4294967295.0));\n"
"        } else if (is_signed) {\n"
"            float v = hd_div(float(int(c)), float((1u << (bits - 1u)) - 1u));\n"
"\n"
"            return U(v > -1.0 ? v : -1.0);\n"
"        } else if (bits == 8u) {\n"
"            return U(hd_unorm8(c));\n"
"        }\n"
"        return U(hd_div(float(c), float((1u << bits) - 1u)));\n"
"    case 1u:\n"
"        return c;\n"
"    default:\n"
"        return U(is_signed ? float(int(c)) : float(c));\n"
"    }\n"
"}\n"
"\n"
"/* ---- surfaces ---- */\n"
"\n"
"uint hd_pipe(uint tx, uint ty, uint pipes)\n"
"{\n"
"    switch (pipes) {\n"
"    case 2u:\n"
"        return (tx & 1u) ^ (ty & 1u);\n"
"    case 4u:\n"
"        return ((tx & 1u) ^ ((ty >> 1u) & 1u)) | (((tx >> 1u) & 1u) ^ (ty & 1u)) << 1u;\n"
"    case 8u:\n"
"        return ((tx & 1u) ^ ((ty >> 2u) & 1u)) |\n"
"               (((tx >> 1u) & 1u) ^ ((ty >> 1u) & 1u) ^ ((ty >> 2u) & 1u)) << 1u |\n"
"               (((tx >> 2u) & 1u) ^ (ty & 1u)) << 2u;\n"
"    default:\n"
"        return 0u;\n"
"    }\n"
"}\n"
"\n"
"uint hd_bank(uint tx, uint ty, uint banks)\n"
"{\n"
"    uint x3 = tx & 1u, x4 = (tx >> 1u) & 1u, x5 = (tx >> 2u) & 1u, x6 = (tx >> 3u) & 1u;\n"
"    uint y3 = ty & 1u, y4 = (ty >> 1u) & 1u, y5 = (ty >> 2u) & 1u, y6 = (ty >> 3u) & 1u;\n"
"\n"
"    switch (banks) {\n"
"    case 16u:\n"
"        return (x3 ^ y6) | (x4 ^ y5 ^ y6) << 1u | (x5 ^ y4) << 2u | (x6 ^ y3) << 3u;\n"
"    case 8u:\n"
"        return (x3 ^ y5) | (x4 ^ y4 ^ y5) << 1u | (x5 ^ y3) << 2u;\n"
"    case 4u:\n"
"        return (x3 ^ y4) | (x4 ^ y3) << 1u;\n"
"    case 2u:\n"
"        return x3 ^ y3;\n"
"    default:\n"
"        return 0u;\n"
"    }\n"
"}\n"
"\n"
"/*\n"
" * Storage buffer 0 offset of the byte at offset o of surface si: in VRAM,\n"
" * or in guest RAM, whose pages are at voff + slot * 4096 (their slots in\n"
" * words from pt on)\n"
" */\n"
"uint hd_at(uint si, uint o)\n"
"{\n"
"    if (surfs[si].space != 0u) {\n"
"        uint q = surfs[si].pg0 + o;\n"
"\n"
"        return surfs[si].voff + words[surfs[si].pt + (q >> 12u)] * 4096u + (q & 4095u);\n"
"    }\n"
"    return surfs[si].voff + o;\n"
"}\n"
"\n"
"/* hd5770_surf_offset_fast(): the offset of element (x, y) of surface si */\n"
"uint hd_addr(uint si, uint x, uint y)\n"
"{\n"
"    uint tx = x >> 3u, ty = y >> 3u, total, pipe, bank, t;\n"
"\n"
"    if (surfs[si].mode < 2u) {\n"
"        return hd_at(si, y * surfs[si].row_bytes + x * surfs[si].bpe);\n"
"    }\n"
"    if (surfs[si].mode == 2u) {\n"
"        t = (ty * surfs[si].mt_per_row + tx) * surfs[si].tile_bytes;\n"
"    } else {\n"
"        total = ((ty >> (surfs[si].mt_h_sh - 3u)) * surfs[si].mt_per_row +\n"
"                 (tx >> (surfs[si].mt_w_sh - 3u))) * surfs[si].mt_bytes +\n"
"                (((ty & (surfs[si].bankh - 1u)) << surfs[si].bankw_sh) +\n"
"                 ((tx >> surfs[si].pipes_sh) & (surfs[si].bankw - 1u))) *\n"
"                surfs[si].tile_bytes;\n"
"        pipe = hd_pipe(tx, ty, surfs[si].pipes);\n"
"        bank = hd_bank(tx >> (surfs[si].bankw_sh + surfs[si].pipes_sh),\n"
"                       ty >> surfs[si].bankh_sh, surfs[si].banks) &\n"
"               (surfs[si].banks - 1u);\n"
"        t = (total & (surfs[si].pib - 1u)) | pipe << surfs[si].pib_sh |\n"
"            bank << (surfs[si].pib_sh + surfs[si].pipes_sh) |\n"
"            (total >> surfs[si].pib_sh) << surfs[si].hi_sh;\n"
"    }\n"
"    return hd_at(si, t + surfs[si].eoff[(y & 7u) * 8u + (x & 7u)]);\n"
"}\n"
"\n"
"/* ---- ALU: eval() and post() ---- */\n"
"\n"
"float hd_mul(float a, float b)\n"
"{\n"
"    if (a == 0.0 || b == 0.0) {\n"
"        return 0.0;\n"
"    }\n"
"    return hd_fmul(a, b);\n"
"}\n"
"\n"
"float hd_mulieee(float a, float b)\n"
"{\n"
"    return hd_fmul(a, b);\n"
"}\n"
"\n"
"float hd_max9(float a, float b)\n"
"{\n"
"    return a >= b ? a : (hd_isnan(b) ? a : b);\n"
"}\n"
"\n"
"float hd_min9(float a, float b)\n"
"{\n"
"    return a < b ? a : (hd_isnan(b) ? a : b);\n"
"}\n"
"\n"
"float hd_cl01(float v)\n"
"{\n"
"    return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : (hd_isnan(v) ? 0.0 : v));\n"
"}\n"
"\n"
"/* mods: bit k neg of source k, bit 4 + k abs */\n"
"float hd_fmod(uint mods, uint k, uint v)\n"
"{\n"
"    float f = F(v);\n"
"\n"
"    if (((mods >> (4u + k)) & 1u) != 0u) {\n"
"        f = F(U(f) & 0x7fffffffu);\n"
"    }\n"
"    if (((mods >> k) & 1u) != 0u) {\n"
"        f = F(U(f) ^ 0x80000000u);\n"
"    }\n"
"    return f;\n"
"}\n"
"\n"
"int hd_f2i(float a)\n"
"{\n"
"    return hd_isnan(a) ? 0 : (a >= 2147483647.0 ? 2147483647 :\n"
"           (a <= -2147483648.0 ? int(0x80000000u) : int(a)));\n"
"}\n"
"\n"
"float hd_neg(float a)\n"
"{\n"
"    return F(U(a) ^ 0x80000000u);\n"
"}\n"
"\n"
"uint hd_eval(uint op, uint mods, uint s0, uint s1, uint s2, inout bool kill)\n"
"{\n"
"    float a = 0.0, b = 0.0, c = 0.0;\n"
"\n"
"    if ((op & 0x800u) != 0u) {\n"
"        a = F(s0);\n"
"        b = F(s1);\n"
"        c = F(s2);\n"
"        if ((mods & 1u) != 0u) {\n"
"            a = hd_neg(a);\n"
"        }\n"
"        if ((mods & 2u) != 0u) {\n"
"            b = hd_neg(b);\n"
"        }\n"
"        if ((mods & 4u) != 0u) {\n"
"            c = hd_neg(c);\n"
"        }\n"
"        switch (op & 0x1fu) {\n"
"        case 0x14u: return U(hd_fadd(hd_mul(a, b), c));\n"
"        case 0x15u: return U(hd_fmul(hd_fadd(hd_mul(a, b), c), 2.0));\n"
"        case 0x16u: return U(hd_fmul(hd_fadd(hd_mul(a, b), c), 4.0));\n"
"        case 0x17u: return U(hd_fmul(hd_fadd(hd_mul(a, b), c), 0.5));\n"
"        case 0x18u: return U(hd_mad(a, b, c));\n"
"        case 0x07u: { precise float r = fma(a, b, c); return U(r); }\n"
"        case 0x19u: return a == 0.0 ? ((mods & 2u) != 0u ? U(b) : s1) : ((mods & 4u) != 0u ? U(c) : s2);\n"
"        case 0x1au: return a > 0.0 ? ((mods & 2u) != 0u ? U(b) : s1) : ((mods & 4u) != 0u ? U(c) : s2);\n"
"        case 0x1bu: return a >= 0.0 ? ((mods & 2u) != 0u ? U(b) : s1) : ((mods & 4u) != 0u ? U(c) : s2);\n"
"        case 0x1cu: return I(s0) == 0 ? s1 : s2;\n"
"        case 0x1du: return I(s0) > 0 ? s1 : s2;\n"
"        case 0x1eu: return I(s0) >= 0 ? s1 : s2;\n"
"        case 0x04u: {\n"
"            uint off = s1 & 31u, w = s2 & 31u;\n"
"            return w != 0u ? (s0 >> off) & ((1u << w) - 1u) : 0u;\n"
"        }\n"
"        case 0x05u: {\n"
"            uint off = s1 & 31u, w = s2 & 31u;\n"
"            return w != 0u ? uint(int(s0 << (32u - off - w)) >> int(32u - w)) : 0u;\n"
"        }\n"
"        case 0x06u: return (s0 & s1) | (~s0 & s2);\n"
"        default: return 0u;\n"
"        }\n"
"    }\n"
"    a = hd_fmod(mods, 0u, s0);\n"
"    b = hd_fmod(mods, 1u, s1);\n"
"    switch (op) {\n"
"    case 0x00u: return U(hd_fadd(a, b));\n"
"    case 0x01u: return U(hd_mul(a, b));\n"
"    case 0x02u: return U(hd_fmul(a, b));\n"
"    case 0x03u: return U(hd_max9(a, b));\n"
"    case 0x04u: return U(hd_min9(a, b));\n"
"    case 0x05u: return U(hd_fmax(a, b));\n"
"    case 0x06u: return U(hd_fmin(a, b));\n"
"    case 0x08u: return U(a == b ? 1.0 : 0.0);\n"
"    case 0x09u: return U(a > b ? 1.0 : 0.0);\n"
"    case 0x0au: return U(a >= b ? 1.0 : 0.0);\n"
"    case 0x0bu: return U(a != b ? 1.0 : 0.0);\n"
"    case 0x0cu: return a == b ? ~0u : 0u;\n"
"    case 0x0du: return a > b ? ~0u : 0u;\n"
"    case 0x0eu: return a >= b ? ~0u : 0u;\n"
"    case 0x0fu: return a != b ? ~0u : 0u;\n"
"    case 0x10u: return U(hd_fsub(a, floor(a)));\n"
"    case 0x11u: return U(trunc(a));\n"
"    case 0x12u: return U(ceil(a));\n"
"    case 0x13u: return U(roundEven(a));\n"
"    case 0x14u: return U(floor(a));\n"
"    case 0x15u: return uint(I(s0) >> int(s1 & 31u));\n"
"    case 0x16u: return s0 >> (s1 & 31u);\n"
"    case 0x17u: return s0 << (s1 & 31u);\n"
"    case 0x19u: return (mods & 0x11u) != 0u ? U(a) : s0;\n"
"    case 0x1au: return 0u;\n"
"    case 0x2cu: kill = a == b; return U(kill ? 1.0 : 0.0);\n"
"    case 0x2du: kill = a > b; return U(kill ? 1.0 : 0.0);\n"
"    case 0x2eu: kill = a >= b; return U(kill ? 1.0 : 0.0);\n"
"    case 0x2fu: kill = a != b; return U(kill ? 1.0 : 0.0);\n"
"    case 0x40u: kill = s0 > s1; return kill ? 1u : 0u;\n"
"    case 0x41u: kill = s0 >= s1; return kill ? 1u : 0u;\n"
"    case 0x46u: kill = I(s0) == I(s1); return kill ? 1u : 0u;\n"
"    case 0x47u: kill = I(s0) > I(s1); return kill ? 1u : 0u;\n"
"    case 0x48u: kill = I(s0) >= I(s1); return kill ? 1u : 0u;\n"
"    case 0x49u: kill = I(s0) != I(s1); return kill ? 1u : 0u;\n"
"    case 0x30u: return s0 & s1;\n"
"    case 0x31u: return s0 | s1;\n"
"    case 0x32u: return s0 ^ s1;\n"
"    case 0x33u: return ~s0;\n"
"    case 0x34u: return s0 + s1;\n"
"    case 0x35u: return s0 - s1;\n"
"    case 0x36u: return I(s0) > I(s1) ? s0 : s1;\n"
"    case 0x37u: return I(s0) < I(s1) ? s0 : s1;\n"
"    case 0x38u: return s0 > s1 ? s0 : s1;\n"
"    case 0x39u: return s0 < s1 ? s0 : s1;\n"
"    case 0x3au: return I(s0) == I(s1) ? ~0u : 0u;\n"
"    case 0x3bu: return I(s0) > I(s1) ? ~0u : 0u;\n"
"    case 0x3cu: return I(s0) >= I(s1) ? ~0u : 0u;\n"
"    case 0x3du: return I(s0) != I(s1) ? ~0u : 0u;\n"
"    case 0x3eu: return s0 > s1 ? ~0u : 0u;\n"
"    case 0x3fu: return s0 >= s1 ? ~0u : 0u;\n"
"    case 0x50u: return uint(hd_f2i(trunc(a)));\n"
"    case 0xb1u: return uint(hd_f2i(floor(a)));\n"
"    case 0xb0u: return uint(hd_f2i(floor(hd_fadd(a, 0.5))));\n"
"    case 0x9au: return hd_isnan(a) || a <= 0.0 ? 0u :\n"
"                       (a >= 4294967295.0 ? 0xffffffffu : uint(trunc(a)));\n"
"    case 0x9bu: return U(float(I(s0)));\n"
"    case 0x9cu: return U(float(s0));\n"
"    case 0x81u: return U(exp2(a));\n"
"    case 0x82u: return U(a == 0.0 ? -FLT_MAX : log2(a));\n"
"    case 0x83u: return U(log2(a));\n"
"    case 0x84u: return U(a == 0.0 ? FLT_MAX : hd_div(1.0, a));\n"
"    case 0x85u: return U(a == 0.0 ? 0.0 : hd_div(1.0, a));\n"
"    case 0x86u: return U(hd_div(1.0, a));\n"
"    case 0x87u: return U(a == 0.0 ? FLT_MAX : hd_div(1.0, hd_sqrt(a)));\n"
"    case 0x88u: return U(a == 0.0 ? 0.0 : hd_div(1.0, hd_sqrt(a)));\n"
"    case 0x89u: return U(hd_div(1.0, hd_sqrt(a)));\n"
"    case 0x8au: return U(hd_sqrt(a));\n"
"    case 0x8du: return U(sin(hd_fmul(hd_fmul(a, 2.0), 3.14159265358979)));\n"
"    case 0x8eu: return U(cos(hd_fmul(hd_fmul(a, 2.0), 3.14159265358979)));\n"
"    case 0x8fu: case 0x91u: return s0 * s1;\n"
"    case 0x90u: { int hi, lo; imulExtended(I(s0), I(s1), hi, lo); return uint(hi); }\n"
"    case 0x92u: { uint hi, lo; umulExtended(s0, s1, hi, lo); return hi; }\n"
"    case 0xccu: return s0;\n"
"    default: return 0u;\n"
"    }\n"
"}\n"
"\n"
"bool hd_isfloat(uint op, uint mods, bool clamp_, uint omod)\n"
"{\n"
"    if ((op & 0x800u) != 0u) {\n"
"        return (op & 0x1fu) >= 0x07u && (op & 0x1fu) <= 0x1bu;\n"
"    }\n"
"    switch (op) {\n"
"    case 0x15u: case 0x16u: case 0x17u: case 0x30u: case 0x31u: case 0x32u:\n"
"    case 0x33u: case 0x34u: case 0x35u: case 0x36u: case 0x37u: case 0x38u:\n"
"    case 0x39u: case 0x3au: case 0x3bu: case 0x3cu: case 0x3du: case 0x3eu:\n"
"    case 0x3fu: case 0x50u: case 0x9au: case 0xb0u: case 0xb1u: case 0x8fu:\n"
"    case 0x90u: case 0x91u: case 0x92u: case 0xccu: case 0x0cu: case 0x0du:\n"
"    case 0x0eu: case 0x0fu: case 0x42u: case 0x43u: case 0x44u: case 0x45u:\n"
"    case 0x40u: case 0x41u: case 0x46u: case 0x47u: case 0x48u: case 0x49u:\n"
"        return false;\n"
"    case 0x19u:\n"
"        return omod != 0u || clamp_ || (mods & 0x11u) != 0u;\n"
"    default:\n"
"        return true;\n"
"    }\n"
"}\n"
"\n"
"uint hd_post(uint op, uint mods, bool clamp_, uint omod, uint v)\n"
"{\n"
"    float f;\n"
"\n"
"    if (!hd_isfloat(op, mods, clamp_, omod)) {\n"
"        return v;\n"
"    }\n"
"    f = F(v);\n"
"    if (omod == 1u) {\n"
"        f = hd_fmul(f, 2.0);\n"
"    } else if (omod == 2u) {\n"
"        f = hd_fmul(f, 4.0);\n"
"    } else if (omod == 3u) {\n"
"        f = hd_fmul(f, 0.5);\n"
"    }\n"
"    if (clamp_) {\n"
"        f = hd_cl01(f);\n"
"    }\n"
"    return U(f);\n"
"}\n"
"\n"
"/* ---- fetches ---- */\n"
"\n"
"/* C's a % b and a / b (truncating), b > 0 */\n"
"int hd_rem(int a, int b)\n"
"{\n"
"    int r = int(uint(abs(a)) % uint(b));\n"
"\n"
"    return a < 0 ? -r : r;\n"
"}\n"
"\n"
"int hd_div2(int a)\n"
"{\n"
"    return a < 0 ? -((-a) / 2) : a / 2;\n"
"}\n"
"\n"
"bool hd_wrap(inout int c, int size, uint mode)\n"
"{\n"
"    int v = c;\n"
"\n"
"    switch (mode) {\n"
"    case 0u:\n"
"        v = hd_rem(v, size);\n"
"        if (v < 0) {\n"
"            v += size;\n"
"        }\n"
"        break;\n"
"    case 1u: {\n"
"        int p = hd_rem(v, 2 * size);\n"
"\n"
"        if (p < 0) {\n"
"            p += 2 * size;\n"
"        }\n"
"        v = p < size ? p : 2 * size - 1 - p;\n"
"        break;\n"
"    }\n"
"    case 2u: case 3u: case 5u: case 7u:\n"
"        if (mode != 2u) {\n"
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
"void hd_texel(uint ti, int x, int y, out uint o[4])\n"
"{\n"
"    uint off = hd_addr(texs[ti].s, uint(x), uint(y)), v[6], c[4];\n"
"\n"
"    if (texs[ti].unorm8 != 0u) {\n"
"        uint e = hd_rd(false, off, 4u);\n"
"\n"
"        for (uint k = 0u; k < 4u; k++) {\n"
"            v[k] = U(hd_unorm8(e >> (8u * k)));\n"
"        }\n"
"    } else {\n"
"        hd_unpack(texs[ti].fmt, false, off, c);\n"
"        for (uint k = 0u; k < 4u; k++) {\n"
"            if (k < texs[ti].comps) {\n"
"                v[k] = hd_comp(c[k], texs[ti].bits[k], texs[ti].is_float != 0u,\n"
"                               ((texs[ti].sgn >> k) & 1u) != 0u, texs[ti].num);\n"
"            } else {\n"
"                v[k] = U(k == 3u ? 1.0 : 0.0);\n"
"            }\n"
"        }\n"
"    }\n"
"    v[4] = U(0.0);\n"
"    v[5] = U(1.0);\n"
"    for (uint k = 0u; k < 4u; k++) {\n"
"        uint s = texs[ti].dst_sel[k];\n"
"\n"
"        o[k] = s < 6u ? v[s] : v[4];\n"
"    }\n"
"}\n"
"\n"
"void hd_point(uint ti, int x, int y, out uint o[4])\n"
"{\n"
"    if (!hd_wrap(x, int(texs[ti].w), texs[ti].samp0 & 7u) ||\n"
"        !hd_wrap(y, int(texs[ti].h), (texs[ti].samp0 >> 3u) & 7u)) {\n"
"        for (uint k = 0u; k < 4u; k++) {\n"
"            o[k] = U(texs[ti].border[k]);\n"
"        }\n"
"        return;\n"
"    }\n"
"    hd_texel(ti, x, y, o);\n"
"}\n"
"\n"
"int hd_floori(float f)\n"
"{\n"
"    return hd_isnan(f) ? 0 : int(clamp(floor(f), -2147483648.0, 2147483520.0));\n"
"}\n"
"\n"
"float hd_lerp(float a, float b, float t)\n"
"{\n"
"    /* a + (b - a) * t */\n"
"#if HD_FUSED\n"
"    precise float d = b - a;\n"
"    precise float r = fma(d, t, a);\n"
"#else\n"
"    precise float d = b - a;\n"
"    precise float m = d * t;\n"
"    precise float r = a + m;\n"
"#endif\n"
"    return r;\n"
"}\n"
"\n"
"/* tfetch_stage() for the pixel shader */\n"
"void hd_tfetch(uint ti, uint w0, uint w1, uint w2, uint c[4], out uint v[4])\n"
"{\n"
"    uint op = w0 & 0x1fu;\n"
"    int ox = int(w2 << 27u) >> 27, oy = int(w2 << 22u) >> 27;\n"
"    float u, vv;\n"
"\n"
"    v[0] = U(0.0); v[1] = U(0.0); v[2] = U(0.0); v[3] = U(0.0);\n"
"    if (op == 0x03u) {\n"
"        int x = I(c[0]) + hd_div2(ox), y = I(c[1]) + hd_div2(oy);\n"
"\n"
"        if (x >= 0 && y >= 0 && x < int(texs[ti].w) && y < int(texs[ti].h)) {\n"
"            hd_texel(ti, x, y, v);\n"
"        }\n"
"        return;\n"
"    }\n"
"    u = F(c[0]);\n"
"    vv = F(c[1]);\n"
"    if (((w1 >> 28u) & 1u) != 0u) {\n"
"        u = hd_fmul(u, float(texs[ti].w));\n"
"    }\n"
"    if (((w1 >> 29u) & 1u) != 0u) {\n"
"        vv = hd_fmul(vv, float(texs[ti].h));\n"
"    }\n"
"    u = hd_mad(float(ox), 0.5, u);\n"
"    vv = hd_mad(float(oy), 0.5, vv);\n"
"    if (texs[ti].dim == 0u) {\n"
"        vv = 0.5;\n"
"    }\n"
"    if (((texs[ti].samp0 >> 9u) & 1u) == 0u && ((texs[ti].samp0 >> 11u) & 1u) == 0u) {\n"
"        hd_point(ti, hd_floori(u), hd_floori(vv), v);\n"
"        return;\n"
"    }\n"
"    {\n"
"        float fx = hd_fsub(u, 0.5), fy = hd_fsub(vv, 0.5);\n"
"        int x0 = hd_floori(fx), y0 = hd_floori(fy);\n"
"        float ax = hd_fsub(fx, float(x0)), ay = hd_fsub(fy, float(y0));\n"
"        uint t00[4], t10[4], t01[4], t11[4];\n"
"\n"
"        hd_point(ti, x0, y0, t00);\n"
"        hd_point(ti, x0 + 1, y0, t10);\n"
"        hd_point(ti, x0, y0 + 1, t01);\n"
"        hd_point(ti, x0 + 1, y0 + 1, t11);\n"
"        for (uint k = 0u; k < 4u; k++) {\n"
"            float top = hd_lerp(F(t00[k]), F(t10[k]), ax);\n"
"            float bot = hd_lerp(F(t01[k]), F(t11[k]), ax);\n"
"\n"
"            v[k] = U(hd_lerp(top, bot, ay));\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"/* vfetch_stage() for the pixel shader */\n"
"void hd_vfetch(uint di, uint w0, uint w1, uint w2, uint index, out uint v[4])\n"
"{\n"
"    uint bid = (w0 >> 8u) & 0xffu, off = w2 & 0xffffu, fmt, num, sgn, n, k;\n"
"    uint c[4];\n"
"\n"
"    if (bid == 128u) {\n"
"        uint e = index + off / 16u;\n"
"\n"
"        for (k = 0u; k < 4u; k++) {\n"
"            v[k] = e < 256u ? words[draws[di].cfile + e * 4u + k] : 0u;\n"
"        }\n"
"        return;\n"
"    }\n"
"    v[0] = U(0.0); v[1] = U(0.0); v[2] = U(0.0);\n"
"    v[3] = U(1.0);\n"
"    for (k = 0u; k < draws[di].nvb && draws[di].vb[k].id != bid; k++) {\n"
"    }\n"
"    if (k == draws[di].nvb) {\n"
"        return;\n"
"    }\n"
"    uint size = draws[di].vb[k].res[1] + 1u, stride = (draws[di].vb[k].res[2] >> 8u) & 0x7ffu;\n"
"    uint r2 = draws[di].vb[k].res[2];\n"
"    if (((w1 >> 21u) & 1u) != 0u) {\n"
"        fmt = (r2 >> 20u) & 0x3fu;\n"
"        num = (r2 >> 26u) & 3u;\n"
"        sgn = (r2 >> 28u) & 1u;\n"
"    } else {\n"
"        fmt = (w1 >> 22u) & 0x3fu;\n"
"        num = (w1 >> 28u) & 3u;\n"
"        sgn = (w1 >> 30u) & 1u;\n"
"    }\n"
"    n = hd_fmt_bytes(fmt);\n"
"    uint mhi, at;\n"
"    umulExtended(index, stride, mhi, at);\n"
"    uint at2 = at + off;\n"
"    if (mhi != 0u || at2 < at || at2 + n < at2 || at2 + n > size ||\n"
"        at2 + n > draws[di].vb[k].len) {\n"
"        v[0] = 0u; v[1] = 0u; v[2] = 0u;\n"
"        return;\n"
"    }\n"
"    hd_unpack(fmt, true, draws[di].vb[k].off * 4u + at2, c);\n"
"    for (k = 0u; k < 4u && hd_fbits(fmt, k) != 0u; k++) {\n"
"        v[k] = hd_comp(c[k], hd_fbits(fmt, k), hd_fmt_float(fmt), sgn != 0u, num);\n"
"    }\n"
"}\n"
"\n"
"/* ---- rasterisation ---- */\n"
"\n"
"float hd_edge(float ax, float ay, float bx, float by, float px, float py)\n"
"{\n"
"    if (ax > bx || (ax == bx && ay > by)) {\n"
"        return hd_neg(hd_msub(hd_fsub(ax, bx), hd_fsub(py, by),\n"
"                              hd_fsub(ay, by), hd_fsub(px, bx)));\n"
"    }\n"
"    return hd_msub(hd_fsub(bx, ax), hd_fsub(py, ay), hd_fsub(by, ay),\n"
"                   hd_fsub(px, ax));\n"
"}\n"
"\n"
"bool hd_owns(float ax, float ay, float bx, float by)\n"
"{\n"
"    return (ay == by && bx < ax) || by < ay;\n"
"}\n"
"\n"
"/* interp_n() */\n"
"float hd_interp(uint di, uint ti, uint pr, uint p, uint chan, bool flat_,\n"
"                float b0, float b1, float b2, float pw)\n"
"{\n"
"    float a, b, c;\n"
"\n"
"    if (p >= draws[di].ninputs) {\n"
"        return 0.0;\n"
"    }\n"
"    a = F(words[pr + p * 12u + chan]);\n"
"    if (flat_ || ((draws[di].flat_ >> p) & 1u) != 0u) {\n"
"        return a;\n"
"    }\n"
"    b = F(words[pr + p * 12u + 4u + chan]);\n"
"    c = F(words[pr + p * 12u + 8u + chan]);\n"
"    if (draws[di].persp != 0u) {\n"
"        return hd_fmul(hd_sum3(hd_fmul(a, tris[ti].w[0]), b0,\n"
"                               hd_fmul(b, tris[ti].w[1]), b1,\n"
"                               hd_fmul(c, tris[ti].w[2]), b2), pw);\n"
"    }\n"
"    return hd_sum3(a, b0, b, b1, c, b2);\n"
"}\n"
"\n"
"/* ---- output: alpha_pass(), put_pixel() ---- */\n"
"\n"
"bool hd_alpha(uint di, float a)\n"
"{\n"
"    float r = draws[di].alpha_ref;\n"
"\n"
"    switch (draws[di].alpha_func) {\n"
"    case 0u: return false;\n"
"    case 1u: return a < r;\n"
"    case 2u: return a == r;\n"
"    case 3u: return a <= r;\n"
"    case 4u: return a > r;\n"
"    case 5u: return a != r;\n"
"    case 6u: return a >= r;\n"
"    default: return true;\n"
"    }\n"
"}\n"
"\n"
"float hd_bfactor(uint di, uint f, float src[4], float dst[4], uint ch)\n"
"{\n"
"    switch (f) {\n"
"    case 0u: return 0.0;\n"
"    case 1u: return 1.0;\n"
"    case 2u: return src[ch];\n"
"    case 3u: return hd_fsub(1.0, src[ch]);\n"
"    case 4u: return src[3];\n"
"    case 5u: return hd_fsub(1.0, src[3]);\n"
"    case 6u: return dst[3];\n"
"    case 7u: return hd_fsub(1.0, dst[3]);\n"
"    case 8u: return dst[ch];\n"
"    case 9u: return hd_fsub(1.0, dst[ch]);\n"
"    case 10u: return ch == 3u ? 1.0 : hd_fmin(src[3], hd_fsub(1.0, dst[3]));\n"
"    case 13u: return draws[di].blend_const[ch];\n"
"    case 14u: return hd_fsub(1.0, draws[di].blend_const[ch]);\n"
"    case 19u: return draws[di].blend_const[3];\n"
"    case 20u: return hd_fsub(1.0, draws[di].blend_const[3]);\n"
"    default: return 0.0;\n"
"    }\n"
"}\n"
"\n"
"float hd_combine(uint fn, float s, float sf, float dv, float df)\n"
"{\n"
"    switch (fn) {\n"
"    case 0u: return hd_sum2(s, sf, dv, df);\n"
"    case 1u: return hd_msub(s, sf, dv, df);\n"
"    case 2u: return hd_fmin(s, dv);\n"
"    case 3u: return hd_fmax(s, dv);\n"
"    case 4u: return hd_msub(dv, df, s, sf);\n"
"    default: return s;\n"
"    }\n"
"}\n"
"\n"
"/* bytes [off, off + n) of VRAM, n < 4, within one word */\n"
"void hd_store_sub(uint off, uint n, uint v)\n"
"{\n"
"    uint sh = (off & 3u) * 8u, m = ((1u << (n * 8u)) - 1u) << sh;\n"
"\n"
"    atomicAnd(vram[off >> 2u], ~m);\n"
"    atomicOr(vram[off >> 2u], (v << sh) & m);\n"
"}\n"
"\n"
"void hd_pack_store(uint off, uint fmt, uint c[4])\n"
"{\n"
"    uint n = hd_fmt_bytes(fmt), sh = 0u, lo = 0u, hi = 0u;\n"
"\n"
"    if (n > 8u) {\n"
"        for (uint i = 0u; i < 4u && hd_fbits(fmt, i) != 0u; i++) {\n"
"            vram[(off >> 2u) + i] = c[i];\n"
"        }\n"
"        return;\n"
"    }\n"
"    for (uint i = 0u; i < 4u && hd_fbits(fmt, i) != 0u; i++) {\n"
"        uint b = hd_fbits(fmt, i);\n"
"        uint v = b >= 32u ? c[i] : c[i] & ((1u << b) - 1u);\n"
"\n"
"        if (sh < 32u) {\n"
"            lo |= v << sh;\n"
"            if (sh != 0u && sh + b > 32u) {\n"
"                hi |= v >> (32u - sh);\n"
"            }\n"
"        } else {\n"
"            hi |= v << (sh - 32u);\n"
"        }\n"
"        sh += b;\n"
"    }\n"
"    switch (n) {\n"
"    case 1u: case 2u:\n"
"        hd_store_sub(off, n, lo);\n"
"        break;\n"
"    case 4u:\n"
"        vram[off >> 2u] = lo;\n"
"        break;\n"
"    case 8u:\n"
"        vram[off >> 2u] = lo;\n"
"        vram[(off >> 2u) + 1u] = hi;\n"
"        break;\n"
"    default:\n"
"        break;\n"
"    }\n"
"}\n"
"\n"
"void hd_put(uint di, int x, int y, uint col[4])\n"
"{\n"
"    uint off = hd_addr(draws[di].rt, uint(x), uint(y)), c[4], old[4];\n"
"    uint n = draws[di].rt_comps, blend = draws[di].blend;\n"
"    float src[4], dst[4];\n"
"\n"
"    c[0] = 0u; c[1] = 0u; c[2] = 0u; c[3] = 0u;\n"
"    hd_unpack(draws[di].rt_fmt, false, off, old);\n"
"    if (draws[di].rt_num == 4u || draws[di].rt_float != 0u) {\n"
"        for (uint i = 0u; i < n; i++) {\n"
"            uint s = draws[di].rt_src[i];\n"
"\n"
"            c[i] = ((draws[di].wmask >> s) & 1u) != 0u ? col[s] : old[i];\n"
"        }\n"
"        hd_pack_store(off, draws[di].rt_fmt, c);\n"
"        return;\n"
"    }\n"
"    for (uint i = 0u; i < 4u; i++) {\n"
"        src[i] = F(col[i]);\n"
"        if (hd_isnan(src[i])) {\n"
"            src[i] = 0.0;\n"
"        }\n"
"        dst[i] = i == 3u ? 1.0 : 0.0;\n"
"    }\n"
"    for (uint i = 0u; i < n; i++) {\n"
"        uint b = draws[di].rt_bits[i];\n"
"\n"
"        dst[draws[di].rt_src[i]] = b == 8u ? hd_unorm8(old[i])\n"
"                                           : hd_div(float(old[i]), float((1u << b) - 1u));\n"
"    }\n"
"    if ((blend & (1u << 30u)) != 0u && draws[di].rt_blend != 0u) {\n"
"        uint csrc = blend & 0x1fu, cfn = (blend >> 5u) & 7u;\n"
"        uint cdst = (blend >> 8u) & 0x1fu;\n"
"        uint asrc = csrc, afn = cfn, adst = cdst;\n"
"        float o[4], sc[4];\n"
"\n"
"        if (((blend >> 29u) & 1u) != 0u) {\n"
"            asrc = (blend >> 16u) & 0x1fu;\n"
"            afn = (blend >> 21u) & 7u;\n"
"            adst = (blend >> 24u) & 0x1fu;\n"
"        }\n"
"        for (uint i = 0u; i < 4u; i++) {\n"
"            sc[i] = hd_fmin(hd_fmax(src[i], 0.0), 1.0);\n"
"        }\n"
"        for (uint i = 0u; i < 4u; i++) {\n"
"            uint fs = i == 3u ? asrc : csrc, fd = i == 3u ? adst : cdst;\n"
"            uint fn = i == 3u ? afn : cfn;\n"
"\n"
"            o[i] = hd_combine(fn, sc[i], hd_bfactor(di, fs, sc, dst, i), dst[i],\n"
"                              hd_bfactor(di, fd, sc, dst, i));\n"
"        }\n"
"        for (uint i = 0u; i < 4u; i++) {\n"
"            src[i] = o[i];\n"
"        }\n"
"    }\n"
"    for (uint i = 0u; i < n; i++) {\n"
"        uint s = draws[di].rt_src[i];\n"
"        uint mx = (1u << draws[di].rt_bits[i]) - 1u;\n"
"        float v = hd_fmin(hd_fmax(src[s], 0.0), 1.0);\n"
"\n"
"        c[i] = ((draws[di].wmask >> s) & 1u) != 0u ? uint(hd_mad(v, float(mx), 0.5)) : old[i];\n"
"    }\n"
"    hd_pack_store(off, draws[di].rt_fmt, c);\n"
"}\n"
"\n"
"bool hd_clip(uint di, int x, int y)\n"
"{\n"
"    uint idx = 0u;\n"
"\n"
"    for (uint i = 0u; i < 4u; i++) {\n"
"        if (x >= draws[di].cr[i * 4u] && x < draws[di].cr[i * 4u + 2u] &&\n"
"            y >= draws[di].cr[i * 4u + 1u] && y < draws[di].cr[i * 4u + 3u]) {\n"
"            idx |= 1u << i;\n"
"        }\n"
"    }\n"
"    return ((draws[di].cliprule >> idx) & 1u) != 0u;\n"
"}\n"
;

/* names of register set q: the pixel's (0), right (1), below (2) */
static const char *const set_sfx[3] = { "", "_1", "_2" };

static void src_expr(GString *o, const HD5770IRSrc *s, unsigned q)
{
    switch (s->kind) {
    case HD5770_IRS_GPR:
        g_string_append_printf(o, "r%s[%u][%u]", set_sfx[q], s->idx, s->chan);
        break;
    case HD5770_IRS_PV:
        g_string_append_printf(o, "pv%s[%u]", set_sfx[q], s->chan);
        break;
    case HD5770_IRS_CONST:
        g_string_append_printf(o, "words[kvo + %uu]", s->k);
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
    const char *x = set_sfx[q];
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
        g_string_append_printf(o, "        uint q%u = 0u;\n", i);
    }
    if (g->dot) {
        g_string_append(o, "        float dot_ = 0.0;\n");
    }
    for (i = 0; i < g->n; i++) {
        const HD5770IRInst *in = &ir->insts[g->first + i];
        unsigned m = src_mods(in);

        if (!in->needed) {
            continue;
        }
        if (in->dot) {
            g_string_append_printf(o, "        dot_ = hd_fadd(dot_, %s(hd_fmod(%uu, 0u, "
                                   "s%u_0), hd_fmod(%uu, 1u, s%u_1)));\n",
                                   in->op == 0xbe ? "hd_mul" : "hd_mulieee",
                                   m, i, m, i);
        } else if (in->interp) {
            g_string_append_printf(o, "        q%u = U(hd_interp(di, ti, pr, %uu, %uu, "
                                   "%s, b0%s, b1%s, b2%s, pw%s));\n", i, in->prm,
                                   in->dst_chan, in->flat ? "true" : "false",
                                   x, x, x, x);
        } else if (in->max4) {
            continue;
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
            /* only the pixel's own set kills */
            g_string_append(o, q ? ", k); }\n"
                                 : ", k); if (k) { killed = true; } }\n");
        }
    }
    if (g->dot) {
        for (i = 0; i < g->n; i++) {
            if (ir->insts[g->first + i].dot && ir->insts[g->first + i].needed) {
                g_string_append_printf(o, "        q%u = U(dot_);\n", i);
            }
        }
    }
    if (g->max4) {
        g_string_append(o, "        { float m = -FLT_MAX;\n");
        for (i = 0; i < g->n; i++) {
            const HD5770IRInst *in = &ir->insts[g->first + i];

            if (in->max4) {
                g_string_append_printf(o, "          m = hd_max9(m, hd_fmod(%uu, 0u, "
                                       "s%u_0));\n", src_mods(in), i);
            }
        }
        for (i = 0; i < g->n; i++) {
            if (ir->insts[g->first + i].max4) {
                g_string_append_printf(o, "          q%u = U(m);\n", i);
            }
        }
        g_string_append(o, "        }\n");
    }
    for (i = 0; i < g->n; i++) {
        const HD5770IRInst *in = &ir->insts[g->first + i];

        if (!in->needed) {
            continue;
        }
        if (in->omod || in->clamp) {
            g_string_append_printf(o, "        q%u = hd_post(0x%xu, %uu, %s, %uu, q%u);\n",
                                   i, in->op, src_mods(in),
                                   in->clamp ? "true" : "false", in->omod, i);
        }
        g_string_append_printf(o, "        pv%s[%u] = q%u;\n", x, in->slot, i);
    }
    for (i = 0; i < g->n; i++) {
        const HD5770IRInst *in = &ir->insts[g->first + i];

        if (in->wr) {
            g_string_append_printf(o, "        r%s[%u][%u] = q%u;\n", x, in->dst,
                                   in->dst_chan, i);
        }
    }
    g_string_append(o, "    }\n");
}

static void emit_fetch(GString *o, const HD5770IR *ir, const HD5770IRFetch *f,
                       unsigned q)
{
    const char *x = set_sfx[q];
    unsigned k, unit = 0;

    g_string_append(o, "    {\n        uint o[4] = uint[4](0u, 0u, 0u, 0u);\n");
    if (f->kind == HD5770_IRF_GRAD_H || f->kind == HD5770_IRF_GRAD_V) {
        /* the neighbour minus this pixel; in a neighbour, itself */
        const char *nb = set_sfx[q ? q : f->kind == HD5770_IRF_GRAD_H ? 1 : 2];

        for (k = 0; k < 4; k++) {
            unsigned sel = (f->w[2] >> (20 + 3 * k)) & 7;

            if (sel < 4) {
                g_string_append_printf(o, "        o[%u] = U(hd_fsub(F(r%s[%u][%u]), "
                                       "F(r%s[%u][%u])));\n", k, nb, f->src, sel,
                                       x, f->src, sel);
            } else {
                g_string_append_printf(o, "        o[%u] = U(0.0);\n", k);
            }
        }
    } else if (f->kind == HD5770_IRF_VTX) {
        g_string_append_printf(o, "        hd_vfetch(di, 0x%08xu, 0x%08xu, 0x%08xu, "
                               "r%s[%u][%u], o);\n", f->w[0], f->w[1], f->w[2],
                               x, f->src, (f->w[0] >> 24) & 3);
    } else {
        for (unit = 0; unit < ir->nres && ir->res[unit] != f->res; unit++) {
            /* search */
        }
        g_string_append(o, "        uint c[4] = uint[4](");
        for (k = 0; k < 4; k++) {
            unsigned sel = (f->w[2] >> (20 + 3 * k)) & 7;

            if (sel < 4) {
                g_string_append_printf(o, "r%s[%u][%u]", x, f->src, sel);
            } else {
                g_string_append(o, sel == 5 ? "U(1.0)" : "U(0.0)");
            }
            g_string_append(o, k < 3 ? ", " : ");\n");
        }
        g_string_append_printf(o, "        hd_tfetch(tx + %uu, 0x%08xu, 0x%08xu, "
                               "0x%08xu, c, o);\n", unit, f->w[0], f->w[1], f->w[2]);
    }
    for (k = 0; k < 4; k++) {
        unsigned sel = (f->w[1] >> (9 + 3 * k)) & 7;

        if (sel < 4) {
            g_string_append_printf(o, "        r%s[%u][%u] = o[%u];\n", x, f->dst,
                                   k, sel);
        } else if (sel == 4 || sel == 5) {
            g_string_append_printf(o, "        r%s[%u][%u] = U(%s);\n", x, f->dst, k,
                                   sel == 5 ? "1.0" : "0.0");
        }
    }
    g_string_append(o, "    }\n");
}

/* the version line and the arithmetic choice */
static void emit_head(GString *o)
{
    g_string_append_printf(o, "#version 430 core\n#define HD_FUSED %d\n",
                           HD5770_GLSL_FUSED);
}

/* the contents of storage buffer 7 */
size_t hd5770_glsl_tables(uint32_t *out, size_t n)
{
    uint8_t z[16] = { 0 };
    uint32_t c[4];
    unsigned bits[4], i, k;
    size_t m = 256 + 64 * 4;

    if (n < m) {
        return m;
    }
    for (i = 0; i < 256; i++) {
        float f = i / (float)((1u << 8) - 1);

        memcpy(&out[i], &f, 4);
    }
    for (i = 0; i < 64; i++) {
        memset(bits, 0, sizeof(bits));
        if (hd5770_fmt_bytes(i)) {
            hd5770_fmt_unpack(i, z, c, bits);
        }
        for (k = 0; k < 4; k++) {
            out[256 + i * 4 + k] = bits[k];
        }
    }
    return m;
}

char *hd5770_glsl_lib_only(void)
{
    GString *o = g_string_new(NULL);

    emit_head(o);
    g_string_append(o, hd5770_glsl_lib);
    g_string_append(o, "void main()\n{\n}\n");
    return g_string_free(o, false);
}

char *hd5770_glsl_program(const HD5770IR *ir)
{
    GString *o;
    unsigned i, j, k, q, nq = ir->grad ? 3 : 1, ng = MAX(ir->ngpr, 1u);

    o = g_string_new(NULL);
    emit_head(o);
    g_string_append(o, hd5770_glsl_lib);
    g_string_append(o,
        "void main()\n"
        "{\n"
        "    uint te = (tile0 + gl_WorkGroupID.x) * 8u, l = gl_LocalInvocationID.x;\n"
        "    uint di = tiles[te], ti = tiles[te + 1u], pr = tiles[te + 2u];\n"
        "    uint tx = tiles[te + 3u], kvo = tiles[te + 4u], sw = tiles[te + 7u];\n"
        "    int x = int(tiles[te + 5u]) + int(l % sw), y = int(tiles[te + 6u]) + int(l / sw);\n"
        "    if (x > tris[ti].x1 || y > tris[ti].y1) {\n        return;\n    }\n"
        "    float pc = draws[di].pix_center, px = hd_fadd(float(x), pc), py = hd_fadd(float(y), pc);\n"
        "    float tx0 = tris[ti].x[0], ty0 = tris[ti].y[0];\n"
        "    float tx1 = tris[ti].x[1], ty1 = tris[ti].y[1];\n"
        "    float tx2 = tris[ti].x[2], ty2 = tris[ti].y[2];\n"
        "    float w0 = hd_edge(tx1, ty1, tx2, ty2, px, py);\n"
        "    float w1 = hd_edge(tx2, ty2, tx0, ty0, px, py);\n"
        "    float w2 = hd_edge(tx0, ty0, tx1, ty1, px, py);\n"
        "    if (w0 < 0.0 || w1 < 0.0 || w2 < 0.0) {\n        return;\n    }\n"
        "    if ((w0 == 0.0 && !hd_owns(tx1, ty1, tx2, ty2)) ||\n"
        "        (w1 == 0.0 && !hd_owns(tx2, ty2, tx0, ty0)) ||\n"
        "        (w2 == 0.0 && !hd_owns(tx0, ty0, tx1, ty1))) {\n"
        "        return;\n    }\n"
        "    if (draws[di].cliprule != 0xffffu && !hd_clip(di, x, y)) {\n        return;\n    }\n"
        "    float area = tris[ti].area;\n"
        "    float b0 = hd_div(w0, area), b1 = hd_div(w1, area), b2 = hd_div(w2, area);\n"
        "    float pw = 0.0;\n"
        "    if (draws[di].persp != 0u) {\n"
        "        float s = hd_sum3(b0, tris[ti].w[0], b1, tris[ti].w[1], b2, tris[ti].w[2]);\n"
        "        pw = s != 0.0 ? hd_div(1.0, s) : 0.0;\n"
        "    }\n");
    g_string_append(o,
        "    uint col[4] = uint[4](0u, 0u, 0u, 0u);\n"
        "    bool killed = false, c0 = false;\n"
        "    int pg = draws[di].pos_gpr;\n");
    for (q = 0; q < nq; q++) {
        const char *x = set_sfx[q];

        if (q) {
            /* bary() at the pixel to the right (1) or below (2) */
            g_string_append_printf(o,
                "    float px%s = hd_fadd(float(x + %d), pc), py%s = hd_fadd(float(y + %d), pc);\n"
                "    float b0%s = hd_div(hd_edge(tx1, ty1, tx2, ty2, px%s, py%s), area);\n"
                "    float b1%s = hd_div(hd_edge(tx2, ty2, tx0, ty0, px%s, py%s), area);\n"
                "    float b2%s = hd_div(hd_edge(tx0, ty0, tx1, ty1, px%s, py%s), area);\n"
                "    float pw%s = 0.0;\n"
                "    if (draws[di].persp != 0u) {\n"
                "        float s = hd_sum3(b0%s, tris[ti].w[0], b1%s, tris[ti].w[1], b2%s, tris[ti].w[2]);\n"
                "        pw%s = s != 0.0 ? hd_div(1.0, s) : 0.0;\n"
                "    }\n",
                x, q == 1, x, q == 2, x, x, x, x, x, x, x, x, x, x, x, x, x, x);
        }
        g_string_append_printf(o,
            "    uvec4 r%s[%u];\n"
            "    for (uint i = 0u; i < %uu; i++) {\n"
            "        r%s[i] = uvec4(0u);\n"
            "    }\n"
            "    uint pv%s[5] = uint[5](0u, 0u, 0u, 0u, 0u);\n",
            x, ng, ng, x, x);
        for (i = 0; i < ir->ngpr; i++) {
            g_string_append_printf(o, "    if (pg == %u) {\n        r%s[%u] = uvec4(U(px%s), "
                                   "U(py%s), U(0.0), U(1.0));\n    }\n",
                                   i, x, i, x, x);
        }
        g_string_append_printf(o, "    r%s[%u][0] = U(b1%s);\n    r%s[%u][1] = U(b2%s);\n",
                               x, ir->map[0], x, x, ir->map[0], x);
    }
    for (i = 0; i < ir->nops; i++) {
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
                    emit_fetch(o, ir, &ir->fetches[op->first + j], q);
                }
            }
            break;
        default: {
            const HD5770IRExport *e = &ir->exports[op->first];

            for (k = 0; k < 4; k++) {
                unsigned sel = e->sel[k];

                if (sel < 4) {
                    g_string_append_printf(o, "    col[%u] = r[%u][%u];\n", k,
                                           e->gpr, sel);
                } else {
                    g_string_append_printf(o, "    col[%u] = %s;\n", k,
                                           sel == 4 ? "U(0.0)" :
                                           sel == 5 ? "U(1.0)" : "0u");
                }
            }
            g_string_append(o, "    c0 = true;\n");
            break;
        }
        }
    }
    g_string_append(o,
        "    if (killed || !c0) {\n        return;\n    }\n"
        "    if (draws[di].alpha_test != 0u && !hd_alpha(di, F(col[3]))) {\n"
        "        return;\n    }\n"
        "    hd_put(di, x, y, col);\n"
        "}\n");
    return g_string_free(o, false);
}
