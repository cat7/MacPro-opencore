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
"    uint space, pt_off, pg0;\n"
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
"    case 0x81: return U(exp2(a));\n"
"    case 0x82: return U(a == 0.0f ? -FLT_MAX : log2(a));\n"
"    case 0x83: return U(log2(a));\n"
"    case 0x84: return U(a == 0.0f ? FLT_MAX : 1.0f / a);\n"
"    case 0x85: return U(a == 0.0f ? 0.0f : 1.0f / a);\n"
"    case 0x86: return U(1.0f / a);\n"
"    case 0x87: return U(a == 0.0f ? FLT_MAX : 1.0f / sqrt(a));\n"
"    case 0x88: return U(a == 0.0f ? 0.0f : 1.0f / sqrt(a));\n"
"    case 0x89: return U(1.0f / sqrt(a));\n"
"    case 0x8a: return U(sqrt(a));\n"
"    case 0x8d: return U(sin(a * 2.0f * 3.14159265358979f));\n"
"    case 0x8e: return U(cos(a * 2.0f * 3.14159265358979f));\n"
"    case 0x8f: case 0x91: return s0 * s1;\n"
"    case 0x90: return (uint)(((long)I(s0) * I(s1)) >> 32);\n"
"    case 0x92: return (uint)(((ulong)s0 * s1) >> 32);\n"
"    case 0xcc: return s0;\n"
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
"static __attribute__((always_inline)) void hd_texel_s(device uchar *vram, device uchar *ram, device const uchar *st,\n"
"                     device const GTex &t, device const GSurf &ls, int x, int y,\n"
"                     thread uint *out)\n"
"{\n"
"    device uchar *p = hd_at(ls, hd_addr(ls, x, y), vram, ram, st);\n"
"    uint4 v;\n"
"    uint c[4];\n"
"\n"
"    if (t.unorm8) {\n"
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
"    for (uint k = 0; k < 4; k++) {\n"
"        uint s = t.dst_sel[k];\n"
"\n"
"        out[k] = s < 4 ? v[s] : s == 5 ? U(1.0f) : U(0.0f);\n"
"    }\n"
"}\n"
"\n"
"static __attribute__((always_inline)) void hd_texel(device uchar *vram, device uchar *ram, device const uchar *st,\n"
"                     device const GTex &t, int x, int y,\n"
"                     thread uint *out)\n"
"{\n"
"    hd_texel_s(vram, ram, st, t, t.s, x, y, out);\n"
"}\n"
"\n"
"static __attribute__((always_inline)) void hd_point_s(device uchar *vram, device uchar *ram, device const uchar *st,\n"
"                     device const GTex &t, device const GSurf &ls, uint lw, uint lh,\n"
"                     int x, int y, thread uint *out)\n"
"{\n"
"    if (!hd_wrap(x, lw, t.samp0 & 7) || !hd_wrap(y, lh, (t.samp0 >> 3) & 7)) {\n"
"        for (uint k = 0; k < 4; k++) {\n"
"            out[k] = U(t.border[k]);\n"
"        }\n"
"        return;\n"
"    }\n"
"    hd_texel_s(vram, ram, st, t, ls, x, y, out);\n"
"}\n"
"\n"
"static __attribute__((always_inline)) void hd_point(device uchar *vram, device uchar *ram, device const uchar *st,\n"
"                     device const GTex &t, int x, int y,\n"
"                     thread uint *out)\n"
"{\n"
"    hd_point_s(vram, ram, st, t, t.s, t.w, t.h, x, y, out);\n"
"}\n"
"\n"
"static __attribute__((always_inline)) int hd_floori(float f)\n"
"{\n"
"    return isnan(f) ? 0 : (int)clamp(floor(f), -2147483648.0f, 2147483520.0f);\n"
"}\n"
"\n"
"/* tfetch_stage() for the pixel shader */\n"
"static __attribute__((always_inline)) void hd_tfetch(device uchar *vram, device uchar *ram, device const uchar *st,\n"
"                      device const GTex &t, uint w0,\n"
"                      uint w1, uint w2, thread const uint *c, thread uint *v)\n"
"{\n"
"    uint op = w0 & 0x1f;\n"
"    int ox = ((int)(w2 << 27)) >> 27, oy = ((int)(w2 << 22)) >> 27;\n"
"    float u, vv;\n"
"\n"
"    v[0] = v[1] = v[2] = v[3] = U(0.0f);\n"
"    if (op == 0x04) {\n"
"        uint lv = min(c[0] + t.base_lv, 31u);\n"
"\n"
"        v[0] = max(t.w >> lv, 1u);\n"
"        v[1] = t.dim == 0 ? 1u : max(t.h >> lv, 1u);\n"
"        v[2] = t.depth;\n"
"        v[3] = t.levels;\n"
"        return;\n"
"    }\n"
"    if (op == 0x03) {\n"
"        int x = I(c[0]) + ox / 2, y = I(c[1]) + oy / 2;\n"
"\n"
"        if (x >= 0 && y >= 0 && x < (int)t.w && y < (int)t.h) {\n"
"            hd_texel(vram, ram, st, t, x, y, v);\n"
"        }\n"
"        return;\n"
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
"    if (((t.samp0 >> 9) & 1) == 0 && ((t.samp0 >> 11) & 1) == 0) {\n"
"        hd_point(vram, ram, st, t, hd_floori(u), hd_floori(vv), v);\n"
"        return;\n"
"    }\n"
"    {\n"
"        float fx = u - 0.5f, fy = vv - 0.5f;\n"
"        int x0 = hd_floori(fx), y0 = hd_floori(fy);\n"
"        float ax = fx - x0, ay = fy - y0;\n"
"        uint t00[4], t10[4], t01[4], t11[4];\n"
"\n"
"        hd_point(vram, ram, st, t, x0, y0, t00);\n"
"        hd_point(vram, ram, st, t, x0 + 1, y0, t10);\n"
"        hd_point(vram, ram, st, t, x0, y0 + 1, t01);\n"
"        hd_point(vram, ram, st, t, x0 + 1, y0 + 1, t11);\n"
"        for (uint k = 0; k < 4; k++) {\n"
"            float top = fma(F(t10[k]) - F(t00[k]), ax, F(t00[k]));\n"
"            float bot = fma(F(t11[k]) - F(t01[k]), ax, F(t01[k]));\n"
"\n"
"            v[k] = U(fma(bot - top, ay, top));\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"/* sample_lv(): point or bilinear sample of a level at (u, vv) in its texels */\n"
"static __attribute__((always_inline)) void hd_sample_lv(device uchar *vram, device uchar *ram, device const uchar *st,\n"
"                      device const GTex &t, uint l, float u, float vv, bool lin,\n"
"                      thread uint *v)\n"
"{\n"
"    device const GSurf &ls = l ? ((device const GSurf *)(st + t.lev_off))[l] : t.s;\n"
"    uint lw = t.lw[l], lh = t.lh[l];\n"
"\n"
"    if (!lin) {\n"
"        hd_point_s(vram, ram, st, t, ls, lw, lh, hd_floori(u), hd_floori(vv), v);\n"
"        return;\n"
"    }\n"
"    float fx = u - 0.5f, fy = vv - 0.5f;\n"
"    int x0 = hd_floori(fx), y0 = hd_floori(fy);\n"
"    float ax = fx - x0, ay = fy - y0;\n"
"    uint t00[4], t10[4], t01[4], t11[4];\n"
"\n"
"    hd_point_s(vram, ram, st, t, ls, lw, lh, x0, y0, t00);\n"
"    hd_point_s(vram, ram, st, t, ls, lw, lh, x0 + 1, y0, t10);\n"
"    hd_point_s(vram, ram, st, t, ls, lw, lh, x0, y0 + 1, t01);\n"
"    hd_point_s(vram, ram, st, t, ls, lw, lh, x0 + 1, y0 + 1, t11);\n"
"    for (uint k = 0; k < 4; k++) {\n"
"        float top = fma(F(t10[k]) - F(t00[k]), ax, F(t00[k]));\n"
"        float bot = fma(F(t11[k]) - F(t01[k]), ax, F(t01[k]));\n"
"\n"
"        v[k] = U(fma(bot - top, ay, top));\n"
"    }\n"
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
"/* tfetch_stage() with derivatives, for a mipmapped texture */\n"
"static __attribute__((always_inline)) void hd_tfetch_d(device uchar *vram, device uchar *ram, device const uchar *st,\n"
"                      device const GTex &t, uint w0,\n"
"                      uint w1, uint w2, thread const uint *c, thread const float *dv,\n"
"                      thread uint *v)\n"
"{\n"
"    uint op = w0 & 0x1f, lv;\n"
"    int ox = ((int)(w2 << 27)) >> 27, oy = ((int)(w2 << 22)) >> 27;\n"
"    float lod = 0.0f, l0, f;\n"
"    bool lin;\n"
"    uint v1[4];\n"
"\n"
"    if (op == 0x03 || op == 0x04 || !t.mip || !((w1 >> 28) & 1) || !((w1 >> 29) & 1)) {\n"
"        hd_tfetch(vram, ram, st, t, w0, w1, w2, c, v);\n"
"        return;\n"
"    }\n"
"    if (op == 0x11) {\n"
"        lod = F(c[3]);\n"
"    } else if (op != 0x13) {\n"
"        float ax = dv[0] * t.w, ay = dv[1] * t.h, bx = dv[4] * t.w, by = dv[5] * t.h;\n"
"        float a = fma(ax, ax, ay * ay), b = fma(bx, bx, by * by);\n"
"\n"
"        lod = 0.5f * hd_lodlog2(max(a, b));\n"
"        if (op == 0x12) {\n"
"            lod += F(c[3]);\n"
"        }\n"
"    }\n"
"    lod = min(max(lod + t.lod_bias, t.min_lod), t.max_lod);\n"
"    if (!(lod > 0.0f)) {\n"
"        hd_sample_lv(vram, ram, st, t, 0, fma((float)ox, 0.5f, F(c[0]) * t.w),\n"
"                     fma((float)oy, 0.5f, F(c[1]) * t.h), (t.samp0 >> 9) & 1, v);\n"
"        return;\n"
"    }\n"
"    lin = (t.samp0 >> 11) & 1;\n"
"    if (t.mip_filter == 1) {\n"
"        lv = min((uint)floor(lod + 0.5f), t.last_lv);\n"
"        hd_sample_lv(vram, ram, st, t, lv, fma((float)ox, 0.5f, F(c[0]) * t.lw[lv]),\n"
"                     fma((float)oy, 0.5f, F(c[1]) * t.lh[lv]), lin, v);\n"
"        return;\n"
"    }\n"
"    l0 = floor(lod);\n"
"    f = lod - l0;\n"
"    lv = min((uint)l0, t.last_lv);\n"
"    hd_sample_lv(vram, ram, st, t, lv, fma((float)ox, 0.5f, F(c[0]) * t.lw[lv]),\n"
"                 fma((float)oy, 0.5f, F(c[1]) * t.lh[lv]), lin, v);\n"
"    if (lv < t.last_lv) {\n"
"        lv++;\n"
"        hd_sample_lv(vram, ram, st, t, lv, fma((float)ox, 0.5f, F(c[0]) * t.lw[lv]),\n"
"                     fma((float)oy, 0.5f, F(c[1]) * t.lh[lv]), lin, v1);\n"
"        for (uint k = 0; k < 4; k++) {\n"
"            v[k] = U(fma(F(v1[k]) - F(v[k]), f, F(v[k])));\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"/* vfetch_stage() for the pixel shader */\n"
"static __attribute__((always_inline)) void hd_vfetch(device const GDraw &u, device const uchar *st, uint w0,\n"
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
"static __attribute__((always_inline)) bool hd_db(device uchar *vram, device uchar *ram,\n"
"                  device const uchar *st, device const GDraw &u,\n"
"                  int x, int y, float z, bool tfront)\n"
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
"        uint ref = (u.rctl & 2) ? u.sclear : srm & 0xff;\n"
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
"static __attribute__((always_inline)) void hd_put(device uchar *vram, device uchar *ram, device const uchar *st,\n"
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

char *hd5770_msl_program(const HD5770IR *ir)
{
    return hd5770_msl_program_mip(ir, 0);
}

char *hd5770_msl_program_mip(const HD5770IR *ir, uint32_t mipmask)
{
    GString *o = g_string_new(hd5770_msl_lib);
    GString *tab = g_string_new("constant float hd_u8[256] = {");
    const char *mark = strstr(o->str, "HD_U8_TABLE");
    unsigned i, j, k, q, nq = ir->grad || mipmask ? 3 : 1;
    unsigned ng = MAX(ir->ngpr, 1u);

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
        "static float hd_mulieee(float a, float b) { return a * b; }\n"
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
        "}\n"
        "kernel void hd_main(device uchar *vram [[buffer(0)]],\n"
        "    device const uchar *st [[buffer(1)]], device const uint *tiles [[buffer(2)]],\n"
        "    device uchar *ram [[buffer(3)]],\n"
        "    uint tg [[threadgroup_position_in_grid]],\n"
        "    uint l [[thread_index_in_threadgroup]])\n"
        "{\n"
        "    device const uint *te = tiles + tg * 8;\n"
        "    device const GDraw &u = *(device const GDraw *)(st + te[0]);\n"
        "    device const GTri *tp = (device const GTri *)(st + te[1]);\n"
        "    device const float *prm = (device const float *)(st + te[2]);\n"
        "    device const GTex *tex = (device const GTex *)(st + te[3]);\n"
        "    device const uint *kv = (device const uint *)(st + te[4]);\n"
        "    uint sw = te[7] & 0xff;\n"
        "    int x = (int)te[5] + (int)(l % sw), y = (int)te[6] + (int)(l / sw);\n"
        "    float pc = u.pix_center, px = x + pc, py = y + pc;\n"
        "    float w0, w1, w2;\n"
        "    bool in = hd_cover(*tp, x, y, px, py, w0, w1, w2);\n"
        "    if (!in && (te[7] & 0x100)) {\n"
        "        /* the other half of the primitive; they share no pixel */\n"
        "        tp = (device const GTri *)(st + te[1] + 128);\n"
        "        prm += (te[7] >> 16) * 4;\n"
        "        in = hd_cover(*tp, x, y, px, py, w0, w1, w2);\n"
        "    }\n"
        "    if (!in) {\n        return;\n    }\n"
        "    device const GTri &t = *tp;\n"
        "    if (u.cliprule != 0xffff && !hd_clip(u, x, y)) {\n        return;\n    }\n"
        "    uint col[4] = { 0, 0, 0, 0 };\n"
        "    bool killed = false, c0 = false;\n");
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
            "    }\n"
            "    uint R%u[%u][4], P%u[5] = { 0, 0, 0, 0, 0 };\n"
            "    for (uint i = 0; i < %u; i++) {\n"
            "        R%u[i][0] = R%u[i][1] = R%u[i][2] = R%u[i][3] = 0;\n"
            "    }\n",
            q, q == 1, q, q == 2, q, q, q, q, q, q, q, q, q, q, q, q, q, q,
            q, ng, q, ng, q, q, q, q);
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
                    emit_fetch(o, ir, &ir->fetches[op->first + j], q, mipmask,
                               op->first + j);
                }
            }
            break;
        default: {
            const HD5770IRExport *e = &ir->exports[op->first];

            for (k = 0; k < 4; k++) {
                unsigned sel = e->sel[k];

                if (sel < 4) {
                    g_string_append_printf(o, "    col[%u] = R0[%u][%u];\n", k,
                                           e->gpr, sel);
                } else {
                    g_string_append_printf(o, "    col[%u] = %s;\n", k,
                                           sel == 4 ? "U(0.0f)" :
                                           sel == 5 ? "U(1.0f)" : "0u");
                }
            }
            g_string_append(o, "    c0 = true;\n");
            break;
        }
        }
    }
    g_string_append(o,
        "    if (killed) {\n        return;\n    }\n"
        "    if (c0 && u.alpha_test && !hd_alpha(u, F(col[3]))) {\n        return;\n    }\n"
        "    if (u.db && !hd_db(vram, ram, st, u, x, y,\n"
        "                       fma(t.z[2], B2_0, fma(t.z[0], B0_0, t.z[1] * B1_0)),\n"
        "                       t.front != 0)) {\n        return;\n    }\n"
        "    if (!c0 || !u.wmask) {\n        return;\n    }\n"
        "    hd_put(vram, ram, st, u, x, y, col);\n"
        "}\n");
    return g_string_free(o, false);
}
