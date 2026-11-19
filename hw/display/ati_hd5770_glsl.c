/*
 * ATI Radeon HD 5770 (Juniper) shader IR to Vulkan GLSL: the stages of
 * the rasterisation pipeline (ati_hd5770_vk.c).
 *
 * The library texts are the port of hd5770_msl_lib (ati_hd5770_msl.c):
 * as functions, what the CPU path computes. Every float operation is
 * precise and rounded as the CPU path rounds it: a * b + c is fused only
 * where the CPU path's compiler fuses it (HD_FUSED: on AArch64 hosts,
 * not on x86, whose build has no FMA), and in a vertex stage division,
 * square root, exp2 and log2 are corrected to the IEEE result through
 * doubles (HD_EXACT). A fragment stage takes the host's: doubles cost a
 * full-screen pass milliseconds.
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

static const char glsl_arith[] =
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
"#if HD_EXACT\n"
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
"#else\n"
"    return q;\n"
"#endif\n"
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
"#if HD_EXACT\n"
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
"#endif\n"
"    return s;\n"
"}\n"
"\n"
"/* exp2f() and log2f(), correctly rounded through doubles for finite\n"
" * arguments (the driver's are approximations) */\n"
"float hd_exp2(float a)\n"
"{\n"
"#if HD_EXACT\n"
"    if (hd_isnan(a) || a < -151.0 || a >= 128.0) {\n"
"        return exp2(a);\n"
"    }\n"
"    precise double x = double(a), n = roundEven(x), t = (x - n) * 0.69314718055994530942LF;\n"
"    precise double p = 1.0LF / 6227020800.0LF;\n"
"    p = p * t + 1.0LF / 479001600.0LF;\n"
"    p = p * t + 1.0LF / 39916800.0LF;\n"
"    p = p * t + 1.0LF / 3628800.0LF;\n"
"    p = p * t + 1.0LF / 362880.0LF;\n"
"    p = p * t + 1.0LF / 40320.0LF;\n"
"    p = p * t + 1.0LF / 5040.0LF;\n"
"    p = p * t + 1.0LF / 720.0LF;\n"
"    p = p * t + 1.0LF / 120.0LF;\n"
"    p = p * t + 1.0LF / 24.0LF;\n"
"    p = p * t + 1.0LF / 6.0LF;\n"
"    p = p * t + 0.5LF;\n"
"    p = p * t + 1.0LF;\n"
"    p = p * t + 1.0LF;\n"
"    return float(ldexp(p, int(n)));\n"
"#else\n"
"    return exp2(a);\n"
"#endif\n"
"}\n"
"\n"
"float hd_log2(float a)\n"
"{\n"
"#if HD_EXACT\n"
"    if (hd_isnan(a) || a <= 0.0 || isinf(a)) {\n"
"        return log2(a);\n"
"    }\n"
"    int e;\n"
"    precise double m = frexp(double(a), e);\n"
"    if (m < 0.70710678118654752440LF) {\n"
"        m = m * 2.0LF;\n"
"        e--;\n"
"    }\n"
"    precise double s = (m - 1.0LF) / (m + 1.0LF), z = s * s;\n"
"    precise double q = 1.0LF / 25.0LF;\n"
"    q = q * z + 1.0LF / 23.0LF;\n"
"    q = q * z + 1.0LF / 21.0LF;\n"
"    q = q * z + 1.0LF / 19.0LF;\n"
"    q = q * z + 1.0LF / 17.0LF;\n"
"    q = q * z + 1.0LF / 15.0LF;\n"
"    q = q * z + 1.0LF / 13.0LF;\n"
"    q = q * z + 1.0LF / 11.0LF;\n"
"    q = q * z + 1.0LF / 9.0LF;\n"
"    q = q * z + 1.0LF / 7.0LF;\n"
"    q = q * z + 1.0LF / 5.0LF;\n"
"    q = q * z + 1.0LF / 3.0LF;\n"
"    q = q * z + 1.0LF;\n"
"    return float(double(e) + 2.0LF * s * q * 1.44269504088896340736LF);\n"
"#else\n"
"    return log2(a);\n"
"#endif\n"
"}\n"
"\n"
"/* float pairs (hi, lo) */\n"
"vec2 hd_df_norm(float s, float e)\n"
"{\n"
"    precise float h = s + e;\n"
"    precise float l = e - (h - s);\n"
"    return vec2(h, l);\n"
"}\n"
"vec2 hd_df_add(vec2 x, vec2 y)\n"
"{\n"
"    precise float s = x.x + y.x;\n"
"    precise float bb = s - x.x;\n"
"    precise float e = (x.x - (s - bb)) + (y.x - bb);\n"
"    precise float l = e + x.y + y.y;\n"
"    return hd_df_norm(s, l);\n"
"}\n"
"vec2 hd_df_mul(vec2 x, vec2 y)\n"
"{\n"
"    precise float p = x.x * y.x;\n"
"    precise float e = fma(x.x, y.x, -p);\n"
"    precise float l = e + (x.x * y.y + x.y * y.x);\n"
"    return hd_df_norm(p, l);\n"
"}\n"
"\n"
"/* sinf(t) (c 0) and cosf(t) (c 1) rounded as the host's: t - k pi/2 and\n"
" * the series in float pairs, as hd_sincos in the MSL library */\n"
"float hd_sincos(float t, uint c)\n"
"{\n"
"    if (!(abs(t) <= 1.0e4)) {\n"
"        return c != 0u ? cos(t) : sin(t);\n"
"    }\n"
"    if (t == 0.0) {\n"
"        return c != 0u ? 1.0 : t;\n"
"    }\n"
"    precise float tk = t * 6.366197467e-01;\n"
"    precise float k = roundEven(tk);\n"
"    precise float h1 = k * 1.570796371e+00;\n"
"    precise float e1 = fma(k, 1.570796371e+00, -h1);\n"
"    precise float h2 = k * -4.371138829e-08;\n"
"    precise float e2 = fma(k, -4.371138829e-08, -h2);\n"
"    precise float h3 = k * 1.715124510e-15;\n"
"    precise float r0 = t - h1;\n"
"    vec2 r = vec2(r0, 0.0);\n"
"\n"
"    r = hd_df_add(r, vec2(-e1, 0.0));\n"
"    r = hd_df_add(r, vec2(-h2, 0.0));\n"
"    r = hd_df_add(r, vec2(-e2, 0.0));\n"
"    r = hd_df_add(r, vec2(h3, 0.0));\n"
"    vec2 r2 = hd_df_mul(r, r), p, s;\n"
"    float x = r2.x;\n"
"    uint q = (uint(int(k)) + c) & 3u;\n"
"\n"
"    if ((q & 1u) != 0u) {\n"
"        precise float a = fma(fma(fma(fma(-1.561920681e-16, x, 4.779477256e-14), x, -1.147074536e-11),\n"
"                                  x, 2.087675588e-09), x, -2.755731998e-07);\n"
"        p = vec2(a, 0.0);\n"
"        p = hd_df_add(hd_df_mul(p, r2), vec2(2.480158764e-05, -3.406996094e-13));\n"
"        p = hd_df_add(hd_df_mul(p, r2), vec2(-1.388888923e-03, 3.363109444e-11));\n"
"        p = hd_df_add(hd_df_mul(p, r2), vec2(4.166666791e-02, -1.241763470e-09));\n"
"        p = hd_df_add(hd_df_mul(p, r2), vec2(-5.000000000e-01, 0.0));\n"
"        s = hd_df_add(vec2(1.0, 0.0), hd_df_mul(p, r2));\n"
"    } else {\n"
"        precise float a = fma(fma(fma(fma(2.811457359e-15, x, -7.647163610e-13), x, 1.605904437e-10),\n"
"                                  x, -2.505210794e-08), x, 2.755731884e-06);\n"
"        p = vec2(a, 0.0);\n"
"        p = hd_df_add(hd_df_mul(p, r2), vec2(-1.984127011e-04, 2.725596875e-12));\n"
"        p = hd_df_add(hd_df_mul(p, r2), vec2(8.333333768e-03, -4.346172033e-10));\n"
"        p = hd_df_add(hd_df_mul(p, r2), vec2(-1.666666716e-01, 4.967053879e-09));\n"
"        s = hd_df_add(r, hd_df_mul(hd_df_mul(p, r2), r));\n"
"    }\n"
"    precise float v = s.x + s.y;\n"
"    return q >= 2u ? -v : v;\n"
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
;

static const char glsl_formats[] =
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
"/* float_to_half(): round to nearest even */\n"
"uint hd_f2h(float f)\n"
"{\n"
"    uint x = U(f), s = (x >> 16u) & 0x8000u, m = x & 0x7fffffu;\n"
"    int e = int((x >> 23u) & 0xffu) - 127 + 15;\n"
"\n"
"    if (((x >> 23u) & 0xffu) == 0xffu) {\n"
"        return s | 0x7c00u | (m != 0u ? 0x200u | (m >> 13u) : 0u);\n"
"    }\n"
"    if (e >= 31) {\n"
"        return s | 0x7c00u;\n"
"    }\n"
"    if (e <= 0) {\n"
"        if (e < -10) {\n"
"            return s;\n"
"        }\n"
"        m |= 0x800000u;\n"
"        uint sh = uint(14 - e), h = m >> sh, r = m & ((1u << sh) - 1u);\n"
"        if (r > (1u << (sh - 1u)) || (r == (1u << (sh - 1u)) && (h & 1u) != 0u)) {\n"
"            h++;\n"
"        }\n"
"        return s | h;\n"
"    }\n"
"    uint h = s | (uint(e) << 10u) | (m >> 13u), r = m & 0x1fffu;\n"
"    if (r > 0x1000u || (r == 0x1000u && (h & 1u) != 0u)) {\n"
"        h++;\n"
"    }\n"
"    return h;\n"
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
;

/* eval(), post(), hd_pred(), hd_cube() */
static const char glsl_alu[] =
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
"    case 0x81u: return U(hd_exp2(a));\n"
"    case 0x82u: return U(a == 0.0 ? -FLT_MAX : hd_log2(a));\n"
"    case 0x83u: return U(hd_log2(a));\n"
"    case 0x84u: return U(a == 0.0 ? FLT_MAX : hd_div(1.0, a));\n"
"    case 0x85u: return U(a == 0.0 ? 0.0 : hd_div(1.0, a));\n"
"    case 0x86u: return U(hd_div(1.0, a));\n"
"    case 0x87u: return U(a == 0.0 ? FLT_MAX : hd_div(1.0, hd_sqrt(a)));\n"
"    case 0x88u: return U(a == 0.0 ? 0.0 : hd_div(1.0, hd_sqrt(a)));\n"
"    case 0x89u: return U(hd_div(1.0, hd_sqrt(a)));\n"
"    case 0x8au: return U(hd_sqrt(a));\n"
"    case 0x8du: return U(hd_sincos(hd_fmul(hd_fmul(a, 2.0), 3.14159265358979), 0u));\n"
"    case 0x8eu: return U(hd_sincos(hd_fmul(hd_fmul(a, 2.0), 3.14159265358979), 1u));\n"
"    case 0x8fu: case 0x91u: return s0 * s1;\n"
"    case 0x90u: { int hi, lo; imulExtended(I(s0), I(s1), hi, lo); return uint(hi); }\n"
"    case 0x92u: { uint hi, lo; umulExtended(s0, s1, hi, lo); return hi; }\n"
"    case 0xccu: return uint(clamp(I(s0), -256, 255));\n"
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
"/* PRED_SET*: the value eval() returns, and the predicate */\n"
"uint hd_pred(uint op, uint mods, uint s0, uint s1, out bool p)\n"
"{\n"
"    float a = hd_fmod(mods, 0u, s0), b = hd_fmod(mods, 1u, s1);\n"
"\n"
"    switch (op) {\n"
"    case 0x20u: p = a == b; break;\n"
"    case 0x21u: p = a > b; break;\n"
"    case 0x22u: p = a >= b; break;\n"
"    case 0x23u: p = a != b; break;\n"
"    case 0x1eu: p = s0 > s1; break;\n"
"    case 0x1fu: p = s0 >= s1; break;\n"
"    case 0x42u: p = I(s0) == I(s1); return p ? 0u : 1u;\n"
"    case 0x43u: p = I(s0) > I(s1); return p ? 0u : 1u;\n"
"    case 0x44u: p = I(s0) >= I(s1); return p ? 0u : 1u;\n"
"    case 0x45u: p = I(s0) != I(s1); return p ? 0u : 1u;\n"
"    default: p = false; return 0u;\n"
"    }\n"
"    return U(p ? 0.0 : 1.0);\n"
"}\n"
"\n"
"/* cube(): CUBE's results in slots x, y, z, w for the vector (x, y, z) */\n"
"void hd_cube(float x, float y, float z, out uint o[4])\n"
"{\n"
"    float ax = abs(x), ay = abs(y), az = abs(z), sc, tc, ma, face;\n"
"\n"
"    if (ax >= ay && ax >= az) {\n"
"        ma = x;\n"
"        face = x >= 0.0 ? 0.0 : 1.0;\n"
"        sc = x >= 0.0 ? hd_neg(z) : z;\n"
"        tc = hd_neg(y);\n"
"    } else if (ay >= az) {\n"
"        ma = y;\n"
"        face = y >= 0.0 ? 2.0 : 3.0;\n"
"        sc = x;\n"
"        tc = y >= 0.0 ? z : hd_neg(z);\n"
"    } else {\n"
"        ma = z;\n"
"        face = z >= 0.0 ? 4.0 : 5.0;\n"
"        sc = z >= 0.0 ? x : hd_neg(x);\n"
"        tc = hd_neg(y);\n"
"    }\n"
"    o[0] = U(tc);\n"
"    o[1] = U(sc);\n"
"    o[2] = U(hd_fmul(2.0, ma));\n"
"    o[3] = U(face);\n"
"}\n"
"\n";

static void src_expr(GString *o, const HD5770IRSrc *s)
{
    switch (s->kind) {
    case HD5770_IRS_GPR:
        g_string_append_printf(o, "r[%u][%u]", s->idx, s->chan);
        break;
    case HD5770_IRS_PV:
        g_string_append_printf(o, "pv[%u]", s->chan);
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

static void emit_group(GString *o, const HD5770IR *ir, const HD5770IRGroup *g)
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
            src_expr(o, &in->src[k]);
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
                                   "%s, b0, b1, b2, pw));\n", i, in->prm,
                                   in->dst_chan, in->flat ? "true" : "false");
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
            g_string_append(o, ", k); if (k) { killed = true; } }\n");
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
    if (g->cube) {
        const char *v[3] = { "0.0", "0.0", "0.0" };
        char b[3][48];

        for (i = 0; i < g->n; i++) {
            const HD5770IRInst *in = &ir->insts[g->first + i];
            int at = in->slot == 2 ? 0 : in->slot == 3 ? 1 : in->slot == 0 ? 2 : -1;

            if (in->cube && at >= 0) {
                snprintf(b[at], sizeof(b[at]), "hd_fmod(%uu, 0u, s%u_0)",
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
        g_string_append_printf(o, "        pv[%u] = q%u;\n", in->slot, i);
    }
    for (i = 0; i < g->n; i++) {
        const HD5770IRInst *in = &ir->insts[g->first + i];

        if (in->wr) {
            g_string_append_printf(o, "        r[%u][%u] = q%u;\n", in->dst,
                                   in->dst_chan, i);
        }
    }
    g_string_append(o, "    }\n");
}

/* the contents of the tables buffer */
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
                snprintf(v[k], sizeof(v[k]), "r[%u][%u]", e->gpr, sel);
            } else {
                snprintf(v[k], sizeof(v[k]), "%s", sel == 4 ? "U(0.0)" :
                         sel == 5 ? "U(1.0)" : "0u");
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

/*
 * The fetches of a vertex shader for the vertex stage: its arguments are
 * words from push constant vso on, a buffer's bytes lie in the words.
 */
static const char glsl_vs_lib[] =
"#define VW(i) words[vso + (i)]\n"
"/* storage buffer 0 offset of byte o of the buffer whose words are at b */\n"
"uint hd_bat(uint b, uint o)\n"
"{\n"
"    if (words[b + 1u] != 0u) {\n"
"        uint q = words[b + 3u] + o;\n"
"\n"
"        return words[b] + words[words[b + 2u] + (q >> 12u)] * 4096u + (q & 4095u);\n"
"    }\n"
"    return words[b] + o;\n"
"}\n"
"uint hd_brd(uint b, uint o, uint n)\n"
"{\n"
"    uint v = 0u;\n"
"\n"
"    if (n == 4u && (o & 3u) == 0u) {\n"
"        return vram[hd_bat(b, o) >> 2u];\n"
"    }\n"
"    for (uint i = 0u; i < n; i++) {\n"
"        uint a = hd_bat(b, o + i);\n"
"\n"
"        v |= ((vram[a >> 2u] >> ((a & 3u) * 8u)) & 0xffu) << (8u * i);\n"
"    }\n"
"    return v;\n"
"}\n"
"/* vfetch_vs(): the fetch type's offset, then vfetch_stage() */\n"
"void hd_vsfetch(uint w0, uint w1, uint w2, uint index, uint fsb, out uint o[4])\n"
"{\n"
"    uint bid = (w0 >> 8u) & 0xffu, off = w2 & 0xffffu, fmt, num, sgn, n, k;\n"
"    uint c[4], w[4];\n"
"\n"
"    if (((w0 >> 5u) & 3u) == 0u) {\n"
"        index += VW(7u);\n"
"    } else if (((w0 >> 5u) & 3u) == 1u) {\n"
"        index += VW(8u);\n"
"    }\n"
"    if (bid == 128u) {\n"
"        uint e = index + off / 16u;\n"
"\n"
"        for (k = 0u; k < 4u; k++) {\n"
"            o[k] = e < 256u ? words[VW(5u) + e * 4u + k] : 0u;\n"
"        }\n"
"        return;\n"
"    }\n"
"    o[0] = U(0.0); o[1] = U(0.0); o[2] = U(0.0);\n"
"    o[3] = U(1.0);\n"
"    for (k = 0u; k < VW(6u) && VW(24u + 16u * k) != bid + fsb; k++) {\n"
"    }\n"
"    if (k == VW(6u)) {\n"
"        return;\n"
"    }\n"
"    uint vb = vso + 24u + 16u * k;\n"
"    uint size = words[vb + 5u] + 1u, r2 = words[vb + 6u], stride = (r2 >> 8u) & 0x7ffu;\n"
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
"    if (mhi != 0u || at2 < at || at2 + n < at2 || at2 + n > size) {\n"
"        o[0] = 0u; o[1] = 0u; o[2] = 0u;\n"
"        return;\n"
"    }\n"
"    for (k = 0u; k < 4u; k++) {\n"
"        w[k] = 4u * k < n ? hd_brd(vb + 12u, at2 + 4u * k, min(n - 4u * k, 4u)) : 0u;\n"
"    }\n"
"    c[0] = 0u; c[1] = 0u; c[2] = 0u; c[3] = 0u;\n"
"    if (n > 8u) {\n"
"        for (k = 0u; k < 4u && hd_fbits(fmt, k) != 0u; k++) {\n"
"            c[k] = w[k];\n"
"        }\n"
"    } else {\n"
"        uint sh = 0u;\n"
"\n"
"        for (k = 0u; k < 4u && hd_fbits(fmt, k) != 0u; k++) {\n"
"            uint bb = hd_fbits(fmt, k);\n"
"\n"
"            c[k] = hd_ext(w[0], w[1], sh, bb);\n"
"            sh += bb;\n"
"        }\n"
"    }\n"
"    for (k = 0u; k < 4u && hd_fbits(fmt, k) != 0u; k++) {\n"
"        o[k] = hd_comp(c[k], hd_fbits(fmt, k), hd_fmt_float(fmt), sgn != 0u, num);\n"
"    }\n"
"}\n"
"/* the vertex index of slot s, as hd5770_gfx_draw(); false: a restart */\n"
"bool hd_vsindex(uint s, out uint x)\n"
"{\n"
"    uint raw;\n"
"\n"
"    x = 0u;\n"
"    if (VW(9u) == 0u) {\n"
"        x = s + VW(13u);\n"
"        return true;\n"
"    }\n"
"    if (VW(9u) == 2u) {\n"
"        raw = words[VW(18u) + s];\n"
"    } else if (s >= VW(12u)) {\n"
"        raw = 0u;\n"
"    } else if (VW(10u) != 0u) {\n"
"        raw = hd_brd(vso + 20u, s * 4u, 4u);\n"
"        if (VW(11u) == 1u) {\n"
"            raw = ((raw & 0x00ff00ffu) << 8u) | ((raw >> 8u) & 0x00ff00ffu);\n"
"        } else if (VW(11u) == 2u) {\n"
"            raw = (raw << 24u) | ((raw << 8u) & 0xff0000u) | ((raw >> 8u) & 0xff00u) | (raw >> 24u);\n"
"        } else if (VW(11u) == 3u) {\n"
"            raw = (raw << 16u) | (raw >> 16u);\n"
"        }\n"
"    } else {\n"
"        raw = hd_brd(vso + 20u, s * 2u, 2u);\n"
"        if (VW(11u) == 1u) {\n"
"            raw = ((raw & 0xffu) << 8u) | (raw >> 8u);\n"
"        }\n"
"    }\n"
"    if (VW(16u) != 0u && raw == VW(17u)) {\n"
"        return false;\n"
"    }\n"
"    x = min(max(raw + VW(13u), VW(14u)), VW(15u));\n"
"    return true;\n"
"}\n";

/* the operations of a vertex shader: records to recs[rec ..] */
static void vs_body(GString *o, const HD5770IR *ir)
{
    unsigned i, j, k;

    for (i = 0; i < ir->nops; i++) {
        const HD5770IROp *op = &ir->ops[i];

        switch (op->kind) {
        case HD5770_IRO_ALU:
            for (j = 0; j < op->n; j++) {
                emit_group(o, ir, &ir->groups[op->first + j]);
            }
            break;
        case HD5770_IRO_FETCH:
            for (j = 0; j < op->n; j++) {
                const HD5770IRFetch *f = &ir->fetches[op->first + j];

                g_string_append_printf(o,
                    "    {\n        uint o[4];\n"
                    "        hd_vsfetch(0x%08xu, 0x%08xu, 0x%08xu, r[%u][%u], %uu, o);\n",
                    f->w[0], f->w[1], f->w[2], f->src, (f->w[0] >> 24) & 3,
                    f->fs ? 160 : 0);
                for (k = 0; k < 4; k++) {
                    unsigned sel = (f->w[1] >> (9 + 3 * k)) & 7;

                    if (sel < 4) {
                        g_string_append_printf(o, "        r[%u][%u] = o[%u];\n",
                                               f->dst, k, sel);
                    } else if (sel == 4 || sel == 5) {
                        g_string_append_printf(o, "        r[%u][%u] = U(%s);\n",
                                               f->dst, k, sel == 5 ? "1.0" : "0.0");
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
                        g_string_append_printf(o, "    recs[rec + %uu] = r[%u][%u];\n",
                                               base + k, e->gpr, sel);
                    } else if (sel == 4 || sel == 5) {
                        g_string_append_printf(o, "    recs[rec + %uu] = U(%s);\n",
                                               base + k, sel == 5 ? "1.0" : "0.0");
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
}

/*
 * The rasterisation pipeline (ati_hd5770_vk.c): a program as the stages
 * of vk_raster.h. Push constant 0 (uo) is the word offset of the draw's
 * block, HD5770_NU_* below; push constant 1 (vso) that of its vertex
 * stage, laid out as glsl_vs_lib reads it, the set-up from word
 * HD5770_NV_X on.
 */
/*
 * exact: division, square root, exp2 and log2 rounded as the CPU path's,
 * through doubles; a vertex stage's positions must be the CPU path's
 */
static void nat_head(GString *o, bool formats, bool exact)
{
    g_string_append_printf(o,
        "#version 450\n#define HD_FUSED %d\n#define HD_EXACT %d\n"
        "layout(std430, set = 0, binding = 0) readonly buffer Words { uint words[]; };\n"
        "layout(std430, set = 0, binding = 1) readonly buffer Tables {"
        " uint hd_u8[256]; uint hd_fmt_bits[]; };\n"
        "layout(push_constant) uniform PC { uint uo; uint vso; };\n"
        "#define vram words\n"
        "#define F(x) uintBitsToFloat(x)\n"
        "#define U(x) floatBitsToUint(x)\n"
        "#define I(x) int(x)\n"
        "#define FLT_MAX uintBitsToFloat(0x7f7fffffu)\n", HD5770_GLSL_FUSED,
        exact);
    g_string_append(o, glsl_arith);
    if (formats) {
        g_string_append(o, glsl_formats);
    }
    g_string_append(o, glsl_alu);
}

/* hd_eval() for opcode op alone, as hd_e<op>(): its case of the library's */
static void nat_eval_fn(GString *w, unsigned op)
{
    const char *ev = strstr(glsl_alu, "uint hd_eval(");
    const char *sw3 = strstr(ev, "switch (op & 0x1fu) {");
    const char *sw2 = strstr(sw3, "    switch (op) {");
    const char *end = strstr(sw2, "\nbool hd_isfloat");
    const char *lo = op & 0x800 ? sw3 : sw2, *hi = op & 0x800 ? sw2 : end;
    const char *p, *q;
    char label[24];

    snprintf(label, sizeof(label), "case 0x%02xu:",
             op & 0x800 ? op & 0x1f : op);
    g_string_append_printf(w,
        "uint hd_e%x(uint mods, uint s0, uint s1, uint s2, inout bool kill)\n"
        "{\n    float a, b, c;\n", op);
    if (op & 0x800) {
        g_string_append(w,
            "    a = F(s0);\n    b = F(s1);\n    c = F(s2);\n"
            "    if ((mods & 1u) != 0u) {\n        a = hd_neg(a);\n    }\n"
            "    if ((mods & 2u) != 0u) {\n        b = hd_neg(b);\n    }\n"
            "    if ((mods & 4u) != 0u) {\n        c = hd_neg(c);\n    }\n");
    } else {
        g_string_append(w,
            "    a = hd_fmod(mods, 0u, s0);\n    b = hd_fmod(mods, 1u, s1);\n"
            "    c = 0.0;\n");
    }
    g_string_append_printf(w, "    switch (0x%xu) {\n",
                           op & 0x800 ? op & 0x1f : op);
    p = g_strstr_len(lo, hi - lo, label);
    while (p && p < hi) {
        /* the lines of the case, to the next label */
        while (p > lo && p[-1] != '\n') {
            p--;
        }
        q = strchr(p, '\n');
        g_string_append_len(w, p, q + 1 - p);
        p = q + 1;
        while (*p == ' ') {
            p++;
        }
        if (g_str_has_prefix(p, "case ") || g_str_has_prefix(p, "default:")) {
            break;
        }
    }
    g_string_append(w, "    default: return 0u;\n    }\n}\n");
}

/*
 * hd_eval() with a constant opcode becomes a function of that opcode
 * alone: the host's compiler is not given the whole opcode switch for
 * every instruction. from: where the program's own text begins.
 */
static void nat_eval(GString *o, size_t from)
{
    static const char call[] = "hd_eval(0x";
    GString *w = g_string_new(NULL);
    uint64_t seen[64] = { 0 };
    char *p;

    while ((p = strstr(o->str + from, call))) {
        char *e;
        unsigned op = strtoul(p + sizeof(call) - 1, &e, 16);
        size_t at = p - o->str;
        char name[24];

        if (e[0] != 'u' || e[1] != ',' || e[2] != ' ' || op >= 4096) {
            break;
        }
        if (!((seen[op >> 6] >> (op & 63)) & 1)) {
            seen[op >> 6] |= 1ull << (op & 63);
            nat_eval_fn(w, op);
        }
        snprintf(name, sizeof(name), "hd_e%x(", op);
        g_string_erase(o, at, e + 3 - p);
        g_string_insert(o, at, name);
    }
    g_string_insert(o, from, w->str);
    g_string_free(w, true);
}

uint32_t hd5770_glsl_native_flat(const HD5770IR *ir, const HD5770NatKey *k)
{
    uint32_t iflat = 0;
    unsigned i;

    for (i = 0; i < ir->ninsts; i++) {
        const HD5770IRInst *in = &ir->insts[i];

        if (in->interp && in->needed && in->prm < k->ninputs && in->flat) {
            iflat |= 1u << in->prm;
        }
    }
    return k->flat | iflat;
}

static void nat_io(GString *o, unsigned ninputs, uint32_t flat, bool out)
{
    unsigned i;

    for (i = 0; i < ninputs; i++) {
        bool fl = (flat >> i) & 1;

        g_string_append_printf(o, "layout(location = %u) %s%s %s p%u;\n", i,
                               fl ? "flat " : "", out ? "out" : "in",
                               fl ? "uvec4" : "vec4", i);
    }
}

char *hd5770_glsl_native_pvs(unsigned ninputs, uint32_t flat)
{
    GString *o = g_string_new(NULL);
    unsigned i;

    nat_head(o, false, true);
    nat_io(o, ninputs, flat, true);
    g_string_append_printf(o,
        "void main()\n{\n"
        "    uint v = words[uo + %uu] + uint(gl_VertexIndex) * %uu;\n"
        "    float pc = F(words[uo + %uu]);\n"
        "    float cw = words[uo + %uu] != 0u ? 1.0 / F(words[v + 3u]) : 1.0;\n"
        "    gl_Position = vec4(((F(words[v]) - pc + 0.5) * F(words[uo + %uu]) - 1.0) * cw,\n"
        "        ((F(words[v + 1u]) - pc + 0.5) * F(words[uo + %uu]) - 1.0) * cw,\n"
        "        F(words[v + 2u]) * cw, cw);\n",
        HD5770_NU_VTX, 4 + 4 * ninputs, HD5770_NU_PC, HD5770_NU_PERSP,
        HD5770_NU_VPX, HD5770_NU_VPY);
    for (i = 0; i < ninputs; i++) {
        bool fl = (flat >> i) & 1;

        g_string_append_printf(o,
            "    p%u = %s(words[v + %uu], words[v + %uu], words[v + %uu],"
            " words[v + %uu])%s;\n", i, fl ? "uvec4" : "uintBitsToFloat(uvec4",
            4 + 4 * i, 5 + 4 * i, 6 + 4 * i, 7 + 4 * i, fl ? "" : ")");
    }
    g_string_append(o, "}\n");
    return g_string_free(o, false);
}

char *hd5770_glsl_native_vs(const HD5770IR *ir, unsigned ninputs,
                            uint32_t flat, const int32_t *src)
{
    unsigned i, k, ng = MAX(ir->ngpr, 1u), np = hd5770_vs_np(ir);
    unsigned rw = HD5770_VS_REC(np);
    size_t from;
    GString *o;

    if (!ir->vs || ninputs > 32) {
        return NULL;
    }
    for (i = 0; i < ir->nfetches; i++) {
        if (ir->fetches[i].kind != HD5770_IRF_VTX) {
            return NULL;
        }
    }
    o = g_string_new(NULL);
    nat_head(o, true, true);
    g_string_append(o, glsl_vs_lib);
    nat_io(o, ninputs, flat, true);
    from = o->len;
    g_string_append_printf(o,
        "#define VX(i) words[vso + %uu + (i)]\n"
        "void main()\n{\n"
        "    uint kvo = VW(4u), inst = uint(gl_InstanceIndex);\n"
        "    uint index = uint(gl_VertexIndex), pm = 0u, fl = 0u, rec = 0u;\n"
        "    uint recs[%u];\n"
        "    for (uint i = 0u; i < %uu; i++) {\n        recs[i] = 0u;\n    }\n"
        "    bool killed = false;\n"
        "    uvec4 r[%u];\n"
        "    for (uint i = 0u; i < %uu; i++) {\n        r[i] = uvec4(0u);\n    }\n"
        "    uint pv[5] = uint[5](0u, 0u, 0u, 0u, 0u);\n"
        "    r[%u][0] = index;\n    r[%u][3] = inst;\n",
        HD5770_NV_X, rw, rw, ng, ng, ir->map[0], ir->map[0]);
    vs_body(o, ir);
    /* clip_dist(), vtx_window(), then the host's clip space */
    g_string_append(o,
        "    float x = F(recs[0]), y = F(recs[1]), z = F(recs[2]), w = F(recs[3]);\n"
        "    uint clip = VX(3u), vte = VX(0u);\n"
        "    float dn = (clip & 2u) != 0u ? z : z + w, df = w - z;\n"
        "    if ((clip & 5u) != 5u) {\n        dn = 0.0;\n    }\n"
        "    if ((clip & 9u) != 9u) {\n        df = 0.0;\n    }\n"
        "    x /= w;\n    y /= w;\n    z /= w;\n"
        "    if ((vte & 1u) != 0u) {\n        x *= F(VX(4u));\n    }\n"
        "    if ((vte & 2u) != 0u) {\n        x += F(VX(5u));\n    }\n"
        "    if ((vte & 4u) != 0u) {\n        y *= F(VX(6u));\n    }\n"
        "    if ((vte & 8u) != 0u) {\n        y += F(VX(7u));\n    }\n"
        "    if ((vte & 16u) != 0u) {\n        z *= F(VX(8u));\n    }\n"
        "    if ((vte & 32u) != 0u) {\n        z += F(VX(9u));\n    }\n"
        "    z += F(VX(13u));\n"
        "    if ((VX(1u) & (1u << 16)) != 0u) {\n"
        "        x += float(int(VX(2u) << 16) >> 16);\n"
        "        y += float(int(VX(2u)) >> 16);\n"
        "    }\n"
        "    float pc = F(VX(12u));\n"
        "    gl_Position = vec4(((x - pc + 0.5) * F(VX(10u)) - 1.0) * w,\n"
        "                       ((y - pc + 0.5) * F(VX(11u)) - 1.0) * w, z * w, w);\n"
        "    gl_ClipDistance[0] = dn;\n    gl_ClipDistance[1] = df;\n");
    for (i = 0; i < ninputs; i++) {
        bool fl = (flat >> i) & 1;

        g_string_append_printf(o, "    p%u = %s(", i, fl ? "uvec4" : "vec4");
        for (k = 0; k < 4; k++) {
            if (src[i] < 0) {
                g_string_append_printf(o, "%s%swords[vso + %uu]%s",
                                       k ? ", " : "", fl ? "" : "F(",
                                       HD5770_NV_DEF + 4 * i + k,
                                       fl ? "" : ")");
            } else if (src[i] >= (int)np) {
                g_string_append_printf(o, "%s%s", k ? ", " : "",
                                       fl ? "0u" : "0.0");
            } else {
                g_string_append_printf(o, "%s%srecs[%u]%s", k ? ", " : "",
                                       fl ? "" : "F(", 4 + 4 * src[i] + k,
                                       fl ? "" : ")");
            }
        }
        g_string_append(o, ");\n");
    }
    g_string_append(o, "}\n");
    nat_eval(o, from);
    return g_string_free(o, false);
}

/* the TEX operations taken: LD, RESINFO, SAMPLE*, SAMPLE_C* */
static bool nat_tex_op(unsigned op)
{
    return op == 0x03 || op == 0x04 || (op >= 0x10 && op <= 0x13) ||
           (op >= 0x18 && op <= 0x1b);
}

/*
 * hd_nt<unit>(): a fetch of the unit by its words, as tfetch_stage() in
 * ati_hd5770_draw.c reads them, through the host's sampler; dv: the
 * coordinates' change to the pixel on the right and below
 */
static void nat_unit(GString *o, unsigned unit, bool padded, unsigned dim,
                     unsigned nsl)
{
    unsigned tu = HD5770_NU_TEX + HD5770_NU_TEX_WORDS * unit;
    bool vol = dim == 2, arr = nsl > 1 || vol;
    const char *tcs = arr ? "vec3(tc, sl)" : "tc";

    g_string_append_printf(o,
        "void hd_nt%u(uint w0, uint w1, uint w2, uint c[4], vec4 dv, out uint o[4])\n"
        "{\n"
        "    uint op = w0 & 0x1fu;\n"
        "    int ox = int(w2 << 27u) >> 27, oy = int(w2 << 22u) >> 27;\n"
        "    vec2 ts = vec2(F(words[uo + %uu]), F(words[uo + %uu]));\n"
        "    vec4 v = vec4(0.0);\n"
        "    bool cmp = false;\n"
        "    float ref = 0.0;\n"
        "    o[0] = 0u; o[1] = 0u; o[2] = 0u; o[3] = 0u;\n"
        "    if (op >= 0x18u && op <= 0x1bu) {\n"
        "        cmp = true;\n"
        "        ref = op == 0x19u || op == 0x1au ? F(c[2]) : F(c[3]);\n"
        "        op -= 8u;\n"
        "    }\n"
        "    if (op == 4u) {\n"
        "        uint lv = min(c[0] + words[uo + %uu], 31u);\n"
        "        o[0] = max(uint(ts.x) >> lv, 1u);\n"
        "        o[1] = max(uint(ts.y) >> lv, 1u);\n"
        "        o[2] = words[uo + %uu];\n"
        "        o[3] = words[uo + %uu];\n"
        "        return;\n"
        "    }\n"
        "    float sl = 0.0;\n", unit, tu, tu + 1, tu + 6, tu + 8, tu + 7);
    /* tex_slice() */
    if (vol) {
        /* the slice as a coordinate, the host wraps and filters it */
        g_string_append_printf(o,
            "    sl = op == 3u ? float(I(c[2])) :\n"
            "         ((w1 >> 30u) & 1u) != 0u ? F(c[2]) : F(c[2]) / %u.0;\n",
            MAX(nsl, 1u));
    } else if (arr && dim == 3) {
        g_string_append(o,
            "    sl = F(c[2]) > 0.0 ? min(floor(F(c[2])), 5.0) : 0.0;\n"
            "    c[0] = U(F(c[0]) - 1.0);\n    c[1] = U(F(c[1]) - 1.0);\n");
    } else if (arr) {
        g_string_append_printf(o,
            "    sl = float(clamp(op == 3u ? I(c[2]) : int(roundEven(F(c[2]))), 0, %d));\n",
            (int)nsl - 1);
    }
    g_string_append_printf(o,
        "    if (op == 3u) {\n"
        "        ivec2 xy = ivec2(I(c[0]) + ox / 2, I(c[1]) + oy / 2);\n"
        "        if (xy.x >= 0 && xy.y >= 0 && xy.x < int(ts.x) && xy.y < int(ts.y)) {\n"
        "            v = texelFetch(t%u, %s, 0);\n"
        "        }\n"
        "    } else {\n"
        "        vec2 tc = vec2(F(c[0]), F(c[1]));\n"
        "        if (((w1 >> 28u) & 1u) == 0u) {\n"
        "            tc.x /= ts.x;\n            dv.xz /= ts.x;\n        }\n"
        "        if (((w1 >> 29u) & 1u) == 0u) {\n"
        "            tc.y /= ts.y;\n            dv.yw /= ts.y;\n        }\n"
        "        tc += vec2(float(ox), float(oy)) * 0.5 / ts;\n",
        unit, arr ? "ivec3(xy, int(sl))" : "xy");
    if (padded) {
        g_string_append_printf(o,
            "        vec2 sc = vec2(F(words[uo + %uu]), F(words[uo + %uu]));\n"
            "        tc = vec2(hd_nwrap(tc.x, ts.x, words[uo + %uu]),\n"
            "                  hd_nwrap(tc.y, ts.y, words[uo + %uu])) * sc;\n"
            "        dv *= sc.xyxy;\n", tu + 11, tu + 12, tu + 13, tu + 14);
    }
    g_string_append_printf(o,
        "        if (op == 0x11u) {\n"
        "            v = textureLod(t%u, %s, F(c[3]));\n"
        "        } else if (op == 0x13u) {\n"
        "            v = textureLod(t%u, %s, 0.0);\n"
        "        } else {\n"
        "            float bias = float(int(w1 << 4u) >> 25) / 16.0;\n"
        "            if (op == 0x12u) {\n                bias += F(c[3]);\n            }\n"
        "            dv *= exp2(bias);\n"
        "            v = textureGrad(t%u, %s, %s);\n"
        "        }\n"
        "    }\n"
        "    if (cmp) {\n"
        "        v = vec4(hd_ncmp(words[uo + %uu], words[uo + %uu], ref, v.x));\n"
        "    }\n"
        "    v = hd_nsel(v, uo + %uu);\n"
        "    o[0] = U(v.x); o[1] = U(v.y); o[2] = U(v.z); o[3] = U(v.w);\n"
        "}\n", unit, tcs, unit, tcs, unit, tcs,
        vol ? "vec3(dv.xy, 0.0), vec3(dv.zw, 0.0)" : "dv.xy, dv.zw", tu + 9,
        tu + 10, tu + 2);
}

/*
 * a TEX fetch through its unit's function; a VTX fetch as vfetch_stage()
 * in ati_hd5770_draw.c reads an 8 or 32 bit element
 */
static bool nat_fetch(GString *o, const HD5770IR *ir, const HD5770IRFetch *f,
                      const HD5770NatKey *key)
{
    unsigned k, unit, op = f->w[0] & 0x1f;

    g_string_append(o, "    {\n        uint o[4] = uint[4](0u, 0u, 0u, 0u);\n");
    if (f->kind == HD5770_IRF_GRAD_H || f->kind == HD5770_IRF_GRAD_V) {
        for (k = 0; k < 4; k++) {
            unsigned sel = (f->w[2] >> (20 + 3 * k)) & 7;

            if (sel < 4) {
                g_string_append_printf(o, "        o[%u] = U(%s(F(r[%u][%u])));\n",
                                       k, f->kind == HD5770_IRF_GRAD_H ?
                                       "dFdx" : "dFdy", f->src, sel);
            } else {
                g_string_append_printf(o, "        o[%u] = U(0.0);\n", k);
            }
        }
    } else if (f->kind == HD5770_IRF_VTX) {
        unsigned bid = (f->w[0] >> 8) & 0xff, fmt, num, sgn, n, vb;

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
        vb = HD5770_NU_VB + 3 * k;
        g_string_append_printf(o,
            "        uint ix = r[%u][%u], st = words[uo + %uu], ln = words[uo + %uu];\n"
            "        o[3] = U(1.0);\n"
            "        if ((st == 0u || ix <= ln / st) && ix * st + %uu <= ln) {\n"
            "            uint at = words[uo + %uu] + ix * st + %uu, sh = (at & 3u) * 8u;\n"
            "            uint c = words[at >> 2] >> sh;\n",
            f->src, (f->w[0] >> 24) & 3, vb, vb + 1, (f->w[2] & 0xffff) + n,
            vb + 2, f->w[2] & 0xffff);
        if (n == 1) {
            g_string_append(o, sgn ? "            c = uint(int(c << 24) >> 24);\n"
                                   : "            c &= 0xffu;\n");
        } else {
            g_string_append(o,
                "            if (sh != 0u) {\n"
                "                c |= words[(at >> 2) + 1u] << (32u - sh);\n"
                "            }\n");
        }
        if (num == 1) {
            g_string_append(o, "            o[0] = c;\n");
        } else if (num != 0) {
            g_string_append_printf(o, "            o[0] = U(float(%sc%s));\n",
                                   sgn ? "int(" : "", sgn ? ")" : "");
        } else if (n == 4) {
            g_string_append_printf(o, "            o[0] = U(float(%sc%s) / %s);\n",
                                   sgn ? "int(" : "", sgn ? ")" : "",
                                   sgn ? "2147483647.0" : "4294967295.0");
        } else if (sgn) {
            g_string_append(o, "            o[0] = U(max(float(int(c)) / 127.0, -1.0));\n");
        } else {
            g_string_append(o, "            o[0] = U(float(c) / 255.0);\n");
        }
        g_string_append(o, "        }\n");
    } else if (f->kind != HD5770_IRF_TEX || !nat_tex_op(op)) {
        return false;
    } else {
        for (unit = 0; unit < ir->nres && ir->res[unit] != f->res; unit++) {
            /* search */
        }
        if (unit == ir->nres) {
            return false;
        }
        g_string_append(o, "        uint c[4] = uint[4](");
        for (k = 0; k < 4; k++) {
            unsigned sel = (f->w[2] >> (20 + 3 * k)) & 7;

            if (sel < 4) {
                g_string_append_printf(o, "r[%u][%u]", f->src, sel);
            } else {
                g_string_append(o, sel == 5 ? "U(1.0)" : "U(0.0)");
            }
            g_string_append(o, k < 3 ? ", " : ");\n");
        }
        g_string_append_printf(o,
            "        vec2 cn = vec2(F(c[0]), F(c[1]));\n"
            "        hd_nt%u(0x%08xu, 0x%08xu, 0x%08xu, c, vec4(dFdx(cn), dFdy(cn)), o);\n",
            unit, f->w[0], f->w[1], f->w[2]);
    }
    for (k = 0; k < 4; k++) {
        unsigned sel = (f->w[1] >> (9 + 3 * k)) & 7;

        if (sel < 4) {
            g_string_append_printf(o, "        r[%u][%u] = o[%u];\n", f->dst, k, sel);
        } else if (sel == 4 || sel == 5) {
            g_string_append_printf(o, "        r[%u][%u] = U(%s);\n", f->dst, k,
                                   sel == 5 ? "1.0" : "0.0");
        }
    }
    g_string_append(o, "    }\n");
    return true;
}

char *hd5770_glsl_native_fs(const HD5770IR *ir, const HD5770NatKey *k,
                            uint32_t padded, bool bary)
{
    GString *o;
    unsigned i, j, ng = MAX(ir->ngpr, 1u);
    uint32_t iflat = 0, ivar = 0, flat;
    /* the compact register of the position */
    int pos = k->pos_gpr >= 0 && ir->map[k->pos_gpr] != 0xff ?
              ir->map[k->pos_gpr] : -1;
    bool kills = false;
    size_t from;

    if (ir->vs || k->ninputs > 32 || ir->nres > HD5770_GPU_TEX ||
        k->nvb > HD5770_GPU_VB) {
        return NULL;
    }
    for (i = 0; i < ir->nfetches; i++) {
        const HD5770IRFetch *f = &ir->fetches[i];

        /* the CF form: texture fetches only, no gradients */
        if (ir->cf && (f->kind != HD5770_IRF_TEX ||
                       !nat_tex_op(f->w[0] & 0x1f))) {
            return NULL;
        }
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
    flat = k->flat | iflat;
    o = g_string_new(NULL);
    nat_head(o, false, false);
    if (bary) {
        g_string_insert(o, strchr(o->str, '\n') + 1 - o->str,
            "#extension GL_EXT_fragment_shader_barycentric : require\n");
    }
    nat_io(o, k->ninputs, flat, false);
    for (i = 0; i < ir->nres; i++) {
        g_string_append_printf(o,
            "layout(set = 0, binding = %u) uniform sampler%s t%u;\n", 2 + i,
            k->dim[i] == 2 ? "3D" : k->nsl[i] > 1 ? "2DArray" : "2D", i);
    }
    for (i = 0; i < MAX(k->ncol, 1u); i++) {
        g_string_append_printf(o, "layout(location = %u) out vec4 o%u;\n", i, i);
    }
    g_string_append(o, "float hd_ni(uint p, uint ch)\n{\n    switch (p) {\n");
    for (i = 0; i < k->ninputs; i++) {
        g_string_append_printf(o, "    case %uu: return %s(p%u[ch]);\n", i,
                               (flat >> i) & 1 ? "F" : "", i);
    }
    g_string_append_printf(o,
        "    default: return 0.0;\n    }\n}\n"
        "#define hd_interp(di, ti, pr, p, ch, fl, b0, b1, b2, pw) hd_ni(p, ch)\n"
        "vec4 hd_nsel(vec4 v, uint at)\n"
        "{\n"
        "    vec4 q;\n"
        "    for (uint k = 0u; k < 4u; k++) {\n"
        "        uint s = words[at + k];\n"
        "        q[k] = s < 4u ? v[s] : s == 5u ? 1.0 : 0.0;\n"
        "    }\n"
        "    return q;\n"
        "}\n"
        "bool hd_nalpha(uint fn, float r, float a)\n"
        "{\n"
        "    switch (fn) {\n"
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
        "/* a normalised coordinate of a texture n texels wide, wrapped */\n"
        "float hd_nwrap(float t, float n, uint m)\n"
        "{\n"
        "    switch (m) {\n"
        "    case 0u: t = fract(t); break;\n"
        "    case 1u: {\n"
        "        float p = t - floor(t * 0.5) * 2.0;\n"
        "        t = p < 1.0 ? p : 2.0 - p;\n"
        "        break;\n"
        "    }\n"
        "    case 3u: case 5u: case 7u: t = abs(t); break;\n"
        "    default: break;\n"
        "    }\n"
        "    return clamp(t, 0.5 / n, 1.0 - 0.5 / n);\n"
        "}\n"
        "float hd_ncmp(uint fn, uint mx, float ref, float f)\n"
        "{\n"
        "    if (mx != 0u) {\n"
        "        ref = floor(clamp(ref, 0.0, 1.0) * float(mx) + 0.5) / float(mx);\n"
        "    }\n"
        "    return hd_nalpha(fn, f, ref) ? 1.0 : 0.0;\n"
        "}\n");
    for (i = 0; i < ir->nres; i++) {
        nat_unit(o, i, (padded >> i) & 1, k->dim[i], k->nsl[i]);
    }
    from = o->len;
    if (ir->cf) {
        /* the program's functions (ati_hd5770_emit.c) */
        g_string_append(o,
            "void hd_ntex(uint fu, uint w0, uint w1, uint w2, uint c[4], vec4 dv,\n"
            "             out uint o[4])\n"
            "{\n"
            "    switch (fu) {\n");
        for (i = 0; i < ir->nres; i++) {
            g_string_append_printf(o,
                "    case %uu: hd_nt%u(w0, w1, w2, c, dv, o); break;\n", i, i);
        }
        g_string_append_printf(o,
            "    default: o[0] = 0u; o[1] = 0u; o[2] = 0u; o[3] = 0u; break;\n"
            "    }\n"
            "}\n"
            "#define HD_CTX_PARAMS uint ctx_\n"
            "#define HD_CTX_ARGS 0u\n"
            "#define HD_K(k) words[words[uo + %uu] + (k)]\n"
            "#define HD_BCONST words[uo + %uu]\n"
            "#define HD_LCONST(i) words[uo + %uu + (i)]\n"
            "#define HD_POSGPR %d\n"
            "#define HD_TEX(fu, w0, w1, w2, c, dv, o) { \\\n"
            "    vec2 cn_ = vec2(F(c[0]), F(c[1])); \\\n"
            "    hd_ntex(fu, w0, w1, w2, c, vec4(dFdx(cn_), dFdy(cn_)), o); }\n"
            "#define HD_ZERO int(words[uo + %uu])\n"
            "#define HD_VTX(w0, w1, w2, i, o)\n"
            "#define HD_INTERP(p, ch, fl, b0, b1, b2, pw) hd_ni(p, ch)\n"
            "#define HD_OUTA(n, a, b) out uint n[a][b]\n"
            "#define HD_OUTBA(n, a) out bool n[a]\n"
            "#define HD_OUTB(n) out bool n\n"
            "#define HD_OUTU(n) out uint n\n"
            "#define HD_OUTC(n) out uint n[8][4]\n",
            HD5770_NU_KV, HD5770_NU_BCONST, HD5770_NU_LCONST,
            pos, HD5770_NU_ZERO);
        hd5770_emit_cf(o, ir, 0);
    }
    g_string_append_printf(o,
        "void main()\n{\n"
        "    uint C[8][4];\n"
        "    for (uint i = 0u; i < 8u; i++) {\n"
        "        C[i][0] = 0u; C[i][1] = 0u; C[i][2] = 0u; C[i][3] = 0u;\n"
        "    }\n"
        "    uint cm = 0u, zd = 0u, sd = 0u, kvo = words[uo + %uu];\n"
        "    bool killed = false, zw = false;\n"
        "    float pc = F(words[uo + %uu]);\n"
        "    uvec4 r[%u];\n"
        "    for (uint i = 0u; i < %uu; i++) {\n        r[i] = uvec4(0u);\n    }\n"
        "    uint pv[5] = uint[5](0u, 0u, 0u, 0u, 0u);\n",
        HD5770_NU_KV, HD5770_NU_PC, ng, ng);
    if (ir->cf) {
        g_string_append_printf(o,
            "    hd_run(HD_CTX_ARGS, %s, 0.0,\n"
            "           gl_FragCoord.x - 0.5 + pc, gl_FragCoord.y - 0.5 + pc,\n"
            "           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,\n"
            "           killed, C, cm, zd, sd, zw);\n",
            bary ? "gl_BaryCoordNoPerspEXT.x, gl_BaryCoordNoPerspEXT.y,"
                   " gl_BaryCoordNoPerspEXT.z" : "0.0, 0.0, 0.0");
    } else if (pos >= 0) {
        g_string_append_printf(o,
            "    r[%d] = uvec4(U(gl_FragCoord.x - 0.5 + pc), U(gl_FragCoord.y - 0.5 + pc),"
            " U(0.0), U(1.0));\n", pos);
    }
    if (bary && !ir->cf) {
        g_string_append_printf(o,
            "    r[%u][0] = U(gl_BaryCoordNoPerspEXT.y);\n"
            "    r[%u][1] = U(gl_BaryCoordNoPerspEXT.z);\n", ir->map[0],
            ir->map[0]);
    }
    for (i = 0; i < ir->nops && !ir->cf; i++) {
        const HD5770IROp *op = &ir->ops[i];

        switch (op->kind) {
        case HD5770_IRO_ALU:
            for (j = 0; j < op->n; j++) {
                emit_group(o, ir, &ir->groups[op->first + j]);
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
        g_string_append_printf(o,
            "    {\n"
            "        ivec2 p = ivec2(gl_FragCoord.xy);\n"
            "        uint idx = 0u;\n"
            "        for (uint i = 0u; i < 4u; i++) {\n"
            "            uint a = uo + %uu + i * 4u;\n"
            "            if (p.x >= int(words[a]) && p.x < int(words[a + 2u]) &&\n"
            "                p.y >= int(words[a + 1u]) && p.y < int(words[a + 3u])) {\n"
            "                idx |= 1u << i;\n"
            "            }\n"
            "        }\n"
            "        killed = killed || ((words[uo + %uu] >> idx) & 1u) == 0u;\n"
            "    }\n", HD5770_NU_CR, HD5770_NU_CLIPRULE);
    }
    /*
     * shade(): the alpha test reads export 0. A function that cannot
     * discard leaves the host its early depth test.
     */
    for (i = 0; i < ir->ninsts; i++) {
        kills |= ir->insts[i].kill;
    }
    if (k->atest) {
        g_string_append_printf(o,
            "    if (killed || %s ||\n"
            "        ((cm & 1u) != 0u && !hd_nalpha(words[uo + %uu], F(words[uo + %uu]),"
            " F(C[0][3])))) {\n"
            "        discard;\n"
            "    }\n", k->ncol ? "cm == 0u" : "(cm & 1u) == 0u",
            HD5770_NU_AFUNC, HD5770_NU_AREF);
    } else if (kills || ir->cf || k->clip ||
               !(k->ncol ? ir->cmask : ir->cmask & 1)) {
        g_string_append_printf(o,
            "    if (killed || %s) {\n"
            "        discard;\n"
            "    }\n", k->ncol ? "cm == 0u" : "(cm & 1u) == 0u");
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
                "    o%u = vec4(float(v%u & 0xffu), float((v%u >> 8) & 0xffu),\n"
                "              float((v%u >> 16) & 0xffu), float(v%u >> 24)) / 255.0;\n",
                i, cb, src[0] & 3, i, i, i, i, i);
        } else if (comps == 1) {
            g_string_append_printf(o,
                "    o%u = vec4(F(C[%u][%u]), 0.0, 0.0, 0.0);\n", i, cb,
                src[0] & 3);
        } else {
            /* channel i of the target is component src[i] */
            g_string_append_printf(o,
                "    o%u = vec4(F(C[%u][%u]), F(C[%u][%u]), F(C[%u][%u]), F(C[%u][%u]));\n",
                i, cb, src[0] & 3, cb, src[1] & 3, cb, src[2] & 3, cb,
                src[3] & 3);
        }
        if (out != 2) {
            g_string_append_printf(o,
                "    o%u = mix(o%u, vec4(0.0), isnan(o%u));\n", i, i, i);
        }
    }
    if (k->zout) {
        g_string_append_printf(o,
            "    gl_FragDepth = min(max(%s, F(words[uo + %uu])), F(words[uo + %uu]));\n",
            k->zexp ? "(zw ? F(zd) : gl_FragCoord.z)" : "gl_FragCoord.z",
            HD5770_NU_ZMIN, HD5770_NU_ZMAX);
    }
    g_string_append(o, "}\n");
    nat_eval(o, from);
    return g_string_free(o, false);
}
