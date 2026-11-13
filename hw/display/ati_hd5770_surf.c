/*
 * ATI Radeon HD 5770 (Juniper) surfaces: memory access, Evergreen tiling,
 * element formats.
 *
 * Tiling follows AMD's address library for the Evergreen-based family
 * (EgBasedLib::ComputeSurfaceAddrFromCoordMicroTiled/MacroTiled): 8x8
 * micro tiles, displayable or non-displayable pixel order inside them;
 * 2D tiling places micro tiles in macro tiles of
 * (8 * bankw * pipes * mtilea) x (8 * bankh * banks / mtilea) pixels and
 * moves pipe and bank bits into the address above the pipe interleave.
 * Pipe and bank are XOR patterns of the micro tile coordinates.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/bswap.h"
#include "ati_hd5770_gfx.h"

/* ---------------------------------------------------------------- */
/* Memory                                                           */

uint8_t *hd5770_span_ptr(HD5770Mem *m, HD5770Span *c, uint64_t mc,
                         unsigned n, bool write)
{
    uint64_t len;
    uint8_t *p;

    if (mc >= c->lo && mc + n <= c->hi && (c->write || !write)) {
        return c->p + (mc - c->lo);
    }
    p = m->map(m->opaque, mc, &len, write);
    if (!p || len < n) {
        return NULL;
    }
    c->lo = mc;
    c->hi = mc + len;
    c->p = p;
    c->write = write;
    return p;
}

void hd5770_mem_read(HD5770Mem *m, uint64_t mc, void *buf, uint64_t len)
{
    uint8_t *d = buf;

    while (len) {
        uint64_t n;
        uint8_t *p = m->map(m->opaque, mc, &n, false);

        if (!p || !n) {
            n = MIN(len, 0x1000 - (mc & 0xfff));
            memset(d, 0, n);
        } else {
            n = MIN(n, len);
            memcpy(d, p, n);
        }
        mc += n;
        d += n;
        len -= n;
    }
}

/* ---------------------------------------------------------------- */
/* Tiling                                                           */

bool hd5770_surf_mode_ok(uint32_t mode)
{
    return mode == HD5770_ARRAY_LINEAR_GENERAL ||
           mode == HD5770_ARRAY_LINEAR_ALIGNED ||
           mode == HD5770_ARRAY_1D_TILED_THIN1 ||
           mode == HD5770_ARRAY_2D_TILED_THIN1;
}

/* GB_ADDR_CONFIG: NUM_PIPES [2:0], PIPE_INTERLEAVE_SIZE [6:4] */
void hd5770_surf_set_addr_config(HD5770Surf *s, uint32_t cfg)
{
    s->pipes = 1u << (cfg & 7);
    s->pipe_interleave = 256u << ((cfg >> 4) & 7);
}

/* pixel index inside an 8x8 micro tile */
static uint32_t micro_index(uint32_t x, uint32_t y, uint32_t bpe,
                            bool nondisp)
{
    uint32_t x0 = x & 1, x1 = (x >> 1) & 1, x2 = (x >> 2) & 1;
    uint32_t y0 = y & 1, y1 = (y >> 1) & 1, y2 = (y >> 2) & 1;
    uint32_t b[6];

    if (nondisp) {
        b[0] = x0; b[1] = y0; b[2] = x1; b[3] = y1; b[4] = x2; b[5] = y2;
    } else {
        switch (bpe) {
        case 1:
            b[0] = x0; b[1] = x1; b[2] = x2; b[3] = y1; b[4] = y0; b[5] = y2;
            break;
        case 2:
            b[0] = x0; b[1] = x1; b[2] = x2; b[3] = y0; b[4] = y1; b[5] = y2;
            break;
        case 4:
            b[0] = x0; b[1] = x1; b[2] = y0; b[3] = x2; b[4] = y1; b[5] = y2;
            break;
        case 8:
            b[0] = x0; b[1] = y0; b[2] = x1; b[3] = x2; b[4] = y1; b[5] = y2;
            break;
        default:
            b[0] = y0; b[1] = x0; b[2] = x1; b[3] = x2; b[4] = y1; b[5] = y2;
            break;
        }
    }
    return b[0] | b[1] << 1 | b[2] << 2 | b[3] << 3 | b[4] << 4 | b[5] << 5;
}

static uint32_t bit(uint32_t v, unsigned n)
{
    return (v >> n) & 1;
}

/* pipe of micro tile (tx, ty) */
static uint32_t pipe_of(uint32_t tx, uint32_t ty, uint32_t pipes)
{
    switch (pipes) {
    case 2:
        return bit(tx, 0) ^ bit(ty, 0);
    case 4:
        return (bit(tx, 0) ^ bit(ty, 1)) | (bit(tx, 1) ^ bit(ty, 0)) << 1;
    case 8:
        return (bit(tx, 0) ^ bit(ty, 2)) |
               (bit(tx, 1) ^ bit(ty, 1) ^ bit(ty, 2)) << 1 |
               (bit(tx, 2) ^ bit(ty, 0)) << 2;
    default:
        return 0;
    }
}

/* bank of the bank-sized block (tx, ty) */
static uint32_t bank_of(uint32_t tx, uint32_t ty, uint32_t banks)
{
    uint32_t x3 = bit(tx, 0), x4 = bit(tx, 1), x5 = bit(tx, 2);
    uint32_t x6 = bit(tx, 3), y3 = bit(ty, 0), y4 = bit(ty, 1);
    uint32_t y5 = bit(ty, 2), y6 = bit(ty, 3);

    switch (banks) {
    case 16:
        return (x3 ^ y6) | (x4 ^ y5 ^ y6) << 1 | (x5 ^ y4) << 2 |
               (x6 ^ y3) << 3;
    case 8:
        return (x3 ^ y5) | (x4 ^ y4 ^ y5) << 1 | (x5 ^ y3) << 2;
    case 4:
        return (x3 ^ y4) | (x4 ^ y3) << 1;
    case 2:
        return x3 ^ y3;
    default:
        return 0;
    }
}

/* byte offset of element (x, y) of slice 0 from the surface base */
uint64_t hd5770_surf_offset(const HD5770Surf *s, uint32_t x, uint32_t y)
{
    uint32_t bpe = s->bpe, tile_bytes, elem, pipe, bank, split = 0;
    uint32_t mt_w, mt_h, mt_per_row, pipes, banks, pib, pbits, bbits;
    uint64_t mt_bytes, total;

    switch (s->mode) {
    case HD5770_ARRAY_LINEAR_GENERAL:
    case HD5770_ARRAY_LINEAR_ALIGNED:
        return (uint64_t)y * s->pitch * bpe + (uint64_t)x * bpe;
    case HD5770_ARRAY_1D_TILED_THIN1:
        return ((uint64_t)(y / 8) * (s->pitch / 8) + x / 8) * 64 * bpe +
               micro_index(x, y, bpe, s->nondisp) * bpe;
    default:
        break;
    }

    pipes = s->pipes;
    banks = s->banks;
    pib = s->pipe_interleave;
    tile_bytes = 64 * bpe;
    elem = micro_index(x, y, bpe, s->nondisp) * bpe;
    if (tile_bytes > s->tile_split) {
        split = elem / s->tile_split;
        elem %= s->tile_split;
        tile_bytes = s->tile_split;
    }
    mt_w = 8 * s->bankw * pipes * s->mtilea;
    mt_h = 8 * s->bankh * banks / s->mtilea;
    mt_bytes = (uint64_t)tile_bytes * (mt_w / 8) * (mt_h / 8) /
               (pipes * banks);
    mt_per_row = s->pitch / mt_w;
    total = ((uint64_t)(y / mt_h) * mt_per_row + x / mt_w) * mt_bytes +
            elem +
            (((y / 8) % s->bankh) * s->bankw +
             ((x / 8) / pipes) % s->bankw) * tile_bytes;
    if (split) {
        total += (uint64_t)split * mt_per_row *
                 (s->height / mt_h) * mt_bytes;
    }

    pipe = pipe_of(x / 8, y / 8, pipes);
    bank = bank_of(x / 8 / (s->bankw * pipes), y / 8 / s->bankh, banks);
    bank ^= ((banks / 2) + 1) * split;
    bank &= banks - 1;

    pbits = ctz32(pipes);
    bbits = ctz32(banks);
    return (total & (pib - 1)) |
           (uint64_t)pipe << ctz32(pib) |
           (uint64_t)bank << (ctz32(pib) + pbits) |
           (total >> ctz32(pib)) << (ctz32(pib) + pbits + bbits);
}

/*
 * Every quantity of the 2D layout is a power of two except the pitch, so
 * an element's offset is the offset of its micro tile plus a table entry
 * for its place inside the tile: the tile part of the address is a
 * multiple of the tile size, and an element offset below the tile size
 * only reaches the bits a tile offset does not use (no carry). Surfaces
 * whose tiles are split are left to hd5770_surf_offset().
 */
void hd5770_surf_prep(HD5770Surf *s)
{
    uint32_t mt_w, mt_h, i, pbits, bbits;

    s->fast = false;
    s->row_bytes = s->pitch * s->bpe;
    if (s->mode == HD5770_ARRAY_LINEAR_GENERAL ||
        s->mode == HD5770_ARRAY_LINEAR_ALIGNED) {
        s->fast = true;
        return;
    }
    if (!s->bpe || (s->bpe & (s->bpe - 1)) || s->bpe > 16) {
        return;
    }
    s->tile_bytes = 64 * s->bpe;
    for (i = 0; i < 64; i++) {
        s->eoff[i] = micro_index(i & 7, i >> 3, s->bpe, s->nondisp) * s->bpe;
    }
    if (s->mode == HD5770_ARRAY_1D_TILED_THIN1) {
        s->mt_per_row = s->pitch / 8;
        s->fast = true;
        return;
    }
    if (s->mode != HD5770_ARRAY_2D_TILED_THIN1 ||
        s->tile_bytes > s->tile_split || !s->pipes || !s->banks ||
        !s->bankw || !s->bankh || !s->mtilea ||
        (s->pipes & (s->pipes - 1)) || (s->banks & (s->banks - 1)) ||
        (s->bankw & (s->bankw - 1)) || (s->bankh & (s->bankh - 1)) ||
        (s->mtilea & (s->mtilea - 1)) ||
        (s->pipe_interleave & (s->pipe_interleave - 1))) {
        return;
    }
    mt_w = 8 * s->bankw * s->pipes * s->mtilea;
    mt_h = 8 * s->bankh * s->banks / s->mtilea;
    if (mt_h < 8) {
        return;
    }
    pbits = ctz32(s->pipes);
    bbits = ctz32(s->banks);
    s->mt_w_sh = ctz32(mt_w);
    s->mt_h_sh = ctz32(mt_h);
    s->pipes_sh = pbits;
    s->banks_sh = bbits;
    s->bankw_sh = ctz32(s->bankw);
    s->bankh_sh = ctz32(s->bankh);
    s->pib_sh = ctz32(s->pipe_interleave);
    s->hi_sh = s->pib_sh + pbits + bbits;
    s->mt_bytes = (uint64_t)s->tile_bytes * (mt_w / 8) * (mt_h / 8) /
                  (s->pipes * s->banks);
    s->mt_per_row = s->pitch / mt_w;
    for (i = 0; i < 64; i++) {
        uint32_t e = s->eoff[i];

        s->eoff[i] = (e & (s->pipe_interleave - 1)) |
                     (e >> s->pib_sh) << s->hi_sh;
    }
    s->fast = true;
}

uint64_t hd5770_surf_tile(const HD5770Surf *s, uint32_t tx, uint32_t ty)
{
    uint64_t total;
    uint32_t pipe, bank;

    if (s->mode == HD5770_ARRAY_1D_TILED_THIN1) {
        return ((uint64_t)ty * s->mt_per_row + tx) * s->tile_bytes;
    }
    total = ((uint64_t)(ty >> (s->mt_h_sh - 3)) * s->mt_per_row +
             (tx >> (s->mt_w_sh - 3))) * s->mt_bytes +
            (((ty & (s->bankh - 1)) << s->bankw_sh) +
             ((tx >> s->pipes_sh) & (s->bankw - 1))) * s->tile_bytes;
    pipe = pipe_of(tx, ty, s->pipes);
    bank = bank_of(tx >> (s->bankw_sh + s->pipes_sh), ty >> s->bankh_sh,
                   s->banks) & (s->banks - 1);
    return (total & (s->pipe_interleave - 1)) |
           (uint64_t)pipe << s->pib_sh |
           (uint64_t)bank << (s->pib_sh + s->pipes_sh) |
           (total >> s->pib_sh) << s->hi_sh;
}

void hd5770_surf_row(const HD5770Surf *s, uint32_t x0, uint32_t y, uint32_t n,
                     uint64_t *off)
{
    uint32_t x = x0, end = x0 + n;

    if (!s->fast) {
        for (; x < end; x++) {
            *off++ = hd5770_surf_offset(s, x, y);
        }
        return;
    }
    if (s->mode < HD5770_ARRAY_1D_TILED_THIN1) {
        uint64_t o = (uint64_t)y * s->row_bytes + (uint64_t)x * s->bpe;

        for (; x < end; x++, o += s->bpe) {
            *off++ = o;
        }
        return;
    }
    while (x < end) {
        uint64_t t = hd5770_surf_tile(s, x >> 3, y >> 3);
        const uint32_t *e = &s->eoff[(y & 7) * 8];
        uint32_t stop = MIN(end, (x | 7) + 1);

        for (; x < stop; x++) {
            *off++ = t + e[x & 7];
        }
    }
}

void hd5770_surf_rows(const HD5770Surf *s, uint32_t y0, uint32_t y1,
                      uint64_t *lo, uint64_t *hi)
{
    uint64_t a, b;

    if (y1 <= y0) {
        *lo = *hi = s->base;
        return;
    }
    if (s->fast && s->mode < HD5770_ARRAY_1D_TILED_THIN1) {
        a = (uint64_t)y0 * s->row_bytes;
        b = (uint64_t)y1 * s->row_bytes;
    } else if (s->fast && s->mode == HD5770_ARRAY_1D_TILED_THIN1) {
        a = (uint64_t)(y0 / 8) * s->mt_per_row * s->tile_bytes;
        b = (uint64_t)DIV_ROUND_UP(y1, 8) * s->mt_per_row * s->tile_bytes;
    } else if (s->fast) {
        uint64_t band = (uint64_t)s->mt_per_row * s->mt_bytes;

        /* a pitch that is not a whole number of macro tiles overhangs */
        a = (uint64_t)(y0 >> s->mt_h_sh) * band;
        b = (uint64_t)((y1 - 1) >> s->mt_h_sh) * band +
            (uint64_t)DIV_ROUND_UP(s->pitch, 1u << s->mt_w_sh) * s->mt_bytes;
        a = (a >> s->pib_sh) << s->hi_sh;
        b = b ? (((b - 1) >> s->pib_sh) + 1) << s->hi_sh : 0;
    } else if (s->mode == HD5770_ARRAY_2D_TILED_THIN1) {
        /* split tiles: the whole surface, padded to whole macro tiles */
        uint32_t mt_w = 8 * s->bankw * s->pipes * MAX(s->mtilea, 1);
        uint32_t mt_h = MAX(8 * s->bankh * s->banks / MAX(s->mtilea, 1), 8);

        a = 0;
        b = 2 * (uint64_t)ROUND_UP(s->pitch, mt_w) * ROUND_UP(s->height, mt_h) *
            s->bpe + s->pipe_interleave * s->pipes * s->banks;
    } else {
        a = 0;
        b = hd5770_surf_size(s);
    }
    *lo = s->base + a;
    *hi = s->base + b;
}

/* bytes from the base the surface's slice 0 covers */
uint64_t hd5770_surf_size(const HD5770Surf *s)
{
    uint64_t sz = (uint64_t)s->pitch * s->height * s->bpe;

    if (s->mode == HD5770_ARRAY_2D_TILED_THIN1) {
        uint32_t mt_h = 8 * s->bankh * s->banks / MAX(s->mtilea, 1);
        uint64_t rows = ROUND_UP(s->height, mt_h);

        sz = (uint64_t)s->pitch * rows * s->bpe;
    } else if (s->mode == HD5770_ARRAY_1D_TILED_THIN1) {
        sz = (uint64_t)s->pitch * ROUND_UP(s->height, 8) * s->bpe;
    }
    return sz;
}

/* ---------------------------------------------------------------- */
/* Formats                                                          */

unsigned hd5770_fmt_bytes(unsigned fmt)
{
    switch (fmt) {
    case HD5770_FMT_8:
    case HD5770_FMT_4_4:
    case HD5770_FMT_3_3_2:
        return 1;
    case HD5770_FMT_16:
    case HD5770_FMT_16_FLOAT:
    case HD5770_FMT_8_8:
    case HD5770_FMT_5_6_5:
    case HD5770_FMT_6_5_5:
    case HD5770_FMT_1_5_5_5:
    case HD5770_FMT_4_4_4_4:
    case HD5770_FMT_5_5_5_1:
        return 2;
    case HD5770_FMT_32:
    case HD5770_FMT_32_FLOAT:
    case HD5770_FMT_16_16:
    case HD5770_FMT_16_16_FLOAT:
    case HD5770_FMT_8_24:
    case HD5770_FMT_24_8:
    case HD5770_FMT_10_11_11:
    case HD5770_FMT_11_11_10:
    case HD5770_FMT_2_10_10_10:
    case HD5770_FMT_8_8_8_8:
    case HD5770_FMT_10_10_10_2:
        return 4;
    case HD5770_FMT_32_32:
    case HD5770_FMT_32_32_FLOAT:
    case HD5770_FMT_16_16_16_16:
    case HD5770_FMT_16_16_16_16_FLOAT:
        return 8;
    case HD5770_FMT_32_32_32_FLOAT:
        return 12;
    case HD5770_FMT_32_32_32_32:
    case HD5770_FMT_32_32_32_32_FLOAT:
        return 16;
    default:
        return 0;
    }
}

/* component widths from component 0 (least significant) up */
static const uint8_t fmt_bits[64][4] = {
    [HD5770_FMT_8] = { 8 },
    [HD5770_FMT_4_4] = { 4, 4 },
    [HD5770_FMT_3_3_2] = { 3, 3, 2 },
    [HD5770_FMT_16] = { 16 },
    [HD5770_FMT_16_FLOAT] = { 16 },
    [HD5770_FMT_8_8] = { 8, 8 },
    [HD5770_FMT_5_6_5] = { 5, 6, 5 },
    [HD5770_FMT_6_5_5] = { 6, 5, 5 },
    [HD5770_FMT_1_5_5_5] = { 1, 5, 5, 5 },
    [HD5770_FMT_4_4_4_4] = { 4, 4, 4, 4 },
    [HD5770_FMT_5_5_5_1] = { 5, 5, 5, 1 },
    [HD5770_FMT_32] = { 32 },
    [HD5770_FMT_32_FLOAT] = { 32 },
    [HD5770_FMT_16_16] = { 16, 16 },
    [HD5770_FMT_16_16_FLOAT] = { 16, 16 },
    [HD5770_FMT_8_24] = { 8, 24 },
    [HD5770_FMT_24_8] = { 24, 8 },
    [HD5770_FMT_10_11_11] = { 10, 11, 11 },
    [HD5770_FMT_11_11_10] = { 11, 11, 10 },
    [HD5770_FMT_2_10_10_10] = { 2, 10, 10, 10 },
    [HD5770_FMT_8_8_8_8] = { 8, 8, 8, 8 },
    [HD5770_FMT_10_10_10_2] = { 10, 10, 10, 2 },
    [HD5770_FMT_32_32] = { 32, 32 },
    [HD5770_FMT_32_32_FLOAT] = { 32, 32 },
    [HD5770_FMT_16_16_16_16] = { 16, 16, 16, 16 },
    [HD5770_FMT_16_16_16_16_FLOAT] = { 16, 16, 16, 16 },
    [HD5770_FMT_32_32_32_32] = { 32, 32, 32, 32 },
    [HD5770_FMT_32_32_32_32_FLOAT] = { 32, 32, 32, 32 },
    [HD5770_FMT_32_32_32_FLOAT] = { 32, 32, 32 },
};

unsigned hd5770_fmt_comps(unsigned fmt)
{
    unsigned i;

    for (i = 0; i < 4 && fmt < 64 && fmt_bits[fmt][i]; i++) {
        /* count */
    }
    return i;
}

bool hd5770_fmt_is_float(unsigned fmt)
{
    switch (fmt) {
    case HD5770_FMT_16_FLOAT:
    case HD5770_FMT_32_FLOAT:
    case HD5770_FMT_16_16_FLOAT:
    case HD5770_FMT_32_32_FLOAT:
    case HD5770_FMT_16_16_16_16_FLOAT:
    case HD5770_FMT_32_32_32_32_FLOAT:
    case HD5770_FMT_32_32_32_FLOAT:
        return true;
    default:
        return false;
    }
}

void hd5770_fmt_unpack(unsigned fmt, const uint8_t *p, uint32_t c[4],
                       unsigned bits[4])
{
    unsigned i, n = hd5770_fmt_bytes(fmt), sh = 0;
    uint64_t v;

    c[0] = c[1] = c[2] = c[3] = 0;
    for (i = 0; i < 4; i++) {
        bits[i] = fmt < 64 ? fmt_bits[fmt][i] : 0;
    }
    if (n > 8) {
        for (i = 0; i < 4 && bits[i]; i++) {
            c[i] = ldl_le_p(p + 4 * i);
        }
        return;
    }
    switch (n) {
    case 1:
        v = p[0];
        break;
    case 2:
        v = lduw_le_p(p);
        break;
    case 4:
        v = ldl_le_p(p);
        break;
    case 8:
        v = ldq_le_p(p);
        break;
    default:
        return;
    }
    for (i = 0; i < 4 && bits[i]; i++) {
        c[i] = (v >> sh) & (bits[i] == 32 ? 0xffffffffu
                                          : (1u << bits[i]) - 1);
        sh += bits[i];
    }
}

void hd5770_fmt_pack(unsigned fmt, uint8_t *p, const uint32_t c[4])
{
    unsigned i, n = hd5770_fmt_bytes(fmt), sh = 0;
    uint64_t v = 0;

    if (fmt >= 64) {
        return;
    }
    if (n > 8) {
        for (i = 0; i < 4 && fmt_bits[fmt][i]; i++) {
            stl_le_p(p + 4 * i, c[i]);
        }
        return;
    }
    for (i = 0; i < 4 && fmt_bits[fmt][i]; i++) {
        unsigned b = fmt_bits[fmt][i];
        uint64_t m = b == 32 ? 0xffffffffull : (1ull << b) - 1;

        v |= ((uint64_t)c[i] & m) << sh;
        sh += b;
    }
    switch (n) {
    case 1:
        p[0] = v;
        break;
    case 2:
        stw_le_p(p, v);
        break;
    case 4:
        stl_le_p(p, v);
        break;
    case 8:
        stq_le_p(p, v);
        break;
    default:
        break;
    }
}
