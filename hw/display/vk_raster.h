/*
 * A rasterisation pipeline on Vulkan for emulated graphics cards.
 *
 * The renderer knows no card. A card's adapter gives it
 *  - images: host copies of guest surfaces, written and read as rows of
 *    tightly packed pixels in the image's format;
 *  - shaders: a vertex and a fragment stage as Vulkan GLSL 4.50 text;
 *  - draws: targets, state, textures with their samplers, and words.
 *
 * SHADER CONTRACT. Set 0, binding 0 is `readonly buffer { uint words[]; }`:
 * everything a draw reads besides textures (constants, vertex data,
 * indices) lies in it, put there with vkr_data(); a block stays where it
 * is while vkr_data_gen() does not change. Binding 1 is a second buffer
 * of words that never change (vkr_tables()). Binding 2 + i is texture
 * unit i, a sampler2D (sampler2DArray for an image with layers, sampler3D
 * for a volume). The push constants are VkrDraw.pc. Vertices are
 * not fed: the vertex stage reads words by gl_VertexIndex and
 * gl_InstanceIndex. Fragment output location i is colour target i.
 *
 * ORDER. Calls take effect in call order. A draw is queued; vkr_flush()
 * hands the queue to the GPU, vkr_image_read() waits for it. One thread
 * at a time calls in.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef HW_DISPLAY_VK_RASTER_H
#define HW_DISPLAY_VK_RASTER_H

typedef struct VkRaster VkRaster;
typedef struct VkrImage VkrImage;
typedef struct VkrShader VkrShader;

enum {
    VKR_FMT_RGBA8, VKR_FMT_R8, VKR_FMT_RG8, VKR_FMT_RGBA16, VKR_FMT_R16,
    VKR_FMT_RGBA16F, VKR_FMT_RGBA32F, VKR_FMT_R32F,
    VKR_FMT_BC1, VKR_FMT_BC2, VKR_FMT_BC3, VKR_FMT_BC4, VKR_FMT_BC5,
    VKR_FMT_BC4S, VKR_FMT_BC5S,
    VKR_FMT_ABGR4,              /* 16 bits, red in the low four */
    VKR_FMT_D32F,               /* depth, read and written as floats */
    VKR_FMT_N,
};

/* comparisons: never, <, ==, <=, >, !=, >=, always */
enum {
    VKR_BF_ZERO, VKR_BF_ONE, VKR_BF_SRC_COLOR, VKR_BF_INV_SRC_COLOR,
    VKR_BF_SRC_ALPHA, VKR_BF_INV_SRC_ALPHA, VKR_BF_DST_ALPHA,
    VKR_BF_INV_DST_ALPHA, VKR_BF_DST_COLOR, VKR_BF_INV_DST_COLOR,
    VKR_BF_SRC_ALPHA_SAT, VKR_BF_CONST_COLOR, VKR_BF_INV_CONST_COLOR,
    VKR_BF_CONST_ALPHA, VKR_BF_INV_CONST_ALPHA,
};
enum { VKR_BO_ADD, VKR_BO_SUB, VKR_BO_RSUB, VKR_BO_MIN, VKR_BO_MAX };
enum {
    VKR_WRAP_REPEAT, VKR_WRAP_MIRROR, VKR_WRAP_CLAMP, VKR_WRAP_MIRROR_ONCE,
    VKR_WRAP_BORDER,
};
/* border colours: transparent black, opaque black, opaque white */
enum { VKR_BORDER_ZERO, VKR_BORDER_BLACK, VKR_BORDER_WHITE };

typedef struct VkrBlend {
    uint8_t on, op, src, dst, aop, asrc, adst;
    uint8_t mask;               /* bit i: channel i is written */
} VkrBlend;

typedef struct VkrSampler {
    uint8_t wrap_u, wrap_v, wrap_w, border;
    uint8_t mag_linear, min_linear, mip_linear;
    float lod_bias, min_lod, max_lod;
} VkrSampler;

#define VKR_COLORS      8
#define VKR_TEXTURES    16
#define VKR_PC_WORDS    8

typedef struct VkrDraw {
    VkrShader *vs, *fs;
    unsigned ncolor;
    VkrImage *color[VKR_COLORS];
    VkrBlend blend[VKR_COLORS];
    float blend_const[4];
    VkrImage *depth;
    bool ztest, zwrite;
    uint8_t zfunc;
    uint8_t cull;               /* bit 0 front, bit 1 back */
    bool front_cw;
    float zbias_const, zbias_slope;
    /* the area drawn to, the viewport's size; x1 and y1 exclusive */
    uint32_t width, height;
    int32_t scissor[4];
    unsigned ntex;
    struct {
        VkrImage *img;
        VkrSampler s;
    } tex[VKR_TEXTURES];
    uint32_t pc[VKR_PC_WORDS];
    /* nindex 0: vertices 0..nvert-1; else the 32-bit indices at index */
    uint32_t nvert, ninst, nindex;
    uint32_t index;             /* word offset from vkr_data() */
} VkrDraw;

/* cache: file the built pipelines are kept in, or NULL */
VkRaster *vkr_open(const char *cache, const char **err);
void vkr_close(VkRaster *r);
const char *vkr_device(VkRaster *r);
/* fragment stages may read gl_BaryCoordNoPerspEXT */
bool vkr_barycentric(VkRaster *r);
void vkr_tables(VkRaster *r, const uint32_t *words, size_t n);

/* images of the format can be made, as targets too */
bool vkr_format_ok(VkRaster *r, unsigned fmt, bool target);
/*
 * layers > 1: an array, its unit a sampler2DArray; with volume the
 * layers are the depth of a 3D image (halving with the levels), its unit
 * a sampler3D. Targets have one layer. NULL: the format, or the size, is
 * not available.
 */
VkrImage *vkr_image_new(VkRaster *r, unsigned fmt, uint32_t w, uint32_t h,
                        uint32_t levels, uint32_t layers, bool volume,
                        bool target);
void vkr_image_free(VkRaster *r, VkrImage *i);
/* block formats: x, y, w, h in texels, whole blocks */
void vkr_image_write(VkRaster *r, VkrImage *i, uint32_t level, uint32_t layer,
                     uint32_t x, uint32_t y, uint32_t w, uint32_t h,
                     const void *px);
void vkr_image_read(VkRaster *r, VkrImage *i, uint32_t x, uint32_t y,
                    uint32_t w, uint32_t h, void *px);

/* stage 0 vertex, 1 fragment; NULL and *log (g_free) when it fails */
VkrShader *vkr_shader_new(VkRaster *r, unsigned stage, const char *glsl,
                          char **log);

/*
 * room for n bytes in the words buffer, *off their word offset; blocks
 * handed out before stay valid while vkr_data_gen() is the same.
 * vkr_data_room(): the blocks of the next n bytes share one generation.
 */
void *vkr_data(VkRaster *r, size_t n, uint32_t *off);
void vkr_data_room(VkRaster *r, size_t n);
uint32_t vkr_data_gen(VkRaster *r);

void vkr_draw(VkRaster *r, const VkrDraw *d);
void vkr_flush(VkRaster *r, bool wait);
void vkr_stats(VkRaster *r, GString *out);

#endif
