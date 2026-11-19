/*
 * A rasterisation pipeline on Vulkan for emulated graphics cards
 * (interface in vk_raster.h).
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/host-utils.h"
#include <vulkan/vulkan.h>
#include <glslang/Include/glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h>
#include "vk_raster.h"

#define VKR_RING        (256u << 20)
#define VKR_CBS         128

struct VkrImage {
    VkImage img;
    VkDeviceMemory mem;
    VkImageView view;           /* all levels */
    VkImageView rt;             /* level 0 */
    unsigned fmt;
    uint32_t w, h, levels, layers;
    bool volume;
};

struct VkrShader {
    VkShaderModule mod;
    unsigned stage;
};

typedef struct VkrPipeKey {
    VkrShader *vs, *fs;
    uint32_t ncolor;
    uint32_t fmt[VKR_COLORS];
    VkrBlend blend[VKR_COLORS];
    uint32_t depth;
} VkrPipeKey;

typedef struct VkrPipe {
    VkrPipeKey key;
    VkPipeline pipe;
} VkrPipe;

typedef struct VkrSamp {
    VkrSampler s;
    VkSampler samp;
} VkrSamp;

struct VkRaster {
    VkInstance inst;
    VkPhysicalDevice pdev;
    VkDevice dev;
    VkQueue queue;
    uint32_t qfi;
    VkPhysicalDeviceMemoryProperties mp;
    char name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
    bool bary, mirror_once, abgr4, abgr4_target;
    PFN_vkCmdPushDescriptorSetKHR push;

    VkCommandPool pool;
    VkCommandBuffer cb;
    bool cb_used;
    VkCommandBuffer done[VKR_CBS];
    unsigned ndone;

    VkBuffer ring, tables, rb;
    VkDeviceMemory ring_mem, tables_mem, rb_mem;
    uint8_t *ring_map, *rb_map;
    size_t ring_at, rb_size, tables_size;
    uint32_t ring_gen;

    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipelineCache pcache;
    char *cache_path;
    GHashTable *pipes;
    GArray *samps;

    bool rendering;
    unsigned cur_n;
    VkrImage *cur[VKR_COLORS], *cur_z;
    uint32_t cur_w, cur_h;
    VkPipeline cur_pipe;

    uint64_t n_draws, n_scopes, n_submits, n_waits, n_pipes, n_shaders;
    uint64_t n_writes, n_reads, n_wraps, saved_pipes;
    int64_t saved_at;
    int64_t pipe_ns, shader_ns, wait_ns;
};

static const struct {
    VkFormat f;
    uint8_t bytes, blk;
} vkr_fmt[VKR_FMT_N] = {
    [VKR_FMT_RGBA8] = { VK_FORMAT_R8G8B8A8_UNORM, 4, 1 },
    [VKR_FMT_R8] = { VK_FORMAT_R8_UNORM, 1, 1 },
    [VKR_FMT_RG8] = { VK_FORMAT_R8G8_UNORM, 2, 1 },
    [VKR_FMT_RGBA16] = { VK_FORMAT_R16G16B16A16_UNORM, 8, 1 },
    [VKR_FMT_R16] = { VK_FORMAT_R16_UNORM, 2, 1 },
    [VKR_FMT_RGBA16F] = { VK_FORMAT_R16G16B16A16_SFLOAT, 8, 1 },
    [VKR_FMT_RGBA32F] = { VK_FORMAT_R32G32B32A32_SFLOAT, 16, 1 },
    [VKR_FMT_R32F] = { VK_FORMAT_R32_SFLOAT, 4, 1 },
    [VKR_FMT_BC1] = { VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 8, 4 },
    [VKR_FMT_BC2] = { VK_FORMAT_BC2_UNORM_BLOCK, 16, 4 },
    [VKR_FMT_BC3] = { VK_FORMAT_BC3_UNORM_BLOCK, 16, 4 },
    [VKR_FMT_BC4] = { VK_FORMAT_BC4_UNORM_BLOCK, 8, 4 },
    [VKR_FMT_BC5] = { VK_FORMAT_BC5_UNORM_BLOCK, 16, 4 },
    [VKR_FMT_BC4S] = { VK_FORMAT_BC4_SNORM_BLOCK, 8, 4 },
    [VKR_FMT_BC5S] = { VK_FORMAT_BC5_SNORM_BLOCK, 16, 4 },
    [VKR_FMT_ABGR4] = { VK_FORMAT_A4B4G4R4_UNORM_PACK16, 2, 1 },
    [VKR_FMT_D32F] = { VK_FORMAT_D32_SFLOAT, 4, 1 },
};

static bool vkr_mem(VkRaster *r, VkMemoryRequirements *mr, VkFlags want,
                    VkDeviceMemory *mem)
{
    VkMemoryAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mr->size,
    };
    uint32_t i;

    for (i = 0; i < r->mp.memoryTypeCount; i++) {
        if ((mr->memoryTypeBits >> i) & 1 &&
            (r->mp.memoryTypes[i].propertyFlags & want) == want) {
            ai.memoryTypeIndex = i;
            return vkAllocateMemory(r->dev, &ai, NULL, mem) == VK_SUCCESS;
        }
    }
    return false;
}

static bool vkr_buffer(VkRaster *r, size_t size, VkFlags usage, VkBuffer *b,
                       VkDeviceMemory *mem, uint8_t **map)
{
    VkBufferCreateInfo bi = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
    };
    VkMemoryRequirements mr;
    void *p;

    if (vkCreateBuffer(r->dev, &bi, NULL, b) != VK_SUCCESS) {
        return false;
    }
    vkGetBufferMemoryRequirements(r->dev, *b, &mr);
    if (!vkr_mem(r, &mr, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, mem) ||
        vkBindBufferMemory(r->dev, *b, *mem, 0) != VK_SUCCESS ||
        vkMapMemory(r->dev, *mem, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS) {
        return false;
    }
    *map = p;
    return true;
}

static void vkr_begin(VkRaster *r)
{
    VkCommandBufferAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = r->pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    vkAllocateCommandBuffers(r->dev, &ai, &r->cb);
    vkBeginCommandBuffer(r->cb, &bi);
    r->cb_used = false;
    r->cur_pipe = VK_NULL_HANDLE;
}

static void vkr_end_scope(VkRaster *r)
{
    if (r->rendering) {
        vkCmdEndRendering(r->cb);
        r->rendering = false;
    }
}

static void vkr_cache_save(VkRaster *r);

void vkr_flush(VkRaster *r, bool wait)
{
    vkr_end_scope(r);
    if (r->n_pipes != r->saved_pipes &&
        g_get_monotonic_time() - r->saved_at > 10 * G_USEC_PER_SEC) {
        /* a run that is cut short keeps what it built */
        vkr_cache_save(r);
    }
    if (r->cb_used) {
        VkSubmitInfo si = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &r->cb,
        };

        vkEndCommandBuffer(r->cb);
        vkQueueSubmit(r->queue, 1, &si, VK_NULL_HANDLE);
        r->done[r->ndone++] = r->cb;
        r->n_submits++;
        vkr_begin(r);
    }
    if ((wait || r->ndone >= VKR_CBS - 1) && r->ndone) {
        int64_t t0 = g_get_monotonic_time();

        vkQueueWaitIdle(r->queue);
        vkFreeCommandBuffers(r->dev, r->pool, r->ndone, r->done);
        r->ndone = 0;
        r->n_waits++;
        r->wait_ns += (g_get_monotonic_time() - t0) * 1000;
    }
}

void vkr_data_room(VkRaster *r, size_t n)
{
    n = ROUND_UP(n, 16) + 4096;
    g_assert(n <= VKR_RING);
    if (r->ring_at + n > VKR_RING) {
        vkr_flush(r, true);
        r->ring_at = 0;
        r->ring_gen++;
        r->n_wraps++;
    }
}

void *vkr_data(VkRaster *r, size_t n, uint32_t *off)
{
    void *p;

    vkr_data_room(r, n);
    p = r->ring_map + r->ring_at;
    *off = r->ring_at / 4;
    r->ring_at += ROUND_UP(n, 16);
    return p;
}

uint32_t vkr_data_gen(VkRaster *r)
{
    return r->ring_gen;
}

const char *vkr_device(VkRaster *r)
{
    return r->name;
}

bool vkr_barycentric(VkRaster *r)
{
    return r->bary;
}

void vkr_tables(VkRaster *r, const uint32_t *words, size_t n)
{
    uint8_t *map;

    g_assert(!r->tables);
    r->tables_size = MAX(n, 1) * 4;
    if (vkr_buffer(r, r->tables_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                   &r->tables, &r->tables_mem, &map)) {
        memcpy(map, words, n * 4);
    }
}

bool vkr_format_ok(VkRaster *r, unsigned fmt, bool target)
{
    if (fmt >= VKR_FMT_N || (target && vkr_fmt[fmt].blk != 1)) {
        return false;
    }
    return fmt != VKR_FMT_ABGR4 || (target ? r->abgr4_target : r->abgr4);
}

static bool vkr_depth(const VkrImage *i)
{
    return i->fmt == VKR_FMT_D32F;
}

VkrImage *vkr_image_new(VkRaster *r, unsigned fmt, uint32_t w, uint32_t h,
                        uint32_t levels, uint32_t layers, bool volume,
                        bool target)
{
    VkImageCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = volume ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D,
        .format = vkr_fmt[fmt].f,
        .extent = { w, h, volume ? layers : 1 },
        .mipLevels = levels,
        .arrayLayers = volume ? 1 : layers,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                 VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkImageViewCreateInfo vi = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .viewType = volume ? VK_IMAGE_VIEW_TYPE_3D :
                    layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                               : VK_IMAGE_VIEW_TYPE_2D,
        .format = vkr_fmt[fmt].f,
    };
    VkImageMemoryBarrier bar = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    };
    VkImageAspectFlags asp = fmt == VKR_FMT_D32F ? VK_IMAGE_ASPECT_DEPTH_BIT
                                                 : VK_IMAGE_ASPECT_COLOR_BIT;
    VkMemoryRequirements mr;
    VkrImage *i;

    if (fmt >= VKR_FMT_N || !w || !h || w > 16384 || h > 16384 || !levels ||
        !layers || layers > 2048 || !vkr_format_ok(r, fmt, target) ||
        levels > 32 - clz32(MAX(w, h))) {
        return NULL;
    }
    if (target) {
        ci.usage |= fmt == VKR_FMT_D32F
                    ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                    : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    }
    i = g_new0(VkrImage, 1);
    i->fmt = fmt;
    i->w = w;
    i->h = h;
    i->levels = levels;
    i->layers = layers;
    i->volume = volume;
    if (volume) {
        layers = 1;
        target = false;
    }
    if (vkCreateImage(r->dev, &ci, NULL, &i->img) != VK_SUCCESS) {
        g_free(i);
        return NULL;
    }
    vkGetImageMemoryRequirements(r->dev, i->img, &mr);
    if (!vkr_mem(r, &mr, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &i->mem) ||
        vkBindImageMemory(r->dev, i->img, i->mem, 0) != VK_SUCCESS) {
        vkDestroyImage(r->dev, i->img, NULL);
        if (i->mem) {
            vkFreeMemory(r->dev, i->mem, NULL);
        }
        g_free(i);
        return NULL;
    }
    vi.image = i->img;
    vi.subresourceRange = (VkImageSubresourceRange){ asp, 0, levels, 0,
                                                     layers };
    vkCreateImageView(r->dev, &vi, NULL, &i->view);
    if (!volume) {
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        vkCreateImageView(r->dev, &vi, NULL, &i->rt);
    }
    vkr_end_scope(r);
    bar.image = i->img;
    bar.subresourceRange = (VkImageSubresourceRange){ asp, 0, levels, 0,
                                                      layers };
    vkCmdPipelineBarrier(r->cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0,
                         NULL, 1, &bar);
    r->cb_used = true;
    return i;
}

void vkr_image_free(VkRaster *r, VkrImage *i)
{
    unsigned k;

    if (!i) {
        return;
    }
    vkr_flush(r, true);
    for (k = 0; k < VKR_COLORS; k++) {
        if (r->cur[k] == i) {
            r->cur[k] = NULL;
        }
    }
    if (r->cur_z == i) {
        r->cur_z = NULL;
    }
    vkDestroyImageView(r->dev, i->view, NULL);
    vkDestroyImageView(r->dev, i->rt, NULL);
    vkDestroyImage(r->dev, i->img, NULL);
    vkFreeMemory(r->dev, i->mem, NULL);
    g_free(i);
}

/* writes, reads and draws to one image are kept apart */
static void vkr_settle(VkRaster *r)
{
    VkMemoryBarrier mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
    };

    vkCmdPipelineBarrier(r->cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                         NULL, 0, NULL);
}

void vkr_image_write(VkRaster *r, VkrImage *i, uint32_t level, uint32_t layer,
                     uint32_t x, uint32_t y, uint32_t w, uint32_t h,
                     const void *px)
{
    unsigned blk = vkr_fmt[i->fmt].blk;
    size_t n = (size_t)DIV_ROUND_UP(w, blk) * DIV_ROUND_UP(h, blk) *
               vkr_fmt[i->fmt].bytes;
    uint32_t lw = MAX(i->w >> level, 1u), lh = MAX(i->h >> level, 1u);
    VkBufferImageCopy c = {
        .bufferRowLength = ROUND_UP(w, blk),
        .bufferImageHeight = ROUND_UP(h, blk),
        .imageSubresource = {
            vkr_depth(i) ? VK_IMAGE_ASPECT_DEPTH_BIT
                         : VK_IMAGE_ASPECT_COLOR_BIT, level,
            i->volume ? 0 : layer, 1 },
        .imageOffset = { x, y, i->volume ? layer : 0 },
        .imageExtent = { MIN(w, lw - x), MIN(h, lh - y), 1 },
    };
    uint32_t off;
    void *p;

    if (level >= i->levels || layer >= i->layers || x >= lw || y >= lh || n > VKR_RING / 2) {
        return;
    }
    p = vkr_data(r, n, &off);
    memcpy(p, px, n);
    c.bufferOffset = (VkDeviceSize)off * 4;
    vkr_end_scope(r);
    vkr_settle(r);
    vkCmdCopyBufferToImage(r->cb, r->ring, i->img, VK_IMAGE_LAYOUT_GENERAL,
                           1, &c);
    vkr_settle(r);
    r->cb_used = true;
    r->n_writes++;
}

void vkr_image_read(VkRaster *r, VkrImage *i, uint32_t x, uint32_t y,
                    uint32_t w, uint32_t h, void *px)
{
    size_t n = (size_t)w * h * vkr_fmt[i->fmt].bytes;
    VkBufferImageCopy c = {
        .imageSubresource = {
            vkr_depth(i) ? VK_IMAGE_ASPECT_DEPTH_BIT
                         : VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .imageOffset = { x, y, 0 },
        .imageExtent = { w, h, 1 },
    };

    if (n > r->rb_size) {
        vkr_flush(r, true);
        if (r->rb) {
            vkDestroyBuffer(r->dev, r->rb, NULL);
            vkFreeMemory(r->dev, r->rb_mem, NULL);
        }
        r->rb_size = ROUND_UP(n, 1 << 20);
        vkr_buffer(r, r->rb_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &r->rb,
                   &r->rb_mem, &r->rb_map);
    }
    vkr_end_scope(r);
    vkr_settle(r);
    vkCmdCopyImageToBuffer(r->cb, i->img, VK_IMAGE_LAYOUT_GENERAL, r->rb, 1,
                           &c);
    r->cb_used = true;
    vkr_flush(r, true);
    memcpy(px, r->rb_map, n);
    r->n_reads++;
}

VkrShader *vkr_shader_new(VkRaster *r, unsigned stage, const char *glsl,
                          char **log)
{
    glslang_input_t in = {
        .language = GLSLANG_SOURCE_GLSL,
        .stage = stage ? GLSLANG_STAGE_FRAGMENT : GLSLANG_STAGE_VERTEX,
        .client = GLSLANG_CLIENT_VULKAN,
        .client_version = GLSLANG_TARGET_VULKAN_1_3,
        .target_language = GLSLANG_TARGET_SPV,
        .target_language_version = GLSLANG_TARGET_SPV_1_6,
        .code = glsl,
        .default_version = 450,
        .default_profile = GLSLANG_NO_PROFILE,
        .messages = GLSLANG_MSG_DEFAULT_BIT,
        .resource = glslang_default_resource(),
    };
    glslang_shader_t *sh = glslang_shader_create(&in);
    glslang_program_t *pg = NULL;
    VkShaderModuleCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    };
    /* the SPIR-V optimiser takes seconds over a stage */
    glslang_spv_options_t so = { .disable_optimizer = true };
    VkrShader *s = NULL;
    int64_t t0 = g_get_monotonic_time();

    *log = NULL;
    if (!glslang_shader_preprocess(sh, &in) || !glslang_shader_parse(sh, &in)) {
        *log = g_strdup(glslang_shader_get_info_log(sh));
        goto out;
    }
    pg = glslang_program_create();
    glslang_program_add_shader(pg, sh);
    if (!glslang_program_link(pg, GLSLANG_MSG_SPV_RULES_BIT |
                              GLSLANG_MSG_VULKAN_RULES_BIT)) {
        *log = g_strdup(glslang_program_get_info_log(pg));
        goto out;
    }
    glslang_program_SPIRV_generate_with_options(pg, in.stage, &so);
    ci.codeSize = glslang_program_SPIRV_get_size(pg) * 4;
    ci.pCode = glslang_program_SPIRV_get_ptr(pg);
    s = g_new0(VkrShader, 1);
    s->stage = stage;
    if (vkCreateShaderModule(r->dev, &ci, NULL, &s->mod) != VK_SUCCESS) {
        *log = g_strdup("the module was not accepted");
        g_free(s);
        s = NULL;
    }
out:
    if (pg) {
        glslang_program_delete(pg);
    }
    glslang_shader_delete(sh);
    r->n_shaders++;
    r->shader_ns += (g_get_monotonic_time() - t0) * 1000;
    return s;
}

static VkBlendFactor vkr_factor(unsigned f)
{
    static const VkBlendFactor t[] = {
        VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_SRC_COLOR,
        VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR, VK_BLEND_FACTOR_SRC_ALPHA,
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_FACTOR_DST_ALPHA,
        VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA, VK_BLEND_FACTOR_DST_COLOR,
        VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
        VK_BLEND_FACTOR_SRC_ALPHA_SATURATE, VK_BLEND_FACTOR_CONSTANT_COLOR,
        VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
        VK_BLEND_FACTOR_CONSTANT_ALPHA,
        VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA,
    };

    return f < ARRAY_SIZE(t) ? t[f] : VK_BLEND_FACTOR_ZERO;
}

static VkBlendOp vkr_blend_op(unsigned op)
{
    static const VkBlendOp t[] = {
        VK_BLEND_OP_ADD, VK_BLEND_OP_SUBTRACT, VK_BLEND_OP_REVERSE_SUBTRACT,
        VK_BLEND_OP_MIN, VK_BLEND_OP_MAX,
    };

    return op < ARRAY_SIZE(t) ? t[op] : VK_BLEND_OP_ADD;
}

static guint vkr_key_hash(gconstpointer p)
{
    const uint8_t *b = p;
    guint h = 2166136261u;
    size_t i;

    for (i = 0; i < sizeof(VkrPipeKey); i++) {
        h = (h ^ b[i]) * 16777619u;
    }
    return h;
}

static gboolean vkr_key_equal(gconstpointer a, gconstpointer b)
{
    return !memcmp(a, b, sizeof(VkrPipeKey));
}

static VkPipeline vkr_pipe(VkRaster *r, const VkrDraw *d)
{
    static const VkDynamicState dyn[] = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_DEPTH_BIAS,
        VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE,
        VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
        VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE,
    };
    VkrPipeKey key;
    VkrPipe *p;
    VkPipelineShaderStageCreateInfo st[2] = {
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = d->vs->mod,
          .pName = "main" },
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = d->fs->mod,
          .pName = "main" },
    };
    VkPipelineVertexInputStateCreateInfo vin = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    };
    VkPipelineInputAssemblyStateCreateInfo ia = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };
    VkPipelineViewportStateCreateInfo vp = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };
    VkPipelineRasterizationStateCreateInfo rs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_TRUE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .lineWidth = 1.0f,
    };
    VkPipelineMultisampleStateCreateInfo ms = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    VkPipelineDepthStencilStateCreateInfo ds = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .maxDepthBounds = 1.0f,
    };
    VkPipelineColorBlendAttachmentState ba[VKR_COLORS];
    VkPipelineColorBlendStateCreateInfo bs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = d->ncolor,
        .pAttachments = ba,
    };
    VkPipelineDynamicStateCreateInfo dy = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = ARRAY_SIZE(dyn),
        .pDynamicStates = dyn,
    };
    VkFormat cf[VKR_COLORS];
    VkPipelineRenderingCreateInfo ri = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = d->ncolor,
        .pColorAttachmentFormats = cf,
        .depthAttachmentFormat = d->depth ? VK_FORMAT_D32_SFLOAT
                                          : VK_FORMAT_UNDEFINED,
    };
    VkGraphicsPipelineCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &ri,
        .stageCount = 2,
        .pStages = st,
        .pVertexInputState = &vin,
        .pInputAssemblyState = &ia,
        .pViewportState = &vp,
        .pRasterizationState = &rs,
        .pMultisampleState = &ms,
        .pDepthStencilState = &ds,
        .pColorBlendState = &bs,
        .pDynamicState = &dy,
        .layout = r->layout,
    };
    int64_t t0;
    unsigned i;

    memset(&key, 0, sizeof(key));
    key.vs = d->vs;
    key.fs = d->fs;
    key.ncolor = d->ncolor;
    key.depth = d->depth != NULL;
    for (i = 0; i < d->ncolor; i++) {
        const VkrBlend *b = &d->blend[i];

        key.fmt[i] = d->color[i]->fmt;
        key.blend[i].mask = b->mask;
        if (b->on) {
            key.blend[i] = *b;
            key.blend[i].on = 1;
        }
    }
    p = g_hash_table_lookup(r->pipes, &key);
    if (p) {
        return p->pipe;
    }
    for (i = 0; i < d->ncolor; i++) {
        const VkrBlend *b = &key.blend[i];

        cf[i] = vkr_fmt[key.fmt[i]].f;
        ba[i] = (VkPipelineColorBlendAttachmentState){
            .blendEnable = b->on,
            .srcColorBlendFactor = vkr_factor(b->src),
            .dstColorBlendFactor = vkr_factor(b->dst),
            .colorBlendOp = vkr_blend_op(b->op),
            .srcAlphaBlendFactor = vkr_factor(b->asrc),
            .dstAlphaBlendFactor = vkr_factor(b->adst),
            .alphaBlendOp = vkr_blend_op(b->aop),
            .colorWriteMask = b->mask & 0xf,
        };
    }
    t0 = g_get_monotonic_time();
    p = g_new0(VkrPipe, 1);
    p->key = key;
    if (vkCreateGraphicsPipelines(r->dev, r->pcache, 1, &ci, NULL,
                                  &p->pipe) != VK_SUCCESS) {
        p->pipe = VK_NULL_HANDLE;
    }
    r->n_pipes++;
    r->pipe_ns += (g_get_monotonic_time() - t0) * 1000;
    g_hash_table_insert(r->pipes, &p->key, p);
    return p->pipe;
}

static VkSampler vkr_samp(VkRaster *r, const VkrSampler *s)
{
    static const VkSamplerAddressMode wrap[] = {
        VK_SAMPLER_ADDRESS_MODE_REPEAT,
        VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
    };
    static const VkBorderColor border[] = {
        VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
        VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
    };
    VkSamplerCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = s->mag_linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .minFilter = s->min_linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .mipmapMode = s->mip_linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                    : VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = wrap[MIN(s->wrap_u, 4)],
        .addressModeV = wrap[MIN(s->wrap_v, 4)],
        .addressModeW = wrap[MIN(s->wrap_w, 4)],
        .mipLodBias = s->lod_bias,
        .minLod = s->min_lod,
        .maxLod = s->max_lod,
        .borderColor = border[MIN(s->border, 2)],
    };
    VkrSamp e;
    unsigned i;

    for (i = 0; i < r->samps->len; i++) {
        VkrSamp *q = &g_array_index(r->samps, VkrSamp, i);

        if (!memcmp(&q->s, s, sizeof(*s))) {
            return q->samp;
        }
    }
    if (!r->mirror_once) {
        if (s->wrap_u == VKR_WRAP_MIRROR_ONCE) {
            ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        }
        if (s->wrap_v == VKR_WRAP_MIRROR_ONCE) {
            ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        }
        if (s->wrap_w == VKR_WRAP_MIRROR_ONCE) {
            ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        }
    }
    e.s = *s;
    e.samp = VK_NULL_HANDLE;
    vkCreateSampler(r->dev, &ci, NULL, &e.samp);
    g_array_append_val(r->samps, e);
    return e.samp;
}

void vkr_draw(VkRaster *r, const VkrDraw *d)
{
    VkPipeline pipe = vkr_pipe(r, d);
    VkDescriptorImageInfo ii[VKR_TEXTURES];
    VkWriteDescriptorSet wr[VKR_TEXTURES + 2];
    VkDescriptorBufferInfo bi[2] = {
        { r->ring, 0, VK_WHOLE_SIZE }, { r->tables, 0, VK_WHOLE_SIZE },
    };
    VkViewport vp = { 0.0f, 0.0f, d->width, d->height, 0.0f, 1.0f };
    VkRect2D sc = {
        { d->scissor[0], d->scissor[1] },
        { d->scissor[2] - d->scissor[0], d->scissor[3] - d->scissor[1] },
    };
    bool same = r->rendering && r->cur_n == d->ncolor &&
                r->cur_z == d->depth && r->cur_w == d->width &&
                r->cur_h == d->height;
    unsigned i, n = 0;

    if (!pipe) {
        return;
    }
    for (i = 0; i < d->ncolor && same; i++) {
        same = r->cur[i] == d->color[i];
    }
    if (!same) {
        VkRenderingAttachmentInfo ca[VKR_COLORS], za;
        VkRenderingInfo ri = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = { { 0, 0 }, { d->width, d->height } },
            .layerCount = 1,
            .colorAttachmentCount = d->ncolor,
            .pColorAttachments = ca,
        };

        vkr_end_scope(r);
        vkr_settle(r);
        for (i = 0; i < d->ncolor; i++) {
            ca[i] = (VkRenderingAttachmentInfo){
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .imageView = d->color[i]->rt,
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            };
            r->cur[i] = d->color[i];
        }
        if (d->depth) {
            za = (VkRenderingAttachmentInfo){
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .imageView = d->depth->rt,
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            };
            ri.pDepthAttachment = &za;
        }
        vkCmdBeginRendering(r->cb, &ri);
        r->rendering = true;
        r->cur_n = d->ncolor;
        r->cur_z = d->depth;
        r->cur_w = d->width;
        r->cur_h = d->height;
        r->n_scopes++;
    }
    r->cb_used = true;
    if (r->cur_pipe != pipe) {
        vkCmdBindPipeline(r->cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        r->cur_pipe = pipe;
    }
    for (i = 0; i < 2; i++) {
        wr[n++] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &bi[i],
        };
    }
    for (i = 0; i < d->ntex; i++) {
        if (!d->tex[i].img) {
            continue;
        }
        ii[i] = (VkDescriptorImageInfo){
            vkr_samp(r, &d->tex[i].s), d->tex[i].img->view,
            VK_IMAGE_LAYOUT_GENERAL,
        };
        wr[n++] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 2 + i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &ii[i],
        };
    }
    r->push(r->cb, VK_PIPELINE_BIND_POINT_GRAPHICS, r->layout,
                           0, n, wr);
    vkCmdPushConstants(r->cb, r->layout, VK_SHADER_STAGE_VERTEX_BIT |
                       VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(d->pc), d->pc);
    vkCmdSetViewport(r->cb, 0, 1, &vp);
    vkCmdSetScissor(r->cb, 0, 1, &sc);
    vkCmdSetBlendConstants(r->cb, d->blend_const);
    vkCmdSetCullMode(r->cb, (d->cull & 1 ? VK_CULL_MODE_FRONT_BIT : 0) |
                     (d->cull & 2 ? VK_CULL_MODE_BACK_BIT : 0));
    vkCmdSetFrontFace(r->cb, d->front_cw ? VK_FRONT_FACE_CLOCKWISE
                                         : VK_FRONT_FACE_COUNTER_CLOCKWISE);
    vkCmdSetDepthTestEnable(r->cb, d->depth && d->ztest);
    vkCmdSetDepthWriteEnable(r->cb, d->depth && d->ztest && d->zwrite);
    vkCmdSetDepthCompareOp(r->cb, d->zfunc & 7);
    vkCmdSetDepthBiasEnable(r->cb, d->zbias_const != 0.0f ||
                            d->zbias_slope != 0.0f);
    vkCmdSetDepthBias(r->cb, d->zbias_const, 0.0f, d->zbias_slope);
    if (d->nindex) {
        vkCmdBindIndexBuffer(r->cb, r->ring, (VkDeviceSize)d->index * 4,
                             VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(r->cb, d->nindex, MAX(d->ninst, 1), 0, 0, 0);
    } else {
        vkCmdDraw(r->cb, d->nvert, MAX(d->ninst, 1), 0, 0);
    }
    r->n_draws++;
}

void vkr_stats(VkRaster *r, GString *out)
{
    g_string_append_printf(out,
        "vk: %s: draws %" PRIu64 " scopes %" PRIu64 " submits %" PRIu64
        " waits %" PRIu64 " (%" PRId64 " ms) image writes %" PRIu64
        " reads %" PRIu64 " ring wraps %" PRIu64 "\n"
        "vk: shaders %" PRIu64 " (%" PRId64 " ms) pipelines %" PRIu64
        " (%" PRId64 " ms)\n", r->name, r->n_draws, r->n_scopes,
        r->n_submits, r->n_waits, r->wait_ns / 1000000, r->n_writes,
        r->n_reads, r->n_wraps, r->n_shaders, r->shader_ns / 1000000,
        r->n_pipes, r->pipe_ns / 1000000);
}

static void vkr_cache_save(VkRaster *r)
{
    size_t n = 0;
    void *p;

    r->saved_pipes = r->n_pipes;
    r->saved_at = g_get_monotonic_time();
    if (!r->cache_path ||
        vkGetPipelineCacheData(r->dev, r->pcache, &n, NULL) != VK_SUCCESS) {
        return;
    }
    p = g_malloc(n);
    if (vkGetPipelineCacheData(r->dev, r->pcache, &n, p) == VK_SUCCESS) {
        g_file_set_contents(r->cache_path, p, n, NULL);
    }
    g_free(p);
    r->saved_pipes = r->n_pipes;
    r->saved_at = g_get_monotonic_time();
}

static void vkr_pipe_free(gpointer v)
{
    g_free(v);
}

void vkr_close(VkRaster *r)
{
    GHashTableIter it;
    gpointer v;
    unsigned i;

    if (r->dev) {
        vkDeviceWaitIdle(r->dev);
        vkr_cache_save(r);
        if (r->pipes) {
            g_hash_table_iter_init(&it, r->pipes);
            while (g_hash_table_iter_next(&it, NULL, &v)) {
                vkDestroyPipeline(r->dev, ((VkrPipe *)v)->pipe, NULL);
            }
            g_hash_table_destroy(r->pipes);
        }
        for (i = 0; r->samps && i < r->samps->len; i++) {
            vkDestroySampler(r->dev, g_array_index(r->samps, VkrSamp, i).samp,
                             NULL);
        }
        if (r->samps) {
            g_array_free(r->samps, true);
        }
        vkDestroyPipelineCache(r->dev, r->pcache, NULL);
        vkDestroyPipelineLayout(r->dev, r->layout, NULL);
        vkDestroyDescriptorSetLayout(r->dev, r->dsl, NULL);
        vkDestroyBuffer(r->dev, r->ring, NULL);
        vkDestroyBuffer(r->dev, r->tables, NULL);
        vkDestroyBuffer(r->dev, r->rb, NULL);
        vkFreeMemory(r->dev, r->ring_mem, NULL);
        vkFreeMemory(r->dev, r->tables_mem, NULL);
        vkFreeMemory(r->dev, r->rb_mem, NULL);
        vkDestroyCommandPool(r->dev, r->pool, NULL);
        vkDestroyDevice(r->dev, NULL);
    }
    if (r->inst) {
        vkDestroyInstance(r->inst, NULL);
    }
    g_free(r->cache_path);
    g_free(r);
}

VkRaster *vkr_open(const char *cache, const char **err)
{
    static const char *const ext[] = {
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
        VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME,
        VK_EXT_4444_FORMATS_EXTENSION_NAME,
    };
    const char *use[ARRAY_SIZE(ext)];
    VkPhysicalDevice4444FormatsFeaturesEXT f4 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_4444_FORMATS_FEATURES_EXT,
        .formatA4B4G4R4 = VK_TRUE,
    };
    VkFormatProperties fp;
    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "qemu",
        .apiVersion = VK_API_VERSION_1_3,
    };
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR fb = {
        .sType =
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR,
    };
    VkPhysicalDeviceVulkan13Features f13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = &fb,
    };
    VkPhysicalDeviceVulkan12Features f12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &f13,
    };
    VkPhysicalDeviceFeatures2 f2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &f12,
    };
    VkPhysicalDeviceFeatures2 want = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
    };
    VkPhysicalDeviceVulkan12Features w12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
    };
    VkPhysicalDeviceVulkan13Features w13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .dynamicRendering = VK_TRUE,
    };
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueCount = 1,
        .pQueuePriorities = &prio,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &want,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
        .ppEnabledExtensionNames = use,
    };
    VkCommandPoolCreateInfo pci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    };
    VkDescriptorSetLayoutBinding lb[VKR_TEXTURES + 2];
    VkDescriptorSetLayoutCreateInfo lci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
        .bindingCount = ARRAY_SIZE(lb),
        .pBindings = lb,
    };
    VkPushConstantRange pcr = {
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
        VKR_PC_WORDS * 4,
    };
    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcr,
    };
    VkPipelineCacheCreateInfo cci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
    };
    VkPhysicalDevice pd[8];
    VkQueueFamilyProperties qf[16];
    VkExtensionProperties *ep;
    uint32_t npd = ARRAY_SIZE(pd), nqf = ARRAY_SIZE(qf), nep = 0, i, k;
    int best = -1, score = -1;
    gchar *blob = NULL;
    gsize blob_len = 0;
    bool push = false;
    VkRaster *r;

    *err = "no Vulkan 1.3 device";
    r = g_new0(VkRaster, 1);
    if (vkCreateInstance(&ici, NULL, &r->inst) != VK_SUCCESS) {
        r->inst = VK_NULL_HANDLE;
        goto fail;
    }
    vkEnumeratePhysicalDevices(r->inst, &npd, pd);
    for (i = 0; i < npd; i++) {
        VkPhysicalDeviceProperties pp;
        int s;

        vkGetPhysicalDeviceProperties(pd[i], &pp);
        if (pp.apiVersion < VK_API_VERSION_1_3) {
            continue;
        }
        s = pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 3 :
            pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2 : 1;
        if (s > score) {
            score = s;
            best = i;
        }
    }
    if (best < 0) {
        goto fail;
    }
    r->pdev = pd[best];
    {
        VkPhysicalDeviceProperties pp;

        vkGetPhysicalDeviceProperties(r->pdev, &pp);
        g_strlcpy(r->name, pp.deviceName, sizeof(r->name));
    }
    vkGetPhysicalDeviceMemoryProperties(r->pdev, &r->mp);
    vkGetPhysicalDeviceQueueFamilyProperties(r->pdev, &nqf, qf);
    for (i = 0; i < nqf && !(qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT); i++) {
        /* search */
    }
    if (i == nqf) {
        goto fail;
    }
    r->qfi = i;
    vkEnumerateDeviceExtensionProperties(r->pdev, NULL, &nep, NULL);
    ep = g_new0(VkExtensionProperties, nep + 1);
    vkEnumerateDeviceExtensionProperties(r->pdev, NULL, &nep, ep);
    for (k = 0; k < nep; k++) {
        push |= !strcmp(ep[k].extensionName, ext[0]);
        r->bary |= !strcmp(ep[k].extensionName, ext[1]);
        r->abgr4 |= !strcmp(ep[k].extensionName, ext[2]);
    }
    g_free(ep);
    if (!r->bary) {
        f13.pNext = NULL;
    }
    vkGetPhysicalDeviceFeatures2(r->pdev, &f2);
    *err = "the Vulkan device lacks a feature the renderer needs";
    if (!push || !f13.dynamicRendering || !f2.features.depthClamp ||
        !f2.features.shaderClipDistance || !f2.features.shaderFloat64 ||
        !f2.features.independentBlend || !f2.features.textureCompressionBC) {
        goto fail;
    }
    r->bary = r->bary && fb.fragmentShaderBarycentric;
    r->mirror_once = f12.samplerMirrorClampToEdge;
    vkGetPhysicalDeviceFormatProperties(r->pdev,
                                        VK_FORMAT_A4B4G4R4_UNORM_PACK16, &fp);
    r->abgr4 = r->abgr4 && (fp.optimalTilingFeatures &
                            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT);
    r->abgr4_target = r->abgr4 && (fp.optimalTilingFeatures &
                                   VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT);
    want.features.depthClamp = VK_TRUE;
    want.features.shaderClipDistance = VK_TRUE;
    want.features.shaderFloat64 = VK_TRUE;
    want.features.independentBlend = VK_TRUE;
    want.features.textureCompressionBC = VK_TRUE;
    w12.samplerMirrorClampToEdge = r->mirror_once;
    want.pNext = &w12;
    w12.pNext = &w13;
    use[dci.enabledExtensionCount++] = ext[0];
    if (r->bary) {
        fb.pNext = NULL;
        w13.pNext = &fb;
        use[dci.enabledExtensionCount++] = ext[1];
    }
    if (r->abgr4) {
        f4.pNext = (void *)w13.pNext;
        w13.pNext = &f4;
        use[dci.enabledExtensionCount++] = ext[2];
    }
    qci.queueFamilyIndex = r->qfi;
    if (vkCreateDevice(r->pdev, &dci, NULL, &r->dev) != VK_SUCCESS) {
        r->dev = VK_NULL_HANDLE;
        goto fail;
    }
    vkGetDeviceQueue(r->dev, r->qfi, 0, &r->queue);
    r->push = (PFN_vkCmdPushDescriptorSetKHR)
              vkGetDeviceProcAddr(r->dev, "vkCmdPushDescriptorSetKHR");
    pci.queueFamilyIndex = r->qfi;
    for (i = 0; i < ARRAY_SIZE(lb); i++) {
        lb[i] = (VkDescriptorSetLayoutBinding){
            .binding = i,
            .descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                              : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
                          VK_SHADER_STAGE_FRAGMENT_BIT,
        };
    }
    *err = "cannot set up the Vulkan renderer";
    if (!r->push ||
        vkCreateCommandPool(r->dev, &pci, NULL, &r->pool) != VK_SUCCESS ||
        vkCreateDescriptorSetLayout(r->dev, &lci, NULL,
                                    &r->dsl) != VK_SUCCESS) {
        goto fail;
    }
    plci.pSetLayouts = &r->dsl;
    if (cache && g_file_get_contents(cache, &blob, &blob_len, NULL)) {
        cci.initialDataSize = blob_len;
        cci.pInitialData = blob;
    }
    r->cache_path = g_strdup(cache);
    if (vkCreatePipelineLayout(r->dev, &plci, NULL, &r->layout) != VK_SUCCESS ||
        vkCreatePipelineCache(r->dev, &cci, NULL, &r->pcache) != VK_SUCCESS ||
        !vkr_buffer(r, VKR_RING, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &r->ring, &r->ring_mem,
                    &r->ring_map)) {
        g_free(blob);
        goto fail;
    }
    g_free(blob);
    r->pipes = g_hash_table_new_full(vkr_key_hash, vkr_key_equal, NULL,
                                     vkr_pipe_free);
    r->samps = g_array_new(false, false, sizeof(VkrSamp));
    glslang_initialize_process();
    vkr_begin(r);
    *err = NULL;
    return r;
fail:
    vkr_close(r);
    return NULL;
}
