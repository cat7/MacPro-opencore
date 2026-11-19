/*
 * A rasterisation pipeline on Vulkan for emulated graphics cards
 * (interface in vk_raster.h).
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/host-utils.h"
#include "qemu/atomic.h"
#include <vulkan/vulkan.h>
#include <glslang/Include/glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h>
#include "vk_raster.h"

#define VKR_RING        (256u << 20)
#define VKR_KEEP        (384u << 20)    /* behind the ring, vkr_keep() */

#define VKR_BLOCK       (128u << 20)    /* image memory comes in these */

typedef struct VkrSpan {
    VkDeviceSize off, size;
} VkrSpan;

/* device memory images share: its free spans, in order */
typedef struct VkrBlock {
    VkDeviceMemory mem;
    uint32_t type;
    VkDeviceSize size;
    GArray *free;
} VkrBlock;

struct VkrImage {
    VkImage img;
    VkrBlock *blk;
    VkrSpan span;
    VkImageView view;           /* all levels */
    VkImageView rt;             /* level 0 */
    unsigned fmt;
    uint32_t w, h, levels, layers;
    bool volume;
};

/* host memory image pixels are read into */
typedef struct VkrRead {
    VkBuffer buf;
    VkDeviceMemory mem;
    uint8_t *map;
    size_t size;
} VkrRead;

/* a command buffer handed to the GPU; done and held under VkRaster.lock */
struct VkrMark {
    VkFence fence;
    VkCommandBuffer cb;
    GPtrArray *reads;           /* the reads queued in it */
    bool done;                  /* reached, its command buffer freed */
    bool held;                  /* the caller has it */
    /* its words lie from here to where the ring was when it was handed over */
    size_t ring_from;
    uint32_t ring_gen;
};

struct VkrShader {
    VkShaderModule mod;
    unsigned stage;
};

typedef VkrPipeDesc VkrPipeKey;

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
    GQueue subs;                /* VkrMark: with the GPU, oldest first */
    GPtrArray *reads;           /* VkrRead: queued in cb */
    GPtrArray *idle;            /* VkrRead: free, under lock */
    GMutex lock;

    VkBuffer ring, tables;
    VkDeviceMemory ring_mem, tables_mem;
    uint8_t *ring_map;
    size_t ring_at, keep_at, tables_size;
    uint32_t ring_gen;
    /* where the command buffer's words begin; the ring is free up to ring_end */
    size_t ring_from, ring_end;

    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipelineCache pcache;
    char *cache_dir, *cache_path;
    GHashTable *pipes;
    GArray *samps;
    GPtrArray *blocks, *dead;
    uint64_t image_bytes;

    bool rendering;
    unsigned cur_n;
    VkrImage *cur[VKR_COLORS], *cur_z;
    uint32_t cur_w, cur_h;
    VkPipeline cur_pipe;
    /* what the command buffer has set and bound; set: it has any */
    struct {
        bool set;
        uint32_t width, height;
        int32_t scissor[4];
        float blend_const[4], zbias_const, zbias_slope;
        uint8_t cull, front_cw, ztest, zwrite, zfunc;
        struct {
            VkSampler samp;
            VkImageView view;
        } tex[VKR_TEXTURES];
    } cur_state;

    uint64_t n_draws, n_scopes, n_submits, n_waits, n_pipes, n_shaders;
    uint64_t n_compiled, n_ahead;
    uint64_t n_writes, n_reads, n_copies, n_wraps, saved_pipes;
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

/* a mapped buffer; cached: memory the host reads fast */
static bool vkr_buffer(VkRaster *r, size_t size, VkFlags usage, bool cached,
                       VkBuffer *b, VkDeviceMemory *mem, uint8_t **map)
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
    if (!(cached && vkr_mem(r, &mr, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                            VK_MEMORY_PROPERTY_HOST_CACHED_BIT, mem)) &&
        !vkr_mem(r, &mr, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, mem)) {
        return false;
    }
    if (vkBindBufferMemory(r->dev, *b, *mem, 0) != VK_SUCCESS ||
        vkMapMemory(r->dev, *mem, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS) {
        return false;
    }
    *map = p;
    return true;
}

/* room for an image in a block of its memory type */
static bool vkr_image_mem(VkRaster *r, VkrImage *i)
{
    VkMemoryRequirements mr;
    VkMemoryAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    };
    VkrBlock *b = NULL;
    VkrSpan sp;
    uint32_t t;
    unsigned k, j;

    vkGetImageMemoryRequirements(r->dev, i->img, &mr);
    for (t = 0; t < r->mp.memoryTypeCount; t++) {
        if ((mr.memoryTypeBits >> t) & 1 &&
            r->mp.memoryTypes[t].propertyFlags &
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            break;
        }
    }
    if (t == r->mp.memoryTypeCount) {
        return false;
    }
    for (k = 0; k <= r->blocks->len; k++) {
        if (k == r->blocks->len) {
            b = g_new0(VkrBlock, 1);
            b->type = t;
            b->size = MAX(mr.size, VKR_BLOCK);
            ai.allocationSize = b->size;
            ai.memoryTypeIndex = t;
            if (vkAllocateMemory(r->dev, &ai, NULL, &b->mem) != VK_SUCCESS) {
                g_free(b);
                return false;
            }
            b->free = g_array_new(false, false, sizeof(VkrSpan));
            sp = (VkrSpan){ 0, b->size };
            g_array_append_val(b->free, sp);
            g_ptr_array_add(r->blocks, b);
        }
        b = g_ptr_array_index(r->blocks, k);
        for (j = 0; b->type == t && j < b->free->len; j++) {
            VkrSpan *f = &g_array_index(b->free, VkrSpan, j);
            VkDeviceSize at = ROUND_UP(f->off, mr.alignment);
            VkDeviceSize end = f->off + f->size;

            if (at + mr.size > end) {
                continue;
            }
            /* the span taken starts where the free one did */
            i->span = (VkrSpan){ f->off, at + mr.size - f->off };
            f->off += i->span.size;
            f->size -= i->span.size;
            if (!f->size) {
                g_array_remove_index(b->free, j);
            }
            i->blk = b;
            r->image_bytes += i->span.size;
            return vkBindImageMemory(r->dev, i->img, b->mem,
                                     at) == VK_SUCCESS;
        }
    }
    return false;
}

static void vkr_image_mem_free(VkRaster *r, VkrImage *i)
{
    VkrBlock *b = i->blk;
    VkrSpan *f;
    unsigned j;

    if (!b) {
        return;
    }
    for (j = 0; j < b->free->len &&
                g_array_index(b->free, VkrSpan, j).off < i->span.off; j++) {
        /* search */
    }
    g_array_insert_val(b->free, j, i->span);
    r->image_bytes -= i->span.size;
    f = &g_array_index(b->free, VkrSpan, j);
    if (j + 1 < b->free->len && f->off + f->size == f[1].off) {
        f->size += f[1].size;
        g_array_remove_index(b->free, j + 1);
    }
    if (j && f[-1].off + f[-1].size == f->off) {
        f[-1].size += f->size;
        g_array_remove_index(b->free, j);
    }
    if (b->free->len == 1 &&
        g_array_index(b->free, VkrSpan, 0).size == b->size) {
        vkFreeMemory(r->dev, b->mem, NULL);
        g_array_free(b->free, true);
        g_ptr_array_remove_fast(r->blocks, b);
        g_free(b);
    }
}

static void vkr_image_destroy(VkRaster *r, VkrImage *i)
{
    vkDestroyImageView(r->dev, i->view, NULL);
    vkDestroyImageView(r->dev, i->rt, NULL);
    vkDestroyImage(r->dev, i->img, NULL);
    vkr_image_mem_free(r, i);
    g_free(i);
}

/* nothing queued uses the images freed: they go */
static void vkr_dead(VkRaster *r)
{
    unsigned k;

    for (k = 0; k < r->dead->len; k++) {
        vkr_image_destroy(r, g_ptr_array_index(r->dead, k));
    }
    g_ptr_array_set_size(r->dead, 0);
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
    memset(&r->cur_state, 0, sizeof(r->cur_state));
}

static void vkr_end_scope(VkRaster *r)
{
    if (r->rendering) {
        vkCmdEndRendering(r->cb);
        r->rendering = false;
    }
}

static void vkr_cache_save(VkRaster *r);

/* with the lock: neither the GPU nor the caller needs the mark */
static void vkr_mark_drop(VkRaster *r, VkrMark *k)
{
    unsigned i;

    for (i = 0; i < k->reads->len; i++) {
        g_ptr_array_add(r->idle, g_ptr_array_index(k->reads, i));
    }
    g_ptr_array_free(k->reads, true);
    vkDestroyFence(r->dev, k->fence, NULL);
    g_free(k);
}

/* the command buffers the GPU is done with are freed; all: every one is */
static void vkr_reap(VkRaster *r, bool all)
{
    VkrMark *k;

    while ((k = g_queue_peek_head(&r->subs)) &&
           (all || vkGetFenceStatus(r->dev, k->fence) == VK_SUCCESS)) {
        g_queue_pop_head(&r->subs);
        vkFreeCommandBuffers(r->dev, r->pool, 1, &k->cb);
        g_mutex_lock(&r->lock);
        k->done = true;
        if (!k->held) {
            vkr_mark_drop(r, k);
        }
        g_mutex_unlock(&r->lock);
    }
    if (g_queue_is_empty(&r->subs)) {
        vkr_dead(r);
    }
}

static VkrMark *vkr_submit(VkRaster *r)
{
    VkFenceCreateInfo fi = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &r->cb,
    };
    VkrMark *k = g_new0(VkrMark, 1);

    vkr_end_scope(r);
    vkCreateFence(r->dev, &fi, NULL, &k->fence);
    k->cb = r->cb;
    k->ring_from = r->ring_from;
    k->ring_gen = r->ring_gen;
    r->ring_from = r->ring_at;
    k->reads = r->reads;
    r->reads = g_ptr_array_new();
    vkEndCommandBuffer(r->cb);
    vkQueueSubmit(r->queue, 1, &si, k->fence);
    g_queue_push_tail(&r->subs, k);
    r->n_submits++;
    vkr_begin(r);
    return k;
}

void vkr_flush(VkRaster *r, bool wait)
{
    vkr_end_scope(r);
    if (r->n_pipes != r->saved_pipes &&
        g_get_monotonic_time() - r->saved_at > 10 * G_USEC_PER_SEC) {
        /* a run that is cut short keeps what it built */
        vkr_cache_save(r);
    }
    if (r->cb_used) {
        vkr_submit(r);
    }
    if (wait && !g_queue_is_empty(&r->subs)) {
        int64_t t0 = g_get_monotonic_time();

        vkQueueWaitIdle(r->queue);
        r->n_waits++;
        r->wait_ns += (g_get_monotonic_time() - t0) * 1000;
    }
    vkr_reap(r, wait);
}

VkrMark *vkr_mark(VkRaster *r)
{
    VkrMark *k = vkr_submit(r);

    k->held = true;
    return k;
}

void vkr_mark_wait(VkRaster *r, VkrMark *k)
{
    vkWaitForFences(r->dev, 1, &k->fence, VK_TRUE, UINT64_MAX);
}

void vkr_mark_free(VkRaster *r, VkrMark *k)
{
    g_mutex_lock(&r->lock);
    k->held = false;
    if (k->done) {
        vkr_mark_drop(r, k);
    }
    g_mutex_unlock(&r->lock);
}

/*
 * At its end the ring starts again; the words of command buffers the GPU
 * still has, from the oldest one's on, are waited for only when the room
 * before them runs out.
 */
void vkr_data_room(VkRaster *r, size_t n)
{
    n = ROUND_UP(n, 16) + 4096;
    g_assert(n <= VKR_RING);
    if (r->ring_at + n > VKR_RING) {
        vkr_flush(r, false);
        r->ring_at = r->ring_from = r->ring_end = 0;
        r->ring_gen++;
        r->n_wraps++;
    }
    while (r->ring_at + n > r->ring_end) {
        VkrMark *k;

        vkr_reap(r, false);
        k = g_queue_peek_head(&r->subs);
        if (!k || k->ring_gen == r->ring_gen) {
            r->ring_end = VKR_RING;
        } else if (r->ring_at + n <= k->ring_from) {
            r->ring_end = k->ring_from;
        } else {
            int64_t t0 = g_get_monotonic_time();

            vkWaitForFences(r->dev, 1, &k->fence, VK_TRUE, UINT64_MAX);
            r->n_waits++;
            r->wait_ns += (g_get_monotonic_time() - t0) * 1000;
        }
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

void *vkr_keep(VkRaster *r, size_t n, uint32_t *off)
{
    void *p = r->ring_map + VKR_RING + r->keep_at;

    n = ROUND_UP(n, 16);
    if (n > VKR_KEEP - r->keep_at) {
        return NULL;
    }
    *off = (VKR_RING + r->keep_at) / 4;
    r->keep_at += n;
    return p;
}

void vkr_keep_reset(VkRaster *r)
{
    vkr_flush(r, true);
    r->keep_at = 0;
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
                   false, &r->tables, &r->tables_mem, &map)) {
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
    if (!vkr_image_mem(r, i)) {
        vkr_image_destroy(r, i);
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
    vkr_end_scope(r);
    for (k = 0; k < VKR_COLORS; k++) {
        if (r->cur[k] == i) {
            r->cur[k] = NULL;
        }
    }
    if (r->cur_z == i) {
        r->cur_z = NULL;
    }
    /* destroyed once the draws queued are done */
    g_ptr_array_add(r->dead, i);
    if (r->dead->len >= 256) {
        vkr_flush(r, true);
    }
}

uint64_t vkr_image_bytes(VkRaster *r)
{
    return r->image_bytes;
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

void vkr_image_copy(VkRaster *r, VkrImage *from, VkrImage *to, uint32_t level,
                    uint32_t layer, uint32_t w, uint32_t h)
{
    VkImageAspectFlags a = vkr_depth(to) ? VK_IMAGE_ASPECT_DEPTH_BIT
                                         : VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageCopy c = {
        .srcSubresource = { a, 0, 0, 1 },
        .dstSubresource = { a, level, to->volume ? 0 : layer, 1 },
        .dstOffset = { 0, 0, to->volume ? layer : 0 },
        .extent = { w, h, 1 },
    };

    if (from->fmt != to->fmt || level >= to->levels || layer >= to->layers ||
        w > from->w || h > from->h || w > MAX(to->w >> level, 1u) ||
        h > MAX(to->h >> level, 1u)) {
        return;
    }
    vkr_end_scope(r);
    vkr_settle(r);
    vkCmdCopyImage(r->cb, from->img, VK_IMAGE_LAYOUT_GENERAL, to->img,
                   VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    vkr_settle(r);
    r->cb_used = true;
    r->n_copies++;
}

const void *vkr_image_read_later(VkRaster *r, VkrImage *i, uint32_t x,
                                 uint32_t y, uint32_t w, uint32_t h)
{
    size_t n = (size_t)w * h * vkr_fmt[i->fmt].bytes;
    VkBufferImageCopy c = {
        .imageSubresource = {
            vkr_depth(i) ? VK_IMAGE_ASPECT_DEPTH_BIT
                         : VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .imageOffset = { x, y, 0 },
        .imageExtent = { w, h, 1 },
    };
    VkrRead *rd = NULL;
    unsigned k;

    g_mutex_lock(&r->lock);
    for (k = 0; k < r->idle->len; k++) {
        VkrRead *q = g_ptr_array_index(r->idle, k);

        if (q->size >= n && q->size / 4 <= n) {
            rd = g_ptr_array_remove_index_fast(r->idle, k);
            break;
        }
    }
    g_mutex_unlock(&r->lock);
    if (!rd) {
        rd = g_new0(VkrRead, 1);
        rd->size = MAX(pow2ceil(n), 65536);
        vkr_buffer(r, rd->size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true,
                   &rd->buf, &rd->mem, &rd->map);
    }
    vkr_end_scope(r);
    vkr_settle(r);
    vkCmdCopyImageToBuffer(r->cb, i->img, VK_IMAGE_LAYOUT_GENERAL, rd->buf, 1,
                           &c);
    r->cb_used = true;
    g_ptr_array_add(r->reads, rd);
    r->n_reads++;
    return rd->map;
}

void vkr_image_read(VkRaster *r, VkrImage *i, uint32_t x, uint32_t y,
                    uint32_t w, uint32_t h, void *px)
{
    const void *p = vkr_image_read_later(r, i, x, y, w, h);
    VkrMark *k = vkr_mark(r);

    vkr_flush(r, true);
    memcpy(px, p, (size_t)w * h * vkr_fmt[i->fmt].bytes);
    vkr_mark_free(r, k);
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
    glslang_shader_t *sh = NULL;
    glslang_program_t *pg = NULL;
    VkShaderModuleCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    };
    /* the SPIR-V optimiser takes seconds over a stage */
    glslang_spv_options_t so = { .disable_optimizer = true };
    g_autofree char *sum = NULL, *path = NULL, *spv = NULL;
    gsize spv_len = 0;
    VkrShader *s = NULL;
    int64_t t0 = g_get_monotonic_time();

    *log = NULL;
    if (r->cache_dir) {
        /* the stage's SPIR-V is kept under the text's checksum */
        sum = g_compute_checksum_for_string(G_CHECKSUM_SHA256, glsl, -1);
        path = g_strdup_printf("%s/%s.%u.spv", r->cache_dir, sum, stage);
        g_file_get_contents(path, &spv, &spv_len, NULL);
    }
    if (spv_len && !(spv_len & 3)) {
        ci.codeSize = spv_len;
        ci.pCode = (const uint32_t *)spv;
    } else {
        sh = glslang_shader_create(&in);
        if (!glslang_shader_preprocess(sh, &in) ||
            !glslang_shader_parse(sh, &in)) {
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
        if (path) {
            g_file_set_contents(path, (const char *)ci.pCode, ci.codeSize,
                                NULL);
        }
        qatomic_inc(&r->n_compiled);
    }
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
    if (sh) {
        glslang_shader_delete(sh);
    }
    qatomic_inc(&r->n_shaders);
    qatomic_add(&r->shader_ns, (g_get_monotonic_time() - t0) * 1000);
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

/* the pipeline of p->key; any thread */
static void vkr_pipe_build(VkRaster *r, VkrPipe *p)
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
    const VkrPipeKey *key = &p->key;
    VkPipelineShaderStageCreateInfo st[2] = {
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = key->vs->mod,
          .pName = "main" },
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = key->fs->mod,
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
        .attachmentCount = key->ncolor,
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
        .colorAttachmentCount = key->ncolor,
        .pColorAttachmentFormats = cf,
        .depthAttachmentFormat = key->depth ? VK_FORMAT_D32_SFLOAT
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

    for (i = 0; i < key->ncolor; i++) {
        const VkrBlend *b = &key->blend[i];

        cf[i] = vkr_fmt[key->fmt[i]].f;
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
    if (vkCreateGraphicsPipelines(r->dev, r->pcache, 1, &ci, NULL,
                                  &p->pipe) != VK_SUCCESS) {
        p->pipe = VK_NULL_HANDLE;
    }
    qatomic_inc(&r->n_pipes);
    qatomic_add(&r->pipe_ns, (g_get_monotonic_time() - t0) * 1000);
}

/* the description as the key: what does not matter is zero */
static void vkr_pipe_key(const VkrPipeDesc *d, VkrPipeKey *key)
{
    unsigned i;

    memset(key, 0, sizeof(*key));
    key->vs = d->vs;
    key->fs = d->fs;
    key->ncolor = d->ncolor;
    key->depth = d->depth != 0;
    for (i = 0; i < d->ncolor; i++) {
        const VkrBlend *b = &d->blend[i];

        key->fmt[i] = d->fmt[i];
        key->blend[i].mask = b->mask;
        if (b->on) {
            key->blend[i] = *b;
            key->blend[i].on = 1;
        }
    }
}

static VkPipeline vkr_pipe(VkRaster *r, const VkrDraw *d)
{
    VkrPipeDesc pd = { .vs = d->vs, .fs = d->fs, .ncolor = d->ncolor,
                       .depth = d->depth != NULL };
    VkrPipeKey key;
    VkrPipe *p;
    unsigned i;

    for (i = 0; i < d->ncolor; i++) {
        pd.fmt[i] = d->color[i]->fmt;
        pd.blend[i] = d->blend[i];
    }
    vkr_pipe_key(&pd, &key);
    p = g_hash_table_lookup(r->pipes, &key);
    if (!p) {
        p = g_new0(VkrPipe, 1);
        p->key = key;
        vkr_pipe_build(r, p);
        g_hash_table_insert(r->pipes, &p->key, p);
    }
    return p->pipe;
}

typedef struct VkrJob {
    unsigned stage;
    const char *glsl;
    VkrShader **out;
    VkrPipe *pipe;
} VkrJob;

static void vkr_job(gpointer data, gpointer opaque)
{
    VkrJob *j = data;
    char *log;

    if (j->pipe) {
        vkr_pipe_build(opaque, j->pipe);
    } else {
        *j->out = vkr_shader_new(opaque, j->stage, j->glsl, &log);
        g_free(log);
    }
}

static void vkr_jobs(VkRaster *r, VkrJob *j, unsigned n)
{
    GThreadPool *pool;
    unsigned i;

    if (!n) {
        return;
    }
    pool = g_thread_pool_new(vkr_job, r, MIN(n, MAX(g_get_num_processors(),
                                                    2u) - 1), false, NULL);
    for (i = 0; i < n; i++) {
        g_thread_pool_push(pool, &j[i], NULL);
    }
    g_thread_pool_free(pool, false, true);
    r->n_ahead += n;
}

void vkr_shaders_new(VkRaster *r, unsigned n, const unsigned *stage,
                     char *const *glsl, VkrShader **out)
{
    g_autofree VkrJob *j = g_new0(VkrJob, n);
    unsigned i;

    for (i = 0; i < n; i++) {
        j[i].stage = stage[i];
        j[i].glsl = glsl[i];
        j[i].out = &out[i];
    }
    vkr_jobs(r, j, n);
}

void vkr_pipes_ahead(VkRaster *r, const VkrPipeDesc *d, unsigned n)
{
    g_autofree VkrJob *j = g_new0(VkrJob, n);
    unsigned i, k = 0;

    for (i = 0; i < n; i++) {
        VkrPipe *p = g_new0(VkrPipe, 1);

        vkr_pipe_key(&d[i], &p->key);
        if (g_hash_table_contains(r->pipes, &p->key)) {
            g_free(p);
            continue;
        }
        /* in the table at once: one of a kind is built */
        g_hash_table_insert(r->pipes, &p->key, p);
        j[k++].pipe = p;
    }
    vkr_jobs(r, j, k);
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
    bool ztest = d->depth && d->ztest, zwrite = ztest && d->zwrite;
    bool set = r->cur_state.set;
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
    for (i = 0; i < 2 && !r->cur_state.set; i++) {
        wr[n++] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &bi[i],
        };
    }
    for (i = 0; i < d->ntex; i++) {
        VkSampler samp;

        if (!d->tex[i].img) {
            continue;
        }
        samp = vkr_samp(r, &d->tex[i].s);
        if (r->cur_state.tex[i].samp == samp &&
            r->cur_state.tex[i].view == d->tex[i].img->view) {
            continue;
        }
        r->cur_state.tex[i].samp = samp;
        r->cur_state.tex[i].view = d->tex[i].img->view;
        ii[i] = (VkDescriptorImageInfo){
            samp, d->tex[i].img->view, VK_IMAGE_LAYOUT_GENERAL,
        };
        wr[n++] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 2 + i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &ii[i],
        };
    }
    if (n) {
        r->push(r->cb, VK_PIPELINE_BIND_POINT_GRAPHICS, r->layout, 0, n, wr);
    }
    vkCmdPushConstants(r->cb, r->layout, VK_SHADER_STAGE_VERTEX_BIT |
                       VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(d->pc), d->pc);
    if (!set || r->cur_state.width != d->width ||
        r->cur_state.height != d->height) {
        vkCmdSetViewport(r->cb, 0, 1, &vp);
        r->cur_state.width = d->width;
        r->cur_state.height = d->height;
    }
    if (!set || memcmp(r->cur_state.scissor, d->scissor, sizeof(d->scissor))) {
        vkCmdSetScissor(r->cb, 0, 1, &sc);
        memcpy(r->cur_state.scissor, d->scissor, sizeof(d->scissor));
    }
    if (!set || memcmp(r->cur_state.blend_const, d->blend_const,
                       sizeof(d->blend_const))) {
        vkCmdSetBlendConstants(r->cb, d->blend_const);
        memcpy(r->cur_state.blend_const, d->blend_const,
               sizeof(d->blend_const));
    }
    if (!set || r->cur_state.cull != d->cull) {
        vkCmdSetCullMode(r->cb, (d->cull & 1 ? VK_CULL_MODE_FRONT_BIT : 0) |
                         (d->cull & 2 ? VK_CULL_MODE_BACK_BIT : 0));
        r->cur_state.cull = d->cull;
    }
    if (!set || r->cur_state.front_cw != d->front_cw) {
        vkCmdSetFrontFace(r->cb, d->front_cw ? VK_FRONT_FACE_CLOCKWISE
                                             : VK_FRONT_FACE_COUNTER_CLOCKWISE);
        r->cur_state.front_cw = d->front_cw;
    }
    if (!set || r->cur_state.ztest != ztest) {
        vkCmdSetDepthTestEnable(r->cb, ztest);
        r->cur_state.ztest = ztest;
    }
    if (!set || r->cur_state.zwrite != zwrite) {
        vkCmdSetDepthWriteEnable(r->cb, zwrite);
        r->cur_state.zwrite = zwrite;
    }
    if (!set || r->cur_state.zfunc != (d->zfunc & 7)) {
        vkCmdSetDepthCompareOp(r->cb, d->zfunc & 7);
        r->cur_state.zfunc = d->zfunc & 7;
    }
    if (!set || memcmp(&r->cur_state.zbias_const, &d->zbias_const, 4) ||
        memcmp(&r->cur_state.zbias_slope, &d->zbias_slope, 4)) {
        vkCmdSetDepthBiasEnable(r->cb, d->zbias_const != 0.0f ||
                                d->zbias_slope != 0.0f);
        vkCmdSetDepthBias(r->cb, d->zbias_const, 0.0f, d->zbias_slope);
        r->cur_state.zbias_const = d->zbias_const;
        r->cur_state.zbias_slope = d->zbias_slope;
    }
    r->cur_state.set = true;
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
        " reads %" PRIu64 " copies %" PRIu64 " ring wraps %" PRIu64 "\n"
        "vk: stages %" PRIu64 " (%" PRIu64 " compiled, %" PRId64
        " ms) pipelines %" PRIu64
        " (%" PRId64 " ms); built ahead %" PRIu64 "\n", r->name, r->n_draws,
        r->n_scopes, r->n_submits, r->n_waits, r->wait_ns / 1000000,
        r->n_writes, r->n_reads, r->n_copies, r->n_wraps, r->n_shaders,
        r->n_compiled,
        r->shader_ns / 1000000, r->n_pipes, r->pipe_ns / 1000000,
        r->n_ahead);
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
        if (r->dead) {
            unsigned k;

            vkr_reap(r, true);
            g_ptr_array_free(r->dead, true);
            g_ptr_array_free(r->blocks, true);
            g_ptr_array_extend_and_steal(r->idle, r->reads);
            for (k = 0; k < r->idle->len; k++) {
                VkrRead *rd = g_ptr_array_index(r->idle, k);

                vkDestroyBuffer(r->dev, rd->buf, NULL);
                vkFreeMemory(r->dev, rd->mem, NULL);
                g_free(rd);
            }
            g_ptr_array_free(r->idle, true);
            g_mutex_clear(&r->lock);
        }
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
        vkFreeMemory(r->dev, r->ring_mem, NULL);
        vkFreeMemory(r->dev, r->tables_mem, NULL);
        vkDestroyCommandPool(r->dev, r->pool, NULL);
        vkDestroyDevice(r->dev, NULL);
    }
    if (r->inst) {
        vkDestroyInstance(r->inst, NULL);
    }
    g_free(r->cache_path);
    g_free(r->cache_dir);
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
    if (cache) {
        r->cache_dir = g_strdup(cache);
        r->cache_path = g_build_filename(cache, "pipelines.bin", NULL);
        if (g_file_get_contents(r->cache_path, &blob, &blob_len, NULL)) {
            cci.initialDataSize = blob_len;
            cci.pInitialData = blob;
        }
    }
    if (vkCreatePipelineLayout(r->dev, &plci, NULL, &r->layout) != VK_SUCCESS ||
        vkCreatePipelineCache(r->dev, &cci, NULL, &r->pcache) != VK_SUCCESS ||
        !vkr_buffer(r, (size_t)VKR_RING + VKR_KEEP,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false, &r->ring,
                    &r->ring_mem, &r->ring_map)) {
        g_free(blob);
        goto fail;
    }
    g_free(blob);
    r->pipes = g_hash_table_new_full(vkr_key_hash, vkr_key_equal, NULL,
                                     vkr_pipe_free);
    r->samps = g_array_new(false, false, sizeof(VkrSamp));
    r->blocks = g_ptr_array_new();
    r->dead = g_ptr_array_new();
    r->reads = g_ptr_array_new();
    r->idle = g_ptr_array_new();
    g_queue_init(&r->subs);
    g_mutex_init(&r->lock);
    glslang_initialize_process();
    vkr_begin(r);
    *err = NULL;
    return r;
fail:
    vkr_close(r);
    return NULL;
}
