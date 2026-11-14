/*
 * ATI Radeon HD 5770 (Juniper) 3D engine on the host GPU through OpenGL
 * (Windows, WGL).
 *
 * The kernels are those of the Metal backend, as GLSL compute shaders
 * (ati_hd5770_glsl.c): one invocation per pixel of a triangle's box
 * runs coverage, the program, the alpha test, blending and packing at
 * the target's tiled address. They address a copy of VRAM in one
 * shader storage buffer of VRAM size, at the same offsets.
 *
 * A host shadow holds what the copy holds, in 4 KiB pages. The first
 * request after a sync that reads a page (its textures, the rows of its
 * target) compares VRAM with the shadow and uploads the page if they
 * differ. sync() reads back the target rows drawn since the last sync
 * and writes to VRAM, and to the shadow, the bytes that differ from the
 * shadow, so memory the draws did not change is left as it is in VRAM.
 *
 * Surfaces in guest RAM: the buffer holds 4 KiB slots after VRAM, one per
 * guest RAM page a request touches, checked, uploaded and read back as
 * VRAM pages are; a surface's pages are found through a table of slots.
 * The slots are given out again once nothing queued uses them.
 *
 * Requests with depth and stencil planes are taken; the planes are
 * written surfaces like the target (checked, uploaded, read back, and in
 * the hazards). The levels of a mipmapped texture are read surfaces like
 * level 0.
 *
 * Triangles of consecutive draws with the same program run in one
 * dispatch, a work group per 64-pixel tile of a triangle's box; a memory
 * barrier goes before a triangle that can cover pixels an earlier one
 * wrote (the halves of one primitive cannot), writes bytes an earlier one
 * read, or reads bytes an earlier one wrote.
 *
 * GL runs on a thread of its own, which owns the context: draw() builds
 * the per-draw data and the commands on the caller's thread, and hands
 * them over in segments. A context of at least 4.4 core (compute
 * shaders, storage buffers and atomics are 4.3, an immutable buffer
 * the driver keeps in video memory 4.4) from a hardware driver is required;
 * a GL error or a lost context is reported once, and from then on every
 * draw is declined.
 *
 * Linked programs are kept on disk (%LOCALAPPDATA%\qemu-x86-opencore\
 * hd5770-gl), one file per SHA-256 of the IR hash, the GLSL text and the
 * renderer and version strings, as the driver's program binaries. A binary the driver
 * rejects is compiled from the text again and replaced.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include <math.h>
#include "qemu/thread.h"
#include "qemu/timer.h"
#include "qemu/error-report.h"
#include "ati_hd5770_gpu.h"

#define GL_TILE         64      /* invocations per work group */
#define GL_ACC          128     /* accesses tracked between barriers */
#define GL_SEG_DRAWS    64      /* draws per segment */
#define GL_MAX_TILES    (4 * 1024 * 1024)
#define GL_GROUPS       65535   /* work groups per dispatch call */
#define GL_POOL         (128u << 20)    /* slots for guest RAM pages */

/* ---- GL, spelled out: opengl32.dll exports 1.1 only ---- */

typedef unsigned int GLenum;
typedef unsigned int GLuint;
typedef unsigned int GLbitfield;
typedef unsigned char GLubyte;
typedef unsigned char GLboolean;
typedef int GLint;
typedef int GLsizei;
typedef char GLchar;
typedef int64_t GLint64;
typedef ptrdiff_t GLintptr;
typedef ptrdiff_t GLsizeiptr;

#define GL_NO_ERROR                         0
#define GL_CONTEXT_LOST                     0x0507
#define GL_RENDERER                         0x1F01
#define GL_VERSION                          0x1F02
#define GL_MAJOR_VERSION                    0x821B
#define GL_MINOR_VERSION                    0x821C
#define GL_COMPILE_STATUS                   0x8B81
#define GL_LINK_STATUS                      0x8B82
#define GL_COMPUTE_SHADER                   0x91B9
#define GL_SHADER_STORAGE_BUFFER            0x90D2
#define GL_MAX_SHADER_STORAGE_BLOCK_SIZE    0x90DE
#define GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT 0x90DF
#define GL_SHADER_STORAGE_BARRIER_BIT       0x00002000
#define GL_ALL_BARRIER_BITS                 0xFFFFFFFF
#define GL_STREAM_DRAW                      0x88E0
#define GL_STREAM_READ                      0x88E1
#define GL_COPY_READ_BUFFER                 0x8F36
#define GL_COPY_WRITE_BUFFER                0x8F37
#define GL_PROGRAM_BINARY_RETRIEVABLE_HINT  0x8257
#define GL_PROGRAM_BINARY_LENGTH            0x8741
#define GL_NUM_PROGRAM_BINARY_FORMATS       0x87FE
#define GL_TRUE                             1

#define HD_GL_PROCS(X) \
    X(const GLubyte *, glGetString, (GLenum)) \
    X(void, glGetIntegerv, (GLenum, GLint *)) \
    X(void, glGetInteger64v, (GLenum, GLint64 *)) \
    X(GLenum, glGetError, (void)) \
    X(void, glFlush, (void)) \
    X(void, glFinish, (void)) \
    X(GLuint, glCreateShader, (GLenum)) \
    X(void, glShaderSource, (GLuint, GLsizei, const GLchar *const *, \
                             const GLint *)) \
    X(void, glCompileShader, (GLuint)) \
    X(void, glGetShaderiv, (GLuint, GLenum, GLint *)) \
    X(void, glGetShaderInfoLog, (GLuint, GLsizei, GLsizei *, GLchar *)) \
    X(void, glDeleteShader, (GLuint)) \
    X(GLuint, glCreateProgram, (void)) \
    X(void, glAttachShader, (GLuint, GLuint)) \
    X(void, glLinkProgram, (GLuint)) \
    X(void, glGetProgramiv, (GLuint, GLenum, GLint *)) \
    X(void, glGetProgramInfoLog, (GLuint, GLsizei, GLsizei *, GLchar *)) \
    X(void, glDeleteProgram, (GLuint)) \
    X(void, glUseProgram, (GLuint)) \
    X(GLint, glGetUniformLocation, (GLuint, const GLchar *)) \
    X(void, glUniform1ui, (GLint, GLuint)) \
    X(void, glGenBuffers, (GLsizei, GLuint *)) \
    X(void, glDeleteBuffers, (GLsizei, const GLuint *)) \
    X(void, glBindBuffer, (GLenum, GLuint)) \
    X(void, glBufferData, (GLenum, GLsizeiptr, const void *, GLenum)) \
    X(void, glBufferSubData, (GLenum, GLintptr, GLsizeiptr, const void *)) \
    X(void, glGetBufferSubData, (GLenum, GLintptr, GLsizeiptr, void *)) \
    X(void, glBindBufferBase, (GLenum, GLuint, GLuint)) \
    X(void, glBindBufferRange, (GLenum, GLuint, GLuint, GLintptr, \
                                GLsizeiptr)) \
    X(void, glCopyBufferSubData, (GLenum, GLenum, GLintptr, GLintptr, \
                                  GLsizeiptr)) \
    X(void, glBufferStorage, (GLenum, GLsizeiptr, const void *, \
                              GLbitfield)) \
    X(void, glDispatchCompute, (GLuint, GLuint, GLuint)) \
    X(void, glMemoryBarrier, (GLbitfield)) \
    X(void, glProgramParameteri, (GLuint, GLenum, GLint)) \
    X(void, glGetProgramBinary, (GLuint, GLsizei, GLsizei *, GLenum *, \
                                 void *)) \
    X(void, glProgramBinary, (GLuint, GLenum, const void *, GLsizei))

#define HD_GL_PTR(ret, name, args) static ret (APIENTRY *p_##name) args;
HD_GL_PROCS(HD_GL_PTR)

typedef struct GlProc {
    const char *name;
    PROC *slot;
} GlProc;

#define HD_GL_ENT(ret, name, args) { #name, (PROC *)&p_##name },
static const GlProc gl_procs[] = {
    HD_GL_PROCS(HD_GL_ENT)
};

/*
 * wglGetProcAddress answers for entry points newer than 1.1 (some drivers
 * return 1, 2, 3 or -1 for none), GetProcAddress for the 1.1 exports.
 */
static bool gl_load(char *err, size_t n)
{
    HMODULE gl32 = LoadLibraryA("opengl32.dll");
    unsigned k;

    if (!gl32) {
        snprintf(err, n, "opengl32.dll could not be loaded");
        return false;
    }
    for (k = 0; k < ARRAY_SIZE(gl_procs); k++) {
        PROC p = wglGetProcAddress(gl_procs[k].name);
        uintptr_t v = (uintptr_t)p;

        if (v <= 3 || v == (uintptr_t)-1) {
            p = GetProcAddress(gl32, gl_procs[k].name);
        }
        if (!p) {
            snprintf(err, n, "host GL is missing %s()", gl_procs[k].name);
            return false;
        }
        *gl_procs[k].slot = p;
    }
    return true;
}

/* ---- the data the kernels read: the layouts of hd5770_glsl_lib ---- */

typedef struct GSurf {
    uint32_t voff, mode, bpe, row_bytes, mt_per_row, tile_bytes, mt_bytes;
    uint32_t pipes, banks, bankw, bankh, pib;
    uint32_t mt_w_sh, mt_h_sh, pipes_sh, bankw_sh, bankh_sh, pib_sh, hi_sh;
    uint32_t fast;
    uint32_t space, pt, pg0;
    uint32_t eoff[64];
} GSurf;

typedef struct GTex {
    uint32_t s, fmt, num, w, h, dim, sgn, comps, is_float, unorm8, samp0;
    uint32_t dst_sel[4];
    uint32_t bits[4];
    float border[4];
    uint32_t mip, last_lv, mip_filter, lev;
    float min_lod, max_lod, lod_bias;
    uint32_t lw[16], lh[16];
    uint32_t base_lv, levels, depth, pad;
} GTex;

typedef struct GVb {
    uint32_t id, off, len, pad;
    uint32_t res[8];
} GVb;

typedef struct GDraw {
    uint32_t rt, rt_fmt, rt_num, rt_comps, rt_unorm8, rt_float, rt_blend;
    uint32_t wmask, blend;
    uint32_t rt_bits[4];
    uint32_t rt_src[4];
    float blend_const[4];
    uint32_t alpha_test, alpha_func;
    float alpha_ref;
    uint32_t cliprule;
    int32_t cr[16];
    float pix_center;
    uint32_t persp;
    int32_t pos_gpr;
    uint32_t ninputs, flat, cfile, ntex, nvb;
    GVb vb[8];
    uint32_t zs, ss, db, zfmt, has_s, dctl, rctl, srm0, srm1, sclear;
    float zclear, zmin, zmax;
} GDraw;

typedef struct GTri {
    float x[3], y[3], w[3];
    float area;
    int32_t x0, y0, x1, y1;
    float z[3];
    uint32_t front;
} GTri;

QEMU_BUILD_BUG_ON(sizeof(GSurf) != 4 * 87);
QEMU_BUILD_BUG_ON(sizeof(GTex) != 4 * 66);
QEMU_BUILD_BUG_ON(sizeof(GDraw) != 4 * 158);
QEMU_BUILD_BUG_ON(sizeof(GTri) != 4 * 18);

/* storage buffers; B_UP: upload bytes (not bound); B_STAGE, B_RB: buffers */
enum {
    B_VRAM, B_SURFS, B_DRAWS, B_TEXS, B_TRIS, B_WORDS, B_TILES, B_UP,
    B_STAGE, B_RB, B_TAB, B_N
};
#define B_TAB_BINDING 7
#define B_SEG_N B_STAGE

/* ---- what the caller hands to the GL thread ---- */

typedef struct GlProg {
    int state;                  /* GLP_*, set by the thread that compiles */
    GLuint prog;
    GLint loc;                  /* tile0 */
    uint64_t hash;
    char *src;                  /* until compiled */
} GlProg;

enum { GLP_QUEUED, GLP_READY, GLP_FAILED };

enum { CMD_UPLOAD, CMD_DISPATCH };

typedef struct GlCmd {
    uint32_t kind;
    bool barrier;
    GlProg *prog;
    uint32_t tile0, ntiles;
    uint64_t off, len;
    uint64_t src_off;           /* CMD_UPLOAD: in the segment's B_UP */
} GlCmd;

/* draws between two hand-overs */
typedef struct GlSeg {
    GArray *a[B_N];             /* B_SURFS..B_UP: the buffers' data */
    GArray *cmds;
    uint32_t cfile;             /* words of the last constant file, or ~0 */
} GlSeg;

#define GL_PAGE_SH      12
#define GL_BLK_SH       16      /* shadow allocation unit */

enum { JOB_SEG, JOB_COMPILE, JOB_SYNC, JOB_QUIT };

typedef struct GlJob {
    int kind;
    GlSeg *seg;
    GlProg *prog;
    bool done;
} GlJob;

/* a pixel box one dispatch wrote, or bytes it read */
typedef struct GlAcc {
    uint64_t lo, hi, key;
    uint32_t mode, row_bytes, bpe;
    int32_t x0, y0, x1, y1;
    uint64_t draw;
    uint32_t prim;
} GlAcc;

enum {
    GLD_PROG, GLD_ROOM, GLD_SURF, GLD_DEAD, GLD_COMPILING, GLD_N,
};

typedef struct HD5770Gl {
    HD5770Gpu pub;
    uint8_t *vram_host;
    uint64_t vram_size;
    uint64_t size;              /* of the buffer: VRAM, then the slots */
    uint8_t *ram_host;          /* guest RAM, once ops->ram() gave it */
    uint64_t ram_size;

    /* the GL thread */
    QemuThread thread;
    QemuMutex lock;
    QemuCond cond;              /* a job queued or done */
    GQueue *jobs;
    QemuSemaphore up;
    HWND win;
    HDC dc;
    HGLRC ctx;
    /* programs are compiled on a thread of their own, in a shared context */
    QemuThread cthread;
    QemuCond ccond;
    GQueue *cq;                 /* GlProg */
    bool cquit, cthread_on;
    HWND cwin;
    HDC cdc;
    HGLRC cctx;
    GLuint buf[B_N];
    GLint align;
    char renderer[160];
    char *cdir;                 /* program binaries; NULL: none kept */
    char err[320];
    bool dead;                  /* a GL error: no more draws */
    bool dead_told;
    char dead_why[160];

    /* the caller's */
    GHashTable *progs;
    GlSeg *seg;
    uint8_t **blk;              /* the shadow, per 64 KiB, NULL: none yet */
    uint32_t *pchk;             /* per page: epoch it was last checked */
    uint8_t *pval;              /* per page: the copy holds the shadow */
    uint32_t epoch;
    GArray *ranges;             /* uint64_t pairs, VRAM written */
    GlAcc wr[GL_ACC], rd[GL_ACC];
    unsigned nwr, nrd;
    GHashTable *slot_of;        /* guest RAM page offset -> slot */
    uint32_t *slot_ram;         /* slot -> guest RAM page offset */
    uint32_t nslots, nused;     /* slot 0: pages not drawn */
    bool pool_busy;             /* slots written since the last sync */
    GlProg *bprog;              /* the batch: tiles from btile on */
    uint32_t btile;
    bool bbar;
    unsigned seg_draws;

    /* statistics */
    uint64_t draws, tris, declined[GLD_N], syncs, segs, dispatches;
    uint64_t barriers, up_bytes, chk_bytes, rb_bytes, wb_bytes, compiles;
    uint64_t st_bytes;
    uint64_t failed;
    uint64_t compile_ns, wait_ns, encode_ns, gpu_ns, rb_ns;
    uint64_t pb_hits, pb_rejected, pb_stored;
    uint64_t ram_draws, pool_resets, slots_max;
} HD5770Gl;

/* ---- the GL thread ---- */

static void seg_free(GlSeg *s);

static bool gl_check(HD5770Gl *g, const char *where)
{
    GLenum e = p_glGetError();

    if (e == GL_NO_ERROR || g->dead) {
        return e == GL_NO_ERROR;
    }
    snprintf(g->dead_why, sizeof(g->dead_why), "GL error 0x%04x%s %s", e,
             e == GL_CONTEXT_LOST ? " (context lost)" : "", where);
    qatomic_set(&g->dead, true);
    return false;
}

/* ---- program binaries on disk ---- */

#define PB_MAGIC        "HD5770GLPB\x01\x00"
#define PB_MAX          (64u << 20)

/* header of a program binary file; the binary follows */
typedef struct PbHead {
    char magic[12];
    uint8_t key[32];            /* SHA-256 of the IR hash, source, renderer */
    uint32_t fmt, len;
    uint64_t sum;               /* FNV-1a of the binary */
} PbHead;

static uint64_t pb_sum(const uint8_t *p, size_t n)
{
    uint64_t h = 0xcbf29ce484222325ull;

    while (n--) {
        h = (h ^ *p++) * 0x100000001b3ull;
    }
    return h;
}

static char *pb_path(HD5770Gl *g, const GlProg *p, uint8_t key[32])
{
    GChecksum *c = g_checksum_new(G_CHECKSUM_SHA256);
    gsize n = 32;
    char hex[33];
    unsigned k;

    g_checksum_update(c, (const guchar *)&p->hash, sizeof(p->hash));
    g_checksum_update(c, (const guchar *)p->src, strlen(p->src) + 1);
    g_checksum_update(c, (const guchar *)g->renderer, strlen(g->renderer) + 1);
    g_checksum_get_digest(c, key, &n);
    g_checksum_free(c);
    for (k = 0; k < 16; k++) {
        snprintf(hex + 2 * k, 3, "%02x", key[k]);
    }
    return g_strdup_printf("%s\\%s.bin", g->cdir, hex);
}

/* the program from its binary, or 0 */
static GLuint pb_load(HD5770Gl *g, const char *path, const uint8_t key[32])
{
    gchar *data = NULL;
    gsize n = 0;
    PbHead h;
    GLuint pr = 0;
    GLint ok = 0;

    if (!g_file_get_contents(path, &data, &n, NULL)) {
        return 0;
    }
    if (n < sizeof(h)) {
        goto bad;
    }
    memcpy(&h, data, sizeof(h));
    if (memcmp(h.magic, PB_MAGIC, sizeof(h.magic)) ||
        memcmp(h.key, key, 32) || h.len != n - sizeof(h) ||
        pb_sum((uint8_t *)data + sizeof(h), h.len) != h.sum) {
        goto bad;
    }
    pr = p_glCreateProgram();
    p_glProgramBinary(pr, h.fmt, data + sizeof(h), h.len);
    p_glGetProgramiv(pr, GL_LINK_STATUS, &ok);
    /* a binary the driver refuses may leave an error behind */
    while (p_glGetError() != GL_NO_ERROR) {
        ok = 0;
    }
    if (!ok) {
        p_glDeleteProgram(pr);
        pr = 0;
        goto bad;
    }
    g->pb_hits++;
    g_free(data);
    return pr;
bad:
    g->pb_rejected++;
    g_free(data);
    return 0;
}

static void pb_store(HD5770Gl *g, GLuint pr, const char *path,
                     const uint8_t key[32])
{
    GLint len = 0;
    GLsizei got = 0;
    GLenum fmt = 0;
    uint8_t *buf;
    PbHead h;

    p_glGetProgramiv(pr, GL_PROGRAM_BINARY_LENGTH, &len);
    if (len <= 0 || (uint32_t)len > PB_MAX) {
        return;
    }
    buf = g_malloc(sizeof(h) + len);
    p_glGetProgramBinary(pr, len, &got, &fmt, buf + sizeof(h));
    if (p_glGetError() == GL_NO_ERROR && got > 0 && got <= len) {
        memset(&h, 0, sizeof(h));
        memcpy(h.magic, PB_MAGIC, sizeof(h.magic));
        memcpy(h.key, key, 32);
        h.fmt = fmt;
        h.len = got;
        h.sum = pb_sum(buf + sizeof(h), got);
        memcpy(buf, &h, sizeof(h));
        if (g_mkdir_with_parents(g->cdir, 0700) == 0 &&
            g_file_set_contents(path, (gchar *)buf, sizeof(h) + got, NULL)) {
            g->pb_stored++;
        }
    }
    g_free(buf);
}

static void gl_compile(HD5770Gl *g, GlProg *p)
{
    GLuint sh, pr;
    GLint ok = 0, n = 0;
    const GLchar *src = p->src;
    int64_t t0 = get_clock();
    uint8_t key[32];
    char *path = g->cdir ? pb_path(g, p, key) : NULL;

    if (path) {
        pr = pb_load(g, path, key);
        if (pr) {
            p->prog = pr;
            p->loc = p_glGetUniformLocation(pr, "tile0");
            goto out;
        }
    }
    sh = p_glCreateShader(GL_COMPUTE_SHADER);
    p_glShaderSource(sh, 1, &src, NULL);
    p_glCompileShader(sh);
    p_glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[600] = "";

        p_glGetShaderInfoLog(sh, sizeof(log), NULL, log);
        snprintf(g->err, sizeof(g->err), "%s", log);
        p_glDeleteShader(sh);
        p->prog = 0;
        g->failed++;
        goto out;
    }
    pr = p_glCreateProgram();
    p_glAttachShader(pr, sh);
    if (path) {
        p_glProgramParameteri(pr, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, GL_TRUE);
    }
    p_glLinkProgram(pr);
    p_glDeleteShader(sh);
    p_glGetProgramiv(pr, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[600] = "";

        p_glGetProgramInfoLog(pr, sizeof(log), &n, log);
        snprintf(g->err, sizeof(g->err), "%s", log);
        p_glDeleteProgram(pr);
        p->prog = 0;
        g->failed++;
        goto out;
    }
    p->prog = pr;
    p->loc = p_glGetUniformLocation(pr, "tile0");
    if (path) {
        pb_store(g, pr, path, key);
    }
out:
    g_free(path);
    g->compiles++;
    g->compile_ns += get_clock() - t0;
}

/* compile, and publish the result to the caller's thread */
static void gl_build(HD5770Gl *g, GlProg *p)
{
    gl_compile(g, p);
    /* the program is complete before another context uses it */
    p_glFinish();
    if (!p->prog) {
        warn_report("ati-hd5770: gl: program %016" PRIx64 " not built: %.200s",
                    p->hash, g->err);
    }
    g_free(p->src);
    p->src = NULL;
    qatomic_store_release(&p->state, p->prog ? GLP_READY : GLP_FAILED);
}

static void *gl_cthread(void *opaque)
{
    HD5770Gl *g = opaque;

    wglMakeCurrent(g->cdc, g->cctx);
    qemu_mutex_lock(&g->lock);
    for (;;) {
        GlProg *p;

        while (g_queue_is_empty(g->cq) && !g->cquit) {
            qemu_cond_wait(&g->ccond, &g->lock);
        }
        if (g->cquit) {
            break;
        }
        p = g_queue_pop_head(g->cq);
        qemu_mutex_unlock(&g->lock);
        gl_build(g, p);
        qemu_mutex_lock(&g->lock);
    }
    qemu_mutex_unlock(&g->lock);
    wglMakeCurrent(NULL, NULL);
    return NULL;
}

static GLintptr gl_up_region(HD5770Gl *g, GArray *a, GLintptr *o)
{
    GLintptr at = *o;
    size_t n = (size_t)a->len * g_array_get_element_size(a);

    *o = ROUND_UP(at + MAX(n, 16), g->align);
    return at;
}

static void gl_run_seg(HD5770Gl *g, GlSeg *s)
{
    GLintptr at[B_SEG_N], o = 0;
    GlProg *cur = NULL;
    guint i, k;

    if (qatomic_read(&g->dead)) {
        return;
    }
    for (k = B_SURFS; k < B_SEG_N; k++) {
        at[k] = gl_up_region(g, s->a[k], &o);
    }
    p_glBindBuffer(GL_SHADER_STORAGE_BUFFER, g->buf[B_STAGE]);
    p_glBufferData(GL_SHADER_STORAGE_BUFFER, o, NULL, GL_STREAM_DRAW);
    g->st_bytes += o;
    for (k = B_SURFS; k < B_SEG_N; k++) {
        size_t n = (size_t)s->a[k]->len * g_array_get_element_size(s->a[k]);

        if (n) {
            p_glBufferSubData(GL_SHADER_STORAGE_BUFFER, at[k], n,
                              s->a[k]->data);
        }
        if (k != B_UP) {
            p_glBindBufferRange(GL_SHADER_STORAGE_BUFFER, k, g->buf[B_STAGE],
                                at[k], MAX(n, 16));
        }
    }
    p_glBindBufferBase(GL_SHADER_STORAGE_BUFFER, B_VRAM, g->buf[B_VRAM]);
    p_glBindBufferBase(GL_SHADER_STORAGE_BUFFER, B_TAB_BINDING, g->buf[B_TAB]);
    /* what an earlier segment wrote */
    p_glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    for (i = 0; i < s->cmds->len; i++) {
        GlCmd *c = &g_array_index(s->cmds, GlCmd, i);

        if (c->kind == CMD_UPLOAD) {
            p_glBindBuffer(GL_COPY_READ_BUFFER, g->buf[B_STAGE]);
            p_glBindBuffer(GL_COPY_WRITE_BUFFER, g->buf[B_VRAM]);
            p_glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
                                  at[B_UP] + c->src_off, c->off, c->len);
            continue;
        }
        if (c->barrier) {
            p_glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
        if (cur != c->prog) {
            cur = c->prog;
            p_glUseProgram(cur->prog);
        }
        for (k = 0; k < c->ntiles; k += GL_GROUPS) {
            p_glUniform1ui(cur->loc, c->tile0 + k);
            p_glDispatchCompute(MIN(c->ntiles - k, GL_GROUPS), 1, 1);
        }
    }
    p_glFlush();
    gl_check(g, "drawing");
}

/*
 * The rows drawn since the last sync, back to VRAM: the bytes that differ
 * from what was uploaded.
 */
static uint8_t *gl_shadow(HD5770Gl *g, uint64_t off)
{
    uint8_t **b = &g->blk[off >> GL_BLK_SH];

    if (!*b) {
        *b = g_malloc0(1u << GL_BLK_SH);
    }
    return *b + (off & ((1u << GL_BLK_SH) - 1));
}

/* host memory of buffer offset off: VRAM, or the guest RAM page of a slot */
static uint8_t *gl_host(HD5770Gl *g, uint64_t off)
{
    if (off < g->vram_size) {
        return g->vram_host + off;
    }
    off -= g->vram_size;
    return g->ram_host + g->slot_ram[off >> GL_PAGE_SH] +
           (off & ((1u << GL_PAGE_SH) - 1));
}

/* read back bytes, to memory and the shadow where they differ from it */
static void gl_merge(HD5770Gl *g, uint64_t lo, const uint8_t *rb, uint64_t n)
{
    uint8_t *dst = gl_host(g, lo), *ref = gl_shadow(g, lo);
    uint64_t k = 0;

    while (k < n) {
        uint64_t a, b, e;

        if (k + 8 <= n) {
            memcpy(&a, rb + k, 8);
            memcpy(&b, ref + k, 8);
            if (a == b) {
                k += 8;
                continue;
            }
        }
        for (e = MIN(k + 8, n); k < e; k++) {
            if (rb[k] != ref[k]) {
                dst[k] = rb[k];
                ref[k] = rb[k];
                g->wb_bytes++;
            }
        }
    }
}

static int gl_cmp_range(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;

    return x < y ? -1 : x > y;
}

/* the pages of the rows drawn since the last sync, back to VRAM */
static void gl_run_sync(HD5770Gl *g)
{
    GArray *rl = g_array_new(false, false, sizeof(uint64_t));
    uint64_t total = 0, at = 0;
    int64_t t0 = get_clock(), t1;
    uint8_t *tmp;
    guint i, k;

    /* page aligned, sorted, merged */
    for (i = 0; i + 1 < g->ranges->len; i += 2) {
        uint64_t r[2] = {
            g_array_index(g->ranges, uint64_t, i) &
            ~(uint64_t)((1u << GL_PAGE_SH) - 1),
            MIN(ROUND_UP(g_array_index(g->ranges, uint64_t, i + 1),
                         1u << GL_PAGE_SH), g->size),
        };

        g_array_append_vals(rl, r, 2);
    }
    qsort(rl->data, rl->len / 2, 2 * sizeof(uint64_t), gl_cmp_range);
    for (i = 0, k = 0; i + 1 < rl->len; i += 2) {
        uint64_t lo = g_array_index(rl, uint64_t, i);
        uint64_t hi = g_array_index(rl, uint64_t, i + 1);

        if (k && lo <= g_array_index(rl, uint64_t, k - 1)) {
            g_array_index(rl, uint64_t, k - 1) =
                MAX(hi, g_array_index(rl, uint64_t, k - 1));
            continue;
        }
        g_array_index(rl, uint64_t, k) = lo;
        g_array_index(rl, uint64_t, k + 1) = hi;
        k += 2;
    }
    g_array_set_size(rl, k);
    for (i = 0; i + 1 < rl->len; i += 2) {
        total += g_array_index(rl, uint64_t, i + 1) -
                 g_array_index(rl, uint64_t, i);
    }
    p_glMemoryBarrier(GL_ALL_BARRIER_BITS);
    if (total) {
        p_glBindBuffer(GL_COPY_READ_BUFFER, g->buf[B_VRAM]);
        p_glBindBuffer(GL_COPY_WRITE_BUFFER, g->buf[B_RB]);
        p_glBufferData(GL_COPY_WRITE_BUFFER, total, NULL, GL_STREAM_READ);
        for (i = 0; i + 1 < rl->len; i += 2) {
            uint64_t lo = g_array_index(rl, uint64_t, i);
            uint64_t hi = g_array_index(rl, uint64_t, i + 1);

            p_glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
                                  lo, at, hi - lo);
            at += hi - lo;
        }
    }
    p_glFinish();
    t1 = get_clock();
    g->gpu_ns += t1 - t0;
    if (total) {
        tmp = g_malloc(total);
        p_glGetBufferSubData(GL_COPY_WRITE_BUFFER, 0, total, tmp);
        if (gl_check(g, "reading back")) {
            g->rb_bytes += total;
            for (at = 0, i = 0; i + 1 < rl->len; i += 2) {
                uint64_t lo = g_array_index(rl, uint64_t, i);
                uint64_t hi = g_array_index(rl, uint64_t, i + 1);

                while (lo < hi) {
                    /* slots: page by page */
                    uint64_t end = MIN(hi, (lo | ((1u << (lo < g->vram_size ?
                                       GL_BLK_SH : GL_PAGE_SH)) - 1)) + 1);

                    gl_merge(g, lo, tmp + at, end - lo);
                    at += end - lo;
                    lo = end;
                }
            }
        }
        g_free(tmp);
    }
    g->rb_ns += get_clock() - t1;
    g_array_free(rl, true);
}

static void gl_close_ctx(HD5770Gl *g)
{
    if (g->cctx) {
        wglDeleteContext(g->cctx);
        g->cctx = NULL;
    }
    if (g->cdc) {
        ReleaseDC(g->cwin, g->cdc);
        g->cdc = NULL;
    }
    if (g->cwin) {
        DestroyWindow(g->cwin);
        g->cwin = NULL;
    }
    if (g->ctx) {
        wglMakeCurrent(NULL, NULL);
        wglDeleteContext(g->ctx);
        g->ctx = NULL;
    }
    if (g->dc) {
        ReleaseDC(g->win, g->dc);
        g->dc = NULL;
    }
    if (g->win) {
        DestroyWindow(g->win);
        g->win = NULL;
    }
}

#define WGL_CONTEXT_MAJOR_VERSION_ARB    0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB    0x2092
#define WGL_CONTEXT_PROFILE_MASK_ARB     0x9126
#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB 0x00000001

/* a renderer that draws on the CPU is no faster than the CPU path */
static bool gl_soft(const char *r)
{
    static const char *const soft[] = {
        "GDI Generic", "llvmpipe", "softpipe", "SwiftShader", "Microsoft",
    };
    unsigned k;

    for (k = 0; k < ARRAY_SIZE(soft); k++) {
        if (strstr(r, soft[k])) {
            return true;
        }
    }
    return false;
}

/*
 * An invisible 1x1 window (WGL has no windowless context), the driver's
 * default context to reach wglCreateContextAttribsARB, then a core
 * context of 4.6 down to 4.4.
 */
static bool gl_open_ctx(HD5770Gl *g)
{
    static const char cls[] = "qemu-ati-hd5770-gl";
    static const int ver[][2] = { { 4, 6 }, { 4, 5 }, { 4, 4 } };
    union {
        PROC any;
        HGLRC (APIENTRY *mk)(HDC, HGLRC, const int *);
    } arb = { NULL };
    PIXELFORMATDESCRIPTOR pfd;
    WNDCLASSA wc;
    HGLRC boot;
    char legacy[200] = "?";
    GLint major = 0, minor = 0;
    GLint64 maxsz = 0;
    int fmt;
    unsigned k;

    memset(&wc, 0, sizeof(wc));
    wc.lpfnWndProc = DefWindowProcA;
    wc.hInstance = GetModuleHandleA(NULL);
    wc.lpszClassName = cls;
    if (!RegisterClassA(&wc) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
        snprintf(g->err, sizeof(g->err), "cannot register a GL window class");
        return false;
    }
    g->win = CreateWindowExA(0, cls, cls, WS_POPUP, 0, 0, 1, 1, NULL, NULL,
                             GetModuleHandleA(NULL), NULL);
    g->dc = g->win ? GetDC(g->win) : NULL;
    if (!g->dc) {
        snprintf(g->err, sizeof(g->err), "cannot create a GL window");
        return false;
    }
    memset(&pfd, 0, sizeof(pfd));
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 24;
    pfd.cAlphaBits = 8;
    fmt = ChoosePixelFormat(g->dc, &pfd);
    if (!fmt || !SetPixelFormat(g->dc, fmt, &pfd)) {
        snprintf(g->err, sizeof(g->err), "no GL pixel format on this host");
        return false;
    }
    boot = wglCreateContext(g->dc);
    if (!boot || !wglMakeCurrent(g->dc, boot)) {
        snprintf(g->err, sizeof(g->err), "wglCreateContext failed");
        if (boot) {
            wglDeleteContext(boot);
        }
        return false;
    }
    {
        const GLubyte *(APIENTRY *gs)(GLenum) =
            (const GLubyte *(APIENTRY *)(GLenum))(void (*)(void))
            GetProcAddress(GetModuleHandleA("opengl32.dll"), "glGetString");

        if (gs && gs(GL_RENDERER)) {
            snprintf(legacy, sizeof(legacy), "%s, %s",
                     (const char *)gs(GL_RENDERER),
                     gs(GL_VERSION) ? (const char *)gs(GL_VERSION) : "?");
        }
    }
    arb.any = wglGetProcAddress("wglCreateContextAttribsARB");
    wglMakeCurrent(NULL, NULL);
    wglDeleteContext(boot);
    for (k = 0; arb.mk && k < ARRAY_SIZE(ver) && !g->ctx; k++) {
        const int attr[] = {
            WGL_CONTEXT_MAJOR_VERSION_ARB, ver[k][0],
            WGL_CONTEXT_MINOR_VERSION_ARB, ver[k][1],
            WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
            0
        };

        g->ctx = arb.mk(g->dc, NULL, attr);
        if (g->ctx) {
            /* the compile thread's, on a window of its own */
            g->cwin = CreateWindowExA(0, cls, cls, WS_POPUP, 0, 0, 1, 1, NULL,
                                      NULL, GetModuleHandleA(NULL), NULL);
            g->cdc = g->cwin ? GetDC(g->cwin) : NULL;
            if (g->cdc && SetPixelFormat(g->cdc, fmt, &pfd)) {
                g->cctx = arb.mk(g->cdc, g->ctx, attr);
            }
        }
    }
    if (!g->ctx) {
        snprintf(g->err, sizeof(g->err),
                 "host GL has no 4.4 core context (renderer %s)", legacy);
        return false;
    }
    if (!wglMakeCurrent(g->dc, g->ctx)) {
        snprintf(g->err, sizeof(g->err), "cannot make the GL context current");
        return false;
    }
    if (!gl_load(g->err, sizeof(g->err))) {
        return false;
    }
    snprintf(g->renderer, sizeof(g->renderer), "%s, %s",
             (const char *)p_glGetString(GL_RENDERER),
             (const char *)p_glGetString(GL_VERSION));
    p_glGetIntegerv(GL_MAJOR_VERSION, &major);
    p_glGetIntegerv(GL_MINOR_VERSION, &minor);
    if (major * 10 + minor < 44) {
        snprintf(g->err, sizeof(g->err), "renderer %s is below GL 4.4",
                 g->renderer);
        return false;
    }
    if (gl_soft(g->renderer)) {
        snprintf(g->err, sizeof(g->err), "renderer %s draws on the CPU",
                 g->renderer);
        return false;
    }
    p_glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxsz);
    g->size = g->vram_size;
    if ((uint64_t)maxsz >= g->vram_size + GL_POOL &&
        g->vram_size + GL_POOL <= UINT32_MAX) {
        g->size += GL_POOL;
    }
    if ((uint64_t)maxsz < g->vram_size) {
        snprintf(g->err, sizeof(g->err), "renderer %s: storage buffers of "
                 "%" PRId64 " bytes, VRAM is larger", g->renderer, maxsz);
        return false;
    }
    p_glGetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &g->align);
    g->align = MAX(g->align, 16);
    p_glGenBuffers(B_N, g->buf);
    p_glBindBuffer(GL_SHADER_STORAGE_BUFFER, g->buf[B_VRAM]);
    p_glBufferStorage(GL_SHADER_STORAGE_BUFFER, g->size, NULL, 0);
    if (p_glGetError() != GL_NO_ERROR) {
        snprintf(g->err, sizeof(g->err), "renderer %s: no storage buffer of "
                 "VRAM size", g->renderer);
        return false;
    }
    {
        size_t n = hd5770_glsl_tables(NULL, 0);
        uint32_t *t = g_new(uint32_t, n);

        hd5770_glsl_tables(t, n);
        p_glBindBuffer(GL_SHADER_STORAGE_BUFFER, g->buf[B_TAB]);
        p_glBufferStorage(GL_SHADER_STORAGE_BUFFER, n * 4, t, 0);
        g_free(t);
    }
    /* the library alone, so that a compiler that cannot take it says so */
    {
        GlProg t = { .src = hd5770_glsl_lib_only() };

        gl_compile(g, &t);
        g_free(t.src);
        if (!t.prog) {
            char log[320];

            snprintf(log, sizeof(log), "%s", g->err);
            snprintf(g->err, sizeof(g->err), "renderer %.120s cannot compile "
                     "the shaders: %.150s", g->renderer, log);
            return false;
        }
        p_glDeleteProgram(t.prog);
        g->compiles = g->failed = 0;
        g->compile_ns = 0;
    }
    {
        GLint nfmt = 0;
        const char *base = g_getenv("LOCALAPPDATA");

        p_glGetIntegerv(GL_NUM_PROGRAM_BINARY_FORMATS, &nfmt);
        if (nfmt > 0 && base && *base) {
            g->cdir = g_strdup_printf("%s\\qemu-x86-opencore\\hd5770-gl",
                                      base);
        }
    }
    return true;
}

static void *gl_thread(void *opaque)
{
    HD5770Gl *g = opaque;
    bool ok = gl_open_ctx(g);
    bool quit = false;

    if (!ok) {
        gl_close_ctx(g);
        qemu_sem_post(&g->up);
        return NULL;
    }
    g->err[0] = 0;
    qemu_sem_post(&g->up);
    while (!quit) {
        GlJob *j;

        qemu_mutex_lock(&g->lock);
        while (g_queue_is_empty(g->jobs)) {
            qemu_cond_wait(&g->cond, &g->lock);
        }
        j = g_queue_pop_head(g->jobs);
        qemu_mutex_unlock(&g->lock);

        switch (j->kind) {
        case JOB_SEG:
            /* nobody waits for a segment: it is the thread's */
            gl_run_seg(g, j->seg);
            seg_free(j->seg);
            g_free(j);
            continue;
        case JOB_COMPILE:
            gl_build(g, j->prog);
            gl_check(g, "compiling a program");
            break;
        case JOB_SYNC:
            if (!qatomic_read(&g->dead)) {
                gl_run_sync(g);
            }
            break;
        default:
            if (g->buf[0]) {
                p_glDeleteBuffers(B_N, g->buf);
            }
            quit = true;
            break;
        }
        qemu_mutex_lock(&g->lock);
        j->done = true;
        qemu_cond_broadcast(&g->cond);
        qemu_mutex_unlock(&g->lock);
    }
    gl_close_ctx(g);
    return NULL;
}

/* queue a job; wait for it if it is on the stack */
static void gl_job(HD5770Gl *g, GlJob *j, bool wait)
{
    qemu_mutex_lock(&g->lock);
    g_queue_push_tail(g->jobs, j);
    qemu_cond_broadcast(&g->cond);
    while (wait && !j->done) {
        qemu_cond_wait(&g->cond, &g->lock);
    }
    qemu_mutex_unlock(&g->lock);
}

/* ---- the caller's side ---- */

static void seg_free(GlSeg *s)
{
    unsigned k;

    for (k = B_SURFS; k < B_SEG_N; k++) {
        g_array_free(s->a[k], true);
    }
    g_array_free(s->cmds, true);
    g_free(s);
}

static GlSeg *seg_new(void)
{
    static const size_t esz[B_N] = {
        [B_SURFS] = sizeof(GSurf), [B_DRAWS] = sizeof(GDraw),
        [B_TEXS] = sizeof(GTex), [B_TRIS] = sizeof(GTri),
        [B_WORDS] = 4, [B_TILES] = 4, [B_UP] = 1,
    };
    GlSeg *s = g_new0(GlSeg, 1);
    unsigned k;

    for (k = B_SURFS; k < B_SEG_N; k++) {
        s->a[k] = g_array_new(false, false, esz[k]);
    }
    s->cmds = g_array_new(false, false, sizeof(GlCmd));
    s->cfile = ~0u;
    return s;
}

/* the batch as one dispatch */
static void gl_flush(HD5770Gl *g)
{
    GArray *t = g->seg->a[B_TILES];
    GlCmd c = { .kind = CMD_DISPATCH };

    if (!g->bprog || t->len / 8 == g->btile) {
        return;
    }
    c.barrier = g->bbar;
    c.prog = g->bprog;
    c.tile0 = g->btile;
    c.ntiles = t->len / 8 - g->btile;
    g_array_append_val(g->seg->cmds, c);
    g->btile = t->len / 8;
    g->bbar = false;
    g->dispatches++;
}

/* hand the segment to the GL thread */
static void gl_submit(HD5770Gl *g)
{
    GlJob *j;

    gl_flush(g);
    if (!g->seg->cmds->len) {
        return;
    }
    j = g_new0(GlJob, 1);
    j->kind = JOB_SEG;
    j->seg = g->seg;
    gl_job(g, j, false);
    g->seg = seg_new();
    g->bprog = NULL;
    g->btile = 0;
    g->bbar = false;
    g->nwr = g->nrd = 0;
    g->seg_draws = 0;
    g->segs++;
}

static void gl_sync(void *opaque)
{
    HD5770Gl *g = opaque;
    GlJob j = { .kind = JOB_SYNC };
    int64_t t0;
    guint i;

    gl_submit(g);
    if (!g->ranges->len) {
        g->epoch++;
        return;
    }
    t0 = get_clock();
    gl_job(g, &j, true);
    g->wait_ns += get_clock() - t0;
    g->syncs++;
    for (i = 0; i + 1 < g->ranges->len; i += 2) {
        uint64_t lo = g_array_index(g->ranges, uint64_t, i);
        uint64_t hi = g_array_index(g->ranges, uint64_t, i + 1);

        if (lo < g->vram_size) {
            g->pub.done(g->pub.opaque, lo, MIN(hi, g->vram_size));
            lo = g->vram_size;
        }
        for (; lo < hi && g->pub.done_ram;
             lo = (lo | ((1u << GL_PAGE_SH) - 1)) + 1) {
            uint64_t r = gl_host(g, lo) - g->ram_host;
            uint64_t e = MIN(hi, (lo | ((1u << GL_PAGE_SH) - 1)) + 1);

            g->pub.done_ram(g->pub.opaque, r, r + (e - lo));
        }
    }
    g_array_set_size(g->ranges, 0);
    g->pool_busy = false;
    /* VRAM may change now: every page is checked again */
    g->epoch++;
    if (qatomic_read(&g->dead) && !g->dead_told) {
        g->dead_told = true;
        warn_report("ati-hd5770: gl: %s; drawing on the CPU", g->dead_why);
    }
}

/*
 * The program of a request; queued for the compile thread the first time
 * (the request goes to the CPU until it is built), compiled on the GL
 * thread meanwhile if there is no compile thread.
 */
static GlProg *gl_prog(HD5770Gl *g, const HD5770IR *ir, uint32_t mipmask)
{
    /* a program with mipmapped units is its own variant */
    uint64_t hash = ir->hash ^ (mipmask * 0x9e3779b97f4a7c15ull);
    GlProg *p = g_hash_table_lookup(g->progs, &hash);
    GlJob j = { .kind = JOB_COMPILE };

    if (p) {
        return p;
    }
    p = g_new0(GlProg, 1);
    p->hash = hash;
    g_hash_table_insert(g->progs, g_memdup2(&hash, sizeof(hash)), p);
    p->src = hd5770_glsl_program_mip(ir, mipmask);
    if (!p->src) {
        p->state = GLP_FAILED;
        return p;
    }
    if (g->cthread_on) {
        qemu_mutex_lock(&g->lock);
        g_queue_push_tail(g->cq, p);
        qemu_cond_signal(&g->ccond);
        qemu_mutex_unlock(&g->lock);
        return p;
    }
    j.prog = p;
    gl_job(g, &j, true);
    return p;
}

static void gl_upload(HD5770Gl *g, uint64_t lo, uint64_t hi)
{
    GlCmd c = { .kind = CMD_UPLOAD, .off = lo, .len = hi - lo };
    GArray *up = g->seg->a[B_UP];

    c.src_off = up->len;
    g_array_append_vals(up, gl_shadow(g, lo), hi - lo);
    g_array_append_val(g->seg->cmds, c);
    g->up_bytes += hi - lo;
}

/*
 * VRAM bytes [lo, hi) in the copy: the pages not checked since the last
 * sync are compared with the shadow, and uploaded if they differ.
 */
static void gl_need(HD5770Gl *g, uint64_t lo, uint64_t hi)
{
    uint64_t p, p1 = (hi - 1) >> GL_PAGE_SH, run = 0;
    bool in = false;

    for (p = lo >> GL_PAGE_SH; p <= p1; p++) {
        uint64_t off = p << GL_PAGE_SH;
        uint8_t *sh, *host = gl_host(g, off);
        bool up;

        if (in && (g->pchk[p] == g->epoch ||
                   !(off & ((1u << GL_BLK_SH) - 1)))) {
            gl_upload(g, run, off);
            in = false;
        }
        if (g->pchk[p] == g->epoch) {
            continue;
        }
        g->pchk[p] = g->epoch;
        sh = gl_shadow(g, off);
        g->chk_bytes += 1u << GL_PAGE_SH;
        up = !g->pval[p] || memcmp(sh, host, 1u << GL_PAGE_SH);
        if (up) {
            memcpy(sh, host, 1u << GL_PAGE_SH);
            g->pval[p] = 1;
            if (!in) {
                run = off;
                in = true;
            }
        } else if (in) {
            gl_upload(g, run, off);
            in = false;
        }
    }
    if (in) {
        gl_upload(g, run, (p1 + 1) << GL_PAGE_SH);
    }
}

static void gl_range(HD5770Gl *g, uint64_t lo, uint64_t hi)
{
    guint n = g->ranges->len;

    if (n && g_array_index(g->ranges, uint64_t, n - 2) <= hi &&
        lo <= g_array_index(g->ranges, uint64_t, n - 1)) {
        g_array_index(g->ranges, uint64_t, n - 2) =
            MIN(lo, g_array_index(g->ranges, uint64_t, n - 2));
        g_array_index(g->ranges, uint64_t, n - 1) =
            MAX(hi, g_array_index(g->ranges, uint64_t, n - 1));
        return;
    }
    g_array_append_val(g->ranges, lo);
    g_array_append_val(g->ranges, hi);
}

static bool gl_overlap(uint64_t a0, uint64_t a1, uint64_t b0, uint64_t b1)
{
    return a0 < b1 && b0 < a1;
}

/* the buffer bytes a surface may occupy: in guest RAM, any slot */
static void gl_span(HD5770Gl *g, const HD5770GpuSurf *s, uint64_t *lo,
                    uint64_t *hi)
{
    if (s->space) {
        *lo = g->vram_size;
        *hi = g->size;
    } else {
        *lo = s->voff;
        *hi = s->voff + hd5770_surf_size(&s->s);
    }
}

/* the bytes a texture reads: level 0 and its mip chain */
static void gl_tex_span(HD5770Gl *g, const HD5770GpuTex *t, uint64_t *lo,
                        uint64_t *hi)
{
    uint64_t l0, h0;
    unsigned l;

    gl_span(g, &t->surf, lo, hi);
    for (l = 1; t->mip && l <= t->last_lv; l++) {
        gl_span(g, &t->lv[l], &l0, &h0);
        *lo = MIN(*lo, l0);
        *hi = MAX(*hi, h0);
    }
}

/* where a surface starts, VRAM and guest RAM apart */
static uint64_t gl_key(const HD5770GpuSurf *s)
{
    return s->space ? (1ull << 63) | ((uint64_t)s->pages[0] + s->pg0)
                    : s->voff;
}

/* the surfaces a request writes: the target, the depth and stencil planes */
static unsigned gl_writes(const HD5770GpuDraw *d, const HD5770GpuSurf **ws)
{
    unsigned n = 0;

    ws[n++] = &d->rt;
    if (d->db && d->zfmt) {
        ws[n++] = &d->zs;
    }
    if (d->db && d->has_s) {
        ws[n++] = &d->ss;
    }
    return n;
}

/* the bytes rows y0..y1 of a written surface take */
static void gl_wrows(HD5770Gl *g, const HD5770GpuSurf *s, int32_t y0,
                     int32_t y1, uint64_t *lo, uint64_t *hi)
{
    if (s->space) {
        gl_span(g, s, lo, hi);
        return;
    }
    hd5770_surf_rows(&s->s, y0, y1 + 1, lo, hi);
    *lo = *lo - s->s.base + s->voff;
    *hi = *hi - s->s.base + s->voff;
}

/*
 * A barrier before a dispatch writing the pixel box of triangle t in the
 * request's written surfaces, if an earlier one since the last barrier
 * conflicts. reads: the draw's textures, checked once per draw (first).
 */
static void gl_hazard(HD5770Gl *g, const HD5770GpuDraw *d,
                      const HD5770GpuTri *t, bool first)
{
    const HD5770GpuSurf *ws[3];
    unsigned nw = gl_writes(d, ws), i, j, k;
    bool bar = g->nwr + nw > GL_ACC || g->nrd + d->ntex > GL_ACC;
    uint64_t lo[3], hi[3], key[3], tlo, thi;
    float pc = d->pix_center, mg = 1.0f / 16;
    int32_t x0 = ceilf(MIN(t->x[0], MIN(t->x[1], t->x[2])) - pc - mg);
    int32_t x1 = floorf(MAX(t->x[0], MAX(t->x[1], t->x[2])) - pc + mg);
    int32_t y0 = ceilf(MIN(t->y[0], MIN(t->y[1], t->y[2])) - pc - mg);
    int32_t y1 = floorf(MAX(t->y[0], MAX(t->y[1], t->y[2])) - pc + mg);

    for (j = 0; j < nw; j++) {
        gl_wrows(g, ws[j], t->y0, t->y1, &lo[j], &hi[j]);
        key[j] = gl_key(ws[j]);
    }
    for (i = 0; !bar && i < g->nwr; i++) {
        const GlAcc *a = &g->wr[i];

        for (j = 0; !bar && j < nw; j++) {
            const HD5770GpuSurf *w = ws[j];

            if (a->key == key[j] && a->mode == w->s.mode &&
                a->row_bytes == w->s.row_bytes && a->bpe == w->s.bpe) {
                bar = a->x0 <= x1 && x0 <= a->x1 && a->y0 <= y1 &&
                      y0 <= a->y1 &&
                      !(a->draw == g->draws && a->prim == t->prim);
            } else {
                bar = gl_overlap(a->lo, a->hi, lo[j], hi[j]);
            }
        }
        for (k = 0; first && !bar && k < d->ntex; k++) {
            gl_tex_span(g, &d->tex[k], &tlo, &thi);
            bar = gl_overlap(a->lo, a->hi, tlo, thi);
        }
    }
    for (i = 0; !bar && i < g->nrd; i++) {
        for (j = 0; !bar && j < nw; j++) {
            bar = gl_overlap(g->rd[i].lo, g->rd[i].hi, lo[j], hi[j]);
        }
    }
    if (bar) {
        gl_flush(g);
        g->bbar = true;
        g->nwr = g->nrd = 0;
        g->barriers++;
    }
    if (first || bar) {
        for (k = 0; k < d->ntex; k++) {
            gl_tex_span(g, &d->tex[k], &g->rd[g->nrd].lo, &g->rd[g->nrd].hi);
            g->nrd++;
        }
    }
    for (j = 0; j < nw; j++) {
        g->wr[g->nwr] = (GlAcc) {
            .lo = lo[j], .hi = hi[j], .key = key[j], .mode = ws[j]->s.mode,
            .row_bytes = ws[j]->s.row_bytes, .bpe = ws[j]->s.bpe,
            .x0 = x0, .y0 = y0, .x1 = x1, .y1 = y1,
            .draw = g->draws, .prim = t->prim,
        };
        g->nwr++;
    }
}

/* pt: the word offset of the slot table of a surface in guest RAM */
static uint32_t gl_surf(HD5770Gl *g, const HD5770GpuSurf *s, uint32_t pt)
{
    const HD5770Surf *f = &s->s;
    GArray *a = g->seg->a[B_SURFS];
    GSurf gs;

    memset(&gs, 0, sizeof(gs));
    gs.voff = s->voff;
    gs.mode = f->mode;
    gs.bpe = f->bpe;
    gs.row_bytes = f->row_bytes;
    gs.mt_per_row = f->mt_per_row;
    gs.tile_bytes = f->tile_bytes;
    gs.mt_bytes = f->mt_bytes;
    gs.pipes = f->pipes;
    gs.banks = f->banks;
    gs.bankw = f->bankw;
    gs.bankh = f->bankh;
    gs.pib = f->pipe_interleave;
    gs.mt_w_sh = f->mt_w_sh;
    gs.mt_h_sh = f->mt_h_sh;
    gs.pipes_sh = f->pipes_sh;
    gs.bankw_sh = f->bankw_sh;
    gs.bankh_sh = f->bankh_sh;
    gs.pib_sh = f->pib_sh;
    gs.hi_sh = f->hi_sh;
    gs.fast = f->fast;
    if (s->space) {
        gs.voff = g->vram_size;
        gs.space = 1;
        gs.pt = pt;
        gs.pg0 = s->pg0;
    }
    if (f->mode >= HD5770_ARRAY_1D_TILED_THIN1) {
        memcpy(gs.eoff, f->eoff, sizeof(gs.eoff));
    }
    g_array_append_val(a, gs);
    return a->len - 1;
}

/* ---- guest RAM pages in slots ---- */

/* every slot free again; nothing queued writes one */
static void gl_pool_reset(HD5770Gl *g)
{
    uint64_t p0 = g->vram_size >> GL_PAGE_SH;

    g_hash_table_remove_all(g->slot_of);
    g->nused = 1;
    memset(g->pval + p0, 0, g->nslots);
    memset(g->pchk + p0, 0, g->nslots * sizeof(*g->pchk));
    g->pool_resets++;
}

static uint32_t gl_slot_find(HD5770Gl *g, uint32_t ram_off)
{
    return GPOINTER_TO_UINT(g_hash_table_lookup(g->slot_of,
                            GUINT_TO_POINTER(ram_off >> GL_PAGE_SH)));
}

static uint32_t gl_slot(HD5770Gl *g, uint32_t ram_off)
{
    uint32_t sl = gl_slot_find(g, ram_off);

    if (!sl) {
        sl = g->nused++;
        g->slot_ram[sl] = ram_off;
        g_hash_table_insert(g->slot_of, GUINT_TO_POINTER(ram_off >> GL_PAGE_SH),
                            GUINT_TO_POINTER(sl));
        g->slots_max = MAX(g->slots_max, g->nused);
    }
    return sl;
}

/* pages [i0, i1] of a surface in guest RAM without a slot (upper bound) */
static uint32_t gl_slots_new(HD5770Gl *g, const HD5770GpuSurf *s, uint32_t i0,
                             uint32_t i1)
{
    uint32_t i, n = 0;

    for (i = i0; i <= i1 && i < s->npages; i++) {
        n += !gl_slot_find(g, s->pages[i]);
    }
    return n;
}

/*
 * The slot table of a surface in guest RAM, in the words: pages [i0, i1]
 * get slots, in the copy before the dispatch (and to be read back if the
 * request writes them), the others slot 0.
 */
static uint32_t gl_ram_surf(HD5770Gl *g, const HD5770GpuSurf *s, uint32_t i0,
                            uint32_t i1, bool wr)
{
    GArray *a = g->seg->a[B_WORDS];
    uint32_t at = a->len, i;

    g_array_set_size(a, at + MAX(s->npages, 1));
    g_array_index(a, uint32_t, at) = 0;
    for (i = 0; i < s->npages; i++) {
        uint32_t sl = 0;

        if (i >= i0 && i <= i1) {
            uint64_t off;

            sl = gl_slot(g, s->pages[i]);
            off = g->vram_size + ((uint64_t)sl << GL_PAGE_SH);
            gl_need(g, off, off + (1u << GL_PAGE_SH));
            if (wr) {
                gl_range(g, off, off + (1u << GL_PAGE_SH));
            }
        }
        g_array_index(a, uint32_t, at + i) = sl;
    }
    return at;
}

/* the pages of the target rows y0..y1 */
static void gl_rt_pages(const HD5770GpuDraw *d, int y0, int y1, uint32_t *i0,
                        uint32_t *i1)
{
    uint64_t lo, hi;

    hd5770_surf_rows(&d->rt.s, y0, y1 + 1, &lo, &hi);
    *i0 = (d->rt.pg0 + (lo - d->rt.s.base)) >> GL_PAGE_SH;
    *i1 = (d->rt.pg0 + (hi - d->rt.s.base) - 1) >> GL_PAGE_SH;
}

static uint32_t gl_words(HD5770Gl *g, const void *p, size_t n)
{
    GArray *a = g->seg->a[B_WORDS];
    uint32_t at = a->len;

    g_array_set_size(a, at + DIV_ROUND_UP(MAX(n, 4), 4));
    memcpy(&g_array_index(a, uint32_t, at), p, n);
    return at;
}

static bool gl_draw(void *opaque, const HD5770GpuDraw *d)
{
    HD5770Gl *g = opaque;
    GlProg *p;
    GArray *tiles;
    GDraw u;
    uint64_t lo, hi;
    size_t prm_tri = (size_t)d->ninputs * 12 * sizeof(float), ntiles = 0;
    uint32_t di, tex0, kvo, prm, ri0 = 1, ri1 = 0, rpt = 0;
    uint32_t tpt[HD5770_GPU_TEX] = { 0 };
    int64_t t0;
    int miny = INT_MAX, maxy = -1;
    unsigned i, k;
    bool first = true, ram = false;
    uint32_t mipmask = 0;

    if (qatomic_read(&g->dead)) {
        g->declined[GLD_DEAD]++;
        return false;
    }
    for (i = 0; i < d->ntex; i++) {
        mipmask |= (uint32_t)d->tex[i].mip << i;
    }
    p = gl_prog(g, d->ir, mipmask);
    switch (qatomic_load_acquire(&p->state)) {
    case GLP_QUEUED:
        g->declined[GLD_COMPILING]++;
        return false;
    case GLP_FAILED:
        g->declined[GLD_PROG]++;
        return false;
    }
    if ((d->rt.space ? !g->ram_host :
         d->rt.voff + hd5770_surf_size(&d->rt.s) > g->vram_size) ||
        !d->rt.s.fast) {
        g->declined[GLD_SURF]++;
        return false;
    }
    for (i = 0; i < d->ntex; i++) {
        const HD5770GpuSurf *ts = &d->tex[i].surf;

        if (ts->space ? !g->ram_host :
            ts->voff + hd5770_surf_size(&ts->s) > g->vram_size) {
            g->declined[GLD_SURF]++;
            return false;
        }
        for (k = 1; d->tex[i].mip && k <= d->tex[i].last_lv; k++) {
            ts = &d->tex[i].lv[k];
            if (ts->space ||
                ts->voff + hd5770_surf_size(&ts->s) > g->vram_size) {
                g->declined[GLD_SURF]++;
                return false;
            }
        }
    }
    for (i = 0; d->db && i < 2; i++) {
        const HD5770GpuSurf *ps = i ? &d->ss : &d->zs;

        if ((i ? d->has_s : d->zfmt != 0) &&
            (ps->space || !ps->s.fast ||
             ps->voff + hd5770_surf_size(&ps->s) > g->vram_size)) {
            g->declined[GLD_SURF]++;
            return false;
        }
    }
    for (i = 0; i < d->ntri; i++) {
        const HD5770GpuTri *t = &d->tri[i];

        if (t->x1 >= t->x0 && t->y1 >= t->y0) {
            ntiles += ((size_t)(t->x1 - t->x0) / 8 + 2) *
                      ((size_t)(t->y1 - t->y0) + 2);
            miny = MIN(miny, t->y0);
            maxy = MAX(maxy, t->y1);
        }
    }
    if (ntiles > GL_MAX_TILES) {
        g->declined[GLD_ROOM]++;
        return false;
    }
    /* slots for the pages in guest RAM; all of them again if none is written */
    if (d->rt.space && maxy >= miny) {
        gl_rt_pages(d, miny, maxy, &ri0, &ri1);
    }
    for (k = 0; k < 2; k++) {
        uint32_t need = d->rt.space ? gl_slots_new(g, &d->rt, ri0, ri1) : 0;

        for (i = 0; i < d->ntex; i++) {
            if (d->tex[i].surf.space) {
                need += gl_slots_new(g, &d->tex[i].surf, 0, UINT32_MAX);
            }
        }
        if (g->nused + need <= g->nslots) {
            break;
        }
        if (k || g->pool_busy) {
            g->declined[GLD_ROOM]++;
            return false;
        }
        gl_pool_reset(g);
    }
    t0 = get_clock();
    if (g->seg->a[B_TILES]->len / 8 + ntiles > GL_MAX_TILES ||
        g->seg_draws >= GL_SEG_DRAWS) {
        gl_submit(g);
    }

    /* the bytes it reads, in the copy before its dispatch */
    for (i = 0; i < d->ntex; i++) {
        const HD5770GpuSurf *ts = &d->tex[i].surf;

        if (ts->space) {
            tpt[i] = gl_ram_surf(g, ts, 0, UINT32_MAX, false);
            ram = true;
        } else {
            gl_need(g, ts->voff, ts->voff + hd5770_surf_size(&ts->s));
        }
        for (k = 1; d->tex[i].mip && k <= d->tex[i].last_lv; k++) {
            ts = &d->tex[i].lv[k];
            gl_need(g, ts->voff, ts->voff + hd5770_surf_size(&ts->s));
        }
    }
    if (d->rt.space) {
        rpt = gl_ram_surf(g, &d->rt, ri0, ri1, true);
        g->pool_busy |= ri1 >= ri0;
        ram = true;
    } else if (maxy >= miny) {
        hd5770_surf_rows(&d->rt.s, miny, maxy + 1, &lo, &hi);
        lo = lo - d->rt.s.base + d->rt.voff;
        hi = MIN(hi - d->rt.s.base + d->rt.voff, g->vram_size);
        gl_need(g, lo, hi);
        gl_range(g, lo, hi);
    }
    for (i = 0; d->db && maxy >= miny && i < 2; i++) {
        const HD5770GpuSurf *ps = i ? &d->ss : &d->zs;

        if (i ? d->has_s : d->zfmt != 0) {
            gl_wrows(g, ps, miny, maxy, &lo, &hi);
            hi = MIN(hi, g->vram_size);
            gl_need(g, lo, hi);
            gl_range(g, lo, hi);
        }
    }

    memset(&u, 0, sizeof(u));
    u.rt = gl_surf(g, &d->rt, rpt);
    u.rt_fmt = d->rt_fmt;
    u.rt_num = d->rt_num;
    u.rt_comps = d->rt_comps;
    u.rt_unorm8 = d->rt_unorm8;
    u.rt_float = d->rt_float;
    u.rt_blend = d->rt_blend;
    u.wmask = d->wmask;
    u.blend = d->blend;
    memcpy(u.rt_bits, d->rt_bits, sizeof(u.rt_bits));
    memcpy(u.rt_src, d->rt_src, sizeof(u.rt_src));
    memcpy(u.blend_const, d->blend_const, sizeof(u.blend_const));
    u.alpha_test = d->alpha_test;
    u.alpha_func = d->alpha_func;
    u.alpha_ref = d->alpha_ref;
    u.cliprule = d->cliprule;
    memcpy(u.cr, d->cr, sizeof(u.cr));
    u.pix_center = d->pix_center;
    u.persp = d->persp;
    u.pos_gpr = d->pos_gpr >= 0 && d->ir->map[d->pos_gpr] != 0xff
                ? d->ir->map[d->pos_gpr] : -1;
    u.ninputs = d->ninputs;
    u.flat = d->flat;
    if (g->seg->cfile == ~0u ||
        memcmp(&g_array_index(g->seg->a[B_WORDS], uint32_t, g->seg->cfile),
               d->cfile, 4096)) {
        g->seg->cfile = gl_words(g, d->cfile, 4096);
    }
    u.cfile = g->seg->cfile;
    u.ntex = d->ntex;
    u.nvb = d->nvb;
    if (d->db) {
        u.db = 1;
        u.zfmt = d->zfmt;
        u.has_s = d->has_s;
        u.zs = d->zfmt ? gl_surf(g, &d->zs, 0) : 0;
        u.ss = d->has_s ? gl_surf(g, &d->ss, 0) : 0;
        u.dctl = d->dctl;
        u.rctl = d->rctl;
        u.srm0 = d->srm[0];
        u.srm1 = d->srm[1];
        u.sclear = d->sclear;
        u.zclear = d->zclear;
        u.zmin = d->zmin;
        u.zmax = d->zmax;
    }
    for (i = 0; i < d->nvb; i++) {
        u.vb[i].id = d->vb[i].id;
        u.vb[i].off = gl_words(g, d->vb[i].data, d->vb[i].len);
        u.vb[i].len = d->vb[i].len;
        memcpy(u.vb[i].res, d->vb[i].res, sizeof(u.vb[i].res));
    }
    tex0 = g->seg->a[B_TEXS]->len;
    for (i = 0; i < d->ntex; i++) {
        const HD5770GpuTex *t = &d->tex[i];
        GTex gt;

        memset(&gt, 0, sizeof(gt));
        gt.s = gl_surf(g, &t->surf, tpt[i]);
        gt.fmt = t->fmt;
        gt.num = t->num;
        gt.w = t->w;
        gt.h = t->h;
        gt.dim = t->dim;
        gt.sgn = t->sgn;
        gt.comps = t->comps;
        gt.is_float = t->is_float;
        gt.unorm8 = t->unorm8;
        gt.samp0 = t->samp[0];
        for (k = 0; k < 4; k++) {
            gt.dst_sel[k] = t->dst_sel[k];
            gt.bits[k] = t->bits[k];
            gt.border[k] = t->border[k];
        }
        gt.base_lv = t->base_lv;
        gt.levels = t->levels;
        gt.depth = t->depth;
        gt.lw[0] = t->w;
        gt.lh[0] = t->h;
        if (t->mip) {
            /* levels 1..last_lv in consecutive surfaces from lev */
            for (k = 1; k <= t->last_lv; k++) {
                uint32_t si = gl_surf(g, &t->lv[k], 0);

                if (k == 1) {
                    gt.lev = si;
                }
                gt.lw[k] = t->lw[k];
                gt.lh[k] = t->lh[k];
            }
            gt.mip = 1;
            gt.last_lv = t->last_lv;
            gt.mip_filter = t->mip_filter;
            gt.min_lod = t->min_lod;
            gt.max_lod = t->max_lod;
            gt.lod_bias = t->lod_bias;
        }
        g_array_append_val(g->seg->a[B_TEXS], gt);
    }
    kvo = gl_words(g, d->kv, (size_t)d->ir->nconsts * 4);
    g_array_append_val(g->seg->a[B_DRAWS], u);
    di = g->seg->a[B_DRAWS]->len - 1;

    if (g->bprog != p) {
        gl_flush(g);
        g->bprog = p;
    }
    tiles = g->seg->a[B_TILES];
    for (i = 0; i < d->ntri; i++) {
        const HD5770GpuTri *t = &d->tri[i];
        GTri gt;
        unsigned tw, th;
        uint32_t ti;
        int x, y;

        if (t->x1 < t->x0 || t->y1 < t->y0) {
            continue;
        }
        gl_hazard(g, d, t, first);
        first = false;
        memcpy(gt.x, t->x, sizeof(gt.x));
        memcpy(gt.y, t->y, sizeof(gt.y));
        memcpy(gt.w, t->w, sizeof(gt.w));
        gt.area = t->area;
        gt.x0 = t->x0;
        gt.y0 = t->y0;
        gt.x1 = t->x1;
        gt.y1 = t->y1;
        memcpy(gt.z, t->z, sizeof(gt.z));
        gt.front = t->front;
        g_array_append_val(g->seg->a[B_TRIS], gt);
        ti = g->seg->a[B_TRIS]->len - 1;
        prm = gl_words(g, d->prm + (size_t)i * d->ninputs * 12, prm_tri);
        /* tiles of 64 x 1 for thin boxes, up to 8 x 8 */
        th = t->y1 - t->y0 + 1;
        th = th >= 8 ? 8 : th >= 4 ? 4 : th >= 2 ? 2 : 1;
        tw = GL_TILE / th;
        for (y = t->y0; y <= t->y1; y += th) {
            for (x = t->x0; x <= t->x1; x += tw) {
                uint32_t te[8] = { di, ti, prm, tex0, kvo, x, y, tw };

                g_array_append_vals(tiles, te, 8);
            }
        }
        g->tris++;
    }
    g->draws++;
    g->ram_draws += ram;
    g->seg_draws++;
    g->encode_ns += get_clock() - t0;
    return true;
}

static void gl_stats(void *opaque, GString *out)
{
    HD5770Gl *g = opaque;

    g_string_append_printf(out,
        "gl: %s draws %" PRIu64 " triangles %" PRIu64 " declined: compiling %"
        PRIu64 " program %" PRIu64 " room %" PRIu64 " surface %" PRIu64
        " gl error %" PRIu64 "; in guest RAM: draws %" PRIu64 " slots max %"
        PRIu64 " of %u, freed %" PRIu64 "%s%s\n", g->renderer, g->draws, g->tris,
        g->declined[GLD_COMPILING], g->declined[GLD_PROG],
        g->declined[GLD_ROOM], g->declined[GLD_SURF], g->declined[GLD_DEAD],
        g->ram_draws, g->slots_max, g->nslots, g->pool_resets,
        qatomic_read(&g->dead) ? " STOPPED: " : "",
        qatomic_read(&g->dead) ? g->dead_why : "");
    g_string_append_printf(out,
        "gl: syncs %" PRIu64 " segments %" PRIu64 " dispatches %" PRIu64
        " barriers %" PRIu64 " staged MB %" PRIu64 " checked MB %" PRIu64
        " uploaded MB %" PRIu64 " read back MB %" PRIu64 " written back KB %"
        PRIu64 " wait ms %" PRIu64 " (gpu %" PRIu64 " read back %" PRIu64
        ") encode ms %" PRIu64 " programs %u compiled %" PRIu64
        " (failed %" PRIu64 ") in ms %" PRIu64 " binaries: loaded %" PRIu64
        " rejected %" PRIu64 " stored %" PRIu64 "\n",
        g->syncs, g->segs, g->dispatches,
        g->barriers, g->st_bytes >> 20, g->chk_bytes >> 20, g->up_bytes >> 20,
        g->rb_bytes >> 20, g->wb_bytes >> 10, g->wait_ns / 1000000,
        g->gpu_ns / 1000000, g->rb_ns / 1000000, g->encode_ns / 1000000,
        g_hash_table_size(g->progs), g->compiles, g->failed,
        g->compile_ns / 1000000, g->pb_hits, g->pb_rejected, g->pb_stored);
}

static void gl_prog_free(gpointer p)
{
    GlProg *gp = p;

    g_free(gp->src);
    g_free(gp);
}

static void gl_close(void *opaque)
{
    HD5770Gl *g = opaque;
    GlJob j = { .kind = JOB_QUIT };
    uint64_t i;

    gl_sync(g);
    if (g->cthread_on) {
        qemu_mutex_lock(&g->lock);
        g->cquit = true;
        qemu_cond_signal(&g->ccond);
        qemu_mutex_unlock(&g->lock);
        qemu_thread_join(&g->cthread);
    }
    gl_job(g, &j, true);
    qemu_thread_join(&g->thread);
    g_queue_free(g->cq);
    qemu_cond_destroy(&g->ccond);
    g_hash_table_destroy(g->progs);
    seg_free(g->seg);
    for (i = 0; i < DIV_ROUND_UP(g->size, 1u << GL_BLK_SH); i++) {
        g_free(g->blk[i]);
    }
    g_free(g->slot_ram);
    g_hash_table_destroy(g->slot_of);
    g_free(g->blk);
    g_free(g->pchk);
    g_free(g->pval);
    g_free(g->cdir);
    g_array_free(g->ranges, true);
    g_queue_free(g->jobs);
    qemu_sem_destroy(&g->up);
    qemu_cond_destroy(&g->cond);
    qemu_mutex_destroy(&g->lock);
    g_free(g);
}

/* guest RAM, for surfaces there: one block, page aligned */
static bool gl_ram(void *opaque, uint8_t *host, uint64_t size)
{
    HD5770Gl *g = opaque;

    if (g->ram_host || g->nslots < 2 || (uintptr_t)host & 4095) {
        return false;
    }
    g->ram_host = host;
    g->ram_size = size;
    return true;
}

static const HD5770GpuOps gl_ops = {
    .name = "gl",
    .draw = gl_draw,
    .sync = gl_sync,
    .stats = gl_stats,
    .close = gl_close,
    .ram = gl_ram,
    .depth = true,
    .mipmaps = true,
    .resinfo = true,
};

HD5770Gpu *hd5770_gl_open(uint8_t *vram, uint64_t vram_size, const char **err)
{
    static char why[sizeof(((HD5770Gl *)0)->err)];
    HD5770Gl *g = g_new0(HD5770Gl, 1);

    g->vram_host = vram;
    g->vram_size = vram_size;
    qemu_mutex_init(&g->lock);
    qemu_cond_init(&g->cond);
    qemu_sem_init(&g->up, 0);
    g->jobs = g_queue_new();
    qemu_thread_create(&g->thread, "ati-hd5770-gl", gl_thread, g,
                       QEMU_THREAD_JOINABLE);
    qemu_sem_wait(&g->up);
    if (g->err[0]) {
        snprintf(why, sizeof(why), "%s", g->err);
        *err = why;
        qemu_thread_join(&g->thread);
        g_queue_free(g->jobs);
        qemu_sem_destroy(&g->up);
        qemu_cond_destroy(&g->cond);
        qemu_mutex_destroy(&g->lock);
        g_free(g);
        return NULL;
    }
    *err = NULL;
    g->progs = g_hash_table_new_full(g_int64_hash, g_int64_equal, g_free,
                                     gl_prog_free);
    qemu_cond_init(&g->ccond);
    g->cq = g_queue_new();
    if (g->cctx) {
        g->cthread_on = true;
        qemu_thread_create(&g->cthread, "ati-hd5770-glc", gl_cthread, g,
                           QEMU_THREAD_JOINABLE);
    }
    g->seg = seg_new();
    g->blk = g_new0(uint8_t *, DIV_ROUND_UP(g->size, 1u << GL_BLK_SH));
    g->pchk = g_new0(uint32_t, DIV_ROUND_UP(g->size, 1u << GL_PAGE_SH));
    g->pval = g_new0(uint8_t, DIV_ROUND_UP(g->size, 1u << GL_PAGE_SH));
    g->nslots = (g->size - vram_size) >> GL_PAGE_SH;
    g->slot_ram = g_new0(uint32_t, MAX(g->nslots, 1));
    g->slot_of = g_hash_table_new(g_direct_hash, g_direct_equal);
    g->nused = 1;
    g->epoch = 1;
    g->ranges = g_array_new(false, false, sizeof(uint64_t));
    g->pub.ops = &gl_ops;
    g->pub.b = g;
    return &g->pub;
}
