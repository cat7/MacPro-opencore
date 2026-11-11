/*
 * ATI R580 3D engine, R500 pipeline on the host GPU: the OpenGL 4.1
 * core backend (ati_r580_r5gpu.h). Included by ati_r580_gl.c for its
 * platform seam; plain GL otherwise.
 *
 * Images are R8UI / R16UI / R32UI textures. A colour buffer is drawn
 * through an FBO; a draw that reads its destination copies each pass's
 * bounding box to a scratch image first and reads that.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/error-report.h"
#include "ati_r580_r5gpu.h"

#define R5GL_PROGS      128
#define R5GL_UNIT_LUT   R5G_MAX_BOUND
#define R5GL_UNIT_DST   (R5G_MAX_BOUND + 1)
#define R5GL_UNIT_PLANE (R5G_MAX_BOUND + 2)
/* a unit no shader reads: image set-up and copies bind there */
#define R5GL_UNIT_WORK  31
#define R5GL_LUT_W      1024
#define R5GL_LUT_H      10

struct R5GpuImage {
    GLuint tex, fbo;
    int w, h, levels, bpp;
};

typedef struct R5GlProg {
    uint64_t key;
    GLuint prog;
    GLint vp, tu[R5G_UNITS];
    bool failed;
    uint64_t lru;
} R5GlProg;

/* std140 mirror of the shader's uniform block */
typedef struct R5GlBlock {
    float kc[R5G_CONSTS][4];
    int32_t ki[R5G_KI][4];
    float kf[R5G_KF][4];
    int32_t ti[R5G_UNITS * R5G_TI][4];
    float tf[R5G_UNITS * R5G_TF][4];
} R5GlBlock;

typedef struct R5Gl {
    R5GlProg prog[R5GL_PROGS];
    uint64_t tick;
    GLuint vao, vbo, ubo, lut, plbo, pltex;
    R5GpuImage *dst;
    GLuint rfbo;
    R5GlBlock blk;
    uint64_t links, fails, draws, link_us, link_max;
    unsigned reports;
} R5Gl;

static const char r5gl_vs_head[] =
"#version 410 core\n"
"layout(location = 0) in vec4 a_pos;\n"
"uniform vec4 u_vp;\n";

static GLenum r5gl_ifmt(int bpp)
{
    return bpp == 1 ? GL_R8UI : bpp == 2 ? GL_R16UI : GL_R32UI;
}

static GLenum r5gl_type(int bpp)
{
    return bpp == 1 ? GL_UNSIGNED_BYTE :
           bpp == 2 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT;
}

static R5Gl *r5gl_get(R580GlCtx *g)
{
    R5Gl *r = g->r5;
    float *lut;
    int nb, v;

    if (r) {
        return r;
    }
    r = g->r5 = g_new0(R5Gl, 1);
    glGenVertexArrays(1, &r->vao);
    glGenBuffers(1, &r->vbo);
    glGenBuffers(1, &r->ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, r->ubo);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(R5GlBlock), NULL, GL_DYNAMIC_DRAW);
    glGenFramebuffers(1, &r->rfbo);
    glGenBuffers(1, &r->plbo);
    glBindBuffer(GL_TEXTURE_BUFFER, r->plbo);
    glBufferData(GL_TEXTURE_BUFFER, 64, NULL, GL_STREAM_DRAW);
    glGenTextures(1, &r->pltex);
    glActiveTexture(GL_TEXTURE0 + R5GL_UNIT_PLANE);
    glBindTexture(GL_TEXTURE_BUFFER, r->pltex);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, r->plbo);
    lut = g_new0(float, R5GL_LUT_W * R5GL_LUT_H);
    for (nb = 1; nb <= R5GL_LUT_H; nb++) {
        for (v = 0; v < (1 << nb) && v < R5GL_LUT_W; v++) {
            lut[(nb - 1) * R5GL_LUT_W + v] = (float)v / (float)((1u << nb) - 1);
        }
    }
    glGenTextures(1, &r->lut);
    glActiveTexture(GL_TEXTURE0 + R5GL_UNIT_LUT);
    glBindTexture(GL_TEXTURE_2D, r->lut);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, R5GL_LUT_W, R5GL_LUT_H, 0,
                 GL_RED, GL_FLOAT, lut);
    glActiveTexture(GL_TEXTURE0);
    g_free(lut);
    return r;
}

static void r5gl_image_destroy(R5GpuImage *im)
{
    if (im->fbo) {
        glDeleteFramebuffers(1, &im->fbo);
    }
    glDeleteTextures(1, &im->tex);
    g_free(im);
}

static void gl_r5_close(R580GlCtx *g)
{
    R5Gl *r = g->r5;
    unsigned k;

    if (!r) {
        return;
    }
    for (k = 0; k < R5GL_PROGS; k++) {
        if (r->prog[k].prog) {
            glDeleteProgram(r->prog[k].prog);
        }
    }
    if (r->dst) {
        r5gl_image_destroy(r->dst);
    }
    glDeleteTextures(1, &r->lut);
    glDeleteTextures(1, &r->pltex);
    glDeleteBuffers(1, &r->plbo);
    glDeleteFramebuffers(1, &r->rfbo);
    glDeleteBuffers(1, &r->vbo);
    glDeleteBuffers(1, &r->ubo);
    glDeleteVertexArrays(1, &r->vao);
    g_free(r);
    g->r5 = NULL;
}

static R5GpuImage *r5gl_image_new(int w, int h, int levels, int bpp)
{
    R5GpuImage *im = g_new0(R5GpuImage, 1);
    int l;

    im->w = w;
    im->h = h;
    im->levels = levels;
    im->bpp = bpp;
    glGenTextures(1, &im->tex);
    glActiveTexture(GL_TEXTURE0 + R5GL_UNIT_WORK);
    glBindTexture(GL_TEXTURE_2D, im->tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                    levels > 1 ? GL_NEAREST_MIPMAP_NEAREST : GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, levels - 1);
    for (l = 0; l < levels; l++) {
        glTexImage2D(GL_TEXTURE_2D, l, r5gl_ifmt(bpp), MAX(w >> l, 1),
                     MAX(h >> l, 1), 0, GL_RED_INTEGER, r5gl_type(bpp), NULL);
    }
    return im;
}

struct R5GpuImage *ati_r580_gl_r5_image(R580GlCtx *g, int w, int h,
                                        int levels, int bpp)
{
    R5GpuImage *im;

    if (!g || w <= 0 || h <= 0 || w > 16384 || h > 16384 || levels < 1 ||
        levels > 16 || (bpp != 1 && bpp != 2 && bpp != 4)) {
        return NULL;
    }
    r580_gl_makecurrent(&g->plat);
    r5gl_get(g);
    im = r5gl_image_new(w, h, levels, bpp);
    if (glGetError() != GL_NO_ERROR) {
        r5gl_image_destroy(im);
        im = NULL;
    }
    r580_gl_done(&g->plat);
    return im;
}

void ati_r580_gl_r5_image_free(R580GlCtx *g, struct R5GpuImage *im)
{
    if (!g || !im) {
        return;
    }
    r580_gl_makecurrent(&g->plat);
    r5gl_image_destroy(im);
    r580_gl_done(&g->plat);
}

bool ati_r580_gl_r5_upload(R580GlCtx *g, struct R5GpuImage *im, int level,
                           int x, int y, int w, int h, const void *data,
                           unsigned stride)
{
    bool ok;

    if (!g || !im || level < 0 || level >= im->levels || w <= 0 || h <= 0 ||
        stride % im->bpp) {
        return false;
    }
    r580_gl_makecurrent(&g->plat);
    glActiveTexture(GL_TEXTURE0 + R5GL_UNIT_WORK);
    glBindTexture(GL_TEXTURE_2D, im->tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, stride / im->bpp);
    glTexSubImage2D(GL_TEXTURE_2D, level, x, y, w, h, GL_RED_INTEGER,
                    r5gl_type(im->bpp), data);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    ok = glGetError() == GL_NO_ERROR;
    r580_gl_done(&g->plat);
    return ok;
}

static bool r5gl_fbo(R5GpuImage *im)
{
    if (!im->fbo) {
        glGenFramebuffers(1, &im->fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, im->fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, im->tex, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) !=
            GL_FRAMEBUFFER_COMPLETE) {
            return false;
        }
        return true;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, im->fbo);
    return true;
}

bool ati_r580_gl_r5_read(R580GlCtx *g, struct R5GpuImage *im, int x, int y,
                         int w, int h, void *data, unsigned stride)
{
    bool ok;

    if (!g || !im || w <= 0 || h <= 0 || x < 0 || y < 0 ||
        x + w > im->w || y + h > im->h || stride % im->bpp) {
        return false;
    }
    r580_gl_makecurrent(&g->plat);
    ok = r5gl_fbo(im);
    if (ok) {
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glPixelStorei(GL_PACK_ROW_LENGTH, stride / im->bpp);
        glReadPixels(x, y, w, h, GL_RED_INTEGER, r5gl_type(im->bpp), data);
        glPixelStorei(GL_PACK_ROW_LENGTH, 0);
        ok = glGetError() == GL_NO_ERROR;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, g->fbo);
    r580_gl_done(&g->plat);
    return ok;
}

/* level-0 rectangle of src into dst; the current context is held */
static bool r5gl_copy(R5GpuImage *dst, int dx, int dy, R5GpuImage *src,
                      int sx, int sy, int w, int h)
{
    if (!r5gl_fbo(src)) {
        return false;
    }
    glActiveTexture(GL_TEXTURE0 + R5GL_UNIT_WORK);
    glBindTexture(GL_TEXTURE_2D, dst->tex);
    glCopyTexSubImage2D(GL_TEXTURE_2D, 0, dx, dy, sx, sy, w, h);
    return true;
}

bool ati_r580_gl_r5_copy(R580GlCtx *g, struct R5GpuImage *dst, int dx, int dy,
                         struct R5GpuImage *src, int sx, int sy, int w, int h)
{
    bool ok;

    if (!g || !dst || !src || src->bpp != dst->bpp || w <= 0 || h <= 0 ||
        sx < 0 || sy < 0 || sx + w > src->w || sy + h > src->h ||
        dx < 0 || dy < 0 || dx + w > dst->w || dy + h > dst->h) {
        return false;
    }
    r580_gl_makecurrent(&g->plat);
    ok = r5gl_copy(dst, dx, dy, src, sx, sy, w, h) &&
         glGetError() == GL_NO_ERROR;
    glBindFramebuffer(GL_FRAMEBUFFER, g->fbo);
    r580_gl_done(&g->plat);
    return ok;
}

bool ati_r580_gl_r5_dst_fetch(R580GlCtx *g)
{
    return false;
}

void ati_r580_gl_r5_stats(R580GlCtx *g, uint64_t *progs, uint64_t *fails,
                          uint64_t *draws, uint64_t *link_us,
                          uint64_t *link_max)
{
    R5Gl *r = g ? g->r5 : NULL;

    *progs = r ? r->links : 0;
    *fails = r ? r->fails : 0;
    *draws = r ? r->draws : 0;
    *link_us = r ? r->link_us : 0;
    *link_max = r ? r->link_max : 0;
}

static GLuint r5gl_shader(R5Gl *r, GLenum type, const char *src)
{
    GLuint sh = glCreateShader(type);
    GLint ok = 0;

    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        if (r->reports++ < 4) {
            char log[2048];

            log[0] = 0;
            glGetShaderInfoLog(sh, sizeof(log), NULL, log);
            error_report("ati-radeon-x1900: R500 %s shader: %s",
                         type == GL_VERTEX_SHADER ? "vertex" : "fragment",
                         log);
        }
        glDeleteShader(sh);
        return 0;
    }
    return sh;
}

static GLuint r5gl_link(R5Gl *r, const R5GpuDraw *d)
{
    GString *vs = g_string_new(r5gl_vs_head);
    GString *fs = g_string_new("#version 410 core\n");
    GLuint v, f, p = 0;
    GLint ok = 0;
    unsigned n, u;

    for (n = 0; n < d->ninputs; n++) {
        bool fl = (d->flat >> n) & 1;

        g_string_append_printf(vs, "layout(location = %u) in vec4 a_in%u;\n"
                               "%sout vec4 fi%u;\n", n + 1, n,
                               fl ? "flat " : "", n);
        g_string_append_printf(fs, "%sin vec4 fi%u;\n", fl ? "flat " : "", n);
    }
    g_string_append(vs, "void main()\n{\n    float w = a_pos.w;\n");
    for (n = 0; n < d->ninputs; n++) {
        g_string_append_printf(vs, "    fi%u = a_in%u;\n", n, n);
    }
    g_string_append(vs,
        "    gl_Position = vec4(((a_pos.x + u_vp.z) * u_vp.x - 1.0) * w,\n"
        "                       ((a_pos.y + u_vp.w) * u_vp.y - 1.0) * w,\n"
        "                       0.0, w);\n}\n");

    g_string_append_printf(fs,
        "layout(std140) uniform R5K {\n"
        "    vec4 kc[%d]; ivec4 ki[%d]; vec4 kf[%d];\n"
        "    ivec4 ti[%d]; vec4 tf[%d];\n"
        "};\n"
        "#define R5_KC(i) kc[i]\n"
        "#define R5_KI(i) ki[i]\n"
        "#define R5_KF(i) kf[i]\n"
        "#define R5_TI(u, i) ti[(u) * %d + (i)]\n"
        "#define R5_TF(u, i) tf[(u) * %d + (i)]\n"
        "#define R5_IN(n) fi##n\n"
        "#define R5_DFDX(e) dFdx(e)\n"
        "#define R5_DFDY(e) dFdy(e)\n"
        "#define R5_FRAG gl_FragCoord.xy\n"
        "#define R5_PRIM (gl_PrimitiveID + R5_KI(4).y)\n"
        "uniform samplerBuffer r5_pl;\n"
        "#define R5_PLANE(i) texelFetch(r5_pl, i)\n"
        "#define R5_THIS\n"
        "uniform sampler2D r5_lut;\n"
        "#define R5_LUT(nb, v) texelFetch(r5_lut, ivec2(int(v), (nb) - 1), 0).r\n"
        "uniform usampler2D r5_dst;\n"
        "#define R5_DST texelFetch(r5_dst, ivec2(gl_FragCoord.xy) - "
        "R5_KI(4).zw, 0).r\n"
        "layout(location = 0) out uint o_c;\n",
        R5G_CONSTS, R5G_KI, R5G_KF, R5G_UNITS * R5G_TI, R5G_UNITS * R5G_TF,
        R5G_TI, R5G_TF);
    for (u = 0; u < R5G_UNITS; u++) {
        if (d->units & (1u << u)) {
            g_string_append_printf(fs, "uniform usampler2D r5_t%u;\n"
                "#define R5_FETCH%u(x, y, l) texelFetch(r5_t%u, "
                "ivec2(x, y), l).r\n", u, u, u);
        }
    }
    g_string_append(fs, d->fs);
    g_string_append(fs,
        "void main()\n{\n"
        "    uint p = om_pixel();\n"
        "    if (!us_keep) discard;\n"
        "    o_c = p;\n"
        "}\n");

    v = r5gl_shader(r, GL_VERTEX_SHADER, vs->str);
    f = v ? r5gl_shader(r, GL_FRAGMENT_SHADER, fs->str) : 0;
    if (v && f) {
        p = glCreateProgram();
        glAttachShader(p, v);
        glAttachShader(p, f);
        glLinkProgram(p);
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok) {
            if (r->reports++ < 4) {
                char log[2048];

                log[0] = 0;
                glGetProgramInfoLog(p, sizeof(log), NULL, log);
                error_report("ati-radeon-x1900: R500 program link: %s", log);
            }
            glDeleteProgram(p);
            p = 0;
        }
    }
    if (v) {
        glDeleteShader(v);
    }
    if (f) {
        glDeleteShader(f);
    }
    g_string_free(vs, true);
    g_string_free(fs, true);
    if (p) {
        GLuint bi = glGetUniformBlockIndex(p, "R5K");
        char nm[16];

        if (bi != GL_INVALID_INDEX) {
            glUniformBlockBinding(p, bi, 0);
        }
        glUseProgram(p);
        glUniform1i(glGetUniformLocation(p, "r5_lut"), R5GL_UNIT_LUT);
        glUniform1i(glGetUniformLocation(p, "r5_dst"), R5GL_UNIT_DST);
        glUniform1i(glGetUniformLocation(p, "r5_pl"), R5GL_UNIT_PLANE);
        (void)nm;
    }
    return p;
}

static R5GlProg *r5gl_prog(R5Gl *r, const R5GpuDraw *d)
{
    R5GlProg *pr, *victim = &r->prog[0];
    unsigned k;

    for (k = 0; k < R5GL_PROGS; k++) {
        pr = &r->prog[k];
        if ((pr->prog || pr->failed) && pr->key == d->key) {
            pr->lru = ++r->tick;
            return pr->failed ? NULL : pr;
        }
        if (pr->lru < victim->lru) {
            victim = pr;
        }
    }
    pr = victim;
    if (pr->prog) {
        glDeleteProgram(pr->prog);
    }
    memset(pr, 0, sizeof(*pr));
    pr->key = d->key;
    pr->lru = ++r->tick;
    {
        int64_t t0 = g_get_monotonic_time();

        pr->prog = r5gl_link(r, d);
        t0 = g_get_monotonic_time() - t0;
        r->link_us += t0;
        r->link_max = MAX(r->link_max, (uint64_t)t0);
    }
    if (!pr->prog) {
        pr->failed = true;
        r->fails++;
        return NULL;
    }
    r->links++;
    pr->vp = glGetUniformLocation(pr->prog, "u_vp");
    for (k = 0; k < R5G_UNITS; k++) {
        char nm[16];

        snprintf(nm, sizeof(nm), "r5_t%u", k);
        pr->tu[k] = glGetUniformLocation(pr->prog, nm);
    }
    return pr;
}

int ati_r580_gl_r5_prog_ready(R580GlCtx *g, const struct R5GpuDraw *d,
                              int bpp)
{
    R5GlProg *pr;

    if (!g) {
        return -1;
    }
    r580_gl_makecurrent(&g->plat);
    pr = r5gl_prog(r5gl_get(g), d);
    glUseProgram(0);
    r580_gl_done(&g->plat);
    return pr ? 1 : -1;
}

/* the scratch the destination is copied to, at least w x h */
static R5GpuImage *r5gl_dst(R5Gl *r, int w, int h, int bpp)
{
    if (r->dst && (r->dst->w < w || r->dst->h < h || r->dst->bpp != bpp)) {
        w = MAX(w, r->dst->w);
        h = MAX(h, r->dst->h);
        r5gl_image_destroy(r->dst);
        r->dst = NULL;
    }
    if (!r->dst) {
        r->dst = r5gl_image_new(ROUND_UP(w, 64), ROUND_UP(h, 64), 1, bpp);
    }
    return r->dst;
}

bool ati_r580_gl_r5_draw(R580GlCtx *g, const struct R5GpuDraw *d)
{
    R5GpuImage *t = d ? d->target : NULL;
    unsigned stride, n, u, k;
    R5GlProg *pr;
    R5Gl *r;
    bool ok = false;

    if (!g || !t || !d->nvert || d->sx1 <= d->sx0 || d->sy1 <= d->sy0) {
        return false;
    }
    r580_gl_makecurrent(&g->plat);
    gl_flush_queue(g);
    r = r5gl_get(g);
    pr = r5gl_prog(r, d);
    if (!pr || !r5gl_fbo(t)) {
        goto out;
    }
    glUseProgram(pr->prog);
    glUniform4f(pr->vp, 2.0f / t->w, 2.0f / t->h, 0.0f, 0.0f);

    memcpy(r->blk.kc, d->consts, sizeof(r->blk.kc));
    memcpy(r->blk.ki, d->ki, sizeof(r->blk.ki));
    memcpy(r->blk.kf, d->kf, sizeof(r->blk.kf));
    memcpy(r->blk.ti, d->ti, sizeof(r->blk.ti));
    memcpy(r->blk.tf, d->tf, sizeof(r->blk.tf));
    glBindBuffer(GL_UNIFORM_BUFFER, r->ubo);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(r->blk), &r->blk);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, r->ubo);

    for (u = 0, k = 0; u < R5G_UNITS; u++) {
        if (d->units & (1u << u)) {
            if (k == R5G_MAX_BOUND) {
                goto out;
            }
            glActiveTexture(GL_TEXTURE0 + k);
            glBindTexture(GL_TEXTURE_2D, d->img[u]->tex);
            glUniform1i(pr->tu[u], k);
            k++;
        }
    }
    if (!(d->ki[0][3] & R5G_DF_DST)) {
        /* the sampler is declared either way: give it a texture */
        R5GpuImage *dst = r5gl_dst(r, 64, 64, t->bpp);

        glActiveTexture(GL_TEXTURE0 + R5GL_UNIT_DST);
        glBindTexture(GL_TEXTURE_2D, dst ? dst->tex : 0);
    }
    glActiveTexture(GL_TEXTURE0 + R5GL_UNIT_LUT);
    glBindTexture(GL_TEXTURE_2D, r->lut);
    if (d->planes) {
        glBindBuffer(GL_TEXTURE_BUFFER, r->plbo);
        glBufferData(GL_TEXTURE_BUFFER,
                     (GLsizeiptr)(d->nvert / 3) * (d->ninputs * 3 + 1) * 16,
                     d->planes, GL_STREAM_DRAW);
    }
    glActiveTexture(GL_TEXTURE0 + R5GL_UNIT_PLANE);
    glBindTexture(GL_TEXTURE_BUFFER, r->pltex);

    glBindFramebuffer(GL_FRAMEBUFFER, t->fbo);
    glViewport(0, 0, t->w, t->h);
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_CULL_FACE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glEnable(GL_SCISSOR_TEST);
    glScissor(d->sx0, d->sy0, d->sx1 - d->sx0, d->sy1 - d->sy0);

    stride = (4 + 4 * d->ninputs) * sizeof(float);
    glBindVertexArray(r->vao);
    glBindBuffer(GL_ARRAY_BUFFER, r->vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)d->nvert * stride, d->verts,
                 GL_STREAM_DRAW);
    for (n = 0; n <= d->ninputs; n++) {
        glEnableVertexAttribArray(n);
        glVertexAttribPointer(n, 4, GL_FLOAT, GL_FALSE, stride,
                              (const void *)(uintptr_t)(n * 16));
    }

    if (d->ki[0][3] & R5G_DF_DST) {
        for (k = 0; k < d->npass; k++) {
            const int *b = &d->pbox[4 * k];
            int w = b[2] - b[0], h = b[3] - b[1];
            R5GpuImage *dst;

            if (w <= 0 || h <= 0) {
                continue;
            }
            dst = r5gl_dst(r, w, h, t->bpp);
            if (!dst || !r5gl_copy(dst, 0, 0, t, b[0], b[1], w, h)) {
                goto out;
            }
            glActiveTexture(GL_TEXTURE0 + R5GL_UNIT_DST);
            glBindTexture(GL_TEXTURE_2D, dst->tex);
            r->blk.ki[4][1] = d->pass[k] / 3;
            r->blk.ki[4][2] = b[0];
            r->blk.ki[4][3] = b[1];
            glBindBuffer(GL_UNIFORM_BUFFER, r->ubo);
            glBufferSubData(GL_UNIFORM_BUFFER,
                            offsetof(R5GlBlock, ki[4]), 16, r->blk.ki[4]);
            glBindFramebuffer(GL_FRAMEBUFFER, t->fbo);
            glDrawArrays(GL_TRIANGLES, d->pass[k], d->pass[k + 1] - d->pass[k]);
        }
    } else {
        glDrawArrays(GL_TRIANGLES, 0, d->nvert);
    }
    for (n = 0; n <= d->ninputs; n++) {
        glDisableVertexAttribArray(n);
    }
    r->draws++;
    ok = glGetError() == GL_NO_ERROR;
out:
    glDisable(GL_SCISSOR_TEST);
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(g->vao);
    glBindBuffer(GL_ARRAY_BUFFER, g->vbo);
    glBindFramebuffer(GL_FRAMEBUFFER, g->fbo);
    glUseProgram(0);
    r580_gl_done(&g->plat);
    return ok;
}
