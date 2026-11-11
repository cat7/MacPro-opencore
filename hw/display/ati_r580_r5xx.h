/*
 * ATI R580 3D engine, R500 pipeline (CPU path).
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef ATI_R580_R5XX_H
#define ATI_R580_R5XX_H

#include "ati_r580_r5raster.h"

void ati_r580_r5xx_init(ATIR580State *s, unsigned threads);
void ati_r580_r5xx_reset(ATIR580State *s);
void ati_r580_r5xx_fini(ATIR580State *s);
void ati_r580_r5xx_reg_write(ATIR580State *s, uint32_t off, uint32_t val);
void ati_r580_r5xx_draw_immd(ATIR580State *s, const uint32_t *dw, unsigned n);
void ati_r580_r5xx_draw_vbuf(ATIR580State *s, uint32_t vf);
void ati_r580_r5xx_draw_indx(ATIR580State *s, const uint32_t *dw, unsigned n);
void ati_r580_r5xx_indx_buffer(ATIR580State *s, uint32_t port, uint32_t addr,
                               uint32_t n);
void ati_r580_r5xx_clear_zmask(ATIR580State *s, uint32_t first, uint32_t n,
                               uint32_t val);
char *ati_r580_r5xx_stats(ATIR580State *s);

/* host-GPU path (ati_r580_r5gpu.c) */
typedef struct R5GpuIn {
    const R5xxRaster *raster;           /* set up for the draw */
    const R500UsDecoded *us;
    const float *consts;                /* R500_US_MAX_CONST vec4 */
    const R5xxTexSampler *samp;         /* R5XX_NUM_TEX_UNITS */
    uint32_t used, bound;               /* units sampled; of those, valid */
    const struct R5Prim *prims;
    uint32_t nprims;
} R5GpuIn;

enum { R5G_CPU, R5G_DONE, R5G_VERIFY };

void ati_r580_r5gpu_init(ATIR580State *s);
void ati_r580_r5gpu_fini(ATIR580State *s);
void ati_r580_r5gpu_reset(ATIR580State *s);
int ati_r580_r5gpu_draw(ATIR580State *s, const R5GpuIn *in);
void ati_r580_r5gpu_verify_end(ATIR580State *s);
void ati_r580_r5gpu_cpu_begin(ATIR580State *s, const R5xxRaster *r);
void ati_r580_r5gpu_cpu_end(ATIR580State *s);
void ati_r580_r5gpu_release(ATIR580State *s);
void ati_r580_r5gpu_sync(ATIR580State *s, uint32_t off, uint32_t len);
void ati_r580_r5gpu_wrote(ATIR580State *s, uint32_t off, uint32_t len);
void ati_r580_r5gpu_epoch(ATIR580State *s);
char *ati_r580_r5gpu_stats(ATIR580State *s);

#endif
