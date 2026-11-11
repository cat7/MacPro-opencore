/*
 * ATI R580 3D engine, R500 pipeline (CPU path).
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef ATI_R580_R5XX_H
#define ATI_R580_R5XX_H

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

#endif
