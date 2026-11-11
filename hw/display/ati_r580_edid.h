/*
 * The monitor's EDID on DDC (ati_r580_edid.c): EDID 1.3 base block plus a
 * CEA-861 extension, served by an I2C slave at 0x50.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 * from rv530-qemu 0b06c9f (qemu/hw/display/rv530/r5xx_edid.h)
 */
#ifndef ATI_R580_EDID_H
#define ATI_R580_EDID_H

#include "hw/i2c/i2c.h"
#include "qom/object.h"

/* the monitor's EDID on DDC (ati_r580_edid.c): base block + CEA extension */
#define ATI_R580_EDID_SIZE 256
#define TYPE_ATI_R580_EDID "ati-r580-edid"
typedef struct ATIR580EdidState {
    I2CSlave i2c;
    bool firstbyte;
    uint8_t reg;
    uint8_t blob[ATI_R580_EDID_SIZE];
} ATIR580EdidState;
OBJECT_DECLARE_SIMPLE_TYPE(ATIR580EdidState, ATI_R580_EDID)

void ati_r580_edid_build(uint8_t edid[ATI_R580_EDID_SIZE], uint32_t prefx,
                      uint32_t prefy, uint32_t maxx, uint32_t maxy, uint32_t serial,
                      uint32_t hz);

#endif
