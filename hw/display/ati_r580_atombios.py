#!/usr/bin/env python3
#
# SPDX-License-Identifier: GPL-2.0-or-later
# Adapted from rv530-qemu 0b06c9f (scripts/atombios/mkrom.py).
"""Build the synthetic AtomBIOS option ROM of the emulated ATI R580.

    ati_r580_atombios.py OUT.rom [--device-id 0x7249] [--subsys 106b:0000]
                                 [--efi DRIVER.efi]

Nothing here comes from a real VBIOS: the layouts are those of the public
atombios.h (Linux drivers/gpu/drm/radeon, MIT licensed) and the values are
chosen for the device model.

Command tables are paravirtual: each copies its parameter block (PS) into
the device's AtomBIOS mailbox registers, writes the table index to the
doorbell, and copies results back to PS for the tables that return some.
The device performs the operation (ati_r580_atom.c).

Data tables: FirmwareInfo 1.4 (27 MHz reference, pixel PLL limits, clocks),
SupportedDevicesInfo 2.1 (two DVI-I connectors: CRT1 + DFP1 on DDC1, CRT2 +
DFP2 on DDC2), GPIO_I2C_Info (the three AVIVO DDC GPIO pairs), GPIO_Pin_LUT
(HPD1/2), VRAM_UsageByFirmware (none) and an empty IndirectIOAccess.

--efi appends a UEFI driver as a second image (PCI code type 3), the layout
edk2's EfiRom produces, so firmware loads it from the card.

Every ROM written is first parsed back by verify(): the PCI image chain,
the AtomBIOS signature and header, both master tables, every command
table's code down to its EOT, every data table's bounds and the image
checksum.
"""
import argparse
import struct
import sys

# device mailbox (ati_r580_avivo.h)
MAILBOX = 0xff00            # 16 dwords of parameters / results
DOORBELL = 0xff7c           # write: table index | dword count << 8

# command table index (atombios.h ATOM_MASTER_LIST_OF_COMMAND_TABLES)
CMD_NAMES = """ASIC_Init GetDisplaySurfaceSize ASIC_RegistersInit VRAM_BlockVenderDetection
DIGxEncoderControl MemoryControllerInit EnableCRTCMemReq MemoryParamAdjust DVOEncoderControl
GPIOPinControl SetEngineClock SetMemoryClock SetPixelClock EnableDispPowerGating ResetMemoryDLL
ResetMemoryDevice MemoryPLLInit AdjustDisplayPll AdjustMemoryController EnableASIC_StaticPwrMgt
SetUniphyInstance DAC_LoadDetection LVTMAEncoderControl HW_Misc_Operation DAC1EncoderControl
DAC2EncoderControl DVOOutputControl CV1OutputControl GetConditionalGoldenSetting TVEncoderControl
TMDSAEncoderControl LVDSEncoderControl Gfx_Harvesting EnableScaler BlankCRTC EnableCRTC
GetPixelClock EnableVGA_Render GetSCLKOverMCLKRatio SetCRTC_Timing SetCRTC_OverScan
SetCRTC_Replication SelectCRTC_Source EnableGraphSurfaces UpdateCRTC_DoubleBufferRegisters
LUT_AutoFill EnableHW_IconCursor GetMemoryClock GetEngineClock SetCRTC_UsingDTDTiming
ExternalEncoderControl LVTMAOutputControl VRAM_BlockDetectionByStrap MemoryCleanUp
ProcessI2cChannelTransaction WriteOneByteToHWAssistedI2C ReadHWAssistedI2CStatus SpeedFanControl
PowerConnectorDetection MC_Synchronization ComputeMemoryEnginePLL MemoryRefreshConversion
VRAM_GetCurrentInfoBlock DynamicMemorySettings MemoryTraining EnableSpreadSpectrumOnPPLL
TMDSAOutputControl SetVoltage DAC1OutputControl DAC2OutputControl ComputeMemoryClockParam
ClockSource MemoryDeviceInit GetDispObjectInfo DIG1EncoderControl DIG2EncoderControl
DIG1TransmitterControl DIG2TransmitterControl ProcessAuxChannelTransaction DPEncoderService
GetVoltageInfo""".split()
assert len(CMD_NAMES) == 81

DATA_NAMES = """UtilityPipeLine MultimediaCapabilityInfo MultimediaConfigInfo StandardVESA_Timing
FirmwareInfo PaletteData LCD_Info DIGTransmitterInfo AnalogTV_Info SupportedDevicesInfo
GPIO_I2C_Info VRAM_UsageByFirmware GPIO_Pin_LUT VESA_ToInternalModeLUT ComponentVideoInfo
PowerPlayInfo CompassionateData SaveRestoreInfo PPLL_SS_Info OemInfo XTMDS_Info MclkSS_Info
Object_Header IndirectIOAccess MC_InitParameter ASIC_VDDC_Info ASIC_InternalSS_Info
TV_VideoMode VRAM_Info MemoryTrainingInfo IntegratedSystemInfo ASIC_ProfilingInfo
VoltageObjectInfo PowerSourceInfo""".split()
assert len(DATA_NAMES) == 34

# name: (parameter bytes the device reads, bytes it returns, format rev,
# content rev). Sizes are the atombios.h parameter structs, never the
# larger *_PS_ALLOCATION unions: copying past the caller's struct would
# read or overwrite its stack.
CMDS = {
    "ASIC_Init":                        (0, 0, 1, 1),
    "EnableCRTCMemReq":                 (4, 0, 1, 1),
    "SetEngineClock":                   (4, 0, 1, 1),
    "SetMemoryClock":                   (4, 0, 1, 1),
    "SetPixelClock":                    (12, 0, 1, 2),
    "EnableASIC_StaticPwrMgt":          (4, 0, 1, 1),
    "DAC_LoadDetection":                (4, 0, 1, 2),
    "LVTMAEncoderControl":              (4, 0, 1, 1),
    "DAC1EncoderControl":               (4, 0, 1, 1),
    "DAC2EncoderControl":               (4, 0, 1, 1),
    "TMDSAEncoderControl":              (4, 0, 1, 1),
    "LVDSEncoderControl":               (4, 0, 1, 1),
    "EnableScaler":                     (4, 0, 1, 1),
    "BlankCRTC":                        (8, 0, 1, 1),
    "EnableCRTC":                       (4, 0, 1, 1),
    "GetPixelClock":                    (0, 0, 1, 1),
    "EnableVGA_Render":                 (4, 0, 1, 1),
    "SetCRTC_Timing":                   (24, 0, 1, 1),
    "SetCRTC_OverScan":                 (12, 0, 1, 1),
    "SelectCRTC_Source":                (4, 0, 1, 1),
    "EnableGraphSurfaces":              (4, 0, 1, 1),
    "UpdateCRTC_DoubleBufferRegisters": (4, 0, 1, 1),
    "LUT_AutoFill":                     (4, 0, 1, 1),
    "EnableHW_IconCursor":              (4, 0, 1, 1),
    "GetMemoryClock":                   (0, 4, 1, 1),
    "GetEngineClock":                   (0, 4, 1, 1),
    "SetCRTC_UsingDTDTiming":           (24, 0, 1, 1),
    "LVTMAOutputControl":               (4, 0, 1, 1),
    "ComputeMemoryEnginePLL":           (8, 8, 1, 1),
    "EnableSpreadSpectrumOnPPLL":       (8, 0, 1, 2),
    "TMDSAOutputControl":               (4, 0, 1, 1),
    "SetVoltage":                       (4, 0, 1, 1),
    "DAC1OutputControl":                (4, 0, 1, 1),
    "DAC2OutputControl":                (4, 0, 1, 1),
    "MemoryDeviceInit":                 (0, 0, 1, 1),
    "DynamicMemorySettings":            (0, 0, 1, 1),
}
assert all(n in CMD_NAMES for n in CMDS)

# AtomBIOS opcodes (atom.c): MOVE to REG = 1, MOVE to PS = 2, EOT = 0x5b
OP_MOVE_REG, OP_MOVE_PS, OP_EOT = 0x01, 0x02, 0x5b
ARG_REG, ARG_PS, ARG_IMM = 0, 1, 5

ATOM_ROM_TABLE_PTR = 0x48
ATOM_ATI_MAGIC_PTR = 0x30
ATOM_ATI_MAGIC = b' 761295520'
PCIR_OFF = 0x60
ROM_HEADER_OFF = 0x80
ROM_HEADER_LEN = 36


def cmd_table(index, ps_in, ps_out, frev, crev):
    code = bytearray()
    for i in range(ps_in // 4):                      # MOVE REG[mb + i], PS[i]
        code += struct.pack('<BBHB', OP_MOVE_REG, ARG_PS, (MAILBOX >> 2) + i, i)
    # MOVE REG[doorbell], index | dwords << 8
    code += struct.pack('<BBHI', OP_MOVE_REG, ARG_IMM, DOORBELL >> 2,
                        index | (ps_in // 4) << 8)
    for i in range(ps_out // 4):                     # MOVE PS[i], REG[mb + i]
        code += struct.pack('<BBBH', OP_MOVE_PS, ARG_REG, i, (MAILBOX >> 2) + i)
    code.append(OP_EOT)
    # header: size, revisions, WS bytes, PS bytes
    return struct.pack('<HBBBB', 6 + len(code), frev, crev, 0, ps_in) + code


def hdr(body, frev, crev):
    return struct.pack('<HBB', 4 + len(body), frev, crev) + body


def firmware_info():
    """ATOM_FIRMWARE_INFO_V1_4 (89 bytes); clocks in 10 kHz units."""
    b = struct.pack('<IIIIIIIIII', 0x00010000,      # firmware revision
                    50000, 50000,                    # default engine/memory clock
                    50000, 50000,                    # driver target clocks
                    100000, 100000,                  # max engine/memory PLL output
                    110000,                          # max pixel PLL output (1.1 GHz)
                    60000, 60000)                    # ASIC max engine/memory clock
    b += struct.pack('<BBHHH', 90, 0, 1200, 0, 0)  # max temp, BL level, VDDC mV, LCD PLL
    b += struct.pack('<II', 50000,                   # 3D engine clock
                     64800)                          # min pixel PLL output (648 MHz)
    b += struct.pack('<HHHHHH', 100, 1350, 20000,    # engine PLL input min/max, output min
                     100, 1350, 20000)               # memory PLL input min/max, output min
    b += struct.pack('<HHHH', 40000,                 # max pixel clock 400 MHz
                     100, 1350,                      # pixel PLL input 1-13.5 MHz
                     64800 & 0xffff)                 # min pixel PLL output (low 16 bits)
    b += struct.pack('<HHHBBB', 0x0000,              # firmware capability
                     2700,                           # 27 MHz reference clock
                     0, 0, 0, 0)                     # PM RTS, design/memory module ID
    t = hdr(b, 1, 4)
    assert len(t) == 89, len(t)
    return t


# ATOM_DEVICE_*_INDEX order for asConnInfo[]
(DEV_CRT1, DEV_LCD1, DEV_TV1, DEV_DFP1, DEV_CRT2, DEV_LCD2, DEV_TV2,
 DEV_DFP2, DEV_CV, DEV_DFP3) = range(10)
CONN_DVI_I = 2              # legacy connector type
DAC_NONE, DAC_A, DAC_B = 0, 1, 2
I2C_DDC1 = 0x90             # HW capable, engine 1, line 0: GPIO_I2C_Info entry 0
I2C_DDC2 = 0x91


def supported_devices():
    """ATOM_SUPPORTED_DEVICES_INFO_2: two DVI-I connectors."""
    conn = [(0, 0)] * 16
    conn[DEV_CRT1] = ((CONN_DVI_I << 4) | DAC_A, I2C_DDC1)
    conn[DEV_DFP1] = ((CONN_DVI_I << 4) | DAC_NONE, I2C_DDC1)
    conn[DEV_CRT2] = ((CONN_DVI_I << 4) | DAC_B, I2C_DDC2)
    conn[DEV_DFP2] = ((CONN_DVI_I << 4) | DAC_NONE, I2C_DDC2)
    b = struct.pack('<H', (1 << DEV_CRT1) | (1 << DEV_DFP1) |
                    (1 << DEV_CRT2) | (1 << DEV_DFP2))
    b += b''.join(struct.pack('<BB', c, i) for c, i in conn)
    b += bytes([0xff] * 16)                          # no interrupt sources mapped
    return hdr(b, 2, 1)


def gpio_i2c_info():
    """ATOM_GPIO_I2C_INFO: DC_GPIO_DDC1..3 (MASK, A, EN, Y at +0, +4, +8,
    +0xc); clock on bit 0, data on bit 8, as the device's GPIO model."""
    out = b''
    for line, base in enumerate((0x7e40, 0x7e50, 0x7e60)):
        mask, a, en, y = base >> 2, (base + 4) >> 2, (base + 8) >> 2, (base + 0xc) >> 2
        out += struct.pack('<HHHHHHHH', mask, en, y, a, mask, en, y, a)
        out += struct.pack('<B', 0x90 | line)        # HW capable, engine 1, line n
        out += struct.pack('<BBBBBBBBBB', 0, 0, 0, 0, 8, 8, 8, 8, 0, 0)
    return hdr(out, 1, 1)


def gpio_pin_lut():
    """ATOM_GPIO_PIN_LUT: HPD1/HPD2 sense bits (DC_HOT_PLUG_DETECTn_INT_STATUS
    bit 1), GPIO ids as the AVIVO HPD pins."""
    b = struct.pack('<HBB', 0x7d04 >> 2, 1, 0x14)
    b += struct.pack('<HBB', 0x7d14 >> 2, 1, 0x15)
    return hdr(b, 1, 1)


def vram_usage():
    """ATOM_VRAM_USAGE_BY_FIRMWARE: firmware reserves nothing."""
    return hdr(struct.pack('<IHH', 0, 0, 0), 1, 2)


def iio():
    """IndirectIOAccess with no programs (atom_index_iio stops at once)."""
    return hdr(b'\x00\x00\x00\x00', 1, 1)


def pcir(vendor, device, class_code, blocks, code_type, last):
    """PCI 3.0 data structure (0x1c bytes)."""
    return (b'PCIR' + struct.pack('<HHHHB', vendor, device, 0, 0x1c, 3) +
            struct.pack('<I', class_code)[:3] +
            struct.pack('<HHBBHHH', blocks, 0, code_type, 0x80 if last else 0,
                        0, 0, 0))


def atom_image(device_id, ss_vendor, ss_id, last):
    rom = bytearray(0x10000)
    rom[0:3] = b'\x55\xaa\x00'
    rom[3:6] = b'\xcb\x90\x90'                       # legacy entry: RETF
    struct.pack_into('<H', rom, 0x18, PCIR_OFF)
    rom[ATOM_ATI_MAGIC_PTR:ATOM_ATI_MAGIC_PTR + len(ATOM_ATI_MAGIC)] = ATOM_ATI_MAGIC
    struct.pack_into('<H', rom, ATOM_ROM_TABLE_PTR, ROM_HEADER_OFF)

    pos = 0xc0

    def put(blob, align=2):
        nonlocal pos
        pos = (pos + align - 1) & ~(align - 1)
        o = pos
        rom[o:o + len(blob)] = blob
        pos += len(blob)
        return o

    msg = put(b'\r\nATI R580 QEMU synthetic AtomBIOS (paravirtual)\r\n\x00')
    put(b'R580-QEMU\x00')
    cmd_off = []
    for i, name in enumerate(CMD_NAMES):
        ps_in, ps_out, frev, crev = CMDS.get(name, (0, 0, 1, 1))
        cmd_off.append(put(cmd_table(i, ps_in, ps_out, frev, crev)))
    data = {"FirmwareInfo": firmware_info(),
            "SupportedDevicesInfo": supported_devices(),
            "GPIO_I2C_Info": gpio_i2c_info(), "GPIO_Pin_LUT": gpio_pin_lut(),
            "VRAM_UsageByFirmware": vram_usage(), "IndirectIOAccess": iio()}
    data_off = [put(data[n]) if n in data else 0 for n in DATA_NAMES]
    cmd_master = put(hdr(b''.join(struct.pack('<H', o) for o in cmd_off), 1, 1))
    data_master = put(hdr(b''.join(struct.pack('<H', o) for o in data_off), 1, 1))

    h = struct.pack('<HBB', ROM_HEADER_LEN, 1, 1) + b'ATOM'
    h += struct.pack('<HHHHHHHHHHHHHBB', 0, 0, 0, 0, msg, 0, 0, 0, ss_vendor,
                     ss_id, PCIR_OFF, cmd_master, data_master, 0, 0)
    assert len(h) == ROM_HEADER_LEN
    rom[ROM_HEADER_OFF:ROM_HEADER_OFF + ROM_HEADER_LEN] = h

    blocks = (pos + 1 + 511) // 512                  # room for the checksum byte
    size = blocks * 512
    rom = rom[:size]
    rom[2] = blocks
    rom[PCIR_OFF:PCIR_OFF + 0x1c] = pcir(0x1002, device_id, 0x030000, blocks,
                                        0, last)
    rom[size - 1] = 0
    rom[size - 1] = (-sum(rom)) & 0xff
    assert sum(rom) & 0xff == 0
    return bytes(rom)


def efi_image(efi, device_id):
    """EFI_PCI_EXPANSION_ROM_HEADER + PCIR + the PE image at 0x200."""
    if efi[:2] != b'MZ':
        raise ValueError('not a PE image')
    pe = struct.unpack_from('<I', efi, 0x3c)[0]
    if efi[pe:pe + 4] != b'PE\0\0':
        raise ValueError('not a PE image')
    machine = struct.unpack_from('<H', efi, pe + 4)[0]
    opt = pe + 24
    subsystem = struct.unpack_from('<H', efi, opt + 68)[0]
    if machine != 0x8664 or subsystem not in (11, 12):
        raise ValueError('want an x64 EFI driver, got machine 0x%x subsystem %d'
                         % (machine, subsystem))
    blocks = (0x200 + len(efi) + 511) // 512
    img = bytearray(b'\xff' * 0x200) + efi
    img += bytes(blocks * 512 - len(img))
    struct.pack_into('<HHIHHH', img, 0, 0xaa55, blocks, 0x0ef1, subsystem,
                     machine, 0)
    img[0x0e:0x16] = bytes(8)
    img[0x1a:0x1c] = bytes(2)
    struct.pack_into('<HH', img, 0x16, 0x200, 0x1c)
    img[0x1c:0x1c + 0x1c] = pcir(0x1002, device_id, 0x030000, blocks, 3, True)
    return bytes(img)


def build(device_id, ss_vendor, ss_id, efi=None):
    rom = atom_image(device_id, ss_vendor, ss_id, efi is None)
    if efi is not None:
        rom += efi_image(efi, device_id)
    verify(rom, device_id)
    return rom


def verify(rom, device_id):
    """Parse the ROM back; raise AssertionError on any inconsistency."""
    def u8(p):
        return rom[p]

    def u16(p):
        return struct.unpack_from('<H', rom, p)[0]

    # PCI image chain
    off, images = 0, []
    while True:
        assert u16(off) == 0xaa55, 'image at 0x%x: no 55AA' % off
        p = off + u16(off + 0x18)
        assert rom[p:p + 4] == b'PCIR', 'image at 0x%x: no PCIR' % off
        assert u16(p + 4) == 0x1002 and u16(p + 6) == device_id
        assert rom[p + 0xd:p + 0x10] == b'\x00\x00\x03', 'class'
        length = u16(p + 0x10) * 512
        assert length and off + length <= len(rom), 'image length'
        code_type, indicator = u8(p + 0x14), u8(p + 0x15)
        images.append((off, length, code_type))
        if code_type == 0:
            assert u8(off + 2) * 512 == length, 'legacy size byte'
            assert sum(rom[off:off + length]) & 0xff == 0, 'checksum'
        elif code_type == 3:
            assert struct.unpack_from('<I', rom, off + 4)[0] == 0x0ef1
            assert u16(off + 2) * 512 == length
            e = off + u16(off + 0x16)
            assert rom[e:e + 2] == b'MZ', 'EFI image offset'
        if indicator & 0x80:
            break
        off += length
    assert off + length == len(rom), 'bytes after the last image'
    assert images[0][2] == 0, 'AtomBIOS must be the first image'

    # AtomBIOS
    end = images[0][1]
    assert rom[ATOM_ATI_MAGIC_PTR:ATOM_ATI_MAGIC_PTR + 10] == ATOM_ATI_MAGIC
    base = u16(ATOM_ROM_TABLE_PTR)
    assert rom[base + 4:base + 8] == b'ATOM'
    assert u16(base) == ROM_HEADER_LEN
    assert u16(base + 0x1c) == u16(0x18), 'PCI info offset'
    cmd_master, data_master = u16(base + 0x1e), u16(base + 0x20)

    def table(o, what):
        assert 0 < o and o + 4 <= end, '%s at 0x%x out of range' % (what, o)
        size = u16(o)
        assert size >= 4 and o + size <= end, '%s at 0x%x: size %d' % (what, o, size)
        return size

    size = table(cmd_master, 'command master table')
    assert size == 4 + 2 * len(CMD_NAMES), 'command master size'
    for i, name in enumerate(CMD_NAMES):
        o = u16(cmd_master + 4 + 2 * i)
        if not o:
            continue
        size = table(o, name)
        ws, ps = u8(o + 4), u8(o + 5)
        assert ws == 0 and ps % 4 == 0, name
        ps_in, ps_out, frev, crev = CMDS.get(name, (0, 0, 1, 1))
        assert (u8(o + 2), u8(o + 3)) == (frev, crev), name
        assert ps == ps_in, name
        # code: the MOVE forms this builder emits, then EOT at the end
        p, door = o + 6, False
        while True:
            assert p < o + size, '%s: no EOT' % name
            op = u8(p)
            if op == OP_EOT:
                assert p + 1 == o + size, '%s: code after EOT' % name
                break
            if op == OP_MOVE_REG and u8(p + 1) == ARG_PS:
                assert MAILBOX >> 2 <= u16(p + 2) < (MAILBOX >> 2) + 16
                assert u8(p + 4) < ps // 4, name
                p += 5
            elif op == OP_MOVE_REG and u8(p + 1) == ARG_IMM:
                assert u16(p + 2) == DOORBELL >> 2, name
                val = struct.unpack_from('<I', rom, p + 4)[0]
                assert val & 0xff == i and (val >> 8) & 0xff == ps // 4, name
                door = True
                p += 8
            elif op == OP_MOVE_PS and u8(p + 1) == ARG_REG:
                assert MAILBOX >> 2 <= u16(p + 3) < (MAILBOX >> 2) + 16
                p += 5
            else:
                raise AssertionError('%s: opcode 0x%02x at 0x%x' % (name, op, p))
        assert door, '%s: never rings the doorbell' % name

    size = table(data_master, 'data master table')
    assert size == 4 + 2 * len(DATA_NAMES), 'data master size'
    for i, name in enumerate(DATA_NAMES):
        o = u16(data_master + 4 + 2 * i)
        if o:
            table(o, name)
    fw = u16(data_master + 4 + 2 * DATA_NAMES.index('FirmwareInfo'))
    assert u16(fw) == 89 and (u8(fw + 2), u8(fw + 3)) == (1, 4), 'FirmwareInfo'
    return images


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('out')
    ap.add_argument('--device-id', type=lambda v: int(v, 0), default=0x7249)
    ap.add_argument('--subsys', default='106b:0000')
    ap.add_argument('--efi', help='UEFI driver appended as a second image')
    ap.add_argument('--verify', action='store_true',
                    help='parse OUT as an existing ROM instead of building it')
    a = ap.parse_args()
    if a.verify:
        with open(a.out, 'rb') as f:
            images = verify(f.read(), a.device_id)
        for off, length, code_type in images:
            print('image at 0x%05x: %6d bytes, code type %d' % (off, length,
                                                              code_type))
        return
    sv, si = (int(x, 16) for x in a.subsys.split(':'))
    efi = None
    if a.efi:
        with open(a.efi, 'rb') as f:
            efi = f.read()
    rom = build(a.device_id, sv, si, efi)
    with open(a.out, 'wb') as f:
        f.write(rom)


if __name__ == '__main__':
    sys.exit(main())
