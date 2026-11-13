#!/usr/bin/env python3
#
# SPDX-License-Identifier: GPL-2.0-or-later
# Adapted from ati_r580_atombios.py.
"""Build the synthetic AtomBIOS option ROM of the emulated ATI Radeon HD 5770.

    ati_hd5770_atombios.py OUT.rom [--device-id 0x68b8] [--subsys 106b:0000]
                                   [--ddc N] [--hpd N] [--efi DRIVER.efi]

The DVI connector defaults to DDC3 / HPD5, the DVI port of the Mac Pro
card's connector table (ATY,Langur), so both ways in agree.

Nothing here comes from a real VBIOS: the layouts are those of the public
atombios.h (Linux drivers/gpu/drm/radeon, MIT licensed) and the values are
chosen for the device model.

Command tables are paravirtual: each copies its parameter block (PS) into
the device's AtomBIOS mailbox registers, writes the table index to the
doorbell, and copies results back to PS for the tables that return some.
The device performs the operation (ati_hd5770_atom.c).

The part number string sits below 0x100: the display driver takes its
offset from the single byte at 0x6e.

Data tables: FirmwareInfo 2.1 (27 MHz reference, clocks, PLL limits),
Object_Header 1.3 (one single-link DVI-D connector driven by UNIPHY, with
its DDC line and hot plug pin, one display path), GPIO_I2C_Info (the six
DCE 4 DDC GPIO pairs), GPIO_Pin_LUT (HPD1-6 in DC_GPIO_HPD_A),
PowerPlayInfo (PPLib layout, one set of clocks),
VRAM_UsageByFirmware (none) and an empty IndirectIOAccess.

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

# device mailbox (ati_hd5770_int.h)
MAILBOX = 0x1ff00           # 16 dwords of parameters / results
DOORBELL = 0x1ff7c          # write: table index | dword count << 8

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
    "ASIC_Init":                        (8, 0, 1, 2),
    "DIGxEncoderControl":               (8, 0, 1, 3),
    "EnableCRTCMemReq":                 (4, 0, 1, 1),
    "SetEngineClock":                   (4, 0, 1, 1),
    "SetMemoryClock":                   (4, 0, 1, 1),
    "SetPixelClock":                    (16, 0, 1, 5),
    "AdjustDisplayPll":                 (8, 8, 1, 3),
    "EnableASIC_StaticPwrMgt":          (4, 0, 1, 1),
    "DAC1EncoderControl":               (4, 0, 1, 1),
    "DAC2EncoderControl":               (4, 0, 1, 1),
    "EnableScaler":                     (4, 0, 1, 1),
    "BlankCRTC":                        (8, 0, 1, 1),
    "EnableCRTC":                       (4, 0, 1, 1),
    "EnableVGA_Render":                 (4, 0, 1, 1),
    "SelectCRTC_Source":                (4, 0, 1, 2),
    "EnableGraphSurfaces":              (4, 0, 1, 1),
    "UpdateCRTC_DoubleBufferRegisters": (4, 0, 1, 1),
    "LUT_AutoFill":                     (4, 0, 1, 1),
    "EnableHW_IconCursor":              (4, 0, 1, 1),
    "GetMemoryClock":                   (0, 4, 1, 1),
    "GetEngineClock":                   (0, 4, 1, 1),
    "ComputeMemoryEnginePLL":           (8, 8, 1, 1),
    "EnableSpreadSpectrumOnPPLL":       (8, 0, 1, 2),
    "SetVoltage":                       (4, 0, 1, 1),
    "DAC1OutputControl":                (4, 0, 1, 1),
    "DAC2OutputControl":                (4, 0, 1, 1),
    "MemoryDeviceInit":                 (0, 0, 1, 1),
    "DynamicMemorySettings":            (0, 0, 1, 1),
    "DIG1EncoderControl":               (8, 0, 1, 3),
    "DIG2EncoderControl":               (8, 0, 1, 3),
    "DIG1TransmitterControl":           (8, 0, 1, 3),
    "DIG2TransmitterControl":           (8, 0, 1, 3),
    "ProcessAuxChannelTransaction":     (8, 8, 1, 2),
    "DPEncoderService":                 (8, 8, 1, 1),
}
assert all(n in CMD_NAMES for n in CMDS)

# AtomBIOS opcodes (atom.c): MOVE to REG = 1, MOVE to PS = 2, EOT = 0x5b
OP_MOVE_REG, OP_MOVE_PS, OP_EOT = 0x01, 0x02, 0x5b
ARG_REG, ARG_PS, ARG_IMM = 0, 1, 5

ATOM_ROM_TABLE_PTR = 0x48
ATOM_ATI_MAGIC_PTR = 0x30
ATOM_ATI_MAGIC = b' 761295520'
ROM_HEADER_OFF = 0x80
ROM_HEADER_LEN = 36
PCIR_OFF = 0xa8
PART_NUMBER_PTR = 0x6e      # one byte: offset of the part number string
PART_NUMBER = b'113-JUNIPER-QEMU\x00'
DATA_START = 0xc8


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
    """ATOM_FIRMWARE_INFO_V2_1 (92 bytes); clocks in 10 kHz units; the
    pixel PLLs run from 27 MHz, the reference the display driver assumes."""
    b = struct.pack('<IIIII', 0x00010000,            # firmware revision
                    85000, 120000,                   # default engine/memory clock
                    0, 0)
    b += struct.pack('<IIIII', 100000, 150000,       # max engine/memory PLL output
                     120000,                         # max pixel PLL output
                     0,                              # binary altered info
                     60000)                          # display engine clock
    b += struct.pack('<BBHHHI', 0, 0, 1000, 0, 0, 0) # BL level, VDDC mV, LCD PLL
    b += struct.pack('<I', 60000)                    # min pixel PLL output
    b += struct.pack('<HHHHHH', 100, 10000, 10000,   # engine PLL in min/max, out min
                     100, 10000, 10000)              # memory PLL in min/max, out min
    b += struct.pack('<HHHH', 40000,                 # max pixel clock 400 MHz
                     100, 10000,                     # pixel PLL input 1-100 MHz
                     60000 & 0xffff)                 # min pixel PLL output (low 16)
    b += struct.pack('<HHHH', 0x0003,                # firmware posted, dual CRTC
                     2700, 10000,                    # pixel PLL / memory reference
                     0)                              # DP mode clock from the PPLL
    b += struct.pack('<BBBB', 0, 0, 0, 0)            # memory module ID
    t = hdr(b, 2, 1)
    assert len(t) == 92, len(t)
    return t


# object ids (ObjectID.h): type << 12 | enum << 8 | id
GPU_OBJ = 0x1101                                     # GRAPH_OBJECT_TYPE_GPU
ENCODER_UNIPHY = 0x211E                              # ENCODER_INTERNAL_UNIPHY_ENUM_ID1
CONNECTOR_DVI_D = 0x3103                             # CONNECTOR_SINGLE_LINK_DVI_D_ENUM_ID1
DEVICE_DFP1 = 0x0008                                 # ATOM_DEVICE_DFP1_SUPPORT
ACPI_DFP1 = 0x0210
HPD_GPIO_SHIFT = (0, 8, 16, 24, 26, 28)              # DC_GPIO_HPD_A bits of HPD1-6
HPD_GPIO_ID = (0x14, 0x15, 0x16, 0x17, 0x18, 0x19)


def i2c_id(ddc):
    """ATOM_I2C_ID_CONFIG: HW capable, engine 1, line mux ddc - 1."""
    return 0x90 | (ddc - 1)


def object_header(ddc, hpd):
    """ATOM_OBJECT_HEADER_V3 with its object tables; record and table
    offsets are from the header's start."""
    hsize = 4 + 2 * 7
    body = bytearray()

    def at():
        return hsize + len(body)

    def table(objs):
        """ATOM_OBJECT_TABLE of (id, src/dst offset, record offset)."""
        o = at()
        body.extend(struct.pack('<B3x', len(objs)))
        for oid, sd, rec in objs:
            body.extend(struct.pack('<HHHH', oid, sd, rec, 0))
        return o

    # connector: source is the encoder; records: device tag, I2C, HPD
    conn_sd = at()
    body.extend(struct.pack('<BHB', 1, ENCODER_UNIPHY, 0))
    conn_rec = at()
    body.extend(struct.pack('<BBBBIHH', 4, 12, 1, 0, ACPI_DFP1, DEVICE_DFP1, 0))
    body.extend(struct.pack('<BBBB', 1, 4, i2c_id(ddc), 0xa0))
    body.extend(struct.pack('<BBBB', 2, 4, HPD_GPIO_ID[hpd - 1], 1))
    body.extend(b'\xff\x00')
    # encoder: source is the GPU, no records
    enc_sd = at()
    body.extend(struct.pack('<BHB', 1, GPU_OBJ, 0))
    enc_rec = at()
    body.extend(b'\xff\x00')
    if len(body) & 1:
        body.append(0)
    conn_tab = table([(CONNECTOR_DVI_D, conn_sd, conn_rec)])
    enc_tab = table([(ENCODER_UNIPHY, enc_sd, enc_rec)])
    router_tab = table([])
    # display path: DFP1 = GPU -> UNIPHY -> DVI-D
    path_tab = at()
    body.extend(struct.pack('<BB2x', 1, 1))
    body.extend(struct.pack('<HHHHH', DEVICE_DFP1, 10, CONNECTOR_DVI_D, GPU_OBJ,
                            ENCODER_UNIPHY))
    h = struct.pack('<HHHHHHH', DEVICE_DFP1, conn_tab, router_tab, enc_tab, 0,
                    path_tab, 0)
    return hdr(h + bytes(body), 1, 3)


def gpio_i2c_info():
    """ATOM_GPIO_I2C_INFO: DC_GPIO_DDC1..6 (MASK, A, EN, Y at +0, +4, +8,
    +0xc from 0x6430 + 0x10 n); clock on bit 0, data on bit 8, as the
    device's GPIO model."""
    out = b''
    for line in range(6):
        base = 0x6430 + 0x10 * line
        mask, a, en, y = base >> 2, (base + 4) >> 2, (base + 8) >> 2, (base + 0xc) >> 2
        out += struct.pack('<HHHHHHHH', mask, en, y, a, mask, en, y, a)
        out += struct.pack('<B', i2c_id(line + 1))
        out += struct.pack('<BBBBBBBBBB', 0, 0, 0, 0, 8, 8, 8, 8, 0, 0)
    return hdr(out, 1, 1)


def gpio_pin_lut():
    """ATOM_GPIO_PIN_LUT: HPD1-6 as bits of DC_GPIO_HPD_A."""
    b = b''.join(struct.pack('<HBB', 0x64b4 >> 2, sh, gid)
                 for sh, gid in zip(HPD_GPIO_SHIFT, HPD_GPIO_ID))
    return hdr(b, 1, 1)


# ATOM_PPLIB_CLASSIFICATION_*
PP_CLASS_UI_BATTERY, PP_CLASS_UI_PERFORMANCE, PP_CLASS_BOOT = 1, 5, 0x0008


def power_play_info():
    """PowerPlayInfo in the PPLib layout: ATOM_PPLIB_POWERPLAYTABLE3, three
    states (boot, battery, performance) of three levels each, every level
    the default clocks (850 MHz engine, 1200 MHz memory, 1000 mV), no
    thermal controller, no voltage control, PCIe x16."""
    body_size = 46 - 4                               # TABLE3 after the header
    states = [(PP_CLASS_BOOT, 0), (PP_CLASS_UI_BATTERY, 1),
              (PP_CLASS_UI_PERFORMANCE, 2)]
    state_size, clock_size, nonclock_size = 4, 16, 24
    state_off = 46
    clock_off = state_off + state_size * len(states)
    nonclock_off = clock_off + clock_size
    b = struct.pack('<BBBBB', 0, len(states), state_size, clock_size,
                    nonclock_size)
    b += struct.pack('<HHH', state_off, clock_off, nonclock_off)
    b += struct.pack('<HHH', 0, 0, 46)               # backbias, voltage time, size
    b += struct.pack('<I', 0)                        # platform caps
    b += bytes(8)                                    # thermal controller: none
    b += struct.pack('<HH', 0, 0)                    # boot clock / non-clock info
    b += struct.pack('<BH', 0, 0)                    # custom thermal policies
    b += struct.pack('<HHH', 0, 0, 0)                # format id, fan table, ext header
    assert len(b) == body_size, len(b)
    for i, (_cls, nc) in enumerate(states):          # ATOM_PPLIB_STATE
        b += struct.pack('<BBBB', nc, 0, 0, 0)
    eng, mem = 85000, 120000                         # ATOM_PPLIB_EVERGREEN_CLOCK_INFO
    b += struct.pack('<HBHBHHHI', eng & 0xffff, eng >> 16, mem & 0xffff,
                     mem >> 16, 1000, 0, 0, 0)
    for cls, _nc in states:                          # ATOM_PPLIB_NONCLOCK_INFO
        b += struct.pack('<HBBIBHII5x', cls, 0, 0xff, (16 - 1) << 3, 0, 0, 0, 0)
    t = hdr(b, 5, 1)
    assert len(t) == nonclock_off + nonclock_size * len(states), len(t)
    return t


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


def atom_image(device_id, ss_vendor, ss_id, last, ddc=3, hpd=5):
    rom = bytearray(0x10000)
    rom[0:3] = b'\x55\xaa\x00'
    rom[3:6] = b'\xcb\x90\x90'                       # legacy entry: RETF
    struct.pack_into('<H', rom, 0x18, PCIR_OFF)
    rom[ATOM_ATI_MAGIC_PTR:ATOM_ATI_MAGIC_PTR + len(ATOM_ATI_MAGIC)] = ATOM_ATI_MAGIC
    struct.pack_into('<H', rom, ATOM_ROM_TABLE_PTR, ROM_HEADER_OFF)

    pos = DATA_START

    def put(blob, align=2):
        nonlocal pos
        pos = (pos + align - 1) & ~(align - 1)
        o = pos
        rom[o:o + len(blob)] = blob
        pos += len(blob)
        return o

    part = put(PART_NUMBER)
    assert part < 0x100
    rom[PART_NUMBER_PTR] = part
    msg = put(b'\r\nATI HD 5770 QEMU synthetic AtomBIOS (paravirtual)\r\n\x00')
    cmd_off = []
    for i, name in enumerate(CMD_NAMES):
        ps_in, ps_out, frev, crev = CMDS.get(name, (0, 0, 1, 1))
        cmd_off.append(put(cmd_table(i, ps_in, ps_out, frev, crev)))
    data = {"FirmwareInfo": firmware_info(),
            "Object_Header": object_header(ddc, hpd),
            "PowerPlayInfo": power_play_info(),
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


def build(device_id, ss_vendor, ss_id, efi=None, ddc=3, hpd=5):
    rom = atom_image(device_id, ss_vendor, ss_id, efi is None, ddc, hpd)
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
    pn = u8(PART_NUMBER_PTR)
    assert pn and rom.index(b'\x00', pn) - pn < 31, 'part number'
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
    assert u16(fw) == 92 and (u8(fw + 2), u8(fw + 3)) == (2, 1), 'FirmwareInfo'
    obj = u16(data_master + 4 + 2 * DATA_NAMES.index('Object_Header'))
    assert (u8(obj + 2), u8(obj + 3)) == (1, 3), 'Object_Header revision'
    conn = obj + u16(obj + 6)
    assert u8(conn) >= 1, 'no connector object'
    for i in range(u8(conn)):
        e = conn + 4 + 8 * i
        assert u16(e) >> 12 == 3, 'connector object id'
        sd, rec = obj + u16(e + 2), obj + u16(e + 4)
        assert u8(sd) >= 1 and u16(sd + 1) >> 12 == 2, 'connector source'
        types = []
        while u8(rec) != 0xff:
            assert u8(rec + 1) >= 2 and rec < end, 'record'
            types.append(u8(rec))
            rec += u8(rec + 1)
        assert 4 in types and 1 in types and 2 in types, 'connector records'
    return images


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('out')
    ap.add_argument('--device-id', type=lambda v: int(v, 0), default=0x68b8)
    ap.add_argument('--subsys', default='106b:0000')
    ap.add_argument('--efi', help='UEFI driver appended as a second image')
    ap.add_argument('--ddc', type=int, default=3, choices=range(1, 7),
                    help='DDC line of the DVI connector')
    ap.add_argument('--hpd', type=int, default=5, choices=range(1, 7),
                    help='hot plug pin of the DVI connector')
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
    rom = build(a.device_id, sv, si, efi, a.ddc, a.hpd)
    with open(a.out, 'wb') as f:
        f.write(rom)


if __name__ == '__main__':
    sys.exit(main())
