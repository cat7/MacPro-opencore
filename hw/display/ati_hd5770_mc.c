/*
 * ATI Radeon HD 5770 (Juniper) memory controller as the engines see it.
 *
 * An engine address ("MC address", 40 bits) resolves in this order:
 * - with accel on and VM context 0 enabled, a page inside
 *   VM_CONTEXT0_PAGE_TABLE_START..END goes through the flat table at
 *   VM_CONTEXT0_PAGE_TABLE_BASE: 64-bit entries, bit 0 valid, bit 1
 *   system page (bus address) else a VRAM page (MC address), address in
 *   bits 39:12. An invalid entry is a fault: reads return zero, writes
 *   are dropped, both counted.
 * - inside the FB window (MC_VM_FB_LOCATION): VRAM.
 * - anything else: a bus address.
 *
 * No TLB is kept, so the invalidate requests of VM_CONTEXTn_REQUEST_RESPONSE
 * complete at once: the response field reads 1 (done).
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/bswap.h"
#include "hw/pci/pci_device.h"
#include "system/dma.h"
#include "system/memory.h"
#include "ati_hd5770_int.h"
#include "ati_hd5770_regs.h"
#include "trace.h"

typedef struct MCSpan {
    bool vram;
    bool fault;
    uint64_t addr;              /* VRAM offset or bus address */
    uint64_t len;               /* bytes valid from addr */
} MCSpan;

static uint8_t *vram_ptr(ATIHD5770State *s)
{
    return memory_region_get_ram_ptr(&s->vram);
}

/* MC address outside the GART: VRAM inside the FB window, else the bus */
static void mc_direct(ATIHD5770State *s, uint64_t mc, MCSpan *sp)
{
    uint64_t off;

    if (ati_hd5770_mc_to_vram(s, mc, &off)) {
        sp->vram = true;
        sp->addr = off;
        sp->len = s->vram_size - off;
    } else {
        sp->vram = false;
        sp->addr = mc;
        sp->len = UINT64_MAX - mc;
    }
    sp->fault = false;
}

static uint64_t pte_read(ATIHD5770State *s, uint64_t mc)
{
    MCSpan sp;
    uint64_t pte = 0;

    mc_direct(s, mc, &sp);
    if (sp.vram) {
        if (sp.len >= 8) {
            pte = ldq_le_p(vram_ptr(s) + sp.addr);
        }
    } else if (pci_dma_read(PCI_DEVICE(s), sp.addr, &pte, 8) == MEMTX_OK) {
        pte = le64_to_cpu(pte);
    } else {
        pte = 0;
    }
    return pte;
}

static void mc_resolve(ATIHD5770State *s, uint64_t mc, MCSpan *sp)
{
    uint64_t page = mc >> 12;
    uint64_t start = s->regs[EG_VM_CONTEXT0_PAGE_TABLE_START >> 2];
    uint64_t end = s->regs[EG_VM_CONTEXT0_PAGE_TABLE_END >> 2];
    uint64_t pte, pa;

    if (!s->accel ||
        !(s->regs[EG_VM_CONTEXT0_CNTL >> 2] & EG_VM_ENABLE_CONTEXT) ||
        page < start || page > end) {
        mc_direct(s, mc, sp);
        return;
    }
    s->mc.gart++;
    pte = pte_read(s, ((uint64_t)s->regs[EG_VM_CONTEXT0_PAGE_TABLE_BASE >> 2]
                       << 12) + (page - start) * 8);
    if (!(pte & EG_PTE_VALID)) {
        sp->fault = true;
        sp->vram = false;
        sp->addr = 0;
        sp->len = 0x1000 - (mc & 0xfff);
        s->mc.gart_invalid++;
        trace_ati_hd5770_gart_fault(mc, pte);
        return;
    }
    pa = (pte & EG_PTE_ADDR_MASK) | (mc & 0xfff);
    if (pte & EG_PTE_SYSTEM) {
        sp->vram = false;
        sp->fault = false;
        sp->addr = pa;
    } else {
        mc_direct(s, pa, sp);
        if (!sp->vram && pa < s->vram_size) {
            /* a VRAM page given as an offset */
            sp->vram = true;
            sp->addr = pa;
        }
    }
    sp->len = 0x1000 - (mc & 0xfff);
}

/*
 * fetch: a read that nothing holds state across, so a wait for the host
 * GPU may release the BQL (ati_hd5770_gpu_sync)
 */
static void mc_access(ATIHD5770State *s, uint64_t mc, void *buf, unsigned len,
                      bool write, bool fetch)
{
    uint8_t *p = buf;

    while (len) {
        MCSpan sp;
        unsigned n;

        mc_resolve(s, mc, &sp);
        n = MIN((uint64_t)len, sp.len);
        if (!n) {
            /* past the end of VRAM */
            sp.fault = true;
            n = len;
        }
        if (sp.fault) {
            if (write) {
                s->mc.fault_writes++;
            } else {
                s->mc.fault_reads++;
                memset(p, 0, n);
            }
        } else if (sp.vram) {
            if (fetch && hd5770_gfx_pending(&s->gfx3d, sp.addr, n)) {
                ati_hd5770_gpu_sync(s);
            }
            hd5770_gfx_touch(&s->gfx3d, sp.addr, n);
            if (write) {
                memcpy(vram_ptr(s) + sp.addr, p, n);
                memory_region_set_dirty(&s->vram, sp.addr, n);
            } else {
                memcpy(p, vram_ptr(s) + sp.addr, n);
            }
        } else {
            if (s->gfx3d.ram_pend) {
                /* queued draws write guest RAM */
                if (fetch) {
                    ati_hd5770_gpu_sync(s);
                } else {
                    hd5770_gfx_sync(&s->gfx3d);
                }
            }
            if (write) {
                pci_dma_write(PCI_DEVICE(s), sp.addr, p, n);
            } else if (pci_dma_read(PCI_DEVICE(s), sp.addr, p, n) !=
                       MEMTX_OK) {
                memset(p, 0, n);
            }
        }
        mc += n;
        p += n;
        len -= n;
    }
}

void ati_hd5770_mc_read(ATIHD5770State *s, uint64_t mc, void *buf,
                        unsigned len)
{
    mc_access(s, mc, buf, len, false, false);
}

void ati_hd5770_mc_fetch(ATIHD5770State *s, uint64_t mc, void *buf,
                         unsigned len)
{
    mc_access(s, mc, buf, len, false, true);
}

void ati_hd5770_mc_write(ATIHD5770State *s, uint64_t mc, const void *buf,
                         unsigned len)
{
    mc_access(s, mc, (void *)buf, len, true, false);
}

uint32_t ati_hd5770_mc_rd32(ATIHD5770State *s, uint64_t mc)
{
    uint8_t b[4];

    mc_access(s, mc, b, 4, false, false);
    return ldl_le_p(b);
}

void ati_hd5770_mc_wr32(ATIHD5770State *s, uint64_t mc, uint32_t val)
{
    uint8_t b[4];

    stl_le_p(b, val);
    mc_access(s, mc, b, 4, true, false);
}

void ati_hd5770_mc_wr64(ATIHD5770State *s, uint64_t mc, uint64_t val)
{
    uint8_t b[8];

    stq_le_p(b, val);
    mc_access(s, mc, b, 8, true, false);
}

/* VM block registers with behaviour; the rest are stored */
bool ati_hd5770_mc_reg_write(ATIHD5770State *s, uint32_t off, uint32_t val)
{
    switch (off) {
    case EG_VM_CONTEXT0_REQUEST_RESPONSE:
    case EG_VM_CONTEXT1_REQUEST_RESPONSE:
        /* the invalidate completes at once */
        s->regs[off >> 2] = (val & ~EG_VM_RESPONSE_MASK) |
            ((val & EG_VM_REQUEST_TYPE_MASK) ? 1u << EG_VM_RESPONSE_SHIFT : 0);
        s->mc.tlb_inval++;
        trace_ati_hd5770_tlb_invalidate(off, val);
        return true;
    case EG_VM_CONTEXT0_CNTL:
    case EG_VM_CONTEXT0_PAGE_TABLE_BASE:
    case EG_VM_CONTEXT0_PAGE_TABLE_START:
    case EG_VM_CONTEXT0_PAGE_TABLE_END:
        s->regs[off >> 2] = val;
        trace_ati_hd5770_vm_setup(off, val);
        return true;
    default:
        return false;
    }
}

/* ---------------------------------------------------------------- */
/* Direct access for the 3D engine                                  */

static uint8_t *gfx_map(void *opaque, uint64_t mc, uint64_t *len, bool write)
{
    ATIHD5770State *s = opaque;
    MCSpan sp;
    HD5770Map *m;
    dma_addr_t plen;
    void *p;

    mc_resolve(s, mc, &sp);
    if (sp.fault || !sp.len) {
        return NULL;
    }
    if (sp.vram) {
        if (sp.addr >= s->vram_size) {
            return NULL;
        }
        *len = MIN(sp.len, s->vram_size - sp.addr);
        return vram_ptr(s) + sp.addr;
    }
    /* system memory: one page at a time, released after the draw */
    plen = MIN(sp.len, 0x1000 - (sp.addr & 0xfff));
    p = pci_dma_map(PCI_DEVICE(s), sp.addr, &plen,
                    write ? DMA_DIRECTION_FROM_DEVICE
                          : DMA_DIRECTION_TO_DEVICE);
    if (!p) {
        return NULL;
    }
    if (s->n_maps == s->max_maps) {
        s->max_maps = MAX(64, s->max_maps * 2);
        s->maps = g_renew(HD5770Map, s->maps, s->max_maps);
    }
    m = &s->maps[s->n_maps++];
    m->p = p;
    m->len = plen;
    m->write = write;
    *len = plen;
    return p;
}

static void gfx_dirty(void *opaque, uint64_t mc, uint64_t len)
{
    ATIHD5770State *s = opaque;
    uint64_t off;

    if (ati_hd5770_mc_to_vram(s, mc, &off) && off < s->vram_size) {
        memory_region_set_dirty(&s->vram, off, MIN(len, s->vram_size - off));
    }
}

void ati_hd5770_mc_unmap_all(ATIHD5770State *s)
{
    uint32_t i;

    for (i = 0; i < s->n_maps; i++) {
        HD5770Map *m = &s->maps[i];

        pci_dma_unmap(PCI_DEVICE(s), m->p, m->len,
                      m->write ? DMA_DIRECTION_FROM_DEVICE
                               : DMA_DIRECTION_TO_DEVICE,
                      m->write ? m->len : 0);
    }
    s->n_maps = 0;
}

void ati_hd5770_mc_gfx_init(ATIHD5770State *s)
{
    s->gfx3d.mem.opaque = s;
    s->gfx3d.mem.map = gfx_map;
    s->gfx3d.mem.dirty = gfx_dirty;
}

/* ---------------------------------------------------------------- */
/* HDP surfaces                                                     */

/*
 * The host data path tiles CPU accesses through BAR 0 that fall into an
 * HDP surface: the bytes of a linear picture of the surface's pitch go
 * to their places in its tiled layout. The controller lays surface n over
 * CRTC n's scanout at mode set (0x2c14 + n * 0x18: first and last byte
 * >> 8, base, INFO, SIZE), so a CPU writing the screen linearly -- the
 * kernel console -- draws on the tiled scanout correctly. A window traps
 * BAR accesses over each surface whose array mode is tiled; linear
 * surfaces need none.
 */
static bool hdp_find(ATIHD5770State *s, uint64_t addr, unsigned *n)
{
    unsigned i;

    for (i = 0; i < EG_HDP_SURFACES; i++) {
        if (addr >= s->hdp_lo[i] && addr < s->hdp_hi[i]) {
            *n = i;
            return true;
        }
    }
    return false;
}

/* VRAM offset of BAR 0 byte addr */
static uint64_t hdp_byte(ATIHD5770State *s, uint64_t addr)
{
    const HD5770Surf *sf;
    uint64_t o, e;
    unsigned n;

    if (!hdp_find(s, addr, &n)) {
        return addr;
    }
    sf = &s->hdp_surf[n];
    o = addr - s->hdp_lo[n];
    e = o / sf->bpe;
    return s->hdp_lo[n] + hd5770_surf_offset_fast(sf, e % sf->pitch,
                                                   e / sf->pitch) +
           o % sf->bpe;
}

static uint64_t hdp_read(void *opaque, hwaddr addr, unsigned size)
{
    ATIHD5770State *s = opaque;
    uint8_t *v = vram_ptr(s);
    uint64_t val = 0;
    unsigned i;

    s->hdp_accesses++;
    ati_hd5770_cp_drain(s);
    for (i = 0; i < size; i++) {
        uint64_t a = hdp_byte(s, addr + i);

        if (a < s->vram_size) {
            hd5770_gfx_touch(&s->gfx3d, a, 1);
            val |= (uint64_t)v[a] << (8 * i);
        }
    }
    return val;
}

static void hdp_write(void *opaque, hwaddr addr, uint64_t val, unsigned size)
{
    ATIHD5770State *s = opaque;
    uint8_t *v = vram_ptr(s);
    unsigned i;

    s->hdp_accesses++;
    ati_hd5770_cp_drain(s);
    for (i = 0; i < size; i++) {
        uint64_t a = hdp_byte(s, addr + i);

        if (a < s->vram_size) {
            hd5770_gfx_touch(&s->gfx3d, a, 1);
            v[a] = val >> (8 * i);
            memory_region_set_dirty(&s->vram, a, 1);
        }
    }
}

static const MemoryRegionOps hdp_ops = {
    .read = hdp_read,
    .write = hdp_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {
        .min_access_size = 1,
        .max_access_size = 8,
        .unaligned = true,
    },
    .impl = {
        .min_access_size = 1,
        .max_access_size = 8,
        .unaligned = true,
    },
};

void ati_hd5770_hdp_init(ATIHD5770State *s)
{
    Object *obj = OBJECT(s);
    uint64_t size = memory_region_size(&s->aper);
    unsigned i;

    memory_region_init(&s->bar0, obj, "ati-hd5770-bar0", size);
    memory_region_add_subregion(&s->bar0, 0, &s->aper);
    memory_region_init_io(&s->hdp_io, obj, &hdp_ops, s, "ati-hd5770-hdp",
                          size);
    /* an access may wait for the engine with the BQL released */
    s->hdp_io.disable_reentrancy_guard = true;
    for (i = 0; i < EG_HDP_SURFACES; i++) {
        memory_region_init_alias(&s->hdp_win[i], obj, "ati-hd5770-hdp-win",
                                 &s->hdp_io, 0, 0x1000);
        memory_region_set_enabled(&s->hdp_win[i], false);
        memory_region_add_subregion_overlap(&s->bar0, 0, &s->hdp_win[i], 1);
    }
}

/* the windows of the tiled surfaces, from the registers */
void ati_hd5770_hdp_update(ATIHD5770State *s)
{
    uint64_t size = memory_region_size(&s->aper);
    uint32_t cfg = s->regs[EG_GB_ADDR_CONFIG >> 2];
    unsigned i;

    memory_region_transaction_begin();
    for (i = 0; i < EG_HDP_SURFACES; i++) {
        const uint32_t *r = &s->regs[(EG_HDP_SURFACE0 +
                                      i * EG_HDP_SURFACE_STRIDE) >> 2];
        uint32_t info = r[EG_HDP_SURF_INFO >> 2];
        uint64_t lo = (uint64_t)r[EG_HDP_SURF_LOWER >> 2] << 8;
        uint64_t hi = ((uint64_t)r[EG_HDP_SURF_UPPER >> 2] << 8) + 0x100;
        HD5770Surf *sf = &s->hdp_surf[i];

        memset(sf, 0, sizeof(*sf));
        /* INFO: array mode 4:1, log2 bytes per element 9:7, then the
         * bank fields encoded as in GRPH_CONTROL */
        sf->mode = (info >> 1) & 0xf;
        sf->bpe = 1u << ((info >> 7) & 7);
        sf->tile_split = 64u << ((info >> 17) & 7);
        sf->banks = 2u << ((info >> 20) & 3);
        sf->bankw = 1u << ((info >> 22) & 3);
        sf->bankh = 1u << ((info >> 24) & 3);
        sf->mtilea = 1u << ((info >> 26) & 3);
        sf->pitch = ((r[EG_HDP_SURF_SIZE >> 2] & 0x7ff) + 1) * 8;
        sf->height = sf->pitch ? (hi - lo) / sf->bpe / sf->pitch : 0;
        hd5770_surf_set_addr_config(sf, cfg ? cfg
                                            : HD5770_GB_ADDR_CONFIG_JUNIPER);
        hd5770_surf_prep(sf);
        if (sf->mode < HD5770_ARRAY_1D_TILED_THIN1 ||
            !hd5770_surf_mode_ok(sf->mode) || !r[EG_HDP_SURF_UPPER >> 2] ||
            hi <= lo || hi > size || hi > s->vram_size) {
            lo = hi = 0;
        }
        s->hdp_lo[i] = lo;
        s->hdp_hi[i] = hi;
        memory_region_set_enabled(&s->hdp_win[i], false);
        if (hi > lo) {
            memory_region_set_alias_offset(&s->hdp_win[i], lo);
            memory_region_set_size(&s->hdp_win[i], hi - lo);
            memory_region_set_address(&s->hdp_win[i], lo);
            memory_region_set_enabled(&s->hdp_win[i], true);
        }
    }
    memory_region_transaction_commit();
}
