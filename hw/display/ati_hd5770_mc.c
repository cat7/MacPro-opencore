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

static void mc_access(ATIHD5770State *s, uint64_t mc, void *buf, unsigned len,
                      bool write)
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
            if (write) {
                memcpy(vram_ptr(s) + sp.addr, p, n);
                memory_region_set_dirty(&s->vram, sp.addr, n);
            } else {
                memcpy(p, vram_ptr(s) + sp.addr, n);
            }
        } else if (write) {
            pci_dma_write(PCI_DEVICE(s), sp.addr, p, n);
        } else if (pci_dma_read(PCI_DEVICE(s), sp.addr, p, n) != MEMTX_OK) {
            memset(p, 0, n);
        }
        mc += n;
        p += n;
        len -= n;
    }
}

void ati_hd5770_mc_read(ATIHD5770State *s, uint64_t mc, void *buf,
                        unsigned len)
{
    mc_access(s, mc, buf, len, false);
}

void ati_hd5770_mc_write(ATIHD5770State *s, uint64_t mc, const void *buf,
                         unsigned len)
{
    mc_access(s, mc, (void *)buf, len, true);
}

uint32_t ati_hd5770_mc_rd32(ATIHD5770State *s, uint64_t mc)
{
    uint8_t b[4];

    mc_access(s, mc, b, 4, false);
    return ldl_le_p(b);
}

void ati_hd5770_mc_wr32(ATIHD5770State *s, uint64_t mc, uint32_t val)
{
    uint8_t b[4];

    stl_le_p(b, val);
    mc_access(s, mc, b, 4, true);
}

void ati_hd5770_mc_wr64(ATIHD5770State *s, uint64_t mc, uint64_t val)
{
    uint8_t b[8];

    stq_le_p(b, val);
    mc_access(s, mc, b, 8, true);
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
