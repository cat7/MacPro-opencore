/** @file
  Graphics Output Protocol for the emulated ATI Radeon HD 5770 (Juniper).

  Adapted from AtiR580GopDxe.c.

  SPDX-License-Identifier: BSD-2-Clause-Patent
**/
#include <Uefi.h>
#include <IndustryStandard/Pci.h>
#include <IndustryStandard/Acpi.h>
#include <Protocol/PciIo.h>
#include <Protocol/GraphicsOutput.h>
#include <Protocol/DevicePath.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/UefiLib.h>
#include <Library/BaseMemoryLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/DevicePathLib.h>
#include <Library/DebugLib.h>
#include <Library/FrameBufferBltLib.h>
#include <Library/BaseLib.h>

#define HD5770_VENDOR  0x1002
#define HD5770_DEVICE  0x68B8   /* HD 5770, Juniper XT */

/* DCE 4 CRTC 0 registers */
#define CRTC_CONTROL                    0x6E70
#define GRPH_ENABLE                     0x6800
#define GRPH_CONTROL                    0x6804
#define GRPH_PRIMARY_SURFACE_ADDRESS    0x6810
#define GRPH_PITCH                      0x6818
#define GRPH_X_START                    0x682C
#define GRPH_Y_START                    0x6830
#define GRPH_X_END                      0x6834
#define GRPH_Y_END                      0x6838
#define VIEWPORT_SIZE                   0x6D74
#define MC_VM_FB_LOCATION               0x2024

typedef struct {
  UINT64                                Signature;
  EFI_HANDLE                            Handle;
  EFI_PCI_IO_PROTOCOL                   *PciIo;
  UINT8                                 MmioBar;
  UINT8                                 VramBar;
  UINT64                                OriginalAttributes;
  EFI_GRAPHICS_OUTPUT_PROTOCOL          Gop;
  EFI_GRAPHICS_OUTPUT_PROTOCOL_MODE     Mode;
  EFI_GRAPHICS_OUTPUT_MODE_INFORMATION  Info;
  EFI_DEVICE_PATH_PROTOCOL              *DevicePath;
  FRAME_BUFFER_CONFIGURE                *BltConfig;
  UINTN                                 BltConfigSize;
} HD5770_GOP;

#define HD5770_GOP_SIG  SIGNATURE_64 ('J','U','N','P','G','O','P','0')
#define FROM_GOP(a)   CR (a, HD5770_GOP, Gop, HD5770_GOP_SIG)

STATIC
UINT32
RRead (
  IN HD5770_GOP  *R,
  IN UINT32    Offset
  )
{
  UINT32  Value = 0;

  R->PciIo->Mem.Read (R->PciIo, EfiPciIoWidthUint32, R->MmioBar, Offset, 1, &Value);
  return Value;
}

STATIC
VOID
RWrite (
  IN HD5770_GOP  *R,
  IN UINT32    Offset,
  IN UINT32    Value
  )
{
  R->PciIo->Mem.Write (R->PciIo, EfiPciIoWidthUint32, R->MmioBar, Offset, 1, &Value);
}

STATIC
EFI_STATUS
EFIAPI
RQueryMode (
  IN  EFI_GRAPHICS_OUTPUT_PROTOCOL          *This,
  IN  UINT32                                ModeNumber,
  OUT UINTN                                 *SizeOfInfo,
  OUT EFI_GRAPHICS_OUTPUT_MODE_INFORMATION  **Info
  )
{
  HD5770_GOP  *R = FROM_GOP (This);

  if ((ModeNumber != 0) || (SizeOfInfo == NULL) || (Info == NULL)) {
    return EFI_INVALID_PARAMETER;
  }

  *Info = AllocateCopyPool (sizeof (R->Info), &R->Info);
  if (*Info == NULL) {
    return EFI_OUT_OF_RESOURCES;
  }

  *SizeOfInfo = sizeof (R->Info);
  return EFI_SUCCESS;
}

STATIC
EFI_STATUS
EFIAPI
RSetMode (
  IN EFI_GRAPHICS_OUTPUT_PROTOCOL  *This,
  IN UINT32                        ModeNumber
  )
{
  HD5770_GOP                       *R = FROM_GOP (This);
  EFI_GRAPHICS_OUTPUT_BLT_PIXEL  Black;
  EFI_STATUS                     Status;

  if (ModeNumber != 0) {
    return EFI_UNSUPPORTED;
  }

  if (R->BltConfig == NULL) {
    R->BltConfigSize = 0;
    Status = FrameBufferBltConfigure ((VOID *)(UINTN)R->Mode.FrameBufferBase,
               &R->Info, R->BltConfig, &R->BltConfigSize);
    if (Status == RETURN_BUFFER_TOO_SMALL) {
      R->BltConfig = AllocatePool (R->BltConfigSize);
      if (R->BltConfig == NULL) {
        return EFI_OUT_OF_RESOURCES;
      }
      Status = FrameBufferBltConfigure ((VOID *)(UINTN)R->Mode.FrameBufferBase,
                 &R->Info, R->BltConfig, &R->BltConfigSize);
    }
    if (EFI_ERROR (Status)) {
      return Status;
    }
  }

  ZeroMem (&Black, sizeof (Black));
  FrameBufferBlt (R->BltConfig, &Black, EfiBltVideoFill, 0, 0, 0, 0,
    R->Info.HorizontalResolution, R->Info.VerticalResolution, 0);
  return EFI_SUCCESS;
}

STATIC
EFI_STATUS
EFIAPI
RBlt (
  IN  EFI_GRAPHICS_OUTPUT_PROTOCOL       *This,
  IN  EFI_GRAPHICS_OUTPUT_BLT_PIXEL      *BltBuffer  OPTIONAL,
  IN  EFI_GRAPHICS_OUTPUT_BLT_OPERATION  BltOperation,
  IN  UINTN                              SourceX,
  IN  UINTN                              SourceY,
  IN  UINTN                              DestinationX,
  IN  UINTN                              DestinationY,
  IN  UINTN                              Width,
  IN  UINTN                              Height,
  IN  UINTN                              Delta
  )
{
  HD5770_GOP  *R = FROM_GOP (This);

  if (R->BltConfig == NULL) {
    return EFI_NOT_READY;
  }

  return FrameBufferBlt (R->BltConfig, BltBuffer, BltOperation, SourceX,
           SourceY, DestinationX, DestinationY, Width, Height, Delta);
}

STATIC
EFI_STATUS
EFIAPI
RSupported (
  IN EFI_DRIVER_BINDING_PROTOCOL  *This,
  IN EFI_HANDLE                   Controller,
  IN EFI_DEVICE_PATH_PROTOCOL     *RemainingDevicePath
  )
{
  EFI_STATUS           Status;
  EFI_PCI_IO_PROTOCOL  *PciIo;
  UINT16               Ids[2];

  Status = gBS->OpenProtocol (Controller, &gEfiPciIoProtocolGuid, (VOID **)&PciIo,
                  This->DriverBindingHandle, Controller, EFI_OPEN_PROTOCOL_BY_DRIVER);
  if (EFI_ERROR (Status)) {
    return Status;
  }

  Status = PciIo->Pci.Read (PciIo, EfiPciIoWidthUint16, 0, 2, Ids);
  if (!EFI_ERROR (Status) &&
      ((Ids[0] != HD5770_VENDOR) || (Ids[1] != HD5770_DEVICE))) {
    Status = EFI_UNSUPPORTED;
  }

  gBS->CloseProtocol (Controller, &gEfiPciIoProtocolGuid, This->DriverBindingHandle, Controller);
  return Status;
}

STATIC
EFI_STATUS
EFIAPI
RStart (
  IN EFI_DRIVER_BINDING_PROTOCOL  *This,
  IN EFI_HANDLE                   Controller,
  IN EFI_DEVICE_PATH_PROTOCOL     *RemainingDevicePath
  )
{
  EFI_STATUS                         Status;
  HD5770_GOP                           *R;
  EFI_DEVICE_PATH_PROTOCOL           *ParentPath;
  ACPI_ADR_DEVICE_PATH               AcpiAdr;
  EFI_ACPI_ADDRESS_SPACE_DESCRIPTOR  *Vram;
  UINT64                             Supports;
  UINT32                             Width, Height, Pitch, Surface, FbLocation;
  EFI_PCI_IO_PROTOCOL                *ChildPciIo;

  R = AllocateZeroPool (sizeof (*R));
  if (R == NULL) {
    return EFI_OUT_OF_RESOURCES;
  }
  R->Signature = HD5770_GOP_SIG;

  Status = gBS->OpenProtocol (Controller, &gEfiPciIoProtocolGuid, (VOID **)&R->PciIo,
                  This->DriverBindingHandle, Controller, EFI_OPEN_PROTOCOL_BY_DRIVER);
  if (EFI_ERROR (Status)) {
    FreePool (R);
    return Status;
  }

  R->PciIo->Attributes (R->PciIo, EfiPciIoAttributeOperationGet, 0, &R->OriginalAttributes);
  R->PciIo->Attributes (R->PciIo, EfiPciIoAttributeOperationSupported, 0, &Supports);
  Status = R->PciIo->Attributes (R->PciIo, EfiPciIoAttributeOperationEnable,
             Supports & EFI_PCI_DEVICE_ENABLE, NULL);
  if (EFI_ERROR (Status)) {
    goto Fail;
  }

  /* PciIo BAR indices are sequential; tell the BARs apart by size */
  {
    UINT8                              Index;
    EFI_ACPI_ADDRESS_SPACE_DESCRIPTOR  *Desc;

    R->MmioBar = 0xFF;
    R->VramBar = 0xFF;
    for (Index = 0; Index < 6; Index++) {
      if (EFI_ERROR (R->PciIo->GetBarAttributes (R->PciIo, Index, NULL, (VOID **)&Desc))) {
        continue;
      }
      if (Desc->ResType == ACPI_ADDRESS_SPACE_TYPE_MEM) {
        if ((Desc->AddrLen == SIZE_128KB) && (R->MmioBar == 0xFF)) {
          R->MmioBar = Index;
        } else if ((Desc->AddrLen >= SIZE_16MB) && (R->VramBar == 0xFF)) {
          R->VramBar = Index;
        }
      }
      FreePool (Desc);
    }
    if ((R->MmioBar == 0xFF) || (R->VramBar == 0xFF)) {
      Status = EFI_DEVICE_ERROR;
      goto Fail;
    }
  }

  Status = R->PciIo->GetBarAttributes (R->PciIo, R->VramBar, NULL, (VOID **)&Vram);
  if (EFI_ERROR (Status)) {
    goto Fail;
  }

  /* the mode the card resets to: CRTC 0, linear ARGB8888 */
  Width  = RRead (R, GRPH_X_END) - RRead (R, GRPH_X_START);
  Height = RRead (R, GRPH_Y_END) - RRead (R, GRPH_Y_START);
  Pitch  = RRead (R, GRPH_PITCH) & 0x7FFF;
  if ((Width == 0) || (Height == 0) || (Width > 4096) || (Height > 4096)) {
    Width  = 1024;
    Height = 768;
    RWrite (R, GRPH_X_START, 0);
    RWrite (R, GRPH_Y_START, 0);
    RWrite (R, GRPH_X_END, Width);
    RWrite (R, GRPH_Y_END, Height);
    RWrite (R, VIEWPORT_SIZE, (Width << 16) | Height);
  }
  if (Pitch < Width) {
    Pitch = Width;
    RWrite (R, GRPH_PITCH, Pitch);
  }
  RWrite (R, GRPH_CONTROL, 2);
  RWrite (R, GRPH_ENABLE, 1);
  RWrite (R, CRTC_CONTROL, (RRead (R, CRTC_CONTROL) | 1) & ~(1u << 24));

  /* the surface address is an MC address; VRAM starts at the FB base */
  FbLocation = (RRead (R, MC_VM_FB_LOCATION) & 0xFFFF) << 24;
  Surface    = RRead (R, GRPH_PRIMARY_SURFACE_ADDRESS) - FbLocation;

  DEBUG ((DEBUG_INFO, "AtiHd5770Gop: %ux%u pitch %u fb 0x%lx surface 0x%x\n",
    Width, Height, Pitch, Vram->AddrRangeMin, Surface));

  R->Info.Version              = 0;
  R->Info.HorizontalResolution = Width;
  R->Info.VerticalResolution   = Height;
  R->Info.PixelFormat          = PixelBlueGreenRedReserved8BitPerColor;
  R->Info.PixelsPerScanLine    = Pitch;
  R->Mode.MaxMode              = 1;
  R->Mode.Mode                 = 0;
  R->Mode.Info                 = &R->Info;
  R->Mode.SizeOfInfo           = sizeof (R->Info);
  R->Mode.FrameBufferBase      = Vram->AddrRangeMin + Surface;
  R->Mode.FrameBufferSize      = (UINTN)Pitch * Height * 4;
  FreePool (Vram);

  R->Gop.QueryMode = RQueryMode;
  R->Gop.SetMode   = RSetMode;
  R->Gop.Blt       = RBlt;
  R->Gop.Mode      = &R->Mode;

  Status = RSetMode (&R->Gop, 0);
  if (EFI_ERROR (Status)) {
    goto Fail;
  }

  Status = gBS->HandleProtocol (Controller, &gEfiDevicePathProtocolGuid, (VOID **)&ParentPath);
  if (EFI_ERROR (Status)) {
    goto Fail;
  }
  ZeroMem (&AcpiAdr, sizeof (AcpiAdr));
  AcpiAdr.Header.Type    = ACPI_DEVICE_PATH;
  AcpiAdr.Header.SubType = ACPI_ADR_DP;
  AcpiAdr.ADR            = ACPI_DISPLAY_ADR (1, 0, 0, 1, 0, ACPI_ADR_DISPLAY_TYPE_EXTERNAL_DIGITAL, 0, 0);
  SetDevicePathNodeLength (&AcpiAdr.Header, sizeof (AcpiAdr));
  R->DevicePath = AppendDevicePathNode (ParentPath, (EFI_DEVICE_PATH_PROTOCOL *)&AcpiAdr);
  if (R->DevicePath == NULL) {
    Status = EFI_OUT_OF_RESOURCES;
    goto Fail;
  }

  Status = gBS->InstallMultipleProtocolInterfaces (&R->Handle,
                  &gEfiDevicePathProtocolGuid, R->DevicePath,
                  &gEfiGraphicsOutputProtocolGuid, &R->Gop, NULL);
  if (EFI_ERROR (Status)) {
    goto Fail;
  }

  gBS->OpenProtocol (Controller, &gEfiPciIoProtocolGuid, (VOID **)&ChildPciIo,
    This->DriverBindingHandle, R->Handle, EFI_OPEN_PROTOCOL_BY_CHILD_CONTROLLER);
  return EFI_SUCCESS;

Fail:
  R->PciIo->Attributes (R->PciIo, EfiPciIoAttributeOperationSet, R->OriginalAttributes, NULL);
  gBS->CloseProtocol (Controller, &gEfiPciIoProtocolGuid, This->DriverBindingHandle, Controller);
  FreePool (R);
  return Status;
}

STATIC
EFI_STATUS
EFIAPI
RStop (
  IN EFI_DRIVER_BINDING_PROTOCOL  *This,
  IN EFI_HANDLE                   Controller,
  IN UINTN                        NumberOfChildren,
  IN EFI_HANDLE                   *ChildHandleBuffer
  )
{
  /* the GOP lives as long as boot services */
  return EFI_UNSUPPORTED;
}

STATIC EFI_DRIVER_BINDING_PROTOCOL  mHd5770Binding = {
  RSupported, RStart, RStop, 0x10, NULL, NULL
};

EFI_STATUS
EFIAPI
AtiHd5770GopEntry (
  IN EFI_HANDLE        ImageHandle,
  IN EFI_SYSTEM_TABLE  *SystemTable
  )
{
  return EfiLibInstallDriverBinding (ImageHandle, SystemTable, &mHd5770Binding, ImageHandle);
}
