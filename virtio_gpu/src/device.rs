//! virtio-gpu 2D driver. Single control queue, single scanout, one
//! in-flight command at a time. Synchronous-poll completion — safe to
//! call from boot-time code before IRQs are armed; for runtime calls
//! `k_gpu` serializes access itself so blocking briefly is fine.

use tracing::{error, info, warn};
use virtio::mmio::Mmio;
use virtio::queue::{Buf, Virtqueue, VirtqBacking};
use virtio::transport::{self, InitError};

use crate::proto::*;

pub const CTRL_QUEUE: u32 = 0;

pub const REQ_OFFSET: usize = 0;
pub const REQ_MAX: usize = 2048;
pub const RESP_OFFSET: usize = 2048;
pub const RESP_MAX: usize = 2048;
pub const ARENA_SIZE: usize = REQ_MAX + RESP_MAX;

#[derive(Debug)]
pub enum GpuError {
    Init(InitError),
    QueueFull,
    DeviceError(u32),
    NoCompletion,
    ArenaTooSmall,
}

impl From<InitError> for GpuError {
    fn from(e: InitError) -> Self { GpuError::Init(e) }
}

pub struct DisplayInfo {
    pub width: u32,
    pub height: u32,
}

/// Backing handed to [`Gpu::new`]. Arena is a single contiguous region
/// holding request + response slots; caller allocates from
/// `kernel_pages::alloc_kdmap` (or equivalent) and hands in both the
/// device-visible PA and the kernel-writable VA.
pub struct GpuBacking {
    pub mmio: Mmio,
    pub ctrl: VirtqBacking,
    pub arena_pa: u64,
    pub arena_kva: *mut u8,
    pub arena_size: usize,
}

pub struct Gpu {
    mmio: Mmio,
    ctrl: Virtqueue,
    arena_pa: u64,
    arena_kva: *mut u8,
    next_resource_id: u32,
}

impl Gpu {
    /// Run the virtio status + feature handshake, program the control
    /// queue, and flip DRIVER_OK. After this returns the device is
    /// ready for commands.
    ///
    /// # Safety
    /// - `mmio` must alias a live virtio-mmio register region for a
    ///   device whose `device_id == 16`.
    /// - `ctrl` backing must be zero-initialized with the required
    ///   alignments and exclusive to this queue.
    /// - `arena_kva` / `arena_pa` must cover at least `ARENA_SIZE`
    ///   contiguous bytes.
    pub unsafe fn new(backing: GpuBacking) -> Result<Self, GpuError> {
        if backing.arena_size < ARENA_SIZE {
            return Err(GpuError::ArenaTooSmall);
        }

        let mmio = backing.mmio;
        let ctrl = unsafe { Virtqueue::new(backing.ctrl) };

        unsafe {
            transport::init_device(&mmio, |dev| {
                // Accept only VIRTIO_F_VERSION_1. No 3D, no EDID, no
                // optimization features — MVP wants the simplest
                // negotiation that works.
                dev & transport::VIRTIO_F_VERSION_1
            })?;

            mmio.select_queue(CTRL_QUEUE);
            let qmax = mmio.queue_num_max();
            if qmax < ctrl.size() as u32 {
                warn!(
                    "virtio-gpu: QueueNumMax={} smaller than requested {}",
                    qmax, ctrl.size()
                );
                return Err(GpuError::Init(InitError::FailedHandshake));
            }
            mmio.set_queue_num(ctrl.size() as u32);
            mmio.set_queue_desc(ctrl.desc_pa());
            mmio.set_queue_driver(ctrl.avail_pa());
            mmio.set_queue_device(ctrl.used_pa());
            mmio.set_queue_ready(1);

            transport::set_driver_ok(&mmio);
        }

        info!("virtio-gpu: device ready (qsize={})", ctrl.size());

        Ok(Self {
            mmio,
            ctrl,
            arena_pa: backing.arena_pa,
            arena_kva: backing.arena_kva,
            next_resource_id: 1,
        })
    }

    pub fn next_resource_id(&mut self) -> u32 {
        let id = self.next_resource_id;
        self.next_resource_id += 1;
        id
    }

    /// Submit `req`, block until completion, return the response
    /// struct the device wrote.
    unsafe fn roundtrip<Req: Copy, Resp: Copy + Default>(
        &mut self,
        req: Req,
    ) -> Result<Resp, GpuError> {
        assert!(core::mem::size_of::<Req>() <= REQ_MAX);
        assert!(core::mem::size_of::<Resp>() <= RESP_MAX);

        unsafe {
            let req_kva = self.arena_kva.add(REQ_OFFSET) as *mut Req;
            let resp_kva = self.arena_kva.add(RESP_OFFSET) as *mut Resp;

            // Zero the response slot so a misbehaving device can't
            // hand us back stale bytes from the last command.
            core::ptr::write_bytes(resp_kva, 0, 1);
            req_kva.write_volatile(req);

            self.ctrl.push_chain(&[
                Buf {
                    pa: self.arena_pa + REQ_OFFSET as u64,
                    len: core::mem::size_of::<Req>() as u32,
                    write: false,
                },
                Buf {
                    pa: self.arena_pa + RESP_OFFSET as u64,
                    len: core::mem::size_of::<Resp>() as u32,
                    write: true,
                },
            ]).map_err(|_| GpuError::QueueFull)?;

            self.mmio.notify_queue(CTRL_QUEUE);

            // Poll until completion. Commands are microsecond-scale
            // against QEMU so a bounded spin is fine.
            for _ in 0..10_000_000 {
                if self.ctrl.pop_used().is_some() {
                    return Ok(resp_kva.read_volatile());
                }
                core::hint::spin_loop();
            }

            error!("virtio-gpu: command timed out");
            Err(GpuError::NoCompletion)
        }
    }

    pub unsafe fn get_display_info(&mut self) -> Result<DisplayInfo, GpuError> {
        let req = CtrlHdr {
            ty: CMD_GET_DISPLAY_INFO,
            ..CtrlHdr::default()
        };
        let resp: RespDisplayInfo = unsafe { self.roundtrip(req)? };
        if resp.hdr.ty != RESP_OK_DISPLAY_INFO {
            return Err(GpuError::DeviceError(resp.hdr.ty));
        }
        let first = resp.pmodes[0];
        if first.enabled == 0 {
            error!("virtio-gpu: scanout 0 disabled");
            return Err(GpuError::DeviceError(0));
        }
        Ok(DisplayInfo {
            width: first.r.width,
            height: first.r.height,
        })
    }

    pub unsafe fn create_2d_resource(
        &mut self,
        resource_id: u32,
        width: u32,
        height: u32,
        format: u32,
    ) -> Result<(), GpuError> {
        let req = ResourceCreate2d {
            hdr: CtrlHdr { ty: CMD_RESOURCE_CREATE_2D, ..CtrlHdr::default() },
            resource_id,
            format,
            width,
            height,
        };
        let resp: CtrlHdr = unsafe { self.roundtrip(req)? };
        check_ok(resp.ty)
    }

    pub unsafe fn attach_backing(
        &mut self,
        resource_id: u32,
        backing_pa: u64,
        backing_len: u32,
    ) -> Result<(), GpuError> {
        let req = ResourceAttachBacking {
            hdr: CtrlHdr { ty: CMD_RESOURCE_ATTACH_BACKING, ..CtrlHdr::default() },
            resource_id,
            nr_entries: 1,
            entry: MemEntry {
                addr: backing_pa,
                length: backing_len,
                _padding: 0,
            },
        };
        let resp: CtrlHdr = unsafe { self.roundtrip(req)? };
        check_ok(resp.ty)
    }

    pub unsafe fn set_scanout(
        &mut self,
        scanout_id: u32,
        resource_id: u32,
        width: u32,
        height: u32,
    ) -> Result<(), GpuError> {
        let req = SetScanout {
            hdr: CtrlHdr { ty: CMD_SET_SCANOUT, ..CtrlHdr::default() },
            r: Rect { x: 0, y: 0, width, height },
            scanout_id,
            resource_id,
        };
        let resp: CtrlHdr = unsafe { self.roundtrip(req)? };
        check_ok(resp.ty)
    }

    pub unsafe fn transfer_to_host_2d(
        &mut self,
        resource_id: u32,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> Result<(), GpuError> {
        let req = TransferToHost2d {
            hdr: CtrlHdr { ty: CMD_TRANSFER_TO_HOST_2D, ..CtrlHdr::default() },
            r: Rect { x, y, width, height },
            offset: 0,
            resource_id,
            _padding: 0,
        };
        let resp: CtrlHdr = unsafe { self.roundtrip(req)? };
        check_ok(resp.ty)
    }

    pub unsafe fn flush(
        &mut self,
        resource_id: u32,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> Result<(), GpuError> {
        let req = ResourceFlush {
            hdr: CtrlHdr { ty: CMD_RESOURCE_FLUSH, ..CtrlHdr::default() },
            r: Rect { x, y, width, height },
            resource_id,
            _padding: 0,
        };
        let resp: CtrlHdr = unsafe { self.roundtrip(req)? };
        check_ok(resp.ty)
    }
}

fn check_ok(ty: u32) -> Result<(), GpuError> {
    if ty == RESP_OK_NODATA {
        Ok(())
    } else {
        Err(GpuError::DeviceError(ty))
    }
}
