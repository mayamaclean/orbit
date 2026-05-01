//! virtio-mmio register block. Layout per the virtio spec (MMIO v2).
//!
//! All accessors are `unsafe` — the underlying pointer is a device VA
//! that must be kept mapped and kept inside the register region's
//! range; reads/writes use `read_volatile` / `write_volatile`.

/// Little-endian `"virt"`.
pub const MAGIC: u32 = 0x74726976;
/// Legacy MMIO v1. We don't support it; the `virt` machine uses v2.
pub const VERSION_LEGACY: u32 = 1;
/// Modern MMIO v2.
pub const VERSION_MODERN: u32 = 2;

// Register offsets, in bytes.
pub const REG_MAGIC: usize = 0x000;
pub const REG_VERSION: usize = 0x004;
pub const REG_DEVICE_ID: usize = 0x008;
pub const REG_VENDOR_ID: usize = 0x00c;
pub const REG_DEVICE_FEATURES: usize = 0x010;
pub const REG_DEVICE_FEATURES_SEL: usize = 0x014;
pub const REG_DRIVER_FEATURES: usize = 0x020;
pub const REG_DRIVER_FEATURES_SEL: usize = 0x024;
pub const REG_QUEUE_SEL: usize = 0x030;
pub const REG_QUEUE_NUM_MAX: usize = 0x034;
pub const REG_QUEUE_NUM: usize = 0x038;
pub const REG_QUEUE_READY: usize = 0x044;
pub const REG_QUEUE_NOTIFY: usize = 0x050;
pub const REG_INTERRUPT_STATUS: usize = 0x060;
pub const REG_INTERRUPT_ACK: usize = 0x064;
pub const REG_STATUS: usize = 0x070;
pub const REG_QUEUE_DESC_LOW: usize = 0x080;
pub const REG_QUEUE_DESC_HIGH: usize = 0x084;
pub const REG_QUEUE_DRIVER_LOW: usize = 0x090;
pub const REG_QUEUE_DRIVER_HIGH: usize = 0x094;
pub const REG_QUEUE_DEVICE_LOW: usize = 0x0a0;
pub const REG_QUEUE_DEVICE_HIGH: usize = 0x0a4;
pub const REG_CONFIG_GENERATION: usize = 0x0fc;
pub const REG_CONFIG: usize = 0x100;

// Device status bits, written to REG_STATUS.
pub const STATUS_ACKNOWLEDGE: u32 = 1;
pub const STATUS_DRIVER: u32 = 2;
pub const STATUS_DRIVER_OK: u32 = 4;
pub const STATUS_FEATURES_OK: u32 = 8;
pub const STATUS_NEEDS_RESET: u32 = 64;
pub const STATUS_FAILED: u32 = 128;

// InterruptStatus bits.
pub const INT_USED_BUFFER: u32 = 1;
pub const INT_CONFIG_CHANGE: u32 = 2;

/// Thin wrapper over a KMMIO-aliased virtio-mmio register region.
#[derive(Clone, Copy)]
pub struct Mmio {
    base: *mut u8,
}

// SAFETY: the underlying pointer is into KMMIO device memory; reads
// and writes are volatile 32-bit accesses with well-defined device
// semantics. No interior Rust state.
unsafe impl Send for Mmio {}
unsafe impl Sync for Mmio {}

impl Mmio {
    /// # Safety
    /// `base_kva` must be a KMMIO mapping covering at least
    /// `REG_CONFIG + device_config_len` bytes.
    pub const unsafe fn new(base_kva: u64) -> Self {
        Self {
            base: base_kva as *mut u8,
        }
    }

    #[inline]
    unsafe fn r32(&self, off: usize) -> u32 {
        unsafe { (self.base.add(off) as *const u32).read_volatile() }
    }

    #[inline]
    unsafe fn w32(&self, off: usize, val: u32) {
        unsafe {
            (self.base.add(off) as *mut u32).write_volatile(val);
        }
    }

    pub unsafe fn magic(&self) -> u32 {
        unsafe { self.r32(REG_MAGIC) }
    }
    pub unsafe fn version(&self) -> u32 {
        unsafe { self.r32(REG_VERSION) }
    }
    pub unsafe fn device_id(&self) -> u32 {
        unsafe { self.r32(REG_DEVICE_ID) }
    }
    pub unsafe fn vendor_id(&self) -> u32 {
        unsafe { self.r32(REG_VENDOR_ID) }
    }

    pub unsafe fn status(&self) -> u32 {
        unsafe { self.r32(REG_STATUS) }
    }
    pub unsafe fn set_status(&self, v: u32) {
        unsafe {
            self.w32(REG_STATUS, v);
        }
    }
    pub unsafe fn add_status(&self, flag: u32) {
        unsafe {
            let s = self.status();
            self.set_status(s | flag);
        }
    }

    pub unsafe fn device_features(&self, sel: u32) -> u32 {
        unsafe {
            self.w32(REG_DEVICE_FEATURES_SEL, sel);
            self.r32(REG_DEVICE_FEATURES)
        }
    }

    pub unsafe fn set_driver_features(&self, sel: u32, val: u32) {
        unsafe {
            self.w32(REG_DRIVER_FEATURES_SEL, sel);
            self.w32(REG_DRIVER_FEATURES, val);
        }
    }

    pub unsafe fn select_queue(&self, idx: u32) {
        unsafe {
            self.w32(REG_QUEUE_SEL, idx);
        }
    }

    pub unsafe fn queue_num_max(&self) -> u32 {
        unsafe { self.r32(REG_QUEUE_NUM_MAX) }
    }
    pub unsafe fn set_queue_num(&self, n: u32) {
        unsafe {
            self.w32(REG_QUEUE_NUM, n);
        }
    }
    pub unsafe fn set_queue_ready(&self, v: u32) {
        unsafe {
            self.w32(REG_QUEUE_READY, v);
        }
    }

    pub unsafe fn set_queue_desc(&self, pa: u64) {
        unsafe {
            self.w32(REG_QUEUE_DESC_LOW, pa as u32);
            self.w32(REG_QUEUE_DESC_HIGH, (pa >> 32) as u32);
        }
    }
    pub unsafe fn set_queue_driver(&self, pa: u64) {
        unsafe {
            self.w32(REG_QUEUE_DRIVER_LOW, pa as u32);
            self.w32(REG_QUEUE_DRIVER_HIGH, (pa >> 32) as u32);
        }
    }
    pub unsafe fn set_queue_device(&self, pa: u64) {
        unsafe {
            self.w32(REG_QUEUE_DEVICE_LOW, pa as u32);
            self.w32(REG_QUEUE_DEVICE_HIGH, (pa >> 32) as u32);
        }
    }

    pub unsafe fn notify_queue(&self, idx: u32) {
        // Order prior main-memory writes (descriptor ring, avail.idx)
        // before this MMIO store. Without `iorw` in the successor set
        // the device could observe an incremented avail.idx and DMA
        // stale descriptor bytes. `fence rw, iorw` is a superset of
        // what core's SeqCst fence emits on RISC-V (`fence rw, rw`),
        // which does not cover I/O.
        unsafe {
            core::arch::asm!("fence rw, iorw", options(nostack, preserves_flags));
            self.w32(REG_QUEUE_NOTIFY, idx);
        }
    }

    pub unsafe fn interrupt_status(&self) -> u32 {
        unsafe { self.r32(REG_INTERRUPT_STATUS) }
    }
    pub unsafe fn interrupt_ack(&self, bits: u32) {
        unsafe {
            self.w32(REG_INTERRUPT_ACK, bits);
        }
    }

    pub unsafe fn config_generation(&self) -> u32 {
        unsafe { self.r32(REG_CONFIG_GENERATION) }
    }

    /// KMMIO VA of the device-specific config region at offset 0x100.
    pub fn config_base(&self) -> *mut u8 {
        unsafe { self.base.add(REG_CONFIG) }
    }
}
