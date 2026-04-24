//! virtio status handshake and feature negotiation.
//!
//! Runs the `RESET → ACK → DRIVER → feature-neg → FEATURES_OK →
//! DRIVER_OK` dance described in virtio §3.1. Device-specific feature
//! selection is delegated to a caller-provided closure so gpu / blk /
//! … can each nominate their own wanted bits.

use tracing::{error, info};

use crate::mmio::{
    self, Mmio, STATUS_ACKNOWLEDGE, STATUS_DRIVER, STATUS_DRIVER_OK, STATUS_FAILED,
    STATUS_FEATURES_OK,
};

/// Minimum required feature: VIRTIO_F_VERSION_1 (bit 32). We reject
/// legacy-mode devices.
pub const VIRTIO_F_VERSION_1: u64 = 1 << 32;

#[derive(Debug, Clone, Copy)]
pub enum InitError {
    BadMagic,
    UnsupportedVersion(u32),
    DeviceNotPresent,
    FeaturesUnacceptable,
    FailedHandshake,
}

/// Drive the virtio status handshake. `negotiate` receives the 64-bit
/// device-features bitmap and returns the driver-features subset it
/// wants enabled. The returned value MUST be a subset of the device
/// features AND must include `VIRTIO_F_VERSION_1` or the handshake
/// fails.
///
/// # Safety
/// `mmio` must point at a valid virtio-mmio register region; caller
/// must serialize concurrent probes on the same device.
pub unsafe fn init_device<F>(mmio: &Mmio, negotiate: F) -> Result<u64, InitError>
where
    F: FnOnce(u64) -> u64,
{
    unsafe {
        // Magic + version sanity check.
        let magic = mmio.magic();
        if magic != mmio::MAGIC {
            error!("virtio: bad magic {:#x}", magic);
            return Err(InitError::BadMagic);
        }
        let version = mmio.version();
        if version != mmio::VERSION_MODERN {
            error!("virtio: unsupported version {} (need {})", version, mmio::VERSION_MODERN);
            return Err(InitError::UnsupportedVersion(version));
        }
        let device_id = mmio.device_id();
        if device_id == 0 {
            return Err(InitError::DeviceNotPresent);
        }

        // 3.1.1 — reset, then ACKNOWLEDGE + DRIVER.
        mmio.set_status(0);
        mmio.add_status(STATUS_ACKNOWLEDGE);
        mmio.add_status(STATUS_DRIVER);

        // Read both feature words (0 = bits 0..32, 1 = bits 32..64).
        let lo = mmio.device_features(0) as u64;
        let hi = mmio.device_features(1) as u64;
        let device_features = (hi << 32) | lo;
        let driver_features = negotiate(device_features);

        if driver_features & !device_features != 0 {
            error!(
                "virtio: driver requested unoffered features: device={:#x} driver={:#x}",
                device_features, driver_features
            );
            mmio.add_status(STATUS_FAILED);
            return Err(InitError::FeaturesUnacceptable);
        }
        if driver_features & VIRTIO_F_VERSION_1 == 0 {
            error!("virtio: driver did not accept VIRTIO_F_VERSION_1");
            mmio.add_status(STATUS_FAILED);
            return Err(InitError::FeaturesUnacceptable);
        }

        mmio.set_driver_features(0, driver_features as u32);
        mmio.set_driver_features(1, (driver_features >> 32) as u32);
        mmio.add_status(STATUS_FEATURES_OK);

        // Spec requires re-reading status to confirm FEATURES_OK was
        // accepted.
        if mmio.status() & STATUS_FEATURES_OK == 0 {
            error!("virtio: device refused FEATURES_OK");
            mmio.add_status(STATUS_FAILED);
            return Err(InitError::FailedHandshake);
        }

        info!(
            "virtio: device_id={} vendor={:#x} feats={:#x} accepted",
            device_id, mmio.vendor_id(), driver_features
        );

        Ok(driver_features)
    }
}

/// Flip DRIVER_OK after queue setup. Spec §3.1.1 step 8.
pub unsafe fn set_driver_ok(mmio: &Mmio) {
    unsafe { mmio.add_status(STATUS_DRIVER_OK); }
}

/// Read + ack InterruptStatus. Returns which interrupt kinds fired.
#[derive(Debug, Clone, Copy, Default)]
pub struct InterruptBits {
    pub used_ring: bool,
    pub config_change: bool,
}

pub unsafe fn ack_interrupts(mmio: &Mmio) -> InterruptBits {
    unsafe {
        let status = mmio.interrupt_status();
        if status != 0 {
            mmio.interrupt_ack(status);
        }
        InterruptBits {
            used_ring: status & mmio::INT_USED_BUFFER != 0,
            config_change: status & mmio::INT_CONFIG_CHANGE != 0,
        }
    }
}
