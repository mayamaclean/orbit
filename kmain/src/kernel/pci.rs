use alloc::vec::Vec;

use tracing::info;

#[derive(Debug, Clone, Copy)]
pub struct PciDevice {
    pub address: usize,
    pub device_id: u16,
    pub vendor_id: u16,
    pub status: u16,
    pub command: u16,
    pub class: u32,
    pub rev: u8,
    pub htype: u8,
    pub cacheline: u8
}

impl PciDevice {
    pub fn from_address(address: usize) -> Self {
        unsafe {
            let base = address as *mut u32;

            let dvid = base.read_volatile();
            let device_id = ((0xFFFF_0000 & dvid) >> 16) as u16;
            let vendor_id = (0xFFFF & dvid) as u16;

            base.add(1).write_volatile(0xFF);
            let statcom = base.add(1).read_volatile();
            let status = ((0xFFFF_0000 & statcom) >> 16) as u16;
            let command = (0xFFFF & statcom) as u16;

            let classrev = base.add(2).read_volatile();
            let class = (0xFFFF_FF00 & classrev) >> 8;
            let rev = (0xFF & classrev) as u8;

            let header = base.add(3).read_volatile();
            let htype = ((0x00FF_0000 & header) >> 16) as u8;

            base.add(3).write_volatile(header | 0x10);
            let header = base.add(3).read_volatile();
            let cacheline = (0xFF & header) as u8;

            Self {
                address,
                device_id, vendor_id,
                status, command,
                class, rev,
                htype, cacheline
            }
        }
    }

    pub fn get_bar_size(&self, bar: usize) -> usize {
        unsafe {
            let bar_base = (self.address as *mut u32).add(4);
            
            let bar_ptr = bar_base.add(bar);
            bar_ptr.write_volatile(0xFFFF_FFFF);
            (!(bar_ptr.read_volatile() & 0xFFFF_FFF0)).saturating_add(1) as usize
        }
    }

    pub fn print_info(&self) {
        unsafe {
            info!("vid:{:04X?},did:{:04X?},status:{:04X?},command:{:04X?},class:{:06X?},rev:{:02?},htype:{:02X?},cacheline:{:02X?}",
                self.vendor_id,
                self.device_id,
                self.status,
                self.command,
                self.class,
                self.rev,
                self.htype,
                self.cacheline);

            let base = self.address as *mut u32;
            if self.htype == 0 {
                let bar_base = base.add(4);
                for bar_num in 0..6 {
                    let bar = bar_base.add(bar_num);
                    
                    let bar_orig = bar.read_volatile();
                    
                    bar.write_volatile(0xFFFF_FFFF);
                    let bar_len = (!(bar.read_volatile() & 0xFFFF_FFF0)).saturating_add(1);
                    let bar_val = bar.read_volatile();

                    bar.write_volatile(bar_orig);

                    info!("\tbar{bar_num} v=0x{bar_val:08X?},l=0x{bar_len:08X?}");
                }
            }
        }
    }

    pub fn write_bar(&self, bar: usize, val: u32) {
        unsafe {
            let base = self.address as *mut u32;
            let bar = base.add(4 + bar);
            bar.write_volatile(val);
        }
    }
}

pub fn scan_pci(base: usize, interests: &[(u16, u16)]) -> Vec<PciDevice> {
    let mut ret = Vec::new();

    for bus in 0..256 {
        for dev in 0..32 {
            for func in 0..8 {
                // ECAM Address Formula: Base + (Bus << 20) | (Dev << 15) | (Func << 12)
                let address = base | 
                            ((bus as usize) << 20) | 
                            ((dev as usize) << 15) | 
                            ((func as usize) << 12);

                let device = PciDevice::from_address(address);
                if interests.contains(&(device.vendor_id, device.device_id)) {
                    ret.push(device);
                }
            }
        }
    }
    ret
}