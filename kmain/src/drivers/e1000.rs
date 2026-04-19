use core::{mem::size_of};

use mem::{round_usize_up};
use mmu::PAGE_SIZE;
use smoltcp::phy::{Checksum, DeviceCapabilities, Medium};
use tracing::{info, warn, error};

use crate::kernel::memmap::virt_to_phys_dmap;

const SW_MTU: usize = 2048;

pub const TX_RING_LEN: usize = 8;
pub const TX_RING_BYTES: usize = round_usize_up(TX_RING_LEN * size_of::<TxDesc>(), PAGE_SIZE);
pub const TX_RING_PAGES: usize = TX_RING_BYTES / PAGE_SIZE;
pub const TX_RING_BUFS_BYTES: usize = round_usize_up(SW_MTU * TX_RING_LEN, PAGE_SIZE);

pub const RX_RING_LEN: usize = 8;
pub const RX_RING_BYTES: usize = round_usize_up(RX_RING_LEN * size_of::<RxDesc>(), PAGE_SIZE);
pub const RX_RING_PAGES: usize = RX_RING_BYTES / PAGE_SIZE;
pub const RX_RING_BUFS_BYTES: usize = round_usize_up(SW_MTU * RX_RING_LEN, PAGE_SIZE);

const CTRL_REG_ADDR: usize = 0;
const STATUS_REG_ADDR: usize = 0x8 / size_of::<u32>();
const EEPROM_READ_REG_ADDR: usize = 0x14 / size_of::<u32>();
const ICR_REG_ADDR: usize = 0xC0 / size_of::<u32>();
const IMS_REG_ADDR: usize = 0xD0 / size_of::<u32>();
const RCTL_REG_ADDR: usize = 0x100 / size_of::<u32>();
const TCTL_REG_ADDR: usize = 0x400 / size_of::<u32>();
const TIPG_REG_ADDR: usize = 0x410 / size_of::<u32>();
const RDBA_REG_ADDR: usize = 0x2800 / size_of::<u32>(); // 64 bits
const RDLEN_REG_ADDR: usize = 0x2808 / size_of::<u32>();
const RDH_REG_ADDR: usize = 0x2810 / size_of::<u32>();
const RDT_REG_ADDR: usize = 0x2818 / size_of::<u32>();
const TDBA_REG_ADDR: usize = 0x3800 / size_of::<u32>(); // 64 bits
const TDLEN_REG_ADDR: usize = 0x3808 / size_of::<u32>();
const TDH_REG_ADDR: usize = 0x3810 / size_of::<u32>();
const TDT_REG_ADDR: usize = 0x3818 / size_of::<u32>();
const MTA_ADDR: usize = 0x5200 / size_of::<u32>();
const RECV_ADDR_TABLE_ADDR: usize = 0x5400 / size_of::<u32>(); // 64 bits * 8

#[repr(packed)]
pub struct E1000Pbuf {
    b: [u8; SW_MTU]
}

// E1000 3.3.3
#[derive(Debug, Clone)]
#[repr(packed)]
pub struct TxDesc {
    addr: u64,
    length: u16,
    cso: u8,
    cmd: u8,
    status: u8,
    css: u8,
    special: u16,
}

// E1000 3.2.3
#[derive(Debug, Clone)]
#[repr(packed)]
pub struct RxDesc {
    addr: u64,
    length: u16,
    csum: u16, 
    status: u8,
    errors: u8,
    special: u16,
}

pub struct E1000 {
    pub(super) bar: *mut u32,

    tx_ring: &'static mut [TxDesc; TX_RING_LEN],
    tx_bufs: &'static mut [E1000Pbuf; TX_RING_LEN],

    rx_next: usize,
    rx_ring: &'static mut [RxDesc; RX_RING_LEN],
    rx_bufs: &'static mut [E1000Pbuf; RX_RING_LEN]
}

impl E1000 {
    /// everything should be id mapped and page aligned
    pub fn new(
        bar: *mut u32,
        tx_ring: &'static mut [TxDesc; TX_RING_LEN], 
        tx_bufs: &'static mut [E1000Pbuf; TX_RING_LEN],
        rx_ring: &'static mut [RxDesc; RX_RING_LEN],
        rx_bufs: &'static mut [E1000Pbuf; RX_RING_LEN])
    -> Self
    {
        for idx in 0..RX_RING_LEN {
            let e = &mut rx_ring[idx];

            // Ring buffers are allocated from kernel_pages (KDMAP VAs); the
            // NIC DMAs by physical address, so translate.
            e.addr = virt_to_phys_dmap(rx_bufs[idx].b.as_mut_ptr() as u64);
            e.csum = 0;
            e.errors = 0;
            e.length = 0;
            e.special = 0;
            e.status = 0;
        }

        for idx in 0..TX_RING_LEN {
            let e = &mut tx_ring[idx];

            e.addr = virt_to_phys_dmap(tx_bufs[idx].b.as_mut_ptr() as u64);
            e.cmd = 0;
            e.cso = 0;
            e.css = 0;
            e.length = 0;
            e.special = 0;
            e.status = 0x1;
        }

        unsafe { core::arch::asm!("fence iorw, iorw"); }

        Self {
            bar,
            tx_ring, tx_bufs,
            rx_next: 0, rx_ring, rx_bufs
        }
    }

    pub fn set_mac(&mut self, mac_idx: usize, mac: [u8; 6]) {
        if mac_idx >= 8 {
            error!("e1000: attempt to set mac{mac_idx} failed: max mac_idx=7");
            return
        }

        unsafe {
            let ral = (mac[0] as u32) | ((mac[1] as u32) << 8) | ((mac[2] as u32) << 16) | ((mac[3] as u32) << 24);
            let rah = 0x8000_0000u32 | (mac[4] as u32) | ((mac[5] as u32) << 8);

            let base = RECV_ADDR_TABLE_ADDR + mac_idx * 2;
            self.bar.add(base).write_volatile(ral);
            self.bar.add(base + 1).write_volatile(rah);
        }
    }

    pub fn read_eeprom(&mut self, addr: u32) -> Result<u16, ()> {
        unsafe {
            let eeprom_read_reg = self.bar.add(EEPROM_READ_REG_ADDR);
            eeprom_read_reg.write_volatile(1 | (addr << 8));

            let timeout = riscv::register::time::read64() + 1_000_000;
            loop {
                let reg = eeprom_read_reg.read_volatile();
                if (reg & 0x10) > 0 {
                    return Ok(((reg & 0xFFFF_0000) >> 16) as u16)
                }

                if riscv::register::time::read64() > timeout {
                    return Err(())
                }
            };
        }
    }

    pub fn read_subsystem_vendor_id(&mut self) -> Result<u16, ()> {
        self.read_eeprom(0xC)
    }

    pub fn read_mac(&mut self) -> Result<[u8; 6], ()> {
        let m0 = self.read_eeprom(0)?;
        let m1 = self.read_eeprom(1)?;
        let m2 = self.read_eeprom(2)?;

        let mut mac = [0u8; 6];
        
        mac[0] = m0 as u8 & 0xFF;
        mac[1] = ((m0 & 0xFF00) >> 8) as u8;

        mac[2] = m1 as u8 & 0xFF;
        mac[3] = ((m1 & 0xFF00) >> 8) as u8;

        mac[4] = m2 as u8 & 0xFF;
        mac[5] = ((m2 & 0xFF00) >> 8) as u8;

        Ok(mac)
    }

    pub fn set_interrupts_enabled(&mut self, mask: u16) {
        unsafe {
            self.bar.add(IMS_REG_ADDR).write_volatile(mask as u32);
            if mask > 0 {
                self.bar.add(ICR_REG_ADDR)
                    .write_volatile(0);
            }
        }
    }

    pub fn read_interrupt_status(&mut self) -> u32 {
        unsafe {
            self.bar.add(ICR_REG_ADDR)
                .read_volatile()
        }
    }

    pub fn init_hw(&mut self, mac: [u8; 6]) -> Result<(), ()> {
        unsafe {
            let svid = 0x8086; //self.read_subsystem_vendor_id()?;

            info!("e1000: svid: {svid:04X?}");

            if svid != 0x8086 {
                return Err(())
            }

            self.bar.add(CTRL_REG_ADDR)
                .write_volatile(0x18000060);

            // not using for now, just zero it
            let mta_base = self.bar.add(MTA_ADDR);
            for mta in 0..128 {
                mta_base.add(mta).write_volatile(0);
            }

            self.set_mac(0, mac);

            info!("e1000: mac: {mac:02X?}");

            // give device rx ring + rx ring len. self.rx_ring is a KDMAP VA;
            // the NIC expects physical, so translate at the boundary.
            let rx_phys = virt_to_phys_dmap(self.rx_ring.as_ptr() as u64);
            self.bar.add(RDBA_REG_ADDR)
                .write_volatile(rx_phys as u32);

            self.bar.add(RDBA_REG_ADDR + 1)
                .write_volatile((rx_phys >> 32) as u32);

            self.bar.add(RDLEN_REG_ADDR)
                .write_volatile(RX_RING_BYTES as u32);

            // give device tx ring + tx ring len
            let tx_phys = virt_to_phys_dmap(self.tx_ring.as_ptr() as u64);
            self.bar.add(TDBA_REG_ADDR)
                .write_volatile(tx_phys as u32);

            self.bar.add(TDBA_REG_ADDR + 1)
                .write_volatile((tx_phys >> 32) as u32);

            self.bar.add(TDLEN_REG_ADDR)
                .write_volatile((self.tx_ring.len() * size_of::<TxDesc>()) as u32);

            const IPG: u32 = 10 | (8 << 10) | (6 << 20);
            self.bar.add(TIPG_REG_ADDR)
                .write_volatile(IPG);

            self.bar.add(RDH_REG_ADDR)
                .write_volatile(0);

            self.bar.add(RDT_REG_ADDR)
                .write_volatile(RX_RING_LEN as u32 - 1);

            self.bar.add(TDH_REG_ADDR)
                .write_volatile(0);

            self.bar.add(TDT_REG_ADDR)
                .write_volatile(0);

            const RX_INT_MASK: u16 = 0x004C;
            const TX_INT_MASK: u16 = 0x0001;

            // set interrupts
            self.set_interrupts_enabled(RX_INT_MASK | TX_INT_MASK);

            let interrupts = self.read_interrupt_status();

            core::arch::asm!("fence iorw, iorw");

            const RX_CTL: u32 = 0x8002;
            self.bar.add(RCTL_REG_ADDR)
                .write_volatile(RX_CTL);

            const TX_CTL: u32 = 0b0110000000000111111000011111010; //0x100000A | (0xF << 4) | (0x40 << 12);
            self.bar.add(TCTL_REG_ADDR)
                .write_volatile(TX_CTL);

            let status = self.bar.add(STATUS_REG_ADDR)
                .read_volatile();

            info!("e1000: status={status:08X?},int={interrupts:08X?}");
        }
        Ok(())
    }

    pub fn get_next_rxtx<'e>(&'e mut self) -> Option<(E1000RxToken<'e>, E1000TxToken<'e>)> {
        unsafe {
            let rindex = self.rx_next;

            if (self.rx_ring[rindex].status & 0x1u8) == 0 {
                return None
            }

            let tindex = self.bar.add(TDT_REG_ADDR).read_volatile() as usize;

            if (self.tx_ring[tindex].status & 0x1u8) == 0 {
                warn!("previous transmission request still in progress");
                return None
            }

            self.rx_next = (rindex + 1) % RX_RING_LEN;

            let rxlen = self.rx_ring[rindex].length as usize;
            let next_tdt = ((tindex + 1) % TX_RING_LEN) as u32;
            let bar = self.bar;

            let rxdesc = ((&mut self.rx_ring[rindex]) as *mut RxDesc).as_mut_unchecked();
            let txdesc = ((&mut self.tx_ring[tindex]) as *mut TxDesc).as_mut_unchecked();
            let rxbuf = core::slice::from_raw_parts_mut(self.rx_bufs[rindex].b.as_mut_ptr(), rxlen);
            let txbuf = core::slice::from_raw_parts_mut(self.tx_bufs[tindex].b.as_mut_ptr(), SW_MTU);

            Some((
                E1000RxToken {
                    bar,
                    buf: rxbuf,
                    desc: rxdesc,
                    next_rdt: rindex as u32
                },
                E1000TxToken {
                    bar,
                    buf: txbuf,
                    desc: txdesc,
                    next_tdt
                }
            ))
        }
    }

    pub fn get_next_tx<'e>(&'e mut self) -> Option<E1000TxToken<'e>> {
        unsafe {
            let tindex = self.bar.add(TDT_REG_ADDR).read_volatile() as usize;

            if (self.tx_ring[tindex].status & 0x1u8) == 0 {
                warn!("previous transmission request still in progress");
                return None
            }

            let next_tdt = ((tindex + 1) % TX_RING_LEN) as u32;

            let desc = ((&mut self.tx_ring[tindex]) as *mut TxDesc).as_mut_unchecked();
            let buf = core::slice::from_raw_parts_mut(self.tx_bufs[tindex].b.as_mut_ptr(), SW_MTU);

            Some(E1000TxToken {
                bar: self.bar,
                buf,
                desc,
                next_tdt
            })
        }
    }

    pub fn get_next_rx<'e>(&'e mut self) -> Option<E1000RxToken<'e>> {
        unsafe {
            let rindex = self.rx_next;

            if (self.rx_ring[rindex].status & 0x1u8) == 0 {
                warn!("previous receive request still in progress");
                return None
            }

            self.rx_next = (rindex + 1) % RX_RING_LEN;

            let len = self.rx_ring[rindex].length as usize;

            let desc = ((&mut self.rx_ring[rindex]) as *mut RxDesc).as_mut_unchecked();
            let buf = core::slice::from_raw_parts_mut(self.rx_bufs[rindex].b.as_mut_ptr(), len);

            Some(E1000RxToken {
                bar: self.bar,
                buf,
                desc,
                next_rdt: rindex as u32
            })
        }
    }
}

pub struct E1000TxToken<'e> {
    bar: *mut u32,
    buf: &'e mut [u8],
    desc: &'e mut TxDesc,
    next_tdt: u32
}

impl<'e> smoltcp::phy::TxToken for E1000TxToken<'e> {
    fn consume<R, F>(self, len: usize, f: F) -> R
        where F: FnOnce(&mut [u8]) -> R
    {
        let r = f(&mut self.buf[..len]);

        self.desc.length = len as u16;
        self.desc.status = 0;
        self.desc.cmd = 0x0B;

        unsafe {
            let icr0 = self.bar.add(ICR_REG_ADDR)
                .read_volatile();

            core::arch::asm!("fence iorw, iorw");

            self.bar.add(TDT_REG_ADDR)
                .write_volatile(self.next_tdt);

            let icr1 = self.bar.add(ICR_REG_ADDR)
                .read_volatile();

            let status = self.bar.add(STATUS_REG_ADDR)
                .read_volatile();

            let rdt = self.bar.add(RDT_REG_ADDR).read_volatile();
            let rdh = self.bar.add(RDH_REG_ADDR).read_volatile();

            info!("e1000: status={:08X?}, icr0={:08X?}, icr1={:08X?}, rdt={rdt:08X?}, rdh={rdh:08X?}", status, icr0, icr1);
        }
        r
    }
}

pub struct E1000RxToken<'e> {
    bar: *mut u32,
    buf: &'e mut [u8],
    desc: &'e mut RxDesc,
    next_rdt: u32
}

impl<'e> smoltcp::phy::RxToken for E1000RxToken<'e> {
    fn consume<R, F>(self, f: F) -> R
        where F: FnOnce(&[u8]) -> R
    {
        let r = f(self.buf);

        self.desc.status = 0;

        unsafe {
            core::arch::asm!("fence iorw, iorw");

            self.bar.add(RDT_REG_ADDR)
                .write_volatile(self.next_rdt);

            self.bar.add(STATUS_REG_ADDR)
                .read_volatile();
        }
        r
    }
}

impl smoltcp::phy::Device for E1000 {
    type RxToken<'e> = E1000RxToken<'e> where Self: 'e;
    type TxToken<'e> = E1000TxToken<'e> where Self: 'e;

    fn capabilities<'e>(&'e self) -> DeviceCapabilities {
        let mut caps = DeviceCapabilities::default();
        caps.medium = Medium::Ethernet;
        caps.max_transmission_unit = 1500;
        caps.checksum.icmpv4 = Checksum::Both;
        caps.checksum.ipv4 = Checksum::Both;
        caps.checksum.tcp = Checksum::Both;
        caps.checksum.udp = Checksum::Both;
        caps
    }

    fn receive<'e>(&'e mut self, _timestamp: smoltcp::time::Instant) -> Option<(Self::RxToken<'e>, Self::TxToken<'e>)> {
        self.get_next_rxtx()
    }

    fn transmit<'e>(&'e mut self, _timestamp: smoltcp::time::Instant) -> Option<Self::TxToken<'e>> {
        self.get_next_tx()
    }
}
