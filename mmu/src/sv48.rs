use core::sync::atomic::{self as atomic, AtomicU64};
use atomic::Ordering;

type BaseUnit = atomic::AtomicU64;

#[repr(transparent)]
#[derive(Debug)]
pub struct VirtAddr {
    pub(crate) a: BaseUnit
}

impl VirtAddr {
    pub const PAGE_OFFSET_MASK: u64 = 0xFFF;
    pub const VIRT_PAGE_NUM_MASK: u64 = 0x1FF;

    const VPN_OFFSETS: [u64; 4] = [
        12, 21, 30, 39
    ];

    pub const fn new(raw: u64) -> Self {
        Self {
            a: AtomicU64::new(raw)
        }
    }

    pub fn copy(&self) -> Self {
        Self {
            a: BaseUnit::new(self.get_raw())
        }
    }
    
    #[inline(always)]
    pub fn get_raw(&self) -> u64 {
        self.a.load(Ordering::Acquire)
    }

    #[inline(always)]
    pub fn set_raw(&self, vaddr: u64) {
        self.a.store(vaddr, Ordering::Release);
    }

    pub fn page_offset(&self) -> u64 {
        self.get_raw() & Self::PAGE_OFFSET_MASK
    }

    pub fn set_page_offset(&self, offset: u16) {
        let o = offset as u64 & Self::PAGE_OFFSET_MASK;
        let mut c = self.get_raw();
        c &= !Self::PAGE_OFFSET_MASK;
        c |= o;
        self.set_raw(c);
    }

    pub fn vpn_n(&self, n: usize) -> u64 {
        let i = Self::VPN_OFFSETS[n];
        let m = Self::VIRT_PAGE_NUM_MASK << i as u64;
        (self.get_raw() & m) >> Self::VPN_OFFSETS[n]
    }

    pub fn vpn0(&self) -> u64 {
        let i = Self::VPN_OFFSETS[0];
        let m = Self::VIRT_PAGE_NUM_MASK << i as u64;
        (self.get_raw() & m) >> Self::VPN_OFFSETS[0]
    }

    pub fn vpn1(&self) -> u64 {
        let i = Self::VPN_OFFSETS[1];
        let m = Self::VIRT_PAGE_NUM_MASK << i as u64;
        (self.get_raw() & m) >> Self::VPN_OFFSETS[1]
    }

    pub fn vpn2(&self) -> u64 {
        let i = Self::VPN_OFFSETS[2];
        let m = Self::VIRT_PAGE_NUM_MASK << i as u64;
        (self.get_raw() & m) >> Self::VPN_OFFSETS[2]
    }

    pub fn vpn3(&self) -> u64 {
        let i = Self::VPN_OFFSETS[3];
        let m = Self::VIRT_PAGE_NUM_MASK << i as u64;
        (self.get_raw() & m) >> Self::VPN_OFFSETS[3]
    }

    pub fn set_vpn_n(&self, n: usize, vpn: u64) {
        let i = Self::VPN_OFFSETS[n];
        let m = Self::VIRT_PAGE_NUM_MASK << i as u64;
        let mut c = self.get_raw();
        c &= !m;
        c |= (vpn as u64 & Self::VIRT_PAGE_NUM_MASK) << i;
        self.set_raw(c);
    }

    pub fn set_vpn0(&self, vpn: u16) {
        let i = Self::VPN_OFFSETS[0];
        let m = Self::VIRT_PAGE_NUM_MASK << i as u64;
        let mut c = self.get_raw();
        c &= !m;
        c |= (vpn as u64 & Self::VIRT_PAGE_NUM_MASK) << i;
        self.set_raw(c);
    }

    pub fn set_vpn1(&self, vpn: u16) {
        let i = Self::VPN_OFFSETS[1];
        let m = Self::VIRT_PAGE_NUM_MASK << i as u64;
        let mut c = self.get_raw();
        c &= !m;
        c |= (vpn as u64 & Self::VIRT_PAGE_NUM_MASK) << i;
        self.set_raw(c);
    }

    pub fn set_vpn2(&self, vpn: u16) {
        let i = Self::VPN_OFFSETS[2];
        let m = Self::VIRT_PAGE_NUM_MASK << i as u64;
        let mut c = self.get_raw();
        c &= !m;
        c |= (vpn as u64 & Self::VIRT_PAGE_NUM_MASK) << i;
        self.set_raw(c);
    }

    pub fn set_vpn3(&self, vpn: u32) {
        let i = Self::VPN_OFFSETS[3];
        let m = Self::VIRT_PAGE_NUM_MASK << i as u64;
        let mut c = self.get_raw();
        c &= !m;
        c |= (vpn as u64 & Self::VIRT_PAGE_NUM_MASK) << i;
        self.set_raw(c);
    }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct PhysAddr {
    pub(crate) a: BaseUnit
}

impl PhysAddr {
    pub const PAGE_OFFSET_MASK: u64 = 0xFFF;
    pub const PHYS_PAGE_NUM_MASK: u64 = 0x1FF;
    pub const PHYS_PAGE_NUM_MASK3: u64 = 0x1FFFF;

    pub const PPN_OFFSETS: [u64; 4] = [
        12, 21, 30, 39
    ];

    pub const fn new(raw: u64) -> Self {
        Self {
            a: AtomicU64::new(raw)
        }
    }

    pub fn copy(&self) -> Self {
        Self {
            a: BaseUnit::new(self.get_raw())
        }
    }

    #[inline(always)]
    pub fn get_raw(&self) -> u64 {
        self.a.load(Ordering::Acquire)
    }

    #[inline(always)]
    pub fn set_raw(&self, paddr: u64) {
        self.a.store(paddr, Ordering::Release);
    }

    pub fn page_offset(&self) -> u64 {
        self.get_raw() & Self::PAGE_OFFSET_MASK
    }

    pub fn set_page_offset(&self, offset: u16) {
        let o = offset as u64 & Self::PAGE_OFFSET_MASK;
        let mut c = self.get_raw();
        c &= !Self::PAGE_OFFSET_MASK;
        c |= o;
        self.set_raw(c);
    }

    pub fn ppn(&self) -> u64 {
        (self.get_raw() & !Self::PAGE_OFFSET_MASK) >> Self::PPN_OFFSETS[0]
    }

    pub fn set_ppn(&self, ppn: u64) {
        let mut c = self.get_raw();
        c &= Self::PAGE_OFFSET_MASK;
        c |= ppn << Self::PPN_OFFSETS[0];
        self.set_raw(c);
    }

    pub fn ppn_n(&self, n: usize) -> u64 {
        let i = Self::PPN_OFFSETS[n];
        let pm = if n == 3 { Self::PHYS_PAGE_NUM_MASK3 } else { Self::PHYS_PAGE_NUM_MASK };
        let m = pm << i as u64;
        (self.get_raw() & m) >> Self::PPN_OFFSETS[n]
    }

    pub fn ppn0(&self) -> u64 {
        let i = Self::PPN_OFFSETS[0];
        let m = Self::PHYS_PAGE_NUM_MASK << i as u64;
        (self.get_raw() & m) >> Self::PPN_OFFSETS[0]
    }

    pub fn ppn1(&self) -> u64 {
        let i = Self::PPN_OFFSETS[1];
        let m = Self::PHYS_PAGE_NUM_MASK << i as u64;
        (self.get_raw() & m) >> Self::PPN_OFFSETS[1]
    }

    pub fn ppn2(&self) -> u64 {
        let i = Self::PPN_OFFSETS[2];
        let m = Self::PHYS_PAGE_NUM_MASK << i as u64;
        (self.get_raw() & m) >> Self::PPN_OFFSETS[2]
    }

    pub fn ppn3(&self) -> u64 {
        let i = Self::PPN_OFFSETS[3];
        let m = Self::PHYS_PAGE_NUM_MASK << i as u64;
        (self.get_raw() & m) >> Self::PPN_OFFSETS[3]
    }

    pub fn set_ppn_n(&self, n: usize, ppn: u16) {
        let i = Self::PPN_OFFSETS[n];
        let pm = if n == 3 { Self::PHYS_PAGE_NUM_MASK3 } else { Self::PHYS_PAGE_NUM_MASK };
        let m = pm << i as u64;
        let mut c = self.get_raw();
        c &= !m;
        c |= (ppn as u64 & Self::PHYS_PAGE_NUM_MASK) << i;
        self.set_raw(c);
    }

    pub fn set_ppn0(&self, ppn: u16) {
        let i = Self::PPN_OFFSETS[0];
        let m = Self::PHYS_PAGE_NUM_MASK << i as u64;
        let mut c = self.get_raw();
        c &= !m;
        c |= (ppn as u64 & Self::PHYS_PAGE_NUM_MASK) << i;
        self.set_raw(c);
    }

    pub fn set_ppn1(&self, ppn: u16) {
        let i = Self::PPN_OFFSETS[1];
        let m = Self::PHYS_PAGE_NUM_MASK << i as u64;
        let mut c = self.get_raw();
        c &= !m;
        c |= (ppn as u64 & Self::PHYS_PAGE_NUM_MASK) << i;
        self.set_raw(c);
    }

    pub fn set_ppn2(&self, ppn: u16) {
        let i = Self::PPN_OFFSETS[2];
        let m = Self::PHYS_PAGE_NUM_MASK << i as u64;
        let mut c = self.get_raw();
        c &= !m;
        c |= (ppn as u64 & Self::PHYS_PAGE_NUM_MASK) << i;
        self.set_raw(c);
    }

    pub fn set_ppn3(&self, ppn: u32) {
        let i = Self::PPN_OFFSETS[3];
        let m = Self::PHYS_PAGE_NUM_MASK3 << i as u64;
        let mut c = self.get_raw();
        c &= !m;
        c |= (ppn as u64 & Self::PHYS_PAGE_NUM_MASK3) << i;
        self.set_raw(c);
    }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct PageTableEntry {
    pub(crate) e: BaseUnit
}

impl PageTableEntry {
    pub const STATUS_BITS_MASK: u64 = 0x3FF;

    pub const PPN_OFFSETS: [u64; 4] = [
        10, 19, 28, 37
    ];

    pub const fn new(raw: u64) -> Self {
        Self {
            e: AtomicU64::new(raw)
        }
    }

    #[inline(always)]
    pub fn get_raw(&self) -> u64 {
        self.e.load(Ordering::Acquire)
    }

    #[inline(always)]
    pub fn set_raw(&self, e: u64) {
        self.e.store(e, Ordering::Release);
    }

    pub fn get_ppn(&self) -> u64 {
        let r = self.get_raw();
        r & 0x3F_FFFF_FFFF_FC00
    }

    fn get_bit(&self, bit: u64) -> bool {
        let m = 1 << bit;
        (self.get_raw() & m) > 0
    }

    fn set_bit(&self, bit: u64, b: bool) {
        let m = 1 << bit;
        let mut c = self.get_raw();
        c &= !m;
        c |= (b as u64) << bit;
        self.set_raw(c);
    }
    
    pub fn is_valid(&self) -> bool {
        self.get_bit(0)
    }

    pub fn set_valid(&self, valid: bool) {
        self.set_bit(0, valid);
    }

    pub fn is_readable(&self) -> bool {
        self.get_bit(1)
    }

    pub fn set_readable(&self, readable: bool) {
        self.set_bit(1, readable);
    }

    pub fn is_writeable(&self) -> bool {
        self.get_bit(2)
    }

    pub fn set_writeable(&self, writeable: bool) {
        self.set_bit(2, writeable);
    }

    pub fn is_executable(&self) -> bool {
        self.get_bit(3)
    }

    pub fn set_executable(&self, executable: bool) {
        self.set_bit(3, executable);
    }

    pub fn is_leaf(&self) -> bool {
        let s = self.get_raw();
        (s & 0xE) > 0
    }

    pub fn is_user_page(&self) -> bool {
        self.get_bit(4)
    }

    pub fn set_user_page(&self, user_page: bool) {
        self.set_bit(4, user_page);
    }

    pub fn is_global_page(&self) -> bool {
        self.get_bit(5)
    }

    pub fn set_global_page(&self, global_page: bool) {
        self.set_bit(5, global_page);
    }

    pub fn was_accessed(&self) -> bool {
        self.get_bit(6)
    }

    pub fn set_accessed(&self, accessed: bool) {
        self.set_bit(6, accessed);
    }

    pub fn is_dirty(&self) -> bool {
        self.get_bit(7)
    }

    pub fn set_dirty(&self, dirty: bool) {
        self.set_bit(7, dirty);
    }

    /// get the 2 rsw bits
    pub fn get_supervisor_bits(&self) -> u8 {
        const M: u64 = 3 << 8;
        ((self.get_raw() & M) >> 8) as u8
    }

    /// set the 2 rsw bits
    pub fn set_supervisor_bits(&self, bits: u8) {
        const M: u64 = 3 << 8;
        let mut c = self.get_raw();
        c &= !M;
        c |= (bits as u64 & 3) << 8;
        self.set_raw(c);
    }

    pub fn ppn(&self) -> u64 {
        (self.get_raw() & !Self::STATUS_BITS_MASK) >> Self::PPN_OFFSETS[0]
    }

    pub fn set_ppn(&self, ppn: u64) {
        let mut c = self.get_raw();
        c &= Self::STATUS_BITS_MASK;
        c |= ppn << Self::PPN_OFFSETS[0];
        self.set_raw(c);
    }

    pub fn ppn_n(&self, n: usize) -> u64 {
        let i = Self::PPN_OFFSETS[n];
        let m = PhysAddr::PHYS_PAGE_NUM_MASK << i as u64;
        (self.get_raw() & m) >> Self::PPN_OFFSETS[n]
    }

    pub fn ppn0(&self) -> u64 {
        let i = Self::PPN_OFFSETS[0];
        let m = PhysAddr::PHYS_PAGE_NUM_MASK << i as u64;
        (self.get_raw() & m) >> Self::PPN_OFFSETS[0]
    }

    pub fn ppn1(&self) -> u64 {
        let i = Self::PPN_OFFSETS[1];
        let m = PhysAddr::PHYS_PAGE_NUM_MASK << i as u64;
        (self.get_raw() & m) >> Self::PPN_OFFSETS[1]
    }

    pub fn ppn2(&self) -> u64 {
        let i = Self::PPN_OFFSETS[2];
        let m = PhysAddr::PHYS_PAGE_NUM_MASK << i as u64;
        (self.get_raw() & m) >> Self::PPN_OFFSETS[2]
    }

    pub fn ppn3(&self) -> u64 {
        let i = Self::PPN_OFFSETS[3];
        let m = PhysAddr::PHYS_PAGE_NUM_MASK3 << i as u64;
        (self.get_raw() & m) >> Self::PPN_OFFSETS[3]
    }

    pub fn set_ppn_n(&self, n: usize, ppn: u16) {
        let i = Self::PPN_OFFSETS[n];
        let m = PhysAddr::PHYS_PAGE_NUM_MASK << i as u64;
        let mut c = self.get_raw();
        c &= !m;
        c |= (ppn as u64 & PhysAddr::PHYS_PAGE_NUM_MASK) << i;
        self.set_raw(c);
    }

    pub fn set_ppn0(&self, ppn: u16) {
        let i = Self::PPN_OFFSETS[0];
        let m = PhysAddr::PHYS_PAGE_NUM_MASK << i as u64;
        let mut c = self.get_raw();
        c &= !m;
        c |= (ppn as u64 & PhysAddr::PHYS_PAGE_NUM_MASK) << i;
        self.set_raw(c);
    }

    pub fn set_ppn1(&self, ppn: u16) {
        let i = Self::PPN_OFFSETS[1];
        let m = PhysAddr::PHYS_PAGE_NUM_MASK << i as u64;
        let mut c = self.get_raw();
        c &= !m;
        c |= (ppn as u64 & PhysAddr::PHYS_PAGE_NUM_MASK) << i;
        self.set_raw(c);
    }

    pub fn set_ppn2(&self, ppn: u16) {
        let i = Self::PPN_OFFSETS[2];
        let m = PhysAddr::PHYS_PAGE_NUM_MASK << i as u64;
        let mut c = self.get_raw();
        c &= !m;
        c |= (ppn as u64 & PhysAddr::PHYS_PAGE_NUM_MASK) << i;
        self.set_raw(c);
    }

    pub fn set_ppn3(&self, ppn: u32) {
        let i = Self::PPN_OFFSETS[3];
        let m = PhysAddr::PHYS_PAGE_NUM_MASK3 << i as u64;
        let mut c = self.get_raw();
        c &= !m;
        c |= (ppn as u64 & PhysAddr::PHYS_PAGE_NUM_MASK3) << i;
        self.set_raw(c);
    }
}

pub const PAGE_TABLE_ENTRY_COUNT: usize = 4096 / core::mem::size_of::<PageTableEntry>();

#[repr(C, align(4096))]
pub struct PageTable {
    pub(super) entries: [PageTableEntry; PAGE_TABLE_ENTRY_COUNT]
}
