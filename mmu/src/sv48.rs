use core::sync::atomic::{self as atomic, AtomicU64};
use atomic::Ordering;

/// Atomic storage for page-table entries. Hardware page-table walkers on
/// any hart can race with our writes, so PTE reads/writes go through
/// `AtomicU64`. Plain address wrappers (`VirtAddr` / `PhysAddr`) don't —
/// those are single-threaded scratch values built up during a mapping
/// call, so they're `Copy`.
type AtomicPte = atomic::AtomicU64;

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct VirtAddr {
    pub(crate) a: u64,
}

impl VirtAddr {
    pub const PAGE_OFFSET_MASK: u64 = 0xFFF;
    pub const VIRT_PAGE_NUM_MASK: u64 = 0x1FF;

    const VPN_OFFSETS: [u64; 4] = [
        12, 21, 30, 39
    ];

    pub const fn new(raw: u64) -> Self {
        Self { a: raw }
    }

    #[inline(always)]
    pub fn get_raw(&self) -> u64 {
        self.a
    }

    pub fn page_offset(&self) -> u64 {
        self.a & Self::PAGE_OFFSET_MASK
    }

    pub fn vpn_n(&self, n: usize) -> u64 {
        let i = Self::VPN_OFFSETS[n];
        let m = Self::VIRT_PAGE_NUM_MASK << i as u64;
        (self.a & m) >> Self::VPN_OFFSETS[n]
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
        (self.a & m) >> Self::VPN_OFFSETS[3]
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct PhysAddr {
    pub(crate) a: u64,
}

impl PhysAddr {
    pub const PAGE_OFFSET_MASK: u64 = 0xFFF;
    pub const PHYS_PAGE_NUM_MASK: u64 = 0x1FF;
    pub const PHYS_PAGE_NUM_MASK3: u64 = 0x1FFFF;

    pub const PPN_OFFSETS: [u64; 4] = [
        12, 21, 30, 39
    ];

    pub const fn new(raw: u64) -> Self {
        Self { a: raw }
    }

    #[inline(always)]
    pub fn get_raw(&self) -> u64 {
        self.a
    }

    pub fn page_offset(&self) -> u64 {
        self.a & Self::PAGE_OFFSET_MASK
    }

    pub fn ppn(&self) -> u64 {
        (self.a & !Self::PAGE_OFFSET_MASK) >> Self::PPN_OFFSETS[0]
    }

    pub fn ppn_n(&self, n: usize) -> u64 {
        let i = Self::PPN_OFFSETS[n];
        let pm = if n == 3 { Self::PHYS_PAGE_NUM_MASK3 } else { Self::PHYS_PAGE_NUM_MASK };
        let m = pm << i as u64;
        (self.a & m) >> Self::PPN_OFFSETS[n]
    }

    pub fn ppn0(&self) -> u64 {
        let i = Self::PPN_OFFSETS[0];
        let m = Self::PHYS_PAGE_NUM_MASK << i as u64;
        (self.a & m) >> Self::PPN_OFFSETS[0]
    }

    pub fn ppn1(&self) -> u64 {
        let i = Self::PPN_OFFSETS[1];
        let m = Self::PHYS_PAGE_NUM_MASK << i as u64;
        (self.a & m) >> Self::PPN_OFFSETS[1]
    }

    pub fn ppn2(&self) -> u64 {
        let i = Self::PPN_OFFSETS[2];
        let m = Self::PHYS_PAGE_NUM_MASK << i as u64;
        (self.a & m) >> Self::PPN_OFFSETS[2]
    }

    pub fn ppn3(&self) -> u64 {
        let i = Self::PPN_OFFSETS[3];
        let m = Self::PHYS_PAGE_NUM_MASK << i as u64;
        (self.a & m) >> Self::PPN_OFFSETS[3]
    }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct PageTableEntry {
    pub(crate) e: AtomicPte,
}

impl PageTableEntry {
    pub const STATUS_BITS_MASK: u64 = 0x3FF;

    pub const PPN_OFFSETS: [u64; 4] = [
        10, 19, 28, 37
    ];

    // Single-bit flag masks. Using these (over repeated `1 << N`) keeps
    // `pack_leaf` and the atomic setters readable.
    pub const VALID:      u64 = 1 << 0;
    pub const READABLE:   u64 = 1 << 1;
    pub const WRITEABLE:  u64 = 1 << 2;
    pub const EXECUTABLE: u64 = 1 << 3;
    pub const USER_PAGE:  u64 = 1 << 4;
    pub const GLOBAL:     u64 = 1 << 5;
    pub const ACCESSED:   u64 = 1 << 6;
    pub const DIRTY:      u64 = 1 << 7;
    // R|W|X|U|G — the bits PagePermissions encodes.
    pub const PERMS_MASK: u64 = 0x3E;

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

    /// Build a fully-formed leaf PTE value in one `u64`. Store with a single
    /// `set_raw` so a remote hardware table walker observes either the old
    /// value or the fully-constructed new one — never a half-built PTE.
    ///
    /// `ppn` is the 44-bit physical page number (paddr / PAGE_SIZE).
    /// `perms` carries R/W/X/U/G bits (matches `PagePermissions`). `rsw`
    /// is stashed into PTE[8:9] (the two reserved-for-supervisor-software
    /// bits); orbit policy assigns meaning to specific values (e.g.
    /// "Shared-pool user mapping, revocable"). `V`, `A`, `D` are set
    /// unconditionally.
    #[inline]
    pub const fn pack_leaf(ppn: u64, perms: u64, rsw: u8) -> u64 {
        (ppn << Self::PPN_OFFSETS[0])
            | (perms & Self::PERMS_MASK)
            | (((rsw as u64) & 0b11) << 8)
            | Self::VALID
            | Self::ACCESSED
            | Self::DIRTY
    }

    /// Build a non-leaf (table-pointer) PTE value. The only permission bit
    /// set is `V`; R/W/X zero makes it an interior PTE per the spec.
    #[inline]
    pub const fn pack_table(ppn: u64) -> u64 {
        (ppn << Self::PPN_OFFSETS[0]) | Self::VALID
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
        if b {
            self.e.fetch_or(m, Ordering::AcqRel);
        } else {
            self.e.fetch_and(!m, Ordering::AcqRel);
        }
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
        let new = (bits as u64 & 3) << 8;
        let _ = self.e.fetch_update(Ordering::AcqRel, Ordering::Acquire, |c| {
            Some((c & !M) | new)
        });
    }

    pub fn ppn(&self) -> u64 {
        (self.get_raw() & !Self::STATUS_BITS_MASK) >> Self::PPN_OFFSETS[0]
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
}

pub const PAGE_TABLE_ENTRY_COUNT: usize = 4096 / core::mem::size_of::<PageTableEntry>();

#[repr(C, align(4096))]
pub struct PageTable {
    pub entries: [PageTableEntry; PAGE_TABLE_ENTRY_COUNT]
}
