//! User-mode address-space layout (Sv48, low→high).
//!
//! The kernel lives entirely in the high half (KTEXT / KDMAP / KMMIO) so
//! the low half belongs to user processes minus a null guard.
//!
//! Sv48 canonical low half is `[0, 2^47)` = 128 TiB. Two layers carve
//! it up:
//!
//! 1. *Kernel-managed user mappings.* Stacks and the ELF image are
//!    installed by the kernel at process creation (via direct
//!    `map_address_range` calls, not the mmap syscall). Their VA
//!    ranges live below [`UPROC_PRIV_BASE`] precisely so a user mmap
//!    can't aim at them — the syscall layer rejects any vaddr below
//!    `UPROC_PRIV_BASE` regardless of permissions.
//!
//! 2. *User-controlled mappings.* `mmap` and `create_netch` install
//!    these at user-supplied VAs. Split into two disjoint ranges:
//!    private (anonymous mmap heap) and shared (mappings the kernel
//!    keeps a KDMAP alias for — NetChannels,
//!    `mmap(share_with_kernel=true)`). Keeping them disjoint means
//!    the syscall layer can reject a private-mmap request that names
//!    a shared VA (and vice versa) without consulting per-process
//!    allocator state.
//!
//!   0..USER_NULL_GUARD_END                         null guard (2 MiB, megapage-aligned)
//!   UPROC_STACK_BASE..+8 GiB                       256 * 32 MiB stack slots (kernel-mapped)
//!   USER_TEXT_BASE..                               user ELF image (kernel-mapped)
//!     (gap)
//!   UPROC_PRIV_BASE..UPROC_PRIV_END                ~64 TiB private (user mmap heap)
//!   UPROC_SHARED_BASE..UPROC_SHARED_END            62 TiB shared (NetChannels, shared mmap)
//!   USER_TRAP_FRAME_BASE..                         256 * 4 KiB TrapFrames (S-only)

pub const PAGE_SIZE:  u64 = 4096;
pub const LARGE_PAGE: u64 = 2 * 1024 * 1024;

/// Catches NULL-region derefs as page faults. Bumped to a megapage so the
/// L1 PTE for the null region is simply absent — accidental dereferences
/// of small struct offsets through a null base pointer all fault, and the
/// page tables don't pay for the guard.
pub const USER_NULL_GUARD_END: u64 = LARGE_PAGE;

/// Per-thread stack region. 256 slots * 32 MiB stride = 8 GiB total —
/// chosen so the whole stack region fits below [`USER_TEXT_BASE`]
/// (8.5 GiB). Each slot holds a stack sized per-thread (2..=30 MiB,
/// multiples of 2 MiB), anchored at the high end of the slot; the
/// remainder is an unmapped guard.
pub const UPROC_STACK_BASE:    u64 = 0x1000_0000;
pub const UPROC_STACK_STRIDE:  u64 = 16 * LARGE_PAGE;
pub const UPROC_STACK_GRAIN:   u64 = LARGE_PAGE;
pub const UPROC_STACK_MIN:     u64 = UPROC_STACK_GRAIN;
/// One grain reserved for the guard at the low end of each slot.
pub const UPROC_STACK_MAX:     u64 = UPROC_STACK_STRIDE - UPROC_STACK_GRAIN;
pub const UPROC_STACK_DEFAULT: u64 = UPROC_STACK_GRAIN;

pub const fn user_stack_slot_base(slot: u16) -> u64 {
    UPROC_STACK_BASE + (slot as u64) * UPROC_STACK_STRIDE
}

pub const fn user_stack_slot_top(slot: u16) -> u64 {
    user_stack_slot_base(slot) + UPROC_STACK_STRIDE
}

/// Low end of the writable stack; stack grows down from `user_stack_slot_top`.
pub const fn user_stack_vaddr(slot: u16, stack_size: u64) -> u64 {
    user_stack_slot_top(slot) - stack_size
}

pub const fn user_stack_guard_vaddr(slot: u16) -> u64 {
    user_stack_slot_base(slot)
}

pub const fn user_stack_guard_size(stack_size: u64) -> u64 {
    UPROC_STACK_STRIDE - stack_size
}

pub const fn validate_user_stack_size(size: u64) -> bool {
    size >= UPROC_STACK_MIN
        && size <= UPROC_STACK_MAX
        && size % UPROC_STACK_GRAIN == 0
}

/// User ELF image sits just above the 8 GiB stack region.
pub const USER_TEXT_BASE: u64 = 0x2_2000_0000;

/// User-controlled private range — `mmap(share_with_kernel=false)`
/// must land here. Sits *above* the kernel-managed stacks (8 GiB
/// region anchored at [`UPROC_STACK_BASE`]) and the ELF image (at
/// [`USER_TEXT_BASE`]) so a user mmap can't aim into either. orbit-rt's
/// global allocator uses this as the start of its heap cursor.
pub const UPROC_PRIV_BASE: u64 = 0x3_0000_0000;             // 12 GiB
pub const UPROC_PRIV_END:  u64 = 0x4000_0000_0000;          // 64 TiB

/// Shared user range — `mmap(share_with_kernel=true)` regions and
/// NetChannels (anything the kernel needs a KDMAP alias for after the
/// user mapping is installed). Disjoint from [`UPROC_PRIV_BASE`] so a
/// private-mmap request can't be aimed into a shared VA range and vice
/// versa.
pub const UPROC_SHARED_BASE: u64 = UPROC_PRIV_END;
pub const UPROC_SHARED_END:  u64 = 0x7E00_0000_0000;        // 126 TiB (= UPROC_SHARED_BASE + 62 TiB)

/// Kernel-private per-thread TrapFrame region (no U bit). One page per slot.
/// Sits at the top of the Sv48 low half above the user-shared range.
pub const USER_TRAP_FRAME_BASE:   u64 = UPROC_SHARED_END;
pub const USER_TRAP_FRAME_STRIDE: u64 = PAGE_SIZE;

pub const fn user_trap_frame_vaddr(slot: u16) -> u64 {
    USER_TRAP_FRAME_BASE + (slot as u64) * USER_TRAP_FRAME_STRIDE
}

/// Exclusive upper bound on user-mappable VAs. The trap-frame region at
/// [`USER_TRAP_FRAME_BASE`] is kernel-private (no U bit), and everything
/// above it is either unused low-half or reserved kernel space. Syscalls
/// taking a user VA reject anything at or above this point.
pub const USER_VA_END: u64 = USER_TRAP_FRAME_BASE;

/// True iff `[vaddr, vaddr + len)` lies anywhere a user mapping might
/// legally exist: above the null guard, below the kernel-private
/// trap-frame region, no overflow. Used by syscalls that *read or write*
/// a user buffer (`serial_print`, `console_write`, `read_stdin`,
/// `create_process`'s elf_ptr) — the buffer can be on the stack, in
/// the ELF data section, in the priv heap, or in a shared region, and
/// the syscall just needs to know it isn't being pointed at a kernel
/// VA. The translate check that follows catches the address-not-mapped
/// case via `user_va_translates`.
///
/// For syscalls that *install* a new mapping (`mmap`, `create_netch`),
/// use [`user_priv_range_ok`] / [`user_shared_range_ok`] instead so the
/// new mapping can't be aimed at a kernel-managed region (stacks, ELF)
/// or the wrong pool.
pub const fn user_range_ok(vaddr: u64, len: u64) -> bool {
    if vaddr < USER_NULL_GUARD_END {
        return false;
    }
    match vaddr.checked_add(len) {
        Some(end) => end <= USER_VA_END,
        None => false,
    }
}

/// True iff `[vaddr, vaddr + len)` lies entirely within the private user
/// range. `mmap(share_with_kernel=false)` callers must satisfy this.
pub const fn user_priv_range_ok(vaddr: u64, len: u64) -> bool {
    if vaddr < UPROC_PRIV_BASE {
        return false;
    }
    match vaddr.checked_add(len) {
        Some(end) => end <= UPROC_PRIV_END,
        None => false,
    }
}

/// True iff `[vaddr, vaddr + len)` lies entirely within the shared user
/// range. `mmap(share_with_kernel=true)` and `create_netch` callers must
/// satisfy this.
pub const fn user_shared_range_ok(vaddr: u64, len: u64) -> bool {
    if vaddr < UPROC_SHARED_BASE {
        return false;
    }
    match vaddr.checked_add(len) {
        Some(end) => end <= UPROC_SHARED_END,
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- layout constants ----

    #[test]
    fn stack_max_leaves_room_for_guard() {
        // A max-size stack still leaves one `UPROC_STACK_GRAIN` for a
        // low-end guard — if this changes, users can stack-overflow into
        // the next slot.
        assert_eq!(UPROC_STACK_MAX + UPROC_STACK_GRAIN, UPROC_STACK_STRIDE);
    }

    #[test]
    fn min_max_are_grain_multiples() {
        assert_eq!(UPROC_STACK_MIN % UPROC_STACK_GRAIN, 0);
        assert_eq!(UPROC_STACK_MAX % UPROC_STACK_GRAIN, 0);
        assert_eq!(UPROC_STACK_STRIDE % UPROC_STACK_GRAIN, 0);
    }

    #[test]
    fn large_page_is_2mib_as_documented() {
        assert_eq!(LARGE_PAGE, 2 * 1024 * 1024);
        assert_eq!(PAGE_SIZE, 4096);
    }

    // ---- user_stack_slot_{base,top} ----

    #[test]
    fn slot_0_base_is_stack_base() {
        assert_eq!(user_stack_slot_base(0), UPROC_STACK_BASE);
    }

    #[test]
    fn slot_top_minus_base_is_stride() {
        for &slot in &[0u16, 1, 7, 128, 255] {
            assert_eq!(
                user_stack_slot_top(slot) - user_stack_slot_base(slot),
                UPROC_STACK_STRIDE
            );
        }
    }

    #[test]
    fn adjacent_slots_abut_cleanly() {
        for slot in 0u16..16 {
            assert_eq!(
                user_stack_slot_top(slot),
                user_stack_slot_base(slot + 1),
                "slot {slot} top should equal slot {} base",
                slot + 1
            );
        }
    }

    // ---- user_stack_vaddr + guard ----

    #[test]
    fn stack_vaddr_anchors_at_top() {
        let slot = 3u16;
        let size = UPROC_STACK_DEFAULT;
        let top = user_stack_slot_top(slot);
        let stack = user_stack_vaddr(slot, size);
        assert_eq!(stack + size, top, "stack grows down from slot top");
    }

    #[test]
    fn stack_and_guard_cover_full_slot_nonoverlapping() {
        let slot = 5u16;
        let stack_size = UPROC_STACK_DEFAULT;
        let stack_lo = user_stack_vaddr(slot, stack_size);
        let guard_lo = user_stack_guard_vaddr(slot);
        let guard_size = user_stack_guard_size(stack_size);

        assert_eq!(guard_lo, user_stack_slot_base(slot));
        assert_eq!(guard_lo + guard_size, stack_lo, "guard ends where stack starts");
        assert_eq!(
            stack_lo + stack_size,
            user_stack_slot_top(slot),
            "stack ends at slot top"
        );
    }

    #[test]
    fn max_size_stack_has_minimum_guard() {
        let g = user_stack_guard_size(UPROC_STACK_MAX);
        assert_eq!(g, UPROC_STACK_GRAIN, "max stack leaves exactly one grain for guard");
    }

    // ---- validate_user_stack_size ----

    #[test]
    fn validate_accepts_min_and_max() {
        assert!(validate_user_stack_size(UPROC_STACK_MIN));
        assert!(validate_user_stack_size(UPROC_STACK_MAX));
    }

    #[test]
    fn validate_rejects_below_min() {
        assert!(!validate_user_stack_size(0));
        assert!(!validate_user_stack_size(UPROC_STACK_MIN - 1));
        assert!(!validate_user_stack_size(PAGE_SIZE));
    }

    #[test]
    fn validate_rejects_above_max() {
        assert!(!validate_user_stack_size(UPROC_STACK_MAX + UPROC_STACK_GRAIN));
        assert!(!validate_user_stack_size(u64::MAX));
    }

    #[test]
    fn validate_rejects_non_grain_multiples() {
        // Within [MIN, MAX] but not grain-aligned.
        assert!(!validate_user_stack_size(UPROC_STACK_MIN + PAGE_SIZE));
        assert!(!validate_user_stack_size(UPROC_STACK_MIN + 1));
    }

    // ---- user_trap_frame_vaddr ----

    #[test]
    fn trap_frame_slots_are_page_apart() {
        for slot in 0u16..8 {
            assert_eq!(
                user_trap_frame_vaddr(slot + 1) - user_trap_frame_vaddr(slot),
                USER_TRAP_FRAME_STRIDE
            );
        }
    }

    #[test]
    fn trap_frame_slot_0_is_base() {
        assert_eq!(user_trap_frame_vaddr(0), USER_TRAP_FRAME_BASE);
    }

    #[test]
    fn trap_frame_last_slot_fits_one_page_below_the_next_region() {
        // 256 slots * 4 KiB = 1 MiB. Not a tight bound — but pin the
        // top so a future increase in slot count has to reconcile.
        let top = user_trap_frame_vaddr(255) + USER_TRAP_FRAME_STRIDE;
        assert_eq!(top - USER_TRAP_FRAME_BASE, 256 * PAGE_SIZE);
    }

    // ---- range layout invariants ----

    #[test]
    fn priv_and_shared_are_disjoint_and_abutting() {
        assert!(UPROC_PRIV_BASE < UPROC_PRIV_END);
        assert!(UPROC_SHARED_BASE < UPROC_SHARED_END);
        assert_eq!(UPROC_PRIV_END, UPROC_SHARED_BASE,
            "ranges should be adjacent so user_range_ok's union has no gap");
        assert_eq!(USER_TRAP_FRAME_BASE, UPROC_SHARED_END,
            "trap frame region sits right above the shared range");
    }

    #[test]
    fn kernel_managed_regions_sit_below_priv_base() {
        // Stacks and the ELF image are installed by the kernel at
        // process creation, not via mmap. Their full extents must sit
        // below UPROC_PRIV_BASE so the syscall gate
        // (`vaddr >= UPROC_PRIV_BASE`) rejects any user mmap aimed at
        // them. The stack region is sized for all 256 slots — if the
        // stride or slot count grows past UPROC_PRIV_BASE this assert
        // catches it before umode threads start trampling the heap.
        let stack_top = UPROC_STACK_BASE + 256u64 * UPROC_STACK_STRIDE;
        assert!(stack_top <= USER_TEXT_BASE,
            "stack region must end below USER_TEXT_BASE");
        assert!(stack_top <= UPROC_PRIV_BASE,
            "stack region must end below UPROC_PRIV_BASE");
        assert!(USER_TEXT_BASE < UPROC_PRIV_BASE);
    }

    #[test]
    fn stack_max_is_30_mib_per_doc() {
        // Doc on UPROC_STACK_STRIDE pins the per-thread stack max at
        // 30 MiB (= STRIDE - one GRAIN guard). Catching divergence here
        // saves a debugging session when someone adjusts STRIDE
        // without updating the docs.
        assert_eq!(UPROC_STACK_MAX, 30 * 1024 * 1024);
    }

    #[test]
    fn null_guard_is_megapage_aligned() {
        // Whole-megapage guard so the L1 PTE for the null region is
        // simply absent rather than having to be a page-table-pointing
        // entry full of L0 invalid PTEs.
        assert_eq!(USER_NULL_GUARD_END, LARGE_PAGE);
        assert_eq!(USER_NULL_GUARD_END % LARGE_PAGE, 0);
    }

    // ---- user_priv_range_ok ----

    #[test]
    fn priv_ok_accepts_inside_priv() {
        assert!(user_priv_range_ok(UPROC_PRIV_BASE, PAGE_SIZE));
        assert!(user_priv_range_ok(UPROC_PRIV_END - PAGE_SIZE, PAGE_SIZE));
    }

    #[test]
    fn priv_ok_rejects_kernel_managed_regions() {
        // Stacks and the ELF image are mapped by the kernel at process
        // creation, not via the mmap syscall. The priv-range gate must
        // reject any user mmap aimed at them.
        assert!(!user_priv_range_ok(UPROC_STACK_BASE, PAGE_SIZE));
        assert!(!user_priv_range_ok(USER_TEXT_BASE, PAGE_SIZE));
    }

    #[test]
    fn priv_ok_rejects_null_guard() {
        assert!(!user_priv_range_ok(0, PAGE_SIZE));
        assert!(!user_priv_range_ok(UPROC_PRIV_BASE - 1, 1));
    }

    #[test]
    fn priv_ok_rejects_shared_range() {
        // A shared VA must never satisfy the private check.
        assert!(!user_priv_range_ok(UPROC_SHARED_BASE, PAGE_SIZE));
        assert!(!user_priv_range_ok(UPROC_SHARED_END - PAGE_SIZE, PAGE_SIZE));
    }

    #[test]
    fn priv_ok_rejects_boundary_cross() {
        // Range that starts in priv and reaches into shared.
        assert!(!user_priv_range_ok(UPROC_PRIV_END - PAGE_SIZE, 2 * PAGE_SIZE));
    }

    #[test]
    fn priv_ok_rejects_overflow() {
        assert!(!user_priv_range_ok(u64::MAX, 1));
        assert!(!user_priv_range_ok(UPROC_PRIV_END - 1, 2));
    }

    // ---- user_shared_range_ok ----

    #[test]
    fn shared_ok_accepts_inside_shared() {
        assert!(user_shared_range_ok(UPROC_SHARED_BASE, PAGE_SIZE));
        assert!(user_shared_range_ok(UPROC_SHARED_END - PAGE_SIZE, PAGE_SIZE));
    }

    #[test]
    fn shared_ok_rejects_priv_range() {
        // A private VA must never satisfy the shared check.
        assert!(!user_shared_range_ok(UPROC_PRIV_BASE, PAGE_SIZE));
        assert!(!user_shared_range_ok(USER_TEXT_BASE, PAGE_SIZE));
    }

    #[test]
    fn shared_ok_rejects_trap_frame_region() {
        // USER_TRAP_FRAME_BASE == UPROC_SHARED_END — kernel-private,
        // no U bit. Shared check must not let a user name it.
        assert!(!user_shared_range_ok(UPROC_SHARED_END, PAGE_SIZE));
        assert!(!user_shared_range_ok(user_trap_frame_vaddr(0), PAGE_SIZE));
    }

    #[test]
    fn shared_ok_rejects_overflow() {
        assert!(!user_shared_range_ok(u64::MAX, 1));
        assert!(!user_shared_range_ok(UPROC_SHARED_END - 1, 2));
    }

    // ---- user_range_ok (buffer-pointer ops) ----

    #[test]
    fn user_range_ok_accepts_kernel_managed_user_regions() {
        // Read/write syscalls (serial_print, console_write, read_stdin,
        // create_process's elf_ptr) take buffer pointers that can land
        // anywhere a user mapping might legally exist — including on
        // the stack and inside the ELF's data section. user_range_ok
        // accepts these; the translate check catches unmapped pages.
        assert!(user_range_ok(UPROC_STACK_BASE, PAGE_SIZE));
        assert!(user_range_ok(USER_TEXT_BASE, PAGE_SIZE));
        assert!(user_range_ok(UPROC_PRIV_BASE, PAGE_SIZE));
        assert!(user_range_ok(UPROC_SHARED_BASE, PAGE_SIZE));
    }

    #[test]
    fn user_range_ok_rejects_null_guard() {
        assert!(!user_range_ok(0, PAGE_SIZE));
        assert!(!user_range_ok(USER_NULL_GUARD_END - 1, 1));
        assert!(!user_range_ok(USER_NULL_GUARD_END - PAGE_SIZE, PAGE_SIZE));
    }

    #[test]
    fn user_range_ok_rejects_trap_frame_region() {
        // The kernel-private TrapFrame region sits at USER_VA_END.
        assert!(!user_range_ok(USER_VA_END, PAGE_SIZE));
        assert!(!user_range_ok(user_trap_frame_vaddr(0), PAGE_SIZE));
    }

    #[test]
    fn user_range_ok_rejects_kernel_high_half() {
        assert!(!user_range_ok(0xFFFF_FFC0_0000_0000, PAGE_SIZE));
    }

    #[test]
    fn user_range_ok_rejects_overflow() {
        assert!(!user_range_ok(u64::MAX, 1));
        assert!(!user_range_ok(USER_VA_END - 1, 2));
        assert!(!user_range_ok(USER_VA_END - PAGE_SIZE, PAGE_SIZE + 1));
    }

    #[test]
    fn user_range_ok_zero_len_at_boundary() {
        assert!(user_range_ok(USER_NULL_GUARD_END, 0));
        assert!(user_range_ok(USER_VA_END, 0));
    }

    #[test]
    fn user_range_ok_accepts_range_ending_at_va_end() {
        assert!(user_range_ok(USER_VA_END - PAGE_SIZE, PAGE_SIZE));
    }
}
