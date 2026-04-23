//! User-mode address-space layout (Sv48, low→high).
//!
//! The kernel lives entirely in the high half (KTEXT / KDMAP / KMMIO) so
//! the low half belongs to user processes minus a null guard.
//!
//!   0..USER_NULL_GUARD_END                         null guard
//!   UPROC_STACK_BASE..+8 GiB                       256 * 32 MiB stack slots
//!   USER_TEXT_BASE..                               user ELF image + mmap arena
//!   USER_TRAP_FRAME_BASE..                         256 * 4 KiB TrapFrames (S-only)

pub const PAGE_SIZE:  u64 = 4096;
pub const LARGE_PAGE: u64 = 2 * 1024 * 1024;

/// Matches Linux-style mmap_min_addr; catches NULL-region derefs as faults.
pub const USER_NULL_GUARD_END: u64 = 0x1_0000;

/// Per-thread stack region. 256 slots * 32 MiB stride = 8 GiB total.
/// Each slot holds a stack sized per-thread (2..=30 MiB, multiples of 2 MiB),
/// anchored at the high end of the slot; the remainder is an unmapped guard.
pub const UPROC_STACK_BASE:    u64 = 0x1000_0000;
pub const UPROC_STACK_STRIDE:  u64 = 32 * LARGE_PAGE;
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

/// Kernel-private per-thread TrapFrame region (no U bit). One page per slot.
pub const USER_TRAP_FRAME_BASE:   u64 = 0x100_0000_0000;
pub const USER_TRAP_FRAME_STRIDE: u64 = PAGE_SIZE;

pub const fn user_trap_frame_vaddr(slot: u16) -> u64 {
    USER_TRAP_FRAME_BASE + (slot as u64) * USER_TRAP_FRAME_STRIDE
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
}
