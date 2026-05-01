use orbit_core::PAGE_SIZE;
use orbit_core::manager::{MEGAPAGE_SIZE, MappingGeometry, select_mapping_geometry};

fn megapage() -> MappingGeometry {
    MappingGeometry {
        align: MEGAPAGE_SIZE,
        levels: 3,
    }
}

fn page() -> MappingGeometry {
    MappingGeometry {
        align: PAGE_SIZE,
        levels: 4,
    }
}

#[test]
fn both_megapage_aligned_picks_megapage() {
    assert_eq!(
        select_mapping_geometry(MEGAPAGE_SIZE, MEGAPAGE_SIZE),
        Some(megapage())
    );
    assert_eq!(
        select_mapping_geometry(MEGAPAGE_SIZE * 5, MEGAPAGE_SIZE * 3),
        Some(megapage())
    );
}

#[test]
fn page_aligned_only_picks_4k() {
    // vaddr is megapage-aligned but size isn't; falls through to 4K.
    assert_eq!(
        select_mapping_geometry(MEGAPAGE_SIZE, PAGE_SIZE),
        Some(page())
    );
    // size is megapage-aligned but vaddr isn't.
    assert_eq!(
        select_mapping_geometry(PAGE_SIZE, MEGAPAGE_SIZE),
        Some(page())
    );
    // Both only 4K-aligned.
    assert_eq!(
        select_mapping_geometry(PAGE_SIZE * 3, PAGE_SIZE * 7),
        Some(page())
    );
}

#[test]
fn misaligned_returns_none() {
    // Not aligned to PAGE_SIZE.
    assert_eq!(select_mapping_geometry(0x1001, PAGE_SIZE), None);
    assert_eq!(select_mapping_geometry(PAGE_SIZE, PAGE_SIZE + 1), None);
    // Unaligned both ways.
    assert_eq!(select_mapping_geometry(7, 31), None);
}

#[test]
fn exactly_megapage_boundary_is_megapage() {
    // 2 MiB exactly: megapage.
    assert_eq!(
        select_mapping_geometry(MEGAPAGE_SIZE, MEGAPAGE_SIZE),
        Some(megapage())
    );
    // 2 MiB + 4 KiB: size isn't megapage-aligned, fall back to 4K.
    assert_eq!(
        select_mapping_geometry(MEGAPAGE_SIZE, MEGAPAGE_SIZE + PAGE_SIZE),
        Some(page())
    );
}

#[test]
fn zero_size_is_aligned_to_everything() {
    // A zero-size request is trivially aligned both ways; the live code
    // preserves this (the subsequent Layout::from_size_align check in
    // the kmain handler is what catches size=0 as an error). Tests pin
    // the pure function's behaviour so future refactors don't silently
    // flip the check.
    assert_eq!(select_mapping_geometry(MEGAPAGE_SIZE, 0), Some(megapage()));
    assert_eq!(select_mapping_geometry(0, 0), Some(megapage()));
}

#[test]
fn zero_vaddr_is_megapage_aligned() {
    assert_eq!(select_mapping_geometry(0, MEGAPAGE_SIZE), Some(megapage()));
}

/// 5×5 grid over boundary values, with each expected outcome computed
/// by hand and inlined as a literal. An independent oracle: a sign /
/// constant / branch-order bug in `select_mapping_geometry` shows up
/// as a single mismatched cell, not as table+code agreeing on the
/// wrong answer.
///
/// Legend: M = megapage, P = 4 KiB page, N = misaligned (None).
/// Rows are vaddr; columns are size.
///
/// ```text
///                size=0(2M)  4K  2M-1(none)  2M  2M+4K(4K)
/// vaddr=0(2M)        M       P       N       M       P
///       4K           P       P       N       P       P
///     2M-1(none)     N       N       N       N       N
///       2M           M       P       N       M       P
///     2M+4K(4K)      P       P       N       P       P
/// ```
#[test]
fn alignment_grid_matches_hand_table() {
    let values = [
        0,
        PAGE_SIZE,
        MEGAPAGE_SIZE - 1,
        MEGAPAGE_SIZE,
        MEGAPAGE_SIZE + PAGE_SIZE,
    ];

    // Hand-computed expected outcomes — see the table in the docstring.
    // None of these are derived from the implementation's predicates.
    let m = Some(megapage());
    let p = Some(page());
    let n = None;
    let expected: [[Option<MappingGeometry>; 5]; 5] = [
        // size:    0   4K   2M-1  2M   2M+4K
        /* 0     */ [m, p, n, m, p],
        /* 4K    */ [p, p, n, p, p],
        /* 2M-1  */ [n, n, n, n, n],
        /* 2M    */ [m, p, n, m, p],
        /* 2M+4K */ [p, p, n, p, p],
    ];

    for (vi, &vaddr) in values.iter().enumerate() {
        for (si, &size) in values.iter().enumerate() {
            let got = select_mapping_geometry(vaddr, size);
            assert_eq!(
                got, expected[vi][si],
                "select_mapping_geometry(vaddr={vaddr:#x}, size={size:#x})"
            );
        }
    }
}
