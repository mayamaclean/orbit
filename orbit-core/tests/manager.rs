use orbit_core::PAGE_SIZE;
use orbit_core::manager::{MEGAPAGE_SIZE, MappingGeometry, select_mapping_geometry};

fn megapage() -> MappingGeometry {
    MappingGeometry { align: MEGAPAGE_SIZE, levels: 3 }
}

fn page() -> MappingGeometry {
    MappingGeometry { align: PAGE_SIZE, levels: 4 }
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
    assert_eq!(
        select_mapping_geometry(MEGAPAGE_SIZE, 0),
        Some(megapage())
    );
    assert_eq!(select_mapping_geometry(0, 0), Some(megapage()));
}

#[test]
fn zero_vaddr_is_megapage_aligned() {
    assert_eq!(
        select_mapping_geometry(0, MEGAPAGE_SIZE),
        Some(megapage())
    );
}

/// Brute 5x5 grid over common boundary values. Catches regressions in
/// the modular-arithmetic predicates that a hand-picked handful would
/// miss. The expected geometry is computed by the same predicate a
/// reader would check mentally, so a symmetric bug in both would still
/// escape — but sign/off-by-one errors against the constants won't.
#[test]
fn alignment_grid_matches_predicate() {
    let values = [
        0,
        PAGE_SIZE,
        MEGAPAGE_SIZE - 1,
        MEGAPAGE_SIZE,
        MEGAPAGE_SIZE + PAGE_SIZE,
    ];

    for &vaddr in &values {
        for &size in &values {
            let expected = if vaddr % MEGAPAGE_SIZE == 0 && size % MEGAPAGE_SIZE == 0 {
                Some(megapage())
            } else if vaddr % PAGE_SIZE == 0 && size % PAGE_SIZE == 0 {
                Some(page())
            } else {
                None
            };
            let got = select_mapping_geometry(vaddr, size);
            assert_eq!(
                got, expected,
                "select_mapping_geometry(vaddr={vaddr:#x}, size={size:#x})"
            );
        }
    }
}
