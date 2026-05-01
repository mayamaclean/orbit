#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
pub mod frame;

pub const fn prev_power_of_two(num: usize) -> usize {
    1 << (usize::BITS as usize - num.leading_zeros() as usize - 1)
}

pub const fn round_u64_up(n: u64, modulo: u64) -> u64 {
    let n_ok = (n % modulo) == 0;
    if n_ok {
        return n;
    }
    ((n / modulo) + 1) * modulo
}

pub const fn round_u64_down(n: u64, modulo: u64) -> u64 {
    let n_ok = (n % modulo) == 0;
    if n_ok {
        return n;
    }
    (n / modulo) * modulo
}

pub const fn round_usize_up(n: usize, modulo: usize) -> usize {
    let n_ok = (n % modulo) == 0;
    if n_ok {
        return n;
    }
    ((n / modulo) + 1) * modulo
}

pub const fn round_usize_down(n: usize, modulo: usize) -> usize {
    let n_ok = (n % modulo) == 0;
    if n_ok {
        return n;
    }
    (n / modulo) * modulo
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- prev_power_of_two ----

    #[test]
    fn prev_power_of_two_exact_power() {
        assert_eq!(prev_power_of_two(1), 1);
        assert_eq!(prev_power_of_two(2), 2);
        assert_eq!(prev_power_of_two(4), 4);
        assert_eq!(prev_power_of_two(1 << 20), 1 << 20);
    }

    #[test]
    fn prev_power_of_two_non_power() {
        assert_eq!(prev_power_of_two(3), 2);
        assert_eq!(prev_power_of_two(5), 4);
        assert_eq!(prev_power_of_two(7), 4);
        assert_eq!(prev_power_of_two(9), 8);
        assert_eq!(prev_power_of_two((1 << 20) + 1), 1 << 20);
    }

    // ---- round_u64_up ----

    #[test]
    fn round_up_exact_multiple_is_identity() {
        assert_eq!(round_u64_up(0, 4096), 0);
        assert_eq!(round_u64_up(4096, 4096), 4096);
        assert_eq!(round_u64_up(8192, 4096), 8192);
    }

    #[test]
    fn round_up_rounds_up_to_next_multiple() {
        assert_eq!(round_u64_up(1, 4096), 4096);
        assert_eq!(round_u64_up(4095, 4096), 4096);
        assert_eq!(round_u64_up(4097, 4096), 8192);
    }

    // ---- round_u64_down ----

    #[test]
    fn round_down_exact_multiple_is_identity() {
        assert_eq!(round_u64_down(0, 4096), 0);
        assert_eq!(round_u64_down(4096, 4096), 4096);
    }

    #[test]
    fn round_down_rounds_down_to_prev_multiple() {
        assert_eq!(round_u64_down(4095, 4096), 0);
        assert_eq!(round_u64_down(4097, 4096), 4096);
        assert_eq!(round_u64_down(8191, 4096), 4096);
    }

    // ---- round_usize_{up,down} symmetry with u64 ----

    #[test]
    fn usize_variants_match_u64_on_shared_values() {
        for &n in &[0usize, 1, 4095, 4096, 4097, 8192, 1 << 20] {
            for &m in &[1usize, 2, 4096, 2 * 1024 * 1024] {
                assert_eq!(
                    round_usize_up(n, m) as u64,
                    round_u64_up(n as u64, m as u64),
                    "round_up mismatch n={n} m={m}"
                );
                assert_eq!(
                    round_usize_down(n, m) as u64,
                    round_u64_down(n as u64, m as u64),
                    "round_down mismatch n={n} m={m}"
                );
            }
        }
    }

    // ---- round_up ∘ round_down invariants ----

    #[test]
    fn round_down_is_idempotent() {
        for &n in &[0u64, 1, 4095, 4096, 4097, 8192, 9999] {
            for &m in &[1u64, 2, 4096] {
                let r = round_u64_down(n, m);
                assert_eq!(round_u64_down(r, m), r);
                assert!(r <= n);
                assert!(r % m == 0);
            }
        }
    }

    #[test]
    fn round_up_is_idempotent() {
        for &n in &[0u64, 1, 4095, 4096, 4097, 8192, 9999] {
            for &m in &[1u64, 2, 4096] {
                let r = round_u64_up(n, m);
                assert_eq!(round_u64_up(r, m), r);
                assert!(r >= n);
                assert!(r % m == 0);
            }
        }
    }
}
