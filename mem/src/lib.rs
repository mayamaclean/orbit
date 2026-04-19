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
        return n
    }
    ((n / modulo) + 1) * modulo
}

pub const fn round_u64_down(n: u64, modulo: u64) -> u64 {
    let n_ok = (n % modulo) == 0;
    if n_ok {
        return n
    }
    (n / modulo) * modulo
}

pub const fn round_usize_up(n: usize, modulo: usize) -> usize {
    let n_ok = (n % modulo) == 0;
    if n_ok {
        return n
    }
    ((n / modulo) + 1) * modulo
}

pub const fn round_usize_down(n: usize, modulo: usize) -> usize {
    let n_ok = (n % modulo) == 0;
    if n_ok {
        return n
    }
    (n / modulo) * modulo
}
