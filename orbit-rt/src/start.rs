//! §13b — process startup ABI.
//!
//! orbit-rt provides the `_start` symbol named in `memory.x`'s
//! `ENTRY(_start)`. Downstream binaries write `#![no_main]` and
//! provide a `main` symbol; orbit-rt's `_start` then runs:
//!
//! 1. **Eager argv resolve.** Calls [`crate::argv::args`] once so the
//!    one-syscall round-trip lands deterministically at startup
//!    rather than the first `args()` call. Callers that don't need
//!    argv pay a single ecall; the win is that a kernel-side argv
//!    failure surfaces here, before `main` has done anything
//!    user-visible.
//! 2. **Calls user `main` → i32.** Declared `extern "C"`, resolved
//!    at link time. The convention is the upstream-Rust shape:
//!    `#[unsafe(no_mangle)] extern "C" fn main() -> i32 { ... }`.
//! 3. **Exits.** Calls [`orbit_abi::user::exit`] with the return
//!    code so a `return 0;` from `main` cleanly winds down the
//!    process.
//!
//! ## Panic hook
//!
//! v1 keeps the `panic = "abort"` shape — each binary supplies its
//! own `#[panic_handler]`. orbit-std (§13d) introduces
//! `set_hook` / `take_hook` and the default-hook-prints-then-aborts
//! behavior; until then the binary's panic handler is the abort
//! point.
//!
//! ## Why a static reference to `_start`?
//!
//! Without an external use, an rlib's `#[unsafe(no_mangle)]` symbol
//! can be discarded by the linker before `ENTRY(_start)` resolves
//! against it. The function-pointer `pub static` below (also
//! `#[used]`) forces the symbol into the binary's symbol table from
//! the rlib side, so a downstream `use orbit_rt as _;` is enough to
//! land both `_start` and the static.

use orbit_abi::user;

unsafe extern "C" {
    /// User-supplied entry. Must be defined in the binary as
    /// `#[unsafe(no_mangle)] extern "C" fn main() -> i32`.
    fn main() -> i32;
}

/// Entry point named by `memory.x`'s `ENTRY(_start)`. Provided by
/// orbit-rt so a downstream binary only needs `fn main() -> i32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _start() -> ! {
    // Eagerly resolve the argv blob. Cheap (one syscall, cached on
    // the static after the first call); deterministic point of
    // failure if the kernel ever returns a malformed blob.
    let _ = crate::argv::args();

    let code = unsafe { main() };
    user::exit(code as isize);
}

/// Linker anchor — see module doc. Forces the `_start` symbol to
/// survive cross-rlib linking so a downstream binary's `ENTRY(_start)`
/// resolves without each binary having to mention `_start` directly.
#[used]
#[unsafe(no_mangle)]
pub static __ORBIT_RT_ENTRY: unsafe extern "C" fn() -> ! = _start;
