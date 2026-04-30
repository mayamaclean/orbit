//! User-side accessors for the §13a.3 argv blob (mapped at
//! [`orbit_abi::layout::USER_ARGV_BASE`]) and §13e envp blob (mapped
//! at [`orbit_abi::layout::USER_ENVP_BASE`]). Both share a wire
//! format, so the same [`Argv`]-shaped view serves both.
//!
//! The kernel's `argv_envp` syscall returns the pair `(argv_va,
//! envp_va)` in one trap; `0` in either slot means "not installed."
//! We resolve once on first access and cache both VAs in static
//! atomics so subsequent reads are register-cheap.

use core::sync::atomic::{AtomicUsize, Ordering};

use orbit_abi::argv::{Argv, ArgvHeader};
use orbit_abi::user::argv_envp;

/// 0 = uninit, `usize::MAX` = no blob (cached miss), other = blob VA.
static ARGV_VA: AtomicUsize = AtomicUsize::new(0);
static ENVP_VA: AtomicUsize = AtomicUsize::new(0);

const NO_BLOB: usize = usize::MAX;

fn resolve_pair() {
    let (argv, envp) = argv_envp();
    let argv_store = if argv == 0 { NO_BLOB } else { argv };
    let envp_store = if envp == 0 { NO_BLOB } else { envp };
    // Race-tolerant: the kernel's answer is stable for the process
    // lifetime, so concurrent first-callers all write the same value.
    ARGV_VA.store(argv_store, Ordering::Release);
    ENVP_VA.store(envp_store, Ordering::Release);
}

fn argv_va() -> usize {
    let cached = ARGV_VA.load(Ordering::Acquire);
    if cached != 0 {
        return cached;
    }
    resolve_pair();
    ARGV_VA.load(Ordering::Acquire)
}

fn envp_va() -> usize {
    let cached = ENVP_VA.load(Ordering::Acquire);
    if cached != 0 {
        return cached;
    }
    resolve_pair();
    ENVP_VA.load(Ordering::Acquire)
}

/// View into the process's argv. Empty if no argv was provided at
/// process creation (i.e. spawned via the bare `create_process`
/// syscall, or via `create_process_with_argv` with an empty blob).
pub fn args() -> Args {
    let va = argv_va();
    if va == NO_BLOB {
        return Args { inner: None };
    }
    // SAFETY: kernel guarantees the page at `va` is mapped R+U for
    // this process's lifetime if argv_envp returned non-zero.
    let inner = unsafe { Argv::from_ptr(va as *const ArgvHeader) };
    Args { inner }
}

/// View into the process's envp blob. Wire format identical to argv;
/// entries are conventionally NUL-terminated `KEY=VALUE` byte
/// strings. Empty if no envp was installed at process creation.
pub fn envp() -> Args {
    let va = envp_va();
    if va == NO_BLOB {
        return Args { inner: None };
    }
    // SAFETY: same guarantee as `args()` — the kernel maps the envp
    // page R+U for the process lifetime when `argv_envp` reports a
    // non-zero envp VA.
    let inner = unsafe { Argv::from_ptr(va as *const ArgvHeader) };
    Args { inner }
}

/// Iterable wrapper around the optional [`Argv`] view. `len() == 0`
/// when there's no blob (and also when argv_envp returned a corrupt
/// blob — `Argv::from_ptr` validates).
pub struct Args {
    inner: Option<Argv<'static>>,
}

impl Args {
    pub fn len(&self) -> usize {
        self.inner.as_ref().map_or(0, |a| a.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow the i-th entry as a NUL-stripped byte slice.
    /// Returns `None` if `i` is out of range or there's no blob.
    pub fn get(&self, i: usize) -> Option<&'static [u8]> {
        self.inner.as_ref().and_then(|a| a.get(i))
    }

    /// Iterate over all entries as byte slices.
    pub fn iter(&self) -> ArgsIter<'_> {
        ArgsIter { args: self, i: 0 }
    }
}

pub struct ArgsIter<'a> {
    args: &'a Args,
    i: usize,
}

impl<'a> Iterator for ArgsIter<'a> {
    type Item = &'static [u8];

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.args.get(self.i)?;
        self.i += 1;
        Some(item)
    }
}
