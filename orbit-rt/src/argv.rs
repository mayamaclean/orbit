//! User-side accessor for the §13a.3 argv blob mapped at
//! [`orbit_abi::layout::USER_ARGV_BASE`]. Lazily resolved on first
//! call; result cached in a static atomic for cheap subsequent
//! reads.

use core::sync::atomic::{AtomicUsize, Ordering};

use orbit_abi::argv::{Argv, ArgvHeader};
use orbit_abi::user::argv_envp;

/// 0 = uninit, `usize::MAX` = no argv (cached miss), other = blob VA.
static ARGV_VA: AtomicUsize = AtomicUsize::new(0);

const NO_ARGV: usize = usize::MAX;

fn resolve() -> usize {
    let cached = ARGV_VA.load(Ordering::Acquire);
    if cached != 0 {
        return cached;
    }
    let va = argv_envp();
    let to_store = if va == 0 { NO_ARGV } else { va };
    // Race-tolerant: first writer wins; later writers see same value
    // anyway since the kernel's answer is stable for the process.
    ARGV_VA.store(to_store, Ordering::Release);
    to_store
}

/// View into the process's argv. Empty if no argv was provided at
/// process creation (i.e. spawned via the bare `create_process`
/// syscall instead of `create_process_ex`).
pub fn args() -> Args {
    let va = resolve();
    if va == NO_ARGV {
        return Args { inner: None };
    }
    // SAFETY: kernel guarantees the page at `va` is mapped R+U for
    // this process's lifetime if argv_envp returned non-zero.
    let inner = unsafe { Argv::from_ptr(va as *const ArgvHeader) };
    Args { inner }
}

/// Iterable wrapper around the optional [`Argv`] view. `len() == 0`
/// when there's no argv (and also when argv_envp returned a
/// corrupt blob — `Argv::from_ptr` validates).
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

    /// Borrow the i-th argument as a NUL-stripped byte slice.
    /// Returns `None` if `i` is out of range or there's no argv.
    pub fn get(&self, i: usize) -> Option<&'static [u8]> {
        self.inner.as_ref().and_then(|a| a.get(i))
    }

    /// Iterate over all arguments as byte slices.
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
