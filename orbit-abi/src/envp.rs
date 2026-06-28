//! Layout of the envp blob.
//!
//! Wire format is **identical** to the argv blob — see
//! [`crate::argv`] for the on-the-wire layout, packer, and reader. The
//! only differences are:
//!
//! - the kernel maps the page at [`crate::layout::USER_ENVP_BASE`]
//!   instead of [`crate::layout::USER_ARGV_BASE`];
//! - entries are conventionally NUL-terminated `KEY=VALUE` byte
//!   strings (Linux-style), but the format itself doesn't enforce
//!   the `=` separator — the orbit-rt env wrapper splits on `=` at
//!   parse time.
//!
//! Sharing the format means the producer-side packer
//! ([`crate::argv::pack`]) and the consumer-side parser
//! ([`crate::argv::Argv`]) work identically for envp; this module
//! re-exports them under env-flavored names so call sites that work
//! with environment data read naturally:
//!
//! ```ignore
//! use orbit_abi::envp::{Envp, EnvpHeader, ENVP_BLOB_MAX, pack};
//!
//! let mut buf = [0u8; ENVP_BLOB_MAX];
//! let n = pack(&[b"PATH=/bin", b"HOME=/", b"TERM=dumb"], &mut buf).unwrap();
//! // ... copy into a page-aligned buffer and pass its VA to
//! //     create_process_v2's envp arg (a non-page-aligned envp VA is
//! //     rejected with EINVAL) ...
//!
//! // Consumer side:
//! let envp = unsafe { Envp::from_ptr(USER_ENVP_BASE as *const EnvpHeader) }
//!     .expect("malformed envp blob");
//! for i in 0..envp.len() {
//!     let entry = envp.get(i).unwrap(); // "KEY=VALUE" bytes
//!     // ... split on b'=' and stash in the env table ...
//! }
//! ```
//!
//! The aliases below are zero-cost: there is no separate envp Rust
//! type. Any code that already speaks `Argv` can speak envp by
//! pointing at the right VA.

pub use crate::argv::ARGV_BLOB_MAX as ENVP_BLOB_MAX;
pub use crate::argv::ARGV_OFFSETS_OFFSET as ENVP_OFFSETS_OFFSET;
pub use crate::argv::Argv as Envp;
pub use crate::argv::ArgvHeader as EnvpHeader;
pub use crate::argv::argv_strings_offset as envp_strings_offset;
pub use crate::argv::pack;
