use alloc::vec::Vec;

use process::PhysBacking;

/// Static TLS image extracted from a binary's `PT_TLS` program header.
/// Snapshotted at ELF-load time onto the `Process` so per-thread create
/// can copy-init the TLS block without re-walking the user PT.
///
/// `template` carries the first `filesz` bytes of the TLS image
/// (`.tdata` initial values); the trailing `memsz - filesz` bytes are
/// implicitly zero and never stored. Binaries with no `#[thread_local]`
/// produce `memsz == 0`, in which case `ElfInfo::tls` is `None`.
#[derive(Debug)]
pub struct TlsTemplate {
    pub template: Vec<u8>,
    pub memsz: usize,
    pub align: usize,
}

#[derive(Debug)]
pub struct ElfInfo {
    pub entrypoint: usize,
    pub segments: Vec<PhysBacking>,
    /// `Some` iff the ELF declares a non-empty `PT_TLS`. `None` skips
    /// per-thread TLS allocation entirely; `tp` defaults to 0 and any
    /// `#[thread_local]` access in the binary would have been a link
    /// error before getting here.
    pub tls: Option<TlsTemplate>,
}
