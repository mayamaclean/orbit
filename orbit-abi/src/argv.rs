//! Layout of the §13a.3 argv blob.
//!
//! Producer side: userland packs `[ArgvHeader][argv_offsets;
//! argc][string_table]` into a buffer, hands it to
//! `CREATE_PROCESS_EX`. Kernel copies the buffer into a fresh
//! kernel_pages page, fixes up the offsets into absolute pointers
//! (since the new process maps the page at the fixed
//! [`crate::layout::USER_ARGV_BASE`]), and installs the page R+U+S
//! in the new process's PT.
//!
//! Consumer side: orbit-rt's startup calls the `argv_envp` syscall;
//! a non-zero return is the VA of the mapped blob (always
//! `USER_ARGV_BASE` in v1). [`Argv::parse`] walks `[ArgvHeader]
//! [argv_ptrs][string_table]` and yields per-arg byte slices.
//!
//! v1 carries argv only — no envp. envp lands when there's a
//! consumer that wants it.

use core::mem::size_of;

/// Header at the start of the blob (8 bytes). `argc` precedes a
/// `[*const u8; argc]` array of absolute user-VA pointers into the
/// trailing string table; each pointed-to string is NUL-terminated.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ArgvHeader {
    pub argc: u32,
    pub _reserved: u32,
}

/// Producer-side packed blob format:
///
/// ```text
/// +0:                       [ArgvHeader { argc, _reserved }]
/// +8:                       [u64; argc] arg_offsets — byte offset
///                              of each NUL-terminated string,
///                              relative to the START of the blob
/// +8 + 8*argc:              [u8; ...] string table (NUL-terminated
///                              UTF-8, concatenated)
/// ```
///
/// Userland builds this. The kernel does the offset → absolute
/// pointer fixup at process creation by replacing each u64 offset
/// with `USER_ARGV_BASE + offset`; the slot widths match
/// (`u64` ↔ `*const u8` on rv64) so the rewrite is in place — see
/// kmain's `run_create_process_ex_req` for the fixup site.
///
/// Maximum blob size is one page (4096 bytes). Realistically holds
/// ~256 short args; the smoke uses 3.
pub const ARGV_BLOB_MAX: usize = 4096;

/// Offset of the offsets array within the blob.
pub const ARGV_OFFSETS_OFFSET: usize = size_of::<ArgvHeader>();

/// Offset of the string table for `argc` entries.
pub const fn argv_strings_offset(argc: u32) -> usize {
    ARGV_OFFSETS_OFFSET + (argc as usize) * size_of::<u64>()
}

/// Pack a slice of byte strings into the producer-side blob format
/// described above. `out` must be at least
/// `argv_strings_offset(args.len() as u32) + sum(arg.len() + 1)`
/// bytes; returns the number of bytes written, or `None` if the
/// inputs would exceed [`ARGV_BLOB_MAX`].
///
/// Each arg is written followed by a NUL terminator. The kernel
/// in-place-fixes the offsets to absolute pointers at process
/// creation; the consumer reads as `[*const u8; argc]`.
pub fn pack(args: &[&[u8]], out: &mut [u8]) -> Option<usize> {
    let argc = args.len();
    let header_end = argv_strings_offset(argc as u32);
    if header_end > out.len() {
        return None;
    }
    let strings_total: usize = args.iter().map(|a| a.len() + 1).sum();
    let total = header_end + strings_total;
    if total > out.len() || total > ARGV_BLOB_MAX {
        return None;
    }

    // Header.
    out[..size_of::<ArgvHeader>()].fill(0);
    let header = ArgvHeader {
        argc: argc as u32,
        _reserved: 0,
    };
    let header_bytes = unsafe {
        core::slice::from_raw_parts(&header as *const _ as *const u8, size_of::<ArgvHeader>())
    };
    out[..size_of::<ArgvHeader>()].copy_from_slice(header_bytes);

    // Offsets + strings.
    let mut cursor = header_end;
    for (i, arg) in args.iter().enumerate() {
        let offset = cursor as u64;
        let slot = ARGV_OFFSETS_OFFSET + i * size_of::<u64>();
        out[slot..slot + 8].copy_from_slice(&offset.to_le_bytes());
        out[cursor..cursor + arg.len()].copy_from_slice(arg);
        out[cursor + arg.len()] = 0;
        cursor += arg.len() + 1;
    }
    Some(total)
}

/// Consumer-side view: the blob the kernel mapped at
/// [`crate::layout::USER_ARGV_BASE`] uses *absolute pointers* (not
/// offsets) so userland walks it without arithmetic. Pointers are
/// stable for the process's lifetime.
///
/// Read with [`Argv::from_ptr`] which validates `argc` against
/// [`ARGV_BLOB_MAX`] and yields per-arg `&[u8]` slices.
pub struct Argv<'a> {
    argc: u32,
    /// Pointer to the start of the absolute-pointer array (not the
    /// header).
    argv: *const *const u8,
    _marker: core::marker::PhantomData<&'a u8>,
}

impl<'a> Argv<'a> {
    /// Parse the blob mapped at `va`. Returns `None` if `argc` is
    /// implausibly large (corrupt blob).
    ///
    /// # Safety
    /// `va` must be a live mapping of a kernel-produced argv blob —
    /// i.e. `USER_ARGV_BASE` of a process spawned via
    /// `CREATE_PROCESS_EX`. Reading from the constant when no argv
    /// was provided would fault; consult `argv_envp()` first.
    pub unsafe fn from_ptr(va: *const ArgvHeader) -> Option<Self> {
        let header = unsafe { va.read() };
        // Sanity cap. argv_strings_offset would overflow well before
        // this, but explicit max keeps the bound visible.
        if argv_strings_offset(header.argc) > ARGV_BLOB_MAX {
            return None;
        }
        let argv = unsafe { (va as *const u8).add(ARGV_OFFSETS_OFFSET) } as *const *const u8;
        Some(Self {
            argc: header.argc,
            argv,
            _marker: core::marker::PhantomData,
        })
    }

    pub fn len(&self) -> usize {
        self.argc as usize
    }

    pub fn is_empty(&self) -> bool {
        self.argc == 0
    }

    /// Borrow the i-th argument as a NUL-terminated byte slice (NUL
    /// excluded). Returns `None` if `i >= argc` or the pointer
    /// fails sanity checks.
    pub fn get(&self, i: usize) -> Option<&'a [u8]> {
        if i >= self.argc as usize {
            return None;
        }
        let p = unsafe { *self.argv.add(i) };
        if p.is_null() {
            return None;
        }
        // Walk to NUL. Cap at ARGV_BLOB_MAX to bound a malformed
        // blob's effect on the consumer.
        let mut len = 0;
        while len < ARGV_BLOB_MAX {
            let b = unsafe { *p.add(len) };
            if b == 0 {
                break;
            }
            len += 1;
        }
        Some(unsafe { core::slice::from_raw_parts(p, len) })
    }
}
