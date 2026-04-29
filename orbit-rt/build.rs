//! Generate the user-app linker script (`memory.x`) from
//! `orbit_abi::layout` constants and emit a `-L` so dependent binaries
//! pick it up via their `-Clink-arg=-Tmemory.x` rustflag.
//!
//! Centralizing here keeps the ELF base in lockstep with `USER_TEXT_BASE`
//! and bounds the text region by the next user range so an oversized
//! image fails to link rather than silently overlapping the priv heap.
//! Each downstream app crate (umode, orbit-loader, console) just needs
//! to depend on `orbit-rt`; no per-crate `memory.x` or `build.rs`.

use std::env;
use std::fs;
use std::path::PathBuf;

use orbit_abi::layout::{USER_ARGV_BASE, USER_TEXT_BASE};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // The ELF region runs from USER_TEXT_BASE up to (but not into) the
    // §13a.3 argv blob page just below UPROC_PRIV_BASE. Capping LENGTH
    // here turns an oversized image into a link error instead of a
    // runtime collision with the kernel-mapped argv page.
    let length = USER_ARGV_BASE - USER_TEXT_BASE;

    let script = format!(
        r#"OUTPUT_ARCH( "riscv" )

ENTRY( _start )

MEMORY
{{
  RAM : ORIGIN = {origin:#x}, LENGTH = {length:#x}
}}

PHDRS
{{
  text PT_LOAD;
  data PT_LOAD;
  bss PT_LOAD;
  rodata PT_LOAD;
  tls PT_TLS;
}}

SECTIONS
{{
  .text : ALIGN(4096) {{
    PROVIDE(_text_start = .);
    *(.text.init) *(.text .text.*)
    PROVIDE(_text_end = .);
  }} >RAM AT>RAM :text

  PROVIDE(_global_pointer = .);

  .rodata : ALIGN(4096) {{
    PROVIDE(_rodata_start = .);
    *(.rodata .rodata.*)
    PROVIDE(_rodata_end = .);
  }} >RAM AT>RAM :rodata

  .data : ALIGN(4096) {{
    PROVIDE(_data_start = .);
    *(.sdata .sdata.*) *(.data .data.*)
    PROVIDE(_data_end = .);
  }} >RAM AT>RAM :data

  /* TLS template — variant-I model. .tdata is in both :data (loaded
     for the static-image case) and :tls (so the linker emits PT_TLS
     covering it); .tbss is only in :tls (zero-init, no file backing).
     The kernel reads PT_TLS at ELF load and snapshots p_filesz bytes
     onto the Process for per-thread copy-init. Binaries with no
     `#[thread_local]` end up with empty .tdata/.tbss → PT_TLS with
     p_memsz=0, and thread-create skips the TLS mapping entirely.

     Placed *immediately after* .data — same :data PHDR — so the
     `data` PT_LOAD covers a contiguous run [.data .tdata) without
     spanning the .bss segment that follows. PT_TLS p_vaddr lands
     mid-page in practice; the kernel reads the segment data via
     elf.segment_data() (file-relative) so page alignment doesn't
     matter for the snapshot. */
  .tdata : ALIGN(8) {{
    PROVIDE(_tdata_start = .);
    *(.tdata .tdata.*)
    PROVIDE(_tdata_end = .);
  }} >RAM AT>RAM :data :tls

  .tbss (NOLOAD) : ALIGN(8) {{
    PROVIDE(_tbss_start = .);
    *(.tbss .tbss.*) *(.tcommon)
    PROVIDE(_tbss_end = .);
  }} >RAM AT>RAM :tls

  .bss : ALIGN(4096) {{
    PROVIDE(_bss_start = .);
    *(.sbss .sbss.*) *(.bss .bss.*)
    PROVIDE(_bss_end = .);
  }} >RAM AT>RAM :bss
}}
"#,
        origin = USER_TEXT_BASE,
    );

    fs::write(out_dir.join("memory.x"), script).unwrap();

    // -L so a dependent binary's `-Clink-arg=-Tmemory.x` rustflag
    // resolves to this generated file. cargo propagates rustc-link-search
    // emissions from dependency build scripts to the link step of any
    // binary that pulls in orbit-rt.
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rerun-if-changed=build.rs");
}
