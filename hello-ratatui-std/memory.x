/* Linker script for `riscv64gc-unknown-orbit` user binaries. Mirrors
   the orbit-rt `memory.x` shape: ELF base at `USER_TEXT_BASE` and an
   upper bound that stops at the kernel-mapped argv page just below
   `UPROC_PRIV_BASE`. Length = UPROC_PRIV_BASE - USER_TEXT_BASE - PAGE_SIZE
   = 0x3_0000_0000 - 0x2_2000_0000 - 0x1000 = 0xDFFFF000. */

OUTPUT_ARCH( "riscv" )

ENTRY( _start )

MEMORY
{
  RAM : ORIGIN = 0x220000000, LENGTH = 0xDFFFF000
}

PHDRS
{
  text PT_LOAD;
  data PT_LOAD;
  bss PT_LOAD;
  rodata PT_LOAD;
  tls PT_TLS;
}

SECTIONS
{
  .text : ALIGN(4096) {
    PROVIDE(_text_start = .);
    *(.text.init) *(.text .text.*)
    PROVIDE(_text_end = .);
  } >RAM AT>RAM :text

  /* Both names: `_global_pointer` is the historical orbit-rt name,
     `__global_pointer$` is what lld auto-provides and what
     library/std/src/sys/pal/orbit/start.rs references. Bind them to
     the same address so gp-relative loads target a stable location. */
  PROVIDE(_global_pointer = .);
  PROVIDE(__global_pointer$ = .);

  .rodata : ALIGN(4096) {
    PROVIDE(_rodata_start = .);
    *(.rodata .rodata.*)
    PROVIDE(_rodata_end = .);
  } >RAM AT>RAM :rodata

  .data : ALIGN(4096) {
    PROVIDE(_data_start = .);
    *(.sdata .sdata.*) *(.data .data.*)
    PROVIDE(_data_end = .);
  } >RAM AT>RAM :data

  /* TLS template — same shape as orbit-rt's memory.x. PT_TLS gives
     the kernel a snapshot it copies per-thread at create_thread time. */
  .tdata : ALIGN(8) {
    PROVIDE(_tdata_start = .);
    *(.tdata .tdata.*)
    PROVIDE(_tdata_end = .);
  } >RAM AT>RAM :data :tls

  .tbss (NOLOAD) : ALIGN(8) {
    PROVIDE(_tbss_start = .);
    *(.tbss .tbss.*) *(.tcommon)
    PROVIDE(_tbss_end = .);
  } >RAM AT>RAM :tls

  .bss : ALIGN(4096) {
    PROVIDE(_bss_start = .);
    *(.sbss .sbss.*) *(.bss .bss.*)
    PROVIDE(_bss_end = .);
  } >RAM AT>RAM :bss
}
