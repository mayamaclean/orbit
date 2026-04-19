OUTPUT_ARCH( "riscv" )

ENTRY( _start )

MEMORY
{
  RAM : ORIGIN = 0x1000, LENGTH = 2048M - 16M
}

PHDRS
{
  text PT_LOAD FILEHDR PHDRS FLAGS(5); /* R E */
  rela.dyn PT_LOAD FLAGS(4);           /* R */
  rodata PT_LOAD FLAGS(4);             /* R */
  gnu.hash PT_LOAD FLAGS(4);           /* R */
  dynsym PT_LOAD FLAGS(4);             /* R */
  hash PT_LOAD FLAGS(4);               /* R */
  dynstr PT_LOAD FLAGS(4);             /* R */
  eh_frame PT_LOAD FLAGS(4);           /* R */
  got PT_LOAD FLAGS(6);                /* RW */
  data PT_LOAD FLAGS(6);               /* RW */
  bss PT_LOAD FLAGS(6);                /* RW */
  dynamic_load PT_LOAD FLAGS(6);       /* RW — makes .dynamic actually land in memory */
  dynamic PT_DYNAMIC;                  /* CRITICAL */
}

SECTIONS
{
  . = 0x1000;

  .text : ALIGN(4096) {
    _text_start = .;
    KEEP(*(.text.init)) *(.text .text.*)
    _text_end = .;
  } >RAM AT>RAM :text

  .rela.dyn : ALIGN(4096) {
    _reladyn_start = .;
      *(.rela .rela*)
    _reladyn_end = .;
  } >RAM AT>RAM :rela.dyn

  .rodata : ALIGN(4096) {
    _rodata_start = .;
    *(.rodata .rodata.*)
    _rodata_end = .;
  } >RAM AT>RAM :rodata

  .gnu.hash : ALIGN(4096) {
    _gnuhash_start = .;
      *(.gnu.hash .gnu.hash*)
    _gnuhash_end = .;
  } >RAM AT>RAM :gnu.hash

  .dynsym : ALIGN(4096) {
    _dynsym_start = .;
      *(.dynsym .dynsym*)
    _dynsym_end = .;
  } >RAM AT>RAM :dynsym

  .hash : ALIGN(4096) {
    _hash_start = .;
      *(.hash .hash*)
    _hash_end = .;
  } >RAM AT>RAM :hash

  .dynstr : ALIGN(4096) {
    _dynstr_start = .;
      *(.dynstr .dynstr*)
    _dynstr_end = .;
  } >RAM AT>RAM :dynstr

  .eh_frame : ALIGN(4096) {
    _ehframe_start = .;
      *(.eh_frame .eh_frame*)
    _ehframe_end = .;
  } >RAM AT>RAM :eh_frame

  .got : ALIGN(4096) {
    _got_start = .;
      *(.got .got*)
    _got_end = .;
  } >RAM AT>RAM :got

  .data : ALIGN(4096) {
    _data_start = .;
    *(.sdata .sdata.*) *(.data .data.*)
    _data_end = .;
  } >RAM AT>RAM :data

  .data.rel.ro : ALIGN(4096) {
      *(.data.rel.ro .data.rel.ro*)
  } >RAM AT>RAM :data

  .bss : ALIGN(4096) {
    _bss_start = .;
    *(.sbss .sbss.*) *(.bss .bss.*)
    _bss_end = .;
  } >RAM AT>RAM :bss

  .dynamic : ALIGN(4096) {
    PROVIDE(_DYNAMIC = .);
    *(.dynamic .dynamic*)
    PROVIDE(_DYNAMIC_END = .);
  } >RAM AT>RAM :dynamic_load :dynamic

   PROVIDE(_global_pointer = .);

  . = ALIGN(4096);
  PROVIDE(_stack_start = .);
  PROVIDE(_stack_end = _stack_start + 0x80000);
  PROVIDE(_memory_start = ORIGIN(RAM));
  PROVIDE(_memory_end = ORIGIN(RAM) + LENGTH(RAM));
}
