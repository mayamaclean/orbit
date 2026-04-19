OUTPUT_ARCH( "riscv" )

ENTRY( _start )

MEMORY
{
  RAM : ORIGIN = 0x80000000, LENGTH = 64M
}

/*
PHDRS is short for "program headers", which we specify three here:
text - CPU instructions (executable sections)
data - Global, initialized variables
bss  - Global, uninitialized variables (all will be set to 0 by boot.S)

The command PT_LOAD tells the linker that these sections will be loaded
from the file into memory.

We can actually stuff all of these into a single program header, but by
splitting it up into three, we can actually use the other PT_* commands
such as PT_DYNAMIC, PT_INTERP, PT_NULL to tell the linker where to find
additional information.

However, for our purposes, every section will be loaded from the program
headers.
*/
PHDRS
{
  text PT_LOAD FLAGS(5);
  rodata PT_LOAD FLAGS(4);
  data PT_LOAD FLAGS(6);
  bss PT_LOAD FLAGS(6);
}

SECTIONS
{
  .text : ALIGN(4096) {
    PROVIDE(_text_start = .);
    *(.text.init) *(.text .text.*)
    PROVIDE(_text_end = .);
  } >RAM AT>RAM :text

  PROVIDE(_global_pointer = .);

  .rodata : ALIGN(4096) {
    PROVIDE(_rodata_start = .);
    *(.rodata .rodata.*)
    PROVIDE(_rodata_end = .);
  } >RAM AT>RAM :rodata

  .bss : ALIGN(4096) {
    PROVIDE(_bss_start = .);
    *(.sbss .sbss.*) *(.bss .bss.*)
    PROVIDE(_bss_end = .);
  } >RAM AT>RAM :bss

  .data : ALIGN(4096) {
    PROVIDE(_data_start = .);
    *(.sdata .sdata.*) 
    *(.data .data.*)
    PROVIDE(_data_end = .);
  } >RAM AT>RAM :data

  . = ALIGN(4096);
  PROVIDE(_stack_start = .);
  PROVIDE(_stack_end = _stack_start + 0x80000);
  PROVIDE(_memory_start = ORIGIN(RAM));
  PROVIDE(_memory_end = ORIGIN(RAM) + LENGTH(RAM));
}
