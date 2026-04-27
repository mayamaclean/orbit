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
  /* Stack at the BOTTOM of RAM, ahead of any loaded sections. SP grows
     down from `_stack_end` toward `_stack_start = ORIGIN(RAM)`; an
     overflow walks past 0x80000000 into qemu-virt MMIO/unmapped and
     traps with an instruction-/load-access fault. Previously the stack
     sat above .data/.bss and overflows silently chewed through
     statics — exactly the failure mode that wiped `serial::SERIAL`
     when kmain grew enough to push BSS into the page-table slot. */
  . = ORIGIN(RAM);
  PROVIDE(_stack_start = .);
  . = . + 0x80000;            /* 512 KiB stack region */
  PROVIDE(_stack_end = .);

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

  /* Page-table pool: 128 KiB after all loaded sections, capped well
     below `TRAP_FRAME_ADDR` (0x80800000). Self-locating so kmain
     growth doesn't push it into anything else. */
  . = ALIGN(4096);
  PROVIDE(_id_map_tables = .);

  PROVIDE(_memory_start = ORIGIN(RAM));
  PROVIDE(_memory_end = ORIGIN(RAM) + LENGTH(RAM));
}

/* Page-table pool stays clear of the M-mode trap frames at 0x80800000.
   If kmain ever grows enough to push `_id_map_tables` past
   `0x80800000 - 128 KiB`, the link fails loudly. */
ASSERT(_id_map_tables + 0x20000 <= 0x80800000,
       "bl page-table pool collides with TRAP_FRAME_ADDR — kmain too large?")
