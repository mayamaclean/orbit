OUTPUT_ARCH( "riscv" )

ENTRY( _start )

MEMORY
{
  RAM : ORIGIN = 0x90000000, LENGTH = 64M
}

PHDRS
{
  text PT_LOAD;
  data PT_LOAD;
  bss PT_LOAD;
  rodata PT_LOAD;
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

  .data : ALIGN(4096) {
    PROVIDE(_data_start = .);
    *(.sdata .sdata.*) *(.data .data.*)
    PROVIDE(_data_end = .);
  } >RAM AT>RAM :data

  .bss : ALIGN(4096) {
    PROVIDE(_bss_start = .);
    *(.sbss .sbss.*) *(.bss .bss.*)
    PROVIDE(_bss_end = .);
  } >RAM AT>RAM :bss
}
