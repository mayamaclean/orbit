// Derived from Stephen Marz's osblog (The Adventures of OS):
// https://github.com/sgmarz/osblog — MIT, Copyright (c) 2019 Stephen Marz.
// See THIRD_PARTY_NOTICES.md at the repo root.
//
// mem.S
// Importation of linker symbols

.global _text_start
_text_start: .dword _text_start

.global _text_end
_text_end: .dword _text_end

.global _data_start
_data_start: .dword _data_start

.global _data_end
_data_end: .dword _data_end

.global _rodata_start
_rodata_start: .dword _rodata_start

.global _rodata_end
_rodata_end: .dword _rodata_end

.global _bss_start
_bss_start: .dword _bss_start

.global _bss_end
_bss_end: .dword _bss_end

.global _stack_start
_stack_start: .dword _stack_start

.global _stack_end
_stack_end: .dword _stack_end

.global _reladyn_end
_reladyn_end: .dword _reladyn_end

.global _reladyn_start
_reladyn_start: .dword _reladyn_start