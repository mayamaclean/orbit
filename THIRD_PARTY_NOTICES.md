# Third-party notices

Orbit vendors or adapts the following third-party code and assets. Each entry
lists the files affected and the applicable license; full license texts are
reproduced at the bottom of this file.

Crates pulled from crates.io as ordinary Cargo dependencies (e.g.
`linked_list_allocator`, `ns16550a`) are not listed here; they are distributed
under their own licenses via the registry.

---

## buddy_system_allocator

- **Upstream:** <https://github.com/rcore-os/buddy_system_allocator> (~v0.11)
- **Copyright:** Copyright 2019-2020 Jiajie Chen
- **License:** MIT (text below)
- **Files:**
  - `mem/src/frame.rs` — the buddy-system `FrameAllocator`, adapted (dropped
    `LockedFrameAllocator` and `alloc_at`, const-fn `new()`, added
    `allocated()`/`total()` accounting and host unit tests)
  - `mem/src/lib.rs` — the `prev_power_of_two` helper

## Terminus Font

- **Upstream:** <https://terminus-font.sourceforge.net/>
- **Copyright:** Copyright (C) 2020 Dimitar Toshkov Zhekov, with Reserved Font
  Name "Terminus Font"
- **License:** SIL Open Font License, Version 1.1 (text below)
- **Files:**
  - `ter-u16n.bdf` — the 8x16 normal-weight BDF, unmodified
  - `kmain/src/drivers/fonts/terminus.rs` — glyph bitmaps for codepoints
    0..256, generated from `ter-u16n.bdf` by `kmain/tools/bdf_to_rust.py`
    (a format conversion; per OFL this is a Modified Version and does not
    present the Reserved Font Name as a primary font name)

## Liberation Mono

- **Upstream:** <https://github.com/liberationfonts/liberation-fonts>
  (Version 2.1.5)
- **Copyright:** Digitized data copyright (c) 2010 Google Corporation with
  Reserved Font Arimo, Tinos and Cousine. Copyright (c) 2012 Red Hat, Inc.
  with Reserved Font Name Liberation.
- **License:** SIL Open Font License, Version 1.1 (text below)
- **Files:**
  - `rootfs/usr/share/fonts/LiberationMono-Regular.ttf` — unmodified;
    staged into `disk.img` by `tools/build-disk.sh` and loaded at runtime by
    the std-on-orbit framebuffer demos (`hello-fb-std`, `hello-ratatui-std`,
    `orbit-top-std`)

---

## MIT License (buddy_system_allocator)

```
MIT License

Copyright 2019-2020 Jiajie Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## SIL Open Font License, Version 1.1 (Terminus Font, Liberation Mono)

Applies to the Terminus Font and Liberation Mono entries above, each under
its own copyright notice as listed in its entry.

```
This Font Software is licensed under the SIL Open Font License, Version 1.1.
This license is copied below, and is also available with a FAQ at:
https://openfontlicense.org


-----------------------------------------------------------
SIL OPEN FONT LICENSE Version 1.1 - 26 February 2007
-----------------------------------------------------------

PREAMBLE
The goals of the Open Font License (OFL) are to stimulate worldwide
development of collaborative font projects, to support the font creation
efforts of academic and linguistic communities, and to provide a free and
open framework in which fonts may be shared and improved in partnership
with others.

The OFL allows the licensed fonts to be used, studied, modified and
redistributed freely as long as they are not sold by themselves. The
fonts, including any derivative works, can be bundled, embedded,
redistributed and/or sold with any software provided that any reserved
names are not used by derivative works. The fonts and derivatives,
however, cannot be released under any other type of license. The
requirement for fonts to remain under this license does not apply
to any document created using the fonts or their derivatives.

DEFINITIONS
"Font Software" refers to the set of files released by the Copyright
Holder(s) under this license and clearly marked as such. This may
include source files, build scripts and documentation.

"Reserved Font Name" refers to any names specified as such after the
copyright statement(s).

"Original Version" refers to the collection of Font Software components as
distributed by the Copyright Holder(s).

"Modified Version" refers to any derivative made by adding to, deleting,
or substituting -- in part or in whole -- any of the components of the
Original Version, by changing formats or by porting the Font Software to a
new environment.

"Author" refers to any designer, engineer, programmer, technical
writer or other person who contributed to the Font Software.

PERMISSION & CONDITIONS
Permission is hereby granted, free of charge, to any person obtaining
a copy of the Font Software, to use, study, copy, merge, embed, modify,
redistribute, and sell modified and unmodified copies of the Font
Software, subject to the following conditions:

1) Neither the Font Software nor any of its individual components,
in Original or Modified Versions, may be sold by itself.

2) Original or Modified Versions of the Font Software may be bundled,
redistributed and/or sold with any software, provided that each copy
contains the above copyright notice and this license. These can be
included either as stand-alone text files, human-readable headers or
in the appropriate machine-readable metadata fields within text or
binary files as long as those fields can be easily viewed by the user.

3) No Modified Version of the Font Software may use the Reserved Font
Name(s) unless explicit written permission is granted by the corresponding
Copyright Holder. This restriction only applies to the primary font name as
presented to the users.

4) The name(s) of the Copyright Holder(s) or the Author(s) of the Font
Software shall not be used to promote, endorse or advertise any
Modified Version, except to acknowledge the contribution(s) of the
Copyright Holder(s) and the Author(s) or with their explicit written
permission.

5) The Font Software, modified or unmodified, in part or in whole,
must be distributed entirely under this license, and must not be
distributed under any other license. The requirement for fonts to
remain under this license does not apply to any document created
using the Font Software.

TERMINATION
This license becomes null and void if any of the above conditions are
not met.

DISCLAIMER
THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
OF COPYRIGHT, PATENT, TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL THE
COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
INCLUDING ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL
DAMAGES, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM
OTHER DEALINGS IN THE FONT SOFTWARE.
```
