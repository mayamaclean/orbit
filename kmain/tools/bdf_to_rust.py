#!/usr/bin/env python3
"""Convert a BDF bitmap font into a Rust static-array module.

Reads a BDF file and emits a `pub static {NAME}: [[u8; HEIGHT]; 256]`
indexed by Unicode codepoint. Codepoints outside 0..=255 are skipped;
missing codepoints in range are filled with a zero glyph.

BDF stores each row of a glyph as a big-endian hex byte with bit 7 =
leftmost pixel. We emit those bytes verbatim — the blit routine is
expected to use MSB-first column order, which is the standard
convention for BDF / PSF / VGA bitmap fonts.

Usage:
    bdf_to_rust.py <input.bdf> <output.rs> <NAME> [--height N]

Example:
    bdf_to_rust.py ter-u16n.bdf terminus.rs TERMINUS_8X16 --height 16
"""

import argparse
import re
import sys
from pathlib import Path


def parse_attribution(text: str) -> list[str]:
    """Return the BDF's COPYRIGHT and NOTICE property values, if present.

    These carry the font's license attribution (e.g. Terminus ships
    'Copyright (C) 2020 Dimitar Toshkov Zhekov' + the OFL notice), which
    must survive into the generated Rust so the embedded glyphs stay
    attributed. See THIRD_PARTY_NOTICES.md at the repo root.
    """
    lines = []
    for key in ("COPYRIGHT", "NOTICE"):
        m = re.search(rf'^{key}\s+"(.*)"\s*$', text, re.MULTILINE)
        if m:
            lines.append(m.group(1))
    return lines


def parse_bdf(text: str, expected_height: int):
    """Return a dict of {codepoint: [row0, row1, ...]} for cp < 256."""
    glyphs: dict[int, list[int]] = {}

    # STARTCHAR ... ENDCHAR sections. Non-greedy so we don't span
    # across chars. DOTALL because BDF is multi-line.
    char_re = re.compile(r'STARTCHAR\s.*?ENDCHAR', re.DOTALL)
    enc_re = re.compile(r'ENCODING\s+(-?\d+)')
    bitmap_re = re.compile(r'^BITMAP\s*$(.*?)^ENDCHAR', re.DOTALL | re.MULTILINE)

    for m in char_re.finditer(text):
        chunk = m.group(0)

        enc_m = enc_re.search(chunk)
        if not enc_m:
            continue
        cp = int(enc_m.group(1))
        if cp < 0 or cp >= 256:
            continue

        bm_m = bitmap_re.search(chunk)
        if not bm_m:
            continue

        rows = [
            int(line.strip(), 16)
            for line in bm_m.group(1).strip().splitlines()
            if line.strip()
        ]
        if len(rows) != expected_height:
            # Pad short glyphs with empty rows; truncate anything longer.
            if len(rows) < expected_height:
                rows = rows + [0] * (expected_height - len(rows))
            else:
                rows = rows[:expected_height]

        glyphs[cp] = rows

    return glyphs


def emit_rust(
    glyphs: dict[int, list[int]],
    out_path: Path,
    bdf_path: Path,
    name: str,
    height: int,
    attribution: list[str],
) -> None:
    with out_path.open("w") as f:
        f.write(f"// Generated from {bdf_path.name} by kmain/tools/bdf_to_rust.py\n")
        f.write("// Do not hand-edit — regenerate if you swap fonts.\n")
        if attribution:
            f.write("//\n")
            for line in attribution:
                f.write(f"// {line}\n")
            f.write("// Full license text in THIRD_PARTY_NOTICES.md at the repo root.\n")
        f.write("//\n")
        f.write("// Each glyph is MSB-first: row byte bit 7 = leftmost pixel.\n")
        f.write("// Missing codepoints in 0..256 render as blank.\n\n")
        f.write(f"pub static {name}: [[u8; {height}]; 256] = [\n")

        for cp in range(256):
            rows = glyphs.get(cp, [0] * height)
            row_str = ", ".join(f"0x{b:02X}" for b in rows)
            # Annotate with a readable glyph for printable ASCII so
            # diffs are skimmable. Escape the quote-sensitive bits.
            if 0x20 <= cp <= 0x7E:
                ch = chr(cp)
                if ch == "'":
                    annot = "'\\''"
                elif ch == "\\":
                    annot = "'\\\\'"
                else:
                    annot = f"'{ch}'"
                f.write(f"    [{row_str}],  // U+{cp:04X} {annot}\n")
            else:
                f.write(f"    [{row_str}],  // U+{cp:04X}\n")

        f.write("];\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="source BDF file")
    ap.add_argument("output", type=Path, help="destination Rust source file")
    ap.add_argument("name", help="static array identifier (e.g. TERMINUS_8X16)")
    ap.add_argument("--height", type=int, default=16, help="glyph height in rows")
    args = ap.parse_args()

    text = args.input.read_text()
    glyphs = parse_bdf(text, args.height)
    attribution = parse_attribution(text)
    print(f"extracted {len(glyphs)} glyphs in U+0000..U+00FF", file=sys.stderr)
    if not attribution:
        print("warning: no COPYRIGHT/NOTICE properties in BDF", file=sys.stderr)
    emit_rust(glyphs, args.output, args.input, args.name, args.height, attribution)


if __name__ == "__main__":
    main()
