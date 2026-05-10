#!/usr/bin/env python3
"""Send an ELF to a running orbit-loader instance.

Wire protocol (matches orbit-loader/src/main.rs):
    [u32 LE len]  [u32 LE !len]  [cbor body: len bytes]

where the CBOR body is a map (kept open-ended via #[cbor(map)] on the
loader side so missing optional keys are accepted):
    { 0: <elf bytes>, 1: <name utf-8>,
      2: <allowed_affinity u64>?, 3: <affinity u64>?,
      4: <argv: [utf-8 string, ...]>? }

Usage:
    send-payload.py PATH [--name NAME] [--host HOST] [--port PORT]
                    [--allowed-affinity MASK] [--affinity MASK]
                    [--arg ARG]...

Pass argv as `--arg foo --arg bar ...` (repeatable). The loader
packs them with `orbit_abi::argv::pack` and hands the blob to
`create_process_with_argv_envp`. By convention the first --arg is
argv[0] (the program name); the loader's --name (which defaults to
basename(PATH)) is *not* auto-prepended — pass it explicitly when
the consumer expects argv[0] (most do).

Defaults: --host 127.0.0.1, --port 7777, --name = basename(PATH).
Affinity masks default to 0 ("all harts" sentinel — kernel substitutes
the real cpu_count mask). Pass e.g. --affinity 0x4 to pin a process to
hart 2 (used by §10's TLB-shootdown stress test).
QEMU is expected to forward host:7777 to guest:7777 via -netdev
user,...,hostfwd=tcp::7777-:7777 (already wired in bl/.cargo/config.toml).
"""

import argparse
import os
import socket
import struct
import sys


def cbor_uint_header(major: int, value: int) -> bytes:
    """Emit the initial byte(s) for a CBOR item of `major` type and
    length/value `value`. Only covers the short + uint32 additional-info
    paths, which is all we need for names and ELFs up to ~4 GiB."""
    assert 0 <= major <= 7
    hi = major << 5
    if value < 24:
        return bytes([hi | value])
    if value < 0x100:
        return bytes([hi | 24, value])
    if value < 0x10000:
        return bytes([hi | 25]) + struct.pack(">H", value)
    if value < 0x1_0000_0000:
        return bytes([hi | 26]) + struct.pack(">I", value)
    return bytes([hi | 27]) + struct.pack(">Q", value)


def encode_payload(elf: bytes, name: str,
                   allowed_affinity: int, affinity: int,
                   argv: list[str]) -> bytes:
    """Encode the CBOR map minicbor's derive macro expects:
    keys are the `#[n(N)]` indices, so 0=elf (bytes), 1=name (text),
    2=allowed_affinity (u64), 3=affinity (u64), 4=argv (array of text
    strings). Optional entries (2, 3, 4) are omitted when at their
    default — the loader's #[cbor(default)] leaves them at the
    all-harts-default sentinel / empty argv, identical to the pre-
    feature wire shape so older senders still work."""
    name_bytes = name.encode("utf-8")
    include_affinity = allowed_affinity != 0 or affinity != 0
    include_argv = bool(argv)
    n_entries = 2 + (2 if include_affinity else 0) + (1 if include_argv else 0)
    out = bytearray()
    # CBOR map header: major 5, length n_entries. n_entries ≤ 5 fits
    # in the short additional-info range so a single byte suffices.
    out.append(0xA0 | n_entries)
    out += cbor_uint_header(0, 0)                               # key: 0
    out += cbor_uint_header(2, len(elf)) + elf                  # value: byte string
    out += cbor_uint_header(0, 1)                               # key: 1
    out += cbor_uint_header(3, len(name_bytes)) + name_bytes    # value: text string
    if include_affinity:
        out += cbor_uint_header(0, 2)                           # key: 2
        out += cbor_uint_header(0, allowed_affinity)            # value: uint
        out += cbor_uint_header(0, 3)                           # key: 3
        out += cbor_uint_header(0, affinity)                    # value: uint
    if include_argv:
        out += cbor_uint_header(0, 4)                           # key: 4
        # Major 4 = array. Each element is a text string (major 3).
        out += cbor_uint_header(4, len(argv))
        for a in argv:
            ab = a.encode("utf-8")
            out += cbor_uint_header(3, len(ab)) + ab
    return bytes(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Send an ELF to orbit-loader.")
    ap.add_argument("path", help="path to ELF to send")
    ap.add_argument("--name", default=None, help="override logical name (default: basename)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7777)
    ap.add_argument(
        "--allowed-affinity", default="0",
        help="immutable hart-permission cap as u64 mask (e.g. 0x4 = hart 2 only). "
             "0 = let the kernel default to all harts.",
    )
    ap.add_argument(
        "--affinity", default="0",
        help="initial hart-eligibility mask as u64. Must be a subset of "
             "--allowed-affinity once both resolve. 0 = inherit allowed mask.",
    )
    ap.add_argument(
        "--arg", action="append", default=[], metavar="ARG",
        help="add one argv entry. Repeatable; e.g. `--arg rg --arg hello "
             "--arg /bin/hello.txt`. By convention the first --arg is "
             "argv[0] (the program name).",
    )
    args = ap.parse_args()

    allowed_aff = int(args.allowed_affinity, 0)
    aff = int(args.affinity, 0)
    argv = list(args.arg)

    with open(args.path, "rb") as f:
        elf = f.read()
    name = args.name or os.path.basename(args.path)

    body = encode_payload(elf, name, allowed_aff, aff, argv)
    length = len(body)
    header = struct.pack("<II", length, (~length) & 0xFFFFFFFF)

    print(f"send-payload: elf={len(elf)}B name={name!r} body={length}B "
          f"allowed_aff={allowed_aff:#x} aff={aff:#x} argv={argv!r} "
          f"→ {args.host}:{args.port}",
          file=sys.stderr)

    with socket.create_connection((args.host, args.port)) as s:
        s.sendmsg([header, body])
        s.shutdown(socket.SHUT_WR)
        resp = s.recv(1, socket.MSG_WAITALL)
        print(f"resp={resp}")
        #s.shutdown(socket.SHUT_RD)

    print("send-payload: done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
