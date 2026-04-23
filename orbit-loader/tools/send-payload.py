#!/usr/bin/env python3
"""Send an ELF to a running orbit-loader instance.

Wire protocol (matches orbit-loader/src/main.rs):
    [u32 LE len]  [u32 LE !len]  [cbor body: len bytes]

where the CBOR body is a two-entry map:
    { 0: <elf bytes>, 1: <name utf-8> }

Usage:
    send-payload.py PATH [--name NAME] [--host HOST] [--port PORT]

Defaults: --host 127.0.0.1, --port 7777, --name = basename(PATH).
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


def encode_payload(elf: bytes, name: str) -> bytes:
    """Encode the CBOR map minicbor's derive macro expects:
    keys are the `#[n(N)]` indices, so 0=elf (bytes), 1=name (text)."""
    name_bytes = name.encode("utf-8")
    out = bytearray()
    out.append(0xA2)                                            # map of 2
    out += cbor_uint_header(0, 0)                               # key: 0
    out += cbor_uint_header(2, len(elf)) + elf                  # value: byte string
    out += cbor_uint_header(0, 1)                               # key: 1
    out += cbor_uint_header(3, len(name_bytes)) + name_bytes    # value: text string
    return bytes(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Send an ELF to orbit-loader.")
    ap.add_argument("path", help="path to ELF to send")
    ap.add_argument("--name", default=None, help="override logical name (default: basename)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7777)
    args = ap.parse_args()

    with open(args.path, "rb") as f:
        elf = f.read()
    name = args.name or os.path.basename(args.path)

    body = encode_payload(elf, name)
    length = len(body)
    header = struct.pack("<II", length, (~length) & 0xFFFFFFFF)

    print(f"send-payload: elf={len(elf)}B name={name!r} body={length}B → {args.host}:{args.port}",
          file=sys.stderr)

    with socket.create_connection((args.host, args.port)) as s:
        s.sendall(header)
        s.sendall(body)
        # Half-close so the guest sees FIN and leaves ESTABLISHED.
        try:
            s.shutdown(socket.SHUT_WR)
        except OSError:
            pass

    print("send-payload: done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
