# Orbit

A hobby RISC-V 64 kernel written in Rust, running on the QEMU `virt` machine
with 4 harts and 2 GiB of RAM. Three privilege levels, three ELFs: an M-mode
bootloader ([bl/](bl/)), a fully relocatable PIE S-mode kernel
([kmain/](kmain/)), and U-mode userspace loaded over TCP at runtime
([orbit-loader/](orbit-loader/)). It's largely written to experiment with
kernel development, LLM-assisted development, rustc internals, permissions,
project organization, and anything else that I'm feeling interested in.

Highlights:

- **SMP** with a greedy, unpinned scheduler — every hart contends for the
  scheduler lock and whoever wins runs the manager pass; cross-hart wakeups
  via ACLINT SSWI, timers via Sstc.
- **Sv48 paging** with a high-half kernel (KTEXT / KDMAP / KMMIO), built by an
  early trampoline in the kernel's own `_start`; the kernel is a
  self-relocating PIE.
- **TCP/IP** via [smoltcp](https://github.com/mayamaclean/smoltcp) on an
  e1000 NIC, exposed to userspace through shared-memory SPSC NetChannels —
  the boot process is a TCP listener you send ELF payloads to.
- **Typestate capability layer** gating all thread mutation
  ([process/src/cap.rs](process/src/cap.rs)); syscall handlers never touch
  manager-owned state from the trap path (no trap-path locks — blocking
  syscalls are manager round-trips).
- **virtio** gpu (damage-tracked compositor + per-process surfaces), blk
  (tarfs root with a page cache), and keyboard (structured key events);
  Goldfish RTC for wall-clock time.
- **POSIX-shaped surface**: uid/gid/groups/login credentials with `vaccess`
  enforcement, cwd, fds, eventfds, waitpid, argv/envp — plus a **`std` port**
  (custom `riscv64gc-unknown-orbit` rustc target) that runs
  [mio](https://github.com/mayamaclean/mio)- and ratatui-based demos.

## Requirements

- **rustup** — the pinned nightly toolchain and components
  ([rust-toolchain.toml](rust-toolchain.toml)) install automatically on first
  build. All crates build with `-Zbuild-std`.
- **qemu-system-riscv64** — needs `-machine virt,aclint=on` and
  `-cpu max,sstc=true` (QEMU ≥ 7.1). The default runner opens an SDL window;
  for headless machines change `-display sdl,gl=off` to `-display none` in
  [bl/.cargo/config.toml](bl/.cargo/config.toml).
- **python3** — payload delivery and the disk-image build script.

There is no top-level workspace; `cargo` runs from inside individual crate
directories, and each of `bl`, `kmain`, `umode`, `orbit-loader` carries its
own `.cargo/config.toml`.

## Build and run

Each layer embeds the one above it (`bl` embeds `kmain`, `kmain` embeds
`orbit-loader`), so order matters. The easy path:

```sh
./build-all.sh     # loader → kernel → bootloader → disk.img, in order
cd bl && cargo run # builds `launch` and boots QEMU via the cargo runner
```

Or manually:

```sh
(cd orbit-loader && cargo build --release)  # 1. TCP-listening init process
(cd kmain && ./build.sh)                    # 2. S-mode kernel (PIE link flags — don't bare `cargo build`)
tools/build-disk.sh                         # 3. tarfs disk.img (staged from rootfs/)
(cd bl && cargo run)                        # 4. M-mode bootloader + QEMU
```

Once the guest is up, the loader listens on TCP :7777 (forwarded to host
:7777). Send it a user program:

```sh
(cd umode && cargo build)
python3 orbit-loader/tools/send_payload.py umode/target/riscv64gc-unknown-none-elf/debug/umode
```

A `console` shell binary is staged into the disk image as the default init.
QEMU's serial console is on stdio (`-serial mon:stdio`); the virtio-gpu
window shows the framebuffer console and any per-process surfaces.

To debug: the runner opens `-gdb tcp::1234`; `./debug kmain orbit` (or
`./debug bl launch`) attaches rust-lldb against the release ELF.

## Tests

```sh
./test
```

Host-side unit tests across the crates (notably `orbit-core`, `process`,
`manager`, `orbit-abi`, `net_channel`, `mem`), plus a miri hammer pass over
the concurrency-sensitive suites.

## std on orbit (optional)

Building std-based user programs (`hello-std`, `hello-ratatui-std`,
`orbit-top-std`) requires the rustc fork with the `riscv64gc-unknown-orbit`
target, cloned to `rust/` inside this repo:

```sh
git clone --filter=blob:none -b orbit-std https://github.com/mayamaclean/rust rust
cd rust && ./x build library --stage 1
rustup toolchain link orbit-stage1 "$(pwd)/build/x86_64-unknown-linux-gnu/stage1"
```

(Don't `--depth 1` the rust clone — bootstrap walks git history to locate the
CI LLVM artifact and fails without it; `--filter=blob:none` keeps the commit
graph at a fraction of the size. Don't pass `--target riscv64gc-unknown-orbit`
either: `bootstrap.toml` already targets both orbit and the host, and the
host std is required for build scripts when cargo runs under `+orbit-stage1`.
The orbit target's `cc` is `riscv64-unknown-elf-gcc`, so a RISC-V bare-metal
GNU toolchain must be on `PATH`.)

Prebuilt copies of the std demos are checked in under [rootfs/bin/](rootfs/bin/),
so the disk image includes them even without the fork. See
[CLAUDE.md](CLAUDE.md) ("Building Orbit's std") for the full loop, including
the stale-rlib gotcha and `./rebuild-std.sh`.

## More docs

- [docs/architecture.md](docs/architecture.md) — system architecture
- [docs/boot.md](docs/boot.md) — the M→S→U boot path
- [docs/kernel-threads.md](docs/kernel-threads.md) — k_net / k_gpu / k_serial
- [CLAUDE.md](CLAUDE.md) — dense per-subsystem map (written for AI agents,
  useful for humans)

## License

MIT — see [LICENSE](LICENSE). Vendored third-party code and fonts are
attributed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
