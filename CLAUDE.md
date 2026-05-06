# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Orbit is a RISC-V 64 (`rv64gc`) kernel written in Rust `no_std`, running on `qemu-system-riscv64` with the `virt` machine, 4 harts, 2 GiB RAM, an e1000 NIC, and M/S privilege modes. Target triple is `riscv64gc-unknown-none-elf`; toolchain is pinned in [rust-toolchain.toml](rust-toolchain.toml) to the nightly in the file (nightly-2026-02-21 at time of writing). `-Zbuild-std=core,alloc` is required — do not run `cargo` without it.

There is no top-level `Cargo.toml`. An old/invalid root manifest exists as [asdasdascCargo.toml.nottoml](asdasdascCargo.toml.nottoml) and is intentionally disabled. `cargo` must be invoked from inside one of the individual crate directories. Each of `bl`, `kmain`, `umode`, `orbit-loader` has its own `.cargo/config.toml` carrying the `runner`, `rustflags`, and `build-std` settings.

## Build and run

Because each upper layer embeds the lower layer's ELF via `include_bytes!`, builds must happen in a specific order:

1. `cd orbit-loader && cargo build --release` — compiles the TCP-listening loader that kmain boots into. `kmain` embeds `orbit-loader/target/riscv64gc-unknown-none-elf/release/orbit-loader` (see [kmain/src/kernel/mod.rs:60](kmain/src/kernel/mod.rs#L60)). Test payloads like `umode` are *not* compiled into the kernel — they're built separately and sent over the wire via [orbit-loader/tools/send-payload.py](orbit-loader/tools/send-payload.py) once the guest is up.
2. `cd kmain && ./build.sh` — builds the S-mode kernel as PIE with the required link flags (`-pie`, `-Bsymbolic`, `-znotext`, `--no-dynamic-linker`, `--pack-dyn-relocs=none`, `--export-dynamic`). Running `cargo build` directly in `kmain` will NOT produce a usable kernel — use the script, which sets the `RUSTFLAGS` the kernel's self-relocation stub depends on. `bl` embeds `kmain/target/riscv64gc-unknown-none-elf/release/orbit` (see [bl/src/lib.rs:17](bl/src/lib.rs#L17)).
3. `cd bl && cargo run` — builds the M-mode bootloader (binary name `launch`) and launches QEMU via the runner in [bl/.cargo/config.toml](bl/.cargo/config.toml). The runner also opens `-gdb tcp::1234` and passes `-S` (freeze on entry) so you can attach a debugger before `_start` runs. Guest port 7777 is forwarded to host 7777 (`hostfwd`) so `send-payload.py` can reach the loader.
4. (Optional, per iteration) `cd umode && cargo build && python3 orbit-loader/tools/send-payload.py umode/target/riscv64gc-unknown-none-elf/debug/umode` — rebuild a test payload and ship it to the running guest. Avoids re-linking kmain+bl.

To debug a running QEMU: `./debug <crate> <exec>` — e.g. `./debug bl launch` or `./debug kmain orbit`. This launches `rust-lldb` and issues `gdb-remote localhost:1234` against the chosen release ELF.

There are no tests or lints configured for this project.

### Building Orbit's std (forked rustc at [rust/](rust/))

Std-on-orbit binaries (currently only [hello-std/](hello-std/), eventually any user app that wants `std`) require a custom rustc fork that knows about the `riscv64gc-unknown-orbit` target. The fork lives at [rust/](rust/); its bootstrap config in [rust/bootstrap.toml](rust/bootstrap.toml) targets both the host and orbit. Build flow:

1. `cd rust && ./x build library --stage 1 --target riscv64gc-unknown-orbit` — compiles `core`/`alloc`/`std` for orbit. Stage 1 is enough; stage 2 adds nothing relevant. Output lives under [rust/build/x86_64-unknown-linux-gnu/stage1/lib/rustlib/riscv64gc-unknown-orbit/lib/](rust/build/x86_64-unknown-linux-gnu/stage1/lib/rustlib/riscv64gc-unknown-orbit/lib/).
2. The toolchain is rustup-linked as `orbit-stage1` (run once: `rustup toolchain link orbit-stage1 /home/maya/projects/orbit/rust/build/x86_64-unknown-linux-gnu/stage1`).
3. `cd hello-std && cargo +orbit-stage1 build` — builds against the freshly-built std. The `+orbit-stage1` selects the linked toolchain; without it cargo defaults to the project nightly which has no `riscv64gc-unknown-orbit` spec. `[build] target = "riscv64gc-unknown-orbit"` in `hello-std/.cargo/config.toml` pins the triple.

Whenever a `library/std/src/sys/...` file changes (PAL surface — fs, net, thread, alloc, futex, …) you must rerun step 1 before rebuilding the consumer. Skipping this is silent: cargo reuses the stale rlib in the toolchain sysroot and the new code never lands.

The `riscv64gc-unknown-orbit` target spec is built into the fork ([rust/compiler/rustc_target/src/spec/base/orbit.rs](rust/compiler/rustc_target/src/spec/base/orbit.rs)). It pins `linker = "rust-lld"`, static reloc model, no PIE, no eh_frame_header. orbit-abi's `rustc-dep-of-std` feature wires `core` to `rustc-std-workspace-core` so the kernel's pure-Rust types can be compiled into std.

For an as-built map of the orbit-side std PAL — what's wired, what's stubbed, file paths, build-loop gotchas, and the priority order of remaining holes — see [docs/std-on-orbit.md](docs/std-on-orbit.md).

External dependency: `smoltcp` is a path dependency at `../smoltcp` (sibling of this repo), not a crates.io version. If it is missing the workspace will fail to build `kmain`, `process`, and `net_channel`.

## Architecture

Three privilege levels run three separate ELF artifacts, each with its own linker script and `.cargo/config.toml`:

- **M-mode bootloader — [bl/](bl/)** (binary `launch`, linked at `0x80000000` per [bl/memory.x](bl/memory.x)). Hart 0 initializes the UART from the DTB, sets up an identity-mapped Sv48 page table at `bl::ID_MAP_TABLES` (`0x80800000 - 2 MiB`), parses the embedded kernel ELF (`KERNEL_ELF`), copies its `PT_LOAD` segments into physical RAM at `0x80000000 + 64 MiB` (see `VBASE` in [bl/src/lib.rs](bl/src/lib.rs)), and `sret`s into S-mode. Other harts spin in `kinit_hart` waiting for `HART_ROOT` (set via an M-mode `ecall` from the kernel) before jumping into their assigned kernel entry point. The kernel also publishes a **KDMAP bias** via `ecall(4)` so `kinit_hart` can hand secondary harts a `sscratch` / `sp` that resolves under the kernel satp. Trap frames for all harts live at `0x80800000` (one per hart) so the M-mode trap handler can route interrupts while paging is still off. `m_trap_vector` switches `sp` to bl's own `KERNEL_STACK_END` before calling the Rust handler, so spills don't end up on the kernel's KDMAP stack (which M-mode would dereference bare).

- **S-mode kernel — [kmain/](kmain/)** (binary `orbit`). Built as a fully relocatable PIE. Linked at low VA `0x1000`; an early trampoline in the naked `_start` in [kmain/src/bin/orbit.rs](kmain/src/bin/orbit.rs) builds a temporary Sv48 table (identity for RAM/MMIO plus `KTEXT_NOMINAL → load_addr` and a `KDMAP_NOMINAL → RAM` direct-map), `csrw satp`, and jumps into the high-half VA. Post-jump `post_trampoline_entry` applies `R_RISCV_RELATIVE` with `slide = ktext_base - LINK_BASE` before any Rust code touches a relocated global, then tail-calls `rust_main`. `rust_main` initializes the linked-list heap (`KHEAP` at its KDMAP VA), frame allocators for page tables (ktables) and general-purpose kernel pages (kpages) — both returning KDMAP VAs via `add_frame_with_va_base` — installs `OrbitLogger`/`OrbitSubscriber`, builds the final Sv48 table via `map_kernel_self` (KTEXT / KDMAP / KMMIO, no identity left), allocates per-hart `HartContext` structs, and `sret`s into `k_smpstart`. `k_smpstart` re-`init_serial`s at `kmmio_uart()` before any print, signals bl (`ecall(4)` then `ecall(1)`), and kicks harts 1..N via KMMIO CLINT MSIPs. Hart 0 then runs `k_manage`; others `k_idle`/WFI. See [kmain/src/kernel/memmap.rs](kmain/src/kernel/memmap.rs) for layout atomics, constants, `KernelLayout`, and `RootTable` helpers.

- **U-mode boot process — [orbit-loader/](orbit-loader/)** linked at `0x2_2000_0000` (`USER_TEXT_BASE`). The single process kmain spawns from its embedded ELF in `k_smpstart`. Listens on TCP :7777 via a NetChannel in listen mode; each incoming connection delivers a framed CBOR payload (`[u32 LE len][u32 LE !len][cbor {0: elf, 1: name}]`) that the loader hands to `create_process` (syscall 4099). This replaces a build-time `include_bytes!` of a hardcoded sample; rebuilding a test binary no longer requires rebuilding kmain or bl.

- **U-mode sample — [umode/](umode/)** linked at `0x2_2000_0000` (same `USER_TEXT_BASE` — only one user ELF occupies that VA at a time today). A hardcoded demo ([umode/src/main.rs](umode/src/main.rs)) that prints to serial, sleeps, `mmap`s a shared region, registers a NetChannel, and opens a TCP connection. No longer embedded in kmain — now shipped to a running guest via `orbit-loader/tools/send-payload.py`.

### U-mode VA layout

Sv48 low half is 128 TiB. The user-mappable space is split between kernel-managed regions (stacks, ELF — installed at process creation, never via `mmap`) and user-controllable regions (priv heap, shared mmap/NetChannels). Constants in [orbit-abi/src/layout.rs](orbit-abi/src/layout.rs):

- `0..0x20_0000` — null guard (2 MiB, megapage-aligned).
- `UPROC_STACK_BASE = 0x1000_0000` — 256 per-thread slots, 32 MiB stride, 8 GiB total. Each slot holds `[stack at slot bottom (≤ 28 MiB)] [unmapped gap] [TLS reservation 2 MiB] [guard 2 MiB at slot top]`. Stack grows down; overflow falls into the previous slot's guard (or the unmapped span below `UPROC_STACK_BASE` for slot 0). Kernel-mapped at thread creation. See [docs/user-thread-region.md](docs/user-thread-region.md).
- `USER_TEXT_BASE = 0x2_2000_0000` — ELF image. Kernel-mapped at process creation.
- `UPROC_PRIV_BASE = 0x3_0000_0000 .. UPROC_PRIV_END = 0x4000_0000_0000` — user-controlled private range (~64 TiB). `mmap(share_with_kernel=false)` and orbit-rt's `#[global_allocator]` claim from here.
- `UPROC_SHARED_BASE = 0x4000_0000_0000 .. UPROC_SHARED_END = 0x7E00_0000_0000` — user-controlled shared range (62 TiB). `mmap(share_with_kernel=true)` and `create_netch` (NetChannels) claim from here. orbit-rt's `SHARED_HEAP` is a separate talc cell with its own VA cursor here.
- `USER_TRAP_FRAME_BASE = 0x7E00_0000_0000` — kernel-private per-thread TrapFrames (S-only, no U bit).

The syscall layer (`orbit-core::syscall`) enforces the priv/shared split at the boundary: `mmap_req` rejects a private mmap aimed at a shared VA (and vice versa), `nc_create_req` rejects any VA outside the shared range, and the buffer-pointer syscalls (`serial_print`, `console_write`, `read_stdin`, `create_process`'s elf_ptr) accept anywhere user-mappable but reject kernel half / null guard / overflow via `user_range_ok`.

### Syscall ABI

`ecall` with syscall number in `a0`, args in `a1..a4`. Dispatched by the `cause == 8` arm of `s_trap` in [kmain/src/bin/orbit.rs](kmain/src/bin/orbit.rs). Canonical list in [orbit-abi/src/syscall.rs](orbit-abi/src/syscall.rs):

- `0` — exit (noreturn)
- `1` — serial_print(ptr, len)
- `2` — sleep_ms(ms)
- `3` — console_write(ptr, len)
- `4` — read_stdin(ptr, len, flags)
- `4096` — mmap(vaddr, len, perms, share_with_kernel) — vaddr must be in `UPROC_PRIV_BASE..UPROC_PRIV_END` when `share_with_kernel=false`, in `UPROC_SHARED_BASE..UPROC_SHARED_END` when `=true`
- `4097` — create_netch(vaddr_hint, region_size, sock_type) → (user_va, fd) — vaddr_hint must be in `UPROC_SHARED_BASE..UPROC_SHARED_END`
- `4098` — close_handle(fd)
- `4099` — create_process(elf_ptr, elf_len) → pid
- `4105` — create_process_v2(args: *const CreateProcessV2Args) → pid — role-aware spawn with explicit perms narrowing, optional cwd/argv/envp install, optional stdout capture, optional identity stamping, and a two-front-door spawn source. **Spawn source**: `spawn_path_vaddr/spawn_path_len != 0` selects path mode — kernel does `fs.open + vaccess(X)` against the caller's effective creds, reads the ELF off disk via `k_io` (the generic "kernel reads/waits and reports back to manager" kthread, see [kmain/src/lib.rs](kmain/src/lib.rs) `k_io`), then bounces `PendingWork::SpawnReady` back to the manager which finishes the install via the canonical `install_spawn` helper. `spawn_path_vaddr == 0` selects bytes mode — kernel reads `elf_vaddr[..elf_len]` from caller user memory; restricted to `role::LOADER` because there's no file to check the X bit against. Non-LOADER bytes-mode → `-EPERM`. orbit-loader (LOADER) uses bytes mode for network-payload spawns; console/std PAL/etc. use path mode. **Identity stamping**: `setuid_uid` / `setuid_gid` (`-1` = inherit, `0..=u32::MAX` = stamp on all three triplet slots), `setlogin_vaddr/len` (`0` = inherit), `groups_vaddr/count` (`0` = inherit; non-zero installs the listed supp groups, capped at `process::NGROUPS_MAX`). Stamping any non-inherit value requires the *parent* to be running with `role::LOADER` — other callers get `-EPERM`. orbit-loader stamps `uid=1000`/`gid=1000` on every payload it spawns; that's the only non-root identity path until a real login pipeline lands.
- `4107` — chdir(path_ptr, path_len) → 0 — replace caller's cwd with the absolute UTF-8 path; kernel validates the dir exists in the active fs before mutating
- `4108` — getcwd(buf_ptr, buf_len) → bytes_written — copy the caller's cwd (no NUL) into the user buffer; ERANGE if too small. Children inherit the parent's cwd at spawn (`CreateProcessV2Args` carries an optional `cwd_vaddr`/`cwd_len` override; legacy `CREATE_PROCESS` / `CREATE_PROCESS_EX` always inherit). Relative-path fs syscalls (`fs_open`, `fs_stat`) are prefixed by `Process.cwd` kernel-side.
- `4109` — getuid() → uid — POSIX `getuid(2)`. Real uid of the calling process. Reads `Thread.uid` (snapshotted from `Process.uid` at thread creation) without locking. Identity-only — roles + permissions own authorization; uid/gid feed `vaccess`-style fs checks once writable-fs ownership lands. Currently every process runs as uid 0; spawn-time stamping ships in the next milestone.
- `4110` — geteuid() → euid — POSIX `geteuid(2)`. Effective uid. Split from `getuid` (rather than bundled into `(uid << 32) | euid`) because uids ≥ 0x8000_0000 would shift into the high bit of the `isize` return and be misread as a negative errno.
- `4111` — getgid() → gid — POSIX `getgid(2)`. Real gid; mirror of `getuid`.
- `4112` — getegid() → egid — POSIX `getegid(2)`. Effective gid; mirror of `geteuid`.
- `4113` — getgroups(buf_ptr, count) → count | -ERANGE — POSIX `getgroups(2)`. Copy the caller's supplementary group list into the user buffer (one `u32` per slot). `count == 0` returns the current group count without writing (POSIX sizing call). Capped at `process::NGROUPS_MAX = 16`.
- `4114` — getlogin(buf_ptr, buf_len) → bytes | -errno — POSIX `getlogin_r(3)`. Copy the calling process's session login name (no NUL terminator) into the user buffer. `ENOENT` if no login name has been installed (initial process state).
- `4115` — setuid(uid) → 0 | -EPERM — POSIX `setuid(2)`. `euid == 0` stamps `uid` on all three triplet slots (privilege drop). `euid != 0` sets only euid, IFF `uid ∈ {ruid, suid}` (POSIX privilege-toggle). Sync handler: refreshes per-thread credential snapshots so sibling threads' `getuid` reads observe the new identity. Gated on `class::PROC_CRED` (separate from PROC_LIFE so a daemon can pledge it away after privsep startup).
- `4116` — setgid(gid) → 0 | -EPERM — POSIX `setgid(2)`. Gid mirror of `setuid` (POSIX still keys the privilege slot off `euid == 0`, not gid).
- `4117` — setgroups(buf_ptr, count) → 0 | -errno — POSIX `setgroups(2)`. Replace the caller's supplementary group list. Requires `euid == 0`; capped at `process::NGROUPS_MAX = 16`. `count == 0` is legal (empties the list). Gated on `class::PROC_CRED`.
- `4118` — setlogin(name_ptr, name_len) → 0 | -errno — POSIX `setlogin(2)`. Stamp the calling process's session login name (UTF-8, ≤ 32 bytes, requires `euid == 0`). Gated on `class::PROC_CRED`.
- VFS access enforcement: `fs_open` runs `vaccess()` against the calling process's `(euid, egid, supplementary groups)` and the inode's `(st_uid, st_gid, st_mode)` after path resolution. Owner→group→other check, first match wins; root (`euid == 0`) bypasses. Returns `EACCES` on the deny path. Pure rule logic + unit tests live in [orbit-abi/src/fs.rs](orbit-abi/src/fs.rs) (`vaccess` fn); the kmain adapter is `Orbit::vaccess_pid` in [kernel/mod.rs](kmain/src/kernel/mod.rs). Path-walk traversal (X on each parent dir) isn't modeled — tarfs's `open(path)` is a single-pass internal walk; only the final inode is checked. `fs_stat` doesn't run vaccess (POSIX); `fs_read` doesn't re-check (perm checked at open time). Test fixture: `/etc/secret` is staged as `0o600 root:root` in tarfs via [tools/build-disk.sh](tools/build-disk.sh)'s second-pass `--owner=0 --group=0` for the `etc/` subtree.
- `6004` — fs_seek(fd, offset, whence) → new_offset — POSIX-shaped: `SEEK_SET = 0`, `SEEK_CUR = 1`, `SEEK_END = 2`. Mutates the per-fd `OpenFile.offset` only; rejects directory fds (`EBADF`) and resolved-negative offsets (`EINVAL`).
- `6005` — fs_fstat(fd, &mut Stat) → 0 — fill `*stat` with metadata for the file backing `fd`. Mirror of `fs_stat` keyed on an open fd.

### Supporting crates (all under the same workspace)

- [mmu/](mmu/) — Sv48 page tables, `PagePermissions`, `MappingConfig`, `id_map_range`, `map_address_range`, `map_va_range`, `virt_to_phys`. Walkers take `&RootTable<'_>`, which pairs a `&PageTable` with a PA→VA bias so they can follow PPNs (always physical) back into the supervisor's KDMAP view. bl/early-trampoline use `RootTable::identity(...)` (bias 0); kmain uses `memmap::kernel_root(...)` / `kernel_root_from_pa(...)` (bias `= kdmap_base - ram_phys_base`). `alloc` feature wires it up against `alloc::Vec`-backed page pools.
- [mem/](mem/) — `FrameAllocator`, `round_u64_up`/`down` helpers, `prev_power_of_two`.
- [device/](device/) — `HartContext` (cache-line-aligned, field offsets are load-bearing — consumed by `asm/trap.S` and `boot.S`), `TrapFrame`, `Stack` (2 MiB), `SysInfo`, DTB walkers (`find_serial_port`, `find_ram`). The trap assembly hard-codes offsets into `HartContext`; changing field order requires updating the `.S` files.
- [process/](process/) — `Thread`, `Process`, `ThreadState`, and the `CompletionHandle` / `AckCounter` primitives that suspend a thread on a kernel-allocated slot. A blocking syscall stashes a `CompletionHandle` in `Thread::handle`, parks the thread `Blocking`, and pushes a `PendingWork` variant onto kmain's `MANAGER_WORK` thingbuf; whichever hart next holds `MANAGER_LOCK` runs `Orbit::drain_pending_work` and `signal_n`s the handle, which the scheduler consumes on resume to write up to 4 a-regs into the parked frame. New blocking syscalls add a variant to [orbit-core/src/pending_work.rs](orbit-core/src/pending_work.rs) and a handler arm in `drain_pending_work` in kmain — no per-reason field on `Thread`.
- [net_channel/](net_channel/) — shared-memory SPSC queues between user programs and the kernel networking thread. `kernel` feature enables the kernel-side `update_tcp` methods that drive smoltcp sockets. Layout of `NetChannel` / `NetChannelState` / `NetChannelQueue` is part of the user/kernel ABI — do not change field order casually.
- [serial/](serial/) — `MpUart` wrapping `ns16550a::Uart` with a spinlock; exposes `println!` macro and a global init.
- [kmain/src/drivers/e1000.rs](kmain/src/drivers/e1000.rs) — PCI-discovered e1000 NIC; ring buffers fed by DMA from kernel pages, integrated with smoltcp via `k_net` (a dedicated kernel thread).
- `k_io` (in [kmain/src/lib.rs](kmain/src/lib.rs)) — generic "kernel reads / waits on something and reports back to manager" kthread. Owns `IO_QUEUE` (`StaticThingBuf<Option<Box<IoWork>>, 16>`); woken via `WAKE_QUEUE.push(WakeEvent::Io)`. Today's only variant is `IoWork::Spawn`: load an ELF off disk for a path-mode `create_process_v2`, then push `PendingWork::SpawnReady` back to the manager. Mirrors the `k_net` pattern (latched `io_thread_tid`, `kthread_park` shape) so the manager and trap path stay fast — long-running / I/O-blocking work moves to dedicated kthreads with private queues.

### SMP and scheduling

`HartContext` is the per-hart state structure. `sscratch` always points to the current hart's `HartContext` (as a KDMAP VA under the kernel satp). Cross-hart wakeups go via ACLINT SSWI at `kmmio_sswi()` (physical `0x02F00000`; one `u32` per hart — `write_sswi`/`supervisor_wake_hart`). Timer interrupts use Sstc (`stimecmp` at CSR `0x14D`). The scheduler (hart 0 only) assigns runnable threads to idle harts and sends an IPI; harts receive S-mode software interrupts in `s_trap` (async cause `1`) and context-switch via `check_context_and_switch`. `s_trap_vector` switches `satp` to the kernel's own on entry — syscall handlers that reach user memory walk the user PT manually and go through the KDMAP alias (`memmap::user_va_to_kdmap`); a future SUM-gate milestone inverts this.
