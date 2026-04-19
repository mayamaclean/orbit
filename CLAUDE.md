# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Orbit is a RISC-V 64 (`rv64gc`) kernel written in Rust `no_std`, running on `qemu-system-riscv64` with the `virt` machine, 4 harts, 2 GiB RAM, an e1000 NIC, and M/S privilege modes. Target triple is `riscv64gc-unknown-none-elf`; toolchain is pinned in [rust-toolchain.toml](rust-toolchain.toml) to the nightly in the file (nightly-2026-02-21 at time of writing). `-Zbuild-std=core,alloc` is required — do not run `cargo` without it.

There is no top-level `Cargo.toml`. An old/invalid root manifest exists as [asdasdascCargo.toml.nottoml](asdasdascCargo.toml.nottoml) and is intentionally disabled. `cargo` must be invoked from inside one of the individual crate directories. Each of `bl`, `kmain`, `umode` has its own `.cargo/config.toml` carrying the `runner`, `rustflags`, and `build-std` settings.

## Build and run

Because each upper layer embeds the lower layer's ELF via `include_bytes!`, builds must happen in a specific order:

1. `cd umode && cargo build` — compiles the sample user program. `kmain` embeds `umode/target/riscv64gc-unknown-none-elf/debug/umode` (see [kmain/src/kernel/mod.rs:56](kmain/src/kernel/mod.rs#L56)).
2. `cd kmain && ./build.sh` — builds the S-mode kernel as PIE with the required link flags (`-pie`, `-Bsymbolic`, `-znotext`, `--no-dynamic-linker`, `--pack-dyn-relocs=none`, `--export-dynamic`). Running `cargo build` directly in `kmain` will NOT produce a usable kernel — use the script, which sets the `RUSTFLAGS` the kernel's self-relocation stub depends on. `bl` embeds `kmain/target/riscv64gc-unknown-none-elf/release/orbit` (see [bl/src/lib.rs:17](bl/src/lib.rs#L17)).
3. `cd bl && cargo run` — builds the M-mode bootloader (binary name `launch`) and launches QEMU via the runner in [bl/.cargo/config.toml](bl/.cargo/config.toml). The runner also opens `-gdb tcp::1234` and passes `-S` (freeze on entry) so you can attach a debugger before `_start` runs.

To debug a running QEMU: `./debug <crate> <exec>` — e.g. `./debug bl launch` or `./debug kmain orbit`. This launches `rust-lldb` and issues `gdb-remote localhost:1234` against the chosen release ELF.

There are no tests or lints configured for this project.

External dependency: `smoltcp` is a path dependency at `../smoltcp` (sibling of this repo), not a crates.io version. If it is missing the workspace will fail to build `kmain`, `process`, and `net_channel`.

## Architecture

Three privilege levels run three separate ELF artifacts, each with its own linker script and `.cargo/config.toml`:

- **M-mode bootloader — [bl/](bl/)** (binary `launch`, linked at `0x80000000` per [bl/memory.x](bl/memory.x)). Hart 0 initializes the UART from the DTB, sets up an identity-mapped Sv48 page table at `bl::ID_MAP_TABLES` (`0x80800000 - 2 MiB`), parses the embedded kernel ELF (`KERNEL_ELF`), copies its `PT_LOAD` segments into physical RAM at `0x80000000 + 64 MiB` (see `VBASE` in [bl/src/lib.rs](bl/src/lib.rs)), and `sret`s into S-mode. Other harts spin in `kinit_hart` waiting for `HART_ROOT` (set via an M-mode `ecall` from the kernel) before jumping into their assigned kernel entry point. The kernel also publishes a **KDMAP bias** via `ecall(4)` so `kinit_hart` can hand secondary harts a `sscratch` / `sp` that resolves under the kernel satp. Trap frames for all harts live at `0x80800000` (one per hart) so the M-mode trap handler can route interrupts while paging is still off. `m_trap_vector` switches `sp` to bl's own `KERNEL_STACK_END` before calling the Rust handler, so spills don't end up on the kernel's KDMAP stack (which M-mode would dereference bare).

- **S-mode kernel — [kmain/](kmain/)** (binary `orbit`). Built as a fully relocatable PIE. Linked at low VA `0x1000`; an early trampoline in the naked `_start` in [kmain/src/bin/orbit.rs](kmain/src/bin/orbit.rs) builds a temporary Sv48 table (identity for RAM/MMIO plus `KTEXT_NOMINAL → load_addr` and a `KDMAP_NOMINAL → RAM` direct-map), `csrw satp`, and jumps into the high-half VA. Post-jump `post_trampoline_entry` applies `R_RISCV_RELATIVE` with `slide = ktext_base - LINK_BASE` before any Rust code touches a relocated global, then tail-calls `rust_main`. `rust_main` initializes the linked-list heap (`KHEAP` at its KDMAP VA), frame allocators for page tables (ktables) and general-purpose kernel pages (kpages) — both returning KDMAP VAs via `add_frame_with_va_base` — installs `OrbitLogger`/`OrbitSubscriber`, builds the final Sv48 table via `map_kernel_self` (KTEXT / KDMAP / KMMIO, no identity left), allocates per-hart `HartContext` structs, and `sret`s into `k_smpstart`. `k_smpstart` re-`init_serial`s at `kmmio_uart()` before any print, signals bl (`ecall(4)` then `ecall(1)`), and kicks harts 1..N via KMMIO CLINT MSIPs. Hart 0 then runs `k_manage`; others `k_idle`/WFI. See [kmain/src/kernel/memmap.rs](kmain/src/kernel/memmap.rs) for layout atomics, constants, `KernelLayout`, and `RootTable` helpers.

- **U-mode sample — [umode/](umode/)** linked at `0x2_2000_0000` (`USER_TEXT_BASE`). Currently a hardcoded demo ([umode/src/main.rs](umode/src/main.rs)) that prints to serial, sleeps, `mmap`s a shared region, registers a `NetChannel`, and opens a TCP connection. The kernel creates exactly one process from this ELF in `k_smpstart`.

### Syscall ABI

`ecall` with syscall number in `a0`, args in `a1..a4`. Dispatched by the `cause == 8` arm of `s_trap` in [kmain/src/bin/orbit.rs](kmain/src/bin/orbit.rs):

- `0` — exit (noreturn)
- `1` — serial_print(ptr, len)
- `2` — sleep_ms(ms)
- `4096` — mmap(vaddr, len, perms, share_with_kernel)
- `4097` — register NetChannel(nc_vaddr, sock_type)

### Supporting crates (all under the same workspace)

- [mmu/](mmu/) — Sv48 page tables, `PagePermissions`, `MappingConfig`, `id_map_range`, `map_address_range`, `map_va_range`, `virt_to_phys`. Walkers take `&RootTable<'_>`, which pairs a `&PageTable` with a PA→VA bias so they can follow PPNs (always physical) back into the supervisor's KDMAP view. bl/early-trampoline use `RootTable::identity(...)` (bias 0); kmain uses `memmap::kernel_root(...)` / `kernel_root_from_pa(...)` (bias `= kdmap_base - ram_phys_base`). `alloc` feature wires it up against `alloc::Vec`-backed page pools.
- [mem/](mem/) — `FrameAllocator`, `round_u64_up`/`down` helpers, `prev_power_of_two`.
- [device/](device/) — `HartContext` (cache-line-aligned, field offsets are load-bearing — consumed by `asm/trap.S` and `boot.S`), `TrapFrame`, `Stack` (2 MiB), `SysInfo`, DTB walkers (`find_serial_port`, `find_ram`). The trap assembly hard-codes offsets into `HartContext`; changing field order requires updating the `.S` files.
- [process/](process/) — `Thread`, `Process`, `ThreadState`, block-reason enums (`MemMapReq`, `NetChannelRegistrationReq`).
- [net_channel/](net_channel/) — shared-memory SPSC queues between user programs and the kernel networking thread. `kernel` feature enables the kernel-side `update_tcp` methods that drive smoltcp sockets. Layout of `NetChannel` / `NetChannelState` / `NetChannelQueue` is part of the user/kernel ABI — do not change field order casually.
- [serial/](serial/) — `MpUart` wrapping `ns16550a::Uart` with a spinlock; exposes `println!` macro and a global init.
- [kmain/src/drivers/e1000.rs](kmain/src/drivers/e1000.rs) — PCI-discovered e1000 NIC; ring buffers fed by DMA from kernel pages, integrated with smoltcp via `k_net` (a dedicated kernel thread).

### SMP and scheduling

`HartContext` is the per-hart state structure. `sscratch` always points to the current hart's `HartContext` (as a KDMAP VA under the kernel satp). Cross-hart wakeups go via ACLINT SSWI at `kmmio_sswi()` (physical `0x02F00000`; one `u32` per hart — `write_sswi`/`supervisor_wake_hart`). Timer interrupts use Sstc (`stimecmp` at CSR `0x14D`). The scheduler (hart 0 only) assigns runnable threads to idle harts and sends an IPI; harts receive S-mode software interrupts in `s_trap` (async cause `1`) and context-switch via `check_context_and_switch`. `s_trap_vector` switches `satp` to the kernel's own on entry — syscall handlers that reach user memory walk the user PT manually and go through the KDMAP alias (`memmap::user_va_to_kdmap`); a future SUM-gate milestone inverts this.
