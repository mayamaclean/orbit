//! PLIC (Platform-Level Interrupt Controller) driver.
//!
//! Two layers:
//! - [`find_plic`]: DTB walk producing a [`PlicInfo`] with geometry +
//!   per-hart S-mode context indices.
//! - [`Plic`]: thin MMIO register-file over a KMMIO VA. All accessors are
//!   `unsafe` since they issue volatile writes to device memory; callers
//!   own sequencing (sfence, context assignment).
//!
//! Register layout (SiFive-style PLIC):
//! - priority: `base + 4 * src` (word per source, u32)
//! - enable bitmap: `base + 0x2000 + 0x80 * ctx`, bit `src % 32` in word
//!   `src / 32`
//! - threshold: `base + 0x200000 + 0x1000 * ctx`
//! - claim/complete: `base + 0x200004 + 0x1000 * ctx`

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::ptr::null_mut;
use core::sync::atomic::{AtomicPtr, Ordering};
use dtoolkit::{
    Node, Property,
    fdt::{Fdt, FdtNode},
};
use mmu::mmap::{PageAlloc, RootTable};
use tracing::{error, info};

use crate::kernel::memmap;

#[derive(Debug)]
pub struct PlicInfo {
    pub pa_base: u64,
    pub size: u64,
    pub ndev: u32,
    /// S-mode PLIC context index per hart id. Indexed by hart_id; an entry
    /// is `None` if the hart has no S-mode context listed in the DTB.
    pub s_contexts: Vec<Option<u32>>,
}

fn plic_node_matches(n: &FdtNode<'_>) -> bool {
    let Some(compat) = n.property("compatible")
    else {
        return false;
    };
    compat
        .as_str_list()
        .any(|s| s == "riscv,plic0" || s == "sifive,plic-1.0.0")
}

fn find_plic_node<'a>(n: FdtNode<'a>) -> Option<FdtNode<'a>> {
    if plic_node_matches(&n) {
        return Some(n);
    }
    for child in n.children() {
        if let Some(m) = find_plic_node(child) {
            return Some(m);
        }
    }
    None
}

fn build_intc_map(fdt: &Fdt<'_>) -> Result<Vec<(u32, u32)>, ()> {
    let cpus = fdt.root().child("cpus").ok_or(())?;
    let mut out = Vec::new();
    for cpu in cpus.children() {
        if !cpu.name().starts_with("cpu@") {
            continue;
        }
        let hart_id = cpu
            .property("reg")
            .and_then(|p| p.as_u32().ok())
            .ok_or(())?;
        let intc = cpu.child("interrupt-controller").ok_or(())?;
        let phandle = intc
            .property("phandle")
            .and_then(|p| p.as_u32().ok())
            .ok_or(())?;
        out.push((phandle, hart_id));
    }
    Ok(out)
}

pub fn find_plic(fdt: &Fdt<'_>) -> Result<PlicInfo, ()> {
    let node = find_plic_node(fdt.root()).ok_or(())?;

    let mut regs = node.reg().map_err(|_| ())?.ok_or(())?;
    let reg = regs.next().ok_or(())?;
    let pa_base = reg.address::<u64>().map_err(|_| ())?;
    let size = reg.size::<u64>().map_err(|_| ())?;

    let ndev = node
        .property("riscv,ndev")
        .ok_or(())?
        .as_u32()
        .map_err(|_| ())?;

    let intc_map = build_intc_map(fdt)?;
    let max_hart = intc_map.iter().map(|(_, h)| *h).max().unwrap_or(0) as usize;
    let mut s_contexts: Vec<Option<u32>> = vec![None; max_hart + 1];

    let ie = node.property("interrupts-extended").ok_or(())?;
    let pairs = ie.as_prop_encoded_array([1usize, 1usize]).map_err(|_| ())?;

    for (i, [intc_cells, irq_cells]) in pairs.enumerate() {
        let intc_phandle: u32 = intc_cells.to_int().map_err(|_| ())?;
        let irq: u32 = irq_cells.to_int().map_err(|_| ())?;
        if irq != 9 {
            continue;
        }
        let Some((_, hart)) = intc_map.iter().find(|(p, _)| *p == intc_phandle)
        else {
            continue;
        };
        let hart = *hart as usize;
        if hart >= s_contexts.len() {
            s_contexts.resize(hart + 1, None);
        }
        s_contexts[hart] = Some(i as u32);
    }

    Ok(PlicInfo {
        pa_base,
        size,
        ndev,
        s_contexts,
    })
}

// Register offsets, expressed in u32 words so they compose with `*mut u32`
// pointer arithmetic.
const PRIORITY_BASE: usize = 0;
const ENABLE_BASE: usize = 0x2000 / 4;
const ENABLE_STRIDE: usize = 0x80 / 4;
const THRESHOLD_BASE: usize = 0x200000 / 4;
const THRESHOLD_STRIDE: usize = 0x1000 / 4;
const CLAIM_OFFSET: usize = 1;

pub struct Plic {
    base: *mut u32,
    ndev: u32,
}

// SAFETY: `Plic` holds a KMMIO VA into PLIC device memory. All accesses
// are 32-bit volatile word reads/writes; the device handles ordering per
// spec. No interior Rust state — sharing across harts is safe.
unsafe impl Sync for Plic {}
unsafe impl Send for Plic {}

impl Plic {
    /// # Safety
    /// `base_kva` must be a KMMIO mapping covering the PLIC register
    /// region, and `ndev` must match the device's source count.
    pub const unsafe fn new(base_kva: u64, ndev: u32) -> Self {
        Self {
            base: base_kva as *mut u32,
            ndev,
        }
    }

    pub fn ndev(&self) -> u32 {
        self.ndev
    }

    #[inline]
    fn priority_ptr(&self, src: u32) -> *mut u32 {
        unsafe { self.base.add(PRIORITY_BASE + src as usize) }
    }

    #[inline]
    fn enable_word_ptr(&self, ctx: u32, src: u32) -> *mut u32 {
        let idx = ENABLE_BASE + (ctx as usize) * ENABLE_STRIDE + (src as usize / 32);
        unsafe { self.base.add(idx) }
    }

    #[inline]
    fn threshold_ptr(&self, ctx: u32) -> *mut u32 {
        let idx = THRESHOLD_BASE + (ctx as usize) * THRESHOLD_STRIDE;
        unsafe { self.base.add(idx) }
    }

    #[inline]
    fn claim_ptr(&self, ctx: u32) -> *mut u32 {
        let idx = THRESHOLD_BASE + (ctx as usize) * THRESHOLD_STRIDE + CLAIM_OFFSET;
        unsafe { self.base.add(idx) }
    }

    /// # Safety
    /// Caller must not race priority writes for the same source across
    /// harts. Writes are 32-bit atomic on the bus; concurrency is a
    /// semantic concern, not a memory-model one.
    pub unsafe fn set_priority(&self, src: u32, prio: u32) {
        unsafe {
            self.priority_ptr(src).write_volatile(prio);
        }
    }

    /// # Safety
    /// `ctx` must be a valid PLIC context for this device.
    pub unsafe fn set_threshold(&self, ctx: u32, thr: u32) {
        unsafe {
            self.threshold_ptr(ctx).write_volatile(thr);
        }
    }

    /// Enable `src` on `ctx`.
    ///
    /// # Safety
    /// RMW on the enable bitmap word — caller must serialize concurrent
    /// enables/disables touching the same 32-source window on the same
    /// context. MVP does all enables from hart 0 at init, so no race.
    pub unsafe fn enable_source(&self, ctx: u32, src: u32) {
        let ptr = self.enable_word_ptr(ctx, src);
        unsafe {
            let v = ptr.read_volatile();
            ptr.write_volatile(v | (1u32 << (src % 32)));
        }
    }

    /// # Safety
    /// See [`Plic::enable_source`] for the RMW concurrency note.
    pub unsafe fn disable_source(&self, ctx: u32, src: u32) {
        let ptr = self.enable_word_ptr(ctx, src);
        unsafe {
            let v = ptr.read_volatile();
            ptr.write_volatile(v & !(1u32 << (src % 32)));
        }
    }

    /// Read the claim register for `ctx`. Returns 0 when no source is
    /// pending (spurious claim). The read atomically takes ownership of
    /// the reported source on this context.
    ///
    /// # Safety
    /// `ctx` must be this hart's own S-mode context; reading another
    /// hart's claim steals its pending IRQ.
    pub unsafe fn claim(&self, ctx: u32) -> u32 {
        unsafe { self.claim_ptr(ctx).read_volatile() }
    }

    /// Signal completion for `src` on `ctx`. Must pair with a claim that
    /// returned the same `src`.
    pub unsafe fn complete(&self, ctx: u32, src: u32) {
        unsafe {
            self.claim_ptr(ctx).write_volatile(src);
        }
    }

    /// Read the pending-bit word containing `src`. Diagnostic only.
    pub fn pending_word_for(&self, src: u32) -> u32 {
        let idx = 0x1000 / 4 + (src as usize / 32);
        unsafe { self.base.add(idx).read_volatile() }
    }

    /// Read the enable-bitmap word for `src` on `ctx`. Diagnostic only.
    pub fn enable_word_for(&self, ctx: u32, src: u32) -> u32 {
        unsafe { self.enable_word_ptr(ctx, src).read_volatile() }
    }
}

// Globals initialized exactly once by `install` on hart 0, read from any
// hart thereafter. Mirrors the `pending_frees::FREE_RINGS` pattern.
static PLIC_PTR: AtomicPtr<Plic> = AtomicPtr::new(null_mut());
static PLIC_INFO_PTR: AtomicPtr<PlicInfo> = AtomicPtr::new(null_mut());

/// Global `Plic` accessor. Returns `None` before [`install`] completes;
/// callers in the trap path should treat that as "no external IRQ
/// handling yet" and ignore the claim.
pub fn plic() -> Option<&'static Plic> {
    let p = PLIC_PTR.load(Ordering::Acquire);
    if p.is_null() {
        None
    }
    else {
        Some(unsafe { &*p })
    }
}

/// Read-only view of the `PlicInfo` captured during install.
pub fn info() -> Option<&'static PlicInfo> {
    let p = PLIC_INFO_PTR.load(Ordering::Acquire);
    if p.is_null() {
        None
    }
    else {
        Some(unsafe { &*p })
    }
}

/// Discover the PLIC via DTB, install a KMMIO alias for its registers,
/// mask every source (priority 0), and zero thresholds on all S-mode
/// contexts. After this returns `Ok`, [`plic`] yields the live device
/// and handlers can be registered with priority > 0 to unmask.
///
/// # Safety
/// Must be called exactly once, from hart 0, after kernel paging is
/// active and before any other hart can reach a cause-9 trap. Installs
/// a mapping in the active root table and issues an `sfence.vma` on
/// the calling hart.
pub unsafe fn install(
    fdt: &Fdt<'_>,
    rt: &RootTable<'_>,
    pa_alloc: &mut PageAlloc,
) -> Result<(), ()> {
    if !PLIC_PTR.load(Ordering::Relaxed).is_null() {
        error!("plic::install: already initialized");
        return Err(());
    }

    let info = find_plic(fdt)?;
    info!(
        "plic@{:#x}..{:#x} ndev={} s_contexts={:?}",
        info.pa_base,
        info.pa_base + info.size,
        info.ndev,
        info.s_contexts,
    );

    let base_kva = unsafe {
        memmap::install_kmmio_alias(rt, pa_alloc, info.pa_base..info.pa_base + info.size)?
    };
    riscv::asm::sfence_vma_all();

    let plic = unsafe { Plic::new(base_kva, info.ndev) };

    // Safety sweep: every source starts masked (priority 0) so only
    // explicit `plic_register` calls unmask. Thresholds at 0 mean any
    // unmasked source with priority > 0 is delivered.
    unsafe {
        for src in 1..=info.ndev {
            plic.set_priority(src, 0);
        }
        for (hart, ctx) in info.s_contexts.iter().enumerate() {
            if let Some(ctx) = ctx {
                plic.set_threshold(*ctx, 0);
                info!("plic: hart{} s-ctx={} threshold=0", hart, ctx);
            }
        }
    }

    let plic_leaked: &'static Plic = Box::leak(Box::new(plic));
    let info_leaked: &'static PlicInfo = Box::leak(Box::new(info));
    PLIC_PTR.store(plic_leaked as *const _ as *mut _, Ordering::Release);
    PLIC_INFO_PTR.store(info_leaked as *const _ as *mut _, Ordering::Release);

    Ok(())
}

/// Handler function invoked when a claimed source is dispatched. Runs
/// in trap context with `sstatus.SIE = 0` — no blocking, no nesting.
pub type Handler = fn(src: u32);

/// Max PLIC source id (exclusive). Sized generously over virt's ndev=95.
pub const MAX_SRC: usize = 128;

// One slot per source. Stored as `*mut ()`; the loader transmutes it
// back to `Handler` on dispatch. Null = unregistered.
static HANDLERS: [AtomicPtr<()>; MAX_SRC] = [const { AtomicPtr::new(null_mut()) }; MAX_SRC];

fn s_context_for(hart: usize) -> Option<u32> {
    info().and_then(|i| i.s_contexts.get(hart).copied().flatten())
}

/// Register `handler` for source `src` and enable it on `hart`'s S-mode
/// PLIC context with priority 1. Each source is pinned to a single
/// hart in MVP — the `hart` parameter is plumbed through so future
/// distribution is a call-site change.
pub fn plic_register(src: u32, handler: Handler, hart: usize) -> Result<(), ()> {
    let Some(plic) = plic()
    else {
        error!("plic_register: plic not initialized");
        return Err(());
    };
    if src == 0 || (src as usize) >= MAX_SRC || src > plic.ndev() {
        error!(
            "plic_register: src {} out of range (ndev={})",
            src,
            plic.ndev()
        );
        return Err(());
    }
    let Some(ctx) = s_context_for(hart)
    else {
        error!("plic_register: hart {} has no S-mode PLIC context", hart);
        return Err(());
    };

    // Store handler before unmasking so a claim arriving the moment
    // after `set_priority` sees a valid pointer. Null loads in
    // `dispatch` are tolerated, but this makes the happy-path race
    // benign.
    HANDLERS[src as usize].store(handler as *mut (), Ordering::Release);

    unsafe {
        plic.set_priority(src, 1);
        plic.enable_source(ctx, src);
    }

    info!("plic: registered src={} hart={} s-ctx={}", src, hart, ctx);
    Ok(())
}

/// Configure the ns16550a to raise RX interrupts. The UART-RX →
/// pane-cycle handler registration is currently disabled (pane cycling
/// is driven by Ctrl+Tab through the virtio-input driver instead), so
/// this only programs the FCR/MCR/IER; it no longer wires a PLIC
/// handler for IRQ 10.
pub fn install_uart_rx_cycle() -> Result<(), ()> {
    const UART_RX_IRQ: u32 = 10;
    //plic_register(UART_RX_IRQ, uart_rx_cycle_handler, 0)?;

    // QEMU's ns16550a only asserts its interrupt line when MCR.OUT2 is
    // set AND the FIFO is enabled with a matched RX trigger level. Our
    // `serial::init_serial` just constructs `Uart::new` (stores base
    // addr, no register writes), so we configure here.
    //
    // Assumes LCR.DLAB is already clear (true for QEMU's post-reset
    // state). Offsets:
    //   1: IER — enable RX available IRQ (bit 0)
    //   2: FCR — enable FIFO, clear RX/TX FIFOs, 1-byte RX trigger
    //   4: MCR — set OUT2 (bit 3) so IRQ line reaches the PLIC
    unsafe {
        let base = memmap::kmmio_uart();
        (base as *mut u8).add(2).write_volatile(0x07); // FCR
        (base as *mut u8).add(4).write_volatile(0x08); // MCR
        (base as *mut u8).add(1).write_volatile(0x01); // IER
    }
    info!("plic: uart rx pane-cycle armed on IRQ {}", UART_RX_IRQ);
    Ok(())
}

#[allow(unused)]
fn uart_rx_cycle_handler(_src: u32) {
    // Draining RBR (offset 0) clears the UART's RX-ready line; without
    // this the source stays asserted and we'd re-trap immediately
    // after complete.
    let rbr = memmap::kmmio_uart() as *const u8;
    let _byte = unsafe { rbr.read_volatile() };

    // Push a CycleActive cmd onto the k_gpu ring. Runs in trap
    // context (SIE=0) — thingbuf push is lock-free, safe here.
    // If the ring is full we drop the keystroke; at human typing
    // rates with `k_gpu` waking every ~50 ms this is not a concern.
    let _ = crate::drivers::k_gpu::push_cycle_active();
}

/// Drain pending sources on this hart's S-mode context. Invoked from
/// the `scause = 9` arm of `s_trap`. Loops until claim returns 0 so a
/// burst of pending IRQs is handled in a single trap.
pub fn dispatch(ctx: u32) {
    let Some(plic) = plic()
    else {
        return;
    };
    if ctx == u32::MAX {
        return;
    }
    loop {
        let src = unsafe { plic.claim(ctx) };
        if src == 0 {
            return;
        }
        if (src as usize) < MAX_SRC {
            let ptr = HANDLERS[src as usize].load(Ordering::Acquire);
            if !ptr.is_null() {
                // SAFETY: only `plic_register` writes `HANDLERS[src]`, and
                // it only writes valid `Handler` pointers. Function-pointer
                // layout matches `*mut ()` on riscv64.
                let handler: Handler = unsafe { core::mem::transmute(ptr) };
                handler(src);
            }
        }
        unsafe {
            plic.complete(ctx, src);
        }
    }
}
