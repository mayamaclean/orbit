//! Kernel-side input dispatch. Sits above input drivers (today
//! virtio-input; tomorrow USB keyboard, virtio-mouse, …) and decides
//! what to do with each event.
//!
//! Today's policy is small:
//! - Track Shift/Ctrl/Alt modifier state across press/release pairs.
//! - On Ctrl+Tab key-down, fan out to `k_gpu` as a `CycleActive` cmd.
//! - Drop everything else. §8 will add a fan-out path to the active
//!   process's stdin pipe; until then non-binding keys are floored
//!   rather than buffered (buffering pre-process keystrokes would be
//!   an unbounded footgun).
//!
//! Trap-context safe: the only mutable state is a static `AtomicU8`
//! and the only side effect is a lock-free `thingbuf` push on bound
//! events. No allocations, no locks.

use core::sync::atomic::{AtomicU8, Ordering};

use virtio_input::proto::{
    EV_KEY, KEY_LEFTALT, KEY_LEFTCTRL, KEY_LEFTSHIFT, KEY_RIGHTALT, KEY_RIGHTCTRL,
    KEY_RIGHTSHIFT, KEY_TAB, VAL_PRESS, VAL_RELEASE, VAL_REPEAT,
};
use virtio_input::InputEvent;

use crate::drivers::k_gpu;

const MOD_SHIFT: u8 = 1 << 0;
const MOD_CTRL: u8 = 1 << 1;
const MOD_ALT: u8 = 1 << 2;

static MODS: AtomicU8 = AtomicU8::new(0);

/// Single entry point for input drivers. Called from trap context.
pub fn dispatch(ev: InputEvent) {
    if ev.ty != EV_KEY {
        // EV_SYN, EV_MSC, axis events, etc. — nothing to do today.
        return;
    }

    // Update modifier state on shift/ctrl/alt transitions. Press =
    // value 1, release = value 0, repeat = value 2 (treat as held).
    if let Some(bit) = modifier_bit(ev.code) {
        match ev.value {
            VAL_PRESS | VAL_REPEAT => {
                MODS.fetch_or(bit, Ordering::Relaxed);
            }
            VAL_RELEASE => {
                MODS.fetch_and(!bit, Ordering::Relaxed);
            }
            _ => {}
        }
        return;
    }

    // Key bindings fire on key-down only. Repeat events are ignored
    // for now — pane cycling on auto-repeat would feel awful.
    if ev.value != VAL_PRESS {
        return;
    }

    let mods = MODS.load(Ordering::Relaxed);
    if ev.code == KEY_TAB && mods & MOD_CTRL != 0 {
        // Floor return value: ring full at human typing rates means a
        // dropped pane switch, which is fine — user can press again.
        let _ = k_gpu::push_cycle_active();
    }

    // Everything else: drop. §8 will route printable + nav keys to the
    // active process's stdin pipe.
}

fn modifier_bit(code: u16) -> Option<u8> {
    match code {
        KEY_LEFTSHIFT | KEY_RIGHTSHIFT => Some(MOD_SHIFT),
        KEY_LEFTCTRL | KEY_RIGHTCTRL => Some(MOD_CTRL),
        KEY_LEFTALT | KEY_RIGHTALT => Some(MOD_ALT),
        _ => None,
    }
}
