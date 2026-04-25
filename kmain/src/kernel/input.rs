//! Kernel-side input dispatch. Sits above input drivers (today
//! virtio-input; tomorrow USB keyboard, virtio-mouse, …) and decides
//! what to do with each event.
//!
//! Behavior:
//! - Track Shift/Ctrl/Alt modifier state across press/release pairs.
//! - On Ctrl+Tab key-down, fan out to `k_gpu` as a `CycleActive` cmd.
//! - For other printable / nav key-down events, look up the active
//!   pid (mirror at [`crate::kernel::stdin::ACTIVE_PID`]); if a
//!   process is active, translate the key via [`evdev_to_bytes`] and
//!   push each byte onto that process's [`ProcessStdin`] ring. Each
//!   push also takes-and-signals any parked reader so a blocked
//!   `read_stdin` thread wakes promptly.
//! - When no process is active (kernel pane), keys are floored.
//!
//! Trap-context safe: modifier state is a static `AtomicU8`, the
//! active-pid lookup is one atomic load, and the ring push +
//! parked-Arc swap are lock-free.

use core::sync::atomic::{AtomicU8, Ordering};

use virtio_input::proto::{
    EV_KEY, KEY_0, KEY_1, KEY_2, KEY_3, KEY_4, KEY_5, KEY_6, KEY_7, KEY_8, KEY_9,
    KEY_A, KEY_APOSTROPHE, KEY_B, KEY_BACKSLASH, KEY_BACKSPACE, KEY_C, KEY_COMMA,
    KEY_D, KEY_DELETE, KEY_DOT, KEY_DOWN, KEY_E, KEY_END, KEY_ENTER, KEY_EQUAL,
    KEY_ESC, KEY_F, KEY_G, KEY_GRAVE, KEY_H, KEY_HOME, KEY_I, KEY_J, KEY_K, KEY_L,
    KEY_LEFT, KEY_LEFTALT, KEY_LEFTBRACE, KEY_LEFTCTRL, KEY_LEFTSHIFT, KEY_M,
    KEY_MINUS, KEY_N, KEY_O, KEY_P, KEY_Q, KEY_R, KEY_RIGHT, KEY_RIGHTALT,
    KEY_RIGHTBRACE, KEY_RIGHTCTRL, KEY_RIGHTSHIFT, KEY_S, KEY_SEMICOLON, KEY_SLASH,
    KEY_SPACE, KEY_T, KEY_TAB, KEY_U, KEY_UP, KEY_V, KEY_W, KEY_X, KEY_Y, KEY_Z,
    VAL_PRESS, VAL_RELEASE, VAL_REPEAT,
};
use virtio_input::InputEvent;

use crate::drivers::k_gpu;
use crate::kernel::stdin;

const MOD_SHIFT: u8 = 1 << 0;
const MOD_CTRL: u8 = 1 << 1;
#[allow(dead_code)]
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

    // Bindings + character translation fire on key-down (and on
    // repeat for typing — repeat at 0.5+ Hz is what the user
    // expects when they hold a key). Pane cycling stays press-only
    // so a held Ctrl+Tab doesn't spam through panes.
    let is_repeat = ev.value == VAL_REPEAT;
    if ev.value != VAL_PRESS && !is_repeat {
        return;
    }

    let mods = MODS.load(Ordering::Relaxed);

    if !is_repeat && ev.code == KEY_TAB && mods & MOD_CTRL != 0 {
        // Floor return value: ring full at human typing rates means a
        // dropped pane switch, which is fine — user can press again.
        let _ = k_gpu::push_cycle_active();
        return;
    }

    // Route printable / nav keys to the active process's stdin.
    // No active process (kernel pane) → drop. No registered stdin
    // for the active pid → drop (race with process teardown).
    let Some(pid) = stdin::active_pid() else { return };
    let Some(stdin_arc) = stdin::get(pid) else { return };

    let mut buf = [0u8; 4];
    let n = evdev_to_bytes(ev.code, mods, &mut buf);
    for &b in &buf[..n] {
        stdin_arc.push_byte(b);
    }
}

/// Translate an evdev key code + modifier mask into UTF-8 bytes. Up
/// to 4 bytes (arrow escape sequences are 3 bytes — `ESC [ A` etc.).
/// Returns 0 for keys we don't know how to encode (function keys,
/// caps lock, etc.) so the caller drops them.
///
/// Ctrl+letter produces the standard control-character byte
/// (Ctrl-C → `\x03`, Ctrl-D → `\x04`, …). Alt is currently ignored.
fn evdev_to_bytes(code: u16, mods: u8, out: &mut [u8; 4]) -> usize {
    let shift = mods & MOD_SHIFT != 0;
    let ctrl = mods & MOD_CTRL != 0;

    // Ctrl+letter → control character (a..z map to 0x01..0x1A).
    if ctrl {
        if let Some(letter_idx) = letter_index(code) {
            out[0] = letter_idx + 1;  // 'a' → 0x01, 'b' → 0x02, ...
            return 1;
        }
    }

    // Special keys.
    match code {
        KEY_ENTER     => { out[0] = b'\n'; return 1; }
        KEY_BACKSPACE => { out[0] = 0x08; return 1; }
        KEY_TAB       => { out[0] = b'\t'; return 1; }
        KEY_ESC       => { out[0] = 0x1b; return 1; }
        KEY_SPACE     => { out[0] = b' '; return 1; }
        KEY_DELETE    => { out[0] = 0x7f; return 1; }
        // Arrow / nav: emit ANSI escape sequences. Same encoding
        // xterm uses, which standard line editors expect.
        KEY_UP    => { out[..3].copy_from_slice(&[0x1b, b'[', b'A']); return 3; }
        KEY_DOWN  => { out[..3].copy_from_slice(&[0x1b, b'[', b'B']); return 3; }
        KEY_RIGHT => { out[..3].copy_from_slice(&[0x1b, b'[', b'C']); return 3; }
        KEY_LEFT  => { out[..3].copy_from_slice(&[0x1b, b'[', b'D']); return 3; }
        KEY_HOME  => { out[..3].copy_from_slice(&[0x1b, b'[', b'H']); return 3; }
        KEY_END   => { out[..3].copy_from_slice(&[0x1b, b'[', b'F']); return 3; }
        _ => {}
    }

    // Letters.
    if let Some(idx) = letter_index(code) {
        out[0] = if shift { b'A' + idx } else { b'a' + idx };
        return 1;
    }

    // Number row + symbol keys.
    let (lo, hi) = match code {
        KEY_1 => (b'1', b'!'),
        KEY_2 => (b'2', b'@'),
        KEY_3 => (b'3', b'#'),
        KEY_4 => (b'4', b'$'),
        KEY_5 => (b'5', b'%'),
        KEY_6 => (b'6', b'^'),
        KEY_7 => (b'7', b'&'),
        KEY_8 => (b'8', b'*'),
        KEY_9 => (b'9', b'('),
        KEY_0 => (b'0', b')'),
        KEY_MINUS      => (b'-',  b'_'),
        KEY_EQUAL      => (b'=',  b'+'),
        KEY_LEFTBRACE  => (b'[',  b'{'),
        KEY_RIGHTBRACE => (b']',  b'}'),
        KEY_BACKSLASH  => (b'\\', b'|'),
        KEY_SEMICOLON  => (b';',  b':'),
        KEY_APOSTROPHE => (b'\'', b'"'),
        KEY_GRAVE      => (b'`',  b'~'),
        KEY_COMMA      => (b',',  b'<'),
        KEY_DOT        => (b'.',  b'>'),
        KEY_SLASH      => (b'/',  b'?'),
        _ => return 0,
    };
    out[0] = if shift { hi } else { lo };
    1
}

/// Map a letter keycode to 0..=25 (a..z). Returns None for non-
/// letter keys. evdev's letter codes aren't contiguous (qwerty
/// row order), so this is a match.
fn letter_index(code: u16) -> Option<u8> {
    Some(match code {
        KEY_A => 0,  KEY_B => 1,  KEY_C => 2,  KEY_D => 3,  KEY_E => 4,
        KEY_F => 5,  KEY_G => 6,  KEY_H => 7,  KEY_I => 8,  KEY_J => 9,
        KEY_K => 10, KEY_L => 11, KEY_M => 12, KEY_N => 13, KEY_O => 14,
        KEY_P => 15, KEY_Q => 16, KEY_R => 17, KEY_S => 18, KEY_T => 19,
        KEY_U => 20, KEY_V => 21, KEY_W => 22, KEY_X => 23, KEY_Y => 24,
        KEY_Z => 25,
        _ => return None,
    })
}

fn modifier_bit(code: u16) -> Option<u8> {
    match code {
        KEY_LEFTSHIFT | KEY_RIGHTSHIFT => Some(MOD_SHIFT),
        KEY_LEFTCTRL | KEY_RIGHTCTRL => Some(MOD_CTRL),
        KEY_LEFTALT | KEY_RIGHTALT => Some(MOD_ALT),
        _ => None,
    }
}
