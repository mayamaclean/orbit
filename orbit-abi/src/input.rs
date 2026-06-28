//! Raw key-event ABI.
//!
//! Companion to the byte-stream `read_stdin` syscall — same source
//! (virtio-input → `kernel::input::dispatch`) but exposes the
//! structured event before lossy encoding into UTF-8 + ANSI escapes.
//! Targets ratatui-shaped TUI consumers that need:
//!
//! - Key release events (the byte path drops them).
//! - Modifiers preserved across all keycodes (the byte path collapses
//!   `Shift+letter → uppercase` and erases `Shift+arrow` entirely).
//! - F-keys, BackTab, Insert, Pause, etc. (the byte path encodes
//!   *some* of these as ANSI escapes; the round-trip is lossy and
//!   asymmetric across keycodes).
//!
//! Wire layout: [`KeyEvent`] is `#[repr(C)]` with explicit field order
//! so the kernel can `core::slice::from_raw_parts` directly into the
//! caller's user buffer. `KeyCode` discriminant + `Modifiers` bits +
//! `KeyEventKind` are all stable u8/u32 reps; do not renumber.
//!
//! Bytes path is unchanged — `read_stdin` still drains UTF-8 + ANSI
//! escape sequences for shells / line editors. Processes that care
//! about structured events read here instead (or in addition).

/// Wire encoding of a single key event. 16 bytes, naturally aligned.
///
/// `code` carries the [`KeyCode`] discriminant in the low byte and
/// associated data in the upper bytes:
/// - `Char`: bytes 1..4 hold the codepoint (LE u32 in bytes 0..4 means
///   `Char` is encoded as `(codepoint << 8) | KeyCode::Char as u8`).
/// - All other variants: bytes 1..4 are zero.
///
/// Construction goes through [`KeyEvent::new`]; decoding goes through
/// the accessors [`KeyEvent::key_code`] / [`KeyEvent::mods`] /
/// [`KeyEvent::event_kind`] (plus the free `decoded_char` / `decoded_fn`
/// helpers); consumers should not reach for the raw `code` field unless
/// they're matching the on-wire bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct KeyEvent {
    /// Tagged code. See type-level docs for layout.
    pub code: u32,
    /// Modifier bitmask (`SHIFT | CONTROL | ALT | SUPER`).
    pub modifiers: u32,
    /// `KeyEventKind` discriminant in the low byte. Other bytes
    /// reserved (kernel writes 0; readers ignore unknown bits).
    pub kind: u32,
    /// Reserved for future use (timestamp, repeat count, …). Kernel
    /// writes 0.
    pub _reserved: u32,
}

impl KeyEvent {
    /// Construct an event from its components.
    pub const fn new(code: KeyCode, modifiers: Modifiers, kind: KeyEventKind) -> Self {
        Self {
            code: code.encode(),
            modifiers: modifiers.bits(),
            kind: kind as u32,
            _reserved: 0,
        }
    }

    /// Decode the event's `KeyCode`. Returns `None` if the wire bits
    /// don't decode to a known variant.
    pub const fn key_code(&self) -> Option<KeyCode> {
        KeyCode::decode(self.code)
    }

    /// Decode the modifier mask.
    pub const fn mods(&self) -> Modifiers {
        Modifiers::from_bits_truncate(self.modifiers)
    }

    /// Decode the event kind. Returns `None` if the discriminant is
    /// unknown.
    pub const fn event_kind(&self) -> Option<KeyEventKind> {
        match self.kind & 0xff {
            0 => Some(KeyEventKind::Press),
            1 => Some(KeyEventKind::Release),
            2 => Some(KeyEventKind::Repeat),
            _ => None,
        }
    }
}

/// Logical key. Mirrors crossterm's `KeyCode` shape so a future
/// crossterm-shim can map 1:1 — but lives in `orbit-abi` so the
/// kernel and pure no_std consumers don't drag in crossterm.
///
/// Discriminants are part of the wire format — do not renumber.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum KeyCode {
    /// Printable character. Codepoint travels in bytes 1..4 of
    /// [`KeyEvent::code`].
    Char = 0,
    Backspace = 1,
    Enter = 2,
    Left = 3,
    Right = 4,
    Up = 5,
    Down = 6,
    Home = 7,
    End = 8,
    PageUp = 9,
    PageDown = 10,
    Tab = 11,
    BackTab = 12,
    Delete = 13,
    Insert = 14,
    Escape = 15,
    /// `F1..F12` — function index in bytes 1..4 of [`KeyEvent::code`]
    /// (1-based to avoid clashing with the all-zero `F1`-default
    /// reading of an uninitialized event).
    F = 16,
    /// Caps lock toggle press/release. Modifiers carry no separate
    /// bit; consumers track lock state from press events.
    CapsLock = 17,
    NumLock = 18,
    ScrollLock = 19,
    PrintScreen = 20,
    Pause = 21,
    Menu = 22,
    /// Null event — used as a sentinel by ABI tests; kernel never
    /// emits it.
    Null = 0xFF,
}

impl KeyCode {
    /// Tag-byte width — the low byte of [`KeyEvent::code`].
    const TAG_MASK: u32 = 0xff;

    /// Encode to wire bits. `Char` and `F` carry their data in the
    /// upper 24 bits; all others encode the discriminant alone.
    pub const fn encode(self) -> u32 {
        self as u32
    }

    /// Encode a printable char. Codepoint goes in bytes 1..4.
    pub const fn encode_char(c: char) -> u32 {
        ((c as u32) << 8) | (Self::Char as u32)
    }

    /// Encode a function key index `n ∈ 1..=24` (room for the F13..24
    /// range some keyboards expose).
    pub const fn encode_fn(n: u8) -> u32 {
        ((n as u32) << 8) | (Self::F as u32)
    }

    /// Decode the discriminant. Returns `None` for unknown tags so
    /// future kernels emitting new variants degrade safely on old
    /// userland.
    pub const fn decode(bits: u32) -> Option<Self> {
        match (bits & Self::TAG_MASK) as u8 {
            0 => Some(Self::Char),
            1 => Some(Self::Backspace),
            2 => Some(Self::Enter),
            3 => Some(Self::Left),
            4 => Some(Self::Right),
            5 => Some(Self::Up),
            6 => Some(Self::Down),
            7 => Some(Self::Home),
            8 => Some(Self::End),
            9 => Some(Self::PageUp),
            10 => Some(Self::PageDown),
            11 => Some(Self::Tab),
            12 => Some(Self::BackTab),
            13 => Some(Self::Delete),
            14 => Some(Self::Insert),
            15 => Some(Self::Escape),
            16 => Some(Self::F),
            17 => Some(Self::CapsLock),
            18 => Some(Self::NumLock),
            19 => Some(Self::ScrollLock),
            20 => Some(Self::PrintScreen),
            21 => Some(Self::Pause),
            22 => Some(Self::Menu),
            0xFF => Some(Self::Null),
            _ => None,
        }
    }
}

/// Decode the codepoint payload from a `Char`-tagged code word.
/// Returns `None` if the upper 24 bits don't form a valid scalar.
pub const fn decoded_char(bits: u32) -> Option<char> {
    char::from_u32(bits >> 8)
}

/// Decode the function-key index `n` from an `F`-tagged code word.
pub const fn decoded_fn(bits: u32) -> u8 {
    (bits >> 8) as u8
}

/// Modifier bitmask. Stored as a u32 on the wire so a future Hyper /
/// Meta split has room without a v2 reshape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Modifiers(u32);

impl Modifiers {
    pub const NONE: Self = Self(0);
    pub const SHIFT: Self = Self(1 << 0);
    pub const CONTROL: Self = Self(1 << 1);
    pub const ALT: Self = Self(1 << 2);
    pub const SUPER: Self = Self(1 << 3);

    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Truncate any unknown bits — keeps forward-compat: a future
    /// kernel adding new modifier bits to events doesn't poison
    /// matches in older user code.
    pub const fn from_bits_truncate(bits: u32) -> Self {
        Self(bits & 0b1111)
    }

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// Press / release / autorepeat. Mirrors crossterm's `KeyEventKind`
/// + the natural distinction the underlying virtio-input driver
/// exposes (`VAL_PRESS` / `VAL_RELEASE` / `VAL_REPEAT`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum KeyEventKind {
    Press = 0,
    Release = 1,
    Repeat = 2,
}

/// Bit set in `read_key_event`'s `flags` arg: return `EAGAIN` instead
/// of blocking when the ring is empty. Mirrors `READ_STDIN_NONBLOCK`.
pub const READ_KEY_EVENT_NONBLOCK: usize = 1;

/// Sentinel for `read_key_event`'s `timeout_ms` arg: block until the
/// next event arrives. Programmed kernel-side as `wake_time =
/// usize::MAX` so the sleep_heap entry never pops; only the
/// wake_override path wakes us. Same value as the kernel-side
/// `READ_KEY_EVENT_INDEFINITE`.
pub const READ_KEY_EVENT_INDEFINITE: usize = usize::MAX;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_char() {
        let ev = KeyEvent::new(
            KeyCode::Char,
            Modifiers::CONTROL.union(Modifiers::SHIFT),
            KeyEventKind::Press,
        );
        // Manually splice in a codepoint — that's what the kernel does.
        let mut ev = ev;
        ev.code = KeyCode::encode_char('A');
        assert_eq!(ev.key_code(), Some(KeyCode::Char));
        assert_eq!(decoded_char(ev.code), Some('A'));
        assert!(ev.mods().contains(Modifiers::CONTROL));
        assert!(ev.mods().contains(Modifiers::SHIFT));
        assert!(!ev.mods().contains(Modifiers::ALT));
        assert_eq!(ev.event_kind(), Some(KeyEventKind::Press));
    }

    #[test]
    fn round_trip_function_key() {
        let mut ev = KeyEvent::new(KeyCode::F, Modifiers::NONE, KeyEventKind::Press);
        ev.code = KeyCode::encode_fn(7);
        assert_eq!(ev.key_code(), Some(KeyCode::F));
        assert_eq!(decoded_fn(ev.code), 7);
    }

    #[test]
    fn unknown_tag_decodes_to_none() {
        // Tag byte 0x80 is unallocated — old userland sees None
        // instead of a misclassified event.
        assert_eq!(KeyCode::decode(0x80), None);
    }

    #[test]
    fn modifier_truncation_drops_unknown_bits() {
        // Bit 31 set (some hypothetical future modifier) — old userland
        // discards it, doesn't poison matches against known bits.
        let m = Modifiers::from_bits_truncate(Modifiers::SHIFT.bits() | (1 << 31));
        assert!(m.contains(Modifiers::SHIFT));
        assert_eq!(m.bits(), Modifiers::SHIFT.bits());
    }

    #[test]
    fn keyevent_size_is_stable() {
        // 4 × u32 — change is wire-breaking.
        assert_eq!(core::mem::size_of::<KeyEvent>(), 16);
        assert_eq!(core::mem::align_of::<KeyEvent>(), 4);
    }
}
