use core::fmt::{self, Write};

use tracing::{Event, Id, Level, Metadata, Subscriber, field::Visit};

pub struct OrbitSubscriber {
    max_level: Level
}

impl OrbitSubscriber {
    pub const fn new(max_level: Level) -> Self {
        Self {
            max_level
        }
    }
}

impl Subscriber for OrbitSubscriber {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        *metadata.level() <= self.max_level
    }

    fn event(&self, event: &Event<'_>) {
        let mut visitor = OrbitVisitor(event.metadata().level().as_str());
        event.record(&mut visitor);
    }

    fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> Id { Id::from_u64(1) }
    fn record(&self, _span: &Id, _values: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}
    fn enter(&self, _span: &Id) {}
    fn exit(&self, _span: &Id) {}
}

struct OrbitVisitor(pub(super) &'static str);

impl Visit for OrbitVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn core::fmt::Debug) {
        if field.name() == "message" {
            emit(format_args!(
                "{}t {}: {:?}\n",
                riscv::register::time::read64(),
                self.0, value,
            ));
        } else {
            emit(format_args!(
                "{}: {}=\"{:?}\"\n",
                self.0, field.name(), value,
            ));
        }
    }
}

pub struct OrbitLogger;

impl log::Log for OrbitLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Trace
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            emit(format_args!(
                "{}t {}: {}\n",
                riscv::register::time::read64(),
                record.level(), record.args(),
            ));
        }
    }

    fn flush(&self) {}
}

/// Max bytes a single trace line occupies in the stack buffer. Lines
/// over this get truncated by the `fmt::Write` impl on `LineBuf`.
/// Also bounds the byte chunk we push to `k_gpu`'s ring.
const LINE_BUF_LEN: usize = 512;

/// Fixed-capacity UTF-8 buffer. Accepts `fmt::Write` and silently
/// drops any overflow past `LINE_BUF_LEN`.
struct LineBuf {
    bytes: [u8; LINE_BUF_LEN],
    len: usize,
}

impl LineBuf {
    const fn new() -> Self {
        Self { bytes: [0; LINE_BUF_LEN], len: 0 }
    }

    fn as_slice(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}

impl Write for LineBuf {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let remaining = LINE_BUF_LEN.saturating_sub(self.len);
        let n = remaining.min(s.len());
        self.bytes[self.len..self.len + n].copy_from_slice(&s.as_bytes()[..n]);
        self.len += n;
        Ok(())
    }
}

/// Format `args` once into a stack buffer, then emit to both the UART
/// (as-is) and — once `k_gpu` is live — the kernel scrollback. Done
/// this way so we don't double-format and so the scrollback sees
/// exactly the bytes the user would see on UART.
fn emit(args: fmt::Arguments<'_>) {
    let mut buf = LineBuf::new();
    let _ = buf.write_fmt(args);
    let bytes = buf.as_slice();

    // Serial is the always-available path. Uses the spinlock'd
    // println-style API; safe from any context that itself isn't
    // already holding the serial lock.
    unsafe {
        if let Ok(s) = core::str::from_utf8(bytes) {
            serial::print!("{}", s);
        }
    }

    // Push to the k_gpu ring if it's initialized. Runs lock-free;
    // dropped on ring-full (the UART path already caught the line).
    if crate::drivers::k_gpu::is_ready() {
        let _ = crate::drivers::k_gpu::push_chunk(
            crate::drivers::display::Source::Kernel,
            bytes,
        );
    }
}
