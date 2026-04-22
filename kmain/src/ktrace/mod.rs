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
            serial::println!("{}: {:?}", self.0, value);
        } else {
            serial::println!("{}: {}=\"{:?}\"", self.0, field.name(), value);
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
            serial::println!("{}: {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}
