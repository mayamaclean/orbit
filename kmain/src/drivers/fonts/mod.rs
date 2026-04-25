//! Embedded bitmap fonts for the kernel framebuffer.
//!
//! Each submodule is a generated `[[u8; H]; 256]` table indexed by
//! Unicode codepoint < 256. Regenerate via `tools/bdf_to_rust.py`
//! if you swap sources.

pub mod terminus;
