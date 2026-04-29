use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Put the linker script somewhere the linker can find it
    fs::write(out_dir.join("memory.x"), include_bytes!("memory.x")).unwrap();
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rerun-if-changed=memory.x");
    println!("cargo::rerun-if-changed=../umode/target/riscv64gc-unknown-none-elf/release/umode");
    println!("cargo::rerun-if-changed=../orbit-loader/target/riscv64gc-unknown-none-elf/release/orbit-loader");
    println!("cargo::rerun-if-changed=../console/target/riscv64gc-unknown-none-elf/release/console");
}
