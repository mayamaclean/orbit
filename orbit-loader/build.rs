fn main() {
    println!(
        "cargo::rerun-if-changed=../console/target/riscv64gc-unknown-none-elf/release/console"
    );
}
