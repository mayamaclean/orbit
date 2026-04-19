#RUSTFLAGS="-Crelocation-model=pie -Ccode-model=medium -Cforce-frame-pointers=yes -Clink-arg=-pie -Clink-arg=-Bsymbolic -Clink-arg=-znotext -Clink-arg=--emit-relocs -Clink-arg=-Tmemory.x" \
RUSTFLAGS="-Crelocation-model=pie -Ccode-model=medium -Cforce-frame-pointers=yes -Clink-arg=-pie -Clink-arg=-Bsymbolic -Clink-arg=-znotext -Clink-arg=-Tmemory.x -Clink-arg=--no-dynamic-linker -Clink-arg=--pack-dyn-relocs=none -Clink-arg=--export-dynamic" \
    cargo build -Zbuild-std=core,alloc --config $(pwd)/.cargo/config.toml  --verbose --release --bin orbit

readelf -SW target/riscv64gc-unknown-none-elf/release/orbit
