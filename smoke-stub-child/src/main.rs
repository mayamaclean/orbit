#![no_std]
#![no_main]

use core::panic::PanicInfo;

use orbit_abi::user::exit;
use orbit_rt as _;

#[unsafe(no_mangle)]
pub extern "C" fn main() -> i32 {
    0
}

#[panic_handler]
fn panic_time(_p: &PanicInfo) -> ! {
    exit(isize::MIN)
}
