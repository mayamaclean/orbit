use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};

fn main() {
    println!("hello from std on orbit!");

    let v: Vec<u32> = (0..10).collect();
    println!("vec sum = {}", v.iter().sum::<u32>());

    let now = std::time::Instant::now();
    let then = std::time::Instant::now();
    println!(
        "instant delta micros = {}",
        then.duration_since(now).as_micros()
    );

    // SystemTime via Goldfish RTC. Sanity check that we got a
    // post-2020 timestamp (anything before that means RTC didn't
    // wire). Year 2020 = 1577836800 secs since epoch.
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) if d.as_secs() > 1_577_836_800 => {
            println!("PASS: SystemTime::now epoch_secs={}", d.as_secs());
        }
        Ok(d) => {
            println!(
                "FAIL: SystemTime::now too early: epoch_secs={}",
                d.as_secs()
            );
        }
        Err(e) => println!("FAIL: SystemTime::now: {e}"),
    }

    // §13e — std::thread::spawn round trip.
    let counter = Arc::new(AtomicU32::new(0));
    let worker_counter = counter.clone();
    let handle = std::thread::spawn(move || {
        for i in 0..5 {
            worker_counter.fetch_add(1, Ordering::Relaxed);
            println!("worker tick {i}");
        }
    });
    handle.join().unwrap();
    println!("post-join counter = {}", counter.load(Ordering::Acquire));

    // §13e — Mutex round trip. Each worker takes the lock, bumps the
    // shared counter, releases. Final value should be N_THREADS *
    // BUMPS_PER_THREAD = 30.
    let m = Arc::new(Mutex::new(0u32));
    let workers: Vec<_> = (0..3)
        .map(|tid| {
            let m = m.clone();
            std::thread::spawn(move || {
                for _ in 0..10 {
                    let mut g = m.lock().unwrap();
                    *g += 1;
                    drop(g);
                    // tiny yield so the threads actually contend
                    std::thread::yield_now();
                }
                println!("mutex worker {tid} done");
            })
        })
        .collect();
    for w in workers {
        w.join().unwrap();
    }
    println!("post-mutex counter = {}", *m.lock().unwrap());

    // §13e — Condvar round trip. Producer flips a flag and signals;
    // consumer waits until the flag is true.
    let pair = Arc::new((Mutex::new(false), Condvar::new()));
    let pair2 = pair.clone();
    let consumer = std::thread::spawn(move || {
        let (lock, cvar) = &*pair2;
        let mut started = lock.lock().unwrap();
        while !*started {
            started = cvar.wait(started).unwrap();
        }
        println!("condvar consumer woke");
    });
    let (lock, cvar) = &*pair;
    {
        let mut started = lock.lock().unwrap();
        *started = true;
        cvar.notify_one();
        drop(started);
    }
    consumer.join().unwrap();
    println!("condvar round trip done");

    // §13e — RwLock smoke. One writer, two readers.
    let rw = Arc::new(std::sync::RwLock::new(0u32));
    let writer = {
        let rw = rw.clone();
        std::thread::spawn(move || {
            let mut g = rw.write().unwrap();
            *g = 99;
        })
    };
    writer.join().unwrap();
    let readers: Vec<_> = (0..2)
        .map(|i| {
            let rw = rw.clone();
            std::thread::spawn(move || {
                let g = rw.read().unwrap();
                println!("rwlock reader {i} sees {}", *g);
            })
        })
        .collect();
    for r in readers {
        r.join().unwrap();
    }
    println!("rwlock final = {}", *rw.read().unwrap());

    // §13e — mpsc round trip. Producer ships 5 messages; consumer
    // sums and reports. Channel internals layer on top of Mutex +
    // Condvar so this is end-to-end coverage of the parking shape.
    let (tx, rx) = std::sync::mpsc::channel::<u32>();
    let producer = std::thread::spawn(move || {
        for i in 1..=5 {
            tx.send(i).unwrap();
        }
    });
    let mut total = 0;
    for v in rx {
        total += v;
    }
    producer.join().unwrap();
    println!("mpsc total = {total}");

    // §13e — args + parallelism (read-only, no thread).
    let args: Vec<_> = std::env::args_os().collect();
    println!("argc = {}", args.len());
    for (i, a) in args.iter().enumerate() {
        println!("argv[{i}] = {}", a.to_string_lossy());
    }

    // §13e env smoke. Boot envp is kmain-installed PATH=/bin / HOME=/
    // / TERM=dumb, propagated by orbit-loader. Validates the std PAL
    // path (env/orbit.rs's OnceLock<RwLock<BTreeMap>>) round-trips
    // both reads and mutations — the latter only mutate the in-process
    // map (children would need explicit envp repack to inherit them).
    match std::env::var("PATH") {
        Ok(v) if v == "/bin" => println!("PASS: std::env PATH=/bin"),
        Ok(v) => println!("FAIL: std::env PATH got {v:?}"),
        Err(e) => println!("FAIL: std::env PATH: {e}"),
    }
    match std::env::var("HOME") {
        Ok(v) if v == "/" => println!("PASS: std::env HOME=/"),
        Ok(v) => println!("FAIL: std::env HOME got {v:?}"),
        Err(e) => println!("FAIL: std::env HOME: {e}"),
    }
    match std::env::var("TERM") {
        Ok(v) if v == "dumb" => println!("PASS: std::env TERM=dumb"),
        Ok(v) => println!("FAIL: std::env TERM got {v:?}"),
        Err(e) => println!("FAIL: std::env TERM: {e}"),
    }
    let n_vars = std::env::vars_os().count();
    if n_vars == 3 {
        println!("PASS: std::env vars_os count=3");
    }
    else {
        println!("FAIL: std::env vars_os count={n_vars} (want 3)");
    }
    // Mutations exist solely in the in-process map — confirm round trip.
    unsafe {
        std::env::set_var("FOO", "bar");
    }
    match std::env::var("FOO") {
        Ok(v) if v == "bar" => println!("PASS: std::env set_var/var round trip"),
        Ok(v) => println!("FAIL: std::env set_var/var got {v:?}"),
        Err(e) => println!("FAIL: std::env set_var/var: {e}"),
    }
    unsafe {
        std::env::remove_var("FOO");
    }
    match std::env::var("FOO") {
        Err(std::env::VarError::NotPresent) => println!("PASS: std::env remove_var clears entry"),
        Ok(v) => println!("FAIL: std::env remove_var still returns {v:?}"),
        Err(e) => println!("FAIL: std::env remove_var unexpected error: {e}"),
    }
    println!(
        "available_parallelism = {}",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0)
    );

    // §13e — HashMap. Backed by `hashmap_random_keys` which on orbit
    // pulls a stack+heap address pair (low-quality entropy, but
    // enough to bring HashMap up before a real RNG lands).
    use std::collections::HashMap;
    let mut h: HashMap<&'static str, u32> = HashMap::new();
    h.insert("alpha", 1);
    h.insert("beta", 2);
    h.insert("gamma", 3);
    let mut keys: Vec<_> = h.keys().copied().collect();
    keys.sort();
    let sum: u32 = h.values().sum();
    println!("hashmap keys = {keys:?}");
    println!("hashmap sum = {sum}");

    // §13e — String formatting + sort + iteration over a heap-allocated
    // slice. Catches any subtle alignment / unwind path that the
    // earlier tests didn't exercise.
    let mut nums: Vec<i32> = vec![5, 2, 8, 1, 9, 3];
    nums.sort();
    println!("sorted = {nums:?}");
    let s: String = nums
        .iter()
        .map(|n| n.to_string())
        .collect::<Vec<_>>()
        .join(",");
    println!("joined = {s}");

    // §13e — current thread identity. ThreadId's u64 accessor is
    // unstable behind `thread_id_value`; print Debug for now.
    let main_id = std::thread::current().id();
    println!("main thread id = {main_id:?}");

    // §13e — std::fs smoke. Runs before the network section so it
    // doesn't depend on DHCP / TCP listener completion.
    {
        use std::fs;
        use std::io::Read as _;

        match fs::metadata("/README") {
            Ok(md) => {
                if md.is_file() && md.len() == 217 {
                    println!("PASS: std::fs::metadata /README is_file size=217");
                }
                else {
                    println!(
                        "FAIL: std::fs::metadata /README is_file={} size={}",
                        md.is_file(),
                        md.len(),
                    );
                }
                match md.modified() {
                    Ok(t) => match t.duration_since(std::time::UNIX_EPOCH) {
                        Ok(d) => println!(
                            "PASS: std::fs::metadata /README modified epoch_secs={}",
                            d.as_secs(),
                        ),
                        Err(e) => println!("FAIL: /README modified pre-epoch: {e}"),
                    },
                    Err(e) => println!("FAIL: /README modified: {e}"),
                }
            }
            Err(e) => println!("FAIL: std::fs::metadata /README: {e}"),
        }

        match fs::File::open("/bin/hello.txt").and_then(|mut f| {
            let mut s = String::new();
            f.read_to_string(&mut s).map(|_| s)
        }) {
            Ok(s) if s == "hello from /bin/hello.txt\n" => {
                println!("PASS: std::fs::File::open /bin/hello.txt read_to_string matches");
            }
            Ok(s) => println!("FAIL: std::fs read_to_string got {s:?}"),
            Err(e) => println!("FAIL: std::fs::File::open /bin/hello.txt: {e}"),
        }

        match fs::read_dir("/") {
            Ok(rd) => {
                let mut names: Vec<String> = rd
                    .filter_map(|r| r.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
                    .collect();
                names.sort();
                if names == ["README", "bin"] {
                    println!("PASS: std::fs::read_dir / yields [README, bin]");
                }
                else {
                    println!("FAIL: std::fs::read_dir / yields {names:?}");
                }
            }
            Err(e) => println!("FAIL: std::fs::read_dir /: {e}"),
        }

        match fs::read_dir("/bin") {
            Ok(rd) => {
                let entries: Vec<(String, bool)> = rd
                    .filter_map(|r| r.ok())
                    .map(|e| {
                        let nm = e.file_name().to_string_lossy().into_owned();
                        let is_file = e.file_type().map(|ft| ft.is_file()).unwrap_or(false);
                        (nm, is_file)
                    })
                    .collect();
                // Disk image grows over time as new in-tree binaries
                // pick up the [package.metadata.disk] marker. Assert
                // the canonical entries are present + are regular
                // files rather than pinning the full list.
                let has_hello = entries.iter().any(|(n, f)| n == "hello" && *f);
                let has_hello_txt = entries.iter().any(|(n, f)| n == "hello.txt" && *f);
                if has_hello && has_hello_txt {
                    println!("PASS: std::fs::read_dir /bin contains hello(file) + hello.txt(file)");
                }
                else {
                    println!("FAIL: std::fs::read_dir /bin missing canonical entries: {entries:?}");
                }
            }
            Err(e) => println!("FAIL: std::fs::read_dir /bin: {e}"),
        }

        match fs::metadata("/does-not-exist") {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                println!("PASS: std::fs::metadata missing path -> NotFound");
            }
            Ok(_) => println!("FAIL: std::fs::metadata /does-not-exist returned Ok"),
            Err(e) => println!("FAIL: std::fs::metadata /does-not-exist unexpected err: {e}"),
        }

        // §13e File::seek round trip. /bin/hello.txt is 26 bytes
        // ("hello from /bin/hello.txt\n"). SeekFrom::Start past the
        // first word, SeekFrom::End(-5), and SeekFrom::Current after
        // a partial read each verify the kernel-side cursor and the
        // post-seek read returns the right slice.
        use std::io::{Seek, SeekFrom};
        match fs::File::open("/bin/hello.txt") {
            Ok(mut f) => {
                // Start: seek 6 bytes in, read 4 → "from"
                match f.seek(SeekFrom::Start(6)) {
                    Ok(p) if p == 6 => {}
                    Ok(p) => println!("FAIL: seek(Start(6)) returned {p}"),
                    Err(e) => println!("FAIL: seek(Start(6)): {e}"),
                }
                let mut buf = [0u8; 4];
                match f.read(&mut buf) {
                    Ok(4) if &buf == b"from" => {
                        println!("PASS: File::seek(Start(6)) + read 4 yields \"from\"");
                    }
                    Ok(n) => println!("FAIL: post-seek read got n={n} buf={buf:?}"),
                    Err(e) => println!("FAIL: post-seek read: {e}"),
                }
                // Current: from offset 10 (after read), seek -4 to land
                // back at 6, then re-read "from".
                match f.seek(SeekFrom::Current(-4)) {
                    Ok(p) if p == 6 => println!("PASS: File::seek(Current(-4)) returns 6"),
                    Ok(p) => println!("FAIL: seek(Current(-4)) returned {p}"),
                    Err(e) => println!("FAIL: seek(Current(-4)): {e}"),
                }
                // End: -4 from the end of a 26-byte file = offset 22,
                // read 4 → "txt\n".
                match f.seek(SeekFrom::End(-4)) {
                    Ok(p) if p == 22 => {}
                    Ok(p) => println!("FAIL: seek(End(-4)) returned {p}"),
                    Err(e) => println!("FAIL: seek(End(-4)): {e}"),
                }
                let mut tail = [0u8; 4];
                match f.read(&mut tail) {
                    Ok(4) if &tail == b"txt\n" => {
                        println!("PASS: File::seek(End(-4)) + read 4 yields \"txt\\n\"");
                    }
                    Ok(n) => println!("FAIL: tail read got n={n} buf={tail:?}"),
                    Err(e) => println!("FAIL: tail read: {e}"),
                }
                // Negative resolved offset is rejected.
                match f
                    .seek(SeekFrom::Start(0))
                    .and_then(|_| f.seek(SeekFrom::Current(-1)))
                {
                    Err(e) if e.kind() == std::io::ErrorKind::InvalidInput => {
                        println!("PASS: File::seek negative resolved offset -> InvalidInput");
                    }
                    Ok(p) => println!("FAIL: seek into negative returned {p}"),
                    Err(e) => println!("FAIL: seek negative unexpected: {e}"),
                }
            }
            Err(e) => println!("FAIL: open /bin/hello.txt for seek tests: {e}"),
        }

        // §13e File::metadata round trip via fstat-by-fd. The same
        // stat data is reachable via fs::metadata(path) but file
        // handles that have outlived their open path (e.g. piped
        // through a function that lost the path) need fstat.
        match fs::File::open("/bin/hello.txt").and_then(|f| f.metadata()) {
            Ok(md) if md.is_file() && md.len() == 26 => {
                println!("PASS: File::metadata fstat /bin/hello.txt is_file size=26");
            }
            Ok(md) => println!(
                "FAIL: File::metadata is_file={} size={}",
                md.is_file(),
                md.len()
            ),
            Err(e) => println!("FAIL: File::metadata: {e}"),
        }
        // fstat on a directory fd should also work and report is_dir.
        match fs::File::open("/bin").and_then(|f| f.metadata()) {
            Ok(md) if md.is_dir() => {
                println!("PASS: File::metadata fstat /bin reports is_dir");
            }
            Ok(md) => println!(
                "FAIL: File::metadata /bin: is_dir={} is_file={}",
                md.is_dir(),
                md.is_file()
            ),
            Err(e) => println!("FAIL: File::metadata /bin: {e}"),
        }
    }

    // §13e — cwd round trip. Boot cwd is `/` (init process default).
    // chdir to /bin → relative "hello.txt" should now resolve to
    // /bin/hello.txt. chdir back to / so the rest of the smoke isn't
    // perturbed.
    {
        use std::env;
        use std::fs;
        match env::current_dir() {
            Ok(p) if p == std::path::PathBuf::from("/") => {
                println!("PASS: std::env::current_dir boot cwd is /");
            }
            Ok(p) => println!("FAIL: std::env::current_dir got {p:?}"),
            Err(e) => println!("FAIL: std::env::current_dir: {e}"),
        }
        if let Err(e) = env::set_current_dir("/bin") {
            println!("FAIL: std::env::set_current_dir(/bin): {e}");
        }
        else {
            match env::current_dir() {
                Ok(p) if p == std::path::PathBuf::from("/bin") => {
                    println!("PASS: std::env::set_current_dir(/bin) round trip");
                }
                Ok(p) => println!("FAIL: post-chdir current_dir got {p:?}"),
                Err(e) => println!("FAIL: post-chdir current_dir: {e}"),
            }
            // Relative-path read should resolve against /bin.
            match fs::metadata("hello.txt") {
                Ok(md) if md.is_file() && md.len() == 26 => {
                    println!("PASS: std::fs::metadata relative `hello.txt` resolves under cwd");
                }
                Ok(md) => println!(
                    "FAIL: relative metadata is_file={} len={}",
                    md.is_file(),
                    md.len()
                ),
                Err(e) => println!("FAIL: relative metadata: {e}"),
            }
            // Restore cwd for the rest of the smoke.
            let _ = env::set_current_dir("/");
        }
        // Negative path: chdir to a non-existent dir should fail.
        match env::set_current_dir("/does-not-exist") {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                println!("PASS: std::env::set_current_dir missing -> NotFound");
            }
            Ok(()) => println!("FAIL: set_current_dir /does-not-exist returned Ok"),
            Err(e) => println!("FAIL: set_current_dir missing unexpected: {e}"),
        }
        // Negative path: chdir to a regular file (not a directory).
        match env::set_current_dir("/README") {
            Err(e) if e.kind() == std::io::ErrorKind::NotADirectory => {
                println!("PASS: std::env::set_current_dir on file -> NotADirectory");
            }
            Ok(()) => println!("FAIL: set_current_dir /README returned Ok"),
            Err(e) => println!("FAIL: set_current_dir /README unexpected: {e}"),
        }
    }

    // §13e — std::process::Command smoke. Spawn /bin/hello (a tiny
    // arg-printing binary on the disk image), wait for it, and check
    // the exit code matches the value its main() returns. Exercises
    // the full Phase-7 path: fs read of program → argv pack → envp
    // pack from current std::env table → create_process_with_argv_envp
    // → wait_pid round trip.
    {
        use std::process::Command;
        // /bin/hello returns 42 from main() and prints its argv via
        // serialln. We pass two extra args so the exec path is
        // exercised end-to-end, then read the exit code back.
        let mut cmd = Command::new("/bin/hello");
        cmd.arg("world");
        cmd.arg("peace");
        match cmd.spawn() {
            Ok(mut child) => {
                let pid = child.id();
                println!("PASS: std::process::Command spawn /bin/hello pid={pid}");
                match child.wait() {
                    Ok(status) => {
                        if status.code() == Some(42) {
                            println!("PASS: std::process::Command wait /bin/hello status=42");
                        }
                        else {
                            println!("FAIL: std::process::Command wait got {status}");
                        }
                    }
                    Err(e) => println!("FAIL: std::process::Command wait: {e}"),
                }
            }
            Err(e) => println!("FAIL: std::process::Command spawn /bin/hello: {e}"),
        }

        // Negative path — missing program surfaces as NotFound from
        // the std::fs::File::open inside spawn().
        match Command::new("/bin/does-not-exist").spawn() {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                println!("PASS: std::process::Command spawn missing -> NotFound");
            }
            Ok(_) => println!("FAIL: std::process::Command spawn missing returned Ok"),
            Err(e) => println!("FAIL: std::process::Command spawn missing unexpected: {e}"),
        }

        // Command::current_dir override. v1 implementation does
        // parent chdir + spawn + restore (the EX path doesn't carry
        // cwd). The child should observe the override; the parent's
        // cwd should be restored afterward.
        let parent_cwd_before = std::env::current_dir().ok();
        let mut cmd = Command::new("/bin/hello");
        cmd.current_dir("/bin");
        match cmd.spawn() {
            Ok(mut child) => match child.wait() {
                Ok(status) if status.code() == Some(42) => {
                    println!("PASS: Command::current_dir(/bin) spawn ok status=42");
                }
                Ok(status) => println!("FAIL: cwd-spawn wait got {status}"),
                Err(e) => println!("FAIL: cwd-spawn wait: {e}"),
            },
            Err(e) => println!("FAIL: Command::current_dir(/bin) spawn: {e}"),
        }
        if std::env::current_dir().ok() == parent_cwd_before {
            println!("PASS: parent cwd restored after Command::current_dir spawn");
        }
        else {
            println!(
                "FAIL: parent cwd not restored — was {:?}, now {:?}",
                parent_cwd_before,
                std::env::current_dir().ok(),
            );
        }
    }

    // §13e — std::net::TcpStream::connect over the kernel's NetChannel
    // primitive. Connect to QEMU's user-net gateway (which maps to host
    // loopback) on a port the smoke harness has nc(1) listening on.
    // Wait for DHCP first — NetChannel::open eats the DHCP-not-ready
    // window in `state >= 2` polling, but a leading sleep is cheaper
    // than 100ms+ of poll churn.
    println!("waiting for net up...");
    std::thread::sleep(std::time::Duration::from_secs(8));
    use std::io::{Read, Write};
    use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpStream};
    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::new(192, 168, 76, 2), 65535));
    match TcpStream::connect(addr) {
        Ok(mut s) => {
            println!("tcp connected to {addr}");
            if let Err(e) = s.write_all(b"hello-std over TcpStream!\n") {
                println!("tcp write failed: {e}");
            }
            let mut buf = [0u8; 64];
            match s.read(&mut buf) {
                Ok(n) => {
                    let txt = String::from_utf8_lossy(&buf[..n]);
                    println!("tcp got {n} bytes: {txt:?}");
                }
                Err(e) => println!("tcp read failed: {e}"),
            }
        }
        Err(e) => println!("tcp connect failed: {e}"),
    }

    // §13f — `os::fd` lift smoke. Validates the std PAL changes from
    // Milestone C: File / TcpStream / TcpListener carry RawFd-shaped
    // handles, expose `AsRawFd` / `AsFd` through the orbit-side
    // `os::fd` plumbing, and `set_nonblocking(true)` is a real
    // userspace operation (no kernel syscall, just an atomic flag).
    {
        use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpListener};
        use std::os::fd::{AsFd, AsRawFd};

        // File: open, check fd > 2 (stdio occupies 0/1/2), check
        // AsFd returns the same raw value.
        match std::fs::File::open("/bin/hello.txt") {
            Ok(f) => {
                let raw = f.as_raw_fd();
                let borrowed = f.as_fd().as_raw_fd();
                if raw > 2 && borrowed == raw {
                    println!("PASS: os::fd File::as_raw_fd={raw} matches AsFd");
                }
                else {
                    println!("FAIL: os::fd File raw={raw} borrowed={borrowed} (stdio is 0/1/2)");
                }
            }
            Err(e) => println!("FAIL: os::fd File::open /bin/hello.txt: {e}"),
        }

        // TcpListener: bind on a separate port, validate as_raw_fd
        // returns a positive non-stdio fd, then flip set_nonblocking
        // and observe `accept()` returns WouldBlock without parking.
        match TcpListener::bind(SocketAddr::V4(SocketAddrV4::new(
            Ipv4Addr::new(0, 0, 0, 0),
            7779,
        ))) {
            Ok(listener) => {
                let raw = listener.as_raw_fd();
                if raw > 2 {
                    println!("PASS: os::fd TcpListener::as_raw_fd={raw}");
                }
                else {
                    println!("FAIL: os::fd TcpListener raw={raw}");
                }

                match listener.set_nonblocking(true) {
                    Ok(()) => println!("PASS: TcpListener::set_nonblocking(true)"),
                    Err(e) => println!("FAIL: set_nonblocking(true): {e}"),
                }

                // Non-blocking accept on a freshly bound listener
                // should return WouldBlock immediately — no peer has
                // connected, and the kernel-side channel state is
                // `IDLE` / `IN_FLIGHT`, not `ACTIVE`.
                match listener.accept() {
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        println!("PASS: nonblocking accept returns WouldBlock");
                    }
                    Err(e) => println!("FAIL: nonblocking accept error: {e}"),
                    Ok(_) => println!("FAIL: nonblocking accept somehow returned a peer"),
                }
            }
            Err(e) => println!("FAIL: os::fd TcpListener::bind :7779: {e}"),
        }
    }

    // §13g — mio compile + minimal API smoke. Validates the orbit
    // mio fork actually links (the major scaffolding payoff for
    // Milestone E) and that `Poll::new` + `Waker::new` + `Registry`
    // operations resolve at runtime against the orbit sys backend.
    // The selector scan loop is still a stub at this point, so we
    // don't exercise `poll()` for real events — just the
    // construction + register/deregister surface that tokio leans
    // on at startup.
    //
    // Markers go through `orbit_abi::serialln!` so they land on the
    // QEMU `-serial` channel (the only place we can read them
    // headlessly); plain `println!` routes to the framebuffer.
    {
        use mio::{Events, Poll, Token, Waker};
        use orbit_abi::serialln;
        use std::sync::Arc;

        let Ok(mut poll) = Poll::new()
        else {
            serialln!("FAIL: mio::Poll::new");
            return;
        };
        serialln!("PASS: mio::Poll::new");

        let Ok(waker) = Waker::new(poll.registry(), Token(42))
        else {
            serialln!("FAIL: mio::Waker::new");
            return;
        };
        serialln!("PASS: mio::Waker::new");

        // Zero-timeout poll before any wake: count is 0, no events.
        let mut events = Events::with_capacity(8);
        match poll.poll(&mut events, Some(core::time::Duration::ZERO)) {
            Ok(()) if events.is_empty() => {
                serialln!("PASS: mio::Poll::poll zero-timeout pre-wake returns empty");
            }
            Ok(()) => serialln!(
                "FAIL: mio::Poll::poll pre-wake returned {} events",
                events.iter().count()
            ),
            Err(e) => serialln!("FAIL: mio::Poll::poll pre-wake: {e}"),
        }

        // Fire wake(): bumps the EventFd's count in shared memory.
        // The reactor isn't parked so no `wake_tid` syscall fires.
        let _ = Arc::new(()); // keep `Arc` import used
        match waker.wake() {
            Ok(()) => serialln!("PASS: mio::Waker::wake"),
            Err(e) => serialln!("FAIL: Waker::wake: {e}"),
        }

        // Poll after wake: the scan should observe count > 0 and
        // emit an event with Token(42)/readable.
        events.clear();
        match poll.poll(&mut events, Some(core::time::Duration::from_millis(100))) {
            Ok(()) => {
                let mut saw = false;
                for ev in events.iter() {
                    if ev.token() == Token(42) && ev.is_readable() {
                        saw = true;
                    }
                }
                if saw {
                    serialln!("PASS: mio scan observed Waker event after wake()");
                }
                else {
                    serialln!(
                        "FAIL: mio scan returned {} events but no Token(42)/readable",
                        events.iter().count(),
                    );
                }
            }
            Err(e) => serialln!("FAIL: poll after wake: {e}"),
        }

        // Explicit deregister + drop ordering: mio's contract is
        // that consumers deregister sources before close. The Waker
        // owns its EventFd so dropping the Waker while it's still
        // registered would leave a stale region pointer in the
        // Selector. Drop Poll first (which drops Registry → drops
        // Selector → invalidates the source set) before the Waker.
        drop(poll);
        drop(waker);
    }

    // §13h — ch_inspect smoke. Validates kind tags + region metadata
    // for every Handle variant the kernel can serve today.
    {
        use orbit_abi::handle::{ChInfo, HandleKind};
        use orbit_abi::layout::UPROC_SHARED_BASE;
        use orbit_abi::serialln;
        use orbit_abi::user::{ch_inspect, close_handle, eventfd};
        use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpListener};
        use std::os::fd::AsRawFd;

        let mut info = ChInfo::default();

        // Stdio kinds — fds 0/1/2 always present after process spawn.
        for (fd, want, label) in [
            (0u32, HandleKind::Stdin as u8, "Stdin"),
            (1u32, HandleKind::Stdout as u8, "Stdout"),
            (2u32, HandleKind::Stderr as u8, "Stderr"),
        ] {
            info = ChInfo::default();
            match ch_inspect(fd, &mut info) {
                Ok(()) if info.kind == want && info.region_va == 0 && info.region_size == 0 => {
                    serialln!("PASS: ch_inspect fd={fd} kind={label}");
                }
                Ok(()) => serialln!(
                    "FAIL: ch_inspect fd={fd} kind={} (wanted {label}={want}) va=0x{:X} size={}",
                    info.kind,
                    info.region_va,
                    info.region_size,
                ),
                Err(e) => serialln!("FAIL: ch_inspect fd={fd} errno={}", e.0),
            }
        }

        // EventFd kind + region echo. The returned va should match
        // both what eventfd() handed us *and* what's mapped into
        // shared memory at that VA.
        let hint = (UPROC_SHARED_BASE as usize) + 0x10_0000;
        match eventfd(hint, 7, orbit_abi::event_fd::EFD_NONBLOCK) {
            Ok((va, fd)) => {
                info = ChInfo::default();
                match ch_inspect(fd, &mut info) {
                    Ok(())
                        if info.kind == HandleKind::EventFd as u8
                            && info.region_va as usize == va
                            && info.region_size == 4096
                            && (info.flags & orbit_abi::event_fd::EFD_NONBLOCK) != 0 =>
                    {
                        serialln!(
                            "PASS: ch_inspect EventFd fd={fd} va=0x{:X} size={} flags=0x{:X}",
                            info.region_va,
                            info.region_size,
                            info.flags,
                        );
                    }
                    Ok(()) => serialln!(
                        "FAIL: ch_inspect EventFd fd={fd} kind={} va=0x{:X}/0x{:X} size={} flags=0x{:X}",
                        info.kind,
                        info.region_va,
                        va as u64,
                        info.region_size,
                        info.flags,
                    ),
                    Err(e) => serialln!("FAIL: ch_inspect EventFd errno={}", e.0),
                }
                let _ = close_handle(fd);
            }
            Err(e) => serialln!("FAIL: eventfd setup for ch_inspect errno={}", e.0),
        }

        // NetChannel kind. Bind a listener on a free port; the
        // resulting NetChannel handle should ch_inspect to kind
        // = NetChannel with a non-zero region_va in the shared range.
        match TcpListener::bind(SocketAddr::V4(SocketAddrV4::new(
            Ipv4Addr::new(0, 0, 0, 0),
            7791,
        ))) {
            Ok(listener) => {
                let fd = listener.as_raw_fd() as u32;
                info = ChInfo::default();
                match ch_inspect(fd, &mut info) {
                    Ok(())
                        if info.kind == HandleKind::NetChannel as u8
                            && info.region_va != 0
                            && info.region_size > 0 =>
                    {
                        serialln!(
                            "PASS: ch_inspect NetChannel fd={fd} va=0x{:X} size={} state={}",
                            info.region_va,
                            info.region_size,
                            info.state,
                        );
                    }
                    Ok(()) => serialln!(
                        "FAIL: ch_inspect NetChannel fd={fd} kind={} va=0x{:X} size={}",
                        info.kind,
                        info.region_va,
                        info.region_size,
                    ),
                    Err(e) => serialln!("FAIL: ch_inspect NetChannel errno={}", e.0),
                }
            }
            Err(e) => serialln!("FAIL: TcpListener bind for ch_inspect: {e}"),
        }

        // EBADF on a fd that was never allocated. 100_000 is way past
        // any plausible slot in a fresh process.
        info = ChInfo::default();
        match ch_inspect(100_000, &mut info) {
            Err(e) if e.0 == orbit_abi::errno::EBADF => {
                serialln!("PASS: ch_inspect bogus fd returns EBADF");
            }
            Ok(()) => serialln!("FAIL: ch_inspect bogus fd unexpectedly succeeded"),
            Err(e) => serialln!("FAIL: ch_inspect bogus fd errno={} (wanted EBADF)", e.0),
        }
    }

    // §13i — Raw-fd round-trip smoke. Exercises the new orbit
    // `FromRawFd` / `IntoRawFd` impls on std::net::TcpListener: a
    // bound listener's fd is extracted, then reconstituted into a
    // fresh wrapper, and the resulting object is asserted to share
    // the same kernel handle / region as the original.
    {
        use orbit_abi::handle::ChInfo;
        use orbit_abi::serialln;
        use orbit_abi::user::ch_inspect;
        use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpListener};
        use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};

        match TcpListener::bind(SocketAddr::V4(SocketAddrV4::new(
            Ipv4Addr::new(0, 0, 0, 0),
            7792,
        ))) {
            Ok(listener) => {
                let original_fd = listener.as_raw_fd();
                let mut info_before = ChInfo::default();
                let _ = ch_inspect(original_fd as u32, &mut info_before);

                // IntoRawFd → mem::forget the wrapper, keep the kernel
                // handle alive.
                let raw = listener.into_raw_fd();
                if raw == original_fd {
                    serialln!("PASS: TcpListener::into_raw_fd preserves fd value");
                }
                else {
                    serialln!("FAIL: TcpListener::into_raw_fd raw={raw} vs original={original_fd}",);
                }

                // FromRawFd → rebuilds the wrapper via ch_inspect.
                let rebuilt = unsafe { TcpListener::from_raw_fd(raw) };
                let rebuilt_fd = rebuilt.as_raw_fd();
                let mut info_after = ChInfo::default();
                let _ = ch_inspect(rebuilt_fd as u32, &mut info_after);

                if rebuilt_fd == original_fd
                    && info_after.region_va == info_before.region_va
                    && info_after.region_size == info_before.region_size
                {
                    serialln!(
                        "PASS: TcpListener::from_raw_fd rehydrated va=0x{:X} size={}",
                        info_after.region_va,
                        info_after.region_size,
                    );
                }
                else {
                    serialln!(
                        "FAIL: rehydration drift fd={}/{} va=0x{:X}/0x{:X} size={}/{}",
                        rebuilt_fd,
                        original_fd,
                        info_after.region_va,
                        info_before.region_va,
                        info_after.region_size,
                        info_before.region_size,
                    );
                }

                // OwnedFd round-trip. Going OwnedFd → TcpListener →
                // OwnedFd should preserve the fd through both
                // conversions.
                let owned: OwnedFd = rebuilt.into();
                let owned_fd = owned.as_raw_fd();
                let again: TcpListener = owned.into();
                if again.as_raw_fd() == owned_fd {
                    serialln!("PASS: OwnedFd ↔ TcpListener round-trip preserves fd");
                }
                else {
                    serialln!(
                        "FAIL: OwnedFd round-trip drift {} → {}",
                        owned_fd,
                        again.as_raw_fd(),
                    );
                }
                drop(again);
            }
            Err(e) => serialln!("FAIL: TcpListener bind for raw-fd smoke: {e}"),
        }
    }

    // §13j — Cross-thread wake_tid doorbell. §13g exercises the
    // shared-mem `count.fetch_add` path with a single thread (wake
    // fires before park), so it never engages the `wake_tid` →
    // `WakeEvent::Tid` → manager-driven unpark chain. Here the
    // parent parks in `poll.poll(events, 5s)` and a child thread
    // delivers `waker.wake()` 50ms later. Pass criteria:
    //  - the parent unparks (Token(99)/readable event observed)
    //  - elapsed wall-clock < 500ms (anything close to 5s means
    //    the doorbell didn't fire and we sat through the timeout)
    {
        use mio::{Events, Poll, Token, Waker};
        use orbit_abi::serialln;
        use std::sync::Arc;
        use std::time::{Duration, Instant};

        let Ok(mut poll) = Poll::new()
        else {
            serialln!("FAIL: §13j Poll::new");
            return;
        };
        let Ok(waker) = Waker::new(poll.registry(), Token(99))
        else {
            serialln!("FAIL: §13j Waker::new");
            return;
        };
        let waker = Arc::new(waker);

        let waker_for_child = Arc::clone(&waker);
        let child = std::thread::spawn(move || {
            // Park budget = 50ms; the parent's poll timeout is
            // 5s so any latency over ~100ms means doorbell missed.
            std::thread::sleep(Duration::from_millis(50));
            waker_for_child.wake()
        });

        let mut events = Events::with_capacity(8);
        let started = Instant::now();
        let result = poll.poll(&mut events, Some(Duration::from_secs(5)));
        let elapsed = started.elapsed();

        match result {
            Ok(()) => {
                let mut saw = false;
                for ev in events.iter() {
                    if ev.token() == Token(99) && ev.is_readable() {
                        saw = true;
                    }
                }
                if saw && elapsed < Duration::from_millis(500) {
                    serialln!(
                        "PASS: cross-thread wake_tid unparked reactor in {}ms",
                        elapsed.as_millis(),
                    );
                }
                else if saw {
                    serialln!(
                        "FAIL: reactor unparked but took {}ms (timeout fire, not doorbell)",
                        elapsed.as_millis(),
                    );
                }
                else {
                    serialln!(
                        "FAIL: §13j poll returned {} events but no Token(99)/readable",
                        events.iter().count(),
                    );
                }
            }
            Err(e) => serialln!("FAIL: §13j poll: {e}"),
        }

        // Child returns Result<(), io::Error> — surface a failed
        // wake call so it can't be silently swallowed.
        match child.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => serialln!("FAIL: §13j child waker.wake() returned {e}"),
            Err(_) => serialln!("FAIL: §13j child thread panicked"),
        }

        drop(poll);
        // `waker` is the last Arc clone now; drops the EventFd.
        drop(waker);
    }

    // §13e — TcpListener round-trip. Bind to port 7778, accept one
    // peer, echo what they send. The host driver sends a single
    // line and disconnects.
    use std::net::TcpListener;
    match TcpListener::bind(SocketAddr::V4(SocketAddrV4::new(
        Ipv4Addr::new(0, 0, 0, 0),
        7778,
    ))) {
        Ok(listener) => {
            println!("listener bound on :7778");
            match listener.accept() {
                Ok((mut peer, peer_addr)) => {
                    println!("accepted peer {peer_addr}");
                    let mut req = [0u8; 128];
                    match peer.read(&mut req) {
                        Ok(n) => {
                            let txt = String::from_utf8_lossy(&req[..n]);
                            println!("listener got {n} bytes: {txt:?}");
                            let _ = peer.write_all(b"echo-back from listener\n");
                        }
                        Err(e) => println!("listener read failed: {e}"),
                    }
                }
                Err(e) => println!("accept failed: {e}"),
            }
        }
        Err(e) => println!("listener bind failed: {e}"),
    }

    // FIXME: `std::process::exit(0)` here faults inside std's
    // at-exit cleanup (cause=2 stval=0 — null fn call, likely an
    // unregistered hook). Plain return-from-main works because the
    // PAL's `_start` calls `abi::exit(code)` directly without
    // walking the cleanup chain. Investigate the cleanup-hook
    // dispatch path before we promote `std::process::exit` to
    // canonical.
    println!("hello-std done; returning cleanly");
}
