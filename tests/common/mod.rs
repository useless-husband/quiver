//! Helpers for tests that run the real `quiver` binary.
#![allow(dead_code)]

use quiver::client::{Response, request};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

pub fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("quiver-it-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A `quiver serve` child process on an ephemeral port.
pub struct Server {
    pub child: Child,
    pub addr: String,
    pub log: PathBuf,
}

impl Server {
    pub fn start(data: &Path, extra: &[&str]) -> Server {
        let port_file = data.with_extension(format!("port-{}", rand_suffix()));
        let _ = std::fs::remove_file(&port_file);
        let log = data.with_extension("log");
        let logf = std::fs::OpenOptions::new().create(true).append(true).open(&log).unwrap();
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_quiver"));
        cmd.arg("serve")
            .arg("--data")
            .arg(data)
            .args(["--listen", "127.0.0.1:0", "--threads", "8"])
            .arg("--port-file")
            .arg(&port_file)
            .args(extra)
            .stdout(Stdio::null())
            .stderr(logf);
        // Owned by `Server` from the start, so its Drop kills and reaps the
        // child even if start-up times out below.
        let mut server = Server { child: cmd.spawn().expect("spawn quiver serve"), addr: String::new(), log };
        let t0 = Instant::now();
        loop {
            if let Ok(s) = std::fs::read_to_string(&port_file)
                && !s.is_empty()
            {
                let _ = std::fs::remove_file(&port_file);
                server.addr = s.trim().to_string();
                return server;
            }
            if t0.elapsed() > Duration::from_secs(60) {
                panic!("server did not start; log:\n{}", std::fs::read_to_string(&server.log).unwrap_or_default());
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    pub fn pid(&self) -> i32 {
        self.child.id() as i32
    }

    pub fn call(&self, method: &str, path: &str, body: Option<Value>) -> Response {
        request(&self.addr, method, path, body.as_ref()).unwrap_or_else(|e| panic!("{method} {path}: {e}"))
    }

    pub fn ok(&self, method: &str, path: &str, body: Option<Value>) -> Value {
        let r = self.call(method, path, body);
        assert_eq!(r.status, 200, "{method} {path}: {}", r.body);
        r.body
    }

    /// SIGKILL by PID: no shutdown code runs.
    pub fn kill9(mut self) {
        // SAFETY: plain kill(2) on our own child.
        unsafe { libc::kill(self.pid(), libc::SIGKILL) };
        let _ = self.child.wait();
    }

    /// SIGTERM by PID and wait for the graceful shutdown.
    pub fn terminate(mut self) {
        unsafe { libc::kill(self.pid(), libc::SIGTERM) };
        let _ = self.child.wait();
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Ok(None) = self.child.try_wait() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn rand_suffix() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    N.fetch_add(1, Ordering::Relaxed)
}
