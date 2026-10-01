//! A minimal blocking HTTP/1.1 JSON client for the quiver API, used by the
//! CLI and by the integration tests (one request per connection).

use serde_json::Value;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

#[derive(Debug)]
pub struct Response {
    pub status: u16,
    pub body: Value,
}

/// Accepts `http://host:port`, `host:port` or `:port`.
pub fn parse_addr(url: &str) -> String {
    let s = url.trim_start_matches("http://");
    let s = s.split('/').next().unwrap_or(s);
    if let Some(p) = s.strip_prefix(':') { format!("127.0.0.1:{p}") } else { s.to_string() }
}

/// Sends one request and parses the JSON response.
pub fn request(addr: &str, method: &str, path: &str, body: Option<&Value>) -> std::io::Result<Response> {
    let mut s = TcpStream::connect(addr)?;
    s.set_read_timeout(Some(Duration::from_secs(300)))?;
    let payload = body.map(|b| serde_json::to_vec(b).unwrap()).unwrap_or_default();
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        payload.len()
    );
    s.write_all(head.as_bytes())?;
    s.write_all(&payload)?;
    let mut raw = Vec::new();
    s.read_to_end(&mut raw)?;
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "no HTTP header terminator"))?;
    let head = String::from_utf8_lossy(&raw[..split]).into_owned();
    let status = head
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse().ok())
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("bad status line: {head}")))?;
    let mut body = raw[split + 4..].to_vec();
    if head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
        body = dechunk(&body)?;
    }
    let body = if body.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&body)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("response is not JSON: {e}")))?
    };
    Ok(Response { status, body })
}

fn dechunk(mut b: &[u8]) -> std::io::Result<Vec<u8>> {
    let err = || std::io::Error::new(std::io::ErrorKind::InvalidData, "bad chunked encoding");
    let mut out = Vec::new();
    loop {
        let line = b.windows(2).position(|w| w == b"\r\n").ok_or_else(err)?;
        let size = usize::from_str_radix(String::from_utf8_lossy(&b[..line]).trim(), 16).map_err(|_| err())?;
        b = &b[line + 2..];
        if size == 0 {
            return Ok(out);
        }
        out.extend_from_slice(b.get(..size).ok_or_else(err)?);
        b = b.get(size + 2..).ok_or_else(err)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addresses_and_chunks() {
        assert_eq!(parse_addr("http://127.0.0.1:6333/x"), "127.0.0.1:6333");
        assert_eq!(parse_addr(":7"), "127.0.0.1:7");
        assert_eq!(dechunk(b"3\r\nabc\r\n2\r\nde\r\n0\r\n\r\n").unwrap(), b"abcde");
        assert!(dechunk(b"zz\r\n").is_err());
    }
}
