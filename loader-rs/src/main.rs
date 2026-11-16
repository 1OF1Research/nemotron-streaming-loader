// Card 13 expert-granular loader daemon (Rust, std-only).
//
// Seam: a long-lived process holding (a) the safetensors index parsed from
// each shard header, (b) a RAM cache of expert tensors keyed by
// (layer, expert, proj), LRU-evicted under a byte budget. The Python cell
// driver talks to it over a Unix socket with a tiny text protocol:
//
//   GET <tensor_name>\n        -> "OK <nbytes>\n" + raw bytes | "ERR <msg>\n"
//   STATS\n                    -> one JSON line of counters
//   PING\n                     -> "PONG\n"
//
// Reads are pread (FileExt::read_exact_at) at the exact header offsets —
// large sequential reads, never page faults. The daemon never writes to the
// checkpoint; it is read-only over the shards.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::fs::FileExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ---------- minimal JSON parsing (safetensors headers only) ----------

fn skip_ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && (b[*i] as char).is_whitespace() {
        *i += 1;
    }
}

fn parse_string(b: &[u8], i: &mut usize) -> Option<String> {
    if *i >= b.len() || b[*i] != b'"' {
        return None;
    }
    *i += 1;
    let mut out = String::new();
    while *i < b.len() {
        let c = b[*i];
        *i += 1;
        match c {
            b'"' => return Some(out),
            b'\\' => {
                if *i < b.len() {
                    let e = b[*i];
                    *i += 1;
                    match e {
                        b'n' => out.push('\n'),
                        b't' => out.push('\t'),
                        b'r' => out.push('\r'),
                        b'u' => {
                            if *i + 4 <= b.len() {
                                let hex = std::str::from_utf8(&b[*i..*i + 4]).ok()?;
                                let cp = u32::from_str_radix(hex, 16).ok()?;
                                out.push(char::from_u32(cp)?);
                                *i += 4;
                            }
                        }
                        other => out.push(other as char),
                    }
                }
            }
            _ => out.push(c as char),
        }
    }
    None
}

// Parse a JSON value loosely: returns the byte range [start,end) of the value.
fn value_range(b: &[u8], i: &mut usize) -> Option<(usize, usize)> {
    skip_ws(b, i);
    let start = *i;
    match *b.get(*i)? {
        b'{' => {
            let mut depth = 0i32;
            while *i < b.len() {
                match b[*i] {
                    b'"' => {
                        parse_string(b, i)?;
                        continue;
                    }
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        depth -= 1;
                        *i += 1;
                        if depth == 0 {
                            return Some((start, *i));
                        }
                        continue;
                    }
                    _ => {}
                }
                *i += 1;
            }
            None
        }
        b'[' => {
            let mut depth = 0i32;
            while *i < b.len() {
                match b[*i] {
                    b'"' => {
                        parse_string(b, i)?;
                        continue;
                    }
                    b'[' | b'{' => depth += 1,
                    b']' | b'}' => {
                        depth -= 1;
                        *i += 1;
                        if depth == 0 {
                            return Some((start, *i));
                        }
                        continue;
                    }
                    _ => {}
                }
                *i += 1;
            }
            None
        }
        b'"' => {
            parse_string(b, i)?;
            Some((start, *i))
        }
        _ => {
            while *i < b.len() && !matches!(b[*i], b',' | b'}' | b']') {
                *i += 1;
            }
            Some((start, *i))
        }
    }
}

fn parse_u64s(text: &str) -> Vec<u64> {
    text.trim_matches(|c| c == '[' || c == ']' || c == ' ')
        .split(',')
        .filter_map(|s| s.trim().parse::<u64>().ok())
        .collect()
}

#[derive(Clone, Debug)]
struct TensorLoc {
    shard: usize,
    begin: u64, // absolute file offset of the tensor's first byte
    end: u64,
    dtype: String,
    shape: Vec<u64>,
}

fn parse_header(header: &[u8], shard: usize, data_base: u64) -> HashMap<String, TensorLoc> {
    let mut out = HashMap::new();
    let mut i = 0usize;
    skip_ws(header, &mut i);
    if header.get(i) != Some(&b'{') {
        return out;
    }
    i += 1;
    loop {
        skip_ws(header, &mut i);
        match header.get(i) {
            Some(&b'}') => break,
            Some(&b',') => {
                i += 1;
                continue;
            }
            Some(&b'"') => {}
            _ => break,
        }
        let key = match parse_string(header, &mut i) {
            Some(k) => k,
            None => break,
        };
        skip_ws(header, &mut i);
        if header.get(i) == Some(&b':') {
            i += 1;
        }
        let (vs, ve) = match value_range(header, &mut i) {
            Some(r) => r,
            None => break,
        };
        if key == "__metadata__" {
            continue;
        }
        let obj = &header[vs..ve];
        // extract "dtype", "shape", "data_offsets" from the object text
        let mut dtype = String::new();
        let mut shape: Vec<u64> = Vec::new();
        let mut offsets: Vec<u64> = Vec::new();
        let text = String::from_utf8_lossy(obj);
        if let Some(p) = text.find("\"dtype\"") {
            if let Some(q) = text[p..].find(':') {
                let rest = &text[p + q + 1..];
                if let Some(e) = rest.find('"') {
                    if let Some(e2) = rest[e + 1..].find('"') {
                        dtype = rest[e + 1..e + 1 + e2].to_string();
                    }
                }
            }
        }
        if let Some(p) = text.find("\"shape\"") {
            if let Some(q) = text[p..].find('[') {
                if let Some(e) = text[p + q..].find(']') {
                    shape = parse_u64s(&text[p + q..p + q + e + 1]);
                }
            }
        }
        if let Some(p) = text.find("\"data_offsets\"") {
            if let Some(q) = text[p..].find('[') {
                if let Some(e) = text[p + q..].find(']') {
                    offsets = parse_u64s(&text[p + q..p + q + e + 1]);
                }
            }
        }
        if offsets.len() == 2 {
            out.insert(
                key,
                TensorLoc {
                    shard,
                    begin: data_base + offsets[0],
                    end: data_base + offsets[1],
                    dtype,
                    shape,
                },
            );
        }
    }
    out
}

// ---------- CRC64 (ECMA polynomial) for byte-identity checks ----------

fn crc64_table() -> [u64; 256] {
    let mut t = [0u64; 256];
    for i in 0..256u64 {
        let mut crc = i;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xC96C_5795_D787_0F42
            } else {
                crc >> 1
            };
        }
        t[i as usize] = crc;
    }
    t
}

fn crc64(data: &[u8], table: &[u64; 256]) -> u64 {
    let mut crc = !0u64;
    for &b in data {
        crc = table[((crc ^ b as u64) & 0xff) as usize] ^ (crc >> 8);
    }
    !crc
}

// ---------- the cache + store ----------

struct Store {
    files: Vec<std::fs::File>,
    index: HashMap<String, TensorLoc>,
    cache: HashMap<String, Arc<Vec<u8>>>,
    order: Vec<String>, // LRU order, front = least recent
    cache_bytes: usize,
    budget: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
    bytes_from_disk: u64,
    bytes_from_cache: u64,
    disk_read_s: f64,
}

impl Store {
    fn touch(&mut self, name: &str) {
        if let Some(p) = self.order.iter().position(|n| n == name) {
            let n = self.order.remove(p);
            self.order.push(n);
        }
    }
    fn get(&mut self, name: &str, table: &[u64; 256]) -> Result<(Arc<Vec<u8>>, bool), String> {
        if let Some(v) = self.cache.get(name) {
            let v = v.clone();
            self.touch(name);
            self.hits += 1;
            self.bytes_from_cache += v.len() as u64;
            return Ok((v, true));
        }
        let loc = self
            .index
            .get(name)
            .ok_or_else(|| format!("no such tensor {name}"))?
            .clone();
        let f = &self.files[loc.shard];
        let len = (loc.end - loc.begin) as usize;
        let mut buf = vec![0u8; len];
        let t0 = Instant::now();
        f.read_exact_at(&mut buf, loc.begin)
            .map_err(|e| format!("pread failed: {e}"))?;
        self.disk_read_s += t0.elapsed().as_secs_f64();
        self.misses += 1;
        self.bytes_from_disk += len as u64;
        let _ = table;
        let arc = Arc::new(buf);
        // insert with LRU eviction under budget
        while self.cache_bytes + len > self.budget && !self.order.is_empty() {
            let victim = self.order.remove(0);
            if let Some(v) = self.cache.remove(&victim) {
                self.cache_bytes -= v.len();
                self.evictions += 1;
            }
        }
        if len <= self.budget {
            self.cache.insert(name.to_string(), arc.clone());
            self.order.push(name.to_string());
            self.cache_bytes += len;
        }
        Ok((arc, false))
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // usage: expert-loader <ckpt_dir> <socket_path> <budget_bytes>
    let ckpt = args.get(1).expect("ckpt dir").clone();
    let sock = args.get(2).expect("socket path").clone();
    let budget: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(9_000_000_000usize);

    // discover shards from the index json (weight_map), in filename order
    let idx_text = std::fs::read_to_string(format!("{ckpt}/model.safetensors.index.json"))
        .expect("index json");
    let mut shards: Vec<String> = Vec::new();
    for part in idx_text.split('"') {
        if part.ends_with(".safetensors") && !shards.contains(&part.to_string()) {
            shards.push(part.to_string());
        }
    }
    shards.sort();

    let table = crc64_table();
    let mut index: HashMap<String, TensorLoc> = HashMap::new();
    let mut files = Vec::new();
    for (si, sh) in shards.iter().enumerate() {
        let path = format!("{ckpt}/{sh}");
        let f = std::fs::File::open(&path).expect("open shard");
        let mut lenbuf = [0u8; 8];
        f.read_exact_at(&mut lenbuf, 0).expect("read header len");
        let hlen = u64::from_le_bytes(lenbuf) as usize;
        let mut header = vec![0u8; hlen];
        f.read_exact_at(&mut header, 8).expect("read header");
        let part = parse_header(&header, si, 8 + hlen as u64);
        index.extend(part);
        files.push(f);
    }
    eprintln!(
        "expert-loader: {} shards, {} tensors indexed, budget {} B",
        shards.len(),
        index.len(),
        budget
    );

    let store = Arc::new(Mutex::new(Store {
        files,
        index,
        cache: HashMap::new(),
        order: Vec::new(),
        cache_bytes: 0,
        budget,
        hits: 0,
        misses: 0,
        evictions: 0,
        bytes_from_disk: 0,
        bytes_from_cache: 0,
        disk_read_s: 0.0,
    }));

    let _ = std::fs::remove_file(&sock);
    let listener = UnixListener::bind(&sock).expect("bind socket");
    eprintln!("expert-loader: listening on {sock}");
    for conn in listener.incoming() {
        let stream = match conn {
            Ok(s) => s,
            Err(_) => continue,
        };
        let store = store.clone();
        let table = table;
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut w = stream;
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {}
                    Err(_) => break,
                }
                let cmd = line.trim();
                if cmd == "PING" {
                    let _ = w.write_all(b"PONG\n");
                } else if cmd == "STATS" {
                    let s = store.lock().unwrap();
                    let msg = format!(
                        "{{\"hits\":{},\"misses\":{},\"evictions\":{},\"cache_bytes\":{},\"cache_entries\":{},\"bytes_from_disk\":{},\"bytes_from_cache\":{},\"disk_read_s\":{:.3}}}\n",
                        s.hits, s.misses, s.evictions, s.cache_bytes,
                        s.cache.len(), s.bytes_from_disk, s.bytes_from_cache, s.disk_read_s
                    );
                    let _ = w.write_all(msg.as_bytes());
                } else if let Some(name) = cmd.strip_prefix("GET ") {
                    let name = name.trim().to_string();
                    let mut s = store.lock().unwrap();
                    match s.get(&name, &table) {
                        Ok((bytes, _hit)) => {
                            let hdr = format!("OK {}\n", bytes.len());
                            if w.write_all(hdr.as_bytes()).is_err() {
                                break;
                            }
                            if w.write_all(&bytes).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = w.write_all(format!("ERR {e}\n").as_bytes());
                        }
                    }
                } else if let Some(name) = cmd.strip_prefix("CRC ") {
                    // stream the tensor's CRC without holding it in cache twice
                    let name = name.trim().to_string();
                    let mut s = store.lock().unwrap();
                    match s.get(&name, &table) {
                        Ok((bytes, _)) => {
                            let c = crc64(&bytes, &table);
                            let _ = w.write_all(format!("CRC {c:016x}\n").as_bytes());
                        }
                        Err(e) => {
                            let _ = w.write_all(format!("ERR {e}\n").as_bytes());
                        }
                    }
                } else if cmd == "QUIT" {
                    break;
                } else {
                    let _ = w.write_all(b"ERR unknown command\n");
                }
            }
        });
    }
}
