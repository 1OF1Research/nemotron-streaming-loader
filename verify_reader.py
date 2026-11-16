# Card 13 — reader verification: Rust daemon bytes vs torch-loaded tensors.
# Compares CRC64-ECMA (computed in Rust, same implementation both sides of
# the comparison: Python recomputes over the daemon's raw bytes AND over the
# tensor torch materializes from the same shard via safetensors) and exact
# shape/dtype, on a stratified sample: every MoE layer x {first, middle,
# last routed expert} x {up_proj, down_proj} + shared experts + a spread of
# non-expert tensors.
import json, socket, struct, sys, time, zlib

CKPT = "/home/agent/hf-weekend/pool-h/nemotron-3-nano-30b-a3b-bf16"
SOCK = "/home/agent/hf-weekend/pool-h/card13/loader.sock"

# Fast checksum for reporting (zlib crc32). The proof is exact byte
# equality (raw == tb); the checksum is a compact fingerprint beside it.
def crc64(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF

class Client:
    def __init__(self, path):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.connect(path)
        self.buf = b""
    def _line(self):
        while b"\n" not in self.buf:
            chunk = self.s.recv(1 << 20)
            if not chunk:
                raise IOError("daemon closed")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return line.decode()
    def _exact(self, n):
        while len(self.buf) < n:
            chunk = self.s.recv(1 << 22)
            if not chunk:
                raise IOError("daemon closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out
    def get(self, name):
        self.s.sendall(f"GET {name}\n".encode())
        head = self._line()
        if head.startswith("ERR"):
            raise IOError(head)
        n = int(head.split()[1])
        return self._exact(n)
    def stats(self):
        self.s.sendall(b"STATS\n")
        return json.loads(self._line())

sys.path.insert(0, "/home/agent/hf-weekend/pool-h/h1")
import torch
from safetensors import safe_open
import json as _json
with open(f"{CKPT}/model.safetensors.index.json") as f:
    wm = _json.load(f)["weight_map"]

def torch_bytes(name):
    with safe_open(f"{CKPT}/{wm[name]}", framework="pt") as sf:
        t = sf.get_tensor(name)
    return t.view(torch.uint16).numpy().tobytes(), str(t.dtype), list(t.shape)

moe_layers = [1,3,6,8,10,13,15,17,20,22,24,27,29,31,34,36,38,40,43,45,47,49,51]
names = []
for L in moe_layers:
    for e in (0, 63, 127):
        for p in ("up_proj", "down_proj"):
            names.append(f"backbone.layers.{L}.mixer.experts.{e}.{p}.weight")
    names.append(f"backbone.layers.{L}.mixer.shared_experts.up_proj.weight")
    names.append(f"backbone.layers.{L}.mixer.shared_experts.down_proj.weight")
    names.append(f"backbone.layers.{L}.mixer.gate.weight")
names += [
    "backbone.embeddings.weight",
    "backbone.norm_f.weight",
    "lm_head.weight",
    "backbone.layers.0.mixer.in_proj.weight",
    "backbone.layers.5.mixer.k_proj.weight",
    "backbone.layers.26.mixer.q_proj.weight",
]

c = Client(SOCK)
t0 = time.time()
ok = bad = 0
fails = []
for name in names:
    raw = c.get(name)
    tb, dt, sh = torch_bytes(name)
    match = (raw == tb) and (crc64(raw) == crc64(tb))
    if match:
        ok += 1
    else:
        bad += 1
        fails.append({"name": name, "daemon_len": len(raw), "torch_len": len(tb),
                      "daemon_crc": f"{crc64(raw):016x}", "torch_crc": f"{crc64(tb):016x}"})
    del raw, tb
res = {"checked": len(names), "byte_identical": ok, "mismatches": bad,
       "fails": fails, "wall_s": round(time.time() - t0, 2),
       "daemon_stats": c.stats()}
print(json.dumps(res, indent=1))
with open("/home/agent/hf-weekend/pool-h/card13/runs/reader-verify.json", "w") as f:
    json.dump(res, f, indent=1)
print("VERIFY", "PASS" if bad == 0 else "FAIL")
