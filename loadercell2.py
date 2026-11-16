# Card 13 loader cell — the probe cell with ONLY the expert fetch changed.
#
# Cell: identical to streamprobe.py (patched arm bytes, BUDGET=0 all-disk
# device map, 4 threads, same prompt, greedy) except the routed-expert
# weights never come from accelerate's disk offload. Every MoE layer's
# experts are rebuilt as meta modules (no offload hook, no disk file), and
# NemotronHMOE.moe is replaced by a version that, per forward:
#   * computes the router-chosen expert set from topk_indices,
#   * fetches exactly those experts' up/down weights from the Rust loader
#     daemon (large sequential preads + RAM cache) and binds them,
#   * runs ONLY the chosen experts (the shipped loop's empty-expert dummy
#     pass over the other 122 experts is dropped — the dummy contributes an
#     identical constant to every token's output in exact arithmetic; the
#     identity gate below is what proves the numerics hold),
#   * unbinds (weights return to meta) so resident memory stays bounded.
# Everything else — attention, mamba, shared experts, embeddings — runs the
# shipped path untouched. The daemon's cache is the frequency-ranked
# resident core: LRU under a byte budget, warmed by prefill's fetches.
import argparse, json, socket, sys, time
sys.path.insert(0, "/home/agent/hf-weekend/pool-h/h1")
import stubs
stubs.install()
import torch
if not torch.cuda.is_available():
    import contextlib
    torch.cuda.default_stream = lambda device=None: None
    torch.cuda.stream = lambda stream=None: contextlib.nullcontext()
from transformers import AutoModelForCausalLM, AutoTokenizer

SOCK = "/home/agent/hf-weekend/pool-h/card13/loader.sock"
CKPT = "/home/agent/hf-weekend/pool-h/nemotron-3-nano-30b-a3b-bf16"

p = argparse.ArgumentParser()
p.add_argument("--arm-dir", required=True)
p.add_argument("--out", required=True)
p.add_argument("--max-new", type=int, default=2)
p.add_argument("--warm", action="store_true",
               help="run one untimed identical generate first (core warmed)")
p.add_argument("--threads", type=int, default=4)
p.add_argument("--budget-gib", type=float, default=0.0,
               help="CPU-resident budget in GiB for the block device map (0 = probe shape, all blocks disk)")
p.add_argument("--retain", action="store_true",
               help="cell-side retention: keep bound expert weights in the cell across steps; fetch only misses from the daemon")
args = p.parse_args()
torch.set_num_threads(args.threads)
res = {"probe": "card13-loader", "threads": args.threads,
       "budget_gib": args.budget_gib, "retain": bool(args.retain),
       "mode": "warm" if args.warm else "cold"}

# ---- daemon client ----
class Client:
    def __init__(self, path):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.connect(path)
        self.buf = b""
        self.fetches = 0
        self.fetch_bytes = 0
        self.fetch_s = 0.0
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
        t0 = time.time()
        self.s.sendall(f"GET {name}\n".encode())
        head = self._line()
        if head.startswith("ERR"):
            raise IOError(head)
        n = int(head.split()[1])
        data = self._exact(n)
        self.fetches += 1
        self.fetch_bytes += n
        self.fetch_s += time.time() - t0
        return data
    def stats(self):
        self.s.sendall(b"STATS\n")
        return json.loads(self._line())

client = Client(SOCK)

tok = AutoTokenizer.from_pretrained(args.arm_dir)
prompt = "The history of the steam engine begins in the first century, when engineers"
ids = tok(prompt, return_tensors="pt")
res["prompt_len"] = ids["input_ids"].shape[1]

# routing census (same wrap as the probe)
route_log = {}
def wrap_router(mod, name):
    orig = mod.get_topk_indices
    def wrapped(scores):
        idx = orig(scores)
        route_log.setdefault(name, []).append(sorted(set(idx.reshape(-1).tolist())))
        return idx
    mod.get_topk_indices = wrapped

from accelerate import init_empty_weights
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(args.arm_dir, trust_remote_code=True)
with init_empty_weights():
    meta_model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.bfloat16)
units = ["backbone.embeddings"] + [f"backbone.layers.{i}" for i in range(cfg.num_hidden_layers)] + ["backbone.norm_f", "lm_head"]
sizes = {}
for name, param in meta_model.named_parameters():
    for u in units:
        if name == u or name.startswith(u + "."):
            sizes[u] = sizes.get(u, 0) + param.numel() * 2
            break
del meta_model
dmap = {}
if args.budget_gib > 0:
    _budget = int(args.budget_gib * 1024**3)
    _used = 0
    for u in units:
        if u in ("backbone.embeddings", "backbone.norm_f", "lm_head") or _used + sizes.get(u, 0) <= _budget:
            dmap[u] = "cpu"; _used += sizes.get(u, 0)
        else:
            dmap[u] = "disk"
else:
    for u in units:
        dmap[u] = "cpu" if u in ("backbone.embeddings", "backbone.norm_f", "lm_head") else "disk"
res["cpu_units"] = sum(1 for v in dmap.values() if v == "cpu")
res["disk_units"] = sum(1 for v in dmap.values() if v == "disk")
res["cpu_gib"] = round(sum(sizes.get(u, 0) for u, v in dmap.items() if v == "cpu") / 1024**3, 2)

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    args.arm_dir, torch_dtype=torch.bfloat16, device_map=dmap,
    offload_folder="/home/agent/hf-weekend/pool-h/card13/offload/loader",
    trust_remote_code=True, low_cpu_mem_usage=True)
model.eval()
res["load_s"] = round(time.time() - t0, 2)

# conv1d substrate repair (same as the probe)
import os
from accelerate.hooks import remove_hook_from_module
from safetensors import safe_open
with open(os.path.join(args.arm_dir, "model.safetensors.index.json")) as f:
    _wm = json.load(f)["weight_map"]
def _load_tensor(name):
    with safe_open(os.path.join(args.arm_dir, _wm[name]), framework="pt") as sf:
        return sf.get_tensor(name)
_fixed = 0
for _mod_name, _mod in model.named_modules():
    if _mod.__class__.__name__ == "NemotronHMamba2Mixer":
        remove_hook_from_module(_mod.conv1d)
        _mod.conv1d.weight = torch.nn.Parameter(_load_tensor(_mod_name + ".conv1d.weight"))
        if _mod.conv1d.bias is not None:
            _mod.conv1d.bias = torch.nn.Parameter(_load_tensor(_mod_name + ".conv1d.bias"))
        _fixed += 1
res["conv1d_materialized"] = _fixed

# ---- rebuild every MoE layer's routed experts as meta modules and swap moe()
from accelerate.hooks import remove_hook_from_submodules

layer_prefix = {}
for name, mod in model.named_modules():
    if mod.__class__.__name__ == "NemotronHMOE":
        layer_prefix[mod] = name  # e.g. backbone.layers.1.mixer

moe_stats = {"forwards": 0, "experts_run": 0, "experts_skipped": 0,
             "retained_hits": 0, "daemon_fetches_here": 0}
retained = {}  # (prefix, expert_idx) -> (up_tensor, dn_tensor), cell-side retention

def bind_expert(expert, up, dn):
    expert.up_proj.weight = torch.nn.Parameter(up, requires_grad=False)
    expert.down_proj.weight = torch.nn.Parameter(dn, requires_grad=False)

def unbind_expert(expert, up, dn):
    expert.up_proj.weight = torch.nn.Parameter(
        torch.empty_like(up, device="meta"), requires_grad=False)
    expert.down_proj.weight = torch.nn.Parameter(
        torch.empty_like(dn, device="meta"), requires_grad=False)

def make_moe(prefix):
    def moe(self, hidden_states, topk_indices, topk_weights):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = expert_mask.permute(2, 0, 1)
        chosen = [i for i in range(len(self.experts)) if expert_mask[i].any()]
        moe_stats["forwards"] += 1
        moe_stats["experts_run"] += len(chosen)
        moe_stats["experts_skipped"] += len(self.experts) - len(chosen)
        for expert_idx in chosen:
            expert = self.experts[expert_idx]
            key = (prefix, expert_idx)
            if args.retain and key in retained:
                up, dn = retained[key]
                moe_stats["retained_hits"] += 1
            else:
                up_name = f"{prefix}.experts.{expert_idx}.up_proj.weight"
                dn_name = f"{prefix}.experts.{expert_idx}.down_proj.weight"
                up = torch.frombuffer(bytearray(client.get(up_name)), dtype=torch.bfloat16)
                dn = torch.frombuffer(bytearray(client.get(dn_name)), dtype=torch.bfloat16)
                up = up.view(expert.up_proj.weight.shape)
                dn = dn.view(expert.down_proj.weight.shape)
                moe_stats["daemon_fetches_here"] += 2
                if args.retain:
                    retained[key] = (up, dn)
            bind_expert(expert, up, dn)
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)
            expert_weights = topk_weights[token_indices, weight_indices]
            expert_input = hidden_states[token_indices]
            expert_output = expert(expert_input)
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            final_hidden_states.index_add_(0, token_indices, weighted_output)
            unbind_expert(expert, up, dn)
        return final_hidden_states.type(hidden_states.dtype)
    return moe

n_moe = 0
for mod, prefix in layer_prefix.items():
    remove_hook_from_submodules(mod.experts)
    for expert in mod.experts:
        for lin_name in ("up_proj", "down_proj"):
            lin = getattr(expert, lin_name)
            w = lin.weight
            lin.weight = torch.nn.Parameter(
                torch.empty(w.shape, dtype=w.dtype, device="meta"), requires_grad=False)
    mod.moe = make_moe(prefix).__get__(mod, mod.__class__)
    n_moe += 1
res["moe_layers_rebuilt"] = n_moe

n_router = 0
for name, mod in model.named_modules():
    if mod.__class__.__name__ == "NemotronHTopkRouter":
        wrap_router(mod, name); n_router += 1
res["routers_wrapped"] = n_router

def ru():
    r = {}
    with open("/proc/self/stat") as f:
        parts = f.read().split()
    r["minflt"] = int(parts[9]); r["majflt"] = int(parts[11])
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM"): r["hwm_kb"] = int(line.split()[1])
    return r

fwd_marks = []
def _pre(mod, inp):
    fwd_marks.append({"t0": time.time(), "fetches0": client.fetches,
                      "bytes0": client.fetch_bytes,
                      "hits0": moe_stats["retained_hits"]})
def _post(mod, inp, outp):
    m = fwd_marks[-1]
    m["s"] = round(time.time() - m["t0"], 2)
    m["fetches"] = client.fetches - m["fetches0"]
    m["fetch_bytes"] = client.fetch_bytes - m["bytes0"]
    m["retained_hits"] = moe_stats["retained_hits"] - m["hits0"]
    del m["t0"], m["fetches0"], m["bytes0"], m["hits0"]
model.register_forward_pre_hook(_pre)
model.register_forward_hook(_post)

def run_generate():
    fwd_marks.clear()
    f0, b0 = client.fetches, client.fetch_bytes
    t = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False)
    steps = [{"i": i, "kind": "prefill" if i == 0 else "decode",
              "s": m["s"], "fetches": m["fetches"],
              "fetch_bytes": m["fetch_bytes"],
              "retained_hits": m["retained_hits"]} for i, m in enumerate(fwd_marks)]
    return out, time.time() - t, client.fetches - f0, client.fetch_bytes - b0, steps

if args.warm:
    w_out, w_s, w_f, w_b, w_steps = run_generate()
    res["warm_pass"] = {"pass_s": round(w_s, 2), "fetches": w_f,
                        "fetch_bytes": w_b, "steps": w_steps,
                        "new_token": w_out[0][ids["input_ids"].shape[1]:].tolist()}

before = ru()
out, pass_s, n_f, n_b, steps = run_generate()
res["pass_s"] = round(pass_s, 2)
res["steps"] = steps
_dec = [s["s"] for s in steps if s["kind"] == "decode"]
if _dec:
    res["decode_step_mean_s"] = round(sum(_dec) / len(_dec), 2)
    res["decode_step_min_s"] = min(_dec); res["decode_step_max_s"] = max(_dec)
res["retained_experts_end"] = len(retained)
res["retained_bytes_end"] = sum(u.numel() * 2 + d.numel() * 2 for u, d in retained.values())
after = ru()
res["minflt_delta"] = after["minflt"] - before["minflt"]
res["majflt_delta"] = after["majflt"] - before["majflt"]
res["hwm_kb"] = after.get("hwm_kb")
res["out_ids"] = out[0].tolist()
res["new_token"] = out[0][ids["input_ids"].shape[1]:].tolist()
res["timed_fetches"] = n_f
res["timed_fetch_bytes"] = n_b
res["fetch_total_s"] = round(client.fetch_s, 2)
res["moe_stats"] = moe_stats
res["daemon_stats"] = client.stats()

union_counts, total_sel = {}, 0
for name, calls in route_log.items():
    u = set()
    for s in calls: u.update(s); total_sel += len(s)
    union_counts[name] = len(u)
res["layers_logged"] = len(route_log)
res["union_per_layer"] = union_counts
if union_counts:
    vals = sorted(union_counts.values())
    res["union_min"] = vals[0]; res["union_max"] = vals[-1]
    res["union_mean"] = round(sum(vals) / len(vals), 2)
with open(args.out, "w") as f:
    json.dump(res, f, indent=1)
print("LOADER DONE", res["pass_s"], "tokens", res["new_token"],
      "fetches", res["timed_fetches"], "fetch_bytes", res["timed_fetch_bytes"])
