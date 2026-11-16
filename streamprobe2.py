# Card 13 probe — instrumented single pass: P-7 fault census + P-2 routing census.
# Separate cell: patched arm bytes, device map BUDGET=0 (every block disk-offloaded),
# 4 threads, max_new=1 (prefill 14 + 1 decode step). Controls untouched.
import argparse, json, sys, time
sys.path.insert(0, "/home/agent/hf-weekend/pool-h/h1")
import stubs
stubs.install()
import torch
if not torch.cuda.is_available():
    import contextlib
    torch.cuda.default_stream = lambda device=None: None
    torch.cuda.stream = lambda stream=None: contextlib.nullcontext()
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--arm-dir", required=True)
p.add_argument("--out", required=True)
p.add_argument("--max-new", type=int, default=1)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--budget-gib", type=float, default=0.0,
               help="CPU-resident budget in GiB for the block device map (0 = probe shape, all blocks disk)")
args = p.parse_args()
torch.set_num_threads(args.threads)
res = {"probe": "card13-stream", "threads": args.threads, "budget_gib": args.budget_gib}

tok = AutoTokenizer.from_pretrained(args.arm_dir)
prompt = "The history of the steam engine begins in the first century, when engineers"
ids = tok(prompt, return_tensors="pt")
res["prompt_len"] = ids["input_ids"].shape[1]

# routing census: wrap every router's get_topk_indices
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
    offload_folder="/home/agent/hf-weekend/pool-h/card13/offload/probe",
    trust_remote_code=True, low_cpu_mem_usage=True)
model.eval()
res["load_s"] = round(time.time() - t0, 2)

# conv1d substrate repair (same as run_h1.py: decode reads conv1d.weight by attribute)
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
    fwd_marks.append({"t0": time.time()})
def _post(mod, inp, outp):
    m = fwd_marks[-1]; m["s"] = round(time.time() - m["t0"], 2); del m["t0"]
model.register_forward_pre_hook(_pre)
model.register_forward_hook(_post)

before = ru()
t1 = time.time()
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False)
res["pass_s"] = round(time.time() - t1, 2)
res["steps"] = [{"i": i, "kind": "prefill" if i == 0 else "decode", "s": m["s"]}
                for i, m in enumerate(fwd_marks)]
_dec = [m["s"] for m in fwd_marks[1:]]
if _dec:
    res["decode_step_mean_s"] = round(sum(_dec) / len(_dec), 2)
    res["decode_step_min_s"] = min(_dec); res["decode_step_max_s"] = max(_dec)
after = ru()
res["minflt_delta"] = after["minflt"] - before["minflt"]
res["majflt_delta"] = after["majflt"] - before["majflt"]
res["hwm_kb"] = after.get("hwm_kb")
res["out_ids"] = out[0].tolist()
res["new_token"] = out[0][ids["input_ids"].shape[1]:].tolist()

# routing summary: per layer, union of routed experts across the pass's forwards
union_counts, total_sel = {}, 0
for name, calls in route_log.items():
    u = set()
    for s in calls: u.update(s); total_sel += len(s)
    union_counts[name] = len(u)
res["forwards_logged"] = {k: len(v) for k, v in route_log.items()}
res["per_call_distinct"] = {k: [len(c) for c in v] for k, v in route_log.items()}
res["layers_logged"] = len(route_log)
res["union_per_layer"] = union_counts
if union_counts:
    vals = sorted(union_counts.values())
    res["union_min"] = vals[0]; res["union_max"] = vals[-1]
    res["union_mean"] = round(sum(vals)/len(vals), 2)
res["note"] = "one pass = prefill(14 tok) + 1 decode step; union counts experts touched per layer across both forwards"
with open(args.out, "w") as f:
    json.dump(res, f, indent=1)
print("PROBE DONE", res["pass_s"], "union_mean", res.get("union_mean"), "majflt", res["majflt_delta"])
