"""Run every distinct Python candidate in the registry through the sandbox on fixed inputs; print JSON outputs."""
import glob, json, os, sys
import numpy as np
from optiforge.isolation.sandbox import SandboxProcess, SandboxConfig
reg = sys.argv[1]
srcs = {}
for g in sorted(glob.glob(os.path.join(reg, "candidates", "*", "*.json"))):
    r = json.load(open(g))
    if r.get("representation") == "python":
        src = r["source"]["code"] if isinstance(r["source"], dict) else r["source"]
        srcs[r["id"]] = src
out = {}
rng = np.random.default_rng(0)
info = {"lower": [-5.0]*4 + [0, 0], "upper": [5.0]*4 + [1, 3], "types": ["real"]*4 + ["binary", "integer"],
        "perm_blocks": [], "sigma": 0.3, "dim": 6}
xs = [[float(v) for v in np.r_[rng.uniform(-5, 5, 4), rng.integers(0, 2), rng.integers(0, 4)]] for _ in range(10)]
for cid, src in sorted(srcs.items()):
    try:
        sb = SandboxProcess(src, config=SandboxConfig(request_timeout=5))
        try:
            out[cid] = [sb.call(x, info, i) for i, x in enumerate(xs)]
        finally:
            sb.close()
    except Exception as e:
        out[cid] = f"ERR {type(e).__name__}: {e}"
print(json.dumps(out, sort_keys=True))
