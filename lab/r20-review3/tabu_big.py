import time, numpy as np, tracemalloc
from optiforge.core.space import Space, Binary
from optiforge.operators import primitives as P
for n in (200, 1000):
    sp = Space([Binary(f"b{i}") for i in range(n)])
    x = (np.arange(n) % 2).astype(float)
    tracemalloc.start(); t=time.time()
    s = list(P.binary_exchanges(x, sp))
    cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    print(n, len(s), f"{time.time()-t:.2f}s", f"peak {peak/1e6:.0f} MB")
