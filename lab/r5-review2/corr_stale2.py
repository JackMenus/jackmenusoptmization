exec(open("/home/claude/dev/review2/corr_stale_cache.py").read().split("algo = I.DSLAlgorithm(CORR_SPEC)")[0])
import gc
algo = I.DSLAlgorithm(CORR_SPEC)
for t in range(300):
    p = make_problem("ellipsoid" if t % 2 else "rastrigin_rotated", t, dim=6)
    r = algo.solve(p, Budget(16 + 3 + (t % 3)), seed=t)
    if t % 5 == 0:
        junk = [np.empty((16, 6)) for _ in range(t % 7)]; del junk
    gc.collect()
print(stats)
