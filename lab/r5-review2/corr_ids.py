exec(open("/home/claude/dev/review2/corr_stale_cache.py").read().split("I._Env.elite_factor = checked")[0])
ids = []
def rec(self, elite):
    ids.append((id(self.X), self.st["gen"]))
    return orig(self, elite)
I._Env.elite_factor = rec
algo = I.DSLAlgorithm(CORR_SPEC)
for t in range(100):
    algo.solve(make_problem("ellipsoid", t, dim=6), Budget(16 + 3), seed=t)
firsts = ids[::3] if False else ids
from collections import Counter
c = Counter(i for i, g in ids)
print(len(ids), len(c), c.most_common(3))
