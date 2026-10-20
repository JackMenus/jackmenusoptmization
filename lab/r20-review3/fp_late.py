import copy, numpy as np
from optiforge.dsl.spec import host_algorithm, DEFAULT_OPERATOR
from optiforge.dsl.interpreter import DSLAlgorithm
from optiforge.generation.fingerprint import behaviour_fingerprint, probe_instances
from optiforge.benchmarks.suites import get_suite
from optiforge.core.budget import Budget
suite = get_suite("continuous_general")
probes = probe_instances(suite.instances("train"))
print("n probes", len(probes))
a = host_algorithm(DEFAULT_OPERATOR, name="a"); a["restart"]["patience"]=15
b = host_algorithm(DEFAULT_OPERATOR, name="b"); b["restart"]["patience"]=200
# also second stage differing
fa = behaviour_fingerprint(lambda: DSLAlgorithm(a, "a"), probes)
fb = behaviour_fingerprint(lambda: DSLAlgorithm(b, "b"), probes)
print("fp equal:", fa[0]==fb[0], fa, fb)
diff=0
for inst in suite.instances("train"):
    ra = DSLAlgorithm(a,"a").solve(inst.make(), Budget(max_evaluations=3000), seed=0)
    rb = DSLAlgorithm(b,"b").solve(inst.make(), Budget(max_evaluations=3000), seed=0)
    print(inst.id, ra.best_f if hasattr(ra,'best_f') else ra.objective, rb.best_f if hasattr(rb,'best_f') else rb.objective)
