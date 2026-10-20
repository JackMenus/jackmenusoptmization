from optiforge.benchmarks.problems import make_problem
bad=[]
for n in (1,2,3):
    for s in range(200):
        try: make_problem("knapsack_correlated", s, n=n)
        except Exception as e: bad.append((n,s,type(e).__name__)); 
print(len(bad), bad[:4])
from collections import Counter; print(Counter(b[0] for b in bad))
