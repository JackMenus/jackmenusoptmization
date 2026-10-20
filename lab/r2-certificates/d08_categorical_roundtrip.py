# Categorical encode uses list.index (==), so equal-but-distinct choices collide
from common import *
sp = Space([Categorical("c", [1, True, 1.0 + 0j][:2])])
for v in [1, True]:
    x = sp.encode({"c": v}); print(repr(v), "->", x, "->", repr(sp.decode(x)["c"]))
sp2 = Space([Categorical("d", [0, False, "a"])])
x = sp2.encode({"d": False}); print("False ->", x, "->", repr(sp2.decode(x)["d"]))
