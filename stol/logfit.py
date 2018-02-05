import numpy as np
from gpfit.fit import fit

N = 100
A = np.linspace(0.5, 10, 100)
B = np.array([0.01, 0.05, 0.4])

x, y = np.meshgrid(A, B)
z = np.log(x/(x-y))

u1 = np.array(list(A)*len(B))
u2 = np.hstack([[v]*len(A) for v in B])
u = [u1, u2]
w = np.hstack(z)

x = np.log(u)
y = np.log(w)

cn, er = fit(x, y, 3, "SMA")
print "RMS error: %.5f" % er
df = cn.get_dataframe()
df.to_csv("logfit.csv", index=False)
