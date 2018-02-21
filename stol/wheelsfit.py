import numpy as np
from gpfit.fit import fit
import matplotlib.pyplot as plt
import numpy as np
N = 100
x = np.linspace(4,20,20)

y = 20.5*x-60.15 #makes an array of results88

u = x
w = y

x = np.log(u)
y = np.log(w)

cn, er = fit(x, y, 1, "MA")
print "RMS error: %.5f" % er
df = cn.get_dataframe()
df.to_csv("wheelsfit2.csv", index=False)

plt.plot(u,w,'o')
yfit = cn.evaluate(x)
plt.plot(u,np.exp(yfit))
plt.show()