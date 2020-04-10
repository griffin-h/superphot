import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-darkgrid')
x = np.linspace(1., 300., 500)
ls = [3., 150.]
us = [100., 250.]
for l, u in zip(ls, us):
    y = np.zeros(500)
    inside = (x<u) & (x>l)
    y[inside] = 1. / ((np.log(u) - np.log(l)) * x[inside])
    plt.plot(x, y, label='lower = {}, upper = {}'.format(l, u))
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.legend(loc=1)
plt.show()