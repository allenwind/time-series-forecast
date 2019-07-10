import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def random_fourier_series(size=1000, n=1000):

    x = np.linspace(0, 10*np.pi, size)
    an = np.random.uniform(-1, 1, n)
    bn = np.random.uniform(-1, 1, n)

    s = 0
    y = 0
    ys = []
    for a, b in zip(an, bn):
        s += 1
        y += a * np.cos(s*x) + b * np.sin(s*x)
        ys.append(y.copy())
    return ys

ys = random_fourier_series()

# for y in ys:
#     plt.plot(y)

# plt.show()

w = np.array(ys)
plt.imshow(w)
plt.show()

y = ys[-1]

plt.plot(y)
plt.show()

plt.hist(y, bins=100)

a = np.min(y)
b = np.max(y)

x = np.linspace(a, b, 1000)
m = np.mean(y)
s = np.var(y)
r = stats.norm.pdf(x, m, s)
plt.plot(x, r)
plt.show()
