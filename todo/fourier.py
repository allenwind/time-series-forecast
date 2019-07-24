import numpy as np
import matplotlib.pyplot as plt

# 谱分析模块, 把时间序列转换到频域后分析
# 确定时序中隐含的周期, 并计算时序的最大周期
# 最大周期作为神经网络的 window size

def find_longest_period(series):
    return (len(series) - 3) // 5

def fourier_series(an, bn, T=1, size=10000):
    x = np.linspace(-T*size, T*size, size)
    y = 0

    n = 1
    for a, b in zip(an, bn):
        y += a * np.cos(2*np.pi*n*x / T) + b * np.sin(2*np.pi*n*x / T)
        n += 1
    return y

def random_fourier_series(size=1000):
    # random fourier series
    x = np.linspace(0, 10*np.pi, size)
    n = 50
    an = np.random.uniform(-1, 1, n)
    bn = np.random.uniform(-1, 1, n)
    noise = np.random.uniform(-0.1, 0.1, len(x)) + np.random.normal(loc=0, scale=0.5, size=len(x))

    s = 0
    y = 0
    ys = []
    for a, b in zip(an, bn):
        s += 1
        y += a * np.cos(s*x) + b * np.sin(s*x)
        ys.append(y.copy()) # y will change next loop
    return y, ys

def test_w():
    from dataset import load_data_by_keyword
    df = load_data_by_keyword("sun")
    series = df.values[:, 1]
    #series = np.diff(series)

    w = np.fft.fft(series)
    # w[w<1500] = 0
    plt.subplot(211)
    plt.plot(series)
    plt.subplot(212)
    plt.plot(w)
    plt.show()

def test():
    y, ys = random_fourier_series()

    size = len(ys)
    fig = 1
    for i in range(size):
        plt.subplot(size, 1, fig)
        fig += 1
        plt.plot(ys[-i-1])
        plt.xticks([])
        plt.yticks([])

    plt.show()


    plt.subplot(211)
    plt.plot(y)

    plt.subplot(212)
    plt.plot(np.fft.fft(y))
    plt.show()

    plt.hist(y)
    plt.show()

if __name__ == "__main__":
    test()
