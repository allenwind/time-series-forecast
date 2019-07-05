import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PowerTransformer

# 时间序列的去燥和离群点模块
# 包括数字信号处理方法和神经网络(自编码器和变分自编码)方法

# 数字滤波器的方法这里不详细了(wavelet, fourier)
# 基于自编码器的思路如下:
# 由于数据本身就带有噪声, 无法使用传统的加噪声
# 去噪方法.
# paper:
# noise2noise https://arxiv.org/pdf/1803.04189.pdf
# PAE denoising https://arxiv.org/pdf/1509.05982.pdf

# 噪声可能是加成或乘成, 后者可以区对数后转为为前者
