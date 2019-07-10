
# 原理和解释

## 原理

以下是 ARMA 模型

$$
X_{t}=c+\varepsilon_{t}+\sum_{i=1}^{p} \varphi_{i} X_{t-i}+\sum_{i=1}^{q} \theta_{i} \varepsilon_{t-i} \tag{1}
$$

AR 部分如下:

$$
X_{t}=c+\varepsilon_{t}+\sum_{i=1}^{p} \varphi_{i} X_{t-i}
$$

它表示过去 p 个时间值的加权和与当前取值有关.

MA 部分如下:

$$
X_{t}=\varepsilon_{t}+\sum_{i=1}^{q} \theta_{i} \varepsilon_{t-i} \tag{1}
$$



而滑动窗口标注如下:

$$
\begin{pmatrix}
y_{t-p-2} & ... & ... & y_{t-3} & y_{t-2}\\ 
y_{t-p-1} & ... & ... & y_{t-2} & y_{t-1}\\ 
y_{t-p} & ... & ... & y_{t-1} & y_{t}\\ 
y_{t-p+1} & ... & ... & y_{t} & y_{t+1}
\end{pmatrix} \tag{2}
$$

p 是我们选择的特征窗口大小. q 又时间序列长度决定. (注意不是 ARMA 模型中的 p,q)

也就是说， 模型的输入为如下形式：

$$
X_{q,p} = 
\begin{pmatrix}
y_{t-p-q+1} & ... & ... & y_{t-q}\\ 
... & ... & ... & ...\\ 
y_{t-p} & ... & ... & y_{t-1}\\ 
y_{t-p+1} & ... & ... & y_{t}
\end{pmatrix} \tag{3}
$$

模型的输出为如下形式：

$$
y = 
\begin{pmatrix}
y_{t-q+1}\\ 
...\\ 
y_{t}\\ 
y_{t+1}
\end{pmatrix} \tag{4}
$$

因此， 滑动窗口本质上是非线性 AR 模型，表达如下

$$
\mathbf{y}_{t+1}=f\left(\mathbf{y}_{t}, \mathbf{y}_{t-1}, \cdots, \mathbf{y}_{t-p+1}\right) \tag{5}
$$


这个 $f$ 就是我们要训练的模型， 如神经网络， XGBoost. 当它为线性模型时， 就化成我们常见的自回归模型. 为直观表示, 可以写成矩阵形式:

$$
\left[\begin{array}{c}{y_{1}} \\ {y_{2}} \\ {y_{3}} \\ {\vdots} \\ {y_{t}}\end{array}\right]=\left[\begin{array}{cccc}{y_{0}} & {y_{-1}} & {y_{-2}} & {\dots} \\ {y_{1}} & {y_{0}} & {y_{-1}} & {\dots} \\ {y_{2}} & {y_{1}} & {y_{0}} & {\dots} \\ {\vdots} & {\vdots} & {\vdots} & {\ddots} \\ {y_{t-1}} & {y_{t-2}} & {y_{t-3}} & {\cdots}\end{array}\right]\left[\begin{array}{l}{f} \\ {f} \\ {f} \\ {\vdots} \\ {f}\end{array}\right]
$$

实际上，训练模型前还有特征提取阶段, 把模型和特征提取函数写成复合函数形式：

$$
\mathbf{y}_{t+1}=f \circ g \left(\mathbf{y}_{t}, \mathbf{y}_{t-1}, \cdots, \mathbf{y}_{t-p+1}\right) \tag{6}
$$

模型训练和特征提取可以表达成复合函数形式.

这种模型有一定的局限, 就是学习长期依赖, 为此, 可以改成如下形式:

$$
\mathbf{y}_{t} , \mathbf{h}_{t}=f\left(\mathbf{h}_{t-1}, \mathbf{y}_{t-1}, \cdots, \mathbf{y}_{t-p}\right) \tag{5}
$$

结构如下:

![srnn](../png/SRNN.png)

它包括两部分: 循环结构和自回归. 这不就是循环神经网络吗!

因此, 我们使用的方法本质上可以分为两类:

1. 线性自回归, 非线性自回归 (Linear XGBoost MLP)
2. 带反馈的非线性自回归 (LSTM)

## 各种预测模型的本质


## 参考资料

[Nonlinear autoregressive exogenous model](https://en.wikipedia.org/wiki/Nonlinear_autoregressive_exogenous_model)
