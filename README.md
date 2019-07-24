# time-series-forecast

本项目实现基于机器学习和深度学习的时间序列预测, 通过训练多个模型, 以一定的组合策略(Bestone or Combiantion) 组合各模型的预测结果.

组合策略目前有两种:

1. 选择模型中的 topk 进行 forecast combination
2. 选择模型中的 top one 进行 forecast

进行时序预测建模我们使用如下模型:

1. 线性模型 ARIMA
2. 集成模型(XGBoost)
3. Prophet(Facebook开源的预测工具)
4. 前馈神经网络
5. 卷积神经网络
6. 循环神经网络

## 基本原理

时间序列预测(forecasting)有两类方法:

1. 时间外推
2. 自回归

时间外推: 建立当前时间戳到与之对应的取值的映射(可以使用统计学方法或机器学习模型), 即 y <- f(t) 的形式. 预测新值外推时间戳即可.

自回归: 建立历史值与当前值的映射(同样可以使用统计学方法或机器学习模型), 即 yt <- f(yt-1, yt-2, ...). 预测新值通过递归方式进行.

时间外推法有很大的局限性, 仅能捕捉到时序中的长期依赖关系. 因此, 接下的建模我们值使用自回归方法. 

## 转化为带标注数据

时间序列数据没有带标注, 而带监督的机器学习学习模型的训练需要定义输入与输出, 那么怎么把时间序列数据转化为带标注的形式呢？

从自会回出发, 时序之间存在自回归关系, 那么, 我们可以把它转化为带标注形式的数据:

![how-to-label-data](./png/how-to-labeling-time-series.png)

## 预处理

预处理包括如下:
1. 缺失值处理
2. 白噪声检验或随机游走检验
3. 去噪
4. 平稳化
5. 数据正太化
6. 归一化

原始时序中包含大量噪声, 可以使用如下方法去噪:

1. 平滑法
2. 数字滤波器

此外, 还要通过白噪声检验或[随机游走检验](https://en.wikipedia.org/wiki/Random_walk_hypothesis)检验数据是否可预测.

stationary  --> detrend

拟合法或差分(对数差分)

正太分布化

https://machinelearningmastery.com/how-to-transform-data-to-fit-the-normal-distribution/

https://en.wikipedia.org/wiki/Log-normal_distribution

## 特征工程

1. 普通特征 (统计特征、拟合特征、分类特征)
2. dtw & wavelet
3. autocorrelation
4. reprecentation learning (深度学习自动特征提取)

目前运用到预测中的特征包括：最大值、最小值、mean、median、方差、标准差、kurtosis、skewness 等等

更丰富的时序特征提取模块见[时间序列特征提取](.md), 该模块包括了特征重要性评估

1~3 的特征提取通过滑动窗口的方式进行,滑动步长通常为 1. 4 为深度学习中的自动特征提取方法, 通过我们会使用 CNN 提取局部特征, LSTM 提取长期依赖特征.下面会展开这方面方法.

特征重要性评估

## 使用统一的评估指标

统一使用 MAPE 作为模型的评估指标.

## 训练与预测

建模方法：

1. 机器学习
2. 机器学习 + 集成方法
3. 深度学习
4. 深度学习 + combination (并行化训练)

以上的数据标注方法仅能让模型预测一个时间步的取值, 如何预测多个时间步呢? 我们有如下策略, 多时间步预测策略：

1. 直接多步预测
2. 递归多步预测
3. 多步输出策略
4. 混合策略

这些策略的详细解释和说明参考 Machine Learning Strategies for Time Series Forecasting.

直接多步预测: 建模获得预测值后, 使用新的模型再预测下一个值, 以此类推
递归多步预测: 
多步输出策略

我们使用第二点: 递归输出所需时间步


递归预测示意图如下:

![how-to-forecast-time-series](./png/how-to-forecast-time-series.png)


或者我们使用动态方式展示：

![how-to-forecast-time-series](./png/how-to-forecast-time-series.gif)


## sliding-window-based linear model

在滑动窗口提取特征的基础上, 我们实现线性模型, 预测效果如下

多周期预测效果:

![linear-1](./png/linear-multi-periodic-1.png)

趋势+多周期预测效果:

![linear-2](./png/linear-multi-periodic-2.png)

趋势+周期:

![linear-3](./png/linear-trend-and-periodic.png)


如何确定输入的大小？因为我们需要模型学习到数据中潜在的规律（增长、周期）

原理部分我们会解释, 这个模型本质上是 ARMA.

处理过拟合

## XGBoost

ensemble 进行时序预测的思路是先对非平稳序列进行平稳化, 然后使用 ensemble 相关的模型进行训练
与预测, 把预测结果再转化为原来的非平稳序列.

平稳序列 + XGBoost

处理过拟合

## MLP (前馈神经网络)

MLP 在时间序列预测中的实现可以看做是非线性自回归模型,

基于 MLP 的预测效果演示1：
![MLP-1](./png/MLP-1.png)

基于 MLP 的预测效果演示2：
![MLP-2](./png/MLP-2.png)

基于 MLP 的预测效果演示3：
![MLP-3](./png/MLP-3.png)

基于 MLP 的预测效果演示4：
![MLP-4](./png/MLP-4.png)

1. 谱正则化 & L1 L2
2. EarlyStopping
3. callback save best model
4. 权重滑动平均

谱正则化是从L约束中推导出来, 可以看做是L2正则化的加强版本. 使用时我们直接把它嵌入到隐层中.

EarlyStopping 通过监控模型训练的收敛情况, 提早跳出不收敛或面临过拟合的训练.

最后通过每一个 epoch 比较上一个 epoch 的验证效果, 选择做好的模型保存. 预测则是使用最好的模型保存.

权重滑动平均提高训练的稳定性.

前馈神经网络
适用谱正则化避免过拟合
监控 epoch weights, 保存最优 weights 到内存中
预测适用最优 weights.
滑动权重让训练稳定

## LSTM

LSTM 的训练效率比 MLP 更慢, 目前先不加入模型集合中.

LSTM 处理过拟合和 MLP 处理过拟合是相似的, 不详细讲述.

## 误差估计

误差是多少取决于我们多大程度上接受错误, 换句话说, 我们容忍多大概率实际取值并没有落到预测区间上。

长时间步预测的问题

## 模型对比



## 预测效果



## 预测失效情况

什么情况下预测不可用

伪随机数

## 解释

以上模型的原理和分析请参考:

详细参考[原理分析部分](./markdown/NAR.md)

有关的数学讨论参考[数学原理部分](./markdown/math.md)




## TODO

参考[TODO部分](./TODO.md)

## 参考资料

1. https://otexts.com/fpp2/
2. http://www.oxford-man.ox.ac.uk/sites/default/files/events/combination_Sofie.pdf
