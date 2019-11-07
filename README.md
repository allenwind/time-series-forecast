# 多步时间序列预测

完成时间序列的多步预测需要三点：
1. 数据预处理实现
2. 预测模型的实现、训练、优化
3. 多步预测算法的实现

这三点应该分离实现。

第一点关注数据的清洗、变换等。具体可见 [time-series-utils]()
第二点根据场景需求而定，模型可能是机器学习模型，也可能是深度学习模型。不存在通用的模型能满足所有的场景。预测结果的好坏取决于模型的实现、训练优化.

本项目则是实现第三点，多步时间序列预测。

思路如下

首先把原始序列（预处理后）化成带标注形式，

![how-to-labeling-time-series](./asset/how-to-labeling-time-series.png)

我们称它为自监督标注。这种做法在 `NLP` 中训练 `RNN` 十分普遍，只不过我们把它引入到时序预测中。时间窗口的大小以超参数的形式存在，也可以考虑自动化地确定时间窗口的大小。



预测时，通过滑动一个固定的窗口完成多步预测，

![how-to-forecast-time-series](./asset/how-to-forecast-time-series.png)

如果使用的预测模型是神经网络，则没有显式的特征计算过程。



从 `deepmind` [wavenet](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio) 一动画更直观理解多步预测，

![how-to-forecast-time-series](./asset/how-to-forecast-time-series.gif)

