## TODO

1. 把先验知识融入到模型中
2. 模型压缩 (使用更小容量的模型达到大容量模型同等或近似的预测效果) (done)
3. 时序到模型的映射 (根据时序特征自动确定适合的模型) (待调研)
4. callback earlystopping、save best model (done)
5. 使用贝叶斯优化寻找最优超参数
6. 无序交叉检验
7. end-to-end 的数据预处理
8. 平稳化后通过 k sigma 去噪, 使用边缘值替代
9. 计算时序潜在最大周期 (done)
10. 处理预测 lag=1 的滞后问题 (done)
11. XGBoost callback

## Q&A

高斯过程和平稳分布过程有什么差别?
如何确定最好的 batch_size 和 epochs train_rate
预测期间是否可以 post fit
自动配置超参数
随机种子对初始化的影响
更好的交叉检验分配方法
季节性数据如何标准化

component 分解后分别有 NN 去 forecast, 后 combination

误差度量中如何融合时间步的影响

如何保证神经网络预测的无偏性

如何把函数簇整合到 NN Dense 中
