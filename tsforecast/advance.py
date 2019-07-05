import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects

# 训练神经网络的高级技巧

def gelu(x):
    """gelu 激活函数
    目前的时序预测中, 进行基本测试
    隐层使用 gelu 比 relu 收敛更快更好, 并且不影响训练效率
    更多信息可参考 paper:
    https://arxiv.org/pdf/1606.08415.pdf
    K.sqrt(2/K.constant(np.pi)) = 0.7978845608028654
    """

    const = K.sqrt(2/K.constant(np.pi))
    cdf = 0.5 * (1 + K.tanh(const * (x + 0.044715 * K.pow(x, 3))))
    return x * cdf

class SGLDOptimizer:
    """Stochastic gradient langevin dynamics optimizer

    wiki: https://en.wikipedia.org/wiki/Stochastic_gradient_Langevin_dynamics
    """

class WeightEMA:

    """使用权重指数滑动平均提供训练稳定性
    """

    def __init__(self, model, momentum=0.9999):
        self.momentum = momentum
        self.model = model
        self.ema_weights = [K.zeros(K.shape(w)) for w in model.weights]
        
    def inject(self):
        self.initialize()
        for w1, w2 in zip(self.ema_weights, self.model.weights):
            op = K.moving_average_update(w1, w2, self.momentum)
            self.model.metrics_updates.append(op)

    def initialize(self):
        self.old_weights = K.batch_get_value(self.model.weights)
        K.batch_set_value(zip(self.ema_weights, self.old_weights))

    def apply_ema_weights(self):
        self.old_weights = K.batch_get_value(self.model.weights)
        ema_weights = K.batch_get_value(self.ema_weights)
        K.batch_set_value(zip(self.model.weights, ema_weights))

    def reset_old_weights(self):
        K.batch_set_value(zip(self.model.weights, self.old_weights))

class SpectralNormalization:

    """谱正则化技巧, 从L约束中推导出来, 可以看作是加强的 L2 正则化, 可以更好地避免
    过拟合. 使用时直接嵌入到神经网络相应的层中.
    """

    def __init__(self, layer):
        self.layer = layer

    def spectral_norm(self, w, r=5):
        w_shape = K.int_shape(w)
        in_dim = np.prod(w_shape[:-1]).astype(int)
        out_dim = w_shape[-1]
        w = K.reshape(w, (in_dim, out_dim))
        u = K.ones((1, in_dim))
        for i in range(r):
            v = K.l2_normalize(K.dot(u, w))
            u = K.l2_normalize(K.dot(v, K.transpose(w)))
        return K.sum(K.dot(K.dot(u, w), K.transpose(v)))

    def spectral_normalization(self, w):
        return w / self.spectral_norm(w)

    def __call__(self, inputs):
        with K.name_scope(self.layer.name):
            if not self.layer.built:
                input_shape = K.int_shape(inputs)
                self.layer.build(input_shape)
                self.layer.built = True
                if self.layer._initial_weights is not None:
                    self.layer.set_weights(self.layer._initial_weights)
        if not hasattr(self.layer, 'spectral_normalization'):
            if hasattr(self.layer, 'kernel'):
                self.layer.kernel = self.spectral_normalization(self.layer.kernel)
            if hasattr(self.layer, 'gamma'):
                self.layer.gamma = self.spectral_normalization(self.layer.gamma)
            self.layer.spectral_normalization = True
        return self.layer(inputs)

class HumanFeaturesExtractingLayer(Layer):
    
    """自定义神经网络的人工特征提取层. 
    features.py 中的特征提取无法直接
    在神经网络中使用, 因为它无法参与到
    梯度流运算中."""

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super().build(input_shape)

    def call(self, x):
        # TODO extract features
        pass

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def variance_error(y_true, y_pred):
    return K.var(y-true-y_pred)

def mean_absolute_percentage_variance_error(y_true, y_pred):
    return 100 * K.mean(K.abs(y_true-y_pred) / y_true) + K.var(y_true-y_pred)

mapve = mean_absolute_percentage_variance_error

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """wiki:
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    """

    return 200 * K.mean(K.abs(y_true-y_pred) / (y_true+y_pred))

smape = symmetric_mean_absolute_percentage_error

def weighted_mean_absolute_percentage_error(y_true, y_pred):
    """paper:
    http://ir.ii.uam.es/rue2012/papers/rue2012-cleger-tamayo.pdf
    """

    alpha = 1
    beta = 0.999
    weights = alpha * K.cumprod(beta * K.ones(y_true.shape))
    return K.sum(weights * K.abs(y_true-y_pred) / y_true)

wmape = weighted_mean_absolute_percentage_error

def time_series_pct(y_true, y_pred):
    """pct 一种度量时间序列预测精度的方法"""
    return K.mean(K.abs(y_true-y_pred) / y_true)

pct = time_series_pct

def lr_schedule(epoch):
    """动态学习率, 目前默认初始值为 1e-4 * 3"""

    lr = 1e-4 * 3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr

class SaveBestModelOnMemory(Callback):
    
    """训练期间, 每个 epoch 检查当前模型是否是最好模型, 
    如果是, 则存放到内存中, 否则丢弃. 训练结束后, 使用最
    好的模型作为训练结果. 这种实现比 keras ModelCheckpoint
    更高效, 因为后者会把模型存储到磁盘上并带来较大的存储空间开销.
    如果要配合 EarlyStopping 使用, 需要合理的 stop 机制.

    由于当前的运用场景是时序预测, 因此只实现了 min mode
    monitor 只支持 val_loss.
    """

    def __init__(self, monitor='val_loss', verbose=0, mode='min', period=1):
        super().__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.period = period
        self.best_epochs = 0
        self.epochs_since_last_save = 0
        self.monitor_op = np.less
        self.best = np.Inf
                
    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            # filepath = self.filepath.format(epoch=epoch + 1, **logs)
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can pick best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    # update weights
                    self.best = current
                    self.best_epochs = epoch + 1
                    self.best_weights = self.model.get_weights()            
                    
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)

get_custom_objects().update({"gelu": Activation(gelu)})
adam = Adam(lr=lr_schedule(0)) # 自调节学习率的 adam 优化算法
