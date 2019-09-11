#### batch normalization
BN层在使用过程中的`training`参数非常重要（参考至[BN原理与实战](https://zhuanlan.zhihu.com/p/34879333)）：    
> BN层在训练的过程中每一层计算的期望$\mu$与方差$\sigma^2$都是基于当前batch中的训练数据，之后更新$Z^{[l]}$；但在测试阶段有可能只需要预测一个样本或很少的样本，此时计算的期望与方差一定是有偏估计，因此，我们需要使用整个样本的统计量来对测试数据进行归一化，即使用均值与方差的无偏估计([无偏估计样本方差为什么分母是m-1而不是m?](https://www.matongxue.com/madocs/607.html))：
> $$\mu_{test}=\Epsilon(\mu_{batch})$$
> $$\sigma_{batch}^{2}=\frac{m}{m-1}\Epsilon(\sigma_{batch}^2)$$
> 得到每个特征的均值与方差的无偏估计后，对test数据才用同样的normalization方法：
> $$BN(X_{test})=\gamma \cdot \frac{X_{test}-\mu_{test}}{\sqrt{\sigma_{test}^2 + \epsilon}} + \beta$$
> 此外，同样可使用train阶段每个batch计算的mean\variance的加权平均数来得到test阶段mean\variance的估计

tf中使用BN层有如下几种方法：

1. tf.layers.batch_normalization()   
tf.layers提供高层的神经网络，主要和卷积相关，是对tf.nn的进一步封装    
示例如下：
```python
tf.layers.batch_normalization(inputs, training=True)
```
2. tf.nn.batch_normalization()    
相对于tf.layers, tf.nn更底层，属于低阶API，提供神经网络相关操作的支持，且tf.nn.batch_normalization()中无`training`参数（需要手动传入期望与方差），因此需要在其基础上继续封装，下面是一个示例：
```python
from tensorflow.python.training import moving_averages
def create_variable(name, shape, initializer, dtype=tf.float32, trainable=True):
    return tf.get_variable(name, shape=shape, dtype=dtype, initializer=initializer, trainable=trainable)
# batchnorm layer
def batchnorm(inputs, scope, epsilon=1e-05, momentum=0.99, is_training=True):
    inputs_shape = inputs.get_shape().as_list()
    params_shape = inputs_shape[-1:]
    axis = list(range(len(inputs_shape) - 1))

    with tf.variable_scope(scope):
        beta = create_variable("beta", params_shape,
                               initializer=tf.zeros_initializer())
        gamma = create_variable("gamma", params_shape,
                                initializer=tf.ones_initializer())
        # for inference
        moving_mean = create_variable("moving_mean", params_shape,
                            initializer=tf.zeros_initializer(), trainable=False)
        moving_variance = create_variable("moving_variance", params_shape,
                            initializer=tf.ones_initializer(), trainable=False)
    if is_training:
        mean, variance = tf.nn.moments(inputs, axes=axis)
        update_move_mean = moving_averages.assign_moving_average(moving_mean,
                                                mean, decay=momentum)
        update_move_variance = moving_averages.assign_moving_average(moving_variance,
                                                variance, decay=momentum)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_move_mean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_move_variance)
    else:
        mean, variance = moving_mean, moving_variance
    return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)
```
3. tf.keras.layers.BatchNormalization()  
   使用keras高阶API
```python
x = tf.keras.layers.BatchNormalization()(x)
```



