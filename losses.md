# LOSS总结
## 分类任务
符号说明：
- $y:$ 预测值
- $\hat y:$ 真实值
- $y_i:$ 多分类任务中第$i$类的预测值
### 二分类交叉熵损失 sigmoid_cross_entropy
$$ sigmoid(y)=\frac1 {1+e^{-y}}$$
$$ L(y,\hat{y})=-\frac 1 2 \times (\hat y \times \log(sigmoid(y))+(1-\hat y)\times \log(1-sigmoid(y)))$$
TF接口：
```python
tf.losses.sigmoid_cross_entropy(
    multi_class_labels,
    logits,
    weights=1.0,
    label_smoothing=0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
)

tf.nn.sigmoid_cross_entropy_with_logits(
    _sentinel=None,
    labels=None,
    logits=None,
    name=None
)
```
keras接口：
```python
keras.losses.binary_crossentropy(y_true, y_pred)
```

### 多分类交叉熵损失 softmax_cross_entropy
> m个数据，n分类。有
$$\hat y_i=
\left\{\begin{matrix}
 0& ,正确分类\\ 
 1& ,错误分类
\end{matrix}\right.
$$
$$softmax(y_i)=\frac {e^{y_i}} {\sum_{j=1}^ne^{y_j}}$$
$$\begin{aligned}
L(y,\hat y)
&=-\frac 1 m \sum_{i=1}^m\hat y_i \times log(softmax(y_i))\\
&= -\frac{1}{m}\sum_{i=1}^m\hat y_i \times\log\frac{e^{W_{y_i}^Tx_i+b_{y_i}}}{\sum_{j=1}^ne^{W_j^Tx_i+b_j}}\\
&= -\frac{1}{m}\sum_{i=1}^m\log\frac{e^{W_{y_i}^Tx_i+b_{y_i}}}{\sum_{j=1}^ne^{W_j^Tx_i+b_j}}
\end{aligned}$$
TF接口：
```python
tf.losses.softmax_cross_entropy(
    onehot_labels,
    logits,
    weights=1.0,
    label_smoothing=0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
)
tf.nn.softmax_cross_entropy_with_logits(
    _sentinel=None,
    labels=None,
    logits=None,
    dim=-1,
    name=None
)
tf.nn.sparse_softmax_cross_entropy_with_logits(
    _sentinel=None,
    labels=None,
    logits=None,
    name=None
)
```
keras接口：
```python
# one-hot 标签
keras.losses.categorical_crossentropy(y_true, y_pred)
# 稀疏标签0,1,2...
keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
```

### focal loss
由何凯明提出[(论文)](https://arxiv.org/pdf/1708.02002.pdf)，主要用于解决多分类任务中样本不平衡的现象，可以获得比softmax_cross_entropy更好的分类效果。
> m个数据，n分类。  
$$softmax(y_i)=\frac {e^{y_i}} {\sum_{j=1}^ne^{y_j}}$$
$$L_{focal\_loss}(y,\hat y)=-\frac 1 m \sum_{i=1}^m\hat y \times \alpha_i \times (1-softmax(y_i))^\gamma \times log(softmax(y_i))$$
论文中$\alpha=0.25$, $\gamma=2$效果最好


### dice loss
图像分割中的sigmoid+dice loss, 比如v-net，只适合二分类，也可推广至多分类，直接优化评价指标。  
TF接口：
```python
def dice_coefficient(y_true_cls, y_pred_cls):
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls )
    union = tf.reduce_sum(y_true_cls ) + tf.reduce_sum(y_pred_cls) + eps
    loss = 1. - (2 * intersection / union)
    tf.summary.scalar('classification_dice_loss', loss)
    return loss
```

### 合页损失 hinge_loss
也叫铰链损失，是svm中使用的损失函数。  
由于合页损失优化到满足小于一定gap距离就会停止优化，而交叉熵损失却是一直在优化，所以，通常情况下，交叉熵损失效果优于合页损失。
$$L(y,\hat y)=\frac 1 n \sum_{i=1}^nmax(0,1-\hat y \times y_i)$$
TF接口：
```python
tf.losses.hinge_loss(
    labels,
    logits,
    weights=1.0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
)
```
keras接口：
```python
keras.losses.hinge(y_true, y_pred)
```

### Connectionisttemporal classification(ctc loss):
对于预测的序列和label序列长度不一致的情况下，可以使用ctc计算该2个序列的loss，主要用于文本分类识别和语音识别中。  
TF接口：
```python
tf.nn.ctc_loss(labels, inputs, sequence_length)
```

### KL散度
$$L_{kl}(y,\hat y)=\sum_{i=1}^n\hat y\times log(\frac {\hat y_i} y_i)=\sum_{i=1}^n y_i\times log(\frac {y_i} {\hat y_i})$$
$$\begin{aligned}
L_{kl}(y,\hat y)&=\sum^n_{i=1}\hat y\times log(\frac {\hat y_i} y_i)\\
&=\sum^n_{i=1}\hat y_i\times(log(\hat y_i)-log(y_i))\\
&=\sum^n_{i=1}\hat y_i\times log(\hat y_i)-\sum^n_{i=1}\hat y_i\times log(y_i)\\
&=\sum^n_{i=1}\hat y_i\times log(\hat y_i)+n\times softmax\_cross\_entrop(y,\hat y)
\end{aligned}$$
从上面式子可以看出，kl散度，也就是相对熵，其实就是交叉熵+一个常数项.  
TF接口：
```python
tf.distributions.kl_divergence(
    distribution_a,
    distribution_b,
    allow_nan_stats=True,
    name=None
)
 
tf.contrib.distributions.kl(
dist_a,
    dist_b,
    allow_nan =False,
    name=None
)
```
### 中心损失
### 人脸中三个loss

## 回归任务
### 均方误差 mean square error(MSE) 和 L2范数
MES表示了预测值与目标值之间差值的平方和然后求平均  
$$MSE=\frac 1 n\sum^n_{i=1}(y_i-\hat y_i)$$
L2损失表示了预测值与目标值之间差值的平方和然后开更方，L2表示的是欧几里得距离。  
$$L2=\sqrt{\sum^n_{i=1}(y_i-\hat y_i)^2}$$
MSE和L2的曲线走势都一样（是一个凸函数，U字形）。区别在于一个是求的平均np.mean()，一个是求的更方np.sqrt()  
TF接口：
```python
tf.losses.mean_squared_error(
    labels,
    predictions,
    weights=1.0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
)
tf.metrics.mean_squared_error(
    labels,
    predictions,
    weights=None,
    metrics_collections=None,
    updates_collections=None,
    name=None
)
```
keras接口:
```python
keras.losses.mean_squared_error(y_true, y_pred)
```

### 平均绝对误差 mean absolution error(MAE) 和L1范数
MAE表示了预测值与目标值之间差值的绝对值然后求平均  
$$MAE=\frac 1 n \sum^n_{i=1}|y_i-\hat y_i|$$
L1范数表示了预测值与目标值之间值得绝对值，L1范数也叫曼哈顿距离。  
$$L1=\sum^n_{i=1}|y_i-\hat y_i|$$
MAE和L1的区别在于一个求了均值np.mean()，一个没有求np.sum()。2者的曲线走势也是完全一致的（是一个V字形）。  
TF接口：
```python
tf.metrics.mean_absolute_error(
    labels,
    predictions,
    weights=None,
    metrics_collections=None,
    updates_collections=None,
    name=None
)
```
keras接口：
```python
keras.losses.mean_absolute_error(y_true, y_pred)
```
> 对比：MAE损失对于局外点更鲁棒，但它的导数不连续使得寻找最优解的过程低效；MSE损失对于局外点敏感，但在优化过程中更为稳定和准确。  

### Huber loss和smooth L1
Huber loss具备了MAE和MSE各自的优点，当δ趋向于0时它就退化成了MAE,而当δ趋向于无穷时则退化为了MSE。  
$$
huber\_loss_\delta(y,\hat y)=
\left\{\begin{matrix}
\frac 1 2\sum^n_{i=1}(y_i-\hat y_i)&,|y_i-\hat y_i|\leq \delta\\
\delta \times \sum^n_{i=1}|y_i-\hat y_i|-\frac 1 2\delta^2&,otherwise
\end{matrix}\right.
$$

## reference：
- [损失函数loss大大总结](https://blog.csdn.net/qq_14845119/article/details/80787753)