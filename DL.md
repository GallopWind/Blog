# DNN

<img src="https://github.com/wanderinwind/blog/raw/master/img/DL/DNN_model.jpg" alt="DNN model" style="zoom: 67%;" />
$$
x_m:输入特征 \quad f_a:激活函数 \quad y_m:输出值
$$

## 特征工程

### 数据预处理

一方面, 清洗, 筛选, 采样出模型需要的训练样本. 另一方面, 将特征归一化, 编码, 便于批处理和训练.

*去除无用变量, 共线性变量.*

#### 特征生成

除手动特征生成外, 根据不同的数据类型有相应的特征生成工具.

关系型特征: Deep Feature Synthesis (DFS). Featuretools, 支持基元规则操作.

时序数据: tsfresh

其他方法: GBDT的早期子节点, CNN, DNN的浅层. 分类任务/自动编码器/无监督训练任务训练的模型

#### tsfresh

## 超参数

### 激活函数

<img src="https://github.com/wanderinwind/blog/raw/master/img/DL/activate_func.jpg" style="zoom:75%;" />
$$
\begin{align}
sigmoid(logic)&:& y&=\frac{1}{1+e^{-x}} \\
tanh&:& y&=\frac{1-e^{-2x}}{1+e^{-2x}} \\
relu&:& y&=\left\{\begin{array}{} x &,x  \geq 0 \\0 &, x \lessdot0  \end{array}\right. \\
leaky\ relu&:& y&=\left\{\begin{array}{} x &,x  \geq 0 \\0 &, \alpha x \lessdot0  \end{array}\right. ,\alpha \in [0.02,\ 0.1] \\
ELU&:& y&=\left\{\begin{array}{}  x &,x  \geq 0 \\ \alpha(e^x-1) &, x \lessdot0  \end{array}\right. ,\alpha 一般取1 \\
softmax &:& y_i&= \frac{e^x_i}{\sum_{i} e^x_i} \quad 多用于分类输出层
\end{align}
$$

## 训练

### 梯度消失, 爆炸

在DNN中, 由于Sigmoid激活函数的导数处于0~1, 在底层会变得极小. 当初始化权重较大时, 又会在训练早期导致梯度爆炸.

#### Xavier和He初始化

每一层的连接权重需要按照一定的要求随机初始化, 如给定$\eta =0, \sigma$ 的正态分布或给定$-r,r$ 的均匀分布随机初始化, 以尽可能接近"保持每一层输入与输出的方差一致" 这一目标.
$$
n_{in},n_{out}:激活函数扇入,扇出 \quad ReLU及其变种的初始化又称He初始化 \\
\begin{gather}
sigmoid & r=\sqrt{\frac{6}{n_{in}+n_{out}}} & \sigma=\sqrt{\frac{2}{n_{in}+n_{out}}} \\
tanh & r=\sqrt[4]{\frac{6}{n_{in}+n_{out}}} & \sigma=\sqrt[4]{\frac{2}{n_{in}+n_{out}}} \\
ReLU,leaky relu,ELU & r=\sqrt[\sqrt2]{\frac{6}{n_{in}+n_{out}}} & \sigma=\sqrt[\sqrt2]{\frac{2}{n_{in}+n_{out}}} \\
\end{gather}
$$

#### 非饱和激活函数

ReLU函数在激活区间导数为1, 不会发生消失/爆炸. 但训练过程中会导致部分神经元"死亡". 于是引入变种leaky relu 和ELU, 使其存在昏迷期, 但仍具备苏醒的可能.

一般而言, ELU> leaky relu >relu >tanh >sigmoid .

#### batch normalization

每一层的***激活函数前*** 添加归一化操作, 确保每一层的信息都稳定在有效传递区间.

#### 梯度裁剪

针对RNN, 在梯度下降过程中限制梯度在一定区域内.

#### 无监督预训练

逐层无监督预训练法, 使用非监督特性检测算法, 如受限玻尔兹曼机(RBM) 或自动编码器.

### 优化器

#### Momentum

$$
\beta : 摩擦系数(0.9) \quad m:下降动量 \\
m \leftarrow \beta m + \eta \nabla J(\theta) \\
\theta \leftarrow \theta-m \
$$

下降值为动量(Momentum), 会累加下降梯度. 当遇到大坡区域时, 能够加速度下降. 而越过谷底就会减速. 由于摩擦$\beta$ 的存在最终会停止在谷底.

*有助于越过局部谷底, 因此采取不同的超参数可能会得到不同的结果*

#### NAG(Nesterov Accelerate Momentum)

$$
\beta : 摩擦系数(0.9) \quad m:下降动量 \\
m \leftarrow \beta m + \eta \nabla J(\theta-\beta m) \\
\theta \leftarrow \theta-m \
$$

将要越过谷底时, 由于采取目标位置附近的梯度, 能够提前减速, 减少收敛的震荡, 同时不影响大坡地区的加速, 能提高收敛速度.

#### AdaGrad

$$
s \leftarrow s + \nabla J(\theta)\otimes \nabla J(\theta) \quad (\otimes 表示矩阵点积) \\
\theta \leftarrow \theta- \eta \nabla J(\theta)/\sqrt{s+\epsilon}
$$

$s$ 累积的是$J(\theta)$ 梯度的平方值, 随着累积始终处于下降的维度的值会增长更快, $s$ 会逐渐接近于全局极小值的下降方向. 随后利用该方向收缩梯度, 使局部最速下降方向偏移向全局最速下降方向, 尤其适合细长碗型的下降收敛. 缺点是下降梯度会迅速衰减, 仅适合简单任务.

#### RMSprop

$$
\beta:滤波系数,一般取0.9 \\
s \leftarrow \beta s +(1-\beta) \nabla J(\theta)\otimes \nabla J(\theta) \quad (\otimes 表示矩阵点积) \\ 
\theta \leftarrow \theta- \eta \nabla J(\theta)/\sqrt{s+\epsilon} \
$$

同AdaGrad算法思想, 防止$s$ 增长且保证其方向, 毫无疑问, 滤波就OK了. 效果很好.

#### Adam

$$
\beta_1:动量衰减超参数,0.9 \quad \beta_2:缩放衰减超参数,0.999 \quad T:迭代次数 \quad \eta:取0.001即可 \\
m \leftarrow \beta_1 m+ (1-\beta_1)\nabla J(\theta) \\
s \leftarrow \beta_2s +(1-\beta_2)\nabla J(\theta)\otimes \nabla J(\theta) \\
m \leftarrow \frac{m}{1-\beta_1^T} \\
s \leftarrow \frac{s}{1-\beta_2^T} \\
\theta \leftarrow \theta - \eta m/\sqrt{s+\epsilon}
$$

算法是以上两种思想: 动量与自适应的综合.

#### 其他方法

- 用L1范数正则化训练稀疏模型.
- 学习速率调度, 如模型性能(正确率)调度, 指数调度, 功率调度, 等.

### 正则化

#### dropout

每一次训练时都将每一层$p$ 比例的神经元"丢弃", 训练结束后再将每个神经元的输出$/(1-p)$ 修正. 这一思想有两种解释, 一: 降低模型对某个神经元的依赖性, 将其任务分配给几个神经元. 二: 每次训练都产生了新的模型, 相当于一个简单高效的针对DNN的集成学习.

*个人更赞成这是一种""浅随机, 深训练""的集成学习, 解决了集成学习无法作用于DNN问题*

#### 最大范数正则化 

剪裁法, 每次训练结束后对神经元的输入连接权重进行 $w\leftarrow w\frac{r}{||w||_2}, r最大范数超参数$ 的权重剪裁. 该正则化并不加入成本函数中.

#### 数据扩充

泛化训练样本. 如针对图片, 平移循环对称, 修改背景等.

# CNN
## CNN

## ResNet

# RNN
## RNN

<img src="https://github.com/wanderinwind/blog/raw/master/img/DL/RNN.jpg" style="zoom:67%;" />
$$
h_t = tanh(W \cdot h_{t-1}+U \cdot X + b) \quad 也可使用其他激活函数如ReLU \\
Y = V \cdot h_t + b
$$
RNN的输入与输出均为时序序列输入输出.一方面其本身可作为一个神经单元, 能够根据输入和记忆进行非线性映射, 另一方面其能够保存记忆信息, 能够根据输入对记忆进行修改. 由于其非线性映射能力有限, 因此实际使用时会在输入和输出添加网络来对信息进行映射, 或者使用深层RNN.

<img src="https://github.com/wanderinwind/blog/raw/master/img/DL/RNN_use.jpg" style="zoom:60%;" />

单层RNN的用法. 一: 用于数据实时预测, 输入一个新数据, 给出一个新预测. 二: 序列数据综合输出, 比如输入一段时间数据给出综合评价. 三: 不常用. 四: 延迟的序列到序列输出, 又称编码解码器.

RNN的训练原理为梯度下降, 即根据时间序列展开成DNN训练. 有时为了避免过深, 仅展开有限时间序列, 又称时间截断反向传播. 

*由于长时记忆不佳, 多采用LSTM与GRU*

## LSTM

<img src="https://github.com/wanderinwind/blog/raw/master/img/DL/LSTM.jpg" style="zoom:60%;" />

## GRU

<img src="https://github.com/wanderinwind/blog/raw/master/img/DL/GRU.jpg" style="zoom:50%;" />

