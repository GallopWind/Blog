# 泛化 数据 训练 特征 

## LR

$$
y:实际值\quad y^{'}:预测值\quad \theta^{T}:模型参数向量\quad X:特征向量\ \\
m:样本数量\quad n:特征数量\quad \eta:学习率 \\
\begin{gather}
y^{'}=h_{\theta}(X)=\theta^{T}\cdot X \tag{1} \\
MSE(h_{\theta},X)=\frac{1}{m}\sum_{i}^{m}\big(\theta^T\cdot X^i-y^i\big)^2 \tag{2} \\
标准方程法:\ \theta^{'}=(X^T\cdot X)^{-1}\cdot X^T\cdot y  \tag{3} \\
\nabla_{\theta}MSE(\theta) = \frac{2}{m}X^T\cdot(X\cdot\theta-y) \tag{4} \\
梯度下降法:\theta^{k+1}=\theta^{k}-\eta\nabla_{\theta}MSE(\theta) \tag{5} \\

\end{gather}
$$

### 标准方程法

算法复杂度 O(n^{2.4}, m), 对于样本多, 优; 对于特征多, 劣.

### 梯度下降法

关键: 运算速度(小批量下降), 局部最优(随机初始化), 收敛最优(针对随机梯度).

批量梯度下降法: 使用全部样本, 可以收敛最优, 但样本较多时计算较慢.

随机梯度下降: 每次随机使用一个样本进行梯度下降. 优点: 速度快, 可以核外运算, 有助于跳出局部最优. 缺点: 训练过程中下降不稳定, 需要使用退火算法, 收敛速度慢.

小批量梯度下降: 每次使用一批样本进行训练. 二者的折中.

## 归一化, 正则化

### 归一化

将输入数据进行中心, 方差归一化处理

意义:

- 正则化需要模型参数处于同一尺寸
- 梯度下降需要下降参数区间位于一定范围内

### 正则化

在训练时对模型参数进行惩罚, 依据不同的惩罚方式, 使模型平滑稀疏, 提高泛化性能
$$
\alpha:正则化系数 \quad \theta:模型参数 \quad r:混合比例 \\
\begin{align}
L1范数&:\alpha\sum|\theta| \\
L2范数&:\alpha\sum\theta^2 \\
弹性网络&:r\alpha\sum|\theta|+\frac{1-r}{2}\alpha\sum\theta^2
\end{align}
$$

#### L1正则化

由于导数不随$\theta$ 的值而变化, 因此L1正则化在"谷底"区域更倾向于将$\theta$ 正则化至0, 使模型稀疏.

注意由于在0处不可导, 需要指定该处梯度为0

#### L2正则化

常用正则化

#### 弹性网络正则化

根据超参数$r$ 来调整.

#### 其他正则化

- 提前停止法, 当学习曲线下降再上升一段时间后, 选择最低处的模型作为训练结果.

- 约束法, 对模型的超参数进行约束, 限制其表征能力.

- 修建法, 如剪枝.

## 降维

### 高维与流形

- 高维空间稀疏, 各元素更接近于边界.

- 流形假设: 大多数情况下, 高维度数据存在低维度数据的流形表示.

- 优先级: 投影>流形>核PCA

### PCA

循环投影到方差最大的轴, 根据方差解释率决定保留哪些轴. 

增量PCA用于计算大型数据集, 可以批次计算.

### kPCA

对数据施加核函数变换, 可以根据不同的核展开集群数据, 扭曲流形等分布特征的高维数据.

### LLE(局部线形嵌入)

对数据的点寻找其临近点, 并尝试用局部线形关系关联二者, 求解整体最近似线形关系来关联所有点. 适用于以一定几何关系发展的线状流形, 对大型数据集而言, 难以求解.


## SVM

$$
简写,向量点积不再转置\\
\gamma:样本到超平面距离 \quad x_i,y_i:样本特征与值 \quad \boldsymbol{w}:高维"斜率" \quad b:高维"偏置" \\
 L(\boldsymbol{w}, b, \boldsymbol{\alpha}):拉格朗日函数,约束转化为维度 \\
\begin{gather}
y_i' = step(\boldsymbol{w}\boldsymbol{x_i}+b) \\
\gamma_{i}=y_{i}\left(\frac{\boldsymbol{w}}{\|\boldsymbol{w}\|} \cdot \boldsymbol{x}_{i}+\frac{b}{\|\boldsymbol{w}\|}\right) \tag1 \\
y_{i}\left(\frac{\boldsymbol{w}}{\|\boldsymbol{w}\|} \cdot \boldsymbol{x}_{i}+\frac{b}{\|\boldsymbol{w}\|}\right) \geq \gamma \tag2 \\
\boldsymbol{w'}=\frac{\boldsymbol{w}}{\|\boldsymbol{w}\| \gamma} \quad b'=\frac{b}{\|\boldsymbol{w}\| \gamma} \tag3 \\
y_{i}\left(\boldsymbol{w'} \cdot \boldsymbol{x}_{i}+b'\right)-1 \geq 0 \tag4 \\ \\
\boldsymbol{w'} \to \boldsymbol{w}, \quad b'\to b \\
\min_{\boldsymbol{w},b}\ L(\boldsymbol{w}, b, \boldsymbol{\alpha})=\frac{1}{2}\|\boldsymbol{w}\|^{2}-\sum_{i=1}^{N} \alpha_{i}\left(y_{i}\left(\boldsymbol{w} \cdot \boldsymbol{x}_{i}+b\right)-1-\beta_i^2\right) \tag{5.1}
\end{gather}
$$

### 原理

考虑线性可分情况下, 通过超平面分类. 为了获得泛化性能, 要求分类间隔尽可能大. 即求式(1)中$\max_{\boldsymbol{w},b}min \gamma_i$ , 由于$i$ 不具备可解性, 将其转换为约束条件式(2), 化为约束式(4)优化问题. 由式(3)可得, 求$\gamma$ 最大值即为求$||\boldsymbol{w}||$ 最小值, 由拉普拉斯方法可将问题化规式(5.1). 考虑到解的复杂度和约束条件, 不使用式(5.1)的松弛变量法求解. 化为如下式(5.2).

*注: 当样本数量极多时,倾向于使用原始问题求解. 对偶问题的复杂度介于$O(m^2),O(m^3)$ 之间.*

### 求解


$$
\alpha_i = \left\{\begin{array}{1}
0 &y_{i}\left(\boldsymbol{w} \cdot \boldsymbol{x}_{i}+b\right)-1 \geq 0\\
+\infty &y_{i}\left(\boldsymbol{w} \cdot \boldsymbol{x}_{i}+b\right)-1 \lessdot 0
\end{array}\right. :拉格朗日算子 \quad  \\
\begin{gather}
\min_{\boldsymbol{w},b}\ L(\boldsymbol{w}, b, \boldsymbol{\alpha})=\frac{1}{2}\|\boldsymbol{w}\|^{2}-\sum_{i=1}^{N} \alpha_{i}\left(y_{i}\left(\boldsymbol{w} \cdot \boldsymbol{x}_{i}+b\right)-1\right) \tag{5.2} \\
\theta(w) = \max _{\alpha_{i} \geq 0} L(\boldsymbol{w}, b, \boldsymbol{\alpha})=
\left\{\begin{matrix} 
  \frac{1}{2}||\boldsymbol{w}^2||&,\boldsymbol{x}\in 可行区域 \\
  +\infty&,\boldsymbol{x}\in不可行区域
\end{matrix}\right. \\ \\
\min _{\boldsymbol{w}, b} \theta(\boldsymbol{w})=\min _{\boldsymbol{w}, b} \max _{\alpha_{i} \geq 0} L(\boldsymbol{w}, b, \boldsymbol{\alpha})=p^{*} \tag6 \\
\max _{\alpha_{i} \geq 0} \min _{\boldsymbol{w}, b} L(\boldsymbol{w}, b, \boldsymbol{\alpha})=d^{*} 
\tag7 \\
\left\{\begin{array}{l}
\alpha_{i} \geq 0 \\
y_{i}\left(\boldsymbol{w}_{i} \cdot \boldsymbol{x}_{i}+b\right)-1 \geq 0 \\
\alpha_{i}\left(y_{i}\left(\boldsymbol{w}_{i} \cdot \boldsymbol{x}_{i}+b\right)-1\right)=0
\end{array}\right. \tag8 \\ \\
\arg _{\boldsymbol{w}, b}L(\boldsymbol{w}, b, \boldsymbol{\alpha}) = 
\left\{\begin{array}{}
\boldsymbol{w}=\sum_{i=1}^{N} \alpha_{i} y_{i} \boldsymbol{x}_{i} \\
\sum_{i=1}^{N} \alpha_{i} y_{i}=0
\end{array}\right. \tag9
\end{gather}
$$

通过添加拉普拉斯算子$\alpha_i$ ,并取值使$L(\boldsymbol{w}, b, \boldsymbol{\alpha})$ 成为对偶函数. 由于$\alpha_i$ 取值, 其定义可转化为对式(5.2)求最大值. 此时式(6), 式(7) 相等的前提为, 为凸函数, 且满足KKT条件, 如式(8) 所示.求解得式(9), $\boldsymbol{w}$ 与$\alpha_i$ 取值有关.

$$
\begin{gather}
\begin{array}{c}
L(\boldsymbol{w}, b, \boldsymbol{\alpha})=\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(\boldsymbol{x}_{i} \cdot \boldsymbol{x}_{j}\right)-\sum_{i=1}^{N} \alpha_{i} y_{i}\left(\left(\sum_{j=1}^{N} \alpha_{j} y_{j} \boldsymbol{x}_{j}\right) \cdot \boldsymbol{x}_{i}+b\right)+\sum_{i=1}^{N} \alpha_{i} \\
=-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(\boldsymbol{x}_{i} \cdot \boldsymbol{x}_{j}\right)+\sum_{i=1}^{N} \alpha_{i}
\end{array} \tag{10} \\
\min _{\boldsymbol{\alpha}} \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(\boldsymbol{x}_{i} \cdot \boldsymbol{x}_{j}\right)-\sum_{i=1}^{N} \alpha_{i} \tag{11} \\
\end{gather}
$$

带入可得式(10), 求式(11), 结合式(9)中第二个约束, 可采用SMO方法求解(为满足式(9)约束, 每次根据偏差取两个值进行梯度下降), 也可通过二次规划问题求解器求解. 求解得到$\alpha_i$, 进而得到$\boldsymbol{w},b$ 解. 
$$
\min _{\boldsymbol{w}, b, \xi_{i}} \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m} \xi_{i} \\
y_{i}\left(\boldsymbol{w} \cdot \boldsymbol{x}_{i}+b\right) \geq 1-\xi_{i}
$$
通过观察式(8)第三个解的必要条件, 可知当$\alpha_i = 0$ 时, 该样本不影响解. 反之, 则该样本在边界上, 称之为"支持向量". 如果对SVM做软间隔分类,则求解以上问题, $C$ 为正则系数.

### 核技巧

$$
\begin{gather} 
\boldsymbol{x} \to \phi(\boldsymbol{x}) \\
\left(\boldsymbol{x}_{i} \cdot \boldsymbol{x}_{j}\right) \to \left(\phi(\boldsymbol{x}_{i}) \cdot \phi(\boldsymbol{x}_{j}\right)) \to K(\boldsymbol{x}_{i} \cdot \boldsymbol{x}_{j}) \tag{12} \\ \\ 
\begin{aligned}
线形核函数 &:& K(a,b) &= a\cdot b \\
多项式核函数 &:& K(a,b) &= (\gamma a^T\cdot b +r)^d \\
RBF核函数 &:& K(a,b) &= exp(-\gamma||a-b||^2) \\
Sigmoid核函数 &:& K(a,b) &=tanh(\gamma a^T \cdot b +r)
\end{aligned} \\ \\
例:二阶核函数 \phi 
\left(\begin{array}{1} x_1 \\ x_2
\end{array}\right) = \left(\begin{gathered}
x_1^2 \\ \sqrt{2}x_1x_2 \\ x_2
\end{gathered}\right) \\
K(a,b) = \phi(a)\phi(b) = (a \cdot b)^2 \\ \\
y' \to y' = \boldsymbol{w} \cdot \phi(\boldsymbol{x})+b  \tag{13}\\
\begin{aligned}
y' &= \boldsymbol{w} \cdot \phi(\boldsymbol{x})+b \\
&= \sum_{i=1}^{N} \alpha_{i} y_{i} \phi(\boldsymbol{x}_{i}) \cdot \phi(\boldsymbol{x})+b \\
&= \sum_{i=1}^{N} \alpha_{i} y_{i}K(\boldsymbol{x}_{i},\boldsymbol{x}) +b
\end{aligned}
\end{gather}
$$

为了提高SVM的非线性分类能力, 对$\boldsymbol{x}$ 施加核变换, 如例所示. 根据求解公式(11), 结合式(12)可知, 可使用"核技巧" 加速计算. 下面列出了常用的"核", 将特征映射到更高维的特征空间, 同时根据式(9) 可知, $\boldsymbol{w}$ 的维度同时也需映射. 需要注意的是, RBF高斯将映射到无限维空间. 此时,  考虑到预测式将作式(13) 变化, 做如下变换即可不必求出 $\boldsymbol{w}$ 值, $b$ 的求值同理.

## 决策树

### 信息理论

$$
p_i:节点k类别中i类的真实概率 \quad q_i:节点k类别中i类的预测概率 \\
\begin{align}
基尼不纯度&:G=\sum_i p_i (1-p_i)=1-\sum_{i}p_i^2 \\
信息熵&:H =\sum_i p_i \cdot log_2 \frac{1}{p_i} = -\sum_i p_i \cdot log_2 p_i \\
交叉熵&:H = -\sum_i p_i \cdot log_2 q_i
\end{align}
$$

基尼不纯度: 衡量系统的混乱度, 其值为样本预测错误率的期望. 系统越混乱, 则不纯度越大.

信息熵: 衡量系统的信息量, 其值为样本分类编码长度的期望. 系统类别越多, 则信息熵越大.

交叉熵: 衡量先验下系统的信息量, 其值为样本分类按先验编码的真实期望. 若先验符合真实分布, 则与信息熵想等; 若不符合, 增加的部分为KL散度, 多用来表征系统预测先验的正确度.

*对成本函数的根本要求在于能够有效区分不同预测的效果. 值得注意的是, 针对不同的应用场景,有不同的要求. 例如 : 当边缘样本预测重要性大时, 要加大对该样本预测错误的惩罚.*

### 训练

$$
J:成本函数 \quad m:当前样本数量 \quad m_{left},m_{right}:左右分支样本数量 \\
G:基尼不纯度 \quad H:信息熵 \\
J = \frac{m_{left}}{m}G_{left}+\frac{m_{right}}{m}G_{right} ,(or\ H)
$$

对于分类树, 一般采用CART(classification and regression Tree)方法, 每个枝节点搜索特征与分割值, 使$J$ 下降最大, 直到一定程度.

***如何搜索??***
$$
i:节点样本i,\ m:节点样本总量 \quad y_i:样本i的值 \quad y_i':样本i的预测值 \\
\begin{gather}
J = \frac{m_{left}}{m}MSE_{left}+\frac{m_{right}}{m}MSE_{right} \\
MSE = \sum_{i}(y_i'-y_i)^2 \\
y_i' = \frac{1}{m}\sum_{i}y_i
\end{gather}
$$
对于回归树, 同上.

### 不稳定性

1. 对数据旋转敏感, 因此有必要采用PCA.
2. 对分割边界过分敏感, 可采取集成算法.

### 正则化

超参数: 最大深度, 最小节点样本数量, 最大叶节点数量, 分裂时最大评估特征数量, 剪枝

## 集成学习

### stacking

训练SVM, LR, RF等模型, 并综合他们的结果. 硬投票器, 直接对结果求均值. 软投票器, 模型输出"确定性参数", 对于确定性高的预测赋予更大的权重. stacking方法, 训练模型来综合结果.

### bagging

用不同的数据训练同一模型, 并集成结果.

*bagging方法是随机放回抽样, pasting方法是不放回抽样(bootstrap), 通常 pasting劣于bagging*

包外评估, 随机放回抽样在样本较多时只抽取63%数据, 未抽取的数据用来包外评估准确率. 一般情况下, 集成模型的准确率接近保外评估准确率的均值.

*random patch*, 对特征与样本都进行抽样. *随机子空间法* , 对特征进行抽样. 

*bagging 方法一般对子模型正则化要求较高*

#### Random Forest

采取random patch 法训练.

极端随机树(**Extrema Trees**): 分裂时对特征使用随机分割值, 速度更快, 效果不差.

*可用来判断特征重要性, 以特征在森林中的平均深度为据*

*可以并行运行*

### Boosting

#### AdaBoost

$$
w_i:样本权重 \quad x_i,y_i:样本值 \quad y_i':预测值 \quad r:子模型误差率 \quad \alpha_j:子模型j权重 \quad \eta:学习率 \\
\begin{gather}
r = \frac{\sum_{y_i'\neq y_i}w_i}{\sum{w_i}} \\
\alpha_j=\eta\  log \frac{1-r}{r} \\
w_i \to w_iexp(\alpha_j) \quad ,y_i' \neq y_i \\
y_i^{res} = \sum \alpha_j y_i'
\end{gather}
$$

初始化样本实例权重为$\frac{1}{m}$ , 训练子模型j , 计算子模型误差率$r$ 和子模型权重$\alpha_j$ , 根据$\alpha_j$更新样本权重$w_i$ (需归一化), 训练子模型j+1, 直到终止.预测输出为各子模型加权集成.

#### GBDT

$$
x_i,y_i:样本数据 \quad y_{i,j}:第j代子模型的训练值 \quad y_{i,j}':第j代子模型的预测值 \\
y_{i,j} = y_{i,j-1} - \eta\  y_{i,j-1}' \quad \eta:学习率超参数
$$

以上一代子模型预测的残差作为下一代子模型的训练值, 通过调节$\eta$ 来控制学习速度, 早代负责拟合大趋势, 晚代负责处理细节.

*早代树节点适合作为深度特征* 



 