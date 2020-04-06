# 从谱图理论的角度谈谈GCN

## spatial domain 和spectral domain

spatial domain，翻译过来就是空间域，是最直观感受GCN逐层传播算法的域，即：节点$v$的Embedding是其所有邻居节点Embedding(包括自己的Embedding)的聚合结果。因此在一些文献上spatial domain又被称做"vertex domain"。但是与CNN所应用的图像不同，图数据中节点邻居的数目不是一定的，这就产生了一个问题，对于不同邻居个数的节点，卷积怎么定义呢？这就引出了spectral domain的概念。spectral domain，即频谱域，旨在借助谱图理论(Spectral Graph Theory)的一些基础实现拓扑图上的卷积操作。

谱图理论主要研究的是图的拉普拉斯特征值和所对应的特征向量对于图拓扑性质的影响。

## 拉普拉斯矩阵

### 定义

拉普拉斯矩阵是定义在无向图上的，其定义如下：

对于无向图图$$\mathcal{G}=(\mathcal{V}, \mathcal{E})$$，其Laplacian矩阵定义为$L=D-A$，其中$D$是节点的度矩阵(只在对角线上有元素)，$A$是图的邻接矩阵。拉普拉斯矩阵还有几种扩展定义：

- $$L^{sys}==D^{-1 / 2} L D^{-1 / 2}$$ 称为对称正规拉普拉斯矩阵(Symmetric Normalized Laplacian)，论文中一般用的是这种Laplacian的定义。
- $$L^{rw}=D^{-1}L$$ 称为随机游走正规拉普拉斯矩阵(Random Walk Normalized Laplacian)。

### 性质

那么拉普拉斯矩阵有哪些优良的性质吗？

- 拉普拉斯矩阵是对称矩阵，可以进行特征分解(谱分解)，那么从**拓扑结构的图**到**拉普拉斯矩阵**再到**谱图**这条链路就形成了，我们就得到了一个从spatial domain到spectral domain的路径。
- 拉普拉斯矩阵只在中心顶点和一度邻居有非0元素，其余之处都为0
- 拉普拉斯矩阵可以类比到频谱域的拉普拉斯算子上

### 谱分解

矩阵的**特征分解**，**对角化**，**谱分解**都是同一个概念，是指将矩阵分解为由其特征值和特征向量表示的矩阵乘积的方法。只有含有n个线性无关的特征向量的n维方阵才可以进行特征分解。

拉普拉斯矩阵是半正定矩阵，有如下三个性质：

1. 是对称矩阵，有n个线性无关的向量

2. 半正定矩阵的特征值非负

3. 对称矩阵的特征向量互相正交，即所有特征向量构成的矩阵为正交矩阵

性质1告诉我们拉普拉斯矩阵一定能进行特征分解(谱分解)，有如下形式：
$$
L=U\begin{pmatrix}
\lambda_1 & & \\
& \ddots & \\
&& \lambda_n
\end{pmatrix}U^{-1}
$$
其中$U=(u_1, u_2, \dots, u_n)$ 为列向量$u_i$组成的单位特征向量矩阵，$\begin{pmatrix}\lambda_1 & & \\& \ddots & \\&& \lambda_n\end{pmatrix}$是$n$个特征值组成的对角阵。

由性质3可知，$U$是正交矩阵，即$UU^\top=E$，所以特征分解又可以写成：
$$
L=U\begin{pmatrix}
\lambda_1 & & \\
& \ddots & \\
&& \lambda_n
\end{pmatrix}U^\top
$$

### 图上的傅里叶变换

### 正变换：从spatial域到spectral域

把传统的傅里叶变换迁移到图上来，核心的工作就是把拉普拉斯算子的特征函数$e^{-i\omega t}$变为图对应的拉普拉斯矩阵的特征向量。可以参考[论文](https://arxiv.org/abs/1211.0053)

传统的傅里叶变换定义为：
$$
F(\omega)=\mathcal{F}[f(t)]=\int f(t)e^{-i\omega t}dt
$$
信号$f(t)$与基函数$e^{-i\omega t}$的积分，$e^{-i\omega t}$是拉普拉斯算子的特征函数，$\omega$和特征值有关。

广义的特征值方程定义为：$Au=\lambda u$ ，其中$A$是变换，$u$是特征向量或者特征函数(无穷维的向量)，$\lambda$是特征值。

$e^{-i\omega t}$满足：
$$
\Delta e^{-i \omega t}=\frac{\partial^{2}}{\partial t^{2}} e^{-i \omega t}=-\omega^{2} e^{-i \omega t}
$$

和广义特征值方程对比，$$\Delta$$ 是变换，$e^{-i\omega t}$是特征函数，$\omega$和特征值密切相关。

因此，如果我们在图上运用拉普拉斯矩阵(拉普拉斯矩阵式离散拉普拉斯算子)，自然就是去找拉普拉斯矩阵的特征向量了。

离散积分就是一种內积的形式，仿上定义图上的傅里叶变换：
$$
F\left(\lambda_{l}\right)=\hat{f}\left(\lambda_{l}\right)=\sum_{i=1}^{N} f(i) u_{l}^{*}(i)
$$
其中$f$是图上的$N$维向量，$f(i)$和图上的节点一一对应，$u_l(i)$表示第$l$个特征向量的第$i$个分量。那么特征值(频率)$\lambda_l$下的，$f$的图傅里叶变换就是与$\lambda_l$对应的特征向量$u_l$进行內积运算。

注：上述的內积运算是在复数空间定义的，所以采用了$u_l^*(i)$，也就是特征向量$u_l$的共轭。

利用矩阵乘法将图上的傅里叶变换推广到矩阵形式：
$$
\left(\begin{array}{c}{\hat{f}\left(\lambda_{1}\right)} \\ {\hat{f}\left(\lambda_{2}\right)} \\ {\vdots} \\ {\hat{f}\left(\lambda_{N}\right)}\end{array}\right)=\left(\begin{array}{cccc}{u_{1}(1)} & {u_{1}(2)} & {\dots} & {u_{1}(N)} \\ {u_{2}(1)} & {u_{2}(2)} & {\dots} & {u_{2}(N)} \\ {\vdots} & {\vdots} & {\ddots} & {\vdots} \\ {u_{N}(1)} & {u_{N}(2)} & {\dots} & {u_{N}(N)}\end{array}\right)\left(\begin{array}{c}{f(1)} \\ {f(2)} \\ {\vdots} \\ {f(N)}\end{array}\right)
$$
即$f$在图上的傅里叶变换的矩阵形式为：
$$
\hat{f}=U^\top f
$$

### 逆变换：从spectral域到spatial域

那么，我们将节点在频率域上变换之后，怎么变换回空间域呢？传统的傅里叶逆变换是对频率$\omega$积分：
$$
\mathcal{F}^{-1}[F(\omega)]=\frac{1}{2 \Pi} \int F(\omega) e^{i \omega t} d \omega
$$
迁移到离散域(图上)则为对特征值$\lambda_l$求和：
$$
f(i)=\sum_{l=1}^{N} \hat{f}\left(\lambda_{l}\right) u_{l}(i)
$$
利用矩阵乘法将图上的傅里叶逆变换推广到矩阵形式：
$$
\left(\begin{array}{c}{f(1)} \\ {f(2)} \\ {\vdots} \\ {f(N)}\end{array}\right)=\left(\begin{array}{cccc}{u_{1}(1)} & {u_{2}(1)} & {\dots} & {u_{N}(1)} \\ {u_{1}(2)} & {u_{2}(2)} & {\dots} & {u_{N}(2)} \\ {\vdots} & {\vdots} & {\ddots} & {\vdots} \\ {u_{1}(N)} & {u_{2}(N)} & {\dots} & {u_{N}(N)}\end{array}\right)\left(\begin{array}{c}{\hat{f}\left(\lambda_{1}\right)} \\ {\hat{f}\left(\lambda_{2}\right)} \\ {\vdots} \\ {\hat{f}\left(\lambda_{N}\right)}\end{array}\right)
$$
即$f$在图上的傅里叶逆变换的矩阵形式为：
$$
f=U\hat{f}
$$


### 图上的卷积

上面我们介绍了图上的傅里叶变换和傅里叶逆变换的定义，在上面的基础上，下面我们利用卷积定理推导图上几个卷积运算的形式：

卷积定理：函数卷积的傅里叶变换是函数傅里叶变换的乘积，即对于函数$f(t)$与$h(t)$两者的卷积是其函数傅里叶变换乘积的逆变换：
$$
f(t) * h(t)=\mathcal{F}^{-1}[\hat{f}(\omega) \hat{h}(\omega)]=\frac{1}{2 \Pi} \int \hat{f}(\omega) \hat{h}(\omega) e^{i \omega t} d \omega
$$
因此$f$与卷积核$h$在图上的卷积可按照如下步骤求得：

1. 计算$f$的傅里叶变换：$\hat{f}=U^\top f$

2. 卷积核$h$的在图上的傅里叶变换为：$\hat{h}=U^\top h$，写成对角矩阵形式为：$\begin{pmatrix}\hat{h}(\lambda_1) & & \\& \ddots & \\&& \hat{h}(\lambda_n)\end{pmatrix}$，其中$\hat{h}\left(\lambda_{l}\right)=\sum_{i=1}^{N} h(i) u_{l}^{*}(i)$ 是根据需要设计的卷积核$h$在图上的傅里叶变换。

3. 两者傅里叶变换的乘积为：$\hat{h}\hat{f}=(U^\top h \odot U^\top f)\begin{pmatrix}\hat{h}(\lambda_1) & & \\& \ddots & \\&& \hat{h}(\lambda_n)\end{pmatrix}U^\top f $

4. 再乘以$U$求两者傅里叶变换乘积的逆变换，则求出卷积：
   $$
   (f*h)_\mathcal{G}=U\begin{pmatrix}\hat{h}(\lambda_1) & & \\& \ddots & \\&& \hat{h}(\lambda_n)\end{pmatrix}U^\top f
   $$


### 拉普拉斯矩阵和傅里叶变换的关系

在上述推导中，我们假设拉普拉斯矩阵的特征向量为傅里叶变换的基，而特征向量对应的特征值表示基对应的频率，下面解释下这种对应关系的原因：

#### 特征向量与基的关系

傅里叶变换的一个本质理解是：把任意一个函数表示成若干个正交函数(由$\sin$，$\cos$构成)的线性组合，而图上的傅里叶变换$f=U^\top \hat{f}$ 也把图上定义的任意向量$f$，表示成了拉普拉斯矩阵特征向量的线性组合，即：
$$
f=\hat{f}\left(\lambda_{1}\right) u_{1}+\hat{f}\left(\lambda_{2}\right) u_{2}+\cdots+\hat{f}\left(\lambda_{n}\right) u_{n}
$$
那么，为什么图上任意向量都可以表示成这样的线性组合?

原因是$(u_1, u_2, \cdots, u_n)$是图上$n$维空间中的$n$个线性无关的正交向量，由线性代数的知识可知：$n$维空间中$n$个线性无关的向量可以构成空间的一组即，而且拉普拉斯矩阵的特征向量还是一组正交基。

#### 特征值和频率的关系

## GCN 结构的演进

从上文中定义的图卷积操作
$$
(f*h)_\mathcal{G}=U\begin{pmatrix}\hat{h}(\lambda_1) & & \\& \ddots & \\&& \hat{h}(\lambda_n)\end{pmatrix}U^\top f
$$
来看，我们可以通过改变$h$


































