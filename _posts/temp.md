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
其中$U=(u_1, u_2, \dots, u_n)$ 为列向量$u_i$组成的单位特征向量矩阵，$\begin{pmatrix}
\lambda_1 & & \\
& \ddots & \\
&& \lambda_n
\end{pmatrix}$是$n$个特征值组成的对角阵。

由性质3可知，$U$是正交矩阵，即$UU^\top=E$，所以特征分解又可以写成：
$$
L=U\begin{pmatrix}
\lambda_1 & & \\
& \ddots & \\
&& \lambda_n
\end{pmatrix}U^\top
$$

### 图上的傅里叶变换

把传统的傅里叶变换迁移到图上来，核心的工作就是把拉普拉斯算子的特征函数$e^{-i\omega t}$变为图对应的拉普拉斯矩阵的特征向量。可以参考[论文](https://arxiv.org/abs/1211.0053)

传统的傅里叶变换定义为：
$$
F(\omega)=\mathcal{F}[f(t)]=\int f(t)e^{-i\omega t}dt
$$
信号$f(t)$与基函数$e^{-i\omega t}$的积分，$e^{-i\omega t}$是拉普拉斯算子的特征函数，$\omega$和特征值有关。

广义的特征值方程定义为：$Av=\lambda v$ ，其中$A$是变换，$v$是特征向量或者特征函数(无穷维的向量)，$\lambda$是特征值。

$e^{-i\omega t}$满足：
$$
\Delta e^{-i \omega t}=\frac{\partial^{2}}{\partial t^{2}} e^{-i \omega t}=-\omega^{2} e^{-i \omega t}
$$




