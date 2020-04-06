---
layout: article
title: GNN 教程：DGL框架-大规模分布式训练
key: GNN_Triplets_GCN
tags: GNN
category: blog
pageview: true
date: 2019-06-01 21:00:00 +08:00
---
**此为原创文章，转载务必保留[出处](https://archwalker.github.io)**

## 引言

前面的文章中我们介绍了DGL如何利用采样的技术缩小计算图的规模来通过mini-batch的方式训练模型，当图特别大的时候，我们会生成非常多的batch串行计算，因此运算时间又成了问题，一个容易想到解决方案是采用并行计算的技术，很多worker同时采样，计算并且更新梯度。这篇博文重点介绍DGL的并行计算框架。

## 多进程方案

概括而言，目前DGL(version 0.3)采用的是多进程的方案，分布式的方案正在开发中。见下图，DGL的并行计算框架分为两个主要部分：`Graph Store`和`Sampler`

- `Sampler`被用来从大图中构建许多计算子图(`NodeFlow`)，DGL能够自动的在多个机器上并行运行多个`Sampler`的实例。
- `Graph Store`存储了大图的embedding信息和结构信息，到目前为止，DGL提供了内存共享式的Graph Store，以用来支持多进程，多GPU的并行训练。DGL未来还将提供分布式的Graph Store，以支持超大规模的图训练。

下面来分别介绍它们。

![image](http://ww3.sinaimg.cn/large/006tNc79ly1g4kplvtvkqj31gu0u0ane.jpg)

### Graph Store

graph store 包含两个部分，server和client，其中server需要作为守护进程(daemon)在训练之前运行起来。比如如下脚本启动了一个graph store server 和 4个worker，并且载入了reddit数据集：

```shell
python3 run_store_server.py --dataset reddit --num-workers 4
```

在训练过程中，这4个worker将会和client交互以取得训练样本。用户需要做的仅仅是编写训练部分的代码。首先需要创建一个client对象连接到对应的server。下面的脚本中用`shared_memory`初始化`store_type`表明client连接的是一个内存共享式的server。

```shell
g = dgl.contrib.graph_store.create_graph_from_store("reddit", store_type="shared_mem")
```

在采样的[博文](<https://archwalker.github.io/blog/2019/06/30/GNN-Framework-DGL-NodeFlow.html>)中，我们已经详细介绍了如何通过采样的技术来减小计算子图的规模。回忆一下，图模型的每一层进行了如下的计算：
$$
z_{v}^{(l+1)}=\sum_{u \in \mathcal{N}^{(l)}(v)} \tilde{A}_{u v} h_{u}^{(l)} \qquad h_{v}^{(l+1)}=\sigma\left(z_{v}^{(l+1)} W^{(l)}\right)
$$
[方差控制采样法](https://arxiv.org/abs/1710.10568)用如下的方法近似了$z_v^{(l+1)}$：
$$
\begin{aligned} \hat{z}_{v}^{(l+1)}=& \frac{|\mathcal{N}(v)|}{\left|\hat{\mathcal{N}}^{(l)}(v)\right|} \sum_{u \in \hat{\mathcal{N}}^{(l)}(v)} \tilde{A}_{u v}\left(\hat{h}_{u}^{(l)}-\overline{h}_{u}^{(l)}\right)+\sum_{u \in \mathcal{N}(v)} \tilde{A}_{u \nu} \overline{h}_{u}^{(l)} \\ & \hat{h}_{v}^{(l+1)}=\sigma\left(\hat{z}_{v}^{(l+1)} W^{(l)}\right) \end{aligned}
$$
除了进行这样的近似，作者还采用了预处理的技巧了把采样的层数减少了1。具体来说，GCN的输入是$X$的原始embedding，预处理之后GCN的输入时$\tilde{A}X$，这种方式使得最早的一层无需进行邻居embedding的融合计算(也就是无需采样)，因为左乘以邻接矩阵已经做了这样的计算，因为，需要采样的层数就减少了1。

对于一个大图来说，$\tilde{A}$和$X$都可能很大。两个矩阵的乘法就要通过分布式计算的方式完成，即每一个trainer(worker)负责计算一部分，然后聚合起来。DGL提供了`update_all`来进行这种计算：

```python
g.update_all(fn.copy_src(src='features', out='m'),
             fn.sum(msg='m', out='preprocess'),
             lambda node : {'preprocess': node.data['preprocess'] * node.data['norm']})
```



### Distributed Sampler

