---
layout: article
title: GNN 教程：图神经网络框架和他们的设计理念对比
key: GNN_tutorial_framework_comparison
tags: GNN
category: blog
pageview: true
date: 2020-04-06 12:00:00 +08:00
---

## 引言

**此为原创文章，未经许可，禁止转载**

最近我们开源了我们在阿里内部场景上使用的超大规模图神经网络计算框架 [graph-learn](https://github.com/alibaba/graph-learn)，graph-learn作为从业务实践角度出发而孵化的GNN框架，原生支持单机多卡，多机多卡，CPU、GPU等分布式集群的超大规模图数据的存储、调度与计算。与此同时，还有很多优秀的图计算框架也已经开源并仍活跃地维护着，他们包括Amazon AI lab的 [DGL](https://www.dgl.ai/)，Matthias Fey 博士的 [pytorch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html)等。我阅读了这些框架的文档，整理一篇文章介绍下各个框架的设计理念以及一些可以互相借鉴的地方。

## Pytorch_geometric (下文将简写成PyG)

Pytorch_geometric 是我最早接触的GNN框架，它将GNN的更新用下面的抽象表示：
$$
\mathbf{x}_{i}^{(k)}=\gamma^{(k)}\left(\mathbf{x}_{i}^{(k-1)}, \square_{j \in \mathcal{N}(i)} \phi^{(k)}\left(\mathbf{x}_{i}^{(k-1)}, \mathbf{x}_{j}^{(k-1)}, \mathbf{e}_{j, i}\right)\right)
$$
其中$\square$用来表示聚合函数，如max，mean等，$\gamma$和$\phi$用来表示一个可微的变换函数，比如多层感知机MLPs。$\mathbf{x}$表示节点的表征，可以是最初的输入特征，或者是更新之后的embeddings。$e$表示边上的特征，总结起来，节点$i$的更新是由两部分组成，一部分来自其本身，一部分来自其邻居节点和关联的边特征。

​	根据这个模型抽象，作者设计的编程接口是这样的，基类`MessagePassing`提供参数`agg`来实现聚合函数$\square$，基类的两个成员函数`message`和`update`分别对应$\phi^{(k)}\left(\mathbf{x}_{i}^{(k-1)}, \mathbf{x}_{j}^{(k-1)}, \mathbf{e}_{j, i}\right)$和$\gamma^{(k)}\left(\mathbf{x}_{i}^{(k-1)},...\right)$的实现，用户继承`MessagePassing`并实现具体的`message`和`update`函数以实现自定义的GNN，最后通过在`forward`方法中调用`propagate`方法驱动整个计算流程的进行。

​	为了说明这样的编程抽象接口对于大多数GNN算法都是通用的，作者在[文档](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html)中以GCN layer和EdgeConv layer为例子，详细说明了实现的步骤。PyG也提供了非常多的已实现的[模型](https://github.com/rusty1s/pytorch_geometric/tree/master/examples)作为通用性的证明。

​	在编程模型中，PyG模型的一个特点是需要显示传递边信息的邻接表，编程中用$e$来表示，它是一个$[2, \textrm{num_edges}]$的二维矩阵，显然，对于超大规模的数据，这样一个矩阵在内存中是存不下的，因此作者提供了split的策略，通过内建的`torch_geometric.data.Dataset`接口，PyG能自动将一个大图切分成多个小图，每个小图用上述的逻辑进行计算，每个小图上的计算仍然需要传递每个小图的边矩阵。

​	PyG目前不提供分布式计算的逻辑，因此大图拆分后的小图计算是串行的，小图内部的计算可以并行起来，因此PyG安装时依赖了torch-cluster, torch-scatter等库。

## Deep Graph Library (下文将简写成DGL)

DGL是目前非常优秀的图计算框架，它将GNN邻居汇聚用”消息传递“这种计算模式统一起来，提供了非常完善的"消息传递"式全图计算框架，DGL 的核心为消息传递机制（message passing），用户在使用的时候可以不考虑当前batch size大小、邻居个数是否对齐等信息，极大得简化了编程流程。

DGL计算的核心是消息函数 （message function）和汇聚函数（reduce function）。如下图所示，假设我们需要更新目的节点的Embedding，那么DGL将计算抽象成了两个部分：

![DGL](https://tva1.sinaimg.cn/large/00831rSTly1gdiya5nj4rj30eh06qq37.jpg)

- 消息函数（message function）：对每个源节点来说，准备他的目的节点需要的信息（比如源节点的Embedding和与目的节点链接的边的Embedding或者weight等），然后把这些信息作为消息传递到目的节点的Mailbox里。如上图所示：对每条边（edge）来说，每个源节点将会将自身的Embedding（src.data）和边的Embedding(edge.data)传递到目的节点；对于每个目的节点来说，它可能会受到多个源节点传过来的消息，它会将这些消息存储在”邮箱”中。
- 汇聚函数（reduce function）：汇聚函数的目的是根据源节点传过来的消息更新更新目的节点Embedding，对目的节点来说，它先从邮箱（Mailbox）中汇聚源节点传递过来的消息（message），并清空邮箱（Mailbox）内消息；然后目的节点结合汇聚后的结果和原Embedding以做一次Embedding更新。

根据这两个抽象，利用DGL的编程接口实现一个自定义图模型只需要提供两个函数，即`message_function`和`reduce_function`，前者用来指导如何对源节点的数据和边数据进行选择与加工然后传递到目的节点的邮箱；后者用来指导目的节点如何利用邮箱中它的邻居传递过来的消息和自身Embedding进行融合以更新自身Embedding。最后通过在`forward`函数中调用`update_all(message_function, reduce_function)`来驱动整个计算流程的进行。DGL一个非常明显的优势是整个编程的抽象中不需要传递任何关于图结构的信息，即不需要边的邻接矩阵或者邻接表，边结构相关的信息在数据载入的过程中被底层的框架捕获并对上提供所需要的信息(主要是邻居的查询)，不需要用户在编程中进行干预。

DGL提供了详细的[教程](https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html)解释了如何利用消息传递的机制实现GCN，为了证明这种编程抽象是合理的，它也提供了很多已实现的[模型](https://github.com/dmlc/dgl/tree/master/examples/pytorch)。

这种”消息传递“的机制需要全图被预先载入到内存中，因此不适合在大规模数据集上训练，为了应对超大规模数据的问题，DGL团队采用和GraphSAGE类似的思路，将一个batch内所需要更新目的节点相关的信息一次性提取出来，构成”计算子图“，称为“NodeFLow”。计算子图被载入到内存中通过"消息传递“这样的方式计算，子图和子图之间的采样和更新都是可并行的。在最新的0.4.3版本中，DGL已经支持了单机多卡的并行，和多机cpu版本的分布式计算。

## Graph-learn （下文将简写成GL)

GL是从阿里内部业务场景出发抽象出来的图神经网络框架，首要的目的是为了解决内部场景应用时候要面对的超大规模图数据的问题，我们面临的图数据规模从几千万到几百亿不等，面对如此大的数据规模，在存储上我们采用分布式的方案，在计算上我们采用了和GraphSAGE类似的子图计算架构，即把一个batch内所需要更新目的节点相关的信息一次性提取出来，构成计算子图，以batch为单位更新梯度，我们把这个流程称为sampling。整个存储和计算都是可以是分布式的，在实际场景上，我们已经实现了单机多卡，多机多卡、CPU、GPU等分布式集群的训练和预测。

我们将GNN的算法流程抽象成以下步骤：

![3](https://tva1.sinaimg.cn/large/007S8ZIlly1gdyy2iset7j30yk0iiaeb.jpg)

其中AGGREGATE的功能类似于PyG中的$\square$和$\phi$，而COMBINE的功能类似于PyG中的$\gamma$，而SAMPLE就是用来返回节点邻居的函数，实现上SAMPLE内部进行了非常多的系统端优化，以至于整个采样的时间相较于训练基本可以忽略不计。当然，虽然它的名字叫SAMPLE采样，但是对于像GCN和GAT这种需要全部邻居参与计算的模型，我们也提供了能够返回全部邻居的接口。对于一个batch的数据，由于每个节点的邻居数据是不定的，这时候全邻居SAMPLE的返回结果将会被封装成一个SparseTensor，并为每个源节点提供必要的邻居定位信息segment_ids以供下游算法使用。

对于每个采样出来的计算子图，我们的编程接口是这么设计的：每个图算法最核心的卷积层更新被统一成

```
def forward(self, self_vecs, neigh_vecs, segment_ids)
```

这样的形式，其中self_vecs是目的节点自身的Embedding，而neigh_vecs是源节点的Embedding，用户需要在这个函数中自定义自己的计算逻辑，即这个函数即做了AGGREGATE的工作，也做了COMBINE的工作，函数的输出即是目的节点更新后的Embedding。segment_ids是在全邻居采样返回的neigh_vecs为SparseTensor时定位每个目的节点的neigh是哪几个用到的，是这里提供一个GCN conv layer的[例子](https://github.com/alibaba/graph-learn/blob/master/graphlearn/python/model/tf/layers/gcn_conv.py)。

定义完卷积层之后，对每个自定义的图算法，我们提供了`LearningBasedModel`这个基类来处理和训练相关的东西，用户需要重写`_encoders`这个函数提供对于多跳的邻居如何逐层更新Embedding的逻辑（即如何堆叠之前实现的conv layer)，最后重写`build`函数，以驱动整个计算流程的进行。

我们提供了详细的[教程](https://github.com/alibaba/graph-learn/blob/master/docs/quick_start.md)解释了如何graph-learn实现GCN，也提供了很多已实现的[模型](https://github.com/alibaba/graph-learn/tree/master/examples/tf)。

graph-learn架构、原理等一些其他的参考文档：

https://yq.aliyun.com/articles/752628

https://yq.aliyun.com/articles/752630

https://yq.aliyun.com/articles/752637

https://yq.aliyun.com/articles/752645



