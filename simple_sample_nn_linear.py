#!/usr/bin/env python
# coding: utf-8

# ## 基于PaddlePaddle2.0-构建线性回归模型
# 作者: 陆平
# ### 1. 线性回归表达式
# 回归模型是根据输入特征来对连续型输出做出预测的模型。
# 
# 现实生活中，有很多问题可以用回归模型来解决，例如，人们通过构建回归模型来预测房屋售价、制定产品销售计划、研判经济走势等。回归模型可以是线性的，即输入特征分别乘以回归系数（权重）后以加和的方式得到输出。回归模型还可以是非线性的，例如，经济学中常用的Cobb-Douglas生产函数，衡量了企业产出与要素投入之间的关系，其输出是输入特征之间的乘积，该模型是非线性的，但经过对输入、输出特征进行对数化处理后，它可以转化为线性模型。
# 
# 当样本数为$n$，特征数为$k$时，线性回归模型的表达式为：
# 
# $\mathbf{\hat{y}}=\mathbf{Xw}+b$
#    
# 其中，线性回归模型的输出形状大小为$\mathbf{\hat{y}}\in \mathbb{R}^{n \times 1}$，样本特征形状大小为$\mathbf{X}\in \mathbb{R}^{n \times k}$，权重形状大小为$\mathbf{w}\in \mathbb{R}^{k \times 1}$，偏置项为$b\in\mathbb{R}^{1}$，设模型的参数为$\mathbf{\theta}= \left [ \mathbf{w},b\right ]^{T}$，加偏置项$b \in \mathbb{R}^{1}$采用了广播机制。
# 
# 为什么模型需要加截距项呢？因为如果模型不添加截距项，估计出来的模型将一定会通过原点，即在$\mathbf{x}$取值为0时，$\mathbf{y}$的估计值也是0。为了消除这个模型设定偏误，我们在模型中添加截距项，这使得模型估计既有可能通过原点，也有可能不通过原点，提升了模型的适用范围。
# 
# ### 2. 均方损失函数
# 
# 在设定好模型和数据准备好之后，通过某种迭代算法得到模型参数$\theta$，使得模型在该组参数上的数据误差尽量小。
# 
# 为了衡量估计值与真实值之间的误差，线性回归模型中通常会选择某种非负实数函数作为损失函数。为什么要采用非负实数函数为损失函数呢？因为非负实数损失函数值如果越小则代表误差也越小，也可以看作为第$i$样本的估计价格值与真实价格值之间的差距越小，当估计值与真实值相等时，表明模型在第$i$样本上的误差为0。在训练数据集给定的情况下，该误差仅与模型参数$\mathbf{w},b$相关，因此$L$可以看成是以$\mathbf{w},b$为参数的函数。二次函数求导会使得项式前乘以2，为了使得导数表达式更简洁，通常会在二次误差项前面乘以1/2。
# 
# $L^{i}\left ( \mathbf{w},\mathbf{b}\right )=\frac{1}{2}\left ( \hat{y}^{i}-y^{i}\right )^{2}$
# 
# 上式用于衡量单个样本$i$的损失，如果要衡量训练数据集上的全部样本损失，一种方式是把所有样本的损失值加起来，然后再求平均。求平均是为了消除样本数量对损失的影响。
# 
# $L \left ( \mathbf{w},\mathbf{b}\right )=\frac{1}{2n}\left ( \hat{y}^{i}-y^{i}\right )^{2}$
# 
# 在模型训练过程中，我们最终的目标是要找到一组参数$\left ( \mathbf{w^{*}},\mathbf{b^{*}}\right )$，使得训练数据集上的全部样本损失尽可能小。
# 
# $\mathbf{w^{*}},\mathbf{b^{*}} = argmin L \left ( \mathbf{w},\mathbf{b} \right )$
# 
# 损失函数又可以写成矢量表达式：
# 
# $L \left ( \mathbf{\theta} \right )=\frac{1}{2n} \left ( \mathbf{\hat{y}}-\mathbf{y} \right )^{T}\left ( \mathbf{\hat{y}}-\mathbf{y} \right )$
# 
# 其中，线性回归模型的输出形状大小为$\mathbf{\hat{y}}\in \mathbb{R}^{n \times 1}$，样本标签的形状大小为$\mathbf{y}\in \mathbb{R}^{n \times 1}$，样本数量为$n$.
# 
# ### 3. 解析优化与随机梯度下降
# 
# 优化算法可以分为解析和数值两类方法。
# 
# 首先，我们来看解析方法，它适用于损失函数形式较为简单的场景，正好适用于线性回归模型。
# 
# $L \left ( \theta \right )=\frac{1}{2n} \left ( \mathbf{\hat{y}}-\mathbf{y} \right )^{T}\left ( \mathbf{\hat{y}}-\mathbf{y} \right ) = \frac{1}{2n} \left ( \mathbf{Xw}+b-\mathbf{y} \right )^{T}\left ( \mathbf{Xw}+b-\mathbf{y} \right )$
# 
# 在样本特征矩阵$\mathbf{X}$上增加一个全为1的列，于是新的样本特征矩阵为$\tilde{X} \in \mathbb{R}^{n \times (k+1)}$，上式可以简化为：
# 
# $L \left ( \theta \right )=\frac{1}{2n} \left ( \mathbf{\tilde{X} \theta}-\mathbf{y} \right )^{T}\left ( \mathbf{\tilde{X} \theta}-\mathbf{y} \right )$
# 
# 其中，$\mathbf{\theta} =\left [ \mathbf{w},\mathbf{b} \right ]^{T}$.
# 
# 将上式对$\mathbf{\theta}$求梯度，得到：
# 
# $\bigtriangledown _{\mathbf{\theta}}L(\mathbf{\theta})=\frac{1}{n}\mathbf{\tilde{X}}^{T}(\mathbf{\tilde{X} \theta}-\mathbf{y})$
# 
# 令上式为零，解出$\mathbf{\theta}$为：
# 
# $\hat{\mathbf{\theta} }=\left ( \mathbf{\tilde{X}}^{T}\mathbf{\tilde{X}}\right )^{-1}\mathbf{\tilde{X}}^{T}\mathbf{y}$
# 
# 接下来，我们来看数值求解方法，它的适用范围更加广泛，适用于函数形式复杂或不存在解析解的场景，该方法通过有限次迭代模型参数使损失函数值不断降低。它不像解析方法一步到位，而是迭代式地向正确方向小步迈进。以小批量随机梯度下降算法为例，该算法在深度学习领域得到了广泛应用。该算法通常有4个步骤：一是选择模型参数值。如果是首轮迭代，可以采用随机方式选取初始值；如果是非首轮迭代，可以选择上一轮迭代更新的参数值。二是在训练数据集中选取一批样本组成小批量集合（Batch Set），小批量中的样本个数通常是固定的，用$m$来代表小批量中样本的个数。三是把模型参数初始值与小批量中的样本数据，都代入模型，$n$替换成$m$，得到损失函数值。损失函数以$(\mathbf{w}, b)$为参数，把损失函数分别对$(\mathbf{w}, b)$参数求偏导数。四是用求出的三个偏导数与预先设定的一个正数（学习率）相乘作为本轮迭代中的减少量。
# 
# $w_{1}\leftarrow w_{1}-\frac{\lambda }{m}\sum_{i=1}^{m}\frac{\partial L^{i}(w_{1},w_{2},b)}{\partial w_{1}}$
# 
# $w_{2}\leftarrow w_{2}-\frac{\lambda }{m}\sum_{i=1}^{m}\frac{\partial L^{i}(w_{1},w_{2},b)}{\partial w_{2}}$
# 
# $b\leftarrow b-\frac{\lambda }{m}\sum_{i=1}^{m}\frac{\partial L^{i}(w_{1},w_{2},b)}{\partial b}$
# 
# 其中，批量大小$m$和学习率$\lambda$是超参数，并不是通过模型训练得出，需要根据经验来提前设定。
# 上述过程的矢量计算表达式为：
# 
# $\theta \leftarrow \theta-\frac{\lambda }{m}\sum_{i=1}^{m}\bigtriangledown _{\mathbf{\theta}}L^{i}(\mathbf{\theta})$
# 
# 模型训练完成后，将得到参数$\mathbf{\hat{\theta}}$，$\mathbf{\hat{\theta}}$参数可以看成真实${\mathbf{\theta}}$的最佳估计。接下来把$\mathbf{\hat{\theta}}$参数代入线性回归模型，待预测样本的输入特征分别乘以回归系数（权重）后加和即可得到输出，该输出便是预测值。
# 
# ### 4. 自定义多元回归模型
# 
# 构建一个自定义的多元线性回归模型，来看看模型能否学习到我们事前自定义的参数。大体流程如下：（1）自定义一些随机样本数据，然后事前设定多元回归模型参数（真实参数），采用该模型为每个样本生成真实标签。（2）假设机器并不知道我们事先设定的真实参数，需要利用样本特征数据和真实标签来训练模型，通过随机梯度下降算法来更新和优化参数，这是参数估计过程。（3）对比估计参数与我们事前设定的真实参数是否趋于一致。如果趋于一致，则表明模型通过学习样本，学习到了真实参数，反之就是没学到。
# 
# 下面采用paddlepaddle 2构建自定义线性回归模型。

# 第一步，生成500个服从均值为0、方差为1正态分布的随机样本，各样本的输入特征数设为2。对于样本$i$，两个特征分别用$x_{1}^{i}$和$x_{2}^{i}$表示，采用以下多元线性方程生成样本$i$的真实标签：
# 
# $y^{i}=x_{1}^{i}w_{1}+x_{2}^{i}w_{2}+b+\varepsilon ^{i}$
# 
# 我们不妨设定$w_{1}=1.2$，$w_{2}=2.5$，$b=6.8$，$\varepsilon$服从均值为0、方差为0.001的正态分布，这即为每个样本的真实标签。

# In[ ]:


# # 23.5.25 代码解析
# 解析如下：

# 导入numpy库，用于生成数据和进行数值计算。
# 定义了两个变量num_inputs和num_examples，分别表示输入特征的维度和样本数量。
# 定义了真实的权重true_w和偏置true_b。
# 使用numpy.random.normal函数生成具有正态分布的随机输入特征，形状为
# (num_examples, num_inputs)，并将数据类型转换为float32。
# 根据线性关系 labels = features[:, 0] * true_w[0] + features[:, 1] * true_w[1] + true_b 生成标签（真实值）。
# labels被赋值后，是什么数据结构？
# features[:,0]是一个一维的数组了，切片时直接0，而不是[0]，就降维了，true_w[0]显然标量
# 这个计算过程可以理解为一个简单的线性回归模型，其中 features 表示输入特征，
# true_w 表示特征的权重，true_b 表示偏置项。通过特征与权重的加权和再加上偏置项，
# 得到预测的标签值。
# 最终生成的 labels 是一个一维数组，形状为 (num_examples,)，表示每个样本对应的标签值。

# 使用numpy.expand_dims函数在最后增加一个维度，将标签的形状变为(num_examples, 1)，
# 以符合一般的数据表示习惯。以下详解expand_dims:
# 具体来说，labels 是一个一维数组，形状为 (num_examples,)，表示每个样本对应的标签值。
# 通过 numpy.expand_dims(labels, axis=-1)，将 labels 数组的维度从 (num_examples,) 
# 扩展为 (num_examples, 1)，在最后一个维度上增加了一个维度。

# 这个操作在机器学习和深度学习中常用于处理标签数据。在某些情况下，模型的输入要求标签数据具有二维形状，
# 即每个样本对应一个向量。通过在标签数据的最后一个维度上增加一个维度，可以将一维标签数据转换为二维向量，
# 便于与模型的输出进行对应和计算。


# In[5]:


import numpy

num_inputs=2
num_examples=500
true_w=[1.2,2.5]
true_b=6.8
features = numpy.random.normal(0,1,(num_examples, num_inputs)).astype('float32')
labels = features[:,0]*true_w[0]+features[:,1]*true_w[1]+true_b
labels = labels + numpy.random.normal(0,0.001,labels.shape[0])
labels = labels.astype('float32')
print(labels.shape)
labels = numpy.expand_dims(labels,axis=-1) #注意：需要在最后增加一个维度
print(labels.shape)


# In[7]:


# labels


# 第二步，构建线性回归模型。

# In[8]:


import paddle
train_data=paddle.to_tensor(features)
y_true    =paddle.to_tensor(labels)
model=paddle.nn.Linear(in_features=2, out_features=1)


# 第三步，构建优化器和损失函数

# In[9]:


sgd_optimizer=paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
mse_loss=paddle.nn.MSELoss()


# In[10]:


for i in range(5000):
    y_predict = model(train_data)
    loss=mse_loss(y_predict, y_true)
    loss.backward()
    sgd_optimizer.step()
    sgd_optimizer.clear_grad()

print(model.weight.numpy())
print(model.bias.numpy())
print(loss.numpy())


# 经过5000次迭代优化，模型参数估计值为1.2、2.5，截距项估计值为6.8。这与我们事前设定的模型参数是一致的，这表明机器已经从样本数据中学习到了真实参数。
