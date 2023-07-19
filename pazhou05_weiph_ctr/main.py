#!/usr/bin/env python
# coding: utf-8

# ![](https://bj.bcebos.com/v1/ai-studio-match/file/9930cd280fb24cd58323cc92ff8423677b7eb8d0fc3b44f89a721005509b3018?authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2023-04-24T15%3A51%3A43Z%2F-1%2F%2F5c8caa3b924ac684af88e1f385661ff0e543262c5bd22684d23817472f6691ad)
# 
# # 基于用户行为和商品属性的购买商品预估
# 
# ## 大赛背景
# 第二届广州·琶洲算法大赛是由广州市人民政府主办，海珠区政府、琶洲实验室等多家单位联合承办的人工智能算法领域权威专业赛事。大赛围绕绿色能源、智慧交通、智能制造、电子商务、前沿科技等重点产业，邀请政府单位、科研平台和龙头企业提供应用场景，并设置数百万奖金、提供足额算力支持和丰厚的政策福利，面向全球招募优选算法和解决方案。
# 
# 赛事官网：[https://ai.baidu.com/tech/competition/pazhou](https://ai.baidu.com/tech/competition/pazhou)
# 
# 
# ## 赛题介绍
# 赛题名称：基于用户行为和商品属性的购买商品预估
# 
# 赛题官网：[https://aistudio.baidu.com/aistudio/competition/detail/914/0/introduction](https://aistudio.baidu.com/aistudio/competition/detail/914/0/introduction)
# 
# 
# 随着信息爆炸式增长，用户往往面临着大量的选择和信息过载，推荐算法可以为用户过滤和推荐最有价值的信息，降低信息过载的风险。在电商领域，推荐算法的核心是要解决人货匹配的问题，即为每个用户从海量的候选商品中，挑选出最为感兴趣的部分商品，构建用户的专属商店。本赛题任务为在真实的业务场景下，基于商品属性、用户在商品上的行为等数据，以转化率为目标构建个性化推荐模型。
# 
# ## 赛题描述
# 本题目的技术方向为电商推荐领域的ctr/cvr预估。在电商领域，推荐算法的核心是要解决人货匹配的问题，即为每个用户从海量的候选商品中，挑选出最为感兴趣的部分商品，构建用户的专属商店。本题目使用唯品会真实业务数据，希望参赛队伍能够基于用户行为数据和商品的品牌品类信息，为用户推荐合适的商品。
# 
# ## 赛题任务
# 在真实的业务场景下，以转化率为目标构建个性化推荐模型。
# 
# 具体定义如下数据：
# 
# - UB-用户对商品的行为数据
# - GA-商品属性数据
# - U-用户子集
# - G-商品子集
# 
# 目标是通过GA和UB训练购买模型，给出U中用户在G中可能购买的商品。
# 
# ## 提交与评分
# 参赛者需预测用户可能购买的goods_id，并提交一份csv文档作为结果，文档名为u2i.csv,并压缩为zip文件，选手输出结果包含两列，分别是user_id和goods_id。示例如下：
# 
# ```
# 8da2ec07d8bf9bfe1e849cb7e7f25e5c, f6e4f43d18157cbdcdc653c6e35f01fb
# e873fcfe12d89fc9fe3f3c4425029bae, 305fa40cbcd4a898f92f00e5ca4ee317
# 5c72cfe71eeb24883bb0a0aec656903b, a04fbf8f3f86d9e3c5fa8fa402b75afb
# 8da2ec07d8bf9bfe1e849cb7e7f25e5c, a04fbf8f3f86d9e3c5fa8fa402b75afb
# ```
# 
# 其中user_id只包括选手预测有购买行为的用户id，如认为用户fb26342610257f08fecf9b4b7c0d64b3无购买行为，则不需要包含该用户；当预测一个用户购买多个goods_id时，写作多行，如示例中的用户8da2ec07d8bf9bfe1e849cb7e7f25e5c
# 
# PredictionSet为算法预测的购买数据集合，ReferenceSet为真实的答案购买数据集合。我们以F1值作为最终的唯一评测标准 训练集包括用户行为数据（UB）和商品属性数据(GA)两部分。其中用户行为数据来自部分用户29天内的浏览，收藏，加购，购买行为，涉及51602个用户的7791816条样本；商品数据包括部分商品的品牌品类信息，涉及3465608个商品。 测试集提供用户id数据（U），和作为候选集的商品id数据（G），参赛者需要为U中每个用户，从G中选择其在第30天可能购买的商品，注意用户在当日可能不存在购买行为。 其中数据G包含1367964个商品id，数据U在赛段1涉及用户数5000个，在赛段2涉及用户10000个。
# 
# ## 数据集介绍
# 
# 数据来自唯品会推荐业务的真实数据。
# 
# a榜数据：[https://aistudio.baidu.com/aistudio/datasetdetail/210802](https://aistudio.baidu.com/aistudio/datasetdetail/210802)
# 
# 1.用户行为数据（UB）
# 
# 一行数据为一次用户请求下的行为。每行格式为：
# 
# user_id, goods_id, is_clk, is_like, is_addcart, is_order, expose_start_time, dt
# 
# | 字段              | 字段说明                     |
# | :---------------- | :--------------------------- |
# | user_id           | 用户id                       |
# | goods_id          | 商品id                       |
# | is_clk            | 本次请求下对该商品的点击次数 |
# | is_like           | 本次请求下对该商品的收藏次数 |
# | is_addcart        | 本次请求下对该商品的加购次数 |
# | is_order          | 本次请求下对该商品的购买次数 |
# | expose_start_time | 本次请求的时间戳             |
# | dt                | 日期                         |
# 
# 2.商品属性数据(GA)
# 
# 商品及其属性，每行格式为：
# 
# goods_id, cat_id, brandsn
# 
# | 字段     | 字段说明 |
# | :------- | :------- |
# | goods_id | 商品id   |
# | cat_id   | 品类id   |
# | brandsn  | 品牌id   |
# 
# 用户id数据（U）：每行为一个用户id
# 
# 商品id数据（G）：每行为一个商品id

# ##  数据加载与分析

# ### 加载数据集

# In[1]:


get_ipython().system('ls ./data/data210802/ -lh')


# In[2]:


get_ipython().system('unzip ./data/data210802/训练集.zip > /dev/null')
get_ipython().system('unzip ./data/data210802/测试集a.zip > /dev/null')


# In[3]:


import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import LabelEncoder


# In[4]:


train_goods = pd.concat([
    pd.read_csv('./训练集/traindata_goodsid/part-00000', header=None, names=['goods_id', 'cat_id', 'brandsn']),
    pd.read_csv('./训练集/traindata_goodsid/part-00001', header=None, names=['goods_id', 'cat_id', 'brandsn']),
    pd.read_csv('./训练集/traindata_goodsid/part-00002', header=None, names=['goods_id', 'cat_id', 'brandsn'])
], axis=0)

train_user = pd.concat([
    pd.read_csv(x, header=None, names=['user_id', 'goods_id', 'is_clk', 'is_like', 'is_addcart', 'is_order', 'expose_start_time', 'dt'], nrows=500000)
    for x in glob.glob('./训练集/traindata_user/part*')
], axis=0)


# In[6]:


testa_goods = pd.concat([
    pd.read_csv('./测试集a/predict_goods_id/part-00000', header=None, names=['goods_id', 'cat_id', 'brandsn']),
    pd.read_csv('./测试集a/predict_goods_id/part-00001', header=None, names=['goods_id', 'cat_id', 'brandsn']),
], axis=0)

testa_user = pd.read_excel('./测试集a/a榜需要预测的uid_5000.xlsx')


# In[7]:


user_encode = LabelEncoder()
user_encode.fit(list(train_user['user_id']) + list(train_user['user_id']))

goods_encode = LabelEncoder()
goods_encode.fit(list(train_user['goods_id']) + list(train_goods['goods_id']) + list(train_goods['goods_id']))


# In[8]:


np.mean(testa_user['user_id'].isin(train_user['user_id'])), np.mean(testa_goods['goods_id'].isin(train_goods['goods_id']))


# ### 数据分析

# In[9]:


train_user['user_id'].nunique(), train_user['goods_id'].nunique()


# In[10]:


train_user.describe().round(2)


# In[11]:


train_user['user_id'].value_counts()


# In[ ]:


train_user.loc[(train_user['user_id'] == '71e1a59e90bc7174cf6349761217c627') & (train_user['goods_id'] == '47382b8a57e5b73bdba51de5c230fded')]


# In[ ]:


train_data = pd.merge(train_user.iloc[:], train_goods.iloc[:], on='goods_id')


# In[ ]:


train_data['cat_id'].nunique(), train_data['brandsn'].nunique()


# ## 基础思路

# In[ ]:


train_agg_feat = train_data.loc[
    (train_data['is_order'] == 0) & (train_data['is_addcart'] != 0) 
]

train_agg_feat = train_agg_feat[train_agg_feat['user_id'].isin(testa_user['user_id'])]
train_agg_feat = train_agg_feat[train_agg_feat['goods_id'].isin(testa_goods['goods_id'])]


# In[ ]:


train_agg_feat[['user_id', 'goods_id']].to_csv('u2i.csv', index=None)


# In[ ]:


get_ipython().system('zip u2i.csv.zip u2i.csv')


# ## 进阶模型（改进中）

# In[ ]:


train_agg_feat = train_data.iloc[:].groupby(['user_id', 'goods_id']).agg({
    'is_clk': ['sum', 'max'],
    'is_like': ['sum', 'max'],
    'is_addcart': ['sum', 'max'],
    'is_order': ['sum', 'max'],
})

train_agg_feat = train_agg_feat.reset_index()
train_agg_feat.columns = [
    'user_id',
    'goods_id',
    'is_clk_sum',
    'is_clk_max',
    'is_like_sum',
    'is_like_max',
    'is_addcart_sum',
    'is_addcart_max',
    'is_order_sum',
    'is_order_max'
 ]


# In[ ]:


test_goods_id_agg = train_agg_feat.groupby('goods_id').agg({
    'is_clk_sum': 'sum',
    'is_order_max': 'sum'
})
test_goods_id_agg = test_goods_id_agg[test_goods_id_agg['is_clk_sum'] > 100]
test_valid_goods = test_goods_id_agg.index


# In[ ]:


train_feat_downsmaple = pd.concat([
    train_agg_feat[train_agg_feat['is_order_max'] !=0],
    train_agg_feat[train_agg_feat['is_order_max'] ==0].sample(int(0.03 * len(train_agg_feat)))
], axis=0)


# In[ ]:


1 - train_feat_downsmaple['is_order_max'].mean()


# In[ ]:


1 - train_feat_downsmaple['is_addcart_max'].mean()


# In[ ]:


train_feat_downsmaple['user_id'] = user_encode.transform(train_feat_downsmaple['user_id'])
train_feat_downsmaple['goods_id'] = goods_encode.transform(train_feat_downsmaple['goods_id'])


# In[ ]:


import paddle
import paddle.nn as nn
from paddle.io import Dataset

class SelfDefinedDataset(Dataset):
    def __init__(self, df, mode = 'train'):
        super(SelfDefinedDataset, self).__init__()
        self.df = df
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode == 'predict':
            return (
                self.df['user_id'].iloc[idx], 
                self.df['goods_id'].iloc[idx], 
                self.df['is_clk_max'].iloc[idx], self.df['is_like_max'].iloc[idx], 
                self.df['is_addcart_max'].iloc[idx],
            )
        else:
            return (
                self.df['user_id'].iloc[idx], 
                self.df['goods_id'].iloc[idx], 
                self.df['is_clk_max'].iloc[idx], 
                (self.df['is_like_max'].iloc[idx]!= 0).astype(int),
                (self.df['is_addcart_max'].iloc[idx] != 0).astype(int),
                (self.df['is_order_max'].iloc[idx] != 0).astype(int)
            )

    def __len__(self):
        return len(self.df)


# In[ ]:


train_feat_downsmaple['is_order_max'].max()


# In[ ]:


train_feat_downsmaple = train_feat_downsmaple.sample(frac=1.0)

traindataset = SelfDefinedDataset(train_feat_downsmaple.iloc[:-int(-0.2*len(train_feat_downsmaple))])
train_loader = paddle.io.DataLoader(traindataset, batch_size = 128, shuffle = True)

validdataset = SelfDefinedDataset(train_feat_downsmaple.iloc[-int(-0.2*len(train_feat_downsmaple)):])
valid_loader = paddle.io.DataLoader(validdataset, batch_size = 128, shuffle = True)


# In[ ]:


EMBEDDING_SIZE = 256

# 定义深度学习模型
class RecommenderNet(nn.Layer):
    def __init__(self, num_users, num_goods, embedding_size, numeric_size):
        super(RecommenderNet, self).__init__()
        self.num_users = num_users
        self.num_goods = num_goods
        self.embedding_size = embedding_size

        weight_attr_user = paddle.ParamAttr(
            regularizer = paddle.regularizer.L2Decay(1e-6),
            initializer = nn.initializer.KaimingNormal()
            )
        self.user_embedding = nn.Embedding(
            num_users,
            embedding_size,
            # weight_attr=weight_attr_user
        )
        
        weight_attr_movie = paddle.ParamAttr(
            regularizer = paddle.regularizer.L2Decay(1e-6),
            initializer = nn.initializer.KaimingNormal()
            )
        self.goods_embedding = nn.Embedding(
            num_goods,
            embedding_size,
            # weight_attr=weight_attr_movie
        )

        self.linear = nn.Linear(
            2 * embedding_size, 2
        )
        
    def forward(self, data):
        user, goods, feat = data[0], data[1], [data[idx] for idx in [2, 3,4]]
        feat = paddle.stack(feat, 1).astype(paddle.float32)

        user_vector = self.user_embedding(user)
        goods_vector = self.goods_embedding(goods)
        x = paddle.concat([user_vector, goods_vector], 1)
        return self.linear(x)


# In[32]:


# 定义模型损失函数、优化器和评价指标
model = RecommenderNet(len(user_encode.classes_), len(goods_encode.classes_), 128, 3)
model


# In[ ]:


optimizer = paddle.optimizer.SGD(parameters=model.parameters(), learning_rate=0.01)
loss_fn = nn.BCEWithLogitsLoss()

