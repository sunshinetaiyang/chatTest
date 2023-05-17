#!/usr/bin/env python
# coding: utf-8

# ![](https://ai-studio-static-online.cdn.bcebos.com/8e54bfb21a0e4aa6bec5406824c021f9ce6b81e30c7949248a6ac5681dd59df0)
# 
# # 1.赛题背景

# 随着轨道交通线网规模增大带来的客流数据规模骤增的情况，采用传统运输策划方式开展线网列车运行计划的编制，已经不能较好的匹配实际需求。因此如何精准预测客流大数据作为轨道交通线网运输策划的依据，在满足旅客出行需求的同时降低企业运输成本，成为了轨道交通运营企业越来越关注的问题。    
#   
#   
# 本次大赛提出了构建基于历史客流数据的轨道交通客流预测模型，通过模型更好的指导轨道交通运营企业开展运输组织工作。  
#   
# 赛题链接：https://aistudio.baidu.com/aistudio/competition/detail/958/0/introduction

# # 2.赛题解读
# 本赛题在初赛阶段的主要任务是基于广州地铁进出站客流量历史数据，从数据中挖掘各地铁站点客流量随着时间变化的规律，使用飞桨深度学习框架训练预测模型，以达到根据过去24小时真实客流量记录来预测未来24小时各地铁站点的客流量和相关高低峰期的特征，以更好地指导轨道交通运营企业开展运输组织工作。  
#   
# 本赛题任务属于时序预测，因此我们将使用飞桨深度时序建模库——PaddleTS完成本任务，同时为参赛选手提供赛题基线，帮助选手们快速上手Paddle TS和了解赛题，把更多的精力放在优化模型效果上。  
#   
#   
# `PaddleTS开源项目库：https://github.com/PaddlePaddle/PaddleTS`

# # 3.赛题数据解析
# 本赛题数据由广州地铁集团有限公司提供，数据采集自广州地铁X号线A-G站的真实进出站客流量数据，数据时间跨度从2022年1月1日到2023年3月31日，共234614条记录。每条记录时间间隔为15分钟，数据文件包括 **监测时间段、监测站点、进站人数和出站人数** 字段。  
#   
# 数据集文件结构如下：
# ```
# 地铁数据.zip
# ├── test.csv
# ├── result.csv
# ├── 车站信息表.xlsx
# └── 客流量数据.csv
# ```
# 
# 
# - `客流量数据.csv`  广州地铁X号线A-G站从2022年1月1日到2023年3月31日的真实进出站客流量数据
# - `车站信息表.xlsx` 各站点的关系数据和位置数据
# - `test.csv` 2023年4月15日真实客流量数据，选手需要根据这份已知数据，预测4月16日的客流量
# - `result.csv` 验证数据提交格式模板

# # 4.数据预处理
# ## 4.1 解压数据集
# 
# 

# In[1]:


get_ipython().system('unzip  -qO UTF-8 data/data211908/地铁数据.zip -d ./data')


# 
# ## 4.2 提取有效信息
# 主办方提供的数据集是csv格式的文档文件，包括了`时间`、`时间区段`、`站点`、`进站数量`、`出站数量`这5项信息。  
# 1.我们先通过文本切割的方法，提取时间区段的开始时间，然后将开始时间与时间字段拼接，组成一列frequency为15min的时间序列。  
# 2.利用正则提取站点字母代码，方便后面对数据进行筛选分类（当然也可以不做这一步，但后面筛选字段时会比较麻烦）  
# <a style="color:red;font-weight:bold">3.（重要）将列名转换为英文命名（"Time", "Station", "InNum", "OutNum"），预测完成后数据提交格式也是这4个字段！</a>

# In[25]:


import re
import pandas as pd

df = pd.read_csv("data/客流量数据.csv")
df.head()


# In[23]:


def extract_station(x):
    station = re.compile("([A-Z])站").findall(x)
    # print('in extract_station:', station)
    if len(station)>0:
        return station[0]

# print(extract_station('BZHAN站E站'))


# In[26]:



df["时间区段"] = df["时间区段"].apply(lambda x:x.split("-")[0])
df["站点"] = df["站点"].apply(extract_station)
df["时间"] = df["时间"].apply(lambda x:str(x)).str.cat(df['时间区段'],sep=" ")
df["时间"] = pd.to_datetime(df["时间"])
df = df.drop("时间区段",axis=1)
df.columns =["Time", "Station", "InNum", "OutNum"] 
df.head()


# # 5.模型训练
# ## 5.1 安装PaddleTS环境

# In[27]:


get_ipython().system('pip install paddlets')


# ## 5.2 引入依赖包

# In[28]:


import paddle
from paddlets.datasets.tsdataset import TSDataset
from paddlets.transform import TimeFeatureGenerator, StandardScaler
from paddlets.models.forecasting import LSTNetRegressor
from paddlets.metrics import MAE
import warnings
warnings.filterwarnings('ignore')


# ## 5.3 选取站点数据
# 赛题要求是预测A到G总共7个站点的客流量，选手可以在训练时将站点编码作为协变量，本基线为了更好让新手理解，选择了每个站点数据独立训练模型的方式，将7个站点的数据分别放入模型中训练得到7个模型，再分别调用7个模型对7个站点的数据进行预测，最后将预测数据合并。因此，我们这里以C站点为例，带大家跑通模型训练到模型预测的过程。  
#   
# 首先，我们把C站点的数据单独提取。

# In[29]:


station = "C" # 站点
dataset_df = df[df['Station']==station]
dataset_df.head()


# ## 5.4 构建TSDataset
# `TSDataset` 是 `PaddleTS` 中一个主要的类结构，用于表示绝大多数的时序样本数据，并作为PaddleTS其他算子的输入以及输出对象。TSDataset支持对csv文件、json、dataframe格式的读取，同时支持split切片、plot绘图等操作，详见技术文档：  
# [https://github.com/PaddlePaddle/PaddleTS/blob/main/paddlets/datasets/tsdataset.py](https://github.com/PaddlePaddle/PaddleTS/blob/main/paddlets/datasets/tsdataset.py)  
# 
# 因为时序预测要求数据连续，但广州地铁的运营时间为6:00-24:00，在0:00-6:00时间段存在数据空白，因此我们需要对这一部分的数据进行填充。PaddleTS的TSDataset为我们提供了`fill_missing_dates`参数，我们可以利用这个参数，将缺失值填充为0。

# In[30]:


dataset_df = TSDataset.load_from_dataframe(
    dataset_df,
    time_col='Time', #时间序列
    target_cols=['InNum','OutNum'], #预测目标
    freq='15min',
    fill_missing_dates=True,
    fillna_method='zero'
)


# 我们可以利用TSDataset自带的.plot方法观测一下输入数据。

# In[31]:


dataset_df.plot()


# 可以看出，地铁客流量有一定的波动规律，存在明显的高峰和低谷，我们可以用.summary()函数查看数据的分布情况。

# In[32]:


dataset_df.summary()


# ## 5.5 分割数据集
# 上面我们介绍了`TSDataset`是支持数据切片的，但与python原生和numpy的切片操作不同，TSDataset的切片操作更便捷，可以直接通过时间序列进行切片。下面我们将数据以2023年2月5日23时45分为分界时间节点分为`训练集`和`验证集`，再将验证集以2023年3月29日23时45分为节点分出`测试集`供后面数据模型效果验证。

# In[33]:


dataset_train, dataset_val_test = dataset_df.split("2023-02-05 23:45:00")
dataset_val, dataset_test = dataset_val_test.split("2023-03-29 23:45:00")


# ## 5.6 数据标准化
# 为了让模型训练得到更好的拟合效果以及更高的拟合速度，我们需要对数据进行标准化缩放，这里使用的是PaddleTS自带的`StandardScaler`函数，以训练集数据的最大最小值为基础，对训练集、验证集和测试集分别做标准化处理。

# In[34]:


scaler = StandardScaler()
scaler.fit(dataset_train)
dataset_train_scaled = scaler.transform(dataset_train)
dataset_val_test_scaled = scaler.transform(dataset_val_test)
dataset_val_scaled = scaler.transform(dataset_val)
dataset_test_scaled = scaler.transform(dataset_test)


# ## 5.7 模型训练
# PaddleTS内置了众多的时序预测模型，如`MLP`、`LSTM`、`Transformer`、`Informer`等，本案例使用的是LSTM长短期记忆网络模型，选手可以尝试更换其他模型，或者将几个模型混合得到更好的效果。  
# `PaddleTS时序预测模型库：https://github.com/PaddlePaddle/PaddleTS/tree/main/paddlets/models/forecasting/dl`  
#   
#   
# 
# 参数说明：
# - in_chunk_len：历史窗口的大小，模型输入的时间步数
# - out_chunk_len：预测范围的大小，模型输出的时间步数
# - max_epochs：最大迭代训练次数
# - patience：early stop的手段之一，超过n轮迭代后指标仍无提升即停止训练
# - eval_metrics：评估指标，这里选用的是赛题要求的MAE
# - optimizer_params：训练参数，主要配置learning_rate
# 

# In[35]:


paddle.seed(2023)
model = LSTNetRegressor(
    in_chunk_len = 4*24,
    out_chunk_len = 4*24,
    max_epochs = 200,
    patience = 20,
    eval_metrics =["mae"],
    optimizer_params= dict(learning_rate=3e-3)
)


# 模型配置定义后，通过一行代码就可以开始模型训练了。

# In[36]:


model.fit(dataset_train_scaled, dataset_val_scaled)


# ## 5.8 数据回测
# 模型训练完后，我们用测试集的数据对模型的效果进行回测，以此检验模型预测数据与真实数据之间的误差。这一步我们用的是PaddleTS自带的`backtest`函数，return_predicts参数设置为True，可以返回预测结果方便我们与真实数据进行可视化对比。

# In[37]:


from paddlets.utils import backtest
from paddlets.metrics import MAE
mae, pred = backtest(data=dataset_test_scaled,
                model=model,
                metric=MAE(),
                return_predicts=True
)


# In[38]:


_, ground_truth = dataset_test_scaled.split("2023-03-30 23:45:00")


# In[41]:


ground_truth.plot(y="power")
print(ground_truth.head())


# In[39]:


pred.plot(add_data = ground_truth)


# 我们用2023年4月16日的数据预测4月17日的C站点客流量，再与4月17日的真实客流量数据做对比，可以看出数据趋势整体一致，但在瞬时波动比较大的峰值数据区间上，模型预测的数据离真实数据还有一定误差，为了衡量预测值和真实值之间的差距，我们可以查看一下MAE指标。

# In[ ]:


mae


# 细心的选手可能会发现上面的图像y轴坐标数值都比较小，这是由于没有将数据<a style="color:red;font-weight:bold">反标准化</a>导致的。因此，我们需要用`scaler.inverse_transform`方法对预测数据进行反标准化处理。

# In[28]:


ground_truth_inverse = scaler.inverse_transform(ground_truth)
pred_inverse = scaler.inverse_transform(pred)
pred_inverse.plot(add_data=ground_truth_inverse)


# 将数据反标准化之后，我们会发现有一些数据值是小于0的，同时很多数据值都是浮点型，这是由于我们用的是自回归建模方法，但负值的数据是明显不合理的。所以，在提交数据之前，我们要注意以下两点：  
# <br />
# <a style="color:red;font-weight:bold">1.将负值改为0；<br/>2.将浮点数类型转化为整数类型</a>

# # 6.模型保存
# 模型训练完成后，我们需要将模型固化成文件保存在本地，方便后续的推理工作。这部分使用的是PaddleTS模型自带的`save()`函数。同时，也要注意将数据标准化的参数保存，用于后续推理前进行数据预处理。

# In[ ]:


import os
import pickle

model_save_path = os.path.join("models/lstm",station)
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
model.save(os.path.join(model_save_path, "model"))
pickle.dump(scaler, open(os.path.join(model_save_path,"scaler.pkl"),'wb'))


# # 7.模型推理
# 模型推理相对比较简单，第一步先将测试数据集读入，转化为`TSDataset`类型。

# In[ ]:


import pandas as pd
from paddlets.datasets.tsdataset import TSDataset
from paddlets.transform import TimeFeatureGenerator, StandardScaler

dataset_test_df = pd.read_csv("data/test.csv")
station = "C" # 站点
dataset_test_df = dataset_test_df[dataset_test_df['Station']==station]

dataset_test_df = TSDataset.load_from_dataframe(
    dataset_test_df,
    time_col='Time',
    target_cols=['InNum','OutNum'],
    freq='15min',
    fill_missing_dates=False,
)

dataset_test_df


# 加载标准化处理参数，对测试数据进行标准化预处理。

# In[ ]:


import pickle
with open("models/lstm/C/scaler.pkl", "rb") as r:
    scaler = pickle.load(r)
dataset_test_scaled = scaler.transform(dataset_test_df)
dataset_test_scaled


# 将保存好的模型重新加载，调用`.predict`函数对测试数据进行预测，对预测结果进行反标准化，再处理负值和浮点型问题即可输出为结果文件。

# In[37]:


import os
from paddlets.models.model_loader import load

model = load("models/lstm/{}/model".format(station))
res = model.predict(dataset_test_scaled)
res_inversed = scaler.inverse_transform(res)
res_inversed = res_inversed.to_dataframe(copy=True)
res_inversed["InNum"] = res_inversed["InNum"].apply(lambda x:0 if x<0 else int(x))
res_inversed["OutNum"] = res_inversed["OutNum"].apply(lambda x:0 if x<0 else int(x))
res_inversed["Station"] = station # 添加站点标识
res_inversed.index.name = "Time"
# 输出结果文件
if not os.path.exists("./results"):
    os.makedirs("./results")
res_inversed.to_csv("results/{}_result.csv".format(station))
res_inversed.head()


# 以上为广州地铁C站点单个站点的客流量预测模型训练和推理过程，只需要重复以上步骤，把剩余站点的客流量预测后合并即可提交结果验证。  
# > 特别说明：为了方便数据验证，提交时仍需提交预测日的全时段进出站客流预测数据，但在数据验证时，只验证各站点6:00-24:00的预测结果与真实数据的误差。详见比赛详情页介绍和提交文件示例。

# # 8.提分技巧
# - 虽然本赛题的数据集数据维度较少，但暗藏的信息比较多，比如日期上可以分为工作日和节假日，时间上可以分为早晚高峰，这些可以结合PaddleTS的数据处理模块进行进一步挖掘作为变量，将可能有助于效果大幅度提升。
# - 本基线以比较直接的时序预测方法来完成，但实际上，我们可以将地铁站点比作节点，将地铁之间的关系连线形成边，尝试去探索上下站点之间的联系，应用包括但不限于图的方法，也是一种不错的解题思路。
