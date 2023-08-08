#!/usr/bin/env python
# coding: utf-8

# # 2023CSIG-飞桨、工银瑞信金融科技挑战赛
# 
# - 赛题链接：[https://aistudio.baidu.com/aistudio/competition/detail/1032/0/introduction](https://aistudio.baidu.com/aistudio/competition/detail/1032/0/introduction)
# - 赛题任务：基于AI的量化选股投资策略建模与可视分析
# 
# ## 任务描述
# 量化选股投资策略建模是一种利用数学和计算机方法来分析市场数据，以确定投资组合的投资策略。量化选股投资策略通常采用机器学习（如线性回归、决策树等）和深度学习（如LSTM、Transformer、CNN等）等方法，发现潜在的市场趋势和规律，并据此制定有效的交易决策。这种方式不仅解决了传统主观判断带来的偏差问题，还提升了交易效率与回报水平，是量化投资人员常用的技术手段。
# 
# 在此任务中，我们将提供丰富的股票因子数据，供参赛者进行建模，对股票的未来的收益率进行预测，并制定有效的投资策略。我们鼓励参赛者探索各种创新的算法和模型，在充满噪声的股票市场数据中提升预测的准确性，获得超额收益。同时，为了对股票因子数据进行归因，我们鼓励参赛者进行多维度可视化分析。
# 
# > 注意：允许使用外部数据，但只能使用基础数据，且需说明数据合理性，保证数据合法性。
# 
# ## 评价标准
# 评价标准包括多头组合的最终收益率和信息比率两个指标。
# 
# - 投资组合的最终收益率
# 
# 根据参赛者提交的投资组合权重数据，计算累计的最终收益率。 参赛者可以周为单位进行投资组合的权重调整。组织者将根据参赛者每周预测的投资组合权重和未来一周每日股票vwap价格，计算股票累计收益。累计收益越高，则说明模型效果越好。 累计收益R计算公式如下：
# 
# $$
# \mathrm{R}=\sum_{t=1}^T \sum_{i=1}^N w_t r_t
# $$
# 
# 其中，为参赛者预测的投资组合权重；t为时间，T表示所有的测试集交易时间；i为股票序号，N为所有股票数量。 交易规则：每周一前提交的权重，作为下一周交易日的持仓权重。
# 
# 1.  手续费：单边千分之1.5，即在股票买入和卖出时**各收取1.5‰的费用**。如当天不提交预测权重，则维持权重不变。非交易日**不允许**提交权重。
# 2.  投资组合权重。要求所有股票权重**均为正值，和为1**。
# 3.  允许参赛者在实盘过程中不断调整模型及投资策略。如在实盘过程中模型、投资策略发生调整，需在算法报告中进行详细说明。
# 
# - 信息比率
# 
# 信息比率（IR， information ration）是超额收益与跟踪误差的比值，为某一投资组合优于一个特定指数的风险调整超额报酬。信息比率用来衡量超额风险所带来的超额收益，表示单位主动风险所带来的超额收益。因此，信息比率是投资组合中的一个重要指标。
# 
# 设投资组合为P，大盘指数为B，则超额收益可表示为：
# 
# $$
# \mathrm{R}_{\mathrm{S}}=\frac{1}{T} \sum_{t=1}^T\left(R_{P_t}-R_{B_t}\right)
# $$
# 
# 跟踪误差（TE,tracking error）表示为：
# 
# $$
# \mathrm{TE}=\sqrt{\frac{\sum_{t=1}^T\left(R_{P_t}-R_{B_t}\right)^2}{(T-1)}}
# $$
# 
# 
# 其中，t为时间，T为交易日数量。
# 
# 信息比率IR为：$\mathrm{IR}=\frac{R_s}{T E}$
# 
# 其中，大盘指数可以选择沪深 300、中证 500、中证 1000、国证,2000、创业板指。
# 
# 
# 
# ## 奖项设置
# 
# | 奖项        | 获奖队伍                               | 奖品内容                                |
# | ----------- | -------------------------------------- | --------------------------------------- |
# | 一等奖      | 1支参赛队伍                            | 奖金14000元人民币，颁发对应奖项获奖证书 |
# | 一等奖      | 2支参赛队伍                            | 奖金8000元人民币，颁发对应奖项获奖证书  |
# | 二等奖      | 3支参赛队伍                            | 奖金4000元人民币，颁发对应奖项获奖证书  |
# | 二等奖      | 4支参赛队伍                            | 奖金2000元人民币，颁发对应奖项获奖证书  |
# | 三等奖      | 10支参赛队伍                           | 颁发对应奖项获奖证书                    |
# 
# 所有进入复赛并取得前三十名的优秀选手将有机会获得工银瑞信AI量化投资实习生机会，有效期一年。 优秀成果可发布在公司公众号，在资管行业为个人能力背书。
# 
# 
# ## 比赛时间
# 
# | **时间**                    | **事项** | **备注**                                                     |
# | :-------------------------- | :------- | :----------------------------------------------------------- |
# | 即日起-2023年8月4日         | 开放注册 | 选手报名，数据集下载。                                       |
# | 2023年7月17日-2023年8月4日  | 初赛时间 | 1）2023年7月17日，开放下载数据。 2）2023年7月20日，开放评测通道，选手可提交测试集投资组合权重结果。 3）初赛**前100名**选手进入复赛（视实际情况进行适当增加或减少）。 |
# | 2023年8月5日-2023年9月7日   | 复赛时间 | 1）A股实盘测试，需要提交**投资组合权重结果**。 2）复赛队伍届时需按所给要求提交**算法代码、算法报告、可视化系统代码**。 |
# | 2023年9月8日                | 提交材料 | 复赛排行榜**前30名**选手（视实际情况进行适当增加或减少）按所给要求提交 算法代码、算法报告、可视化系统代码 |
# | 2023年9月15日-2023年9月17日 | 结果公示 | 比赛最终结果公示                                             |
# | 2023年9月22日-2023年9月24日 | 总决赛   | 分赛道冠军团队晋级总决赛                                     |
# 
# ## 综合测评
# 组织者将根据参赛提交的投资组合权重计算投资组合的累计收益排名和信息比率排名。最终复赛排行榜排名为两个评价指标的排名平均值（排名相同的情况下以信息比率高者为先）。 在初赛阶段，排名将在网站排行榜中公布。 在复赛阶段，参赛者需根据组织者公布的最近一周交易日的股票匿名因子特征预测下一周交易日的投资组合权重。组织者将在每周二在参赛者交流群中公布截至上周五的排行榜排名情况。
# 

# ## 训练集
# 
# 训练集包括股票的因子特征和对应的待预测标签。 因子特征共包含三种因子特征和一种风险特征。其中因子特征文件分别为：
# - factor_set_0_20100104_20221130.csv
# - factor_set_1_20100104_20221130.csv
# - factor_set_2_20131231_20221130.csv
# - risk_set_20121231_20221130.csv
# 
# 待预测标签文件为label_20100104_20221130.csv。
# 
# 不同文件字段解释：
# - factor_set_0_20100104_20221130.csv
#     - date：日期，顺序排列
#     - asset：股票编号
#     - AdjPrice_Close：调整后收盘价
#     - AdjPrice_High：调整后最高价
#     - AdjPrice_Open：调整后开盘价
#     - AdjPrice_Low： 调整后最低价
#     - Vwap_Adj_1d： 调整后Vwap价格
#     - Turnover： 换手率
# 
# - factor_set_1_20100104_20221130.csv
#     - date：日期，顺序排列
#     - asset：股票编号
#     - factor_set_1_0:factor_set_1_50: 基于股票相关数据生成的匿名特征
# 
# - factor_set_2_20131231_20221130.csv
#     - date：日期，顺序排列
#     - asset：股票编号
#     - factor_set_2_0:factor_set_2_14: 基于股票相关数据生成的匿名特征
# - risk_set_20100104_20221130.csv
#     - date：日期，顺序排列
#     - asset：股票编号
#     - risk_set_0:risk_set_10: 基于股票相关数据生成的匿名特征
# - label_20100104_20221130.csv
#     - date：日期，顺序排列
#     - asset：股票编号
#     - label：未来一段时间的收益
# 
# 
# 1.  参赛者可根据提供的数据制定自己的label。
# 2.  以上五种数据集通过date和asset进行关联。
# 3.  不是所有的asset在每个交易时间的每个因子上都有值。
# 4.  参赛选手可使用全部或部分因子数据。
# 5.  允许参赛者使用外部数据，但只能使用基础数据，且需说明数据合理性，保证数据合法性。
# 6.  **禁止在模型中使用未来数据，包括未来的收益，整个训练集上的平均值等。**
# 
# ## 测试集
# 
# 测试集提供与训练集相同的四种因子特征文件，分别为
# - factor_set_0_20230103_20230630.csv
# - factor_set_1_20230103_20230630.csv
# - factor_set_2_20230103_20230630.csv
# - risk_set_20230103_20230630.csv
# 
# 四种因子文件含义与训练集完全相同。
# 
# 
# 

# In[1]:


get_ipython().system('pip list | grep paddle')


# ## 数据读取

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

# https://aistudio.baidu.com/aistudio/datasetdetail/230997/0
get_ipython().system('ls /home/aistudio/data/data230997/ -lh')


# In[3]:


train_factor_set_0 = pd.read_csv("/home/aistudio/data/data230997/factor_set_0_20100104_20221130.csv", nrows=None)
train_factor_set_0 = reduce_mem_usage(train_factor_set_0)

train_factor_set_1 = pd.read_csv("/home/aistudio/data/data230997/factor_set_1_20100104_20221130.tar.gz", nrows=None)
train_factor_set_1 = reduce_mem_usage(train_factor_set_1)

train_factor_set_2 = pd.read_csv("/home/aistudio/data/data230997/factor_set_2_20131231_20221130.tar.gz", nrows=None)
train_factor_set_2 = reduce_mem_usage(train_factor_set_2)

train_risk_set = pd.read_csv("/home/aistudio/data/data230997/risk_set_20100104_20221130.tar.gz", nrows=None)
train_risk_set = reduce_mem_usage(train_risk_set)

train_label = pd.read_csv("/home/aistudio/data/data230997/label_20100104_20221130.tar.gz", nrows=None)
train_label = reduce_mem_usage(train_label)


# In[4]:


train_label['label'].max(), train_label['label'].min()


# In[5]:


train_factor_set_0.shape, train_factor_set_1.shape, train_factor_set_2.shape, train_risk_set.shape, train_label.shape


# In[6]:


test_factor_set_0 = pd.read_csv("/home/aistudio/data/data230997/factor_set_0_20230103_20230630.csv", nrows=None)
test_factor_set_0 = reduce_mem_usage(test_factor_set_0)

test_factor_set_1 = pd.read_csv("/home/aistudio/data/data230997/factor_set_1_20230103_20230630.tar.gz", nrows=None)
test_factor_set_1 = reduce_mem_usage(test_factor_set_1)

test_factor_set_2 = pd.read_csv("/home/aistudio/data/data230997/factor_set_2_20230103_20230630.tar.gz", nrows=None)
test_factor_set_2 = reduce_mem_usage(test_factor_set_2)

test_risk_set = pd.read_csv("/home/aistudio/data/data230997/risk_set_20230103_20230630.tar.gz", nrows=None)
test_risk_set = reduce_mem_usage(test_risk_set)


# In[7]:


test_factor_set_0.shape, test_factor_set_1.shape, test_factor_set_2.shape, test_risk_set.shape


# ## 数据分析

# ### factor_set_0

# In[8]:


df1 =  train_factor_set_0[train_factor_set_0['asset'] == '000002.XSHE']
df2 =  test_factor_set_0[test_factor_set_0['asset'] == '000002.XSHE']

plt.plot(range(len(df1)), df1['AdjPrice_Open'])

plt.plot(range(len(df1), len(df1) + len(df2)), df2['AdjPrice_Open'])


# ### factor_set_1

# In[9]:


df1 =  train_factor_set_1[train_factor_set_1['asset'] == '000002.XSHE']
df2 =  test_factor_set_1[test_factor_set_1['asset'] == '000002.XSHE']

plt.plot(range(len(df1)), df1['factor_set_1_0'])
plt.plot(range(len(df1), len(df1) + len(df2)), df2['factor_set_1_0'])


# ### factor_set_2

# In[10]:


df1 =  train_factor_set_2[train_factor_set_2['asset'] == '000002.XSHE']
df2 =  test_factor_set_2[test_factor_set_2['asset'] == '000002.XSHE']

plt.plot(range(len(df1)), df1['factor_set_2_0'])
plt.plot(range(len(df1), len(df1) + len(df2)), df2['factor_set_2_0'])


# ### risk_set

# In[11]:


df1 =  train_risk_set[train_risk_set['asset'] == '000002.XSHE']
df2 =  test_risk_set[test_risk_set['asset'] == '000002.XSHE']

plt.plot(range(len(df1)), df1['risk_set_0'])
plt.plot(range(len(df1), len(df1) + len(df2)), df2['risk_set_0'])


# ## 搭建模型

# - 全连接网络

# In[12]:


import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim
from paddle.io import Dataset, DataLoader

model = nn.Sequential(
    nn.Linear(50 + 12, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

paddle.summary(model, (1, 62))

model = paddle.Model(model)
model.prepare(optimizer=paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()), 
              loss=paddle.nn.MSELoss())


# - 1D卷积网络

# In[13]:


model = nn.Sequential(
    nn.Conv1D(7, 20, 3),
    nn.ReLU(),
    nn.MaxPool1D(3),
    nn.Conv1D(20, 20, 3),
    nn.ReLU(),
    nn.MaxPool1D(3),
    nn.Flatten(),
    nn.Linear(120, 1),
    nn.ReLU()
)

paddle.summary(model, (1024, 7, 62))

model = paddle.Model(model)
model.prepare(optimizer=paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()), 
              loss=paddle.nn.MSELoss())


# - 1D 卷积 + 全连接网络

# In[14]:


class Model3(nn.Layer):
    def __init__(self):
        super(Model3, self).__init__()

        self.cnn_layer = nn.Sequential(
            nn.Conv1D(7, 20, 3),
            nn.ReLU(),
            nn.MaxPool1D(3),
            nn.Conv1D(20, 20, 3),
            nn.ReLU(),
            nn.MaxPool1D(3),
            nn.Flatten(),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(50 + 12 + 120, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
    def forward(self, data):
        x1 = self.cnn_layer(data)
        x2 = paddle.concat([x1, data[:, -1, :]], 1)
        out = self.fc_layer(x2)
        return out

model = Model3()
paddle.summary(model, (10, 7, 62))

model = paddle.Model(model)
model.prepare(optimizer=paddle.optimizer.Adam(learning_rate=0.0005, parameters=model.parameters()), 
              loss=paddle.nn.MSELoss())


# In[15]:


class Model3(nn.Layer):
    def __init__(self):
        super(Model3, self).__init__()

        self.cnn_layer = nn.Sequential(
            nn.Conv1D(7, 20, 3),
            nn.ReLU(),
            nn.MaxPool1D(3),
            nn.Conv1D(20, 20, 3),
            nn.ReLU(),
            nn.MaxPool1D(3),
            nn.Flatten(),
        )

        self.lstm_layer = nn.LSTM(62, 128, 2)

        self.fc_layer = nn.Sequential(
            nn.Linear(50 + 12 + 120 + 128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
    def forward(self, data):
        x1 = self.cnn_layer(data)

        x2, _ = self.lstm_layer(data)
        x2 = x2[:, -1, :]

        x3 = paddle.concat([x1, x2, data[:, -1, :]], 1)
        out = self.fc_layer(x3)
        return out

model = Model3()
paddle.summary(model, (10, 7, 62))

model = paddle.Model(model)
model.prepare(optimizer=paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters()), 
              loss=paddle.nn.MSELoss())


# In[16]:


import numpy as np
from paddle.io import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, factor_set_1, factor_set_2, risk_set, label):
        self.factor_set_1 = factor_set_1
        # self.factor_set_2 = factor_set_2
        self.risk_set = risk_set
        self.stack_data = np.hstack([
                self.factor_set_1.values,
                self.risk_set.values,
        ]).astype(np.float32)

        self.label = label.astype(np.float32).values
    
    def __getitem__(self, idx):
        if idx > 7:
            feat = self.stack_data[idx-7:idx]
        else:
            pad = np.zeros((7-idx, 62)).astype(np.float32)
            feat = self.stack_data[:idx]
            feat = np.vstack([pad, feat])

        return feat, self.label[idx]

    def __len__(self):
        return len(self.label)

train_dataset = CustomDataset(
    train_factor_set_1.iloc[:-761192, 2:].fillna(0),
    train_factor_set_2.iloc[:-761192, 2:].fillna(0),
    train_risk_set.iloc[:-761192, 2:].fillna(0),
    np.clip(train_label.iloc[:-761192, 2:].fillna(0), -1, 1),
)

val_dataset = CustomDataset(
    train_factor_set_1.iloc[-761192:, 2:].fillna(0),
    train_factor_set_2.iloc[-761192:, 2:].fillna(0),
    train_risk_set.iloc[-761192:, 2:].fillna(0),
    np.clip(train_label.iloc[-761192:, 2:].fillna(0), -1, 1)
)

train_loader = DataLoader(train_dataset,
                    batch_size=1024,
                    shuffle=True,
                    num_workers=1)

val_loader = DataLoader(val_dataset,
                    batch_size=1024,
                    shuffle=True,
                    num_workers=1)


# In[17]:


for data, label in train_loader:
    break


# In[18]:


data.shape, label.shape


# In[19]:


np.clip(train_label.iloc[:-761192, 2:].fillna(0), -1, 1)


# ## 模型训练

# In[20]:


# 启动模型训练，指定训练数据集，设置训练轮次，设置每次数据集计算的批次大小，设置日志格式
model.fit(train_dataset, val_dataset,
          epochs=1, 
          batch_size=1024,
          verbose=1)


# In[21]:


model.save("model", True)
model.load("model")


# ## 模型预测

# In[22]:


test_risk_set = pd.read_csv("/home/aistudio/data/data230997/risk_set_20230103_20230630.tar.gz", nrows=None)
test_dataset = CustomDataset(
    test_factor_set_1.iloc[:, 2:].fillna(0),
    test_factor_set_2.iloc[:, 2:].fillna(0),
    test_risk_set.iloc[:, 2:].fillna(0),
    pd.DataFrame([0] * len(test_factor_set_1))
)
test_predict = model.predict(test_dataset, batch_size=1024 * 4, verbose=1)
test_predict = np.vstack(test_predict[0]).reshape(-1)


# In[23]:


np.min(test_predict), np.max(test_predict)


# In[24]:


test_risk_set = pd.read_csv("/home/aistudio/data/data230997/risk_set_20230103_20230630.tar.gz", nrows=None)
test_risk_set = reduce_mem_usage(test_risk_set)
test_risk_set['weight'] = test_predict
test_risk_set = test_risk_set[test_risk_set['weight'] > 0]


# In[25]:


test_risk_set = pd.merge(test_risk_set,
    test_risk_set.groupby('date')['weight'].sum().reset_index().rename({'weight': 'sum_weight'}, axis=1),
    on='date', how='left')

test_risk_set['weight'] /= test_risk_set['sum_weight']
test_risk_set['weight'] = test_risk_set['weight'].apply(lambda x: round(x, 6))
test_risk_set[['date', 'asset',  'weight']].to_csv('submit.csv', index=None)


# In[26]:


test_risk_set['weight'].min(), test_risk_set['weight'].max()


# In[27]:


get_ipython().system('head submit.csv')


# ## 改进方向
# 
# 1. 特征工程优化：针对提供的股票因子数据，进行更加深入的特征工程，寻找更能反映股票未来收益率的有效特征。可以考虑使用统计学指标、技术指标、基本面数据等来构建更全面的特征集，还可以尝试使用领域知识来创造新的特征。特征工程在量化选股中非常重要，优秀的特征能够大幅提升模型性能。
# 2. 时间序列建模：由于股票数据是时间序列数据，可以尝试使用更加专业的时间序列建模方法，如ARIMA、SARIMA、Prophet等，结合股票市场的季节性和周期性特征，进一步提高预测精度。
# 3. Transformer模型在自然语言处理中取得了巨大的成功，它可以并行处理序列数据，具有较低的计算复杂度。将Transformer模型应用于量化选股任务中，可能会带来更高效的模型训练和预测。

# In[ ]:




