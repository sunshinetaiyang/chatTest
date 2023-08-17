#!/usr/bin/env python
# coding: utf-8

# # UFO:Unified Feature Optimization [[arXiv]](https://arxiv.org/pdf/2207.10341v1.pdf) 
# ```BibTex
# @inproceedings{
#   xi2022ufo,
#   title={UFO:Unified Feature Optimization},
#   author={Teng Xi, Yifan Sun, Deli Yu, Bi Li, Nan Peng, Gang Zhang et al.},
#   booktitle={European Conference on Computer Vision},
#   year={2022},
#   url={https://arxiv.org/pdf/2207.10341v1.pdf}
# }
# ```
# 
# # **UFO比赛背景**
# 
# 近年来预训练大模型一次次刷新记录，展现出惊人的效果，但对于产业界而言，势必要面对如何应用落地的问题。当前预训练模型的落地流程可被归纳为：针对只有少量标注数据的特定任务，使用任务数据 fine-tune 预训练模型并部署上线。然而，当预训练模型参数量不断增大后，该流程面临两个严峻的挑战。首先，随着模型参数量的急剧增加，大模型 fine-tuning 所需要的计算资源将变得非常巨大，普通开发者通常无法负担。其次，随着 AIoT 的发展，越来越多 AI 应用从云端往边缘设备、端设备迁移，而大模型却无法直接部署在这些存储和算力都极其有限的硬件上。
# 
# 针对预训练大模型落地所面临的问题，百度提出统一特征表示优化技术（[UFO：Unified Feature Optimization](https://arxiv.org/pdf/2207.10341v1.pdf)），在充分利用大数据和大模型的同时，兼顾大模型落地成本及部署效率。VIMER-UFO 2.0 技术方案的主要内容包括：
# 
# * All in One：行业最大 170 亿参数视觉多任务模型，覆盖人脸、人体、车辆、商品、食物细粒度分类等 20+ CV 基础任务，单模型 28 个公开测试集效果 SOTA。
# * One for All：首创针对视觉多任务的超网络与训练方案，支持各类任务、各类硬件的灵活部署，解决大模型参数量大，推理性能差的问题。
# 
# 
# <img src="https://bj.bcebos.com/v1/ai-studio-match/file/fe0849b846fd4a6bb7ab361a1bc6c470e932a10b66714fd9a5ed862a8df2d554?authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-08-01T08%3A04%3A20Z%2F-1%2F%2F3403d8f635cb0a410b50d4dd52197327a329cdfee2f28cce1ef96c716fa52686 " width = "800"  alt="未标题-2.png" align=center /><br>
# 
# # **原理介绍**
# 
# ## **All in One功能更强大更通用的视觉模型**
# 
# 之前主流的视觉模型生产流程，通常采用单任务 “train from scratch” 方案。每个任务都从零开始训练，各个任务之间也无法相互借鉴。由于单任务数据不足带来偏置问题，实际效果过分依赖任务数据分布，场景泛化效果往往不佳。近两年蓬勃发展的大数据预训练技术，通过使用大量数据学到更多的通用知识，然后迁移到下游任务当中，本质上是不同任务之间相互借鉴了各自学到的知识。基于海量数据获得的预训练模型具有较好的知识完备性，在下游任务中基于少量数据 fine-tuning 依然可以获得较好的效果。不过基于预训练+下游任务 fine-tuning 的模型生产流程，需要针对各个任务分别训练模型，存在较大的研发资源消耗。
# 
# 百度提出的 VIMER-UFO All in One 多任务训练方案，通过使用多个任务的数据训练一个功能强大的通用模型，可被直接应用于处理多个任务。不仅通过跨任务的信息提升了单个任务的效果，并且免去了下游任务 fine-tuning 过程。VIMER-UFO All in One 研发模式可被广泛应用于各类多任务 AI 系统，以智慧城市场景为例，VIMER-UFO 可以用单模型实现人脸识别、人体和车辆ReID等多个任务的 SOTA 效果，同时多任务模型可获得显著优于单任务模型的效果，证明了多任务之间信息借鉴机制的有效性。
# 
# 针对大模型的开发和部署问题，UFO给出了One for All的解决方案，通过引入超网络的概念，超网络由众多稀疏的子网络构成，每个子网络是超网络中的一条路径，将不同参数量、不同任务功能和不同精度的模型训练过程变为训练一个超网络模型。训练完成的One for All UFO超网络大模型即可针对不同的任务和设备低成本生成相应的可即插即用的小模型，实现One for All Tasks 和 One for All Chips的能力
# 
# 
# <img src=" https://bj.bcebos.com/v1/ai-studio-match/file/b8ac3afdd7db4ec8ad7800e74fa9f812202303b93de14fc89d7fffdfb6c0fcd5?authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-08-01T08%3A07%3A18Z%2F-1%2F%2Fa345b74cf0163a218a705e8d34dc479d3a8f2b5a3b40ca6234b53adc18d2bfad" width = "800"  alt="未标题-2.png" align=center /><br>
# 
# ## **One For All灵活可伸缩的弹性部署方案**
# 
# 受算力和存储的限制，大模型无法直接部署在边缘设备上。一个针对云端设备开发的模型要部署到边缘设备或端设备时往往要进行模型压缩，或完全重新设计，而预训练大模型的压缩本身需要耗费大量的资源。
# 
# 另外，不同任务对模型的功能和性能要求也不同，例如人脸识别门禁系统只需具备人脸识别功能即可，智慧社区的管控系统则需要同时具备人脸识别和人体分析的能力，部分场景还需要同时具备车型识别及车牌识别能力。即便是同样的人脸识别任务，门禁系统和金融支付系统对模型的精度和性能要求也不同。目前针对这些任务往往需要定制化开发多个单任务模型，加之需要适配不同的硬件平台，AI模型开发的工作量显著增长。
# 
# 针对大模型的开发和部署问题，VIMER-UFO 给出了 One for All 的解决方案，通过引入超网络的概念，超网络由众多稀疏的子网络构成，每个子网络是超网络中的一条路径，将不同参数量、不同任务功能和不同精度的模型训练过程变为训练一个超网络模型。训练完成的 VIMER-UFO One for All 超网络大模型即可针对不同的任务和设备低成本生成相应的可即插即用的小模型，实现 One for All Tasks 和 One for All Chips 的能力。
# 
# ## **超网络设计与训练方案**
# 
# VIMER-UFO 2.0 基于 Vision Transformer 结构设计了多任务多路径超网络。与谷歌 Switch Transformer 以图片为粒度选择路径不同，VIMER-UFO 2.0 以任务为粒度进行路径选择，这样当超网络训练好以后，可以根据不同任务独立抽取对应的子网络进行部署，而不用部署整个大模型。VIMER-UFO 2.0 的超网中不同的路径除了可以选择不同 FFN 单元，Attention 模块和 FFN 模块内部也支持弹性伸缩，实现网络的搜索空间扩展，为硬件部署提供更多可选的子网络，并提升精度。
# 
# VIMER-UFO 2.0 超网络分为多路径 FFN 超网和与可伸缩 Attention 超网两部分。首先针对多路径 FFN 超网模块，每个任务都有两种不同的路径选择，即选择共享 FFN（FFN-shared）或者专属 FFN（FFN-taskX），当选定好 FFN 以后，还可根据放缩系数弹性选择FFN中参数规模；因此FFN超网络中共有（T * ratio）^L 种不同的 FFN 路径，其中 T 为 task 的数量，L 为网络的层数, ratio 为放缩系数的数量。而对于 self-attention 超网，每个子网络可以选择不同的 Head 数量 QKV 矩阵参数量。
# 
# VIMER-UFO 2.0 训练时将模型按层级结构划分为任务超网和芯片超网两个级别。并分别使用不同的训练方案进行优化。
# 
# ## **One For All Tasks**
# 
# 任务超网络训练时，需要同时优化网络参数（FFN）和路由参数（Router）。前面提到，网络参数包含共享 FFN（FFN-shared）和专属 FFN（FFN-taskX），所有任务都会更新共享 FFN 的参数，特定任务只会更新专属的 FFN 参数。而路由参数由于离散不可导，训练时通过 Gumbel Softmax 进行优化。由于在训练超网的过程中多个任务的同时进行优化，同时引入了路由机制，可以让相关的任务共享更多的参数，而不相关的任务之间尽量减少干扰，从而获得针对不同任务最优的子网络模型。在业务应用时，只需要根据不同子网络在特定任务的效果，抽取出对应的任务子网，即可直接部署，无需重复训练。
# 
# 
# <img src=" https://bj.bcebos.com/v1/ai-studio-match/file/c9f5068604c44c48862f3ec6d84bda48488a62fa89c64978a7eacefef9b5ac60?authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-08-01T08%3A11%3A34Z%2F-1%2F%2F889afd878d993aa741bce76077ba45ee76f2d74607532adc22c733945ccbff49" width = "800"  alt="未标题-2.png" align=center /><br>
# 
# ## **One For All Chips**
# 
# 在任务超网训练完成以后，针对每个任务抽取的子网络进行芯片子网络的训练。经过上述训练以后便得到了每个任务的芯片超网。在业务应用时，针对不同平台存储容量和算力不同，可以抽取不同深度和宽度的子网络进行部署，进一步压缩模型的参数和计算量。由于超网络中子网络的数据众多，每个子网逐一测试精度和延时并不现实，因此在 VIMER-UFO 2.0 中，使用了 GP-NAS中的基于高斯过程的超参数超参估计技术，只需采样超网络中少了子网络进行评估，即可准确预测出其他网络的精度和速度。
# 
# 
# <img src=" https://bj.bcebos.com/v1/ai-studio-match/file/89d457b606cf4788b16190dc8c627fd8bc0056e356e34f40a3c9b48380f04633?authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-08-01T08%3A12%3A14Z%2F-1%2F%2F5aa0fa1d04b9cb913d6d24847e24cd93ddde680cd263d15a6477829c401b5d9d" width = "800"  alt="未标题-2.png" align=center /><br>
# 
# # **模型效果**
# 170亿参数，全球最大CV大模型，基于Task MoE架构，稀疏激活，支持抽取轻量级小模型，兼顾大模型效果和小模型推理性能，单模型覆盖20+ CV基础任务，在28个公开测试集上效果SOTA，根据任务的不同自动选择激活最优的区域，从而实现100倍参数压缩 ，同时支持下游任务快速扩展 。
# 
# 
# 在背景介绍中我们知道，受算力和存储的限制，大模型无法直接部署在边缘设备上。一个针对云端设备开发的模型要部署到边缘设备或端设备时往往要进行模型压缩，或完全重新设计，而预训练大模型的压缩本身需要耗费大量的资源。
# 
# 另外，不同任务对模型的功能和性能要求也不同，例如人脸识别门禁系统只需具备人脸识别功能即可，智慧社区的管控系统则需要同时具备人脸识别和人体分析的能力，部分场景还需要同时具备车型识别及车牌识别能力。即便是同样的人脸识别任务，门禁系统和金融支付系统对模型的精度和性能要求也不同。目前针对这些任务往往需要定制化开发多个单任务模型，加之需要适配不同的硬件平台，AI模型开发的工作量显著增长。
# 
# 针对大模型的开发和部署问题，VIMER-UFO 给出了 One for All 的解决方案，通过引入超网络的概念，超网络由众多稀疏的子网络构成，每个子网络是超网络中的一条路径，将不同参数量、不同任务功能和不同精度的模型训练过程变为训练一个超网络模型。训练完成的 VIMER-UFO One for All 超网络大模型即可针对不同的任务和设备低成本生成相应的可即插即用的小模型，实现 One for All Tasks 和 One for All Chips 的能力。
# 
# 我们从垂类应用出发，选择了人脸、人体、车辆、商品四个任务来训练视觉模型大一统模型。
# 
# ## 数据集介绍
# 我们使用了脸、人体、车辆、商品的公开数据集具体如下:
# 
# ### 训练集
# 
# | **任务**                      | **数据集**                     | **图片数**                     | **类别数**                     |
# | :-----------------------------| :----------------------------: | :----------------------------: | :----------------------------: |
# | 人脸                          |           MS1M-V3              |           5,179,510            |           93,431               |
# | 人体                          |           Market1501-Train     |           12,936               |           751                  |
# | 人体                          |           MSMT17-Train         |           30,248               |           1,041                |
# | 车辆                          |           Veri-776-Train       |           37,778               |           576                  |
# | 车辆                          |           VehicleID-Train      |           113,346              |           13,164               |
# | 车辆                          |           VeriWild-Train       |           277,797              |           30,671               |
# | 商品                          |           SOP-Train            |           59,551               |           11,318               |
# 
# 
# ### 测试集
# 
# | **任务**                      | **数据集**                     | **图片数**                     | **类别数**                     |
# | :-----------------------------| :----------------------------: | :----------------------------: | :----------------------------: |
# | 人脸                          |           LFW                  |           12,000               |           -                    |
# | 人脸                          |           CPLFW                |           12,000               |           -                    |
# | 人脸                          |           CFP-FF               |           14,000               |           -                    |
# | 人脸                          |           CFP-FP               |           14,000               |           -                    |
# | 人脸                          |           CALFW                |           12,000               |           -                    |
# | 人脸                          |           AGEDB-30             |           12,000               |           -                    |
# | 人体                          |           Market1501-Test      |           19,281               |           750                  |
# | 人体                          |           MSMT17-Test          |           93,820               |           3,060                |
# | 车辆                          |           Veri-776-Test        |           13,257               |           200                  |
# | 车辆                          |           VehicleID-Test       |           19,777               |           2,400                |
# | 车辆                          |           VeriWild-Test        |           138,517              |           10,000               |
# | 商品                          |           SOP-Test             |           60,502               |           11,316               |
# 
# ## 多任务AutoDL benchmark
# 
# 我们基于ViT-Base构建了搜索空间，搜索空间维度有网络深度（depth）、自注意力头的数目（num_heads）、前向计算网络的膨胀系数（mlp_ratio)，其变化范围为depth \in {10, 11, 12}，num_heads \in {10, 11, 12}，mlp_ratio \in {3.0, 3.5, 4.0}，搜索空间中有 9^10 + 9^11 + 9^12 个不同的子网络。 子网的编码的长度为37，包括1位depth编码，以及12组的3位编码，分别指示为num_heads、mlp_ratio和embed_dim（在本次赛题中embed_dim为768，不作为搜索维度），实际depth小于12，则后尾填充0。对于depth编码，‘j’，'k’和’l’分别表示10，11和12；对于num_heads编码，‘1’，'2’和’3’表示12，11和10；对于mlp_ratio编码，‘1’，'2’和’3’表示4.0, 3.5, 3.0，对于embed_dim编码，'1’表示768。以j111231321311311221231121111231000000为例，子网结构的depth为10，10层模型的num_heads的列表为[12, 11, 10, 10, 10, 11, 11, 12, 12, 11]，mlp_ratio的列表为[4.0, 3, 3.5, 4.0, 4.0, 3.5, 3, 3.5, 4.0, 3]，embed_dim的列表为[768, 768, 768, 768, 768, 768, 768, 768, 768, 768]。
# 
# 为了方便选手参赛，我们直接将基于训练好的超网络的采样的模型结构在各个benchmark上的性能提供给大家作为训练数据，包括500个样本，其中输入为模型结构，标签为每个结构在8个任务上的相对排序rank，rank取值为0到499的整数；测试数据包括99500个样本，选手需要根据样本的结构信息训练多任务预测器（可以每个任务单独训练，也可以联合训练），并预测测试数据的99500个结构在8个任务上的排序，取值范围为0到99499。提交格式见『提交结果』
# 
# 比赛分A/B榜单，A/B榜单都基于选手提交的同一份提交文件，但是计算分数的节点的编号不同。比赛提交截止日期前仅A榜对选手可见，比赛结束后B榜会对选手公布，比赛最终排名按照选手成绩在B榜的排名。

# # 训练集说明

# In[1]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')
get_ipython().system('ls /home/aistudio/data/data162979/')


# In[2]:


# 读取训练数据, 训练集包含500个模型结构，以及这些结构在cplfw，market1501，dukemtmc等8个任务上的性能排序
import json
with open('/home/aistudio/data/data162979/CCF_UFO_train.json', 'r') as f:
    train_data = json.load(f)
print(train_data['arch1'])
print('train_num:',len(train_data.keys()))


# # 处理训练数据

# In[3]:


def convert_X(arch_str):
        temp_arch = []
        total_1 = 0
        total_2 = 0
        ts = ''
        for i in range(len(arch_str)):
            if i % 3 != 0 and i != 0 and i <= 30:
                elm = arch_str[i]
                ts = ts + elm
                if elm == 'l' or elm == '1':
                    temp_arch = temp_arch + [1, 1, 0, 0]
                elif elm == 'j' or elm == '2':
                    temp_arch = temp_arch + [0, 1, 1, 0]
                elif elm == 'k' or elm == '3':
                    temp_arch = temp_arch + [0, 0, 1, 1]
                else:
                    temp_arch = temp_arch + [0, 0, 0, 0]
            
            elif i % 3 != 0 and i != 0 and i > 30:
                elm = arch_str[i]
                if elm == 'l' or elm == '1':
                    temp_arch = temp_arch + [1, 1, 0, 0, 0]
                elif elm == 'j' or elm == '2':
                    temp_arch = temp_arch + [0, 1, 1, 0, 0]
                elif elm == 'k' or elm == '3':
                    temp_arch = temp_arch + [0, 0, 1, 1, 0]
                else:
                    temp_arch = temp_arch + [0, 0, 0, 0, 1]
            
        return temp_arch

train_list = [[],[],[],[],[],[],[],[]]
arch_list_train = []
name_list = ['cplfw_rank', 'market1501_rank', 'dukemtmc_rank', 'msmt17_rank', 'veri_rank', 'vehicleid_rank', 'veriwild_rank', 'sop_rank']
for key in train_data.keys():
    for idx, name in enumerate(name_list):
        train_list[idx].append(train_data[key][name])
    arch_list_train.append(convert_X(train_data[key]['arch']))
print(arch_list_train[0])


# # 训练各任务预测器

# In[4]:


# 本demo基于GP-NAS
# GP-NAS已经集成在PaddleSlim模型压缩工具中
# GP-NAS论文地址 https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_GP-NAS_Gaussian_Process_Based_Neural_Architecture_Search_CVPR_2020_paper.pdf
get_ipython().system('pip install paddleslim')


# In[5]:


from paddleslim.nas import GPNAS 
import numpy as np
import scipy
import scipy.stats

gp_list = []

for i in range(len(train_list[:])):
    # 每个任务有该任务专属的gpnas预测器
    gp_list.append(GPNAS(2,2))

train_num = 400


for i in range(len(train_list[:])):
    # 划分训练及测试集
    X_all_k, Y_all_k  = np.array(arch_list_train), np.array(train_list[i])
    X_train_k, Y_train_k, X_test_k, Y_test_k = X_all_k[0:train_num:1], Y_all_k[0:train_num:1], X_all_k[train_num::1], Y_all_k[train_num::1]
    # 初始该任务的gpnas预测器参数
    gp_list[i].get_initial_mean(X_train_k[0::2],Y_train_k[0::2])
    init_cov = gp_list[i].get_initial_cov(X_train_k)
    # 更新（训练）gpnas预测器超参数
    gp_list[i].get_posterior_mean(X_train_k[1::2],Y_train_k[1::2])  
   
    # 基于测试评估预测误差   
    #error_list_gp = np.array(Y_test_k.reshape(len(Y_test_k),1)-gp_list[i].get_predict(X_test_k))
    #error_list_gp_j = np.array(Y_test_k.reshape(len(Y_test_k),1)-gp_list[i].get_predict_jiont(X_test_k, X_train_k[::1], Y_train_k[::1]))
    #print('AVE mean gp :',np.mean(abs(np.divide(error_list_gp,Y_test_k.reshape(len(Y_test_k),1) ))))
    #print('AVE mean gp jonit :',np.mean(abs(np.divide(error_list_gp_j,Y_test_k.reshape(len(Y_test_k),1) ))))
    #y_predict = gp_list[i].get_predict_jiont(X_test_k, X_train_k[::1], Y_train_k[::1])
    y_predict = gp_list[i].get_predict(X_test_k)

    #基于测试集评估预测的Kendalltau
    print('Kendalltau:',scipy.stats.stats.kendalltau( y_predict,Y_test_k))


# # 查看测试集

# In[6]:


with open('/home/aistudio/data/data162979/CCF_UFO_test.json', 'r') as f:
    test_data = json.load(f)
test_data['arch99997']


# # 处理测试集数据

# In[7]:


test_arch_list = []
for key in test_data.keys():
    test_arch =  convert_X(test_data[key]['arch'])
    test_arch_list.append(test_arch)
print(test_arch_list[99499])


# # 预测各任务上的测试集的结果

# In[8]:


rank_all = []
for task in range(len(name_list)):
    print('Predict the rank of:', name_list[task])
    # slow mode
    #rank_all.append(gp_list[task].get_predict_jiont(np.array(test_arch_list), np.array(arch_list_train), np.array(train_list[task])))
    # fast mode
    rank_all.append(gp_list[task].get_predict(np.array(test_arch_list)))


# # 生成提交结果

# In[9]:


for idx,key in enumerate(test_data.keys()):
    test_data[key]['cplfw_rank'] = int(rank_all[0][idx][0])
    test_data[key]['market1501_rank'] = int(rank_all[1][idx][0])
    test_data[key]['dukemtmc_rank'] = int(rank_all[2][idx][0])
    test_data[key]['msmt17_rank'] = int(rank_all[3][idx][0])
    test_data[key]['veri_rank'] = int(rank_all[4][idx][0])
    test_data[key]['vehicleid_rank'] = int(rank_all[5][idx][0])
    test_data[key]['veriwild_rank'] = int(rank_all[6][idx][0])
    test_data[key]['sop_rank'] = int(rank_all[7][idx][0])
print('Ready to save results!')
with open('./CCF_UFO_submit_A.json', 'w') as f:
    json.dump(test_data, f)

## 基线fast mode指标如下:
#'avg_tau': 0.7757232329923527
#'tau_cplfw': 0.2882839060350976
#'tau_market1501': 0.8590128953239777
#'tau_dukemtmc': 0.8847439316188985
# 'tau_msmt17': 0.9385055385574282
# 'tau_veri': 0.8901155512936689
# 'tau_vehicleid': 0.6493515264090904
# 'tau_veriwild': 0.9009226525633378
# 'tau_sop': 0.7948498621373224


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
