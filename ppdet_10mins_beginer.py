#!/usr/bin/env python
# coding: utf-8

# # PaddleDetection 快速上手
# 
# 本项目以路标数据集roadsign为例，详细说明了如何使用PaddleDetection训练一个目标检测模型，并对模型进行评估和预测。
# 
# 本项目提供voc格式的roadsign数据集和coco格式的roadsign数据集。
# 
# 本项目提供 YOLOv3、FasterRCNN、FCOS这几个算法的配置文件。
# 
# 您可以选择其中一个配置开始训练，快速体验PaddleDeteciton。
# 
# 效果请戳这里：
# 
# [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/master/README_cn.md)
# 
# ### 欢迎到[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)主页查看更快更好的模型。
# 
# ### 您也可以扫下面的二维码访问PaddleDetection github主页，欢迎关注和点赞^_^。
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/e4a76d874e3c4fec844a262fc571062ff57cadb02e8c49e4a8c5624808a90e93)
# 
# 

# ## 环境安装
# 
# ### 1.  AiStudio环境设置

# In[4]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# In[2]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. All changes under this directory will be kept even after reset. Please clean unnecessary files in time to speed up environment loading.
get_ipython().system('ls /home/aistudio/work')


# In[ ]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, you need to use the persistence path as the following:
get_ipython().system('mkdir /home/aistudio/external-libraries')
get_ipython().system('pip install beautifulsoup4 -t /home/aistudio/external-libraries')


# In[ ]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可:
# Also add the following code, so that every time the environment (kernel) starts, just run the following code:
import sys
sys.path.append('/home/aistudio/external-libraries')


# 
# ### 2. 安装Paddle
# 
# AIStudio上已经安装好paddlepaddle 1.8.4。

# In[3]:


import paddle
print(paddle.__version__)


# ### 3. 克隆PaddleDetection
# 
# 通过以下命令克隆最新的PaddleDetection代码库。
# 
# `! git clone https://github.com/PaddlePaddle/PaddleDetection`
# 
# 
# 如果因为网络问题clone较慢，可以： 
# 1. 通过github加速通道clone
# 
# `git clone https://hub.fastgit.org/PaddlePaddle/PaddleDetection.git`
# 
# 
# 2. 选择使用码云上的托管
# 
# `git clone https://gitee.com/paddlepaddle/PaddleDetection`
# 
# 注：码云托管代码可能无法实时同步本github项目更新，存在3~5天延时，请优先从github上克隆。
# 
# 3. 使用本项目提供的代码库，存放路径`work/PaddleDetection.zip`

# 这里采用项目提供的代码库

# In[5]:


get_ipython().system(' ls ~/work/PaddleDetection.zip')


# In[6]:


get_ipython().run_line_magic('cd', '~/work/')
get_ipython().system(' unzip -o PaddleDetection.zip')


# ### 4. PaddleDetection依赖安装及设置
# 
# 通过如下方式安装PaddleDetection依赖，并设置环境变量

# 安装 cocoapi
# 
# 
# 如果因为网络问题clone较慢，可以： 
# 1. 通过github加速通道clone
# 
# `pip install "git+https://hub.fastgit.org/cocodataset/cocoapi.git#subdirectory=PythonAPI"`

# In[7]:


# github
#! pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"

# fast github
# ! pip install "git+https://hub.fastgit.org/cocodataset/cocoapi.git#subdirectory=PythonAPI"

# 
get_ipython().system(' pip install pycocotools')


# 设置环境

# In[8]:


get_ipython().run_line_magic('cd', '~/work/PaddleDetection/')
get_ipython().system('pip install -r requirements.txt')

get_ipython().run_line_magic('env', 'PYTHONPATH=.:$PYTHONPATH')
get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=0')


# 验证安装

# In[9]:


get_ipython().system(' python ppdet/modeling/tests/test_architectures.py')


# ## 准备数据
# 
# 本项目使用[road-sign-detection](https://www.kaggle.com/andrewmvd/road-sign-detection) 比赛数据，检测4种路标：
# * speedlimit
# * crosswalk
# * trafficlight
# * stop
# 
# 划分成训练集和测试集，总共877张图，其中训练集701张图、测试集176张图。
# 
# 本项目提供voc格式和coco格式的数据：
# 
# 1. voc格式：
# 
# 	划分好的数据下载地址为： [roadsign_voc.tar](https://paddlemodels.bj.bcebos.com/object_detection/roadsign_voc.tar)。
# 
# 	AiStudio上数据地址：[roadsign_voc](https://aistudio.baidu.com/aistudio/datasetdetail/49531)
# 
# 2. coco格式：
# 
# 	划分好的数据下载地址为：：[roadsign_coco.tar](https://paddlemodels.bj.bcebos.com/object_detection/roadsign_coco.tar)。
# 
# 	AiStudio上数据地址：[roadsign_coco](https://aistudio.baidu.com/aistudio/datasetdetail/52968)
# 

# 将`~/data/`文件夹下的数据解压到`PaddleDetection/dataset/`文件夹下。

# In[10]:


get_ipython().run_line_magic('cd', '~/work/PaddleDetection/dataset/')
get_ipython().system(' pwd')
get_ipython().system(' ls ~/data -l')


# #### 1. voc格式数介绍
# 
# 
# VOC数据格式的目标检测数据，是指每个图像文件对应一个同名的xml文件，xml文件中标记物体框的坐标和类别等信息。
# 
# Pascal VOC比赛对目标检测任务，对目标物体是否遮挡、是否被截断、是否是难检测物体进行了标注。对于用户自定义数据可根据实际情况对这些字段进行标注。
# 
# xml文件中包含以下字段：
# 
# * filename，表示图像名称。
# ```
# <filename>road650.png</filename>
# ```
# 
# * size，表示图像尺寸。包括：图像宽度、图像高度、图像深度
# ```
# <size>
# 	<width>300</width>
# 	<height>400</height>
# 	<depth>3</depth>
# </size>
# ```
# * object字段，表示每个物体。包括
# 	
# 	* `name`: 目标物体类别名称
#     * `pose`: 关于目标物体姿态描述（非必须字段）
#     * `truncated`: 目标物体目标因为各种原因被截断（非必须字段）
#     * `occluded`:  目标物体是否被遮挡（非必须字段）
#     * `difficult`: 目标物体是否是很难识别（非必须字段）
#     * `bndbox`: 物体位置坐标，用左上角坐标和右下角坐标表示：`xmin`、`ymin`、`xmax`、`ymax`
# 
# 
# 将`~/data/data49531/roadsign_voc.tar`解压到`PaddleDetection/dataset/roadsign_voc`下

# In[12]:


get_ipython().run_line_magic('cd', '~/work/PaddleDetection/dataset/roadsign_voc/')
get_ipython().system(' pwd')
get_ipython().system(' ls -h')


# In[13]:


# copy roadsign_voc.tar and extract
get_ipython().system(' cp ~/data/data49531/roadsign_voc.tar .')
get_ipython().system(' tar -xvf roadsign_voc.tar')
get_ipython().system(' rm -rf roadsign_voc.tar')


# In[14]:


# 查看一条数据
get_ipython().system(' cat ./annotations/road650.xml')


# #### 2. coco格式数介绍
# 
# coco数据格式，是指将所有训练图像的标注都存放到一个json文件中。数据以字典嵌套的形式存放。
# 
# json文件中存放了 `info licenses images annotations categories`的信息:
# 
# * info中存放标注文件标注时间、版本等信息。
# * licenses中存放数据许可信息。
# * images中存放一个list，存放所有图像的图像名，下载地址，图像宽度，图像高度，图像在数据集中的id等信息。
# * annotations中存放一个list，存放所有图像的所有物体区域的标注信息，每个目标物体标注以下信息：
#     
# ```
#     {
#     	'area': 899, 
#     	'iscrowd': 0, 
#         'image_id': 839, 
#         'bbox': [114, 126, 31, 29], 
#         'category_id': 0, 'id': 1, 
#         'ignore': 0, 
#         'segmentation': []
#     }
# ```
# 
# 将`~/data/data49531/roadsign_coco.tar`解压到`PaddleDetection/dataset/roadsign_coco`下

# In[15]:


get_ipython().run_line_magic('cd', '~/work/PaddleDetection/dataset/')
get_ipython().system(' mkdir roadsign_coco')
get_ipython().run_line_magic('cd', '~/work/PaddleDetection/dataset/roadsign_coco/')
get_ipython().system(' pwd')


# In[16]:


# copy roadsign_coco.tar and extract
get_ipython().system(' cp ~/data/data52968/roadsign_coco.tar .')
get_ipython().system(' tar -xf roadsign_coco.tar')
get_ipython().system(' rm -rf roadsign_coco.tar')


# In[17]:


get_ipython().system(' cat ./annotations/train.json')


# In[18]:


# 查看一条数据
import json
coco_anno = json.load(open('./annotations/train.json'))

# coco_anno.keys
print('\nkeys:', coco_anno.keys())

# 查看类别信息
print('\n物体类别:', coco_anno['categories'])

# 查看一共多少张图
print('\n图像数量：', len(coco_anno['images']))

# 查看一共多少个目标物体
print('\n标注物体数量：', len(coco_anno['annotations']))

# 查看一条目标物体标注信息
print('\n查看一条目标物体标注信息：', coco_anno['annotations'][0])


# ## 开始训练
# 本项目在`work/hw_configs/`目录下提供以下配置文件
# 
# * yolov3_mobilenet_v1_roadsign_voc_template.yml
# * yolov3_mobilenet_v1_roadsign_coco_template.yml
# * ppyolo_resnet50_vd_roadsign_coco_template.yml
# * faster_rcnn_r50_roadsign_coco_template.yml
# * faster_rcnn_r50_vd_fpn_roadsign_coco_template.yml
# * fcos_r50_roadsign_coco_template.yml

# 将`~/work/hw_configs.zip`解压到 `configs` 文件夹下

# In[19]:


get_ipython().run_line_magic('cd', '~/work/PaddleDetection/')

get_ipython().system('unzip -o ~/work/hw_configs.zip -d configs/')

get_ipython().system(' ls configs/hw_configs/')


# In[ ]:


# 选择配置开始训练。可以通过 -o 选项覆盖配置文件中的参数

# faster_rcnn_r50_vd_fpn
get_ipython().system(' python -u tools/train.py -c configs/hw_configs/faster_rcnn_r50_vd_fpn_roadsign_coco_template.yml -o use_gpu=True --eval')

# yolov3
#! python -u tools/train.py -c configs/hw_configs/yolov3_mobilenet_v1_roadsign_voc_template.yml -o use_gpu=True --eval

# fcos
#! python -u tools/train.py -c configs/hw_configs/fcos_r50_roadsign_coco_template.yml -o use_gpu=True --eval


# 您可以通过指定visualDL可视化工具，对loss变化曲线可视化。您仅需要指定 `use_vdl` 参数和 `vdl_log_dir` 参加即可。
# 
# 点击左侧 **可视化** 按钮，设置 `logdir` 和模型文件，就可以对训练过程loss变化曲线和模型进行可视化。

# In[20]:


# 选择配置开始训练。可以通过 -o 选项覆盖配置文件中的参数 vdl_log_dir 设置vdl日志文件保存路径

# faster_rcnn_r50_vd_fpn
get_ipython().system(' python -u tools/train.py -c configs/hw_configs/faster_rcnn_r50_vd_fpn_roadsign_coco_template.yml -o use_gpu=True --use_vdl=True --vdl_log_dir=vdl_dir/scalar --eval')

# yolov3
#! python -u tools/train.py -c configs/hw_configs/yolov3_mobilenet_v1_roadsign_voc_template.yml -o use_gpu=True --use_vdl=True --vdl_log_dir=vdl_dir/scalar --eval

# fcos
#! python -u tools/train.py -c configs/hw_configs/fcos_r50_roadsign_coco_template.yml -o use_gpu=True --use_vdl=True --vdl_log_dir=vdl_dir/scalar --eval


# ## 评估和预测

# PaddleDetection也提供了`tools/eval.py`脚本用于评估模型，评估是可以通过`-o weights=`指定待评估权重。
# 
# PaddleDetection训练过程中若开始了`--eval`，会将所有checkpoint中评估结果最好的checkpoint保存为`best_model.pdparams`，可以通过如下命令一键式评估最优checkpoint
# 
# 这里我们加载预训练好的权重进行预测：
# * https://paddlemodels.bj.bcebos.com/object_detection/yolov3_best_model_roadsign.pdparams
# * https://paddlemodels.bj.bcebos.com/object_detection/faster_r50_fpn_best_model_roadsign.pdparams
# * https://paddlemodels.bj.bcebos.com/object_detection/fcos_best_model_roadsign.pdparams

# In[21]:


# 评估

# faster_rcnn_r50_vd_fpn
get_ipython().system(' python -u tools/eval.py -c configs/hw_configs/faster_rcnn_r50_vd_fpn_roadsign_coco_template.yml -o use_gpu=True weights=https://paddlemodels.bj.bcebos.com/object_detection/faster_r50_fpn_best_model_roadsign.pdparams')

# yolov3
#! python -u tools/eval.py -c configs/hw_configs/yolov3_mobilenet_v1_roadsign_coco_template.yml -o use_gpu=True weights=https://paddlemodels.bj.bcebos.com/object_detection/yolov3_best_model_roadsign.pdparams

# fcos
#! python -u tools/eval.py -c configs/hw_configs/fcos_r50_roadsign_coco_template.yml -o use_gpu=True weights=https://paddlemodels.bj.bcebos.com/object_detection/fcos_best_model_roadsign.pdparams


# PaddleDetection提供了tools/infer.py预测工具，可以使用训练好的模型预测图像并可视化，通过-o weights=指定加载训练过程中保存的权重。
# 
# 预测脚本如下：

# In[22]:


img_path = './dataset/roadsign_voc/images/road554.png'

# faster_rcnn_r50_vd_fpn
get_ipython().system(' python tools/infer.py -c configs/hw_configs/faster_rcnn_r50_vd_fpn_roadsign_coco_template.yml -o use_gpu=True weights=https://paddlemodels.bj.bcebos.com/object_detection/faster_r50_fpn_best_model_roadsign.pdparams --infer_img=dataset/roadsign_voc/images/road554.png')

# yolov3
#! python tools/infer.py -c configs/hw_configs/yolov3_mobilenet_v1_roadsign_voc_template.yml -o use_gpu=True weights=https://paddlemodels.bj.bcebos.com/object_detection/yolov3_best_model_roadsign.pdparams --infer_img=dataset/roadsign_voc/images/road554.png

# fcos
#! python tools/infer.py -c configs/hw_configs/fcos_r50_roadsign_coco_template.yml -o use_gpu=True weights=https://paddlemodels.bj.bcebos.com/object_detection/fcos_best_model_roadsign.pdparams --infer_img=dataset/roadsign_voc/images/road554.png


# In[23]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
import cv2

infer_img = cv2.imread("output/road554.png")
plt.figure(figsize=(15,10))
plt.imshow(cv2.cvtColor(infer_img, cv2.COLOR_BGR2RGB))
plt.show()
# 23.5.17 图中选中了“限速牌”，Bbox预测正确，但是类别“person”应该是有问题的


# ## 模型压缩
# 
# #### 如果您要对模型进行压缩，PaddleDetection中[模型压缩](https://github.com/PaddlePaddle/PaddleDetection/tree/release/0.4/slim)部分提供以下模型压缩方式：
# * [量化](https://github.com/PaddlePaddle/PaddleDetection/tree/release/0.4/slim/quantization)
# * [剪枝](https://github.com/PaddlePaddle/PaddleDetection/tree/release/0.4/slim/prune)
# * [蒸馏](https://github.com/PaddlePaddle/PaddleDetection/tree/release/0.4/slim/distillation)
# * [搜索](https://github.com/PaddlePaddle/PaddleDetection/tree/release/0.4/slim/nas)
# 

# ## 模型部署
# 
# #### 如果您要部署模型，请参考[模型部署](https://github.com/PaddlePaddle/PaddleDetection/tree/release/0.4/slim)部分提供以下部署方式：
# 
# * [服务器端Python部署](https://github.com/PaddlePaddle/PaddleDetection/tree/release/0.4/deploy/python)
# * [服务器端C++部署](dhttps://github.com/PaddlePaddle/PaddleDetection/tree/release/0.4/eploy/cpp)
# * [移动端部署](https://github.com/PaddlePaddle/Paddle-Lite-Demo)
# * [在线Serving部署](https://github.com/PaddlePaddle/Serving)

# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# In[ ]:




