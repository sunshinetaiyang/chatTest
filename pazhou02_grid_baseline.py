#!/usr/bin/env python
# coding: utf-8

# # 第二届广州·琶洲算法大赛-基于复杂场景的输电通道隐患目标检测算法baseline（非官方）
# ![](https://ai-studio-static-online.cdn.bcebos.com/b94334380fb347f18975b5dfe3cf2fa3d7dee70e7a394631af74d009656aa077)
# 

# ## 1 赛题介绍（赛题名称：基于复杂场景的输电通道隐患目标检测算法）
# 
# 本次比赛旨在推动科技创新发展，提高粤港澳大湾区电力系统监测的效率和准确性，减少人为差错和风险，提高电力系统的生产安全和稳定性。竞赛面临的难点在于，输电通道检测具有复杂场景，存在多种干扰因素，例如天气变化、背景环境、器材性能、光线等等问题，这些因素对图像质量的影响较大。此外，不同型号、不同年限的杆塔、设备类型等因素会导致目标的种类和形状有所不同，对算法的识别能力提出了更高的要求。
# 
# 本次比赛对参赛者的吸引点在于，参赛者可以在实战场景中锻炼和提高自己的技术水平，借助比赛平台，通过算法优化和迭代，提高算法识别效率和精度。这不仅可以推进人工智能技术在各个行业的应用，还可为参赛者提供更广阔的职业发展空间。
# 
# 随着我国能源行业的迅速发展，给输电通道的运行维护带来巨大挑战，促进了无人机等智能运维技术的快速发展。为解决大量无人机拍摄的输配电线路图片识别缺陷和隐患人力成本高、效率低的问题，人工智能技术在能源行业被大量运用。本次比赛旨在基于无人机拍摄的图像和深度学习技术，自动化查找隐患、准确定位、返回检测结果并进行可视化。
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/3e24eb636a564314b3c09bafa279b37b1ae041ea90764a3aadb58c099aba7f5c)
# 
# 

# ## 2 赛题解读
# 
# 随着我国能源行业的迅速发展，给输电通道的运行维护带来巨大挑战，促进了无人机等智能运维技术的快速发展。为解决大量无人机拍摄的输配电线路图片识别缺陷和隐患人力成本高、效率低的问题，人工智能技术在能源行业被大量运用。本次比赛旨在基于无人机拍摄的图像和深度学习技术，自动化查找隐患、准确定位、返回检测结果并进行可视化。
# 
# 本赛题技术方向为人工智能计算机视觉，主要为目标检测技术。本赛题中输电通道隐患主要是指鸟巢、导线异物（气球、风筝、垃圾等），这些异物可能导致输电线路出现短路、接地等故障，导致线路停电，影响供电可靠性和稳定性。
# 
# ### 2.1 本赛题难点： 
# 
# 1. 需要找到存在缺陷的图片，并准确定位缺陷位置、标明缺陷类型； 
# 
# 2. 缺陷形式多样，对算法鲁棒性要求高，例如气球样式多、垃圾种类多；
# 
# 3. 异物与设备位置关系多样，例如部分风筝并未直接接触导线，而是通过风筝线与导线连接。
# 
# 参赛者需要通过已知的数据集进行整理、清洗，搭建模型，实现目标检测功能，并将结果按照指定格式输出至指定路径。
# 
# 以下是示意图例：
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/e9055be893cc4d82b5a5657efa8597499c37e53b2e924d7eb0dd4e5d838c8aed)
# 
# ### 2.2 评价指标
# 
# 本赛题根据参赛队伍提供的检测结果计算F1进行评分，详细计算公式为：
# 
# * 精确率(Precision)：即真正例(TP)占所有被分类为正例(TP+FP)的比例，公式为: Precision = TP / (TP + FP)
# 
# * 召回率(Recall)：即真正例(TP)占所有真实正例(TP+FN)的比例，公式为: Recall = TP / (TP + FN)
#  
# * F1-Score：综合Precision和Recall，公式为： F1-Score = (2 * Precision * Recall) / (Precision + Recall)

# ## 3 赛题数据解析
# 
# ### 3.1 数据介绍
# 
# 本次赛题的数据集来源多个渠道，使用高清摄像机、普通摄像机等多种设备拍摄。数据来自各类型输电线路设备，背景包括城市、乡村、山地、道路、农田等。 数据集包含1000张输电通道图像，大小约为3GB，由比赛选手自行拆分训练集和验证集。数据集中目标的种类包括鸟巢、风筝、气球、垃圾，图片为jpg格式，标注文件为xml格式。数据集中每个目标都被标注了专业准确的边界框和类别。 数据集的独特优势在于：数据来源广泛、数量庞大、种类清晰、边界框标注准确。这些特点使得该数据集适用于复杂场景下的隐患目标检测算法的开发和应用。同时，数据量足够大，可以有效提高算法的鲁棒性和泛化性能，从而达到更好的应用效果。 参赛者可以基于该数据集进行算法的训练和测试，并结合数据集的独特特点进行算法设计，以期达到更好的检测效果和性能。
# 
# <style>
# .center 
# {
#   width: auto;
#   display: table;
#   margin-left: auto;
#   margin-right: auto;
# }
# </style>
# 
# <div class="center">
# 
# | 数据集名称 | 输电通道的隐患目标数据集 |
# | :--------: | :--------: |
# | 数据来源 | 多种设备拍摄 |
# | 数据量 | 1000张图像，大小约为3GB |
# | 数据内容 | 各类型输电线路设备，不同背景 |
# | 标注信息 | 准确的边界框和类别标注 |
# | 目标种类 | 鸟巢、风筝、气球、垃圾等 | 
# | 优势 | 来源广泛、数量庞大、标注准确 |
# | 使用价值 | 开发隐患目标检测算法 |
# | 针对参赛者 | 提供基础数据，便于开发算法 |
#     
# </div>
# 
# ### 3.2 数据说明
# 
# 具体格式说明如下：
# 
# * 文件名称说明： 数据集中的图像文件以“.jpg”为后缀，标注文件同名，只是后缀名为“.xml”。
# 
# * 文件编码说明： 标注文件采用UTF-8编码。
# 
# * 文件中的数据格式说明： 标注文件采用XML格式，格式如下：
# 
# <style>
# .center 
# {
#   width: auto;
#   display: table;
#   margin-left: auto;
#   margin-right: auto;
# }
# </style>
# 
# <div class="center">
# 
# ```xml
# <?xml version="1.0" encoding="utf-8"?>
# <annotation>
#     <filename> 0000001.jpg</filename>
#     <size>
#         <width>1080</width>
#         <height>1080</height>
#         <depth>3</depth>
#     </size>
#     <object>
#         <name>balloon</name>
#         <bndbox>
#             <xmin>100</xmin>
#             <ymin>200</ymin>
#             <xmax>300</xmax>
#             <ymax>400</ymax>
#         </bndbox>
#     </object>
#     <object>
#         <name>trash</name>
#         <bndbox>
#             <xmin>700</xmin>
#             <ymin>500</ymin>
#             <xmax>800</xmax>
#             <ymax>700</ymax>
#         </bndbox>
#     </object>
# </annotation>
# ```  
# </div>
# 
# 标注文件中各字段含义：
# 
# <style>
# .center 
# {
#   width: auto;
#   display: table;
#   margin-left: auto;
#   margin-right: auto;
# }
# </style>
# 
# <div class="center">
# 
# | < filename > | 每张图像对应的文件名，格式为“xxx.jpg” | 
# | :--------: | :--------: |
# | < size > | 存放图像尺寸信息，包括宽、高、通道数 |
# | < object > | 存放检测到的目标信息 |
# | < name > | 存放目标类别名称 |
# | < bndbox > | 存放目标的位置信息 |
# | < xmin > | 目标边界框左上角的横坐标 |
# | < ymin > | 目标边界框左上角的纵坐标 |
# | < xmax > | 目标边界框右下角的横坐标 |
# | < ymax > | 目标边界框右下角的纵坐标 |
# 
# </div>

# In[2]:


# 训练环境准备
!git clone https://gitee.com/paddlepaddle/PaddleDetection.git # 23.5.23仅需run一次


# ## 4 数据预处理
# 
# 

# ### 4.1 解压数据集
# 

# In[4]:


# 解压数据集
!unzip -oq ~/data/data212110/train.zip -d ~/PaddleDetection/dataset/voc
!unzip -oq ~/data/data212110/val.zip -d ~/PaddleDetection/dataset/voc


# In[5]:


# # 将标注和图片分开
%cd ~/PaddleDetection/dataset/voc
!mkdir JPEGImages Annotations
!cp -r train/*.xml Annotations
!cp -r train/*.jpg JPEGImages

# In[6]:
!rm -rf /home/aistudio/PaddleDetection/dataset/voc/train
!rm -rf /home/aistudio/PaddleDetection/dataset/voc/val # 23.5.25 不能删


# ### 4.2 划分数据集

# In[8]:


# import random
# import os
# #生成trainval.txt和val.txt
random.seed(2020)
xml_dir  = '/home/aistudio/PaddleDetection/dataset/voc/Annotations'#标签文件地址
img_dir = '/home/aistudio/PaddleDetection/dataset/voc/JPEGImages'#图像文件地址
path_list = list()
for img in os.listdir(img_dir):
    img_path = os.path.join(img_dir,img)
    xml_path = os.path.join(xml_dir,img.replace('jpg', 'xml'))
    path_list.append((img_path, xml_path))
random.shuffle(path_list)
ratio = 0.9
train_f = open('/home/aistudio/PaddleDetection/dataset/voc/trainval.txt','w') #生成训练文件
val_f = open('/home/aistudio/PaddleDetection/dataset/voc/val.txt' ,'w')#生成验证文件

for i ,content in enumerate(path_list):
    img, xml = content
    text = img + ' ' + xml + '\n'
    if i < len(path_list) * ratio:
        train_f.write(text)
    else:
        val_f.write(text)
train_f.close()
val_f.close()

#生成标签文档
label = ['nest', 'kite', 'balloon', 'trash']#设置你想检测的类别
with open('/home/aistudio/PaddleDetection/dataset/voc/label_list.txt', 'w') as f:
    for text in label:
        f.write(text+'\n')


# ### 4.3 环境准备
# 
# 由于PP-YOLOE还在快速迭代中，因此，对框架的稳定性有一定的要求，PaddlePaddle的框架不要选择最新版。本文使用的单卡训练环境如下：
# 
# * 框架版本：PaddlePaddle 2.4.0
# 
# * CUDA Version: 11.2
# 
# * 模型库版本：PaddleDetection(release/2.5分支)
# 

# ### 4.4 VOC2COCO
# 
# 若需要COCO数据集格式，可以运行以下命令进行转换

# In[1]:


import os
import os.path
import xml.dom.minidom
path = r'/home/aistudio/PaddleDetection/dataset/voc/Annotations'
files = os.listdir(path)  # 得到文件夹下所有文件名称
s = []
count = 0
for xmlFile in files:  # 遍历文件夹
    if not os.path.isdir(xmlFile):  # 判断是否是文件夹,不是文件夹才打开
            name1 = xmlFile.split('.')[0]
            dom = xml.dom.minidom.parse(path + '/' + xmlFile)
            root = dom.documentElement
            newfolder = root.getElementsByTagName('folder')
            newpath = root.getElementsByTagName('path')
            newfilename = root.getElementsByTagName('filename')
            newfilename[0].firstChild.data = name1 + '.jpg'
            with open(os.path.join(path, xmlFile), 'w') as fh:
                dom.writexml(fh)
                print('写入成功')
            count = count + 1


# In[10]:


# # 将训练集转换为COCO格式
%cd ~/PaddleDetection
!python tools/x2coco.py \
        --dataset_type voc \
        --voc_anno_dir dataset/voc/Annotations/ \
        --voc_anno_list dataset/voc/trainval.txt \
        --voc_label_list dataset/voc/label_list.txt \
        --voc_out_name dataset/voc/train.json


# In[11]:


# 将验证集转换为COCO格式
%cd ~/PaddleDetection
!python tools/x2coco.py \
        --dataset_type voc \
        --voc_anno_dir dataset/voc/Annotations/ \
        --voc_anno_list dataset/voc/val.txt \
        --voc_label_list dataset/voc/label_list.txt \
        --voc_out_name dataset/voc/val.json


# ### 4.5 统计数据集分布
# 
# 在切图前，我们首先需要统计所用数据集标注框的平均宽高占图片真实宽高的比例分布：
# 
# * --json_path ：待统计数据集COCO 格式 annotation 的json文件路径
# 
# * --out_img ：输出的统计分布图路径

# In[1]:


# # # 23.5.23 安装pycocotools
# !mkdir /home/aistudio/external-libraries
# !pip install beautifulsoup4 -t /home/aistudio/external-libraries
get_ipython().system('pip install pycocotools ')
# !pip install pycocotools  -t /home/aistudio/external-libraries
# !pip install -U scikit-image imagecodecs  -t /home/aistudio/external-libraries


# In[1]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可:
import sys
sys.path.append('/home/aistudio/external-libraries')


# In[1]:


get_ipython().run_line_magic('cd', '~/PaddleDetection')

# 统计数据集分布
get_ipython().system(
'python tools/box_distribution.py     --json_path dataset/voc/train.json     --out_img /home/aistudio/PaddleDetection/dataset/voc/box_distribution.jpg')


# ### 4.6 基于SAHI切图
# 
# 针对需要切图的数据集，使用SAHI库进行切分：
# 
# * --image_dir：原始数据集图片文件夹的路径
# 
# * --json_path：原始数据集COCO格式的json标注文件的路径
# 
# 
# * --output_dir：切分后的子图及其json标注文件保存的路径
# 
# * --slice_size：切分以后子图的边长尺度大小(默认切图后为正方形)
# 
# * --overlap_ratio：切分时的子图之间的重叠率
# 
# 以上述代码为例，切分后的子图文件夹与json标注文件共同保存在MyDataset/IMG_sliced文件夹下，比如训练集图片和标注就命名为train_images_640_025、train_images_640_025.json

# In[15]:


# !mkdir dataset/voc/IMG_sliced


# In[2]:


# !pip install -U scikit-image imagecodecs


# In[17]:


# 对训练集标注进行切图
!python tools/slice_image.py \
        --image_dir /home/aistudio/PaddleDetection/dataset/voc/JPEGImages \
        --json_path /home/aistudio/PaddleDetection/dataset/voc/train.json \
        --output_dir /home/aistudio/PaddleDetection/dataset/voc/IMG_sliced \
        --slice_size 640 \
        --overlap_ratio 0.25

# In[18]:
# 对验证集标注进行切图
!python tools/slice_image.py \
        --image_dir /home/aistudio/PaddleDetection/dataset/voc/JPEGImages \
        --json_path /home/aistudio/PaddleDetection/dataset/voc/val.json \
        --output_dir /home/aistudio/PaddleDetection/dataset/voc/IMG_sliced \
        --slice_size 640 \
        --overlap_ratio 0.25


# ## 5 模型训练
# 
# 由于图像尺寸较大，且训练模型时，读取图像和预处理时耗时较长，因此我们选择拼图模型进行训练。

# ### 5.1 原图模型训练
# 
# PP-YOLOE+_l在COCO test-dev2017达到了53.3的mAP, 同时其速度在Tesla V100上达到了78.1 FPS。PP-YOLOE+_s/m/x同样具有卓越的精度速度性价比, 其精度速度可以在模型库中找到。

# In[19]:


# !cp -r /home/aistudio/work/eda /home/aistudio/PaddleDetection/
# !cp -r /home/aistudio/work/output /home/aistudio/PaddleDetection/


# In[6]:


# %cd ~/PaddleDetection
# !python setup.py install


# In[8]:


get_ipython().system('cat eda/ppyoloe/ppyoloe_plus_crn_s_80e_eda.yml')


# In[3]:


# 23.5.24 修改了datasets/eda_detection.yml到正确的大写PaddleDetection目录，但是内存不够挂死
get_ipython().run_line_magic('cd', '~/PaddleDetection')
get_ipython().system('
python tools/train.py -c eda/ppyoloe/ppyoloe_plus_crn_s_80e_eda.yml --use_vdl=True --vdl_log_dir=./ori_log --eval --amp')


# ### 5.2 拼图模型选型
# PaddleDetection团队提供的基于PP-YOLOE的检测模型，以及提供了一套使用SAHI(Slicing Aided Hyper Inference)工具切图和拼图的方案，其效果如下：
# 
# |模型|	数据集|	SLICE_SIZE	|OVERLAP_RATIO	|类别数	|mAP 0.5:0.95	|AP 0.5	|
# | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
# |PP-YOLOE-l|	VisDrone-DET|	640|	0.25|	10	|29.7	|48.5|	
# |PP-YOLOE-l (Assembled)	|VisDrone-DET	|640	|0.25|	10	|37.2	|59.4|	
# 
# Assembled表示自动切图和拼图后模型的表现，从中我们可以看出，mAP较原图预测有了非常显著的提升，因此，接下来就基于PP-YOLOE-l，看看自动切图和拼图后模型在这个比赛检测数据集上的表现。

# ### 5.3 模型训练
# 
# 切图后模型的训练是要基于切图数据集的，配置如下：
# 
```
metric: COCO
num_classes: 4

TrainDataset:
  !COCODataSet
    image_dir: /home/aistudio/paddledetection/dataset/voc/IMG_sliced/train_images_640_025
    anno_path: /home/aistudio/paddledetection/dataset/voc/IMG_sliced/train_640_025.json
    dataset_dir: /home/aistudio/paddledetection/dataset/voc
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

EvalDataset:
  !COCODataSet
    image_dir: /home/aistudio/paddledetection/dataset/voc/IMG_sliced/val_images_640_025
    anno_path: /home/aistudio/paddledetection/dataset/voc/IMG_sliced/val_640_025.json
    dataset_dir: /home/aistudio/paddledetection/dataset/voc

TestDataset:
  !ImageFolder
    anno_path: val_640_025.json
    dataset_dir: /home/aistudio/paddledetection/dataset/voc/IMG_sliced

```
# In[3]:


# 23.5.25 分析拼图模型并尝试运行
get_ipython().run_line_magic('cd', '~/PaddleDetection')
get_ipython().system('cat eda/eda_model/ppyoloe_crn_l_80e_sliced_eda.yml')
# In[5]:
get_ipython().system('
python tools/train.py -c eda/eda_model/ppyoloe_crn_l_80e_sliced_eda.yml --use_vdl=True --vdl_log_dir=./log --eval ')


# ### 5.4 模型评估
# 我们对训练80个epoch后，切图模型的效果进行一下评估。
# 
# #### 5.4.1 子图评估
# 
# 对于子图评估和原图评估，差别仅仅在于验证集路径的配置：
# 
# * 子图评估
# 
#   配置切图后的子图存放目录和子图验证集标注文件
#   
# * 原图评估
# 
#   配置原图存放目录和验证集标注文件

# In[6]:


# 训练80个epoch后，子图评估
get_ipython().run_line_magic('cd', '~/PaddleDetection')
get_ipython().system('python tools/eval.py     -c eda/eda_model/ppyoloe_crn_l_80e_sliced_eda.yml     -o weights=output/ppyoloe_crn_l_80e_sliced_eda/best_model.pdparams')


# In[ ]:


# 训练80个epoch后，原图评估
get_ipython().system('python tools/eval.py         -c eda/eda_model/ppyoloe_crn_l_80e_sliced_eda.yml         -o weights=output/ppyoloe_crn_l_80e_sliced_eda/best_model.pdparams')


# #### 5.4.2 子图拼图评估
# 
# 执行命令时，需要关注下面这些参数的设置：
# 
# * 设置--combine_method表示子图结果重组去重的方式，默认是nms；
# 
# * 设置--match_threshold表示子图结果重组去重的阈值，默认是0.6；
# 
# * 设置--match_metric表示子图结果重组去重的度量标准，默认是ios表示交小比(两个框交集面积除以更小框的面积)，也可以选择交并比iou(两个框交集面积除以并集面积)，精度效果因数据集而而异，但选择ios预测速度会更快一点。
# 
# 在本项目中，我们把交小比和交并比两种方式都试验一番：

# In[8]:


# 训练80个epoch后，子图拼图评估，交小比
get_ipython().system('python tools/eval.py     -c eda/eda_model/ppyoloe_crn_l_80e_sliced_eda_slice_infer.yml     -o weights=output/ppyoloe_crn_l_80e_sliced_eda/best_model.pdparams     --slice_infer     --combine_method=nms     --match_threshold=0.6     --match_metric=ios')


# In[3]:

get_ipython().run_line_magic('cd', '~/PaddleDetection')

# 训练80个epoch后，子图拼图评估，交并比
get_ipython().system('python tools/eval.py     -c eda/eda_model/ppyoloe_crn_l_80e_sliced_eda_slice_infer.yml     -o weights=output/ppyoloe_crn_l_80e_sliced_eda/best_model.pdparams     --slice_infer     --combine_method=nms     --match_threshold=0.6     --match_metric=iou')


# #### 5.4.3 评估结论
# 
# 将80个epoch后，切图训练子图评估、切图训练原图评估、切图训练子图拼图评估的效果进行了对比，结果如下：
# 
# |训练评估方式	|SLICE_SIZE	|OVERLAP_RATIO|	mAP0.5:0.95	|AP0.5|
# | :--------: | :--------: | :--------: |:--------: |:--------: |
# |子图训练子图评估|640	|0.25	|78.0	|94.8|
# |子图训练原图评估|640	|0.25	|69.3	|81.5|
# |子图训练拼图评估-IoS|640	|0.25	|47.6	|66.8|
# |子图训练拼图评估-IoU|640	|0.25	|47.1	|71.1|
# 
# 从上面的简单表格可以明显看出，相比原图直接训练，子图训练拼图评估精度提升明显。同时，子图结果重组去重的度量标准用交并比在该数据集上表现更好，猜测可能是因为，数据集的目标大小差异较大，对于中大型目标，去重标准如果用交小比，在评估效果上，会比较吃亏。

# ### 5.5 模型预测

# #### 5.5.1 单图预测

# In[4]:


# 挑一张测试集的图片展示预测效果
get_ipython().system('python tools/infer.py     -c eda/eda_model/ppyoloe_crn_l_80e_sliced_eda.yml     -o weights=output/ppyoloe_crn_l_80e_sliced_eda/best_model.pdparams     --infer_img=/home/aistudio/PaddleDetection/dataset/voc/val/i1wJLsAZbpvD3mNWeK8Hfl7xrPC9cMqT02So4YyF.jpg     --draw_threshold=0.4     # --slice_infer \\')
    # --slice_size 640 640 \
    # --overlap_ratio 0.25 0.25 \
    # --combine_method=nms \
    # --match_threshold=0.6 \
    # --match_metric=iou \
    # --save_results=True


# #### 5.5.2 批量预测
# 

# In[5]:


# 执行批量预测
get_ipython().system('
python tools/infer.py     
-c eda/eda_model/ppyoloe_crn_l_80e_sliced_eda.yml     
-o weights=output/ppyoloe_crn_l_80e_sliced_eda/best_model.pdparams     
--infer_dir=/home/aistudio/PaddleDetection/dataset/voc/val     
--save_results=True')


# ![](https://ai-studio-static-online.cdn.bcebos.com/6a9d5cc3f03847d38f44e91e27abf361a87f29bc0ebd4a1e99c39580ca104de6)
# 

# ### 5.6 模型导出
# 
# 

# In[ ]:


get_ipython().run_line_magic('cd', '~/PaddleDetection')
get_ipython().system('
python tools/export_model.py     
-c eda/eda_model/ppyoloe_crn_l_80e_sliced_eda.yml     
--output_dir=./inference_model     
-o weights=output/ppyoloe_crn_l_80e_sliced_eda/best_model')


# ### 5.7 结果文件生成

# In[ ]:
get_ipython().run_line_magic('cd', '~/PaddleDetection')
get_ipython().system('cp -r /home/aistudio/work/infer.py /home/aistudio/PaddleDetection/deploy/python/')
get_ipython().system('
python deploy/python/infer.py     
--model_dir=inference_model/ppyoloe_crn_l_80e_sliced_eda     
--image_dir=/home/aistudio/PaddleDetection/dataset/voc/val     
--device=GPU     --output_dir infer_output     --save_results')
get_ipython().system('mkdir submit/')
get_ipython().system('mv infer_output/bbox.json submit/')

输出：
  File "deploy/python/infer.py", line 426, in predict_image
    image_list, results, use_coco_category=FLAGS.use_coco_category)
  File "deploy/python/infer.py", line 471, in save_coco_results
    fr = open('/home/aistudio/val_imgID.txt', 'r')
FileNotFoundError: [Errno 2] No such file or directory: '/home/aistudio/val_imgID.txt'
# ## 6 提分技巧
# 
# * 针对本赛题的数据集特点，可以选用更不同尺度的图片切片大小，进行模型训练。
# 
# * 针对训练环境适当调整部分参数，以提升模型精度。
# 
# * 选择精度更高的目标检测模型或适当优化模型的部分结构。
# 

# ## 7 小结
# 
# 模型还支持GPU上的推理部署，对于时间要求不高、精度要求极高的大图目标检测场景，具有良好的潜在应用价值。

# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
