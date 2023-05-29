#!/usr/bin/env python
# coding: utf-8

# ![](https://ai-studio-static-online.cdn.bcebos.com/4c15a9d3efad44c18c17d0893c53142f918f079771b74fd1a8418648a9421c6a)

# ## 一、 赛题背景

# 随着我国能源行业的迅速发展，给输电通道的运行维护带来巨大挑战，促进了无人机等智能运维技术的快速发展。为解决大量无人机拍摄的输配电线路图片识别缺陷和隐患人力成本高、效率低的问题，人工智能技术在能源行业被大量运用。本次比赛旨在基于无人机拍摄的图像和深度学习技术，自动化查找隐患、准确定位、返回检测结果并进行可视化。
# 
# 本次比赛旨在推动科技创新发展，提高粤港澳大湾区电力系统监测的效率和准确性，减少人为差错和风险，提高电力系统的生产安全和稳定性。竞赛面临的难点在于，输电通道检测具有复杂场景，存在多种干扰因素，例如天气变化、背景环境、器材性能、光线等等问题，这些因素对图像质量的影响较大。此外，不同型号、不同年限的杆塔、设备类型等因素会导致目标的种类和形状有所不同，对算法的识别能力提出了更高的要求。
# 
# 本次比赛对参赛者的吸引点在于，参赛者可以在实战场景中锻炼和提高自己的技术水平，借助比赛平台，通过算法优化和迭代，提高算法识别效率和精度。这不仅可以推进人工智能技术在各个行业的应用，还可为参赛者提供更广阔的职业发展空间。
# 
# 以下是示意图例：
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/dd5280a1347844bb9ea15c7c883495e07022ac64ff104b4b9f16c4fac71a6cfb)

# ## 二、 赛题介绍

# 本赛题技术方向为人工智能计算机视觉，主要为目标检测技术。本赛题中输电通道隐患主要是指鸟巢、导线异物（气球、风筝、垃圾等），这些异物可能导致输电线路出现短路、接地等故障，导致线路停电，影响供电可靠性和稳定性。
# 
# **本赛题难点：** 
# 1. 需要找到存在缺陷的图片，并准确定位缺陷位置、标明缺陷类型； 
# 2. 缺陷形式多样，对算法鲁棒性要求高，例如气球样式多、垃圾种类多；
# 3. 异物与设备位置关系多样，例如部分风筝并未直接接触导线，而是通过风筝线与导线连接。
# 
# 参赛者需要通过已知的数据集进行整理、清洗，搭建模型，实现目标检测功能，并将结果按照指定格式输出至指定路径。
# 
# 以下是示意图例：
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/fde175111b55406f97f8f469239264fdb1737533d28a4081baef8810db27c7cd)

# **评价指标：**
# 
# 本赛题根据参赛队伍提供的检测结果计算F1进行评分，详细计算公式为：
# 1. 精确率(Precision)：即真正例(TP)占所有被分类为正例(TP+FP)的比例，公式为: Precision = TP / (TP + FP)
# 2. 召回率(Recall)：即真正例(TP)占所有真实正例(TP+FN)的比例，公式为: Recall = TP / (TP + FN)
# 3. F1-Score：综合Precision和Recall，公式为： F1-Score = (2 * Precision * Recall) / (Precision + Recall)

# ## 三、 赛题数据解析

# 本次赛题的数据集来源多个渠道，使用高清摄像机、普通摄像机等多种设备拍摄。数据来自各类型输电线路设备，背景包括城市、乡村、山地、道路、农田等。 数据集包含1000张输电通道图像，大小约为3GB，由比赛选手自行拆分训练集和验证集。数据集中目标的种类包括鸟巢、风筝、气球、垃圾，图片为jpg格式，标注文件为xml格式。数据集中每个目标都被标注了专业准确的边界框和类别。 数据集的独特优势在于：数据来源广泛、数量庞大、种类清晰、边界框标注准确。这些特点使得该数据集适用于复杂场景下的隐患目标检测算法的开发和应用。同时，数据量足够大，可以有效提高算法的鲁棒性和泛化性能，从而达到更好的应用效果。 参赛者可以基于该数据集进行算法的训练和测试，并结合数据集的独特特点进行算法设计，以期达到更好的检测效果和性能。
# 
# 初赛数据集（训练数据集与测试数据集）下载链接：[链接](https://aistudio.baidu.com/aistudio/datasetdetail/212110)
# 
# | 数据集名称 | 输电通道的隐患目标数据集  |
# | -------- | -------- |
# | 数据来源 | 多种设备拍摄|
# | 数据量 | 1000张图像，大小约为3GB |
# | 数据内容 | 各类型输电线路设备，不同背景 |
# | 标注信息 | 准确的边界框和类别标注 |
# | 目标种类 | 鸟巢、风筝、气球、垃圾等 |
# | 优势 | 来源广泛、数量庞大、标注准确 |
# | 使用价值 | 开发隐患目标检测算法 |
# | 针对参赛者 | 提供基础数据，便于开发算法 |
# 
# 
# 部分数据可视化如下：
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/605988285e6b45179ba5298bd3787593043db8246c03421b81101f2f7a5583c9)

# **数据说明：**
# 
# 具体格式说明如下：
# 1. 文件名称说明： 数据集中的图像文件以“.jpg”为后缀，标注文件同名，只是后缀名为“.xml”。
# 2. 文件编码说明： 标注文件采用UTF-8编码。
# 3. 文件中的数据格式说明： 标注文件采用XML格式，格式如下：
# ```
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
# 
# 标注文件中各字段含义：
# 
# | \<filename> | 每张图像对应的文件名，格式为“ xxx.jpg” |
# | -------- | -------- |
# | \<width> | 目标图像的宽度，单位为像素 |
# | \<height> | 目标图像的高度，单位为像素 |
# | \<depth> | 目标图像的通道数，一般为3，分别表示红、绿、蓝三个通道 |
# | \<name> | 目标的类别名称，如“nest”、“balloon”、“trash”、“kite” |
# | \<xmin> | 目标边界框左上角的横坐标，单位为像素 |
# | \<ymin> | 目标边界框左上角的纵坐标，单位为像素 |
# | \<xmax> | 目标边界框右下角的横坐标，单位为像素 |
# | \<ymax> | 目标边界框右下角的纵坐标，单位为像素 |
#     
# 注意：在实际数据中，一个图像可能包含多个目标，因此一个标注文件中可能会有多个\<object>标签。

# ## 四、数据预处理

# ### 4.1 解压数据集

# In[2]:


# !unzip /home/aistudio/data/data212110/train.zip -d /home/aistudio/work/
# !unzip /home/aistudio/data/data212110/val.zip -d /home/aistudio/work/


# ### 4.2 数据集划分

# **Step01：** 首先分隔开.jpg文件和.xml文件。

# In[3]:


# %cd /home/aistudio/work/train/
# !mkdir JPEGImages
# !mkdir Annotations
# !mv /home/aistudio/work/train/*.jpg /home/aistudio/work/train/JPEGImages/
# !mv /home/aistudio/work/train/*.xml /home/aistudio/work/train/Annotations/


# **Step02：** 使用 PaddleX 划分数据集

# 首先安装 PaddleX 。

# In[2]:


get_ipython().system('pip install paddlex')


# 通过 split_dataset 这个 API 按照 0.8：0.2 的比例划分训练集和验证集。

# In[7]:


# !paddlex --split_dataset --format VOC --dataset_dir /home/aistudio/work/train --val_value 0.2


# **Step03（非必须）：** PaddleX 划分数据集会生成一个 labels.txt 文件，这时我们可以选择将 labels.txt 中的类别顺序按照官方给出的 categoryID.txt 中的顺序进行整理，这样后面模型预测出的类别标签就和官方给出的类别标签相对应。

# In[6]:


# !rm /home/aistudio/work/train/labels.txt
# !cp /home/aistudio/coco_config/labels.txt /home/aistudio/work/train/


# 处理后的结果如图所示：
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/06397cfbbe3749b48b3831f42bd3e1d32297c6ba47174dc58e8efb3e78a14ab0)

# ## 五、代码实现

# ### 5.1 安装PaddleDetection

# In[10]:


# # 克隆PaddleDetection仓库
# %cd /home/aistudio/
# # !git clone -b develop https://gitee.com/PaddlePaddle/PaddleDetection.git

# # 安装其他依赖
# %cd /home/aistudio/PaddleDetection/
# !pip install -r requirements.txt --user

# # 编译安装paddledet
# !python setup.py install


# ### 5.2 检测数据分析

# 该数据集总共包含 4 个标签，各类标签的数量分别为：
# * trash: 76
# * balloon: 88
# * nest: 540
# * kite: 103

# In[12]:


import os
from unicodedata import name
import xml.etree.ElementTree as ET
import glob

def count_num(indir):
    # 提取xml文件列表
    os.chdir(indir)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations) + '*.xml')

    dict = {} # 新建字典，用于存放各类标签名及其对应的数目
    for i, file in enumerate(annotations): # 遍历xml文件
       
        # actual parsing
        in_file = open(file, encoding = 'utf-8')
        tree = ET.parse(in_file)
        root = tree.getroot()

        # 遍历文件的所有标签
        for obj in root.iter('object'):
            name = obj.find('name').text
            if(name in dict.keys()): dict[name] += 1 # 如果标签不是第一次出现，则+1
            else: dict[name] = 1 # 如果标签是第一次出现，则将该标签名对应的value初始化为1

    # 打印结果
    print("各类标签的数量分别为：")
    for key in dict.keys(): 
        print(key + ': ' + str(dict[key]))            

indir='/home/aistudio/work/train/Annotations/'   # xml文件所在的目录
count_num(indir) # 调用函数统计各类标签数目


# **图像尺寸分析：** 通过图像尺寸分析，我们可以看到该数据集图片的尺寸不一。

# In[1]:


import os
from unicodedata import name
import xml.etree.ElementTree as ET
import glob

def Image_size(indir):
    # 提取xml文件列表
    os.chdir(indir)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations) + '*.xml')
    width_heights = []

    for i, file in enumerate(annotations): # 遍历xml文件
        # actual parsing
        in_file = open(file, encoding = 'utf-8')
        tree = ET.parse(in_file)
        root = tree.getroot()
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)
        if [width, height] not in width_heights: width_heights.append([width, height])
    print("数据集中，有{}种不同的尺寸，分别是：".format(len(width_heights)))
    for item in width_heights:
        print(item)

indir='/home/aistudio/work/train/Annotations/'   # xml文件所在的目录
Image_size(indir)


# ### 5.3 模型训练

# （非必须）：我们可以通过以下代码块将VOC格式数据集转换成COCO格式数据集。
# 
# 注意：由于标注文件中filename字段和xml格式标注文件本身的名字不相符，我们直接通过x2coco.py转换数据集格式，会发现在训练的时候会找不到数据集图片，原因就是因为转换后的标注文件误以为文件名是xml格式标注文件的filename字段。
# 
# 解决方法：可以直接修改x2coco.py的源码，修改后的文件存放在/home/aistudio/coco_config/路径下，大家可以自行查看！

# In[4]:


get_ipython().system('rm /home/aistudio/PaddleDetection/tools/x2coco.py')
get_ipython().system('cp /home/aistudio/coco_config/x2coco.py /home/aistudio/PaddleDetection/tools/')


# In[5]:


get_ipython().run_line_magic('cd', '/home/aistudio/PaddleDetection/')
get_ipython().system('
python tools/x2coco.py         
--dataset_type voc         
--voc_anno_dir /home/aistudio/work/train/         
--voc_anno_list /home/aistudio/work/train/train_list.txt         
--voc_label_list /home/aistudio/work/train/labels.txt         
--voc_out_name /home/aistudio/work/train/voc_train.json')
# In[6]:
get_ipython().system('
python tools/x2coco.py         
--dataset_type voc         
--voc_anno_dir /home/aistudio/work/train/         
--voc_anno_list /home/aistudio/work/train/val_list.txt         
--voc_label_list /home/aistudio/work/train/labels.txt         
--voc_out_name /home/aistudio/work/train/voc_val.json')


# In[7]:
# 替换配置文件
get_ipython().system('
rm -rf /home/aistudio/PaddleDetection/configs/rtdetr/_base_')
get_ipython().system('
rm /home/aistudio/PaddleDetection/configs/rtdetr/rtdetr_hgnetv2_x_6x_coco.yml')
get_ipython().system('
rm /home/aistudio/PaddleDetection/configs/runtime.yml')
get_ipython().system('
rm /home/aistudio/PaddleDetection/configs/datasets/coco_detection.yml')
get_ipython().system('
cp -r /home/aistudio/coco_config/_base_  /home/aistudio/PaddleDetection/configs/rtdetr/')
get_ipython().system('
cp /home/aistudio/coco_config/rtdetr_hgnetv2_x_6x_coco.yml /home/aistudio/PaddleDetection/configs/rtdetr/')
get_ipython().system('
cp /home/aistudio/coco_config/runtime.yml /home/aistudio/PaddleDetection/configs/')
get_ipython().system('
cp /home/aistudio/coco_config/coco_detection.yml /home/aistudio/PaddleDetection/configs/datasets/')


# In[8]:


# training on multi-GPU
get_ipython().run_line_magic('cd', '/home/aistudio/PaddleDetection/')
get_ipython().system('export CUDA_VISIBLE_DEVICES=0,1,2,3')
get_ipython().system('
python -m paddle.distributed.launch 
--gpus 0,1,2,3 tools/train.py 
-c configs/rtdetr/rtdetr_hgnetv2_x_6x_coco.yml 
--fleet --eval')


# In[22]:


# # 23.5.23 只有一个GPU
# %cd /home/aistudio/PaddleDetection/
# # !export CUDA_VISIBLE_DEVICES=0
# !python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_coco.yml --eval


# ### 5.4 模型评估

# In[23]:


get_ipython().system('python tools/eval.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_coco.yml -o weights=output/rtdetr_hgnetv2_x_6x_coco/best_model.pdparams')


# **指标如下：**
# * Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.748
# * Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.922
# * Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.819
# * Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
# * Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.627
# * Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.761
# * Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.772
# * Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.805
# * Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.842
# * Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
# * Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.750
# * Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.852
# * Total sample number: 160, average FPS: 11.625377601635861

# ### 5.5 模型导出

# In[ ]:


get_ipython().system('python tools/export_model.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_coco.yml --output_dir=./inference_model -o weights=output/rtdetr_hgnetv2_x_6x_coco/best_model')


# ### 5.6 结果文件生成

# baseline目前已经成功提交，如果在提交方面有问题的话可以参考下这部分的内容，最后祝愿大家在本次比赛中都取得好成绩！
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/cf055da106ed4123becf8e4037d8bf8c9fbed16ae6ab47b5abd2f1351b55d994)

# 最后提交结果的时候，我们需要生成一个coco数据集标注格式的文件，我们可以通过PaddleDetection中Python部署的demo来生成该文件，但需要注意以下几个问题。
# 1. 官方给出的categoryID.txt中类别ID是从1开始，而PaddleDetection的数据标注是从0开始，实际上就相当于bbox[0]+1;
# 2. 官方给出的val_imgID.txt对ID和file_name都有明确的限制，我们可以通过读入val_imgID.txt生成一个字典，通过键值对的映射关系来解决。
# 
# 修改后的infer.py文件存放在/home/aistudio/coco_config/目录下，大家可以自行查看！

# In[1]:


# !rm /home/aistudio/PaddleDetection/deploy/python/infer.py
# !cp /home/aistudio/coco_config/infer.py /home/aistudio/PaddleDetection/deploy/python/


# In[ ]:


get_ipython().system('python deploy/python/infer.py --model_dir=inference_model/rtdetr_hgnetv2_x_6x_coco --image_dir=/home/aistudio/data/val --device=GPU --output_dir infer_output --save_results')


# 最后json文件中的一条数据如下图所示：
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/34c763c7f2314a00b453f067399ddafbf46e3d74d02a41f1a67a4b063344cded)

# ## 六、总结与提高

# 以上是[第二届广州·琶洲算法大赛]基于复杂场景的输电通道隐患目标检测算法的baseline，大家将更多的精力花在改进模型提升模型的性能上。
# 1. 数据增强层面：该数据集的样本不是很多，所以大家需要做一些针对性的数据增强。同时训练的时候大家可能需要划分数据集，但是最后选定好模型提交的时候我们可以将所有的标注数据集中起来去训练。PaddleDetection目前支持的数据增强包括：
#     * Resize
#     * Lighting
#     * Flipping
#     * Expand
#     * Crop
#     * Color Distort
#     * Random Erasing
#     * Mixup
#     * AugmentHSV
#     * Mosaic
#     * Cutmix
#     * Grid Mask
#     * Auto Augment
#     * Random Perspective
# 2. 模型结构方面：本赛题是通过F1-Score去进行评分。那是否我们在改进的时候可以牺牲部分速度来换取精度呢？所以在这块我们其实可以去尝试更多的大模型。但是需要注意本比赛不可以用训练集和测试集做交叉训练、交叉测试（比如不可以对测试集打伪标签用于训练，或者半监督学习等），因此千万不能用semi_det中的模型。
