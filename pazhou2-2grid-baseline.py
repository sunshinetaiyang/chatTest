#!/usr/bin/env python
# coding: utf-8

# ## 一、 赛题背景。5.25作者更新版本。PaddleDetection官方最新模型rtd，跑分72分

# ## 二、 赛题介绍

# ## 三、 数据集分析

# ### 3.1 先把原始xml文件中文件名不匹配的问题解决，这是官方提供数据集的一个小问题

# In[3]:


# 23.6.9 准备后台运行，只运行一次
# !unzip -oq ~/data/data212110/val.zip -d ~/PaddleDetection/dataset/grid_coco
# !unzip -oq ~/data/data212110/train.zip -d ~/PaddleDetection/dataset/grid_coco
# val目录中300张照片，未标注，用来infer提交result打榜；train中800张照片，800个xml，用于训练

# %cd ~/PaddleDetection/dataset/grid_coco/train/
# !mkdir JPEGImages
# !mkdir Annotations
# !mv *.jpg JPEGImages/
# !mv *.xml Annotations/


# In[4]:


# # 解决原数据filename问题
# import os
# from xml.etree import ElementTree as ET

# # 把所有PaddleDetection2.5/dataset/coco 换成新目录 PaddleDetection/dataset/grid_coco
# # 标注文件夹路径
# annotation_dir = '/home/aistudio/PaddleDetection/dataset/grid_coco/train/Annotations'

# # 遍历标注文件夹，修改文件名
# for filename in os.listdir(annotation_dir):
#     xml_path = os.path.join(annotation_dir, filename)
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
#     annotation_filename = os.path.splitext(filename)[0] + '.jpg'
#     root.find('filename').text = annotation_filename

#     # 保存修改后的标注文件
#     tree.write(xml_path)


# ### 3.2 查看train数据集图片和标注框情况

# In[90]:


# # 23.6.7 查看数据集中图片与标准信息
# import os
# import random
# import xml.etree.ElementTree as ET
# from PIL import Image
# import matplotlib.pyplot as plt
# %matplotlib inline
# %cd


# # 图片和标注文件夹路径
# image_folder = 'PaddleDetection/dataset/grid_coco/train/JPEGImages'
# annotation_folder = 'PaddleDetection/dataset/grid_coco/train/Annotations'

# # 获取图片文件列表
# image_files = os.listdir(image_folder)

# # 随机选择一张图片
# # image_file = random.choice(image_files)
# image_file = 'gD70hFn1TxmVd5HYOB4AirLNq3oSkyPCIscR8juw.jpg'

# # 构建标注文件路径
# annotation_file = os.path.join(annotation_folder, os.path.splitext(image_file)[0] + '.xml')

# # 打开图片
# image_path = os.path.join(image_folder, image_file)
# image = Image.open(image_path)

# # 获取图片宽度和高度
# width, height = image.size

# # 打开标注文件
# tree = ET.parse(annotation_file)
# root = tree.getroot()

# # 解析标注文件，提取边界框信息
# bboxes = []
# for obj in root.iter('object'):
#     # 获取边界框坐标
#     bbox = obj.find('bndbox')
#     xmin = int(bbox.find('xmin').text)
#     ymin = int(bbox.find('ymin').text)
#     xmax = int(bbox.find('xmax').text)
#     ymax = int(bbox.find('ymax').text)
    
#     # 将边界框坐标转换为左上角和右下角坐标形式
#     bbox_coords = (xmin, ymin, xmax, ymax)
#     bboxes.append(bbox_coords)

# # 在图片上绘制边界框和坐标轴
# fig, ax = plt.subplots()
# ax.imshow(image)

# for bbox in bboxes:
#     xmin, ymin, xmax, ymax = bbox
#     # 绘制边界框矩形
#     rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='red', linewidth=2)
#     ax.add_patch(rect)

# # 添加图片尺寸的坐标轴
# ax.set_xlim(0, width)
# ax.set_ylim(height, 0)
# ax.set_aspect('equal')
# ax.set_title('Image with Bounding Boxes')
# ax.set_xlabel('Width')
# ax.set_ylabel('Height')

# plt.show()


# ### 3.3 分析train数据集category分布并做增广处理

# In[85]:


# 6.9 标注信息分析
import os
import xml.etree.ElementTree as ET

get_ipython().system('rm -rf PaddleDetection/dataset/grid_coco/train/Annotations/.ipynb_checkpoints')
# 标注文件夹路径
annotation_folder = 'PaddleDetection/dataset/grid_coco/train/Annotations'

# 类别字典
class_dict = {'nest': 1, 'kite': 2, 'balloon': 3, 'trash': 4}

# 统计类型数量的字典
class_count = {class_name: 0 for class_name in class_dict.keys()}

# 遍历标注文件夹中的所有.xml文件
xml_files = os.listdir(annotation_folder)

# 用于观察一个图片多个标注框的情况
multi_objs_files = []

for xml_file in xml_files:
    xml_path = os.path.join(annotation_folder, xml_file)
    
    # 解析标注文件
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    obj_num = 0
    # 遍历标注文件中的所有目标
    for obj in root.iter('object'):
        # 获取目标类别名称
        class_name = obj.find('name').text
        # 统计类型数量
        if class_name in class_dict:
            class_count[class_name] += 1

        # 统计目标的bbox信息
        obj_num+=1
    # 完成obj遍历后，如果obj_num>1，记住这个文件
    if obj_num > 1:
        multi_objs_files.append(xml_file)            

# 打印类型分布统计结果
for class_name, count in class_count.items():
    print(f'{class_name}: {count}')

for file in multi_objs_files:
    print(file)
# nest: 540
# kite: 103
# balloon: 88
# trash: 76 数据分布800个标注文件，807个标注框


# In[44]:


# 23.6.7 由于各标注类别数目比例不平衡，对数量少的进行离线增广

import os
import random
import shutil
from xml.etree import ElementTree as ET

get_ipython().run_line_magic('cd', '')
# 源图片文件夹路径
image_dir = '/home/aistudio/PaddleDetection2.5/dataset/coco/train/JPEGImages'

# 源标注文件夹路径
annotation_dir = '/home/aistudio/PaddleDetection2.5/dataset/coco/train/Annotations'

# 目标文件夹路径
# output_dir = 'PaddleDetection2.5/dataset/coco/train/OversampledData'

# 定义需要过采样的类别和对应的样本数量
classes_to_oversample = {
    'kite': 400,
    'balloon': 400,
    'trash': 400
}

# 创建目标文件夹
# os.makedirs(output_dir, exist_ok=True)

# 对需要过采样的类别进行样本复制
for class_name, num_samples in classes_to_oversample.items():
    # 获取该类别下的所有XML文件路径
    xml_files = [file for file in os.listdir(annotation_dir) if file.endswith('.xml')]
    
    # 获取该类别下的XML文件中包含的样本
    class_files = []
    for xml_file in xml_files:
        xml_path = os.path.join(annotation_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objects = root.findall('object')
        for obj in objects:
            obj_class = obj.find('name').text
            if obj_class == class_name:
                image_file = root.find('filename').text
                class_files.append(image_file)
                break
    
    # 计算需要复制的次数
    num_copies = num_samples - len(class_files)
    
    # 从该类别的样本中随机选择并复制到目标文件夹中，直到达到指定数量
    for _ in range(num_copies):
        # 随机选择源文件
        source_file = random.choice(class_files)
        
        # 获取源文件路径
        source_image_path = os.path.join(image_dir, source_file)
        source_annotation_path = os.path.join(annotation_dir, source_file.replace('.jpg', '.xml'))
        
        # 生成新的文件名
        target_name = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=len(source_file.split('.')[0])))
        target_image_file = target_name + '.jpg'
        target_annotation_file = target_name + '.xml'
        
        # 目标文件路径
        target_image_path = os.path.join(image_dir, target_image_file)
        target_annotation_path = os.path.join(annotation_dir, target_annotation_file)
        
        # 复制图片文件和对应的标注文件到目标文件夹
        shutil.copy(source_image_path, target_image_path)
        shutil.copy(source_annotation_path, target_annotation_path)


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

# In[6]:


# !unzip -oq /home/aistudio/data/data212110/train.zip -d /home/aistudio/PaddleDetection/dataset/coco
# !unzip -oq /home/aistudio/data/data212110/val.zip -d /home/aistudio/PaddleDetection/dataset/coco


# ### 4.2 数据集划分

# **Step01：** 首先分隔开.jpg文件和.xml文件。

# In[8]:


# %cd /home/aistudio/PaddleDetection/dataset/coco/train/
# !mkdir JPEGImages
# !mkdir Annotations
# !mv ./*.jpg ./JPEGImages/
# !mv ./*.xml ./Annotations/


# **Step02：** 使用 PaddleX 划分数据集

# 首先安装 PaddleX 。

# In[1]:


get_ipython().system('pip install paddlex')


# 通过 split_dataset 这个 API 按照 0.8：0.2 的比例划分训练集和验证集。

# In[6]:


get_ipython().system('paddlex --split_dataset --format VOC --dataset_dir /home/aistudio/PaddleDetection2.5/dataset/coco/train/ --val_value 0.2')


# **Step03（非必须）：** PaddleX 划分数据集会生成一个 labels.txt 文件，这时我们可以选择将 labels.txt 中的类别顺序按照官方给出的 categoryID.txt 中的顺序进行整理，这样后面模型预测出的类别标签就和官方给出的类别标签相对应。

# In[11]:


# !pwd


# In[12]:


# !rm labels.txt
# !cp /home/aistudio/coco_config/labels.txt .


# 处理后的结果如图所示：
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/06397cfbbe3749b48b3831f42bd3e1d32297c6ba47174dc58e8efb3e78a14ab0)

# ## 五、代码实现

# ### 5.1 安装PaddleDetection

# In[13]:


# %cd /home/aistudio/
# !git clone -b develop https://gitee.com/PaddlePaddle/PaddleDetection.git
# !git clone https://gitee.com/PaddlePaddle/PaddleDetection.git
# !git clone -b develop  https://github.com/PaddlePaddle/PaddleDetection.git
# ! git clone https://hub.fastgit.xyz/PaddlePaddle/PaddleDetection.git
# 23.5.29 使用AIstudio套件安装，需要改名，但是train的时候出错，所以再次尝试git
# ! mv PaddleDetection-2.5.0 PaddleDetection


# In[3]:


# 克隆PaddleDetection仓库
# %cd /home/aistudio/
# !git clone -b develop https://gitee.com/PaddlePaddle/PaddleDetection.git

# 安装其他依赖
get_ipython().run_line_magic('cd', '/home/aistudio/PaddleDetection/')
get_ipython().system('pip install -r requirements.txt --user')

# 编译安装paddledet
get_ipython().system('python setup.py install')


# In[4]:


# 安装后确认测试通过：23.5.29
get_ipython().system('python ppdet/modeling/tests/test_architectures.py')


# In[16]:


# # 在GPU上预测一张图片
# !export CUDA_VISIBLE_DEVICES=0
# !python tools/infer.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -o use_gpu=true weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams --infer_img=/home/aistudio/PaddleDetection/dataset/coco/train/JPEGImages/dE9X2lfzPkYvj4UNrZWDOGycw8LAeQoiSxmuJgRB.jpg


# In[17]:


# 23.5.29 试验GitHub的quick start代码
# 边训练边测试 CPU需要约1小时(use_gpu=false)，1080Ti GPU需要约10分钟
# -c 参数表示指定使用哪个配置文件
# -o 参数表示指定配置文件中的全局变量（覆盖配置文件中的设置），这里设置使用gpu
# --eval 参数表示边训练边评估，最后会自动保存一个名为model_final.pdparams的模型

# !python tools/train.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml --eval -o use_gpu=true


# In[18]:


# 评估 默认使用训练过程中保存的model_final.pdparams
# -c 参数表示指定使用哪个配置文件
# -o 参数表示指定配置文件中的全局变量（覆盖配置文件中的设置）
# 目前只支持单卡评估

# !python tools/eval.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml -o use_gpu=true


# In[19]:


# -c 参数表示指定使用哪个配置文件
# -o 参数表示指定配置文件中的全局变量（覆盖配置文件中的设置）
# --infer_img 参数指定预测图像路径
# 预测结束后会在output文件夹中生成一张画有预测结果的同名图像
# !pwd
# !python tools/infer.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml -o use_gpu=true --infer_img=demo/road572.png


# ### 5.2 检测数据分析

# 该数据集总共包含 4 个标签，各类标签的数量分别为：
# * trash: 76
# * balloon: 88
# * nest: 540
# * kite: 103

# In[2]:


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

indir='/home/aistudio/PaddleDetection2.5/dataset/coco/train/Annotations/'   # xml文件所在的目录
count_num(indir) # 调用函数统计各类标签数目


# **图像尺寸分析：** 通过图像尺寸分析，我们可以看到该数据集图片的尺寸不一。
# **数据处理是关键**

# In[ ]:


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

indir='/home/aistudio/PaddleDetection2.5/dataset/coco/train/Annotations/'   # xml文件所在的目录
Image_size(indir)


# ### 5.3 模型训练

# （非必须）：我们可以通过以下代码块将VOC格式数据集转换成COCO格式数据集。
# 
# 注意：由于标注文件中filename字段和xml格式标注文件本身的名字不相符，我们直接通过x2coco.py转换数据集格式，会发现在训练的时候会找不到数据集图片，原因就是因为转换后的标注文件误以为文件名是xml格式标注文件的filename字段。
# 
# 解决方法：可以直接修改x2coco.py的源码，修改后的文件存放在/home/aistudio/coco_config/路径下，大家可以自行查看！

# In[22]:


# !rm /home/aistudio/PaddleDetection/tools/x2coco.py
# !cp /home/aistudio/coco_config/x2coco.py /home/aistudio/PaddleDetection/tools/
# # 23.5.29 已经copy，修改执行的话，直接在tools/x2coco.py中修改并立即生效


# In[7]:


get_ipython().run_line_magic('cd', '/home/aistudio/PaddleDetection/')
get_ipython().system('python tools/x2coco.py         --dataset_type voc         --voc_anno_dir /home/aistudio/PaddleDetection2.5/dataset/coco/train/         --voc_anno_list /home/aistudio/PaddleDetection2.5/dataset/coco/train/train_list.txt         --voc_label_list /home/aistudio/PaddleDetection2.5/dataset/coco/train/labels.txt         --voc_out_name /home/aistudio/PaddleDetection2.5/dataset/coco/train/voc_train.json')


# In[8]:


get_ipython().system('python tools/x2coco.py         --dataset_type voc         --voc_anno_dir /home/aistudio/PaddleDetection2.5/dataset/coco/train/         --voc_anno_list /home/aistudio/PaddleDetection2.5/dataset/coco/train/val_list.txt         --voc_label_list /home/aistudio/PaddleDetection2.5/dataset/coco/train/labels.txt         --voc_out_name /home/aistudio/PaddleDetection2.5/dataset/coco/train/voc_val.json')


# In[25]:


# # 替换配置文件
# !rm -rf /home/aistudio/PaddleDetection/configs/rtdetr/_base_
# !rm /home/aistudio/PaddleDetection/configs/rtdetr/rtdetr_hgnetv2_x_6x_coco.yml
# !rm /home/aistudio/PaddleDetection/configs/runtime.yml
# !rm /home/aistudio/PaddleDetection/configs/datasets/coco_detection.yml
# !cp -r /home/aistudio/coco_config/_base_  /home/aistudio/PaddleDetection/configs/rtdetr/
# !cp /home/aistudio/coco_config/rtdetr_hgnetv2_x_6x_coco.yml /home/aistudio/PaddleDetection/configs/rtdetr/
# !cp /home/aistudio/coco_config/runtime.yml /home/aistudio/PaddleDetection/configs/
# !cp /home/aistudio/coco_config/coco_detection.yml /home/aistudio/PaddleDetection/configs/datasets/
# # 23.5.29 不替换，直接看GitHub进行二次开发
# 23.5.30 git可以下载最新了，xml拷贝一次即可


# In[ ]:


# 6.8 单卡继续训练，要修改yml学习率
get_ipython().run_line_magic('cd', '/home/aistudio/PaddleDetection/')
# !python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_coco.yml --fleet --eval
get_ipython().system(' export CUDA_VISIBLE_DEVICES=0 #windows和Mac下不需要执行该命令')
get_ipython().system(' python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_coco.yml -r output/rtdetr_hgnetv2_x_6x_coco/98 --eval')


# In[9]:


# training on multi-GPU
get_ipython().run_line_magic('cd', '/home/aistudio/PaddleDetection/')
get_ipython().system('export CUDA_VISIBLE_DEVICES=0,1,2,3')
get_ipython().system('python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_coco.yml --fleet --eval')


# ### 5.4 模型评估

# In[4]:


get_ipython().run_line_magic('cd', 'PaddleDetection')
get_ipython().system('python tools/eval.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_coco.yml -o weights=output/rtdetr_hgnetv2_x_6x_coco/98.pdparams')


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
# 
# 

# In[28]:


# DONE (t=0.29s).
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.726
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.894
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.811
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.513
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.774
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.756
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.779
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.809
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.571
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.860
# [05/30 14:33:05] ppdet.engine INFO: Total sample number: 160, average FPS: 16.74370932608211

# DONE (t=0.66s). 23.6.8 离线增强+在线增强，98epoch
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.875
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.998
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.918
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.886
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.878
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.881
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.906
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.931
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.951
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.933
# [06/08 18:53:55] ppdet.engine INFO: Total sample number: 347, average FPS: 18.17841402004979


# ### 5.5 模型导出

# In[15]:


get_ipython().run_line_magic('cd', '~/PaddleDetection')
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

# In[30]:


# !rm /home/aistudio/PaddleDetection/deploy/python/infer.py
# !cp /home/aistudio/coco_config/infer.py /home/aistudio/PaddleDetection/deploy/python/


# In[16]:


get_ipython().run_line_magic('cd', '/home/aistudio/PaddleDetection/')
get_ipython().system('python deploy/python/infer.py --model_dir=inference_model/rtdetr_hgnetv2_x_6x_coco --image_dir=/home/aistudio/PaddleDetection2.5/dataset/coco/val --device=GPU --output_dir infer_output --save_results')


# In[17]:


# %cd
# !pwd
# !python change_id.py
get_ipython().system(' cp /home/aistudio/PaddleDetection/infer_output/bbox.json ~/')


# In[10]:


# 23.6.8 特意保留所有的bbox在all_bbox.json以便后处理分析只用，就不需要再次推理等待时间

import json

get_ipython().run_line_magic('cd', '~/json')

# 读取原始数据
with open('all_bbox.json', 'r') as file:
    data = json.load(file)

# 过滤掉"score"值小于0.3的字典，并重新排列id
filtered_data = [d for d in data if d.get('score', 0) >= 0.3]

# 重新排列id
for i, d in enumerate(filtered_data):
    d['id'] = i + 1

# 保存结果到新文件
with open('bbox_3.json', 'w') as file:
    json.dump(filtered_data, file)

print("Filtered data with re-arranged ids saved to bbox_3.json")


# In[70]:


import os
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image

# 文件夹路径
folder_path = '/home/aistudio/PaddleDetection/infer_output/'

# 获取文件夹中所有图片文件的路径
image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.jpg')]

# 随机选择一张图片
random_image_path = random.choice(image_files)

# 加载图片
image = Image.open(random_image_path)

# 获取图片的宽度和高度
w, h = image.size

# 显示图片和坐标
plt.imshow(image)
plt.title(f'Image Size: {w} x {h}')
plt.axis('off')
plt.show()


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
