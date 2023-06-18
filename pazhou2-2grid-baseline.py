#!/usr/bin/env python
# coding: utf-8

# ## 一、 赛题背景。5.25作者更新版本。PaddleDetection官方最新模型rtd，跑分72分

# ## 三、 数据集分析

# ### 3.1 先把原始xml文件中文件名不匹配的问题解决，这是官方提供数据集的一个小问题

# #### 3.1.1解压，把img和annotation分装到对应文件夹

# In[1]:


# 23.6.9 准备后台运行，只运行一次
# !unzip -oq ~/data/data212110/val.zip -d ~/PaddleDetection/dataset/grid_coco
# !unzip -oq ~/data/data212110/train.zip -d ~/PaddleDetection/dataset/grid_coco
# val目录中300张照片，未标注，用来infer提交result打榜；train中800张照片，800个xml，用于训练

# %cd ~/PaddleDetection/dataset/grid_coco/train/
# !mkdir JPEGImages
# !mkdir Annotations
# !mv *.jpg JPEGImages/
# !mv *.xml Annotations/


# In[2]:


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

# In[93]:


# 23.6.7 查看数据集中图片与标准信息
import os
import random
import xml.etree.ElementTree as ET
from PIL import Image
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('cd', '')


# 图片和标注文件夹路径
image_folder = 'PaddleDetection/dataset/grid_coco/train/JPEGImages'
annotation_folder = 'PaddleDetection/dataset/grid_coco/train/Annotations'

# 获取图片文件列表
image_files = sorted(os.listdir(image_folder))
pic_id = 700


# In[177]:




# 随机选择一张图片
# image_file = random.choice(image_files)
image_file = image_files[pic_id]
pic_id+=1
# image_file = 'gD70hFn1TxmVd5HYOB4AirLNq3oSkyPCIscR8juw.jpg'

# 构建标注文件路径
annotation_file = os.path.join(annotation_folder, os.path.splitext(image_file)[0] + '.xml')

# 打开图片
image_path = os.path.join(image_folder, image_file)
image = Image.open(image_path)

# 获取图片宽度和高度
width, height = image.size

# 打开标注文件
tree = ET.parse(annotation_file)
root = tree.getroot()

# 解析标注文件，提取边界框信息
bboxes = []
for obj in root.iter('object'):
    # 获取边界框坐标
    bbox = obj.find('bndbox')
    xmin = int(bbox.find('xmin').text)
    ymin = int(bbox.find('ymin').text)
    xmax = int(bbox.find('xmax').text)
    ymax = int(bbox.find('ymax').text)
    
    # 将边界框坐标转换为左上角和右下角坐标形式
    bbox_coords = (xmin, ymin, xmax, ymax)
    bboxes.append(bbox_coords)

# 在图片上绘制边界框和坐标轴
fig, ax = plt.subplots()
ax.imshow(image)

for bbox in bboxes:
    xmin, ymin, xmax, ymax = bbox
    # 绘制边界框矩形
    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)

# 添加图片尺寸的坐标轴
ax.set_xlim(0, width)
ax.set_ylim(height, 0)
ax.set_aspect('equal')
ax.set_title('Image with Bounding Boxes')
ax.set_xlabel('Width')
ax.set_ylabel('Height')

plt.show()


# ### 3.3 分析train数据集category分布并做增广处理

# In[4]:


# # 6.9 标注信息分析
# import os
# import xml.etree.ElementTree as ET

# !rm -rf PaddleDetection/dataset/grid_coco/train/Annotations/.ipynb_checkpoints
# # 标注文件夹路径
# annotation_folder = 'PaddleDetection/dataset/grid_coco/train/Annotations'

# # 类别字典
# class_dict = {'nest': 1, 'kite': 2, 'balloon': 3, 'trash': 4}

# # 统计类型数量的字典
# class_count = {class_name: 0 for class_name in class_dict.keys()}

# # 遍历标注文件夹中的所有.xml文件
# xml_files = os.listdir(annotation_folder)

# # 用于观察一个图片多个标注框的情况
# multi_objs_files = []

# for xml_file in xml_files:
#     xml_path = os.path.join(annotation_folder, xml_file)
    
#     # 解析标注文件
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
    
#     obj_num = 0
#     # 遍历标注文件中的所有目标
#     for obj in root.iter('object'):
#         # 获取目标类别名称
#         class_name = obj.find('name').text
#         # 统计类型数量
#         if class_name in class_dict:
#             class_count[class_name] += 1

#         # 统计目标的bbox信息
#         obj_num+=1
#     # 完成obj遍历后，如果obj_num>1，记住这个文件
#     if obj_num > 1:
#         multi_objs_files.append(xml_file)            

# # 打印类型分布统计结果
# for class_name, count in class_count.items():
#     print(f'{class_name}: {count}')

# for file in multi_objs_files:
#     print(file)
# # nest: 540
# # kite: 103
# # balloon: 88
# # trash: 76 数据分布800个标注文件，807个标注框


# In[5]:


# # 23.6.7 由于各标注类别数目比例不平衡，对数量少的进行离线增广

# import os
# import random
# import shutil
# from xml.etree import ElementTree as ET

# %cd
# # 源图片文件夹路径
# image_dir = '/home/aistudio/PaddleDetection2.5/dataset/coco/train/JPEGImages'

# # 源标注文件夹路径
# annotation_dir = '/home/aistudio/PaddleDetection2.5/dataset/coco/train/Annotations'

# # 目标文件夹路径
# # output_dir = 'PaddleDetection2.5/dataset/coco/train/OversampledData'

# # 定义需要过采样的类别和对应的样本数量
# classes_to_oversample = {
#     'kite': 400,
#     'balloon': 400,
#     'trash': 400
# }

# # 创建目标文件夹
# # os.makedirs(output_dir, exist_ok=True)

# # 对需要过采样的类别进行样本复制
# for class_name, num_samples in classes_to_oversample.items():
#     # 获取该类别下的所有XML文件路径
#     xml_files = [file for file in os.listdir(annotation_dir) if file.endswith('.xml')]
    
#     # 获取该类别下的XML文件中包含的样本
#     class_files = []
#     for xml_file in xml_files:
#         xml_path = os.path.join(annotation_dir, xml_file)
#         tree = ET.parse(xml_path)
#         root = tree.getroot()
#         objects = root.findall('object')
#         for obj in objects:
#             obj_class = obj.find('name').text
#             if obj_class == class_name:
#                 image_file = root.find('filename').text
#                 class_files.append(image_file)
#                 break
    
#     # 计算需要复制的次数
#     num_copies = num_samples - len(class_files)
    
#     # 从该类别的样本中随机选择并复制到目标文件夹中，直到达到指定数量
#     for _ in range(num_copies):
#         # 随机选择源文件
#         source_file = random.choice(class_files)
        
#         # 获取源文件路径
#         source_image_path = os.path.join(image_dir, source_file)
#         source_annotation_path = os.path.join(annotation_dir, source_file.replace('.jpg', '.xml'))
        
#         # 生成新的文件名
#         target_name = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=len(source_file.split('.')[0])))
#         target_image_file = target_name + '.jpg'
#         target_annotation_file = target_name + '.xml'
        
#         # 目标文件路径
#         target_image_path = os.path.join(image_dir, target_image_file)
#         target_annotation_path = os.path.join(annotation_dir, target_annotation_file)
        
#         # 复制图片文件和对应的标注文件到目标文件夹
#         shutil.copy(source_image_path, target_image_path)
#         shutil.copy(source_annotation_path, target_annotation_path)


# ## 四、数据预处理

# ### 4.2 数据集划分

# **Step02：** 使用 PaddleX 划分数据集

# In[1]:


get_ipython().system('pip install paddlex')


# 通过 split_dataset 这个 API 按照 0.8：0.2 的比例划分训练集和验证集。

# In[8]:


# !paddlex --split_dataset --format VOC --dataset_dir /home/aistudio/PaddleDetection/dataset/grid_coco/train/ --val_value 0.1


# **Step03：** PaddleX 划分数据集会生成一个 labels.txt 文件，这时我们可以选择将 labels.txt 中的类别顺序按照官方给出的 categoryID.txt 中的顺序进行整理，这样后面模型预测出的类别标签就和官方给出的类别标签相对应。

# In[9]:


# !pwd
# %cd PaddleDetection/dataset/grid_coco/train


# In[10]:


# !rm labels.txt
# !cp /home/aistudio/coco_config/labels.txt .


# ## 五、代码实现

# ### 5.1 安装PaddleDetection

# In[11]:


# %cd /home/aistudio/
# !git clone -b develop https://gitee.com/PaddlePaddle/PaddleDetection.git
# !git clone https://gitee.com/PaddlePaddle/PaddleDetection.git
# !git clone -b develop  https://github.com/PaddlePaddle/PaddleDetection.git
# ! git clone https://hub.fastgit.xyz/PaddlePaddle/PaddleDetection.git
# 23.5.29 使用AIstudio套件安装，需要改名，但是train的时候出错，所以再次尝试git
# ! mv PaddleDetection-2.5.0 PaddleDetection


# In[2]:


# 克隆PaddleDetection仓库
# %cd /home/aistudio/
# !git clone -b develop https://gitee.com/PaddlePaddle/PaddleDetection.git

# 安装其他依赖
get_ipython().run_line_magic('cd', '/home/aistudio/PaddleDetection/')
get_ipython().system('pip install -r requirements.txt --user')

# 编译安装paddledet
get_ipython().system('python setup.py install')


# In[3]:


# 安装后确认测试通过：23.5.29
get_ipython().system('python ppdet/modeling/tests/test_architectures.py')


# ### 5.2 检测数据分析

# In[18]:


# import os
# from unicodedata import name
# import xml.etree.ElementTree as ET
# import glob

# def count_num(indir):
#     # 提取xml文件列表
#     os.chdir(indir)
#     annotations = os.listdir('.')
#     annotations = glob.glob(str(annotations) + '*.xml')

#     dict = {} # 新建字典，用于存放各类标签名及其对应的数目
#     for i, file in enumerate(annotations): # 遍历xml文件
       
#         # actual parsing
#         in_file = open(file, encoding = 'utf-8')
#         tree = ET.parse(in_file)
#         root = tree.getroot()

#         # 遍历文件的所有标签
#         for obj in root.iter('object'):
#             name = obj.find('name').text
#             if(name in dict.keys()): dict[name] += 1 # 如果标签不是第一次出现，则+1
#             else: dict[name] = 1 # 如果标签是第一次出现，则将该标签名对应的value初始化为1

#     # 打印结果
#     print("各类标签的数量分别为：")
#     for key in dict.keys(): 
#         print(key + ': ' + str(dict[key]))            

# indir='/home/aistudio/PaddleDetection2.5/dataset/coco/train/Annotations/'   # xml文件所在的目录
# count_num(indir) # 调用函数统计各类标签数目


# **图像尺寸分析：** 通过图像尺寸分析，我们可以看到该数据集图片的尺寸不一。
# **数据处理是关键**

# In[19]:


# import os
# from unicodedata import name
# import xml.etree.ElementTree as ET
# import glob

# def Image_size(indir):
#     # 提取xml文件列表
#     os.chdir(indir)
#     annotations = os.listdir('.')
#     annotations = glob.glob(str(annotations) + '*.xml')
#     width_heights = []

#     for i, file in enumerate(annotations): # 遍历xml文件
#         # actual parsing
#         in_file = open(file, encoding = 'utf-8')
#         tree = ET.parse(in_file)
#         root = tree.getroot()
#         width = int(root.find('size').find('width').text)
#         height = int(root.find('size').find('height').text)
#         if [width, height] not in width_heights: width_heights.append([width, height])
#     print("数据集中，有{}种不同的尺寸，分别是：".format(len(width_heights)))
#     for item in width_heights:
#         print(item)

# indir='/home/aistudio/PaddleDetection2.5/dataset/coco/train/Annotations/'   # xml文件所在的目录
# Image_size(indir)


# ### 5.3 模型训练

# （非必须）：我们可以通过以下代码块将VOC格式数据集转换成COCO格式数据集。
# 
# 注意：由于标注文件中filename字段和xml格式标注文件本身的名字不相符，我们直接通过x2coco.py转换数据集格式，会发现在训练的时候会找不到数据集图片，原因就是因为转换后的标注文件误以为文件名是xml格式标注文件的filename字段。
# 
# 解决方法：可以直接修改x2coco.py的源码，修改后的文件存放在/home/aistudio/coco_config/路径下，大家可以自行查看！

# In[20]:


# !rm /home/aistudio/PaddleDetection/tools/x2coco.py
# !cp /home/aistudio/coco_config/x2coco.py /home/aistudio/PaddleDetection/tools/
# # 23.5.29 已经copy，修改执行的话，直接在tools/x2coco.py中修改并立即生效


# In[21]:


# %cd /home/aistudio/PaddleDetection/
# !python tools/x2coco.py \
#         --dataset_type voc \
#         --voc_anno_dir /home/aistudio/PaddleDetection/dataset/grid_coco/train/ \
#         --voc_anno_list /home/aistudio/PaddleDetection/dataset/grid_coco/train/train_list.txt \
#         --voc_label_list /home/aistudio/PaddleDetection/dataset/grid_coco/train/labels.txt \
#         --voc_out_name /home/aistudio/PaddleDetection/dataset/grid_coco/train/voc_train.json


# In[22]:


# !python tools/x2coco.py \
#         --dataset_type voc \
#         --voc_anno_dir /home/aistudio/PaddleDetection/dataset/grid_coco/train/ \
#         --voc_anno_list /home/aistudio/PaddleDetection/dataset/grid_coco/train/val_list.txt \
#         --voc_label_list /home/aistudio/PaddleDetection/dataset/grid_coco/train/labels.txt \
#         --voc_out_name /home/aistudio/PaddleDetection/dataset/grid_coco/train/voc_val.json


# In[23]:


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


# In[24]:


# # 6.8 单卡继续训练，要修改yml学习率
# %cd /home/aistudio/PaddleDetection/
# # !python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_coco.yml --fleet --eval
# ! export CUDA_VISIBLE_DEVICES=0 #windows和Mac下不需要执行该命令
# ! python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_coco.yml -r output/rtdetr_hgnetv2_x_6x_coco/98 --eval


# In[3]:


# 6.15 使用ppyoloe训练试试
get_ipython().run_line_magic('cd', '~/PaddleDetection')
# !cp faster
# ! python tools/train.py -c configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml \
# -r output/ppyoloe_plus_crn_x_80e_coco/4 --eval --amp 
get_ipython().system('export CUDA_VISIBLE_DEVICES=0,1,2,3')
get_ipython().system('python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml -r output/ppyoloe_plus_crn_x_80e_coco/39 --fleet --eval')


# In[69]:


# 6.9进行小样本label_co_tuning尝试
get_ipython().run_line_magic('cd', '~/PaddleDetection')
# !cp faster
get_ipython().system(' python tools/train.py -c configs/few-shot/faster_rcnn_r50_vd_fpn_1x_coco_cotuning_grid.yml -r output/faster_rcnn_r50_vd_fpn_1x_coco_cotuning_grid/49 --eval')


# In[26]:


# # training on multi-GPU
# %cd /home/aistudio/PaddleDetection/
# !export CUDA_VISIBLE_DEVICES=0,1,2,3
# !python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_coco.yml --fleet --eval


# ### 5.4 模型评估

# In[4]:


# 6.16 使用ppyoloe训练结果评估
get_ipython().run_line_magic('cd', '~/PaddleDetection')
get_ipython().system('python tools/eval.py -c configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml -o weights=output/ppyoloe_plus_crn_x_80e_coco/best_model.pdparams')


# In[ ]:


get_ipython().run_line_magic('cd', '~/PaddleDetection')
get_ipython().system('python tools/eval.py -c configs/few-shot/faster_rcnn_r50_vd_fpn_1x_coco_cotuning_grid.yml  -o weights=output/faster_rcnn_r50_vd_fpn_1x_coco_cotuning_grid/best_model')


# In[28]:


# %cd ~/PaddleDetection
# !python tools/eval.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_coco.yml -o weights=output/rtdetr_hgnetv2_x_6x_coco/best_model.pdparams


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

# In[29]:



# ppyoloe 6.16
# DONE (t=0.11s).
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.815
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.995
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.889
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.900
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.815
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.836
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.865
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.866
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.900
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.864
# [06/16 09:42:49] ppdet.engine INFO: Total sample number: 80, average FPS: 12.129655432787684

# DONE (t=0.29s). lable_co_finetune 分数提交68左右 faster rcnn
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

# DONE (t=0.66s). 23.6.8 离线增强+在线增强，98epoch，划分数据集时，val有很多是从train复制的原图
# 所以ap值很高，但实际提交分数只有37，一般左右
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


# ### 5.5 不导出预测

# In[ ]:


# 6.16 不导出预测的图片连个框都没有，也不知道哪里出了问题，暂时先放下
get_ipython().system('export CUDA_VISIBLE_DEVICES=0')
get_ipython().system('python tools/infer.py -c configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml -o weights=output/ppyoloe_plus_crn_x_80e_coco/best_model.pdparams --infer_dir=/home/aistudio/PaddleDetection/dataset/grid_coco/test  --output_dir infer_output_grid --save_results True --draw_threshold 0.5')


# ### 5.6 模型导出

# In[7]:


# 6.16 使用ppyoloe导出模型
get_ipython().run_line_magic('cd', '~/PaddleDetection')
get_ipython().system('python tools/export_model.py -c configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml -o weights=output/ppyoloe_plus_crn_x_80e_coco/best_model')


# In[39]:


get_ipython().run_line_magic('cd', '~/PaddleDetection')
get_ipython().system('python tools/export_model.py -c configs/few-shot/faster_rcnn_r50_vd_fpn_1x_coco_cotuning_grid.yml --output_dir=./inference_model -o weights=output/faster_rcnn_r50_vd_fpn_1x_coco_cotuning_grid/best_model')


# In[30]:


# %cd ~/PaddleDetection
# !python tools/export_model.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_coco.yml --output_dir=./inference_model -o weights=output/rtdetr_hgnetv2_x_6x_coco/best_model


# ### 5.6 结果文件生成

# baseline目前已经成功提交，如果在提交方面有问题的话可以参考下这部分的内容，最后祝愿大家在本次比赛中都取得好成绩！
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/cf055da106ed4123becf8e4037d8bf8c9fbed16ae6ab47b5abd2f1351b55d994)

# 最后提交结果的时候，我们需要生成一个coco数据集标注格式的文件，我们可以通过PaddleDetection中Python部署的demo来生成该文件，但需要注意以下几个问题。
# 1. 官方给出的categoryID.txt中类别ID是从1开始，而PaddleDetection的数据标注是从0开始，实际上就相当于bbox[0]+1;
# 2. 官方给出的val_imgID.txt对ID和file_name都有明确的限制，我们可以通过读入val_imgID.txt生成一个字典，通过键值对的映射关系来解决。
# 
# 修改后的infer.py文件存放在/home/aistudio/coco_config/目录下，大家可以自行查看！

# In[31]:


# !rm /home/aistudio/PaddleDetection/deploy/python/infer.py
# !cp /home/aistudio/coco_config/infer.py /home/aistudio/PaddleDetection/deploy/python/


# In[4]:


# 6.16 ppyoloe+ 的推理
get_ipython().run_line_magic('cd', '/home/aistudio/PaddleDetection/')
get_ipython().system('python deploy/python/infer.py --model_dir=output_inference/ppyoloe_plus_crn_x_80e_coco --image_dir=/home/aistudio/PaddleDetection/dataset/grid_coco/test --device=GPU --output_dir infer_output_grid_depoly --save_results')


# In[40]:


get_ipython().run_line_magic('cd', '/home/aistudio/PaddleDetection/')
get_ipython().system('python deploy/python/infer.py --model_dir=inference_model/faster_rcnn_r50_vd_fpn_1x_coco_cotuning_grid --image_dir=/home/aistudio/PaddleDetection/dataset/grid_coco/test --device=GPU --output_dir infer_output --save_results')


# In[32]:


# %cd /home/aistudio/PaddleDetection/
# !python deploy/python/infer.py --model_dir=inference_model/rtdetr_hgnetv2_x_6x_coco \
# --image_dir=/home/aistudio/PaddleDetection/dataset/grid_coco/val \
# --device=GPU --output_dir infer_output --save_results


# In[2]:


get_ipython().run_line_magic('cd', '/home/aistudio/PaddleDetection/')
get_ipython().system("paddlex.det.coco_error_analysis(eval_details_file=None, gt=None, pred_bbox=None, pred_mask=None, save_dir='./infer_output/')")


# In[5]:


get_ipython().run_line_magic('cd', '')
# !pwd
# !python change_id.py
# ! cp /home/aistudio/PaddleDetection/infer_output/bbox.json ~/
get_ipython().system(' cp /home/aistudio/PaddleDetection/infer_output_grid_depoly/bbox.json ~/')


# In[34]:


# # 23.6.8 特意保留所有的bbox在all_bbox.json以便后处理分析只用，就不需要再次推理等待时间

# import json

# %cd ~/json

# # 读取原始数据
# with open('all_bbox.json', 'r') as file:
#     data = json.load(file)

# # 过滤掉"score"值小于0.3的字典，并重新排列id
# filtered_data = [d for d in data if d.get('score', 0) >= 0.3]

# # 重新排列id
# for i, d in enumerate(filtered_data):
#     d['id'] = i + 1

# # 保存结果到新文件
# with open('bbox_3.json', 'w') as file:
#     json.dump(filtered_data, file)

# print("Filtered data with re-arranged ids saved to bbox_3.json")


# ## 6 后处理，观察分析预测结果

# ### 6.1 推理文件夹中paddle生成图片的可视化

# In[5]:


# import os
# import json
# import random
# import matplotlib.pyplot as plt
# %matplotlib inline
# from PIL import Image

# 文件夹路径
# folder_path = '/home/aistudio/PaddleDetection/infer_output/' # lable_co 的模型
# INFER_IMAGES_PATH = '/home/aistudio/PaddleDetection/infer_output_grid_depoly/' # ppyoloe+ 的模型
# BBOX_JSON_PATH = '/home/aistudio/bbox.json' # 上面图像文件夹对应的json文件，可查看标注信息

# 获取文件夹中所有图片文件的路径
# image_files =        [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.jpg')]
# image_files = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.jpg')])

# 6.17 已组合进函数
# 6.16 测试集就这么一个取到字典是最方便
# with open('/home/aistudio/val_imgID.txt', 'r') as f:
#     image_data = json.load(f)
# image_dict = {image['id']: image['file_name'] for image in image_data}

# # 6.16 准备获取图片对应的检测框信息
# import json

# with open(BBOX_JSON_PATH, 'r') as f:
#     anno_data = json.load(f)

# pic_id = 20230000045

# 6.11 重点图片分析
# 1，pic_id:148, pic_name:jCBgPvSALlEWcVmRKFiIfaQGs1NM42wyeOpYXh6H.jpg,Image Size: 8688 x 5792
# 两个预测框重叠了，分别是88和90，这肯定被扣分，但是这样保存的逻辑是什么。ppyoloe+ 模型解决这个问题。

# 2,pic_id:1, pic_name:HZKRDyfxg4juOvF2kTC0cWLzibwpdqa61ne9It3V.jpg,Image Size: 4125 x 2987
# 一个nest有3个预测框，98,82,62的prob，这就是nms处理的问题，模型的预测效果还是可以的，改进nms呈现就能加分。同时要看看目前paddle的实现逻辑
# ppyoloe+ 模型解决了这个问题 6.16

# 3，pic_id:2, pic_name:HZxPEN9YTjueWpgy8b5FMtiROGQB4faUJXhcSkIl.jpg,Image Size: 2501 x 1988
# 同样的问题，改进好这个问题就能提分，模型和数据啥都不用变 ppyoloe+ 模型解决了这个问题 6.16

# 4,pic_id:6, pic_name:HxgqCP6MYuoNOEULy41nTWsfXJRQwlkhdp7tVB0Z.jpg,Image Size: 3074 x 2047
# 还是预测框重叠的问题 ppyoloe+ 模型解决了这个问题 6.16

# 5,pic_id:9, pic_name:I70pyxz2aQ5GitwnmklLPe6JcSBbCZfr3vKuVshT.jpg,Image Size: 8688 x 5792
# 这个大图的nest没有识别出来，模型是肯定有预测框的，但可能是p不够，给删掉了，这些都是需要看paddle的作图逻辑
# 模型解决了这个问题 6.16

# 6，pic_id:10, pic_name:I7TEB0tGfcNY491kJaZdM2hgnoqDrseVvmAUzCly.jpg,Image Size: 277 x 184
# 拿着风筝的小女孩。两个错误，一是把小女孩当做风筝的一部分，一是把衣服看成trash，65的prob当做trash，这个只能说是样本数据太少了
# 模型解决了这个问题 6.16

# 7，pic_id:43, pic_name:Ivzq1530tecfpDyXCRQxPTNLJhoamEsSW9nGUVud.jpg,Image Size: 8688 x 5792
# 这幅图把高压电塔柱上的接头板看成nest了，置信度高达97.31，这个就需要考虑模型问题了
# 模型解决了这个问题 6.16

# 8，pic_id:66, pic_name:JmHgaKjqMQfuA7r5hVv0ZIb469sF2tCw3oDcnklx.jpg,Image Size: 289 x 193
# 风筝问题，又是典型的预测框重叠
# 模型解决了这个问题 6.16

# 9，pic_id:108, pic_name:L12ifUrzO5EQt7cjn9ywPDgJNXeHv0AuIqM8Y3lF.jpg,Image Size: 3303 x 2274
# 预测框重叠 # 模型解决了这个问题 6.16

# 10，pic_id:123, pic_name:LUgwKYyd1cCbXqZ4pD0kFVHv9Bt3T6Oe2ArlPIsN.jpg,Image Size: 3123 x 2448
# 预测框重叠，小框大框重叠，小框prob高的情况
# 模型解决了这个问题 6.16 但是有树枝没有框住，要看看是怎么计算分值

# =============================6.16================================
# 6.16 结果分析列表
# pic_id:20230000124, pic_name:jO1T2mlhpSYysMkiqZ3tNo68PrxRWIvDa0Xf5KgA,Image Size: 4149 x 2890
# 未识别出来，要去看看thresh总体情况

# pic_id:20230000052, pic_name:IMHyhtAizuO1v72VJbRcds9XCpxPSk63EYgUZfLo,Image Size: 600 x 400
# 垃圾没有认出来，要去看看thresh总体情况，有可能标出来了，但是conf比较低导致未显示


# In[98]:


# # 6.16 如需通过filename显示，先找到pic_id，这样如下的代码逻辑进行统一，不需要修改
# target_filename = 'LUgwKYyd1cCbXqZ4pD0kFVHv9Bt3T6Oe2ArlPIsN'
# pic_id = 0
# for key, value in image_dict.items():
#     if value == target_filename:
#         pic_id = key
#         break
# pic_id


# In[14]:


analysis_pic_list = [20230000052,20230000056,20230000107,20230000124,20230000158,20230000230,20230000270]
# analysis_pic_list_two = [20230000029,
# 20230000110,
# 20230000120,
# 20230000121,
# 20230000123,
# 20230000144,
# 20230000156,
# 20230000183,
# 20230000268
# ]
idx = 0


# In[18]:


idx+=1


# In[19]:


# 6.16 封装统一函数后，维护方便非常多
pic_id=analysis_pic_list[idx]
# idx+=1
INFER_IMAGES_PATH = '/home/aistudio/PaddleDetection/infer_output_grid_depoly/' # ppyoloe+ 的模型
test_images_folder = '/home/aistudio/PaddleDetection/dataset/grid_coco/test/'
BBOX_JSON_PATH = '/home/aistudio/bbox.json' # 上面图像文件夹对应的json文件，可查看标注信息
id_name_file = '/home/aistudio/val_imgID.txt'
drawed_path = '/home/aistudio/work/pic_analysis'

draw_image_anno_from_json(pic_id, id_name_file, INFER_IMAGES_PATH, BBOX_JSON_PATH, drawed_path)


# ### 6.2 result.json文件的图片可视化

# In[21]:


INFER_IMAGES_PATH = '/home/aistudio/PaddleDetection/infer_output_grid_depoly/' # ppyoloe+ 的模型
test_images_folder = '/home/aistudio/PaddleDetection/dataset/grid_coco/test/'
BBOX_JSON_PATH = '/home/aistudio/0.2bbox.json' # 上面图像文件夹对应的json文件，可查看标注信息
id_name_file = '/home/aistudio/val_imgID.txt'
drawed_path = '/home/aistudio/work/pic_analysis'

draw_image_anno_from_json(pic_id, id_name_file, test_images_folder, BBOX_JSON_PATH, drawed_path)


# In[1]:


# 【重要功能函数】draw_image_anno_from_json
# 可视化infer的图像，CV的可视化函数要封装起来，便于维护与迭代
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 23.6.16 根据id和infer的json框作图，id_name_file提供文件名映射,imgs_path提供图片目录路径
# 23.6.16 修改为封装函数的好处还是很大的，要坚持封装
# 23.7.18 分析图片时都进行单独cp出来分析，将结果保存到save_save_path，逻辑区分换为是不是dataset目录
def draw_image_anno_from_json(image_id, id_name_file, imgs_path, bbox_json_file, drawed_save_path):
    # 6.16 测试集就这么一个取到字典是最方便
    with open(id_name_file, 'r') as f:
        id_name_json_data = json.load(f)
    # ID和文件名对应dict    {202300001：dajkjdlerDFDFS}
    id_name_dict = {image['id']: image['file_name'] for image in id_name_json_data}
    # 获取指定图像ID对应的文件名
    file_name = id_name_dict[image_id]
    print(f'image_id:{image_id},pic_name:{file_name}')

    # 6.16 准备获取图片对应的检测框信息，pazhou02项目的json文件只有annotation信息
    with open(bbox_json_file, 'r') as f:
        anno_data = json.load(f)
    # 获取指定图像ID对应的所有边界框
    bboxes = [bbox for bbox in anno_data if bbox['image_id'] == image_id]

    # 加载并显示图像
    image_path = os.path.join(imgs_path, file_name + '.jpg')
    image = plt.imread(image_path)
    plt.imshow(image)

    # 绘制每个边界框
    for bbox in bboxes:
        print(f'bbox:{bbox}')
        # 23.6.18 如果图片来自原图dataset，则需要自行绘制；否则是infer时自动生成的
        if 'dataset' in imgs_path:
            bbox_values = bbox['bbox']
            x, y, w, h = bbox_values[0], bbox_values[1], bbox_values[2], bbox_values[3]
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)

    # 获取图片的宽度和高度
    # w, h = image.size
    # 6.16 注意这里必须在plt.show()之前调用
    # 6.18 自行生成bbox框的，需要savefig
    if 'dataset' in imgs_path:
        plt.savefig(os.path.join(drawed_save_path, file_name + '.jpg'))
    else:
        # 6.18 结合上一句代码，如果是依据Bbox自行绘制rect，则要保存到文件夹分析
        # 当不需要plt.savefig()时，也把文件cp到pic_analysis进行分析
        # 6.11 拷贝到/work/pic_analysis进行分析
        import shutil
        src_file_path = image_path
        save_file_path = os.path.join(drawed_save_path, file_name + '.jpg')
        shutil.copy(src_file_path, save_file_path)        
    # 显示图片和坐标
    plt.show()


# In[27]:


# 【重要功能函数】adjust_jsons 用于json文件的手动修改
# 6.16 合并json文件，用于手动增加未识别的图片
# 6.18 把类似逻辑的功能函数进行封装，便于维护merge_jsons->adjust，变为可增可删
import json

def adjust_jsons(bbox_file, out_file, txt_file='', list_for_delete=None):
    if '' != txt_file and list_for_delete != None:
        # 6.18 避免添加的id导致可能的di错乱，增、删分开操作
        print('to avoid id error, one time one op')
        return

    # 读取bbox.json文件内容
    with open(bbox_file, 'r') as f:
        bbox_data = json.load(f)

    # 读取manual.txt文件内容
    manual_data = []
    if '' != txt_file:
        with open(txt_file, 'r') as f:
            manual_data = [json.loads(line.replace("'", '"')) for line in f]

    # 处理增加未找到的图片的bbox，合并两个列表
    merged_data = bbox_data + manual_data
    
    # 6.18 删除重复的anno_id
    if list_for_delete != []:
        # 根据id删除字典
        merged_data = [item for item in merged_data if item['id'] not in list_for_delete]

    # 6.16 重新整理id
    for i, data in enumerate(merged_data):
        data['id'] = i + 1
    # 将合并后的数据写入新的文件
    with open(out_file, 'w') as f:
        json.dump(merged_data, f)
    print('File adjusted.')


# In[28]:


bbox_file = '/home/aistudio/merged_bbox.json'
txt_file = '/home/aistudio/manual_add.txt'
out_file = '/home/aistudio/adjusted_bbox.json'

list_for_delete = [12,179,222,178,
30,
113
]
# adjust_jsons(bbox_file, out_file, txt_file) # 6.17 增
adjust_jsons(bbox_file, out_file, '', list_for_delete) # 6.18 删


# ### 6.3 手动增删提分试验

# 新建文本文件进行记录，比cell方便

# In[88]:





# In[1]:


# 6.16 bbox.json文件分析，
# id	        file_name	                                bbox_num
# 20230000001	HuXQEyIAeRq70Z6F4gDTOwh9zPnkBmaoCiNb2f8l	1
# 20230000002	HvWgFtQ7mSlTr0uKYw2MGIfApbjZcizRN5sdkEX6	1
# 20230000003	HWGTxO7U09BdAqroS3i4JPNCwpFb2MkZunXtjLVa	2
# 20230000004	HxgqCP6MYuoNOEULy41nTWsfXJRQwlkhdp7tVB0Z	2



import json
import csv

def count_bbox_dicts(json_file, txt_file, output_file):
    with open(json_file, 'r') as f:
        bbox_data = json.load(f)

    with open(txt_file, 'r') as f:
        val_data = json.load(f)

    id_dict = {item['id']: item for item in val_data}

    bbox_counts = {}
    for bbox_dict in bbox_data:
        image_id = bbox_dict['image_id']
        if image_id in id_dict:
            id_val = id_dict[image_id]
            bbox_counts[image_id] = bbox_counts.get(image_id, 0) + 1
            id_val['bbox_num'] = bbox_counts[image_id]

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['id', 'file_name', 'bbox_num']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in val_data:
            bbox_num = item.get('bbox_num', 0)
            writer.writerow({'id': item['id'], 'file_name': item['file_name'], 'bbox_num': bbox_num})


# 示例用法
json_file = '/home/aistudio/merged_bbox.json'
txt_file = '/home/aistudio/val_imgID.txt'
output_file = '/home/aistudio/merged_82b_num.csv'
count_bbox_dicts(json_file, txt_file, output_file)


# ## 七、总结与提高

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
