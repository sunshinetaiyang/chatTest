#!/usr/bin/env python
# coding: utf-8

# # 基于VOC Dataset的目标检测实验
# <center><img src='https://ai-studio-static-online.cdn.bcebos.com/848ac04bf5ba41b49b7713968b61a57f270583eb93344136a834e09a26982302' width=900></center>
#   
# # 1. 实验介绍
# ## 1.1 实验目的
# 1. 理解并掌握目标检测任务中的基础知识点，包括：[边界框](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/object_detection/Bounding_Box_Anchor.html)、[锚框](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/object_detection/Bounding_Box_Anchor.html#anchor-box)、[交并比](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/object_detection/IOU.html)、[非极大值抑制](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/object_detection/NMS.html)、[mAP](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/metrics/mAP.html)；
# 1. 掌握YOLOv3目标检测算法的设计原理以及构建流程。

# ## 1.2 实验内容
# 目标检测是计算机视觉领域的核心任务之一，其主要目的是让计算机可以自动识别图片中所有目标的类别，并在该目标周围绘制边界框，标示出每个目标的位置。
# 
# 深度学习中的卷积网络可以根据输入的图像，自动学习包含丰富语义信息的特征，得到更为全面的图像特征描述，更精确地完成目标检测任务。如今目标检测的应用场景非常广泛。在生活中，有大家非常熟悉的商品、车辆、行人检测，在真正的工业生产中也有非常多的应用场景：比如说常用的手机零件质量检测、厂房安全检测，以及在遥感、医疗等都有广泛的应用。在今年疫情期间，目标检测还可以进行人脸口罩的检测。 
# <br></br>
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/24af9157a7c54789960fe6bee511734608a7dca034c545b19df3ed8e3d2e158a" width = "700"></center>
# <center><br>图2 目标检测应用场景示意图</br></center>
# <br></br>

# ## 1.3 实验环境
# 
# 本实验支持在实训平台或本地环境操作，建议使用实训平台。
# 
# * **实训平台**：如果选择在实训平台上操作，仅需安装少部分实验环境。实训平台集成了实验必须的大部分相关环境，代码可在线运行，同时还提供了免费算力，即使实践复杂模型也无算力之忧。
# * **本地环境**：如果选择在本地环境上操作，需要安装Python3.7、飞桨开源框架2.0等实验必须的环境，具体要求及实现代码请参见[《本地环境安装说明》](https://aistudio.baidu.com/aistudio/projectdetail/1793329)。

# ## 1.4 实验设计
# 
# 本实验的实现方案如 **图3** 所示。
# * 在训练阶段：
# 1. 按一定规则在图片上产生一系列的候选区域，然后根据这些候选区域与图片上物体真实框之间的位置关系对候选区域进行标注。跟真实框足够接近的那些候选区域会被标注为正样本，同时将真实框的位置作为正样本的位置目标。偏离真实框较大的那些候选区域则会被标注为负样本，负样本不需要预测位置或者类别。
# 1. 使用卷积神经网络提取图片特征并对候选区域的位置和类别进行预测。这样每个预测框就可以看成是一个样本，根据真实框相对它的位置和类别进行了标注而获得标签值，通过网络模型预测其位置和类别，将网络预测值和标签值进行比较，就可以建立起损失函数。
# * 在预测阶段，根据预先定义的锚框和提取到的图片特征计算预测框，然后使用多分类非极大值抑制消除重合较大的框，得到最终结果。
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/e4a6f2ee77c645de9ddd4aca24b6e34f8490f2670c51493b8dcf01e18b1f4de8" width = "900"></center>
# <center><br>图3 目标检测设计方案</br></center>

# # 2. 基于VOC Dataset的目标检测实验详细实现
# 
# 基于VOC Dataset的目标检测实验流程如 **图4** 所示，包含如下9个步骤：
# 
# **1.数据处理**：根据网络接收的数据格式，完成相应的预处理操作，保证模型正常读取，同时，对于训练数据，使用数据增广策略来提升模型泛化性能；
# 
# **2.模型构建**：设计深度神经网络结构（模型的假设空间）；
# 
# **3.模型后处理**：通过模型预测得到的概率图，经过一系列后处理操作得到真实的输出值；
# 
# **4.损失函数定义**：根据预测值和真实值构建损失函数，神经网络通过最小化损失函数使得网络的输出值更接近真实值；
# 
# **5.训练配置**：实例化模型，加载模型参数，指定模型采用的寻解算法（优化器）；
# 
# **6.模型训练**：执行多轮训练不断调整参数，以达到较好的效果；
# 
# **7.模型保存**：将模型参数保存到指定位置，便于后续推理或继续训练使用；
# 
# **8.模型评估**：对训练好的模型进行评估测试，观察准确率和Loss；
# 
# **9.模型推理及可视化**：使用一张真实图片来验证模型识别的效果，并可视化推理结果。
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/fc58e8f1fa1943448d5077d7bff25afa94c4d41925274f3985592cc935f253c9" width = "900"></center>
# <center><br>图4 基于VOC Dataset的目标检测实验流程</br></center>
# <br></br>

# ## 2.1 数据处理
# 
# ### 2.1.1 数据集准备
# 
# 本次实验选取了学术经典的[VOC2007 Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)数据集。VOC2007 Dataset 是 PASCAL VOC挑战赛使用的数据集，包含了20种常见类别的图片。其中训练集共5011幅，测试集4952幅，是目标检测领域的经典学术数据集之一。VOC2007 Dataset 如 **图5** 所示。
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/df1723b2be33482c94b1b6d28ffab7ee4c1a1089488e41e0982ae28c796aa466" width = "600"></center>
# <center><br>图5 VOC2007 Dataset示意图</br></center>
# <br></br>
# 
# 
# 已经预先将原始数据集中图片以及标注文件抽取出来，按照更为清晰的目录结构进行了整理。解压缩后的目录结构如下：
# 
# ```
# VOC
#     |---train
#     |         |---annotations
#     |         |         |---xmls
#     |         |                  |---000005.xml
#     |         |                  |---000007.xml
#     |         |                  |---...
#     |         |
#     |         |---images
#     |                   |---000005.jpg
#     |                   |---000007.jpg
#     |                   |---...
#     |
#     |---test
#     |        |---annotations
#     |        |         |---xmls
#     |        |                  |---000008.xml
#     |        |                  |---000015.xml
#     |        |                  |---...
#     |        |
#     |        |---images
#     |                  |---000008.jpg
#     |                  |---000015.jpg
#     |                  |---...
# ```
# 
# 其中，train/annotations/xmls目录下存放着图片的标注。每个xml文件是对一张图片的说明，包括图片尺寸、包含的目标类别、在图片上出现的位置等信息。
# 
# 文件解压操作代码实现如下：

# In[1]:


get_ipython().run_line_magic('cd', '/home/aistudio/work/')
# 解压数据脚本，第一次运行时打开注释，将文件解压到work目录下
get_ipython().system('unzip -q /home/aistudio/data1/data80490/VOC.zip')


# 每一个xml标注文件中，保存有如下信息：
# 
# * filename：图片名称。
# * size：图片尺寸。
# * object：图片中包含的物体，一张图片可能中包含多个物体。
# 
#  -- name：类别名称；
#  
#  -- bndbox：物体真实框；
#  
#  -- difficult：识别是否困难。
#  
#  这里以其中的一个xml为例，可以简单观察一下xml文件的格式。
# ```
# <annotation>
# 	<folder>VOC2007</folder>
# 	<filename>000010.jpg</filename>
# 	<source>
# 		<database>The VOC2007 Database</database>
# 		<annotation>PASCAL VOC2007</annotation>
# 		<image>flickr</image>
# 		<flickrid>227250080</flickrid>
# 	</source>
# 	<owner>
# 		<flickrid>genewolf</flickrid>
# 		<name>whiskey kitten</name>
# 	</owner>
# 	<size>
# 		<width>354</width>
# 		<height>480</height>
# 		<depth>3</depth>
# 	</size>
# 	<segmented>0</segmented>
# 	<object>
# 		<name>horse</name>
# 		<pose>Rear</pose>
# 		<truncated>0</truncated>
# 		<difficult>0</difficult>
# 		<bndbox>
# 			<xmin>87</xmin>
# 			<ymin>97</ymin>
# 			<xmax>258</xmax>
# 			<ymax>427</ymax>
# 		</bndbox>
# 	</object>
# 	<object>
# 		<name>person</name>
# 		<pose>Unspecified</pose>
# 		<truncated>0</truncated>
# 		<difficult>0</difficult>
# 		<bndbox>
# 			<xmin>133</xmin>
# 			<ymin>72</ymin>
# 			<xmax>245</xmax>
# 			<ymax>284</ymax>
# 		</bndbox>
# 	</object>
# </annotation>
# ```

# 使用PIL库，随机选取一张图片可视化，观察该数据集的图片数据。

# In[3]:


# coding=utf-8
# 导入环境
import os
import random
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
# 在notebook中使用matplotlib.pyplot绘图时，需要添加该命令进行显示
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.image import imread
import matplotlib.patches as patches
import cv2
from PIL import Image, ImageEnhance
import paddle
import paddle.nn.functional as F

# 使用PIL库读取图片，并转为numpy array的格式
# （思考）您也可以尝试修改下面代码中images文件夹包含的图片标号，可视化数据集中的其他图片。

img = Image.open('./VOC/train/images/008150.jpg')
img = np.array(img)

# 画出读取的图片
plt.figure(figsize=(10, 10))
plt.imshow(img)


# ### 2.1.2 数据读取
# 
# 数据准备好后，就可以从数据集中读取xml文件，将每张图片的标注信息读取出来。从而获取图片路径、边界框等实验所需的信息。
# 
# 在读取具体的标注文件之前，先完成一件事情，就是将类别名字（字符串）转化成数字id。因为神经网络里面计算时需要的输入类型是数值型的，所以需要将字符串表示的类别转化成具体的数字。使用下面的程序可以得到表示名称字符串和数字类别之间映射关系的字典。

# In[4]:


# VOC类别标签列表
VOC_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# 将VOC数据集中的标签名称转为id值
def get_voc_names():
    voc_category2id = {}
    for i, item in enumerate(VOC_NAMES):
        voc_category2id[item] = i

    return voc_category2id


# 调用get_voc_names函数可以返回一个描述了数据集中类别名称和类别id之间映射关系的dict。然后可以通过下面的程序，从annotations/xml目录下面读取所有文件标注信息。

# In[5]:


def get_annotations(cname2cid, datadir):
    filenames = os.listdir(os.path.join(datadir, 'annotations', 'xmls'))
    records = []
    ct = 0
    for fname in filenames:
        fid = fname.split('.')[0]
        fpath = os.path.join(datadir, 'annotations', 'xmls', fname)
        img_file = os.path.join(datadir, 'images', fid + '.jpg')
        # 解析标注文件
        tree = ET.parse(fpath)

        if tree.find('id') is None:
            im_id = np.array([ct])
        else:
            im_id = np.array([int(tree.find('id').text)])
        # 解析图片信息
        objs = tree.findall('object')
        im_w = float(tree.find('size').find('width').text)
        im_h = float(tree.find('size').find('height').text)
        gt_bbox = np.zeros((len(objs), 4), dtype=np.float32)
        gt_class = np.zeros((len(objs), ), dtype=np.int32)
        is_crowd = np.zeros((len(objs), ), dtype=np.int32)
        difficult = np.zeros((len(objs), ), dtype=np.int32)
        # 获取并解析边界框信息
        for i, obj in enumerate(objs):
            cname = obj.find('name').text
            gt_class[i] = cname2cid[cname]
            _difficult = int(obj.find('difficult').text)
            x1 = float(obj.find('bndbox').find('xmin').text)
            y1 = float(obj.find('bndbox').find('ymin').text)
            x2 = float(obj.find('bndbox').find('xmax').text)
            y2 = float(obj.find('bndbox').find('ymax').text)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(im_w - 1, x2)
            y2 = min(im_h - 1, y2)
            # 这里使用xywh格式来表示目标物体真实框
            gt_bbox[i] = [(x1+x2)/2.0 , (y1+y2)/2.0, x2-x1+1., y2-y1+1.]
            is_crowd[i] = 0
            difficult[i] = _difficult

        voc_rec = {
            'im_file': img_file,
            'im_id': im_id,
            'h': im_h,
            'w': im_w,
            'is_crowd': is_crowd,
            'gt_class': gt_class,
            'gt_bbox': gt_bbox,
            'gt_poly': [],
            'difficult': difficult
            }
        if len(objs) != 0:
            records.append(voc_rec)
        ct += 1
    return records


# 通过上面的程序，将所有训练数据集的标注数据全部读取出来了，存放在records列表下面，其中每一个元素是一张图片的标注数据，包含了`图片存放地址`，`图片id`，`图片高度和宽度`，图片中所包含的`目标物体的种类和位置`。下面的程序展示了如何根据records里面的描述读取图片及标注。

# In[6]:


# 数据读取
def get_bbox(gt_bbox, gt_class):
    # 对于一般的检测任务来说，一张图片上往往会有多个目标物体
    # 设置参数MAX_NUM = 50， 即一张图片最多取50个真实框；如果真实
    # 框的数目少于50个，则将不足部分的gt_bbox, gt_class和gt_score的各项数值全设置为0
    MAX_NUM = 50
    gt_bbox2 = np.zeros((MAX_NUM, 4))
    gt_class2 = np.zeros((MAX_NUM,))
    for i in range(len(gt_bbox)):
        gt_bbox2[i, :] = gt_bbox[i, :]
        gt_class2[i] = gt_class[i]
        if i >= MAX_NUM:
            break
    return gt_bbox2, gt_class2


# In[7]:


# 返回图片数据的数据，包括图像img，真实框坐标gt_boxes，真实框包含的物体类别gt_labels，图像尺寸(h, w)。
def get_img_data_from_file(record):
    im_file = record['im_file']
    h = record['h']
    w = record['w']
    is_crowd = record['is_crowd']
    gt_class = record['gt_class']
    gt_bbox = record['gt_bbox']
    difficult = record['difficult']
    img = cv2.imread(im_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gt_boxes, gt_labels = get_bbox(gt_bbox, gt_class)

    # gt_bbox 用相对值
    gt_boxes[:, 0] = gt_boxes[:, 0] / float(w)
    gt_boxes[:, 1] = gt_boxes[:, 1] / float(h)
    gt_boxes[:, 2] = gt_boxes[:, 2] / float(w)
    gt_boxes[:, 3] = gt_boxes[:, 3] / float(h)
  
    return img, gt_boxes, gt_labels, (h, w), im_file


# ### 2.1.3 数据预处理
# 
# 本实验中，目标检测算法对输入图片的格式、大小有一定的要求，数据灌入模型前，需要对数据进行预处理操作，使图片满足网络训练以及预测的需要。另外，为了扩大训练数据集、抑制过拟合，提升模型的泛化能力，实验中还使用了几种基础的数据增广方法。
# 
# 本实验的数据预处理共包括如下方法：
# 
# * **随机改变亮暗、对比度和颜色**：使用PIL图像处理库完成对亮度、对比度、颜色的随机变换；
# * **随机填充**：在图像外围进行填充，填充后的新图像大小为随机数；
# * **随机裁剪**：随机裁剪图像中的一块子区域作为新图像。在裁剪的过程中，随时观察新图像中是否还保留有真实框，要避免裁剪得到的新图像中没有物体的情况；
# * **随机缩放**：随机选择差值方式，根据传入的尺寸参数进行图片缩放；
# * **随机翻转**：使用一定的概率进行图片翻转；
# * **随机打乱真实框排列顺序**：通过大量实验发现，模型对最后出现的数据印象更加深刻。训练数据导入后，越接近模型训练结束，最后几个批次数据对模型参数的影响越大。为了避免模型记忆影响训练效果，需要进行样本乱序操作。
# * **归一化与通道变换**：将图像数据中的数值除以255，并且减去均值、除以方差，改变为标准正态分布，使得最优解的寻优过程变得平缓，训练过程更容易收敛；同时，由于图像的数据格式为[H, W, C]（即高度、宽度和通道数），而神经网络使用的训练数据的格式为[C, H, W]，因此需要对图像数据重新排列，例如[224, 224, 3]变为[3, 224, 224]；
# 
# 下面分别介绍数据预处理方法的代码实现。

#  **随机改变亮暗、对比度和颜色等**

# In[8]:


# 随机改变亮暗、对比度和颜色等
def random_distort(img):
    # 随机改变亮度
    def random_brightness(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(e)
    # 随机改变对比度
    def random_contrast(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(e)
    # 随机改变颜色
    def random_color(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(e)

    ops = [random_brightness, random_contrast, random_color]
    np.random.shuffle(ops)

    img = Image.fromarray(img)
    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)
    img = np.asarray(img)

    return img


# **随机填充**

# In[9]:


# 随机填充
def random_expand(img, gtboxes, max_ratio=4., fill=None, keep_ratio=True, thresh=0.5):
    if random.random() > thresh:
        return img, gtboxes

    if max_ratio < 1.0:
        return img, gtboxes

    h, w, c = img.shape
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)
    oh = int(h * ratio_y)
    ow = int(w * ratio_x)
    off_x = random.randint(0, ow - w)
    off_y = random.randint(0, oh - h)
    # 产生大小为(oh, ow, c)的新图片,先将其数值全部填充为fill指定的数值
    out_img = np.zeros((oh, ow, c))
    if fill and len(fill) == c:
        for i in range(c):
            out_img[:, :, i] = fill[i] * 255.0
    # 将原图赋值到新图片对应区域
    out_img[off_y:off_y + h, off_x:off_x + w, :] = img
    # 计算在新图片上真实框的坐标
    gtboxes[:, 0] = ((gtboxes[:, 0] * w) + off_x) / float(ow)
    gtboxes[:, 1] = ((gtboxes[:, 1] * h) + off_y) / float(oh)
    gtboxes[:, 2] = gtboxes[:, 2] / ratio_x
    gtboxes[:, 3] = gtboxes[:, 3] / ratio_y

    return out_img.astype('uint8'), gtboxes


# **随机裁剪**

# In[10]:


# 计算不同box之间的IOU
def multi_box_iou_xywh(box1, box2):
    assert box1.shape[-1] == 4, "Box1 shape[-1] should be 4."
    assert box2.shape[-1] == 4, "Box2 shape[-1] should be 4."


    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    # 计算交集
    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_y2 = np.minimum(b1_y2, b2_y2)
    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1
    inter_w = np.clip(inter_w, a_min=0., a_max=None)
    inter_h = np.clip(inter_h, a_min=0., a_max=None)

    inter_area = inter_w * inter_h
    # 计算两个区域面积
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # 计算IOU
    return inter_area / (b1_area + b2_area - inter_area)


# In[11]:


# 计算裁剪之后的真实框
def box_crop(boxes, labels, crop, img_shape):
    x, y, w, h = map(float, crop)
    im_w, im_h = map(float, img_shape)

    boxes = boxes.copy()
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] - boxes[:, 2] / 2) * im_w, (boxes[:, 0] + boxes[:, 2] / 2) * im_w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] - boxes[:, 3] / 2) * im_h, (boxes[:, 1] + boxes[:, 3] / 2) * im_h

    crop_box = np.array([x, y, x + w, y + h])
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    mask = np.logical_and(crop_box[:2] <= centers, centers <= crop_box[2:]).all(axis=1)

    boxes[:, :2] = np.maximum(boxes[:, :2], crop_box[:2])
    boxes[:, 2:] = np.minimum(boxes[:, 2:], crop_box[2:])
    boxes[:, :2] -= crop_box[:2]
    boxes[:, 2:] -= crop_box[:2]

    mask = np.logical_and(mask, (boxes[:, :2] < boxes[:, 2:]).all(axis=1))
    boxes = boxes * np.expand_dims(mask.astype('float32'), axis=1)
    labels = labels * mask.astype('float32')
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) / 2 / w, (boxes[:, 2] - boxes[:, 0]) / w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) / 2 / h, (boxes[:, 3] - boxes[:, 1]) / h

    return boxes, labels, mask.sum()


# In[12]:


# 随机裁剪
def random_crop(img, boxes, labels, scales=[0.3, 1.0], max_ratio=2.0, constraints=None, max_trial=50):
    if len(boxes) == 0:
        return img, boxes

    if not constraints:
        constraints = [(0.1, 1.0), (0.3, 1.0), (0.5, 1.0), (0.7, 1.0), (0.9, 1.0), (0.0, 1.0)]

    img = Image.fromarray(img)
    w, h = img.size
    crops = [(0, 0, w, h)]
    for min_iou, max_iou in constraints:
        for _ in range(max_trial):
            scale = random.uniform(scales[0], scales[1])
            aspect_ratio = random.uniform(max(1 / max_ratio, scale * scale), min(max_ratio, 1 / scale / scale))
            crop_h = int(h * scale / np.sqrt(aspect_ratio))
            crop_w = int(w * scale * np.sqrt(aspect_ratio))
            crop_x = random.randrange(w - crop_w)
            crop_y = random.randrange(h - crop_h)
            crop_box = np.array([[(crop_x + crop_w / 2.0) / w, (crop_y + crop_h / 2.0) / h, crop_w / float(w), crop_h / float(h)]])
            # 计算多个框之间的IoU
            iou = multi_box_iou_xywh(crop_box, boxes)
            if min_iou <= iou.min() and max_iou >= iou.max():
                crops.append((crop_x, crop_y, crop_w, crop_h))
                break
    # 计算裁剪之后的真实框如果裁的太偏了，则重新裁一次直到裁剪出合适的新图片
    while crops:
        crop = crops.pop(np.random.randint(0, len(crops)))
        crop_boxes, crop_labels, box_num = box_crop(boxes, labels, crop, (w, h))
        if box_num < 1:
            continue
        img = img.crop((crop[0], crop[1], crop[0] + crop[2], crop[1] + crop[3])).resize(img.size, Image.LANCZOS)
        img = np.asarray(img)
        return img, crop_boxes, crop_labels
    img = np.asarray(img)
    return img, boxes, labels


# **随机缩放**

# In[13]:


# 随机缩放
def random_interp(img, size, interp=None):
    interp_method = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
    # 随机选择缩放时的插值方法
    if not interp or interp not in interp_method:
        interp = interp_method[random.randint(0, len(interp_method) - 1)]
    h, w, _ = img.shape
    im_scale_x = size / float(w)
    im_scale_y = size / float(h)
    # 调整图像大小
    img = cv2.resize(img, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=interp)
    return img


# **随机翻转**

# In[14]:


# 随机翻转
def random_flip(img, gtboxes, thresh=0.5):
    if random.random() > thresh:
        img = img[:, ::-1, :]
        gtboxes[:, 0] = 1.0 - gtboxes[:, 0]
    return img, gtboxes


# **随机打乱真实框排列顺序**

# In[15]:


# 随机打乱真实框排列顺序
def shuffle_gtbox(gtbox, gtlabel):
    gt = np.concatenate([gtbox, gtlabel[:, np.newaxis]], axis=1)
    idx = np.arange(gt.shape[0])
    # 随机打乱顺序
    np.random.shuffle(idx)
    gt = gt[idx, :]
    return gt[:, :4], gt[:, 4]


# **图像增广方法汇总**

# In[16]:


# 图像增广方法汇总
def image_augment(img, gtboxes, gtlabels, size, means=None):
    # 随机改变亮暗、对比度和颜色等
    img = random_distort(img)
    # 随机填充
    img, gtboxes = random_expand(img, gtboxes, fill=means)
    # 随机裁剪
    img, gtboxes, gtlabels, = random_crop(img, gtboxes, gtlabels)
    # 随机缩放
    img = random_interp(img, size)
    # 随机翻转
    img, gtboxes = random_flip(img, gtboxes)
    # 随机打乱真实框排列顺序
    gtboxes, gtlabels = shuffle_gtbox(gtboxes, gtlabels)

    return img.astype('float32'), gtboxes.astype('float32'), gtlabels.astype('int32')


# 
# 使用图像增广方法处理后得到的图像需要进行归一化，并将维度从[H, W, C]调整为[C, H, W]。

# In[17]:


def get_img_data(record, size=640, mode='train'):
    # 在标注数据中获取图片信息
    img, gt_boxes, gt_labels, scales, im_file = get_img_data_from_file(record)
    if mode == 'train':
        # 图像增广
        img, gt_boxes, gt_labels = image_augment(img, gt_boxes, gt_labels, size)
    else:
        img = cv2.resize(img, (size, size))
    # 图像归一化
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = np.array(mean).reshape((1, 1, -1))
    std = np.array(std).reshape((1, 1, -1))
    img = (img / 255.0 - mean) / std
    # 调整维度
    img = img.astype('float32').transpose((2, 0, 1))
    return img, gt_boxes, gt_labels, scales, os.path.split(im_file)[1].split('.')[0]


# ### 2.1.4 批量数据读取
# 
# 上面的代码仅展示了读取一张图片和预处理的方法，但在真实场景的模型训练与评估过程中，通常会使用批量数据读取和预处理的方式。
# 
# 在本实验中，网络训练时每个 batch 中的数据会随机选择一个尺寸进行缩放，尺寸范围在[320, 608]之间。在测试时，图像统一被缩放至608。
# 
# 获取一个批次内图像随机缩放的尺寸的代码如下。

# In[18]:


# 获取一个批次内样本随机缩放的尺寸
def get_img_size(mode):
    if mode == 'train':
        #inds = np.array([0,1,2,3,4,5,6,7,8,9])
        #ii = np.random.choice(inds)
        #img_size = 320 + ii * 32
        img_size = 640
    else:
        img_size = 608
    return img_size


# 定义数据读取器Dataset，实现数据批量读取和预处理。具体代码如下：

# In[19]:


# 定义数据读取类，继承Paddle.io.Dataset
class Dataset(paddle.io.Dataset):
    def  __init__(self, datadir, mode='train'):
        self.datadir = datadir
        cname2cid = get_voc_names()
        self.records = get_annotations(cname2cid, datadir)
        self.img_size = get_img_size(mode)
        self.mode = mode

    def __getitem__(self, idx):
        record = self.records[idx]
        img, gt_bbox, gt_labels, im_shape, img_name = get_img_data(record, size=self.img_size, mode=self.mode)
        if self.mode == 'train':
            return img, gt_bbox, gt_labels, np.array(im_shape)
        else:
            return int(img_name), img, np.array(im_shape).astype('int32')

    def __len__(self):
        return len(self.records)


# 数据预处理耗时较长，推荐使用 [paddle.io.DataLoader](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DataLoader_cn.html#dataloader) API中的``num_workers``参数，设置进程数量，实现多进程读取数据。
# 
# > *class* paddle.io.DataLoader(*dataset,  batch_size=2,  num_workers=2*) 
# 
# 关键参数含义如下：
# 
# * batch_size (int|None) - 每个mini-batch中样本个数；
# * num_workers (int) - 加载数据的子进程个数 。
# 
# 多线程读取实现代码如下。

# In[20]:


TRAINDIR = '/home/aistudio/work/VOC/train'
VALIDDIR = '/home/aistudio/work/VOC/test'

# 创建数据读取类
train_dataset = Dataset(TRAINDIR, mode='train')
valid_dataset = Dataset(VALIDDIR, mode='valid')
# 使用paddle.io.DataLoader创建数据读取器，并设置batchsize，进程数量num_workers等参数
train_loader = paddle.io.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0, drop_last=True)
valid_loader = paddle.io.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)


# 至此，完成了如何查看数据集中的数据、提取数据标注信息、从文件读取图像和标注数据、图像增广、批量读取和加速等过程，通过`paddle.io.Dataset`可以返回img, gt_boxes, gt_labels, im_shape等数据，接下来就可以将它们输入到神经网络，应用到具体算法上了。

# ## 2.2 模型构建
# 
# 经典的R-CNN系列算法也被称为两阶段目标检测算法，由于这种方法需要先产生候选区域，再对候选区域做分类和位置坐标的预测，因此算法速度非常慢。与此对应的是以YOLO算法为代表的单阶段检测算法，只需要一个网络即可同时产生候选区域并预测出物体的类别和位置坐标。
# 
# 与R-CNN系列算法不同，YOLOv3使用单个网络结构，在产生候选区域的同时即可预测出物体类别和位置，不需要分成两阶段来完成检测任务。另外，YOLOv3算法产生的预测框数目比Faster R-CNN少很多。Faster R-CNN中每个真实框可能对应多个标签为正的候选区域，而YOLOv3里面每个真实框只对应一个正的候选区域。这些特性使得YOLOv3算法具有更快的速度，能到达实时响应的水平。
# 
# Joseph Redmon等人在2015年提出YOLO（You Only Look Once，YOLO）算法，通常也被称为YOLOv1；2016年，他们对算法进行改进，又提出YOLOv2版本；2018年发展出YOLOv3版本。

# ### 2.2.1 YOLOv3 训练流程
# YOLOv3算法的训练流程可以分成两部分，如 **图6** 所示。
# 
# * 按一定规则在图片上产生一系列的候选区域，然后根据这些候选区域与图片上物体真实框之间的位置关系对候选区域进行标注。跟真实框足够接近的那些候选区域会被标注为正样本，同时将真实框的位置作为正样本的位置目标。偏离真实框较大的那些候选区域则会被标注为负样本，负样本不需要预测位置或者类别。
# * 使用卷积神经网络提取图片特征并对候选区域的位置和类别进行预测。这样每个预测框就可以看成是一个样本，根据真实框相对它的位置和类别进行了标注而获得标签值，通过网络模型预测其位置和类别，将网络预测值和标签值进行比较，就可以建立起损失函数。
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/3aaf17739c2948a78d0b86b994c8455468587ee936f14b6186c9bdb8b28873ba" width = "800"></center>
# <center><br>图6 YOLOv3算法训练流程图 </br></center>

# ### 2.2.2 产生候选区域
# 
# 如何产生候选区域，是检测模型的核心设计方案。目前大多数基于卷积神经网络的模型所采用的方式大体如下：
# 
# 1. 按一定的规则在图片上生成一系列位置固定的锚框，将这些锚框看作是可能的候选区域。
# 1. 对锚框是否包含目标物体进行预测，如果包含目标物体，还需要预测所包含物体的类别，以及预测框相对于锚框位置需要调整的幅度。

# **1. 生成锚框**
# 
# 将原始图片划分成$m\times n$个区域，如 **图7** 所示，原始图片高度$H=640$, 宽度$W=480$，如果选择小块区域的尺寸为$32 \times 32$，则$m$和$n$分别为：
# 
# $$m = \frac{640}{32} = 20$$
# 
# $$n = \frac{480}{32} = 15$$
# 
# 也就是说，将原始图像分成了20行15列小方块区域。
# 
# <br></br>
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/2dd1cbeb53644552a8cb38f3f834dbdda5046a489465454d93cdc88d1ce65ca5" width = "400"></center>
# <center><br>图7 将图片划分成多个32x32的小方块 </br></center>
# <br></br>
# 
# 
# YOLOv3算法会在每个区域的中心，生成一系列锚框。为了展示方便，仅在图中第十行第四列的小方块位置附近画出生成的锚框，如 **图8** 所示。
# 
# <br></br>
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/6dd42b9138364a379b6231ac2247d3cb449d612e17be4896986bca2703acbb29" width = "400"></center>
# <center><br>图8 在第10行第4列的小方块区域生成3个锚框 </br></center>
# <br></br>
# 
# ------
# **说明：**
# 
# 这里为了跟程序中的编号对应，最上面的行号是第0行，最左边的列号是第0列。
# 
# ------

# **2. 生成预测框**
# 
# 在前面已经指出，锚框的位置都是固定好的，不可能刚好跟物体边界框重合，需要在锚框的基础上进行位置的微调以生成预测框。预测框相对于锚框会有不同的中心位置和大小，采用什么方式能得到预测框呢？先来考虑如何生成其中心位置坐标。
# 
# 比如上面图中在第10行第4列的小方块区域中心生成的一个锚框，如绿色虚线框所示。以小方格的宽度为单位长度，
# 
# 此小方块区域左上角的位置坐标是：
# $$c_x = 4$$
# $$c_y = 10$$
# 
# 此锚框的区域中心坐标是：
# $$center\_x = c_x + 0.5 = 4.5$$
# $$center\_y = c_y + 0.5 = 10.5$$
# 
# 可以通过下面的方式生成预测框的中心坐标：
# $$b_x = c_x + \sigma(t_x)$$
# $$b_y = c_y + \sigma(t_y)$$
# 
# 其中$t_x$和$t_y$为实数，$\sigma(x)$是之前学过的Sigmoid函数，其定义如下：
# 
# $$\sigma(x) = \frac{1}{1 + exp(-x)}$$
# 
# 由于Sigmoid的函数值在$0 \thicksim 1$之间，因此由上面公式计算出来的预测框的中心点总是落在第十行第四列的小区域内部。
# 
# 当$t_x=t_y=0$时，$b_x = c_x + 0.5$，$b_y = c_y + 0.5$，预测框中心与锚框中心重合，都是小区域的中心。
# 
# 锚框的大小是预先设定好的，在模型中可以当作是超参数，下图中画出的锚框尺寸是
# 
# $$p_h = 350$$
# $$p_w = 250$$
# 
# 通过下面的公式生成预测框的大小：
# 
# $$b_h = p_h e^{t_h}$$
# $$b_w = p_w e^{t_w}$$
# 
# 如果$t_x=t_y=0, t_h=t_w=0$，则预测框跟锚框重合。
# 
# 如果给$t_x, t_y, t_h, t_w$随机赋值如下：
# 
# $$t_x = 0.2,  t_y = 0.3, t_w = 0.1, t_h = -0.12$$
# 
# 则可以得到预测框的坐标是(154.98, 357.44, 276.29, 310.42)，如 **图9** 中蓝色框所示。
# 
# <br></br>
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/f4b33522eb5a45f0804b94a5c66b76a0a2d13345d6de499399580a031b6ccc74" width = "400"></center>
# <center><br>图9 生成预测框 </br></center>
# <br></br>
# 
# ------
# **说明：**
# 这里坐标采用$xywh$的格式。
# 
# -------
# 
# 这里：当$t_x, t_y, t_w, t_h$取值为多少的时候，预测框能够跟真实框重合？为了回答问题，只需要将上面预测框坐标中的$b_x, b_y, b_h, b_w$设置为真实框的位置，即可求解出$t$的数值。
# 
# 令：
# $$\sigma(t^*_x) + c_x = gt_x$$
# $$\sigma(t^*_y) + c_y = gt_y$$
# $$p_w e^{t^*_w} = gt_h$$
# $$p_h e^{t^*_h} = gt_w$$
# 
# 可以求解出：$(t^*_x, t^*_y, t^*_w, t^*_h)$
# 
# 如果$t$是网络预测的输出值，将$t^*$作为目标值，以他们之间的差距作为损失函数，则可以建立起一个回归问题，通过学习网络参数，使得$t$足够接近$t^*$，从而能够求解出预测框的位置坐标和大小。
# 
# 预测框可以看作是在锚框基础上的一个微调，每个锚框会有一个跟它对应的预测框，需要确定上面计算式中的$t_x, t_y, t_w, t_h$，从而计算出与锚框对应的预测框的位置和形状。

# **3. 对候选区域进行标注**
# 
# 在YOLOv3中，每个区域会产生3种不同形状的锚框，每个锚框都是一个可能的候选区域，对这些候选区域需要了解如下几件事情：
# 
# - 锚框是否包含物体，这可以看成是一个二分类问题，使用标签objectness来表示。当锚框包含了物体时，objectness=1，表示预测框属于正类；当锚框不包含物体时，设置objectness=0，表示锚框属于负类；还有一种情况，有些预测框跟真实框之间的IoU很大，但并不是最大的那个，那么直接将其objectness标签设置为0当作负样本，可能并不妥当，为了避免这种情况，YOLOv3算法设置了一个IoU阈值iou_threshold，当预测框的objectness不为1，但是其与某个真实框的IoU大于iou_threshold时，就将其objectness标签设置为-1，不参与损失函数的计算。
# 
# - 如果锚框包含了物体，那么就需要计算对应的预测框中心位置和大小应该是多少，或者说上文中的$t_x, t_y, t_w, t_h$应该是多少。
# 
# - 如果锚框包含了物体，那么就需要计算具体类别是什么，这里使用变量label来表示其所属类别的标签。
# 
# 总结起来，如 **图10** 所示。
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/cdb2db48bdb24fcf98950fdbbfce98580193364c8dcb4fd0b57eabab22c59bfa" width = "700"></center>
# <center><br>图10 标注流程示意图 </br></center>
# <br></br>
# 
# 通过上述介绍，初步了解了YOLOv3中候选区域的标注方式，通过这种方式，就可以获取到真实的预测框标签。在 Paddle 中，这些操作都已经被封装到了 [paddle.vision.ops.yolo_loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/ops/yolo_loss_cn.html) API中，计算损失时，只需要调用这个API即可简单实现上述过程。接下来，再了解一下YOLOv3的网络结构，看网络是如何计算得到对应的预测值的。

# ### 2.2.3 YOLOv3 网络结构
# 
# **1. backbone**
# 
# YOLOv3算法使用的骨干网络是Darknet53。Darknet53网络的具体结构如 **图11** 所示，在ImageNet图像分类任务上取得了很好的成绩。在检测任务中，将图中C0后面的平均池化、全连接层和Softmax去掉，保留从输入到C0部分的网络结构，作为检测模型的基础网络结构，也称为骨干网络。YOLOv3模型会在骨干网络的基础上，再添加检测相关的网络模块。
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/dec8826ad94c4d1cac5832aca9aa51dc1d1824ace48c44d09646ee85f39a857e" width = "400"></center>
# <center><br>图11 Darknet53网络结构 </br></center>
# <br></br>
# 
# 下面的程序是Darknet53骨干网络的实现代码，这里将上图中C0、C1、C2所表示的输出数据取出，并查看它们的形状分别是，$C0 [1, 1024, 20, 20]$，$C1 [1, 512, 40, 40]$，$C2 [1, 256, 80, 80]$。

# In[21]:


# 将卷积和批归一化封装为ConvBNLayer，方便后续复用
class ConvBNLayer(paddle.nn.Layer):
    def __init__(self, ch_in, ch_out,  kernel_size=3, stride=1, groups=1, padding=0, act="leaky"):
        # 初始化函数
        super(ConvBNLayer, self).__init__()
        # 创建卷积层
        self.conv = paddle.nn.Conv2D(in_channels=ch_in, out_channels=ch_out, kernel_size=kernel_size, stride=stride, padding=padding, 
            groups=groups, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Normal(0., 0.02)), bias_attr=False)
        # 创建批归一化层
        self.batch_norm = paddle.nn.BatchNorm2D(num_features=ch_out,
            weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Normal(0., 0.02), regularizer=paddle.regularizer.L2Decay(0.)),
            bias_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0), regularizer=paddle.regularizer.L2Decay(0.)))
        self.act = act

    def forward(self, inputs):
        # 前向计算
        out = self.conv(inputs)
        out = self.batch_norm(out)
        if self.act == 'leaky':
            out = F.leaky_relu(x=out, negative_slope=0.1)
        return out


# In[29]:


# 定义下采样模块，使图片尺寸减半
class DownSample(paddle.nn.Layer):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=2, padding=1):
        # 初始化函数
        super(DownSample, self).__init__()
        # 使用 stride=2 的卷积，可以使图片尺寸减半
        self.conv_bn_layer = ConvBNLayer(ch_in=ch_in, ch_out=ch_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.ch_out = ch_out
        
    def forward(self, inputs):
        # 前向计算
        out = self.conv_bn_layer(inputs)
        return out


# In[22]:


# 定义残差块
class BasicBlock(paddle.nn.Layer):
    def __init__(self, ch_in, ch_out):
        # 初始化函数
        super(BasicBlock, self).__init__()
        # 定义两个卷积层
        self.conv1 = ConvBNLayer(ch_in=ch_in, ch_out=ch_out, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBNLayer(ch_in=ch_out, ch_out=ch_out*2, kernel_size=3, stride=1, padding=1)
        
    def forward(self, inputs):
        # 前向计算
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        # 将第二个卷积层的输出和最初的输入值相加
        out = paddle.add(x=inputs, y=conv2)
        return out


# In[23]:


# 将多个残差块封装为一个层级，方便后续复用
class LayerWarp(paddle.nn.Layer):
    def __init__(self, ch_in, ch_out, count, is_test=True):
        # 初始化函数
        super(LayerWarp,self).__init__()
        self.basicblock0 = BasicBlock(ch_in, ch_out)
        self.res_out_list = []
        for i in range(1, count):
            # 使用add_sublayer添加子层
            res_out = self.add_sublayer("basic_block_%d" % (i), BasicBlock(ch_out*2, ch_out))
            self.res_out_list.append(res_out)

    def forward(self,inputs):
        # 前向计算
        y = self.basicblock0(inputs)
        for basic_block_i in self.res_out_list:
            y = basic_block_i(y)
        return y


# In[24]:


# DarkNet 每组残差块的个数，来自DarkNet的网络结构图
DarkNet_cfg = {53: ([1, 2, 8, 8, 4])}
# 创建DarkNet53骨干网络
class DarkNet53_conv_body(paddle.nn.Layer):
    def __init__(self):
        # 初始化函数
        super(DarkNet53_conv_body, self).__init__()
        self.stages = DarkNet_cfg[53]
        self.stages = self.stages[0:5]

        # 第一层卷积
        self.conv0 = ConvBNLayer(ch_in=3, ch_out=32, kernel_size=3, stride=1, padding=1)

        # 下采样，使用stride=2的卷积来实现
        self.downsample0 = DownSample(ch_in=32, ch_out=32 * 2)

        # 添加各个层级的实现
        self.darknet53_conv_block_list = []
        self.downsample_list = []
        for i, stage in enumerate(self.stages):
            conv_block = self.add_sublayer("stage_%d" % (i), LayerWarp(32*(2**(i+1)), 32*(2**i), stage))
            self.darknet53_conv_block_list.append(conv_block)
        # 两个层级之间使用DownSample将尺寸减半
        for i in range(len(self.stages) - 1):
            downsample = self.add_sublayer("stage_%d_downsample" % i, DownSample(ch_in=32*(2**(i+1)), ch_out=32*(2**(i+2))))
            self.downsample_list.append(downsample)

    def forward(self,inputs):
        # 前向计算
        out = self.conv0(inputs)
        out = self.downsample0(out)
        blocks = []
        # 依次将各个层级作用在输入上面
        for i, conv_block_i in enumerate(self.darknet53_conv_block_list): 
            out = conv_block_i(out)
            blocks.append(out)
            if i < len(self.stages) - 1:
                out = self.downsample_list[i](out)
        # 将C0, C1, C2作为返回值
        return blocks[-1:-4:-1] 


# **2. 预测模块**
# 
# 上文中，了解到在YOLOv3算法中，网络需要输出3组结果，分别是：
# 
# - 预测框是否包含物体。也可理解为 objectness=1 的概率是多少，这里可以让网络输出一个实数 $x$ ，然后用 $Sigmoid(x)$ 表示 objectness 为正的概率 $P_{obj}$ 。
# 
# - 预测物体位置和形状。可以用网络输出4个实数来表示物体位置和形状 $t_x, t_y, t_w, t_h$ 。
# 
# - 预测物体类别。预测图像中物体的具体类别是什么，或者说其属于每个类别的概率分别是多少。总的类别数为 C ，需要预测物体属于每个类别的概率 $(P_1, P_2, ..., P_C)$ ，可以用网络输出C个实数 $(x_1, x_2, ..., x_C)$ ，对每个实数分别求 Sigmoid 函数，让 $P_i = Sigmoid(x_i)$ ，则可以表示出物体属于每个类别的概率。
# 
# 
# 因此，对于一个预测框，网络需要输出 $(5 + C)$ 个实数来表征它是否包含物体、位置和形状尺寸以及属于每个类别的概率。
# 
# 由于在每个小方块区域都生成了 3 个预测框，则所有预测框一共需要网络输出的预测值数目是：
# 
# $$[3 \times (5 + C)] \times m \times n $$
# 
# 还有更重要的一点是网络输出必须要能区分出小方块区域的位置来，不能直接将特征图连接一个输出大小为 $[3 \times (5 + C)] \times m \times n$ 的全连接层。
# 
# 这里继续使用上文中的图片，现在观察特征图，经过多次卷积核池化之后，其步幅 stride=32，$640 \times 480$ 大小的输入图片变成了 $20\times15$ 的特征图；而小方块区域的数目正好是$20\times15$，也就是说可以让特征图上每个像素点分别跟原图上一个小方块区域对应。这也是为什么最开始将小方块区域的尺寸设置为32的原因，这样可以巧妙的将小方块区域跟特征图上的像素点对应起来，解决了空间位置的对应关系。
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/79ed292d500d451bb0fe804b8a9e1c962c825f53616b4422afd8f3e5b536a589" width = "600"></center>
# <center><br>图12 特征图C0与小方块区域形状对比 </br></center>
# <br></br>
# 
# 下面还需要将像素点$(i,j)$与第i行第j列的小方块区域所需要的预测值关联起来，每个小方块区域产生3个预测框，每个预测框需要$(5 + C)$个实数预测值，则每个像素点相对应的要有$3 \times (5 + C)$个实数。为了解决这一问题，对特征图进行多次卷积，并将最终的输出通道数设置为$3 \times (5 + C)$，即可将生成的特征图与每个预测框所需要的预测值巧妙的对应起来。经过卷积后，保证输出的特征图尺寸变为$[1, 75, 20, 15]$。每个小方块区域生成的锚框或者预测框的数量是3，物体类别数目是20，每个区域需要的预测值个数是$3 \times (5 + 20) = 75$，正好等于输出的通道数。
# 
# 此时，输出特征图与候选区域关联方式如 **图13** 所示。
# 
# 将$P0[t, 0:25, i, j]$与输入的第t张图片上小方块区域$(i, j)$第1个预测框所需要的25个预测值对应，$P0[t, 25:50, i, j]$与输入的第t张图片上小方块区域$(i, j)$第2个预测框所需要的25个预测值对应，$P0[t, 50:75, i, j]$与输入的第t张图片上小方块区域$(i, j)$第3个预测框所需要的25个预测值对应。
# 
# $P0[t, 0:4, i, j]$与输入的第t张图片上小方块区域$(i, j)$第1个预测框的位置对应，$P0[t, 4, i, j]$与输入的第t张图片上小方块区域$(i, j)$第1个预测框的objectness对应，$P0[t, 5:25, i, j]$与输入的第t张图片上小方块区域$(i, j)$第1个预测框的类别对应。
# 
# 通过这种方式可以巧妙的将网络输出特征图，与每个小方块区域生成的预测框对应起来了。
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/94e03942bc97416a9a3fdfe58e75ec211edb37ec4e584f45a7504fa216ce79d0" width = "800"></center>
# <center><br>图13 特征图P0与候选区域的关联 </br></center>
# <br></br>
# 
# 骨干网络的输出特征图是C0，下面的程序是对C0进行多次卷积以得到跟预测框相关的特征图P0。

# In[25]:


class YoloDetectionBlock(paddle.nn.Layer):
    # define YOLOv3 detection head
    # 使用多层卷积和BN提取特征
    def __init__(self,ch_in,ch_out,is_test=True):
        super(YoloDetectionBlock, self).__init__()

        assert ch_out % 2 == 0,             "channel {} cannot be divided by 2".format(ch_out)

        self.conv0 = ConvBNLayer(ch_in=ch_in, ch_out=ch_out, kernel_size=1, stride=1, padding=0)
        self.conv1 = ConvBNLayer(ch_in=ch_out, ch_out=ch_out*2, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBNLayer(ch_in=ch_out*2, ch_out=ch_out, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvBNLayer(ch_in=ch_out, ch_out=ch_out*2, kernel_size=3, stride=1, padding=1)
        self.route = ConvBNLayer(ch_in=ch_out*2, ch_out=ch_out, kernel_size=1, stride=1, padding=0)
        self.tip = ConvBNLayer(ch_in=ch_out, ch_out=ch_out*2, kernel_size=3, stride=1, padding=1)
        
    def forward(self, inputs):
        out = self.conv0(inputs)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        route = self.route(out)
        tip = self.tip(route)
        return route, tip


# ### 2.2.4 多尺度检测
# 
# 上文中讲解的运算过程都是在特征图P0的基础上进行的，它的步幅stride=32。特征图的尺寸比较小，像素点数目比较少，每个像素点的感受野很大，具有非常丰富的高层级语义信息，可能比较容易检测到较大的目标。为了能够检测到尺寸较小的那些目标，需要在尺寸较大的特征图上面建立预测输出。如果在C2或者C1这种层级的特征图上直接产生预测输出，可能面临新的问题，它们没有经过充分的特征提取，像素点包含的语义信息不够丰富，有可能难以提取到有效的特征模式。在目标检测中，解决这一问题的方式是，将高层级的特征图尺寸放大之后跟低层级的特征图进行融合，得到的新特征图既能包含丰富的语义信息，又具有较多的像素点，能够描述更加精细的结构。
# 
# 具体的网络实现方式如 **图14** 所示：
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/295bbf5cfbad47d2b9833baf5e0babc775566c5638c64f6aa685c631f6fc95f2" width = "800"></center>
# <center><br>图14 生成多层级的输出特征图P0、P1、P2 </br></center>
# <br></br>
# 
# YOLOv3在每个区域的中心位置产生3个锚框，在3个层级的特征图上产生锚框的大小分别为P2 [(10×13),(16×30),(33×23)]，P1 [(30×61),(62×45),(59× 119)]，P0[(116 × 90), (156 × 198), (373 × 326]。越往后的特征图上用到的锚框尺寸也越大，能捕捉到大尺寸目标的信息；越往前的特征图上锚框尺寸越小，能捕捉到小尺寸目标的信息。
# 
# 所以，最终的损失函数计算以及模型预测都是在这3个层级上进行的。接下来就可以进行完整网络结构的定义了。
# 
# ---
# 
# 
# **说明：**
# 
# YOLOv3中，损失函数主要包括3个部分，分别是：
# 
# - 表征是否包含目标物体的损失函数，使用二值交叉熵损失函数进行计算。
# 
# - 表征物体位置的损失函数，其中，$t_x, t_y$ 使用二值交叉熵损失函数进行计算，$t_w, t_h$ 使用L1损失进行计算。
# 
# - 表征物体类别的损失函数，使用二值交叉熵损失函数进行计算。
# ---

# 在完整网络定义时，需要使用 [paddle.vision.ops.yolo_loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/ops/yolo_loss_cn.html) API来计算损失函数，该API 将上述候选区域的标注以及多尺度的损失函数计算统一地进行了封装。同时，需要使用[paddle.vision.ops.yolo_box](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/ops/yolo_box_cn.html)来计算三个层级的特征图对应的预测框和得分。
#    
# 返回值包括两项，boxes和scores，其中boxes是所有预测框的坐标值，scores是所有预测框的得分。
# 
# 预测框得分的定义是所属类别的概率乘以其预测框是否包含目标物体的objectness概率，即
# 
# $$score = P_{obj} \cdot P_{classification}$$
# 
# 通过调用`paddle.vision.ops.yolo_box`获得P0、P1、P2三个层级的特征图对应的预测框和得分，并将他们拼接在一块，即可得到所有的预测框及其属于各个类别的得分。
# 
# 定义完整网络的具体实现代码如下：

# In[26]:


# 定义上采样模块
class Upsample(paddle.nn.Layer):
    def __init__(self, scale=2):
        # 初始化函数
        super(Upsample,self).__init__()
        self.scale = scale

    def forward(self, inputs):
        # 前向计算
        # 获得动态的上采样输出形状
        shape_nchw = paddle.shape(inputs)
        shape_hw = paddle.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = paddle.cast(shape_hw, dtype='int32')
        out_shape = in_shape * self.scale
        out_shape.stop_gradient = True

        # 上采样计算
        out = paddle.nn.functional.interpolate(x=inputs, scale_factor=self.scale, mode="NEAREST")
        return out


# In[27]:


# 定义完整的YOLOv3模型
class YOLOv3(paddle.nn.Layer):
    def __init__(self, num_classes=20):
        # 初始化函数
        super(YOLOv3,self).__init__()

        self.num_classes = num_classes
        # 提取图像特征的骨干代码
        self.block = DarkNet53_conv_body()
        self.block_outputs = []
        self.yolo_blocks = []
        self.route_blocks_2 = []
        # 生成3个层级的特征图P0, P1, P2
        for i in range(3):
            # 添加从ci生成ri和ti的模块
            yolo_block = self.add_sublayer("yolo_detecton_block_%d" % (i),
                YoloDetectionBlock(ch_in=512//(2**i)*2 if i==0 else 512//(2**i)*2 + 512//(2**i), ch_out = 512//(2**i)))
            self.yolo_blocks.append(yolo_block)

            num_filters = 3 * (self.num_classes + 5)

            # 添加从ti生成pi的模块，这是一个Conv2D操作，输出通道数为3 * (num_classes + 5)
            block_out = self.add_sublayer("block_out_%d" % (i),
                paddle.nn.Conv2D(in_channels=512//(2**i)*2, out_channels=num_filters, kernel_size=1, stride=1, padding=0,
                       weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Normal(0., 0.02)),
                       bias_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0), regularizer=paddle.regularizer.L2Decay(0.))))
            self.block_outputs.append(block_out)
            if i < 2:
                # 对ri进行卷积
                route = self.add_sublayer("route2_%d"%i, ConvBNLayer(ch_in=512//(2**i), ch_out=256//(2**i), kernel_size=1, stride=1, padding=0))
                self.route_blocks_2.append(route)
            # 将ri放大以便跟c_{i+1}保持同样的尺寸
            self.upsample = Upsample()

    def forward(self, inputs):
        # 前向运算
        outputs = []
        blocks = self.block(inputs)
        for i, block in enumerate(blocks):
            if i > 0:
                # 将r_{i-1}经过卷积和上采样之后得到特征图，与这一级的ci进行拼接
                block = paddle.concat([route, block], axis=1)
            # 从ci生成ti和ri
            route, tip = self.yolo_blocks[i](block)
            # 从ti生成pi
            block_out = self.block_outputs[i](tip)
            # 将pi放入列表
            outputs.append(block_out)

            if i < 2:
                # 对ri进行卷积调整通道数
                route = self.route_blocks_2[i](route)
                # 对ri进行放大，使其尺寸和c_{i+1}保持一致
                route = self.upsample(route)

        return outputs

    def get_loss(self, outputs, gtbox, gtlabel, gtscore=None, anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
                 anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]], ignore_thresh=0.7, use_label_smooth=False):
        # 损失计算函数
        self.losses = []
        downsample = 32
        # 对三个层级分别求损失函数
        for i, out in enumerate(outputs): 
            anchor_mask_i = anchor_masks[i]
            # 使用paddle.vision.ops.yolo_loss 直接计算损失函数
            loss = paddle.vision.ops.yolo_loss(x=out, gt_box=gtbox, gt_label=gtlabel, gt_score=gtscore, anchors=anchors, anchor_mask=anchor_mask_i, 
                    class_num=self.num_classes, ignore_thresh=ignore_thresh, downsample_ratio=downsample, use_label_smooth=False)
            self.losses.append(paddle.mean(loss)) 
            # 下一级特征图的缩放倍数会减半
            downsample = downsample // 2 
        # 对所有层级求和
        return sum(self.losses) 

    def get_pred(self, outputs, im_shape=None, anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
                 anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]], valid_thresh = 0.01):
        # 预测函数
        downsample = 32
        total_boxes = []
        total_scores = []
        for i, out in enumerate(outputs):
            anchor_mask = anchor_masks[i]
            anchors_this_level = []
            for m in anchor_mask:
                anchors_this_level.append(anchors[2 * m])
                anchors_this_level.append(anchors[2 * m + 1])
            # 使用paddle.vision.ops.yolo_box 直接计算损失函数
            boxes, scores = paddle.vision.ops.yolo_box(x=out, img_size=im_shape, anchors=anchors_this_level, class_num=self.num_classes,
                   conf_thresh=valid_thresh, downsample_ratio=downsample, name="yolo_box" + str(i))
            total_boxes.append(boxes)
            total_scores.append(paddle.transpose( scores, perm=[0, 2, 1]))
            downsample = downsample // 2
        # 将三个层级的结果进行拼接
        yolo_boxes = paddle.concat(total_boxes, axis=1)
        yolo_scores = paddle.concat(total_scores, axis=2)
        return yolo_boxes, yolo_scores


# ## 2.3 训练配置
# 
# 1）声明定义好的模型实例。

# In[30]:


# 类别总数
NUM_CLASSES = 20
# 创建模型
model = YOLOv3(num_classes = NUM_CLASSES)  


# 2）定义优化器
# 
# 本实验使用Momentum优化器，其中，使用了分段设置学习率的策略。其中，在定义学习率时，需要使用[paddle.optimizer.lr.PiecewiseDecay](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/PiecewiseDecay_cn.html) API。
# 
# 具体代码如下所示：

# In[31]:


# 定义学习率
def get_lr(base_lr = 0.0001, lr_decay = 0.1):
    bd = [10000, 20000]
    lr = [base_lr, base_lr * lr_decay, base_lr * lr_decay * lr_decay]
    learning_rate = paddle.optimizer.lr.PiecewiseDecay(boundaries=bd, values=lr)
    return learning_rate


# In[32]:


# 定义学习率参数
learning_rate = get_lr()
# 创建优化器
opt = paddle.optimizer.Momentum(learning_rate=learning_rate, momentum=0.9, weight_decay=paddle.regularizer.L2Decay(0.0005), parameters=model.parameters())  


# 3）定义后处理过程
# 
# 在前边的计算过程中，网络对同一个目标可能会进行多次检测。这也就导致对于同一个物体，预测会产生多个预测框。因此，得到模型输出后，需要使用非极大值抑制（non-maximum suppression, nms）来消除冗余框。思想是，如果有多个预测框都对应同一个物体，则只选出得分最高的那个预测框，剩下的预测框被丢弃掉。
# 
# 如何判断两个预测框对应的是同一个物体呢，标准该怎么设置？
# 
# 如果两个预测框的类别一样，而且他们的位置重合度比较大，则可以认为他们是在预测同一个目标。非极大值抑制的做法是，选出某个类别得分最高的预测框，然后看哪些预测框跟它的IoU大于阈值，就把这些预测框给丢弃掉。这里IoU的阈值是超参数，需要提前设置，YOLOv3模型里面设置的是0.5。

# In[33]:


# 计算IoU，其中边界框的坐标形式为xyxy
def box_iou_xyxy(box1, box2):
    # 获取box1左上角和右下角的坐标
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    # 计算box1的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    # 获取box2左上角和右下角的坐标
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    # 计算box2的面积
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)
    
    # 计算相交矩形框的坐标
    xmin = np.maximum(x1min, x2min)
    ymin = np.maximum(y1min, y2min)
    xmax = np.minimum(x1max, x2max)
    ymax = np.minimum(y1max, y2max)
    # 计算相交矩形行的高度、宽度、面积
    inter_h = np.maximum(ymax - ymin + 1., 0.)
    inter_w = np.maximum(xmax - xmin + 1., 0.)
    intersection = inter_h * inter_w
    # 计算相并面积
    union = s1 + s2 - intersection
    # 计算交并比
    iou = intersection / union
    return iou


# 非极大值抑制的具体实现代码如下面的`nms`函数的定义，需要说明的是数据集中含有多个类别的物体，所以这里需要做多分类非极大值抑制，其实现原理与非极大值抑制相同，区别在于需要对每个类别都做非极大值抑制，实现代码如下面的`multiclass_nms`所示。

# In[34]:


# 非极大值抑制
def nms(bboxes, scores, score_thresh, nms_thresh):
    # 对预测框得分进行排序
    inds = np.argsort(scores)
    inds = inds[::-1]
    keep_inds = []
    # 循环遍历预测框
    while(len(inds) > 0):
        cur_ind = inds[0]
        cur_score = scores[cur_ind]
        # 如果预测框得分低于阈值，则退出循环
        if cur_score < score_thresh:
            break
        # 计算当前预测框与保留列表中的预测框IOU，如果小于阈值则保留该预测框，否则丢弃该预测框
        keep = True
        for ind in keep_inds:
            current_box = bboxes[cur_ind]
            remain_box = bboxes[ind]
            # 计算当前预测框与保留列表中的预测框IOU
            iou = box_iou_xyxy(current_box, remain_box)
            if iou > nms_thresh:
                keep = False
                break
        if keep:
            keep_inds.append(cur_ind)
        inds = inds[1:]
    return np.array(keep_inds)


# In[35]:


# 多分类非极大值抑制
def multiclass_nms(bboxes, scores, score_thresh=0.01, nms_thresh=0.45, pos_nms_topk=100):
    batch_size = bboxes.shape[0]
    class_num = scores.shape[1]
    rets = []
    for i in range(batch_size):
        bboxes_i = bboxes[i]
        scores_i = scores[i]
        ret = []
        # 遍历所有类别进行单分类非极大值抑制
        for c in range(class_num):
            scores_i_c = scores_i[c]
            # 单分类非极大值抑制
            keep_inds = nms(bboxes_i, scores_i_c, score_thresh, nms_thresh)
            if len(keep_inds) < 1:
                continue
            keep_bboxes = bboxes_i[keep_inds]
            keep_scores = scores_i_c[keep_inds]
            keep_results = np.zeros([keep_scores.shape[0], 6])
            keep_results[:, 0] = c
            keep_results[:, 1] = keep_scores[:]
            keep_results[:, 2:6] = keep_bboxes[:, :]
            ret.append(keep_results)
        if len(ret) < 1:
            rets.append(ret)
            continue
        ret_i = np.concatenate(ret, axis=0)
        scores_i = ret_i[:, 1]
        # 如果保留的预测框超过100个，只保留得分最高的100个
        if len(scores_i) > pos_nms_topk:
            inds = np.argsort(scores_i)[::-1]
            inds = inds[:pos_nms_topk]
            ret_i = ret_i[inds]
        rets.append(ret_i)
    return rets


# 4）定义评估方式
# 
# 本实验使用目标检测任务中最常用的评估指标mAP来观察模型效果。具体代码如下所示。

# In[36]:


# 计算边界框面积
def bbox_area(bbox, is_bbox_normalized):
    norm = 1. - float(is_bbox_normalized)
    width = bbox[2] - bbox[0] + norm
    height = bbox[3] - bbox[1] + norm
    return width * height


# In[37]:


# 计算两个边界框之间的jaccard重叠率
def jaccard_overlap(pred, gt, is_bbox_normalized=False):
    if pred[0] >= gt[2] or pred[2] <= gt[0] or pred[1] >= gt[3] or pred[3] <= gt[1]:
        return 0.
    inter_xmin = max(pred[0], gt[0])
    inter_ymin = max(pred[1], gt[1])
    inter_xmax = min(pred[2], gt[2])
    inter_ymax = min(pred[3], gt[3])
    inter_size = bbox_area([inter_xmin, inter_ymin, inter_xmax, inter_ymax], is_bbox_normalized)
    pred_size = bbox_area(pred, is_bbox_normalized)
    gt_size = bbox_area(gt, is_bbox_normalized)
    overlap = float(inter_size) / (pred_size + gt_size - inter_size)
    return overlap


# In[38]:


# 计算mAP
class DetectionMAP(object):
    def __init__(self, class_num, overlap_thresh=0.5, is_bbox_normalized=False):
        # 初始化
        self.class_num = class_num
        self.overlap_thresh = overlap_thresh
        self.is_bbox_normalized = is_bbox_normalized
        self.reset()

    def update(self, bbox, gt_box, gt_label, difficult=None):
        # 更新统计的metric
        if difficult is None:
            difficult = np.zeros_like(gt_label)
        # 记录每个类别真实框的数量
        for gtl, diff in zip(gt_label, difficult):
            if int(diff) == 0:
                self.class_gt_counts[int(np.array(gtl))] += 1
        # 统计分数列表，用于mAP计算
        visited = [False] * len(gt_label)
        for b in bbox:
            label, score, xmin, ymin, xmax, ymax = b.tolist()
            pred = [xmin, ymin, xmax, ymax]
            max_idx = -1
            max_overlap = -1.0
            # 遍历所有同类别的真实框
            for i, gl in enumerate(gt_label):
                if int(gl) == int(label):
                    # 计算jaccard重叠率
                    overlap = jaccard_overlap(pred, gt_box[i], self.is_bbox_normalized)
                    if overlap > max_overlap:
                        max_overlap = overlap
                        max_idx = i
            # 如果重叠率大于阈值，保存到分数列表中
            if max_overlap > self.overlap_thresh:
                if int(np.array(difficult[max_idx])) == 0:
                    if not visited[max_idx]:
                        self.class_score_poss[int(label)].append([score, 1.0])
                        visited[max_idx] = True
                    else:
                        self.class_score_poss[int(label)].append([score, 0.0])
            else:
                self.class_score_poss[int(label)].append([score, 0.0])

    def reset(self):
        # 初始化参数
        self.class_score_poss = [[] for _ in range(self.class_num)]
        self.class_gt_counts = [0] * self.class_num
        self.mAP = None

    def accumulate(self):
        # 计算mAP
        mAP = 0.
        valid_cnt = 0
        for score_pos, count in zip(self.class_score_poss, self.class_gt_counts):
            if count == 0 or len(score_pos) == 0:
                continue
            # 计算累积tp、fp
            accum_tp_list, accum_fp_list = self._get_tp_fp_accum(score_pos)
            precision = []
            recall = []
            # 计算precision、recall
            for ac_tp, ac_fp in zip(accum_tp_list, accum_fp_list):
                precision.append(float(ac_tp) / (ac_tp + ac_fp))
                recall.append(float(ac_tp) / count)
            max_precisions = [0.] * 11
            start_idx = len(precision) - 1
            for j in range(10, -1, -1):
                for i in range(start_idx, -1, -1):
                    if recall[i] < float(j) / 10.:
                        start_idx = i
                        if j > 0:
                            max_precisions[j - 1] = max_precisions[j]
                            break
                    else:
                        if max_precisions[j] < precision[i]:
                            max_precisions[j] = precision[i]
            mAP += sum(max_precisions) / 11.
            valid_cnt += 1
        # 计算mAP
        self.mAP = mAP / float(valid_cnt) if valid_cnt > 0 else mAP

    def get_map(self):
        # 获取mAP值
        if self.mAP is None:
            logger.error("mAP is not calculated.")
        return self.mAP

    def _get_tp_fp_accum(self, score_pos_list):
        # 计算累积tp、fp
        sorted_list = sorted(score_pos_list, key=lambda s: s[0], reverse=True)
        accum_tp = 0
        accum_fp = 0
        accum_tp_list = []
        accum_fp_list = []
        for (score, pos) in sorted_list:
            accum_tp += int(pos)
            accum_tp_list.append(accum_tp)
            accum_fp += 1 - int(pos)
            accum_fp_list.append(accum_fp)
        return accum_tp_list, accum_fp_list


# In[39]:


VOC_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# 计算mAP值并打印结果
def mAP(results, annotations_dir='./VOC/test/annotations/xmls'):
    cname2cid = {}
    for i, item in enumerate(VOC_names):
        cname2cid[item] = i
    num_classes = len(VOC_names)
    # 声明 DetectionMAP 实例，用于计算mAP
    detection_map = DetectionMAP(class_num=num_classes, overlap_thresh=0.5, is_bbox_normalized=False)
    # 遍历所有的预测结果
    for result in results:
        # 获取图片名称
        image_name = int(result[0])
        # 获取预测框列表
        bboxes = np.array(result[1]).astype('float32')
        # 获取标注文件
        anno_file = os.path.join(annotations_dir, str(image_name).zfill(6) + '.xml')
        # 解析标注文件
        tree = ET.parse(anno_file)
        objs = tree.findall('object')
        im_w = float(tree.find('size').find('width').text)
        im_h = float(tree.find('size').find('height').text)
        gt_bbox = np.zeros((len(objs), 4), dtype=np.float32)
        gt_class = np.zeros((len(objs), 1), dtype=np.int32)
        difficult = np.zeros((len(objs), 1), dtype=np.int32)
        for i, obj in enumerate(objs):
            cname = obj.find('name').text
            gt_class[i][0] = cname2cid[cname]
            _difficult = int(obj.find('difficult').text)
            x1 = float(obj.find('bndbox').find('xmin').text)
            y1 = float(obj.find('bndbox').find('ymin').text)
            x2 = float(obj.find('bndbox').find('xmax').text)
            y2 = float(obj.find('bndbox').find('ymax').text)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(im_w - 1, x2)
            y2 = min(im_h - 1, y2)
            gt_bbox[i] = [x1, y1, x2, y2]
            difficult[i][0] = _difficult
        detection_map.update(bboxes, gt_bbox, gt_class, difficult)

    print("Accumulating evaluatation results...")
    # 计算mAP
    detection_map.accumulate()
    # 获取最终mAP
    map_stat = 100. * detection_map.get_map()
    print("mAP(0.5, 11point) = {:.2f}".format(map_stat))


# 5）配置全局变量

# In[40]:


# 锚框尺寸
ANCHORS = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
# 锚框掩膜，用于指定哪些锚框用于哪些层级
ANCHOR_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
# 区分正负样本的IOU阈值
IGNORE_THRESH = .7
# 训练轮数
MAX_EPOCH = 20
# 模型验证时的得分阈值
VALID_THRESH = 0.01
# NMS保留的最大预测框个数
NMS_POSK = 100
# NMS阈值
NMS_THRESH = 0.45


# ## 2.4 模型训练
# 
# 训练模型并调整参数的过程，观察模型学习的过程是否正常，如损失函数值是否在降低。由于资源有限，在本实验中，模型总共训练20轮。其中，每隔100个批次会在训练集上计算loss并打印；每隔10轮会在验证集上计算mAP并打印，同时保存模型参数。

# In[39]:


# 设置运行设备
# 开启0号GPU
use_gpu = True
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

# 将模型调整为训练状态
model.train()

# 模型训练
for epoch in range(MAX_EPOCH):
    for i, data in enumerate(train_loader()):
        img, gt_boxes, gt_labels, img_scale = data
        gt_scores = np.ones(gt_labels.shape).astype('float32')
        gt_scores = paddle.to_tensor(gt_scores)
        img = paddle.to_tensor(img)
        gt_boxes = paddle.to_tensor(gt_boxes)
        gt_labels = paddle.to_tensor(gt_labels)
        # 前向传播，输出[P0, P1, P2]
        outputs = model(img)  
        # 计算损失函数
        loss = model.get_loss(outputs, gt_boxes, gt_labels, gtscore=gt_scores, anchors = ANCHORS, anchor_masks = ANCHOR_MASKS, ignore_thresh=IGNORE_THRESH, use_label_smooth=False)
        # 反向传播计算梯度
        loss.backward() 
        # 更新参数   
        opt.step()  
        opt.clear_grad()
        if i % 100 == 0:
            print('[TRAIN]epoch {}, iter {}, output loss: {}'.format(epoch, i, loss.numpy()))

    if (epoch+1) % 10 == 0:
        # 将模型调整为测试状态
        model.eval()
        total_results = []
        for i, data in enumerate(valid_loader()):
            img_name, img_data, img_scale_data = data
            img = paddle.to_tensor(img_data)
            img_scale = paddle.to_tensor(img_scale_data)
            # 前向传播，输出[P0, P1, P2]
            outputs = model(img)
            # 计算预测框和得分
            bboxes, scores = model.get_pred(outputs, im_shape=img_scale, anchors=ANCHORS, anchor_masks=ANCHOR_MASKS, valid_thresh = VALID_THRESH)
            bboxes_data = bboxes.numpy()
            scores_data = scores.numpy()
            # 多分类非极大值抑制
            result = multiclass_nms(bboxes_data, scores_data, score_thresh=VALID_THRESH, nms_thresh=NMS_THRESH, pos_nms_topk=NMS_POSK)
            # 汇总所有预测结果
            for j in range(len(result)):
                result_j = result[j]
                img_name_j = img_name[j]
                if len(result_j) == 0:
                    total_results.append([img_name_j, result_j])
                else:
                    total_results.append([img_name_j, result_j.tolist()])
        # 计算mAP值并打印结果
        mAP(total_results, annotations_dir='./VOC/test/annotations/xmls')

        # 保存模型参数
        paddle.save(model.state_dict(), 'yolo_epoch{}.pdparams'.format(epoch))
        paddle.save(opt.state_dict(), 'yolo_epoch{}.pdopt'.format(epoch))

        model.train()


# ## 2.5 模型保存
# 
# 训练完成后，可以将模型参数保存到磁盘，用于模型推理或继续训练。

# In[40]:


# 保存模型参数
paddle.save(model.state_dict(), 'last_model.pdparams')
# 保存优化器信息和相关参数，方便继续训练
paddle.save(opt.state_dict(), 'last_model.pdopt')


# ## 2.6 模型评估
# 
# 使用保存的模型参数评估在验证集上的准确率，代码实现如下：

# In[41]:


# 创建模型
model = YOLOv3(num_classes=NUM_CLASSES)
# 指定模型路径
params_file_path = '/home/aistudio/work/last_model.pdparams'
# 加载模型权重
model_state_dict = paddle.load(params_file_path)
model.load_dict(model_state_dict)

total_results = []
# 将模型调整为测试状态
model.eval()
for i, data in enumerate(valid_loader()):
    img_name, img_data, img_scale_data = data
    img = paddle.to_tensor(img_data)
    img_scale = paddle.to_tensor(img_scale_data)
    # 前向传播，输出[P0, P1, P2]
    outputs = model.forward(img)
    # 计算预测框和得分
    bboxes, scores = model.get_pred(outputs, im_shape=img_scale, anchors=ANCHORS, anchor_masks=ANCHOR_MASKS, valid_thresh = VALID_THRESH)
    bboxes_data = bboxes.numpy()
    scores_data = scores.numpy()
    # 多分类非极大值抑制
    result = multiclass_nms(bboxes_data, scores_data, score_thresh=VALID_THRESH, nms_thresh=NMS_THRESH, pos_nms_topk=NMS_POSK)
    # 汇总所有预测结果
    for j in range(len(result)):
        result_j = result[j]
        img_name_j = img_name[j]
        if len(result_j) == 0:
            total_results.append([img_name_j, result_j])
        else:
            total_results.append([img_name_j, result_j.tolist()])
# 计算mAP值并打印结果
mAP(total_results, annotations_dir='./VOC/test/annotations/xmls')


# ## 2.7 模型推理及可视化
# 
# 同样地，也可以使用保存好的模型，对数据集中的某一张图片进行模型推理，观察模型效果。考虑到上边模型训练轮数较少，难以达到较好的结果，为大家提供了预先训练了200轮的模型进行推理和可视化。具体代码实现如下：
# 
# 1. 创建数据读取器以读取单张图片的数据

# In[42]:


# 将 list形式的batch数据 转化成多个array构成的tuple
def make_test_array(batch_data):
    img_name_array = np.array([item[0] for item in batch_data])
    img_data_array = np.array(
        [item[1] for item in batch_data], dtype='float32')
    img_scale_array = np.array([item[2] for item in batch_data], dtype='int32')
    return img_name_array, img_data_array, img_scale_array


# In[43]:


# 读取单张测试图片
def single_image_data_loader(filename, test_image_size=608, mode='test'):
    """
    加载测试用的图片，测试数据没有groundtruth标签
    """
    batch_size= 1
    def reader():
        batch_data = []
        img_size = test_image_size
        file_path = os.path.join(filename)
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H = img.shape[0]
        W = img.shape[1]
        img = cv2.resize(img, (img_size, img_size))

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = np.array(mean).reshape((1, 1, -1))
        std = np.array(std).reshape((1, 1, -1))
        out_img = (img / 255.0 - mean) / std
        out_img = out_img.astype('float32').transpose((2, 0, 1))
        img = out_img 
        im_shape = [H, W]

        batch_data.append((image_name.split('.')[0], img, im_shape))
        if len(batch_data) == batch_size:
            yield make_test_array(batch_data)
            batch_data = []

    return reader


# 2. 定义绘制预测框的画图函数，代码如下。

# In[44]:


# 定义画图函数
VOC_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# 定义画矩形框的函数 
def draw_rectangle(currentAxis, bbox, edgecolor = 'k', facecolor = 'y', fill=False, linestyle='-'):
    rect=patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1, linewidth=1, edgecolor=edgecolor,facecolor=facecolor,fill=fill, linestyle=linestyle)
    currentAxis.add_patch(rect)

# 定义绘制预测结果的函数
def draw_results(result, filename, draw_thresh=0.5):
    plt.figure(figsize=(10, 10))
    im = imread(filename)
    plt.imshow(im)
    currentAxis=plt.gca()
    colors = ['r', 'g', 'b', 'k', 'y', 'c', 'purple']
    for item in result:
        box = item[2:6]
        label = int(item[0])
        name = VOC_NAMES[label]
        if item[1] > draw_thresh:
            draw_rectangle(currentAxis, box, edgecolor = colors[label%7])
            plt.text(box[0], box[1], name, fontsize=12, color=colors[label%7])


# 3. 使用上面定义的single_image_data_loader函数读取指定的图片，输入网络并计算出预测框和得分，然后使用多分类非极大值抑制消除冗余的框。将最终结果画图展示出来。

# In[46]:


image_name = '/home/aistudio/work/VOC/test/images/002007.jpg'
params_file_path = '/home/aistudio/work/last_model.pdparams'

model = YOLOv3(num_classes=NUM_CLASSES)
model_state_dict = paddle.load(params_file_path)
model.load_dict(model_state_dict)
model.eval()

total_results = []
test_loader = single_image_data_loader(image_name, mode='test')
for i, data in enumerate(test_loader()):
    img_name, img_data, img_scale_data = data
    img = paddle.to_tensor(img_data)
    img_scale = paddle.to_tensor(img_scale_data)

    outputs = model(img)
    bboxes, scores = model.get_pred(outputs, im_shape=img_scale, anchors=ANCHORS, anchor_masks=ANCHOR_MASKS, valid_thresh = VALID_THRESH)

    bboxes_data = bboxes.numpy()
    scores_data = scores.numpy()
    results = multiclass_nms(bboxes_data, scores_data, score_thresh=VALID_THRESH, nms_thresh=NMS_THRESH, pos_nms_topk=NMS_POSK)

result = results[0]
draw_results(result, image_name, draw_thresh=0.5)


# **拓展思考：**
# 
# 如果想预测VOC2007 Dataset中另外一张图片的类别，只需在“VOC/test/images/”路径下，将任意一个图片名称赋值给`image_name`，然后重新执行“模型推理”代码即可。可以观察下可视化的结果有何不同？

# # 3. 实验总结
# 
# 本次实验使用飞桨框架构建了经典的YOLOv3目标检测网络，并在VOC2007 Dataset上实现了目标检测任务。通过本次实验，不但掌握了《人工智能导论：模型与算法》- 6.6深度学习在自然语言和计算机视觉上的应用（P238-P243）中介绍的相关原理的实践方法，还熟悉了通过开源框架实现目标检测任务的实验流程和代码实现。大家可以在此实验的基础上，尝试开发自己感兴趣的目标检测任务。

# # 4. 实验拓展
# 
# * 尝试加载ImageNet上的预训练权重、调整模型优化策略和训练轮数等超参数，观察是否能够得到更高的指标；
# * 尝试使用其他算法（Faster RCNN等）实现目标检测任务。
