#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from IPython.display import display
# from PIL import Image

# # 读取图片文件
# img = Image.open('/home/aistudio/work/cat_12_train/5nKsehtjrXCZqbAcSW13gxB8E6z2Luy7.jpg')
#                                                 #    5nKsehtjrXCZqbAcSW13gxB8E6z2Luy7
# # 获取图片信息
# print('格式:', img.format)
# print('大小:', img.size)
# print('模式:', img.mode)
# # 显示图片
# display(img)

# img = Image.open('/home/aistudio/work/cat_12_train/5Kb2qj1pru6cLNPWReYT3vtEMFzi87IA.jpg')
# # 获取图片信息
# print('格式:', img.format)
# print('大小:', img.size)
# print('模式:', img.mode)
# # 显示图片
# display(img)


# In[2]:


# import numpy as np
# from PIL import Image

# # 生成随机的 RGB 图像
# image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

# # 创建 PIL Image 对象
# pil_image = Image.fromarray(image)

# # 显示图像
# pil_image.show()


# In[3]:


# # 2.1 计算图像尺寸大小
# from PIL import Image
# import os

# image_dir = './work/cat_12_predict'  # 图片目录
# image_names = os.listdir(image_dir)[:]  # 获取前100张图片的文件名

# width_sum = 0  # 宽度总和
# height_sum = 0  # 高度总和
# for name in image_names:
#     # 打开图片并获取尺寸大小
#     with Image.open(os.path.join(image_dir, name)) as img:
#         width, height = img.size
#         # print('宽度：{}, 高度：{}'.format(width, height))
#         width_sum += width
#         height_sum += height

# # 计算平均尺寸
# print(len(image_names))
# avg_width = width_sum / len(image_names)
# avg_height = height_sum / len(image_names)
# print('平均宽度：', avg_width)
# print('平均高度：', avg_height)

# # train:2160
# # 平均宽度： 438.08842592592595 438
# # 平均高度： 388.6138888888889  389

# # predict：240
# # 平均宽度： 435.44166666666666
# # 平均高度： 417.5708333333333


# In[4]:


# # 23.5.9 升级paddle版本
# ! pip uninstall paddlepaddle -y
# ! pip uninstall paddlehub -y

# ! pip install paddlepaddle
# ! pip install paddlehub


# In[5]:


# !unzip -q -d /home/aistudio/work /home/aistudio/data/data10954/cat_12_test.zip
# !unzip -q -d /home/aistudio/work /home/aistudio/data/data10954/cat_12_train.zip


# In[6]:


# 23.5.9 数据文件清洗。在transform时发现一些cv2.readim()错误的文件，导致训练挂死，需要转换清洗
# 5.9 根据GPT思路，改进如下
import os
import cv2
from PIL import Image

# 执行一遍再次执行，可以发现已经没有cv2.imread为空的文件了，清洗完成
directory = "/home/aistudio/work"
for subdir in ["cat_12_predict", "cat_12_train"]:
    subdir_path = os.path.join(directory, subdir)
    for filename in os.listdir(subdir_path):
        filepath = os.path.join(subdir_path, filename)
        # 23.5.9 遇到可能的cv2.imread问题的图像文件，先进行转换，避免transform的时候出错
        if cv2.imread(filepath) is None:
            print(filepath)
            img = Image.open(filepath).convert('RGB')
            # 23.5.9 先完全遍历并变换格式，让cv2可以识别
            img.save(filepath)
            print('finish saving')


# In[7]:


# 23.5.14 划分已标记数据的集合
# 23.6.7 训练集提升到.95，去掉test集，分数提升到0.92083，目前最高分
# 按比例随机切割数据集
# import os
import random

# 读取原始的训练数据列表
with open('/home/aistudio/data/data10954/train_list.txt', 'r') as f:
    lines = f.readlines()

# 打乱数据集
random.shuffle(lines)

# 计算划分的数量
num_train = int(len(lines) * 0.95)
num_validate = int(len(lines) * 0.05)
num_test = len(lines) - num_train - num_validate

# 划分数据集
train_lines = lines[:num_train]
validate_lines = lines[num_train:num_train+num_validate]
test_lines = lines[num_train+num_validate:]

# 将数据集保存到文件
with open('/home/aistudio/work/train_list.txt', 'w') as f:
    f.writelines(train_lines)

with open('/home/aistudio/work/validate_list.txt', 'w') as f:
    f.writelines(validate_lines)

with open('/home/aistudio/work/test_list.txt', 'w') as f:
    f.writelines(test_lines)

# 复制原始的训练数据列表到 /work 目录下
# os.system('cp /home/aistudio/data/data10954/train_list.txt /work')




# 23.5.9 定义数据集类，加载训练的数据和标签
# 23.5.14 桃子分类的案例中，train,val,test三个数据集的transform居然是同一个，显然不合理，太简单粗暴
import paddle
import paddlehub as hub
import os

class DemoDataset(paddle.io.Dataset):
    def __init__(self, transforms, num_classes=12, mode='train'):	
        # 数据集存放位置
        self.dataset_dir = "/home/aistudio/work"  #dataset_dir为数据集实际路径，需要填写全路径
        self.transforms = transforms
        self.num_classes = num_classes
        self.mode = mode

        if self.mode == 'train':
            self.file = 'train_list.txt'
            print('train_list: ')
        elif self.mode == 'test':
            self.file = 'test_list.txt'
            print('test_list: ')
        else:
            self.file = 'validate_list.txt'
            print('validate_list: ')
        
        self.file = os.path.join(self.dataset_dir , self.file)
        self.data = []
        
        # 23.5.8 单行line示例：cat_12_train/twJmVosEvIGg9BhkFUdYHxprz8yWcM2C.jpg       1
        with open(self.file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line != '':
                    self.data.append(line)
        print('list size:', len(self.data))
            
    def __getitem__(self, idx):
        # 23.5.8 这个函数从“桃子”分类移植，split符合要修改为'\t
        img_path, grt = self.data[idx].split('\t')
        img_path = os.path.join(self.dataset_dir, img_path)
        im = self.transforms(img_path)
        # 23.5.14 如果是val和test数据，做完归一化可以直接返回，但train数据集必须增强
        # 原桃子分类hub案例代码，三集同样的transform，太简单粗暴了
        # 23.5.14 通过分析，还是使用hub的曾广方法，在本类外层区分transform
        # if self.mode == 'train':
        #     # 对train数据集进行增强处理
        #     print('train_list: ')
        return im, int(grt)

    def __len__(self):
        return len(self.data)

# 第二部分：数据预处理
'''将训练数据输入模型之前，我们通常还需要对原始数据做一些数据处理的工作，比如数据格式的规范化处理，或增加一些数据增强策略。
构建图像分类模型的数据读取器，负责将桃子dataset的数据进行预处理，以特定格式组织并输入给模型进行训练。
如下数据处理策略，只做了三种操作：
1.指定输入图片的尺寸，并将所有样本数据统一处理成该尺寸。
2.对输入图像进行裁剪，并且保持图片中心点不变。
3.对所有输入图片数据进行归一化处理。

对数据预处理及加载数据集的示例如下：'''
# 23.5.14 hub的transforms只做简单归一化
import paddlehub.vision.transforms as T
# 23.5.14 paddle.vision.transforms版本的增强能力更好
# import paddle.vision.transforms as T       # 数据增强

# 23.5.8 以下是桃子分类的预处理，这里需要根据猫的图像数据改进
# transforms = T.Compose(
#         [T.Resize((256, 256)),
#          T.CenterCrop(224),
#          T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
#         to_rgb=True)


# 23.5.8 遇到一张图片格式: GIF 大小: (216, 188) 模式: P，导致im = self.transforms(img_path)挂死
# 要提升程序的健壮性
# 23.5.8 GPT推荐的增强方式
# transforms = T.Compose([
#     T.Resize((400, 400)),
#     T.RandomHorizontalFlip(),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
#     to_rgb=True)

# 23.5.9 为了提升准确率，考虑增加数据增强的方法
# 23.5.14 继续考虑增强数据，提升准确率。
# transforms = T.Compose(
#         [T.Resize((256, 256)),
#          T.RandomCrop(224, pad_if_needed=True),
#          T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
#          T.RandomHorizontalFlip(),
#          T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# 23.5.14 把增强移植到get_item中，这里只做归一化，即使是test集为了predict也需要统一格式
transforms = T.Compose(
        [T.Resize((224, 224)),
        #  T.CenterCrop(224),
         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
        to_rgb=True)
# 23.5.14 对train数据集进行“图像数据增广”，具体如下
'''
在PaddleHub中，常用有效的图像数据增广方法包括：
随机翻转：使用RandomFlip类对图像进行随机水平或垂直翻转，增加数据的多样性。
随机剪裁：使用RandomCrop类对图像进行随机剪裁，可以获得不同尺寸的图像。
随机缩放：使用RandomResize类对图像进行随机缩放，可以增加数据的多样性，同时可以获得不同尺寸的图像。
随机扭曲：使用RandomDistort类对图像进行随机扭曲，包括随机调整亮度、对比度、饱和度和色相等。
随机模糊：使用RandomBlur类对图像进行随机模糊，可以模拟真实场景中的模糊效果，同时可以增加数据的多样性。
随机旋转：使用RandomRotation类对图像进行随机旋转，可以增加数据的多样性，同时可以获得不同角度的图像。
随机裁剪并resize：使用RandomCrop和Resize类进行组合，实现先随机剪裁再resize的操作，可以获得不同尺寸的图像。
通过组合这些方法，可以获得更多多样的图像数据，并增加模型的泛化能力。
'''

import cv2
import numpy as np

# 23.6.13 新增随机宽高比
import random

class RandomAspectRatio(object):
    """
    Randomly adjust the aspect ratio of the input image.

    Args:
        aspect_ratio_range (tuple): A tuple (min_ratio, max_ratio) specifying the range of aspect ratios.
    """

    def __init__(self, aspect_ratio_range=(0.5, 2.0)):
        self.aspect_ratio_range = aspect_ratio_range

    def __call__(self, img):
        # if random.random() < 0.5:
        # Randomly select an aspect ratio within the specified range
        min_ratio, max_ratio = self.aspect_ratio_range
        aspect_ratio = random.uniform(min_ratio, max_ratio)
        # print(f'in RandomAspectRatio: {aspect_ratio}' )

        # Resize the image by adjusting the width and height based on the aspect ratio
        width = img.shape[1] # 保持宽度不变
        height = int(width / aspect_ratio)
        img = cv2.resize(img, (width, height))
        # print(f'width:{width}, height:{height}')

        # 6.14 采用在形变函数中就补成正方形
        long_side = max(width, height)
        short_side = min(width, height)
        # Calculate the padding size to make the image a square
        pad_size = long_side - short_side
        pad_before = pad_size // 2
        pad_after = pad_size - pad_before
        # Pad the image to a square shape
        if width > height:
            img = cv2.copyMakeBorder(img, pad_before, pad_after, 0, 0, cv2.BORDER_CONSTANT)
        else:
            img = cv2.copyMakeBorder(img, 0, 0, pad_before, pad_after, cv2.BORDER_CONSTANT)

        return img


# 23.6.6 新增转黑白的增强，这个措施使成绩从89提升到91
def convert_to_gray(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    return gray_image

class RandomGray(object):
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = convert_to_gray(img)
        return img

# 23.6.7 新增黑色遮挡块，从0.92083提升到0.925
class RandomBlackRectanglePaste:
    def __init__(self, min_width=10, max_width=50, min_height=10, max_height=50, paste_prob=0.5):
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        self.paste_prob = paste_prob

    def __call__(self, image):
        if random.random() < self.paste_prob:
            # 创建一个与输入图像大小相同的空白图像
            result_image = np.copy(image)

            # 随机生成要拼贴的长方形块的宽度和高度
            width = random.randint(self.min_width, self.max_width)
            height = random.randint(self.min_height, self.max_height)

            # 随机选择要将长方形块拼贴到结果图像的位置
            x = random.randint(0, image.shape[1] - width)
            y = random.randint(0, image.shape[0] - height)

            # 将黑色长方形块拼贴到结果图像中
            # 6.12 发现未识别的都有被青草绿植遮挡的情况
            result_image[y:y+height, x:x+width] = [15, 15, 15]  # 黑色像素的RGB值为[0, 0, 0]

            return result_image
        else:
            return image

class RandomGreenRectanglePaste:
    def __init__(self, min_width=10, max_width=50, min_height=10, max_height=50, paste_prob=0.5):
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        self.paste_prob = paste_prob

    def __call__(self, image):
        if random.random() < self.paste_prob:
            # 创建一个与输入图像大小相同的空白图像
            result_image = np.copy(image)

            # 随机生成要拼贴的长方形块的宽度和高度
            width = random.randint(self.min_width, self.max_width)
            height = random.randint(self.min_height, self.max_height)

            # 随机选择要将长方形块拼贴到结果图像的位置
            x = random.randint(0, image.shape[1] - width)
            y = random.randint(0, image.shape[0] - height)

            # 将黑色长方形块拼贴到结果图像中
            # 6.12 发现未识别的都有被青草绿植遮挡的情况
            result_image[y:y+height, x:x+width] = [56, 94, 15]  # 叶绿像素的RGB值为[0, 0, 0]

            return result_image
        else:
            return image

# 定义拼贴黑色长方形块的概率和大小范围
paste_prob = 0.5
min_width = 10
max_width = 150
min_height = 10
max_height = 50

# 6.13 增加压缩变形
target_width = np.random.randint(150, 300)

transforms_train = T.Compose(
        [
         T.Resize((256, 256)),
         RandomAspectRatio(aspect_ratio_range=(0.5, 2.0)), # 6.13 完成后width256，height256/(0.45--1)，height确定比256大
        #  T.CenterCrop(256),
         T.Resize((224, 224)),
        # 6.13 以上四步是经典的增加压缩变形的流程
        # 6.14 在函数RandomAspectRatio已经进行padding，原代码padding不好用
         T.RandomHorizontalFlip(0.5),
         T.RandomVerticalFlip(0.2),
        #  T.RandomRotation(35), 效果一般
         T.RandomDistort(brightness_prob=0.25, contrast_prob=0.25, saturation_prob=0.25, hue_prob=0.25),
         T.RandomBlur(0.1), #  反复和92分的resnet50对比，这个效果不好，去掉    
         RandomGray(p=0.3),  # 添加转换为黑白图像的方法，概率为0.3
         RandomBlackRectanglePaste(min_width=min_width, max_width=max_width, min_height=min_height, max_height=max_height, paste_prob=paste_prob),
         RandomGreenRectanglePaste(min_width=min_width, max_width=max_width, min_height=min_height, max_height=max_height, paste_prob=paste_prob),
         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ],
        to_rgb=True)

cat_train = DemoDataset(transforms_train)
cat_validate =  DemoDataset(transforms, mode='val')
cat_test =  DemoDataset(transforms, mode='test')


# In[8]:


# 6.12 进行数据增强的可视化，看看都进行了哪些变换
# 要进行反标准化操作
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# 6.13 取增强后的图像
img, lable = cat_train.__getitem__(32)

# 6.13 反标准化操作
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
img = img.transpose([1, 2, 0])
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # 6.13 貌似这句话加了反而错
img = img * np.array(std)[np.newaxis, np.newaxis, :] + np.array(mean)[np.newaxis, np.newaxis, :]
img = (img * 255).astype(np.uint8)
plt.imshow(img)

# 6.13 不进行标准化的话，可以直接用下面一句代码显示
# plt.imshow(img.astype(np.uint8).transpose([1, 2, 0]))

plt.show()


# In[9]:


# 第三部分：模型搭建
'''我们要在PaddleHub中选择合适的预训练模型来Fine-tune，由于桃子分类是一个图像分类任务，这里采用Resnet50模型，并且是采用ImageNet数据集预训练过的版本。这个预训练模型是在图像任务中的一个“万金油”模型，Resnet是目前较为有效的处理图像的网络结构，50层是一个精度和性能兼顾的选择，而ImageNet又是计算机视觉领域公开的最大的分类数据集。所以，在不清楚选择什么模型好的时候，可以优先以这个模型作为baseline。

使用PaddleHub加载ResNet50模型，十分简单，只需一行代码即可实现。关于更多预训练模型信息参见PaddleHub模型介绍
'''
#安装预训练模型
# !hub install resnet50_vd_imagenet_ssld==1.1.0
# !hub install resnet50_vd_imagenet_ssld
# ! hub install resnet50_vd_10w
# ! hub install resnet50_vd_wildanimals
# 5.11 换一个。ResNet-50是基于有标签数据集进行监督训练的，而resnet50_vd_imagenet_ssld使用自监督学习方法进行预训练
# !hub install resnet50_vd_imagenet
# 5.14 resnet50_vd_imagenet_ssld模型是在ImageNet上进行预训练的，虽然ImageNet包含了一些动物类别的图片，
# 但是它的主要目的是用于自然场景物体识别，对于动物分类任务可能效果不如在动物数据集上进行预训练的模型。如果要进行
# 动物分类，建议选择在动物数据集上预训练过的模型，比如使用在ImageNet和Places365等数据集上进行预训练的ResNet50_vd等模型。
# 5.14 resnet50_vd_animals：使用大规模的动物图像数据集（Animal-10）在ImageNet预训练得到的，适用于各种动物分类任务。
# EfficientNet系列模型：在ImageNet上预训练，同时使用了自动化网络结构搜索和网络缩放等技术，
# 拥有更高的精度和更少的参数量，适用于各种图像分类任务。
# 同时，您也可以根据实际需求选择不同的预训练模型，比如需要对特定的动物进行分类，可以选择针对该类动物预训练的模型。


# In[10]:


# 加载模型
import paddlehub as hub
# ! cp -r ~/work/se_hrnet64_imagenet_ssld /home/aistudio/.paddlehub/module # 避免线上速度太慢
# 23.5.8 这是绝对高级的API，用hub封装好的，已经是非常高层次的调用了。
# 23.6.12 继续试试 se_hrnet64_imagenet_ssld 终于又提高一分到93.33
# resnet50_vd_imagenet_ssld
model = hub.Module(name='se_hrnet64_imagenet_ssld',label_list=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"])

# 23.5.11 常规赛有一个baseline使用resnet50轻松做到acc91.25，这里也试一下看看
# model = paddle.vision.models.resnet50(pretrained=True, num_classes=12)

# 23.5.11 继续换
# model = hub.Module(name="resnet50_vd_imagenet", num_classes=12)
# 23.5.14 resnet50_vd_animals 
# model = hub.Module(name='resnet50_vd_animals')
# 23.5.14 好可惜啊，animal类模型不能finetune
# model = hub.Module(name='resnet50_vd_wildanimals')


# In[11]:


# 第四部分：模型训练
'''本案例中，我们使用Adam优化器，2014年12月，Kingma和Lei Ba提出了Adam优化器。该优化器对梯度的均值，即一阶矩估计（First Moment Estimation）和梯度的未中心化的方差，即二阶矩估计（Second Moment Estimation）进行综合计算，获得更新步长。Adam优化器实现起来较为简单，且计算效率高，需要的内存更少，梯度的伸缩变换不会影响更新梯度的过程， 超参数的可解释性强，且通常超参数无需调整或仅需微调等优点。我们将学习率设置为0.001，训练10个epochs。'''
from paddlehub.finetune.trainer import Trainer

import paddle

# 23.5.8 use_gpu=True在这个环境下老是出现如下问题：
# ValueError: The device should not be 'gpu', since PaddlePaddle is not compiled with CUDA
# 无法使用GPU导致训练时间非常长
# 23.5.8 在数据集上新建最新版的CodeLab，基于GPU训练，果然很快，3小时缩短到7分钟

# 23.5.14 以下2行是GPT对于hub的理解
# strategy = hub.AdamWeightDecayStrategy(weight_decay=0.01)
# optimizer = hub.AdamOptimizer(learning_rate=0.0001, strategy=strategy)
# 23.5.14 在切换模型至resnet50_vd_animals时，原代码的model.parameters()保持，说是animals模型没有这个属性
# print(model.parameters())

# 23.6.13 
# 学习率（learning_rate）：在你的代码中，学习率设置为0.000005。较小的学习率通常适用于训练较复杂的模型或较大的数据集。如果你的模型较小或数据集较小，
# 可能需要适当增加学习率，以加快模型的训练速度。
# 权重衰减系数（weight_decay）：你选择了0.000005作为权重衰减系数，这是一个较小的值。权重衰减用于控制模型参数的收缩，以防止过拟合。
# 较小的权重衰减系数通常适用于数据集较小、模型复杂度较低的情况。如果你的模型较复杂或数据集较大，可能需要适当增加权重衰减系数，以增强正则化效果。

# 23.5.14 
# 定义L2正则化
reg = paddle.regularizer.L2Decay(0.000005)
# # 定义优化器，并将正则化传递给weight_decay参数
# optimizer = paddle.optimizer.Adam(learning_rate=0.001, weight_decay=reg)

params = model.parameters()
# model.train() 
# params = model.trainable_parameters()

optimizer = paddle.optimizer.Adam(learning_rate=0.000005, weight_decay=reg, parameters=params)
# optimizer = paddle.optimizer.Adam(learning_rate=0.00001)
# 6.13 这个学习率对训练作用很大，0.001和0.0001训了半天loss和acc都很差，改为0.00001收敛快多了
trainer = Trainer(model, optimizer, checkpoint_dir='img_classification_ckpt', use_gpu=True) 
# 5.14 增加shuffle=True，执行报错，去GitHub上查了定义，shuffle集成到函数内部，默认全shuffle了
# PaddleHub/paddlehub/finetune/trainer L195:shuffle=True,


# In[ ]:


trainer.train(cat_train, epochs=30, batch_size=16, eval_dataset=cat_validate, save_interval=1)
# 5.11 出现错误AttributeError: 'ResNet' object has no attribute 'training_step'
# hub为什么反而还不如paddle？resnet50_vd_imagenet_ssld 咋还不如resnet50的原始模型？
# 5.14 现在可以回答上面的问题了，因为没有做数据增广，train、val、test全都同一个transform


# In[13]:


# 第五部分：模型评估
# 模型评估
trainer.evaluate(cat_validate, 16)

# 5.9 早上评估结果：{'loss': 1.1632198539045122, 'metrics': defaultdict(int, {'acc': 0.6})}
# 准确率太低，还需要想办法提升
# [2023-05-09 08:39:05,031] 再训10个epoch，[Evaluation result] avg_loss=1.1610 avg_acc=0.7222
# [2023-05-09 09:08:44,005]'acc': 0.7333333333333333
# [2023-05-09 09:57:54,150]'acc': 0.6055555555555555 堪忧。
# 5.10 保持住transform参数，训了30个epoch，准确率提升很明显，多轮还是有效果，又训了10轮，又回到0.66了
# 继续训练[2023-05-10 18:23:03,277][Evaluation result] avg_loss=0.9856 avg_acc=0.7611
# [2023-05-11 19:54:25,952][Evaluation result] avg_loss=0.8390 avg_acc=0.8333 50个epoch，
# 效果好了一点，但还是不如直接paddle的resnet50达到92.15
# [2023-05-11 20:53:24,170] [Evaluation result] avg_loss=0.6026 avg_acc=0.8389 学习率调为0.0001后
# 5.14 重新区分定义了train的transform函数，进行特定的图像数据增广，现在明白了之前的问题，train和test居然用了同样的transform
# [2023-05-14 10:42:46,165] [Evaluation result] avg_loss=0.3400 avg_acc=0.9259
# [2023-05-15 06:14:25,374] [Evaluation result] avg_loss=0.1824 avg_acc=0.9352
# [2023-05-16 13:03:16,497] [Evaluation result] avg_loss=0.1264 avg_acc=0.9630 目前最好
# [2023-05-17 11:36:28,287] [Evaluation result] avg_loss=0.0215 avg_acc=0.9907 去掉随机左右上下对称evaluate更好了
# 23.5.17 但是提交predict的成绩还只有90
# [2023-05-17 11:51:45,416] [Evaluation result] avg_loss=0.0154 avg_acc=1.0000 这是成精了，测试满分，但predict只有87
# 明显是过拟合了。
# [2023-05-17 12:24:46,407] [Evaluation result] avg_loss=0.2500 avg_acc=0.9074
# [2023-06-06 20:35:30,551] [Evaluation result] avg_loss=0.0041 avg_acc=1.0000 修改学习率1/10


# In[18]:


# # 第六部分：模型推理
# # 考试要求：考试提交，需要提交模型代码项目版本和结果文件。结果文件为CSV文件格式，命名为result.csv，
# # 文件内的字段需要按照指定格式写入。

# # 文件格式：WMgOhwZzacY023lCusqnBxIdibpkT5GP.jpg,0 其中，前半部分为【图片路径】，后半部分为【类别编号】，数据列以逗号分隔，每一行数据都以回车符结束。

# # 23.5.9 预测
# import os
# import csv
# import paddlehub as hub

# # 设置要预测的图片文件夹路径
# img_dir = '/home/aistudio/work/cat_12_predict'
# img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
# results = []
# # 23.5.18 train,val,test都需要做归一化，predict当然也要做
# for i in range(len(img_paths)):
#     im = transforms(img_paths[i])
#     result = model.predict(im)
#     print(result)
#     results.append(result)



# # print(max_key)  # 输出值最大的键


# # 创建 CSV 文件
# with open('result.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     for i in range(len(img_paths)):
#         file_name = os.path.basename(img_paths[i])
        
#         # 提取预测结果中的标签和概率
#         # 使用max函数和lambda表达式获取值最大的字典
#         max_dict = max(results[i], key=lambda x: list(x.values())[0])

#         # 获取字典中的键
#         predicted_label = list(max_dict.keys())[0]
#         # predicted_label = list(results[i].keys())[0]
#         print(file_name ,predicted_label)
#         # predicted_prob = list(result[0].values())[0]

#         # 将预测结果写入 CSV 文件
#         writer.writerow([file_name, predicted_label])


# In[224]:


# import paddlehub as hub

# 23.6.13 使用se_hrnet64_imagenet_ssld微调发现后面好的参数没有替代best，直接load epoch50试试
# 23.6.4 正确的load方法，load custom parameters success，load_checkpoint='/home/aistudio/img_classification_ckpt/best_model/model.pdparams'
# model = hub.Module(name='se_hrnet64_imagenet_ssld', 
#                    label_list=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
#                    load_checkpoint='/home/aistudio/img_classification_ckpt/epoch_50/model.pdparams')
# results = model.predict(data, max_seq_len=50, batch_size=1, use_gpu=False)


# In[14]:


import os
import csv
import paddlehub as hub

# 设置要预测的图片文件夹路径
img_dir = '/home/aistudio/work/cat_12_predict'
img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]

results = model.predict(img_paths)

# 创建 CSV 文件
with open('result.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(len(img_paths)):
        file_name = os.path.basename(img_paths[i])
        # 提取预测结果中的标签和概率
        predicted_label = list(results[i].keys())[0]
        predicted_prob = list(results[i].values())[0]
        # 将预测结果写入 CSV 文件
        writer.writerow([file_name, predicted_label, predicted_prob])

import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('result.csv', sep=',', header=None)
df.columns = ['file_name', 'predicted_label', 'predicted_prob']


# In[15]:


import matplotlib.pyplot as plt
import math
get_ipython().run_line_magic('matplotlib', 'inline')

# 筛选 predicted_prob < 0.7 的行
df_low = df[df['predicted_prob'] < 0.7]
data_list = df_low.values.tolist()
# 打印结果
print(df_low)
print(len(df_low))
# data_list = [['BGrEOpbnU5tIVwXqz32SHFujMLmTP079.jpg', 0, 0.64460254], ['GlozKf7Nt4126DudiTQHwM9gXF5RB3SP.jpg', 6, 0.5626435000000001]]
# 定义每行的图像数量
images_per_row = 2

# 计算总行数
num_rows = math.ceil(len(data_list) / images_per_row)

# 创建一个包含子图的画布
fig, axs = plt.subplots(num_rows, images_per_row, figsize=(12*images_per_row, 18*num_rows))

# 遍历图像数据并显示
for i, data in enumerate(data_list):
    # print(data)
    img_path = '/home/aistudio/work/cat_12_predict/'+data[0]
    label = str(data[1]) + ' prob:' + str(data[2])
    
    # 计算当前图像在子图中的位置
    row = i // images_per_row
    col = i % images_per_row
    
    # 读取并显示图像
    img = plt.imread(img_path)
    axs[row, col].imshow(img)
    axs[row, col].set_title(f"Label: {label}")
    axs[row, col].axis('off')

# 调整子图之间的间距
plt.tight_layout()

# 显示图像
plt.show()


# In[23]:


# 6.13 对比train集图片
import matplotlib.pyplot as plt
import math
get_ipython().run_line_magic('matplotlib', 'inline')

# 23.6.7 通过筛选作图
df_low = df[df['predicted_label'] == 0][:20]
data_list = df_low.values.tolist()
print(df_low)
print(len(df_low))

# 定义每行的图像数量
images_per_row = 2

# 计算总行数
num_rows = math.ceil(len(data_list) / images_per_row)

# 创建一个包含子图的画布
fig, axs = plt.subplots(num_rows, images_per_row, figsize=(12*images_per_row, 18*num_rows))

# 遍历图像数据并显示
for i, data in enumerate(data_list):
    # print(data)
    img_path = '/home/aistudio/work/cat_12_predict/'+data[0]
    label = str(data[1]) + ' prob:' + str(data[2])
    
    # 计算当前图像在子图中的位置
    row = i // images_per_row
    col = i % images_per_row
    
    # 读取并显示图像
    img = plt.imread(img_path)
    axs[row, col].imshow(img)
    axs[row, col].set_title(f"Label: {label}")
    axs[row, col].axis('off')

# 调整子图之间的间距
plt.tight_layout()

# 显示图像
plt.show()


# In[24]:


# # 23.5.20 抽取12个类型的照片用于显示
# # 定义要抽取的行数
# num_rows = 12

# # 存储抽取的数据
# extracted_data = {}

# # 打开标记文件并逐行读取数据
# with open('/home/aistudio/work/validate_list.txt', 'r') as file:
#     lines = file.readlines()
    
#     # 遍历每一行数据
#     for line in lines:
#         # 提取文件路径和标签
#         file_path, label = line.strip().split('\t')
        
#         # 如果该标签还没有被抽取过，则将该行数据添加到抽取的数据中
#         if label not in extracted_data:
#             extracted_data[label] = (file_path, label)
        
#         # 如果已经抽取了足够数量的数据，则结束循环
#         if len(extracted_data) == num_rows:
#             break

# # 打印抽取的数据
# # for label, (file_path, label) in extracted_data.items():
# #     print(f"文件路径：{file_path}，标签：{label}")
# extracted_data


# In[ ]:





# In[25]:


# # 23.6.4 测试了一下，太牛逼了，resnet50_vd_animals居然把cat12的名字早就区分开了
# # 但是肉眼可见，有些的置信度比较低，0.23这种可能精确度达不到
# import paddle
# import paddlehub as hub
# import cv2
# import os

# classifier = hub.Module(name="resnet50_vd_animals")
# train_path = '/home/aistudio/work/'
# for key in extracted_data:
#     # print(extracted_data[data][0])
#     jpg_name = os.path.join(train_path, extracted_data[key][0])
#     print(jpg_name)
#     result = classifier.classification(images=[cv2.imread(jpg_name)])
#     print(result)


# In[26]:


# # 23.6.4 用resnet50_vd_animals来预测一下，再上传答案看看精度，不用上传了，过滤了一下概率0.8,202个不合格
# import os
# import cv2
# import csv
# import paddlehub as hub

# # 设置要预测的图片文件夹路径
# img_dir = '/home/aistudio/work/cat_12_predict'
# img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
# cv2_imgs = [cv2.imread(img) for img in img_paths]

# classifier = hub.Module(name="resnet50_vd_animals")
# results = classifier.classification(images=cv2_imgs)

# print(results[0])

# possible_miss=[]
# # 创建 CSV 文件
# with open('result.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     for i in range(len(img_paths)):
#         file_name = os.path.basename(img_paths[i])
        
#         # 提取预测结果中的标签和概率
#         predicted_label = list(results[i].keys())[0]
#         predicted_prob = list(results[i].values())[0]
#         if list(results[i].values())[0] < 0.8:
#             possible_miss.append([file_name, predicted_label, predicted_prob])

#         # 将预测结果写入 CSV 文件
#         writer.writerow([file_name, predicted_label, predicted_prob])


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
