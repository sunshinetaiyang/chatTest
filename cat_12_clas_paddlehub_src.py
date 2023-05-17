#!/usr/bin/env python
# coding: utf-8

# In[4]:

# 23.5.13 本周关于paddlehub的观察总结：hub在GitHub已经半年没人维护了，
# 从cat_12的分类看，还不如直接用paddle调用resnet50的准确率高，封装的貌似并不好
# 23.5.14 看了李宏毅老师讲到的bias和variance，想起hub训练的bias还是很小的，模型是
# 没有问题的，那么问题就在于data的量太少了，还是在数据曾广上hub没有做好，这是一个
# 可以考虑的方向

#!/usr/bin/env python
# coding: utf-8

# In[4]:


from IPython.display import display
from PIL import Image

# 读取图片文件
img = Image.open('/home/aistudio/work/cat_12_train/5nKsehtjrXCZqbAcSW13gxB8E6z2Luy7.jpg')
                                                #    5nKsehtjrXCZqbAcSW13gxB8E6z2Luy7
# 获取图片信息
print('格式:', img.format)
print('大小:', img.size)
print('模式:', img.mode)
# 显示图片
display(img)

img = Image.open('/home/aistudio/work/cat_12_train/5Kb2qj1pru6cLNPWReYT3vtEMFzi87IA.jpg')
# 获取图片信息
print('格式:', img.format)
print('大小:', img.size)
print('模式:', img.mode)
# 显示图片
display(img)


# In[1]:


# 2.1 计算图像尺寸大小
from PIL import Image
import os

image_dir = './work/cat_12_train'  # 图片目录
image_names = os.listdir(image_dir)[:100]  # 获取前100张图片的文件名

width_sum = 0  # 宽度总和
height_sum = 0  # 高度总和
for name in image_names:
    # 打开图片并获取尺寸大小
    with Image.open(os.path.join(image_dir, name)) as img:
        width, height = img.size
        # print('宽度：{}, 高度：{}'.format(width, height))
        width_sum += width
        height_sum += height

# 计算平均尺寸
print(len(image_names))
avg_width = width_sum / len(image_names)
avg_height = height_sum / len(image_names)
print('平均宽度：', avg_width)
print('平均高度：', avg_height)


# In[2]:


# 23.5.9 升级paddle版本
get_ipython().system(' pip uninstall paddlepaddle -y')
get_ipython().system(' pip uninstall paddlehub -y')

get_ipython().system(' pip install paddlepaddle')
get_ipython().system(' pip install paddlehub')


# In[5]:


# !unzip -q -d /home/aistudio/work /home/aistudio/data/data10954/cat_12_test.zip
# !unzip -q -d /home/aistudio/work /home/aistudio/data/data10954/cat_12_train.zip


# In[2]:


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


# In[1]:


# 23.5.14 划分已标记数据的集合
# 按比例随机切割数据集
# import os
import random

# 读取原始的训练数据列表
with open('/home/aistudio/data/data10954/train_list.txt', 'r') as f:
    lines = f.readlines()

# 打乱数据集
random.shuffle(lines)

# 计算划分的数量
num_train = int(len(lines) * 0.9)
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



# In[2]:


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


# In[3]:


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
        # [T.Resize((256, 256)),
         # T.CenterCrop(224),
         # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
        # to_rgb=True)


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
        [T.Resize((256, 256)),
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
transforms_train = T.Compose(
        [
         T.RandomHorizontalFlip(),
         T.RandomVerticalFlip(),
         T.RandomRotation(),
         T.RandomDistort(brightness_prob=0.05, contrast_prob=0.05, saturation_prob=0.05, hue_prob=0.05),
        #  T.RandomBlur(0.05),   反复和92分的resnet50对比，这个效果不好，去掉    
         T.Resize((256, 256)),
        #  T.RandomPaddingCrop(224),
         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
        to_rgb=True)

cat_train = DemoDataset(transforms_train)
cat_validate =  DemoDataset(transforms, mode='val')
cat_test =  DemoDataset(transforms, mode='test')


# In[4]:


# 第三部分：模型搭建
'''我们要在PaddleHub中选择合适的预训练模型来Fine-tune，由于桃子分类是一个图像分类任务，这里采用Resnet50模型，并且是采用ImageNet数据集预训练过的版本。这个预训练模型是在图像任务中的一个“万金油”模型，Resnet是目前较为有效的处理图像的网络结构，50层是一个精度和性能兼顾的选择，而ImageNet又是计算机视觉领域公开的最大的分类数据集。所以，在不清楚选择什么模型好的时候，可以优先以这个模型作为baseline。

使用PaddleHub加载ResNet50模型，十分简单，只需一行代码即可实现。关于更多预训练模型信息参见PaddleHub模型介绍
'''
#安装预训练模型
# !hub install resnet50_vd_imagenet_ssld==1.1.0
get_ipython().system('hub install resnet50_vd_imagenet_ssld')
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


# In[4]:


# 加载模型
import paddlehub as hub
# 23.5.8 这是绝对高级的API，用hub封装好的，已经是非常高层次的调用了。
model = hub.Module(name='resnet50_vd_imagenet_ssld',label_list=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"])

# 23.5.11 常规赛有一个baseline使用resnet50轻松做到acc91.25，这里也试一下看看
# model = paddle.vision.models.resnet50(pretrained=True, num_classes=12)

# 23.5.11 继续换
# model = hub.Module(name="resnet50_vd_imagenet", num_classes=12)
# 23.5.14 resnet50_vd_animals 
# model = hub.Module(name='resnet50_vd_animals')
# 23.5.14 好可惜啊，animal类模型不能finetune
# model = hub.Module(name='resnet50_vd_wildanimals')


# In[5]:


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

# 23.5.14 
# 定义L2正则化
reg = paddle.regularizer.L2Decay(0.0001)
# # 定义优化器，并将正则化传递给weight_decay参数
# optimizer = paddle.optimizer.Adam(learning_rate=0.001, weight_decay=reg)

params = model.parameters()
# model.train() 
# params = model.trainable_parameters()

optimizer = paddle.optimizer.Adam(learning_rate=0.0001, weight_decay=reg, parameters=params)
# optimizer = paddle.optimizer.Adam(learning_rate=0.0001)
trainer = Trainer(model, optimizer, checkpoint_dir='img_classification_ckpt', use_gpu=True) 
# 5.14 增加shuffle=True，执行报错，去GitHub上查了定义，shuffle集成到函数内部，默认全shuffle了
# PaddleHub/paddlehub/finetune/trainer L195:shuffle=True,


# In[6]:


trainer.train(cat_train, epochs=10, batch_size=16, eval_dataset=cat_validate, save_interval=1)
# 5.11 出现错误AttributeError: 'ResNet' object has no attribute 'training_step'
# hub为什么反而还不如paddle？resnet50_vd_imagenet_ssld 咋还不如resnet50的原始模型？
# 5.14 现在可以回答上面的问题了，因为没有做数据增广，train、val、test全都同一个transform


# In[ ]:


# 23.5.14 继续再训20轮，观察eval效果
trainer.train(cat_train, epochs=20, batch_size=16, eval_dataset=cat_validate, save_interval=1)


# In[8]:


# 第五部分：模型评估
# 模型评估
trainer.evaluate(cat_test, 16)

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


# In[ ]:


# 第六部分：模型推理
# 考试要求：考试提交，需要提交模型代码项目版本和结果文件。结果文件为CSV文件格式，命名为result.csv，
# 文件内的字段需要按照指定格式写入。

# 文件格式：WMgOhwZzacY023lCusqnBxIdibpkT5GP.jpg,0 其中，前半部分为【图片路径】，后半部分为【类别编号】，数据列以逗号分隔，每一行数据都以回车符结束。

# 23.5.9 预测
import os
import csv
import paddlehub as hub

# 设置要预测的图片文件夹路径
img_dir = '/home/aistudio/work/cat_12_predict'
img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]

results = model.predict(img_paths)

# 创建 CSV 文件
with open('results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(len(img_paths)):
        file_name = os.path.basename(img_paths[i])
        
        # 提取预测结果中的标签和概率
        predicted_label = list(results[i].keys())[0]
        print(file_name ,predicted_label)
        # predicted_prob = list(result[0].values())[0]

        # 将预测结果写入 CSV 文件
        writer.writerow([file_name, predicted_label])


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# In[1]:


import paddlehub.vision.transforms as T
transforms = T.Compose([T.Resize((256, 256)), 
                        T.CenterCrop(224), 
                        T.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])], 
                        to_rgb=True)


# In[2]:


from paddlehub.datasets import Flowers
flowers = Flowers(transforms)
flowers_validate = Flowers(transforms, mode='val')


# In[3]:


import paddlehub as hub
model = hub.Module(name="resnet50_vd_imagenet_ssld", label_list=["roses", "tulips", "daisy", "sunflowers", "dandelion"])


# In[4]:


import paddle
from paddlehub.finetune.trainer import Trainer

optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
# trainer = Trainer(model, optimizer, checkpoint_dir='img_classification_ckpt')
trainer = Trainer(model, optimizer, checkpoint_dir='img_classification_ckpt', use_gpu=True)
trainer.train(flowers, epochs=10, batch_size=16, eval_dataset=flowers_validate, log_interval=10, save_interval=1)


# In[9]:


# model_predict = hub.Module(name='resnet50_vd_imagenet_ssld', label_list=["roses", "tulips", "daisy", "sunflowers", "dandelion"], load_checkpoint='img_classification_ckpt/epoch_9/model.pdparams')
result = model.predict(['flower.jpg'])
print(result)

# 23.5.13 这是用paddlehub，hub.Module(name="resnet50_vd_imagenet_ssld"的第二个案例，跑起来是不错，但精度就是上不到95，有些遗憾。


