#!/usr/bin/env python
# coding: utf-8

# # 猫猫可爱吗，今天我们就来撸他！使用残差网络对猫猫进行分类！
# 
# 赛题链接[猫十二分类新人赛](https://aistudio.baidu.com/aistudio/competition/detail/136/0/introduction)
# 难度：3(比一般猫狗分类要难，因为这个分类属于同一物种猫，和人脸识别处于同一级别的猫脸识别)
#     
# ![](https://ai-studio-static-online.cdn.bcebos.com/2c273609398c42b999f62caf9c9973fedf0d6e2bf56643b3ba4de69666578cb8)
# 
# 
# [baseline视频解析](https://aistudio.baidu.com/aistudio/competition/detail/136/0/related-material)

# # 1、首先导入包

# In[2]:


# 数据科学包
import random                      # 随机切分数据集
import numpy as np                 # 常用数据科学包
import os
from PIL import Image              # 图像读取
import matplotlib.pyplot as plt    # 代码中快速验证
import cv2                         # 图像包

# 深度学习包
import paddle
import paddle.vision.transforms as T       # 数据增强
from paddle.io import Dataset, DataLoader  # 定义数据集


# # 2、解压数据集，对数据集进行增广，并创建dataset

# In[3]:


# 解压数据集
if not os.path.exists('cat_12_train'):
    get_ipython().system('unzip data/data10954/cat_12_train.zip')
    get_ipython().system('unzip data/data10954/cat_12_test.zip')


# In[4]:


# ! cp data/data10954/train_list.txt work/


# In[4]:


# 按比例随机切割数据集
train_ratio = 0.9  # 训练集占0.9，验证集占0.1

train_paths, train_labels = [], []
valid_paths, valid_labels = [], []
with open('data/data10954/train_list.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if random.uniform(0, 1) < train_ratio:
            train_paths.append(line.split('	')[0])
            label = line.split('	')[1]
            train_labels.append(int(line.split('	')[1]))
        else:
            valid_paths.append(line.split('	')[0])
            valid_labels.append(int(line.split('	')[1]))


# In[5]:


# 定义训练数据集
class TrainData(Dataset):
    def __init__(self):
        super().__init__()
        # 5.11 这里进行了比hub更多的图像增强
        self.color_jitter = T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)
        self.normalize = T.Normalize(mean=0, std=1)
        self.random_crop = T.RandomCrop(224, pad_if_needed=True)
    
    def __getitem__(self, index):
        # 读取图片
        image_path = train_paths[index]

        image = np.array(Image.open(image_path))    # H, W, C
        try:
            # 5.11 由于深度学习框架通常要求输入的图片维度为(C, H, W)，因此需要对数组进行转置，
            # 使其变为(C, H, W)的形式。如果该图片只有一个通道，则需要将其复制3次，使其变为3通道的形式。
            image = image.transpose([2, 0, 1])[:3]  # C, H, W
        except:
            image = np.array([image, image, image]) # C, H, W
        
        # 图像增广
        features = self.color_jitter(image.transpose([1, 2, 0]))
        features = self.random_crop(features)
        features = self.normalize(features.transpose([2, 0, 1])).astype(np.float32)

        # 读取标签
        labels = train_labels[index]

        return features, labels
    
    def __len__(self):
        return len(train_paths)

    
# 定义验证数据集
class ValidData(Dataset):
    def __init__(self):
        super().__init__()
        self.normalize = T.Normalize(mean=0, std=1)
    
    def __getitem__(self, index):
        # 读取图片
        image_path = valid_paths[index]

        image = np.array(Image.open(image_path))    # H, W, C
        try:
            image = image.transpose([2, 0, 1])[:3]  # C, H, W
        except:
            image = np.array([image, image, image]) # C, H, W
        
        # 图像变换
        features = cv2.resize(image.transpose([1, 2, 0]), (256, 256)).transpose([2, 0, 1]).astype(np.float32)
        features = self.normalize(features)

        # 读取标签
        labels = valid_labels[index]

        return features, labels
    
    def __len__(self):
        return len(valid_paths)


# # 3、查看我们对猫猫进行的变换，从直觉上判断增广是否合理
# 

# In[6]:


train_data = TrainData()
img, labels = train_data.__getitem__(22)
plt.figure(dpi=40,figsize=(16,16))
plt.imshow(img.astype(np.uint8).transpose([1, 2, 0]))
plt.show()


# In[7]:


valid_data = ValidData()
img, label = valid_data.__getitem__(33)
plt.figure(dpi=40,figsize=(16,16))
plt.imshow(img.astype(np.uint8).transpose([1, 2, 0]))


# # 4、搭建网络模型，进行模型的训练与评估保存

# In[8]:


# 调用resnet50模型
paddle.vision.set_image_backend('cv2')
model = paddle.vision.models.resnet50(pretrained=True, num_classes=12)

# 定义数据迭代器
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=False)

# 定义优化器
opt = paddle.optimizer.Adam(learning_rate=1e-4, parameters=model.parameters(), weight_decay=paddle.regularizer.L2Decay(1e-4))

# 定义损失函数
loss_fn = paddle.nn.CrossEntropyLoss()

# 设置gpu环境
paddle.set_device('gpu:0')

# 整体训练流程
for epoch_id in range(15):
    model.train()
    for batch_id, data in enumerate(train_dataloader()):
        # 读取数据
        features, labels = data
        features = paddle.to_tensor(features)
        labels = paddle.to_tensor(labels)

        # 前向传播
        predicts = model(features)

        # 损失计算
        loss = loss_fn(predicts, labels)

        # 反向传播
        avg_loss = paddle.mean(loss)
        avg_loss.backward()

        # 更新
        opt.step()

        # 清零梯度
        opt.clear_grad()

        # 打印损失
        if batch_id % 2 == 0:
            print('epoch_id:{}, batch_id:{}, loss:{}'.format(epoch_id, batch_id, avg_loss.numpy()))
    model.eval()
    print('开始评估')
    i = 0
    acc = 0
    for image, label in valid_data:
        image = paddle.to_tensor([image])

        pre = list(np.array(model(image)[0]))
        max_item = max(pre)
        pre = pre.index(max_item)

        i += 1
        if pre == label:
            acc += 1
        if i % 10 == 0:
            print('精度：', acc / i)
    
    paddle.save(model.state_dict(), 'acc{}.model'.format(acc / i))


# # 5、加载模型进行预测，并提交结果
# 
# ## >> 直接提交submit.csv，得分0.9125！

# In[9]:


# 进行预测和提交
# 首先拿到预测文件的路径列表

def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
test_path = []
listdir('cat_12_test', test_path)

# 加载训练好的模型
pre_model = paddle.vision.models.resnet50(pretrained=True, num_classes=12)
pre_model.set_state_dict(paddle.load('acc0.9285714285714286.model'))
pre_model.eval()

pre_classes = []
normalize = T.Normalize(mean=0, std=1)
# 生成预测结果
for path in test_path:
    image_path = path

    image = np.array(Image.open(image_path))    # H, W, C
    try:
        image = image.transpose([2, 0, 1])[:3]  # C, H, W
    except:
        image = np.array([image, image, image]) # C, H, W
    
    # 图像变换
    features = cv2.resize(image.transpose([1, 2, 0]), (256, 256)).transpose([2, 0, 1]).astype(np.float32)
    features = normalize(features)

    features = paddle.to_tensor([features])
    pre = list(np.array(pre_model(features)[0]))
    # print(pre)
    max_item = max(pre)
    pre = pre.index(max_item)
    print("图片：", path, "预测结果：", pre)
    pre_classes.append(pre)

print(pre_classes)


# In[10]:


# 导入csv模块
import csv

# 1、创建文件对象
with open('result.csv', 'w', encoding='gbk', newline="") as f:
    # 2、基于文件对象构建csv写入对象
    csv_writer = csv.writer(f)
    for i in range(240):
        csv_writer.writerow([test_path[i].split('/')[1], pre_classes[i]])
    print('写入数据完成')


# 
# 
# 想要解锁更多技能？B站主页，求关注[一心炼银](https://space.bilibili.com/173706050)
