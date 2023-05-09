# 23.5.8 百度大脑和中山大学联合的近视眼PM数据集的分类检测
# 调用方法不如“桃子”案例，但也可以参考部分数据处理
# 这个案例没有用paddlehub，所以可用性不如hub那么简单直接
# ============================================================
# 图像分类示例 中山大学眼疾数据集示例
# 第一部分：数据集相关
# 1，解压数据集
# 如果已经解压过，不需要运行此段代码，否则由于文件已经存在，解压时会报错
!unzip -o -q -d /home/aistudio/work/palm /home/aistudio/data/data19065/training.zip
%cd /home/aistudio/work/palm/PALM-Training400/
!unzip -o -q PALM-Training400.zip
!unzip -o -q -d /home/aistudio/work/palm /home/aistudio/data/data19065/validation.zip
!unzip -o -q -d /home/aistudio/work/palm /home/aistudio/data/data19065/valid_gt.zip
#返回家目录，生成模型文件位于/home/aistudio/
%cd /home/aistudio/


# 2，数据预处理，调整图片大小：将每张图缩放到 224×224 大小，
# 归一化：将像素值调整到 [−1,1] 之间
import cv2
import numpy as np

# 对读入的图像数据进行预处理
def transform_img(img):
    # 将图片尺寸缩放道 224x224
    img = cv2.resize(img, (224, 224))
    # 读入的图像数据格式是[H, W, C]
    # 使用转置操作将其变成[C, H, W]
    img = np.transpose(img, (2,0,1))
    img = img.astype('float32')
    # 将数据范围调整到[-1.0, 1.0]之间
    img = img / 255.
    img = img * 2.0 - 1.0
    return img

# 3，定义数据读取器
import cv2
import random
import numpy as np
import os

# 定义训练集数据读取器
def data_loader(datadir, batch_size=10, mode = 'train'):
    # 将datadir目录下的文件列出来，每条文件都要读入
    filenames = os.listdir(datadir)
    def reader():
        if mode == 'train':
            # 训练时随机打乱数据顺序
            random.shuffle(filenames)
        batch_imgs = []
        batch_labels = []
        for name in filenames:
            filepath = os.path.join(datadir, name)
            img = cv2.imread(filepath)
            img = transform_img(img)
            if name[0] == 'H' or name[0] == 'N':
                # H开头的文件名表示高度近似，N开头的文件名表示正常视力
                # 高度近视和正常视力的样本，都不是病理性的，属于负样本，标签为0
                label = 0
            elif name[0] == 'P':
                # P开头的是病理性近视，属于正样本，标签为1
                label = 1
            else:
                raise('Not excepted file name')
            # 每读取一个样本的数据，就将其放入数据列表中
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                # 当数据列表的长度等于batch_size的时候，
                # 把这些数据当作一个mini-batch，并作为数据生成器的一个输出
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).reshape(-1, 1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            # 剩余样本数目不足一个batch_size的数据，一起打包成一个mini-batch
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).reshape(-1, 1)
            yield imgs_array, labels_array

    return reader

# 定义验证集数据读取器
def valid_data_loader(datadir, csvfile, batch_size=10, mode='valid'):
    # 训练集读取时通过文件名来确定样本标签，验证集则通过csvfile来读取每个图片对应的标签
    # 请查看解压后的验证集标签数据，观察csvfile文件里面所包含的内容
    # csvfile文件所包含的内容格式如下，每一行代表一个样本，
    # 其中第一列是图片id，第二列是文件名，第三列是图片标签，
    # 第四列和第五列是Fovea的坐标，与分类任务无关
    # ID,imgName,Label,Fovea_X,Fovea_Y
    # 1,V0001.jpg,0,1157.74,1019.87
    # 2,V0002.jpg,1,1285.82,1080.47
    # 打开包含验证集标签的csvfile，并读入其中的内容
    filelists = open(csvfile).readlines()
    def reader():
        batch_imgs = []
        batch_labels = []
        for line in filelists[1:]:
            line = line.strip().split(',')
            name = line[1]
            label = int(line[2])
            # 根据图片文件名加载图片，并对图像数据作预处理
            filepath = os.path.join(datadir, name)
            img = cv2.imread(filepath)
            img = transform_img(img)
            # 每读取一个样本的数据，就将其放入数据列表中
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                # 当数据列表的长度等于batch_size的时候，
                # 把这些数据当作一个mini-batch，并作为数据生成器的一个输出
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).reshape(-1, 1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            # 剩余样本数目不足一个batch_size的数据，一起打包成一个mini-batch
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).reshape(-1, 1)
            yield imgs_array, labels_array

    return reader

# 第二部分，模型构建。
# 1，模型，直接使用飞桨高层API中的Resnet50进行图像分类实验
from paddle.vision.models import resnet50
model = resnet50()

# 2，损失函数
import paddle.nn.functional as F
loss_fn = F.cross_entropy

# 第三部分，模型训练
# 使用交叉熵损失函数，并用SGD作为优化器来训练ResNet网络
# -*- coding: utf-8 -*-
# LeNet 识别眼疾图片
import os
import random
import paddle
import numpy as np

class Runner(object):
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
        # 记录全局最优指标
        self.best_acc = 0
    
    # 定义训练过程
    def train_pm(self, train_datadir, val_datadir, **kwargs):
        print('start training ... ')
        self.model.train()
        
        num_epochs = kwargs.get('num_epochs', 0)
        csv_file = kwargs.get('csv_file', 0)
        save_path = kwargs.get("save_path", "/home/aistudio/output/")

        # 定义数据读取器，训练数据读取器
        train_loader = data_loader(train_datadir, batch_size=10, mode='train')
        
        for epoch in range(num_epochs):
            for batch_id, data in enumerate(train_loader()):
                x_data, y_data = data
                img = paddle.to_tensor(x_data)
                label = paddle.to_tensor(y_data)
                # 运行模型前向计算，得到预测值
                logits = model(img) 
                avg_loss = self.loss_fn(logits, label)
                
                if batch_id % 20 == 0:
                    print("epoch: {}, batch_id: {}, loss is: {:.4f}".format(epoch, batch_id, float(avg_loss.numpy())))
                # 反向传播，更新权重，清除梯度
                avg_loss.backward()
                self.optimizer.step()
                self.optimizer.clear_grad()
            
            acc = self.evaluate_pm(val_datadir, csv_file)
            self.model.train()
            if acc > self.best_acc:
                self.save_model(save_path)
                self.best_acc = acc

    # 模型评估阶段，使用'paddle.no_grad()'控制不计算和存储梯度
    @paddle.no_grad()
    def evaluate_pm(self, val_datadir, csv_file):
        self.model.eval()
        accuracies = []
        losses = []
        # 验证数据读取器
        valid_loader = valid_data_loader(val_datadir, csv_file)

        for batch_id, data in enumerate(valid_loader()):
            x_data, y_data = data
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            # 运行模型前向计算，得到预测值
            logits = self.model(img)
            # 多分类，使用softmax计算预测概率
            pred = F.softmax(logits)
            loss = self.loss_fn(pred, label)
            acc = paddle.metric.accuracy(pred, label)
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())
        print("[validation] accuracy/loss: {:.4f}/{:.4f}".format(np.mean(accuracies), np.mean(losses)))
        return np.mean(accuracies)    
    
    # 模型评估阶段，使用'paddle.no_grad()'控制不计算和存储梯度
    @paddle.no_grad()
    def predict_pm(self, x, **kwargs):
        # 将模型设置为评估模式
        self.model.eval()
        # 运行模型前向计算，得到预测值
        logits = self.model(x)
        return logits
    
    def save_model(self, save_path):
        paddle.save(self.model.state_dict(), save_path + 'palm.pdparams')
        paddle.save(self.optimizer.state_dict(), save_path + 'palm.pdopt')
    
    def load_model(self, model_path):
        model_state_dict = paddle.load(model_path)
        self.model.set_state_dict(model_state_dict)
        
        
# 开始实例化并训练
# 开启0号GPU训练
use_gpu = True
paddle.device.set_device('gpu:0') if use_gpu else paddle.device.set_device('cpu')

# 定义优化器
# opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters(), weight_decay=0.001)
opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())

runner = Runner(model, opt, loss_fn)

import os
# 数据集路径
DATADIR = '/home/aistudio/work/palm/PALM-Training400/PALM-Training400'
DATADIR2 = '/home/aistudio/work/palm/PALM-Validation400'
CSVFILE = '/home/aistudio/labels.csv'
# 设置迭代轮数
EPOCH_NUM = 5
# 模型保存路径
PATH='/home/aistudio/output/'
if not os.path.exists(PATH):
    os.makedirs(PATH)
# 启动训练过程
runner.train_pm(DATADIR, DATADIR2, 
                num_epochs=EPOCH_NUM, csv_file=CSVFILE, save_path=PATH)
                
                
# 第四部分，模型评估
# 使用测试数据对在训练过程中保存的最佳模型进行评价，观察模型在评估集上的准确率。代码实现如下：
# 加载最优模型
runner.load_model('/home/aistudio/output/palm.pdparams')
# 模型评价
score = runner.evaluate_pm(DATADIR2, CSVFILE)

# 第五部分，模型预测
# 使用保存好的模型，对测试集中的某一个数据进行模型预测，观察模型效果
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import paddle
import paddle.nn.functional as F
%matplotlib inline

# 加载最优模型
runner.load_model('/home/aistudio/output/palm.pdparams')

# 获取测试集中第一条数据
DATADIRv2 = '/home/aistudio/work/palm/PALM-Validation400'
filelists = open('/home/aistudio/labels.csv').readlines()
# 可以通过修改filelists列表的数字获取其他测试图片，可取值1-400
line = filelists[1].strip().split(',')
name, label = line[1], int(line[2])

# 读取测试图片
img = cv2.imread(os.path.join(DATADIRv2, name))
# 测试图片预处理
trans_img = transform_img(img)
unsqueeze_img = paddle.unsqueeze(paddle.to_tensor(trans_img), axis=0)

# 模型预测
logits = runner.predict_pm(unsqueeze_img)
result=F.softmax(logits)
pred_class = paddle.argmax(result).numpy()

# 输出真实类别与预测类别
print("The true category is {} and the predicted category is {}".format(label, pred_class))

# 图片可视化
show_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.imshow(show_img)
plt.show()

