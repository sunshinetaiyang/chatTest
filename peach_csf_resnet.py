# 飞桨案例，桃子分类，高级别API，resnet50_vd
# 第一部分：数据读取
#coding:utf-8
import os


# 23.5.8 注意保证版本的最新状态
# 23.5.8 由于案例时间久远，paddle的版本较旧，在加载hub.module时出错错误，需要重装
! pip uninstall paddlepaddle -y
! pip uninstall paddlehub -y


! pip install paddlepaddle
! pip install paddlehub

import paddle
import paddlehub as hub


class DemoDataset(paddle.io.Dataset):
    def __init__(self, transforms, num_classes=4, mode='train'):	
        # 数据集存放位置
        self.dataset_dir = "./work/peach-classification"  #dataset_dir为数据集实际路径，需要填写全路径
        self.transforms = transforms
        self.num_classes = num_classes
        self.mode = mode

        if self.mode == 'train':
            self.file = 'train_list.txt'
        elif self.mode == 'test':
            self.file = 'test_list.txt'
        else:
            self.file = 'validate_list.txt'
        
        self.file = os.path.join(self.dataset_dir , self.file)
        self.data = []
        
        with open(self.file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line != '':
                    self.data.append(line)
            
    def __getitem__(self, idx):
        img_path, grt = self.data[idx].split(' ')
        img_path = os.path.join(self.dataset_dir, img_path)
        im = self.transforms(img_path)
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
import paddlehub.vision.transforms as T

transforms = T.Compose(
        [T.Resize((256, 256)),
         T.CenterCrop(224),
         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
        to_rgb=True)
peach_train = DemoDataset(transforms)
peach_validate =  DemoDataset(transforms, mode='val')
peach_test =  DemoDataset(transforms, mode='test')


# 第三部分：模型搭建
'''我们要在PaddleHub中选择合适的预训练模型来Fine-tune，由于桃子分类是一个图像分类任务，这里采用Resnet50模型，并且是采用ImageNet数据集预训练过的版本。这个预训练模型是在图像任务中的一个“万金油”模型，Resnet是目前较为有效的处理图像的网络结构，50层是一个精度和性能兼顾的选择，而ImageNet又是计算机视觉领域公开的最大的分类数据集。所以，在不清楚选择什么模型好的时候，可以优先以这个模型作为baseline。

使用PaddleHub加载ResNet50模型，十分简单，只需一行代码即可实现。关于更多预训练模型信息参见PaddleHub模型介绍
'''
#安装预训练模型
!hub install resnet50_vd_imagenet_ssld==1.1.0

# 加载模型

import paddlehub as hub
# 23.5.8 这是绝对高级的API，用hub封装好的，已经是非常高层次的调用了。
model = hub.Module(name='resnet50_vd_imagenet_ssld', label_list=["R0", "B1", "M2", "S3"])

# 第四部分：模型训练
'''本案例中，我们使用Adam优化器，2014年12月，Kingma和Lei Ba提出了Adam优化器。该优化器对梯度的均值，即一阶矩估计（First Moment Estimation）和梯度的未中心化的方差，即二阶矩估计（Second Moment Estimation）进行综合计算，获得更新步长。Adam优化器实现起来较为简单，且计算效率高，需要的内存更少，梯度的伸缩变换不会影响更新梯度的过程， 超参数的可解释性强，且通常超参数无需调整或仅需微调等优点。我们将学习率设置为0.001，训练10个epochs。'''
from paddlehub.finetune.trainer import Trainer

import paddle

optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
trainer = Trainer(model, optimizer, checkpoint_dir='img_classification_ckpt', use_gpu=True) 
trainer.train(peach_train, epochs=10, batch_size=16, eval_dataset=peach_validate, save_interval=1)
'''其中paddle.optimizer.Adam:

learning_rate: 全局学习率。默认为1e-3；
parameters: 待优化模型参数。
运行配置
Trainer 主要控制Fine-tune的训练，包含以下可控制的参数:

model: 被优化模型；
optimizer: 优化器选择；
use_vdl: 是否使用vdl可视化训练过程；
checkpoint_dir: 保存模型参数的地址；
compare_metrics: 保存最优模型的衡量指标；
trainer.train 主要控制具体的训练过程，包含以下可控制的参数：

train_dataset: 训练时所用的数据集；
epochs: 训练轮数；
batch_size: 训练的批大小，如果使用GPU，请根据实际情况调整batch_size；
num_workers: works的数量，默认为0；
eval_dataset: 验证集；
log_interval: 打印日志的间隔， 单位为执行批训练的次数。
save_interval: 保存模型的间隔频次，单位为执行训练的轮数。
当Fine-tune完成后，我们使用模型来进行预测，实现如下：
'''

# 第五部分：模型评估
# 模型评估
trainer.evaluate(peach_test, 16)

# 第六部分：模型推理
import paddle
import paddlehub as hub
from PIL import Image
import matplotlib.pyplot as plt

img_path = './work/test.jpg'
img = Image.open(img_path)
plt.imshow(img)
plt.axis('off')
plt.show()
result = model.predict([img_path])
print("桃子的类别被预测为:{}".format(result))

# 第七部分：模型部署
由于AIStudio不支持ip访问，以下代码仅做示例，如有需要，请在本地机器运行。
想用我们自己训练的分拣桃子的网络参数，先配置config.json文件：
{
  "modules_info": {
    "resnet50_vd_imagenet_ssld": {
      "init_args": {
          "version": "1.1.0",
          "label_list":["R0", "B1", "M2", "S3"],
          "load_checkpoint": "img_classification_ckpt/best_model/model.pdparams"
      },
      "predict_args": {
          "batch_size": 1
      }

    }
  },
  "port": 8866,
  "gpu": "0"
}

借助 PaddleHub，服务器端的部署也非常简单，直接用一条命令行在服务器启动resnet50分类模型就行了：

$ hub serving start --config config.json
是的，在服务器端这就完全没问题了。相比手动配置各种参数或者调用各种框架，PaddleHub 部署服务器实在是太好用了。

NOTE: 如使用GPU预测，则需要在启动服务之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。

在服务端发送请求，请求脚本如下：


import requests
import json
import cv2
import base64

import numpy as np


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')

def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data

# 发送HTTP请求
org_im = cv2.imread('/PATH/TO/IMAGE')
data = {'images':[cv2_to_base64(org_im)], 'top_k':1}
headers = {"Content-type": "application/json"}
url = "http://127.0.0.1:8866/predict/resnet50_vd_imagenet_ssld"
r = requests.post(url=url, headers=headers, data=json.dumps(data))
data =r.json()["results"]['data']

相信只要有一些 Python 基础，在本地预测、以及部署到服务器端都是没问题的，飞桨的 PaddleHub 已经帮我们做好了各种处理过程。

# 第八部分：资源
资源
更多资源请参考：

更多深度学习知识、产业案例，请参考：awesome-DeepLearning

更多预训练模型(图像、文本、语音、视频等)，请参考：PaddleHub

飞桨框架相关资料，请参考：飞桨深度学习平台

数据来源

本案例数据集来源于：https://aistudio.baidu.com/aistudio/datasetdetail/67225


