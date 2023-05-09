

# 23.5.8 提交作业赚积分，猫的12分类问题
# 23.5.8 TODO
# 1 预测时候遇到GIF图片无法加载的问题
# 2 训练时候遇到GIF图片无法加载的问题
# 3 精度提升问题
# 4 5月9号提交了一版，系统自动算出了result.csv的准确率，积分69.5，闯关失败，还得继续提升准确率啊
jupyter nbconvert --to=python main.ipynb

# 第一部分：环境准备和数据读取
#coding:utf-8
import os


# 23.5.8 注意保证版本的最新状态
# 23.5.8 由于案例时间久远，paddle的版本较旧，在加载hub.module时出错错误，需要重装
! pip uninstall paddlepaddle -y
! pip uninstall paddlehub -y

! pip install paddlepaddle
! pip install paddlehub

!unzip -q -d /home/aistudio/work /home/aistudio/data/data10954/cat_12_test.zip
!unzip -q -d /home/aistudio/work /home/aistudio/data/data10954/cat_12_train.zip

# 23.5.8 拷贝标注到工作目录work下
! cp /home/aistudio/data/data10954/train_list.txt /home/aistudio/work

# 这个train_list.txt需要划分为train集和val集，一则是模型训练的需要，二则和hub调用匹配
! ls -l /home/aistudio/work/cat_12_train | grep "^-" | wc -l
! ls -l /home/aistudio/work/cat_12_test | grep "^-" | wc -l
2160
240
# 2023.5.8 对比cat数据集和peach数据集，应该修改cat_test为cat_predict，因为cat_test没有标签，是用来做推理
# 按照2160和240的书目，拆分240个到val集，需要建立val目录，和validate_list.txt
grep -c "11" train_list.txt
# 因为现有cat_12_test的图片是没有标签的，无法用来训练和finetune模型，只能用来推理，所以改名
! mv cat_12_test cat_12_predict

# 23.5.8 对于目前的标记文件train_list.txt，要三分类7:1:1，分别取140张，20张和20张
# 借助GPT进行划分
要将标记文件train_list.txt划分为训练集（train_list.txt）、验证集（validate_list.txt）和测试集（test_list.txt），可以按照以下步骤进行操作：

1. 打乱标记文件中的行顺序，以便随机选择行来分配到不同的集合中。可以使用`shuf`命令来随机化文件中的行：

   ```
   shuf train_list.txt -o train_list_shuf.txt
   ```

   这将把原始标记文件train_list.txt的行打乱，然后将结果保存到新文件train_list_shuf.txt中。

2. 确定每个集合中应该有多少行。可以根据您的需要和数据集大小来设置这些值。例如，您可能希望将80％的数据分配给训练集，10％分配给验证集，10％分配给测试集。在这种情况下，您可以计算出每个集合应该有的行数：

   ```
   # 计算训练集行数
   num_total=$(wc -l train_list_shuf.txt | cut -d' ' -f1)
   num_train=$(echo "($num_total * 80 + 99) / 100" | bc)

   # 计算验证集行数
   num_val=$(echo "($num_total - $num_train + 1) / 2" | bc)

   # 计算测试集行数
   num_test=$(echo "$num_total - $num_train - $num_val + 1" | bc)
   ```
# 因为已知会用每个分类的7:1:1所以：
num_train=$((140*12))
num_val=$((20*12))
num_test=$((20*12))
echo $num_val

# 5.9 鉴于训练和验证比重有些偏，增加训练比重
num_train=$((150*12))
num_val=$((15*12))
num_test=$((15*12))

   在这个示例中，`wc -l`命令用于计算文件中的行数，`cut`命令用于提取结果中的行数。`bc`命令用于执行整数除法并向上舍入。

3. 将随机排序的文件分配到不同的集合中。可以使用`head`和`tail`命令来选择文件中的前几行和后几行：

   ```
   # 选择前num_train行作为训练集
   head -n $num_train train_list_shuf.txt > train_list.txt

   # 选择接下来的num_val行作为验证集
   tail -n +$(($num_train + 1)) train_list_shuf.txt | head -n $num_val > validate_list.txt

   # 选择接下来的num_test行作为测试集
   tail -n +$(($num_train + $num_val + 1)) train_list_shuf.txt > test_list.txt
   ```

   在这个示例中，`head`命令用于选择文件的前几行，`tail`命令用于选择文件的后几行。`+`符号表示从指定行号开始选择行。

4. 删除中间文件train_list_shuf.txt。

   ```
   rm train_list_shuf.txt
   ```

这样，您就可以将标记文件train_list.txt划分为训练集、验证集和测试集，并将它们保存到train_list.txt、validate_list.txt和test_list.txt中。

# 23.5.8 百度官方文档关于loaddata案例：
    def __getitem__(self, index):
        """
        步骤三：实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        image_path, label = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # cv2.imread(im)
        image = image.astype('float32')
        # 3. 应用数据处理方法到图像上
        if self.transform is not None:
            image = self.transform(image)
        label = int(label)
        return image, label
        
        
# 23.5.8 解决图像读取的问题，使用PIL读取而避免 im = cv2.imread(im) 错误
from PIL import Image
import paddle.vision.transforms as T

transforms = T.Compose([
    T.Resize((400, 400)),
    T.RandomHorizontalFlip(),
])

# 读取图像
img_path = "example.jpg"
with open(img_path, 'rb') as f:
    img = Image.open(f).convert('RGB')

# 应用变换
img = transforms(img)

# 转换为 Tensor
img_tensor = T.ToTensor()(img)


import paddle
import paddlehub as hub


# 23.5.8 奇怪的是这段代码在cpu运行环境下paddle版本错误，GPU是OK
# 23.5.9 定义数据集类，加载训练的数据和标签
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
        print('list size:', len(data))
            
    def __getitem__(self, idx):
        # 23.5.8 这个函数从“桃子”分类移植，split符合要修改为'\t
        img_path, grt = self.data[idx].split('\t')
        img_path = os.path.join(self.dataset_dir, img_path)
        im = self.transforms(img_path)
        return im, int(grt)

    def __len__(self):
        return len(self.data)
        
        
# 23.5.8 解决cv2.readim()出错的思路
# 使用AIstudio论坛也可以有很多的借鉴
# 1，可以遍历文件夹，把文件名后缀不是.jpg、.jpeg、.png的文件找出来，删除即可。
# 2，“已经解决了，用Image谢谢哈”，意思是用PIL是可以解决的。方案如下：
import cv2
import glob
from PIL import Image

pl = glob.glob("cat_12_test/" + "*.jpg")
for i in pl:
    if cv2.imread(i) is not None:
        pass
    else:
        print(i)
        img = Image.open(path).convert('RGB')
        # 23.5.9 先完全遍历并变换格式，让cv2可以识别
        img.save(i)
# 5.9 根据GPT思路，改进如下
import os
from PIL import Image
import cv2


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

# 还有这样处理的：
        #获取单个样本数据和标签
        image_file,label = self.data[index]
        image = Image.open(image_file)
        #非RGB格式图像转化为RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        #图像预处理
        image = self.transforms(image)
        #将标签转换为numpy形式
        return image,np.array(label, dtype = 'int64')


# 第二部分：数据预处理
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
        print('宽度：', width)
        print('高度：', height)
        width_sum += width
        height_sum += height

# 计算平均尺寸
avg_width = width_sum / len(image_names)
avg_height = height_sum / len(image_names)
print('平均宽度：', avg_width)
print('平均高度：', avg_height)



'''将训练数据输入模型之前，我们通常还需要对原始数据做一些数据处理的工作，比如数据格式的规范化处理，或增加一些数据增强策略。

构建图像分类模型的数据读取器，负责将桃子dataset的数据进行预处理，以特定格式组织并输入给模型进行训练。

如下数据处理策略，只做了三种操作：

1.指定输入图片的尺寸，并将所有样本数据统一处理成该尺寸。

2.对输入图像进行裁剪，并且保持图片中心点不变。

3.对所有输入图片数据进行归一化处理。

对数据预处理及加载数据集的示例如下：'''
import paddlehub.vision.transforms as T

# 23.5.8 遇到一张图片格式: GIF 大小: (216, 188) 模式: P，导致im = self.transforms(img_path)挂死
# 要提升程序的健壮性
# transforms = T.Compose(
        # [T.Resize((256, 256)),
         # T.CenterCrop(224),
         # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
        # to_rgb=True)
        
# 23.5.8 GPT推荐的增强方式
transforms = T.Compose([
    T.Resize((256, 256)),
    T.RandomResizedCrop(224), # 这个paddlehub不支持了
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


cat_train = DemoDataset(transforms)
cat_validate =  DemoDataset(transforms, mode='val')
cat_test =  DemoDataset(transforms, mode='test')


# 第三部分：模型搭建
'''我们要在PaddleHub中选择合适的预训练模型来Fine-tune，由于桃子分类是一个图像分类任务，这里采用Resnet50模型，并且是采用ImageNet数据集预训练过的版本。这个预训练模型是在图像任务中的一个“万金油”模型，Resnet是目前较为有效的处理图像的网络结构，50层是一个精度和性能兼顾的选择，而ImageNet又是计算机视觉领域公开的最大的分类数据集。所以，在不清楚选择什么模型好的时候，可以优先以这个模型作为baseline。

使用PaddleHub加载ResNet50模型，十分简单，只需一行代码即可实现。关于更多预训练模型信息参见PaddleHub模型介绍
'''
#安装预训练模型
!hub install resnet50_vd_imagenet_ssld==1.1.0

# 加载模型
import paddlehub as hub
# 23.5.8 这是绝对高级的API，用hub封装好的，已经是非常高层次的调用了。
model = hub.Module(name='resnet50_vd_imagenet_ssld',label_list=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"])

# 第四部分：模型训练
'''本案例中，我们使用Adam优化器，2014年12月，Kingma和Lei Ba提出了Adam优化器。该优化器对梯度的均值，即一阶矩估计（First Moment Estimation）和梯度的未中心化的方差，即二阶矩估计（Second Moment Estimation）进行综合计算，获得更新步长。Adam优化器实现起来较为简单，且计算效率高，需要的内存更少，梯度的伸缩变换不会影响更新梯度的过程， 超参数的可解释性强，且通常超参数无需调整或仅需微调等优点。我们将学习率设置为0.001，训练10个epochs。'''
from paddlehub.finetune.trainer import Trainer

import paddle

# 23.5.8 use_gpu=True在这个环境下老是出现如下问题：
# ValueError: The device should not be 'gpu', since PaddlePaddle is not compiled with CUDA
# 无法使用GPU导致训练时间非常长

optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
trainer = Trainer(model, optimizer, checkpoint_dir='img_classification_ckpt', use_gpu=True) 
trainer.train(cat_train, epochs=10, batch_size=16, eval_dataset=cat_validate, save_interval=1)

'''
训练的打印信息如下：
这是一个深度学习模型的训练输出日志，包含了当前训练的一些重要信息。

具体来说，这段日志的含义如下：

- `[TRAIN]`：表示当前输出是针对训练集的统计信息，还可能会有`[VAL]`表示验证集的统计信息。
- `Epoch=1/10`：表示当前处于第1个epoch，总共需要训练10个epoch。
- `Step=20/105`：表示当前处于第20个batch，总共需要训练105个batch。一般来说，一个epoch的训练次数（即batch数量）等于训练集中的样本数除以batch_size，如果无法整除则需要进行取整或者舍去。
- `loss=2.1858`：表示当前batch的损失值为2.1858，损失值一般用来衡量模型训练的好坏，通常希望损失值越小越好。
- `acc=0.3500`：表示当前batch的准确率为35.00%，准确率一般用来衡量模型的分类精度，通常希望准确率越高越好。
- `lr=0.001000`：表示当前的学习率为0.001000，学习率一般用来控制模型参数的更新速度，通常需要调整来达到更好的训练效果。
- `step/sec=0.08`：表示当前每秒钟可以处理0.08个batch的数据，一般来说这个值越大越好，可以说明模型训练的速度越快。
- `ETA 03:28:20`：表示预计还需要3小时28分钟20秒才能完成当前epoch的训练，ETA一般用来估计模型训练的时间，可以帮助调整训练的参数和策略，以提高训练效率。


'''

# 第五部分：模型评估
# 模型评估
trainer.evaluate(cat_test, 16)
# 【重要】23.5.9 训练10个epochs才只有0.66的准确率，如何提升是一门学问
# 1 增加训练次数，好在GPU训练，10个epoch只需要5分钟
# 2 图片的transform，各种增强，每10个epoch用不同的增强

# 第六部分：模型推理
# 考试要求：考试提交，需要提交模型代码项目版本和结果文件。结果文件为CSV文件格式，命名为result.csv，
# 文件内的字段需要按照指定格式写入。

# 文件格式：WMgOhwZzacY023lCusqnBxIdibpkT5GP.jp,0 其中，前半部分为【图片路径】，后半部分为【类别编号】，数据列以逗号分隔，每一行数据都以回车符结束。

import paddle
import paddlehub as hub
import matplotlib.pyplot as plt
import os

# 构建待预测的文件路径列表
predict_dir = '/home/aistudio/work/cat_12_predict'
predict_files = os.listdir(predict_dir)
predict_file_paths = [os.path.join(predict_dir, f) for f in predict_files]

# 对文件路径列表进行预测
results = model.predict(predict_file_paths)

# 将预测结果保存为csv文件
with open('results.csv', 'w') as f:
    for i, result in enumerate(results):
        filename = predict_files[i]
        label = result['label']
        f.write(f"{filename},{label}\n")
        
# 23.5.8 用于单次单个图片的训练，便于处理异常
import os
import csv

# 获取所有待预测图片的文件路径
img_dir = '/home/aistudio/work/cat_12_predict'
img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]

# 预测并将结果输出到CSV文件中
with open('predictions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for img_path in img_paths:
        try:
            # result = model.predict([img_path])[0]
            results = model.predict([img_path])
            print('results: ', results)
            result = results[0]
            file_name = os.path.basename(img_path)
            label = result[0]
            writer.writerow([file_name, label])            
        except Exception as e:
            print(f"An error occurred while predicting image {img_path}: {e}")
            file_name = os.path.basename(img_path)
            label = 0            
            writer.writerow([file_name, label])  
            
# 23.5.9 GPT代码预测
import os
import csv
import paddlehub as hub

# 设置要预测的图片文件夹路径
img_dir = '/home/aistudio/work/cat_12_predict'
img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]

results = model.predict(img_paths)

# 创建 CSV 文件
with open('predict_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(len(img_paths)):
        file_name = os.path.basename(img_paths[i])
        
        # 提取预测结果中的标签和概率
        predicted_label = list(results[i].keys())[0]
        print(file_name ,predicted_label)
        # predicted_prob = list(result[0].values())[0]

        # 将预测结果写入 CSV 文件
        writer.writerow([file_name, predicted_label])





