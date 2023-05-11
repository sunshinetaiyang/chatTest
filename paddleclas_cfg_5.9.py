# 下面以PaddleClas中的configs/quick_start/ResNet50_vd.yaml为例，介绍一下训练的配置参数。

mode: 'train' # 当前所处的模式，支持训练与评估模式

ARCHITECTURE:
name: 'ResNet50_vd' # 模型结构，可以通过这个这个名称，使用模型库中其他支持的模型
pretrained_model: "" # 预训练模型，因为这个配置文件演示的是不加载预训练模型进行训练，因此配置为空。
model_save_dir: "./output/" # 模型保存的路径
classes_num: 102 # 类别数目，需要根据数据集中包含的类别数目来进行设置
total_images: 1020 # 训练集的图像数量，用于设置学习率变换策略等。
save_interval: 1 # 保存的间隔，每隔多少个epoch保存一次模型
validate: True # 是否进行验证，如果为True，则配置文件中需要包含VALID字段
valid_interval: 1 # 每隔多少个epoch进行验证
epochs: 20 # 训练的总得的epoch数量
topk: 5  # 除了top1 acc之外，还输出topk的准确率，注意该值不能大于classes_num
image_shape: [3, 224, 224] # 图像形状信息

LEARNING_RATE: # 学习率变换策略，目前支持Linear/Cosine/Piecewise/CosineWarmup
    function: 'Cosine'
    params:
        lr: 0.0125

OPTIMIZER: # 优化器设置
    function: 'Momentum'
    params:
        momentum: 0.9
    regularizer:
        function: 'L2'
        factor: 0.00001

LOSS:
    function: "CELoss"
    params:

TRAIN: # 训练配置
    batch_size: 32 # 训练的batch size
    num_workers: 0 # 每个trainer(1块GPU上可以视为1个trainer)的进程数量
    file_list: "./dataset/flowers102/train_list.txt" # 训练集标签文件，每一行由"image_name label"组成
    data_dir: "./dataset/flowers102/" # 训练集的图像数据路径
    shuffle_seed: 0 # 数据打散的种子
    transforms: # 训练图像的数据预处理
        - DecodeImage: # 解码
            to_rgb: True
            channel_first: False
        - RandCropImage: # 随机裁剪
            size: 224
        - RandFlipImage: # 随机水平翻转
            flip_code: 1
        - NormalizeImage: # 归一化
            scale: 1./255.
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
        - ToCHWImage: # 通道转换

VALID: # 验证配置，validate为True时有效
    batch_size: 20 # 验证集batch size
    num_workers: 0  # 每个trainer(1块GPU上可以视为1个trainer)的进程数量
    file_list: "./dataset/flowers102/val_list.txt" # 验证集标签文件，每一行由"image_name label"组成
    data_dir: "./dataset/flowers102/" # 验证集的图像数据路径
    shuffle_seed: 0 # 数据打散的种子
    transforms:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - ResizeImage:
            resize_short: 256
        - CropImage:
            size: 224
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
        - ToCHWImage:
        
# 基于ResNet50_vd模型，训练命令如下所示。
#添加环境变量，并启动不加载预训练模型的训练
#!python3 -m paddle.distributed.launch --gpus="0" tools/train.py -c ./configs/quick_start/ResNet50_vd.yaml
!python tools/train.py -c ./configs/quick_start/ResNet50_vd.yaml

# 3.2 模型微调-基于ResNet50_vd预训练模型(准确率79.12%)
# 基于ImageNet1k分类预训练模型进行微调，训练命令如下所示。
#下载预训练模型
!mkdir pretrained
%cd pretrained
!wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_pretrained.pdparams
%cd ..

#启动基于ResNet50_vd预训练模型的训练
#!python3 -m paddle.distributed.launch --gpus="0" tools/train.py  -c ./configs/quick_start/ResNet50_vd_finetune.yaml
!python3 tools/train.py -c ./configs/quick_start/ResNet50_vd_finetune.yaml -o use_gpu=True

训练完成后，查看训练集验证集的loss与准确率等信息。
最终验证集准确率为93.82%，加载预训练模型之后，flowers102数据集精度大幅提升，绝对精度涨幅超过63%。


3.3 SSLD模型微调-基于ResNet50_vd_ssld预训练模型(准确率82.39%)
基于ImageNet1k分类SSLD预训练模型进行微调，这里需要注意的是，在使用通过知识蒸馏得到的预训练模型进行微调时，我们推荐使用相对较小的网络中间层学习率。对应地，配置文件中需要进行以下配置。
ARCHITECTURE:
    name: 'ResNet50_vd'
    params: # 使用该模型时额外传入的参数
        lr_mult_list: [0.5, 0.5, 0.6, 0.6, 0.8] # 每个res-block的学习率倍数，默认均为1
pretrained_model: "./pretrained/ResNet50_vd_ssld_pretrained" # 预训练模型地址
PaddleClas已经配置好了该部分的配置，加载配置脚本直接启动训练即可。具体的训练脚本如下所示。
#下载预训练模型
!mkdir pretrained
%cd pretrained
!wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_ssld_pretrained.pdparams
%cd ..
# 启动SSLD模型微调-基于ResNet50_vd_ssld预训练模型的训练
#!python3 -m paddle.distributed.launch  --gpus="0"  tools/train.py -c ./configs/quick_start/ResNet50_vd_ssld_finetune.yaml
!python3 tools/train.py -c ./configs/quick_start/ResNet50_vd_ssld_finetune.yaml -o use_gpu=True

训练完成后，查看训练集验证集的loss与准确率等信息。
最终flowers102验证集上精度指标为95.29%，相对于79.12%预训练模型的微调结果，新数据集指标可以再次提升约1%。

3.4 数据增广的尝试-RandomErasing

训练数据量较小时，使用数据增广可以进一步提升模型精度，基于3.3节中的训练方法，结合RandomErasing的数据增广方式进行训练，配置文件中的训练集配置如下所示。
TRAIN:
    batch_size: 32
    num_workers: 0
    file_list: "./dataset/flowers102/train_list.txt"
    data_dir: "./dataset/flowers102/"
    shuffle_seed: 0
    transforms:
        - DecodeImage:
            to_rgb: True
            to_np: False
            channel_first: False
        - RandCropImage:
            size: 224
        - RandFlipImage:
            flip_code: 1
        - NormalizeImage:
            scale: 1./255.
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
        - RandomErasing: # 在归一化之后使用RandomErasing方法进行数据增广
            EPSILON: 0.5
        - ToCHWImage:
具体的训练命令如下所示。

#启动含数据增广RandomErasing的训练
#!python3 -m paddle.distributed.launch  --gpus="0"   tools/train.py -c ./configs/quick_start/ResNet50_vd_ssld_random_erasing_finetune.yaml
!python3 tools/train.py -c ./configs/quick_start/ResNet50_vd_ssld_random_erasing_finetune.yaml  -o use_gpu=True

训练完成后，可以在终端中实时查看训练集验证集的loss与准确率等信息。

最终flowers102验证集上的精度为95.39%，使用数据增广可以使得模型精度再次提升0.1%
将该模型保存模型到./pretrained/flowers102_R50_vd_final/文件夹下，用于作为后续SSLD知识蒸馏实验的教师模型的预训练模型，需要运行如下命令：

#保存模型
!cp -r output/ResNet50_vd/19/ ./pretrained/flowers102_R50_vd_final/

3.5 尝试更多的模型结构-MobileNetV3

基于ImageNet1k分类预训练模型进行微调，训练命令如下所示。
#下载预训练模型
!mkdir pretrained
!wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x1_0_pretrained.pdparams -P pretrained/
#启动模型结构为MobileNetV3的训练
!python3 -m paddle.distributed.launch  --gpus="0"  tools/train.py -c ./configs/quick_start/MobileNetV3_large_x1_0_finetune.yaml
训练完成后，查看训练集验证集的loss与准确率等信息。

最终flowers102验证集上的精度为90.39%，比加载了预训练模型的ResNet50_vd的精度差了5%。但是模型大小减少了70%，不同模型结构的网络在相同数据集上的性能表现不同，需要根据预测耗时以及存储的需求选择合适的模型。

3.6 SSLD知识蒸馏小试牛刀

使用flowers102数据集进行模型蒸馏，为了进一步提提升模型的精度，使用extra_list.txt充当无标签数据，在这里有几点需要注意：
extra_list.txt与val_list.txt的样本没有重复，因此可以用于扩充知识蒸馏任务的训练数据。
即使引入了有标签的extra_list.txt中的图像，但是代码中没有使用标签信息，因此仍然可以视为无标签的模型蒸馏。
蒸馏过程中，教师模型使用的预训练模型为flowers102数据集上的训练结果，学生模型使用的是ImageNet1k数据集上精度为75.32%的MobileNetV3_large_x1_0预训练模型。
配置文件中的主要变化主要如下所示。
total_images: 7169 # 图像数量，在这里使用了更多的额外数据，因此图像数量有所增加
use_distillation: True # 使用蒸馏
ARCHITECTURE:
    name: 'ResNet50_vd_distill_MobileNetV3_large_x1_0' # 蒸馏模型会同时输出教师与学生模型的预测结果
pretrained_model: # 预训练模型地址，在这里因为需要加载两个预训练模型，因此以列表形式给出，无先后顺序
    - "./pretrained/flowers102_R50_vd_final/ppcls" # 教师模型的预训练模型
    - "./pretrained/MobileNetV3_large_x1_0_pretrained” # 学生模型的预训练模型
TRAIN:
    file_list: "./dataset/flowers102/train_extra_list.txt" # 训练集的标签文件
首先将刚才保存的ResNet50_vd预训练拷贝，作为教师模型的预训练参数。

!cp -r output/ResNet50_vd/best_model/ ./pretrained/flowers102_R50_vd_final/
最终的训练脚本如下所示。
#启动含SSLD知识蒸馏的训练
!python3 -m paddle.distributed.launch  --gpus="0"  tools/train.py  -c ./configs/quick_start/R50_vd_distill_MV3_large_x1_0.yaml

训练完成后，查看训练集验证集的loss与准确率等信息。

最终flowers102验证集上的精度为0.9549，结合更多的无标签数据，使用教师模型进行知识蒸馏，MobileNetV3的精度涨幅高达5%，甚至超越了教师模型，从而具有了同时兼顾速度和精度的优势

