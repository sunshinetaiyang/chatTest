下面以PaddleClas中的configs/quick_start/ResNet50_vd.yaml为例，介绍一下训练的配置参数。

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