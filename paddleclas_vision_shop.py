#!/usr/bin/env python
# coding: utf-8

# # 1. 项目场景说明
# ![](https://ai-studio-static-online.cdn.bcebos.com/fb4d5b47f401468d9cd565b23cf289792835142451e24f2e9bc752c933852db4)
# 
# 在超市等无人零售场景中，目前主要是结算方式，主要有以下几种
# - 条形码方式
# - RFID等射频码
# - 称重方法
# 
# 但是以上几种方法存在如下缺点：
# 1）针对条形码方式，对于成品包装的商品，较为成熟，但是对与生鲜产品等商品，并不能满足需求。
# 2）RFID等方式，虽然对生鲜等产品能够支持，但是额外生成标签，增加成本
# 3）称重方法，对于相同重量的山商品，不能很好的区分，同时重量称等精密仪器在长时间的负重和使用过程中，精度会发生变化，需要工作人员定期调教，以满足精度需求。
# 
# 因此，如何选择一种既能大规模支持各种商品识别，又能方便管理，同时维护成本不高的识别系统，显得尤为重要。
# 
# 深圳市银歌云技术有限公司基于飞桨的图像识别开发套件PaddleClas，提供了一套基于计算机视觉的完整生鲜品自主结算方案，其通过结算平台的摄像头拍摄的图像，自动的识别称上的商品，整个流程在1秒内完成，无需售卖人员的操作及称重。整个流程，实现了精度高、速度快，无需人工干预的自动结算效果。减少人工成本的同时，大大提高了效率和用户体验。
# 
# 读者可以通过本项目熟悉AI项目的落地流程，了解解决问题的思路。如果能对大家今后的实际项目操作有一定启发，那我们就颇感欣慰了。😊
# 在此过程中，我们先来看一下项目面临的挑战
# - 物体形状千差万别，如何找到待检测的商品？
# - 商品及生鲜品种类繁多，如何准备的识别出对应种类？
# - 使用过程中，商品及生鲜品类迭代速度快，如何减少模型更新成本？

# # 2. 技术方案选型
# 针对以上难点，我们的方案如下PP—ShiTuV2
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/10fc391e32e54a4ab27bfa11f2e2ec307ca64b0d27d147fcb716c8af16035296)
# 
# 如上图所示，针对以上问题，我们使用图中的PipeLine进行解决上述问题。整个PipeLine中，主要分为三部分
# 1. 主体检测：检测出待识别的商品，去掉冗余的背景信息，提高生鲜品识别的精度
# 2. 特征提取：将待识别的生鲜品图像，提取特征
# 3. 检索模块：将待检索的特征与库中的生鲜品特征比对，得到待检索生鲜品的标签。
# 
# 在此方案中，用户只需要训练一套模型，之后在应用过程中，只需要在检索库中，添加少量有代表性的新增生鲜品及商品类别图像，就能够很好的解决新增商品问题。同时，在使用的过程中，无需频繁重新训练模型，能够极大的降低用户使用成本。同时本套方案中，用户在后续使用中，无需添加辅助设备，降低了维护及使用成本。

# # 3. 安装说明
# 环境要求
# - PaddlePaddle = 2.2.2
# - Python = 3.7
# - PaddleClas = 2.5
# - PaddleDetection = 2.3
# 

# In[1]:


# 安装PaddleClas
get_ipython().run_line_magic('cd', '~')
get_ipython().system('git clone https://github.com/PaddlePaddle/PaddleClas.git')
get_ipython().run_line_magic('cd', 'PaddleClas')
# 切换到2.5版本
get_ipython().system('git checkout release/2.5')
# 安装好相关依赖
get_ipython().system('pip install -r requirements.txt')
# 安装PaddleClas
get_ipython().system('python setup.py install')
get_ipython().run_line_magic('cd', '..')


# In[2]:


# 安装PaddleDetection
get_ipython().run_line_magic('cd', '~')
get_ipython().system('git clone https://github.com/PaddlePaddle/PaddleDetection.git')
get_ipython().run_line_magic('cd', 'PaddleDetection')
get_ipython().system('git checkout release/2.3')
get_ipython().system('pip install -r requirements.txt')


# # 4. 数据准备
# ## 4.1 主体检测数据准备
# 在PP-ShiTuV2中，主体检测训练数据集如下：
# | 数据集       | 数据量 | 主体检测任务中使用的数据量 | 场景         | 数据集地址                                                 |
# | ------------ | ------ | -------------------------- | ------------ | ---------------------------------------------------------- |
# | Objects365   | 170W   | 6k                         | 通用场景     | [地址](https://www.objects365.org/overview.html)           |
# | COCO2017     | 12W    | 5k                         | 通用场景     | [地址](https://cocodataset.org/)                           |
# | iCartoonFace | 2k     | 2k                         | 动漫人脸检测 | [地址](https://github.com/luxiangju-PersonAI/iCartoonFace) |
# | LogoDet-3k   | 3k     | 2k                         | Logo 检测    | [地址](https://github.com/Wangjing1551/LogoDet-3K-Dataset) |
# | RPC          | 3k     | 3k                         | 商品检测     | [地址](https://rpc-dataset.github.io/)  
# 
# 在PP-ShiTu中，使用上述数据集进行模型主体检测模型训练。在整个PP-ShiTu的Pipeline中，主体检测只是需要把待检测图像中的待检测物体检测出来，即需要区分物体的类别，只需要框出物体位置。因此，需要对下载的数据进行处理一下，将所有的检测数据集中物体都统一成一个类别，并修改成COCO数据集格式。具体操作可以根据[自定义检测数据集教程](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/docs/tutorials/PrepareDataSet.md)进行操作。
# 
# 在此文档中，我们准备了整理好的demo数据，位于`~/work/data/detection_demo_dataset.tar`
# 
# ## 4.2 识别模型数据集准备
# 在PP-ShiTuV2中，识别模型训练数据集如下：
# | 数据集                 | 数据量  |  类别数  | 场景  |                                      数据集地址                                      |
# | :--------------------- | :-----: | :------: | :---: | :----------------------------------------------------------------------------------: |
# | Aliproduct             | 2498771 |  50030   | 商品  |      [地址](https://retailvisionworkshop.github.io/recognition_challenge_2020/)      |
# | GLDv2                  | 1580470 |  81313   | 地标  |               [地址](https://github.com/cvdfoundation/google-landmark)               |
# | VeRI-Wild              | 277797  |  30671   | 车辆  |                    [地址](https://github.com/PKU-IMRE/VERI-Wild)                     |
# | LogoDet-3K             | 155427  |   3000   | Logo  |              [地址](https://github.com/Wangjing1551/LogoDet-3K-Dataset)              |
# | SOP                    |  59551  |  11318   | 商品  |              [地址](https://cvgl.stanford.edu/projects/lifted_struct/)               |
# | Inshop                 |  25882  |   3997   | 商品  |            [地址](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)             |
# | bird400                |  58388  |   400    | 鸟类  |          [地址](https://www.kaggle.com/datasets/gpiosenka/100-bird-species)          |
# | 104flows               |  12753  |   104    | 花类  |              [地址](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)              |
# | Cars                   |  58315  |   112    | 车辆  |            [地址](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)            |
# | Fashion Product Images |  44441  |    47    | 商品  | [地址](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) |
# | flowerrecognition      |  24123  |    59    | 花类  |         [地址](https://www.kaggle.com/datasets/aymenktari/flowerrecognition)         |
# | food-101               | 101000  |   101    | 食物  |         [地址](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)          |
# | fruits-262             | 225639  |   262    | 水果  |            [地址](https://www.kaggle.com/datasets/aelchimminut/fruits262)            |
# | inaturalist            | 265213  |   1010   | 自然  |           [地址](https://github.com/visipedia/inat_comp/tree/master/2017)            |
# | indoor-scenes          |  15588  |    67    | 室内  |       [地址](https://www.kaggle.com/datasets/itsahmad/indoor-scenes-cvpr-2019)       |
# | Products-10k           | 141931  |   9691   | 商品  |                       [地址](https://products-10k.github.io/)                        |
# | CompCars               |  16016  |   431    | 车辆  |     [地址](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)      |
# | **Total**              | **6M**  | **192K** |   -   |                                          -                                           |
# 
# 为了保证识别模型的通用特征能力，我们使用上述16个公开数据集，覆盖商品、地标、车辆、logo、植物、动物等多种场景。因为训练数据丰富多样，在实际使用场景，可以直接使用我们公开的预训练模型，就能得到一个比较好的效果。在下载完成数据集后，需要对数据集进行整理，并同时生成`image_list.txt`的文件，文件中每一行格式如下：
# ```
# image_path imagel_label_id
# ```
# 在本实验中，我们也准备了整理好的demo数据，位置在`~/work/data/rec_demo_dataset.tar`

# # 5. 模型选择
# 
# ## 5.1 主体检测模型选择
# 
# PaddleDetection 提供了非常丰富的目标检测模型，但是我们需要从项目实际情况出发，选择适合部署条件的模型。项目要求模型体积小、精度高、速度达标，因此我们将候选模型锁定在PP-PicoDet上。
# 我们先来看一下 PicoDet 的性能指标对比：![](https://ai-studio-static-online.cdn.bcebos.com/24c977f3b0774dc9a1f4d0c2cb3931ff7fa6fac70cb340cb9c6ca0abecee6e27)
# 如图所示，PicoDet无论从速度和精度都有较明显的优势。同时为了能够较好的兼顾兼顾速度与精度，我们使用PP-LCNet2.5x作为backbone的Picodet_m_640模型。
# 
# ## 5.2 识别模型选择
# 
# 
# | Model | Params(M) | FLOPs(M) | Top-1 Acc(\%) | Top-5 Acc(\%) | Latency(ms) |
# |:--:|:--:|:--:|:--:|:--:|:--:|
# | MobileNetV3_Large_x1_25 | 7.4 | 714  | 76.4 | 93.00 | 5.19 |
# | PPLCNetV1_x2_5  | 9 | 906  | 76.60 | 93.00 | 7.25 |
# | <b>PPLCNetV2_base<b>  | <b>6.6<b> | <b>604<b>  | <b>77.04<b> | <b>93.27<b> | <b>4.32<b> |
# | <b>PPLCNetV2_base_ssld<b>  | <b>6.6<b> | <b>604<b>  | <b>80.07<b> | <b>94.87<b> | <b>4.32<b> |
# 
# 在PP-ShiTuV1中，使用PP-LCNetV1_x2_5作为backbone，而PP-ShiTuV2使用了PP-LCNetV2作为Backbone。在不使用额外数据的前提下，PPLCNetV2_base 模型在图像分类 ImageNet 数据集上能够取得超过 77% 的 Top1 Acc，同时在 Intel CPU 平台的推理时间在 4.4 ms 以下，如下表所示，其中推理时间基于 Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz 硬件平台，OpenVINO 推理平台。
# 
# 此外，在PP—LCNetV2_base作为backbone的基础上，我们又加了一些其他的策略，如`Last Stride=1`、`BNNeck`、`ThripletAngularMarginLoss`等，进一步提升模型精度
# 

# # 6. 主体检测模型训练、评测及模型导出
# 
# ## 6.1 主体检测模型训练
# 主体检测模型配置文件位于`PaddleDetection/configs/picodet/application/mainbody_detection/picodet_lcnetv2_base_640_mainbody.yml`，主要内容如下
# ![](https://ai-studio-static-online.cdn.bcebos.com/eb7e6a40904847788a2fea8debf06002ccab79e2a9ae48cd97c21b80399f93b6)
# 用户在实际使用过程中，可以适当修改其中参数，如根据显存的大小，修改`batch_size`等参数，同时不要忘记等比例扩大或者缩小`learning_rate`。
# 同时在实际使用过程中，建议将`pretrain_weights`修改为`https://paddledet.bj.bcebos.com/models/picodet_lcnet_x2_5_640_mainbody.pdparams`，即使用我们训练好的模型进行finetune
# 具体的训练过程请参考[主体检测训练文档](https://github.com/HydrogenSulfate/PaddleClas/blob/refine_ShiTuV2_doc/docs/zh_CN/image_recognition_pipeline/mainbody_detection.md)

# In[3]:


# 解压数据集
get_ipython().run_line_magic('cd', '~/PaddleDetection/dataset/')
get_ipython().run_line_magic('cp', '../../work/data/detection_demo_dataset.tar .')
get_ipython().system('tar -xf detection_demo_dataset.tar')
get_ipython().run_line_magic('cd', 'detection_demo_dataset')
get_ipython().system('ln -s eval.json val.json')
get_ipython().run_line_magic('cd', '..')
get_ipython().system('ln -s detection_demo_dataset mainbody')
get_ipython().run_line_magic('cd', '..')

# 微调coco数据代码
get_ipython().run_line_magic('cd', 'ppdet/data/source')
get_ipython().run_line_magic('cp', '../../../../work/data/coco.py .')
get_ipython().run_line_magic('cd', '../../../')

# 注意，在开始训练的时候，请对yaml文件做好适配，如修改好训练集、测试集的路径
# 此时需要打开终端，修改好yaml文件中对应的配置

# 开始训练，单卡训练
get_ipython().run_line_magic('cd', '~/PaddleDetection')
get_ipython().system('python tools/train.py -c configs/picodet/application/mainbody_detection/picodet_lcnet_x2_5_640_mainbody.yml -o pretrain_weights=https://paddledet.bj.bcebos.com/models/picodet_lcnet_x2_5_640_mainbody.pdparams TrainReader.batch_size=28 epoch=10')

# 如果有多张卡，可以进行多卡训练
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/picodet/application/mainbody_detection/picodet_lcnet_x2_5_640_mainbody.yml  -o pretrain_weights=https://paddledet.bj.bcebos.com/models/picodet_lcnet_x2_5_640_mainbody.pdparams


# ## 6.2 主体检测模型评估
# 训练好主体检测模型后，就需要对其进行评估

# In[4]:


# 由于数据问题，需要修改下代码
get_ipython().run_line_magic('cp', '~/work/cocoeval.py /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/pycocotools/cocoeval.py')
# 单卡GPU评估
get_ipython().system('export CUDA_VISIBLE_DEVICES=0')
get_ipython().system('python tools/eval.py -c configs/picodet/application/mainbody_detection/picodet_lcnet_x2_5_640_mainbody.yml -o weights=output/picodet_lcnet_x2_5_640_mainbody/model_final')


# ## 6.3 主体检测模型导出
# 训练完成后，可以将模型导出为`inference model`，具体操作如下,其中模型导出后，存储在`PaddleDetection/inference_model/picodet_lcnet_x2_5_640_mainbody`下,默认生成文件格式如下
# ```
# infer_cfg.yml
# model.pdiparams
# model.pdiparams.info
# model.pdmodel
# ```
# 将文件经过下面语句重命名后，就可以进行PP-ShiTuV2的部署了

# In[5]:


get_ipython().system('python tools/export_model.py -c configs/picodet/application/mainbody_detection/picodet_lcnet_x2_5_640_mainbody.yml --output_dir=./inference_model -o weights=output/picodet_lcnet_x2_5_640_mainbody/model_final.pdparams')
# 将inference model重新命名，生成变成PP—ShiTu的应用格式
get_ipython().run_line_magic('cd', 'inference_model/picodet_lcnet_x2_5_640_mainbody')
get_ipython().run_line_magic('mv', 'model.pdiparams inference.pdiparams')
get_ipython().run_line_magic('mv', 'model.pdmodel inference.pdmodel')


# # 7 识别模型训练、评估及模型导出
# ## 7.1 识别模型训练
# 
# 识别模型配置文件位于`PaddleClas/ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml`，其中包括了训练所需的所有配置信息，主要有：
# - Global模块：模型存储、日志、训练方式等辅助训练信息
# - AMP：混合精度训练模块
# - Arch：具体模型信息
# 	- Backbone
# 	- Neck
# 	- Head
# - Loss： 损失函数信息
# - Optimizer： 优化器
# - DataLoader： 数据信息
# 	- Train
#     - Eval
#     	- Query
#         - Gallery
# - Metric: 评价指标
# 
# 用户在实际使用过程中，可以适当修改其中参数，如根据显存的大小，修改`batch_size`等参数，同时不要忘记等比例扩大或者缩小`learning_rate`。同时将`Global.pretrained_model`，设置为`https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/PPShiTuV2/general_PPLCNetV2_base_pretrained_v1.0.pdparams`，即在自己的数据集上进行微调。
# 具体模型训练、评估等可以参考[识别模型文档](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/image_recognition_pipeline/feature_extraction.md)

# In[7]:


get_ipython().run_line_magic('cd', '~/PaddleClas/')
# 拷贝好数据集
get_ipython().run_line_magic('cd', 'dataset/')
get_ipython().run_line_magic('cp', '-r ../../work/data/rec_demo_dataset.tar .')
get_ipython().system('tar -xf rec_demo_dataset.tar')
get_ipython().run_line_magic('cp', 'rec_demo_dataset/* .')
get_ipython().run_line_magic('mv', 'rec_demo_dataset Aliproduct')
get_ipython().run_line_magic('cd', '..')

# 打开终端，修改好yaml文件中对应训练、测试数据集路径

# 单卡训练
get_ipython().run_line_magic('cd', '~/PaddleClas')
get_ipython().run_line_magic('cp', '~/work/GeneralRecognitionV2_PPLCNetV2_base.yaml ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml')
get_ipython().system('export CUDA_VISIBLE_DEVICES=0')
get_ipython().system('python tools/train.py -c ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml -o Global.pretrained_model=https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/PPShiTuV2/general_PPLCNetV2_base_pretrained_v1.0.pdparams -o DataLoader.Train.sampler.batch_size=64 -o Global.epochs=1')

# 多卡训练
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml -o Global.pretrained_model=https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/PPShiTuV2/general_PPLCNetV2_base_pretrained_v1.0.pdparams


# ## 7.2 识别模型评测

# In[8]:


# 单卡GPU评估
get_ipython().system('export CUDA_VISIBLE_DEVICES=0')
get_ipython().system('python tools/eval.py -c ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml -o Global.pretrained_model=output/RecModel/best_model')


# ## 7.3 识别模型导出inference model
# 模型导出后，默认存储在`PaddleClas/inference`下，生成文件如下：
# ```
# inference.pdiparams
# inference.pdiparams.info
# inference.pdmodel
# ```
# 生成inference model后，就可以进行PP-ShiTu部署使用了

# In[9]:


get_ipython().system('python tools/export_model.py -c ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml -o Global.pretrained_model=output/RecModel/best_model')


# # 8 模型部署
# 此项目最终部署在rk3566开发板上, 其搭载了A55架构arm处理器，同时搭载G52图形处理器，能够支持NPU加速。基于arm的部署可以体验PP-ShiTuV2 的安卓demo，可以下载体验，预装饮商品检索库，同时支持加库等操作
# ![](https://ai-studio-static-online.cdn.bcebos.com/778bc86d53b446e38e47ac65ba26489fadb66ad35f26474ea35ba1f6488d3434)
# 
# 在本此展示中，以python inference 为例，展示整个PipeLine的运行结果，具体请参考[python inference部署文档](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/inference_deployment/python_deploy.md#2)
# 实际部署过程中，为了达到更好的速度，可以使用C++inference 部署，请参考[PP-ShiTu C++ inference 部署文档](https://github.com/PaddlePaddle/PaddleClas/blob/develop/deploy/cpp_shitu/readme.md)

# In[ ]:


get_ipython().run_line_magic('cd', '~/PaddleClas/deploy/')

# 下载相应的inference model
get_ipython().run_line_magic('mkdir', 'models')
get_ipython().run_line_magic('cd', 'models')
get_ipython().system('wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar')
get_ipython().system('tar -xf picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar')
get_ipython().system('wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/PP-ShiTuV2/general_PPLCNetV2_base_pretrained_v1.0_infer.tar')
get_ipython().system('tar -xf general_PPLCNetV2_base_pretrained_v1.0_infer.tar')
get_ipython().run_line_magic('cd', '..')

# 下载对应的数据集
get_ipython().system('wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/PP-ShiTuV2_application_dataset/Veg200.tar')
get_ipython().system('tar -xf Veg200.tar')

# 下载的数据集中已经生成了对应的index索引文件
# 如需重新生成对应的index文件 请参考8.1


# ## 8.1 生成index文件
# 在上述代码中，已经生成好了对应的检索库文件，用户如果想要重新生成库文件，需要对配置文件进行修改适配，配置文件位于`~/work/PaddleClas/deploy/configs/inference_general.yaml`
# 首先需修改模型路径部分，修改如下：
# ```
# Global:
#   infer_imgs: './Veg/Query'
#   det_inference_model_dir: './models/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer'
#   rec_inference_model_dir: './models/general_PPLCNetV2_base_pretrained_v1.0_infer'
# ```
# 另外Index等参数配置如下：
# ```
# IndexProcess:
#   index_method: "HNSW32" # supported: HNSW32, IVF, Flat
#   image_root: "./Veg200/Gallery/"
#   index_dir: "./Veg200/Index"
#   data_file: "./Veg200/gallery_list.txt"
#   index_operation: "new" # suported: "append", "remove", "new"
#   delimiter: " "
#   dist_type: "IP"
#   embedding_size: 512
#   batch_size: 32
#   return_k: 5
#   score_thres: 0.5
# ```
# 需要修改的部分包括：
# - `image_root`: 库图像存储目录
# - `index_dir`: 生成index的存储目录
# - `data_file`: 图像的list文件
# - `delimiter`: 图像list文件中每行的分隔符。根据不同list修改
# 
# 如需重新生成index文件，则需如下操作

# In[ ]:


# 进行操作目录
get_ipython().run_line_magic('cd', 'd ~/work/PaddleClas/deploy/')
# 生成index文件
get_ipython().system('python python/build_gallery.py -c configs/inference_general.yaml')


# ## 8.2 推理部署
# 生成好对应的index文件后，可进行推理部署，代码如下

# In[ ]:


# 此代码是将文件夹下所有的图像进行预测
get_ipython().system('python python/predict_system.py -c configs/inference_general.yaml -o Global.infer_imgs=Veg200/Query/')

# 如果只是预测单张图像，则可以如下操作
get_ipython().system('python python/predict_system.py -c configs/inference_general.yaml -o Global.infer_imgs=Veg200/Query/allium_v_15_04_0019.jpg')


# # 9 检索库调优
# 在PP-ShiTu中，挑选具有代表性的检索库，是非常重要的，这对模型的效果有着巨大的影响，因此用户可以通过手动调节生成index库的图像，来进一步调节模型精度。
# 在PP-ShiTuV2中，提供了可视化的库管理工具，具体可以查看库[管理工具文档](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/inference_deployment/shitu_gallery_manager.md)
# 其中界面如下：
# ![](https://ai-studio-static-online.cdn.bcebos.com/7b6d4cf6c88c49d99b073fdbb1a08c7cf31aa8905c1846b899ff08b28af1371a)
# 注意：此工具需要在有界面的系统中打开，在AiStudio中无法展示。
# 用户可以根据对图像进行增删改查，制作高质量的index库，提升模型效果。
# 
# # 10 总结
# 
# 本案例旨在通过一个实际的项目带大家熟悉从数据集建立、到模型训练优化、到最终部署的完整的快速落地流程，并在这个过程中了解分析问题、解决问题的思路。结合飞桨提供的全生态的AI套件提升项目的实践体验和效率。
# 值得一提的是，在数据准备阶段，需要的是耐心。在模型优化阶段需要的是大胆的尝试。在部署阶段需要的是细心。希望这些经验能对大家今后的实际项目操作有所帮助。
# 具体方案链接：https://gitee.com/yingeo-pro/yingeo-ai
# 
# 
# # 11 企业介绍
# 
# 深圳市银歌云技术有限公司
# 深圳市银歌云技术有限公司是国内商业信息化、数字门店、数字餐厅及智能化设备方案提供商与践行者，专注零售餐饮信息化创新应用，系统整合了商业大数据、智能设备、生态流量，推出银歌云商计划。用智能技术“新引擎”赋能零售商，以科技服务市场，AI视觉结算、刷脸支付、自助收银、会员画像、AI互动等技术产品带给消费者全新的购物体验，塑造科技零售美好生活。
# 
# 公司官网：https://www.yingeo.com/
# 
