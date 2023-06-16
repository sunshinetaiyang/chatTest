#!/usr/bin/env python
# coding: utf-8

# # 第二届广州·琶洲算法大赛-基于复杂场景的输电通道隐患目标检测算法baseline（非官方）切图版本，后处理prob>0.5过滤后73分
# 
# 

# In[2]:


# 训练环境准备
# !git clone https://gitee.com/paddlepaddle/PaddleDetection.git # 23.5.23仅需run一次


# ## 4 数据预处理
# 
# 

# ### 4.1 解压数据集
# 

# In[1]:


# 解压数据集
# !unzip -oq ~/data/data212110/train.zip -d ~/PaddleDetection/dataset/voc
# !unzip -oq ~/data/data212110/val.zip -d ~/PaddleDetection/dataset/voc
# 23.5.26 以下unzip用来观察分析原始数据库
# ! rm -rf ~/data/CHECK
# !unzip -oq ~/data/data212110/train.zip -d ~/data/CHECK
# !unzip -oq ~/data/data212110/val.zip -d ~/data/CHECK
# !unzip -l ~/data/data212110/val.zip


# In[5]:


# # 将标注和图片分开
# %cd ~/PaddleDetection/dataset/voc
# !mkdir JPEGImages Annotations
# !cp -r train/*.xml Annotations
# !cp -r train/*.jpg JPEGImages


# In[6]:


# !rm -rf /home/aistudio/PaddleDetection/dataset/voc/train
# !rm -rf /home/aistudio/PaddleDetection/dataset/voc/val


# ### 4.2 划分数据集

# In[8]:


# 23.5.26 划分train/val数据集，都有标注，train图片720张，val图片80张
# 生成文件trainval.txt和val.txt以及标签类比文件label_list.txt
# import random
# import os
# #生成trainval.txt和val.txt
# random.seed(2020)
# xml_dir  = '/home/aistudio/PaddleDetection/dataset/voc/Annotations'#标签文件地址
# img_dir = '/home/aistudio/PaddleDetection/dataset/voc/JPEGImages'#图像文件地址
# path_list = list()
# for img in os.listdir(img_dir):
#     img_path = os.path.join(img_dir,img)
#     xml_path = os.path.join(xml_dir,img.replace('jpg', 'xml'))
#     path_list.append((img_path, xml_path))
# random.shuffle(path_list)
# ratio = 0.9
# train_f = open('/home/aistudio/PaddleDetection/dataset/voc/trainval.txt','w') #生成训练文件
# val_f = open('/home/aistudio/PaddleDetection/dataset/voc/val.txt' ,'w')#生成验证文件

# for i ,content in enumerate(path_list):
#     img, xml = content
#     text = img + ' ' + xml + '\n'
#     if i < len(path_list) * ratio:
#         train_f.write(text)
#     else:
#         val_f.write(text)
# train_f.close()
# val_f.close()

# #生成标签文档
# label = ['nest', 'kite', 'balloon', 'trash']#设置你想检测的类别
# with open('/home/aistudio/PaddleDetection/dataset/voc/label_list.txt', 'w') as f:
#     for text in label:
#         f.write(text+'\n')


# ### 4.3 环境准备
# 
# 由于PP-YOLOE还在快速迭代中，因此，对框架的稳定性有一定的要求，PaddlePaddle的框架不要选择最新版。本文使用的单卡训练环境如下：
# 
# * 框架版本：PaddlePaddle 2.4.0
# 
# * CUDA Version: 11.2
# 
# * 模型库版本：PaddleDetection(release/2.5分支)
# 

# ### 4.4 VOC2COCO
# 
# 若需要COCO数据集格式，可以运行以下命令进行转换

# In[3]:


# # 23.5.26 因为原始数据中xml文件的<filename>字段无意义，所以要修改为正确的图片名.jpg
# import os
# import os.path
# import xml.dom.minidom
# path = r'/home/aistudio/PaddleDetection/dataset/voc/Annotations'
# files = os.listdir(path)  # 得到文件夹下所有文件名称
# s = []
# count = 0
# for xmlFile in files:  # 遍历文件夹
#     if not os.path.isdir(xmlFile):  # 判断是否是文件夹,不是文件夹才打开
#             name1 = xmlFile.split('.')[0]
#             dom = xml.dom.minidom.parse(path + '/' + xmlFile)
#             root = dom.documentElement
#             newfolder = root.getElementsByTagName('folder')
#             newpath = root.getElementsByTagName('path')
#             newfilename = root.getElementsByTagName('filename')
#             newfilename[0].firstChild.data = name1 + '.jpg'
#             with open(os.path.join(path, xmlFile), 'w') as fh:
#                 dom.writexml(fh)
#                 print('写入成功')
#             count = count + 1


# In[10]:


# # 将训练集转换为COCO格式，生成train.json文件
# %cd ~/PaddleDetection
# !python tools/x2coco.py \
#         --dataset_type voc \
#         --voc_anno_dir dataset/voc/Annotations/ \
#         --voc_anno_list dataset/voc/trainval.txt \
#         --voc_label_list dataset/voc/label_list.txt \
#         --voc_out_name dataset/voc/train.json


# In[11]:


# # 将验证集转换为COCO格式，生成coco格式的标注文件val.json
# %cd ~/PaddleDetection
# !python tools/x2coco.py \
#         --dataset_type voc \
#         --voc_anno_dir dataset/voc/Annotations/ \
#         --voc_anno_list dataset/voc/val.txt \
#         --voc_label_list dataset/voc/label_list.txt \
#         --voc_out_name dataset/voc/val.json


# ### 4.5 统计数据集分布
# 
# 在切图前，我们首先需要统计所用数据集标注框的平均宽高占图片真实宽高的比例分布：
# 
# * --json_path ：待统计数据集COCO 格式 annotation 的json文件路径
# 
# * --out_img ：输出的统计分布图路径

# In[1]:


# # # 23.5.23 安装pycocotools
# !mkdir /home/aistudio/external-libraries
# !pip install beautifulsoup4 -t /home/aistudio/external-libraries
get_ipython().system('pip install pycocotools ')
# !pip install pycocotools  -t /home/aistudio/external-libraries
# !pip install -U scikit-image imagecodecs  -t /home/aistudio/external-libraries


# ### 4.6 分布研究
# 6.15 
# Mean of all img_w is 2895.152777777778
# Mean of all img_h is 2061.6291666666666
# Median of ratio_w is 0.2673318275939969 # 由于是大小混合，全部使用同一模型的效率低，改进为分开训
# Median of ratio_h is 0.3123932133447108
# all_img with box:  720
# all_ann:  723

# In[2]:


# 6.15 单独对nest类数据进行distribution统计
import json

# 读取train.json文件并解析内容
with open('/home/aistudio/PaddleDetection/dataset/voc/train.json', 'r') as f:
    data = json.load(f)

# 获取annotations和images列表
annotations = data['annotations']
images = data['images']

# 将images列表转换为字典，以image_id作为键
image_dict = {image['id']: image for image in images}

# 筛选出category_id为1的annotations和对应的images内容
filtered_annotations = []
filtered_images = {}

for annotation in annotations:
    if annotation['category_id'] == 1:
        filtered_annotations.append(annotation)
        image_id = annotation['image_id']
        filtered_images[image_id] = image_dict[image_id]

# 创建新的数据字典，并将filtered_annotations和filtered_images存入其中
filtered_data = {'images': list(filtered_images.values()), 'annotations': filtered_annotations}

# 将filtered_data转换为JSON格式并写入新文件
with open('/home/aistudio/PaddleDetection/dataset/voc/nest_train.json', 'w') as f:
    json.dump(filtered_data, f)


# In[3]:


get_ipython().run_line_magic('cd', '~/PaddleDetection')

# 统计数据集分布
get_ipython().system('python tools/box_distribution.py     --json_path dataset/voc/nest_train.json     --out_img /home/aistudio/PaddleDetection/dataset/voc/nest_box_distribution.jpg')


# nest类的图片distribution结果：
# Suggested reg_range[1] is 16
# Mean of all img_w is 3834.351966873706
# Mean of all img_h is 2734.304347826087
# Median of ratio_w is 0.22495527728085868 # 居然也有0.225这么高，基本上没有切分的必要
# Median of ratio_h is 0.23942652329749103
# all_img with box:  483
# all_ann:  483
# Distribution saved as /home/aistudio/PaddleDetection/dataset/voc/nest_box_distribution.jpg
# Figure(640x480)

# ### 4.6 基于SAHI切图
# 
# 针对需要切图的数据集，使用SAHI库进行切分：
# 
# * --image_dir：原始数据集图片文件夹的路径
# 
# * --json_path：原始数据集COCO格式的json标注文件的路径
# 
# 
# * --output_dir：切分后的子图及其json标注文件保存的路径
# 
# * --slice_size：切分以后子图的边长尺度大小(默认切图后为正方形)
# 
# * --overlap_ratio：切分时的子图之间的重叠率
# 
# 以上述代码为例，切分后的子图文件夹与json标注文件共同保存在MyDataset/IMG_sliced文件夹下，比如训练集图片和标注就命名为train_images_640_025、train_images_640_025.json

# In[15]:


# !mkdir dataset/voc/IMG_sliced


# In[2]:


# !pip install -U scikit-image imagecodecs


# In[17]:


# # 对训练集标注进行切图
# !python tools/slice_image.py \
#         --image_dir /home/aistudio/PaddleDetection/dataset/voc/JPEGImages \
#         --json_path /home/aistudio/PaddleDetection/dataset/voc/train.json \
#         --output_dir /home/aistudio/PaddleDetection/dataset/voc/IMG_sliced \
#         --slice_size 640 \
#         --overlap_ratio 0.25


# In[18]:


# # 对验证集标注进行切图
# !python tools/slice_image.py \
#         --image_dir /home/aistudio/PaddleDetection/dataset/voc/JPEGImages \
#         --json_path /home/aistudio/PaddleDetection/dataset/voc/val.json \
#         --output_dir /home/aistudio/PaddleDetection/dataset/voc/IMG_sliced \
#         --slice_size 640 \
#         --overlap_ratio 0.25


# ## 5 模型训练
# 
# 由于图像尺寸较大，且训练模型时，读取图像和预处理时耗时较长，因此我们选择拼图模型进行训练。

# ### 5.1 原图模型训练
# 
# PP-YOLOE+_l在COCO test-dev2017达到了53.3的mAP, 同时其速度在Tesla V100上达到了78.1 FPS。PP-YOLOE+_s/m/x同样具有卓越的精度速度性价比, 其精度速度可以在模型库中找到。

# In[19]:


# !cp -r /home/aistudio/work/eda /home/aistudio/PaddleDetection/
# !cp -r /home/aistudio/work/output /home/aistudio/PaddleDetection/


# In[6]:


# %cd ~/PaddleDetection
# !python setup.py install


# In[8]:


get_ipython().system('cat eda/ppyoloe/ppyoloe_plus_crn_s_80e_eda.yml')


# In[3]:


# 23.5.24 修改了datasets/eda_detection.yml到正确的大写PaddleDetection目录，但是内存不够挂死
get_ipython().run_line_magic('cd', '~/PaddleDetection')
get_ipython().system('python tools/train.py -c eda/ppyoloe/ppyoloe_plus_crn_s_80e_eda.yml --use_vdl=True --vdl_log_dir=./ori_log --eval --amp')


# ### 5.2 拼图模型选型
# PaddleDetection团队提供的基于PP-YOLOE的检测模型，以及提供了一套使用SAHI(Slicing Aided Hyper Inference)工具切图和拼图的方案，其效果如下：
# 
# |模型|	数据集|	SLICE_SIZE	|OVERLAP_RATIO	|类别数	|mAP 0.5:0.95	|AP 0.5	|
# | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
# |PP-YOLOE-l|	VisDrone-DET|	640|	0.25|	10	|29.7	|48.5|	
# |PP-YOLOE-l (Assembled)	|VisDrone-DET	|640	|0.25|	10	|37.2	|59.4|	
# 
# Assembled表示自动切图和拼图后模型的表现，从中我们可以看出，mAP较原图预测有了非常显著的提升，因此，接下来就基于PP-YOLOE-l，看看自动切图和拼图后模型在这个比赛检测数据集上的表现。

# ### 5.3 模型训练
# 
# 切图后模型的训练是要基于切图数据集的，配置如下：
# 
# ```
# metric: COCO
# num_classes: 4
# 
# TrainDataset:
#   !COCODataSet
#     image_dir: /home/aistudio/paddledetection/dataset/voc/IMG_sliced/train_images_640_025
#     anno_path: /home/aistudio/paddledetection/dataset/voc/IMG_sliced/train_640_025.json
#     dataset_dir: /home/aistudio/paddledetection/dataset/voc
#     data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']
# 
# EvalDataset:
#   !COCODataSet
#     image_dir: /home/aistudio/paddledetection/dataset/voc/IMG_sliced/val_images_640_025
#     anno_path: /home/aistudio/paddledetection/dataset/voc/IMG_sliced/val_640_025.json
#     dataset_dir: /home/aistudio/paddledetection/dataset/voc
# 
# TestDataset:
#   !ImageFolder
#     anno_path: val_640_025.json
#     dataset_dir: /home/aistudio/paddledetection/dataset/voc/IMG_sliced
# 
# ```
# 

# In[3]:


# 23.5.25 分析拼图模型并尝试运行
get_ipython().run_line_magic('cd', '~/PaddleDetection')
get_ipython().system('cat eda/eda_model/ppyoloe_crn_l_80e_sliced_eda.yml')


# In[5]:



get_ipython().system('python tools/train.py -c eda/eda_model/ppyoloe_crn_l_80e_sliced_eda.yml --use_vdl=True --vdl_log_dir=./log --eval ')


# ### 5.4 模型评估
# 我们对训练80个epoch后，切图模型的效果进行一下评估。
# 
# #### 5.4.1 子图评估
# 
# 对于子图评估和原图评估，差别仅仅在于验证集路径的配置：
# 
# * 子图评估
# 
#   配置切图后的子图存放目录和子图验证集标注文件
#   
# * 原图评估
# 
#   配置原图存放目录和验证集标注文件

# In[6]:


# 训练80个epoch后，子图评估
get_ipython().run_line_magic('cd', '~/PaddleDetection')
get_ipython().system('python tools/eval.py     -c eda/eda_model/ppyoloe_crn_l_80e_sliced_eda.yml     -o weights=output/ppyoloe_crn_l_80e_sliced_eda/best_model.pdparams')


# In[ ]:


# 训练80个epoch后，原图评估
get_ipython().system('python tools/eval.py         -c eda/eda_model/ppyoloe_crn_l_80e_sliced_eda.yml         -o weights=output/ppyoloe_crn_l_80e_sliced_eda/best_model.pdparams')


# #### 5.4.2 子图拼图评估
# 
# 执行命令时，需要关注下面这些参数的设置：
# 
# * 设置--combine_method表示子图结果重组去重的方式，默认是nms；
# 
# * 设置--match_threshold表示子图结果重组去重的阈值，默认是0.6；
# 
# * 设置--match_metric表示子图结果重组去重的度量标准，默认是ios表示交小比(两个框交集面积除以更小框的面积)，也可以选择交并比iou(两个框交集面积除以并集面积)，精度效果因数据集而而异，但选择ios预测速度会更快一点。
# 
# 在本项目中，我们把交小比和交并比两种方式都试验一番：

# In[8]:


# 训练80个epoch后，子图拼图评估，交小比
get_ipython().system('python tools/eval.py     -c eda/eda_model/ppyoloe_crn_l_80e_sliced_eda_slice_infer.yml     -o weights=output/ppyoloe_crn_l_80e_sliced_eda/best_model.pdparams     --slice_infer     --combine_method=nms     --match_threshold=0.6     --match_metric=ios')


# In[3]:



get_ipython().run_line_magic('cd', '~/PaddleDetection')

# 训练80个epoch后，子图拼图评估，交并比
get_ipython().system('python tools/eval.py     -c eda/eda_model/ppyoloe_crn_l_80e_sliced_eda_slice_infer.yml     -o weights=output/ppyoloe_crn_l_80e_sliced_eda/best_model.pdparams     --slice_infer     --combine_method=nms     --match_threshold=0.6     --match_metric=iou')


# #### 5.4.3 评估结论
# 
# 将80个epoch后，切图训练子图评估、切图训练原图评估、切图训练子图拼图评估的效果进行了对比，结果如下：
# 
# |训练评估方式	|SLICE_SIZE	|OVERLAP_RATIO|	mAP0.5:0.95	|AP0.5|
# | :--------: | :--------: | :--------: |:--------: |:--------: |
# |子图训练子图评估|640	|0.25	|78.0	|94.8|
# |子图训练原图评估|640	|0.25	|69.3	|81.5|
# |子图训练拼图评估-IoS|640	|0.25	|47.6	|66.8|
# |子图训练拼图评估-IoU|640	|0.25	|47.1	|71.1|
# 
# 从上面的简单表格可以明显看出，相比原图直接训练，子图训练拼图评估精度提升明显。同时，子图结果重组去重的度量标准用交并比在该数据集上表现更好，猜测可能是因为，数据集的目标大小差异较大，对于中大型目标，去重标准如果用交小比，在评估效果上，会比较吃亏。

# ### 5.5 模型预测

# #### 5.5.1 单图预测

# ib9s83aGpq5PXOW41vRBNKemHylw6JFcLgMSxD7h.jpg

# In[18]:


# 挑一张测试集的图片展示预测效果
get_ipython().system('python tools/infer.py     -c eda/eda_model/ppyoloe_crn_l_80e_sliced_eda.yml     -o weights=output/ppyoloe_crn_l_80e_sliced_eda/best_model.pdparams     --infer_img=/home/aistudio/PaddleDetection/dataset/voc/val/ib9s83aGpq5PXOW41vRBNKemHylw6JFcLgMSxD7h.jpg     --draw_threshold=0.4     --output_dir test_one_infer_output')
    # --slice_infer \
    # --slice_size 640 640 \
    # --overlap_ratio 0.25 0.25 \
    # --combine_method=nms \
    # --match_threshold=0.6 \
    # --match_metric=iou \
    # --save_results=True


# #### 5.5.2 批量预测
# 

# In[4]:


# 执行批量预测
get_ipython().system('python tools/infer.py     -c eda/eda_model/ppyoloe_crn_l_80e_sliced_eda.yml     -o weights=output/ppyoloe_crn_l_80e_sliced_eda/best_model.pdparams     --infer_dir=/home/aistudio/PaddleDetection/dataset/voc/val     --save_results=True')


# ![](https://ai-studio-static-online.cdn.bcebos.com/6a9d5cc3f03847d38f44e91e27abf361a87f29bc0ebd4a1e99c39580ca104de6)
# 

# ### 5.6 模型导出
# 
# 

# In[5]:


get_ipython().run_line_magic('cd', '~/PaddleDetection')
get_ipython().system('python tools/export_model.py     -c eda/eda_model/ppyoloe_crn_l_80e_sliced_eda.yml     --output_dir=./inference_model     -o weights=output/ppyoloe_crn_l_80e_sliced_eda/best_model')


# ### 5.7 结果文件生成

# In[26]:


# deploy slice infer
# 单张图
# 6.12 无人机项目预测命令
get_ipython().run_line_magic('cd', '~/PaddleDetection')
get_ipython().system('CUDA_VISIBLE_DEVICES=0 ')
get_ipython().system(' python deploy/python/infer.py --model_dir=inference_model/ppyoloe_crn_l_80e_sliced_eda --image_dir=/home/aistudio/PaddleDetection/dataset/voc/val --device=GPU --threshold=0.5  --slice_infer --slice_size 640 640 --overlap_ratio 0.25 0.25 --combine_method=concat --match_threshold=0.2 --match_metric=iou --output_dir /home/aistudio/work/test_sliced_output --save_results')
# ! python deploy/python/infer.py --model_dir=output_inference/ppyoloe_crn_l_80e_sliced_vi --image_dir=demo/                                          --device=GPU --save_images --threshold=0.25  --slice_infer --slice_size 640 640 --overlap_ratio 0.25 0.25 --combine_method=nms --match_threshold=0.6 --match_metric=ios


# In[ ]:


get_ipython().system('mkdir submit/')
get_ipython().system('mv /home/aistudio/work/demo/deploy_output/bbox.json submit/')


# In[1]:


get_ipython().run_line_magic('cd', '~/PaddleDetection')
# !cp -r /home/aistudio/work/infer.py /home/aistudio/PaddleDetection/deploy/python/
get_ipython().system('python deploy/python/infer.py     --model_dir=inference_model/ppyoloe_crn_l_80e_sliced_eda     --image_dir=/home/aistudio/PaddleDetection/dataset/voc/val     --device=GPU     --output_dir infer_output     --save_results')
get_ipython().system('mkdir submit/')
get_ipython().system('mv infer_output/bbox.json submit/')


# In[2]:


# 23.5.27 观察输出数据
import json

# 读取原始文件内容
with open('submit/bbox.json', 'r') as f:
    data = f.read()

# 替换并分行
formatted_data = data.replace('},', '},\n')

# 写入新文件
with open('submit/b_lines.json', 'w') as f:
    f.write(formatted_data)
    print('完成b_lines.json')


# ## 6 提分技巧
# 
# * 针对本赛题的数据集特点，可以选用更不同尺度的图片切片大小，进行模型训练。
# 
# * 针对训练环境适当调整部分参数，以提升模型精度。
# 
# * 选择精度更高的目标检测模型或适当优化模型的部分结构。
# 

# ## 7 小结
# 
# 模型还支持GPU上的推理部署，对于时间要求不高、精度要求极高的大图目标检测场景，具有良好的潜在应用价值。

# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
