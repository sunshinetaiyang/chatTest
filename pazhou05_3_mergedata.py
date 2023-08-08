#!/usr/bin/env python
# coding: utf-8

# # 一、环境准备

# In[1]:


# # 克隆PaddleDetection仓库
# %cd /home/aistudio/
# !git clone -b develop https://gitee.com/PaddlePaddle/PaddleDetection.git


# In[1]:


# 安装其他依赖
get_ipython().run_line_magic('cd', '/home/aistudio/PaddleDetection/')
get_ipython().system('pip install -r requirements.txt --user')

# 编译安装paddledet
get_ipython().system('python setup.py install')

# 安装后确认测试通过：23.5.29
get_ipython().system('python ppdet/modeling/tests/test_architectures.py')


# # 二、数据准备

# In[3]:


# # 23.6.9 准备后台运行，只运行一次
# !unzip -oq ~/data/data212110/val.zip -d ~/PaddleDetection/dataset/grid_coco
# !unzip -oq ~/data/data212110/train.zip -d ~/PaddleDetection/dataset/grid_coco
# # val目录中300张照片，未标注，用来infer提交result打榜；train中800张照片，800个xml，用于训练

# %cd ~/PaddleDetection/dataset/grid_coco/train/
# !mkdir JPEGImages
# !mkdir Annotations
# !mv *.jpg JPEGImages/
# !mv *.xml Annotations/


# In[4]:


# # 解决原数据filename问题
# import os
# from xml.etree import ElementTree as ET

# # 把所有PaddleDetection2.5/dataset/coco 换成新目录 PaddleDetection/dataset/grid_coco
# # 标注文件夹路径
# annotation_dir = '/home/aistudio/PaddleDetection/dataset/grid_coco/train/Annotations'

# def fix_filename(annotation_dir):
#     # 遍历标注文件夹，修改文件名
#     for filename in os.listdir(annotation_dir):
#         xml_path = os.path.join(annotation_dir, filename)
#         tree = ET.parse(xml_path)
#         root = tree.getroot()
#         annotation_filename = os.path.splitext(filename)[0] + '.jpg'
#         root.find('filename').text = annotation_filename

#         # 保存修改后的标注文件
#         tree.write(xml_path)


# # 三、训练

# In[ ]:


# 6.15 使用ppyoloe训练试试
get_ipython().run_line_magic('cd', '~/PaddleDetection')
# !export CUDA_VISIBLE_DEVICES=0
# !python tools/train.py -c configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml --eval 
# -r output/ppyoloe_plus_crn_x_80e_coco/4 
#  --amp # --amp可能降低精度，暂不使用
# !export FLAGS_use_cuda_managed_memory=false
get_ipython().system('export CUDA_VISIBLE_DEVICES=0,1,2,3')
get_ipython().system('python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml --fleet --eval -r output/ppyoloe_plus_crn_x_80e_coco/67')


# # 评估

# ## 分图片进行评估的json构造

# In[11]:


import json  
  
def filter_data(base_json_file, id, new_json_file):  
    with open(base_json_file, 'r') as f:  
        data = json.load(f)  
  
    # 保留 "type" 字段和 "categories"  
    filtered_data = {  
        "images": [image for image in data['images'] if image['id'] == id], 
        "type": data["type"],   
        "annotations": [annotation for annotation in data['annotations'] if annotation['image_id'] == id],
        "categories": data["categories"],
    }  

    with open(new_json_file, 'w') as f:  
        json.dump(filtered_data, f)  
  
    print(f"Filtered data saved to {new_json_file}")

base_json_file = '/home/aistudio/PaddleDetection/dataset/grid_coco/train/grid_val.json'
img_id = 16
new_json_file = '/home/aistudio/PaddleDetection/dataset/grid_coco/train/grid_val_id' + str(img_id) + '.json'
filter_data(base_json_file, img_id, new_json_file)


# ## 评估执行

# In[12]:



# 6.16 使用ppyoloe训练结果评估
get_ipython().run_line_magic('cd', '~/PaddleDetection')
get_ipython().system('python tools/eval.py -c configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml  -o weights=output/ppyoloe_plus_crn_x_80e_coco/best_model.pdparams  --classwise  --output_eval eval_output')


# # 导出和推理

# In[2]:



# 6.16 使用ppyoloe导出模型
get_ipython().run_line_magic('cd', '~/PaddleDetection')
# !python tools/export_model.py -c configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml -o weights=output/ppyoloe_plus_crn_x_80e_coco/best_model
get_ipython().system('python tools/export_model.py -c configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml -o weights=output/ppyoloe_plus_crn_x_80e_coco/best_model')


# In[4]:


# 6.16 ppyoloe+ 的推理
get_ipython().run_line_magic('cd', '/home/aistudio/PaddleDetection/')
get_ipython().system('python deploy/python/infer.py --model_dir=output_inference/ppyoloe_plus_crn_x_80e_coco --image_dir=/home/aistudio/PaddleDetection/dataset/grid_coco/val --device=GPU --output_dir infer_output_grid_depoly --save_results')


# # 不同category不同的threshold过滤

# In[2]:


# 6.20 yiyan的代码
import json  
  
def filter_data(base_json, threshold_list, new_json):  
    # 读取base_json文件  
    with open(base_json, 'r') as f:  
        data = json.load(f)  
  
    # 遍历数据并过滤  
    filtered_data = []  
    id = 0  # 初始化id为0  
    for item in data:  
        if item['category_id'] == 1:  
            if item['score'] > threshold_list[0]:  
                item['id'] = id  # 设置id为递增的值  
                id += 1  
                filtered_data.append(item)  
        elif item['category_id'] == 2:  
            if item['score'] > threshold_list[1]:  
                item['id'] = id  
                id += 1  
                filtered_data.append(item)  
        elif item['category_id'] == 3:  
            if item['score'] > threshold_list[2]:  
                item['id'] = id  
                id += 1  
                filtered_data.append(item)  
        elif item['category_id'] == 4:  
            if item['score'] > threshold_list[3]:  
                item['id'] = id  
                id += 1  
                filtered_data.append(item)  
  
    # 将过滤后的数据写入到new_json文件中  
    with open(new_json, 'w') as f:  
        json.dump(filtered_data, f)
        print('after threshold json saved!')


# base_json = '/home/aistudio/post_processing/720epoch_best/bbox.json'  
base_json = '/home/aistudio/PaddleDetection/eval_output/bbox.json'
threshold_list = [0.5, 0.4, 0.4, 0.3]  
# new_json = '/home/aistudio/post_processing/720epoch_best/720new.json'  
new_json = '/home/aistudio/PaddleDetection/eval_output/bbox_new.json'
filter_data(base_json, threshold_list, new_json)


# In[2]:


# 6.20 chat的代码
import json

def bbox_filter_scores(json_file, thresh_list, out_file):
    # 读取bbox.json文件
    with open(json_file, 'r') as file:
        bbox_data = json.load(file)

    filtered_data = []
    bbox_count = {}  # 用于存储每个image_id对应的bbox数目
    image_ids = set()  # 用于收集所有不相同的image_id
    zero_count_ids = []  # 存储bbox数目为0的image_id列表
    multiple_count_ids = []  # 存储bbox数目大于1的image_id列表

    for bbox in bbox_data:
        image_id = bbox['image_id']
        image_ids.add(image_id)  # 将image_id添加到集合中

        category_id = bbox['category_id']
        score = bbox['score']
        thresh = thresh_list[category_id - 1]  # 因为索引是从0开始，所以需要减去1
        if score >= thresh:
            filtered_data.append(bbox)

            # 统计每个image_id对应的bbox数目
            if image_id in bbox_count:
                bbox_count[image_id] += 1
            else:
                bbox_count[image_id] = 1

    with open(out_file, 'w') as file:
        json.dump(filtered_data, file)

    print(f"Filtered data has been written to {out_file}.")

    # 输出过滤后的每个image_id对应的bbox数目
    print("Bbox counts per image_id (after filtering):")
    for i, image_id in enumerate(sorted(image_ids), start=1):
        count = bbox_count.get(image_id, 0)
        # print(f"No{i}: Image ID: {image_id}, Bbox Count: {count}")

        if count == 0:
            zero_count_ids.append(image_id)
        elif count > 1:
            multiple_count_ids.append(image_id)

    print("Image IDs with bbox count 0:")
    print(zero_count_ids)

    print("Image IDs with bbox count > 1:")
    print(multiple_count_ids)


json_file = '/home/aistudio/post_processing/720epoch_best/bbox.json'  
# base_json = '/home/aistudio/PaddleDetection/eval_output/bbox.json'
thresh_list = [0.5, 0.4, 0.4, 0.3]  
out_file = '/home/aistudio/post_processing/720epoch_best/720new.json' 


# json_file = '/home/aistudio/post_processing/704FirstBigData/bbox.json'
# 设置阈值列表
# thresh_list = [0.5, 0.5, 0.2, 0.2]
# 将过滤后的数据写入到输出文件
# out_file = '/home/aistudio/post_processing/704FirstBigData/filtered_bbox.json'
bbox_filter_scores(json_file, thresh_list, out_file)


# In[ ]:





# In[4]:


# 7.1 格式修正，太牛逼了，没想到一句话可以搞定
import json

def format_josn_for_easydata(src_json, des_json):
    # 读取JSON文件
    with open(src_json, 'r') as file:
        data = json.load(file)

    # 格式化数据
    formatted_data = json.dumps(data, indent=2)

    # 保存修改后的JSON文件
    with open(des_json, 'w') as file:
        file.write(formatted_data)

# src_json='/home/aistudio/epoch92_05_imgs_anno.json'
src_json = '/home/aistudio/PaddleDetection/dataset/grid_coco/train/grid_val.json'
des_json='/home/aistudio/PaddleDetection/dataset/grid_coco/train/grid_val_indent.json'
format_josn_for_easydata(src_json, des_json)


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
