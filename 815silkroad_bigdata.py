#!/usr/bin/env python
# coding: utf-8

# # 2023IKCEST第五届“一带一路”国际大数据竞赛
# # 一、背景介绍
# 
# 本届大数据竞赛在中国工程院、教育部高等学校大学计算机课程教学指导委员会及丝绸之路大学联盟的指导下由联合国教科文组织国际工程科技知识中心（IKCEST）、中国工程科技知识中心（CKCEST）、百度公司及西安交通大学共同主办，旨在放眼“一带一路”倡议沿线国家，通过竞赛方式挖掘全球大数据人工智能尖端人才，实现政府—产业—高校合力推动大数据产业研究、应用、发展的目标，进一步夯实赛事的理论基础与实践基础，加快拔尖AI创新人才培养。

# # 二、赛题介绍
# 随着新媒体时代信息媒介的多元化发展，各种内容大量活跃在媒体内中，与此同时各类虚假信息也充斥着社交媒体，影响着公众的判断和决策。如何在大量的文本、图像等多模态信息中，通过大数据与人工智能技术，纠正和消除虚假错误信息，对于网络舆情及社会治理有着重大意义。
# 
# 本次赛题要求选手基于官方指定数据集，通过建模同一事实跨模态数据之间的关系 （主要是文本和图像），实现对任一模态信息能够进行虚假和真实性的检测。鼓励参赛选手通过大模型解决问题，进行技术探索。

# In[ ]:


#环境安装
get_ipython().system('pip install paddlenlp==2.4.2')


# # 三、数据集介绍
# 本次比赛提供从国内外主流社交媒体平台上爬取的含有不同领域声明的数据集。
# 
# 初赛：训练集与验证集： 提供中文训练集5694条以及英文数据4893条，同时公开英文验证集611条与中文验证集711条供选手优化模型。
# 
# 初赛评测数据： 提供文娱、经济、健康领域的测试数据，这些领域的数据较容易区分。英文与中文数据集的测试集各600条。参赛队伍上传的结果文本的每一行就是对应的分类结果，该数据不公布，用于评测。
# 
# 
# | 0 | 1 | 2 |
# | -------- | -------- | -------- |
# | non-rumor | rumor  | unverified |
# 
# 
# 
# [复赛数据后续见官网通知](https://aistudio.baidu.com/aistudio/competition/detail/1030/0/task-definition)

# # 四、数据预处理
# **数据集过大，右键选择解压/home/aistudio/data/data229919/data.zip数据集，耐心等待30分钟，直到出现以下文件夹和文件,解压之后硬盘达到约80g（压缩包27g、解压文件之后50g，可以将项目挂载的数据集取消，空余出27g）**
# * test
# * train
# * val
# * dataset_items_test.json
# * dataset_items_train.json
# * dataset_items_val.json
# 
# 此处将数据集已经放置在queries_dataset_merge文件夹

# In[1]:


from functools import partial
import numpy as np
import time
import os 
import copy
import json
import random
from tqdm import tqdm 

import paddle
from paddlenlp.datasets import load_dataset
import paddle.nn.functional as F
import paddle.nn as nn
import paddlenlp as ppnlp
from paddlenlp.transformers import LinearDecayWithWarmup
import pandas as pd


# In[2]:


#读取数据
import json
data_items_train = json.load(open("/home/aistudio/queries_dataset_merge/dataset_items_train.json"))
data_items_val = json.load(open("/home/aistudio/queries_dataset_merge/dataset_items_val.json"))
data_items_test = json.load(open("/home/aistudio/queries_dataset_merge/dataset_items_test.json"))


# 读取数据中的每一个样本：图像img、文本caption、对应的img_html_news、inverse_search为支持图像img和文本caption的证据材料

# In[3]:


import paddle
from paddle.vision import transforms as T
from paddle.io import Dataset
import json
from urllib.parse import urlparse
from PIL import Image
import os 
import imghdr

def process_string(input_str):
    input_str = input_str.replace('&#39;', ' ')
    input_str = input_str.replace('<b>','')
    input_str = input_str.replace('</b>','')
    #input_str = unidecode(input_str)  
    return input_str
    
class NewsContextDatasetEmbs(Dataset):
    def __init__(self, context_data_items_dict, queries_root_dir, split):
        self.context_data_items_dict = context_data_items_dict
        self.queries_root_dir = queries_root_dir
        self.idx_to_keys = list(context_data_items_dict.keys())
        self.transform =T.Compose([
                        T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
        self.split=split
    def __len__(self):
        return len(self.context_data_items_dict)   


    def load_img_pil(self,image_path):
        if imghdr.what(image_path) == 'gif': 
            try:
                with open(image_path, 'rb') as f:
                    img = Image.open(f)
                    return img.convert('RGB')
            except:
                return None 
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    def load_imgs_direct_search(self,item_folder_path,direct_dict):   
        list_imgs_tensors = []
        count = 0   
        keys_to_check = ['images_with_captions','images_with_no_captions','images_with_caption_matched_tags']
        for key1 in keys_to_check:
            if key1 in direct_dict.keys():
                for page in direct_dict[key1]:
                    image_path = os.path.join(item_folder_path,page['image_path'].split('/')[-1])
                    try:
                        pil_img = self.load_img_pil(image_path)
                    except Exception as e:
                        print(e)
                        print(image_path)
                    if pil_img == None: continue
                    transform_img = self.transform(pil_img)
                    count = count + 1 
                    list_imgs_tensors.append(transform_img)
        stacked_tensors = paddle.stack(list_imgs_tensors, axis=0)
        return stacked_tensors
    def load_captions(self,inv_dict):
        captions = ['']
        pages_with_captions_keys = ['all_fully_matched_captions','all_partially_matched_captions']
        for key1 in pages_with_captions_keys:
            if key1 in inv_dict.keys():
                for page in inv_dict[key1]:
                    if 'title' in page.keys():
                        item = page['title']
                        item = process_string(item)
                        captions.append(item)
                    
                    if 'caption' in page.keys():
                        sub_captions_list = []
                        unfiltered_captions = []
                        for key2 in page['caption']:
                            sub_caption = page['caption'][key2]
                            sub_caption_filter = process_string(sub_caption)
                            if sub_caption in unfiltered_captions: continue 
                            sub_captions_list.append(sub_caption_filter) 
                            unfiltered_captions.append(sub_caption) 
                        captions = captions + sub_captions_list 
                    
        pages_with_title_only_keys = ['partially_matched_no_text','fully_matched_no_text']
        for key1 in pages_with_title_only_keys:
            if key1 in inv_dict.keys():
                for page in inv_dict[key1]:
                    if 'title' in page.keys():
                        title = process_string(page['title'])
                        captions.append(title)
        return captions

    def load_captions_weibo(self,direct_dict):
        captions = ['']
        keys = ['images_with_captions','images_with_no_captions','images_with_caption_matched_tags']
        for key1 in keys:
            if key1 in direct_dict.keys():
                for page in direct_dict[key1]:
                    if 'page_title' in page.keys():
                        item = page['page_title']
                        item = process_string(item)
                        captions.append(item)
                    if 'caption' in page.keys():
                        sub_captions_list = []
                        unfiltered_captions = []
                        for key2 in page['caption']:
                            sub_caption = page['caption'][key2]
                            sub_caption_filter = process_string(sub_caption)
                            if sub_caption in unfiltered_captions: continue 
                            sub_captions_list.append(sub_caption_filter) 
                            unfiltered_captions.append(sub_caption) 
                        captions = captions + sub_captions_list 
        #print(captions)
        return captions
        #加载img文件夹
    def load_queries(self,key):
        caption = self.context_data_items_dict[key]['caption']
        image_path = os.path.join(self.queries_root_dir,self.context_data_items_dict[key]['image_path'])
        pil_img = self.load_img_pil(image_path)
        transform_img = self.transform(pil_img)
        return transform_img, caption
    def __getitem__(self, idx):
        #print(idx)
        #print(self.context_data_items_dict)      
        #idx = idx.tolist()               
        key = self.idx_to_keys[idx]
        #print(key)
        item=self.context_data_items_dict.get(str(key))
        #print(item)
        # 如果为test没有label属性
        #print(self.split)
        if self.split=='train' or self.split=='val':
            label = paddle.to_tensor(int(item['label']))
            direct_path_item = os.path.join(self.queries_root_dir,item['direct_path'])
            inverse_path_item = os.path.join(self.queries_root_dir,item['inv_path'])
            inv_ann_dict = json.load(open(os.path.join(inverse_path_item, 'inverse_annotation.json')))
            direct_dict = json.load(open(os.path.join(direct_path_item, 'direct_annotation.json')))
            captions= self.load_captions(inv_ann_dict)
            captions += self.load_captions_weibo(direct_dict)
            imgs = self.load_imgs_direct_search(direct_path_item,direct_dict)     
            qImg,qCap =  self.load_queries(key)
            sample = {'label': label, 'caption': captions,'imgs': imgs,  'qImg': qImg, 'qCap': qCap}
        else:
            direct_path_item = os.path.join(self.queries_root_dir,item['direct_path'])
            inverse_path_item = os.path.join(self.queries_root_dir,item['inv_path'])
            inv_ann_dict = json.load(open(os.path.join(inverse_path_item, 'inverse_annotation.json')))
            direct_dict = json.load(open(os.path.join(direct_path_item, 'direct_annotation.json')))
            captions= self.load_captions(inv_ann_dict)
            captions += self.load_captions_weibo(direct_dict)
            imgs = self.load_imgs_direct_search(direct_path_item,direct_dict)     
            qImg,qCap =  self.load_queries(key)
            sample = {'caption': captions,'imgs': imgs,  'qImg': qImg, 'qCap': qCap}
        #print(sample)
        #print(len(captions)) 
        #print(type(imgs))
        #print(imgs.size)
        #print(imgs.shape)  
        return sample,  len(captions), imgs.shape[0]


# In[4]:


#### load Datasets ####
train_dataset = NewsContextDatasetEmbs(data_items_train, '/home/aistudio/queries_dataset_merge','train')
val_dataset = NewsContextDatasetEmbs(data_items_val,'/home/aistudio/queries_dataset_merge','val')
test_dataset = NewsContextDatasetEmbs(data_items_test,'/home/aistudio/queries_dataset_merge','test')


# In[ ]:


# 打印数据
for step, batch in enumerate(test_dataset, start=1):
    print(batch)
    break


# In[ ]:


import paddle 
def collate_context_bert_train(batch):
    #print(batch)
    samples = [item[0] for item in batch]
    max_captions_len = max([item[1] for item in batch])
    max_images_len = max([item[2] for item in batch])
    qCap_batch = []
    qImg_batch = []
    img_batch = []
    cap_batch = []
    labels = [] 
    for j in range(0,len(samples)):  
        sample = samples[j]    
        labels.append(sample['label'])
        captions = sample['caption']
        cap_len = len(captions)
        for i in range(0,max_captions_len-cap_len):
            captions.append("")
        if len(sample['imgs'].shape) > 2:
            padding_size = (max_images_len-sample['imgs'].shape[0], sample['imgs'].shape[1], sample['imgs'].shape[2], sample['imgs'].shape[3])
        else:
            padding_size = (max_images_len-sample['imgs'].shape[0],sample['imgs'].shape[1])
        padded_mem_img = paddle.concat((sample['imgs'], paddle.zeros(padding_size)),axis=0)
        #print(1)
        img_batch.append(padded_mem_img)#pad证据图片
        cap_batch.append(captions)
        qImg_batch.append(sample['qImg'])#[3, 224, 224]
        qCap_batch.append(sample['qCap'])     
    #print(labels)   
    #print(img_batch)
    img_batch = paddle.stack(img_batch, axis=0)
    qImg_batch = paddle.stack(qImg_batch, axis=0)
    labels = paddle.stack(labels, axis=0) 
    #print(3)  
    return labels, cap_batch, img_batch, qCap_batch, qImg_batch

def collate_context_bert_test(batch):
    samples = [item[0] for item in batch]
    max_captions_len = max([item[1] for item in batch])
    max_images_len = max([item[2] for item in batch])
    qCap_batch = []
    qImg_batch = []
    img_batch = []
    cap_batch = []
    for j in range(0,len(samples)):  
        sample = samples[j]    
        captions = sample['caption']
        cap_len = len(captions)
        for i in range(0,max_captions_len-cap_len):
            captions.append("")
        if len(sample['imgs'].shape) > 2:
            padding_size = (max_images_len-sample['imgs'].shape[0],sample['imgs'].shape[1],sample['imgs'].shape[2],sample['imgs'].shape[3])
        else:
            padding_size = (max_images_len-sample['imgs'].shape[0],sample['imgs'].shape[1])
        padded_mem_img = paddle.concat((sample['imgs'], paddle.zeros(padding_size)),axis=0)
        img_batch.append(padded_mem_img)
        cap_batch.append(captions)
        qImg_batch.append(sample['qImg'])
        qCap_batch.append(sample['qCap'])        
    img_batch = paddle.stack(img_batch, axis=0)
    qImg_batch = paddle.stack(qImg_batch, axis=0)
    return cap_batch, img_batch, qCap_batch, qImg_batch


# In[ ]:


# load DataLoader
from paddle.io import DataLoader
# 8.18 change
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn = collate_context_bert_train, return_list=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn = collate_context_bert_train,  return_list=True)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn = collate_context_bert_test, return_list=True)


# In[ ]:


# 打印数据
for step, batch in enumerate(train_dataloader, start=1):
    print(batch)
    break


# # 四、模型构建
# **本次赛题为一个NLP与多模态的分类赛题，整体建模采用特征提取、特征交互、预测分类三个阶段**
# 
# **特征提取：** 对于图像数据，使用ResNet模型进行特征提取、对于文本数据，使用预训练模型Ernie-m多语言模型对中文和英文同时处理，qCap,qImg,（需要验证的标题或图像材料）、caps,imgs（支持验证的文本、图像证据材料）
# 
# **特征交互**：使用多头自注意力机制，将标题与文本证据材料交互、图像与图像证据材料交互，输出与需要验证的标题和图像的相关证据特征caps_feature、imgs_features
# 
# **预测分类：** 最后使用全连接层将标题特征、图像特征、相关的文本证据特征、相关的图像证据特征拼接输入到分类器得到最终结果
# ![](https://ai-studio-static-online.cdn.bcebos.com/3f29e3f853b9445fbeb24189103cdbbcb8364498dc484593a891839994dadbd6)
# 
# 

# ## 多语言预训练模型ERNIE-M
# 2021年，百度发布多语言预训练模型ERNIE-M。ERNIE-M通过对96门语言的学习，使得一个模型能同时理解96种语言，该项技术在5类典型跨语言理解任务上刷新世界最好效果。
# 
# ## ERNIE-M原理
# ERNIE-M基于飞桨PaddlePaddle框架训练，该模型构建了大小为25万的多语言词表，涵盖了96种语言的大多数常见词汇，训练语料包含了汉语、英语、法语、南非语、阿尔巴尼亚语、阿姆哈拉语、梵语、阿拉伯语、亚美尼亚语、阿萨姆语、阿塞拜疆语等96种语言，约1.5万亿字符。
# 
# ERNIE-M的学习过程由两阶段组成。第一阶段从少量的双语语料中学习跨语言理解能力，使模型学到初步的语言对齐关系；第二阶段使用回译的思想，通过大量的单语语料学习，增强模型的跨语言理解能力。
# 
# [百度NLP知乎介绍](https://zhuanlan.zhihu.com/p/344810337)
# 

# In[7]:


from paddle.vision import models
import paddle
from paddlenlp.transformers import ErnieMModel,ErnieMTokenizer
from paddle.nn import functional as F
from paddle import nn
import matplotlib.pyplot as plt
import numpy as np
class EncoderCNN(nn.Layer):
    def __init__(self, resnet_arch = 'resnet101'):
        super(EncoderCNN, self).__init__()
        if resnet_arch == 'resnet101':
            resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2D((1, 1))
    def forward(self, images, features='pool'):
        out = self.resnet(images)
        if features == 'pool':
            out = self.adaptive_pool(out)
            out = paddle.reshape(out, (out.shape[0],out.shape[1]))
        return out

class NetWork(nn.Layer):
    def __init__(self, mode):
        super(NetWork, self).__init__()
        self.mode = mode           
        self.ernie = ErnieMModel.from_pretrained('ernie-m-base')
        self.tokenizer = ErnieMTokenizer.from_pretrained('ernie-m-base')
        self.resnet = EncoderCNN()
        self.classifier1 = nn.Linear(2*(768+2048),1024)
        self.classifier2 = nn.Linear(1024,3)
        self.attention_text = nn.MultiHeadAttention(768,16)
        self.attention_image = nn.MultiHeadAttention(2048,16)
        if self.mode == 'text':
            self.classifier = nn.Linear(768,3)
        self.resnet.eval()

    def forward(self,qCap,qImg,caps,imgs):
        self.resnet.eval()
        encode_dict_qcap = self.tokenizer(text = qCap ,max_length = 128 ,truncation=True, padding='max_length')
        input_ids_qcap = encode_dict_qcap['input_ids']
        input_ids_qcap = paddle.to_tensor(input_ids_qcap)
        qcap_feature, pooled_output= self.ernie(input_ids_qcap) #(b,length,dim)
        if self.mode == 'text':
            logits = self.classifier(qcap_feature[:,0,:].squeeze(1))
            return logits
        caps_feature = []
        for i,caption in enumerate (caps):
            encode_dict_cap = self.tokenizer(text = caption ,max_length = 128 ,truncation=True, padding='max_length')
            input_ids_caps = encode_dict_cap['input_ids']
            input_ids_caps = paddle.to_tensor(input_ids_caps)
            cap_feature, pooled_output= self.ernie(input_ids_caps) #(b,length,dim)
            caps_feature.append(cap_feature)
        caps_feature = paddle.stack(caps_feature,axis=0) #(b,num,length,dim)
        caps_feature = caps_feature.mean(axis=1)#(b,length,dim)
        caps_feature = self.attention_text(qcap_feature,caps_feature,caps_feature) #(b,length,dim)
        imgs_features = []
        for img in imgs:
            imgs_feature = self.resnet(img) #(length,dim)
            imgs_features.append(imgs_feature)
        imgs_features = paddle.stack(imgs_features,axis=0) #(b,length,dim)
        qImg_features = []
        for qImage in qImg:
            qImg_feature = self.resnet(qImage.unsqueeze(axis=0)) #(1,dim)
            qImg_features.append(qImg_feature)
        qImg_feature = paddle.stack(qImg_features,axis=0) #(b,1,dim)
        imgs_features = self.attention_image(qImg_feature,imgs_features,imgs_features) #(b,1,dim)
        # [1, 128, 768] [1, 128, 768] [1, 1, 2048] [1, 1, 2048] origin
        # print(qcap_feature.shape,caps_feature.shape,qImg_feature.shape,imgs_features.shape)
        # print((qcap_feature[:,0,:].shape,caps_feature[:,0,:].shape,qImg_feature.squeeze(1).shape,imgs_features.squeeze(1).shape))
        # ([1,768], [1 , 768], [1, 2048], [1,  2048])
        feature = paddle.concat(x=[qcap_feature[:,0,:], caps_feature[:,0,:], qImg_feature.squeeze(1), imgs_features.squeeze(1)], axis=-1) 
        logits = self.classifier1(feature)
        logits = self.classifier2(logits)
        return logits


# In[ ]:


# 声明模型
model = NetWork("image")
print(model)


# # 六、训练配置

# In[ ]:


epochs = 2
num_training_steps = len(train_dataloader) * epochs
warmup_steps = int(num_training_steps*0.1)
print(num_training_steps,warmup_steps)
# 定义 learning_rate_scheduler，负责在训练过程中对 lr 进行调度
lr_scheduler = LinearDecayWithWarmup(1e-6, num_training_steps, warmup_steps)
# 训练结束后，存储模型参数
save_dir ="checkpoint/"
best_dir = "best_model"
# 创建保存的文件夹
os.makedirs(save_dir,exist_ok=True)
os.makedirs(best_dir,exist_ok=True)

decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]

# 定义 Optimizer
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=1.2e-4,
    apply_decay_param_fun=lambda x: x in decay_params)

# 交叉熵损失
criterion = paddle.nn.loss.CrossEntropyLoss()

# 评估的时候采用准确率指标
metric = paddle.metric.Accuracy()


# # 七、模型训练

# In[10]:


# 定义线下评估 评价指标为acc 线上评估是f1score
@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:      
        labels, cap_batch, img_batch, qCap_batch, qImg_batch = batch
        logits = model(qCap=qCap_batch,qImg=qImg_batch,caps=cap_batch,imgs=img_batch)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("eval loss: %.5f, accu: %.5f" % (np.mean(losses), accu))
    model.train()
    metric.reset()
    return np.mean(losses), accu


# In[ ]:


# 定义训练
def do_train(model, criterion, metric, val_dataloader,train_dataloader):
    print("train run start")
    global_step = 0
    tic_train = time.time()
    best_accuracy=0.0
    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(train_dataloader, start=1):
            labels, cap_batch, img_batch, qCap_batch, qImg_batch = batch
            probs = model(qCap=qCap_batch,qImg=qImg_batch,caps=cap_batch,imgs=img_batch)
            loss = criterion(probs, labels)
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()

            global_step += 1 
            # 每间隔 100 step 输出训练指标
            if global_step % 100 == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, acc,
                        10 / (time.time() - tic_train)))
                tic_train = time.time()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            # 每间隔一个epoch 在验证集进行评估
            if global_step % len(train_dataloader) == 0:
                eval_loss,eval_accu=evaluate(model, criterion, metric, val_dataloader)
                save_param_path = os.path.join(save_dir+str(epoch), 'model_state.pdparams')
                paddle.save(model.state_dict(), save_param_path)
                if(best_accuracy<eval_accu):
                    best_accuracy=eval_accu
                    # 保存模型
                    save_param_path = os.path.join(best_dir, 'model_best.pdparams')
                    paddle.save(model.state_dict(), save_param_path)
do_train(model, criterion, metric, val_dataloader,train_dataloader) 


# # 八、模型预测
# **模型预测前，请重启内核，清空占用的显存**

# In[ ]:


# 根据实际运行情况，更换加载的参数路径
import os
import paddle

params_path = 'checkpoint/model_best.pdparams'
if params_path and os.path.isfile(params_path):
    # 加载模型参数
    state_dict = paddle.load(params_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s" % params_path)


# In[ ]:


results = []
# 切换model模型为评估模式，关闭dropout等随机因素
model.eval()
count=0
for batch in test_dataloader:
    count+=1
    cap_batch, img_batch, qCap_batch, qImg_batch = batch
    logits = model(qCap=qCap_batch,qImg=qImg_batch,caps=cap_batch,imgs=img_batch)
    # 预测分类
    probs = F.softmax(logits, axis=-1)
    label = paddle.argmax(probs, axis=1).numpy()
    results += label.tolist()
    print(count)
print(results[:5])
print(len(results))


# In[ ]:


# 输出结果
import pandas as pd
#id/label
#字典中的key值即为csv中的列名
id_list=range(len(results))
print(id_list)
frame = pd.DataFrame({'id':id_list,'label':results})
frame.to_csv("result.csv",index=False,sep=',')


# # 九、后续优化
# 
# baseline分数只有65分，还有很大的改进地方，大家多多尝试，下面是一些想法
# 
# 参数调优：学习率、优化器以及其他超参数等
# 
# 特征提取：更换预训练权重更大的图像特征提取器or文本特征提取器（Ernie or Bert系列）
# 
# 特征交互：目前使用多头自注意力机制对文本与文本证据交互、图像与图像证据交互，可以尝试文本与图像之间的跨模态交互
# 
