#!/usr/bin/env python
# coding: utf-8

# # 效果展示
# ![](https://ai-studio-static-online.cdn.bcebos.com/87ced28595054ada86e3c0bd5bb8e7a6b6d27ff605dc48519c535341079faf8d)
# 
# # 项目背景
# - 风景园林专业是一门科学和艺术结合的专业，旨在研究土地、水体、植物、空气和它们之间的关系，为人类提供健康、舒适和美观的空间。专业结合了生态学、建筑学、美学、文化学等多个学科，通过研究和规划城市和城市绿地系统，为人们创造出宜居的环境。  
# - 但给世界描绘美好蓝图的同时，画图，是风景园林等建筑土木类专业学生和从业人员的痛。
# - 无数设计师加班加点耗费在那一张张明天早上汇报就可能被甲方要求回炉从造的方案图上。常常因为甲方的几条修改意见，让设计师们绞尽脑汁画通宵。有些人甚至为此付出生命的代价，一批批新生力量前赴后继扑上设计院的战场，同时一批批饱受折磨的老人转行考公读研。
# - 作为跟房地产关系密切的行业之一，随着房地产快速扩张发展的红利期过去，风景园林行业各家公司也开始降本增效，于是，网传已经有设计研究院要求员工掌握Stable Diffusion等AI绘图技术，并与员工的晋升和年终奖挂钩，相信在不久的未来，AI设计将对我们的行业、生活产生巨大的影响。  
# 

# > 为了您能更好体验，请在V100 32G环境下运行本项目。
# # 快速上手
# ## PPDiffusers介绍
# PPDiffusers是一款支持多种模态（如文本图像跨模态、图像、语音）扩散模型（Diffusion Model）训练和推理的国产化工具箱，依托于PaddlePaddle框架和PaddleNLP自然语言处理开发库。该库提供了超过50种SOTA扩散模型Pipelines集合，支持文图生成（Text-to-Image Generation）、文本引导的图像编辑（Text-Guided Image Inpainting）、文本引导的图像变换（Image-to-Image Text-Guided Generation）、超分（Super Superresolution）在内的10+任务，覆盖文本图像跨模态、图像、音频等多种模态。  
# 更多详细介绍见PPDiffusers代码仓库：https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers

# ## 环境安装

# ### 安装PPDiffusers

# In[10]:


get_ipython().system('pip install --upgrade ppdiffusers -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html --user')


# **> 请“重启内核”后继续下面操作！**

# # 版本更新
# ## 20230519:发布公园环境LoRA权重

# In[5]:


import paddle
from ppdiffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

paddle.seed(54235333)
pipe = StableDiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V2.0")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.unet.load_attn_procs("CityParkLoraV1")
image = pipe("A wetland park in the suburbs of the city, some birds, high resolution,hyper quality,full details, natural, communtiy park, outdoor, grassland", num_inference_steps=50).images[0]


# In[6]:


image.save("/home/aistudio/wetland_park.png")
image


# ## 开始画图
# ### 1.根据甲方需求生成图片
# 本项目中，我们主要用到`Realistic_Vision_V2.0`扩散模型，它是基于Stable Diffsion模型finetune而形成的。  
# 在设计专业，我们经常会收到甲方很多文字性的（无理）需求，例如五彩斑斓的黑、五颜六色的白等等，下面我们就用`Realistic_Vision_V2.0`模型帮我们根据甲方爸爸的需求出图，看看五彩斑斓的黑到底是长什么样子的。
# 
# 
# | SD Model | Lora | ControlNet Model | Prompt |
# | -------- | -------- | -------- | -------- |
# | Realistic_Vision_V2.0     | None     | None     | a house in village, colorful black wall     |
# 
# 

# In[3]:


import paddle
from ppdiffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V2.0")
paddle.seed(123321120)
prompt = "a house in village, colorful black wall"
image = pipe(prompt, guidance_scale=7.5, height=768, width=768).images[0]

image.save("village_house.png")


# In[4]:


from PIL import Image
Image.open("village_house.png")


# ### 2.毛坯变精装
# 新时代美丽乡村建设也是当下时期风景园林专业的重要课题之一，在广袤的乡村土地中，有很多年久失修的旧房子，我们设计师需要将这些房子进行翻新升级改造，让它们重焕活力，成为新时代的旅游景点为大众服务。下面就给大家展示一下，通过SD大模型，旧房子如何换了一个新的面貌。
# 
# | SD Model | Lora | ControlNet Model | Prompt |
# | -------- | -------- | -------- | -------- |
# | Realistic_Vision_V2.0     | None     | sd-controlnet-canny     | beautiful village,shrubs and flowers around the building,countryside,country road,blue sky,modern house,white wall,glass window, wooden roof,high resolution,hyper quality,full details     |

# In[9]:


import requests

model_name = "lllyasviel/sd-controlnet-canny"

def check_hf_model_availability(model_name):
    # url = f"https://huggingface.co/{model_name}/resolve/main/config.json"
    url = f"https://huggingface.co/models"
    try:
        response = requests.head(url)
        if response.status_code == 200:
            print(f"Model '{model_name}' is available in the Hugging Face model hub.")
        else:
            print(f"Model '{model_name}' is not available in the Hugging Face model hub.")
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while connecting to the Hugging Face model hub: {e}")

check_hf_model_availability(model_name)


# In[4]:


import os
import cv2
import random
import paddle

from annotator.canny import CannyDetector
from annotator.util import HWC3, resize_image

from paddlenlp.trainer import set_seed as seed_everything
from ppdiffusers import ControlNetModel, StableDiffusionControlNetPipeline, StableDiffusionPipeline

apply_canny = CannyDetector()

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")

# 23.6.3 网络下载有问题，尝试从本地进行预训练
pipe = StableDiffusionControlNetPipeline.from_pretrained(
   "SG161222/Realistic_Vision_V2.0", controlnet=controlnet, from_hf_hub=True, from_diffusers=True
)
# 23.6.3 还是有问题，尝试从云上进行预训练
# pipe = StableDiffusionControlNetPipeline.from_pretrained("SG161222/Realistic_Vision_V2.0")


def process(
    input_image,
    prompt,
    a_prompt,
    n_prompt,
    num_samples,
    image_resolution,
    ddim_steps,
    guess_mode,
    strength,
    scale,
    seed,
    eta,
    low_threshold,
    high_threshold,
):
    with paddle.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape
        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = paddle.to_tensor(detected_map.copy(), dtype=paddle.float32) / 255.0
        control = control.unsqueeze(0).transpose([0, 3, 1, 2])

        control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        )  
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)
        results = []
        for _ in range(num_samples):
            img = pipe(
                prompt + ", " + a_prompt,
                negative_prompt=n_prompt,
                image=control,
                num_inference_steps=ddim_steps,
                height=H,
                width=W,
                eta=eta,
                controlnet_conditioning_scale=control_scales,
                guidance_scale=scale,
            ).images[0]
            results.append(img)

    return [255 - detected_map] + results

inputImage = cv2.imread('test_img/village.jpg')
results = process(input_image=inputImage,
                  prompt="beautiful village,shrubs and flowers around the building,countryside,country road,blue sky,modern house,white wall,glass window, wooden roof,high resolution,hyper quality,full details",
                  a_prompt="",
                  n_prompt="",
                  num_samples=1,
                  image_resolution=512,
                  ddim_steps=20,
                  guess_mode=False,
                  strength=1.0,
                  scale=9.0,
                  seed=33241531,
                  eta=0.0,
                  low_threshold=20,
                  high_threshold=200,
                  )

savePath = "./outputImg/village"
if not os.path.exists(savePath):
    os.makedirs(savePath)
    
for i in range(1, len(results)):
    results[i].save(os.path.join(savePath, "{}.jpg".format(i)))


# In[10]:


# 原图
from PIL import Image
Image.open("test_img/village.jpg")


# In[20]:


# 生成效果图
from PIL import Image
Image.open("outputImg/1.jpg")


# ### 3.手绘秒转效果图
# 设计师会在效果图建模和渲染上面花费比较多的时间，下面让我们来看看如何用大模型对手绘图纸进行渲染直出效果图，节约设计师建模和渲染的时间，提高生产效率吧。这里用的参数和模型与上面的“毛坯变精装”基本相同，唯一不同的点是加了一个根据地产景观优化的`Fair-faced-concrete-V1`Lora参数模型，使生成效果更佳符合渲染器效果。
# | SD Model | Lora | ControlNet Model | Prompt |
# | -------- | -------- | -------- | -------- |
# | Realistic_Vision_V2.0     | Fair-faced-concrete-V1     | sd-controlnet-canny     | garden in residential area,large grassland,adults and children walking,people sit under umbrellas chatting,glass window,blue sky,high resolution,hyper quality,full details,<lora:architecturalconcrete-v1:1>modern architecture,outside,facade     |

# In[2]:


pipe.apply_lora("/home/aistudio/data/data214847/Fair-faced-concrete-V1.safetensors")

inputImage = cv2.imread('test_img/draw.jpg')
results = process(input_image=inputImage,
                  prompt="garden in residential area,large grassland,adults and children walking,people sit under umbrellas chatting,glass window,blue sky,high resolution,hyper quality,full details,modern architecture,outside,facade",
                  a_prompt="",
                  n_prompt="water,lake",
                  num_samples=1,
                  image_resolution=512,
                  ddim_steps=20,
                  guess_mode=False,
                  strength=1.0,
                  scale=9.0,
                  seed=12332,
                  eta=0.0,
                  low_threshold=20,
                  high_threshold=200,
                  )

savePath = "./outputImg/"
if not os.path.exists(savePath):
    os.makedirs(savePath)
    
for i in range(1, len(results)):
    results[i].save(os.path.join(savePath, "{}.jpg".format(i)))


# In[39]:


# 原图
from PIL import Image
Image.open("test_img/draw.jpg")


# In[38]:


# 生成效果图
from PIL import Image
Image.open("outputImg/1.jpg")


# # 总结
# 以上为Stable Diffusion大模型在风景园林专业的一些探索，借助PPDiffusers套件的能力，我们可以快速地应用SD模型以及对SD模型进行针对化finetune调优，也欢迎大家多探索prompt，欢迎将一些效果好的prompt提示词分享到评论区！
# # 参考资料
# https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers
# https://huggingface.co/SG161222/Realistic_Vision_V2.0
# 
# # 版权声明
# 本项目展示所用原图素材由华南农业大学陈崇贤副教授提供，未经许可请勿随意转载。
