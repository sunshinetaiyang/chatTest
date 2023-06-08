import os
import random
import shutil
from xml.etree import ElementTree as ET

# 源图片文件夹路径
image_dir = 'PaddleDetection2.5/dataset/coco/train/JPEGImages'

# 源标注文件夹路径
annotation_dir = 'PaddleDetection2.5/dataset/coco/train/Annotations'

# 目标文件夹路径
output_dir = 'PaddleDetection2.5/dataset/coco/train/OversampledData'

# 定义需要过采样的类别和对应的样本数量
classes_to_oversample = {
    'kite': 200,
    'balloon': 200,
    'trash': 200
}

# 创建目标文件夹
os.makedirs(output_dir, exist_ok=True)

# 对需要过采样的类别进行样本复制
for class_name, num_samples in classes_to_oversample.items():
    # 获取该类别下的所有XML文件路径
    xml_files = [file for file in os.listdir(annotation_dir) if file.endswith('.xml')]
    
    # 获取该类别下的XML文件中包含的样本
    class_files = []
    for xml_file in xml_files:
        xml_path = os.path.join(annotation_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objects = root.findall('object')
        for obj in objects:
            obj_class = obj.find('name').text
            if obj_class == class_name:
                image_file = root.find('filename').text
                class_files.append(image_file)
                break
    
    # 计算需要复制的次数
    num_copies = num_samples - len(class_files)
    
    # 从该类别的样本中随机选择并复制到目标文件夹中，直到达到指定数量
    for _ in range(num_copies):
        # 随机选择源文件
        source_file = random.choice(class_files)
        
        # 获取源文件路径
        source_image_path = os.path.join(image_dir, source_file)
        source_annotation_path = os.path.join(annotation_dir, source_file.replace('.jpg', '.xml'))
        
        # 生成新的文件名
        target_name = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=len(source_file.split('.')[0])))
        target_image_file = target_name + '.jpg'
        target_annotation_file = target_name + '.xml'
        
        # 目标文件路径
        target_image_path = os.path.join(image_dir, target_image_file)
        target_annotation_path = os.path.join(annotation_dir, target_annotation_file)
        
        # 复制图片文件和对应的标注文件到目标文件夹
        shutil.copy(source_image_path, target_image_path)
        shutil.copy(source_annotation_path, target_annotation_path)
        
        
        
import os
from xml.etree import ElementTree as ET

# 标注文件夹路径
annotation_dir = '/home/aistudio/PaddleDetection2.5/dataset/coco/train/Annotations'

# 遍历标注文件夹，修改文件名
for filename in os.listdir(annotation_dir):
    xml_path = os.path.join(annotation_dir, filename)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotation_filename = os.path.splitext(filename)[0] + '.jpg'
    root.find('filename').text = annotation_filename

    # 保存修改后的标注文件
    tree.write(xml_path)


# 23.6.7 遍历标注类别，代码写得真是优美，ChatGPT牛逼
import os
import xml.etree.ElementTree as ET

!rm -rf PaddleDetection2.5/dataset/coco/train/Annotations/.ipynb_checkpoints
# 标注文件夹路径
annotation_folder = 'PaddleDetection2.5/dataset/coco/train/Annotations'

# 类别字典
class_dict = {'nest': 1, 'kite': 2, 'balloon': 3, 'trash': 4}

# 统计类型数量的字典
class_count = {class_name: 0 for class_name in class_dict.keys()}

# 遍历标注文件夹中的所有.xml文件
xml_files = os.listdir(annotation_folder)
for xml_file in xml_files:
    xml_path = os.path.join(annotation_folder, xml_file)
    
    # 解析标注文件
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 遍历标注文件中的所有目标
    for obj in root.iter('object'):
        # 获取目标类别名称
        class_name = obj.find('name').text
        
        # 统计类型数量
        if class_name in class_dict:
            class_count[class_name] += 1

# 打印类型分布统计结果
for class_name, count in class_count.items():
    print(f'{class_name}: {count}')


