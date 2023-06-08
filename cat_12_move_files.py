import os
import shutil

# 定义文件夹路径和train_list.txt文件路径
folder_path = 'cat_12_train'
list_file_path = 'train_list.txt'

# 创建12个分类文件夹
for i in range(12):
    category_folder = os.path.join(folder_path, str(i))
    os.makedirs(category_folder, exist_ok=True)

# 读取train_list.txt文件并处理每一行
with open(list_file_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip() # cat_12_train/cgeh7E0wtiTxJDbPWoa9f8KZBsU1z65A.jpg       11
        image_path, category = line.split('\t')
        # image_path:cat_12_train/cgeh7E0wtiTxJDbPWoa9f8KZBsU1z65A.jpg
        # category:11
        category = int(category)

        # 构建目标文件夹路径和目标文件路径
        target_folder = os.path.join(folder_path, str(category))
        image_file = os.path.basename(image_path)
        target_path = os.path.join(target_folder, image_file)

        # 移动文件到目标文件夹
        # image_path:cat_12_train/cgeh7E0wtiTxJDbPWoa9f8KZBsU1z65A.jpg
        # target_path:cat_12_train/11/cgeh7E0wtiTxJDbPWoa9f8KZBsU1z65A.jpg
        shutil.move(image_path, target_path)

# 23.6.5 一次成型，真是牛逼，chat大幅提升效率
print("图片已按分类放置到对应文件夹中。")
