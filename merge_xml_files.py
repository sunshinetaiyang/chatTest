import sys
import os

# 检查命令行参数
if len(sys.argv) < 2:
    print("请提供要合并的 YAML 文件的路径！")
    sys.exit(1)

# 获取输入文件路径
input_file = sys.argv[1]

# 检查输入文件是否存在
if not os.path.exists(input_file):
    print("输入文件不存在！")
    sys.exit(1)

# 生成输出文件路径
output_file = os.path.splitext(input_file)[0] + "_all_in_one.yml"

# 打开输入文件并读取内容
with open(input_file, 'r') as file:
    input_data = file.read()

# 移除末尾的逗号和空格
# input_data = input_data.rstrip(',\n')

# 拼接所有 .yml 文件的内容
merged_data = ''

# 添加输入文件的内容
merged_data += f"# 以下是{os.path.basename(input_file)}的内容\n\n"
merged_data += input_data + '\n'

# 获取 .yml 文件的内容
base_files = [file.strip()[1:-2] for file in input_data.split('\n') if '.yml' in file]
print(base_files)

for file in base_files:
    # 读取 .yml 文件的内容
    with open(file, 'r') as f:
        file_data = f.read()

    # 添加文件信息行
    merged_data += f"# 以下是{os.path.basename(file)}的内容\n"
    # 添加文件内容
    merged_data += file_data + '\n\n'

# 将合并后的内容写入输出文件
with open(output_file, 'w') as file:
    file.write(merged_data)

print(f"合并完成！结果保存在 {output_file} 文件中。")

