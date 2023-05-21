# Linux常用命令集合
# du 命令用于显示指定目录或文件所占用磁盘空间的大小，常用于查看文件和目录的大小以及占用磁盘空间的情况。
# du -h path/to/file_or_directory
# $ du -sh img_classification_ckpt/
# 命令可以查询指定dir下所有文件的总size

jupyter nbconvert --to=python main.ipynb

修改 vim 配置文件 ~/.vimrc，在其中添加以下命令：
set number

# 23.5.16
find . -maxdepth 1 -type f ! -name 'acc0.9285714285714286.model' -name '*.model' -delete


print('in infer_mot.py, os.path.join', os.path.join(__file__, *(['..'] * 2)))

!mkdir /home/aistudio/external-libraries
!pip install beautifulsoup4 -t /home/aistudio/external-libraries
# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可:
import sys
sys.path.append('/home/aistudio/external-libraries')

# Linux中统计当前目录中的文件数目
ls -l | grep "^-" | wc -l

$ pip list | grep paddle