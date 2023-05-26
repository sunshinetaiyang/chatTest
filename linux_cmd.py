# Linux常用命令集合
# du 命令用于显示指定目录或文件所占用磁盘空间的大小，常用于查看文件和目录的大小以及占用磁盘空间的情况。
du -h path/to/file_or_directory
$ du -sh img_classification_ckpt/
命令可以查询指定dir下所有文件的总size

jupyter nbconvert --to=python main.ipynb

# 查找查询类
# 按文件名查询
find / -name "a.txt"
# 查询字符串
grep "val_imgID.txt" /path/to/dir_a/*


修改 vim 配置文件 ~/.vimrc，在其中添加以下命令：
set number
:%s/aaa/bbb/g

# Jupiter notebook中增加显示
pd.set_option('display.max_rows', 1000)  # 设置显示的最大行数为1000行

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
ls -lh
# 查看文件总大小
du -sh dir_a
# 查看分列大小
du -h dir_a


$ pip list | grep paddle

# 23.5.22 查看完整路径
$realpath result.csv
/home/aistudio/work/result.csv

# -d ~/data/CHECK 表示将解压缩后的文件放置在指定目录，会自动新建
# -o 参数表示在解压缩时覆盖已存在的文件，-o覆盖
# -q 参数表示安静模式，即不显示任何输出信息
!unzip -oq ~/data/data212110/val.zip -d ~/data/CHECK
unzip -lh yourfile.zip

