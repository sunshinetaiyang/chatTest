# Linux常用命令集合
# du 命令用于显示指定目录或文件所占用磁盘空间的大小，常用于查看文件和目录的大小以及占用磁盘空间的情况。
du -h path/to/file_or_directory
$ du -sh img_classification_ckpt/
命令可以查询指定dir下所有文件的总size

jupyter nbconvert --to=python main.ipynb

paddle.fluid.dygraph.guard.release()

git clone https://hub.fastgit.xyz/author/repo
https://doc.fastgit.org/zh-cn/guide.html#web-%E7%9A%84%E4%BD%BF%E7%94%A8

# !cp -r /home/aistudio/coco_config/_base_  /home/aistudio/PaddleDetection/configs/rtdetr/
# 查找查询类
# 按文件名查询
find dir_a -name "*xyz*"

# 删除dir_A目录下除了*.txt之外的所有文件
find dir_A ! -name '*.txt' -type f -delete
find . ! -name '*odel*' -type f -delete

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
# sort -hr 会将结果按照人类可读的格式和逆序（从大到小）排序。
du -h ./* | sort -hr
# sort 从小到大
du -h ./* | sort -h
# 查找大于1GB的文件
find /home/user/Documents -type f -size +1G




$ pip list | grep paddle

# 23.5.22 查看完整路径
$realpath result.csv
/home/aistudio/work/result.csv

# -d ~/data/CHECK 表示将解压缩后的文件放置在指定目录，会自动新建
# -o 参数表示在解压缩时覆盖已存在的文件，-o覆盖
# -q 参数表示安静模式，即不显示任何输出信息
!unzip -oq ~/data/data212110/val.zip -d ~/data/CHECK
unzip -lh yourfile.zip


usage: infer.py [-h] --model_dir MODEL_DIR [--image_file IMAGE_FILE]
                [--image_dir IMAGE_DIR] [--batch_size BATCH_SIZE]
                [--video_file VIDEO_FILE] [--camera_id CAMERA_ID]
                [--threshold THRESHOLD] [--output_dir OUTPUT_DIR]
                [--run_mode RUN_MODE] [--device DEVICE] [--use_gpu USE_GPU]
                [--run_benchmark RUN_BENCHMARK]
                [--enable_mkldnn ENABLE_MKLDNN]
                [--enable_mkldnn_bfloat16 ENABLE_MKLDNN_BFLOAT16]
                [--cpu_threads CPU_THREADS] [--trt_min_shape TRT_MIN_SHAPE]
                [--trt_max_shape TRT_MAX_SHAPE]
                [--trt_opt_shape TRT_OPT_SHAPE]
                [--trt_calib_mode TRT_CALIB_MODE] [--save_images SAVE_IMAGES]
                [--save_mot_txts] [--save_mot_txt_per_img] [--scaled SCALED]
                [--tracker_config TRACKER_CONFIG]
                [--reid_model_dir REID_MODEL_DIR]
                [--reid_batch_size REID_BATCH_SIZE] [--use_dark USE_DARK]
                [--action_file ACTION_FILE] [--window_size WINDOW_SIZE]
                [--random_pad RANDOM_PAD] [--save_results]
                [--use_coco_category] [--slice_infer]
                [--slice_size SLICE_SIZE [SLICE_SIZE ...]]
                [--overlap_ratio OVERLAP_RATIO [OVERLAP_RATIO ...]]
                [--combine_method COMBINE_METHOD]
                [--match_threshold MATCH_THRESHOLD]
                [--match_metric MATCH_METRIC]
infer.py: error: unrecognized arguments: --save_txt=True

python -m pip install visualdl -i https://mirror.baidu.com/pypi/simple
python -u tools/train.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml \
                        --use_vdl=true \
                        --vdl_log_dir=vdl_dir/scalar \
                        --eval
visualdl --logdir vdl_dir/scalar/ --host <host_IP> --port <port_num>
