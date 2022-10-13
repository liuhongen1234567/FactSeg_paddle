# FactSeg (Implemented by PaddlePaddle 2.3)

## 1. 简介
为了解决在遥感图像中仅占几个像素的小物体检测问题，作者提出了Fact-Seg语义分割网络。该网络首次从结构和优化器
角度提出了前景激活（FA）驱动的小物体语义分割框架，增强了网络对小物体弱特征的感知能力。该模块由对偶解码器分支（FA）和联合
概率损失（CP）组成，其中对偶解码器中的FA模块用于激活小物体特征并抑制大尺度的背景，语义分割（SR）微调分支模块用于进一步区分
小物体。此外，作者还提出了小物体在线挖掘模块，用于解决小物体和背景之间的样本不平衡问题。该方法在两个遥感图像
数据集中取得了SOTA的成绩，并且在精度和速度之间取得了很好的平衡。

![](framework.png)

**论文：** [FactSeg: Foreground Activation-Driven Small
Object Semantic Segmentation in Large-Scale
Remote Sensing Imagery](https://ieeexplore.ieee.org/document/9497514)

**参考repo：** [Wang: FactSeg](https://github.com/Junjue-Wang/FactSeg)

在此非常感谢 [Wang](https://github.com/Junjue-Wang/FactSeg) 等人贡献的FactSeg项目，提高了本repo复现论文的效率。项目已上传到[AI studio](https://aistudio.baidu.com/aistudio/projectdetail/4632057?sUid=711344&shared=1&ts=1665137667176)上，
可使用32G显存部署后台任务训练。若在本地训练，请对数据集路径，预训练权重路径的文件进行相应更改。AI studio上请fork v_4_1 最新版本。若部署后台任务，请先全选全部项目，再删除Step1_5文件夹, final.zip文件，RaddleRS.zip, RaddleRS文件夹，并选择train.ipynb作为执行文件。其他版本将会导致文件无法运行。具体内容可以参考后台任务部署说明文档.pdf。


# 2. 数据集和复现精度

- 航空图像数据集iSAID—语义分割部分：[https://captain-whu.github.io/iSAID/index.html](https://captain-whu.github.io/iSAID/index.html)

复现精度如下。

| method         | iters | bs   | card | loss      | align_corners | mIoU    |
|----------------| ----- | ---- | ---- |-----------| ------------- |---------|
| official_code  | 60k   | 4    | 2    | JointLoss | √             | 64.80   | 
| ours           | 60k   | 8    | 1    | JointLoss | √             | 64.64   | 

关于模型验证指标，模型迭代60k大约需要2天左右。训练日志保存在Log文件夹下。


## 3. 准备数据与环境

### 3.1 准备环境

- 环境：AI Studio & BML CodeLab & Python 3.7；
- 硬件：Nvidia Tesla V100 32G × 1；
- 框架：PaddlePaddle 2.3.2；

```jupyter
!pip install scikit-image
!pip install reprod_log
!pip install paddleseg
!pip install pillow
%cd /home/aistudio/
!mkdir output
%cd /home/aistudio/data/data170962
!mkdir train test
%cd /home/aistudio/data/data170962/train
!mkdir images masks
%cd  /home/aistudio/data/data170962/test
!mkdir images masks
%cd /home/aistudio/
```

### 3.2 解压数据

训练和评估所需数据为iSAID数据集的语义分割任务部分。

图像可以在 <a href="https://captain-whu.github.io/DOTA/dataset.html" target="_blank">DOTA-v1.0</a> (train/val/test)下载，标注可在 <a href="https://captain-whu.github.io/iSAID/dataset.html" target="_blank">iSAID</a> (train/val)下载。

对于下载完成的原始iSAID数据集，使用如下命令进行解压
```jupyter
!unzip -d /home/aistudio/data/data170962/train/images /home/aistudio/data/data170962/part1.zip
!unzip -d /home/aistudio/data/data170962/train/images /home/aistudio/data/data170962/part2.zip
!unzip -d /home/aistudio/data/data170962/train/images /home/aistudio/data/data170962/part3.zip
!unzip -d /home/aistudio/data/data170962/train/masks /home/aistudio/data/data170962/seg_train.zip 
!unzip -d /home/aistudio/data/data170962/test/images /home/aistudio/data/data170962/val_image.zip
!unzip -d /home/aistudio/data/data170962/test/masks /home/aistudio/data/data170962/seg_val.zip
```

按照如下结构在根目录进行准备进行准备：

```diff
├── data
 └── data170962
     ├── train
     │   ├──images
     │   │   └── images
     │   └──masks
     │       └── images
     └── test
         ├──images
         │   └── images
         └──masks
             └── images

```

- [AI Studio: iSAID](https://aistudio.baidu.com/aistudio/datasetdetail/170962/0) ；
- 数据格式：图片为RGB三通道图像，标签为单通道图像，值为INT[0,15]+{255}，二者均为PNG格式存储；


## 4. 开始使用

### 4.1 模型训练

主要训练配置如下：

-   模型训练步长60000，单卡训练批大小设为8；
-   优化器：Momentum（momentum=0.9, weight_decay=1e-4, clip_grad_by_norm ）；
-   学习率策略：多项式衰减PolynomialDecay（begin=0.007，end=0.0, power=0.9）；

下载转换得到的ResNet50预训练权重，保存在`/home/aistudio/data/data170962/resnet50_paddle.pdparams`或者手动更改/home/aistudio/simplecv1/module/_resnet.py
下的resnet state路径，若打印出”Loading model Resnet50“，则加载成功。这里请确保打印出Loading model Resnet50，否则可能会影响最终精度。
参数中的image_dir和mask_dir设置的是测试集的输入图像和分割图像路径；如果要更改训练集路径，请更改`configs/isaid/factseg.py`中的data.train.params.image_dir 和
mask_dir参数。如果AI studio长期不显示打印信息，可能是训练集的路径设置错误导致的。

开始训练，单GPU自动混合精度。

```jupyter
config_path='isaid.factseg'
ckpt_path='/home/aistudio/data/data170962/fact-seg_temp.pdparams'
image_dir='/home/aistudio/data/data170962/test/images/images'
mask_dir='/home/aistudio/data/data170962/test/masks/images'
resume_model_path=''
!python  apex_train.py \
    --config_path={config_path} \
    --ckpt_path={ckpt_path} \
    --image_dir={image_dir} \
    --mask_dir={mask_dir} \
    --patch_size=896\
    --resume_model_path={resume_model_path}\
    --resume_iter=0
```

### 4.2 模型评估

训练时未进行评估，待训练完成后单独对已保存模型进行评估，并写入日志，选择验证集最优模型。根据测试结果10k之后模型的测试指标不再提升，因此选择10k-60k保存的模型均可达到验收指标

```jupyter
config_path='isaid.factseg'
ckpt_path='/home/aistudio/data/data170962/fact-seg_temp.pdparams'
image_dir='/home/aistudio/data/data170962/test/images/images'
mask_dir='/home/aistudio/data/data170962/test/masks/images'

!python isaid_eval.py \
    --config_path={config_path} \
    --ckpt_path={ckpt_path} \
    --image_dir={image_dir} \
    --mask_dir={mask_dir} \
    --patch_size=896
```


FactSeg权重文件和ResNet50预训练权重，请在[百度网盘](https://pan.baidu.com/s/1wI7OjqIkrBvo6gv55GSKyg)下载，密码为`st5l`或直接在AI studio中下载。10k之前的模型采用训练评估一体化部署，后因内存溢出，训练从10k模型恢复训练，10k之后开始保存每5000k的周期模型，因此这里不提供5k模型，10k模型路径为/home/aistudio/data/data170962/fact-seg_temp.pdparams，15k模型路径为/home/aistudio/data/data170962/fact-seg_temp_15k.pdparams，其他模型均在/home/aistudio/data/data171451/val_model文件夹下，测试时，请更换文件名。如果需要运行模型、损失、评估、反向对齐的代码块，即代码块3-6，请从上述[百度网盘](https://pan.baidu.com/s/1wI7OjqIkrBvo6gv55GSKyg)链接下载Step1_5.zip文件，并解压到根目录FactSeg_paddle-main下。


## 5. 模型推理部署
从之前的[百度网盘](https://pan.baidu.com/s/1wI7OjqIkrBvo6gv55GSKyg)链接下载PaddleRS.zip，并解压PaddleRS.zip文件夹
这里手动生成了`PaddleRS/normal_model/model.yml`，其中包含了使PaddleRS成功调用模型的参数。

FactSeg模型迁移在这里 [PaddleRS/paddlers/rs_models/seg]。

将工作目录切换到`Fact-Seg-master/PaddleRS`。然后安装依赖，拷贝[4.2]小节下载的权重，导出部署模型。
这里如果在本地部署还需要更改`/PaddleRS/paddlers/rs_models/seg/backbone/_resnet.py`第202-206行中的预训练权重路径
```jupyter
%cd /home/aistudio/PaddleRS
!pip install -r requirements.txt
!python setup.py install
```

```jupyter
%cd /home/aistudio/PaddleRS
!cp /home/aistudio/data/data170962/factseg50_paddle.pdparams normal_model/model.pdparams
!python deploy/export/export_model.py --model_dir=normal_model --save_dir=inference_model --fixed_input_shape [None,3,896,896]
```

运行动态图、静态图的加载与预测，可视化图像保存在`/PaddleRS/infer_test/`。

```jupyter
!python infer_test/infer.py
```


## 6. TIPC测试

- 模式1：lite_train_lite_infer，使用少量数据训练，用于快速验证训练到预测的流程是否能走通，不验证精度和速度；

```jupyter
!pip install  https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl
!bash ./test_tipc/prepare.sh test_tipc/configs/seg/factseg/train_infer_python.txt lite_train_lite_infer
```

由于AI stuido上无法安装gdal库，请下载PaddleRS.zip到本地进行测试，整个测试约耗时5min左右，本地测试结果已经保存到/home/aistudio/PaddleRS/test_tipc/output/seg/FactSeg/lite_train_lite_infer/results_python.log中。请下载/home/aistudio/data/data170962/resnet50_paddle.pdparams下的Resnet50预训练模型，并再次确保/home/aistudio/PaddleRS/paddlers/rs_models/seg/backbone/_resnet.py第202-206行的resnet50模型路径正确；如果不正确，请更改，并再次运行python setup.py install对更改进行保存。该项目没有在多卡上测试，仅在本地单卡12G上测试成功。

```jupyter
!bash ./test_tipc/test_train_inference_python.sh test_tipc/configs/seg/factseg/train_infer_python.txt lite_train_lite_infer
```

TIPC测试日志文件保存于之前的[百度网盘](https://pan.baidu.com/s/1wI7OjqIkrBvo6gv55GSKyg)链接中，密码为st5l 。日志文件在百度网盘的路径为FactSeg/TIPC日志文件/results_python.log



### Ubuntu 系统安装gdal
```jupyter
conda install -c conda-forge gdal=3.4.1
```
#### 报错一

```py
gdalinfo: /home/pg/anaconda3/envs/abcNet/bin/../lib/./libstdc++.so.6: version `GLIBCXX_3.4.30' not
found (required by /home/pg/anaconda3/envs/abcNet/bin/../lib/./libtiledb.so.2.2)
```
#### 解决方案
```py
conda install -c conda-forge gcc=12.1.0
```
#### 报错二
```py
ImportError: libtiledb.so.2.2: cannot open shared object file: No such file or directory
```
#### 解决方案
```py
ln -s /home/XX/anaconda3/envs/torch17/lib/libtiledb.so.2.3 /home/XX/anaconda3/envs/torch17/lib/libtiledb.so.2.2
```
#### 报错三
```py
GDAL import fails with ImportError: libpoppler.so.71: cannot open shared object file: No such file or directory
```
#### 解决方案
```py
conda install "poppler<0.62"
```
## 7. pytorch下的权重转换脚本（Resnet50、FactSeg50）及各种npy文件生成脚本使用说明
生成对齐数据请运行如下命令：
```py
%cd /home/aistudio/
!python 00_test_data_generate.py
```
生成其他ref.npy文件,请在pytorch和paddle环境下运行以下命令：
```py
cd XX/lhe/Fact-Seg 
# 模型结构对齐
python 01_test_forward_torch.py
# 评估对齐
python 03_test_metric_torch.py
# 损失对齐
python 04_loss_torch.py
# 反向对齐
python 05_test_backward_torch.py
```
对Resnet50、FactSeg50进行权重转换，请在pytorch环境和paddle下运行以下命令
```py
cd XX/lhe/Fact-Seg 
# 对FactSeg50进行权重转换
python torch2paddleFactSeg.py
# 对Resnet50进行权重转换
python torch2paddleResnet50.py
```

这里除了生成对齐数据代码块外，其他脚本中的输入路径、输出路径都要根据本地情况进行具体修改，同时也都无法在AI studio上运行,pytorch下的权重转换和npy生成文件位于torch_ref目录下，这里如果想要运行转换脚本和npy生成脚本，请下载到具有pytorch和paddle的虚拟环境下，同时根据`pytorch下预训练模型权重转换脚本及npy文件生成代码使用说明.pdf文档`和github上[百度网盘](https://pan.baidu.com/s/1wI7OjqIkrBvo6gv55GSKyg)链接搭建相应的目录结构。百度网盘密码为`st5l`。

## 8. 迁移到PaddleRS模型与AI Studio实现模型一致性验证

模型一致性验证前，请确保下载并解压Step1_5文件夹和转化后的PaddleRS迁移模型factseg50_paddle_RS.pdparams，该模型仍位于上述百度网盘链接中。
模型结构对齐验证，请运行如下命令：
```py
%cd /home/aistudio/
!python rs_test_forward.py
```
torch转paddle模型在测试集上的精度验证，请运行如下命令：
```py
%cd /home/aistudio/
config_path='isaid.factseg'
ckpt_path='/home/aistudio/data/data171451/factseg50_paddle_RS.pdparams'
image_dir='/home/aistudio/data/data170962/test/images/images'
mask_dir='/home/aistudio/data/data170962/test/masks/images'

!python isaid_eval_rs.py \
    --config_path={config_path} \
    --ckpt_path={ckpt_path} \
    --image_dir={image_dir} \
    --mask_dir={mask_dir} \
    --patch_size=896
```

## 9. LICENSE

本项目的发布受 [Apache 2.0 license](https://github.com/ucsk/FarSeg/blob/develop/LICENSE) 许可认证。

## 10. 参考链接与文献

- 论文： <a href="https://ieeexplore.ieee.org/abstract/document/9497514" target="_blank">FactSeg: Foreground Activation-Driven Small Object Semantic Segmentation in Large-Scale Remote Sensing Imagery</a>
- 代码： <a href="https://github.com/Junjue-Wang/FactSeg" target="_blank">FactSeg (Junjue-Wang
)</a> 、 <a href="https://github.com/ucsk/FarSeg" target="_blank">FarSeg (DeepRS)</a>
