# train-ssd
本项目旨在任意更改模型，以适用于不同硬件环境

根据Kaiming He的论文，训练数据规模不大时，train from stratch所能达到的精度不如pretrain on imagenet。

下载地址:

原始数据：[ILSVRC2012_img_train](https://pan.baidu.com/s/1TdFvKZJyX_CMkdjWqAlaeg)

裁剪之后：[ILSVRC2012_img_train_224x224](https://pan.baidu.com/s/1PamzHH14wUITchMelT0Fvw)

# 训练分类模型： 

模型定义在symbol\mymodel.py

**(1)准备数据**

下载[ILSVRC2012_img_train_224x224](https://pan.baidu.com/s/1PamzHH14wUITchMelT0Fvw)，放在classify_data里面

	classify_data/anno.txt
	classify_data/ILSVRC2012_img_train_224x224/n01440764
	classify_data/ILSVRC2012_img_train_224x224/n01443537
    ...
	
**(2)双击train_imagenet.bat开始训练**

# 训练SSD

**(3)准备数据**

参照[mxnet-ssd官方](https://github.com/apache/incubator-mxnet/tree/master/example/ssd)

或者你从[此处]()下载打包好的文件, 解压到data文件夹

	data/train.idx
	data/train.lst
	data/train.rec
	data/val.idx
	data/val.lst
	data/val.rec

**(4)双击train_ssd.bat开始训练**

跑起来之后你再研究参数吧

**(5)想自定义模型？**

你改symbol\mymodel.py， 改channel不会造成错误，如果要加减层，别改最后conv12之后的层

# 推荐[ZQCNN](https://github.com/zuoqing1988/ZQCNN)作为PC推理库

