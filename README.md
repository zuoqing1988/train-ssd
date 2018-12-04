# train-ssd
本项目旨在任意更改模型，以适用于不同硬件环境

根据Kaiming He的论文，训练数据规模不大时，train from stratch所能达到的精度不如pretrain on imagenet。

下载地址:

原始数据：[ILSVRC2012_img_train](https://pan.baidu.com/s/1TdFvKZJyX_CMkdjWqAlaeg)

裁剪之后：[ILSVRC2012_img_train_224x224](https://pan.baidu.com/s/1PamzHH14wUITchMelT0Fvw)

# 训练分类模型： 

模型定义在classify_core\symbol.py

**(1)准备数据**

下载[ILSVRC2012_img_train_224x224](https://pan.baidu.com/s/1PamzHH14wUITchMelT0Fvw)，放在classify_data里面

	classify_data/anno.txt
	classify_data/ILSVRC2012_img_train_224x224/n01440764
	classify_data/ILSVRC2012_img_train_224x224/n01443537
    ...
	
**(2)双击train_imagenet.bat开始训练**

# 训练SSD（未完待续...）
