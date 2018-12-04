set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python classify_example\train_imagenet.py --lr 0.01 --end_epoch 16 --prefix model/mymodel --lr_epoch 8,14,100 --batch_size 96 --thread_num 32 --frequent 100 
pause 