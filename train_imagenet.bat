set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python classify_example\train_imagenet.py --network mymodel --lr 0.01 --end_epoch 16 --prefix model/mymodel --lr_epoch 8,14,100 --batch_size 32 --thread_num 8 --frequent 100 
pause 