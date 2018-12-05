set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python train.py --lr 0.01 --network mymodel --data-shape 300 --batch-size 60 --frequent 20 
pause 