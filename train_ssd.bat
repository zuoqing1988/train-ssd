set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python train.py --lr 0.01 --network mymodel --epoch 7 --lr-steps 200,300 --end-epoch 400 --data-shape 192 --batch-size 32 --frequent 20 
pause 