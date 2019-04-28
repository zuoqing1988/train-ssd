set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python train.py --lr 0.01 --network mymodel --epoch 7 --lr-steps 200,300 --end-epoch 400 --data-shape 192 --batch-size 64 --frequent 10 --num-class 1 --class-names hand --train-path gesture_train.rec --val-path gesture_val.rec --prefix mode/ssd_hand
pause 