#!/bin/sh

rm -rf imgs
rm -rf trained_models
mkdir imgs
mkdir trained_models

cd usb_py_env
sh make.sh
cd -

python main.py --use-gae --log-interval 1 --num-stack 1 --num-steps 512 --num-processes 1 --lr 5e-5 --entropy-coef 0.01 --ppo-epoch 20 --batch-size 32 --gamma 0.01 --tau 0.95 --num-frames 1000000
