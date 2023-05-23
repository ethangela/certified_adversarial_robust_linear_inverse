#!/bin/bash

#Ord
python train.py --gpu 0
#Jcb
python train.py --gpu 0 --jcb 1 --gma 20
#Adv
python train.py --gpu 0 --atk 1 --itr 150 --alp 0.2 --eps 1
#Smt-Adv
python train.py --gpu 0 --atk 1 --itr 150 --alp 0.2 --eps 1 --smp 15 --std 10
#Smt-Grad
python train.py --gpu 0 --smt 1 --stp 6 --smp 15 --std 10