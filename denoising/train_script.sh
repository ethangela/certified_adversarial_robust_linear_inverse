#!/bin/sh

#Ord
python train.py --gpu 0
#Jcb
python train.py --gpu 0 --jcb 1
#Adv
python train.py --gpu 0 --atk 1 --alp 0.4--itr 100 --eps 5
#Smt-Adv
python train.py --gpu 0 --atk 1 --alp 0.4--itr 100 --eps 5 --smp 15 --std 10
#Smt-Grad
python train.py --gpu 0 --smt 1 --stp 6 --smp 15 --std 10

