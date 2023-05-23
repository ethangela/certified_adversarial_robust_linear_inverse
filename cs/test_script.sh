#!/bin/bash

#testing script example on Set11 dataset
#testing para: \epsilon=1, \sigma=[0.5 0.7 1.0 1.3 1.6 2.0], m/n=0.4
for i in 0.5 0.7 1.0 1.3 1.6 2.0
do 
python test.py --gpu 0 --pkl set_e1_c40_n0 --tatk 1 --titr 150 --talp 0.2 --teps 1 --tsmt 1 --tsmp 1000 --tstd $i --cs_ratio 40
python test.py --gpu 0 --pkl set_e1_c40_n0 --jcb 1 --gma 20 --tatk 1 --titr 150 --talp 0.2 --teps 1 --tsmt 1 --tsmp 1000 --tstd $i --cs_ratio 40
python test.py --gpu 0 --pkl set_e1_c40_n0 --atk 1 --itr 150 --alp 0.2 --eps 1 --tatk 1 --titr 150 --talp 0.2 --teps 1 --tsmt 1 --tsmp 1000 --tstd $i --cs_ratio 40
python test.py --gpu 0 --pkl set_e1_c40_n0 --atk 1 --itr 150 --alp 0.2 --eps 1 --smp 15 --std 10 --tatk 1 --titr 150 --talp 0.2 --teps 1 --tsmt 1 --tsmp 1000 --tstd $i --cs_ratio 40
python test.py --gpu 0 --pkl set_e1_c40_n0 --smt 1 --smp 15 --std 10 --stp 6 --tatk 1 --titr 150 --talp 0.2 --teps 1 --tsmt 1 --tsmp 1000 --tstd $i --cs_ratio 40
done