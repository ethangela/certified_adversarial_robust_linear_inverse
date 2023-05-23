#!/bin/bash

#testing script example on BSD68 dataset
#testing para: \epsilon=10, \sigma=[5 7 10 13 15 17 19 22], \sigma_b=2(--noi 2)
for j in 5 7 10 13 15 17 19 22
do 
python test_bsd68.py --gpu 0 --load_model_path ./checkpoints/ord_sigma15_data32000_epo100.pth --pkl bsd68_noi2_eps10 --noi 2 --eps 10 --std $j
python test_bsd68.py --gpu 0 --load_model_path ./checkpoints/jcb_sigma15_data32000_epo100.pth --pkl bsd68_noi2_eps10 --noi 2 --eps 10 --std $j
python test_bsd68.py --gpu 0 --load_model_path ./checkpoints/adv_itr100_alp0.4_eps5.0_sigma15_data32000_epo100.pth --pkl bsd68_noi2_eps10 --noi 2 --eps 10 --std $j
python test_bsd68.py --gpu 0 --load_model_path ./checkpoints/smtadv_itr100_alp0.4_eps5.0_smp15_std10.0_sigma15_data32000_epo100.pth --pkl bsd68_noi2_eps10 --noi 2 --eps 10 --std $j --blk 1
python test_bsd68.py --gpu 0 --load_model_path ./checkpoints/smtgrad_stp6_std10.0_smp15_sigma15_data32000_epo100.pth --pkl bsd68_noi2_eps10 --noi 2 --eps 10 --std $j --blk 1
done