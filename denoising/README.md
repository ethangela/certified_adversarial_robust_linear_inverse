# experiments for denoising
The model used is adapted from [DPDNN](https://github.com/WeishengDong/DPDNN/tree/master/DENOISE)

---
### Data
1. DIV2K train and validation set (can be downloaded from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and placed into data directory)
3. BSD68 (can be downloaded from []() and placed into data directory)

---
### Train, Test, and Analysis
1. Run script in Shell and the trained parameters will be saved to model directory:
     - ```sh train_script.sh```  
2. Run script in Shell and the resulted excel file will be saved to main directory:
     - ```sh test_script.sh (can repalce test_bsd68.py with test_vk.py)```  
3. Trends plots will be saved to plt_bsd68 and plt_vk directories: 
     - ```python graphing.py```  
    
---
### Parameters info (with details in ```train.py```)
1. For training usage:
     - ```-- jcb```  
     - ```-- gma``` 
     - ```-- atk```  
     - ```-- itr``` 
     - ```-- alp``` 
     - ```-- eps```
     - ```-- smt```   
     - ```-- smp``` 
     - ```-- std``` 
     - ```-- stp``` 
2. For testing usage:
     - ```-- tatk``` 
     - ```-- titr``` 
     - ```-- talp``` 
     - ```-- teps``` 
     - ```-- tsmt```
     - ```-- tsmp```
     - ```-- tstd```
     - ```-- noi```    
3. For visualization usage: 
     - ```--vis```  
