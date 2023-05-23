# experiments for denoising
The model used is adapted from [DPDNN](https://github.com/WeishengDong/DPDNN/tree/master/DENOISE)

---
### Data
1. DIV2K train and validation set (can be downloaded from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and placed into the ```./data``` directory)
3. BSD68 (can be downloaded from [BSD68 dataset](https://drive.google.com/drive/folders/1igMLxCG2GHcXt5JeChrC7T-xvEHGA1xj?usp=sharing) and placed into the ```./data``` directory)

---
### Train, Test, and Analysis
1. Run script in Shell and the trained parameters will be saved to the ```./checkpoint``` directory (only pre-trained Ord model is provided).
     - ```sh train_script.sh```  
2. Run script in Shell and the resulted excel file will be saved to the current ```./``` directory (can repalce ```test_bsd68.py``` with ```test_div2k.py``` for test data switching). 
     - ```sh test_script.sh```  
3. Run ploting programme and the trend-pattern plots will be saved to the ```./plt_bsd68``` and the ```./plt_vk``` directories.
     - ```python graphing.py```  
    
---
### Tuneable parameters (with details in ```train.py``` and ```test_bsd68.py```)
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
