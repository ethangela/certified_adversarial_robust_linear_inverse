# experiments for compressed sensing
The model used is adapted from [ISTA-Net<sup>+</sup><sup>+</sup>](https://github.com/jianzhangcs/ISTA-Netpp)

---
### Data
1. Train400 (can be download from [BaiduPan [code: 2o7t]](https://pan.baidu.com/s/1iLpTpRAwXF7Eb3aQZ0jv1A) and placed into the auto-generated ```./cs_train400_png``` directory)
2. Set11 (already contained in ```./data/Set11```)
3. ImageNet 300 grayscale-converted samples (can be downloaded from [ImageNet300](https://drive.google.com/drive/folders/1OVNW7MmOaqvHZQW8LA723r_RgziOxDVj?usp=sharing) and placed into the manually-created ```./data/imagenet``` directory)
4. Sampling matrix (more matrice can be downloaded from [BaiduPan [code: rgd9]](https://pan.baidu.com/s/1AFza-XCyTqRIVTdaYwjT3w) and placed into the ```./sampling_matrix``` directory)

---
### Train, Test, and Analysis
1. Run script in Shell and the trained parameters will be saved to the ```./model``` directory (only pre-trained Ord model is provided)
     - ```sh train_script.sh```  
2. Run script in Shell and the resulted excel file will be saved to the current ```./``` directory
     - ```sh test_script.sh```  
3. Run ploting programme and the trend-pattern plots will be saved to the ```./plt_ig``` and the ```./plt_set``` directories
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
     - ```-- data_dir``` 
     - ```-- tatk``` 
     - ```-- titr``` 
     - ```-- talp``` 
     - ```-- teps``` 
     - ```-- tsmt```
     - ```-- tsmp```
     - ```-- tstd```
     - ```-- cs_ratio```    
3. For visualization usage: 
     - ```--vis```  


