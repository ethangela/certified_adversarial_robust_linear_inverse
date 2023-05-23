# experiments for compressed sensing

---
### Data
1. Train400 (can be download from [BaiduPan [code: 2o7t]](https://pan.baidu.com/s/1iLpTpRAwXF7Eb3aQZ0jv1A))
1. Set11 (already contained in data/Set11)
3. ImageNet 300 grayscale-converted samples (can be downloaded from and placed into data directory)
4. Sampling matrix (more matrice can be downloaded from [BaiduPan [code: rgd9]](https://pan.baidu.com/s/1AFza-XCyTqRIVTdaYwjT3w) and placed into sampling_matrix directory)

---
### Train, Test, and Analysis
1. Run script in Shell and the trained parameters will be saved to model directory:
     - ```sh train_script.sh```  
2. Run script in Shell and the resulted excel file will be saved to main directory:
     - ```sh test_script.sh```  
3. Trend plots will be saved to plt_ig and plt_set directories: 
     - ```python graphing.py```  
    
---
### Parameters info
1. For training usage:
     - ```-- jcb```  
     - ```-- gma``` 
     - ```-- itr``` 
     - ```-- alp``` 
     - ```-- eps``` 
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
     - ```-- cs_ratio```    
3. For visualization usage: 
     - ```--vis```  


