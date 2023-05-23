# experiments for compressed sensing

---
### Data
1. Set11 (already contained in data/Set11)
2. ImageNet 300 grayscale-converted samples (can be downloaded from)

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


