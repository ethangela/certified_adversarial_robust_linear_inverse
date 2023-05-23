# robust_inverse_problems
 
This repository provides code to reproduce results of the paper: Certified Adversarial Defense Methods for Inverse Problems.

---
### Requirements
1. Python 3.7
2. PyTorch 

---
### Reproducing quantitative results
1. Main experiment scripts:
     - ```python main.py```  
2. Sensitivity to graph mismatch:
     - ```python varying_graph_test.py```  
3. Sensitivity to parameter mismatch:  
     - ```python varying_lambda_test.py```  
* Facebook network loader:  
     - ```python facebook_network_loader.py```  
* Gibbs sampling:  
     - ```python gibbs_sampling.py```  
     
---
### Additional info
We provide two graph examples (grid and block) produced by Gibbs sampling. Feel free to create your own graph samples in replace of them. 
