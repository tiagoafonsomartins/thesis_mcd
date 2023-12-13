# thesis_mcd

This is the code used for my dissertation, and it was made public to help in the dissemination of the research area known as XAI. The authors of each XAI method used are properly referenced in the dissertation, but the main purpose of the usage of several, novel techniques is to provide practical examples on how these methods can be implemented in real-world scenarios.


## Data processing 
Data processing was made on Jupyter Notebook and can be found in the folder "data processing":
  **German credit** - inside EXP 1-2
  **Default credit** - inside EXP 1-2
  **IAPMEI** - inside IAPMEI

## Results
Obtained results for predictive and XAI models are present in the folder results
**analysis** - summary statistical description of the datasets
**best_parameters** - best hyper-parameters for each predictive model
**explanations** - explanations generated for each experiment through the usage of PDP, SHAP, LIME, DiCE, and PermuteAttack
**model_performance** - performance metrics for all predictive models used (train/test) 
Accompanying excel files contain information on the statistical information of the public datasets as well as the generated counterfactual methods, and are present within this folder

## Files
main.py contains the main pipeline for the predictive/explanatory process
explainers.py contains the methods that generate explanations for a predictive model
auxiliary_functions.py contains functions used for the explanations as well as for the pipeline


