# Source code for paper 'SharpZO: Hybrid Sharpness-Aware Vision Language Model Prompt Tuning via Forward-Only Passes'

## Requirements
### Environment
The environment is based on `torch==2.5.1` and `Python=3.9.21`. Deltaled environment information are provided in `requirements.txt`.


### Dataset

Follow [DATASET.md](DATASET.md) to install ImageNet and other 10 datasets referring to previous CoOp and CraFT work.

## Run the Code

We provide simple scripts to run the code in `run_all_final_exps.sh`. The key arguments are listed in the following sections.

## Key Arguments

Here, we provide the key arguments for running the code. The hyperparameters used to run the results in the paper are included in the example script `run_all_final_exps.sh`.

### General arguments:
- `config`: Path to configuration file for datasets 
- `zo_eps_cge`: Finite difference step size for Conditional Gradient Estimation (CGE), used in both stage 1 and 2 training
- `total_steps`: Total number of training steps
- `prompt_trainer`: Prompt optimization method (`cma-es`, `zo`, `fo`, or `zo-pruned`)  , use `zo-pruned` for sharpzo method
- `backbone`: CLIP model backbone (`RN50`, `RN101`, `ViT-B/32`, `ViT-B/16`)  
- `root_path`: Root directory for data and cache  
- `shots`: Number of few-shot training examples per class  

### For stage 1 training:
- `rho`: Sharpness regularization parameter (radius for the smoothness estimator in SharpZO)  
- `smoothness_estimator`: Method to estimate loss smoothness (`zo`, `fo`, or `none`)  
- `sigma`: Step size for CMA-ES optimization

### For stage 2 training:

- `num_pertub`: Number of queries for gradient estimation, we keep use 1 in our method
- `zo_lr`: Learning rate for zeroth-order (ZO) optimization
- `zo_eps_rge`: Finite difference step size for Random Gradient Estimation (RGE)  
- `prune_interval`: Frequency of pruning (in training steps)  
- `prune_ratio`: Proportion of parameters to prune at each interval  