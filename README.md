# Source Code for paper 'SharpZO: Hybrid Sharpness-Aware Vision Language Model Prompt Tuning via Forward-Only Passes'
in Proceeding of the Thirty-Ninth Annual Conference on Neural Information Processing Systems (NeurIPS 2025)

<h1> <p>🤗 News</p></h1>

**1/15/2026:** The first version of the SharpZO code has been released. We are currently conducting thorough testing on the cleaned-up codebase. Please feel free to raise an issue if you encounter any problems..


## Requirements
### Environment
The environment is based on `torch==2.5.1` and `Python=3.9.21`. Deltaled environment information are provided in `requirements.txt` or environment.yml (for a full list exported by conda).


### Dataset

Follow [DATASET.md](DATASET.md) to install ImageNet and other 10 datasets referring to previous CoOp and CraFT work. The root path of the dataset should be passed to `main_sharpzo.py`. The path of dataset should follow this pattern:
```
$ROOT/datasets
|–– imagenet/
|–– caltech-101/
|–– oxford_pets/
|–– stanford_cars/
```

## Run the Code

We provide simple scripts to run the code in `run_all_final_exps.sh`. The key arguments are listed in the following sections.

## Key Arguments

Here, we provide the key arguments for running the code. The hyperparameters used to run the results in the paper are included in the example script `run_all_final_exps.sh`.

### General arguments:
- `config`: Path to configuration file for datasets 
- `zo_eps_cge`: Finite difference step size for Conditional Gradient Estimation (CGE), used in both stage 1 and 2 training
- `total_steps`: Total number of training steps
- `change_point`: Change point between stage 1 and stage 2
- `prompt_trainer`: Use `zo` for sharpzo method, but pure cma-es or pure zo training can be achieved by editing the `change_point` and `smoothness_estimator` setup.
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



## Citation
If our work or code supports your research, we would greatly appreciate it if you could cite our related papers:

```angular2html
@inproceedings{
yang2025sharpzo,
title={SharpZO: Hybrid Sharpness-Aware Vision Language Model Prompt Tuning via Forward-Only Passes},
author={Yifan Yang and Zhen Zhang and Rupak Vignesh Swaminathan and Jing Liu and Nathan Susanj and Zheng Zhang},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
}
