ROOT=/raid0-data/yifan
export HF_HOME=$ROOT

CUDA_VISIBLE_DEVICES=5 python main_sharpzo.py --root_path $ROOT --config dataset_configs/eurosat.yaml --zo_lr 1e-3 --zo_eps_rge 1e-3 --sigma 0.4 --zo_eps_cge 1e-3 --rho 0.1 --prompt_trainer zo --smoothness_estimator zo  --total_steps 200000 --change_point 100 &
CUDA_VISIBLE_DEVICES=6 python main_sharpzo.py --root_path $ROOT --config dataset_configs/eurosat.yaml --zo_lr 1e-3 --zo_eps_rge 1e-3 --sigma 0.4 --zo_eps_cge 1e-3 --rho 0.1 --prompt_trainer zo --smoothness_estimator zo  --total_steps 200000 --change_point 200 &
CUDA_VISIBLE_DEVICES=5 python main_sharpzo.py --root_path $ROOT --config dataset_configs/eurosat.yaml --zo_lr 1e-3 --zo_eps_rge 1e-3 --sigma 0.4 --zo_eps_cge 1e-3 --rho 0.1 --prompt_trainer zo --smoothness_estimator zo  --total_steps 200000 --change_point 300 &
CUDA_VISIBLE_DEVICES=6 python main_sharpzo.py --root_path $ROOT --config dataset_configs/eurosat.yaml --zo_lr 1e-3 --zo_eps_rge 1e-3 --sigma 0.4 --zo_eps_cge 1e-3 --rho 0.1 --prompt_trainer zo --smoothness_estimator zo  --total_steps 200000 --change_point 400 &   
CUDA_VISIBLE_DEVICES=7 python main_sharpzo.py --root_path $ROOT --config dataset_configs/eurosat.yaml --zo_lr 1e-3 --zo_eps_rge 1e-3 --sigma 0.4 --zo_eps_cge 1e-3 --rho 0.1 --prompt_trainer zo --smoothness_estimator zo  --total_steps 200000 --change_point 500 &

wait
