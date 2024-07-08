#!/bin/sh
env="StarCraft2"
map="5m_vs_6m"
algo="rmappo"
exp="check"
seed_max=1

#__________________________________________________________________________________
# num_env_steps=10000000 #default
# num_env_steps=1920000 #600 Episodes 
# num_env_steps=3200000 #1k Episodes
num_env_steps=16000 #5 Episodes


#__________________________________________________________________________________


echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 \
    --num_env_steps ${num_env_steps} --ppo_epoch 10 --clip_param 0.05 --use_value_active_masks --use_eval --eval_episodes 32 --share_policy \
    --use_wandb 0 \
    --save_models_flag
done

# --save_models_flag
# --naive_training
# --joint_training