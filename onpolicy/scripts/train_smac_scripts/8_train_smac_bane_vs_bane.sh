#!/bin/sh
env="StarCraft2"
map="bane_vs_bane"
algo="mappo"
exp="check"
seed_max=1

#__________________________________________________________________________________
# num_env_steps=10000000 #default
# num_env_steps=6400000 #2k Episodes
# num_env_steps=4800000 #1.5k Episodes
# num_env_steps=3200000 #1k Episodes
# num_env_steps=1920000 #600 Episodes
# num_env_steps=1600000 #500 Episodes 
# num_env_steps=960000 #300 Episodes 
# num_env_steps=640000 #200 Episodes  #TO TRAIN TEAMS 
# num_env_steps=480000 #150 Episodes  
# num_env_steps=320000 #100 Episodes         
# num_env_steps=32000 #10 Episodes
# num_env_steps=16000 #5 Episodes
#__________________________________________________________________________________


echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
# for seed in `seq $(seq 2 3)`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 \
    --num_env_steps ${num_env_steps} --ppo_epoch 15 --use_value_active_masks --use_eval --eval_episodes 32 --share_policy \
    --use_wandb 0 \
    --save_buffer
done


# --save_models_flag
# --multi_agent
# --naive_training
# --naive_training --naive_test
# --joint_training
# --save_buffer
# --use_buffer 
# --use_buffer --buffer_test