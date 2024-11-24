#!/bin/sh
env="StarCraft2"
map="3s_vs_4z"
algo="mappo"
exp="check"
seed_max=1

#__________________________________________________________________________________
# num_env_steps=10000000 #default
# num_env_steps=6400000 #2k Episodes
# num_env_steps=4800000 #1.5k Episodes
# num_env_steps=3200000 #1k Episodes          #TO TRAIN TEAMS 
# num_env_steps=1920000 #600 Episodes
# num_env_steps=1600000 #500 Episodes        #NAIVE TRAINING 
# num_env_steps=960000 #300 Episodes 
# num_env_steps=640000 #200 Episodes 
# num_env_steps=320000 #100 Episodes
# num_env_steps=32000 #10 Episodes
num_env_steps=160000 #50 Episodes
# num_env_steps=16000 #5 Episodes
#__________________________________________________________________________________


echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
# for seed in `seq ${seed_max}`;
# for seed in seq $(seq 2 3);
seed=2
# do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 \
    --num_env_steps ${num_env_steps} --ppo_epoch 15 --use_value_active_masks --use_eval --eval_episodes 32 --stacked_frames 4 --use_stacked_frames --share_policy \
    --use_wandb 0 \
    --ep_no 1000 \
    --alpha 1e-2 \
    --use_lwf
# done

# --save_models_flag
# --naive_training
# --naive_training --naive_test
# --joint_training
# --save_buffer 
# --use_buffer  --pcnt_buffer 50
# --use_buffer --buffer_test
# --use_lwf --usexe
# --use_lwf --lwf_test