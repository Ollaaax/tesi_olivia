#!/bin/sh
env="MPE"
scenario="simple_spread" 
num_landmarks=3
num_agents=3
algo="mappo" #"mappo" "ippo"
exp="check"
seed_max=1

#______________________________________________________________________
# num_env_steps=20000000 #default
# num_env_steps=640000 #200 Episodes
# num_env_steps=480000 #150 episodes
# num_env_steps=160000 #50 Episodes
num_env_steps=80000 #25 Episodes
# num_env_steps=32000 #10 Episodes
#_____________________________________________________________________

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;

do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 25 --num_env_steps ${num_env_steps} \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "xxx" --user_name "yuchao" --use_wandb 0 --share_policy \
    --log_interval 1 \
    --ep_no 25 \
    --naive_training

done

    # --save_models_flag
    # --multi_agent
    # --naive_training
    # --naive_training --naive_test
    # --joint_training
    # --save_buffer
    # --use_buffer --ep_no_rebuf_train 500 --pcnt_buffer 50
    # --use_buffer --buffer_test

