#!/bin/sh
env="MPE"
scenario="simple_reference"
num_landmarks=3
num_agents=2
algo="rmappo" #"mappo" "ippo"
exp="check"
seed_max=2

#________________________________________________________________________________
episode_length=25   
# num_env_steps=3000000 #Default
# num_env_steps=320000 #100 Episodes
num_env_steps=32000 #10 Episodes
# num_env_steps=16000 #5 Episodes
#________________________________________________________________________________
# save_models=0
# naive_training=True

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length ${episode_length} --num_env_steps ${num_env_steps} \
    --ppo_epoch 15 --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --use_wandb 0 --wandb_name "xxx" --user_name "yuchao" --share_policy \
    --save_models_flag 
    


done


    # --save_models_flag 
    # --naive_training  
    # --continual 
    # --joint_training 
    # --n_ep_switch 5 