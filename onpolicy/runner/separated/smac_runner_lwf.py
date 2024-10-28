import time
import numpy as np
from functools import reduce
import torch
from onpolicy.runner.separated.base_runner import Runner
import wandb
import matplotlib.pyplot as plt

import sys
from onpolicy.utils.lwf_utils import LwF_Utils

def _t2n(x):
    return x.detach().cpu().numpy()

class SMACRunner(Runner, LwF_Utils):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(SMACRunner, self).__init__(config)

    def run(self):
        self.warmup()

        #Select the AA 
        self.active_agent = self.active_agent_choice()

        #Create the teacher net 
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        from onpolicy.utils.separated_buffer import SeparatedReplayBuffer
        from onpolicy.algorithms.r_mappo.r_mappo_lwf import R_MAPPO as TrainAlgo

        share_observation_space = self.envs.share_observation_space[self.active_agent] if self.use_centralized_V else self.envs.observation_space[self.active_agent]
        self.teacher = Policy(self.all_args,
                    self.envs.observation_space[self.active_agent],
                    share_observation_space,
                    self.envs.action_space[self.active_agent],
                    device = self.device)
        

        # algorithm
        self.teach_tr = TrainAlgo(self.all_args, self.teacher, device = self.device)
        # buffer

        self.teach_buf = SeparatedReplayBuffer(self.all_args,
                                    self.envs.observation_space[self.active_agent],
                                    share_observation_space,
                                    self.envs.action_space[self.active_agent])


        #Load the Teacher of the AA
        team = 1
        agent_id = self.active_agent
        teacher_actor_state_dict = torch.load(str(self.trained_models_dir) + "/" + str(team) + '/actor_agent' + str(agent_id) + '.pt')
        self.teacher.actor.load_state_dict(teacher_actor_state_dict)

        #Bias Check
        self.extract_biases_from_dict(teacher_actor_state_dict, agent_id, team)

        policy_critic_state_dict = torch.load(str(self.trained_models_dir) + "/" + str(team) + '/critic_agent' + str(agent_id) + '.pt')
        self.teacher.critic.load_state_dict(policy_critic_state_dict)

        policy_vnrom_state_dict = torch.load(str(self.trained_models_dir)  + "/" + str(team) + "/vnrom_agent" + str(agent_id) + ".pt")
        self.teach_tr.value_normalizer.load_state_dict(policy_vnrom_state_dict)
        print(f"Loaded Team no {team}")     

        #Pass the nets to the TrainAlgo
        #_____________________________________________________________________________________________________________________
        
        #Load Teammates
        self.load_teammates()
        #_____________________________________________________________________________________________________________________


        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        print("No of agents", + self.num_agents)
        print("Seed is", + self.all_args.seed)

        #Save log information for performance plotting
        self.win_rate_list = []


        self.episode_count = 0
        self.show_biases = False

######################################################################################
        for episode in range(episodes):
            
            #___________________________________________

            #Sequentially load the teams for Naive Training
            # if self.naive_training and episode % (episodes//len(team_order)) == 0:
            #     index = int(episode / (episodes//len(team_order)))
            #     print(f"Team Loaded is {team_order[index]}")
            #     self.load_teammates(team_order[index])               

            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

            ######################################

            # compute return and update network
            self.compute()
            
            if self.naive_training or self.joint_training is True:
                train_infos = self.freeze_train() if self.naive_test else self.continual_train()
            else:
                train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads 

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

                #Saving the Active Agent
                if self.naive_training or self.joint_training:
                    if not self.multi_agent:
                        self.save_active_agent()
                        if episode == episodes - 1:
                            print(f"Active no {self.active_agent} agent SAVED!")
                    else: 
                        self.save_active_multi_agent()

                #Saving Teams
                if self.save_models_flag:
                    if (episode == episodes - 1  or episode % self.save_interval == 0):
                        self.save_teams_seed()
                        # print('TEAM SAVED')

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.map_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "StarCraft2":
                    battles_won = []
                    battles_game = []
                    incre_battles_won = []
                    incre_battles_game = []                    

                    for i, info in enumerate(infos):
                        if 'battles_won' in info[0].keys():
                            battles_won.append(info[0]['battles_won'])
                            incre_battles_won.append(info[0]['battles_won']-last_battles_won[i])
                        if 'battles_game' in info[0].keys():
                            battles_game.append(info[0]['battles_game'])
                            incre_battles_game.append(info[0]['battles_game']-last_battles_game[i])

                    incre_win_rate = np.sum(incre_battles_won)/np.sum(incre_battles_game) if np.sum(incre_battles_game)>0 else 0.0
                    print("incre win rate is {}.".format(incre_win_rate))

                    #______________________________________________
                    for i in range(self.log_interval):
                        self.win_rate_list.append(incre_win_rate)
                    
                    # self.save_log_infos(self.win_rate_list, increwinrate_path)
                    self.save_log_infos2("incre_win_rate", self.win_rate_list)
                    #______________________________________________


                    if self.use_wandb:
                        wandb.log({"incre_win_rate": incre_win_rate}, step=total_num_steps)
                    else:
                        self.writter.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)
                    
                    last_battles_game = battles_game
                    last_battles_won = battles_won
                # modified

                for agent_id in range(self.num_agents):
                    train_infos[agent_id]['dead_ratio'] = 1 - self.buffer[agent_id].active_masks.sum() /(self.num_agents* reduce(lambda x, y: x*y, list(self.buffer[agent_id].active_masks.shape)))
                
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

            # train_infos[0].update({"average_episode_rewards": np.mean(self.buffer[agent_id].rewards) * self.episode_length})
            # self.episode_reward_list.append(train_infos[0]['average_episode_rewards'])
            
            # # eval
            # if episode % self.eval_interval == 0 and self.use_eval:
            #     self.eval(total_num_steps)

        ########## END EPISODE LOOP ##############
        # #__________________________________
        # #Plot Episode avg rewards
        #     plt.plot(self.win_rate_list, )

        #     plt.ylabel("incremental win rate")

        #     plt.xlabel("Episodes")

        #     if self.save_models_flag:
        #         plt.savefig(str(self.trained_models_dir)  + "/" + str(self.all_args.seed ) + "/team_train0" + ".png")
        #     if self.joint_training:
        #         plt.savefig(str(self.trained_models_dir)  +  "/joint_train1ep" + ".png")
        #     if self.naive_training:
        #         plt.savefig(str(self.trained_models_dir)  +  "/naive_train_A121_0" + str(self.active_agent) + ".png")

        #     print("Image Saved!!")
            # plt.show()
            

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs[:,agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:,agent_id].copy()
            self.buffer[agent_id].available_actions[0] = available_actions[:,agent_id].copy()
    
    @torch.no_grad()
    def collect(self, step):
        value_collector=[]
        action_collector=[]
        action_log_prob_collector=[]
        rnn_state_collector=[]
        rnn_state_critic_collector=[]
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                self.buffer[agent_id].obs[step],
                                                self.buffer[agent_id].rnn_states[step],
                                                self.buffer[agent_id].rnn_states_critic[step],
                                                self.buffer[agent_id].masks[step],
                                                self.buffer[agent_id].available_actions[step])
            value_collector.append(_t2n(value))
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            rnn_state_critic_collector.append(_t2n(rnn_state_critic))
        # [self.envs, agents, dim]
        values = np.array(value_collector).transpose(1, 0, 2)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_state_critic_collector).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        if not self.use_centralized_V:
            share_obs = obs
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(share_obs[:,agent_id], obs[:,agent_id], rnn_states[:,agent_id],
                    rnn_states_critic[:,agent_id],actions[:,agent_id], action_log_probs[:,agent_id],
                    values[:,agent_id], rewards[:,agent_id], masks[:,agent_id], bad_masks[:,agent_id], 
                    active_masks[:,agent_id], available_actions[:,agent_id])

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []
        for eval_i in range(self.n_eval_rollout_threads):
            one_episode_rewards.append([])
            eval_episode_rewards.append([])

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            eval_actions_collector=[]
            eval_rnn_states_collector=[]
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_actions, temp_rnn_state = \
                    self.trainer[agent_id].policy.act(eval_obs[:,agent_id],
                                            eval_rnn_states[:,agent_id],
                                            eval_masks[:,agent_id],
                                            eval_available_actions[:,agent_id],
                                            deterministic=True)
                eval_rnn_states[:,agent_id]=_t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1,0,2)

            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            for eval_i in range(self.n_eval_rollout_threads):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards[eval_i].append(np.sum(one_episode_rewards[eval_i], axis=0))
                    one_episode_rewards[eval_i] = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.concatenate(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}                
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                if self.use_wandb:
                    wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break
