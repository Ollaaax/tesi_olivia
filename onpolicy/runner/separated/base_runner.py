
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch 
from tensorboardX import SummaryWriter

import shutil
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
from scipy.io import savemat

from onpolicy.utils.separated_buffer import SeparatedReplayBuffer
from onpolicy.utils.util import update_linear_schedule

def _t2n(x):
    return x.detach().cpu().numpy()

class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        #___________________________________________
        #CONFIG FLAGS
        self.flag = self.all_args.flag
        self.multi_agent = self.all_args.multi_agent
        # self.continual_flag = self.all_args.continual

        self.use_lwf = self.all_args.use_lwf

        self.naive_training = self.all_args.naive_training
        self.naive_test = self.all_args.naive_test

        self.joint_training = self.all_args.joint_training

        self.save_models_flag = self.all_args.save_models_flag
        self.acquario = self.all_args.acquario
        
        self.save_buffer = self.all_args.save_buffer
        self.use_buffer = self.all_args.use_buffer
        self.buffer_test = self.all_args.buffer_test
        self.ep_no_rebuf_train = self.all_args.ep_no_rebuf_train
        self.pcnt_buffer = self.all_args.pcnt_buffer
        
        #Other Flags
        self.show_biases = False
        self.iamloading = False
        self.iamloadingacquario = False
        #___________________________________________
        #Other
        self.active_agent = 0

        #___________________________________________
        #Directories
        self.trained_models_dir = config["trained_models_dir"]
        if not self.save_buffer:
            self.results_dir = self.create_log_infos2()
        
        if self.multi_agent:
            if self.naive_training:
                agents_dir = self.trained_models_dir  / "MultiAgent" / "NaiveTraining"  / "Agents"
            if self.use_buffer:
                agents_dir = self.trained_models_dir  / "MultiAgent" / "BufferReplay" / "Agents"
            if not agents_dir.exists():
                os.makedirs(str(agents_dir))
            self.agents_dir = agents_dir
        #___________________________________________

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # #_____________________________________________________________________________________________________________________


        if self.use_render:
            import imageio
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

        
        #____________________________________________________________________________________________________________________
        #SET Policy and TrainAlgo
        if self.all_args.algorithm_name == "happo":
            from onpolicy.algorithms.happo.happo_trainer import HAPPO as TrainAlgo
            from onpolicy.algorithms.happo.policy import HAPPO_Policy as Policy
        elif self.all_args.algorithm_name == "hatrpo":
            from onpolicy.algorithms.hatrpo.hatrpo_trainer import HATRPO as TrainAlgo
            from onpolicy.algorithms.hatrpo.policy import HATRPO_Policy as Policy
        else:
            #Load the Rebuf Algo if needed (only with use_buffer)
            if self.use_buffer: 
                from onpolicy.algorithms.r_mappo.r_mappo_rebuf import R_MAPPO as TrainAlgo
            elif self.use_lwf:
                from onpolicy.algorithms.r_mappo.r_mappo_lwf import R_MAPPO as TrainAlgo
            else:
                from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo

            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy


        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)

        #____________________________________________________________________________________________________________________
        ###NETS INITIALIZATION

        self.policy = []
        self.trainer = []
        self.buffer = []

        for agent_id in range(self.num_agents):
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            # policy network
            po = Policy(self.all_args,
                        self.envs.observation_space[agent_id],
                        share_observation_space,
                        self.envs.action_space[agent_id],
                        device = self.device)
            self.policy.append(po)

            # ## TODO ACQUARIO FINIRE IL TRAINING 
            # if self.save_models_flag:
            #     if os.path.exists(str(self.trained_models_dir) + "/" + str(self.all_args.seed)):
            #         self.restore_pretrained_models_acquario(agent_id)
            #         print("MODEL RESTORED")
            #     else:
            #         print("INITIALIZE NEW MODEL")

            #________________________________________________________________________________________________________________________
            #LOAD PRETRAINED MODELS for N/J Training - At this point we suppose to have
            #a directory with the pretrained models
            
            # #Naive Training
            # if self.naive_training:
            #     self.naive_load_teams(agent_id)

            # #Joint Training
            # if self.joint_training and agent_id != self.active_agent:
            #     teammates = 1
            #     self.restore_pretrained_models(agent_id, teammates) 

            # #____________________________________________________________
            #____________________________________________________________
            #LOAD PRETRAINED MODELS for JOINT TRAINING

        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device = self.device)
            # buffer
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            bu = SeparatedReplayBuffer(self.all_args,
                                       self.envs.observation_space[agent_id],
                                       share_observation_space,
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)
            self.trainer.append(tr)

            #_______________________________________________________________________
#             ####LOAD VALUE NORMALIZER
# #________________________________________________________________________________________
#########################################################################################
#########################################################################################
#########################################################################################    
 
    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1], 
                                                                self.buffer[agent_id].rnn_states_critic[-1],
                                                                self.buffer[agent_id].masks[-1])
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = []
        # random update order

        action_dim=self.buffer[0].actions.shape[-1]
        factor = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)

        for agent_id in torch.randperm(self.num_agents):
            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(factor)
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])
            
            if self.all_args.algorithm_name == "hatrpo":
                old_actions_logprob, _, _, _, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            else:
                old_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])

            if self.all_args.algorithm_name == "hatrpo":
                new_actions_logprob, _, _, _, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            else:
                new_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))

            factor = factor*_t2n(torch.prod(torch.exp(new_actions_logprob-old_actions_logprob),dim=-1).reshape(self.episode_length,self.n_rollout_threads,1))
            train_infos.append(train_info)      
            self.buffer[agent_id].after_update()

        return train_infos

    def save(self):
        for agent_id in range(self.num_agents):
            policy_actor = self.trainer[agent_id].policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")

            self.extract_biases_from_dict(policy_actor.state_dict(), agent_id, 1)

            policy_critic = self.trainer[agent_id].policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")
            if self.trainer[agent_id]._use_valuenorm:
                policy_vnrom = self.trainer[agent_id].value_normalizer
                torch.save(policy_vnrom.state_dict(), str(self.save_dir) + "/vnrom_agent" + str(agent_id) + ".pt")

    def restore(self):
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)
            if self.trainer[agent_id]._use_valuenorm:
                policy_vnrom_state_dict = torch.load(str(self.model_dir) + '/vnrom_agent' + str(agent_id) + '.pt')
                self.trainer[agent_id].value_normalizer.load_state_dict(policy_vnrom_state_dict)

    def log_train(self, train_infos, total_num_steps): 
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)


#_____________________________________________________________________________________________________________
                ######### MY FUNCTIONS for CHILD ###########
 #________________________________________________________________________
 ###           TRAIN FUNCTIONS

    def continual_train(self):
        """
        Train a single agent (freezing all the others)
        """

        train_infos = []
        # random update order

        action_dim=self.buffer[0].actions.shape[-1]
        factor = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)

        agents_list = []
        for i in range(self.num_agents):
            if not self.multi_agent:
                update = True if i == self.active_agent else False
            else: 
                update = True if i in self.multi_active_agent else False
            agents_list.append((i, update))

        for i in torch.randperm(self.num_agents):

            agent_id, update = agents_list[i]

            self.trainer[agent_id].prep_training() if update else self.trainer[agent_id].prep_rollout()
            self.buffer[agent_id].update_factor(factor)
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])
            
            if self.all_args.algorithm_name == "hatrpo":
                old_actions_logprob, _, _, _, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            else:
                old_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            train_info = self.trainer[agent_id].train(self.buffer[agent_id], update)

            if self.all_args.algorithm_name == "hatrpo":
                new_actions_logprob, _, _, _, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            else:
                new_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            
            # y = new_actions_logprob[:,0].detach().numpy()
            # # print(y)
            # x = np.random.rand(len(y))
            # plt.scatter(x, np.exp(y))
            # plt.xlabel('x')
            # plt.ylabel('Value')
            # plt.title('Scatter Plot')
            # plt.show(block=False)
            # plt.pause(1)
            # plt.close()

            factor = factor*_t2n(torch.prod(torch.exp(new_actions_logprob-old_actions_logprob),dim=-1).reshape(self.episode_length,self.n_rollout_threads,1))
            train_infos.append(train_info)      
            self.buffer[agent_id].after_update()

        return train_infos

#________________________________________________________________________
###           TEAMS MANAGEMENT

    def ssave(self):
        for agent_id in range(self.num_agents):
            policy_actor = self.trainer[agent_id].policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_continual_dir) + "/actor_agent" + str(agent_id) + ".pt")
            policy_critic = self.trainer[agent_id].policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_continual_dir) + "/critic_agent" + str(agent_id) + ".pt")
            if self.trainer[agent_id]._use_valuenorm:
                policy_vnrom = self.trainer[agent_id].value_normalizer
                torch.save(policy_vnrom.state_dict(), str(self.save_continual_dir) + "/vnrom_agent" + str(agent_id) + ".pt")

    def restore_pretrained_models(self, agent_id, team):
        """
        Load the pretrained teams (as teammates) and value normalizer 
        """

        policy_actor_state_dict = torch.load(str(self.trained_models_dir) + "/" + str(team) + '/actor_agent' + str(agent_id) + '.pt')
        self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)

        #Bias Check
        self.extract_biases_from_dict(policy_actor_state_dict, agent_id, team)

        policy_critic_state_dict = torch.load(str(self.trained_models_dir) + "/" + str(team) + '/critic_agent' + str(agent_id) + '.pt')
        self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

        policy_vnrom_state_dict = torch.load(str(self.trained_models_dir)  + "/" + str(team) + "/vnrom_agent" + str(agent_id) + ".pt")
        self.trainer[agent_id].value_normalizer.load_state_dict(policy_vnrom_state_dict)

    def restore_pretrained_models_acquario(self):
        """
        Load the pretrained teams (as teammates) addressed with the seed to continue the training if the acquario stops
        """
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(str(self.trained_models_dir) + '/' + str(self.all_args.seed) + '/actor_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)

            policy_critic_state_dict = torch.load(str(self.trained_models_dir) + '/' + str(self.all_args.seed) + '/critic_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

            policy_vnrom_state_dict = torch.load(str(self.trained_models_dir)  + "/" + str(self.all_args.seed) + "/vnrom_agent" + str(agent_id) + ".pt")
            self.trainer[agent_id].value_normalizer.load_state_dict(policy_vnrom_state_dict)

        print(f"Team {self.all_args.seed} Restored")

    def load_teammates(self, team):
        """
        Load the pretrained teams (as teammates) and value normalizer except for the active agent
        """
        for agent_id in range(self.num_agents):
            if agent_id != self.active_agent:
                policy_actor_state_dict = torch.load(str(self.trained_models_dir) + "/" + str(team) + '/actor_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)

                #Bias Check
                self.extract_biases_from_dict(policy_actor_state_dict, agent_id, team)

                policy_critic_state_dict = torch.load(str(self.trained_models_dir) + "/" + str(team) + '/critic_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

                policy_vnrom_state_dict = torch.load(str(self.trained_models_dir)  + "/" + str(team) + "/vnrom_agent" + str(agent_id) + ".pt")
                self.trainer[agent_id].value_normalizer.load_state_dict(policy_vnrom_state_dict)
        print(f"Loaded Team no {team}")

    def save_teams_seed(self):
        '''
        Function to be run once, the first time, to create different teams
        '''

        curr_team = self.trained_models_dir / str(self.all_args.seed)
        if not curr_team.exists():
            os.makedirs(str(curr_team))
            print(f"Folder {self.all_args.seed} created!")

        for agent_id in range(self.num_agents):
            policy_actor = self.trainer[agent_id].policy.actor

            torch.save(policy_actor.state_dict(), str(curr_team)  + "/actor_agent" + str(agent_id) + ".pt")
            policy_critic = self.trainer[agent_id].policy.critic
            torch.save(policy_critic.state_dict(), str(curr_team)  + "/critic_agent" + str(agent_id) + ".pt")
            if self.trainer[agent_id]._use_valuenorm:
                policy_vnrom = self.trainer[agent_id].value_normalizer
                torch.save(policy_vnrom.state_dict(), str(curr_team) + "/vnrom_agent" + str(agent_id) + ".pt")

    def create_log_infos(self, base_name):
        '''
        Function to save into .npy file in the current folder the results of the training
        data : list of relevant information
        returns: file_path : path to save the relevant information
        '''        
        #Identify the correct path:
        extension = ".npy"
        extension_mat = ".mat"

        if self.save_models_flag:
            curr_dir = self.trained_models_dir / str(self.all_args.seed)
            second_name = "training"
        
        if self.naive_training:
            curr_dir = self.trained_models_dir
            second_name = "naive_training"

        if self.joint_training:
            curr_dir = self.trained_models_dir
            second_name = "joint_training"

        if self.use_buffer:
            curr_dir = self.trained_models_dir
            second_name = "buffer_replay"

        if not curr_dir.exists():
            os.makedirs(str(curr_dir))

        existing_files = [f for f in os.listdir(curr_dir) if f.endswith(str(second_name) + ".npy")]

        # Extract existing file numbers (if any)
        existing_numbers = []
        for file in existing_files:
            try:
                # Assuming counter is separated by underscore before the extension
                number_str = file.split("_")[-1].split(extension)[0]
                existing_numbers.append(int(number_str))
            except ValueError:
                # Skip files without a counter in the name
                pass

        # Determine the next file number (considering existing files)
        if not existing_numbers:
            file_count = 1
        else:
            file_count = len(existing_numbers) + 1

        print(f"Previous training no is {file_count}")

        file_path = f"{base_name}_{second_name}_{file_count}{extension}"
        file_path_mat = f"{base_name}_{second_name}_{file_count}{extension_mat}"

        file_path = curr_dir / file_path
        file_path_mat = curr_dir / file_path_mat
        
        return [file_path, file_path_mat]

    def save_log_infos(self, data, file_path):

        npy_path = file_path[0]
        mat_path = file_path[1]

        #Convert to array
        data = np.array(data)

        np.save(str(npy_path), data)

        #__ MATLAB save ____________
        savemat(str(mat_path), {'data': data})

        print("LOG SAVED!")
        return

    def create_log_infos2(self, second_name = None):
        '''
        Function to save into .npy file in the current folder the results of the training
        data : list of relevant information
        returns: file_path : path to save the relevant information
        '''        
        #Identify the correct path:

        if self.save_models_flag:
            log_dir = self.trained_models_dir / str(self.all_args.seed)
            second_name = "training"

        location = self.trained_models_dir
        if self.multi_agent:
            location = self.trained_models_dir / "MultiAgent"
        
        if self.naive_training:
            log_dir = location / "NaiveTraining" 
            second_name = "naive_training"

        if self.joint_training:
            log_dir = location
            second_name = "joint_training" / "JointTraining" 

        if self.use_buffer:
            log_dir = location / "BufferReplay"
            second_name = "buffer_replay"

        if self.use_lwf:
            log_dir = location / "LwF"
            second_name = "training"

        if not log_dir.exists():
            os.makedirs(str(log_dir))
            curr_run = 'run_1'
        else:
            exst_run_nums = [int(str(folder.name).split('run_')[1]) for folder in log_dir.iterdir() if
                             str(folder.name).startswith('run_')]
            if len(exst_run_nums) == 0:
                curr_run = 'run_1'
            else:
                curr_run = 'run_%i' % (max(exst_run_nums) + 1)
        log_dir = log_dir / curr_run
        if not log_dir.exists():
            os.makedirs(str(log_dir))
        
        return log_dir

    def save_log_infos2(self, base_name, data, additional_info = None):

        extension = ".npy"
        extension_mat = ".mat"
        extension_plot = ".png"

        if self.save_models_flag:
            second_name = "training"
        
        if self.naive_training:
            second_name = "naive_training"

        if self.joint_training:
            second_name = "joint_training"

        if self.use_buffer:
            second_name = "buffer_replay"

        file_path = f"{self.results_dir}/{base_name}_{second_name}{extension}"
        file_path_mat = f"{self.results_dir}/{base_name}_{second_name}{extension_mat}"
        file_path_plot = f"{self.results_dir}/{base_name}_{second_name}_{additional_info}{extension_plot}"

        #Convert to array
        data = np.array(data)

        np.save(str(file_path), data)

        #__ MATLAB save ____________
        savemat(str(file_path_mat), {'data': data})

        #__Plot Save________________
        plt.plot(data, )

        plt.ylabel("incremental win rate")
        plt.xlabel("Episodes")

        plt.savefig(str(file_path_plot))

        print("LOG SAVED!")
        return

#________________________________________________________________________
###           ACTIVE AGENT MANAGEMENT

    def active_agent_choice(self):
        '''
        This function returns the AA but makes you choose the active agent 
        '''
        while True:
            try:
                active_agent = input(f"Choose the agent to train in MARL: ")
                if active_agent.isdigit() and 1 <= int(active_agent) <= self.num_agents:
                    print(f"You chose to train agent {active_agent}")
                    active_agent = int(active_agent) - 1
                    break
                else: print(f"Insert a valid number between 1 and {self.num_agents}")
            except ValueError:
                print(f"Invalid input. Please enter a number between 1 and {self.num_agents}.")      
        return active_agent

    def active_agent_init(self):
        
        #Ask if we want to initialize 
        answer_map = {"Y": True, "N": False, "y": True, "n": False}

        while True:
            try:
                key_input = input(f"Do you want to start fresh? Y/N ")
                answer = answer_map[key_input]

                if answer is True:
                    self.set_active_agent()
                    break
                else: break
            except ValueError:
                print("Invalid input. Please enter 'y' or 'n'")

    def set_active_agent(self):
        """ 
        Copies the nets of the active agent -taken from team 1 (the one that will learn to cooperate)
        with a different suffix (A). The new nets will be updated, preserving the original one. 
        """
        #TODO checkare che le reti esistono già
        agent = self.active_agent

        aa_team = 1

        #Copy ACTOR
        source_path =  f"{self.trained_models_dir}/{aa_team}/actor_agent{agent}.pt"
        dest_path = str(self.trained_models_dir) + "/actor_agent" + str(agent) + "A" + ".pt"
        shutil.copy2(source_path, dest_path)

        #Copy CRITIC
        source_path = f"{self.trained_models_dir}/{aa_team}/critic_agent{agent}.pt"
        dest_path = str(self.trained_models_dir) + "/critic_agent" + str(agent) + "A" + ".pt"
        shutil.copy2(source_path, dest_path)

        if self.all_args.use_valuenorm:
            source_path = f"{self.trained_models_dir}/{aa_team}/vnrom_agent{agent}.pt"
            dest_path = str(self.trained_models_dir) + '/vnrom_agent' + str(agent) + "A" + ".pt"
            shutil.copy2(source_path, dest_path)

    def load_active_agent(self):

        agent_id = self.active_agent

        policy_actor_state_dict = torch.load(str(self.trained_models_dir) + "/actor_agent" + str(agent_id) + "A" + ".pt")
        self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
        
        #Bias Check
        if self.show_biases:
            self.extract_biases_from_dict(policy_actor_state_dict, agent_id, 1)

        policy_critic_state_dict = torch.load(str(self.trained_models_dir) + "/critic_agent" + str(agent_id) + "A" + ".pt")
        self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

        if self.all_args.use_valuenorm:
            policy_vnrom_state_dict = torch.load(str(self.trained_models_dir) + '/vnrom_agent' + str(agent_id) + "A" + '.pt')
            self.trainer[agent_id].value_normalizer.load_state_dict(policy_vnrom_state_dict)

    def save_active_agent(self):
        agent_id = self.active_agent

        policy_actor = self.trainer[agent_id].policy.actor
        torch.save(policy_actor.state_dict(), str(self.trained_models_dir)  + "/actor_agent" + str(agent_id) + "A" + ".pt")
        
        #Print Biases
        if self.show_biases:
            self.extract_biases_from_dict(policy_actor.state_dict(), agent_id, 1)

        policy_critic = self.trainer[agent_id].policy.critic
        torch.save(policy_critic.state_dict(), str(self.trained_models_dir)  + "/critic_agent" + str(agent_id) + "A" + ".pt")
        
        if self.trainer[agent_id]._use_valuenorm:
            policy_vnrom = self.trainer[agent_id].value_normalizer
            torch.save(policy_vnrom.state_dict(), str(self.trained_models_dir) + "/vnrom_agent" + str(agent_id) + "A" + ".pt")

 #________________________________________________________________________
   ###            MULTI-AGENT TRAINING

    def active_multi_agent_init(self):
        
        #Ask if we want to initialize 
        answer_map = {"Y": True, "N": False, "y": True, "n": False}

        while True:
            try:
                key_input = input(f"Do you want to start fresh? Y/N ")
                answer = answer_map[key_input]

                if answer is True:
                    self.set_active_multi_agent()
                    break
                else: break
            except ValueError:
                print("Invalid input. Please enter 'y' or 'n'")
    
    def set_active_multi_agent(self, aa_team=1):
        """ 
        Copies the nets of the active agent -taken from team 1 (the one that will learn to cooperate)
        with a different suffix (A). The new nets will be updated, preserving the original one. 
        """
        #TODO checkare che le reti esistono già

        for agent in self.multi_active_agent:
            #Copy ACTOR
            source_path =  f"{self.trained_models_dir}/{aa_team}/actor_agent{agent}.pt"
            dest_path = str(self.agents_dir) + "/actor_agent" + str(agent) + "A" + ".pt"
            shutil.copy2(source_path, dest_path)

            #Copy CRITIC
            source_path = f"{self.trained_models_dir}/{aa_team}/critic_agent{agent}.pt"
            dest_path = str(self.agents_dir) + "/critic_agent" + str(agent) + "A" + ".pt"
            shutil.copy2(source_path, dest_path)

            if self.all_args.use_valuenorm:
                source_path = f"{self.trained_models_dir}/{aa_team}/vnrom_agent{agent}.pt"
                dest_path = str(self.agents_dir) + '/vnrom_agent' + str(agent) + "A" + ".pt"
                shutil.copy2(source_path, dest_path)
              
    def save_active_multi_agent(self):
        agent_id = self.active_agent

        policy_actor = self.trainer[agent_id].policy.actor
        torch.save(policy_actor.state_dict(), str(self.agents_dir)  + "/actor_agent" + str(agent_id) + "A" + ".pt")
        
        #Print Biases
        if self.show_biases:
            self.extract_biases_from_dict(policy_actor.state_dict(), agent_id, 1)

        policy_critic = self.trainer[agent_id].policy.critic
        torch.save(policy_critic.state_dict(), str(self.agents_dir)  + "/critic_agent" + str(agent_id) + "A" + ".pt")
        
        if self.trainer[agent_id]._use_valuenorm:
            policy_vnrom = self.trainer[agent_id].value_normalizer
            torch.save(policy_vnrom.state_dict(), str(self.agents_dir) + "/vnrom_agent" + str(agent_id) + "A" + ".pt")
  
    def load_active_multi_agent(self):

        for agent_id in self.multi_active_agent:

            policy_actor_state_dict = torch.load(str(self.agents_dir) + "/actor_agent" + str(agent_id) + "A" + ".pt")
            self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            
            # #Bias Check
            # if self.show_biases:
            #     self.extract_biases_from_dict(policy_actor_state_dict, agent_id, 1)

            policy_critic_state_dict = torch.load(str(self.agents_dir) + "/critic_agent" + str(agent_id) + "A" + ".pt")
            self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

            if self.all_args.use_valuenorm:
                policy_vnrom_state_dict = torch.load(str(self.agents_dir) + '/vnrom_agent' + str(agent_id) + "A" + '.pt')
                self.trainer[agent_id].value_normalizer.load_state_dict(policy_vnrom_state_dict)

            print(f"Loaded agent {agent_id}")

    def save_active_multi_agent(self):

        agent_id = self.active_agent
        for agent_id in self.multi_active_agent:
            policy_actor = self.trainer[agent_id].policy.actor
            torch.save(policy_actor.state_dict(), str(self.agents_dir)  + "/actor_agent" + str(agent_id) + "A" + ".pt")
            
            #Print Biases
            if self.show_biases:
                self.extract_biases_from_dict(policy_actor.state_dict(), agent_id, 1)

            policy_critic = self.trainer[agent_id].policy.critic
            torch.save(policy_critic.state_dict(), str(self.agents_dir)  + "/critic_agent" + str(agent_id) + "A" + ".pt")
            
            if self.trainer[agent_id]._use_valuenorm:
                policy_vnrom = self.trainer[agent_id].value_normalizer
                torch.save(policy_vnrom.state_dict(), str(self.agents_dir) + "/vnrom_agent" + str(agent_id) + "A" + ".pt")
        
    def load_teammates_multi(self, team):
        """
        Load the pretrained teams (as teammates) and value normalizer except for the active agent
        """
        for agent_id in range(self.num_agents):
            if agent_id not in self.multi_active_agent:
                policy_actor_state_dict = torch.load(str(self.trained_models_dir) + "/" + str(team) + '/actor_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)

                #Bias Check
                self.extract_biases_from_dict(policy_actor_state_dict, agent_id, team)

                policy_critic_state_dict = torch.load(str(self.trained_models_dir) + "/" + str(team) + '/critic_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

                policy_vnrom_state_dict = torch.load(str(self.trained_models_dir)  + "/" + str(team) + "/vnrom_agent" + str(agent_id) + ".pt")
                self.trainer[agent_id].value_normalizer.load_state_dict(policy_vnrom_state_dict)

                print(f"Loaded teammate {agent_id}")

 #________________________________________________________________________
    ###           JOINT TRAINING

    def joint_update_teams(self, team):

        for agent_id in range(self.num_agents):

            if agent_id != self.active_agent:
                policy_actor_state_dict = torch.load(str(self.trained_models_dir)  + "/" + str(team) + "/actor_agent" + str(agent_id) + "A" + ".pt")
                self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)

                policy_critic_state_dict = torch.load(str(self.trained_models_dir)  + "/" + str(team) + "/critic_agent" + str(agent_id) + "A" + ".pt")
                self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)


            if self.all_args.use_valuenorm:
                if agent_id == self.active_agent:
                    
                    policy_vnrom_state_dict = torch.load(str(self.trained_models_dir) + "/vnrom_agent" + str(agent_id) + "A" + ".pt")
                    self.trainer[agent_id].value_normalizer.load_state_dict(policy_vnrom_state_dict)
                else:
                    if self.all_args.use_valuenorm:
                        policy_vnrom_state_dict = torch.load(str(self.trained_models_dir) + "/" + str(team) +  '/vnrom_agent' + str(agent_id) + '.pt')
                        self.trainer[agent_id].value_normalizer.load_state_dict(policy_vnrom_state_dict)



                #_______________________________________________________________________
                ####Fare una function o sistemare 

    def freeze_train(self):

        train_infos = []
        update = False

        # random update order not needed

        action_dim=self.buffer[0].actions.shape[-1]
        factor = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(factor)
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])
            

            old_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            


            train_info = self.trainer[agent_id].train(self.buffer[agent_id], update)


            new_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            


            factor = factor*_t2n(torch.prod(torch.exp(new_actions_logprob-old_actions_logprob),dim=-1).reshape(self.episode_length,self.n_rollout_threads,1))
            train_infos.append(train_info)      
            self.buffer[agent_id].after_update()
        
        return train_infos

    #________________________________________________________________________
    #________________________________________________________________________
    ###          BIAS SANITY CHECK

    def extract_biases_from_dict(self, state_dict, agent, team):
        """
        Attempts to extract biases from a loaded state dictionary (**not recommended**).

        This function is fragile and might not work for all model architectures or checkpoint
        saving/loading methods. Use with caution.

        Args:
            state_dict (OrderedDict): The loaded state dictionary from the checkpoint.

        Returns:
            list: A list of bias tensors (if found), or an empty list if no biases are found.
        """
        biases = []
        for key, value in state_dict.items():
            if key.endswith(".bias"):
                biases.append(value)
        if self.show_biases:
            print(f"Biases for Agent {agent + 1} and team {team}")
            print(biases[-1])

