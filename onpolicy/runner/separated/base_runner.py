
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
from pathlib import Path

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
        self.episodes_no = self.all_args.ep_no
        # self.continual_flag = self.all_args.continual

        self.use_lwf = self.all_args.use_lwf
        self.lwf_test = self.all_args.lwf_test

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
            self.results_dir, self.agents_dir = self.create_log_infos2()
        #___________________________________________

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        if self.use_lwf:
            print(f"CrossEntropy? {self.all_args.usexe}")

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
            

            factor = factor*_t2n(torch.prod(torch.exp(new_actions_logprob-old_actions_logprob),dim=-1).reshape(self.episode_length,self.n_rollout_threads,1))
            train_infos.append(train_info)      
            self.buffer[agent_id].after_update()

        return train_infos

#________________________________________________________________________
###           TEAMS MANAGEMENT

    def manually_choice_of_team(self):

        #Check how many teams are trained
        exst_teams_no = [int(str(folder.name)) for folder in self.trained_models_dir.iterdir() if str(folder.name).isdigit()]
        if len(exst_teams_no) == 0:
            print("No TEAMS trained. SORRY")
            sys.exit()

        exst_teams_no = len(exst_teams_no)

        while True:
            try:
                team_no = input(f"Choose the Team ")
                if team_no.isdigit() and 1 <= int(team_no) <= exst_teams_no:
                    print(f"You chose to load team {team_no}")
                    team_no = int(team_no)
                    break
                else: print(f"Insert a valid number between 1 and {exst_teams_no}")
            except ValueError:
                print(f"Invalid input. Please enter a number between 1 and {exst_teams_no}.")  
        return team_no

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

    def create_log_infos2(self):
        '''
        Function to save into .npy file in the current folder the results of the training
        data : list of relevant information
        returns: file_path : path to save the relevant information
        '''        
        #Identify the correct path:

        location = Path(os.path.dirname(self.trained_models_dir))
        ##_____ SAVE TEAMS _______
        if self.save_models_flag:
            log_dir = self.trained_models_dir / str(self.all_args.seed)
    

        ##_____ MULTIAGENT _______
        
        if self.multi_agent:
            log_dir = location / "MultiAgent"
            if self.naive_training:
                log_dir = log_dir / "NaiveTraining" 
            if self.use_buffer:
                log_dir = log_dir / "BufferReplay"               
            
        ##______ NAIVE ____________
        if self.naive_training:
            log_dir = location / "NaiveTraining" 
            

        # if self.joint_training:
        #     log_dir = location
        #     second_name = "joint_training" / "JointTraining" 

        ##______ BUFFER ____________
        if self.use_buffer:
            log_dir = location / "BufferReplay"
            
        ##______ LwF ____________
        if self.use_lwf:
            log_dir = location / "LwF"
            
        #____ AGENTS ______________________
        agent_dir = Path(log_dir) / "Agents"

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
    
        if not agent_dir.exists():
            os.makedirs(str(agent_dir))

        
        return log_dir, agent_dir

    def save_log_infos2OLD(self, base_name, data, additional_info = None):

        extension = ".npy"
        extension_mat = ".mat"
        extension_plot = ".png"

        # if self.save_models_flag:
        #     second_name = "training"
        
        # if self.naive_training:
        #     second_name = "naive_training"

        # if self.joint_training:
        #     second_name = "joint_training"

        # if self.use_buffer:
        #     second_name = "buffer_replay"


        if additional_info is not None: 
            file_path = f"{self.results_dir}/{base_name}_{second_name}{additional_info}{extension}"
            file_path_mat = f"{self.results_dir}/{base_name}_{second_name}{additional_info}{extension_mat}"
            file_path_plot = f"{self.results_dir}/{base_name}_{second_name}{additional_info}{extension_plot}"

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

    def save_log_infos2(self, base_name, data, additional_info = None):

        extension = ".npy"
        extension_mat = ".mat"
        extension_plot = ".png"

        if self.save_models_flag:
            second_name = "team"
            additional_info = self.all_args.seed
        
        if self.naive_training:
            second_name = "naive"

        # if self.joint_training:
        #     second_name = "joint_training"

        if self.use_buffer:
            second_name = "rebuf"

        if self.use_lwf:
            second_name = "rebuf"

        if additional_info is not None: 
            file_path = f"{self.results_dir}/{base_name}_{second_name}{additional_info}{extension}"
            file_path_mat = f"{self.results_dir}/{base_name}_{second_name}{additional_info}{extension_mat}"
            file_path_plot = f"{self.results_dir}/{base_name}_{second_name}{additional_info}{extension_plot}"

        else: 
            file_path = f"{self.results_dir}/{base_name}_{second_name}{extension}"
            file_path_mat = f"{self.results_dir}/{base_name}_{second_name}{extension_mat}"
            file_path_plot = f"{self.results_dir}/{base_name}_{second_name}{extension_plot}"

        #Convert to array
        data = np.array(data)

        np.save(str(file_path), data)

        #__ MATLAB save ____________
        mat_name = "TRAIN" if not (self.buffer_test or self.lwf_test or self.naive_test) else "TEST"
        savemat(str(file_path_mat), {f'{mat_name}': data})

        #__Plot Save________________
        plt.figure()
        plt.plot(data, )

        plt.ylabel(f"{base_name}")
        plt.xlabel("Episodes")

        plt.savefig(str(file_path_plot))

        # print("LOG SAVED!")
        return

    def save_log_losses(self, ppo_loss, l1_rebuf_loss, l2_rebuf_loss, overall_loss):
        
        losses = {
                    "ppo_loss": ppo_loss,
                    "l1_rebuf_loss": l1_rebuf_loss,
                    "l2_rebuf_loss": l2_rebuf_loss,
                    "overall_loss": overall_loss
                }
        extension = ".npy"
        extension_mat = ".mat"
        extension_plot = ".png"

        loss_dir = self.results_dir / "Losses"
        if not loss_dir.exists():
            os.makedirs(str(loss_dir))

        for name, el in losses.items():
            file_path = f"{loss_dir}/{name}{extension}"
            file_path_mat = f"{loss_dir}/{name}{extension_mat}"
            file_path_plot = f"{loss_dir}/{name}{extension_plot}"

            #Convert to array
            el = np.array(el)

            np.save(str(file_path), el)

            #__ MATLAB save ____________
            savemat(str(file_path_mat), {f'{name}': el})

            #__Plot Save________________
            plt.figure()
            plt.plot(el, )

            plt.ylabel(f"{name}")
            plt.xlabel("Episodes")

            plt.savefig(str(file_path_plot))

            # print("Losses LOG SAVED!")

    def save_log_losses_lwf(self, ppo_loss, xentropy_loss, l2_lwf_loss, overall_loss):
        
        losses = {
                    "ppo_loss": ppo_loss,
                    "xentropy_loss": xentropy_loss,
                    "l2_lwf_loss": l2_lwf_loss,
                    "overall_loss": overall_loss
                }
        extension = ".npy"
        extension_mat = ".mat"
        extension_plot = ".png"

        loss_dir = self.results_dir / "Losses"
        if not loss_dir.exists():
            os.makedirs(str(loss_dir))

        for name, el in losses.items():
            file_path = f"{loss_dir}/{name}{extension}"
            file_path_mat = f"{loss_dir}/{name}{extension_mat}"
            file_path_plot = f"{loss_dir}/{name}{extension_plot}"

            #Convert to array
            el = np.array(el)

            np.save(str(file_path), el)

            #__ MATLAB save ____________
            savemat(str(file_path_mat), {f'{name}': el})

            #__Plot Save________________
            plt.figure()
            plt.plot(el, )

            plt.ylabel(f"{name}")
            plt.xlabel("Episodes")

            plt.savefig(str(file_path_plot))

            # print("Losses LOG SAVED!")

       
       
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

    def load_active_agent(self):

        agent_id = self.active_agent

        policy_actor_state_dict = torch.load(str(self.agents_dir) + "/actor_agent" + str(agent_id) + "A" + ".pt")
        self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
        
        #Bias Check
        if self.show_biases:
            self.extract_biases_from_dict(policy_actor_state_dict, agent_id, 1)

        policy_critic_state_dict = torch.load(str(self.agents_dir) + "/critic_agent" + str(agent_id) + "A" + ".pt")
        self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

        if self.all_args.use_valuenorm:
            policy_vnrom_state_dict = torch.load(str(self.agents_dir) + '/vnrom_agent' + str(agent_id) + "A" + '.pt')
            self.trainer[agent_id].value_normalizer.load_state_dict(policy_vnrom_state_dict)

    def save_active_agent(self):
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
    ###           NAIVE/SAVE MODELS TRAINING
    def naive_training_setting(self):

        #_____TRAIN setting___________
        if not self.naive_test:
        #ACTIVE AGENT
            self.active_agent = self.active_agent_choice() #choice -> copy2dir -> init the nets
    
            self.active_agent_init() #Ask if we want to initialize, if yes it does it
            self.load_active_agent() #Load the NN of the AA

            #TEAMMATES
            team = self.manually_choice_of_team()

        #_____TEST setting______________
        else: 
            self.active_agent = self.active_agent_choice()
            self.load_active_agent() #Load the NN of the AA
            
            team = 1
        
        self.load_teammates(team)
        self.num_env_steps = self.episode_length * self.n_rollout_threads * self.episodes_no

    def naive_training_setting_multi(self):

        self.multi_active_agent = [0, 1, 2, 3]

        #_____TRAIN setting___________
        if not self.naive_test:

            self.active_multi_agent_init()
            self.load_active_multi_agent()

            team = self.manually_choice_of_team()

        #_____TEST setting______________
        else:
            self.load_active_multi_agent()
            team = 1
            
        self.load_teammates_multi(team)
        self.num_env_steps = self.episode_length * self.n_rollout_threads * self.episodes_no

    def saving_agents(self):
        #_____ NAIVE TRAINING______________________________
        if self.naive_training and not self.multi_agent:
            self.save_active_agent()
        
        if self.naive_training and self.multi_agent:
            self.ave_active_multi_agent()

        #__________________________________________________________________________________
        #_____SAVING MODELS_________________________________
        if self.save_models_flag:
            self.save_teams_seed()

#__________________________________________________________________________________

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

