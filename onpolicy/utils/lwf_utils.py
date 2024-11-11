import numpy as np
import os
import sys
from pathlib import Path
import random
import torch
import shutil
import random

def _t2n(x):
    return x.detach().cpu().numpy()

class LwF_Utils():
    
    def lwf_train(self):
        train_infos = []
        # random update order

        action_dim=self.buffer[0].actions.shape[-1]
        factor = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        
        agents_list = []
        for i in range(self.num_agents):
            update = True if i == self.active_agent else False
            agents_list.append((i, update))

        for i in torch.randperm(self.num_agents):

            agent_id, update = agents_list[i]

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

            train_info = self.trainer[agent_id].train(self.buffer[agent_id], self.teach_tr, update)


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

    def freeze_train2(self):

        train_infos = []
        update = False

        # random update order not needed

        action_dim=self.buffer[0].actions.shape[-1]
        factor = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            self.buffer[agent_id].update_factor(factor)
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])
            

            old_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            


            train_info = self.trainer[agent_id].train(self.buffer[agent_id], self.teach_tr, update)


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
    
    #_______________________________

    def teacher_creation(self):
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
        share_observation_space = self.envs.share_observation_space[self.active_agent] if self.use_centralized_V else self.envs.observation_space[self.active_agent]
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
        print(f"Loaded Team for the teacher no {team}")  

    def team_assemblation_lwf(self):
        #________TRAIN________________
        if not self.lwf_test:
            self.active_agent = self.active_agent_choice()
            self.active_agent_init()

            self.load_active_agent()
            #________________________________
            team = self.manually_choice_of_team()
            self.load_teammates(team)
        
        #________TEST_____________________
        else: 
            self.active_agent = self.active_agent_choice()
            self.load_active_agent()
            #________________________________
            team = 1
            self.load_teammates(team)

        self.num_env_steps = self.episodes_no * self.episode_length * self.n_rollout_threads
        
        #TEACHER CREATION
        self.teacher_creation()   