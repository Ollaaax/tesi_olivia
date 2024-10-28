import numpy as np
import os
import sys
from pathlib import Path
import random
import torch
import shutil
import random

from onpolicy.utils import buffer_utils

def _t2n(x):
    return x.detach().cpu().numpy()

class Buffer_Utils():

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

    def buffer_foreachteam_creation(self):
        self.rebuf_outs = [[] for _ in range(self.num_agents)]
        self.rebuf_ins = [[] for _ in range(self.num_agents)]

        buffer_team_no = self.manually_choice_of_team()

        #TODO Choose how many samples we want to save -> this will be the name of the directory along with the team
        #25*10*128 numero degli step (delle traiettorie), ogni 25 steps viene fatto il ppo update. Prendiamo 

        #Create the directory to save the buffers
        self.buffer_dir = self.trained_models_dir / str(buffer_team_no) / ("Buffer2T" + str(buffer_team_no))
        if not os.path.exists(self.buffer_dir):
            os.makedirs(self.buffer_dir)
        else: 
            self.ask_if_want_to_reset_buffer()

        for agent_id in range(self.num_agents):
            self.restore_pretrained_models(agent_id, buffer_team_no)

    #____________TRAIN FUNCTIONS________________________________________

    def freeze_save_trainoldold(self):

        train_infos = []
        update = False

        # random update order not needed

        action_dim=self.buffer[0].actions.shape[-1]
        factor = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)

        for agent_id in range(self.num_agents):
            #_____________________________________________________________
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])
            

            self.trainer[agent_id].prep_rollout()
        #1. costruire il sample in 
    
            # obs, rnn_states, masks, available_actions
            input_data = self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]), \
                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]), \
                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]), \
                            available_actions
            
            # print(available_actions.shape)

            self.rebuf_ins[agent_id].append(input_data)
        #3. prendere il sample out da act.get_logits

            sample_logits_out = self.trainer[agent_id].policy.actor.get_logit_forward(*input_data)

            self.rebuf_outs[agent_id].append(sample_logits_out.logits)

            #_____________________________________________________________
            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(factor)


            old_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            


            #come input data abbiamo al momento: obs, rnn_states, actions, masks, available_actions, active masks 
            input_data = self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]), \
                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]), \
                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]), \
                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]), \
                            available_actions, \
                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:])


            # self.rebuf_ins[agent_id].append(input_data)

            train_info = self.trainer[agent_id].train(self.buffer[agent_id], update)


            new_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            

            # self.rebuf_outs[agent_id].append(new_actions_logprob)

            factor = factor*_t2n(torch.prod(torch.exp(new_actions_logprob-old_actions_logprob),dim=-1).reshape(self.episode_length,self.n_rollout_threads,1))
            train_infos.append(train_info)      
            self.buffer[agent_id].after_update()
        
        torch.save(self.rebuf_ins, str(self.buffer_dir) + "/replay_buffer_ins.npy")
        torch.save(self.rebuf_outs, str(self.buffer_dir) + "/replay_buffer_outs.npy")

        return train_infos

    def freeze_save_train(self):

        train_infos = []
        update = False

        # random update order not needed

        action_dim=self.buffer[0].actions.shape[-1]
        factor = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)

        for agent_id in range(self.num_agents):

            #_____________________________________________________________
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])
            

            self.trainer[agent_id].prep_rollout()
        #1. costruire il sample in 
    
            # obs, rnn_states, masks, available_actions
            input_data = self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]), \
                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]), \
                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]), \
                            available_actions
            
            # print(available_actions.shape)

            self.rebuf_ins[agent_id].append(input_data)
        #3. prendere il sample out da act.get_logits

            sample_logits_out = self.trainer[agent_id].policy.actor.get_logit_forward(*input_data)

            self.rebuf_outs[agent_id].append(sample_logits_out.logits)

            #_____________________________________________________________
            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(factor)


            old_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            

            # self.rebuf_ins[agent_id].append(input_data)

            train_info = self.trainer[agent_id].train(self.buffer[agent_id], update)


            new_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            

            # self.rebuf_outs[agent_id].append(new_actions_logprob)

            factor = factor*_t2n(torch.prod(torch.exp(new_actions_logprob-old_actions_logprob),dim=-1).reshape(self.episode_length,self.n_rollout_threads,1))
            train_infos.append(train_info)      
            self.buffer[agent_id].after_update()
        
        torch.save(self.rebuf_ins, str(self.buffer_dir) + "/replay_buffer_ins.npy")
        torch.save(self.rebuf_outs, str(self.buffer_dir) + "/replay_buffer_outs.npy")

        return train_infos


    def freeze_save_trainnew(self):

        train_infos = []
        
        #1. eval mode 
        for agent_id in range(self.num_agents):

            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])

            self.trainer[agent_id].prep_rollout()
        
        
        #1. costruire il sample in 
    
            # obs, rnn_states, masks, available_actions
            input_data = self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]), \
                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]), \
                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]), \
                            available_actions
            
            # print(available_actions.shape)

            self.rebuf_ins[agent_id].append(input_data)
        #3. prendere il sample out da act.get_logits

            sample_logits_out = self.trainer[agent_id].policy.actor.get_logit_forward(*input_data)

            self.rebuf_outs[agent_id].append(sample_logits_out.logits)

        torch.save(self.rebuf_ins, str(self.buffer_dir) + "/replay_buffer_ins.npy")
        torch.save(self.rebuf_outs, str(self.buffer_dir) + "/replay_buffer_outs.npy")


        # print(f"len buff in: {len(self.rebuf_ins)}")
        # print(f"len buffagents in: {len(self.rebuf_ins[0])}")
        # print(f"len buff ep in: {len(self.rebuf_ins[0][0])}")
        # print(f"len buff robe (4): {len(self.rebuf_ins[0][0][0])}")
        # print(f"len buff obs: {len(self.rebuf_ins[0][0][0][0])}")

        # print(f"len buff out: {len(self.rebuf_outs)}")
        # print(f"len buff ag: {len(self.rebuf_outs[0])}")

        # print(f"len buff logt: {self.rebuf_outs[0][0]}")
        # print(f"len buff logt: {(self.rebuf_outs[0][0]).shape}")
        return train_infos

    def train_with_rebuf(self):
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
            
            # print(f"Availbale acltion: {available_actions.shape}")
            
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
    

    def train_with_rebuf_multi(self):
        train_infos = []
        # random update order

        action_dim=self.buffer[0].actions.shape[-1]
        factor = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)

        agents_list = []
        for i in range(self.num_agents):
            update = True if i in self.multi_active_agent else False
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

    def freeze_train2(self):

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

    #____________UTILS FUNCTIONS_________________________________________

    def ask_if_want_to_reset_buffer(self):
    #INITIALIZE AGENTS AND FRESH START 
        answer_map = {"Y": True, "N": False, "y": True, "n": False}

        # TODO Check if an agent is already stored
        while True:
            try:
                key_input = input(f"Do you want to start fresh? Y/N ")
                answer = answer_map[key_input]
                if answer is False:
                    print(f"Buffer for team {self.buffer_team_no} already existing. Aborting")
                    sys.exit()
                else: break
            except ValueError:
                print("Invalid input. Please enter 'y' or 'n'")

    def team_init(self):
        self.active_agent_choice2()

        self.active_agent = self.set_active_agent2()

        team = self.manually_choice_of_team()

        for agent_id in range(self.num_agents):
            if agent_id != self.active_agent: 
                self.load_teammates(team)
            else:
                self.load_active_agent2()

    #___________MANAGEMENT_______________________________________________

    def team_assemblation(self):
        '''Idea: 
        if i want to SAVE the buffer
            > la funzione ci pensa da sé
            ________________________________________________
        if I want to TRAIN with Buffer, then setting is:
            > Choose which agent train 
            > AA copied in the folder from team 1 
            > Net initialized with such agent 
            > Buffer creation and assigning to my team
            > Model of AA saved in the folder at each episode
            > Team Loaded 1 (not necessary) or 2 (to train)
            > Perform 100 episodes 
            _________________________________________________
        if I want to TEST the training: 
            > Take the agent from the folder 
            > Initialize net with such agent
            > Load the original team (= 1)
            > Perform 10 Episodes 
        '''

        #CREATION OF THE BUFFER
        if self.save_buffer:
            #Create the empty Replay Buffer
            self.buffer_foreachteam_creation()
            self.num_env_steps = self.n_rollout_threads * self.episode_length * 1
        #______________________________________________________
        
        if self.use_buffer and not self.multi_agent:  
            # TRAIN WITH BUFFER  
            if not self.buffer_test:
                #Choose, set and load active agent and teams
                self.active_agent = self.active_agent_choice()

                self.set_active_agent() #Copies A from T1 in folder

                #Create buffer
                rebuf_in, rebuf_out = self.shuffle_n_portion_buffer()
                #It must be passed to trainer
                self.trainer[self.active_agent].set_buffers(rebuf_in, rebuf_out)

                self.team = self.manually_choice_of_team() #Choose team

                self.num_env_steps = self.ep_no_rebuf_train * self.episode_length * self.n_rollout_threads

            # TEST with buffer
            else:
                self.team = 1 #For test use team 1 only

                #Perform 10 Episodes only
                self.num_env_steps = 50 * self.episode_length * self.n_rollout_threads

            self.load_teammates(self.team) #Initialize nets with teammates

            self.load_active_agent() #Load the Active Agent copied in folder
            #____________________________________________________
        
        #MULTIAGENT
        if self.use_buffer and self.multi_agent:  
            self.multi_active_agent = [0, 1, 2, 3]

            # TRAIN WITH BUFFER  
            if not self.buffer_test:
                #Choose, set and load active agent and teams
                self.multi_active_agent = [0, 1, 2, 3]

                self.active_multi_agent_init() #Copies A from T1 in folder

                #Create buffer
                rebuf_in, rebuf_out = self.shuffle_n_portion_multi_buffer()
     
                #It must be passed to trainer
                for agent in self.multi_active_agent:
                    self.trainer[agent].set_buffers(rebuf_in[agent], rebuf_out[agent])

                self.team = self.manually_choice_of_team() #Choose team

                self.num_env_steps = self.ep_no_rebuf_train * self.episode_length * self.n_rollout_threads
            
            # TEST with buffer
            else:
                self.team = 1 #For test use team 1 only

                #Perform 50 Episodes only
                self.num_env_steps = 50 * self.episode_length * self.n_rollout_threads

            self.load_teammates_multi(self.team) #Initialize nets with teammates

            #TODO CHANGE 
            self.load_active_multi_agent() #Load the Active Agent copied in folder
            #____________________________________________________


    def rebuf_train(self):
        # compute return and update network
        if self.save_buffer:
            train_infos = self.freeze_save_train()

        elif self.use_buffer:
            # train_infos = (0 if self.buffer_test else self.train_with_rebuf())
            if self.buffer_test: 
                train_infos = self.freeze_train2()
            else:
                if self.multi_agent:
                    train_infos = self.train_with_rebuf_multi()
                else: 
                    train_infos = self.train_with_rebuf()

        return train_infos
    

    
    def OLDshuffle_n_portion_buffer(self):
        """Here I import the buffer, shuffle it along episodes and steps, and take a portion
        Input: semmai buffer_team_no
        Output: Shuffled and portioned Buffers
        """
        #TODO cambiare eventualmente la sqaudra di partenza 
        rebuf_in, rebuf_out = buffer_utils.import_buffer(1)
        # print(f"Buff out len: {len(rebuf_out)}")
        # print(f"Buff out 2: {(rebuf_out[0][0])}")

        #PORZIONO IL BUFFER 
        # 1. seleziono solo la colonna relativa all'active agent
        rebuf_in = rebuf_in[self.active_agent] 
        rebuf_out = rebuf_out[self.active_agent]

        # 2. Setto la percentuale di uso del buffer (tramite config?)
        buff_episodes_len = len(rebuf_in)
        
        index_array = np.arange(buff_episodes_len * self.episode_length)
        np.random.shuffle(index_array)
        # print(f"Indeces array {index_array}")
        # print(f"Index len {len(index_array)}")

        # 3. Shuffle + Portion
        new_buf_in = [] 
        new_buf_out = []
        for i in index_array:
            ep_no = i // self.episode_length
            step_no = i % self.episode_length

            old_sample_in = [   rebuf_in[ep_no][0][step_no:step_no+self.n_rollout_threads],
                                rebuf_in[ep_no][1], 
                                rebuf_in[ep_no][2][step_no:step_no+self.n_rollout_threads], 
                                rebuf_in[ep_no][3][step_no:step_no+self.n_rollout_threads], 
                                rebuf_in[ep_no][4][step_no:step_no+self.n_rollout_threads], 
                                # rebuf_in[ep_no][4], 
                                rebuf_in[ep_no][5][step_no:step_no+self.n_rollout_threads]]
            
            new_buf_in.append(old_sample_in)
            new_buf_out.append(rebuf_out[ep_no][step_no:step_no+self.n_rollout_threads])
            
        rebuf_in = new_buf_in[:len(new_buf_in)*self.pcnt_buffer//100]
        rebuf_out = new_buf_out[:len(new_buf_in)*self.pcnt_buffer//100]

        print(f"The size of the buffer is {len(rebuf_in)}")
        # 4. Voilà nuovo buffer 
        return rebuf_in, rebuf_out

    def shuffle_n_portion_buffer(self):
        """Here I import the buffer, shuffle it along episodes and steps, and take a portion
        Input: semmai buffer_team_no
        Output: Shuffled and portioned Buffers
        """
        #TODO cambiare eventualmente la sqaudra di partenza 
        rebuf_in, rebuf_out = buffer_utils.import_buffer(1)

        #PORZIONO IL BUFFER 
        # 1. seleziono solo la colonna relativa all'active agent
        rebuf_in = rebuf_in[self.active_agent] 
        rebuf_out = rebuf_out[self.active_agent]

        # print(len(rebuf_in[0][3][0]))
        # 2. Setto la percentuale di uso del buffer (tramite config?)
        buff_episodes_len = len(rebuf_in)
        
        index_array = np.arange(buff_episodes_len * self.episode_length)
        np.random.shuffle(index_array)
        # print(f"Indeces array {index_array}")
        # print(f"Index len {len(index_array)}")

        # 3. Shuffle + Portion
        new_buf_in = [] 
        new_buf_out = []

        for i in index_array:
            ep_no = i // self.episode_length
            step_no = i % self.episode_length

            step_no = step_no * self.n_rollout_threads

            # print(rebuf_in[ep_no][3][0])
            old_sample_in = [   rebuf_in[ep_no][0][step_no:step_no+self.n_rollout_threads],  #obs
                                rebuf_in[ep_no][1],  #rnn_states
                                rebuf_in[ep_no][2][step_no:step_no+self.n_rollout_threads],  #masks
                                rebuf_in[ep_no][3][step_no:step_no+self.n_rollout_threads],  #available_actions
                                # rebuf_in[ep_no][3]
                            ]

            # print(len(rebuf_in[ep_no][3][0]))
            # print(rebuf_in[ep_no][0][step_no:step_no+self.n_rollout_threads].shape)
            # sys.exit()
            new_buf_in.append(old_sample_in)
            new_buf_out.append(rebuf_out[ep_no][step_no:step_no+self.n_rollout_threads])
            
        rebuf_in = new_buf_in[:len(new_buf_in)*self.pcnt_buffer//100]
        rebuf_out = new_buf_out[:len(new_buf_in)*self.pcnt_buffer//100]

        print(f"The size of the buffer is {len(rebuf_in)}")
        # print(f"The size of the OSI {len(rebuf_in[0][3][0])}")
        # print(f"The size of the OSO is 400 {len(rebuf_out)}")
        # print(f"The size of the OSO is 1 il logit {rebuf_out[0]}")

        # logit = torch.nn.functional.softmax(rebuf_out[0], dim=1)
        # print(torch.round(logit*100))
        #  print(torch.sum(logit, dim=1))
        # 4. Voilà nuovo buffer 
        return rebuf_in, rebuf_out


    def shuffle_n_portion_multi_bufferOLD(self):
        """Here I import the buffer, shuffle it along episodes and steps, and take a portion
        Input: semmai buffer_team_no
        Output: Shuffled and portioned Buffers
        """
        #TODO cambiare eventualmente la sqaudra di partenza 
        rebuf_in, rebuf_out = buffer_utils.import_buffer(1)

        new_rebuf_outs = [[] for _ in range(len(self.multi_active_agent))]
        new_rebuf_ins = [[] for _ in range(len(self.multi_active_agent))]

        #PORZIONO IL BUFFER 
        # 1. seleziono solo la colonna relativa all'active agent
        for agent in self.multi_active_agent:
            print(f"AGENT is {agent}")

            # 2. Setto la percentuale di uso del buffer (tramite config?)
            buff_episodes_len = len(rebuf_in[agent])
            
            index_array = np.arange(buff_episodes_len * self.episode_length)
            np.random.shuffle(index_array)
            # print(f"Indeces array {index_array}")
            # print(f"Index len {len(index_array)}")

            # 3. Shuffle + Portion
            new_buf_in = [] 
            new_buf_out = []
            for i in index_array:
                ep_no = i // self.episode_length
                step_no = i % self.episode_length

                old_sample_in = [   rebuf_in[agent][ep_no][0][step_no:step_no+self.n_rollout_threads],
                                    rebuf_in[agent][ep_no][1], 
                                    rebuf_in[agent][ep_no][2][step_no:step_no+self.n_rollout_threads], 
                                    rebuf_in[agent][ep_no][3][step_no:step_no+self.n_rollout_threads], 
                                    rebuf_in[agent][ep_no][4][step_no:step_no+self.n_rollout_threads], 
                                    rebuf_in[agent][ep_no][5][step_no:step_no+self.n_rollout_threads]]
                
                new_buf_in.append(old_sample_in)
                new_buf_out.append(rebuf_out[agent][ep_no][step_no:step_no+self.n_rollout_threads])
                
            new_rebuf_ins[agent] = new_buf_in[:len(new_buf_in)*self.pcnt_buffer//100]
            new_rebuf_outs[agent] = new_buf_out[:len(new_buf_in)*self.pcnt_buffer//100]

        print(f"The size of the buffer is {len(new_rebuf_ins)}")
        print(f"The size of the buffer is {len(new_rebuf_ins[0])}")
        # 4. Voilà nuovo buffer 
        return new_rebuf_ins, new_rebuf_outs

    def shuffle_n_portion_multi_buffer(self):
        """Here I import the buffer, shuffle it along episodes and steps, and take a portion
        Input: semmai buffer_team_no
        Output: Shuffled and portioned Buffers
        """
        #TODO cambiare eventualmente la sqaudra di partenza 
        rebuf_in, rebuf_out = buffer_utils.import_buffer(1)

        new_rebuf_outs = [[] for _ in range(len(self.multi_active_agent))]
        new_rebuf_ins = [[] for _ in range(len(self.multi_active_agent))]

        #PORZIONO IL BUFFER 
        # 1. seleziono solo la colonna relativa all'active agent
        for agent in self.multi_active_agent:
            print(f"AGENT is {agent}")

            # 2. Setto la percentuale di uso del buffer (tramite config?)
            buff_episodes_len = len(rebuf_in[agent])
            
            index_array = np.arange(buff_episodes_len * self.episode_length)
            np.random.shuffle(index_array)
            # print(f"Indeces array {index_array}")
            # print(f"Index len {len(index_array)}")

            # 3. Shuffle + Portion
            new_buf_in = [] 
            new_buf_out = []
            for i in index_array:
                ep_no = i // self.episode_length
                step_no = i % self.episode_length

                step_no = step_no * self.n_rollout_threads

                old_sample_in = [   rebuf_in[agent][ep_no][0][step_no:step_no+self.n_rollout_threads],  #obs
                                    rebuf_in[agent][ep_no][1],                                          #rnn_states
                                    rebuf_in[agent][ep_no][2][step_no:step_no+self.n_rollout_threads],  #masks
                                    rebuf_in[agent][ep_no][3][step_no:step_no+self.n_rollout_threads],  #available_actions
                                ]   
                
                new_buf_in.append(old_sample_in)
                new_buf_out.append(rebuf_out[agent][ep_no][step_no:step_no+self.n_rollout_threads])
                
            new_rebuf_ins[agent] = new_buf_in[:len(new_buf_in)*self.pcnt_buffer//100]
            new_rebuf_outs[agent] = new_buf_out[:len(new_buf_in)*self.pcnt_buffer//100]

        print(f"The size of the buffer is {len(new_rebuf_ins)}")
        # print(f"The size of the buffer is {len(new_rebuf_ins[0])}")
        # 4. Voilà nuovo buffer 
        return new_rebuf_ins, new_rebuf_outs

