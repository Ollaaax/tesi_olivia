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
        self.buffer_dir = self.trained_models_dir / str(buffer_team_no) / ("BufferT" + str(buffer_team_no))
        if not os.path.exists(self.buffer_dir):
            os.makedirs(self.buffer_dir)
        else: 
            self.ask_if_want_to_reset_buffer()

        for agent_id in range(self.num_agents):
            self.restore_pretrained_models(agent_id, buffer_team_no)

    #____________TRAIN FUNCTIONS________________________________________

    def freeze_save_train(self):

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
            

            #come input data abbiamo al momento: obs, rnn_states, actions, masks, available_actions, active masks 
            input_data = self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]), \
                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]), \
                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]), \
                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]), \
                            available_actions, \
                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:])

            # print("Size of input0 " + str(len(input_data[0])))
            # print("Size of input1 " + str(len(input_data[1])))
            # print("Size of input2 " + str(len(input_data[2])))
            # print("Size of input3 " + str(len(input_data[3])))

            # print(input_data[0][0])
            # print(input_data[0][0])

            self.rebuf_ins[agent_id].append(input_data)
            print(len(self.rebuf_ins[agent_id]))

            train_info = self.trainer[agent_id].train(self.buffer[agent_id], update)


            new_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            

            self.rebuf_outs[agent_id].append(new_actions_logprob)

            factor = factor*_t2n(torch.prod(torch.exp(new_actions_logprob-old_actions_logprob),dim=-1).reshape(self.episode_length,self.n_rollout_threads,1))
            train_infos.append(train_info)      
            self.buffer[agent_id].after_update()
        
        torch.save(self.rebuf_ins, str(self.buffer_dir) + "/replay_buffer_ins.npy")
        torch.save(self.rebuf_outs, str(self.buffer_dir) + "/replay_buffer_outs.npy")

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
            
            
            old_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                        self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                        self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                        self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                        available_actions,
                                                        self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            train_info = self.trainer[agent_id].train(self.buffer[agent_id], self.active_agent, update)


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
            


            train_info = self.trainer[agent_id].train(self.buffer[agent_id], self.active_agent, update)


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
        
        if self.use_buffer:  
            # TRAIN WITH BUFFER  
            if not self.buffer_test:
                #Choose, set and load active agent and teams
                self.active_agent = self.active_agent_choice()

                self.set_active_agent() #Copies A from T1 in folder

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

    def rebuf_train(self):
        # compute return and update network
        if self.save_buffer:
            train_infos = self.freeze_save_train()

        elif self.use_buffer:
            # train_infos = (0 if self.buffer_test else self.train_with_rebuf())
            if self.buffer_test: 
                train_infos = self.freeze_train2()
            else:
                #here i should extract the buffer and pass it
                #TODO cambiare eventualmente la sqaudra di partenza 
                self.rebuf_in, self.rebuf_out = buffer_utils.import_buffer(1)

                #PORZIONO IL BUFFER 
                # 1. seleziono solo la colonna relativa all'active agent
                self.rebuf_in = self.rebuf_in[self.active_agent] 
                self.rebuf_out = self.rebuf_out[self.active_agent]

                # 2. Setto la percentuale di uso del buffer (tramite config?)
                buff_episodes_len = len(self.rebuf_in)
                
                index_array = np.arange(buff_episodes_len * self.episode_length)
                np.random.shuffle(index_array)
                print(f"Indeces array {index_array}")
                print(f"Index len {len(index_array)}")

                # 3. Shuffle + Portion
                new_buf_in = [] 
                for i in index_array:
                    ep_no = i // self.episode_length
                    step_no = i % self.episode_length
                    print(f"i is {i}")
                    print(f"ep_no is {ep_no}")
                    print(f"step_no is {step_no}")
                    print(f" sample is {self.rebuf_in[ep_no][:][step_no]}")
                    sys.exit()
                # 4. Voilà nuovo buffer 

                #train algorithm 
                train_infos = self.train_with_rebuf()

        return train_infos
    

    def import_buffer(self, buffer_team_no):

        if self.env_name == "MPE":
            buffer_path = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                        0]+ "/scripts/results") / "TRAINING" / self.env_name / self.scenario_name / self.algorithm_name / self.experiment_name / "trained_teams" \
                        / str(buffer_team_no) / ("BufferT" + str(buffer_team_no)) 
            
        if self.env_name == "StarCraft2":
            buffer_path = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                            0] + "/scripts/results") / "TRAINING" / self.env_name / self.map_name / self.algorithm_name / self.experiment_name / "trained_teams" \
                            / str(buffer_team_no) / ("BufferT" + str(buffer_team_no)) 
        
        buffer_in = torch.load(str(buffer_path) + "/replay_buffer_ins.npy") 
        buffer_out = torch.load(str(buffer_path) +  "/replay_buffer_outs.npy")

        return buffer_in, buffer_out