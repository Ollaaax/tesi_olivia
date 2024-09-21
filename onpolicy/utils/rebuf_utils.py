import numpy as np
import os
import sys
from pathlib import Path
import random
import torch
import shutil

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
        self.rebuf_outs = [[], []]
        self.rebuf_ins = [[], []]

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

        for agent_id in torch.randperm(self.num_agents):

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
            train_info = self.trainer[agent_id].train(self.buffer[agent_id], agent_id, update)


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

