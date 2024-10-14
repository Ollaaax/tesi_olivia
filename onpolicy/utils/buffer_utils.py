import numpy as np
import os
import sys
from pathlib import Path
import random
import torch


from onpolicy.config import get_config


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
        
  
    parser.add_argument('--map_name', type=str, default='3m',
                        help="Which smac map to run on")

    all_args = parser.parse_known_args(args)[0]

    return all_args

def import_buffer(buffer_team_no):
    parser = get_config()
    args = sys.argv[1:]
    all_args = parse_args(args, parser)

    if all_args.env_name == "MPE":
        buffer_path = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                    0]+ "/scripts/results") / "TRAINING" / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name / "trained_teams" \
                    / str(buffer_team_no) / ("BufferT" + str(buffer_team_no)) 
        
    if all_args.env_name == "StarCraft2":
        buffer_path = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                        0] + "/scripts/results") / "TRAINING" / all_args.env_name / all_args.map_name / all_args.algorithm_name / all_args.experiment_name / "trained_teams" \
                        / str(buffer_team_no) / ("BufferT" + str(buffer_team_no)) 
    
    buffer_in = torch.load(str(buffer_path) + "/replay_buffer_ins.npy") 
    buffer_out = torch.load(str(buffer_path) +  "/replay_buffer_outs.npy")

    return buffer_in, buffer_out

def pick_sample(self):
    parser = get_config()
    args = sys.argv[1:]
    all_args = parse_args(args, parser)

    if all_args.env_name == "MPE":
        episode_no = random.randint(0, 9)
        s = random.randint(0, 24)

        old_sample_in = self.rebuf_in[self.agent][episode_no]
        old_sample_out = self.rebuf_out[self.agent][episode_no]

        obs_vec = self.rebuf_in[self.agent][episode_no][0][s:s+128]
        old_sample_in = [obs_vec, self.rebuf_in[self.agent][episode_no][1], self.rebuf_in[self.agent][episode_no][2][s:s+128], self.rebuf_in[self.agent][episode_no][3][s:s+128], self.rebuf_in[self.agent][episode_no][4], self.rebuf_in[self.agent][episode_no][5][s:s+128]]

        old_sample_out = self.rebuf_out[self.agent][episode_no][s:s+128]

    if all_args.env_name == "StarCraft2":
        episode_no = 0
        s = random.randint(0, 399)

        old_sample_in = self.rebuf_in[self.agent][episode_no]
        old_sample_out = self.rebuf_out[self.agent][episode_no]

        obs_vec = self.rebuf_in[self.agent][episode_no][0][s:s+8]
        old_sample_in = [obs_vec, self.rebuf_in[self.agent][episode_no][1], self.rebuf_in[self.agent][episode_no][2][s:s+8], self.rebuf_in[self.agent][episode_no][3][s:s+8], self.rebuf_in[self.agent][episode_no][4][s:s+8], self.rebuf_in[self.agent][episode_no][5][s:s+8]]
        # old_sample_in = [obs_vec, self.rebuf_in[self.agent][episode_no][1], self.rebuf_in[self.agent][episode_no][2], self.rebuf_in[self.agent][episode_no][3], self.rebuf_in[self.agent][episode_no][4], self.rebuf_in[self.agent][episode_no][5]]

        old_sample_out = self.rebuf_out[self.agent][episode_no][s:s+8]  

    return old_sample_in, old_sample_out

def pick_sample2(self):
    s = random.randint(0, len(self.rebuf_in))

    old_sample_in = self.rebuf_in[s]
    old_sample_out = self.rebuf_out[s]
    
    return old_sample_in, old_sample_out


