import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check
#######
from onpolicy.utils import buffer_utils

import sys
import random



class R_MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        #__________________________________________________________________________________________________________
        #TODO tirare fuori il buffer
        self.multi_agent = args.multi_agent
        self.alpha = args.alpha
        self.rebuf_in = None

        #Loss fncs 
        self.loss_data_trace = None
        self.ppo_loss = 0
        self.replay_loss_l1 = 0
        self.replay_loss_l2 = 0
        self.policy_loss = 0
        #__________________________________________________________________________________________________________

        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1).to(self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        if len(sample) == 12:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = sample
        else:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = sample[:-1]

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)

        # print(obs_batch)
        # print(action_log_probs)
        #____________________________________________________________________________________________________________
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        # print(f"dim imp pesi: {imp_weights.shape}")

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        #____________________________________________________________________________________________________________
        ##       NEW LOSS FNC
        
        policy_loss = policy_action_loss
        # replay_loss_l1 = 0
        # replay_loss_l2 = 0
        #____________________________________________________________________________________________________________
        #         USE THE REPLAY BUFFER IN PIECES\
        # print("__________________________________________")
        if update_actor:

            old_sample_in, old_sample_out = buffer_utils.pick_sample2(self)

            # print([*old_sample_in[:][0]])
            a = old_sample_in[0]
            b = old_sample_in[1]
            c = old_sample_in[2]
            d = old_sample_in[3]
            
            # print(old_sample_in[0][0][0])
            # print(old_sample_out[:][0])
            # #Run the net with the samples taken from the rebuf
            predicted_out = self.policy.actor.get_logit_forward(a, b, c, d)
            # print(predicted_out)
            predicted_out = predicted_out.logits

            #l1 Replay Loss 
            replay_diff = (old_sample_out - predicted_out)
            # print(f" differenze {replay_diff}")

            #l1
            replay_loss_l1 = torch.sum(torch.abs(replay_diff), dim=-1, keepdim=True).mean()
            # print(replay_loss_l1)
            # print(a[:10])
            # print(f"old out {old_sample_out[:][:10]}")
            # print(f"pred {predicted_out[:][:10]}")

            #l2 Replay Loss #TODO!!
            replay_loss_l2 = torch.sum(replay_diff ** 2, dim=-1, keepdim=True).mean()

            policy_loss = policy_action_loss + self.alpha*replay_loss_l2

            self.ppo_loss = policy_action_loss
            self.replay_loss_l1 = replay_loss_l1
            self.replay_loss_l2 = replay_loss_l2
            self.policy_loss = policy_loss
            # print(f"Policy Loss is {policy_loss}")
        #____________________________________________________________________________________________________________

        self.policy.actor_optimizer.zero_grad()
        # print(policy_loss - dist_entropy * self.entropy_coef)
        if update_actor:
            # (policy_loss - dist_entropy * self.entropy_coef).backward()
            (policy_loss).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())


        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        if update_actor:
            (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        #_________________

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, 

    def train(self, buffer, update_actor):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        
        
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        #_______________________________________________________-
        ## Dict to save loss fncs
        if self.rebuf_in is not None:
            loss_data = {}  

            loss_data['ppo_loss'] = 0
            loss_data['l1_rebuf_loss'] = 0
            loss_data['l2_rebuf_loss'] = 0
            loss_data['overall_loss'] = 0


        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)
            
            for sample in data_generator:
                
                # if self.rebuf_in is None:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                        = self.ppo_update(sample, update_actor)
                # else: 
                #     value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, replay_loss_l1, replay_loss_l2, ppo_loss \
                #         = self.ppo_update(sample, update_actor)                    

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

                if self.rebuf_in is not None:
                    loss_data['ppo_loss'] += self.ppo_loss.item()
                    loss_data['l1_rebuf_loss'] += self.replay_loss_l1.item()
                    loss_data['l2_rebuf_loss'] += self.replay_loss_l2.item()
                    loss_data['overall_loss'] += self.policy_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

       
        if self.rebuf_in is not None:
            for j in loss_data.keys():
                loss_data[j] /= num_updates

            self.loss_data_trace = loss_data
        
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()

#######################################################################
#######################################################################

    def set_buffers(self, rebuf_in, rebuf_out):
        self.rebuf_in = rebuf_in
        self.rebuf_out = rebuf_out

    # def pick_sample(self):
    #     episode_no = random.randint(0, 9)
    #     print("The episode is " + str(episode_no))
    #     s = random.randint(0, 24)

    #     old_sample_in = self.rebuf_in[self.agent][episode_no]
    #     old_sample_out = self.rebuf_out[self.agent][episode_no]

    #     obs_vec = self.rebuf_in[self.agent][episode_no][0][s:s+128]
    #     old_sample_in = [obs_vec, self.rebuf_in[self.agent][episode_no][1], self.rebuf_in[self.agent][episode_no][2][s:s+128], self.rebuf_in[self.agent][episode_no][3][s:s+128], self.rebuf_in[self.agent][episode_no][4], self.rebuf_in[self.agent][episode_no][5][s:s+128]]

    #     old_sample_out = self.rebuf_out[self.agent][episode_no][s:s+128]

    #     return old_sample_in, old_sample_out