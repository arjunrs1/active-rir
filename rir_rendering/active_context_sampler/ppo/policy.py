#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import abc
import os
import pickle
import math
import numpy as np

import torch
import torch.nn as nn
from torchsummary import summary

from ss_baselines.common.utils import CategoricalNet
from ss_baselines.av_nav.models.rnn_state_encoder import RNNStateEncoder
from ss_baselines.av_nav.models.visual_cnn import VisualCNN
from ss_baselines.av_nav.models.audio_cnn import AudioCNN

from rir_rendering.models.visual_cnn import VisualEnc
from rir_rendering.models.global_map_encoder import MapEncoder
from rir_rendering.models.audio_cnn import AudioEnc, AudioDec
from rir_rendering.models.positional_net import PositionalEnc, LowDimPositionalEnc
from rir_rendering.models.fusion_net import FusionNet
from rir_rendering.models.memory_net import TransformerMemory

DUAL_GOAL_DELIMITER = ','


class Policy(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        prev_obs_hidden_states,
        deterministic=False,
    ):
        features, rnn_hidden_states, obs_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks, prev_obs_hidden_states
        )
        # print('Features: ', features.cpu().numpy())
        distribution = self.action_distribution(features)
        # print('Distribution: ', distribution.logits.cpu().numpy())
        value = self.critic(features)
        # print('Value: ', value.item())

        if deterministic:
            action = distribution.mode()
            # print('Deterministic action: ', action.item())
        else:
            action = distribution.sample()
            # print('Sample action: ', action.item())

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states, obs_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks, prev_obs_hidden_states):
        features, rnn_hidden_states, obs_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks, prev_obs_hidden_states
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action, prev_obs_hidden_states
    ):
        features, rnn_hidden_states, obs_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks, prev_obs_hidden_states
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states, prev_obs_hidden_states


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass

class ActiveRIRPolicy(Policy):
    def __init__(
        self,
        cfg,
        action_space,
        hidden_size=512
    ):
        super().__init__(
            ContextEncoderNet(cfg=cfg,
                hidden_size=hidden_size,
            ),
            action_space.n,
        )


class ContextEncoderNet(Net):
    r"""Network which passes the input image/audio/pose through RNN to get context summary, and passes
    context summary vector through RNN.
    """

    def __init__(self, cfg, hidden_size):
        super().__init__()
        self._hidden_size = hidden_size
        rnn_input_size = self._hidden_size
        self._cfg = cfg
        self._sampler_cfg = cfg.UniformContextSampler
        

        #RNN context and state encoders
        self.state_encoder = RNNStateEncoder(rnn_input_size, self._hidden_size)

        #per modality observation encoders
        self.visual_context_enc = VisualEnc()
        self.audio_context_enc = AudioEnc(
            audio_cfg=cfg.TASK_CONFIG.SIMULATOR.AUDIO,
            log_instead_of_log1p_in_logspace=self._sampler_cfg.predict_in_logspace and\
                                             self._sampler_cfg.log_instead_of_log1p_in_logspace,
            log_eps=self._sampler_cfg.log_gt_eps,
        )

        self.pose_context_enc = PositionalEnc(
            positional_enc_cfg=self._sampler_cfg.PositionalEnc,
        )

        self.global_map_enc = MapEncoder()

        #multi-modal fusion network
        if self._sampler_cfg.encode_each_modality_as_independent_context_entry:
            n_input_feats_fusion_context_enc = sum([context_enc.n_out_feats for context_enc in [
                self.visual_context_enc,
                self.audio_context_enc,
                self.pose_context_enc,
                self.global_map_enc
            ]])
            self.fusion_context_enc = FusionNet(trainer_cfg=self._sampler_cfg, n_input_feats=n_input_feats_fusion_context_enc)
            self.fused_context_layer = nn.Sequential(
                    nn.Linear(1024, 512, bias=False),
                )
        else:
            raise ValueError

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_context_enc.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, prev_obs_hidden_states):

        assert 'bin_spect_mag' in observations
        bin_spect_mag = observations['bin_spect_mag']
        assert 'rgb' in observations
        rgb = observations['rgb']
        assert 'depth' in observations
        depth = self.normalize_depth(observations['depth'])
        assert 'pose' in observations
        pose = observations['pose']
        assert 'occupancy_map' in observations
        global_occ_map = observations['occupancy_map']
        assert 'timestep_sensor' in observations
        remaining_trajectory_length = observations['timestep_sensor']
        assert 'context_length_sensor' in observations
        remaining_context_capacity = observations['context_length_sensor']

        context_feats = []

        #Only add audio to policy when collected/sampled
        if self._cfg.UNIFORM_SAMPLE:
            sampling_condition = (remaining_trajectory_length < 1.0) & (remaining_trajectory_length % 0.05 == 0)
            mask = ~sampling_condition
            mask_expanded = mask.view(bin_spect_mag.shape[0], 1, 1, 1).expand_as(bin_spect_mag)
            bin_spect_mag[mask_expanded] = 0
        else:
            prev_actions_simplified = prev_actions.squeeze(-1)
            actions_of_interest = torch.tensor([3, 4, 5], device=prev_actions_simplified.device)
            mask = ~torch.any(prev_actions_simplified[:, None] == actions_of_interest, dim=1)
            bin_spect_mag[mask, ...] = 0

        #encode each modality
        if self._cfg.SENSORS in [["RGB_SENSOR", "DEPTH_SENSOR"], ["DEPTH_SENSOR", "RGB_SENSOR"]]:
            visual_context_feats = self.visual_context_enc({"rgb": rgb,
                                                            "depth": depth})
            context_feats.append(visual_context_feats)
        else:
            raise ValueError
        
        audio_context_feats = self.audio_context_enc({"audio_spect": bin_spect_mag})
        context_feats.append(audio_context_feats)

        pose_context_feats = self.pose_context_enc({"positional_obs": pose})
        context_feats.append(pose_context_feats)

        map_context_feats = self.global_map_enc({"occupancy_map": global_occ_map})
        context_feats.append(map_context_feats)

        #concatenate encoded modalities
        x1 = self.fusion_context_enc(torch.cat(context_feats, dim=1).unsqueeze(0))
        x1 = self.fused_context_layer(x1)

        if remaining_trajectory_length.dim() == 1:
            remaining_trajectory_length = remaining_trajectory_length.unsqueeze(0)
        if remaining_context_capacity.dim() == 1:
            remaining_context_capacity = remaining_context_capacity.unsqueeze(0)
        x1 = torch.cat((x1, remaining_trajectory_length, remaining_context_capacity), dim=1)

        prev_obs_hidden_states1 = None

        #pass encoded_observation state to state encoder
        x2, rnn_hidden_states1 = self.state_encoder(x1, rnn_hidden_states, masks)

        if torch.isnan(x2).any().item():
            for key in observations:
                print(key, torch.isnan(observations[key]).any().item())
            print('rnn_old', torch.isnan(rnn_hidden_states).any().item())
            print('rnn_new', torch.isnan(rnn_hidden_states1).any().item())
            print('mask', torch.isnan(masks).any().item())
            assert True

        return x2, rnn_hidden_states1, prev_obs_hidden_states1
    
    def normalize_depth(self, depth):
        """
        normalize depth
        :param depth: unnormalized depth
        :return: normalized depth
        """
        depth = (depth - self._cfg.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH) / (
            self._cfg.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH - self._cfg.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        )

        return depth