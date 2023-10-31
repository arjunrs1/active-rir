#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import logging
from collections import deque, Counter
from typing import Dict, List
import json
import random
import math
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from numpy.linalg import norm
from gym import spaces
from collections import defaultdict

from habitat import logger, Config
from habitat.utils.visualizations.utils import observations_to_image
from rir_rendering.common.base_trainer import BaseRLTrainer
from rir_rendering.common.baseline_registry import baseline_registry
from ss_baselines.common.env_utils import construct_envs
from rir_rendering.common.environments import get_env_class
from ss_baselines.common.rollout_storage import RolloutStorage
from ss_baselines.common.tensorboard_utils import TensorboardWriter
from ss_baselines.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
    plot_top_down_map,
    resize_observation
)
from rir_rendering.active_context_sampler.ppo.policy import ActiveRIRPolicy
from rir_rendering.active_context_sampler.ppo.ppo import PPO
from rir_rendering.active_context_sampler.ppo.ppo_trainer import load_rir_predictor
from rir_rendering.uniform_context_sampler.policy import UniformContextSamplerPolicy
from rir_rendering.common.eval_metrics import compute_spect_metrics
from habitat_audio.utils import load_points_data
from rir_rendering.datasets.dataset import UniformContextSamplerDataset

class DataParallelPassthrough(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

@baseline_registry.register_trainer(name="active_rir_ppo")
class ActiveRIRTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Exploration"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None

        with open(self.config.TASK_CONFIG.ENVIRONMENT.ARBITRARY_RIR_SEEN_ENV_EVAL_SCENE_NAMES_PATH, "rb") as f:
            seen_scenes = pickle.load(f)

        with open(self.config.TASK_CONFIG.ENVIRONMENT.ARBITRARY_RIR_UNSEEN_ENV_EVAL_SCENE_NAMES_PATH, "rb") as f:
            unseen_scenes = pickle.load(f)

        scene_names = seen_scenes + unseen_scenes
        self.scene_count = dict(Counter(scene_names))

        self._static_smt_encoder = False
        self._encoder = None
        self.num_updates = 0

        if self.config.RL.USE_EARLY_ANNEALING:
            self.early_sampling_penalty = 4.0
            self.penalty_annealing_steps = int(self.config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS * self.config.RL.EARLY_ANNEALING_FRAC)

        if self.config.RL.USE_COOLDOWN_ANNEALING:
            self.cooldown_penalty = 4
            self.cooldown_timer = 0
    
    def get_novelty_reward(self, env_index, curr_obs):
        if curr_obs is not None:
            pose = tuple(curr_obs['pose'])[:2]
            pose = (pose[0]//self.config.RL.NOVELTY_GRID_FACTOR, pose[1]//self.config.RL.NOVELTY_GRID_FACTOR)
            if pose in self.novelty_count[env_index].keys():
                novelty_reward = 1/math.sqrt(self.novelty_count[env_index][pose])
                self.novelty_count[env_index][pose] += 1.0
                return novelty_reward
            else:
                self.novelty_count[env_index][pose] = 2.0
                return 1.0
        else:
            return 0

    def get_reward(self, prev_observations, env_index, curr_obs=None, use_sparse_reward=False, episode_step=None, action=None):
        reward = 0

        if len(prev_observations) == 20:
            return reward #reward for the actions taken after you already captured 20 contextual obs is irrelevant

        if self.config.RL.WITH_NOVELTY_REWARD:
            reward += self.get_novelty_reward(env_index, curr_obs)
            #TO DO: add if statement for self.config.RL.USE_RIR_REWARD, if true then continue, else return current reward
        
        #TO DO: add option for rir error (i.e. if self.config.RL.WITH_RIR_ERROR_REWARD:)
        if self.config.RL.WITH_RIR_REWARD:
            if len(prev_observations) != 1:
                curr_rir_error = self._curr_rir_error[env_index]
            else:
                curr_rir_error = np.abs(np.array(
                    self._get_rir_error(prev_observations, len(prev_observations), env_index)[0]['stft_l1_distance']
                )).mean()

            if curr_obs is not None:
                next_rir_error = np.abs(np.array(
                    self._get_rir_error(prev_observations, len(prev_observations)+1, env_index, curr_obs=curr_obs)[0]['stft_l1_distance']
                )).mean()
            else:
                next_rir_error = 0
            
            reward += (
                next_rir_error - curr_rir_error
            ) * self.config.RL.MEASUREMENT_RIR_REWARD_SCALE

            if use_sparse_reward:
                reward += -self.config.RL.SPARSE_RIR_REWARD_SCALE * next_rir_error

            self._curr_rir_error[env_index] = next_rir_error

        if self.config.RL.USE_EARLY_ANNEALING and episode_step < self.penalty_annealing_steps:
            initial_penalty = self.early_sampling_penalty #removed temporal annealing: * (1-self.num_updates/self.config.NUM_UPDATES)
            current_penalty = initial_penalty #removed spatial annealing: - initial_penalty * (episode_step / self.penalty_annealing_steps)
            if action == 3 or action == 4 or action == 5:
                reward -= current_penalty
            current_penalty = None
            
        if self.config.RL.USE_COOLDOWN_ANNEALING:
            #removed temporal annealing: current_cooldown_window = max(10,int(self.config.RL.SAMPLING_COOLDOWN_PERIOD*(1-(self.num_updates/self.config.NUM_UPDATES))))
            current_cooldown_window = self.config.RL.SAMPLING_COOLDOWN_PERIOD
            if action == 3 or action == 4 or action == 5:
                if self.cooldown_timer > 0:
                    reward -= self.cooldown_penalty

                #reset the cooldown timer  
                self.cooldown_timer = current_cooldown_window
            else:
                self.cooldown_timer = max(0, self.cooldown_timer - 1)

        return reward
    
    def _get_rir_error(self, prev_observations, current_episode_step, env_index, curr_obs=None, return_all_metrics=False):
        
        context = self._format_observation_rollout(prev_observations, current_episode_step, env_index, curr_obs=curr_obs)
        error_metrics = self._current_measurement_error(context, env_index, return_all_metrics=return_all_metrics)
        return error_metrics

    def _format_observation_rollout(self, prev_observations, num_masked_pos, env_index, curr_obs=None):

        #for compatibilty with obs list
        if isinstance(prev_observations, list):
            full_dict = {
                'rgb': [],
                'depth': [],
                'bin_spect_mag': [],
                'pose': []
            }
            for obs_dict in prev_observations:
                full_dict['rgb'].append(obs_dict['rgb'])
                full_dict['depth'].append(self.normalize_depth(obs_dict['depth']))
                full_dict['bin_spect_mag'].append(obs_dict['bin_spect_mag'])
                full_dict['pose'].append(obs_dict['pose'])
            for key in full_dict.keys():
                full_dict[key] = torch.stack(full_dict[key], dim=0)

            pad_amount = int(self.config.TASK_CONFIG.ENVIRONMENT.MAX_CONTEXT_LENGTH-len(prev_observations))
            view_echo_padding = (0,0,0,0,0,0,0,pad_amount)
            pose_padding = (0,0,0,pad_amount)
            full_dict['rgb'] = F.pad(full_dict['rgb'], view_echo_padding)
            full_dict['depth'] = F.pad(full_dict['depth'], view_echo_padding)
            full_dict['bin_spect_mag'] = F.pad(full_dict['bin_spect_mag'], view_echo_padding)
            full_dict['pose'] = F.pad(full_dict['pose'], pose_padding)

            prev_observations = full_dict


        if not isinstance(num_masked_pos, int):
            num_masked_pos = int(num_masked_pos.item()) 

        context_views = torch.unsqueeze(torch.cat((prev_observations['rgb'], prev_observations['depth']), dim=3),0)
        context_echoes = torch.unsqueeze(prev_observations['bin_spect_mag'],0)
        context_poses = torch.unsqueeze(prev_observations['pose'],0)
        context_mask = torch.zeros(1, self.config.TASK_CONFIG.ENVIRONMENT.MAX_CONTEXT_LENGTH)
        context_mask[:, :int(num_masked_pos)] = 1
        query_poses = torch.unsqueeze(torch.tensor(self.query_positions[env_index]),0)
        query_mask = torch.ones(query_poses.shape[0], query_poses.shape[1])
        
        context = {}
        context['context_views'] = context_views
        context['context_echoes'] = context_echoes
        context['context_poses'] = context_poses
        context['context_mask'] = context_mask
        context['query_poses'] = query_poses
        context['query_mask'] = query_mask

        if curr_obs is not None:
            curr_rgb = torch.tensor(curr_obs['rgb']).unsqueeze(0)
            curr_depth = torch.tensor(curr_obs['depth']).unsqueeze(0)
            curr_echo = torch.tensor(curr_obs['bin_spect_mag']).unsqueeze(0)
            curr_pose = torch.tensor(curr_obs['pose']).unsqueeze(0)
            context['context_views'][:,num_masked_pos-1,::] = torch.cat((curr_rgb, curr_depth), dim=3)
            context['context_echoes'][:,num_masked_pos-1,::] = curr_echo
            context['context_poses'][:,num_masked_pos-1,::] = curr_pose

        for key in context:
            if key in ['context_views', 'context_echoes', 'context_poses']:
                context[key] = context[key][:,:self.config.TASK_CONFIG.ENVIRONMENT.MAX_CONTEXT_LENGTH,::]

        return context

    def _current_measurement_error(self, context_observations, env_index, return_all_metrics=False):
        with torch.no_grad():
            preds = self.rir_predictor(context_observations)

        if self.config.UniformContextSampler.predict_in_logspace:
            if self.config.UniformContextSampler.log_instead_of_log1p_in_logspace:
                pred_spect_mag = torch.exp(preds.view(-1, *preds.size()[2:]))\
                                    - self.config.UniformContextSampler.log_gt_eps
            else:
                pred_spect_mag = torch.exp(preds.view(-1, *preds.size()[2:])) - 1
        else:
            pred_spect_mag = preds.view(-1, *preds.size()[2:])
        
        gt_rirs_mag = self.gt_rirs_mag[env_index]
        gt_rirs_phase = self.gt_rirs_phase[env_index]
        pred_spect_mag = pred_spect_mag.detach().cpu()

        eval_metrics_batch = compute_spect_metrics(
                            metric_types=['stft_l1_distance'] if not return_all_metrics else self.config.UniformContextSampler.EvalMetrics.types,
                            gt_spect_mag=gt_rirs_mag,
                            gt_spect_phase=gt_rirs_phase,
                            pred_spect_mag=pred_spect_mag,
                            eval_mode=True,
                            fs=self.config.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE,
                            hop_length=self.config.TASK_CONFIG.SIMULATOR.AUDIO.HOP_LENGTH,
                            n_fft=self.config.TASK_CONFIG.SIMULATOR.AUDIO.N_FFT,
                            win_length=self.config.TASK_CONFIG.SIMULATOR.AUDIO.WIN_LENGTH,
                        )

        return eval_metrics_batch, gt_rirs_mag, pred_spect_mag

    def _setup_actor_critic_agent(self, cfg: Config, observation_space=None) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        ppo_cfg = cfg.RL.PPO

        if observation_space is None:
            observation_space = self.envs.observation_spaces[0]
        self.actor_critic = ActiveRIRPolicy(
            cfg,
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size
        )
        self.actor_critic.to(self.device)

        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
        )

    def save_checkpoint(self, file_name: str) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    def _collect_rollout_step(
        self, rollouts, current_episode_reward, current_episode_step, episode_rewards,
            episode_counts, episode_steps, episode_rir_errors
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
                prev_obs_hidden_states,
            ) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
                rollouts.prev_obs_hidden_states[rollouts.step],
            )

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()

        outputs = self.envs.step([a[0].item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        episode_rir_error = [0] * len(dones)
        for i, done in enumerate(dones):
            if done:
                mask_size = min(20, len(self.context_observations[i]))
                episode_rir_error[i] = np.array(self._get_rir_error(self.context_observations[i], mask_size, i)[0]['stft_l1_distance']).mean()
                observations[i] = self.initialize_queries_gt_rirs_and_observations(observations[i], i)
                rewards[i] = 0.0
            else:
                observations[i]['depth'] = observations[i]['depth'].squeeze(-1) if len(observations[i]['depth'].shape) == 4 else observations[i]['depth']
                if self.config.UNIFORM_SAMPLE:
                    if current_episode_step[i] != 0 and current_episode_step[i] % (self.config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS // 20) == 0 and len(self.context_observations[i]) < 20:
                        context_obs = {k: v[i].cpu() for k, v in step_observation.items()}
                        self.context_observations[i].append(context_obs)
                else:
                    if (actions[i][0].item() == 3) or (actions[i][0].item() == 4) or (actions[i][0].item() == 5):
                        context_obs = {k: v[i].cpu() for k, v in step_observation.items()}
                        self.context_observations[i].append(context_obs)
                rewards[i] = self.get_reward(self.context_observations[i], i, curr_obs=observations[i],
                                             use_sparse_reward=(len(self.context_observations[i])==self.config.TASK_CONFIG.ENVIRONMENT.MAX_CONTEXT_LENGTH-1),
                                             episode_step=current_episode_step[i].item(), action=actions[i][0].item())

        logging.debug('Reward: {}'.format(rewards[0]))

        env_time += time.time() - t_step_env

        
        t_update_stats = time.time()
        rewards = torch.tensor(rewards, dtype=torch.float, device=current_episode_reward.device)
        rewards = rewards.unsqueeze(1)
        episode_rir_error = torch.tensor(episode_rir_error, dtype=torch.float, device=current_episode_reward.device)
        episode_rir_error = episode_rir_error.unsqueeze(1)
        batch = batch_obs(observations, device=self.device)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones], dtype=torch.float, device=current_episode_reward.device
        )

        current_episode_reward += rewards
        current_episode_step += 1
        # current_episode_reward is accumulating rewards across multiple updates,
        # as long as the current episode is not finished
        # the current episode reward is added to the episode rewards only if the current episode is done
        # the episode count will also increase by 1
        episode_rewards += (1 - masks) * current_episode_reward
        episode_steps += (1 - masks) * current_episode_step
        episode_counts += 1 - masks
        episode_rir_errors += (1 - masks) * episode_rir_error
        current_episode_reward *= masks
        current_episode_step *= masks

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards.to(device=self.device),
            masks.to(device=self.device),
            prev_obs_hidden_states=None,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[-1] for k, v in rollouts.observations.items()
            }

            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[-1],
                rollouts.prev_actions[-1],
                rollouts.masks[-1],
                rollouts.prev_obs_hidden_states[-1],
            ).detach()

        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        value_loss, action_loss, dist_entropy = self.agent.update(rollouts)

        rollouts.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
        )

    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """
        logger.info(f"config: {self.config}")
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME), workers_ignore_signals=True
        )

        ppo_cfg = self.config.RL.PPO
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.rir_predictor = load_rir_predictor(self.config.RL.PRETRAINED_RIR_PREDICTOR_PATH, self.device)

        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        self._setup_actor_critic_agent(self.config)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
        )
        rollouts.to(self.device)

        self.query_positions = [
            [] for _ in range(self.envs.num_envs)
        ]

        self.novelty_count = [
            dict() for _ in range(self.envs.num_envs)
        ]

        self.context_observations = [
            [] for _ in range(self.envs.num_envs)
        ]

        self.gt_rirs_mag = torch.zeros(torch.Size([self.envs.num_envs, 60, 256, 259, 2]))
        self.gt_rirs_phase = torch.zeros(torch.Size([self.envs.num_envs, 60, 256, 259, 2]))
        self._curr_rir_error = torch.zeros(self.envs.num_envs, 1)

        #get initial observations
        observations_and_queries = self.envs.reset()
        observations = [self.initialize_queries_gt_rirs_and_observations(obs, i) for i, obs in enumerate(observations_and_queries)]
        batch = batch_obs(observations)
        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        batch = None
        observations = None

        # episode_rewards and episode_counts accumulates over the entire training course
        episode_rewards = torch.zeros(self.envs.num_envs, 1)
        episode_steps = torch.zeros(self.envs.num_envs, 1)
        episode_counts = torch.zeros(self.envs.num_envs, 1)
        episode_rir_errors = torch.zeros(self.envs.num_envs, 1)
        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        current_episode_step = torch.zeros(self.envs.num_envs, 1)
        window_episode_reward = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_step = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_counts = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_rir_error = deque(maxlen=ppo_cfg.reward_window_size)

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(self.config.NUM_UPDATES):
                self.num_updates += 1
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                #collect trajectories
                self.agent.eval()
                for step in tqdm(range(ppo_cfg.num_steps)):
                    delta_pth_time, delta_env_time, delta_steps = self._collect_rollout_step(
                        rollouts,
                        current_episode_reward,
                        current_episode_step,
                        episode_rewards,
                        episode_counts,
                        episode_steps,
                        episode_rir_errors
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps

                self.agent.train()
                delta_pth_time, value_loss, action_loss, dist_entropy = self._update_agent(
                    ppo_cfg, rollouts
                )
                pth_time += delta_pth_time

                window_episode_reward.append(episode_rewards.clone())
                window_episode_step.append(episode_steps.clone())
                window_episode_counts.append(episode_counts.clone())
                window_episode_rir_error.append(episode_rir_errors.clone())

                #losses = [value_loss, action_loss, dist_entropy]
                stats = zip(
                    ["count", "reward", "step", "rir_error"],
                    [window_episode_counts, window_episode_reward, window_episode_step, window_episode_rir_error],
                )
                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in stats if k != "rir_error"
                }
                episode_errors_window = list(window_episode_rir_error)
                deltas["rir_error"] = torch.mean(torch.abs(torch.stack(episode_errors_window)), dim=0).item()

                #deltas["rir_error"] = (window_episode_rir_error[0] - window_episode_rir_error[-1]).sum().item() if len(window_episode_rir_error) > 1 else window_episode_rir_error[0].sum().item()
                deltas["count"] = max(deltas["count"], 1.0)

                # this reward is averaged over all the episodes happened during window_size updates
                # approximately number of steps is window_size * num_steps
                if update % 10 == 0:
                    writer.add_scalar("Environment/Reward", deltas["reward"] / deltas["count"], count_steps)
                    writer.add_scalar("Environment/Episode_length", deltas["step"] / deltas["count"], count_steps)
                    writer.add_scalar("Environment/RIR_Error", deltas["rir_error"] / deltas["count"], count_steps)
                    writer.add_scalar('Policy/Value_Loss', value_loss, count_steps)
                    writer.add_scalar('Policy/Action_Loss', action_loss, count_steps)
                    writer.add_scalar('Policy/Entropy', dist_entropy, count_steps)
                    writer.add_scalar('Policy/Learning_Rate', lr_scheduler.get_lr()[0], count_steps)

                # log stats
                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            update, count_steps / (time.time() - t_start)
                        )
                    )

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            update, env_time, pth_time, count_steps
                        )
                    )

                    window_rewards = (
                        window_episode_reward[-1] - window_episode_reward[0]
                    ).sum()
                    window_rir_errors = (
                        window_episode_rir_error[0] - window_episode_rir_error[-1]
                    ).sum()
                    window_counts = (
                        window_episode_counts[-1] - window_episode_counts[0]
                    ).sum()

                    if window_counts > 0:
                        logger.info(
                            "Average window size {} reward: {:3f}".format(
                                len(window_episode_reward),
                                (window_rewards / window_counts).item(),
                            )
                        )
                        logger.info(
                            "Average window size {} rir error: {:3f}".format(
                                len(window_episode_rir_error),
                                (window_rir_errors / window_counts).item(),
                            )
                        )
                    else:
                        logger.info("No episodes finish in current window")

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(f"ckpt.{count_checkpoints}.pth")
                    count_checkpoints += 1

            self.envs.close()

    def normalize_depth(self, depth):
        """
        normalize depth
        :param depth: unnormalized depth
        :return: normalized depth
        """
        depth = (depth - self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH) / (
            self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH - self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        )

        return depth
    
    def initialize_queries_gt_rirs_and_observations(self, initial_observations, i, reset_context=True):
        observations, query_positions, gt_rirs_mags, gt_rirs_phases, ref_pose = initial_observations
        if len(observations['depth'].shape) == 4:
            observations['depth'] = observations['depth'].squeeze(-1)
        self.query_positions[i] = list(query_positions)
        self.gt_rirs_mag[i] = torch.tensor(gt_rirs_mags)
        self.gt_rirs_phase[i] = torch.tensor(gt_rirs_phases)

        if self.config.SAVE_INTERMEDIATE_FS_RIR_ERRORS:
            self.ref_pose = ref_pose

        if reset_context:
            self.reset_and_initialize_context(observations, i)

        if self.config.RL.WITH_NOVELTY_REWARD:
            self.novelty_count[i] = {}

        return observations
    
    def reset_and_initialize_context(self, observations, env_index):
        #empty the context cache and initialize it with first observation
        self.context_observations[env_index] = [] 
        initial_context = {}
        initial_context['rgb'] = torch.tensor(observations['rgb'], dtype=torch.float32)
        initial_context['depth'] = torch.tensor(observations['depth'])
        initial_context['bin_spect_mag'] = torch.tensor(observations['bin_spect_mag'])
        initial_context['pose'] = torch.tensor(np.array(observations['pose']), dtype=torch.float32)
        self.context_observations[env_index].append(initial_context)
    