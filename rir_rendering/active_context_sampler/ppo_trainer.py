#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import logging
from collections import deque
from typing import Dict, List
import json
import random
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from numpy.linalg import norm
from gym import spaces

from habitat import logger, Config
from habitat.utils.visualizations.utils import observations_to_image
from rir_rendering.common.base_trainer import BaseRLTrainer
from rir_rendering.common.baseline_registry import baseline_registry
from rir_rendering.common.env_utils import construct_envs
from rir_rendering.common.environments import get_env_class
from ss_baselines.common.rollout_storage import RolloutStorage
from rir_rendering.common.tensorboard_utils import TensorboardWriter
from ss_baselines.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
    plot_top_down_map,
    resize_observation
)
from rir_rendering.active_context_sampler.policy import ActiveRIRPolicy
from rir_rendering.active_context_sampler.ppo import PPO
from rir_rendering.uniform_context_sampler.policy import UniformContextSamplerPolicy
from rir_rendering.common.eval_metrics import compute_spect_metrics

@baseline_registry.register_trainer(name="ActiveRIRTrainer")
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
        self.rir_predictor = load_rir_predictor(self.config.RL.PRETRAINED_RIR_PREDICTOR_PATH, self.config)
        self.previous_measurement_error = None

    def get_reward_placeholder(self, prev_observations, current_episode_step, env_index, curr_obs=None, use_sparse_reward=False):
        return 0

    def get_reward(self, prev_observations, current_episode_step, env_index, curr_obs=None, use_sparse_reward=False):
        reward = 0

        #compute RIR measurement delta reward:

        if int(current_episode_step) != 0:
            curr_rir_error = self._curr_rir_error[env_index]
        else:
            curr_rir_error = np.abs(np.array(
                self._get_rir_error(prev_observations, current_episode_step+1, env_index)[0]['stft_l1_distance']
            )).mean()

        if curr_obs is not None:
            next_rir_error = np.abs(np.array(
                self._get_rir_error(prev_observations, current_episode_step+2, env_index, curr_obs=curr_obs)[0]['stft_l1_distance']
            )).mean()
        else:
            next_rir_error = 0
        
        reward += (
            next_rir_error - curr_rir_error
        ) * self.config.RL.MEASUREMENT_RIR_REWARD_SCALE

        if use_sparse_reward:
            assert current_episode_step + 2 == self.config.TASK_CONFIG.ENVIRONMENT.MAX_CONTEXT_LENGTH
            reward += -self.config.RL.SPARSE_RIR_REWARD_SCALE * next_rir_error

        self._curr_rir_error[env_index] = next_rir_error

        return reward
    
    def _get_rir_error(self, prev_observations, current_episode_step, env_index, curr_obs=None):
        
        context = self._format_observation_rollout(prev_observations, current_episode_step, env_index, curr_obs=curr_obs)
        return self._current_measurement_error(context, env_index)


    def _format_observation_rollout(self, prev_observations, num_masked_pos, env_index, curr_obs=None):

        #for compatibilty with obs list passed during evaluation
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

            print("BEFORE")
            for key in full_dict:
                print(key)
                print(full_dict[key].shape)

            pad_amount = int(self.config.TASK_CONFIG.ENVIRONMENT.MAX_CONTEXT_LENGTH-len(prev_observations))
            view_echo_padding = (0,0,0,0,0,0,0,0,0,pad_amount)
            pose_padding = (0,0,0,0,0,pad_amount)
            full_dict['rgb'] = F.pad(full_dict['rgb'], view_echo_padding)#.permute(1,0,2,3,4)
            full_dict['depth'] = F.pad(full_dict['depth'], view_echo_padding)#.permute(1,0,2,3,4)
            full_dict['bin_spect_mag'] = F.pad(full_dict['bin_spect_mag'], view_echo_padding)#.permute(1,0,2,3,4)
            full_dict['pose'] = F.pad(full_dict['pose'], pose_padding)#.permute(1,0,2)

            prev_observations = full_dict
            print("AFTER")
            for key in prev_observations:
                print(key)
                print(prev_observations[key].shape)


        if not isinstance(num_masked_pos, int):
            num_masked_pos = int(num_masked_pos.item())

        context_views = torch.unsqueeze(torch.cat((prev_observations['rgb'], prev_observations['depth']), dim=4).permute(1, 0, 2, 3, 4)[env_index],0)
        context_echoes = torch.unsqueeze(prev_observations['bin_spect_mag'].permute(1,0,2,3,4)[env_index],0)
        context_poses = torch.unsqueeze(prev_observations['pose'].permute(1,0,2)[env_index],0)
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
            print("KUDIYAAN")
            curr_rgb = torch.tensor(curr_obs['rgb']).unsqueeze(0)
            curr_depth = torch.tensor(curr_obs['depth']).unsqueeze(0)
            curr_echo = torch.tensor(curr_obs['bin_spect_mag']).unsqueeze(0)
            curr_pose = torch.tensor(curr_obs['pose']).unsqueeze(0)
            context['context_views'][:,num_masked_pos-1,::] = torch.cat((curr_rgb, curr_depth), dim=3)
            context['context_echoes'][:,num_masked_pos-1,::] = curr_echo
            context['context_poses'][:,num_masked_pos-1,::] = curr_pose

        for key in context:
            #print(key)
            if key in ['context_views', 'context_echoes', 'context_poses']:
                context[key] = context[key][:,:self.config.TASK_CONFIG.ENVIRONMENT.MAX_CONTEXT_LENGTH,::]
            #print(context[key].shape)

        return context

    def _current_measurement_error(self, context_observations, env_index):  
        pred_rirs = self.rir_predictor(context_observations)
        pred_spect_mag = torch.exp(pred_rirs.view(-1, *pred_rirs.size()[2:]))\
                                                     - self.config.UniformContextSampler.log_gt_eps
        
        gt_rirs = self.gt_rirs[env_index]
        gt_rirs_mag = torch.tensor([gt[0] for gt in gt_rirs]).detach().cpu()
        gt_rirs_phase = torch.tensor([gt[1] for gt in gt_rirs]).detach().cpu()
        pred_spect_mag = pred_spect_mag.detach().cpu()

        eval_metrics_batch = compute_spect_metrics(
                            metric_types=self.config.UniformContextSampler.EvalMetrics.types,
                            gt_spect_mag=gt_rirs_mag,
                            gt_spect_phase=gt_rirs_phase,
                            pred_spect_mag=pred_spect_mag,
                            mask=None,
                            eval_mode=True,
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
                observations[i] = self.initialize_queries_gt_rirs_and_observations(observations[i], i)
                rewards[i] = 0.0
                episode_rir_error[i] = np.array(self._get_rir_error(rollouts.observations, current_episode_step[i]+1, i)[0]['stft_l1_distance']).mean()
            else:
                observations[i]['depth'] = observations[i]['depth'].squeeze(-1)
                rewards[i] = self.get_reward(rollouts.observations, current_episode_step[i], i, curr_obs=observations[i], use_sparse_reward=(rollouts.step==18))

        logging.debug('Reward: {}'.format(rewards[0]))

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        rewards = torch.tensor(rewards, dtype=torch.float)
        rewards = rewards.unsqueeze(1)
        episode_rir_error = torch.tensor(episode_rir_error, dtype=torch.float)
        episode_rir_error = episode_rir_error.unsqueeze(1)
        batch = batch_obs(observations)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones], dtype=torch.float
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
            rewards,
            masks,
            prev_obs_hidden_states,
        )

        pth_time += time.time() - t_update_stats

        #clear CUDA memory
        batch = None
        observations = None

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
            self.config, get_env_class(self.config.ENV_NAME)
        )

    
        ppo_cfg = self.config.RL.PPO
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
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
        )
        rollouts.to(self.device)

        self.query_positions = [
            [] for _ in range(self.envs.num_envs)
        ]

        self.gt_rirs = [
            [] for _ in range(self.envs.num_envs)
        ]

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
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                #collect trajectories
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

                delta_pth_time, value_loss, action_loss, dist_entropy = self._update_agent(
                    ppo_cfg, rollouts
                )
                pth_time += delta_pth_time

                window_episode_reward.append(episode_rewards.clone())
                window_episode_step.append(episode_steps.clone())
                window_episode_counts.append(episode_counts.clone())
                window_episode_rir_error.append(episode_rir_errors.clone())

                losses = [value_loss, action_loss, dist_entropy]
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
                    for k, v in stats
                }
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
                        window_episode_rir_error[-1] - window_episode_rir_error[0]
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

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0
    ) -> Dict:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)
            
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        if self.config.DISPLAY_RESOLUTION != config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH:
            model_resolution = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
            config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = \
                config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = \
                self.config.DISPLAY_RESOLUTION
        else:
            model_resolution = self.config.DISPLAY_RESOLUTION
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()
        elif "top_down_map" in self.config.VISUALIZATION_OPTION:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.freeze()
        if "pred_rir_spec" in self.config.VISUALIZATION_OPTION:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("PRED_RIR_SPEC")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(
            config, get_env_class(config.ENV_NAME)
        )
        if self.config.DISPLAY_RESOLUTION != model_resolution:
            observation_space = self.envs.observation_spaces[0]
            observation_space.spaces['depth'] = spaces.Box(low=0, high=1, shape=(model_resolution,
                                                           model_resolution, 1), dtype=np.uint8)
            observation_space.spaces['rgb'] = spaces.Box(low=0, high=1, shape=(model_resolution,
                                                         model_resolution, 3), dtype=np.uint8)
        else:
            observation_space = self.envs.observation_spaces[0]
        self._setup_actor_critic_agent(config, observation_space)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        self.metric_uuids = []
        # get name of performance metric, e.g. "spl"
        for metric_name in self.config.TASK_CONFIG.TASK.MEASUREMENTS:
            metric_cfg = getattr(self.config.TASK_CONFIG.TASK, metric_name)
            measure_type = baseline_registry.get_measure(metric_cfg.TYPE)
            assert measure_type is not None, "invalid measurement type {}".format(
                metric_cfg.TYPE
            )
            self.metric_uuids.append(measure_type(sim=None, task=None, config=None)._get_uuid())

        self._curr_rir_error = torch.zeros(self.envs.num_envs, 1)

        self.query_positions = [
            [] for _ in range(self.envs.num_envs)
        ]

        self.gt_rirs = [
            [] for _ in range(self.envs.num_envs)
        ]

        full_obs = []
        observations_and_queries = self.envs.reset()
        observations = [self.initialize_queries_gt_rirs_and_observations(obs, i) for i, obs in enumerate(observations_and_queries)]
        if self.config.DISPLAY_RESOLUTION != model_resolution:
            resize_observation(observations, model_resolution)

        batch = batch_obs(observations, self.device)
        full_obs.append(batch)


        current_episode_step = torch.zeros(
            self.envs.num_envs, 1
        )
        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )

        prev_obs_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )

        stats_episodes = dict()  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        audios = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        t = tqdm(total=self.config.TEST_EPISODE_COUNT)
        while (
            len(stats_episodes) < self.config.TEST_EPISODE_COUNT
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                _, actions, _, test_recurrent_hidden_states, prev_obs_hidden_states = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    prev_obs_hidden_states,
                    deterministic=False
                )

                prev_actions.copy_(actions)

            outputs = self.envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

            episode_rir_error = [0] * len(dones)
            for i, done in enumerate(dones):
                if done:
                    observations[i] = self.initialize_queries_gt_rirs_and_observations(observations[i], i)
                    rewards[i] = 0.0
                    episode_rir_error[i] = np.array(self._get_rir_error(full_obs, self.config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS, i)[0]['stft_l1_distance']).mean()
                else:
                    observations[i]['depth'] = observations[i]['depth'].squeeze(-1)
                    print("curr ep step:")
                    print(current_episode_step[i])
                    rewards[i] = self.get_reward(full_obs, current_episode_step[i], i, curr_obs=observations[i], use_sparse_reward=(current_episode_step[i]==18))

            current_episode_step += 1
            for i in range(self.envs.num_envs):
                if len(self.config.VIDEO_OPTION) > 0:
                    if config.TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE and 'intermediate' in observations[i]:
                        for observation in observations[i]['intermediate']:
                            frame = observations_to_image(observation, infos[i])
                            rgb_frames[i].append(frame)
                        del observations[i]['intermediate']

                    if "rgb" not in observations[i]:
                        observations[i]["rgb"] = np.zeros((self.config.DISPLAY_RESOLUTION,
                                                           self.config.DISPLAY_RESOLUTION, 3))
                    frame = observations_to_image(observations[i], infos[i])
                    rgb_frames[i].append(frame)

            if config.DISPLAY_RESOLUTION != model_resolution:
                resize_observation(observations, model_resolution)
            batch = batch_obs(observations, self.device)
            full_obs.append(batch)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            for i in range(self.envs.num_envs):
                # pause envs which runs out of episodes
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    episode_stats = dict()
                    for metric_uuid in self.metric_uuids:
                        episode_stats[metric_uuid] = infos[i][metric_uuid]
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats['geodesic_distance'] = current_episodes[i].info['geodesic_distance']
                    episode_stats['euclidean_distance'] = norm(np.array(current_episodes[i].goals[0].position) -
                                                               np.array(current_episodes[i].start_position))
                    spect_metrics, gts, preds = self._get_rir_error(full_obs, self.config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS, i)
                    for metric in spect_metrics:
                        episode_stats[metric] = np.array(spect_metrics[metric]).mean()
                    logging.debug(episode_stats)
                    current_episode_reward[i] = 0
                    current_episode_step[i] = 0
                    full_obs = [full_obs[-1]] #TO DO: fix this - this is resetting all obs after one episode finishes
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats
                    t.update()

                    if len(self.config.VIDEO_OPTION) > 0:
                        fps = int(1 / self.config.TASK_CONFIG.SIMULATOR.STEP_TIME)
                        if 'sound' in current_episodes[i].info:
                            sound = current_episodes[i].info['sound']
                        else:
                            sound = current_episodes[i].sound_id.split('/')[1][:-4]
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i][:-1],
                            scene_name=current_episodes[i].scene_id.split('/')[3],
                            sound=sound,
                            sr=self.config.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE,
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metric_name='no_metric',
                            metric_value=5.0,
                            tb_writer=writer,
                            fps=fps
                        )

                        # observations has been reset but info has not
                        # to be consistent, do not use the last frame
                        rgb_frames[i] = []

                    if "top_down_map" in self.config.VISUALIZATION_OPTION:
                        top_down_map = plot_top_down_map(infos[i],
                                                         dataset=self.config.TASK_CONFIG.SIMULATOR.SCENE_DATASET)
                        scene = current_episodes[i].scene_id.split('/')[3]
                        writer.add_image('{}_{}_{}/{}'.format(config.EVAL.SPLIT, scene, current_episodes[i].episode_id,
                                                              config.BASE_TASK_CONFIG_PATH.split('/')[-1][:-5]),
                                         top_down_map,
                                         dataformats='WHC')
                    if "pred_rir_spec" in self.config.VISUALIZATION_OPTION:
                        #rir_plot = plot_rir_gts_and_preds(gts, preds)
                        top_down_map = plot_top_down_map(infos[i],
                                                         dataset=self.config.TASK_CONFIG.SIMULATOR.SCENE_DATASET)
                        scene = current_episodes[i].scene_id.split('/')[3]
                        writer.add_image('{}_{}_{}/{}'.format(config.EVAL.SPLIT, scene, current_episodes[i].episode_id,
                                                              config.BASE_TASK_CONFIG_PATH.split('/')[-1][:-5]),
                                         rir_plot,
                                         dataformats='WHC')

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = sum(
                [v[stat_key] for v in stats_episodes.values()]
            )
        num_episodes = len(stats_episodes)

        stats_file = os.path.join(config.TENSORBOARD_DIR, '{}_stats_{}.json'.format(config.EVAL.SPLIT, config.SEED))
        new_stats_episodes = {','.join(key): value for key, value in stats_episodes.items()}
        with open(stats_file, 'w') as fo:
            json.dump(new_stats_episodes, fo, indent=4)

        episode_reward_mean = aggregated_stats["reward"] / num_episodes
        episode_metrics_mean = {}
        for metric_uuid in self.metric_uuids:
            episode_metrics_mean[metric_uuid] = aggregated_stats[metric_uuid] / num_episodes

        logger.info(f"Average episode reward: {episode_reward_mean:.6f}")
        for metric_uuid in self.metric_uuids:
            logger.info(
                f"Average episode {metric_uuid}: {episode_metrics_mean[metric_uuid]:.6f}"
            )

        if not config.EVAL.SPLIT.startswith('test'):
            writer.add_scalar("{}/reward".format(config.EVAL.SPLIT), episode_reward_mean, checkpoint_index)
            for metric_uuid in self.metric_uuids:
                writer.add_scalar(f"{config.EVAL.SPLIT}/{metric_uuid}", episode_metrics_mean[metric_uuid],
                                  checkpoint_index)

        self.envs.close()

        result = {
            'episode_reward_mean': episode_reward_mean
        }
        for metric_uuid in self.metric_uuids:
            result['episode_{}_mean'.format(metric_uuid)] = episode_metrics_mean[metric_uuid]

        return result
    
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
    
    def initialize_queries_gt_rirs_and_observations(self, initial_observations, i):
        observations, query_positions, gt_rirs = initial_observations
        observations['depth'] = observations['depth'].squeeze(-1)
        self.query_positions[i] = list(query_positions)
        self.gt_rirs[i] = list(gt_rirs)
        return observations


def load_rir_predictor(rir_pred_ckpt_path: str, cfg):
    device = (
            torch.device("cuda", cfg.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    rir_pred_ckpt = torch.load(rir_pred_ckpt_path, map_location=device)
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(rir_pred_ckpt['state_dict'], "actor_critic.")

    rir_predictor = UniformContextSamplerPolicy(rir_pred_ckpt['config'])
    rir_predictor = torch.nn.DataParallel(rir_predictor, device_ids=list(range(torch.cuda.device_count())),
                                                output_device=0)
    
    rir_predictor.load_state_dict(rir_pred_ckpt['state_dict'])
    rir_predictor.eval()
    rir_predictor.to(device)

    return rir_predictor