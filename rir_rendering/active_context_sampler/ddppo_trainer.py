#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import random
import time
import logging
from collections import defaultdict, deque
from typing import Dict, List
import json
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as distrib
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
from ss_baselines.savi.ddppo.algo.ddp_utils import (
    EXIT,
    REQUEUE,
    add_signal_handlers,
    init_distrib_slurm,
    load_interrupted_state,
    requeue_job,
    save_interrupted_state,
)
from rir_rendering.active_context_sampler.ddppo import DDPPO
from rir_rendering.active_context_sampler.ppo_trainer import ActiveRIRTrainer, load_rir_predictor
from rir_rendering.active_context_sampler.policy import ActiveRIRPolicy
from rir_rendering.active_context_sampler.ppo import PPO
from rir_rendering.uniform_context_sampler.policy import UniformContextSamplerPolicy
from rir_rendering.common.eval_metrics import compute_spect_metrics

@baseline_registry.register_trainer(name="DDPPOTrainer")
class DDPPOTrainer(ActiveRIRTrainer):
    r"""DDPPO Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    SHORT_ROLLOUT_THRESHOLD: float = 1.0

    def __init__(self, config=None):
        
        super().__init__(config)

    def _setup_actor_critic_agent(self, cfg: Config, observation_space=None) -> None:
        r"""Sets up actor critic and agent for DD-PPO.

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

        if self.config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = DDPPO(
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
        rewards = rewards.unsqueeze(1).to(device=self.device)
        episode_rir_error = torch.tensor(episode_rir_error, dtype=torch.float)
        episode_rir_error = episode_rir_error.unsqueeze(1).to(device=self.device)
        batch = batch_obs(observations)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones], dtype=torch.float
        ).to(device=self.device)

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
        episode_rir_error = None

        return pth_time, env_time, self.envs.num_envs

    def train(self) -> None:
        r"""Main method for training DD-PPO.

        Returns:
            None
        """
        self.local_rank, tcp_store = init_distrib_slurm(
            self.config.RL.DDPPO.distrib_backend,
            master_port=self.config.RL.PPO.master_port,
            master_addr=self.config.RL.PPO.master_addr,
        )
        add_signal_handlers()

        # Stores the number of workers that have finished their rollout
        num_rollouts_done_store = distrib.PrefixStore(
            "rollout_tracker", tcp_store
        )
        num_rollouts_done_store.set("num_done", "0")

        self.world_rank = distrib.get_rank()
        print(f'World rank: {self.world_rank}')
        self.world_size = distrib.get_world_size()

        self.config.defrost()
        self.config.TORCH_GPU_ID = self.local_rank
        self.config.SIMULATOR_GPU_ID = self.local_rank
        # Multiply by the number of simulators to make sure they also get unique seeds
        self.config.TASK_CONFIG.SEED += (
            self.world_rank * self.config.NUM_PROCESSES
        )
        self.config.freeze()

        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.rir_predictor = load_rir_predictor(self.config.RL.PRETRAINED_RIR_PREDICTOR_PATH, self.device, distributed=True)

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        ppo_cfg = self.config.RL.PPO
        if (
            not os.path.isdir(self.config.CHECKPOINT_FOLDER)
            and self.world_rank == 0
        ):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent(self.config)
        self.agent.init_distributed(find_unused_params=True)


        
        if self.world_rank == 0:
            logger.info(
                "agent number of trainable parameters: {}".format(
                    sum(
                        param.numel()
                        for param in self.agent.parameters()
                        if param.requires_grad
                    )
                )
            )
            logger.info(f"config: {self.config}")

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

        self.novelty_count = [
            dict() for _ in range(self.envs.num_envs)
        ]

        self.gt_rirs_mag = torch.zeros(torch.Size([self.envs.num_envs, 60, 256, 259, 2]))
        self.gt_rirs_phase = torch.zeros(torch.Size([self.envs.num_envs, 60, 256, 259, 2]))
        self._curr_rir_error = torch.zeros(self.envs.num_envs, 1)

        #get initial observations
        observations_and_queries = self.envs.reset()
        observations = [self.initialize_queries_gt_rirs_and_observations(obs, i) for i, obs in enumerate(observations_and_queries)]
        batch = batch_obs(observations, device=self.device)
        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        batch = None
        observations = None
        

        # episode_rewards and episode_counts accumulates over the entire training course
        episode_rewards = torch.zeros(self.envs.num_envs, 1, device=self.device)
        episode_steps = torch.zeros(self.envs.num_envs, 1, device=self.device)
        episode_counts = torch.zeros(self.envs.num_envs, 1, device=self.device)
        episode_rir_errors = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_step = torch.zeros(self.envs.num_envs, 1, device=self.device)
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

        with (
            TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            )
            if self.world_rank == 0
            else contextlib.suppress()
        ) as writer:
            for update in range(self.config.NUM_UPDATES):
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

                    # This is where the preemption of workers happens.  If a
                    # worker detects it will be a straggler, it preempts itself!
                    if (
                        step
                        >= ppo_cfg.num_steps * self.SHORT_ROLLOUT_THRESHOLD
                    ) and int(num_rollouts_done_store.get("num_done")) > (
                        self.config.RL.DDPPO.sync_frac * self.world_size
                    ):
                        break
                
                num_rollouts_done_store.add("num_done", 1)

                self.agent.train()
                delta_pth_time, value_loss, action_loss, dist_entropy = self._update_agent(
                    ppo_cfg, rollouts
                )
                pth_time += delta_pth_time

                stats = torch.stack(
                    [episode_counts, episode_rewards, episode_steps, episode_rir_errors], 0
                )
                distrib.all_reduce(stats)

                window_episode_counts.append(stats[0].clone())
                window_episode_reward.append(stats[1].clone())
                window_episode_step.append(stats[2].clone())
                window_episode_rir_error.append(stats[3].clone())

                window_episode_stats = zip(
                        ["count", "reward", "step", "rir_error"],
                        [window_episode_counts, window_episode_reward, window_episode_step, window_episode_rir_error],
                )

                stats = torch.tensor(
                    [value_loss, action_loss, dist_entropy],
                    device=self.device,
                )
                distrib.all_reduce(stats)

                if self.world_rank == 0:
                    num_rollouts_done_store.set("num_done", "0")
                    losses = [
                        stats[0].item() / self.world_size,
                        stats[1].item() / self.world_size,
                        stats[2].item() / self.world_size,
                    ]
                    deltas = {
                        k: (
                            (v[-1] - v[0]).sum().item()
                            if len(v) > 1
                            else v[0].sum().item()
                        )
                        for k, v in window_episode_stats
                    }
                    deltas["count"] = max(deltas["count"], 1.0)

                    # this reward is averaged over all the episodes happened during window_size updates
                    # approximately number of steps is window_size * num_steps
                    writer.add_scalar("Environment/Reward", deltas["reward"] / deltas["count"], count_steps)
                    writer.add_scalar("Environment/Episode_length", deltas["step"] / deltas["count"], count_steps)
                    writer.add_scalar("Environment/RIR_Error", deltas["rir_error"] / deltas["count"], count_steps)
                    writer.add_scalar('Policy/Value_Loss', losses[0], count_steps)
                    writer.add_scalar('Policy/Action_Loss', losses[1], count_steps)
                    writer.add_scalar('Policy/Entropy', losses[2], count_steps)
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