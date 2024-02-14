#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import cv2
import random
import time
import logging
from collections import defaultdict, deque, Counter
from typing import Dict, List
import json
import math
import pickle
import h5py
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as distrib
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from numpy.linalg import norm
from gym import spaces
from collections import defaultdict
from einops import rearrange, asnumpy

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
from rir_rendering.active_context_sampler.ddppo.ddppo import DDPPO
from rir_rendering.active_context_sampler.ddppo.ppo_trainer import ActiveRIRTrainer
from rir_rendering.active_context_sampler.ppo.ppo_trainer import load_rir_predictor
from rir_rendering.active_context_sampler.ppo.policy import ActiveRIRPolicy
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
        log_file = self.config.LOG_FILE.split(".log")[0] + "_process_" + str(self.local_rank) + ".log"
        if not os.path.exists(log_file):
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        logger.add_filehandler(log_file)

        # Setup heuristic stop criterion if applicable
        action_space = self.envs.action_spaces[0]
        self.action_space = action_space

        ppo_cfg = cfg.RL.PPO

        if observation_space is None:
            observation_space = self.envs.observation_spaces[0]
        self.actor_critic = ActiveRIRPolicy(
            cfg,
            action_space=self.action_space,
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
            use_normalized_advantage=True
        )


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
        ans_cfg = self.config.RL.ANS
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
            self.action_space,
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
        self._prev_coverage = torch.zeros(self.envs.num_envs, 1)

        M = ans_cfg.overall_map_size
        ground_truth_states = {
            # To measure area seen
            "visible_occupancy": torch.zeros(
                self.envs.num_envs, 2, M, M, device=self.device, requires_grad=False
            ),
        }

        #get initial observations
        observations_and_queries = self.envs.reset()
        observations = [self.initialize_queries_gt_rirs_and_observations(obs, i) for i, obs in enumerate(observations_and_queries)]
        batch = batch_obs(observations, device=self.device)
        prev_pose = batch['pose'].clone()

        ground_truth_states[
            "visible_occupancy"
        ] = self.mapper.ext_register_map(
            ground_truth_states["visible_occupancy"],
            batch["ego_map"].permute(0, 3, 1, 2),
            batch["pose"],
            prev_pose
        )

        with torch.no_grad():
            downsampled_map = F.interpolate(ground_truth_states[
                "visible_occupancy"
            ],
            size=(241, 241),
            mode='bilinear',
            align_corners=False)

        batch['occupancy_map'] = downsampled_map.clone()
        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        batch = None
        observations = None
        torch.cuda.empty_cache() #TODO: This was done for CUDA reasons. May not be necessary anymore.

        # episode_rewards and episode_counts accumulates over the entire training course
        episode_rewards = torch.zeros(self.envs.num_envs, 1, device=self.device, requires_grad=False) #TODO: remove requires_grad=False
        episode_steps = torch.zeros(self.envs.num_envs, 1, device=self.device, requires_grad=False)
        episode_counts = torch.zeros(self.envs.num_envs, 1, device=self.device, requires_grad=False)
        episode_rir_errors = torch.zeros(self.envs.num_envs, 1, device=self.device, requires_grad=False)
        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device, requires_grad=False)
        current_episode_step = torch.zeros(self.envs.num_envs, 1, device=self.device, requires_grad=False)
        window_episode_reward = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_step = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_counts = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_rir_error = deque(maxlen=ppo_cfg.reward_window_size)

        episode_novelty_rewards = torch.zeros(self.envs.num_envs, 1)
        episode_coverage_rewards = torch.zeros(self.envs.num_envs, 1)
        episode_acoustic_rewards = torch.zeros(self.envs.num_envs, 1)
        current_episode_novelty_reward = torch.zeros(self.envs.num_envs, 1)
        current_episode_coverage_reward = torch.zeros(self.envs.num_envs, 1)
        current_episode_acoustic_reward = torch.zeros(self.envs.num_envs, 1)
        window_episode_novelty_reward = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_coverage_reward = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_acoustic_reward = deque(maxlen=ppo_cfg.reward_window_size)

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
                    delta_pth_time, delta_env_time, delta_steps, prev_pose = self._collect_rollout_step(
                        rollouts,
                        current_episode_reward,
                        current_episode_novelty_reward,
                        current_episode_coverage_reward,
                        current_episode_acoustic_reward,
                        current_episode_step,
                        episode_rewards,
                        episode_novelty_rewards,
                        episode_coverage_rewards,
                        episode_acoustic_rewards,
                        episode_counts,
                        episode_steps,
                        episode_rir_errors,
                        ground_truth_states,
                        prev_pose
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
                window_episode_novelty_reward.append(episode_novelty_rewards.clone())
                window_episode_coverage_reward.append(episode_coverage_rewards.clone())
                window_episode_acoustic_reward.append(episode_acoustic_rewards.clone())
                window_episode_step.append(stats[2].clone())
                window_episode_rir_error.append(stats[3].clone())

                window_episode_stats = zip(
                    ["count", "reward", "novelty_reward", "coverage_reward", "acoustic_reward", "step", "rir_error"],
                    [window_episode_counts, window_episode_reward, window_episode_novelty_reward, window_episode_coverage_reward, window_episode_acoustic_reward, window_episode_step, window_episode_rir_error],
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
                        for k, v in window_episode_stats if k != "rir_error"
                    }
                    deltas["rir_error"] = (window_episode_rir_error[0] - window_episode_rir_error[-1]).sum().item() if len(window_episode_rir_error) > 1 else window_episode_rir_error[0].sum().item()
                    #deltas["rir_error"] = total_error = sum(tensor.sum() for tensor in window_episode_rir_error).item()
                    deltas["count"] = max(deltas["count"], 1.0)

                    # this reward is averaged over all the episodes happened during window_size updates
                    # approximately number of steps is window_size * num_steps
                    if update % 10 == 0:
                        writer.add_scalar("Environment/Reward", deltas["reward"] / deltas["count"], count_steps)
                        if self.config.RL.WITH_NOVELTY_REWARD:
                            writer.add_scalar("Environment/Novelty_reward", deltas["novelty_reward"] / deltas["count"], count_steps)
                        if self.config.RL.WITH_COVERAGE_REWARD:
                            writer.add_scalar("Environment/Coverage_reward", deltas["coverage_reward"] / deltas["count"], count_steps)
                        if self.config.RL.WITH_RIR_REWARD:
                            writer.add_scalar("Environment/Acoustic_reward", deltas["acoustic_reward"] / deltas["count"], count_steps)
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