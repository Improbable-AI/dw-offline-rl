import copy
from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

from d3rlpy.gpu import Device
from d3rlpy.models.builders import (
    create_continuous_q_function,
    create_deterministic_policy,
)
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.models.torch import (
    DeterministicPolicy,
    EnsembleContinuousQFunction,
    EnsembleQFunction,
    Policy,
)
from d3rlpy.models.torch import (
    Encoder,
)
from d3rlpy.preprocessing import ActionScaler, RewardScaler, Scaler
from d3rlpy.torch_utility import TorchMiniBatch, soft_sync, torch_api, train_api
from d3rlpy.algos.torch.base import TorchImplBase
from d3rlpy.algos.torch.utility import ContinuousQFunctionMixin

from d3rlpy.algos.torch.ddpg_impl import DDPGImpl
from d3rlpy.algos.torch.td3_impl import TD3Impl



class MeanFlowConserveDiscrimanator(nn.Module):  # type: ignore

    _action_size: int
    _state_action_encoder: Encoder
    _state_encoder: Encoder
    _fc: nn.Linear

    def __init__(self, 
        state_action_encoder: Encoder,
        state_encoder: Encoder,
        action_size: int,
        discriminator_clip_ratio: float = None,
        discriminator_temp: float = None):
        super().__init__()
        self._action_size = action_size
        self._state_action_encoder = state_action_encoder
        self._state_encoder = state_encoder

        self._state_action_fc = nn.Linear(self._state_action_encoder.get_feature_size(), 1)
        self._state_fc = nn.Linear(self._state_encoder.get_feature_size(), 1)

        self._discriminator_clip_ratio = discriminator_clip_ratio
        self._discriminator_temp = discriminator_temp

    def forward(self, x: torch.Tensor, actions: torch.Tensor = None) -> torch.Tensor:
        if actions is not None:
            return cast(torch.Tensor, self._state_action_fc(self._state_action_encoder(x, actions))), cast(torch.Tensor, self._state_fc(self._state_encoder(x)))
        else:
            return cast(torch.Tensor, self._state_fc(self._state_encoder(x)))

    def compute_normalized_ratio_and_logits(self, 
            observation: torch.Tensor, 
            actions: torch.Tensor, 
            next_observation: torch.Tensor):
        
        logits_sa, logits_s = self.forward(observation, actions)
        logits_next_s = self.forward(next_observation)
        logits = logits_sa + logits_s

        dsa_clipped_logits = torch.clip(logits,
          -(1 + self._discriminator_clip_ratio),
          (1 + self._discriminator_clip_ratio))
        normalized_dsa_ratio = F.softmax(dsa_clipped_logits / self._discriminator_temp, dim=0)

        ds_clipped_logits = torch.clip(logits_s,
          -(1 + self._discriminator_clip_ratio),
          (1 + self._discriminator_clip_ratio))
        normalized_ds_ratio = F.softmax(ds_clipped_logits / self._discriminator_temp, dim=0)

        return normalized_dsa_ratio, normalized_ds_ratio, logits_sa, logits_s, logits_next_s

class WeightedDDPGImpl(DDPGImpl):

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        discriminator_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
        discriminator_reward_scaler: Optional[RewardScaler],
        discriminator_kl_penalty_coef: Optional[float] = 1.0,
        discriminator_clip_ratio: Optional[float] = 1.0,
        discriminator_discount: Optional[float] = 1.0,
        discriminator_weight_temp: Optional[float] = 1.0,
        discriminator_lr: Optional[float] = 1e-4,
        discriminator_flow_coef: Optional[float] = 1.0,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,        
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            use_gpu=use_gpu,
        )
        self._discriminator_optim_factory = discriminator_optim_factory
        self._discriminator = None
        self._discriminator_reward_scaler = discriminator_reward_scaler

        # density-ratio weighted
        self.discriminator_kl_penalty_coef = discriminator_kl_penalty_coef
        self.discriminator_clip_ratio = discriminator_clip_ratio
        self.discriminator_discount = discriminator_discount
        self.discriminator_weight_temp = discriminator_weight_temp
        self.discriminator_lr = discriminator_lr
        self.discriminator_flow_coef = discriminator_flow_coef
    
    def _build_discriminator(self) -> None:
        state_action_encoder = self._critic_encoder_factory.create_with_action(
            self._observation_shape, self._action_size)
        state_encoder = self._critic_encoder_factory.create(
            self._observation_shape)
        self._discriminator = MeanFlowConserveDiscrimanator(
            state_action_encoder, state_encoder, self._action_size,
            self.discriminator_clip_ratio,
            self.discriminator_weight_temp)

    def _build_discriminator_optim(self) -> None:
        self._discriminator_optim = self._discriminator_optim_factory.create(
            self._discriminator.parameters(), 
            lr=self.discriminator_lr,
        )
    
    def build(self) -> None:
        # setup torch models
        self._build_critic()
        self._build_actor()
        self._build_discriminator()

        # setup target networks
        self._targ_q_func = copy.deepcopy(self._q_func)
        self._targ_policy = copy.deepcopy(self._policy)

        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

        # setup optimizer after the parameters move to GPU
        self._build_critic_optim()
        self._build_actor_optim()
        self._build_discriminator_optim()

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor, dsa_weights: torch.Tensor, ds_weights: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        td_loss = torch.tensor(
            0.0, dtype=torch.float32, device=batch.observations.device
        )

        for q_func in self._q_func._q_funcs:
            qloss = q_func.compute_error(
                observations=batch.observations,
                actions=batch.actions,
                rewards=batch.rewards,
                target=q_tpn,
                terminals=batch.terminals,
                gamma=self._gamma**batch.n_steps,
                reduction="none",
            )
            td_loss += (qloss * dsa_weights).sum()

        return td_loss

    def compute_actor_loss(self, batch: TorchMiniBatch, dsa_weights: torch.Tensor, ds_weights: torch.Tensor) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        action = self._policy(batch.observations)
        q_t = self._q_func(batch.observations, action, "none")[0]
        return -(q_t * ds_weights).sum()
    

    @train_api
    @torch_api()
    def update_discriminator(self, 
        batch: TorchMiniBatch) -> np.ndarray:
        assert self._discriminator_optim is not None
    
        self._discriminator_optim.zero_grad()
        normalized_dsa_ratios, normalized_ds_ratios, logits_sa, logits_s, logits_next_s = self._discriminator.compute_normalized_ratio_and_logits(
          batch.observations, batch.actions, batch.next_observations)
        
        normalized_rewards = self._discriminator_reward_scaler.transform(batch.rewards)
        reward_loss = -(normalized_dsa_ratios * normalized_rewards).sum()
        kl_loss = (normalized_ds_ratios * torch.log(normalized_ds_ratios)).sum()
        flow_loss = torch.square(
                self.discriminator_discount * torch.exp((logits_sa + logits_s) / self.discriminator_weight_temp) - 
                torch.exp(logits_next_s / self.discriminator_weight_temp)).mean()
        
        loss = reward_loss + \
            self.discriminator_kl_penalty_coef * kl_loss + \
            self.discriminator_flow_coef * flow_loss
        
        loss.backward()
        self._discriminator_optim.step()

        return normalized_dsa_ratios, normalized_ds_ratios, {
            "discriminator_loss": loss.cpu().detach().numpy(),
            "reward_loss": reward_loss.cpu().detach().numpy(),
            "kl_loss": kl_loss.cpu().detach().numpy(),
            "flow_loss": flow_loss.cpu().detach().numpy(),
        }


    @train_api
    @torch_api()
    def update_critic(self, batch: TorchMiniBatch, dsa_weights: torch.Tensor, ds_weights: torch.Tensor) -> np.ndarray:
        assert self._critic_optim is not None

        self._critic_optim.zero_grad()

        q_tpn = self.compute_target(batch)

        loss = self.compute_critic_loss(batch, q_tpn, dsa_weights=dsa_weights, ds_weights=ds_weights)

        loss.backward()
        self._critic_optim.step()

        return loss.cpu().detach().numpy()
    

    @train_api
    @torch_api()
    def update_actor(self, batch: TorchMiniBatch, dsa_weights: torch.Tensor, ds_weights: torch.Tensor) -> np.ndarray:
        assert self._q_func is not None
        assert self._actor_optim is not None

        # Q function should be inference mode for stability
        self._q_func.eval()
        
        self._actor_optim.zero_grad()

        loss = self.compute_actor_loss(batch, dsa_weights=dsa_weights, ds_weights=ds_weights)

        loss.backward()
        self._actor_optim.step()

        return loss.cpu().detach().numpy()


class WeightedTD3Impl(WeightedDDPGImpl):
    
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        discriminator_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        target_smoothing_sigma: float,
        target_smoothing_clip: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
        discriminator_reward_scaler: Optional[RewardScaler],
        discriminator_kl_penalty_coef: Optional[float] = 1.0,
        discriminator_clip_ratio: Optional[float] = 1.0,
        discriminator_discount: Optional[float] = 1.0,
        discriminator_weight_temp: Optional[float] = 1.0,
        discriminator_lr: Optional[float] = 1e-4,
        discriminator_flow_coef: Optional[float] = 1.0,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            discriminator_optim_factory=discriminator_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            discriminator_reward_scaler=discriminator_reward_scaler,
            discriminator_kl_penalty_coef=discriminator_kl_penalty_coef,
            discriminator_clip_ratio=discriminator_clip_ratio,
            discriminator_discount=discriminator_discount,
            discriminator_weight_temp=discriminator_weight_temp,
            discriminator_lr=discriminator_lr,
            discriminator_flow_coef=discriminator_flow_coef,
        )
        self._target_smoothing_sigma = target_smoothing_sigma
        self._target_smoothing_clip = target_smoothing_clip

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_policy is not None
        assert self._targ_q_func is not None
        with torch.no_grad():
            action = self._targ_policy(batch.next_observations)
            # smoothing target
            noise = torch.randn(action.shape, device=batch.device)
            scaled_noise = self._target_smoothing_sigma * noise
            clipped_noise = scaled_noise.clamp(
                -self._target_smoothing_clip, self._target_smoothing_clip
            )
            smoothed_action = action + clipped_noise
            clipped_action = smoothed_action.clamp(-1.0, 1.0)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                clipped_action,
                reduction="min",
            )

class WeightedTD3PlusBCImpl(WeightedTD3Impl):
     
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        discriminator_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        target_smoothing_sigma: float,
        target_smoothing_clip: float,
        alpha: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
        discriminator_reward_scaler: Optional[RewardScaler],
        discriminator_kl_penalty_coef: Optional[float] = 1.0,
        discriminator_clip_ratio: Optional[float] = 1.0,
        discriminator_discount: Optional[float] = 1.0,
        discriminator_weight_temp: Optional[float] = 1.0,
        discriminator_lr: Optional[float] = 1e-4,
        discriminator_flow_coef: Optional[float] = 1.0,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            discriminator_optim_factory=discriminator_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            target_smoothing_sigma=target_smoothing_sigma,
            target_smoothing_clip=target_smoothing_clip,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            discriminator_reward_scaler=discriminator_reward_scaler,
            discriminator_kl_penalty_coef=discriminator_kl_penalty_coef,
            discriminator_clip_ratio=discriminator_clip_ratio,
            discriminator_discount=discriminator_discount,
            discriminator_weight_temp=discriminator_weight_temp,
            discriminator_lr=discriminator_lr,
            discriminator_flow_coef=discriminator_flow_coef,
        )
        self._alpha = alpha

    def compute_actor_loss(self, batch: TorchMiniBatch, dsa_weights: torch.Tensor, ds_weights: torch.Tensor) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        action = self._policy(batch.observations)
        q_t = self._q_func(batch.observations, action, "none")[0]
        lam = self._alpha / (q_t.abs().mean()).detach()
        return lam * -(q_t * ds_weights).sum() + (dsa_weights * ((batch.actions - action) ** 2).mean(dim=1, keepdim=True)).sum()

    # def compute_actor_loss(self, batch: TorchMiniBatch, weights: torch.Tensor) -> torch.Tensor:
    #     assert self._policy is not None
    #     assert self._q_func is not None
    #     action = self._policy(batch.observations)
    #     q_t = self._q_func(batch.observations, action, "none")[0]
    #     lam = self._alpha / (q_t.abs().mean()).detach()
    #     return lam * -(q_t * weights).sum() + (weights * ((batch.actions - action) ** 2)).sum()