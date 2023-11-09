from typing import Any, Dict, Optional, Sequence
import numpy as np
from d3rlpy.argument_utility import (
    ActionScalerArg,
    EncoderArg,
    QFuncArg,
    RewardScalerArg,
    ScalerArg,
    UseGPUArg,
    check_encoder,
    check_q_func,
    check_use_gpu,
)
import torch
from d3rlpy.constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from d3rlpy.dataset import TransitionMiniBatch
from d3rlpy.gpu import Device
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import AdamFactory, OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.algos.base import AlgoBase
from d3rlpy_algos.algos.dw_td3bc_impl import WeightedTD3PlusBCImpl


class DWTD3PlusBC(AlgoBase):
    r"""TD3+BC algorithm.

    TD3+BC is an simple offline RL algorithm built on top of TD3.
    TD3+BC introduces BC-reguralized policy objective function.

    .. math::

        J(\phi) = \mathbb{E}_{s,a \sim D}
            [\lambda Q(s, \pi(s)) - (a - \pi(s))^2]

    where

    .. math::

        \lambda = \frac{\alpha}{\frac{1}{N} \sum_(s_i, a_i) |Q(s_i, a_i)|}

    References:
        * `Fujimoto et al., A Minimalist Approach to Offline Reinforcement
          Learning. <https://arxiv.org/abs/2106.06860>`_

    Args:
        actor_learning_rate (float): learning rate for a policy function.
        critic_learning_rate (float): learning rate for Q functions.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the critic.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        target_smoothing_sigma (float): standard deviation for target noise.
        target_smoothing_clip (float): clipping range for target noise.
        alpha (float): :math:`\alpha` value.
        update_actor_interval (int): interval to update policy function
            described as `delayed policy update` in the paper.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`.
        action_scaler (d3rlpy.preprocessing.ActionScaler or str):
            action preprocessor. The available options are ``['min_max']``.
        reward_scaler (d3rlpy.preprocessing.RewardScaler or str):
            reward preprocessor. The available options are
            ``['clip', 'min_max', 'standard']``.
        impl (d3rlpy.algos.torch.td3_impl.TD3Impl): algorithm implementation.

    """

    _actor_learning_rate: float
    _critic_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _tau: float
    _n_critics: int
    _target_smoothing_sigma: float
    _target_smoothing_clip: float
    _alpha: float
    _update_actor_interval: int
    _use_gpu: Optional[Device]
    _impl: Optional[WeightedTD3PlusBCImpl]

    def __init__(
        self,
        *,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory: OptimizerFactory = AdamFactory(),
        discriminator_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 256,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_critics: int = 2,
        target_smoothing_sigma: float = 0.2,
        target_smoothing_clip: float = 0.5,
        alpha: float = 2.5,
        update_actor_interval: int = 2,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = "standard",
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[WeightedTD3PlusBCImpl] = None,

        # For reward discriminator
        discriminator_reward_scaler: RewardScalerArg = None,
        discriminator_kl_penalty_coef = 0.001,
        discriminator_clip_ratio = 1.0,
        discriminator_weight_temp = 1.0,
        discriminator_lr = 3e-4,
        discriminator_flow_coef = 1.0,
        discriminator_discount = 1.0,


        **kwargs: Any
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            kwargs=kwargs,
        )
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._discriminator_optim_factory = discriminator_optim_factory
        self._actor_encoder_factory = check_encoder(actor_encoder_factory)
        self._critic_encoder_factory = check_encoder(critic_encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._tau = tau
        self._n_critics = n_critics
        self._target_smoothing_sigma = target_smoothing_sigma
        self._target_smoothing_clip = target_smoothing_clip
        self._alpha = alpha
        self._update_actor_interval = update_actor_interval
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl
        
        self.discriminator_reward_scaler = discriminator_reward_scaler
        self.discriminator_kl_penalty_coef = discriminator_kl_penalty_coef
        self.discriminator_clip_ratio = discriminator_clip_ratio
        self.discriminator_weight_temp = discriminator_weight_temp
        self.discriminator_lr = discriminator_lr
        self.discriminator_flow_coef = discriminator_flow_coef
        self.discriminator_discount = discriminator_discount

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = WeightedTD3PlusBCImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            discriminator_optim_factory=self._discriminator_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            tau=self._tau,
            n_critics=self._n_critics,
            target_smoothing_sigma=self._target_smoothing_sigma,
            target_smoothing_clip=self._target_smoothing_clip,
            alpha=self._alpha,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
            discriminator_reward_scaler=self.discriminator_reward_scaler,
            discriminator_kl_penalty_coef=self.discriminator_kl_penalty_coef,
            discriminator_clip_ratio=self.discriminator_clip_ratio,
            discriminator_weight_temp=self.discriminator_weight_temp,
            discriminator_lr=self.discriminator_lr,
            discriminator_flow_coef=self.discriminator_flow_coef,
            discriminator_discount=self.discriminator_discount,
        )
        self._impl.build()

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        metrics = {}

        # update disrcriminator
        normalized_dsa_ratios, normalized_ds_ratios, discriminator_losses = self._impl.update_discriminator(batch)
        metrics.update({**discriminator_losses})
        dsa_weights = normalized_dsa_ratios.detach()
        ds_weights = normalized_ds_ratios.detach()

        weights_dsa_numpy = dsa_weights.cpu().numpy()
        metrics.update({
            "weights_dsa_max": weights_dsa_numpy.max(),
            "weights_dsa_min": weights_dsa_numpy.min(),
            "weights_dsa_median": np.median(weights_dsa_numpy),
            "weights_dsa_mean": weights_dsa_numpy.mean(),
        })

        weights_ds_numpy = ds_weights.cpu().numpy()
        metrics.update({
            "weights_ds_max": weights_ds_numpy.max(),
            "weights_ds_min": weights_ds_numpy.min(),
            "weights_ds_median": np.median(weights_ds_numpy),
            "weights_ds_mean": weights_ds_numpy.mean(),
        })
        
        critic_loss = self._impl.update_critic(batch, dsa_weights, ds_weights)
        metrics.update({"critic_loss": critic_loss})

        # delayed policy update
        if self._grad_step % self._update_actor_interval == 0:
            actor_loss = self._impl.update_actor(batch, dsa_weights, ds_weights)
            metrics.update({"actor_loss": actor_loss})
            self._impl.update_critic_target()
            self._impl.update_actor_target()

        return metrics


    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS