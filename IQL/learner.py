"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.nn as nn
import numpy as np
import optax

import policy
import value_net
from actor import update as awr_update_actor
from actor import update_weighted_sum as awr_update_actor_weighted_sum
from common import Batch, InfoDict, Model, PRNGKey
from critic import update_q, update_v
from critic import update_q_weighted_sum, update_v_weighted_sum
from discriminator import update_discriminator

from tqdm import *


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)


@jax.jit
def _update_jit(
    rng: PRNGKey, actor: Model, critic: Model, value: Model,
    target_critic: Model, batch: Batch, discount: float, tau: float,
    expectile: float, temperature: float, exp_a_clip: float,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:

    new_value, value_info = update_v(target_critic, value, batch, expectile)
    key, rng = jax.random.split(rng)
    new_actor, actor_info = awr_update_actor(key, actor, target_critic,
                                             new_value, batch, temperature, exp_a_clip)

    new_critic, critic_info = update_q(critic, new_value, batch, discount)

    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_actor, new_critic, new_value, new_target_critic, {
        **critic_info,
        **value_info,
        **actor_info
    }


@jax.jit
def _update_rewards_reweighted_jit(
    rng: PRNGKey, actor: Model, critic: Model, value: Model, discriminator: Model,
    target_critic: Model, batch: Batch, discount: float, tau: float,
    expectile: float, temperature: float, exp_a_clip: float,
    weight_temp: float, clip_ratio: float, kl_penalty_coeff:float, 
    flow_coeff: float, flow_discount: float,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:

    sa_ratios, s_ratios = discriminator(batch.observations, batch.actions)
    sa_clipped_ratios = jnp.clip(sa_ratios, -(1 + clip_ratio), 1 + clip_ratio)
    s_clipped_ratios = jnp.clip(s_ratios, -(1 + clip_ratio), 1 + clip_ratio)

    # exp(f(s)) exp(g(a|s)) = exp(f(s) + g(a|s))
    clipped_ratios = sa_clipped_ratios + s_clipped_ratios
    normalized_ratios = nn.softmax(clipped_ratios / weight_temp, axis=0)

    weights = jax.lax.stop_gradient(normalized_ratios)

    new_value, value_info = update_v_weighted_sum(target_critic, value, batch, expectile, weights)
    key, rng = jax.random.split(rng)
    new_actor, actor_info = awr_update_actor_weighted_sum(key, actor, target_critic,
                                             new_value, batch, temperature, exp_a_clip, weights)

    new_discriminator, discriminator_info = update_discriminator(
        discriminator,
        batch,
        flow_discount,
        weight_temp, clip_ratio, kl_penalty_coeff, flow_coeff)

    new_critic, critic_info = update_q_weighted_sum(critic, new_value, batch, discount, weights)

    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_actor, new_critic, new_value, new_discriminator, new_target_critic, {
        **critic_info,
        **value_info,
        **actor_info,
        **discriminator_info,
        "weight_mean": weights.mean(),
        "weight_max": weights.max(),
        "weight_min": weights.min(),
    }



class Learner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 expectile: float = 0.8,
                 temperature: float = 0.1,
                 exp_a_clip: float = 100,
                 dropout_rate: Optional[float] = None,
                 max_steps: Optional[int] = None,
                 opt_decay_schedule: str = "cosine"):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.temperature = temperature
        self.exp_a_clip = exp_a_clip
        self.hidden_dims = hidden_dims
        self.observations = observations
        self.actions = actions
        self.critic_lr = critic_lr

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key, self.discriminator_key = jax.random.split(rng, 5)

        action_dim = actions.shape[-1]
        actor_def = policy.NormalTanhPolicy(hidden_dims,
                                            action_dim,
                                            log_std_scale=1e-3,
                                            log_std_min=-5.0,
                                            dropout_rate=dropout_rate,
                                            state_dependent_std=False,
                                            tanh_squash_distribution=False)

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            optimiser = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            optimiser = optax.adam(learning_rate=actor_lr)

        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optimiser)

        critic_def = value_net.DoubleCritic(hidden_dims)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))

        value_def = value_net.ValueCritic(hidden_dims)
        value = Model.create(value_def,
                             inputs=[value_key, observations],
                             tx=optax.adam(learning_rate=value_lr))

        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        self.actor = actor
        self.critic = critic
        self.value = value
        self.target_critic = target_critic
        self.rng = rng

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policy.sample_actions(self.rng, self.actor.apply_fn,
                                             self.actor.params, observations,
                                             temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        new_rng, new_actor, new_critic, new_value, new_target_critic, info = _update_jit(
            self.rng, self.actor, self.critic, self.value, self.target_critic,
            batch, self.discount, self.tau, self.expectile, self.temperature, self.exp_a_clip)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.target_critic = new_target_critic

        return info


class DensityRatioWeightedLearner(Learner):

    def __init__(self, *args,
      flow_coeff,
      flow_discount,
      weight_temp,
      discriminator_lr,
      clip_ratio, 
      kl_penalty_coeff,
      **kwargs):
      super().__init__(*args, **kwargs)

      self.flow_coeff = flow_coeff
      self.flow_discount = flow_discount
      self.weight_temp = weight_temp
      self.clip_ratio = clip_ratio
      self.kl_penalty_coeff = kl_penalty_coeff
      discriminator_def = value_net.StateActionFlowConserveDiscriminator(self.hidden_dims)
      self.discriminator = Model.create(discriminator_def,
                            inputs=[
                              self.discriminator_key,
                              self.observations,
                              self.actions],
                            tx=optax.adam(learning_rate=discriminator_lr))


    def update(self, batch: Batch) -> InfoDict:
        new_rng, new_actor, new_critic, new_value, new_discriminator, new_target_critic, info = _update_rewards_reweighted_jit(
            self.rng, self.actor, self.critic, self.value, self.discriminator, self.target_critic,
            batch, self.discount, self.tau, self.expectile, self.temperature, self.exp_a_clip,
            self.weight_temp, self.clip_ratio, self.kl_penalty_coeff, self.flow_coeff, self.flow_discount)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.discriminator = new_discriminator
        self.target_critic = new_target_critic

        return info