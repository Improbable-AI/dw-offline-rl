from typing import Tuple

import jax
import jax.nn as nn
import jax.numpy as jnp
from jax.experimental import checkify

from common import Batch, InfoDict, Model, Params, PRNGKey

# from jax.config import config
# config.update("jax_debug_nans", True)


def update_discriminator(
    discriminator: Model,
    batch: Batch,
    gamma: float,
    weight_temp: float,
    clip_ratio: float,
    kl_penalty_coeff: float,
    flow_consistency_coeff: float) -> Tuple[Model, InfoDict]:

    def discriminator_loss_fn(discriminator_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        sa_ratios, s_ratios = discriminator.apply({'params': discriminator_params},
                              batch.observations,
                              batch.actions)
        sa_clipped_ratios = jnp.clip(sa_ratios, -(1 + clip_ratio), 1 + clip_ratio)
        s_clipped_ratios = jnp.clip(s_ratios, -(1 + clip_ratio), 1 + clip_ratio)

        clipped_ratios = sa_clipped_ratios + s_clipped_ratios
        normalized_ratios = nn.softmax(clipped_ratios / weight_temp, axis=0)

        _, next_s_ratios = discriminator.apply({'params': discriminator_params},
                      batch.next_observations,
                      batch.actions)

        reward_loss = -(normalized_ratios * batch.normalized_rewards).sum()
        kl_penalty_loss = (normalized_ratios * jnp.log(normalized_ratios)).sum()
        flow_consistency_loss = jnp.square((gamma * jnp.exp((sa_ratios + s_ratios) / weight_temp)) - jnp.exp(next_s_ratios / weight_temp)).mean()
        discriminator_loss = reward_loss + flow_consistency_coeff * flow_consistency_loss + kl_penalty_coeff * kl_penalty_loss

        return discriminator_loss, {
            'discriminator_loss': discriminator_loss,
            'reward_loss': reward_loss,
            'flow_consistency_loss': flow_consistency_loss,
            "kl_penalty_loss":  kl_penalty_loss,

            "sa_ratios": sa_ratios.mean(),
            "sa_clipped_ratios": sa_clipped_ratios.mean(),

            "s_ratios": s_ratios.mean(),
            "s_clipped_ratios": s_clipped_ratios.mean(),

            "next_s_ratios": next_s_ratios.mean(),

            'clipped_ratio': clipped_ratios.mean(),
            'norm_ratio': normalized_ratios.mean(),
            "discriminator_reward": batch.normalized_rewards.mean()
        }

    new_discriminator, info = discriminator.apply_gradient(discriminator_loss_fn)

    return new_discriminator, info