import absl.app
import absl.flags
import os
import time
from copy import deepcopy
import uuid
import re
import numpy as np
import pprint

import jax
import jax.numpy as jnp
import flax

import gym
import d4rl
import suboptimal_offline_datasets

from .utils import (
    Timer, define_flags_with_default, set_random_seed, print_flags,
    get_user_flags, prefix_metrics, WandBLogger
)

from .replay_buffer import get_d4rl_dataset, subsample_dataset_batch
from .jax_utils import batch_to_jax
from .conservative_sac import ConservativeSAC, DensityRatioWeightedConservativeSAC
from .model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy, FullyConnectedFlowConserveDiscriminator
from .sampler import StepSampler, TrajSampler
from viskit.logging import logger, setup_logger

from .dataset_utils import make_dataset_with_sampler



FLAGS_DEF = define_flags_with_default(
    env='halfcheetah-medium-v2',
    sampler="uniform",
    max_traj_length=1000,
    seed=42,
    save_model=False,
    batch_size=256,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=0.999,

    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    n_epochs=1000,
    bc_epochs=0,
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=5,

    reward_norm="max-min",

    cql=ConservativeSAC.get_default_config(),
    logging=WandBLogger.get_default_config(),
)


def main(argv):    
    FLAGS = absl.flags.FLAGS
    variant = get_user_flags(FLAGS, FLAGS_DEF)

    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=FLAGS.seed,
        base_log_dir=FLAGS.logging.output_dir,
        include_exp_prefix_sub_dir=False
    )

    set_random_seed(FLAGS.seed)

    eval_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length)
    
    # These samplers requires trajectory structure, but `get_d4rl_dataset` converts to qlearning dataset
    # Wrap the dataset with sampler
    dataset = make_dataset_with_sampler(FLAGS.env, eval_sampler.env, FLAGS.sampler,
                max_steps=int(FLAGS.n_epochs * FLAGS.n_train_step_per_epoch),
                batch_size=FLAGS.batch_size, reward_norm=FLAGS.reward_norm)

    dataset.rewards = dataset.rewards * FLAGS.reward_scale + FLAGS.reward_bias
    dataset.actions = np.clip(dataset.actions, -FLAGS.clip_action, FLAGS.clip_action)

    observation_dim = eval_sampler.env.observation_space.shape[0]
    action_dim = eval_sampler.env.action_space.shape[0]

    policy = TanhGaussianPolicy(
        observation_dim, action_dim, FLAGS.policy_arch, FLAGS.orthogonal_init,
        FLAGS.policy_log_std_multiplier, FLAGS.policy_log_std_offset
    )
    qf = FullyConnectedQFunction(observation_dim, action_dim, FLAGS.qf_arch, FLAGS.orthogonal_init)

    if FLAGS.cql.target_entropy >= 0.0:
        FLAGS.cql.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()
    
    if FLAGS.sampler.startswith("DRW") or \
        re.search(r"(AW|RW)-[+-]?([0-9]*[.])?[0-9]+[+]DRW-*", FLAGS.sampler): # e.g., AW-0.1+DRW-...      
        print("Use DensityRatioWeightedCQL")
        discriminator = FullyConnectedFlowConserveDiscriminator(
            observation_dim, action_dim, FLAGS.qf_arch, FLAGS.orthogonal_init)
        if '+' in FLAGS.sampler: # Composite sampler
            _, sampler = FLAGS.sampler.split("+", 2)
        else:
            sampler = FLAGS.sampler
    
        _, param = sampler.split("-")
        kl_weight, \
        flow_weight, \
        flow_discount, \
        discriminator_temp, \
        discriminator_clip_ratio, \
            discriminator_lr = param.split("_")
        sac = DensityRatioWeightedConservativeSAC(FLAGS.cql, policy, qf,
            discriminator=discriminator,
            kl_weight=float(kl_weight[1:]), # e.g., K0.1
            flow_weight=float(flow_weight[1:]), # e.g., F0.1
            flow_discount=float(flow_discount[1:]), # e.g., G1.0
            discriminator_temp=float(discriminator_temp[1:]), # e.g., T1.0
            discriminator_clip_ratio=float(discriminator_clip_ratio[1:]), # e.g., C1.0
            discriminator_lr=float(discriminator_lr[1:]), # e.g., L0.0001 (1e-4)
        )
    else:
        sampler = FLAGS.sampler
        print(f"Use CQL with {sampler}")
        sac = ConservativeSAC(FLAGS.cql, policy, qf)
    sampler_policy = SamplerPolicy(sac.policy, sac.train_params['policy'])

    viskit_metrics = {}
    for epoch in range(FLAGS.n_epochs):
        metrics = {'epoch': epoch}

        with Timer() as train_timer:
            for batch_idx in range(FLAGS.n_train_step_per_epoch):
                batch = batch_to_jax(subsample_dataset_batch(dataset, FLAGS.batch_size))
                metrics.update(prefix_metrics(sac.train(batch, bc=epoch < FLAGS.bc_epochs), 'sac'))
                
        with Timer() as eval_timer:        
            if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                trajs = eval_sampler.sample(
                    sampler_policy.update_params(sac.train_params['policy']),
                    FLAGS.eval_n_trajs, deterministic=True
                )

                metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                metrics['average_normalizd_return'] = np.mean(
                    [eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) for t in trajs]
                )
                if FLAGS.save_model:
                    save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
                    wandb_logger.save_pickle(save_data, 'model.pkl')

        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = train_timer() + eval_timer()
        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(viskit_metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    if FLAGS.save_model:
        save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
        wandb_logger.save_pickle(save_data, 'model.pkl')

if __name__ == '__main__':
    absl.app.run(main)