from typing import Tuple

import collections
import random
from typing import Optional

import d4rl
import gym
import numpy as np
from tqdm import tqdm

from .replay_buffer import subsample_batch, get_d4rl_dataset
from .jax_utils import batch_to_jax


Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int, weights: Optional[np.ndarray] = None,
                 reward_norm: str = "max-min"):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.reward_norm = reward_norm
        
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size
        self.weights = weights
        self.indices = range(size)

        self.max_reward = self.rewards.max()
        self.min_reward = self.rewards.min()
        self.reward_range = self.max_reward - self.min_reward
        if self.reward_norm is None or self.reward_norm == "max-min":        
          self.normalized_rewards = (self.rewards - self.min_reward) / self.reward_range
        elif self.reward_norm == "std":
          self.normalized_rewards = self.rewards / self.rewards.std()

    def set_weight(self, weights):
      self.weights = weights

    def sample(self, batch_size: int) -> Batch:
        if self.weights is not None:
          indx = np.random.choice(self.indices, p=self.weights, size=batch_size)
        else:
          indx = np.random.randint(self.size, size=batch_size)
        return dict(observations=self.observations[indx],
                    actions=self.actions[indx],
                    next_observations=self.next_observations[indx],
                    rewards=self.rewards[indx],
                    dones=self.dones_float[indx],
                    normalized_rewards=self.normalized_rewards[indx])

class D4RLDataset(Dataset):
    def __init__(self, dataset, reward_norm="max-min"):
        if "dones" in dataset:
          dataset["terminals"] = dataset["dones"]
        dones_float = dataset["terminals"]
        dones_float[-1] = 1

        super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(
                             np.float32),
                         size=len(dataset['observations']),
                         reward_norm=reward_norm)

class DatasetWrapper:

  def __init__(self, dataset) -> None:
    self.dataset = dataset
    for attr_key, attr_val in dataset.__dict__.items():
      setattr(self, attr_key, attr_val)

class PreSampleWeightedDataset(DatasetWrapper):

  def __init__(self, dataset, weights, n_batches, batch_size) -> None:
      super().__init__(dataset)
      self.weights = weights
      self.n_batches = n_batches
      self.batch_size = batch_size
      self.num_presampled = n_batches * batch_size

      self.all_indx = np.random.choice(self.indices, p=self.weights, size=(batch_size * n_batches))

  def set_weight(self, weights):
      self.dataset.set_weight(weights)

  def sample(self, batch_size: int) -> Batch:
    indx = self.all_indx[np.random.randint(self.num_presampled, size=batch_size)]
    return dict(observations=self.observations[indx],
              actions=self.actions[indx],
              next_observations=self.next_observations[indx],
              rewards=self.rewards[indx],
              dones=self.dones_float[indx],
              normalized_rewards=self.normalized_rewards[indx])


def qlearning_dataset_with_weights(
    dataset,
    terminate_on_end=False,
    **kwargs):
    """
    Adapt from D4RL
    """

    assert "weights" in dataset
    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    weight_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])
        weight = dataset['weights'][i]

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        weight_.append(weight)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
        'weights': np.array(weight_),
    }



def sequence_dataset(env, dataset=None, **kwargs):
    """
    Returns an iterator through trajectories.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in ['next_observations', 'actions', 'observations', 'rewards', 'terminals', 'timeouts']:
            if k == "next_observations" and "next_observations" not in dataset:
               continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1


def AW(trajs):
  import functools
  from sklearn.linear_model import LinearRegression
  ep_rets = np.asarray(list(map(lambda traj: traj["rewards"].sum(), trajs)))
  ep_lens =  np.asarray(list(map(lambda traj: traj["observations"].shape[0], trajs)))
  s0s = np.array([traj["observations"][0] for traj in trajs])
  v = LinearRegression().fit(s0s, ep_rets).predict(s0s)
  weights = np.asarray(functools.reduce(lambda a, b: a + b,
              [[w] * l for w, l in zip((ep_rets - v), ep_lens)]))
  weights = (weights - weights.min()) / (weights.max() - weights.min())
  dataset = {k: np.concatenate([traj[k] for traj in trajs], axis=0) for k in trajs[0].keys()}
  dataset["weights"] = weights
  return dataset

def RW(trajs):
  import functools
  ep_rets = np.asarray(list(map(lambda traj: traj["rewards"].sum(), trajs)))
  ep_rets = (ep_rets - ep_rets.min()) / (ep_rets.max() - ep_rets.min())
  ep_lens =  np.asarray(list(map(lambda traj: traj["observations"].shape[0], trajs)))
  weights =  np.asarray(functools.reduce(lambda a, b: a + b,
              [[w] * l for w, l in zip(ep_rets, ep_lens)]))
  dataset = {k: np.concatenate([traj[k] for traj in trajs], axis=0) for k in trajs[0].keys()}
  dataset["weights"] = weights
  return dataset

def TopK(trajs, percent):
  traj_rets = list(map(lambda traj: traj["rewards"].sum(), trajs))
  sorted_indices = np.argsort(traj_rets)[::-1]
  topk_trajs = [trajs[i] for i in sorted_indices[:int(len(trajs) * percent)]]
  topk_dataset = {k: np.concatenate([traj[k] for traj in topk_trajs], axis=0) for k in topk_trajs[0].keys()}
  return topk_dataset

def make_dataset_with_sampler(env_name, env, sampler,
                              # For pre-sampling
                              max_steps, batch_size, reward_norm):   
    if sampler.startswith("Top"):
      dataset = d4rl.qlearning_dataset(env, 
          dataset=TopK(list(sequence_dataset(env)), 
          float(sampler.split("-")[-1])))
      dataset = D4RLDataset(dataset, reward_norm=reward_norm)
      return dataset
    elif sampler.startswith("RewW"):
      from scipy.special import softmax
      dataset = get_d4rl_dataset(env)
      dataset = D4RLDataset(dataset, reward_norm=reward_norm)
      temp = float(sampler.split("-")[1])
      normalized_rewards = (dataset.rewards - dataset.rewards.min()) / (dataset.rewards.max() - dataset.rewards.min())
      dataset = PreSampleWeightedDataset(dataset,
        softmax(normalized_rewards / temp),
        max_steps + 1, batch_size)
    elif sampler.startswith("RW") or sampler.startswith("AW"): # ICLR AW/RW
        from scipy.special import softmax
        if '+' in sampler: # Composite sampler
          sampler, _ = sampler.split('+', 2)
        prefix, temp = sampler.split("-")
        temp = float(temp)
        if prefix == "RW":
          dataset = RW(list(sequence_dataset(env)))
        elif prefix == "AW":
          dataset = AW(list(sequence_dataset(env)))
        dataset = qlearning_dataset_with_weights(dataset)
        weights = dataset["weights"]
        dataset = D4RLDataset(dataset, reward_norm=reward_norm)
        dataset = PreSampleWeightedDataset(dataset,
          softmax(weights / temp),
          max_steps + 1, batch_size)
    elif sampler.startswith("DW"):        
        _, optdice_ckpt_iter = sampler.split("-")
        weights = load_sample_weights(env_name, int(optdice_ckpt_iter))
        dataset = get_d4rl_dataset(env)    
        dataset = D4RLDataset(dataset, reward_norm=reward_norm)           
        dataset = PreSampleWeightedDataset(dataset,
                      weights / weights.sum(),
                      max_steps + 1, batch_size)
        return dataset
    elif sampler.startswith("SoftDW"):
        from scipy.special import softmax
        def _normalize(w, norm):
          if norm == "maxmin":
            return (w - w.min()) / (w.max() - w.min())
          elif norm == "none":
            return w
          else:
            raise NotImplemented
        _, param = sampler.split("-")
        optdice_ckpt_iter, norm, temp = param.split("_")
        weights = load_sample_weights(env_name, int(optdice_ckpt_iter))
        dataset = get_d4rl_dataset(env)    
        dataset = D4RLDataset(dataset, reward_norm=reward_norm)
        dataset = PreSampleWeightedDataset(dataset,
                      softmax(_normalize(weights, norm) / float(temp)),
                      max_steps + 1, batch_size)
        return dataset
    elif sampler.startswith("PowerDW+"): # PowerDW+AW-1000-none-0.5-0.2-0.5
        def _normalize(w, norm):
          if norm == "maxmin":
            return (w - w.min()) / (w.max() - w.min())
          elif norm == "none":
            return w
          else:
            raise NotImplemented
        prefix, param = sampler.split("-")
        sampler1, sampler2 = prefix.split("+")
        optdice_ckpt_iter, norm, beta, temp, alpha = param.split("_")

        # weights1
        weights1 = load_sample_weights(env_name, int(optdice_ckpt_iter))
        weights1 = _normalize(weights1, norm)
        weights1 = weights1**float(beta)
        weights1 /= weights1.sum()

        # Weights2
        from scipy.special import softmax
        temp = float(temp)
        if sampler2 == "RW":
          dataset2 = RW(list(sequence_dataset(env)))
        elif sampler2 == "AW":
          dataset2 = AW(list(sequence_dataset(env)))
        dataset2 = qlearning_dataset_with_weights(dataset2)
        weights2 = dataset2["weights"]
        weights2 = softmax(weights2 / temp)

        alpha = float(alpha)
        dataset = get_d4rl_dataset(env)    
        dataset = D4RLDataset(dataset, reward_norm=reward_norm)
        # print((weights1 * alpha + weights2 * (1.0 - alpha)).sum())
        comb_weights =  (weights1 * alpha + weights2 * (1.0 - alpha))

        dataset = PreSampleWeightedDataset(dataset,
                      comb_weights / comb_weights.sum(),
                      max_steps + 1, batch_size)
        return dataset
    elif sampler.startswith("PowerDW"):
        def _normalize(w, norm):
          if norm == "maxmin":
            return (w - w.min()) / (w.max() - w.min())
          elif norm == "none":
            return w
          else:
            raise NotImplemented
        _, param = sampler.split("-")
        optdice_ckpt_iter, norm, beta = param.split("_")
        weights = load_sample_weights(env_name, int(optdice_ckpt_iter))
        weights = _normalize(weights, norm)
        weights = weights**float(beta)

        dataset = get_d4rl_dataset(env)    
        dataset = D4RLDataset(dataset, reward_norm=reward_norm)           
        dataset = PreSampleWeightedDataset(dataset,
                      weights / weights.sum(),
                      max_steps + 1, batch_size)
        return dataset
    elif sampler == "uniform" or sampler.startswith("DRW"):
        dataset = get_d4rl_dataset(env)
        dataset = D4RLDataset(dataset, reward_norm=reward_norm)
        return dataset
    elif sampler.startswith("OptDiCEW"):
        dataset = get_d4rl_dataset(env)
        dataset = OptDiCED4RLDataset(dataset, reward_norm=reward_norm)
    else:
       raise NotImplemented()
    
    return dataset


