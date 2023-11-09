import collections
from typing import Optional

import d4rl
import gym
import numpy as np
from scipy.special import softmax
from tqdm import tqdm

import wrappers
import suboptimal_offline_datasets

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations', "normalized_rewards"])


def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs

def split_dataset_into_trajectories(dataset):
  return split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for (obs, act, rew, mask, done, next_obs) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return np.stack(observations), np.stack(actions), np.stack(
        rewards), np.stack(masks), np.stack(dones_float), np.stack(
            next_observations)


class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int, weights: Optional[np.ndarray] = None):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.normalized_rewards = self.rewards / self.rewards.std()
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size
        self.weights = weights
        self.indices = range(size)

    def set_weight(self, weights):
      self.weights = weights
    
    def normalize_rewards(self, scheme):
      if scheme == "max-min":
        self.normalized_rewards = (self.rewards - self.rewards.min()) / (self.rewards.max() - self.rewards.min())
      elif scheme == "std":
        self.rewards / self.rewards.std()
      else:
        raise NotImplemented()

    def iter(self, batch_size: int) -> Batch:
        n_batches = int(np.ceil(self.size / batch_size))
        new_batch_size = self.size // n_batches
        batch_start_indices = np.arange(0, self.size, new_batch_size)
        for start_idx in batch_start_indices:
           yield Batch(observations=self.observations[start_idx:start_idx + new_batch_size],
                     actions=self.actions[start_idx:start_idx + new_batch_size],
                     rewards=self.rewards[start_idx:start_idx + new_batch_size],
                     masks=self.masks[start_idx:start_idx + new_batch_size],
                     next_observations=self.next_observations[start_idx:start_idx + new_batch_size],
                     normalized_rewards=self.normalized_rewards[start_idx:start_idx + new_batch_size])
       

    def sample(self, batch_size: int) -> Batch:
        if self.weights is not None:
          indx = np.random.choice(self.indices, p=self.weights, size=batch_size)
        else:
          indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx],
                     normalized_rewards=self.normalized_rewards[indx])


class D4RLDataset(Dataset):
    def __init__(self,
                 env: Optional[gym.Env] = None,
                 dataset: Optional[dict] = None,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):
        if dataset is None:
          dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(
                             np.float32),
                         size=len(dataset['observations']))

def get_return(traj):
  return np.sum([e[2] for e in traj])

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
    return Batch(observations=self.observations[indx],
                actions=self.actions[indx],
                rewards=self.rewards[indx],
                masks=self.masks[indx],
                next_observations=self.next_observations[indx],
                normalized_rewards=self.normalized_rewards[indx])
  
  def iter(self, batch_size: int) -> Batch:
        n_batches = int(np.ceil(self.size / batch_size))
        new_batch_size = self.size // n_batches
        batch_start_indices = np.arange(0, self.size, new_batch_size)
        for start_idx in batch_start_indices:
           yield Batch(observations=self.observations[start_idx:start_idx + new_batch_size],
                     actions=self.actions[start_idx:start_idx + new_batch_size],
                     rewards=self.rewards[start_idx:start_idx + new_batch_size],
                     masks=self.masks[start_idx:start_idx + new_batch_size],
                     next_observations=self.next_observations[start_idx:start_idx + new_batch_size],
                     normalized_rewards=self.normalized_rewards[start_idx:start_idx + new_batch_size])
  
  def normalize_rewards(self, scheme):
      return self.dataset.normalize_rewards(scheme)

def AW(trajs, temp):
  import functools
  from scipy.special import softmax
  from sklearn.linear_model import LinearRegression
  ep_rets = np.asarray(list(map(get_return, trajs))) 
  ep_lens =  np.asarray(list(map(len, trajs)))
  s0s = np.array([traj[0][0] for traj in trajs])
  v = LinearRegression().fit(s0s, ep_rets).predict(s0s)
  weights = np.asarray(functools.reduce(lambda a, b: a + b,
              [[w] * l for w, l in zip((ep_rets - v), ep_lens)]))
  weights = (weights - weights.min()) / (weights.max() - weights.min())
  return np.asarray(softmax(weights / temp))

def RW(trajs, temp):
  import functools
  from scipy.special import softmax
  ep_rets = np.asarray(list(map(get_return, trajs)))
  ep_rets = (ep_rets - ep_rets.min()) / (ep_rets.max() - ep_rets.min())
  ep_lens =  np.asarray(list(map(len, trajs)))
  weights =  np.asarray(functools.reduce(lambda a, b: a + b,
              [[w] * l for w, l in zip(ep_rets, ep_lens)]))
  return np.asarray(softmax(weights / temp))

def TopK(trajs, p):
  import functools
  ep_rets = np.asarray(list(map(get_return, trajs)))
  ep_lens =  np.asarray(list(map(len, trajs)))
  sorted_indices = np.argsort(ep_rets)[::-1]
  top_indices = sorted_indices[:int(len(trajs) * p)]
  weights = np.asarray(functools.reduce(lambda a, b: a + b,
              [[float(i in top_indices)] * l for i, l in enumerate(ep_lens)]))
  return weights / weights.sum()

def RewW(trajs, temp):
  import functools
  return np.asarray(functools.reduce(lambda a, b: a + b,
                [[t[2] for t in traj] for traj in trajs]))


def wrap_dataset(env_name, dataset, sampler, 
      max_steps, batch_size):
    if sampler.startswith("uniform"):
      return dataset
    elif sampler.startswith("RW") \
      or sampler.startswith("AW") \
      or sampler.startswith("RewW") \
      or sampler.startswith("Top"):
      
      if '+' in sampler: # Composite sampler
        sampler, _ = sampler.split('+', 2)
      prefix, param = sampler.split("-", 2)
      
      dataset = PreSampleWeightedDataset(
        dataset=dataset,
        weights={
            "RW": RW, # param: temperature
            "AW": AW, # param: temperature
            "RewW": RewW, # param: temperature
            "Top": TopK, # param: percentage
          }[prefix](split_dataset_into_trajectories(dataset), float(param)),
        n_batches=max_steps, 
        batch_size=batch_size
      )
    elif sampler.startswith("DW+"): # DW+AW-{ckpt}_{temp}_{b}_{p}
      prefix, param = sampler.split("-", 2)
      
      sampler1, sampler2 = prefix.split("+")
      ckpt_idx, temp, beta, p = param.split("_")

      weights1 = load_sample_weights(env_name, int(ckpt_idx))
      weights1 = (weights1)**(float(beta))
      weights1 /= weights1.sum() # 

      weights2 = {
        "RW": RW, # param: temperature
        "AW": AW, # param: temperature
        "RewW": RewW, # param: temperature
        "TopK": TopK, # param: percentage
      }[sampler2](split_dataset_into_trajectories(dataset), float(temp))

      dataset = PreSampleWeightedDataset(
        dataset=dataset,
        weights=(weights1 * float(p) + weights2 * (1 - float(p))),
        n_batches=max_steps, 
        batch_size=batch_size
      )
    elif sampler.startswith("DW"):
      _, param = sampler.split("-", 2)
      weights = load_sample_weights(env_name, int(param))
      dataset = PreSampleWeightedDataset(
        dataset=dataset,
        weights=weights / weights.sum(),
        n_batches=max_steps,
        batch_size=batch_size
      )
    elif sampler.startswith("SoftDW"):
      from scipy.special import softmax
      def _normalize(w, norm):
        if norm == "maxmin":
          return (w - w.min()) / (w.max() - w.min())
        elif norm == "none":
          return w
        else:
          raise NotImplemented
      _, param = sampler.split("-", 2)
      ckpt_idx, norm, temp = param.split("_")
      weights = load_sample_weights(env_name, int(ckpt_idx))
      dataset = PreSampleWeightedDataset(
        dataset=dataset,
        weights=softmax(_normalize(weights, norm) / float(temp)),
        n_batches=max_steps,
        batch_size=batch_size
      )
    elif sampler.startswith("PowerDW"):
      _, param = sampler.split("-", 2)
      ckpt_idx, beta = param.split("_")
      weights = load_sample_weights(env_name, int(ckpt_idx))
      weights = weights**(float(beta))
      dataset = PreSampleWeightedDataset(
        dataset=dataset,
        weights=weights / weights.sum(),
        n_batches=max_steps,
        batch_size=batch_size
      )
    else:
      raise NotImplemented()
    return dataset


class ReplayBuffer(Dataset):
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
                 capacity: int):

        observations = np.empty((capacity, *observation_space.shape),
                                dtype=observation_space.dtype)
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity, ), dtype=np.float32)
        masks = np.empty((capacity, ), dtype=np.float32)
        dones_float = np.empty((capacity, ), dtype=np.float32)
        next_observations = np.empty((capacity, *observation_space.shape),
                                     dtype=observation_space.dtype)
        super().__init__(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         masks=masks,
                         dones_float=dones_float,
                         next_observations=next_observations,
                         size=0)

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(self, dataset: Dataset,
                                num_samples: Optional[int]):
        assert self.insert_index == 0, 'Can insert a batch online in an empty replay buffer.'

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert self.capacity >= num_samples, 'Dataset cannot be larger than the replay buffer capacity.'

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[
            indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: np.ndarray):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)



def normalize(dataset):

    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


def make_env_and_dataset(env_name: str,
                         seed: int,
                         sampler: str,
                         max_steps: int,
                         batch_size: int):
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env)

    if 'antmaze' in env_name:
        dataset.rewards -= 1.0
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0
    elif ('halfcheetah' in env_name or 'walker2d' in env_name
          or 'hopper' in env_name):
        normalize(dataset)   

    return env, wrap_dataset(env_name, dataset, sampler, max_steps, batch_size)