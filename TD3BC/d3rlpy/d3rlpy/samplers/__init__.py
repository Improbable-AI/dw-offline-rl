from typing import List, Tuple
import numpy as np
import functools

from scipy.special import softmax
from sklearn.linear_model import LinearRegression

from ..dataset import Episode, Transition

# No longer use optdice wieghts
# from ddos.optdice.sampling_util import load_sample_weights

def get_return(traj: Episode):
  return traj.rewards.sum()

def AW(env_name, trajs: List[Episode], temp: float):
  temp = float(temp)
  ep_rets = np.asarray(list(map(get_return, trajs)))
  ep_lens =  np.asarray(list(map(len, trajs)))
  s0s = np.array([traj.observations[0] for traj in trajs])
  v = LinearRegression().fit(s0s, ep_rets).predict(s0s)
  weights = np.asarray(functools.reduce(lambda a, b: a + b,
              [[w] * l for w, l in zip((ep_rets - v), ep_lens)]))
  weights = (weights - weights.min()) / (weights.max() - weights.min())
  return np.asarray(softmax(weights / temp))

def RW(env_name, trajs: List[Episode], temp: str):
  temp = float(temp)
  ep_rets = np.asarray(list(map(get_return, trajs)))
  ep_rets = (ep_rets - ep_rets.min()) / (ep_rets.max() - ep_rets.min())
  ep_lens =  np.asarray(list(map(len, trajs)))
  weights =  np.asarray(functools.reduce(lambda a, b: a + b,
              [[w] * l for w, l in zip(ep_rets, ep_lens)]))
  return np.asarray(softmax(weights / temp))

def TopK(env_name, trajs, p):
  p = float(p)
  ep_rets = np.asarray(list(map(get_return, trajs)))
  ep_lens =  np.asarray(list(map(len, trajs)))
  sorted_indices = np.argsort(ep_rets)[::-1]
  top_indices = sorted_indices[:int(len(trajs) * p)]
  weights = np.asarray(functools.reduce(lambda a, b: a + b,
              [[float(i in top_indices)] * l for i, l in enumerate(ep_lens)]))
  return weights / weights.sum()

def RewW(env_name, trajs: List[Episode], temp: float):
  return np.asarray(functools.reduce(lambda a, b: a + b,
                [[t.reward for t in traj] for traj in trajs]))

# def PowerDW(env_name, trajs, param: str):
#   ckpt_idx, beta = param.split("_")
#   weights = load_sample_weights(env_name, int(ckpt_idx))
#   weights = weights**(float(beta))
#   return weights / weights.sum()

def generate_sample_weights_from_trajectories(env_name, sampler_name: str, trajs: List[Episode]):
  if '+' in sampler_name: # Composite sampler
    sampler_name, _ = sampler_name.split('+', 2)
  if sampler_name == "uniform":
    return None # None means no using sample_weights in algo  
  sampler_cls, sampler_args = sampler_name.split("-", 2)
  print(f"Use sampler: {sampler_cls}")
  return { 
      "RW": RW, # param: temperature
      "AW": AW, # param: temperature
      "RewW": RewW, # param: temperature
      "Top": TopK, # param: percentage
      # "PowerDW": PowerDW, # param: beta
    }[sampler_cls](env_name, trajs, sampler_args)
  
