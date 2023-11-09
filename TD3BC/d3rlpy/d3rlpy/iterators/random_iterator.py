from typing import List, cast, Optional

import numpy as np

from ..dataset import Transition
from .base import TransitionIterator


class RandomIterator(TransitionIterator):

    _n_steps_per_epoch: int

    def __init__(
        self,
        transitions: List[Transition],
        n_steps_per_epoch: int,
        batch_size: int,
        n_steps: int = 1,
        gamma: float = 0.99,
        n_frames: int = 1,
        real_ratio: float = 1.0,
        generated_maxlen: int = 100000,
        sample_weights: Optional[List[float]] = None,
    ):
        super().__init__(
            transitions=transitions,
            batch_size=batch_size,
            n_steps=n_steps,
            gamma=gamma,
            n_frames=n_frames,
            real_ratio=real_ratio,
            generated_maxlen=generated_maxlen,
        )
        self._n_steps_per_epoch = n_steps_per_epoch
        self._n_transitions_per_epoch = self._batch_size * n_steps_per_epoch
        self._sample_weights = sample_weights       
        # Weights has to be probabilities
        if sample_weights is not None:
            assert np.isclose(np.sum(self._sample_weights), 1.0)
        self._indices_cache = None
        self._indices_count = 0

    def _reset(self) -> None:
        if self._sample_weights is not None:
            self._indices_cache = np.random.choice(
                range(len(self._transitions)), 
                self._n_transitions_per_epoch, 
                p=self._sample_weights)
            self._indices_count = 0

    def _next(self) -> Transition:
        if self._indices_cache is None:           
            index = cast(int, np.random.randint(len(self._transitions)))
        else:
            index = cast(int, self._indices_cache[self._indices_count])
            self._indices_count = (self._indices_count + 1) % self._n_transitions_per_epoch
            
        transition = self._transitions[index]
        return transition

    def _has_finished(self) -> bool:
        return self._count >= self._n_steps_per_epoch

    def __len__(self) -> int:
        return self._n_steps_per_epoch
