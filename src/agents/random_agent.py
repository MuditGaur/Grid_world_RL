"""
Random agent that selects actions from a Dirichlet-parameterized distribution.

The agent exposes the same API as other agents in this package:
- reset(env)
- act(env) -> int action in [0..4]

Dirichlet concentration parameters can be fixed or state-conditioned via a
callable. By default, it uses a symmetric Dirichlet over 5 actions.
"""

from typing import Callable, Optional, Sequence
import numpy as np

from ..environment import Gridworld, STAY


class RandomAgent:
    """Agent that samples actions from a Dirichlet distribution.

    Parameters
    ----------
    alpha : float | Sequence[float]
        Dirichlet concentration(s). If float, uses symmetric [alpha]*5.
    alpha_fn : Optional[Callable[[Gridworld], Sequence[float]]]
        Optional callable returning per-action concentration given the env.
        If provided, this overrides `alpha` each step.
    prefer_stay_bias : float
        Added to the STAY action's concentration to control idleness tendency.
    rng : Optional[np.random.Generator]
        Random generator for reproducibility.
    """

    def __init__(self,
                 alpha: float | Sequence[float] = 1.0,
                 alpha_fn: Optional[Callable[[Gridworld], Sequence[float]]] = None,
                 prefer_stay_bias: float = 0.0,
                 rng: Optional[np.random.Generator] = None):
        if isinstance(alpha, (int, float)):
            base_alpha = np.full(5, float(alpha), dtype=np.float32)
        else:
            arr = np.asarray(alpha, dtype=np.float32)
            assert arr.shape == (5,), "alpha must be length-5 for 5 actions"
            base_alpha = arr
        self.base_alpha = base_alpha
        self.alpha_fn = alpha_fn
        self.prefer_stay_bias = float(prefer_stay_bias)
        self.rng = rng or np.random.default_rng()
        self.last_action = STAY

    def reset(self, env: Gridworld):
        self.last_action = STAY

    def _alphas(self, env: Gridworld) -> np.ndarray:
        if self.alpha_fn is not None:
            al = np.asarray(self.alpha_fn(env), dtype=np.float32)
            assert al.shape == (5,), "alpha_fn must return length-5 sequence"
        else:
            al = self.base_alpha.copy()
        # Bias STAY if configured
        al = al.astype(np.float32)
        al[STAY] = max(1e-4, al[STAY] + self.prefer_stay_bias)
        # Ensure strictly positive concentrations
        al = np.clip(al, 1e-4, None)
        return al

    def act(self, env: Gridworld) -> int:
        al = self._alphas(env)
        probs = self.rng.dirichlet(al)
        a = int(self.rng.choice(5, p=probs))
        self.last_action = a
        return a


