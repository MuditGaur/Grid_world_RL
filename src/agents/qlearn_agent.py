"""
Q-learning agent for gridworld environments.

This agent uses tabular Q-learning with discretized state representation
to learn policies in partially observable gridworld environments.
"""

import random
import numpy as np
from typing import Dict, Tuple
from ..environment import Gridworld, STAY, UP, DOWN, LEFT, RIGHT, ACTIONS

class QLearnAgent:
    """Tiny tabular Q-learning agent with partial observability.
    For tractability, the state is *discretized* to a coarse code:
    (dx_sign_to_pref_obj, dy_sign_to_pref_obj, wall_front, wall_left, wall_right)
    This is intentionally simplistic but yields RL-like data.
    """
    def __init__(self, preferred_kind: int, n: int, fov: int,
                 alpha=0.3, gamma=0.95, eps=0.2):
        self.pref = preferred_kind
        self.n = n
        self.fov = fov
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.Q: Dict[Tuple, np.ndarray] = {}
        self.last_action = STAY

    def _state_code(self, env: Gridworld) -> Tuple:
        # Use believed location as in BeliefAgent to define coarse deltas
        # If unseen, default deltas to 0
        # Also add 3 wall sensors (front/left/right) relative to last action dir
        # Determine belief (nearest visible of preferred kind)
        # For simplicity, look in FOV; if not visible, use last known (None -> (0,0))
        r = env.fov//2
        ax,ay = env.agent_pos
        target = None
        for o in env.objects:
            if o.kind==self.pref and abs(o.pos[0]-ax)<=r and abs(o.pos[1]-ay)<=r:
                target = o.pos
                break
        dx = 0 if target is None else np.sign((target[0]-ax))
        dy = 0 if target is None else np.sign((target[1]-ay))
        # wall sensors relative to facing given last action
        facing = self.last_action
        dirs = {
            UP:    [UP, LEFT, RIGHT],
            DOWN:  [DOWN, RIGHT, LEFT],
            LEFT:  [LEFT, DOWN, UP],
            RIGHT: [RIGHT, UP, DOWN],
            STAY:  [UP, LEFT, RIGHT],
        }
        front,left,right = dirs[facing]
        def blocked(a):
            dx,dy = ACTIONS[a]
            nx,ny = ax+dx, ay+dy
            return 1 if not env._valid((nx,ny)) else 0
        sensors = (blocked(front), blocked(left), blocked(right))
        return (int(dx), int(dy), sensors[0], sensors[1], sensors[2])

    def _get_Q(self, s: Tuple) -> np.ndarray:
        if s not in self.Q:
            self.Q[s] = np.zeros(5, dtype=np.float32)
        return self.Q[s]

    def train_episode(self, env: Gridworld, iters: int = 120):
        self.last_action = STAY
        env.reset()
        total_r = 0.0
        for t in range(iters):
            s = self._state_code(env)
            if random.random() < self.eps:
                a = random.randrange(5)
            else:
                a = int(np.argmax(self._get_Q(s)))
            out = env.step(a)
            # reward: +1 if on preferred object, +0.2 if closer to it, small step cost
            r = -0.01
            # distance change heuristic toward preferred object
            ax,ay = env.agent_pos
            for o in env.objects:
                if o.kind==self.pref:
                    dist = abs(o.pos[0]-ax)+abs(o.pos[1]-ay)
                    break
            # estimated previous position (undo action)
            dx,dy = ACTIONS[a]
            px,py = ax-dx, ay-dy
            prev_dist = abs(o.pos[0]-px)+abs(o.pos[1]-py)
            if dist < prev_dist:
                r += 0.2
            if (ax,ay) == o.pos:
                r += 1.0
            total_r += r
            s2 = self._state_code(env)
            q = self._get_Q(s)
            q2 = self._get_Q(s2)
            q[a] = (1-self.alpha)*q[a] + self.alpha*(r + self.gamma*np.max(q2))
            self.last_action = a
            if out["done"]:
                break
        return total_r

    def reset(self, env: Gridworld):
        self.last_action = STAY

    def act(self, env: Gridworld) -> int:
        s = self._state_code(env)
        q = self._get_Q(s)
        a = int(np.argmax(q))
        self.last_action = a
        return a
