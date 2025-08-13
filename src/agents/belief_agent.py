"""
Rule-based belief agent for gridworld environments.

This agent maintains persistent beliefs about object locations and updates
them only when objects are within its field of view, creating false beliefs
when swaps occur outside the FOV.
"""

import random
from typing import Dict, Tuple
from ..environment import Gridworld, STAY, UP, DOWN, LEFT, RIGHT, ACTIONS

class BeliefAgent:
    """Rule-based agent with persistent beliefs about object locations.
    Prefers a specific object kind; updates beliefs when in FOV, otherwise
    continues to pursue last believed location. This creates *false beliefs*
    when swaps happen outside FOV.
    """
    def __init__(self, preferred_kind: int, fov: int, n: int):
        self.pref = preferred_kind
        self.fov = fov
        self.n = n
        self.beliefs: Dict[int, Tuple[int,int]] = {}
        self.last_action = STAY

    def reset(self, env: Gridworld):
        # initialize beliefs from first partial observation
        self.beliefs = {}
        self.last_action = STAY
        self._update_beliefs(env)

    def _update_beliefs(self, env: Gridworld):
        # If an object is inside FOV, update its believed position
        r = env.fov//2
        x0 = env.agent_pos[0]-r
        y0 = env.agent_pos[1]-r
        for idx,o in enumerate(env.objects):
            if abs(o.pos[0]-env.agent_pos[0])<=r and abs(o.pos[1]-env.agent_pos[1])<=r:
                self.beliefs[o.kind] = o.pos

    def act(self, env: Gridworld) -> int:
        self._update_beliefs(env)
        # Greedy move toward believed target (preferred kind). If unknown, wander.
        target = self.beliefs.get(self.pref, None)
        if target is None:
            # simple exploration: rotate right-hand wall-following-ish
            for a in [RIGHT, DOWN, LEFT, UP, STAY]:
                dx,dy = ACTIONS[a]
                nx,ny = env.agent_pos[0]+dx, env.agent_pos[1]+dy
                if env._valid((nx,ny)):
                    self.last_action = a
                    return a
            self.last_action = STAY
            return STAY
        # Move to minimize Manhattan distance
        ax,ay = env.agent_pos
        tx,ty = target
        candidates = []
        for a in [UP, DOWN, LEFT, RIGHT, STAY]:
            dx,dy = ACTIONS[a]
            nx,ny = ax+dx, ay+dy
            if env._valid((nx,ny)):
                dist = abs(tx-nx)+abs(ty-ny)
                candidates.append((dist, a))
        candidates.sort(key=lambda x: (x[0], random.random()))
        act = candidates[0][1]
        self.last_action = act
        return act
