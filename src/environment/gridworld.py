"""
Gridworld POMDP with hidden swaps for Theory of Mind experiments.

This module implements a partially observable gridworld environment where
objects can be swapped outside the agent's field of view, creating
false-belief scenarios for ToM evaluation.
"""

import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

# Action definitions
Action = int  # 0:UP,1:DOWN,2:LEFT,3:RIGHT,4:STAY
ACTIONS = [(0,-1), (0,1), (-1,0), (1,0), (0,0)]
UP, DOWN, LEFT, RIGHT, STAY = 0,1,2,3,4

@dataclass
class Obj:
    kind: int  # 0..3
    pos: Tuple[int,int]

@dataclass
class StepRec:
    full_state_feat: np.ndarray  # features for observer from full state
    partial_obs: np.ndarray      # agent's egocentric partial observation (FOV)
    action: int
    agent_pos: Tuple[int,int]
    swap_hidden: bool            # whether a recent swap was hidden from agent
    t: int

class Gridworld:
    """Maze with four collectible objects and a subgoal. After subgoal is
    touched, objects may be swapped (permuted) with probability p_swap. If the
    agent's FOV does not cover the swap, it forms a *false belief*.

    The observer (our ToM model) sees full state; the actor sees only a
    FOV-square around its position (partial observability).
    """
    def __init__(self, n: int = 9, fov: int = 3, p_wall: float = 0.1,
                 p_swap: float = 0.25, max_steps: int = 120, rng: Optional[random.Random]=None):
        assert fov % 2 == 1, "FOV must be odd"
        self.n = n
        self.fov = fov
        self.p_wall = p_wall
        self.p_swap = p_swap
        self.max_steps = max_steps
        self.rng = rng or random.Random()

        # Precompute wall grid and spawn points per episode in reset()
        self.grid = None
        self.agent_pos = None
        self.objects: List[Obj] = []
        self.subgoal_pos = None
        self.t = 0
        self.swapped = False
        self._last_swap_step = -999

    def _rand_empty_cell(self) -> Tuple[int,int]:
        while True:
            x = self.rng.randrange(self.n)
            y = self.rng.randrange(self.n)
            if self.grid[y, x] == 0:
                # avoid placing on same cell as subgoal or agent
                return (x, y)

    def reset(self) -> Dict:
        # 0 empty, 1 wall
        self.grid = np.zeros((self.n, self.n), dtype=np.int8)
        for y in range(self.n):
            for x in range(self.n):
                if x==0 or y==0 or x==self.n-1 or y==self.n-1:
                    self.grid[y,x] = 1
                elif self.rng.random() < self.p_wall:
                    self.grid[y,x] = 1

        # Place subgoal, agent, objects
        self.subgoal_pos = self._rand_empty_cell()
        self.agent_pos = self._rand_empty_cell()
        self.objects = []
        taken = {self.subgoal_pos, self.agent_pos}
        for k in range(4):
            while True:
                pos = self._rand_empty_cell()
                if pos not in taken:
                    taken.add(pos)
                    self.objects.append(Obj(kind=k, pos=pos))
                    break

        self.t = 0
        self.swapped = False
        self._last_swap_step = -999
        obs = self._get_partial_obs()
        return {"obs": obs, "pos": self.agent_pos}

    def _get_partial_obs(self) -> np.ndarray:
        r = self.fov//2
        x0 = self.agent_pos[0]-r
        y0 = self.agent_pos[1]-r
        patch = np.ones((self.fov, self.fov), dtype=np.int8)  # 1=wall/pad
        for dy in range(self.fov):
            for dx in range(self.fov):
                x = x0+dx
                y = y0+dy
                if 0 <= x < self.n and 0 <= y < self.n:
                    patch[dy,dx] = self.grid[y,x]
        # Add channels for subgoal and objects (binary masks in FOV)
        ch = []
        def mask_for(pos):
            mx = np.zeros_like(patch)
            dx = pos[0] - x0
            dy = pos[1] - y0
            if 0 <= dx < self.fov and 0 <= dy < self.fov:
                mx[dy,dx] = 1
            return mx
        ch.append(mask_for(self.subgoal_pos))
        for o in self.objects:
            ch.append(mask_for(o.pos))
        # shape: (5, fov, fov)
        return np.stack([patch] + ch, axis=0).astype(np.float32)

    def _valid(self, pos: Tuple[int,int]) -> bool:
        x,y = pos
        if x<0 or y<0 or x>=self.n or y>=self.n: return False
        return self.grid[y,x] == 0

    def step(self, action: Action) -> Dict:
        dx,dy = ACTIONS[action]
        nx, ny = self.agent_pos[0]+dx, self.agent_pos[1]+dy
        if self._valid((nx,ny)):
            self.agent_pos = (nx,ny)
        self.t += 1

        # Swap with probability p after subgoal visited (toggle once)
        if (not self.swapped) and self.agent_pos == self.subgoal_pos and self.rng.random() < self.p_swap:
            # Determine if swap is visible: any object outside FOV becomes hidden-swap if moved
            before = [o.pos for o in self.objects]
            self.rng.shuffle(self.objects)
            # permute positions
            new_positions = before[1:]+before[:1]
            for o, np_ in zip(self.objects, new_positions):
                o.pos = np_
            self.swapped = True
            self._last_swap_step = self.t

        done = self.t >= self.max_steps
        obs = self._get_partial_obs()
        return {"obs": obs, "pos": self.agent_pos, "done": done}

    # --------------------
    # Observer features
    # --------------------
    def full_state_features(self, last_action: int = STAY) -> np.ndarray:
        """Compact full-state features for the *observer* (not seen by the actor):
        - Agent (x,y) normalized
        - Subgoal relative vector
        - For 4 objects: relative (dx,dy) vectors
        - Last action one-hot (5)
        -> Vector size = 2 + 2 + 4*2 + 5 = 17
        """
        ax, ay = self.agent_pos
        axn = (ax/(self.n-1))*2-1
        ayn = (ay/(self.n-1))*2-1
        def rel(pos):
            dx = (pos[0]-ax)/(self.n-1)
            dy = (pos[1]-ay)/(self.n-1)
            return [dx,dy]
        feat = [axn, ayn]
        feat += rel(self.subgoal_pos)
        for o in self.objects:
            feat += rel(o.pos)
        la = [0.0]*5
        la[last_action] = 1.0
        feat += la
        return np.array(feat, dtype=np.float32)
