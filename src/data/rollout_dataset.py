"""
Dataset generation for Theory of Mind training.

This module constructs supervised learning samples for the ToM observer by
rolling out agents in the `Gridworld` environment and packaging the data into
PyTorch tensors. For each training example (taken from the query episode), we
produce five items:

- character: 24-D summary of k context episodes (action histogram + mean full-state features)
- mental: 17-D mean of recent m steps' full-state features (short-term context)
- state: either 17-D observer full-state features, or flattened agent partial observation if requested
- label: next action (int in [0..4])
- swap_hidden: 1.0 if the most recent swap was hidden from the agent (false-belief bucket), else 0.0

Use `build_rollouts(...)` to synthesize (train_samples, val_samples) lists and
`RolloutDataset` to obtain DataLoader-ready tensors. Set
`use_agent_obs_as_state=True` to use the agent's egocentric FOV as the state
input instead of the 17-D observer features (used by the state-encoder comparison).
"""

import random
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from ..environment import Gridworld, StepRec, STAY
from ..agents import BeliefAgent, QLearnAgent, RandomAgent

class RolloutDataset(Dataset):
    """Build context/query samples for ToM-style observer training.

    For each actor (agent_id), we create `k_context` episodes and 1 query episode.
    For each step in the query episode, we produce a training example containing:
      - character trajectory (compact summary of past episodes)
      - mental (recent-step) trajectory from current episode
      - current state features (observer 17-D or flattened agent FOV) -> label next action
      - swap_hidden flag (for evaluation buckets)
    """
    def __init__(self, samples, device: str = "cpu"):
        self.samples = samples
        self.device = device

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # Convert to tensors
        char = torch.tensor(s["char"], dtype=torch.float32)
        mental = torch.tensor(s["mental"], dtype=torch.float32)
        state = torch.tensor(s["state"], dtype=torch.float32)
        label = torch.tensor(s["action"], dtype=torch.long)
        swap_hidden = torch.tensor([1.0 if s["swap_hidden"] else 0.0], dtype=torch.float32)
        return char, mental, state, label, swap_hidden


def build_rollouts(num_agents: int, episodes_per_agent: int, k_context: int,
                   grid=9, fov=3, use_rb_agents=True, use_qlearn_agents=False,
                   use_random_agents: bool = False,
                   random_alpha: float | list[float] = 1.0,
                   qlearn_iters=10000, max_steps=80, seed=1234,
                   p_swap: float = 0.25, swap_window: int = 10,
                   force_swap: bool = False, force_swap_at_step: int | None = None,
                   use_agent_obs_as_state: bool = False) -> Tuple[List, List]:
    rng = random.Random(seed)
    env = Gridworld(n=grid, fov=fov, p_wall=0.10, p_swap=p_swap, max_steps=max_steps, rng=rng)

    agents = []
    kinds = [0,1,2,3]
    # Construct agent population
    for i in range(num_agents):
        pref = kinds[i % 4]
        if use_rb_agents:
            agents.append((f"RB_{i}", BeliefAgent(pref, fov=fov, n=grid)))
        if use_qlearn_agents:
            agents.append((f"QL_{i}", QLearnAgent(pref, n=grid, fov=fov)))
        if use_random_agents:
            # Random agents do not use preferred_kind functionally, but keep naming parity
            agents.append((f"RN_{i}", RandomAgent(alpha=random_alpha)))

    # optional: pre-train Q-learning agents
    if use_qlearn_agents:
        for name, ag in agents:
            if isinstance(ag, QLearnAgent):
                for _ in range(qlearn_iters):
                    ag.train_episode(env, iters=max_steps)

    # Helper: summarize a trajectory into a fixed-size vector for character net
    def summarize_episode(ep_steps: List[StepRec]) -> np.ndarray:
        # statistics over actions and relative object vectors
        acts = np.array([s.action for s in ep_steps], dtype=np.int64)
        hist = np.bincount(acts, minlength=5).astype(np.float32)
        hist = hist/ (hist.sum()+1e-6)
        # mean of full-state features
        mean_full = np.mean(np.stack([s.full_state_feat for s in ep_steps], axis=0), axis=0)
        return np.concatenate([hist, mean_full], axis=0)  # size = 5 + 19 = 24

    # Collect episodes and build training tuples
    train_samples = []
    val_samples = []

    def run_episode(agent) -> List[StepRec]:
        env.reset()
        agent.reset(env)
        steps: List[StepRec] = []
        last_action = STAY
        hidden_since = False
        last_swap_seen = True
        r = env.fov//2
        for t in range(env.max_steps):
            # Optionally force a swap event if the agent hasn't triggered one by reaching subgoal
            if (force_swap and (not env.swapped)):
                trigger_step = force_swap_at_step if force_swap_at_step is not None else (env.max_steps//3)
                if t == trigger_step and env.agent_pos != env.subgoal_pos:
                    # Determine if swap is visible at this time step
                    ax,ay = env.agent_pos
                    vis_any = False
                    for o in env.objects:
                        if abs(o.pos[0]-ax) <= r and abs(o.pos[1]-ay) <= r:
                            vis_any = True
                            break
                    last_swap_seen = vis_any
                    # Perform swap by rotating object positions
                    before_positions = [o.pos for o in env.objects]
                    new_positions = before_positions[1:] + before_positions[:1]
                    for o, np_ in zip(env.objects, new_positions):
                        o.pos = np_
                    env.swapped = True
                    env._last_swap_step = t
            # visibility of objects before step
            # we approximate swap-hidden by checking if any object moved outside FOV since last subgoal
            # Mark swap_hidden for a configurable window of steps after swap if agent could not see moved objects at swap time
            swap_hidden_flag = (env.swapped and (t - env._last_swap_step) <= swap_window)
            # Check if swap cells were visible at swap time: approximate by whether agent was within FOV radius of any object positions before swap; keep heuristic flag
            if env.swapped and t == env._last_swap_step:
                ax,ay = env.agent_pos
                vis_any = False
                for o in env.objects:
                    if abs(o.pos[0]-ax)<=r and abs(o.pos[1]-ay)<=r:
                        vis_any = True
                        break
                last_swap_seen = vis_any
            if env.swapped and (t - env._last_swap_step) <= swap_window:
                swap_hidden_flag = not last_swap_seen

            full_feat = env.full_state_features(last_action)
            partial = env._get_partial_obs()
            steps.append(StepRec(full_state_feat=full_feat, partial_obs=partial,
                                  action=last_action, agent_pos=env.agent_pos,
                                  swap_hidden=swap_hidden_flag, t=t))
            a = agent.act(env)
            out = env.step(a)
            last_action = a
            if out["done"]:
                break
        return steps

    for idx,(name, agent) in enumerate(agents):
        # Gather episodes
        episodes = [run_episode(agent) for _ in range(episodes_per_agent)]
        # Split into context and query episodes in blocks
        for j in range(0, episodes_per_agent, k_context+1):
            if j + k_context >= episodes_per_agent:
                break
            ctx_eps = episodes[j:j+k_context]
            qry_ep = episodes[j+k_context]
            # summarize contexts
            ctx_summ = np.stack([summarize_episode(ep) for ep in ctx_eps], axis=0)  # (k, 24)
            # For each step in query, create sample with mental window of last m steps
            m = 6
            for t in range(min(len(qry_ep)-1, env.max_steps-1)):
                mental_window = qry_ep[max(0,t-m):t+1]
                mental = np.mean(np.stack([s.full_state_feat for s in mental_window], axis=0), axis=0)  # (19,)
                # Choose state representation for the model input
                if use_agent_obs_as_state:
                    # Flatten agent partial observation (C, H, W) -> (C*H*W)
                    po = qry_ep[t].partial_obs
                    state_now = po.reshape(-1).astype(np.float32)
                else:
                    # Observer 17-D compact full-state features
                    state_now = qry_ep[t].full_state_feat
                action_next = qry_ep[t+1].action
                sample = {
                    "char": ctx_summ.mean(axis=0),   # (24,)
                    "mental": mental,               # (19,)
                    "state": state_now,             # (19,)
                    "action": action_next,          # int 0..4
                    "swap_hidden": qry_ep[t].swap_hidden,
                }
                # 80/20 split
                if (idx + t) % 5 == 0:
                    val_samples.append(sample)
                else:
                    train_samples.append(sample)

    return train_samples, val_samples
