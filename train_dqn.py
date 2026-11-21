"""
DQN agent for Gymnasium CarRacing-v3 (discrete actions), built for a multi-core
GPU server: many CarRacing copies run in parallel processes to collect
experience, while the GPU trains on large batches.

  Train (or resume):   python train_dqn.py --run-name baseline --hours 48
  Record a video:      python train_dqn.py --play runs/baseline/best.pt --record videos/
  Watch in a window:   python train_dqn.py --play runs/baseline/best.pt   (needs a display)

Everything for a run lives in runs/<run-name>/:
  config.json  hyperparameters used
  train.csv    one row per training episode
  eval.csv     greedy evaluation score every --eval-every transitions
  last.pt      full checkpoint; re-running the same command resumes from it
  best.pt      weights with the best evaluation score so far

Ctrl+C (or `kill <pid>`) saves a checkpoint before exiting.

Actions: 0 = nothing, 1 = left, 2 = right, 3 = gas, 4 = brake
"""

import argparse
import csv
import json
import os
import random
import signal
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EVAL_SEED_BASE = 1_000_000_000  # fixed evaluation tracks, so scores are comparable

def parse_args():
    p = argparse.ArgumentParser(description="DQN for CarRacing-v3")
    p.add_argument("--run-name", default="baseline")
    p.add_argument("--play", metavar="CHECKPOINT", help="run a trained agent instead of training")
    p.add_argument("--record", metavar="DIR", help="with --play: save mp4 videos here instead of opening a window")
    p.add_argument("--play-episodes", type=int, default=3)
    # parallelism
    p.add_argument("--num-envs", type=int, default=16, help="parallel CarRacing processes (leave a few cores free)")
    p.add_argument("--threads", type=int, default=2, help="CPU threads for PyTorch in the main process")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    # run length (counted in transitions = agent decisions summed over all envs)
    p.add_argument("--hours", type=float, default=None, help="stop after this much total training time")
    p.add_argument("--total-steps", type=int, default=20_000_000)
    # DQN hyperparameters
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--train-freq", type=float, default=4,
                   help="one gradient update per this many new transitions")
    p.add_argument("--buffer-size", type=int, default=100_000, help="~35 MB of RAM per 1,000")
    p.add_argument("--learning-starts", type=int, default=20_000)
    p.add_argument("--target-update-every", type=int, default=5_000, help="in gradient updates")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=500_000)
    # environment
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--max-negative-steps", type=int, default=50)
    # bookkeeping
    p.add_argument("--eval-every", type=int, default=100_000)
    p.add_argument("--eval-episodes", type=int, default=8, help="run in parallel")
    p.add_argument("--checkpoint-every", type=int, default=100_000)
    p.add_argument("--print-every", type=int, default=20_000)
    return p.parse_args()


class CarRacingWrapper(gym.Wrapper):
    def __init__(self, env, frame_skip=4, frame_stack=4, max_negative_steps=50, early_stop=True):
        super().__init__(env)
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.max_negative_steps = max_negative_steps
        self.early_stop = early_stop
        self.frames = deque(maxlen=frame_stack)
        self.negative_steps = 0
        self.observation_space = gym.spaces.Box(0, 255, (frame_stack, 84, 84), np.uint8)

    @staticmethod
    def preprocess(obs):
        # 96x96x3 RGB -> 84x84 grayscale; drops the HUD bar at the bottom.
        gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
        return gray[:84, 6:90].astype(np.uint8)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(50):  # skip the camera zoom-in at the start
            obs, _, _, _, _ = self.env.step(0)
        frame = self.preprocess(obs)
        for _ in range(self.frame_stack):
            self.frames.append(frame)
        self.negative_steps = 0
        return np.stack(self.frames), info

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(int(action))
            total_reward += reward
            if terminated or truncated:
                break
        self.frames.append(self.preprocess(obs))

        # End early if the car keeps losing reward (usually stuck on grass).
        self.negative_steps = self.negative_steps + 1 if total_reward < 0 else 0
        if self.early_stop and self.negative_steps >= self.max_negative_steps:
            truncated = True
        return np.stack(self.frames), total_reward, terminated, truncated, info


def env_factory(args, early_stop=True, render_mode=None, video_dir=None, worker=True):
    frame_skip, frame_stack, max_neg = args.frame_skip, args.frame_stack, args.max_negative_steps

    def make():
        if worker:
            # Ctrl+C goes to the main process only; it shuts the workers down cleanly.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        env = gym.make("CarRacing-v3", continuous=False, render_mode=render_mode)
        if video_dir:
            env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda e: True,
                                           disable_logger=True)
        return CarRacingWrapper(env, frame_skip, frame_stack, max_neg, early_stop)

    return make

class QNetwork(nn.Module):
    def __init__(self, in_channels, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
                                  nn.Linear(512, n_actions))

    def forward(self, x):
        return self.head(self.conv(x.float() / 255.0))


# ---------------------------------------------------------------------------
# Replay buffer. Stores each state stack plus only the *new* frame of the next
# state (the rest of the next state is the current stack shifted by one).
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity, stack, h=84, w=84):
        self.capacity = capacity
        self.states = np.zeros((capacity, stack, h, w), dtype=np.uint8)
        self.next_frames = np.zeros((capacity, h, w), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add_batch(self, s, a, r, s2, done):
        n = len(a)
        i = (self.idx + np.arange(n)) % self.capacity
        self.states[i] = s
        self.next_frames[i] = s2[:, -1]
        self.actions[i] = a
        self.rewards[i] = r
        self.dones[i] = done
        self.idx = (self.idx + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size, device):
        i = np.random.randint(0, self.size, size=batch_size)
        s = self.states[i]
        s2 = np.concatenate([s[:, 1:], self.next_frames[i][:, None]], axis=1)
        t = lambda x: torch.from_numpy(x).to(device, non_blocking=True)
        return t(s), t(self.actions[i]), t(self.rewards[i]), t(s2), t(self.dones[i])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_device(choice):
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def epsilon_at(step, args):
    frac = min(step / args.eps_decay_steps, 1.0)
    return args.eps_start + frac * (args.eps_end - args.eps_start)


def select_actions(net, obs, epsilon, n_actions, device):
    """Epsilon-greedy actions for a batch of observations (one per env)."""
    n = len(obs)
    if epsilon >= 1.0:
        return np.random.randint(0, n_actions, size=n)
    with torch.no_grad():
        q = net(torch.from_numpy(obs).to(device))
        actions = q.argmax(dim=1).cpu().numpy()
    explore = np.random.random(n) < epsilon
    actions[explore] = np.random.randint(0, n_actions, size=explore.sum())
    return actions


def atomic_save(obj, path):
    # Write to a temp file then rename, so a crash mid-save can't corrupt it.
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


class CsvLog:
    def __init__(self, path, header, resume_step=None):
        if resume_step is not None and os.path.exists(path):
            # Drop rows logged after the checkpoint we are resuming from.
            with open(path, newline="") as f:
                rows = list(csv.reader(f))
            keep = [rows[0]] + [r for r in rows[1:] if r and int(r[0]) <= resume_step]
            with open(path, "w", newline="") as f:
                csv.writer(f).writerows(keep)
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(header)
            self.f.flush()

    def write(self, row):
        self.w.writerow(row)
        self.f.flush()


def evaluate(net, eval_envs, n_episodes, n_actions, device):
    """Greedy play (epsilon 0, no early stop) on the same fixed tracks, all in parallel."""
    obs, _ = eval_envs.reset(seed=[EVAL_SEED_BASE + i for i in range(n_episodes)])
    returns = np.zeros(n_episodes)
    finished = np.zeros(n_episodes, dtype=bool)
    while not finished.all():
        actions = select_actions(net, obs, 0.0, n_actions, device)
        obs, rewards, terminated, truncated, _ = eval_envs.step(actions)
        returns[~finished] += rewards[~finished]
        finished |= terminated | truncated
    return float(returns.mean())


STOP_REQUESTED = False


def _request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # no display on a server
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    run_dir = os.path.join("runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    last_path = os.path.join(run_dir, "last.pt")
    best_path = os.path.join(run_dir, "best.pt")

    # Start the env worker processes *before* touching the GPU.
    n_envs = args.num_envs
    envs = gym.vector.AsyncVectorEnv([env_factory(args) for _ in range(n_envs)])
    eval_envs = gym.vector.AsyncVectorEnv(
        [env_factory(args, early_stop=False) for _ in range(args.eval_episodes)])
    n_actions = int(envs.single_action_space.n)

    torch.set_num_threads(args.threads)
    device = get_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    online = QNetwork(args.frame_stack, n_actions).to(device)
    target = QNetwork(args.frame_stack, n_actions).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size, args.frame_stack)

    step, updates, episode, best_eval, elapsed_before = 0, 0, 0, -float("inf"), 0.0
    resume_step = None
    if os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device)
        online.load_state_dict(ckpt["online"])
        target.load_state_dict(ckpt["target"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step, updates, episode = ckpt["step"], ckpt["updates"], ckpt["episode"]
        best_eval, elapsed_before = ckpt["best_eval"], ckpt["elapsed"]
        resume_step = step
        print(f"[{args.run_name}] resuming from {step:,} transitions "
              f"({elapsed_before / 3600:.1f} h done). Replay buffer refills first.")
    else:
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"[{args.run_name}] new run: {n_envs} envs on {device}")

    random.seed(args.seed + step)
    np.random.seed(args.seed + step)
    torch.manual_seed(args.seed + step)

    train_log = CsvLog(os.path.join(run_dir, "train.csv"),
                       ["step", "episode", "return", "length", "epsilon", "loss", "hours"],
                       resume_step)
    eval_log = CsvLog(os.path.join(run_dir, "eval.csv"),
                      ["step", "eval_return", "hours"], resume_step)

    session_start = time.time()

    def hours():
        return (elapsed_before + time.time() - session_start) / 3600

    def save_last():
        atomic_save({
            "online": online.state_dict(), "target": target.state_dict(),
            "optimizer": optimizer.state_dict(), "step": step, "updates": updates,
            "episode": episode, "best_eval": best_eval, "elapsed": hours() * 3600,
        }, last_path)

    def next_multiple(x, every):
        return (x // every + 1) * every

    next_eval = next_multiple(step, args.eval_every)
    next_ckpt = next_multiple(step, args.checkpoint_every)
    next_print = next_multiple(step, args.print_every)

    obs, _ = envs.reset(seed=args.seed * 100_000 + episode)
    ep_return = np.zeros(n_envs)
    ep_len = np.zeros(n_envs, dtype=int)
    # Gymnasium vector envs reset a finished env on the *following* step and
    # ignore that step's action, so that step is not a real transition.
    resetting = np.zeros(n_envs, dtype=bool)
    recent_returns = deque(maxlen=50)
    losses = deque(maxlen=100)
    update_credit = 0.0
    rate_t, rate_step = time.time(), step

    while step < args.total_steps:
        if STOP_REQUESTED:
            print(f"[{args.run_name}] stop requested, saving checkpoint")
            break
        if args.hours is not None and hours() >= args.hours:
            print(f"[{args.run_name}] reached {args.hours} h, stopping")
            break

        eps = epsilon_at(step, args)
        actions = select_actions(online, obs, eps, n_actions, device)
        next_obs, rewards, terminated, truncated, _ = envs.step(actions)

        valid = ~resetting
        n_new = int(valid.sum())
        if n_new:
            # Only a true termination stops bootstrapping; truncation does not.
            buffer.add_batch(obs[valid], actions[valid], rewards[valid].astype(np.float32),
                             next_obs[valid], terminated[valid].astype(np.float32))
            ep_return[valid] += rewards[valid]
            ep_len[valid] += 1
        step += n_new

        for i in np.flatnonzero((terminated | truncated) & valid):
            episode += 1
            recent_returns.append(ep_return[i])
            loss_avg = float(np.mean(losses)) if losses else float("nan")
            train_log.write([step, episode, round(float(ep_return[i]), 2), int(ep_len[i]),
                             round(eps, 4), round(loss_avg, 5), round(hours(), 3)])
            ep_return[i], ep_len[i] = 0.0, 0
        resetting = terminated | truncated
        obs = next_obs

        # --- learn (Double DQN) ---
        if buffer.size >= args.learning_starts:
            update_credit += n_new / args.train_freq
            while update_credit >= 1.0:
                update_credit -= 1.0
                s, a, r, s2, d = buffer.sample(args.batch_size, device)
                q = online(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    best_next = online(s2).argmax(dim=1, keepdim=True)
                    q_next = target(s2).gather(1, best_next).squeeze(1)
                    q_target = r + args.gamma * (1.0 - d) * q_next
                loss = F.smooth_l1_loss(q, q_target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optimizer.step()
                updates += 1
                if updates % 100 == 0:
                    losses.append(loss.item())
                if updates % args.target_update_every == 0:
                    target.load_state_dict(online.state_dict())

        if step >= next_print:
            next_print = next_multiple(step, args.print_every)
            rate = (step - rate_step) / max(time.time() - rate_t, 1e-6)
            rate_t, rate_step = time.time(), step
            avg = np.mean(recent_returns) if recent_returns else float("nan")
            print(f"[{args.run_name}] {step:>11,} transitions | ep {episode:>6} | "
                  f"avg50 return {avg:7.1f} | eps {eps:.3f} | updates {updates:,} | "
                  f"{rate:,.0f} transitions/s | {hours():.2f} h", flush=True)

        if step >= next_eval:
            next_eval = next_multiple(step, args.eval_every)
            score = evaluate(online, eval_envs, args.eval_episodes, n_actions, device)
            eval_log.write([step, round(score, 2), round(hours(), 3)])
            if score > best_eval:
                best_eval = score
                atomic_save({"online": online.state_dict(), "step": step,
                             "eval_return": score}, best_path)
            print(f"[{args.run_name}] EVAL at {step:,}: {score:.1f} (best {best_eval:.1f})",
                  flush=True)

        if step >= next_ckpt:
            next_ckpt = next_multiple(step, args.checkpoint_every)
            save_last()

    save_last()
    envs.close()
    eval_envs.close()
    print(f"[{args.run_name}] saved at {step:,} transitions, best eval {best_eval:.1f}")


# ---------------------------------------------------------------------------
# Run a trained agent (window, or mp4 videos with --record)
# ---------------------------------------------------------------------------
def play(args):
    if args.record:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        render_mode = "rgb_array"
    else:
        render_mode = "human"
    env = env_factory(args, early_stop=False, render_mode=render_mode,
                      video_dir=args.record, worker=False)()
    device = get_device(args.device)
    n_actions = int(env.action_space.n)
    net = QNetwork(args.frame_stack, n_actions).to(device)
    ckpt = torch.load(args.play, map_location=device)
    net.load_state_dict(ckpt["online"] if "online" in ckpt else ckpt)
    net.eval()
    for ep in range(args.play_episodes):
        state, _ = env.reset()
        done, total = False, 0.0
        while not done:
            action = select_actions(net, state[None], 0.0, n_actions, device)[0]
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        print(f"Episode {ep + 1}: return {total:.1f}")
    env.close()
    if args.record:
        print(f"Videos saved in {args.record}/")


if __name__ == "__main__":
    args = parse_args()
    if args.play:
        play(args)
    else:
        train(args)