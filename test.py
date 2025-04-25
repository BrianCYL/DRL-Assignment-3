from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()

env.close()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utils import PrioritizedReplayBuffer, DuelDQN, ICM
from tqdm import tqdm

import gym
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, TimeLimit
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

import argparse
import os

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class DQNAgent:
    def __init__(self, obs_shape, action_dim, lr=1e-4, gamma=0.99,
                 batch_size=32, tau=0.01, beta=0.1,
                 start_epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.9995,
                 target_update_freq=100, buffer_capacity=100000,
                 icm_eta=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DuelDQN(obs_shape, action_dim).to(self.device)
        self.target_network = DuelDQN(obs_shape, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.icm = ICM(obs_shape, action_dim).to(self.device)

        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss(reduction='none')

        self.epsilon = start_epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.beta = beta
        self.icm_eta = icm_eta
        self.action_dim = action_dim
        self.target_update_freq = target_update_freq
        self.update_counter = 0

    def select_action(self, obs):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad():
            q = self.q_network(state)
        return q.argmax(dim=1).item()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None
        # sample batch
        states, actions, rewards, next_states, dones, idxs, weights = \
            self.replay_buffer.sample(self.batch_size)
        # to device and normalize
        states = states.to(self.device) / 255.0
        next_states = next_states.to(self.device) / 255.0
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        # Q values
        q_vals = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # update priorities
        td_errors = (q_vals - target_q).abs().cpu().numpy()
        self.replay_buffer.update_priorities(idxs, td_errors)

        # compute DQN loss with IS weights
        losses = (q_vals - target_q).pow(2)
        dqn_loss = (weights * losses).mean()

        # ICM loss
        a_onehot = F.one_hot(actions, self.action_dim).float().to(self.device)
        inv_loss, fwd_loss, _ = self.icm(states, next_states, a_onehot)
        icm_loss = (1 - self.beta) * inv_loss + self.beta * fwd_loss

        # combined loss and optimize
        total_loss = dqn_loss + icm_loss
        self.optimizer.zero_grad()
        self.icm_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 250.0)
        self.optimizer.step()
        self.icm_optimizer.step()

        # soft update
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            for tp, pp in zip(self.target_network.parameters(), self.q_network.parameters()):
                tp.data.copy_(self.tau * pp.data + (1 - self.tau) * tp.data)

        return dqn_loss.item(), icm_loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epsilon_decay', type=float, default=0.99)
    parser.add_argument('--max_steps', type=int, default=2500)
    parser.add_argument('--target_update_freq', type=int, default=100)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='dqn_mario')
    args = parser.parse_args()

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=(84, 84))
    env = FrameStack(env, num_stack=4)
    env = TimeLimit(env, max_episode_steps=args.max_steps)

    agent = DQNAgent(
        obs_shape=(4, 84, 84), action_dim=env.action_space.n,
        lr=args.lr, gamma=args.gamma, batch_size=args.batch_size,
        epsilon_decay=args.epsilon_decay, target_update_freq=args.target_update_freq,
        buffer_capacity=100000, icm_eta=0.01
    )

    for ep in tqdm(range(args.n_episodes)):
        obs = env.reset()
        obs = torch.tensor(np.array(obs), dtype=torch.float32).squeeze(-1)
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32).squeeze(-1)
            # intrinsic reward
            _, _, ir = agent.icm(
                obs.unsqueeze(0).to(agent.device)/255.0,
                next_obs.unsqueeze(0).to(agent.device)/255.0,
                F.one_hot(torch.tensor([action]), env.action_space.n).float().to(agent.device)
            )
            tr = reward + agent.icm_eta * ir.item()
            agent.replay_buffer.push(obs, action, tr, next_obs, done)

            q_loss, icm_loss = agent.train_step()
            obs = next_obs
            total_reward += reward

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        print(f"Episode {ep+1}, Reward {total_reward}, Epsilon {agent.epsilon:.4f}")

if __name__ == '__main__':
    main()
