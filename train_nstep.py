import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utils import NstepPrioritizedReplayBuffer, NoisyDuelDQN, ICM, FeatureEncoder  # use the new n‐step buffer
from tqdm import tqdm

import gym
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, TimeLimit
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

import argparse
import os
from collections import deque

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
    def __init__(self, obs_shape, action_dim,
                 lr=1e-4, gamma=0.99, n_step=3,
                 batch_size=32, tau=0.01, beta=0.1,
                 replay_buffer_size=100000, icm_eta=0.01, icm_lambda=0.1,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 target_update_freq=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.n_step = n_step
        self.batch_size = batch_size
        self.tau = tau
        self.beta = beta
        self.icm_eta = icm_eta
        self.icm_lambda = icm_lambda
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.action_dim = action_dim

        # n-step PER buffer
        self.replay_buffer = NstepPrioritizedReplayBuffer(
            capacity=replay_buffer_size,
            n_step=n_step,
            gamma=gamma
        )

        # feature encoder
        self.feature_encoder = FeatureEncoder(obs_shape).to(self.device)
        feature_dim = self.feature_encoder.conv_out_dim

        # Q networks
        self.q_network = NoisyDuelDQN(feature_dim, action_dim).to(self.device)
        self.target_network = NoisyDuelDQN(feature_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # ICM
        self.icm = ICM(feature_dim, action_dim).to(self.device)

        # optimizer
        self.optimizer = optim.Adam(
            list(self.feature_encoder.parameters()) +
            list(self.q_network.parameters()) +
            list(self.icm.parameters()), lr=lr)

        self.target_update_counter = 0

    def select_action(self, observation):
        state = observation.clone() if isinstance(observation, torch.Tensor) else torch.tensor(observation)
        self.q_network.reset_noise()
        with torch.no_grad():
            ft = self.feature_encoder(state)
            q  = self.q_network(ft)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        return int(q.argmax(dim=1).item())

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None
        # sample batch
        state, action, reward, next_state, done, idxs, weights = self.replay_buffer.sample(self.batch_size)
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)
        weights = weights.to(self.device)
        idxs = idxs.to(self.device)

        # compute Double DQN n-step target
        gamma_n = self.gamma ** self.n_step
        self.target_network.reset_noise()
        with torch.no_grad():
            next_ft = self.feature_encoder(next_state).detach()
            target_q_values = self.target_network(next_ft)
            target_q = reward + (1 - done) * gamma_n * target_q_values.max(1)[0]

        # current Q
        self.q_network.reset_noise()
        ft = self.feature_encoder(state)
        next_ft = self.feature_encoder(next_state).detach()
        q_vals = self.q_network(ft)
        q_val = q_vals.gather(1, action.unsqueeze(1)).squeeze(1)

        # update priorities
        td_errors = (q_val - target_q).abs().detach().cpu().numpy()
        self.replay_buffer.update_priorities(idxs, td_errors)

        # losses
        dqn_loss = (weights * (q_val - target_q).pow(2)).mean()
        inv_l, fwd_l, _ = self.icm(ft, next_ft, F.one_hot(action, self.action_dim).float().to(self.device))
        icm_loss = (1 - self.beta) * inv_l + self.beta * fwd_l
        loss = dqn_loss + self.icm_lambda * icm_loss

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 250.0)
        self.optimizer.step()

        # soft update
        # self.update_counter += 1
        # if self.update_counter % self.target_update_freq == 0:
        #     for t_p, q_p in zip(self.target_network.parameters(), self.q_network.parameters()):
        #         t_p.data.copy_(self.tau * q_p.data + (1 - self.tau) * t_p.data)

        return dqn_loss.item(), icm_loss.item()

    def update(self):
        for t_param, p_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                t_param.data.copy_(self.tau * p_param.data + (1 - self.tau) * t_param.data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=500, help='Number of episodes to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=1.0, help='Discount factor')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epsilon_decay', type=float, default=0.99, help='Epsilon decay rate')
    parser.add_argument('--max_steps', type=int, default=2500, help='Maximum steps per episode')
    parser.add_argument('--target_update_freq', type=int, default=100, help='Target network update frequency')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='dqn_mario', help='Wandb project name')
    parser.add_argument('--start_episode', type=int, default=0, help='continue training from this episode')
    args = parser.parse_args()

    if not os.path.exists('checkpoints/'):
        os.makedirs('checkpoints/')
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name='dqn_mario')


    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=(84, 84))
    env = FrameStack(env, num_stack=4)
    env = TimeLimit(env, max_episode_steps=args.max_steps)

    agent = DQNAgent(obs_shape=(4,84,84), action_dim=env.action_space.n,
                     lr=args.lr, gamma=args.gamma,
                     batch_size=args.batch_size,
                     epsilon_decay=args.epsilon_decay,
                     target_update_freq=args.target_update_freq,
                     n_step=3)
    if os.path.exists(f"checkpoints/skip_model_{args.start_episode}.pth"):
        print("Loading DQN from checkpoint...")
        agent.q_network.load_state_dict(torch.load(f"checkpoints/skip_model_{args.start_episode}.pth"))
        agent.target_network.load_state_dict(agent.q_network.state_dict())
    if os.path.exists(f"checkpoints/skip_icm_{args.start_episode}.pth"):
        print("Loading ICM from checkpoint...")
        agent.icm.load_state_dict(torch.load(f"checkpoints/skip_icm_{args.start_episode}.pth"))
    if os.path.exists(f"checkpoints/skip_feature_{args.start_episode}.pth"):
        print("Loading Feature Encoder from checkpoint...")
        agent.feature_encoder.load_state_dict(torch.load(f"checkpoints/skip_feature_{args.start_episode}.pth"))
    
    print("device:", agent.device)
    reward_per_eps = []
    for ep in tqdm(range(args.n_episodes)):
        obs = env.reset()
        obs = torch.tensor(np.array(obs),dtype=torch.float32).to(agent.device).squeeze(-1)
        done = False
        total_q_loss = 0
        total_icm_loss = 0
        total_loss = 0
        total_reward = 0
        steps = 0
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            next_obs = torch.tensor(np.array(next_obs),dtype=torch.float32).to(agent.device).squeeze(-1)
            a_one_hot = F.one_hot(torch.tensor(action), agent.action_dim).float().to(agent.device)
            _, _, ir = agent.icm(
                agent.feature_encoder(obs.unsqueeze(0)),
                agent.feature_encoder(next_obs.unsqueeze(0)),
                a_one_hot
            )
            t_reward = reward + agent.icm_eta * ir.item()
            agent.replay_buffer.push(obs.cpu(), action, t_reward, next_obs.cpu(), done)
            q_loss, icm_loss = agent.train()
            obs = next_obs
            total_reward += reward
            if q_loss is not None and icm_loss is not None:
                total_q_loss += q_loss
                total_icm_loss += icm_loss
                total_loss += q_loss + icm_loss            
            steps += 1

            # Update the target network
            agent.target_update_counter += 1
            if agent.target_update_counter % agent.target_update_freq == 0:
                agent.update()
        reward_per_eps.append(total_reward)
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}, Avg Reward: {np.mean(reward_per_eps[-10:])}")
        
        if (ep + 1) % 20 == 0:
            if not os.path.exists('checkpoints/'):
                os.makedirs('checkpoints/')
            torch.save(agent.feature_encoder.state_dict(), f"checkpoints/double_feature_{args.start_episode + ep + 1}.pth")
            torch.save(agent.q_network.state_dict(), f"checkpoints/double_model_{args.start_episode + ep + 1}.pth")
            torch.save(agent.icm.state_dict(), f"checkpoints/doouble_icm_{args.start_episode + ep + 1}.pth")

        if args.use_wandb:
            wandb.log({
                'total_reward': total_reward,
                'q_loss': total_q_loss / steps,
                'icm_loss': total_icm_loss / steps,
                'total_loss': total_loss / steps,
            })


        
if __name__ == '__main__':
    main()
