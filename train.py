import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from utils import NstepPrioritizedReplayBuffer, NoisyDuelDQN, ICM, FeatureEncoder, NoisyLinear
from tqdm import tqdm
import math

import gym
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, TimeLimit
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

import argparse
import os

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(SkipFrame, self).__init__(env)
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

def init_weights(module):
    """
    Applies:
      - Kaiming (He) Normal for Conv2d + ReLU
      - Xavier Uniform for plain Linear layers
      - The NoisyNet μ‐uniform rule for NoisyLinear
      - Zeroes for all biases
    """
    if isinstance(module, nn.Conv2d):
        # He normal for conv+ReLU
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Linear) and not isinstance(module, NoisyLinear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, NoisyLinear):
        fan_in = module.in_features
        bound  = 1.0 / math.sqrt(fan_in)
        module.weight_mu.data.uniform_(-bound, bound)
        module.bias_mu.data.uniform_(-bound, bound)


class DQNAgent:
    def __init__(self, obs_shape, action_dim, lr=0.0001, gamma=1.0, batch_size=32, tau=0.3, beta=0.1,
                 start_epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.9995, target_update_freq=100, replay_buffer_size=1000000,
                 icm_eta=0.1, icm_lambda=0.1):
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize the feature extractor
        self.feature_encoder = FeatureEncoder(obs_shape).to(self.device)
        self.feature_encoder.apply(init_weights)
        feature_dim = self.feature_encoder.proj_dim

        # Initialize the Q-network and target network
        self.q_network = NoisyDuelDQN(feature_dim, action_dim).to(self.device)
        self.q_network.apply(init_weights)
        self.target_network = NoisyDuelDQN(feature_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Initialize the ICM
        self.icm = ICM(feature_dim, action_dim).to(self.device)
        self.icm.apply(init_weights)

        # Initialize the criterion
        self.criterion = nn.SmoothL1Loss()

        # Initialize the target network update counter
        self.target_update_counter = 0

        # Initialize the replay buffer
        self.replay_buffer = NstepPrioritizedReplayBuffer(replay_buffer_size, n_step=9)

        # Initialize the optimizer
        self.optimizer = optim.Adam(
            list(self.feature_encoder.parameters()) +
            list(self.q_network.parameters()) +
            list(self.icm.parameters()),
            lr=lr
        )
        # self.optimizer = optim.Adam(
        #     list(self.feature_encoder.parameters()) +
        #     list(self.q_network.parameters()),
        #     lr=lr
        # )
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10_000, gamma=0.9)

        # Initialize the exploration strategy
        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.beta = beta

        # Initialize ICM hyperparameters
        self.icm_eta = icm_eta
        self.icm_lambda = icm_lambda

    def select_action(self, observation):        
        state = observation.clone() if isinstance(observation, torch.Tensor) else torch.tensor(observation)
        with torch.no_grad():
            self.q_network.reset_noise()
            ft = self.feature_encoder(state)
            q = self.q_network(ft)
        return q.argmax(dim=1).item()
    
    def train(self):
        # self.q_network.train()
        # self.icm.train()
        # self.feature_encoder.train()
        # self.target_network.eval()

        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample a batch from the replay buffer
        state, action, reward, next_state, done, idxs, weights = self.replay_buffer.sample(self.batch_size)
        # Move the batch to the device
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)
        weights = weights.to(self.device)
        idxs = idxs.to(self.device)
        
        # Compute the target Q-values
        with torch.no_grad():
            self.target_network.reset_noise()
            next_ft = self.feature_encoder(next_state).detach()
            online_act = self.q_network(next_ft).argmax(dim=1)
            target_q_value = reward + (1 - done) * (self.gamma ** 9) * self.target_network(next_ft).gather(1, online_act.unsqueeze(1)).squeeze(1)

        # Compute the current Q-values
        self.q_network.reset_noise()
        ft = self.feature_encoder(state)
        q_values = self.q_network(ft)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        # Compute the TD error
        td_errors = (target_q_value - q_value)
        self.replay_buffer.update_priorities(idxs, td_errors.detach().cpu().numpy())

        # Compute the loss
        # loss = self.criterion(q_value, target_q_value)
        loss = F.smooth_l1_loss(td_errors, torch.zeros_like(td_errors), reduction='none', beta=1.0)
        loss = (weights * loss).mean()

        # Update the ICM
        # a_onehot_b = F.one_hot(action, self.action_dim).float().to(self.device)
        # inv_loss, for_loss, _ = self.icm(ft, next_ft, a_onehot_b)
        # icm_loss = (1 - self.beta) * inv_loss + self.beta * for_loss
        
        # all_loss = loss + self.icm_lambda * icm_loss
        
        # gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 250.0)

        # Optimize the Q-network and ICM
        self.optimizer.zero_grad()
        # all_loss.backward()
        loss.backward()
        self.optimizer.step()  
        # self.scheduler.step()

        # return loss.item(), icm_loss.item()
        return loss.item()
    
    def update(self):
        for t_param, p_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                t_param.data.copy_((1 - self.tau) * t_param.data + self.tau * p_param.data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=500, help='Number of episodes to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.1, help='Soft update parameter')
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
    agent = DQNAgent(obs_shape=(4, 84, 84), action_dim=env.action_space.n, lr=args.lr, gamma=args.gamma, tau=args.tau,
                     batch_size=args.batch_size, epsilon_decay=args.epsilon_decay, target_update_freq=args.target_update_freq)
    
    if os.path.exists(f"checkpoints/scheduler_model_{args.start_episode}.pth"):
        print("Loading DQN from checkpoint...")
        agent.q_network.load_state_dict(torch.load(f"checkpoints/scheduler_model_{args.start_episode}.pth"))
        agent.target_network.load_state_dict(agent.q_network.state_dict())
    # if os.path.exists(f"checkpoints/scheduler_icm_{args.start_episode}.pth"):
    #     print("Loading ICM from checkpoint...")
    #     agent.icm.load_state_dict(torch.load(f"checkpoints/scheduler_icm_{args.start_episode}.pth"))
    if os.path.exists(f"checkpoints/scheduler_feature_{args.start_episode}.pth"):
        print("Loading Feature Encoder from checkpoint...")
        agent.feature_encoder.load_state_dict(torch.load(f"checkpoints/scheduler_feature_{args.start_episode}.pth"))
    
    print("device:", agent.device)
    reward_per_eps = []
    for eps in tqdm(range(args.n_episodes)):
        obs = env.reset()
        done = False
        total_reward = 0
        total_intrinsic_reward = 0
        total_extrinsic_reward = 0
        obs = torch.tensor(np.array(obs), dtype=torch.float32).to(agent.device).squeeze(-1)
        total_q_loss = 0
        total_icm_loss = 0
        total_loss = 0
        total_reward = 0
        steps = 0
        while True:
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32).to(agent.device).squeeze(-1)
            # a_one_hot = F.one_hot(torch.tensor(action), agent.action_dim).float().to(agent.device)
            # Compute the intrinsic reward
            # with torch.no_grad():
            #     _, _, intrinsic_reward = agent.icm(
            #         agent.feature_encoder(obs),
            #         agent.feature_encoder(next_obs),
            #         a_one_hot
            #     )
            
            # t_reward = reward + agent.icm_eta * intrinsic_reward.item()
            

            # Store the transition in the replay buffer
            agent.replay_buffer.push(obs.cpu(), action, reward, next_obs.cpu(), done)

            # Update the Q-network and ICM
            # q_loss, icm_loss = agent.train()
            q_loss = agent.train()

            obs = next_obs

            # total_reward += t_reward
            # total_intrinsic_reward += agent.icm_eta * intrinsic_reward.item()
            total_extrinsic_reward += reward

            # if q_loss is not None and icm_loss is not None:
            if q_loss is not None:
                total_q_loss += q_loss
                # total_icm_loss += icm_loss
                # total_loss += q_loss + icm_loss            
            steps += 1

            # Update the target network
            agent.target_update_counter += 1
            if agent.target_update_counter % agent.target_update_freq == 0:
                agent.update()
            
            if done:
                break
            # env.render()
            
        # agent.scheduler.step(total_reward)

        reward_per_eps.append(total_extrinsic_reward)
        
        if (eps + 1) % 10 == 0:
            print(f"Episode {eps + 1}, Avg Reward: {np.mean(reward_per_eps[-10:])}")

        if (eps + 1) % 20 == 0:
            if not os.path.exists('checkpoints/'):
                os.makedirs('checkpoints/')
            torch.save(agent.feature_encoder.state_dict(), f"checkpoints/scheduler_feature_{args.start_episode + eps + 1}.pth")
            torch.save(agent.q_network.state_dict(), f"checkpoints/scheduler_model_{args.start_episode + eps + 1}.pth")
            # torch.save(agent.icm.state_dict(), f"checkpoints/scheduler_icm_{args.start_episode + eps + 1}.pth")

        if args.use_wandb:
            wandb.log({
                # 'instrinsic_reward': total_intrinsic_reward,
                'extrinsic reward': total_extrinsic_reward,
                # 'total_reward': total_reward,
                'q_loss': total_q_loss / steps,
                # 'icm_loss': total_icm_loss / steps,
                # 'total_loss': total_loss / steps,
            })
    # env.close()
if __name__ == "__main__":
    main()