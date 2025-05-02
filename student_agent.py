import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import FeatureEncoder, NoisyDuelDQN
import cv2
import numpy as np
from collections import deque

from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

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

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent with built-in preprocessing: grayscale, resize, and frame-stacking."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)

        # Load pretrained feature encoder
        self.encoder = FeatureEncoder((4, 84, 84))
        self.encoder.load_state_dict(
            torch.load('checkpoints/scheduler_feature_400.pth',map_location=torch.device('cpu'))
            )
        self.encoder.eval()

        # Load pretrained Q-network
        feat_dim = self.encoder.proj_dim
        self.q_net = NoisyDuelDQN(feat_dim, self.action_space.n)
        self.q_net.load_state_dict(
            torch.load('checkpoints/scheduler_model_400.pth', map_location=torch.device('cpu'))
            )
        self.q_net.eval()
        self.q_net.reset_noise()

        # Frame buffer for stacking
        self.frame_deque = deque(maxlen=4)
        self.action_list = []

    def _preprocess(self, observation):
        """
        Convert raw RGB observation to (4,84,84) tensor:
         - Grayscale
         - Resize to 84×84
         - Append to internal deque and stack last 4 frames
        """
        # Raw obs: H×W×3
        arr = np.asarray(observation)
        # Grayscale: H×W
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        # Resize to 84×84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

        # Append and pad if needed
        self.frame_deque.append(resized)
        while len(self.frame_deque) < self.frame_deque.maxlen:
            self.frame_deque.append(resized)

        # Stack
        stacked = np.stack(self.frame_deque, axis=0)
        # To tensor
        tensor = torch.tensor(stacked, dtype=torch.float32)
        return tensor

    def act(self, observation):
        if self.action_list:
            # If the action list is not empty, pop the first action
            return self.action_list.pop()
        # Preprocess raw obs into (4,84,84)
        state = self._preprocess(observation)
        # observation = torch.tensor(np.array(observation), dtype=torch.float32).squeeze(-1)
        # state = observation.clone() if isinstance(observation, torch.Tensor) else torch.tensor(observation)
        # NoisyNet reset
        self.q_net.reset_noise()

        # Forward pass
        with torch.no_grad():
            feat = self.encoder(state)
            q = self.q_net(feat)
            for _ in range(4):
                # Append the action to the action list
                self.action_list.append(q.argmax(dim=1).item())
        return self.action_list.pop()

def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=(84, 84))
    env = FrameStack(env, num_stack=4)
    agent = Agent()

    total_reward = 0
    obs = env.reset()
    done = False
    while not done:
        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        if done:
            break
        total_reward += reward
        env.render()

    print(f'Total reward: {total_reward}')
    env.close()

if __name__ == "__main__":
    main()