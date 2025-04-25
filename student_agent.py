import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import FeatureEncoder, NoisyDuelDQN
import cv2

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
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.encoder = FeatureEncoder((4, 84, 84))
        self.encoder.load_state_dict('skip_feature_440.pth', map='cpu')
        self.encoder.eval()
        feat_dim = self.encoder.conv_out_dim
        self.q_net = NoisyDuelDQN(feat_dim, self.action_space.n)
        self.q_net.load_state_dict('skip_model_440.pth', map='cpu')
        self.q_net.eval()
        self.q_net.reset_noise()

    def act(self, observation):
        state = self._preprocess(observation)
        self.q_net.reset_noise()
        with torch.no_grad():
            feat = self.encoder(state)
            q = self.q_net(feat)
        return q.argmax(dim=1).item()

    def _preprocess(self, observation):
        if observation.shape[1] != 84 or observation.shape[2] != 84:
            observation = cv2.resize(
                observation,
                (84, 84),
            )            
        
        return observation

def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, (84, 84))
    env = FrameStack(env, 4)

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