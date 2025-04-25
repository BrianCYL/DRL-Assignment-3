import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class PrioritizedReplayBuffer:
    """
    Simple array‐based Prioritized Experience Replay (no tree).
    Args:
        capacity (int): max number of transitions to store
        alpha (float): how much prioritization is used (0 = uniform, 1 = full)
        beta_start (float): initial value of β for importance sampling
        beta_frames (int): number of samples over which β is annealed to 1
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100_000):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.pos = 0
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

    def push(self, state, action, reward, next_state, done):
        """Add a new experience with maximum existing priority."""
        max_p = max(self.priorities, default=1.0)
        data = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
            self.priorities.append(max_p)
        else:
            self.buffer[self.pos] = data
            self.priorities[self.pos] = max_p
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch, returning states, actions, rewards, next_states, dones,
           indices, and importance-sampling weights."""
        N = len(self.buffer)
        assert N > 0, "Replay buffer is empty!"
        # compute sampling probabilities
        probs = np.array(self.priorities, dtype=np.float32) ** self.alpha
        probs /= probs.sum()
        # sample indices
        idxs = np.random.choice(N, batch_size, p=probs)
        # anneal beta towards 1
        self.frame += 1
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
        # compute importance weights
        weights = (N * probs[idxs]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        batch = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.stack(states, dim=0),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states, dim=0),
            torch.tensor(dones, dtype=torch.float32),
            torch.tensor(idxs, dtype=torch.long),
            weights
        )

    def update_priorities(self, idxs, td_errors):
        """After learning, update the priorities of sampled transitions."""
        for idx, error in zip(idxs, td_errors):
            self.priorities[idx] = abs(error) + 1e-6

    def __len__(self):
        return len(self.buffer)

class FeatureEncoder(nn.Module):
    def __init__(self, input_shape):
        super(FeatureEncoder, self).__init__()
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            self.conv_out_dim = int(np.prod(self.conv(dummy_input).shape[1:]))

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x / 255.0
        return self.conv(x).view(x.size(0), -1)

class NoisyLinear(nn.Module):
    """
    Noisy linear layer with factorized Gaussian noise (NoisyNet).
    """
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.full((out_features,), sigma_init))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / self.in_features**0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

    def reset_noise(self):
        # factorized Gaussian noise
        eps_in = self._f(torch.randn(self.in_features, device=self.weight_mu.device))
        eps_out = self._f(torch.randn(self.out_features, device=self.weight_mu.device))
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    @staticmethod
    def _f(x):
        return x.sign().mul_(x.abs().sqrt_())

class NoisyDuelDQN(nn.Module):
    def __init__(self, feature_dim, action_size):
        super(NoisyDuelDQN, self).__init__()
        self.advantage = nn.Sequential(
            NoisyLinear(feature_dim, 512),
            nn.ReLU(),
            NoisyLinear(512, action_size)
        )
        self.value = nn.Sequential(
            NoisyLinear(feature_dim, 512),
            nn.ReLU(),
            NoisyLinear(512, 1)
        )

    def forward(self, features):
        adv = self.advantage(features)
        val = self.value(features)
        return val + adv - adv.mean(dim=1, keepdim=True)
    
    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

class ICM(nn.Module):
    def __init__(self, feature_dim, action_size):
        super(ICM, self).__init__()
        self.projector = nn.Linear(feature_dim, 256)

        self.inverse_model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

        self.forward_model = nn.Sequential(
            nn.Linear(256 + action_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, state_feat, next_state_feat, action_label):
        state_proj = self.projector(state_feat)
        next_state_proj = self.projector(next_state_feat)

        if action_label.ndim == 1:
            action_label = action_label.unsqueeze(0)

        # Inverse model
        inv_input = torch.cat((state_proj, next_state_proj), dim=1)
        pred_action = self.inverse_model(inv_input)
        inv_loss = F.cross_entropy(pred_action, action_label.argmax(dim=-1))

        # Forward model
        fwd_input = torch.cat((state_proj, action_label), dim=1)
        pred_next_proj = self.forward_model(fwd_input)
        fwd_loss = F.mse_loss(pred_next_proj, next_state_proj)

        # Intrinsic reward
        intrinsic_reward = 0.5 * ((pred_next_proj - next_state_proj) ** 2).sum(dim=1)

        return inv_loss, fwd_loss, intrinsic_reward