import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

class RunningMeanStd:
    """
    Tracks the running mean and variance, so you can normalize a stream of scalars.
    Uses Welford’s algorithm for numerical stability.
    """
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x):
        """
        Incorporate a new batch of values x (can be a numpy array or a scalar).
        """
        x = np.array(x, dtype=np.float64)
        batch_mean = x.mean()
        batch_var  = x.var()
        batch_count = x.size

        # from OpenAI baselines RunningMeanStd
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        """
        Normalize x using the current mean and standard deviation.
        Returns a numpy array of the same shape as x.
        """
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

class PrioritizedReplayBuffer:
    """
    Simple array based Prioritized Experience Replay (no tree).
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

class NstepPrioritizedReplayBuffer:
    """
    Prioritized replay buffer that stores n-step returns internally.
    On each push(state,action,r,next_state,done) we accumulate into
    an n-step deque and when full emit (s0,a0, R^{(n)}, s_n, done_n).
    """
    def __init__(self, capacity, n_step=3, gamma=0.99,
                 alpha=0.6, beta_start=0.4, beta_frames=100_000):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.pos = 0
        # PER params
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        # n-step params
        self.n_step = n_step
        self.gamma = gamma
        self.n_deque = deque(maxlen=n_step)

    def _get_nstep_info(self):
        """Compute R = sum_{i=0..n-1} γ^i r_i, and last next_state, done."""
        R = 0.0
        for idx, (_,_,r,_,_) in enumerate(self.n_deque):
            R += (self.gamma**idx) * r
        next_state, done = self.n_deque[-1][3], self.n_deque[-1][4]
        return R, next_state, done

    def push(self, state, action, reward, next_state, done):
        self.n_deque.append((state, action, reward, next_state, done))
        if len(self.n_deque) == self.n_step:
            s0, a0, _, _, _ = self.n_deque[0]
            R, s_n, done_n = self._get_nstep_info()
            data = (s0, a0, R, s_n, done_n)
            # store with max‐priority
            max_p = max(self.priorities, default=1.0)
            if len(self.buffer) < self.capacity:
                self.buffer.append(data)
                self.priorities.append(max_p)
            else:
                self.buffer[self.pos] = data
                self.priorities[self.pos] = max_p
                self.pos = (self.pos + 1) % self.capacity

        if done:
            while len(self.n_deque) > 1:
                # pop oldest and emit again with fewer than n steps
                self.n_deque.popleft()
                s0, a0, _, _, _ = self.n_deque[0]
                R, s_n, done_n  = self._get_nstep_info()
                data = (s0, a0, R, s_n, done_n)
                max_p = max(self.priorities, default=1.0)
                if len(self.buffer) < self.capacity:
                    self.buffer.append(data)
                    self.priorities.append(max_p)
                else:
                    self.buffer[self.pos] = data
                    self.priorities[self.pos] = max_p
                    self.pos = (self.pos + 1) % self.capacity
            self.n_deque.clear()

    def sample(self, batch_size):
        N = len(self.buffer)
        assert N > 0, "Empty buffer"
        # PER probabilities
        probs = np.array(self.priorities, dtype=np.float32)**self.alpha
        probs /= probs.sum()
        idxs = np.random.choice(N, batch_size, p=probs)
        # IS weights
        self.frame += 1
        beta = min(1.0, self.beta_start + (1.0-self.beta_start)*self.frame/self.beta_frames)
        weights = (N * probs[idxs])**(-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        batch = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
            torch.tensor(idxs, dtype=torch.long),
            weights
        )

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            self.priorities[idx] = abs(err) + 1e-6

    def __len__(self):
        return len(self.buffer)


class FeatureEncoder(nn.Module):
    def __init__(self, input_shape, proj_dim=512):
        super(FeatureEncoder, self).__init__()
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            self.conv_out_dim = int(np.prod(self.conv(dummy_input).shape[1:]))
        
        self.projector = nn.Sequential(
            nn.Linear(self.conv_out_dim, proj_dim),
            nn.ReLU(),
        )

        self.proj_dim = proj_dim

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x / 255.0
        x = self.conv(x)
        return self.projector(x)
    
class NoisyLinear(nn.Module):
    """
    Noisy linear layer with factorized Gaussian noise (NoisyNet).
    """
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        # learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty((out_features)))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / self.in_features**0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init * mu_range)
        self.bias_sigma.data.fill_(self.sigma_init * mu_range)

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
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_size, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )

    def forward(self, state_feat, next_state_feat, action_label):
        # state_proj = self.projector(state_feat)
        # next_state_proj = self.projector(next_state_feat)

        if action_label.ndim == 1:
            action_label = action_label.unsqueeze(0)

        # Inverse model
        inv_input = torch.cat((state_feat, next_state_feat), dim=1)
        pred_action = self.inverse_model(inv_input)
        inv_loss = F.cross_entropy(pred_action, action_label.argmax(dim=-1))

        # Forward model
        fwd_input = torch.cat((state_feat, action_label), dim=1)
        pred_next_proj = self.forward_model(fwd_input)
        fwd_loss = F.mse_loss(pred_next_proj, next_state_feat)

        # Intrinsic reward
        intrinsic_reward = 0.5 * ((pred_next_proj - next_state_feat) ** 2).sum(dim=1)

        return inv_loss, fwd_loss, intrinsic_reward