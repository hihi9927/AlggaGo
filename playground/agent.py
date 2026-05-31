import math
import torch
import torch.nn as nn
import torch.optim as optim


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.angle_head = nn.Sequential(nn.Linear(64, 1), nn.Tanh())
        self.force_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.log_std_angle = nn.Parameter(torch.tensor(0.3))
        self.log_std_force = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x):
        feat        = self.fc(x)
        mean_angle  = self.angle_head(feat) * math.pi          # [-π, π]
        mean_force  = self.force_head(feat)                     # [0, 1]
        std_angle   = torch.exp(self.log_std_angle).clamp(0.05, 1.5)
        std_force   = torch.exp(self.log_std_force).clamp(0.01, 0.5)
        return mean_angle, std_angle, mean_force, std_force


class Agent:
    def __init__(self, lr=3e-3):
        self.net      = PolicyNet()
        self.opt      = optim.Adam(self.net.parameters(), lr=lr)
        self._lp      = None
        self.baseline = 0.0

    def act(self, obs, greedy=False):
        t = torch.FloatTensor(obs).unsqueeze(0)
        mean_angle, std_angle, mean_force, std_force = self.net(t)
        if greedy:
            return mean_angle.item(), mean_force.item()
        d_angle = torch.distributions.Normal(mean_angle, std_angle)
        d_force = torch.distributions.Normal(mean_force, std_force)
        a = d_angle.sample()
        f = d_force.sample().clamp(0.0, 1.0)
        self._lp = d_angle.log_prob(a) + d_force.log_prob(f)
        return a.item(), f.item()

    def learn(self, reward):
        if self._lp is None:
            return 0.0
        adv = reward - self.baseline
        self.baseline += 0.05 * adv
        loss = -self._lp * adv
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        self._lp = None
        return loss.item()

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, weights_only=True))
        self.net.eval()