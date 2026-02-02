import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feature_size, feature_size), nn.GELU(), nn.Linear(feature_size, feature_size)
        )

    def forward(self, x):
        return self.proj(x)


class LatentPolicy(nn.Module):
    def __init__(self, feature_size, intermediate_size=512, deterministic=False):
        super().__init__()
        self.deterministic = deterministic
        self.fc = nn.Sequential(
            nn.Linear(feature_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, intermediate_size),
            nn.LayerNorm(intermediate_size),
        )

        self.mean = nn.Linear(intermediate_size, feature_size)
        if not deterministic:
            self.log_std = nn.Linear(intermediate_size, feature_size)

    def forward(self, x, temperature=1.0):
        x = self.fc(x)
        mean = self.mean(x)
        if self.deterministic:
            return torch.distributions.Normal(mean, torch.ones_like(mean) * 1e-9)
        log_std = self.log_std(x)
        std = log_std.exp() * temperature
        return torch.distributions.Normal(mean, std)
