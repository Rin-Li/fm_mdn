import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        num_in,
        num_out,
        num_hid=500,
    ):
        super().__init__()
        self.num_in = num_in
        self.num_hid = num_hid
        self.num_out = num_out

        self.rdFrequency = torch.normal(0, 1, (1, 100))

        self.net = nn.Sequential(
            nn.Linear(num_in + 100, num_hid),
            nn.SiLU(),
            nn.Linear(num_hid, num_hid),
            nn.SiLU(),
            nn.Linear(num_hid, num_hid),
            nn.SiLU(),
            nn.Linear(num_hid, num_hid),
            nn.SiLU(),
            nn.Linear(num_hid, num_out),
        )

    def forward(self, noisy_y, timesteps):
        time_feature = torch.cos(torch.matmul(timesteps, self.rdFrequency.to(timesteps)))
        x_in = torch.cat([noisy_y, time_feature], dim=-1)
        out = self.net(x_in)
        return out
