import pathlib
import random
import torch
import numpy as np
from tqdm import tqdm
from model import MLP
import matplotlib.pyplot as plt
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparams
batch_size = 1000
num_epochs = 2000
num_warmup_steps = 10
lr = 1e-3
seed = 1234

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


# Network
net = MLP(num_in=2, num_out=2).to(DEVICE)

# Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# LR Scheduler
schedule_func = TYPE_TO_SCHEDULER_FUNCTION[SchedulerType("cosine")]
lr_scheduler = schedule_func(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_epochs
)

# Model params
pos_emb_scale = 20
k_steps = 50


# Loss
def calculate_loss(noise=None, B=None, reduction="mean"):
    if noise is None:
        z0 = torch.randn((B, 2), device=DEVICE)
    else:
        z0 = noise
        B = noise.shape[0]

    t = torch.rand((B, 1), device=DEVICE)
    z1 = torch.cat((torch.ones(B, 1), torch.zeros(B, 1)), dim=-1).to(DEVICE)
    target_vel = z1 - z0
    zt = z0 + target_vel * t
    timesteps = t * pos_emb_scale
    pred_vel = net(zt, timesteps)
    loss = torch.nn.functional.mse_loss(pred_vel, target_vel, reduction=reduction)
    return loss


# Inference
def infer_y(noise=None, B=None):
    if noise is None:
        z = torch.randn((B, 2), device=DEVICE)
    else:
        z = noise
        B = noise.shape[0]
    t = torch.linspace(0, 1, k_steps + 1)[:-1]
    dt = torch.ones(k_steps) / k_steps
    for k in range(k_steps):
        timesteps = torch.ones((B, 1), device=DEVICE) * t[k]
        timesteps *= pos_emb_scale
        pred_vel = net(z, timesteps)
        z = z + pred_vel * dt[k]

    # Project output to unit circle
    z = z / torch.norm(z, dim=-1, keepdim=True)
    return z


# Training loop
losses = list()
for epoch in tqdm(range(num_epochs), desc="Epoch"):
    loss = calculate_loss(B=batch_size)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    lr_scheduler.step()
    losses.append(loss.item())

# Plot loss curve
plt.figure()
plt.plot(losses)
plt.title("Training Loss")
# plt.show()


# Evaluation
B = 10000
start_noise = torch.randn((B, 2), device=DEVICE)
losses = calculate_loss(noise=start_noise, reduction="none").detach()
print(f"Loss on random noise: {loss.item()}")
with torch.no_grad():
    z = infer_y(start_noise)
angle_errors = torch.atan2(z[:, 1], z[:, 0]) * 180 / torch.pi
angle_abs_errors = torch.abs(angle_errors)

print(f"{angle_abs_errors.mean() = }")

start_noisy_angle_np = torch.atan2(start_noise[:, 1], start_noise[:, 0]).cpu().numpy()
angle_err_np = angle_abs_errors.cpu().numpy()
plt.scatter(start_noisy_angle_np, angle_err_np)
# plt.show()

saving_path = pathlib.Path("toy_circle") / "results" / "euclid"
saving_path.mkdir(exist_ok=True, parents=True)

np.save(saving_path / "start_noise.npy", start_noise.cpu().numpy())
np.save(saving_path / "losses.npy", losses.cpu().numpy())
np.save(saving_path / "angle_abs_errors.npy", angle_abs_errors.cpu().numpy())
