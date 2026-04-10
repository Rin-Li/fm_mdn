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
net = MLP(num_in=2, num_out=1).to(DEVICE)

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


def sample_SO2(B):
    # Sample randomly between -pi, pi
    z0_angle = (torch.rand((B, 1), device=DEVICE) - 0.5) * 2.0 * torch.pi
    # Transform to SO(2)
    z0 = exp(z0_angle)
    return z0


def exp(theta):
    if theta.ndim == 1:
        theta = theta.unsqueeze(-1)
    R = torch.stack(
        (
            torch.cat((torch.cos(theta), -torch.sin(theta)), axis=-1),
            torch.cat((torch.sin(theta), torch.cos(theta)), axis=-1),
        ),
        axis=-2,  # Stack along second dimension (not last)! Very important!
    )
    return R


def log(R: torch.tensor):
    assert R.shape[-2:] == (2, 2)
    return torch.arctan2(R[..., 1, 0], R[..., 0, 0]).unsqueeze(
        -1
    )  # Keep single last dimensions again


def SO2_to_pfp(z_SO2):
    return z_SO2[:, :, 0]


# Loss
def calculate_loss(noise=None, B=None, reduction="mean"):
    if noise is None:
        # Euclidean variant
        # z0 = torch.randn((B, 2), device=DEVICE)

        # SO(2) variant
        z0 = sample_SO2(B)
    else:
        z0 = noise
        B = noise.shape[0]

    t = torch.rand((B, 1), device=DEVICE)
    z1 = torch.eye((2)).unsqueeze(0).repeat(B, 1, 1).to(DEVICE)

    # Euclidean variant
    # target_vel = z1 - z0
    # zt = z0 + target_vel * t

    # SO(2) variant
    # target_vel = log(torch.linalg.pinv(z0) @ z1)
    target_vel = log(torch.linalg.pinv(z0) @ z1)
    zt = z0 @ exp(target_vel * t)

    # Convert to pfp
    zt_pfp = SO2_to_pfp(zt)
    timesteps = t * pos_emb_scale

    pred_vel = net(zt_pfp, timesteps)

    # Use the loss to make a forward prediction of the goal state
    z_pred = zt @ exp(pred_vel * (1 - t))

    loss = torch.nn.functional.mse_loss(z_pred, z1, reduction=reduction)
    return loss


# Inference
def infer_y(noise=None, B=None):
    if noise is None:
        # z = torch.randn((B, 2), device=DEVICE)
        z = sample_SO2(B)
    else:
        z = noise
        B = noise.shape[0]
    t = torch.linspace(0, 1, k_steps + 1)[:-1]
    dt = torch.ones(k_steps) / k_steps
    for k in range(k_steps):
        timesteps = torch.ones((B, 1), device=DEVICE) * t[k]
        timesteps *= pos_emb_scale
        z_pfp = SO2_to_pfp(z)
        pred_vel = net(z_pfp, timesteps)

        # z = z + pred_vel * dt[k]
        z = z @ exp(pred_vel * dt[k])

    # Project output to unit circle
    # z = z / torch.norm(z, dim=-1, keepdim=True)
    z = z[..., 0]
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
plt.show()


# Evaluation
B = 10000
# start_noise = torch.randn((B, 2), device=DEVICE)
start_noise = sample_SO2(B)
losses = calculate_loss(noise=start_noise, reduction="none").detach()
print(f"Loss on random noise: {loss.item()}")
with torch.no_grad():
    z = infer_y(start_noise)
angle_errors = torch.atan2(z[:, 1], z[:, 0]) * 180 / torch.pi
angle_abs_errors = torch.abs(angle_errors)

print(f"{angle_abs_errors.mean() = }")

start_noisy_angle_np = log(start_noise).cpu().numpy()[:, 0]
angle_err_np = angle_abs_errors.cpu().numpy()
plt.scatter(start_noisy_angle_np, angle_err_np)
plt.show()

saving_path = pathlib.Path("toy_circle") / "results" / "so2_forward"
saving_path.mkdir(exist_ok=True, parents=True)


np.save(saving_path / "start_noise.npy", start_noise.cpu().numpy())
np.save(saving_path / "losses.npy", losses.cpu().numpy())
np.save(saving_path / "angle_abs_errors.npy", angle_abs_errors.cpu().numpy())
