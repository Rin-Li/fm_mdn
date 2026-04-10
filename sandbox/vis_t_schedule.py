import torch
import matplotlib.pyplot as plt

plt.figure()

k_steps = 20
t = torch.linspace(0, 1, k_steps + 1)[:-1]

# Linear
dt_lin = torch.ones(k_steps) / k_steps

# Cosine
dt_cos = torch.cos(t * torch.pi) + 1
dt_cos /= torch.sum(dt_cos)
t0_cos = torch.cat((torch.zeros(1), torch.cumsum(dt_cos, dim=0)[:-1]))
plt.scatter(t.numpy(), t0_cos.numpy(), label="cos")

for scaling in [1, 2, 4, 8]:
    dt_exp = torch.exp(-t * scaling)
    dt_exp /= torch.sum(dt_exp)
    t0_exp = torch.cat((torch.zeros(1), torch.cumsum(dt_exp, dim=0)[:-1]))
    plt.scatter(t.numpy(), t0_exp.numpy(), label=f"exp_{scaling}")

plt.legend()
plt.show()

print("Done")
