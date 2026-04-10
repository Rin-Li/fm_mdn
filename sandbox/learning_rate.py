import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from diffusion_policy.model.common.lr_scheduler import get_scheduler


epochs = 2000
len_dataset = 10000

params = torch.Tensor(1, 1, 1, 1)
optimizer = AdamW([params], lr=1.0e-4, betas=[0.95, 0.999], eps=1.0e-8, weight_decay=1.0e-6)
lr_scheduler: LambdaLR = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=(len_dataset * epochs),
    # pytorch assumes stepping LRScheduler every epoch
    # however huggingface diffusers steps it every batch
)

for epoch in range(epochs):
    # for _ in range(len_dataset):
    lr_scheduler.step()
    # print(lr_scheduler.get_last_lr())

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, LR: {lr_scheduler.get_last_lr()}")
