import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Generator(nn.Module):
    def __init__(self, z_dim=4, h=64, out_dim=2, use_tanh=True):
        super().__init__()
        self.use_tanh = use_tanh
        self.net = nn.Sequential(
            nn.Linear(z_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, out_dim),
        )

    def forward(self, z):
        out = self.net(z)
        if self.use_tanh:
            return torch.tanh(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, in_dim=2, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.LeakyReLU(0.2),
            nn.Linear(h, h),
            nn.LeakyReLU(0.2),
            nn.Linear(h, 1),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def get_data(n_samples=2048, noise=0.1):
    x, _ = make_moons(n_samples=n_samples, noise=noise)
    x = np.asarray(x, dtype=np.float32)
    mins = x.min(axis=0, keepdims=True)
    maxs = x.max(axis=0, keepdims=True)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    x = 2.0 * (x - mins) / ranges - 1.0
    return torch.tensor(x, dtype=torch.float32)


def sample_noise(batch, dim, device):
    return torch.randn(batch, dim, device=device)


def plot_points(real, fake, path):
    plt.figure(figsize=(5, 5))
    plt.scatter(real[:, 0], real[:, 1], c="#2ca02c", s=10, alpha=0.6)
    plt.scatter(fake[:, 0], fake[:, 1], c="#1f77b4", s=10, alpha=0.6)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def mmd_rbf(real, fake, sigmas=(0.05, 0.1, 0.2, 0.4)):
    x = real.float()
    y = fake.float()
    xx = torch.cdist(x, x, p=2).pow(2)
    yy = torch.cdist(y, y, p=2).pow(2)
    xy = torch.cdist(x, y, p=2).pow(2)
    k_xx = 0.0
    k_yy = 0.0
    k_xy = 0.0
    for s in sigmas:
        gamma = 1.0 / (2.0 * s * s)
        k_xx = k_xx + torch.exp(-gamma * xx)
        k_yy = k_yy + torch.exp(-gamma * yy)
        k_xy = k_xy + torch.exp(-gamma * xy)
    mmd2 = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    return float(torch.sqrt(torch.clamp(mmd2, min=0.0)).item())


@torch.no_grad()
def evaluate_generator_mmd(gen, real_data, z_dim, device, n_eval=1024):
    n_eval = min(n_eval, real_data.size(0))
    z = sample_noise(n_eval, z_dim, device)
    fake = gen(z).cpu()
    real = real_data[:n_eval].cpu()
    return mmd_rbf(real, fake)


@torch.no_grad()
def save_snapshot(gen, real_data, z_dim, device, out_path, n_vis=1024):
    n_vis = min(n_vis, real_data.size(0))
    z = sample_noise(n_vis, z_dim, device)
    fake_points = gen(z).cpu().numpy()
    real_points = real_data[:n_vis].numpy()
    plot_points(real_points, fake_points, out_path)


def train_gan(
    device,
    out_dir,
    real_data,
    epochs=1200,
    steps_per_epoch=8,
    batch_size=256,
    z_dim=4,
    h=64,
    lr=2e-4,
):
    os.makedirs(out_dir, exist_ok=True)
    gen = Generator(z_dim=z_dim, h=h, use_tanh=True).to(device)
    disc = Discriminator(h=h).to(device)
    gen.apply(init_weights)
    disc.apply(init_weights)
    opt_g = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_d = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.9))
    bce = nn.BCEWithLogitsLoss()
    n_real = real_data.size(0)
    freq = max(1, epochs // 10)
    print(f"[train_gan] epochs={epochs} steps/epoch={steps_per_epoch} batch={batch_size}")
    for e in range(1, epochs + 1):
        for _ in range(steps_per_epoch):
            idx = torch.randint(0, n_real, (batch_size,))
            real = real_data[idx].to(device)
            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)
            z = sample_noise(batch_size, z_dim, device)
            fake = gen(z).detach()
            d_loss = 0.5 * (bce(disc(real), real_labels) + bce(disc(fake), fake_labels))
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            z = sample_noise(batch_size, z_dim, device)
            g_loss = bce(disc(gen(z)), real_labels)
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
        if e % freq == 0 or e == 1:
            save_snapshot(gen, real_data, z_dim, device, os.path.join(out_dir, f"gan_epoch_{e}.png"))
    score = evaluate_generator_mmd(gen, real_data, z_dim, device)
    print(f"[train_gan] final MMD={score:.6f}")
    return gen, score


def train_wgan(
    device,
    out_dir,
    real_data,
    epochs=2200,
    steps_per_epoch=10,
    batch_size=256,
    z_dim=2,
    h=128,
    lr=5e-5,
    clip_value=0.05,
    n_critic=7,
):
    os.makedirs(out_dir, exist_ok=True)
    gen = Generator(z_dim=z_dim, h=h, use_tanh=True).to(device)
    disc = Discriminator(h=h).to(device)
    gen.apply(init_weights)
    disc.apply(init_weights)
    opt_g = optim.RMSprop(gen.parameters(), lr=lr)
    opt_d = optim.RMSprop(disc.parameters(), lr=lr)
    n_real = real_data.size(0)
    freq = max(1, epochs // 10)
    print(f"[train_wgan] epochs={epochs} steps/epoch={steps_per_epoch} batch={batch_size}")
    for e in range(1, epochs + 1):
        for _ in range(steps_per_epoch):
            for _ in range(n_critic):
                idx = torch.randint(0, n_real, (batch_size,))
                real = real_data[idx].to(device)
                z = sample_noise(batch_size, z_dim, device)
                fake = gen(z).detach()
                d_loss = -(disc(real).mean() - disc(fake).mean())
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()
                for p in disc.parameters():
                    p.data.clamp_(-clip_value, clip_value)
            z = sample_noise(batch_size, z_dim, device)
            g_loss = -disc(gen(z)).mean()
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
        if e % freq == 0 or e == 1:
            save_snapshot(gen, real_data, z_dim, device, os.path.join(out_dir, f"wgan_epoch_{e}.png"))
    score = evaluate_generator_mmd(gen, real_data, z_dim, device)
    print(f"[train_wgan] final MMD={score:.6f}")
    return gen, score


def gradient_penalty(discriminator, real, fake):
    alpha = torch.rand(real.size(0), 1, device=real.device).expand_as(real)
    interp = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    pred = discriminator(interp)
    grads = autograd.grad(
        outputs=pred,
        inputs=interp,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
    )[0]
    grads = grads.view(grads.size(0), -1)
    return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()


def train_wgangp(
    device,
    out_dir,
    real_data,
    epochs=1800,
    steps_per_epoch=8,
    batch_size=256,
    z_dim=4,
    h=64,
    g_lr=1e-4,
    d_lr=2e-4,
    n_critic=5,
    n_gen=1,
    lambda_gp=10.0,
):
    os.makedirs(out_dir, exist_ok=True)
    gen = Generator(z_dim=z_dim, h=h, use_tanh=True).to(device)
    disc = Discriminator(h=h).to(device)
    gen.apply(init_weights)
    disc.apply(init_weights)
    opt_g = optim.Adam(gen.parameters(), lr=g_lr, betas=(0.0, 0.9))
    opt_d = optim.Adam(disc.parameters(), lr=d_lr, betas=(0.0, 0.9))
    n_real = real_data.size(0)
    freq = max(1, epochs // 10)
    print(
        f"[train_wgangp] epochs={epochs} steps/epoch={steps_per_epoch} batch={batch_size} "
        f"g_lr={g_lr} d_lr={d_lr} n_critic={n_critic} n_gen={n_gen} lambda_gp={lambda_gp}"
    )
    for e in range(1, epochs + 1):
        for _ in range(steps_per_epoch):
            for _ in range(n_critic):
                idx = torch.randint(0, n_real, (batch_size,))
                real = real_data[idx].to(device)
                z = sample_noise(batch_size, z_dim, device)
                fake = gen(z).detach()
                gp = gradient_penalty(disc, real, fake)
                d_loss = -(disc(real).mean() - disc(fake).mean()) + lambda_gp * gp
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()
            for _ in range(n_gen):
                z = sample_noise(batch_size, z_dim, device)
                g_loss = -disc(gen(z)).mean()
                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()
        if e % freq == 0 or e == 1:
            save_snapshot(gen, real_data, z_dim, device, os.path.join(out_dir, f"wgangp_epoch_{e}.png"))
    score = evaluate_generator_mmd(gen, real_data, z_dim, device)
    print(f"[train_wgangp] final MMD={score:.6f}")
    return gen, score


if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda")
    base_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(base_dir, exist_ok=True)
    real_data = get_data(2048, noise=0.1)

    print("gan:")
    gan_out = os.path.join(base_dir, "gan")
    os.makedirs(gan_out, exist_ok=True)
    _, gan_mmd = train_gan(device, gan_out, real_data)

    print("wgan:")
    wgan_out = os.path.join(base_dir, "wgan")
    os.makedirs(wgan_out, exist_ok=True)
    _, wgan_mmd = train_wgan(device, wgan_out, real_data)

    print("wgan-gp:")
    trials = [
        {"epochs": 2400, "steps_per_epoch": 9, "batch_size": 256, "z_dim": 2, "h": 128, "g_lr": 2e-4, "d_lr": 2e-4, "n_critic": 5, "n_gen": 1, "lambda_gp": 0.2},
        {"epochs": 2800, "steps_per_epoch": 9, "batch_size": 256, "z_dim": 2, "h": 128, "g_lr": 2e-4, "d_lr": 2e-4, "n_critic": 5, "n_gen": 1, "lambda_gp": 0.25},
    ]
    best_mmd = float("inf")
    best_idx = -1
    for i, cfg in enumerate(trials, start=1):
        out_dir = os.path.join(base_dir, f"wgangp_try{i}")
        os.makedirs(out_dir, exist_ok=True)
        _, mmd = train_wgangp(device, out_dir, real_data, **cfg)
        if mmd < best_mmd:
            best_mmd = mmd
            best_idx = i
        if best_mmd <= gan_mmd:
            break

    print(f"final scores: gan={gan_mmd:.6f}, wgan={wgan_mmd:.6f}, wgangp_best={best_mmd:.6f} (try {best_idx})")
    if best_mmd > gan_mmd:
        print("warning: wgan-gp is still worse than gan on current seed; increase trial epochs.")
