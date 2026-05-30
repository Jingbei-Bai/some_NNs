import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def generate_spiral(n_points=1000, noise=0.05, turns=3.0, radius_scale=1.0, seed=42):
    if seed is not None:
        np.random.seed(seed)
    theta = np.random.rand(n_points) * (turns * 2.0 * math.pi)
    r = radius_scale * theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    x += np.random.randn(n_points) * noise
    y += np.random.randn(n_points) * noise
    pts = np.stack([x, y], axis=1).astype(np.float32)
    return torch.from_numpy(pts)


class PointCloudDataset(Dataset):
    def __init__(self, points: torch.Tensor):
        self.points = points

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        return self.points[idx]


def to_numpy(points):
    return points.detach().cpu().numpy()


class AEEncoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, latent_dim))

    def forward(self, x):
        return self.net(x)


class AEDecoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, z):
        return self.net(z)


def train_autoencoder(points: torch.Tensor, latent_dim=2, seq_len=32, n_samples=1500, epochs=80, batch_size=128, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs, targets = build_sequence_dataset(points, seq_len=seq_len, n_samples=n_samples)
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    enc = AEEncoder(latent_dim).to(device)
    dec = AEDecoder(latent_dim).to(device)
    optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(1, epochs + 1):
        enc.train(); dec.train(); train_loss = 0.0; train_total = 0
        for src, tgt in loader:
            src = src.to(device)
            # flatten points to apply point-wise encoder/decoder
            b = src.size(0)
            flat = src.view(b * seq_len, 2)
            z = enc(flat)
            recon_flat = dec(z)
            recon = recon_flat.view(b, seq_len, 2)
            loss = loss_fn(recon, tgt.to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item() * b
            train_total += b
        train_loss = train_loss / train_total
        if epoch % 10 == 0 or epoch == 1:
            print(f"AE(seq) epoch {epoch}/{epochs} loss={train_loss:.6f}")
    enc.eval(); dec.eval()
    with torch.no_grad():
        sample_src = inputs[:4].to(device)
        b = sample_src.size(0)
        flat = sample_src.view(b * seq_len, 2)
        recon = dec(enc(flat)).view(b, seq_len, 2).cpu()
    mse = nn.MSELoss()(recon, targets[:4]).item(); print(f"AE(seq) sample MSE: {mse:.6f}")
    os.makedirs("outputs", exist_ok=True)
    for i in range(min(3, recon.size(0))):
        src_np = to_numpy(sample_src[i].cpu()); pred_np = to_numpy(recon[i]); tgt_np = to_numpy(targets[i])
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].scatter(src_np[:, 0], src_np[:, 1], s=6); axs[0].set_title("input sample")
        axs[1].scatter(pred_np[:, 0], pred_np[:, 1], s=6); axs[1].set_title("AE(seq) recon")
        axs[2].scatter(tgt_np[:, 0], tgt_np[:, 1], s=6); axs[2].set_title("target ordered")
        plt.tight_layout(); out = os.path.join("outputs", f"ae_seq_{i}.png"); plt.savefig(out); plt.close(); print("Saved", out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(0)
        return x + self.pe[:seq_len]

def build_sequence_dataset(points: torch.Tensor, seq_len=32, n_samples=1000, seed=42):
    if seed is not None:
        np.random.seed(seed)
    N = points.shape[0]
    inputs = []
    targets = []
    pts_np = points.numpy()
    for i in range(n_samples):
        idx = np.random.choice(N, size=seq_len, replace=False)
        subset = pts_np[idx]
        angles = np.arctan2(subset[:, 1], subset[:, 0])
        order = np.argsort(angles)
        sorted_subset = subset[order]
        shuffle_order = np.arange(seq_len)
        np.random.shuffle(shuffle_order)
        shuffled = subset[shuffle_order]
        inputs.append(shuffled); targets.append(sorted_subset)
    inputs = torch.tensor(np.stack(inputs, axis=0), dtype=torch.float32)
    targets = torch.tensor(np.stack(targets, axis=0), dtype=torch.float32)
    return inputs, targets

class TransformerAutoEncoder(nn.Module):
    def __init__(self, seq_len, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.input_proj = nn.Linear(2, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, 2)

    def forward(self, src):
        src_emb = self.input_proj(src).permute(1, 0, 2)
        src_pe = self.pos_enc(src_emb)
        mem = self.encoder(src_pe)
        out = mem.permute(1, 0, 2)
        return self.out(out)

def train_transformer(points: torch.Tensor, seq_len=32, n_samples=1000, epochs=40, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs, _ = build_sequence_dataset(points, seq_len=seq_len, n_samples=n_samples)
    targets = inputs.clone()
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = TransformerAutoEncoder(seq_len=seq_len, d_model=64, nhead=4, num_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(1, epochs + 1):
        model.train(); train_loss = 0.0; train_total = 0
        for src, tgt in loader:
            src = src.to(device); tgt = tgt.to(device)
            pred = model(src)
            loss = loss_fn(pred, tgt)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            bs = src.size(0); train_loss += loss.item() * bs; train_total += bs
        train_loss = train_loss / train_total
        if epoch % 10 == 0 or epoch == 1:
            print(f"Transformer epoch {epoch}/{epochs} train loss {train_loss:.6f}")
    model.eval()
    with torch.no_grad():
        sample_src = inputs[:4].to(device); sample_tgt = targets[:4]; pred = model(sample_src).cpu()
    mse = nn.MSELoss()(pred, sample_tgt).item(); print(f"Transformer recon final MSE: {mse:.6f}")
    os.makedirs("outputs", exist_ok=True)
    for i in range(min(3, pred.size(0))):
        src_np = to_numpy(sample_src[i].cpu()); pred_np = to_numpy(pred[i]); tgt_np = to_numpy(sample_tgt[i])
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].scatter(src_np[:, 0], src_np[:, 1], s=6); axs[0].set_title("input sample")
        axs[1].scatter(pred_np[:, 0], pred_np[:, 1], s=6); axs[1].set_title("transformer recon")
        axs[2].scatter(tgt_np[:, 0], tgt_np[:, 1], s=6); axs[2].set_title("target (same as input)")
        plt.tight_layout(); out = os.path.join("outputs", f"transformer_recon_{i}.png"); plt.savefig(out); plt.close(); print("Saved", out)

def train_global_recon(points: torch.Tensor, ae_epochs=80, trans_epochs=80, batch_size=256, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("outputs", exist_ok=True)

    # AE point-wise training (original simple recon)
    dataset = PointCloudDataset(points)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    enc = AEEncoder(latent_dim=2).to(device)
    dec = AEDecoder(latent_dim=2).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(1, ae_epochs + 1):
        enc.train(); dec.train(); running = 0.0; total = 0
        for batch in loader:
            batch = batch.to(device)
            z = enc(batch)
            recon = dec(z)
            loss = loss_fn(recon, batch)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * batch.size(0); total += batch.size(0)
        if epoch % 10 == 0 :
            print(f"Global AE epoch {epoch}/{ae_epochs} loss={running/total:.6f}")
    enc.eval(); dec.eval()
    with torch.no_grad():
        pts = points.to(device)
        z = enc(pts)
        recon_ae = dec(z).cpu()
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].scatter(points[:, 0].numpy(), points[:, 1].numpy(), s=6); ax[0].set_title("global AE original")
    ax[1].scatter(recon_ae[:, 0].numpy(), recon_ae[:, 1].numpy(), s=6); ax[1].set_title("global AE recon")
    plt.tight_layout(); out = os.path.join("outputs", "global_ae_recon.png"); plt.savefig(out); plt.close(); print("Saved", out)

    # Transformer global sequence recon
    N = points.shape[0]
    model = TransformerAutoEncoder(seq_len=N, d_model=64, nhead=4, num_layers=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    src = points.unsqueeze(0).to(device)
    for epoch in range(1, trans_epochs + 1):
        model.train(); opt.zero_grad()
        pred = model(src)
        loss = loss_fn(pred, src)
        loss.backward(); opt.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"Global Transformer epoch {epoch}/{trans_epochs} loss={loss.item():.6f}")
    model.eval()
    with torch.no_grad():
        pred_t = model(src).squeeze(0).cpu()
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].scatter(points[:, 0].numpy(), points[:, 1].numpy(), s=6); ax[0].set_title("global Transformer original")
    ax[1].scatter(pred_t[:, 0].numpy(), pred_t[:, 1].numpy(), s=6); ax[1].set_title("global Transformer recon")
    plt.tight_layout(); out = os.path.join("outputs", "global_transformer_recon.png"); plt.savefig(out); plt.close(); print("Saved", out)

if __name__ == "__main__":
    pts = generate_spiral(n_points=2000, noise=0.08, turns=3.5, radius_scale=0.4)
    train_global_recon(pts, ae_epochs=200, trans_epochs=200, batch_size=256)
    train_autoencoder(pts, latent_dim=2, seq_len=32, n_samples=1500, epochs=200, batch_size=256)
    train_transformer(pts, seq_len=32, n_samples=1500, epochs=200, batch_size=64)

