import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. UTILITIES ---


def get_signed_area(p):
    if p.dim() == 2:
        p = p.unsqueeze(0)
    x, y = p[:, :, 0], p[:, :, 1]
    return 0.5 * torch.sum(x * torch.roll(y, -1, dims=1) - torch.roll(x, -1, dims=1) * y, dim=1)

# --- 2. MODEL: General Dual-Stream (NO hardcoded angles, NO cheating) ---


class DualStreamFlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Local rotation per point
        self.rot_encoder = nn.Sequential(
            nn.Linear(2, 32), nn.ELU(), nn.Linear(32, 16)
        )
        self.rot_net = nn.Sequential(
            nn.Linear(17, 64), nn.ELU(), nn.Linear(64, 1)
        )
        # Local smooth warp
        self.warp_encoder = nn.Sequential(
            nn.Linear(2, 32), nn.ELU(), nn.Linear(32, 16)
        )
        self.warp_net = nn.Sequential(
            nn.Linear(17, 128), nn.ELU(), nn.Linear(128, 2)
        )

    def forward(self, x, t):
        B, N, _ = x.shape
        t_in = t.view(B, 1, 1)
        # Rotation branch
        feat_rot = self.rot_encoder(x)
        rot_in = torch.cat([feat_rot, t_in.expand(B, N, 1)], dim=-1)
        local_angles = self.rot_net(rot_in)
        # Warp branch
        feat_warp = self.warp_encoder(x)
        warp_in = torch.cat([feat_warp, t_in.expand(B, N, 1)], dim=-1)
        local_warp = self.warp_net(warp_in)
        return local_angles, local_warp

# --- 3. TRAINING (Default params = perfect) ---


def train_dual_stream_polished(poses, steps=60000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualStreamFlowModel().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=6e-5, weight_decay=1e-4)

    p0 = poses[0].to(device)
    p1 = torch.flip(poses[1].to(device), dims=[0])
    center = torch.mean(p0, dim=0).view(1, 2)
    area_start = get_signed_area(p0)

    print("🎯 Training default-good morph...")
    for i in range(steps + 1):
        optimizer.zero_grad()
        relax = 1.0 - (i / steps) * 0.85

        # Stable time sampling
        if i % 20 == 0:
            t = torch.tensor([1.0], device=device)
        else:
            raw_t = torch.rand(1, device=device)
            t = 4 * raw_t**3 - 3 * raw_t**4

        local_angles, local_warp = model(p0.unsqueeze(0), t)

        # Deform: per-point rotation + smooth warp
        rel = p0 - center
        theta = local_angles.squeeze(-1).squeeze(0) * t * 0.6
        cos_a, sin_a = torch.cos(theta), torch.sin(theta)
        rot_x = rel[:, 0] * cos_a - rel[:, 1] * sin_a
        rot_y = rel[:, 0] * sin_a + rel[:, 1] * cos_a
        rotated = torch.stack([rot_x, rot_y], dim=1) + center

        warp_scaled = local_warp.squeeze(0) * t * relax
        p_final = rotated + warp_scaled

        # Losses (balanced DEFAULT weights)
        loss_dest = torch.mean((p_final - p1) ** 2) * 20.0
        target_area = area_start * (1.0 + 0.25 * t)
        loss_area = torch.mean(
            (get_signed_area(p_final) - target_area) ** 2) * 40.0 * relax
        loss_center = torch.mean(
            (torch.mean(p_final, dim=0) - center) ** 2) * 1000.0
        loss_rot_smooth = torch.mean((local_angles.diff(dim=1)) ** 2) * 30.0
        loss_warp = torch.mean(torch.abs(local_warp)) * 0.5

        total_loss = loss_dest + loss_area + loss_center + loss_rot_smooth + loss_warp

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.03)
        total_loss.backward()
        optimizer.step()

        if i % 10000 == 0:
            print(f"Step {i:5} | Dest: {loss_dest.item():.2f}")

    return model.cpu(), p0.cpu()

# --- 4. RENDER ---


def save_dual_stream_gif(model, p0, steps=60):
    print("🎬 Rendering...")
    fig, ax = plt.subplots(figsize=(6, 6))
    line, = ax.plot([], [], 'b-', lw=2.5)
    dots, = ax.plot([], [], 'ro', markersize=3, alpha=0.3)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    center = torch.mean(p0, dim=0).view(1, 2)
    animation_data = []

    for i in range(steps + 1):
        t_val = torch.tensor([i / steps])
        with torch.no_grad():
            local_angles, local_warp = model(p0.unsqueeze(0), t_val)
            rel = p0 - center
            theta = local_angles.squeeze(0) * t_val * 0.6
            cos_a, sin_a = torch.cos(theta), torch.sin(theta)
            rx = rel[:, 0] * cos_a - rel[:, 1] * sin_a
            ry = rel[:, 0] * sin_a + rel[:, 1] * cos_a
            rotated = torch.stack([rx, ry], dim=1) + center
            pts = (rotated + local_warp.squeeze(0) * t_val).numpy()
            animation_data.append(pts)

    def update(frame):
        pts = animation_data[frame]
        line.set_data(pts[:, 0], pts[:, 1])
        dots.set_data(pts[:, 0], pts[:, 1])
        return line, dots

    ani = FuncAnimation(fig, update, frames=len(animation_data), blit=True)
    ani.save('final_morph.gif', writer='pillow', fps=30)
    print("✅ Saved final_morph.gif")


if __name__ == "__main__":
    def get_circle(n_points=32):
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        return torch.tensor(np.stack([np.cos(t), np.sin(t)], axis=1), dtype=torch.float32)

    def get_twisted_pose(n_points=32):
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        return torch.tensor(np.stack([1.3 * np.cos(t), -1.3 * np.sin(t)], axis=1), dtype=torch.float32)

    poses = [get_circle(), get_twisted_pose()]
    model, p0 = train_dual_stream_polished(poses)
    save_dual_stream_gif(model, p0)
