import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. SHAPE DATA ---


def get_circle(n_points=32):
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    return torch.tensor(np.stack([np.cos(t), np.sin(t)], axis=1), dtype=torch.float32)


def get_twisted_test_pose(n_points=32):
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    return torch.tensor(np.stack([1.3 * np.cos(t), -1.3 * np.sin(t)], axis=1), dtype=torch.float32)

# --- 2. THE LATENT FLOW ARCHITECTURE ---


class LatentFlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: 2D -> 4D Latent Space
        self.encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ELU(),
            nn.Linear(16, 4)
        )
        # Flow Processor: Operating on the 4D manifold + time
        self.flow_net = nn.Sequential(
            nn.Linear(4 + 1, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 4)
        )
        # Decoder: 4D -> 2D
        self.decoder = nn.Linear(4, 2)

    def forward(self, x, t):
        # x shape: [Batch, 32, 2]
        batch_size = x.size(0)

        # 1. Lift every point to 4D
        h = self.encoder(x.view(-1, 2)).view(batch_size, 32, 4)

        # 2. Predict 4D Velocity
        t_stack = t.view(batch_size, 1, 1).expand(-1, 32, -1)
        v_input = torch.cat([h, t_stack], dim=-1)
        v_latent = self.flow_net(v_input)

        # 3. Project 4D Velocity back to 2D Delta
        v_2d = self.decoder(v_latent.view(-1, 4)).view(batch_size, 32, 2)
        return v_2d.flatten(1)


def get_signed_area(p):
    if p.dim() == 2:
        p = p.unsqueeze(0)
    x, y = p[:, :, 0], p[:, :, 1]
    return 0.5 * torch.sum(x * torch.roll(y, -1, 1) - torch.roll(x, -1, 1) * y, dim=1)

# --- 3. TRAINING (The Latent Lift) ---


def train_latent_flow_hybrid(poses, steps=60000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LatentFlowModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    p0 = poses[0].to(device)
    p1 = poses[1].to(device)
    center = torch.mean(p0, dim=0)

    print(f"🧬 Training Hybrid Morph (Rigid Start -> Shape-Shifting End)...")
    for i in range(steps + 1):
        optimizer.zero_grad()
        t = torch.rand(1, device=device)

        # --- 1. THE RIGID ROTATION COMPONENT ---
        angle = t * 3.14159
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        rel_p0 = p0 - center
        xt_rigid = torch.stack([
            rel_p0[:, 0] * cos_a - rel_p0[:, 1] * sin_a,
            rel_p0[:, 0] * sin_a + rel_p0[:, 1] * cos_a
        ], dim=1) + center

        # --- 2. THE HYBRID VELOCITY ---
        # Pure rotation velocity
        v_rigid = torch.stack([
            -(xt_rigid[:, 1] - center[1]) * 3.14159,
            (xt_rigid[:, 0] - center[0]) * 3.14159
        ], dim=1)

        # Pure destination pull (where do I need to go to hit p1?)
        v_dest = (p1 - xt_rigid) / (1.0 - t + 1e-6)

        # THE MAGIC BLEND:
        # We use a sigmoid-like blend so it's rigid at the start (t=0)
        # and aggressively shape-shifts at the end (t=1)
        # Sharp transition at 70% of the way
        blend = torch.sigmoid(10 * (t - 0.5))
        target_v = (1.0 - blend) * v_rigid + blend * v_dest

        # --- 3. THE TRAINING ---
        # We train on the 'Rigid' path samples so the model never sees a squash
        v_pred = model(xt_rigid.unsqueeze(0), t.view(1, 1))

        loss_flow = torch.mean((v_pred.view(32, 2) - target_v)**2)

        # Hard constraint: Even when shape-shifting, keep the segments stiff!
        dt = 0.02
        p_next = xt_rigid + v_pred.view(32, 2) * dt
        len_now = torch.norm(p_next - torch.roll(p_next, 1, 0), dim=1)
        len_rest = torch.norm(p0 - torch.roll(p0, 1, 0), dim=1)
        loss_stiff = torch.mean((len_now - len_rest)**2) * 200.0

        (loss_flow + loss_stiff).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
        optimizer.step()

        if i % 10000 == 0:
            print(
                f"Step {i:5} | Blend: {blend.item():.2f} | Stiff: {loss_stiff.item():.6f}")

    return model.cpu(), p0.flatten().cpu()


def get_bezier_points(p0, p1, p2, num_steps=15):
    t = np.linspace(0, 1, num_steps)
    return (1-t)[:, None]**2 * p0 + 2*(1-t)[:, None] * t[:, None] * p1 + t[:, None]**2 * p2


def save_bezier_gif(model, x0, steps=60):
    print("🎬 Rendering Latent Morph...")
    fig, ax = plt.subplots(figsize=(6, 6))
    line, = ax.plot([], [], 'b-', lw=2)
    dots, = ax.plot([], [], 'ro', markersize=3, alpha=0.3)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    current_x = x0.view(1, 32, 2)
    dt = 1.0 / steps
    animation_data = []

    for i in range(steps + 1):
        t_val = torch.tensor([i * dt])
        pts = current_x.squeeze().detach().numpy()
        animation_data.append(pts)
        if i < steps:
            with torch.no_grad():
                v_flat = model(current_x, t_val)
                current_x = current_x + v_flat.view(1, 32, 2) * dt

    def update(frame):
        pts = animation_data[frame]
        all_curve_pts = [get_bezier_points(
            pts[i], pts[i+1], pts[(i+2) % len(pts)]) for i in range(0, len(pts), 2)]
        curve_data = np.concatenate(all_curve_pts)
        line.set_data(curve_data[:, 0], curve_data[:, 1])
        dots.set_data(pts[:, 0], pts[:, 1])
        return line, dots

    ani = FuncAnimation(fig, update, frames=len(animation_data), blit=True)
    ani.save('morph_latent_4d.gif', writer='pillow', fps=30)
    print("✅ Done! Check morph_latent_4d.gif")


if __name__ == "__main__":
    poses = [get_circle(), get_twisted_test_pose()]
    model, x0 = train_latent_flow_hybrid(poses)
    save_bezier_gif(model, x0)
