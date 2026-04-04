import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. SHAPE DATA (32 points: Anchor, Control, Anchor...) ---


def get_circle(n_points=32):
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    return torch.tensor(np.stack([np.cos(t), np.sin(t)], axis=1), dtype=torch.float32)


def get_twisted_test_pose(n_points=32):
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    # The flip: Y is inverted. Lerp would collapse this at t=0.5.
    x = 1.3 * np.cos(t)
    y = -1.3 * np.sin(t)
    return torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)


def get_soft_rect(n_curves=16, power=8):
    # 1. Start with linear spacing
    t_linear = np.linspace(0, 2*np.pi, n_curves * 2, endpoint=False)

    # 2. Warp t to cluster points at corners (pi/4, 3pi/4, etc.)
    # The sin(4t) term creates 4 "bunches" of points
    # Adjust 0.1 to a higher value (like 0.15) for even tighter corners
    t = 0.5 + t_linear - 0.1 * np.sin(4 * t_linear)

    # 3. Standard Superellipse Formula
    cos_t = np.cos(t)
    sin_t = np.sin(t)

    # Increase power to make the flat sides flatter
    denom = (np.abs(cos_t)**power + np.abs(sin_t)**power)**(1/power)
    r = 1.0 / denom

    x = r * cos_t
    y = r * sin_t

    return torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)

# --- 2. THE FLOW MODEL ---


class FlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(65, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 64)
        )

    def forward(self, x, t):
        # If x is [64], make it [1, 64]
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # If t is a scalar or [1], make it [Batch, 1]
        if t.dim() == 0:
            t = t.view(1, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)

        # If x and t have different batch sizes (usually 1 vs N),
        # expand t to match x
        if t.size(0) != x.size(0):
            t = t.expand(x.size(0), -1)

        return self.net(torch.cat([x, t], dim=-1))


def get_signed_area(p):
    if p.dim() == 2:
        p = p.unsqueeze(0)
    x, y = p[:, :, 0], p[:, :, 1]
    return 0.5 * torch.sum(x * torch.roll(y, -1, 1) - torch.roll(x, -1, 1) * y, dim=1)

# --- 3. TRAINING --


def get_bezier_points(p0, p1, p2, num_steps=15):
    t = np.linspace(0, 1, num_steps)
    points = (1-t)[:, None]**2 * p0 + 2*(1-t)[:, None] * \
        t[:, None] * p1 + t[:, None]**2 * p2
    return points


def save_bezier_gif(model, x0, steps=60):
    print("🎬 Rendering Smooth Bezier GIF...")
    fig, ax = plt.subplots(figsize=(6, 6))
    line, = ax.plot([], [], 'b-', lw=2.5)
    dots, = ax.plot([], [], 'ro', markersize=3, alpha=0.2)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    # 1. Generate Raw Point Data
    current_x = x0.unsqueeze(0)
    dt = 1.0 / steps
    animation_data = []
    for i in range(steps + 1):
        t_val = torch.tensor([[i * dt]])
        pts = current_x.detach().numpy().reshape(32, 2)
        animation_data.append(pts)
        if i < steps:
            with torch.no_grad():
                v = model(current_x, t_val)
                current_x = current_x + v * dt

    def update(frame):
        pts = animation_data[frame]
        all_curve_pts = []
        for i in range(0, len(pts), 2):
            p0, p1 = pts[i], pts[i+1]
            p2 = pts[(i+2) % len(pts)]
            all_curve_pts.append(get_bezier_points(p0, p1, p2))

        curve_data = np.concatenate(all_curve_pts)
        line.set_data(curve_data[:, 0], curve_data[:, 1])
        dots.set_data(pts[:, 0], pts[:, 1])
        return line, dots

    ani = FuncAnimation(fig, update, frames=len(animation_data), blit=True)
    ani.save('morph_bezier.gif', writer='pillow', fps=30)
    print("✅ Done! Saved as morph_bezier.gif")


def train_flow_model_slow_cook(poses, steps=60000):
    device = torch.device("cuda")
    model = FlowModel().to(device)

    # 1. LOWER LEARNING RATE (The 'Slow-Cook')
    # 5e-5 is much safer for high-tension physics
    optimizer = torch.optim.Adam(
        model.parameters(), lr=5e-5, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps)

    p0 = poses[0].to(device)  # [32, 2]
    p1 = poses[1].to(device)  # [32, 2]
    rest_lengths = torch.norm(p0 - torch.roll(p0, 1, 0), dim=1)

    print(f"🔥 Slow-Cooking Physics (60k steps)...")
    for i in range(steps + 1):
        optimizer.zero_grad()
        t = torch.rand(1, device=device)

        # We use a bit of noise (0.002) to help the model explore 'around' the path
        xt = (1.0 - t) * p0 + t * p1 + torch.randn_like(p0) * 0.002
        xt_flat = xt.flatten().unsqueeze(0)

        v_pred = model(xt_flat, t.view(1, 1))
        v_reshaped = v_pred.view(32, 2)

        # --- THE REFINED PHYSICS ---

        # A. FLOW MATCHING (Baseline)
        target_v = (p1 - p0).flatten().unsqueeze(0)
        loss_flow = torch.mean((v_pred - target_v)**2)

        # B. STIFFNESS (Lowered weight to prevent 'Explosions')
        dt = 0.02  # Smaller look-ahead for stability
        p_next = xt + v_reshaped * dt
        current_lengths = torch.norm(p_next - torch.roll(p_next, 1, 0), dim=1)
        loss_stiff = torch.mean((current_lengths - rest_lengths)**2) * 500.0

        # C. AREA CONSERVATION
        area_now = get_signed_area(p_next)
        loss_area = torch.mean(
            (area_now / get_signed_area(p0) - 1.0)**2) * 50.0

        # D. RADIUS PRESERVATION (Center of Mass)
        center = torch.mean(xt, dim=0)
        dist_to_center = torch.norm(p_next - center, dim=1)
        rest_dist = torch.norm(p0 - center, dim=1)
        loss_radius = torch.mean((dist_to_center - rest_dist)**2) * 200.0

        # TOTAL LOSS
        total_loss = (loss_flow * 0.5) + loss_stiff + loss_area + loss_radius

        total_loss.backward()

        # 2. AGGRESSIVE GRADIENT CLIPPING
        # This is the 'Shield' that stops the explosions.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.05)

        optimizer.step()
        scheduler.step()

        if i % 10000 == 0:
            print(
                f"Step {i:5} | Area: {area_now.item():.4f} | Stiff: {loss_stiff.item():.4f}")

    return model.cpu(), p0.flatten().cpu()


if __name__ == "__main__":
    poses = [get_circle(), get_twisted_test_pose()]
    model, x0 = train_flow_model_slow_cook(poses)
    save_bezier_gif(model, x0)
