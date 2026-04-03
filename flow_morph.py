from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt


def plot_morph(x0, x1, model, steps=5):
    plt.figure(figsize=(8, 8))

    # Reshape for plotting
    x0_p = x0.reshape(32, 2).numpy()
    x1_p = x1.reshape(32, 2).numpy()

    # Plot start and end shapes
    plt.scatter(x0_p[:, 0], x0_p[:, 1], color='blue',
                label='Circle (Start)', alpha=0.5)
    plt.scatter(x1_p[:, 0], x1_p[:, 1], color='red',
                label='Squircle (End)', alpha=0.5)

    # Plot the "velocity" vectors from the model at t=0.5
    t_mid = torch.tensor([[0.5]])
    x_mid = (0.5 * x0 + 0.5 * x1).unsqueeze(0)
    v = model(x_mid, t_mid).detach().numpy().reshape(32, 2)
    x_mid_p = x_mid.detach().numpy().reshape(32, 2)

    plt.quiver(x_mid_p[:, 0], x_mid_p[:, 1], v[:, 0],
               v[:, 1], color='green', label='Flow Velocity')

    plt.legend()
    plt.axis('equal')
    plt.title("Flow Matching: Shape Morphing Vectors")

    # Save instead of show to avoid Docker GUI issues
    plt.savefig("morph_plot.png")
    print("Plot saved as morph_plot.png")

# Call this in your __main__
# plot_morph(x0, x1, model)


# --- 1. SHAPE GENERATION ---
def get_circle(n_curves=16):
    # 32 points total: alternating Anchor and Control
    t = np.linspace(0, 2*np.pi, n_curves * 2, endpoint=False)
    # Standard circle radius 1.0
    x = np.cos(t)
    y = np.sin(t)
    return torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)


def get_soft_rect(n_curves=16, power=8):
    # 1. Start with linear spacing
    t_linear = np.linspace(0, 2*np.pi, n_curves * 2, endpoint=False)

    # 2. Warp t to cluster points at corners (pi/4, 3pi/4, etc.)
    # The sin(4t) term creates 4 "bunches" of points
    # Adjust 0.1 to a higher value (like 0.15) for even tighter corners
    t = t_linear - 0.1 * np.sin(4 * t_linear)

    # 3. Standard Superellipse Formula
    cos_t = np.cos(t)
    sin_t = np.sin(t)

    # Increase power to make the flat sides flatter
    denom = (np.abs(cos_t)**power + np.abs(sin_t)**power)**(1/power)
    r = 1.0 / denom

    x = r * cos_t
    y = r * sin_t

    return torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)


class FlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 32 points * 2 coords + 1 time = 65
        self.net = nn.Sequential(
            nn.Linear(65, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 64)  # Output: velocity for 32 points (dx, dy)
        )

    def forward(self, x, t):
        # Ensure t has the same batch dimension as x
        # If x is [Batch, 64] and t is [1], we need t to be [Batch, 1]
        if t.dim() == 1:
            t = t.unsqueeze(0)

        # This joins them along the last axis: [Batch, 64 + 1]
        return self.net(torch.cat([x, t], dim=-1))

# --- 3. TRAINING ---


# --- 3. TRAINING ---
def train():
    x0 = get_circle().flatten()
    x1 = get_soft_rect().flatten()

    model = FlowModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training Flow Matching Model...")
    for i in range(2000):
        optimizer.zero_grad()
        t = torch.rand(1, 1)  # Shape: [1, 1]

        # xt inherits the [1, 64] shape from t
        xt = (1 - t) * x0 + t * x1

        target_v = x1 - x0

        # REMOVE .unsqueeze(0) HERE
        v_pred = model(xt, t)

        # Ensure target_v matches the shape of v_pred [1, 64]
        loss = torch.mean((v_pred - target_v.unsqueeze(0))**2)

        loss.backward()
        optimizer.step()

        if i % 500 == 0:
            print(f"Step {i}, Loss: {loss.item():.6f}")

    return model, x0, x1
# --- 4. INFERENCE (The Animation) ---


def generate_animation(model, x0, steps=50):
    print("Generating Morph...")
    current_x = x0.unsqueeze(0)
    dt = 1.0 / steps
    frames = []

    for i in range(steps + 1):
        t = torch.tensor([[i * dt]])
        frames.append(current_x.detach().numpy().reshape(32, 2).tolist())

        # Euler Integration
        if i < steps:
            v = model(current_x, t)
            current_x = current_x + v * dt

    return frames


def save_gif(animation_data):
    fig, ax = plt.subplots(figsize=(6, 6))
    line, = ax.plot([], [], 'o-', lw=2)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    def update(frame):
        points = np.array(frame)
        line.set_data(points[:, 0], points[:, 1])
        return line,

    ani = FuncAnimation(fig, update, frames=animation_data, blit=True)
    ani.save('morph.gif', writer='pillow', fps=30)
    print("Animation saved as morph.gif")


def get_bezier_points(p0, p1, p2, num_steps=10):
    """Calculates points along a quadratic Bezier curve."""
    t = np.linspace(0, 1, num_steps)
    # Quadratic Bezier Formula: (1-t)^2*P0 + 2(1-t)t*P1 + t^2*P2
    points = (1-t)[:, None]**2 * p0 + 2*(1-t)[:, None] * \
        t[:, None] * p1 + t[:, None]**2 * p2
    return points


def save_bezier_gif(animation_data):
    fig, ax = plt.subplots(figsize=(6, 6))
    line, = ax.plot([], [], 'b-', lw=2)  # The smooth curve
    dots, = ax.plot([], [], 'ro', markersize=4,
                    alpha=0.3)  # The raw data points

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    def update(frame):
        pts = np.array(frame)
        all_curve_pts = []

        # Loop through points in steps of 2 (Anchor, Control)
        for i in range(0, len(pts), 2):
            p0 = pts[i]
            p1 = pts[i+1]
            p2 = pts[(i+2) % len(pts)]  # Next anchor (wrap around)

            segment = get_bezier_points(p0, p1, p2)
            all_curve_pts.append(segment)

        curve_data = np.concatenate(all_curve_pts)
        line.set_data(curve_data[:, 0], curve_data[:, 1])
        dots.set_data(pts[:, 0], pts[:, 1])
        return line, dots

    from matplotlib.animation import FuncAnimation
    ani = FuncAnimation(fig, update, frames=animation_data, blit=True)
    ani.save('morph_bezier.gif', writer='pillow', fps=30)


if __name__ == "__main__":
    model, x0, x1 = train()
    animation_data = generate_animation(model, x0)
    plot_morph(x0, x1, model)
    save_gif(animation_data)
    save_bezier_gif(animation_data)
    # Save to JSON for your 2D Curve Library
    with open("morph_animation.json", "w") as f:
        json.dump(animation_data, f)
    print("Done! Animation saved to morph_animation.json")
