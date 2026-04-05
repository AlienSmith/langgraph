import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('Agg')

# ==============================
# BÉZIER EVALUATION (Unchanged)
# ==============================


def get_bezier(p0, p1, p2, steps=12):
    t = np.linspace(0, 1, steps)[:, None]
    return (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2


def bezier_line(pts, steps=12):
    out = []
    n = len(pts)
    for i in range(n):
        out.append(get_bezier(pts[i], pts[(i+1) % n], pts[(i+2) % n], steps))
    return np.vstack(out)

# ==============================
# PREPROCESSING (Asset Pipeline)
# ==============================


def resample_by_arc_length(pts, n_out):
    pts_closed = np.vstack([pts, pts[0]])
    dists = np.linalg.norm(np.diff(pts_closed, axis=0), axis=1)
    cumulative = np.concatenate([[0], np.cumsum(dists)])
    total_len = cumulative[-1]
    target_dists = np.linspace(0, total_len, n_out, endpoint=False)
    new_x = np.interp(target_dists, cumulative, pts_closed[:, 0])
    new_y = np.interp(target_dists, cumulative, pts_closed[:, 1])
    return np.c_[new_x, new_y]


def prepare_shape(pts_raw, n_points=32):
    pts = resample_by_arc_length(pts_raw, n_points)
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * np.sum(x[:-1]*y[1:] - x[1:]*y[:-1]) + \
        0.5*(x[-1]*y[0] - x[0]*y[-1])
    if area < 0:  # Clockwise -> Reverse
        pts = pts[::-1]
        pts = np.roll(pts, 1, axis=0)
    return pts

# ==============================
# RUST-LIKE MORPHER STRUCT
# ==============================


class FourierShapeMorpher:
    """
    Matches Rust integration pattern:
    1. `__init__` -> `fn new()`: Precomputes FFT & alignment (runs once at load)
    2. `evaluate` -> `fn sample(t)`: Stateless, O(N) per frame, safe for XPBD/Compute
    """

    def __init__(self, pts_A_raw, pts_B_raw, n_points=32):
        # [Rust] Asset loading & topology enforcement
        self.A = prepare_shape(pts_A_raw, n_points)
        self.B = prepare_shape(pts_B_raw, n_points)
        self.n_points = n_points

        # [Rust] Precompute spectral descriptors (baked into game binary)
        self._precompute()

    def _precompute(self):
        N = self.n_points
        Az = self.A[:, 0] + 1j * self.A[:, 1]
        Bz = self.B[:, 0] + 1j * self.B[:, 1]

        # Store centroids for translation interpolation
        self.centroid_A = Az.mean()
        self.centroid_B = Bz.mean()

        # Center shapes
        A_c = Az - self.centroid_A
        B_c = Bz - self.centroid_B

        # Scale normalization (match B's energy to A's)
        self.scale = np.linalg.norm(A_c) / (np.linalg.norm(B_c) + 1e-8)
        B_c *= self.scale

        # Optimal cyclic shift (phase alignment)
        # [Rust] Precompute `best_k` at compile/load time
        best_err, best_k = np.inf, 0
        for k in range(N):
            err = np.sum(np.abs(A_c - np.roll(B_c, k))**2)
            if err < best_err:
                best_err, best_k = err, k
        B_c = np.roll(B_c, best_k)

        # Precompute FFTs (runs exactly once)
        self.FA = np.fft.fft(A_c)
        self.FB = np.fft.fft(B_c)
        self.phase_shift = best_k

    def evaluate(self, t_ease):
        """
        [Rust] fn sample(&self, t: f32) -> [[f32; 2]; 32]
        Stateless, closed-form, zero allocations, safe for parallel XPBD/Compute.
        """
        # 1. Spectral interpolation (32 complex mul-adds)
        Ft = (1.0 - t_ease) * self.FA + t_ease * self.FB

        # 2. Inverse FFT -> complex coordinates
        z_t = np.fft.ifft(Ft)

        # 3. Reconstruct translation
        centroid = (1.0 - t_ease) * self.centroid_A + t_ease * self.centroid_B
        z_t += centroid

        # Return as (32, 2) float32 array, matching Rust `[[f32; 2]; 32]`
        return np.c_[z_t.real.astype(np.float32), z_t.imag.astype(np.float32)]

# ==============================
# SHAPE GENERATORS (High-Res for Prototyping)
# ==============================


def circle_raw(n=512):
    t = np.linspace(0, 2*np.pi, n, False)
    return np.c_[np.cos(t), np.sin(t)]


def star_raw(n=512, spikes=8):
    t = np.linspace(0, 2*np.pi, n, False)
    r = 0.55 + 0.45 * np.cos(spikes * t)
    return np.c_[r*np.cos(t), r*np.sin(t)]


def heart_raw(n=512):
    t = np.linspace(0, 2*np.pi, n, False)
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    pts = np.c_[x, y]
    pts /= np.max(np.abs(pts))
    return pts

# ==============================
# RENDER & TEST (Game Loop Simulation)
# ==============================


def run_morph_test():
    print("🔹 Initializing morpher (precomputes FFT & alignment)...")
    # Simulates loading assets at game startup
    morpher = FourierShapeMorpher(heart_raw(), star_raw(spikes=8), n_points=32)

    print("🔹 Simulating game loop (61 frames)...")
    anim = []
    for i in range(61):
        t = i / 60.0
        # [Rust] Easing is applied by your animation system, not the morpher
        t_ease = t**2 * (3 - 2*t)
        # [Rust] Stateless per-frame call. Safe to call from render/physics threads.
        pts = morpher.evaluate(t_ease)
        anim.append(pts)

    # Visualization (unchanged)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    ax.set_aspect('equal')

    curve, = ax.plot([], [], 'b-', lw=2.2)
    anc,   = ax.plot([], [], 'ro', ms=4)
    title = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                    ha='center', fontsize=12)

    def update(f):
        pts = anim[f]
        poly = bezier_line(pts)
        curve.set_data(poly[:, 0], poly[:, 1])
        anc.set_data(pts[:, 0], pts[:, 1])
        title.set_text(f"t = {f/60:.2f}")
        return curve, anc, title

    print("🔹 Rendering GIF...")
    FuncAnimation(fig, update, frames=61, blit=False).save(
        "morph_rust_pattern.gif", fps=30, writer='pillow')
    print("✅ Saved: morph_rust_pattern.gif")


if __name__ == "__main__":
    run_morph_test()
