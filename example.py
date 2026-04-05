import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('Agg')

# ==============================
# CONFIGURATION
# ==============================
TARGET_SEGMENTS = 64  # 64 quad Bézier curves -> 128 points (A,C,A,C...)
# Try 32, 64, 96, or 128. Must be a multiple of 16 for clean subdivision.

# ==============================
# BÉZIER EVALUATION & RESAMPLING
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


def resample_by_arc_length(pts, n_out):
    pts_closed = np.vstack([pts, pts[0]])
    dists = np.linalg.norm(np.diff(pts_closed, axis=0), axis=1)
    cumulative = np.concatenate([[0], np.cumsum(dists)])
    total_len = cumulative[-1]
    target_dists = np.linspace(0, total_len, n_out, endpoint=False)
    new_x = np.interp(target_dists, cumulative, pts_closed[:, 0])
    new_y = np.interp(target_dists, cumulative, pts_closed[:, 1])
    return np.c_[new_x, new_y]

# ==============================
# RESOLUTION UPGRADE (Core Feature)
# ==============================


def evaluate_high_res(editor_pts, samples_per_curve=128):
    """Evaluate original 32-pt curve at high resolution for accurate arc-length mapping."""
    pts = []
    n_curves = len(editor_pts) // 2
    for i in range(n_curves):
        p0 = editor_pts[2*i]
        p1 = editor_pts[2*i+1]
        p2 = editor_pts[2*(i+1) % len(editor_pts)]
        for j in range(samples_per_curve):
            t = j / samples_per_curve
            pts.append((1-t)**2*p0 + 2*(1-t)*t*p1 + t**2*p2)
    return np.array(pts, dtype=np.float32)


def upgrade_to_uniform_resolution(editor_pts, target_segments=64):
    """
    1. Samples anchors uniformly by arc-length
    2. Reconstructs control points using exact C = 2*B(0.5) - 0.5*(A0 + A1)
    3. Returns [A0, C0, A1, C1, ...] of length 2*target_segments
    Preserves original shape exactly while increasing resolution for physics/FFT.
    """
    high_res = evaluate_high_res(editor_pts, samples_per_curve=256)

    # 1. Uniform arc-length anchors
    anchors = resample_by_arc_length(high_res, target_segments)

    # 2. Reconstruct controls via midpoints on original curve
    N = target_segments
    controls = np.zeros((N, 2), dtype=np.float32)

    cumulative = np.concatenate(
        [[0], np.cumsum(np.linalg.norm(np.diff(high_res, axis=0), axis=1))])
    total_len = cumulative[-1]

    for i in range(N):
        s_i = (i / N) * total_len
        s_next = ((i + 1) % N / N) * total_len
        if i == N - 1:
            s_next = total_len

        s_mid = (s_i + s_next) / 2.0

        # Interpolate midpoint from high-res curve
        mid_x = np.interp(s_mid, cumulative, high_res[:, 0])
        mid_y = np.interp(s_mid, cumulative, high_res[:, 1])
        mid_pt = np.array([mid_x, mid_y], dtype=np.float32)

        # Exact quadratic control reconstruction: C = 2*M - 0.5*(A0 + A1)
        A0 = anchors[i]
        A1 = anchors[(i+1) % N]
        controls[i] = 2.0 * mid_pt - 0.5 * (A0 + A1)

    # Interleave [A0, C0, A1, C1, ...]
    out = np.zeros((2*N, 2), dtype=np.float32)
    out[0::2] = anchors
    out[1::2] = controls
    return out

# ==============================
# JSON LOADER & PREPROCESSING
# ==============================


def load_and_upgrade(json_path, target_segments=64):
    with open(json_path, 'r') as f:
        editor_pts = np.array(json.load(f), dtype=np.float32)
    assert editor_pts.shape == (
        32, 2), f"Editor JSON must be (32,2), got {editor_pts.shape}"

    # Upgrade to higher resolution [A,C,A,C...]
    pts = upgrade_to_uniform_resolution(editor_pts, target_segments)

    # Enforce CCW winding
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * np.sum(x[:-1]*y[1:] - x[1:]*y[:-1]) + \
        0.5*(x[-1]*y[0] - x[0]*y[-1])
    if area < 0:
        pts = pts[::-1]
        pts = np.roll(pts, 1, axis=0)
    return pts

# ==============================
# FFT MORPHER (Now operates on higher-res)
# ==============================


class FourierShapeMorpher:
    def __init__(self, pts_A, pts_B):
        self.A = pts_A
        self.B = pts_B
        self.n_points = len(pts_A)
        self._precompute()

    def _precompute(self):
        N = self.n_points
        Az = self.A[:, 0] + 1j * self.A[:, 1]
        Bz = self.B[:, 0] + 1j * self.B[:, 1]

        self.centroid_A = Az.mean()
        self.centroid_B = Bz.mean()

        A_c = Az - self.centroid_A
        B_c = Bz - self.centroid_B

        self.scale = np.linalg.norm(A_c) / (np.linalg.norm(B_c) + 1e-8)
        B_c *= self.scale

        best_err, best_k = np.inf, 0
        for k in range(N):
            err = np.sum(np.abs(A_c - np.roll(B_c, k))**2)
            if err < best_err:
                best_err, best_k = err, k
        B_c = np.roll(B_c, best_k)

        self.FA = np.fft.fft(A_c)
        self.FB = np.fft.fft(B_c)
        self.phase_shift = best_k

    def evaluate(self, t_ease):
        Ft = (1.0 - t_ease) * self.FA + t_ease * self.FB
        z_t = np.fft.ifft(Ft)
        centroid = (1.0 - t_ease) * self.centroid_A + t_ease * self.centroid_B
        z_t += centroid
        return np.c_[z_t.real.astype(np.float32), z_t.imag.astype(np.float32)]

# ==============================
# RENDER & TEST
# ==============================


def run_morph_test():
    print(f"🔹 Loading & upgrading shapes to {TARGET_SEGMENTS} segments...")
    # 👇 Replace with your actual filenames
    A = load_and_upgrade("/workspace/Shape_A.json", TARGET_SEGMENTS)
    B = load_and_upgrade("/workspace/Shape_B.json", TARGET_SEGMENTS)

    print(
        f"🔹 Initialized morpher on {len(A)} control points ({TARGET_SEGMENTS} quad curves)")
    morpher = FourierShapeMorpher(A, B)

    print("🔹 Simulating game loop (61 frames)...")
    anim = []
    for i in range(61):
        t = i / 60.0
        t_ease = t**2 * (3 - 2*t)
        pts = morpher.evaluate(t_ease)
        anim.append(pts)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    ax.set_aspect('equal')

    curve, = ax.plot([], [], 'b-', lw=2.2)
    anc,   = ax.plot([], [], 'ro', ms=3)  # Smaller dots for higher res
    title = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                    ha='center', fontsize=12)

    def update(f):
        pts = anim[f]
        poly = bezier_line(pts)
        curve.set_data(poly[:, 0], poly[:, 1])
        anc.set_data(pts[:, 0], pts[:, 1])
        title.set_text(f"t = {f/60:.2f} | {len(pts)} pts")
        return curve, anc, title

    print("🔹 Rendering GIF...")
    FuncAnimation(fig, update, frames=61, blit=False).save(
        "morph_highres.gif", fps=30, writer='pillow')
    print("✅ Saved: morph_highres.gif")


if __name__ == "__main__":
    run_morph_test()
