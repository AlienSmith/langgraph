import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('Agg')

# ==============================
# BÉZIER EVALUATION (Your original)
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
# PREPROCESSING
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
    # Ensure CCW winding
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * np.sum(x[:-1]*y[1:] - x[1:]*y[:-1]) + \
        0.5*(x[-1]*y[0] - x[0]*y[-1])
    if area < 0:
        pts = pts[::-1]
        pts = np.roll(pts, 1, axis=0)
    return pts

# ==============================
# FOURIER SPECTRAL MORPH (Industry Standard)
# ==============================


def fourier_morph(A, B, t):
    """
    Symmetry-preserving shape morph via Fourier Descriptors.
    Eliminates caving, guarantees smooth frequency transition.
    """
    N = len(A)
    Az = A[:, 0] + 1j * A[:, 1]
    Bz = B[:, 0] + 1j * B[:, 1]

    # 1. Center & Scale Match
    A_c = Az - Az.mean()
    B_c = Bz - Bz.mean()
    scale = np.linalg.norm(A_c) / (np.linalg.norm(B_c) + 1e-8)
    B_c *= scale

    # 2. Optimal Phase Alignment (Cyclic Shift)
    best_err, best_k = np.inf, 0
    for k in range(N):
        err = np.sum(np.abs(A_c - np.roll(B_c, k))**2)
        if err < best_err:
            best_err, best_k = err, k
    B_c = np.roll(B_c, best_k)

    # 3. Fourier Transform
    FA = np.fft.fft(A_c)
    FB = np.fft.fft(B_c)

    # 4. Interpolate Spectral Coefficients
    Ft = (1 - t) * FA + t * FB

    # 5. Inverse Transform
    z_t = np.fft.ifft(Ft)

    # 6. Reconstruct Translation & Return
    centroid = (1 - t) * Az.mean() + t * Bz.mean()
    z_t += centroid

    return np.c_[z_t.real, z_t.imag]

# ==============================
# SHAPE GENERATORS
# ==============================


def circle_raw(n=512):
    t = np.linspace(0, 2*np.pi, n, False)
    return np.c_[np.cos(t), np.sin(t)]


def star_raw(n=512, spikes=8):
    t = np.linspace(0, 2*np.pi, n, False)
    # r > 0.4 prevents origin singularity
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
# RENDER & TEST
# ==============================


def run_morph_test():
    # === TEST: Circle -> 8-Point Star ===
    A_raw = heart_raw()
    B_raw = star_raw(spikes=8)

    print("Preprocessing shapes (arc-length + winding fix)...")
    A = prepare_shape(A_raw, n_points=32)
    B = prepare_shape(B_raw, n_points=32)

    print("Generating frames (Fourier Spectral Morph)...")
    anim = []
    for i in range(61):
        t = i / 60.0
        t_ease = t**2 * (3 - 2*t)  # Smooth easing
        pts = fourier_morph(A, B, t_ease)
        anim.append(pts)

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

    print("Rendering GIF...")
    FuncAnimation(fig, update, frames=61, blit=False).save(
        "morph_fourier_star.gif", fps=30, writer='pillow')
    print("✅ Saved: morph_fourier_star.gif")


if __name__ == "__main__":
    run_morph_test()
