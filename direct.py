import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('Agg')

# ==============================
# BÉZIER EVALUATION
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
# CORE MORPHING ALGORITHM
# ==============================


def robust_morph(A, B, t):
    """
    拓扑保持形状插值（全局旋转解耦 + 局部边向量插值）
    彻底解决坍缩、双星旋转、自交问题
    """
    N = len(A)
    Az = A[:, 0] + 1j * A[:, 1]
    Bz = B[:, 0] + 1j * B[:, 1]

    # 1. 中心化（去除平移影响）
    A_c = Az - Az.mean()
    B_c = Bz - Bz.mean()

    # 2. 提取最优全局旋转与缩放 (2D Procrustes / Kabsch)
    dot = np.vdot(A_c, B_c)  # conj(A) * B 的点积
    global_angle = np.angle(dot)
    scale = np.linalg.norm(A_c) / (np.linalg.norm(B_c) + 1e-8)

    # 将 B 对齐到 A 的朝向，剥离出纯局部形变
    B_aligned = B_c * np.exp(1j * global_angle) * scale

    # 3. 局部形变：边向量极坐标插值
    dA = np.roll(A_c, -1) - A_c
    dB = np.roll(B_aligned, -1) - B_aligned

    lenA, angA = np.abs(dA), np.angle(dA)
    lenB, angB = np.abs(dB), np.angle(dB)

    # 最短路径角度差（避免 ±π 跳变导致的局部反转）
    d_ang = (angB - angA + np.pi) % (2 * np.pi) - np.pi
    len_t = (1 - t) * lenA + t * lenB
    ang_t = angA + t * d_ang

    # 4. 重建轮廓
    dz_t = len_t * np.exp(1j * ang_t)
    pts = np.cumsum(dz_t)

    # 闭合修正（线性分配累积误差，防止螺旋断裂）
    err = pts[-1] - pts[0]
    pts -= err * np.linspace(0, 1, N, endpoint=False)

    # 5. 叠加插值后的全局旋转 + 质心轨迹
    centroid = (1 - t) * Az.mean() + t * Bz.mean()
    rot_t = np.exp(1j * t * global_angle)
    pts = rot_t * (pts - pts.mean()) + centroid

    return np.c_[pts.real, pts.imag]

# ==============================
# SHAPE GENERATORS
# ==============================


def circle(n=32):
    t = np.linspace(0, 2*np.pi, n, False)
    return np.c_[np.cos(t), np.sin(t)]


def heart(n=32):
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
    # === 测试 1: Circle -> 180° 旋转 (验证无坍缩/双星) ===
    # A = circle(32)
    # B = -A  # 纯 180° 旋转，保持手性一致

    # === 测试 2: Circle -> Heart (凸 -> 凹) ===
    A = circle(32)
    B = heart(32)

    print("Generating frames...")
    anim = []
    for i in range(61):
        t = i / 60.0
        # 可选：加入缓动曲线让动画更自然
        # t_ease = t**2 * (3 - 2*t)
        # pts = robust_morph(A, B, t_ease)
        pts = robust_morph(A, B, t)
        anim.append(pts)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
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
        "morph_robust.gif", fps=30, writer='pillow')
    print("✅ Saved: morph_robust.gif")


if __name__ == "__main__":
    run_morph_test()
