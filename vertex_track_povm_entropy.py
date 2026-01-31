"""
Visualization of the convergence of the outer-approximating polytope
=======================================================
此脚本用于可视化任意维 4-outcome POVM 外近似凸多面体的收敛过程

作者: [yang]
日期: 2026-1-25
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from scipy.linalg import svd, eigh
from pypoman import compute_polytope_vertices
from qbism import *

# ==========================================
# 0. 全局样式设置
# ==========================================
def set_pub_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 120,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })

set_pub_style()

# ==========================================
# 1. 数学工具 (N=4 POVM)
# ==========================================
def _normalize_povm(POVM):
    d = POVM[0].shape[0]
    S = sum(POVM)
    if np.allclose(S, np.eye(d)): return POVM
    L = np.linalg.cholesky(S)
    Linv = np.linalg.inv(L)
    return [Linv.conj().T @ E @ Linv for E in POVM]

def _vec_real(H):
    return np.hstack([np.real(H).ravel(), np.imag(H).ravel()])

def _affine_basis(POVM):
    d = POVM[0].shape[0]
    n = len(POVM)
    trE = np.array([np.trace(E).real for E in POVM])
    s = trE / d
    rows = []
    for i, E in enumerate(POVM):
        Ei0 = E - (trE[i] / d) * np.eye(d)
        rows.append(_vec_real(Ei0))
    M = np.vstack(rows)
    U, S, Vt = svd(M, full_matrices=False)
    tol = 1e-10
    r = int(np.sum(S > tol * S[0]))
    Q = U[:, :r]
    ones = np.ones(n)
    alpha = (ones @ Q) / n
    Q = Q - np.outer(ones, alpha)
    Q, _ = np.linalg.qr(Q, mode='reduced')
    return s, Q

def _entropy(p):
    p = np.clip(p, 1e-16, 1.0)
    return float(-np.sum(p * np.log(p)))

def _entropy_grad(p):
    p = np.clip(p, 1e-16, 1.0)
    g = -(np.log(p) + 1.0)
    return g / np.linalg.norm(g)

def _support_h(POVM, u):
    R = sum(u[i] * POVM[i] for i in range(len(POVM)))
    return float(np.linalg.eigvalsh(R)[-1])

def _p_from_psi(POVM, psi):
    return np.array([np.real(psi.conj().T @ E @ psi) for E in POVM])

# ==========================================
# 2. 计算核心
# ==========================================
def run_solver(POVM, max_iter=30):
    POVM = _normalize_povm(POVM)
    n = len(POVM)
    s, Q = _affine_basis(POVM)
    
    # 初始 Box 约束 (更宽松一点，防止数值误差导致初始切片丢失)
    A, b = [], []
    # 理论最大值是1，稍微放宽到1.001确保包含边界
    umax = [1.0 for _ in range(n)] 
    
    for i in range(n):
        A.append(-Q[i]); b.append(s[i])       # p_i >= 0
        A.append(Q[i]);  b.append(1.0 - s[i]) # p_i <= 1
        
    A, b = np.array(A), np.array(b)

    history = []
    c_plus = float('inf')
    
    print(f"Start Solver: N={n}, DOF={Q.shape[1]}")
    
    for k in range(1, max_iter + 1):
        try:
            verts_z = np.array(compute_polytope_vertices(A, b))
        except: break
        if len(verts_z) == 0: break

        verts_p = s + verts_z @ Q.T
        
        vals = np.array([_entropy(p) for p in verts_p])
        best_idx = np.argmin(vals)
        c_minus = vals[best_idx]
        p_best = verts_p[best_idx]
        
        g = _entropy_grad(p_best)
        R_op = sum(g[i] * POVM[i] for i in range(n))
        _, v = eigh(R_op)
        psi = v[:, 0]
        p_real = _p_from_psi(POVM, psi)
        c_plus = min(c_plus, _entropy(p_real))
        gap = c_plus - c_minus

        history.append({
            'iter': k,
            'verts': verts_p,
            'lb': c_minus,
            'gap': gap,
            'best_v': p_best
        })
        
        print(f"Iter {k}: V={len(verts_p)}, Gap={gap:.4e}")

        if gap < 1e-9: 
            print("Converged.")
            break

        hval = _support_h(POVM, -g)
        Ai = -(g @ Q)
        bi = hval + float(g @ s)
        A = np.vstack([A, Ai])
        b = np.hstack([b, bi])

    return history

# ==========================================
# 3. 绘图核心 (修正坐标轴范围)
# ==========================================
def style_axis_clean(ax):
    """设置去背景、完整范围坐标轴"""
    # 1. 移除背景板
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    # 2. 网格线样式
    grid_prop = {"color": "#d9d9d9", "linewidth": 0.5, "linestyle": ":"}
    ax.xaxis._axinfo["grid"].update(grid_prop)
    ax.yaxis._axinfo["grid"].update(grid_prop)
    ax.zaxis._axinfo["grid"].update(grid_prop)
    
    # 3. 范围修正：放宽到 [0, 1.0]
    # 因为 p_i 最大可能为 1，加上 0.05 的 buffer 确保不切边
    limit_min = -0.05
    limit_max = 1.05
    
    ax.set_xlim(limit_min, limit_max)
    ax.set_ylim(limit_min, limit_max)
    ax.set_zlim(limit_min, limit_max)
    
    # 4. 标准刻度
    ticks = [0.0, 0.5, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    
    # 5. 轴标签
    ax.set_xlabel(r'$p_1$', labelpad=5)
    ax.set_ylabel(r'$p_2$', labelpad=5)
    ax.set_zlabel(r'$p_3$', labelpad=5)

def plot_2x2_style(history):
    n_total = len(history)
    if n_total == 0: return

    # 自动选择4个关键帧
    if n_total < 4:
        indices = np.arange(n_total)
        while len(indices) < 4: indices = np.append(indices, n_total-1)
    else:
        indices = np.linspace(0, n_total-1, 4).astype(int)
        indices[-1] = n_total - 1
        indices = np.unique(indices)
        while len(indices) < 4: indices = np.append(indices, n_total-1)
        indices = np.sort(indices)

    fig = plt.figure(figsize=(7, 6))
    
    # 颜色定义
    face_color = np.array([0/255, 119/255, 136/255]) 
    edge_color = '#333333'
    best_pt_color = '#EE3377' 

    for plot_i, frame_idx in enumerate(indices):
        data = history[frame_idx]
        
        verts_3d = data['verts'][:, :3]
        best_v_3d = data['best_v'][:3]
        
        ax = fig.add_subplot(2, 2, plot_i + 1, projection='3d')
        
        # 应用新的坐标轴样式
        style_axis_clean(ax)
        
        # --- A. 绘制凸包 ---
        if len(verts_3d) >= 4:
            try:
                hull = ConvexHull(verts_3d)
                faces = [verts_3d[s] for s in hull.simplices]
                poly = Poly3DCollection(faces, alpha=0.25, linewidths=0.5)
                poly.set_facecolor(face_color)
                poly.set_edgecolor(edge_color)
                ax.add_collection3d(poly)
            except: pass
        
        # --- B. 绘制顶点 ---
        ax.scatter(verts_3d[:, 0], verts_3d[:, 1], verts_3d[:, 2], 
                   c='k', s=10, alpha=0.4, linewidth=0, depthshade=True)
        
        # --- C. 绘制最优解 ---
        ax.scatter(best_v_3d[0], best_v_3d[1], best_v_3d[2], 
                   c=best_pt_color, marker='D', s=80, 
                   edgecolors='white', linewidth=1.0, zorder=10)

        # --- D. 信息标注 ---
        lb_val = abs(data['lb']) 
        info_text = (f"Iter: {data['iter']}\n"
                     f"Verts: {len(verts_3d)}\n"
                     f"$h_{{-}}(\\mathcal{{E}})$: {lb_val:.4f}\n"
                     f"$\\epsilon$: {data['gap']:.1e}")
        # f"Iter {data['iter']}\n
        # $H_{{\min}} = {display_h:.4f}$"
        
        ax.text2D(0.05, 0.95, info_text, transform=ax.transAxes,
                  fontsize=11, verticalalignment='top',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor='#cccccc', alpha=0.9))
        
        ax.view_init(elev=32, azim=45)
    
    # 启用真正的LaTeX渲染
    # plt.rcParams['text.usetex'] = True

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.995, wspace=0.01, hspace=0.02)
    plt.show()

# ==========================================
# 4. 主程序
# ==========================================

if __name__ == "__main__":
    np.random.seed(42)
    d = 100
    m = 4
    
    # aux = []
    # for _ in range(n):
    #     M = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    #     M = M @ M.conj().T
    #     aux.append(M)
    
    # S = sum(aux)
    # Sinv = np.linalg.inv(np.linalg.cholesky(S))
    # POVM_4 = [Sinv.conj().T @ M @ Sinv for M in aux]

       # povm = generate_random_povm(d, N)
    ms_qobj = random_haar_povm(d, k=m, n=d, real=False)
    povm = [q.full() for q in ms_qobj]

        # 1. 运行计算
    hist_data = run_solver(povm, max_iter=100)
    
    # 2. 运行绘图 
    plot_2x2_style(hist_data)

