import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from scipy.linalg import eigh
from qbism import *

# ==========================================
# 0. 期刊级样式 (Publication Style)
# ==========================================
def set_pub_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        # 轴刻度设置
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
set_pub_style()
# 依赖检查
try:
    from pypoman import compute_polytope_vertices
except ImportError:
    raise ImportError("Dependencies missing. Run: pip install pypoman cvxopt")

# ==========================================
# 1. 外部逼近算法 (带熵值记录)
# ==========================================
def run_outer_approximation(POVM, max_iter=20):
    n = len(POVM)
    print(f"--- Processing (N={n}, d={POVM[0].shape[0]}) ---")
    
    def h_func(r): 
        # 计算支撑平面
        return np.linalg.eigvalsh(sum(r[i] * POVM[i] for i in range(n)))[0]
    
    def target_grad(p):
        # 熵的梯度
        p = np.clip(p, 1e-16, 1)
        g = -(np.log2(p) + 1.442695)
        return g / np.linalg.norm(g)
    
    # def target_func(x):
    #     """目标函数 f(x) = ||x||^2"""
    #     return np.sum(x**2)
    
    # def target_grad(x):
    #     """目标函数的梯度 ∇f(x) = 2x"""
    #     return 2 * x
    
    # 初始约束: Simplex (sum=1, p>=0)
    Rmat = [np.ones(n), -np.ones(n)]
    hvec = [1.0, -1.0]
    for i in range(n):
        r = np.zeros(n); r[i] = 1.0
        Rmat.append(r); hvec.append(0.0)
    Rmat, hvec = np.array(Rmat), np.array(hvec)
    
    history = []
    for k in range(max_iter + 1):
        try:
            verts = np.array(compute_polytope_vertices(-Rmat, -hvec))
        except: break
        if len(verts) == 0: break
        
        # 计算熵 (防止 log(0))
        entropies = [-np.sum(v * np.log2(np.clip(v, 1e-16, 1))) for v in verts]
        min_h = np.min(entropies)
        
        # 修正 -0.0000 问题
        if abs(min_h) < 1e-10: min_h = 0.0
            
        best_v = verts[np.argmin(entropies)]
        
        history.append({'iter': k, 'verts': verts, 'min_h': min_h, 'best_v': best_v})
        
        grad = target_grad(best_v)
        cut_val = h_func(grad)
        if abs(np.dot(grad, best_v) - cut_val) < 1e-6: break
        Rmat = np.vstack([Rmat, grad])
        hvec = np.append(hvec, cut_val)
    return history
# ==========================================
# 2. 绘图美化 (关键修改部分)
# ==========================================
def style_3d_axis(ax):
    """移除灰色背景，简化刻度，避免文字重叠"""
    # 移除背景板颜色
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    # 细化网格线
    ax.xaxis._axinfo["grid"].update({"color": "lightgray", "linewidth": 0.5, "linestyle": ":"})
    ax.yaxis._axinfo["grid"].update({"color": "lightgray", "linewidth": 0.5, "linestyle": ":"})
    ax.zaxis._axinfo["grid"].update({"color": "lightgray", "linewidth": 0.5, "linestyle": ":"})
    # 关键：设置坐标轴范围，稍微大于1，防止顶点被切
    pad = 0.05
    ax.set_xlim(0 - pad, 1 + pad)
    ax.set_ylim(0 - pad, 1 + pad)
    ax.set_zlim(0 - pad, 1 + pad)
    # 关键：简化刻度，只显示 0, 0.5, 1
    ticks = [0.0, 0.5, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
def plot_polytope_evolution(history, n_outcomes):
    total_frames = len(history)
    if total_frames == 0: return
    # 选帧：保证第一帧(Simplex)和最后一帧被选中
    indices = np.unique(np.linspace(0, total_frames - 1, 6).astype(int))
    
    fig = plt.figure(figsize=(12, 8))
    
    # 颜色配置 (Science 风格)
    # 多面体颜色：通透的蓝绿色
    face_color_base = np.array([0/255, 119/255, 136/255]) 
    edge_color = '#333333'
    best_pt_color = '#EE3377' # Magenta/Red 用于突出重点
    
    for plot_idx, frame_idx in enumerate(indices):
        data = history[frame_idx]
        verts = data['verts']
        min_h = data['min_h']
        best_v = data['best_v']
        
        # 投影处理 (N=4 时只取前3维)
        if n_outcomes == 4:
            plot_verts = verts[:, :3]
            best_v_proj = best_v[:3]
        else:
            plot_verts = verts
            best_v_proj = best_v
        ax = fig.add_subplot(2, 3, plot_idx + 1, projection='3d')
        style_3d_axis(ax)
        
        # 设置视角：稍微太高，便于观察整体形状
        ax.view_init(elev=30, azim=40)
        
        # 关键：增加 labelpad 防止重叠
        label_pad_val = 10 
        ax.set_xlabel(r'$p_1$', labelpad=label_pad_val)
        ax.set_ylabel(r'$p_2$', labelpad=label_pad_val)
        ax.set_zlabel(r'$p_3$', labelpad=label_pad_val)
        # --- 凸包与多面体渲染 ---
        if len(plot_verts) >= 4:
            try:
                if n_outcomes == 3:
                    # N=3: 原点辅助法
                    # 添加原点 (0,0,0) 构造四面体，然后只画不含原点的面
                    verts_origin = np.vstack([plot_verts, [0,0,0]])
                    hull = ConvexHull(verts_origin)
                    # 筛选面：面的顶点索引中不包含最后一个点(即原点)
                    faces = [verts_origin[s] for s in hull.simplices if len(plot_verts) not in s]
                    alpha_val = 0.5
                else:
                    # N=4: 直接计算 3D 凸包
                    hull = ConvexHull(plot_verts)
                    faces = [plot_verts[s] for s in hull.simplices]
                    alpha_val = 0.25 # 实体需要更透明
                
                # 创建多面体
                poly = Poly3DCollection(faces, alpha=alpha_val, linewidths=0.5)
                poly.set_facecolor(face_color_base)
                poly.set_edgecolor(edge_color)
                ax.add_collection3d(poly)
                
            except Exception as e:
                pass # 忽略退化情况
        # --- 绘制顶点 ---
        ax.scatter(plot_verts[:,0], plot_verts[:,1], plot_verts[:,2], 
                   c='k', s=5, alpha=0.5, depthshade=True)
        
        # --- 绘制最优解 (菱形标记) ---
        ax.scatter(best_v_proj[0], best_v_proj[1], best_v_proj[2], 
                   c=best_pt_color, marker='D', s=50, edgecolors='white', linewidth=0.5,
                   zorder=10, label=r'Min Entropy')
        # --- 文字标注 (带边框) ---
        # 使用 abs(min_h) 彻底解决 -0.0000 问题
        display_h = abs(min_h)
        info_txt = f"Iter {data['iter']}\n$H_{{\min}} = {display_h:.4f}$"
        
        ax.text2D(0.05, 0.92, info_txt, transform=ax.transAxes,
                  fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9, lw=0.5))
    plt.tight_layout()
    # 微调子图间距
    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    plt.show()

# ==========================================
# 3. 运行
# ==========================================
if __name__ == "__main__":
    # --- 用户配置区 ---
    N = 4  # 结果数量 (3 或 4)
    d = 30  # 希尔伯特空间维度 (可以是 2, 3, 4...)
    # ------------------
    
    # 1. 生成随机 POVM
    # 固定随机种子以便复现
    # np.random.seed(42) 
    # povm = generate_random_povm(d, N)
    ms_qobj = random_haar_povm(d, k=N, n=d, real=False)
    povm = [q.full() for q in ms_qobj]
    
    # print(f"生成的随机 POVM 示例 (第一个元素):\n{povm[0].round(3)}")
    
    # 2. 运行计算
    history = run_outer_approximation(povm, max_iter=40)
    
    # 3. 绘图
    plot_polytope_evolution(history, N)
