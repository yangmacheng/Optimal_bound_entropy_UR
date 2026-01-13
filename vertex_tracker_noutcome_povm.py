import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from scipy.linalg import eigh
from qbism import *

try:
    from pypoman import compute_polytope_vertices
except ImportError:
    raise ImportError("需要 pypoman: pip install pypoman cvxopt")

# =========================== 
# 2. 高维外部逼近算法
# ===========================
def run_high_dim_approximation(POVM, max_iter=20):
    n = len(POVM)
    print(f"--- 开始高维计算 (N={n}, d={POVM[0].shape[0]}) ---")
    print("注意：随着N增大，顶点数量会指数级增加，请耐心等待...")
    
    # 支撑平面函数
    def h_func(r):
        R = sum(r[i] * POVM[i] for i in range(n))
        return np.linalg.eigvalsh(R)[0]

    # 目标函数(熵)的梯度
    def target_grad(p):
        p_safe = np.clip(p, 1e-16, 1)
        grad = -(np.log2(p_safe) + 1.442695)
        return grad / np.linalg.norm(grad)

    # 初始约束
    # 1. sum(p) = 1
    Rmat = [np.ones(n), -np.ones(n)]
    hvec = [1.0, -1.0]
    # 2. p_i >= 0
    for i in range(n):
        r = np.zeros(n); r[i] = 1.0
        Rmat.append(r); hvec.append(0.0)

    Rmat = np.array(Rmat); hvec = np.array(hvec)
    history = []
    
    for k in range(max_iter + 1):
        try:
            # 高维顶点枚举是计算瓶颈
            verts = compute_polytope_vertices(-Rmat, -hvec)
            verts = np.array(verts)
        except Exception as e:
            print(f"Solver limit reached at iter {k}: {e}")
            break
            
        if len(verts) == 0: break
        history.append(verts)
        
        # 找到当前使得熵最小的顶点 (最尖锐的点)
        vals = [-np.sum(v * np.log2(np.clip(v, 1e-16, 1))) for v in verts]
        best_v = verts[np.argmin(vals)]
        
        # 增加切平面
        grad = target_grad(best_v)
        cut_val = h_func(grad)
        
        Rmat = np.vstack([Rmat, grad])
        hvec = np.append(hvec, cut_val)
        
        # 简单进度条
        if k % 2 == 0: print(f"Iter {k}: {len(verts)} vertices found in {n}D space.")

    return history

# ===========================
# 3. 投影可视化 (Projection Visualizer)
# ===========================
def plot_projections(history, dims_to_plot=[0, 1, 2]):
    """
    history: 历史顶点数据列表
    dims_to_plot: 要展示的三个维度的索引，例如 [0, 1, 2] 表示 p1, p2, p3
    """
    total_frames = len(history)
    # 选 4 个关键帧
    indices = np.unique(np.linspace(0, total_frames - 1, 4).astype(int))
    
    d1, d2, d3 = dims_to_plot
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Projection of {history[0].shape[1]}D Polytope onto axes (p{d1+1}, p{d2+1}, p{d3+1})", fontsize=16)

    for i, idx in enumerate(indices):
        verts_Nd = history[idx] # N维顶点
        
        # --- 核心步骤：投影 ---
        # 只取我们关心的 3 列
        verts_3d = verts_Nd[:, dims_to_plot]
        
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.set_title(f"Iter {idx} (Total Vertices: {len(verts_Nd)})")
        ax.set_xlabel(f'p{d1+1}'); ax.set_ylabel(f'p{d2+1}'); ax.set_zlabel(f'p{d3+1}')
        
        # 坐标轴限制 (概率总是 0-1)
        ax.set_xlim(0, 0.8); ax.set_ylim(0, 0.8); ax.set_zlim(0, 0.8)
        ax.view_init(elev=25, azim=135)

        if len(verts_3d) < 4: continue

        try:
            # 计算投影后点云的凸包 (Convex Hull of the Shadow)
            # 因为凸多面体的线性投影依然是凸多面体
            hull = ConvexHull(verts_3d)
            
            # 绘制面
            faces = [verts_3d[s] for s in hull.simplices]
            
            # 颜色：后期颜色更深，代表更紧致
            col = plt.cm.plasma(i / 3.0)
            
            # alpha 设低一点，因为高维投影内部结构可能很复杂
            poly = Poly3DCollection(faces, alpha=0.3, edgecolor='k', linewidths=0.5)
            poly.set_facecolor(col)
            ax.add_collection3d(poly)
            
            # 绘制投影后的顶点（散点）
            ax.scatter(verts_3d[:,0], verts_3d[:,1], verts_3d[:,2], s=5, c='gray', alpha=0.5)
            
            # 标记最优解在投影空间的位置
            vals = [-np.sum(v * np.log2(np.clip(v, 1e-16, 1))) for v in verts_Nd]
            best_idx = np.argmin(vals)
            best_v_proj = verts_3d[best_idx]
            ax.scatter(best_v_proj[0], best_v_proj[1], best_v_proj[2], 
                       c='red', marker='*', s=120, zorder=10, label='Min Entropy (Proj)')
            
            if i==0: ax.legend()
            
        except Exception as e:
            print(f"Hull error: {e}")

    plt.tight_layout()
    plt.show()

# ===========================
# 4. 主程序
# ===========================
if __name__ == "__main__":
    # --- 用户配置 ---
    # 尝试 N=6 (6个结果的POVM), d=3 (Qutrit)
    # 这意味着我们在寻找 6维 空间中的物体
    N = 8
    d = 8      
    
    # 我们想看哪三个概率分量的关联？比如 p1, p2, p3
    target_dims = [0, 1, 2] 
    # ----------------
    
    # 1. 生成高维 POVM
    ms_qobj = random_haar_povm(d, k=N, n=d, real=False)
    povm = [q.full() for q in ms_qobj]
    
    # 2. 运行高维逼近 (迭代次数不宜过多，否则顶点数爆炸)
    # 对于 N=6，顶点数可能在几百到几千
    hist = run_high_dim_approximation(povm, max_iter=48)
    
    # 3. 画出投影
    if len(hist) > 0:
        plot_projections(hist, dims_to_plot=target_dims)
