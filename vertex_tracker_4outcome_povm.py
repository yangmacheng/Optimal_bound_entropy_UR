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

# ==========================================
# 2. 外部逼近算法 (带熵值记录)
# ==========================================
def run_outer_approximation(POVM, max_iter=20):
    n = len(POVM)
    print(f"--- 开始计算 (N={n}, d={POVM[0].shape[0]}) ---")
    
    def h_func(r):
        R = sum(r[i] * POVM[i] for i in range(n))
        return np.linalg.eigvalsh(R)[0]
    def target_grad(p):
        p_safe = np.clip(p, 1e-16, 1)
        grad = -(np.log2(p_safe) + 1.442695) # log2(e) ≈ 1.442695
        return grad / np.linalg.norm(grad)
    # 初始化约束: sum(p)=1, p_i >= 0
    Rmat = [np.ones(n), -np.ones(n)]
    hvec = [1.0, -1.0]
    for i in range(n):
        r = np.zeros(n); r[i] = 1.0
        Rmat.append(r); hvec.append(0.0)
    Rmat = np.array(Rmat)
    hvec = np.array(hvec)
    
    # 历史记录: 存储 (vertices, min_entropy) 元组
    history = []
    
    for k in range(max_iter + 1):
        try:
            verts = compute_polytope_vertices(-Rmat, -hvec)
            verts = np.array(verts)
        except:
            break
            
        if len(verts) == 0: break
        
        # 计算当前多面体所有顶点的熵
        entropies = [-np.sum(v * np.log2(np.clip(v, 1e-16, 1))) for v in verts]
        min_entropy = np.min(entropies)
        best_idx = np.argmin(entropies)
        best_v = verts[best_idx]
        
        # 记录状态
        history.append({
            'iter': k,
            'verts': verts,
            'min_h': min_entropy,
            'best_v': best_v
        })
        
        # 计算切平面
        grad = target_grad(best_v)
        cut_val = h_func(grad)
        
        # 收敛判定
        if abs(np.dot(grad, best_v) - cut_val) < 1e-6:
            print(f"在第 {k} 次迭代收敛。")
            break
            
        Rmat = np.vstack([Rmat, grad])
        hvec = np.append(hvec, cut_val)
        
    return history
# ==========================================
# 3. 高级可视化 (网格布局 + 凸包渲染)
# ==========================================
def plot_polytope_evolution(history, n_outcomes):
    """
    绘制 2x3 网格的收敛过程图
    """
    total_frames = len(history)
    if total_frames == 0: return
    # 选取 6 个均匀分布的帧 (保证包含第一帧和最后一帧)
    if total_frames <= 6:
        indices = range(total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, 6).astype(int)
        indices = np.unique(indices) # 去重
    rows, cols = 2, 3
    fig = plt.figure(figsize=(18, 10))
    
    title_str = f"Outer Approximation Convergence (N={n_outcomes})"
    if n_outcomes == 4: title_str += " [Projected to p1-p2-p3]"
    fig.suptitle(title_str, fontsize=18, y=0.95)
    for plot_idx, frame_idx in enumerate(indices):
        if plot_idx >= rows * cols: break
        
        data = history[frame_idx]
        verts = data['verts']
        min_h = data['min_h']
        best_v = data['best_v']
        
        ax = fig.add_subplot(rows, cols, plot_idx + 1, projection='3d')
        
        # 设置坐标轴
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        ax.set_xlabel('p1'); ax.set_ylabel('p2'); ax.set_zlabel('p3')
        ax.view_init(elev=30, azim=45)
        
        # 准备绘图数据 (如果N=4，投影到前3维)
        if n_outcomes == 4:
            plot_verts = verts[:, :3]
            best_v_plot = best_v[:3]
        else:
            plot_verts = verts
            best_v_plot = best_v
        # --- 绘制凸多面体 (Convex Hull) ---
        if len(plot_verts) >= 4:
            try:
                if n_outcomes == 3:
                    # N=3 技巧: 加入原点 (0,0,0) 形成锥体，只画底面
                    verts_w_origin = np.vstack([plot_verts, [0,0,0]])
                    hull = ConvexHull(verts_w_origin)
                    # 筛选出不包含原点的面 (即我们需要的平面多边形)
                    faces = [verts_w_origin[s] for s in hull.simplices if len(plot_verts) not in s]
                    face_color = 'cyan'
                    alpha_val = 0.5
                    
                elif n_outcomes == 4:
                    # N=4: 直接计算 3D 凸包
                    hull = ConvexHull(plot_verts)
                    faces = [plot_verts[s] for s in hull.simplices]
                    face_color = 'lime' # 绿色代表体积
                    alpha_val = 0.25
                # 绘制面
                poly = Poly3DCollection(faces, alpha=alpha_val, edgecolor='k', linewidths=0.5)
                poly.set_facecolor(face_color)
                ax.add_collection3d(poly)
                
            except Exception as e:
                print(f"Hull error frame {frame_idx}: {e}")
        # --- 绘制顶点 ---
        ax.scatter(plot_verts[:,0], plot_verts[:,1], plot_verts[:,2], 
                   c='blue', s=10, alpha=0.6, depthshade=False)
        
        # --- 标记最优顶点 (黄色星号) ---
        ax.scatter(best_v_plot[0], best_v_plot[1], best_v_plot[2], 
                   c='gold', marker='*', s=200, edgecolors='k', zorder=10, label='Min Entropy')
        # --- 添加文本标注 ---
        ax.set_title(f"Iter {data['iter']} | Vertices: {len(verts)}")
        # 在 3D 图的左上角 (2D 投影坐标) 添加文字
        ax.text2D(0.05, 0.95, f"$H_{{min}} = {min_h:.4f}$", transform=ax.transAxes, 
                  fontsize=12, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    # --- 用户配置区 ---
    N = 4  # 结果数量 (3 或 4)
    d = 50  # 希尔伯特空间维度 (可以是 2, 3, 4...)
    # ------------------
    
    # 1. 生成随机 POVM
    # 固定随机种子以便复现
    np.random.seed(42) 
    # povm = generate_random_povm(d, N)
    ms_qobj = random_haar_povm(d, k=N, n=d, real=False)
    povm = [q.full() for q in ms_qobj]
    
    print(f"生成的随机 POVM 示例 (第一个元素):\n{povm[0].round(3)}")
    
    # 2. 运行计算
    history = run_outer_approximation(povm, max_iter=40)
    
    # 3. 绘图
    plot_polytope_evolution(history, N)
