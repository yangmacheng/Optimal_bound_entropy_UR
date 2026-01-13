import numpy as np
from scipy.linalg import sqrtm
from qbism import *
import entropy_min
import matplotlib.pyplot as plt

# # 3. 可视化绘图模块
# # ===========================
# def plot_results(history, n, d):
#     iters = history['iter']
#     lbs = history['lb']
#     ubs = history['ub']
#     gaps = history['gap']
    
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
#     # 图1: 上下界收敛
#     ax1.plot(iters, ubs, 'r-o', label='Upper Bound (Best Feasible)', markersize=4)
#     ax1.plot(iters, lbs, 'b-s', label='Lower Bound (MIP Relax)', markersize=4)
#     ax1.fill_between(iters, lbs, ubs, color='gray', alpha=0.1)
    
#     ax1.set_ylabel('Shannon Entropy')
#     ax1.set_title(f'Convergence of Min-Entropy (N={n}, d={d})')
#     ax1.legend()
#     ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    
#     # 图2: 精度 Gap
#     ax2.plot(iters, gaps, 'k-^', label='Gap (UB - LB)', markersize=4)
#     ax2.set_yscale('log') # 关键：用对数坐标看收敛级数
#     ax2.set_ylabel('Gap (Log Scale)')
#     ax2.set_xlabel('Iteration')
#     ax2.set_title('Convergence Precision')
#     ax2.legend()
#     ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    
#     plt.tight_layout()
#     plt.show()

# ===========================
# 4. 执行测试 N=10
# ===========================
if __name__ == "__main__":
    # 构造 N=10, d=3 的 POVM (确保是高维情况)
    N = 4
    d = 100
    
    # np.random.seed(42) # 固定种子方便复现
    # print(f"Generating Random POVM (N={N}, d={d})...")
    
    # # 生成随机 SIC-POVM 风格的算子
    # vectors = (np.random.randn(N, d) + 1j * np.random.randn(N, d))
    # vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    # povm_raw = [np.outer(v, v.conj()) for v in vectors]
    
    # # 归一化使其求和为 I
    # S = sum(povm_raw)
    # S_sqrt_inv = np.linalg.inv(np.linalg.cholesky(S))
    # POVM = [S_sqrt_inv @ E @ S_sqrt_inv.conj().T for E in povm_raw]

    # 生成两个正交归一基（计算基和傅里叶基）
    basis1 = [np.array([[1], [0]]), np.array([[0], [1]])]  # 标准计算基 (|0>, |1>)
    basis2 = [
        np.array([[1], [1]]) / np.sqrt(2),  # |+> = (|0> + |1>)/√2
        np.array([[1], [-1]]) / np.sqrt(2)  # |-> = (|0> - |1>)/√2
    ]
    basis3 = [
        np.array([[np.sqrt(3)], [-1]]) / 2,  # (|0> + i|1>)/√2
        np.array([[1], [np.sqrt(3)]]) / 2# (|0> - i|1>)/√2
    ]
    # 创建POVM元素：每个投影算子乘以1/2
    POVM2 = []
    for v in basis1 + basis3:
        # 创建投影算子 |v><v|
        projector = v @ v.T
        # 乘以1/2得到POVM元素
        povm_element = projector
        POVM2.append(povm_element)

        # d = 5
    ms_qobj = random_haar_povm(d, k=N, n=d, real=False)
    POVM1 = [q.full() for q in ms_qobj]
        
    # 运行
    # best_p, hist = entropy_min.optimize_min_entropy(POVM1, max_iter=50, tol=1e-4)
    
    # # 绘图
    # if len(hist['iter']) > 0:
    #     plot_results(hist, N, d)
    # else:
    #     print("No data to plot.")

    effective_povm_bound = entropy_min.shannon_entropy_min(POVM1, max_iter=150, tol=1e-5)
    # effective_povm_bound = entropy_min.shannon_entropy_min_bnb(POVM1, max_iter=80, tol=1e-6)
    # effective_povm_bound = entropy_min.shannon_entropy_min_gurobi(POVM1, max_iter=100, tol=1e-5)


    

# 生成两个正交归一基（计算基和傅里叶基）
# basis1 = [np.array([[1], [0]]), np.array([[0], [1]])]  # 标准计算基 (|0>, |1>)
# basis2 = [
#     np.array([[1], [1]]) / np.sqrt(2),  # |+> = (|0> + |1>)/√2
#     np.array([[1], [-1]]) / np.sqrt(2)  # |-> = (|0> - |1>)/√2
# ]
# basis3 = [
#     np.array([[np.sqrt(3)], [-1]]) / 2,  # (|0> + i|1>)/√2
#     np.array([[1], [np.sqrt(3)]]) / 2# (|0> - i|1>)/√2
# ]
# # 创建POVM元素：每个投影算子乘以1/2
# POVM2 = []
# for v in basis1 + basis2:
#     # 创建投影算子 |v><v|
#     projector = v @ v.T
#     # 乘以1/2得到POVM元素
#     povm_element = projector / 2
#     POVM2.append(povm_element)

#     d = 3
#     ms_qobj = random_haar_povm(d, k=8, n=d, real=False)
#     POVM1 = [q.full() for q in ms_qobj]

# effective_povm_bound = entropy_min.shannon_entropy_min(
#     POVM1, max_iter=50, tol=1e-6)

# effective_povm_bound = entropy_min.shannon_entropy_min_sbb(POVM1, max_iter=50, tol=1e-5, sbb_timeout=30)

# effective_povm_bound = entropy_min.shannon_entropy_min_milp(POVM1, max_iter=30, segments=15)
 # 运行

# %%
import numpy as np
np.log2(1e-16)


# %%
import numpy as np

def target_func(p):
    """目标函数: H(p)"""
    p_safe = np.clip(p, 1e-16, 1)
    return -np.sum(p_safe * np.log2(p_safe))
def target_grad(p):
    """目标梯度: ∇H(p)"""
    p_safe = np.clip(p, 1e-16, 1) # 去除 log(0)
    grad = -(np.log2(p_safe) + 1.442695) # 1/ln2
    return grad / np.linalg.norm(grad)

target_func(target_grad([1,0,0]))
# %%
44237/3600
# %%
