import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from qbism import *
import time

# ===========================
# 1. 核心 MIP 求解器 (处理子问题)
# ===========================
def solve_milp_subproblem(A_ub, b_ub, A_eq, b_eq, n, num_segments=20):
    """
    使用 MIP 求解 Polytope 上的最小熵近似值。
    min sum(w_i) 
    s.t. p in Polytope
         (p_i, w_i) 在 -x*log(x) 的分段线性曲线上 (SOS2 约束)
    """
    # 1. 准备分段线性网格
    x_grid = np.linspace(0, 1, num_segments + 1)
    # y = -x log2(x), 凹函数
    y_grid = np.zeros_like(x_grid)
    mask = x_grid > 1e-12
    y_grid[mask] = -x_grid[mask] * np.log2(x_grid[mask])
    
    M = num_segments
    
    # 变量结构:
    # Lambda: n * (M+1) 个 (连续, [0,1]) -> 用于插值 p 和 w
    # Z:      n * M     个 (二进制)      -> 用于选择区间
    # p_i = sum_k x_k * lambda_{i,k}
    # w_i = sum_k y_k * lambda_{i,k}
    
    num_lambda = n * (M + 1)
    num_z = n * M
    n_vars = num_lambda + num_z
    
    # --- 目标函数: min sum w_i ---
    c = np.zeros(n_vars)
    # 系数只与 lambda 有关 (w 的插值系数)
    for i in range(n):
        for k in range(M + 1):
            c[i*(M+1) + k] = y_grid[k]
            
    # --- 约束构建 ---
    A_rows = []
    b_l = []
    b_u = []
    
    # 辅助函数: 添加一行约束
    def add_constr(coeffs, lb, ub):
        A_rows.append(coeffs)
        b_l.append(lb)
        b_u.append(ub)
        
    # 1. 物理约束 (A_ub @ p <= b_ub)
    # 将 p 替换为 sum x_k * lambda_{i,k}
    if A_ub is not None:
        for r in range(A_ub.shape[0]):
            row = np.zeros(n_vars)
            for i in range(n):
                coef_pi = A_ub[r, i]
                # 分配给对应的 lambda
                for k in range(M + 1):
                    row[i*(M+1) + k] = coef_pi * x_grid[k]
            add_constr(row, -np.inf, b_ub[r])

    # 2. 物理约束 (A_eq @ p == b_eq)
    if A_eq is not None:
        for r in range(A_eq.shape[0]):
            row = np.zeros(n_vars)
            for i in range(n):
                coef_pi = A_eq[r, i]
                for k in range(M + 1):
                    row[i*(M+1) + k] = coef_pi * x_grid[k]
            add_constr(row, b_eq[r], b_eq[r])
            
    # 3. Lambda Sum = 1 (每个变量 i)
    for i in range(n):
        row = np.zeros(n_vars)
        for k in range(M + 1):
            row[i*(M+1) + k] = 1.0
        add_constr(row, 1.0, 1.0)
        
    # 4. SOS2 逻辑约束 (Lambda 和 Z 的关系)
    # lambda_{i,0} <= z_{i,0}
    # lambda_{i,k} <= z_{i,k-1} + z_{i,k}
    # lambda_{i,M} <= z_{i,M-1}
    # sum z_{i,k} = 1
    
    for i in range(n):
        base_lam = i * (M + 1)
        base_z   = num_lambda + i * M
        
        # Sum Z = 1
        row_z = np.zeros(n_vars)
        for k in range(M):
            row_z[base_z + k] = 1.0
        add_constr(row_z, 1.0, 1.0)
        
        # Link constraints
        # k=0
        row = np.zeros(n_vars); row[base_lam] = 1.0; row[base_z] = -1.0
        add_constr(row, -np.inf, 0.0)
        
        # k=1..M-1
        for k in range(1, M):
            row = np.zeros(n_vars)
            row[base_lam + k] = 1.0
            row[base_z + k - 1] = -1.0
            row[base_z + k] = -1.0
            add_constr(row, -np.inf, 0.0)
            
        # k=M
        row = np.zeros(n_vars); row[base_lam + M] = 1.0; row[base_z + M - 1] = -1.0
        add_constr(row, -np.inf, 0.0)

    # --- 求解设置 ---
    # 变量类型: Lambda 是连续(0), Z 是整数(1)
    integrality = np.zeros(n_vars)
    integrality[num_lambda:] = 1 
    
    constraints = LinearConstraint(np.array(A_rows), np.array(b_l), np.array(b_u))
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=Bounds(0, 1))
    
    if not res.success:
        return None, None
    
    # 恢复 p
    lambdas = res.x[:num_lambda]
    p_opt = np.zeros(n)
    for i in range(n):
        for k in range(M + 1):
            p_opt[i] += lambdas[i*(M+1) + k] * x_grid[k]
            
    return p_opt, res.fun

# ===========================
# 2. 主算法: Outer Approximation
# ===========================
def optimize_min_entropy(POVM, max_iter=30, tol=1e-4):
    n = len(POVM)
    d = POVM[0].shape[0]
    
    print(f"Running Optimization N={n}, d={d}, MaxIter={max_iter}")
    
    # 初始约束
    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])
    
    # 切平面集合
    R_curr = np.empty((0, n))
    h_curr = np.empty(0)
    
    # 初始化一些 Box 约束方向，防止一开始 Polytope 无界/太大
    # 虽然 p_i >= 0 是隐含的，但加上 R方向有助于 MIP
    for i in range(n):
        r = np.zeros(n); r[i] = -1.0
        # h(r) = min Tr(rho * -E_i) = - max_eig(E_i)
        val = -np.linalg.eigvalsh(POVM[i])[-1]
        
        R_curr = np.vstack([R_curr, r]) if R_curr.size else np.atleast_2d(r)
        h_curr = np.append(h_curr, val)

    # 记录历史数据
    history = {
        'iter': [],
        'lb': [],
        'ub': [],
        'gap': []
    }
    
    best_ub = float('inf')
    best_p = None
    
    t_start = time.time()
    
    for k in range(1, max_iter + 1):
        # 1. 求解松弛问题 (MIP) -> Lower Bound
        # A_ub = -R, b_ub = -h  =>  R*p >= h
        p_k, lb_val = solve_milp_subproblem(-R_curr, -h_curr, A_eq, b_eq, n, num_segments=20)
        
        if p_k is None:
            print("Infeasible Subproblem.")
            break
            
        c_minus = lb_val
        
        # 2. 求解真实量子态 (Gradient Mapping) -> Upper Bound
        # 计算当前解 p_k 处的熵的梯度
        p_safe = np.clip(p_k, 1e-12, 1.0)
        grad = -(np.log2(p_safe) + 1.442695) # dH/dp
        
        # 求解 min <grad, p> = min Tr(rho * sum grad_i E_i)
        Sum_Op = np.zeros((d, d), dtype=complex)
        for i in range(n):
            Sum_Op += grad[i] * POVM[i]
            
        evals, evecs = eigh(Sum_Op)
        min_val = evals[0] # 切平面的右端项 h(grad)
        psi = evecs[:, 0]
        
        # 计算对应真实态的熵
        p_real = np.array([np.real(psi.conj().T @ E @ psi) for E in POVM])
        p_real = np.clip(p_real, 1e-16, 1.0) # 修正数值误差
        ub_val = -np.sum(p_real * np.log2(p_real))
        
        # 更新全局最优
        if ub_val < best_ub:
            best_ub = ub_val
            best_p = p_real
            
        c_plus = best_ub
        gap = abs(c_plus - c_minus)
        
        # 记录
        history['iter'].append(k)
        history['lb'].append(c_minus)
        history['ub'].append(c_plus)
        history['gap'].append(gap)
        
        print(f"Iter {k:<3} | LB: {c_minus:.5f} | UB: {c_plus:.5f} | Gap: {gap:.5e} | T: {time.time()-t_start:.1f}s")
        
        if gap < tol:
            print("Converged!")
            break
            
        # 3. 添加切平面 (Cut)
        # linear cut: grad * p >= min_val
        R_curr = np.vstack([R_curr, grad])
        h_curr = np.append(h_curr, min_val)
        
    return best_p, history

# ===========================
# 3. 可视化绘图模块
# ===========================
def plot_results(history, n, d):
    iters = history['iter']
    lbs = history['lb']
    ubs = history['ub']
    gaps = history['gap']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 图1: 上下界收敛
    ax1.plot(iters, ubs, 'r-o', label='Upper Bound (Best Feasible)', markersize=4)
    ax1.plot(iters, lbs, 'b-s', label='Lower Bound (MIP Relax)', markersize=4)
    ax1.fill_between(iters, lbs, ubs, color='gray', alpha=0.1)
    
    ax1.set_ylabel('Shannon Entropy')
    ax1.set_title(f'Convergence of Min-Entropy (N={n}, d={d})')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # 图2: 精度 Gap
    ax2.plot(iters, gaps, 'k-^', label='Gap (UB - LB)', markersize=4)
    ax2.set_yscale('log') # 关键：用对数坐标看收敛级数
    ax2.set_ylabel('Gap (Log Scale)')
    ax2.set_xlabel('Iteration')
    ax2.set_title('Convergence Precision')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

# ===========================
# 4. 执行测试 N=10
# ===========================
if __name__ == "__main__":
    # 构造 N=10, d=3 的 POVM (确保是高维情况)
    N = 3
    d = 3
    
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

        # d = 5
    ms_qobj = random_haar_povm(d, k=N, n=d, real=False)
    POVM1 = [q.full() for q in ms_qobj]
        
    # 运行
    best_p, hist = optimize_min_entropy(POVM1, max_iter=50, tol=1e-4)
    
    # 绘图
    if len(hist['iter']) > 0:
        plot_results(hist, N, d)
    else:
        print("No data to plot.")
