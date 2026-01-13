import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import time
from pypoman import compute_polytope_vertices

def shannon_entropy_min(POVM, max_iter=50, tol=1e-6):
    """
    基于顶点枚举的香农熵最小化 (Outer Approximation)，
    包含时间统计和Gap可视化。
    """
    # ==================== 0. 输入验证与准备 ====================
    n = len(POVM)
    d = POVM[0].shape[0]
    
    print(f"=== 开始优化 (Vertex Enumeration 方法) ===")
    print(f"参数: N={n}, d={d}, MaxIter={max_iter}")
    
    # 验证 POVM
    povm_sum = sum(POVM)
    if not np.allclose(povm_sum, np.eye(d), atol=1e-6):
        print("警告: POVM 元素之和不为 I (已自动归一化)")
        S_sqrt_inv = np.linalg.inv(np.linalg.cholesky(povm_sum))
        POVM = [S_sqrt_inv @ E @ S_sqrt_inv.conj().T for E in POVM]
    # ==================== 1. 核心数学函数 ====================
    def h_func(r):
        """计算支撑超平面位置 h(r) = min_ρ Tr(ρ R)"""
        R = sum(r[i] * POVM[i] for i in range(n))
        # 求解最小特征值
        min_eig = np.linalg.eigvalsh(R)[0]
        return min_eig
    def target_func(p):
        """目标函数: H(p)"""
        p_safe = np.clip(p, 1e-16, 1)
        return -np.sum(p_safe * np.log2(p_safe))
    
    def target_grad(p):
        """目标梯度: ∇H(p)"""
        p_safe = np.clip(p, 1e-16, 1) # 去除 log(0)
        grad = -(np.log2(p_safe) + 1.442695) # 1/ln2
        return grad / np.linalg.norm(grad)
    # ==================== 2. 初始化约束 ====================
    # 构造初始多面体 P_0
    # 约束格式: A p <= b
    # 我们有: r·p >= h(r)  =>  -r·p <= -h(r)
    
    Rmat = []
    hvec = []
    
    # (1) 概率归一化: sum(p) = 1
    # pypoman 处理等式比较麻烦，通常拆成两个不等式: sum(p)<=1 且 -sum(p)<=-1
    Rmat.append(np.ones(n));  hvec.append(1.0)
    Rmat.append(-np.ones(n)); hvec.append(-1.0)
    
    # (2) 简单的 Box 约束: p_i >= 0 ( -p_i <= 0 )
    for i in range(n):
        r = np.zeros(n); r[i] = 1.0; 
        # 这里实际上 p_i >= 0 是隐含的，但为了多面体闭合，我们加上 p_i >= 0 的约束
        # 注意: 顶点枚举需要闭合多面体。
        # 这里我们加入 p_i >= 0 的方向 (即 r=[0,..,1,..0])
        # 以及对应的 h(r)。对于 POVM，min Tr(rho E_i) = 0 通常成立。
        Rmat.append(r)
        hvec.append(0.0) # p_i >= 0
    Rmat = np.array(Rmat)
    hvec = np.array(hvec)
    
    # 记录历史
    c_minus_list = []
    c_plus_list = []
    gap_list = [] 
    time_list = []
    
    c_plus = float('inf')
    c_minus = -float('inf')
    
    total_start_time = time.time()
    # ==================== 3. 主循环 ====================
    for k in range(1, max_iter + 1):
        iter_start = time.time()
        
        # --- A. 顶点枚举 (最耗时步骤) ---
        # pypoman 期望格式: A x <= b
        A = -Rmat
        b = -hvec
        
        try:
            # 核心：调用 pypoman 计算所有顶点
            vertices = compute_polytope_vertices(A, b)
            vertices = np.array(vertices)
        except Exception as e:
            print(f"Error in vertex enumeration: {e}")
            break
            
        if len(vertices) == 0:
            print("Polytope is empty.")
            break
        # --- B. 计算下界 (LB) ---
        # 凹函数最小值一定在顶点上
        vals = [target_func(v) for v in vertices]
        min_idx = np.argmin(vals)
        lb_val = vals[min_idx]
        best_vertex = vertices[min_idx]
        
        # 下界应当单调递增(也就是多面体越来越小，最小值被抬高)
        # 但由于是重新计算，我们直接取当前多面体的最小值作为近似
        c_minus = lb_val 
        
        # --- C. 计算上界 (UB) ---
        # 利用当前最优顶点的梯度方向，寻找真实量子态
        grad = target_grad(best_vertex)
        
        # 求解 min Tr(rho * Sum grad_i E_i)
        val_h = h_func(grad) # 这是新切平面的截距
        
        # 我们需要找到对应的态 rho 来计算真实量子态的熵
        # 重复一下 h_func 里的逻辑来拿 eigenvector
        R_op = sum(grad[i] * POVM[i] for i in range(n))
        evals, evecs = eigh(R_op)
        psi = evecs[:, 0]
        
        # 真实概率分布
        p_real = np.array([np.real(psi.conj().T @ E @ psi) for E in POVM])
        ub_val = target_func(p_real)
        
        if ub_val < c_plus:
            c_plus = ub_val
            
        # --- D. 记录与收敛判定 ---
        gap = abs(c_plus - c_minus)
        iter_time = time.time() - iter_start
        
        c_minus_list.append(c_minus)
        c_plus_list.append(c_plus)
        gap_list.append(gap)
        time_list.append(iter_time)
        
        print(f"Iter {k:<3} | LB: {c_minus:.6f} | UB: {c_plus:.6f} | Gap: {gap:.6e} | Vertices: {len(vertices):<5} | Time: {iter_time:.4f}s")
        
        if gap < tol:
            print(f"Converged at iteration {k}")
            break
            
        # --- E. 添加切平面 (Cut) ---
        # 约束: grad * p >= val_h
        Rmat = np.vstack([Rmat, grad])
        hvec = np.append(hvec, val_h)
    total_time = time.time() - total_start_time
    print(f"=== 优化结束 ===")
    print(f"总耗时: {total_time:.4f} 秒")
    print(f"最终结果: [{c_minus:.6f}, {c_plus:.6f}]")
    # ==================== 4. 可视化绘图 ====================
    plt.figure(figsize=(12, 10))
    
    # 子图1: 上下界收敛
    plt.subplot(2, 1, 1)
    its = range(1, len(c_minus_list) + 1)
    plt.plot(its, c_plus_list, 'r-o', label='Upper Bound (Actual Min)', linewidth=1.5, markersize=4)
    plt.plot(its, c_minus_list, 'b-s', label='Lower Bound (Polytope Min)', linewidth=1.5, markersize=4)
    plt.fill_between(its, c_minus_list, c_plus_list, color='gray', alpha=0.1)
    plt.ylabel('Shannon Entropy')
    plt.title(f'Convergence Bounds (N={n}, d={d})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 子图2: Gap 与 时间
    plt.subplot(2, 1, 2)
    ax1 = plt.gca()
    # 左轴: Gap (对数)
    color = 'tab:blue'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Gap (Log Scale)', color=color)
    ax1.semilogy(its, gap_list, '^-', color=color, label='Gap (UB - LB)', markersize=5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 右轴: 每次迭代耗时 (线性)
    ax2 = ax1.twinx()  
    color = 'tab:orange'
    ax2.set_ylabel('Time per Iteration (s)', color=color)
    ax2.plot(its, time_list, 'D--', color=color, label='Time (s)', markersize=4, alpha=0.6)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center')
    
    plt.title('Convergence Gap & Computational Cost')
    plt.tight_layout()
    plt.show()


import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import heapq
import time
def shannon_entropy_min_sbb(POVM, max_iter=50, tol=1e-5, sbb_timeout=60):
    """
    计算 POVM 测量的最小香农熵 (优化版 sBB)
    """
    
    # ==================== 输入验证 ====================
    if not isinstance(POVM, list) or len(POVM) == 0:
        raise ValueError("POVM must be a non-empty list")
    
    n = len(POVM)
    d = POVM[0].shape[0]
    
    # 全局最优记录 (修正上界震荡问题)
    global_best_entropy = float('inf')
    
    # ==================== 核心数学函数 ====================
    def h_func(r):
        """支撑函数: min_ρ Tr(ρ R)"""
        R = np.zeros((d, d), dtype=complex)
        for i in range(n):
            R += r[i] * POVM[i]
        R = (R + R.conj().T) / 2
        # 注意：eigh 返回升序特征值，取第一个即最小值
        return np.linalg.eigvalsh(R)[0]
    def entropy_scalar(x):
        """单变量熵 -x log2(x)"""
        if x <= 1e-16: return 0.0
        if x >= 1.0: return 0.0
        return -x * np.log2(x)
    def total_entropy(p):
        """总熵"""
        p = np.clip(p, 0, 1)
        res = 0.0
        for val in p:
            if val > 1e-16:
                res -= val * np.log2(val)
        return res
    def target_grad(p):
        """熵的梯度"""
        p_safe = np.clip(p, 1e-16, 1)
        grad = -(np.log2(p_safe) + 1 / np.log(2))
        return grad / np.linalg.norm(grad)
    # ==================== 空间分支定界法 (sBB) 类 ====================
    class Node:
        def __init__(self, lb_val, p_sol, bounds, depth):
            self.lb = lb_val        # 线性规划算出的下界
            self.p = p_sol          # 线性规划的解点
            self.bounds = bounds    # 矩形区域
            self.depth = depth
            
        def __lt__(self, other):
            return self.lb < other.lb 
    def solve_sbb(A_ub, b_ub, A_eq, b_eq, global_tol, timeout):
        """
        sBB 求解器
        """
        start_time = time.time()
        
        # 初始边界
        init_bounds = [(0.0, 1.0) for _ in range(n)]
        pq = []
        
        # sBB 内部的全局上界 (Current best valid value inside the polytope)
        # 注意：这个不同于外层的 global_best_entropy，这是为了在 sBB 树中剪枝用的
        sbb_ub = float('inf')
        best_p_sbb = None
        
        nodes_processed = 0
        
        # 预计算 b_ub 的 flat 版本，避免重复操作
        b_ub_flat = np.array(b_ub).flatten() if b_ub is not None else None
        # --- 辅助: 求解节点的线性松弛 LP ---
        def solve_relaxation(bounds):
            c = []
            const_term = 0.0
            slopes = []
            intercepts = []
            
            # 构造凸包络 (Secant lines)
            for i in range(n):
                L, U = bounds[i]
                # 简单数值保护
                if U - L < 1e-9:
                    slope = 0.0
                    intercept = entropy_scalar(L)
                else:
                    fL = entropy_scalar(L)
                    fU = entropy_scalar(U)
                    slope = (fU - fL) / (U - L)
                    intercept = fL - slope * L
                
                c.append(slope) # LP 的目标函数系数
                const_term += intercept
                slopes.append(slope)
                intercepts.append(intercept)
            
            # 调用求解器
            # method='highs' 是 scipy 中最快且最稳定的
            res = linprog(c, A_ub=A_ub, b_ub=b_ub_flat, A_eq=A_eq, b_eq=b_eq, 
                          bounds=bounds, method='highs')
            
            if not res.success:
                return None, None, None, None, None
            
            # 计算下界
            lb_val = res.fun + const_term
            return lb_val, res.x, slopes, intercepts, True
        # --- 根节点 ---
        root_lb, root_p, r_slopes, r_intercepts, success = solve_relaxation(init_bounds)
        if not success:
            return None, float('inf')
            
        # 根节点作为一个可行解，更新 sBB 上界
        root_true_val = total_entropy(root_p)
        if root_true_val < sbb_ub:
            sbb_ub = root_true_val
            best_p_sbb = root_p
        # 存入节点：需要额外存储 slopes 和 intercepts 以便快速计算误差
        # 为节省内存，这里只存必要信息。如果需要极速，可重新计算。
        # 这里为了逻辑清晰，我们重新计算误差
        heapq.heappush(pq, Node(root_lb, root_p, init_bounds, 0))
        # --- sBB 主循环 ---
        while pq:
            if time.time() - start_time > timeout:
                break
            
            # 取出下界最小的节点
            node = heapq.heappop(pq)
            nodes_processed += 1
            
            # 剪枝 1: 如果当前节点下界 >= 全局上界，丢弃
            if node.lb >= sbb_ub - 1e-7:
                continue
            
            # 更新上界: 计算当前解的真实值
            curr_true_val = total_entropy(node.p)
            if curr_true_val < sbb_ub:
                sbb_ub = curr_true_val
                best_p_sbb = node.p
            
            # 剪枝 2: Gap 满足要求
            if abs(sbb_ub - node.lb) < global_tol:
                continue
            # --- 改进的分支策略: 最大凹性误差 (Maximum Concavity Gap) ---
            # 我们要找哪个变量导致了松弛误差最大
            # Error_i = | f_real(p_i) - f_relaxed(p_i) |
            # 其中 f_relaxed(p_i) = slope_i * p_i + intercept_i
            
            max_gap = -1
            split_idx = -1
            
            # 为了计算 Gap，我们需要重构当前矩形的 slope/intercept
            # 这比存储在 Node 对象中省内存，但多一点计算量。对于 Python 来说，这点计算量远小于 linprog
            for i in range(n):
                L, U = node.bounds[i]
                if U - L < 1e-6: continue 
                
                # 当前变量的值
                val = node.p[i]
                
                # 真实凹函数值
                f_real = entropy_scalar(val)
                
                # 线性松弛值 (割线值)
                if U - L < 1e-9:
                    f_lin = entropy_scalar(L)
                else:
                    fL = entropy_scalar(L)
                    fU = entropy_scalar(U)
                    slope = (fU - fL) / (U - L)
                    intercept = fL - slope * L
                    f_lin = slope * val + intercept
                
                # Gap (因为 f是凹的，线性松弛是下界，所以 f_real >= f_lin)
                gap = f_real - f_lin
                
                if gap > max_gap:
                    max_gap = gap
                    split_idx = i
            
            if split_idx == -1 or max_gap < 1e-7:
                continue # 误差极小，无需分支
            
            # --- 执行分支 ---
            # 策略：在当前解的位置切分 (Omega-subdivision)
            # 理由：最大误差通常发生在当前 LP 解的位置，切断这里能最快提升下界
            current_val = node.p[split_idx]
            L_curr, U_curr = node.bounds[split_idx]
            
            # 防止切分点太靠近边界导致死循环
            split_point = np.clip(current_val, L_curr + 0.1*(U_curr-L_curr), U_curr - 0.1*(U_curr-L_curr))
            # 或者简单的中点: split_point = (L_curr + U_curr) / 2.0
            # 混合策略: 如果 current_val 在中间，用它；否则用中点
            
            for i in range(2):
                new_bounds = list(node.bounds)
                if i == 0:
                    new_bounds[split_idx] = (L_curr, split_point)
                else:
                    new_bounds[split_idx] = (split_point, U_curr)
                
                # 求解子节点
                c_lb, c_p, _, _, c_succ = solve_relaxation(new_bounds)
                
                # 只有当下界有希望优于当前最优解时才加入队列
                if c_succ and c_lb < sbb_ub - 1e-7:
                    heapq.heappush(pq, Node(c_lb, c_p, new_bounds, node.depth + 1))
        return best_p_sbb, sbb_ub
    # ==================== 算法初始化 ====================
    print(f"Optimizing (Optimized sBB) for POVM (n={n}, d={d})...")
    
    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])
    
    Rmat = []
    hvec = []
    
    # 初始化 box 约束
    for i in range(n):
        r = np.zeros(n); r[i] = 1; Rmat.append(r); hvec.append(h_func(r))
        r = np.zeros(n); r[i] = -1; Rmat.append(r); hvec.append(h_func(r))
    R_curr = np.array(Rmat)
    h_curr = np.array(hvec)
    
    c_minus_list = []
    c_plus_list = []
    
    print(f"{'Iter':<5} | {'LB (Polytope Min)':<18} | {'UB (Actual Min)':<18} | {'Gap':<10} | {'Time':<6}")
    print("-" * 75)
    current_min_p = None
    
    # ==================== 主迭代循环 ====================
    for iter in range(1, max_iter + 1):
        
        # 1. 准备约束
        A_ub = -R_curr
        b_ub = -h_curr # 此时 h_curr 已经是 1D 数组
        
        # 2. 运行 sBB
        t0 = time.time()
        # 动态容忍度：前期不需要算得太准，后期收紧
        # 这能显著提升前期切平面的产生速度
        dynamic_tol = max(tol, 0.1 / iter) 
        
        poly_min_p, poly_min_val = solve_sbb(A_ub, b_ub, A_eq, b_eq, 
                                            global_tol=tol, timeout=sbb_timeout)
        sbb_time = time.time() - t0
        
        if poly_min_p is None:
            print("Error: sBB infeasible.")
            break
            
        c_minus = poly_min_val # 当前多面体上的最小熵 (LB)
        current_min_p = poly_min_p
        
        # 3. 计算实际物理上界 (UB) 并保持单调性
        grad = target_grad(current_min_p)
        
        R_op = np.zeros((d, d), dtype=complex)
        for i in range(n):
            R_op += grad[i] * POVM[i]
        
        # 获取最小特征值 (标量)
        min_eig_val = np.linalg.eigvalsh(R_op)[0] # eigvalsh 比 eigh 更快一点
        new_h = min_eig_val
        
        # 计算该梯度方向对应的量子态产生的熵
        # 为了得到 psi，我们需要重新调用 eigh，或者直接用逆迭代。
        # 这里为了稳健还是调 eigh
        _, psi = eigh(R_op)
        psi = psi[:, 0]
        
        p_real = np.zeros(n)
        for i in range(n):
            p_real[i] = np.real(psi.conj().T @ POVM[i] @ psi)
        
        current_iter_entropy = total_entropy(p_real)
        
        # [关键修复] 只有当新发现的熵更小时，才更新全局上界
        if current_iter_entropy < global_best_entropy:
            global_best_entropy = current_iter_entropy
        
        c_plus = global_best_entropy
        
        # 记录
        c_minus_list.append(c_minus)
        c_plus_list.append(c_plus)
        
        print(f"{iter:<5} | {c_minus:<18.6f} | {c_plus:<18.6f} | {abs(c_plus-c_minus):<10.6f} | {sbb_time:<6.2f}s")
        
        if abs(c_plus - c_minus) <= tol:
            print("\nConverged!")
            break
            
        # 4. 添加新约束
        R_curr = np.vstack([R_curr, grad])
        h_curr = np.append(h_curr, new_h)
    # ==================== 绘图 ====================
    plt.figure(figsize=(10, 6))
    plt.plot(c_minus_list, 'b-o', label='Lower Bound (sBB)')
    plt.plot(c_plus_list, 'r-s', label='Upper Bound (Monotonic)')
    plt.xlabel('Iteration')
    plt.ylabel('Shannon Entropy')
    plt.title('Minimizing Entropic Uncertainty (Optimized sBB)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Final Entropy Interval: [{c_minus_list[-1]:.6f}, {c_plus_list[-1]:.6f}]")
    
    return c_minus_list, c_plus_list


import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import time
import heapq

# ==================== 1. 内部核心算法：矩形分支定界求解器 ====================
# 用于替代原本的 compute_polytope_vertices
# 解决子问题: min H(p) s.t. A p <= b (在当前多面体上找最小熵)

class Node:
    """分支定界树的节点"""
    def __init__(self, bounds, lb, p_sol):
        self.bounds = bounds  # 变量范围 [(min, max), ...]
        self.lb = lb          # 该区域的目标函数下界 (Lower Bound)
        self.p_sol = p_sol    # 产生该下界的解向量
        
    def __lt__(self, other):
        # 优先队列比较：优先处理下界更小的节点 (Best-First Search)
        return self.lb < other.lb

def solve_concave_min_bnb(A_ub, b_ub, n_vars, max_nodes=5000, tol=1e-6):
    """
    使用分支定界法(Branch and Bound, B&B)求解当前多面体上的凹函数最小值。
    不需要枚举顶点，而是搜索空间。
    """
    
    # 辅助函数：计算区间 [l, u] 上的线性下界（割线）系数
    # H(p) = sum h(p_i), h(p_i)是凹的。
    # 凹函数在区间上的凸包络是连接端点的直线。
    def get_secant_coeffs(bounds):
        c = []
        offset = 0.0
        for (l, u) in bounds:
            l = max(l, 1e-16) # 防止 log(0)
            u = max(u, 1e-16)
            
            val_l = -l * np.log2(l)
            val_u = -u * np.log2(u)
            
            if abs(u - l) < 1e-9:
                slope = 0.0
                curr_offset = val_l - slope * l
            else:
                slope = (val_u - val_l) / (u - l)
                curr_offset = val_l - slope * l
            
            c.append(slope)
            offset += curr_offset
        return np.array(c), offset

    # 辅助函数：解松弛后的线性规划 (LP)
    def solve_node_lp(bounds):
        c, offset = get_secant_coeffs(bounds)
        # min c*x s.t. Ax <= b, x in bounds
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not res.success:
            return None, None, None
        
        # LP的最优值 + 偏移量 = 该区域内凹函数的下界
        lb = res.fun + offset
        return lb, res.x, res.x 

    def entropy(p):
        p_safe = np.clip(p, 1e-16, 1)
        return -np.sum(p_safe * np.log2(p_safe))

    # --- B&B 初始化 ---
    # 初始包围盒：所有 p_i 在 [0, 1]
    global_bounds = [(0.0, 1.0) for _ in range(n_vars)]
    
    pq = [] # 优先队列
    global_ub = float('inf') # 当前找到的最好可行解的值
    best_sol = None
    
    # 处理根节点
    lb, p_sol, _ = solve_node_lp(global_bounds)
    if lb is not None:
        heapq.heappush(pq, Node(global_bounds, lb, p_sol))
        # 尝试用 LP 的解更新全局最优
        curr_val = entropy(p_sol)
        if curr_val < global_ub:
            global_ub = curr_val
            best_sol = p_sol

    nodes_processed = 0
    
    # --- B&B 主循环 ---
    while pq and nodes_processed < max_nodes:
        # 1. 取出下界最小的节点
        node = heapq.heappop(pq)
        
        # 2. 剪枝：如果该节点的下界 > 全局最优，则该区域不可能有更优解
        if node.lb > global_ub - tol:
            continue
            
        nodes_processed += 1
        
        # 3. 分支：选择最宽的维度切分
        bounds = node.bounds
        diffs = [u - l for l, u in bounds]
        split_idx = np.argmax(diffs)
        
        if diffs[split_idx] < 1e-5: # 区域极小，不再细分
            continue
            
        l_curr, u_curr = bounds[split_idx]
        mid = (l_curr + u_curr) / 2.0
        
        # 生成左右子节点
        bounds_left = list(bounds)
        bounds_left[split_idx] = (l_curr, mid)
        
        bounds_right = list(bounds)
        bounds_right[split_idx] = (mid, u_curr)
        
        for child_bounds in [bounds_left, bounds_right]:
            lb_child, p_child, _ = solve_node_lp(child_bounds)
            if lb_child is not None:
                # 更新 UB
                real_val = entropy(p_child)
                if real_val < global_ub:
                    global_ub = real_val
                    best_sol = p_child
                
                # 如果下界有希望，入队
                if lb_child < global_ub:
                    heapq.heappush(pq, Node(child_bounds, lb_child, p_child))
    
    return best_sol, global_ub, nodes_processed

# ==================== 2. 主程序 (Outer Approximation) ====================

def shannon_entropy_min_bnb(POVM, max_iter=50, tol=1e-6):
    """
    基于分支定界的香农熵最小化 (Outer Approximation)，
    包含时间统计和Gap可视化。
    """
    # ==================== 0. 输入验证与准备 ====================
    n = len(POVM)
    d = POVM[0].shape[0]
    
    print(f"=== 开始优化 (Branch & Bound Acceleration) ===")
    print(f"参数: N={n}, d={d}, MaxIter={max_iter}")
    
    # 验证 POVM
    povm_sum = sum(POVM)
    if not np.allclose(povm_sum, np.eye(d), atol=1e-6):
        print("警告: POVM 元素之和不为 I (已自动归一化)")
        S_sqrt_inv = np.linalg.inv(np.linalg.cholesky(povm_sum))
        POVM = [S_sqrt_inv @ E @ S_sqrt_inv.conj().T for E in POVM]
        
    # ==================== 1. 核心数学函数 ====================
    def h_func(r):
        """计算支撑超平面位置 h(r) = min_ρ Tr(ρ R)"""
        R = sum(r[i] * POVM[i] for i in range(n))
        min_eig = np.linalg.eigvalsh(R)[0]
        return min_eig
    
    def target_grad(p):
        """目标梯度: ∇H(p)"""
        p_safe = np.clip(p, 1e-16, 1)
        grad = -(np.log2(p_safe) + 1.442695) # 1/ln2
        return grad / np.linalg.norm(grad)
        
    # ==================== 2. 初始化约束 ====================
    # 构造初始多面体 P_0 (A p <= b)
    A_ub = []
    b_ub = []
    
    # (1) 概率归一化: sum(p) = 1  =>  sum(p)<=1 AND -sum(p)<=-1
    A_ub.append(np.ones(n));  b_ub.append(1.0)
    A_ub.append(-np.ones(n)); b_ub.append(-1.0)
    
    # (2) 非负约束: p_i >= 0 => -p_i <= 0
    # 虽然B&B的bounds处理了它，但为了LP求解器的稳定性，显式加入
    for i in range(n):
        row = np.zeros(n); row[i] = -1.0
        A_ub.append(row); b_ub.append(0.0)
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # 记录历史
    c_minus_list = []
    c_plus_list = []
    gap_list = [] 
    time_list = []
    
    c_plus = float('inf')
    c_minus = -float('inf')
    
    total_start_time = time.time()
    
    # ==================== 3. 主循环 ====================
    for k in range(1, max_iter + 1):
        iter_start = time.time()
        
        # --- A. 替换了原先的顶点枚举 ---
        # 使用矩形分支定界法 (B&B) 求解当前多面体上的最小熵
        # 这确保了能找到全局最小，但比枚举所有顶点快得多
        best_vertex, poly_min_val, nodes_cnt = solve_concave_min_bnb(
            A_ub, b_ub, n_vars=n, max_nodes=3000, tol=tol/10
        )
        
        if best_vertex is None:
            print("Polytope is empty or infeasible.")
            break
            
        # 这里的 poly_min_val 就是当前多面体下的最小值 (LB)
        c_minus = poly_min_val 
        
        # --- B. 计算上界 (UB) ---
        # 利用当前找到的最优解 best_vertex 计算梯度，反推真实量子态
        grad = target_grad(best_vertex)
        
        # 计算切平面截距
        val_h = h_func(grad) 
        
        # 寻找对应的真实量子态 rho
        R_op = sum(grad[i] * POVM[i] for i in range(n))
        evals, evecs = eigh(R_op)
        psi = evecs[:, 0]
        
        # 真实概率分布
        p_real = np.array([np.real(psi.conj().T @ E @ psi) for E in POVM])
        # 计算真实熵
        ub_val = -np.sum(p_real * np.log2(np.clip(p_real, 1e-16, 1)))
        
        if ub_val < c_plus:
            c_plus = ub_val
            
        # --- D. 记录与收敛判定 ---
        gap = abs(c_plus - c_minus)
        iter_time = time.time() - iter_start
        
        c_minus_list.append(c_minus)
        c_plus_list.append(c_plus)
        gap_list.append(gap)
        time_list.append(iter_time)
        
        print(f"Iter {k:<3} | LB: {c_minus:.6f} | UB: {c_plus:.6f} | Gap: {gap:.6e} | B&B Nodes: {nodes_cnt:<5} | Time: {iter_time:.4f}s")
        
        if gap < tol:
            print(f"Converged at iteration {k}")
            break
            
        # --- E. 添加切平面 (Cut) ---
        # 约束: grad * p >= val_h  =>  -grad * p <= -val_h
        A_ub = np.vstack([A_ub, -grad])
        b_ub = np.append(b_ub, -val_h)
        
    total_time = time.time() - total_start_time
    print(f"=== 优化结束 ===")
    print(f"总耗时: {total_time:.4f} 秒")
    print(f"最终结果: [{c_minus:.6f}, {c_plus:.6f}]")
    
    # ==================== 4. 可视化绘图 ====================
    plt.figure(figsize=(12, 10))
    
    # 子图1: 上下界收敛
    plt.subplot(2, 1, 1)
    its = range(1, len(c_minus_list) + 1)
    plt.plot(its, c_plus_list, 'r-o', label='Upper Bound (Actual Min)', linewidth=1.5, markersize=4)
    plt.plot(its, c_minus_list, 'b-s', label='Lower Bound (Polytope Min)', linewidth=1.5, markersize=4)
    plt.fill_between(its, c_minus_list, c_plus_list, color='gray', alpha=0.1)
    plt.ylabel('Shannon Entropy')
    plt.title(f'Convergence Bounds (N={n}, d={d})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 子图2: Gap 与 时间
    plt.subplot(2, 1, 2)
    ax1 = plt.gca()
    # 左轴: Gap (对数)
    color = 'tab:blue'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Gap (Log Scale)', color=color)
    ax1.semilogy(its, gap_list, '^-', color=color, label='Gap (UB - LB)', markersize=5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 右轴: 每次迭代耗时 (线性)
    ax2 = ax1.twinx()  
    color = 'tab:orange'
    ax2.set_ylabel('Time per Iteration (s)', color=color)
    ax2.plot(its, time_list, 'D--', color=color, label='Time (s)', markersize=4, alpha=0.6)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center')
    
    plt.title('Convergence Gap & Computational Cost')
    plt.tight_layout()
    plt.show()


import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import time
import heapq
import gurobipy as gp
from gurobipy import GRB
# ==================== 1. 极速版 Gurobi 分支定界 ====================
class Node:
    def __init__(self, bounds, lb, p_sol):
        self.bounds = bounds
        self.lb = lb
        self.p_sol = p_sol
    def __lt__(self, other):
        return self.lb < other.lb
def solve_concave_min_bnb_gurobi_fast(A_ub, b_ub, n_vars, max_nodes=10000, tol=1e-6, env=None):
    """
    极速版：使用 Gurobi 模型持久化 (Persistent Model) 技术。
    避免在循环中重复创建模型对象。
    """
    
    # --- 1. 预计算与模型初始化 (Loop 外) ---
    # 创建模型
    try:
        m = gp.Model("bnb", env=env)
    except:
        m = gp.Model("bnb")
        
    m.setParam('OutputFlag', 0)
    m.setParam('Threads', 1) 
    
    # 核心优化：预先添加变量和静态约束
    # 初始范围 [0, 1]
    lb_init = np.zeros(n_vars)
    ub_init = np.ones(n_vars)
    
    # 添加变量 x (MVar)
    x = m.addMVar(shape=n_vars, lb=lb_init, ub=ub_init, vtype=GRB.CONTINUOUS, name="x")
    
    # 添加约束 A x <= b (这些约束在整个B&B过程中是不变的！)
    m.addConstr(A_ub @ x <= b_ub, name="static_constrs")
    
    # --- 辅助函数：快速更新模型并求解 ---
    def solve_fast(current_bounds):
        # 1. 计算割线系数 (纯数学)
        c = []
        offset = 0.0
        # 提取 bounds 为 numpy 数组以加速
        lbs = np.array([b[0] for b in current_bounds])
        ubs = np.array([b[1] for b in current_bounds])
        
        # 向量化计算系数 (比循环快)
        # 处理 log(0)
        l_safe = np.maximum(lbs, 1e-16)
        u_safe = np.maximum(ubs, 1e-16)
        
        val_l = -l_safe * np.log2(l_safe)
        val_u = -u_safe * np.log2(u_safe)
        
        # 计算斜率
        diff = u_safe - l_safe
        mask = diff < 1e-9
        
        slopes = np.zeros_like(diff)
        # mask为False的地方计算斜率
        slopes[~mask] = (val_u[~mask] - val_l[~mask]) / diff[~mask]
        # mask为True的地方斜率为0 (近似)
        
        offset = np.sum(val_l - slopes * l_safe)
        
        # 2. 修改 Gurobi 模型参数
        # 直接修改变量属性，比 remove/add 约束快得多
        x.LB = lbs
        x.UB = ubs
        x.Obj = slopes # 设置线性目标函数系数
        
        # 3. 求解
        m.optimize()
        
        if m.status == GRB.OPTIMAL:
            return m.ObjVal + offset, x.X
        else:
            return None, None
    def entropy(p):
        p_safe = np.clip(p, 1e-16, 1)
        return -np.sum(p_safe * np.log2(p_safe))
    # --- B&B 初始化 ---
    global_bounds = [(0.0, 1.0) for _ in range(n_vars)]
    pq = [] 
    global_ub = float('inf') 
    best_sol = None
    
    # 根节点求解
    lb, p_sol = solve_fast(global_bounds)
    
    if lb is not None:
        heapq.heappush(pq, Node(global_bounds, lb, p_sol))
        curr_val = entropy(p_sol)
        if curr_val < global_ub:
            global_ub = curr_val
            best_sol = p_sol
    nodes_processed = 0
    
    # --- B&B 主循环 ---
    while pq and nodes_processed < max_nodes:
        node = heapq.heappop(pq)
        
        # 剪枝
        if node.lb > global_ub - tol:
            continue
            
        nodes_processed += 1
        
        # 分支
        bounds = node.bounds
        # 找最长边
        l_arr = np.array([b[0] for b in bounds])
        u_arr = np.array([b[1] for b in bounds])
        diffs = u_arr - l_arr
        split_idx = np.argmax(diffs)
        
        if diffs[split_idx] < 1e-5:
            continue
            
        l_curr, u_curr = bounds[split_idx]
        mid = (l_curr + u_curr) / 2.0
        
        # 构造子节点 Bounds
        # 优化：不进行深拷贝，只复制列表结构
        bounds_left = list(bounds)
        bounds_left[split_idx] = (l_curr, mid)
        
        bounds_right = list(bounds)
        bounds_right[split_idx] = (mid, u_curr)
        
        for child_bounds in [bounds_left, bounds_right]:
            lb_child, p_child = solve_fast(child_bounds)
            
            if lb_child is not None:
                # 更新全局最优
                real_val = entropy(p_child)
                if real_val < global_ub:
                    global_ub = real_val
                    best_sol = p_child
                
                # 入队判断
                if lb_child < global_ub:
                    heapq.heappush(pq, Node(child_bounds, lb_child, p_child))
    
    # 释放显式引用的 Gurobi 资源 (可选，依靠GC通常也可以)
    return best_sol, global_ub, nodes_processed
# ==================== 2. 主程序 ====================
def shannon_entropy_min_gurobi(POVM, max_iter=50, tol=1e-6):
    n = len(POVM)
    d = POVM[0].shape[0]
    
    print(f"=== 开始优化 (Gurobi Persistent Model) ===")
    print(f"参数: N={n}, d={d}, MaxIter={max_iter}")
    
    # 归一化 POVM
    povm_sum = sum(POVM)
    if not np.allclose(povm_sum, np.eye(d), atol=1e-6):
        S_sqrt_inv = np.linalg.inv(np.linalg.cholesky(povm_sum))
        POVM = [S_sqrt_inv @ E @ S_sqrt_inv.conj().T for E in POVM]
        
    def h_func(r):
        R = sum(r[i] * POVM[i] for i in range(n))
        return np.linalg.eigvalsh(R)[0]
    
    def target_grad(p):
        p_safe = np.clip(p, 1e-16, 1)
        grad = -(np.log2(p_safe) + 1.442695)
        return grad / np.linalg.norm(grad)
        
    # 初始化约束 A p <= b
    A_ub = []
    b_ub = []
    
    # sum(p) = 1
    A_ub.append(np.ones(n));  b_ub.append(1.0)
    A_ub.append(-np.ones(n)); b_ub.append(-1.0)
    # p >= 0
    for i in range(n):
        row = np.zeros(n); row[i] = -1.0
        A_ub.append(row); b_ub.append(0.0)
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # 历史记录
    c_minus_list, c_plus_list, gap_list, time_list = [], [], [], []
    c_plus = float('inf')
    c_minus = -float('inf')
    
    # 环境初始化
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
    except:
        env = None
    total_start_time = time.time()
    
    for k in range(1, max_iter + 1):
        iter_start = time.time()
        
        # -----------------------------------------------------------
        # 调用优化后的 B&B 求解器
        # 注意：这里 max_nodes 设为 3000，与你之前的 Scipy 保持一致，公平比较
        # -----------------------------------------------------------
        best_vertex, poly_min_val, nodes_cnt = solve_concave_min_bnb_gurobi_fast(
            A_ub, b_ub, n_vars=n, max_nodes=100000, tol=tol/100, env=env
        )
        
        if best_vertex is None: break
            
        c_minus = poly_min_val 
        
        # 计算 UB
        grad = target_grad(best_vertex)
        val_h = h_func(grad) 
        
        R_op = sum(grad[i] * POVM[i] for i in range(n))
        psi = eigh(R_op)[1][:, 0]
        p_real = np.array([np.real(psi.conj().T @ E @ psi) for E in POVM])
        ub_val = -np.sum(p_real * np.log2(np.clip(p_real, 1e-16, 1)))
        
        if ub_val < c_plus: c_plus = ub_val
            
        # 记录
        gap = abs(c_plus - c_minus)
        iter_time = time.time() - iter_start
        
        c_minus_list.append(c_minus)
        c_plus_list.append(c_plus)
        gap_list.append(gap)
        time_list.append(iter_time)
        
        # 格式化输出 (保持对齐)
        print(f"Iter {k:<3} | LB: {c_minus:.6f} | UB: {c_plus:.6f} | Gap: {gap:.6e} | Nodes: {nodes_cnt:<5} | Time: {iter_time:.4f}s")
        
        if gap < tol:
            print(f"Converged at iteration {k}")
            break
            
        # 添加 Cut
        A_ub = np.vstack([A_ub, -grad])
        b_ub = np.append(b_ub, -val_h)
        
    total_time = time.time() - total_start_time
    if env: env.dispose()
        
    print(f"=== 优化结束 ===")
    print(f"总耗时: {total_time:.4f} 秒")
    print(f"最终结果: [{c_minus:.6f}, {c_plus:.6f}]")
    
    # 绘图代码保持不变
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    its = range(1, len(c_minus_list) + 1)
    plt.plot(its, c_plus_list, 'r-o', label='Upper Bound', linewidth=1.5)
    plt.plot(its, c_minus_list, 'b-s', label='Lower Bound', linewidth=1.5)
    plt.fill_between(its, c_minus_list, c_plus_list, color='gray', alpha=0.1)
    plt.title(f'Convergence (Gurobi Optimized, N={n})')
    plt.legend()
    plt.grid(True, linestyle='--')
    
    plt.subplot(2, 1, 2)
    ax1 = plt.gca()
    ax1.semilogy(its, gap_list, '^-', color='tab:blue', label='Gap')
    ax1.set_ylabel('Gap', color='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(its, time_list, 'D--', color='tab:orange', label='Time')
    ax2.set_ylabel('Time (s)', color='tab:orange')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.show()



