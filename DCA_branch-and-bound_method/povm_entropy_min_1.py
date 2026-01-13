# %%
import os
# ========================================================
# 0. 环境配置：锁死底层单线程，把算力留给 Python 多进程
# ========================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import time

# ========================================================
# 1. Worker 定义
# ========================================================

_global_povm = None

def init_worker_process(povm_data):
    global _global_povm
    _global_povm = povm_data
    # 确保随机性差异
    np.random.seed(os.getpid() + int(time.time()*10000) % 9999999)

def task_hybrid_dca(args):
    """
    混合搜索 Worker：
    mode='global': 纯随机全域搜索
    mode='local':  在上一轮最佳点附近进行扰动搜索 (Warm Start)
    """
    num_loops, n, A_ub, b_ub, A_eq, b_eq, center_p, mode = args
    
    local_best_p = None
    local_best_val = float('inf')
    bounds = (0, 1)
    
    # 预分配
    c_vec = np.zeros(n)
    
    for _ in range(num_loops):
        # --- 策略分支 ---
        if mode == 'local' and center_p is not None:
            # 在已知最优点附近加噪声 (局部深挖)
            noise = (np.random.rand(n) - 0.5) * 0.1 # 扰动幅度
            x_current = np.clip(center_p + noise, 1e-6, 1.0)
        else:
            # 全局随机 (广度搜索)
            x_current = np.random.rand(n)
        
        x_current /= np.sum(x_current)
        
        # 内层 DCA 循环
        # 注意：这里适度增加了 max_inner，保证收敛到底
        for _ in range(15): 
            np.clip(x_current, 1e-12, 1.0, out=x_current)
            np.log(x_current, out=c_vec)
            c_vec += 1.0
            c_vec *= -1.0 
            
            res = linprog(c_vec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                          bounds=bounds, method='highs') # highs 是目前最稳健的
            
            if not res.success: break
            
            x_next = res.x
            if np.linalg.norm(x_next - x_current) < 1e-6:
                x_current = x_next
                break
            x_current = x_next
            
        # 计算熵
        np.clip(x_current, 1e-12, 1.0, out=x_current)
        val = -np.sum(x_current * np.log(x_current))
        
        if val < local_best_val:
            local_best_val = val
            local_best_p = x_current.copy()
            
    return local_best_p

def task_oracle_check(p_candidate):
    global _global_povm
    if p_candidate is None: return None
    
    n = len(p_candidate)
    d = _global_povm[0].shape[0]
    
    grad_vec = -(1.0 + np.log(np.clip(p_candidate, 1e-12, 1.0)))
    
    R = np.zeros((d, d), dtype=complex)
    for i in range(n):
        R += grad_vec[i] * _global_povm[i]
    R = (R + R.conj().T) / 2
    
    vals, vecs = eigh(R, subset_by_index=[0, 0])
    return vals[0], vecs[:, 0], grad_vec

# ========================================================
# 2. 优化器主类
# ========================================================

class AdvancedOptimizer:
    def __init__(self, POVM):
        self.POVM = POVM
        self.n = len(POVM)
        
        # 自动检测核心数：优先跑满物理核
        detected = os.cpu_count()
        if detected > 60: 
            self.n_workers = detected // 2 # 针对 52核/104线程 机器
        else:
            self.n_workers = detected
        
        # 确保至少有 4 个 worker
        if self.n_workers < 4: self.n_workers = 4
            
        print(f"\n{'='*60}")
        print(f"🚀 Hybrid Parallel Optimizer")
        print(f"   Workers: {self.n_workers} (Mixed Local/Global Search)")
        print(f"{'='*60}\n")

    def entropy(self, p):
        p = np.clip(p, 1e-16, 1.0)
        return -np.sum(p * np.log(p)) 

    def optimize(self, max_outer_iter=30, base_dca_loops=500, tol=1e-5):
        
        A_ub = np.vstack([-np.eye(self.n), np.eye(self.n)])
        b_ub = np.concatenate([np.zeros(self.n), np.ones(self.n)])
        A_eq = np.ones((1, self.n))
        b_eq = np.array([1.0])
        
        history_lb = []
        history_ub = []
        
        # 记录全局最好的点，用于下一轮 Warm Start
        global_best_p = None 
        c_lower = -float('inf')
        c_upper = float('inf')

        start_time = time.time()
        print(f"{'Iter':<5} | {'DCA(s)':<7} | {'Gap':<9} | {'Cuts':<5} | {'LB':<9} | {'UB':<9} | {'Constraints'}")
        print("-" * 85)

        with ProcessPoolExecutor(max_workers=self.n_workers, 
                                 initializer=init_worker_process, 
                                 initargs=(self.POVM,)) as executor:
            
            for k in range(1, max_outer_iter + 1):
                t1 = time.time()
                
                # --- 1. 自适应负载 & 混合策略 ---
                # 随着迭代增加，搜索次数加倍，防止后期切不动
                current_loops = int(base_dca_loops * (1 + k * 0.1))
                
                tasks = []
                for i in range(self.n_workers):
                    # 50% 的工人做局部深挖 (Local)，50% 做全局探索 (Global)
                    mode = 'local' if (i < self.n_workers // 2) and (global_best_p is not None) else 'global'
                    tasks.append((current_loops, self.n, A_ub, b_ub, A_eq, b_eq, global_best_p, mode))
                
                # --- 2. 并行 DCA ---
                candidates = list(executor.map(task_hybrid_dca, tasks))
                candidates = [c for c in candidates if c is not None]
                
                if not candidates:
                    print("⚠️ No valid candidates found. Terminating.")
                    break

                # 更新 LB
                current_vals = [-np.sum(c * np.log(np.clip(c,1e-12,1))) for c in candidates]
                iter_min = min(current_vals)
                best_idx = np.argmin(current_vals)
                
                # 更新全局最佳点 (给下一轮做种子)
                global_best_p = candidates[best_idx].copy()
                c_lower = iter_min
                
                t2 = time.time()
                
                # --- 3. 并行 Oracle & 切割 ---
                oracle_results = list(executor.map(task_oracle_check, candidates))
                
                cuts_added = 0
                added_grads = [] # 去重缓冲区
                
                for i, res in enumerate(oracle_results):
                    if res is None: continue
                    min_eig, psi_opt, grad_vec = res
                    
                    # 更新 UB
                    p_real = np.array([np.real(psi_opt.conj().T @ E @ psi_opt) for E in self.POVM])
                    p_real /= np.sum(p_real)
                    e_real = self.entropy(p_real)
                    if e_real < c_upper: c_upper = e_real
                    
                    # 检查是否需要添加切割
                    lhs = np.dot(grad_vec, candidates[i])
                    violation = min_eig - lhs
                    
                    if violation > tol:
                        # 简单的余弦相似度去重，避免添加平行约束
                        is_duplicate = False
                        g_norm = grad_vec / (np.linalg.norm(grad_vec) + 1e-20)
                        for existing in added_grads:
                            if np.dot(existing, g_norm) > 0.995: 
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            A_ub = np.vstack([A_ub, -grad_vec])
                            b_ub = np.append(b_ub, -min_eig)
                            added_grads.append(g_norm)
                            cuts_added += 1
                
                gap = c_upper - c_lower
                history_lb.append(c_lower)
                history_ub.append(c_upper)
                
                dca_time = t2 - t1
                print(f"{k:<5d} | {dca_time:6.3f}s | {gap:.2e} | +{cuts_added:<4d} | {c_lower:.5f}  | {c_upper:.5f}  | {len(b_ub)}")
                
                if gap < tol:
                    print("\n🏆 Converged within tolerance!")
                    break
        
        total_time = time.time() - start_time
        print("-" * 85)
        print(f"✅ Total Runtime: {total_time:.4f}s")
        return history_lb, history_ub

# ========================================================
# 3. 运行与绘图
# ========================================================

def plot_convergence(lb, ub):
    plt.figure(figsize=(10, 6))
    iters = range(1, len(lb) + 1)
    plt.plot(iters, ub, 'r-o', label='Upper Bound (Quantum)', linewidth=2)
    plt.plot(iters, lb, 'b-s', label='Lower Bound (Polytope)', linewidth=2)
    plt.fill_between(iters, lb, ub, color='gray', alpha=0.1)
    
    plt.xlabel('Iteration')
    plt.ylabel('Entropy')
    plt.title('Convergence of Outer Approximation')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 为了测试效果，生成一个稍大的问题
    np.random.seed(42)
    dim = 20           
    n_outcomes = 60   
    
    print(f"Generating Problem (d={dim}, n={n_outcomes})...")
    M_list = []
    for _ in range(n_outcomes):
        A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        M_list.append(A @ A.conj().T)
    S = sum(M_list)
    v, u = np.linalg.eigh(S)
    S_inv_sqrt = u @ np.diag(1/np.sqrt(v)) @ u.conj().T
    POVM_test = [S_inv_sqrt @ M @ S_inv_sqrt for M in M_list]

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
    POVM = []
    for v in basis1 + basis2:
        # 创建投影算子 |v><v|
        projector = v @ v.T
        # 乘以1/2得到POVM元素
        povm_element = projector / 2
        POVM.append(povm_element)
    
    # 初始化优化器
    opt = AdvancedOptimizer(POVM)
    
    # 运行优化
    # base_dca_loops=1000: 初始每核跑1000次，后续会自动增加
    lb_hist, ub_hist = opt.optimize(max_outer_iter=40, base_dca_loops=10, tol=1e-4)
    
    # 绘图
    plot_convergence(lb_hist, ub_hist)

# %%
