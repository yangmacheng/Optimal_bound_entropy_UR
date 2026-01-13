# %%
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import time
import os

# ========================================================
# 1. 全局初始化与 Worker
# ========================================================

_global_povm = None

def init_worker_process(povm_data):
    global _global_povm
    _global_povm = povm_data
    # 确保每个核心的随机数不同
    np.random.seed(os.getpid() + int(time.time()*1000) % 123456789)

def task_batch_dca(args):
    """
    极速模式：本地闭环计算，无通信等待
    """
    num_loops, n, A_ub, b_ub, A_eq, b_eq, max_inner = args
    local_best_p = None
    local_best_val = float('inf')
    bounds = (0, 1)

    for _ in range(num_loops):
        x_current = np.random.rand(n)
        x_current /= np.sum(x_current)
        
        for _ in range(max_inner):
            c_vec = -(1.0 + np.log(np.clip(x_current, 1e-12, 1.0)))
            res = linprog(c_vec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                          bounds=bounds, method='highs') # highs solver is fastest
            if not res.success: break
            x_next = res.x
            if np.linalg.norm(x_next - x_current) < 1e-6:
                x_current = x_next
                break
            x_current = x_next
            
        val = -np.sum(x_current * np.log(np.clip(x_current, 1e-12, 1.0)))
        if val < local_best_val:
            local_best_val = val
            local_best_p = x_current.copy()
            
    return local_best_p, local_best_val

def task_separation_oracle(p_candidate):
    global _global_povm
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
# 2. 带有计时统计的主优化器
# ========================================================

class LinuxParallelOptimizer:
    def __init__(self, POVM):
        self.POVM = POVM
        self.n = len(POVM)
        self.n_workers = os.cpu_count() # 自动识别 WSL2 分配的核心数
        print(f"\n{'='*60}")
        print(f"🚀 HPC Environment Detected: {self.n_workers} Cores Active")
        print(f"{'='*60}\n")

    def entropy(self, p):
        p = np.clip(p, 1e-16, 1.0)
        return -np.sum(p * np.log(p)) 

    def optimize(self, max_outer_iter=20, total_dca_starts=52000, tol=1e-4):
        # 初始线性约束
        A_ub = np.vstack([-np.eye(self.n), np.eye(self.n)])
        b_ub = np.concatenate([np.zeros(self.n), np.ones(self.n)])
        A_eq = np.ones((1, self.n))
        b_eq = np.array([1.0])
        
        c_lower = -float('inf')
        c_upper = float('inf')
        history_lb = []
        history_ub = []

        # --- 总体计时开始 ---
        program_start_time = time.time()

        print(f"{'Iter':<5} | {'DCA Time':<10} | {'Oracle Time':<11} | {'Total Time':<10} | {'Gap':<8} | {'Status'}")
        print("-" * 75)

        with ProcessPoolExecutor(max_workers=self.n_workers, 
                                 initializer=init_worker_process, 
                                 initargs=(self.POVM,)) as executor:
            
            for k in range(1, max_outer_iter + 1):
                iter_start_time = time.time()
                
                # --- Phase 1: 并行 DCA (CPU 密集型) ---
                t1 = time.time()
                
                base_load = total_dca_starts // self.n_workers
                remainder = total_dca_starts % self.n_workers
                batch_tasks = []
                for i in range(self.n_workers):
                    count = base_load + (1 if i < remainder else 0)
                    if count > 0:
                        batch_tasks.append((count, self.n, A_ub, b_ub, A_eq, b_eq, 10))
                
                # 这里会瞬间占满 52 个核心
                dca_results = list(executor.map(task_batch_dca, batch_tasks, chunksize=1))
                
                candidates = []
                min_poly_entropy = float('inf')
                for p_final, obj_val in dca_results:
                    if p_final is not None:
                        candidates.append(p_final)
                        if obj_val < min_poly_entropy:
                            min_poly_entropy = obj_val
                
                c_lower = min_poly_entropy
                dca_duration = time.time() - t1

                # --- Phase 2: Oracle 校验 ---
                t2 = time.time()
                oracle_results = list(executor.map(task_separation_oracle, candidates))
                
                new_cuts = 0
                for i, (min_eig, psi_opt, grad_vec) in enumerate(oracle_results):
                    # 更新上界 UB
                    p_real = np.array([np.real(psi_opt.conj().T @ E @ psi_opt) for E in self.POVM])
                    p_real /= np.sum(p_real)
                    e_real = self.entropy(p_real)
                    if e_real < c_upper: c_upper = e_real
                    
                    # 检查是否需要切割
                    lhs = np.dot(grad_vec, candidates[i])
                    if min_eig - lhs > tol:
                        A_ub = np.vstack([A_ub, -grad_vec])
                        b_ub = np.append(b_ub, -min_eig)
                        new_cuts += 1
                
                oracle_duration = time.time() - t2
                
                # --- 统计汇总 ---
                iter_total_time = time.time() - iter_start_time
                gap = c_upper - c_lower
                history_lb.append(c_lower)
                history_ub.append(c_upper)

                # 打印格式化统计
                status = f"Cut +{new_cuts}" if gap > tol else "CONVERGED"
                print(f"{k:<5d} | {dca_duration:6.3f}s   | {oracle_duration:7.3f}s    | {iter_total_time:6.3f}s   | {gap:.1e} | {status}")
                
                if gap < tol:
                    break
        
        # --- 最终统计 ---
        total_program_time = time.time() - program_start_time
        print("-" * 75)
        print(f"✅ Optimization Finished.")
        print(f"⏱️  Total Runtime: {total_program_time:.4f} seconds")
        print(f"📊 Avg Time/Iter: {total_program_time/k:.4f} seconds")
        print(f"💥 Total DCA Tasks Completed: {k * total_dca_starts:,}")
        
        return history_lb, history_ub

# ========================================================
# 3. 运行部分
# ========================================================

if __name__ == "__main__":
    # 为了测试 52 核的威力，我们稍微增加一点计算难度
    np.random.seed(42)
    dim = 25          # 维度
    n_outcomes = 50   # POVM 元素数量
    
    print(f"Generating Problem (Dimension={dim}, Outcomes={n_outcomes})...")
    M_list = []
    for _ in range(n_outcomes):
        A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        M_list.append(A @ A.conj().T)
    S = sum(M_list)
    v, u = np.linalg.eigh(S)
    S_inv_sqrt = u @ np.diag(1/np.sqrt(v)) @ u.conj().T
    POVM_test = [S_inv_sqrt @ M @ S_inv_sqrt for M in M_list]
    
    opt = LinuxParallelOptimizer(POVM_test)
    
    # 52核心 * 1000 = 52000 次DCA/轮，这是巨大的计算量
    opt.optimize(max_outer_iter=30, total_dca_starts=500, tol=1e-5)
    
    # 结果可视化 (WSLg 支持直接弹出)
    plt.show() # 如果之前已经画了图

# %%

# %%
