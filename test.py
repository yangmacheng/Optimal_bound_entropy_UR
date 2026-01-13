import matplotlib
matplotlib.use('TkAgg')  # 指定使用 Tkinter 弹出窗口
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linprog
import matplotlib.pyplot as plt
# 修改这里：导入 ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor 
import time
import os

# ================= 全局辅助函数 =================
# 在多线程模式下，这些函数即使定义在 Notebook 里也能正常工作
def solve_lp_task(args):
    c_vec, A_ub, b_ub, A_eq, b_eq = args
    # 依然使用 'highs'，它底层是 C++，不仅快还能在多线程下释放 GIL
    res = linprog(c_vec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                  bounds=(0, 1), method='highs')
    if res.success:
        return res.x
    return None

def compute_ground_state_task(args):
    r_vec, POVM = args
    n = len(r_vec)
    d = POVM[0].shape[0]
    
    R = np.zeros((d, d), dtype=complex)
    for i in range(n):
        R += r_vec[i] * POVM[i]
    R = (R + R.conj().T) / 2
    
    vals, vecs = eigh(R, subset_by_index=[0, 0]) 
    return vals[0], vecs[:, 0]

# ================= 主优化类 =================

class POVMEntropyOptimizer:
    def __init__(self, POVM, n_workers=None):
        self.POVM = POVM
        self.n = len(POVM)
        self.d = POVM[0].shape[0]
        # 线程数可以设得比核心数多一点，因为有些开销是 IO 或调度的
        self.n_workers = n_workers if n_workers else min(32, (os.cpu_count() or 1) * 2)
        
        if not np.allclose(sum(POVM), np.eye(self.d), atol=1e-6):
            raise ValueError("POVM does not sum to identity")

    def entropy(self, p):
        p = np.clip(p, 1e-16, 1.0)
        return -np.sum(p * np.log2(p))

    def entropy_gradient(self, p):
        p = np.clip(p, 1e-16, 1.0)
        return -(np.log2(p) + 1/np.log(2))

    def optimize(self, max_iter=100, tol=1e-5):
        print(f"Starting Multi-Cut Optimization (n={self.n}, d={self.d}) using {self.n_workers} threads...")
        
        # --- 初始化 ---
        A_ub = np.vstack([-np.eye(self.n), np.eye(self.n)])
        b_ub = np.concatenate([np.zeros(self.n), np.ones(self.n)])
        A_eq = np.ones((1, self.n))
        b_eq = np.array([1.0])
        
        c_minus = 0.0
        c_plus = float('inf')
        best_psi = None
        
        # 初始随机搜索
        seeds = [np.eye(self.d)[i] for i in range(self.d)]
        for _ in range(20):
            psi = np.random.randn(self.d) + 1j * np.random.randn(self.d)
            psi /= np.linalg.norm(psi)
            seeds.append(psi)     
        for psi in seeds:
            p = np.array([np.real(psi.conj().T @ E @ psi) for E in self.POVM])
            e_val = self.entropy(p)
            if e_val < c_plus:
                c_plus = e_val
                best_psi = psi
                
        history_lb = []
        history_ub = []
        p_poly_best = np.ones(self.n) / self.n
        
        # 修改这里：使用 ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            
            for iter_idx in range(1, max_iter + 1):
                t_start = time.time()
                
                # Step 1: 构造搜索方向
                search_vectors = []
                grad = self.entropy_gradient(p_poly_best)
                search_vectors.append(grad)
                for _ in range(5): # 随机扰动
                    noise = np.random.randn(self.n)
                    search_vectors.append(grad + 0.5 * noise)
                for i in range(self.n): # Corner search
                    c = np.zeros(self.n)
                    c[i] = -1.0 
                    search_vectors.append(c)
                
                # Step 2: 并行求解 LP
                lp_tasks = [(vec, A_ub, b_ub, A_eq, b_eq) for vec in search_vectors]
                results = list(executor.map(solve_lp_task, lp_tasks))
                
                # 过滤结果
                candidates = []
                seen_hashes = set()
                min_ent_this_iter = float('inf')
                best_p_this_iter = None
                
                for p_res in results:
                    if p_res is not None:
                        p_hash = tuple(np.round(p_res, 4))
                        if p_hash not in seen_hashes:
                            seen_hashes.add(p_hash)
                            candidates.append(p_res)
                            e = self.entropy(p_res)
                            if e < min_ent_this_iter:
                                min_ent_this_iter = e
                                best_p_this_iter = p_res
                
                if best_p_this_iter is not None:
                    p_poly_best = best_p_this_iter

                if min_ent_this_iter > c_plus:
                    lb_display = c_minus
                    p_current_quantum = np.array([np.real(best_psi.conj().T @ E @ best_psi) for E in self.POVM])
                    candidates.append(p_current_quantum)
                else:
                    c_minus = max(c_minus, min_ent_this_iter)
                    lb_display = min(c_minus, c_plus)

                # Step 3: 并行生成切割
                cut_tasks = []
                valid_candidates = []
                for p_cand in candidates:
                    g = self.entropy_gradient(p_cand)
                    norm = np.linalg.norm(g)
                    if norm > 1e-9:
                        r = g / norm
                        cut_tasks.append((r, self.POVM))
                        valid_candidates.append(p_cand)
                
                ground_results = list(executor.map(compute_ground_state_task, cut_tasks))
                
                # Step 4: 更新
                new_cuts_count = 0
                for i, (min_eig, psi_opt) in enumerate(ground_results):
                    r_vec = cut_tasks[i][0]
                    p_target = valid_candidates[i]
                    
                    p_real = np.array([np.real(psi_opt.conj().T @ E @ psi_opt) for E in self.POVM])
                    e_real = self.entropy(p_real)
                    if e_real < c_plus:
                        c_plus = e_real
                        best_psi = psi_opt
                    
                    violation = min_eig - np.dot(r_vec, p_target)
                    if violation > tol:
                        A_ub = np.vstack([A_ub, -r_vec])
                        b_ub = np.append(b_ub, -min_eig)
                        new_cuts_count += 1
                
                history_lb.append(lb_display)
                history_ub.append(c_plus)
                
                gap = c_plus - lb_display
                print(f"Iter {iter_idx:3d} | LB: {lb_display:.5f} | UB: {c_plus:.5f} | Gap: {gap:.5f} | Cuts: +{new_cuts_count}")
                
                if gap < tol:
                    print("Converged!")
                    break
                    
        return history_lb, history_ub

if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)
    n_outcomes = 50
    dim = 5
    
    print("Generating random POVM...")
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
    
    optimizer = POVMEntropyOptimizer(POVM_test, n_workers=16) 
    lb, ub = optimizer.optimize(max_iter=500, tol=1e-4)
    
    plt.figure(figsize=(10,6))
    plt.plot(lb, label='Lower Bound')
    plt.plot(ub, label='Upper Bound')
    plt.legend()
    plt.grid()
    plt.show()


