"""
Support-Function Based Outer Approximation for the Minimal Entropy of a POVM
=======================================================
此代码用于计算量子理论所允许的任意 POVM 测量的最小 Tsallis 熵。

作者: [yang]
日期: 2026-1-25
"""
import numpy as np
from scipy.linalg import eigh, svd
from pypoman import compute_polytope_vertices
import matplotlib.pyplot as plt
import time

# ======================= 1. 基础工具函数 =======================

def _normalize_povm(POVM, atol=1e-8, verbose=True):
    """
    检查并归一化 POVM。如果 sum(E) != I，则通过白化处理强制归一化。
    """
    d = POVM[0].shape[0]
    S = sum(POVM)
    if np.allclose(S, np.eye(d), atol=atol):
        return POVM

    if verbose:
        print("警告: POVM 不归一，进行归一化使 sum(E)=I")
    
    # Cholesky 分解用于白化：E' = L^{-H} E L^{-1}
    L = np.linalg.cholesky(S)
    Linv = np.linalg.inv(L)
    POVM2 = [Linv.conj().T @ E @ Linv for E in POVM]
    return POVM2

def _vec_real(H):
    """将复 Hermitian 矩阵展平为实向量（用于构建仿射基）。"""
    Hr = np.real(H).ravel()
    Hi = np.imag(H).ravel()
    return np.hstack([Hr, Hi])

def _affine_basis_from_povm(POVM, tol=1e-12):
    """
    计算概率空间 P 的仿射子空间表示：p = s + Q * z
    s: 最大混态对应的概率 (中心点)
    Q: 零空间基向量 (用于降维，去除 sum(p)=1 等等式约束)
    """
    d = POVM[0].shape[0]
    n = len(POVM)
    I = np.eye(d)

    trE = np.array([np.trace(E).real for E in POVM])
    s = trE / d  # 中心点 s

    # 构建线性方程组 M x = 0 寻找自由度
    rows = []
    for i, E in enumerate(POVM):
        Ei0 = E - (trE[i] / d) * I
        rows.append(_vec_real(Ei0))
    M = np.vstack(rows) 

    # SVD 获取零空间基
    U, S, Vt = svd(M, full_matrices=False)
    r = int(np.sum(S > tol * (S[0] if S.size > 0 else 1.0)))
    Q = U[:, :r] if r > 0 else np.zeros((n, 0))

    # 修正 Q 以确保 sum(Q_col) = 0 (数值稳定性)
    if r > 0:
        ones = np.ones(n)
        alpha_vec = (ones @ Q) / (ones @ ones)
        Q = Q - np.outer(ones, alpha_vec)
        Q, _ = np.linalg.qr(Q, mode='reduced')

    return s, Q

def _support_h(POVM, u):
    """
    计算支撑函数 h(u) = max_{p in P} <u, p>
    """
    R = sum(u[i] * POVM[i] for i in range(len(POVM)))
    w = np.linalg.eigvalsh(R)
    return float(w[-1])

def _umax_all(POVM):
    """计算每个 POVM 元素的最大特征值（即 p_i 的理论最大值）。"""
    return np.array([np.linalg.eigvalsh(E).max().real for E in POVM])

def _calc_entropy(p, alpha=1.0, clip=1e-16):
    """
    计算 Tsallis 熵。
    当 alpha=1 时，计算香农熵。
    当 alpha!=1 时，计算 H_a(p) = 1/(a-1) * (1 - sum p^a)。
    """
    p = np.clip(p, clip, 1.0)
    if abs(alpha - 1.0) < 1e-6:
        # 原始代码情形：Base 2 Shannon Entropy
        return float(-np.sum(p * np.log(p)))
    else:
        # Tsallis Entropy
        return float((1.0 - np.sum(p ** alpha)) / (alpha - 1.0))

def _calc_entropy_grad(p, alpha=1.0, clip=1e-16):
    """
    计算 Tsallis 熵的梯度。
    """
    p = np.clip(p, clip, 1.0)
    
    if abs(alpha - 1.0) < 1e-6:
        # Shannon Entropy Gradient
        # invln2 = 1.0 / np.log(2.0)
        g = -(np.log(p) + 1)
    else:
        # Tsallis Entropy Gradient
        # dH/dp_i = (alpha / (1 - alpha)) * p_i^(alpha - 1)
        coeff = alpha / (1.0 - alpha)
        g = coeff * (p ** (alpha - 1.0))
        
    # 归一化梯度以保持数值稳定
    nrm = np.linalg.norm(g)
    if nrm > 0:
        g = g / nrm
    return g

def _rand_pure_states(d, m, seed=0):
    """生成随机纯态用于初始探索。"""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(d, m)) + 1j * rng.normal(size=(d, m))
    Z /= np.linalg.norm(Z, axis=0, keepdims=True)
    return [Z[:, k] for k in range(m)]

def _p_from_state(POVM, psi):
    """计算纯态 psi 对应的概率分布 p。"""
    return np.array([np.real(psi.conj().T @ E @ psi) for E in POVM])


# ======================= 2. 初始约束构建 =======================

def _build_initial_constraints_z(
    POVM, s, Q,
    alpha=1.0,  # 新增 alpha 参数
    n_pair_cuts=100,
    n_rand_states=40,
    n_eigstate_cuts=50,
    seed=42,
    verbose=True
):
    """
    在降维后的 z 空间构建初始的多面体约束 A z <= b。
    """
    n = len(POVM)
    rdim = Q.shape[1]
    if rdim == 0:
        return np.zeros((0, 0)), np.zeros(0)

    A = []
    b = []

    # --- 1) 单元素 Box 约束 ---
    umax = _umax_all(POVM)
    for i in range(n):
        Ai = -Q[i, :]
        bi = s[i]
        A.append(Ai)
        b.append(bi)
        Ai = Q[i, :]
        bi = umax[i] - s[i]
        A.append(Ai)
        b.append(bi)

    # --- 2) 对子 (Pairwise) 上界约束 ---
    if n_pair_cuts and n > 1:
        rng = np.random.default_rng(seed)
        M_cand = min(5000, n * (n - 1) // 2)
        cand = []
        seen = set()
        tries = 0
        while len(cand) < M_cand and tries < 10 * M_cand:
            i, j = rng.choice(n, 2, replace=False)
            if i > j: i, j = j, i
            if (i, j) in seen:
                tries += 1
                continue
            seen.add((i, j))
            lam = np.linalg.eigvalsh(POVM[i] + POVM[j])[-1].real
            cand.append(((i, j), lam))
            tries += 1

        cand.sort(key=lambda x: x[1])
        use = cand[:min(n_pair_cuts, len(cand))]
        for (i, j), lam in use:
            aijQ = Q[i, :] + Q[j, :]
            bij = lam - (s[i] + s[j])
            A.append(aijQ)
            b.append(bij)

    # --- 3) 初始切平面 (Gradient Cuts from States) ---
    def add_cuts_from_states(states_list):
        local_cuts = []
        for psi in states_list:
            p_real = _p_from_state(POVM, psi)
            # 这里调用新的梯度函数
            g = _calc_entropy_grad(p_real, alpha=alpha)
            
            hval = _support_h(POVM, -g)
            
            Ai = -(g @ Q)
            bi = hval + float(g @ s)
            
            strength = -hval - np.min(g) 
            local_cuts.append((Ai, bi, strength))
        return local_cuts

    cuts = []
    if n_eigstate_cuts:
        idx_sorted = np.argsort(-umax)
        take = idx_sorted[:min(n_eigstate_cuts, n)]
        psis = []
        for i in take:
            _, V = eigh(POVM[i])
            psis.append(V[:, -1])
        cuts.extend(add_cuts_from_states(psis))

    if n_rand_states:
        psis = _rand_pure_states(POVM[0].shape[0], n_rand_states, seed=seed + 1)
        cuts.extend(add_cuts_from_states(psis))

    cuts.sort(key=lambda t: t[2], reverse=True)
    for Ai, bi, _ in cuts:
        A.append(Ai)
        b.append(bi)

    A = np.array(A) if len(A) > 0 else np.zeros((0, rdim))
    b = np.array(b) if len(b) > 0 else np.zeros(0)
    return A, b


# ======================= 3. 主优化循环 =======================

def tsallis_entropy_min(
    POVM,
    alpha=1.0,  # 新增 alpha 参数
    max_iter=50,
    tol=1e-6,
    seed=42,
    n_pair_cuts=100,
    n_rand_states=40,
    n_eigstate_cuts=50,
    plot=True,
    verbose=True
):
    """
    基于外逼近 (Outer Approximation) 的最小 Tsallis 熵求解器。
    alpha=1.0 时退化为 Shannon 熵。
    """
    t0 = time.time()
    POVM = _normalize_povm(POVM, verbose=verbose)
    n = len(POVM)
    d = POVM[0].shape[0]
    
    s, Q = _affine_basis_from_povm(POVM)
    rdim = Q.shape[1]

    if rdim == 0:
        Hs = _calc_entropy(s, alpha=alpha)
        print("概率空间维数为0，唯一分布为最大混态分布。")
        return Hs, Hs, s, {'lb': [Hs], 'ub': [Hs], 'gap': [0.0], 'time': [0.0]}

    # 传递 alpha 构建初始多面体
    A_z, b_z = _build_initial_constraints_z(
        POVM, s, Q,
        alpha=alpha,
        n_pair_cuts=n_pair_cuts,
        n_rand_states=n_rand_states,
        n_eigstate_cuts=n_eigstate_cuts,
        seed=seed,
        verbose=verbose
    )

    c_minus_list, c_plus_list, gap_list, time_list = [], [], [], []
    c_plus = float('inf') 
    c_minus = -float('inf')
    p_best_poly = s 

    total_start = time.time()
    
    label_str = "Shannon" if abs(alpha-1)<1e-6 else f"Tsallis(a={alpha})"

    if verbose:
        print(f"=== 开始优化 [{label_str}] (N={n}, d={d}, r={rdim}) ===")

    for k in range(1, max_iter + 1):
        iter_start = time.time()

        # --- 步骤 A: 顶点枚举 ---
        try:
            verts_z = compute_polytope_vertices(A_z, b_z)
            verts_z = np.array(verts_z) if len(verts_z) > 0 else np.zeros((0, rdim))
        except Exception as e:
            print(f"顶点枚举错误: {e}")
            break

        if verts_z.shape[0] == 0:
            print("错误: 当前多面体为空集。")
            break

        # --- 步骤 B: 计算下界 (Lower Bound) ---
        vals = []
        for z in verts_z:
            p = s + Q @ z
            vals.append(_calc_entropy(p, alpha=alpha)) # 使用新的熵函数
        vals = np.array(vals)
        
        best_idx = int(np.argmin(vals))
        lb_val = float(vals[best_idx])
        
        z_best = verts_z[best_idx]
        p_best_poly = s + Q @ z_best
        c_minus = lb_val

        # --- 步骤 C: 计算上界 (Upper Bound) ---
        g = _calc_entropy_grad(p_best_poly, alpha=alpha) # 使用新的梯度函数

        R_op = sum(g[i] * POVM[i] for i in range(n))
        evals, evecs = eigh(R_op)
        psi_min = evecs[:, 0]
        
        p_real = _p_from_state(POVM, psi_min)
        ub_val = _calc_entropy(p_real, alpha=alpha) # 使用新的熵函数
        
        if ub_val < c_plus:
            c_plus = ub_val

        # --- 步骤 D: 收敛检查 ---
        gap = c_plus - c_minus
        it_time = time.time() - iter_start
        c_minus_list.append(c_minus)
        c_plus_list.append(c_plus)
        gap_list.append(gap)
        time_list.append(it_time)

        if verbose:
            print(f"Iter {k:<3} | LB: {c_minus:.6f} | UB: {c_plus:.6f} "
                  f"| Gap: {gap:.3e} | Verts: {verts_z.shape[0]:<5} "
                  f"| Time: {it_time:.3f}s")

        if gap < tol:
            if verbose: print("已达到收敛精度。")
            break

        # --- 步骤 E: 添加切平面 ---
        hval = _support_h(POVM, -g)
        
        Ai_new = -(g @ Q)
        bi_new = hval + float(g @ s)
        
        A_z = np.vstack([A_z, Ai_new.reshape(1, -1)])
        b_z = np.hstack([b_z, bi_new])

    total_time = time.time() - total_start
    if verbose:
        print(f"=== 优化结束, 总耗时: {total_time:.3f} s ===")
        print(f"最终结果区间: [{c_minus:.6f}, {c_plus:.6f}]")

    # --- 可视化 ---
    if plot:
        its = np.arange(1, len(c_minus_list) + 1)
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 1, 1)
        plt.plot(its, c_plus_list, 'r-o', label='Upper Bound')
        plt.plot(its, c_minus_list, 'b-s', label='Lower Bound')
        plt.fill_between(its, c_minus_list, c_plus_list, color='gray', alpha=0.1)
        plt.ylabel(f'{label_str} Entropy')
        plt.title(f'Convergence (N={n}, d={d})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.subplot(2, 1, 2)
        plt.semilogy(its, gap_list, '^-', label='Optimality Gap')
        plt.xlabel('Iteration')
        plt.ylabel('Gap (log scale)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    return c_minus, c_plus, p_best_poly, {
        'lb': c_minus_list, 'ub': c_plus_list,
        'gap': gap_list, 'time': time_list
    }
