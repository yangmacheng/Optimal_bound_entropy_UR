"""
Entropic Uncertainty Relations (EUR) Bounds Visualization
=======================================================
此脚本用于计算并对比三个测量不同熵不确定性关系的下界（Bounds）。
主要包含以下界限的计算：
1. 两个测量 Maassen-Uffink (MU) Bound 简单组合
2. Liu-Mu-Fan (LMF) 界
3. RPZ (Rudnicki-Puchala-Zyczkowski) majorization 界
4. Optimal Bound 

作者: [yang]
日期: 2026-1-25
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import entropy
import entropy_min_outerapproximate 

# ==============================================================================
# 第一部分：辅助工具函数 (Helper Functions)
# ==============================================================================

def observables_to_projector_list(observables, tol=1e-10):
    """
    将一组可观测量（厄米矩阵）转换为投影算符列表。
    
    原理：
    对每个可观测量进行谱分解，提取本征向量。对于简并本征值，构建对应的子空间投影；
    对于非简并本征值，构建秩为1的投影算符。

    参数:
        observables (list of np.ndarray): 包含 n 个 numpy 数组的 list，每个代表一个可观测量。
        tol (float): 判断本征值简并的数值容差。
        
    返回:
        list: 一个扁平的 list，包含所有计算出的投影算符矩阵 P = |v><v|。
    """
    projector_list = []
    
    for i, obs in enumerate(observables):
        # 1. 确保矩阵是厄米矩阵
        if not np.allclose(obs, obs.conj().T, atol=tol):
            raise ValueError(f"第 {i} 个输入矩阵不是厄米矩阵。")
            
        d = obs.shape[0]
        
        # 2. 谱分解: 获取本征值和本征向量 (eigh 保证实数本征值且排序)
        evals, evecs = np.linalg.eigh(obs)
        
        # 3. 遍历本征值，处理简并，构建投影算符
        idx = 0
        while idx < d:
            # 寻找当前本征值的结束索引（处理简并情况）
            end_idx = idx + 1
            while end_idx < d and abs(evals[end_idx] - evals[idx]) < tol:
                end_idx += 1
            
            # 提取对应的一个或多个本征向量 (列向量)
            vectors = evecs[:, idx:end_idx]
            
            # 构建投影算符 P = V * V_dagger
            P = vectors @ vectors.conj().T
            projector_list.append(P)
            
            # 跳过已处理的索引
            idx = end_idx
            
    return projector_list


def get_basis_3(a, phi):
    """
    根据参数 a 和 phi 生成第三组基矩阵 U3 (Example 1)。
    
    参数:
        a (float): 混合参数 [0, 1]
        phi (float): 相位参数 (弧度)
    
    返回:
        np.ndarray: 3x3 的幺正矩阵 (基向量按行排列或按列排列需保持一致，此处上下文暗示为按行)
    """
    # 确保 sqrt 内非负
    a = np.clip(a, 0, 1)
    
    sqrt_a = np.sqrt(a)
    sqrt_1_a = np.sqrt(1 - a)
    
    # 构造矩阵
    return np.array([
        [sqrt_a, np.exp(1j * phi) * sqrt_1_a, 0],
        [sqrt_1_a, -np.exp(1j * phi) * sqrt_a, 0],
        [0,        0,        1]
    ])


def projections_exp1(a=0.5, phi=np.pi/2):
    """
    生成 Example 1 中三组基对应的所有投影算符列表。
    用于自定义的优化算法 (entropy_min_outerapproximate)。
    
    参数:
        a (float): 实数参数, 0 <= a <= 1
        phi (float): 相位参数 (弧度)
    
    返回:
        list: 包含 9 个 3x3 numpy 矩阵的列表
    """
    # --- 内部函数：计算投影算符 |v><v| ---
    def proj(v):
        v = np.array(v)
        return np.outer(v, np.conj(v))
    
    # Group 1 (标准基)
    v1_1 = [1, 0, 0]
    v1_2 = [0, 1, 0]
    v1_3 = [0, 0, 1]
    
    # Group 2 (实数旋转基)
    inv_sq2 = 1.0 / np.sqrt(2)
    v2_1 = [inv_sq2, 0, -inv_sq2]
    v2_2 = [0, 1, 0]
    v2_3 = [inv_sq2, 0, inv_sq2]
    
    # Group 3 (含参复数基)
    v3_1 = [np.sqrt(a), np.exp(1j * phi) * np.sqrt(1 - a), 0]
    v3_2 = [np.sqrt(1 - a), -np.exp(1j * phi) * np.sqrt(a), 0]
    v3_3 = [0, 0, 1]
    
    all_vectors = [v1_1, v1_2, v1_3, v2_1, v2_2, v2_3, v3_1, v3_2, v3_3]
    return [proj(v) for v in all_vectors]


# ==============================================================================
# 第二部分：熵不确定性关系下界计算 (Bound Calculation)
# ==============================================================================

def LMF_bound(basis1, basis2, basis3):
    """
    计算 Liu-Mu-Fan (LMF) 界。
    参考文献: PRA 91, 042133 (2015)
    公式: -ln( max_k { sum_j max_i [ c(u^1_i, u^2_j) ] * c(u^2_j, u^3_k) } )
    """
    U1 = np.array(basis1)
    U2 = np.array(basis2)
    U3 = np.array(basis3)

    # 1. 计算 U1 和 U2 之间的重叠 c(u^1_i, u^2_j) = |<u1|u2>|^2
    # C12[i, j] 代表第 i 个 u1 和第 j 个 u2 的重叠
    inner_prod_12 = U1 @ U2.conj().T
    C12 = np.abs(inner_prod_12)**2

    # 2. 对每列取最大值: M_j = max_i c(u^1_i, u^2_j)
    M_j = np.max(C12, axis=0)

    # 3. 计算 U2 和 U3 之间的重叠
    inner_prod_23 = U2 @ U3.conj().T
    C23 = np.abs(inner_prod_23)**2

    # 4. 加权求和: S_k = sum_j (M_j * C23[j, k])
    S_k = M_j @ C23

    # 5. 取最大值并计算负对数
    result = -np.log(np.max(S_k))
    return result


def SCB_bound(basis1, basis2, basis3):
    """
    计算 simply constructed bound (SCB) 界。
    公式: -0.5 * log2( max(C12) * max(C23) * max(C13) )
    其中 max(C_xy) 是两个基之间所有重叠模方的最大值。
    """
    U1 = np.array(basis1)
    U2 = np.array(basis2)
    U3 = np.array(basis3)

    def get_max_overlap(Ba, Bb):
        inner_prod = Ba @ Bb.conj().T
        c_matrix = np.abs(inner_prod)**2
        return np.max(c_matrix)

    # 计算两两之间的最大重叠
    c_max_12 = get_max_overlap(U1, U2)
    c_max_23 = get_max_overlap(U2, U3)
    c_max_13 = get_max_overlap(U1, U3)
    
    product_term = c_max_12 * c_max_23 * c_max_13
    
    if product_term <= 1e-15:
        return np.inf
    
    result = -0.5 * np.log(product_term)
    return result


def RPZ_maj_bound(U_list, base=np.e):
    """
    计算 RPZ (Rudnicki-Puchala-Zyczkowski) majorization 界。
    利用矩阵拼接和奇异值分解构建优越化向量 s，然后计算其熵。
    """
    # 1. 横向拼接矩阵 [U1, U2, ...]
    M = np.hstack(U_list) 
    d, N = M.shape # N = d * L
    
    # 2. 计算子矩阵最大奇异值平方 S_k
    S = np.zeros(N+1)
    S[0] = 1.0

    cols = range(N)
    for k in range(1, N):
        m = k + 1
        max_sigma2 = 0.0
        # 遍历所有列的组合
        for comb in combinations(cols, m):
            sub = M[:, comb]
            # singular values only
            sigma1 = np.linalg.svd(sub, compute_uv=False)[0]
            sigma2 = sigma1**2
            if sigma2 > max_sigma2:
                max_sigma2 = sigma2
        S[k] = max_sigma2

    S[N] = np.linalg.svd(M, compute_uv=False)[0]**2

    # 3. 构造 majorization 向量 s
    s = np.zeros(N+1)
    s[0] = 1.0
    s[1:] = S[1:] - S[:-1]

    # 4. 计算 Shannon 熵
    mask = s > 0
    if base == 2:
        H = -np.sum(s[mask] * np.log2(s[mask]))
    else:
        H = -np.sum(s[mask] * np.log(s[mask]) / np.log(base))

    return H


# ==============================================================================
# 第三部分：绘图样式设置 (Plotting Style)
# ==============================================================================

def set_pub_style():
    """设置符合期刊发表要求的 matplotlib 绘图样式"""
    # 尝试启用 LaTeX，如果系统未安装 LaTeX 可自动回退
    try:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        })
    except:
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
        })

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 13,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linestyle": ":",
        "legend.frameon": True,
    })


# ==============================================================================
# 第四部分：主程序执行块 (Main Execution)
# ==============================================================================

if __name__ == "__main__":
    
    # ---------------------------------------------------------
    # 1. 定义物理量和基 (Definitions)
    # ---------------------------------------------------------
    
    # Spin-3/2 算符 (示例 2)
    Sx = 0.5 * np.array([
        [0, np.sqrt(3), 0, 0],
        [np.sqrt(3), 0, 2, 0],
        [0, 2, 0, np.sqrt(3)],
        [0, 0, np.sqrt(3), 0]
    ], dtype=complex)

    Sz = 0.5 * np.array([
        [3, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -3]
    ], dtype=complex)
    
    # 测试辅助函数: 生成 Spin-3/2 投影算符
    # (注：这些算符在下面的绘图循环中并未直接使用，仅作演示或预留)
    my_observables = [Sx, Sz]
    all_projectors = observables_to_projector_list(my_observables)
    
    # Example 1 的固定基
    # Basis 1: 标准基
    basis_1 = np.eye(3)
    
    # Basis 2: 实数旋转基
    inv_sqrt2 = 1 / np.sqrt(2)
    basis_2 = [
        [inv_sqrt2, 0, -inv_sqrt2],
        [0, 1, 0],
        [inv_sqrt2, 0, inv_sqrt2]
    ]

    # ---------------------------------------------------------
    # 2. 循环计算 (Simulation Loop)
    # ---------------------------------------------------------
    print("开始计算...")
    
    a_values = np.linspace(0, 1, 500)
    
    LMF_bound_results = []
    SCB_bound_results = []
    RPZ_bound_results = []
    our_bound_results = []

    for a in a_values:
        # 生成参数化的第三组基 (注意：这里使用 phi = pi/2)
        basis_3 = get_basis_3(a, phi=np.pi/2)
        
        # 1. 计算 SCB Bound
        # 注意: 原代码逻辑中输入为 (basis_2, basis_2, basis_3)
        b1 = SCB_bound(basis_2, basis_2, basis_3)
        
        # 2. 计算 LMF Bound
        # 注意: 原代码逻辑中输入顺序为 (basis_2, basis_3, basis_1)
        b2 = LMF_bound(basis_2, basis_3, basis_1)
        
        # 3. 计算 RPZ Bound
        b3 = RPZ_maj_bound([basis_1, basis_2, basis_3])
        
        # 4. 计算 Optimal Bound (自定义算法)
        # 注意: 这里 projections_exp1 使用 phi = pi/4，与上面 basis_3 参数不同，保留原逻辑
        proj_list_opt = projections_exp1(a, phi=np.pi/4)
        c_minus, c_plus, _, _ = entropy_min_outerapproximate.tsallis_entropy_min(
            proj_list_opt, 
            alpha = 1,
            max_iter=60, 
            tol=1e-5,
            plot=False, 
            verbose=False
        )
        b4 = 3 * c_minus - 3 * np.log(3) # 结果转换
        
        SCB_bound_results.append(b1)
        LMF_bound_results.append(b2)
        RPZ_bound_results.append(b3)
        our_bound_results.append(b4)

    print("计算完成，开始绘图。")

    # ---------------------------------------------------------
    # 3. 绘图 (Plotting)
    # ---------------------------------------------------------
    
    set_pub_style() # 应用样式

    fig, ax = plt.subplots(figsize=(7, 5))

    # 配色方案
    color_scb = '#1f77b4'  # Blue
    color_lmf = '#2ca02c'  # Green
    color_rpz = '#ff7f0e'  # Orange
    color_our = '#d62728'  # Red (突出显示)

    # 绘制曲线
    ax.plot(a_values, SCB_bound_results, 
            label=r'$q_{\mathrm{SCB}}$', 
            color=color_scb, linestyle=':', linewidth=2, alpha=0.9)

    ax.plot(a_values, LMF_bound_results, 
            label=r'$q_{\mathrm{LMF}}$', 
            color=color_lmf, linestyle='--', linewidth=2, alpha=0.9)

    ax.plot(a_values, RPZ_bound_results, 
            label=r'$q_{\mathrm{RPZ2}}$', 
            color=color_rpz, linestyle='-.', linewidth=2, alpha=0.9)

    # 优化方法结果（实线，层级最高）
    ax.plot(a_values, our_bound_results, 
            label=r'$q_{\mathrm{Optimal}}$', 
            color=color_our, linestyle='-', linewidth=2, zorder=10)

    # 标签与设置
    ax.set_xlabel(r'Parameter $a$', fontsize=14)
    ax.set_ylabel(r'Bound', fontsize=14)
    ax.set_xlim(0, 1)
    
    # 图例
    ax.legend(loc='upper right', fontsize=12, handlelength=2.5)

    plt.tight_layout()
    
    # 若需保存，取消下面注释
    # plt.savefig('bounds_comparison.pdf', dpi=300, bbox_inches='tight')
    
    plt.show()
