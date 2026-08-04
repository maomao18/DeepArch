import concurrent.futures
import random
import subprocess
import os
import numpy as np
import math
from scipy.stats import qmc
def run_ansys_command(command):
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout, result.stderr, command
    except subprocess.CalledProcessError as e:
        return e.stdout, e.stderr, command
def task_done(future):
    try:
        stdout, stderr, command = future.result()
    except Exception as exc:
        print(f"任务异常: {exc}")
        return
    print("=" * 50)
    print("Task completed.")
    if stdout:
        print("Standard Output snippet:\n", stdout[-500:])
    if stderr:
        print("Standard Error snippet:\n", stderr[-500:])
    # 获取工作目录并清理临时文件
    work_dir = None
    for i, arg in enumerate(command):
        if arg == "-dir":
            work_dir = command[i + 1]
            break
    if work_dir and os.path.isdir(work_dir):
        for filename in os.listdir(work_dir):
            filepath = os.path.join(work_dir, filename)
            # 保留run.txt，删除其他临时文件（.rst, .db, .esav等）
            print(f"Cleaning up directory: {work_dir}")
            if filename != "run.txt" and os.path.isfile(filepath):
                try:
                    os.remove(filepath)
                except OSError:
                    pass
def eta_to_rotational_stiffness(eta, DI, L):
    """
    将 [0,1] 的无量纲刚度系数映射到实际转动刚度

    使用 eta/(1-eta) 映射：
    kth = DI/L * eta/(1-eta)

    特性：
    - eta=0 → kth=0（完全铰接）
    - eta=0.1 → kth≈0.11*DI/L（接近铰接）
    - eta=0.5 → kth=1*DI/L（中等刚度）
    - eta=0.9 → kth≈9*DI/L（接近固结）
    - eta→1 → kth→+∞（完全固结）

    eta接近0时增长慢，接近1时快速增长到无穷大
    """
    k_base = DI / L
    # 使用 eta/(1-eta) 映射，eta=1时发散到无穷大（固结）
    kth = k_base * eta / (1 - eta + 1e-12)  # 加小量避免除零
    return kth


def generate_single_sample(sample_params):
    """
    处理单个LHS样本，生成完整的几何、材料、刚度参数
    返回：(是否有效, 参数字典)

    修改说明：
    - 固定跨径 L = 1
    - 从 lambda_target 和 f_L 反算截面尺寸 h 和 b
    - 使用 b_h（宽高比）作为采样参数
    - 转动刚度系数 etaRot 使用 tanh 映射
    """
    # 解包LHS样本参数（9维）
    (E, rho, b_h, lambda_target, f_L, e0, choice_cont,
     etaRotL, etaRotR) = sample_params

    # 转换离散参数
    choice = 1 if choice_cont < 0.5 else 2  # FGM分布类型选择

    # ========== 固定跨径 L = 1 ==========
    L = 1.0  # 固定跨径

    # ========== 计算几何参数 ==========
    f_arch = L * f_L  # 矢高

    # 计算弧长 S（抛物线拱弧长公式）
    k = 4 * f_L  # 参数
    sqrt_term = math.sqrt(1 + k ** 2)
    log_term = math.log(k + sqrt_term)
    C = 0.5 * sqrt_term + (1 / (2 * k)) * log_term
    S = L * C  # 弧长

    # ========== 计算目标回转半径 ==========
    ix_target = S / lambda_target

    # ========== 计算FGM材料因子（与h无关） ==========
    z_rel = np.array([-0.45, -0.35, -0.25, -0.15, -0.05,
                       0.05,  0.15,  0.25,  0.35,  0.45])  # 相对厚度位置
    type1 = np.cos(np.pi * z_rel)
    type2 = np.cos(np.pi * z_rel / 2 + np.pi / 4)

    if choice == 1:
        factorE_list = 1 - type1 * e0
        factorRho_list = np.power((1 - type1 * e0), 0.4347826087) * 1.121 - 0.121
    else:
        factorE_list = 1 - type2 * e0
        factorRho_list = np.power((1 - type2 * e0), 0.4347826087) * 1.121 - 0.121

    mu = 0.3
    mu_list = [mu] * 10

    # ========== 迭代反算截面尺寸 h 和 b ==========
    # 初始估计（均匀材料近似）
    h = math.sqrt(12) * ix_target
    b = b_h * h

    # 迭代求解，考虑FGM刚度分布和耦合效应
    max_iterations = 20
    tolerance = 0.01  # 长细比误差容限1%

    for iteration in range(max_iterations):
        # 用当前h和b计算刚度
        E_list = factorE_list * E * 1e9
        rho_list = factorRho_list * rho * 1e3

        h_layer = h / 10
        I0 = 0.0
        A11 = 0.0
        B11 = 0.0
        D11 = 0.0

        for j in range(10):
            z_layer = -h / 2 + j * h_layer + h_layer / 2
            I0 += rho_list[j] * b * h_layer
            A11 += E_list[j] * b * h_layer
            B11 += E_list[j] * b * h_layer * z_layer
            D11 += E_list[j] * b * (h_layer * z_layer ** 2 + h_layer ** 3 / 12)

        # 计算等效刚度和实际长细比
        D11_eq = D11 - (B11 ** 2) / A11
        ix_eq = math.sqrt(D11_eq / A11)
        lambda_real = S / ix_eq

        # 检查误差
        error = abs(lambda_real - lambda_target) / lambda_target
        if error < tolerance:
            break  # 满足精度要求，退出迭代

        # 修正h：根据ix比值调整
        h = h * (ix_target / ix_eq)
        b = b_h * h  # 更新b

    # ========== 计算转动弹簧刚度（使用eta/(1-eta)映射） ==========
    kthL = eta_to_rotational_stiffness(etaRotL, D11_eq, L)
    kthR = eta_to_rotational_stiffness(etaRotR, D11_eq, L)

    # ========== 判断是否接近固结（避免数值溢出） ==========
    # 当 eta >= 0.99 或 kth/k_base > 100 时，直接使用固结约束
    FIXED_THRESHOLD = 0.99
    is_fixed_L = etaRotL >= FIXED_THRESHOLD
    is_fixed_R = etaRotR >= FIXED_THRESHOLD

    # ========== 合理性检查 ==========
    valid = True
    if not (abs(lambda_real - lambda_target) / lambda_target < 0.01):  # 长细比误差<1%
        valid = False

    # 打包返回
    params = {
        'E': E, 'rho': rho, 'mu': mu,
        'b': b, 'h': h, 'b_h': b_h,
        'lambda_target': lambda_target, 'lambda_real': lambda_real,
        'f_L': f_L, 'L': L, 'f_arch': f_arch, 'S': S,
        'choice': choice, 'e0': e0,
        'E_list': E_list, 'rho_list': rho_list, 'mu_list': mu_list,
        'I0': I0, 'A11': A11, 'B11': B11, 'D11': D11, 'D11_eq': D11_eq,
        'ix_eq': ix_eq,
        'etaRotL': etaRotL, 'etaRotR': etaRotR,
        'kthL': kthL, 'kthR': kthR,
        'is_fixed_L': is_fixed_L, 'is_fixed_R': is_fixed_R
    }
    return valid, params
def create_commands(work_dir, num_samples=20000):
    MAPDL_PATH = r"C:\Program Files\ANSYS Inc\v211\ansys\bin\winx64\MAPDL.exe"
    number_of_cores = '4'
    commands = []
    # ========== 1. LHS抽样设计（9维：简化边界条件） ==========
    # [E, rho, b_h, lambda_target, f_L, e0, choice_cont, etaRotL, etaRotR]
    # 修改说明：
    # - 固定跨径 L = 1，不再采样
    # - 删除 h 采样，改为反算
    # - b_h 范围 [5, 10]（宽高比）
    # - etaRot 范围 [0, 1]（转动刚度系数，tanh映射）
    # - 删除 KXL, KYL, KXR, KYR（X/Y铰接约束）
    bounds = np.array([
        [60,   210],     # 0  E (GPa)
        [2.7,  8],       # 1  rho (g/cm³)
        [5,    10],      # 2  b_h（宽高比）
        [150,  250],     # 3  lambda_target（长细比）
        [1/12, 1/2],     # 4  f_L（矢跨比）
        [0.1,  0.3],     # 5  e0（FGM梯度系数）
        [0,    1],       # 6  choice_cont（FGM分布类型选择）
        [0,    1],       # 7  etaRotL（左转动刚度系数[0,1]）
        [0,    1],       # 8  etaRotR（右转动刚度系数[0,1]）
    ])
    n_dims = bounds.shape[0]
    # 生成LHS样本（固定seed保证可复现）
    print(f"正在生成 {num_samples} 个LHS样本（{n_dims}维）...")
    sampler = qmc.LatinHypercube(d=n_dims, seed=42)
    sample_unit = sampler.random(n=num_samples)
    samples = qmc.scale(sample_unit, bounds[:, 0], bounds[:, 1])
    print("LHS样本生成完成。")
    # ========== 2. 逐个处理样本，生成APDL命令 ==========
    valid_count = 0
    for i in range(num_samples):
        sample_idx = i + 1
        valid, params = generate_single_sample(samples[i])
        if not valid:
            continue
        valid_count += 1
        if valid_count % 100 == 0 or valid_count <= 10:
            print(f"样本 {sample_idx}/{num_samples} 有效，当前有效样本数：{valid_count}")
        # 生成APDL命令流
        apdl_str = generate_apdl_str(params)
        # 创建工作目录
        new_dir = os.path.join(work_dir, f"FGM_load_disp_{valid_count}")
        os.makedirs(new_dir, exist_ok=True)
        rundata_dir = os.path.join(new_dir, "rundata")
        result_dir = os.path.join(new_dir, "result")
        os.makedirs(rundata_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        # 写入APDL文件
        with open(os.path.join(rundata_dir, "run.txt"), "w") as f:
            f.write(apdl_str)
        with open(os.path.join(result_dir, "run.txt"), "w") as f:
            f.write(apdl_str)
        # 写入输入参数文件（新格式：9个参数）
        # 参数说明：
        # - I0: 截面质量
        # - A11: 压缩刚度
        # - B11: 压-弯耦合刚度
        # - D11_eq: 等效弯曲刚度
        # - f_L: 矢跨比
        # - lambda_real: 实际长细比
        # - b_h: 宽高比
        # - etaRotL: 左转动刚度系数 [0,1]
        # - etaRotR: 右转动刚度系数 [0,1]
        with open(os.path.join(result_dir, "input.txt"), "w") as f:
            f.write(f"{params['I0']}\n")
            f.write(f"{params['A11']}\n")
            f.write(f"{params['B11']}\n")
            f.write(f"{params['D11_eq']}\n")
            f.write(f"{params['f_L']}\n")  # 矢跨比
            f.write(f"{params['lambda_real']}\n")  # 实际长细比
            f.write(f"{params['b_h']}\n")  # 宽高比
            f.write(f"{params['etaRotL']}\n")  # 左转动刚度系数
            f.write(f"{params['etaRotR']}\n")  # 右转动刚度系数

        # 生成命令
        command = [
            MAPDL_PATH, "-p", "ansys", "-smp", "-np", number_of_cores, "-lch",
            "-dir", rundata_dir, "-j", "run", "-s", "read", "-l", "en-us", "-b",
            "-i", os.path.join(rundata_dir, "run.txt"),
            "-o", os.path.join(result_dir, "run.out")
        ]
        commands.append(command)

    print(f"\n样本处理完成！总样本数：{num_samples}，有效样本数：{valid_count}")
    return commands

def generate_apdl_str(params):
    """根据参数字典生成APDL命令流"""
    E_list_str = ','.join([f"{e:.6e}" for e in params['E_list']])
    rho_list_str = ','.join([f"{r:.6e}" for r in params['rho_list']])
    mu_list_str = ','.join([f"{m:.6f}" for m in params['mu_list']])

    # ========== 预处理：根据is_fixed标记生成不同的边界条件代码块 ==========
    is_fixed_L = params['is_fixed_L']
    is_fixed_R = params['is_fixed_R']

    # 左侧边界条件代码块
    if is_fixed_L:
        # 左侧固结：不生成弹簧，后续直接施加ROTZ=0
        left_spring_et_block = "! 左侧固结：不定义转动弹簧单元"
        left_spring_anchor_block = "! 左侧固结：不生成锚固节点"
        left_spring_connect_block = "! 左侧固结：不连接弹簧"
        left_rotz_constraint = """
! 左侧固结约束（直接约束ROTZ）
D,nodeL_1,ROTZ,0
D,nodeL_2,ROTZ,0
D,nodeL_3,ROTZ,0
"""
        left_anchor_constraint_block = "! 左侧固结：无锚固节点需约束"
    else:
        # 左侧弹性转动：生成弹簧单元
        left_spring_et_block = f"""
! 左侧转动弹簧
ET,2,COMBIN14
KEYOPT,2,1,0
KEYOPT,2,2,0
KEYOPT,2,3,1  ! 转动弹簧
R,2,{params['kthL']}/3  ! 转动刚度（均分到3个节点）
"""
        left_spring_anchor_block = """
! 左拱脚转动弹簧锚固节点（Z偏置）
N,nodestart+0,-L_arch/2,0,L0_anchor
N,nodestart+1,-L_arch/2,0,L0_anchor+b_section/2
N,nodestart+2,-L_arch/2,0,L0_anchor+b_section
"""
        left_spring_connect_block = """
! 连接转动弹簧（左侧）
TYPE,2
REAL,2
E,nodeL_1,nodestart+0   ! 对应 (-L/2,0,L0_anchor)
E,nodeL_2,nodestart+1   ! 对应 (-L/2,0,L0_anchor+b/2)
E,nodeL_3,nodestart+2   ! 对应 (-L/2,0,L0_anchor+b)
"""
        left_rotz_constraint = "! 左侧弹性转动：不约束ROTZ（保留转动自由度）"
        left_anchor_constraint_block = """
! 约束左侧锚固节点
D,nodestart+0,ALL,0
D,nodestart+1,ALL,0
D,nodestart+2,ALL,0
"""

    # 右侧边界条件代码块
    if is_fixed_R:
        # 右侧固结：不生成弹簧，后续直接施加ROTZ=0
        right_spring_et_block = "! 右侧固结：不定义转动弹簧单元"
        right_spring_anchor_block = "! 右侧固结：不生成锚固节点"
        right_spring_connect_block = "! 右侧固结：不连接弹簧"
        right_rotz_constraint = """
! 右侧固结约束（直接约束ROTZ）
D,nodeR_1,ROTZ,0
D,nodeR_2,ROTZ,0
D,nodeR_3,ROTZ,0
"""
        right_anchor_constraint_block = "! 右侧固结：无锚固节点需约束"
    else:
        # 右侧弹性转动：生成弹簧单元
        right_spring_et_block = f"""
! 右侧转动弹簧
ET,3,COMBIN14
KEYOPT,3,1,0
KEYOPT,3,2,0
KEYOPT,3,3,1  ! 转动弹簧
R,3,{params['kthR']}/3  ! 转动刚度（均分到3个节点）
"""
        right_spring_anchor_block = """
! 右拱脚转动弹簧锚固节点（Z偏置）
N,nodestart+3,L_arch/2,0,L0_anchor
N,nodestart+4,L_arch/2,0,L0_anchor+b_section/2
N,nodestart+5,L_arch/2,0,L0_anchor+b_section
"""
        right_spring_connect_block = """
! 连接转动弹簧（右侧）
TYPE,3
REAL,3
E,nodeR_1,nodestart+3   ! 对应 (+L/2,0,L0_anchor)
E,nodeR_2,nodestart+4   ! 对应 (+L/2,0,L0_anchor+b/2)
E,nodeR_3,nodestart+5   ! 对应 (+L/2,0,L0_anchor+b)
"""
        right_rotz_constraint = "! 右侧弹性转动：不约束ROTZ（保留转动自由度）"
        right_anchor_constraint_block = """
! 约束右侧锚固节点
D,nodestart+3,ALL,0
D,nodestart+4,ALL,0
D,nodestart+5,ALL,0
"""

    # 计算nodestart基准值（保持编号一致性）
# 无论左侧是否有锚固节点，都定义nodestart = MAX_NODE+1
# 这样右侧锚固节点编号(MAX_NODE+4~6)与连接代码(nodestart+3~5)保持一致

    return rf'''
!##############################################################
! 拱结构屈曲分析程序 (APDL 19.5)
! 功能：包含特征值屈曲分析+弧长法非线性后屈曲分析
! 修改要点（本版v3.0）：
! - 拱脚采用铰接约束（UX=UY=0）+ 转动弹性支撑/固结约束
! - 转动刚度使用 eta/(1-eta) 映射：kth = DI/L * eta/(1-eta)
! - etaRot=0 → kth=0（完全铰接，无转动约束）
! - etaRot=0.5 → kth=DI/L（中等刚度）
! - etaRot→1 → kth→+∞（完全固结）
! - 当eta>=0.99时，直接施加ROTZ=0固结约束，避免数值溢出
! - 固定跨径 L=1，迭代反算截面尺寸 b 和 h（考虑FGM耦合）
! - 为保持平面内屈曲分析，最小化面外约束：仅关键节点施加UZ=0
! 作者：Hao Tang
! 版本：v3.0
! 最后修改：2025-04-24
!##############################################################

!============== 均布压力施加载荷宏（保持原宏） =================
*Create,Arbpres,mac
Save,Arbs1,Db
Esla,S$Nsla,S,1
*Get,Elnum,Elem,,count
*dim,Eleno,,Elnum
*get,e1,elem,,num,min
eleno(1)=e1
*do,i,2,elnum
  e1=elnext(e1)
  eleno(i)=e1
*enddo
Dofsel,S,Fx,Fy,Fz
Fcum,Add
*Do,I,1,Elnum
  *If,Arg1,Eq,1,Then
    Esel,S,,,eleno(i)
    *Get,E_area,Elem,Eleno(i),Aproj,Arg2
  *Else
    *Get,E_area,Elem,Eleno(i),Area
  *Endif
  ArbP=Arg3
  ArbF=ArbP*E_area
  Esel,S,,,eleno(i)
  Nsle,S,Corner
  *Get,N_num,Node,,Count
  F_N=ArbF/N_num
  *Do,J,1,N_Num
    *if,Arg5,eq,1,then
      F,Nelem(Eleno(i),J),Arg4,F_N
    *elseif,Arg5,eq,-1,then
      F,Nelem(Eleno(i),J),Arg4,-F_N
    *endif
  *Enddo
*ENDDO
Esla,s
Fcum,Repl
Dofsel,All
Allsel
ELnum=$ArbP=$ArbF=$F_N=$N_num=$Inum=$E_area=
*End
!================ 宏结束 ================

!==================== 初始化设置 ====================
FINISH
/CLEAR,NOSTART
/uis,msgpop,3
KEYW,PR_SGVOF,1
/NERR,0,999999
/FILNAME,'BucklingAnalysis'
/TITLE,Parabolic Arch Buckling Analysis with Elastic Supports

!==================== 用户参数区 ====================
!-------- 材料参数 --------
E_val   = {params['E']}e9
rho_val = {params['rho']}e3
mu_val  = {params['mu']}

!-------- 几何参数（抛物线拱） --------
L_arch   = {params['L']}          ! 跨径 (m)
f_rise   = {params['f_arch']}        ! 矢高 (m)
h_section= {params['h']}      ! 壳厚 (m)
b_section= {params['b']}      ! 沿Z方向板宽（拉伸方向宽度，用于生成面积）
S_arch   = {params['S']}          ! 拱长度 (m)

!-------- 分析控制参数 --------
n_points          = 21      ! 曲线采样关键点，用于生成样条
n_elements        = 500     ! 建议线单元数（此处已通过LESIZE控制）
imp_factor        = 0.005   ! 初始缺陷系数（基于一阶屈曲模态）
n_buckling_modes  = 5

!-------- 拱脚边界条件参数（铰接+转动弹簧/固结） --------
! 本版本修改说明：
! - 拱脚采用铰接约束（UX=UY=0）+ 转动弹性支撑/固结约束
! - 删除X/Y平移弹簧，仅保留转动弹簧
! - 转动刚度使用 eta/(1-eta) 映射：kth = DI/L * eta/(1-eta)
! - etaRot=0 → kth=0（铰接）
! - etaRot=0.5 → kth=DI/L（中等刚度）
! - etaRot→1 → kth→+∞（固结）
! - 当eta>=0.99时直接施加ROTZ=0，避免数值溢出
! - 左侧固结标志: is_fixed_L = {params['is_fixed_L']}
! - 右侧固结标志: is_fixed_R = {params['is_fixed_R']}
kthL_val = {params['kthL']:.6e}  ! 左侧转动弹簧刚度（仅当is_fixed_L=False时使用）
kthR_val = {params['kthR']:.6e}  ! 右侧转动弹簧刚度（仅当is_fixed_R=False时使用）
L0_anchor = 0.1       ! 锚固节点与拱脚的偏置距离（几何上远离，避免刚度耦合）

!==================== 前处理模块 ====================
/PREP7

!-------- 抛物线拱几何建模（XY面内，沿Z拖拽成面） --------
*DO,i,1,n_points
  x = -L_arch/2 + (i-1)*(L_arch/(n_points-1))
  y = (4*f_rise/(L_arch**2))*( (L_arch/2)**2 - x**2 )
  K,i, x, y, 0
*ENDDO

FLST,3,n_points,3
*DO,i,1,n_points
  FITEM,3,i
*ENDDO
BSPLIN, ,P51X                      ! 生成抛物线样条（线1）

K,n_points+1,-L_arch/2,0,b_section ! Z向拉伸的端点
LSTR,1,n_points+1                  ! 生成线（线2）
ADRAG, 1, , , , , , 2              ! 沿Z方向拖拽生成面（面1）

!-------- 单元与材料 --------
ET,1,SHELL181
KEYOPT,1,1,0
KEYOPT,1,8,2

*dim,Elastic_modulus,array,10
Elastic_modulus(1)={params['E_list'][0]},{params['E_list'][1]},{params['E_list'][2]},{params['E_list'][3]},{params['E_list'][4]},{params['E_list'][5]},{params['E_list'][6]},{params['E_list'][7]},{params['E_list'][8]},{params['E_list'][9]}
*dim,Density,array,10
Density(1)={params['rho_list'][0]},{params['rho_list'][1]},{params['rho_list'][2]},{params['rho_list'][3]},{params['rho_list'][4]},{params['rho_list'][5]},{params['rho_list'][6]},{params['rho_list'][7]},{params['rho_list'][8]},{params['rho_list'][9]}
*dim,mu_val,array,10
mu_val(1)={params['mu_list'][0]},{params['mu_list'][1]},{params['mu_list'][2]},{params['mu_list'][3]},{params['mu_list'][4]},{params['mu_list'][5]},{params['mu_list'][6]},{params['mu_list'][7]},{params['mu_list'][8]},{params['mu_list'][9]}

*DO,i,1,10
MPTEMP,,,,,,,,
MPTEMP,1,0
MPDATA,EX,i,,Elastic_modulus(i)
MPDATA,PRXY,i,,mu_val(i)
MPTEMP,,,,,,,,
MPTEMP,1,0
MPDATA,DENS,i,,Density(i)
*ENDDO


!-------- 壳截面定义 --------
SECTYPE,1,SHELL
*DO,i,1,10
SECDATA,h_section/10,i,0.0,3
*ENDDO
secoffset,MID
seccontrol,,,, , , ,

!-------- 网格划分 --------
TYPE,1
MAT,1
SECNUM,1

LESIZE,1,,,160    ! 曲线密度（与几何线编号保持一致：线1）
LESIZE,3,,,160    ! 辅助线（通常为边线）
LESIZE,4,,,4
LESIZE,5,,,4

ASEL,S,,,1
MSHKEY,1
AMESH,ALL

!==================== 定义拱脚边界条件 ====================
! 本版本：铰接约束 + 转动弹性支撑/固结约束
! 根据etaRot值决定：eta>=0.99时直接固结（ROTZ=0），否则使用转动弹簧

{left_spring_et_block}
{right_spring_et_block}

! 2) 生成转动弹簧锚固节点（仅当需要弹簧时生成）
*GET, MAX_NODE, NODE, 0, NUM, MAX
nodestart = MAX_NODE+1  ! 锚固节点编号基准（保持编号一致性）

{left_spring_anchor_block}
{right_spring_anchor_block}

! 3) 获取拱脚实际连接节点（Z=0, b/2, b）
Xl=-L_arch/2
Yl=0
Xr= L_arch/2
Yr=0
Z0=0
Zm=b_section/2
Z1=b_section

nodeL_1 = NODE(Xl,Yl,Z0)
nodeL_2 = NODE(Xl,Yl,Zm)
nodeL_3 = NODE(Xl,Yl,Z1)

nodeR_1 = NODE(Xr,Yr,Z0)
nodeR_2 = NODE(Xr,Yr,Zm)
nodeR_3 = NODE(Xr,Yr,Z1)

! 4) 连接转动弹簧（仅当需要弹簧时连接）
{left_spring_connect_block}
{right_spring_connect_block}

! 记录锚固节点范围（用于后续约束）
*GET, MAX_NODE, NODE, 0, NUM, MAX
nodeend = MAX_NODE

FINISH

!==================== 边界与分析流程 ====================
/SOLU
! 1) 约束转动弹簧锚固节点（仅当存在锚固节点时）
!    将弹簧另一端”接地”，约束全部自由度
{left_anchor_constraint_block}
{right_anchor_constraint_block}

! 2) 拱脚铰接约束（UX=UY=0）
!    对拱脚节点施加平移约束
D,nodeL_1,UX,0
D,nodeL_1,UY,0
D,nodeL_2,UX,0
D,nodeL_2,UY,0
D,nodeL_3,UX,0
D,nodeL_3,UY,0

D,nodeR_1,UX,0
D,nodeR_1,UY,0
D,nodeR_2,UX,0
D,nodeR_2,UY,0
D,nodeR_3,UX,0
D,nodeR_3,UY,0

! 3) 转动约束（根据etaRot值决定）
!    当eta>=0.99时直接固结（ROTZ=0），否则保留转动自由度
{left_rotz_constraint}
{right_rotz_constraint}

! 4) 最小面外约束（仅关键节点UZ=0），保持平面内问题稳定
node_M = NODE(0, f_rise, b_section/2)
FLST,2,7,1,ORDE,7
FITEM,2,nodeL_1
FITEM,2,nodeL_2
FITEM,2,nodeL_3
FITEM,2,nodeR_1
FITEM,2,nodeR_2
FITEM,2,nodeR_3
FITEM,2,node_M
/GO
D,P51X,,, , , ,UZ
/GO
D,P51X, , , , , ,UZ

ALLSEL

!==================== 模态分析 ====================
/SOLU
ANTYPE,2
MODOPT,LANB,n_buckling_modes
EQSLV,SPAR
MXPAND,n_buckling_modes, , ,0
LUMPM,0
PSTRES,0
MODOPT,LANB,n_buckling_modes,0,11111, ,OFF
/STATUS,SOLU
SOLVE
FINISH
!导出自振频率
*DEL,arch_frequcy
*DIM,arch_frequcy,array,n_buckling_modes,1
*do,i,1,n_buckling_modes,1
!set,,,1,,,,i,
*GET,FREQT,MODE,i,FREQ
arch_frequcy(i)=FREQT
*enddo
*CREATE,scratch,gui
out_path = strcat('..\result\','freq')
/OUTPUT,out_path,'txt'
*VWRITE,arch_frequcy(1,1)
%G
/OUTPUT,TERM
*END
/INPUT,scratch,gui
!==================== 静力预分析（用于预应力/几何刚化） ====================
/SOLU
ANTYPE,STATIC
PSTRES,ON
FDELE,ALL,ALL
ASEL,S, , , 1
Arbpres,1,'Y',1,'FY',-1
SOLVE
FINISH

!==================== 特征值屈曲分析 ====================
/SOLU
ANTYPE,BUCKLE
BUCOPT,LANB,n_buckling_modes
MXPAND,n_buckling_modes
SOLVE
FINISH

!==================== 引入初始缺陷（基于一阶屈曲模态） ====================
/POST1
ALLSEL,ALL

! ---- 读取一阶屈曲模态 ----
! SET, Lstep, SBSTEP, Fact, KIMG, ...
!   Fact=1 → 只取归一化模态位移，不乘特征值
SET,1,1,1                    ! ★ 修复1：三个参数，Fact=1 在第3位

! ---- 获取一阶屈曲荷载因子（特征值） ----
*GET,buckle_load_factor,ACTIVE,,SET,FREQ

! ---- 获取模态最大合位移 ----
NSORT,U,SUM,1                ! ★ 修复2：先按 USUM 降序排列
*GET,U_max,SORT,,MAX         !           再取排序后的最大值

! ---- 调试输出 ----
*STATUS,U_max
*STATUS,buckle_load_factor

! ---- 计算缺陷缩放系数 ----
! 目标：使几何缺陷最大值 = L_arch/1000（论文 S/1000）
! 原理：UPGEOM 会把 RST 中的位移 × Factor 叠加到坐标上
!       RST 中存的是归一化模态，max = U_max
!       所以 Factor × U_max = L_arch/1000  →  Factor = L_arch/1000/U_max
Defect_Max = S_arch/1000
Factor     = Defect_Max/U_max

*STATUS,Factor
*STATUS,Defect_Max

FINISH

! ---- 将缺陷叠加到几何 ----
/PREP7
UPGEOM,Factor,1,1,'BucklingAnalysis','rst'   ! ★ 修复3：用 Factor，不是 imp_factor
FINISH
!==================== 非线性弧长后屈曲分析 ====================
/SOLU
NCNV,2,0,0,0,0
ANTYPE,STATIC
NROPT,UNSYM
NLGEOM,ON
OUTRES,ALL,ALL
ARCLEN,ON,25,0.01
NSUBST,50,200,20
NEQIT,200
CNVTOL,F,0.005
FDELE,ALL,ALL
ASEL,S, , , 1
Arbpres,1,'Y',1.1*buckle_load_factor,'FY',-1
TIME,1.1*buckle_load_factor
SOLVE
FINISH

!==================== 后处理：按原脚本输出 ====================
/post26
RESET
FILE,'BucklingAnalysis','rst','.'
STORE,MERGE
NUMVAR,200

*DO,i,1,22
  *dim,node_num_list,array,22
  node_num_list(1)=1,1,10,18,26,34,42,50,58,66
  node_num_list(11)=74,82,90,98,106,114,122,130,138,146
  node_num_list(21)=154,2
  node_num = node_num_list(i)

  NSOL,2,node_num,U,x
  NSOL,3,node_num,U,Y
  NSOL,4,node_num,ROT,Z

  *IF, i, EQ, 1, THEN
    fname = STRCAT('load-disp_',CHRVAL(1))
  *ELSE
    fname = STRCAT('load-disp_',CHRVAL(i-1))
  *ENDIF

  *GET,num_steps,VARI,0,NSETS
  *DEL,disp_data
  *DIM,disp_data,TABLE,num_steps,4
  VGET,disp_data(1,0),1
  VGET,disp_data(1,1),2
  VGET,disp_data(1,2),3
  VGET,disp_data(1,3),4

  /OUTPUT,fname,'csv','../result'
  *VWRITE,'TIME','UX','UY','ROTZ'
  %C, %C, %C, %C
  *VWRITE,disp_data(1,0),disp_data(1,1),disp_data(1,2),disp_data(1,3)
  %G, %G, %G, %G
  /OUTPUT,TERM
*ENDDO
'''

def main():
    # 调整max_workers，避免占用过多资源
    cpu_count = os.cpu_count() or 4
    max_workers = max(1, cpu_count // 4)
    print(f"CPU核心数：{cpu_count}，并行任务数：{max_workers}")

    # 生成命令（修改num_samples调整总样本数，建议先跑100个测试）
    commands = create_commands(
        r"./LHS_Arch_M1_v4",
        num_samples=5000  # 先改小测试，比如100
    )

    if not commands:
        print("没有生成有效命令，退出。")
        return

    # 多进程提交
    print(f"\n开始提交 {len(commands)} 个计算任务...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        # 提交初始任务
        for _ in range(min(max_workers, len(commands))):
            command = commands.pop(0)
            future = executor.submit(run_ansys_command, command)
            future.add_done_callback(task_done)
            futures.append(future)

        # 监控任务完成并添加新任务
        while futures:
            done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                futures.remove(future)
                if commands:
                    command = commands.pop(0)
                    new_future = executor.submit(run_ansys_command, command)
                    new_future.add_done_callback(task_done)
                    futures.append(new_future)

    print("\n所有任务提交完成！")

if __name__ == "__main__":
    main()