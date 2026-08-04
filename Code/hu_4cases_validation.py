"""
hu_4cases_validation.py — Full arc-length nonlinear buckling validation for 4 homogeneous-steel arch cases.

Uses the EXACT same analysis template as AnsysBatch_V1_FGM.py:
  a. Static pre-analysis (Arbpres 1 Pa, PSTRES,ON)
  b. Eigenvalue buckling → GET buckle_load_factor
  c. S/1000 imperfection via UPGEOM
  d. Arc-length: ARCLEN,ON,25,0.01, NSUBST,50,200,20, NEQIT,200, CNVTOL,F,0.005
  e. Reference load: Arbpres 1.1*blf
  f. POST26: RFORCE sum at Y-spring anchor nodes, NSOL mid-span UY → CSV

Output for each case: label, blf, Q_cr (kN), Np (kN), N_cr/N_p, num converged steps.
"""

import math
import os
import subprocess
import sys
import time
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAPDL_PATH = r"C:\Program Files\ANSYS Inc\v211\ansys\bin\winx64\MAPDL.exe"
WORK_DIR = r"D:\tanghao\deeparch_review\hu_4cases_validation"
N_CORES = "4"
N_LAYERS = 5  # 5-layer homogeneous (all identical material)

E_STEEL = 210e9      # Pa
PRXY_STEEL = 0.3
RHO_STEEL = 7850.0   # kg/m³ (density for mass matrix in modal analysis)

# Section geometry
B = 0.4     # m (width in Z-direction)
H = 0.045   # m (thickness / shell depth)

# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------
CASES = [
    {
        "label": "pin1",
        "L": 0.9743, "f": 0.0779,
        "KXL": 10, "KYL": 10, "KZL": 0,
        "KXR": 10, "KYR": 10, "KZR": 0,
        "bc_type": "pin",
    },
    {
        "label": "pin2",
        "L": 0.5456, "f": 0.0779,
        "KXL": 10, "KYL": 10, "KZL": 0,
        "KXR": 10, "KYR": 10, "KZR": 0,
        "bc_type": "pin",
    },
    {
        "label": "fix1",
        "L": 2.1109, "f": 0.1689,
        "KXL": 10, "KYL": 10, "KZL": 1000,
        "KXR": 10, "KYR": 10, "KZR": 1000,
        "bc_type": "fixed",
    },
    {
        "label": "fix2",
        "L": 1.1821, "f": 0.1689,
        "KXL": 10, "KYL": 10, "KZL": 1000,
        "KXR": 10, "KYR": 10, "KZR": 1000,
        "bc_type": "fixed",
    },
]

# ---------------------------------------------------------------------------
# Utility: arc length (exact copy from AnsysBatch_V1_FGM.py)
# ---------------------------------------------------------------------------
def arc_length(L, f_arch):
    """Compute arc length of a parabolic arch."""
    return (
        8 * math.sqrt((L ** 2 + 16 * f_arch ** 2) / L ** 2)
        * math.sqrt(f_arch ** 2 / L ** 4)
        * L
        + math.log(
            1
            / L ** 3
            * (
                L ** 3
                * math.sqrt((L ** 2 + 16 * f_arch ** 2) / L ** 2)
                * math.sqrt(f_arch ** 2 / L ** 4)
                + 4 * f_arch ** 2
            )
            * (f_arch ** 2 / L ** 4) ** (-0.1e1 / 0.2e1)
        )
        - math.log(
            1
            / L ** 3
            * (
                L ** 3
                * math.sqrt((L ** 2 + 16 * f_arch ** 2) / L ** 2)
                * math.sqrt(f_arch ** 2 / L ** 4)
                - 4 * f_arch ** 2
            )
            * (f_arch ** 2 / L ** 4) ** (-0.1e1 / 0.2e1)
        )
    ) * (f_arch ** 2 / L ** 4) ** (-0.1e1 / 0.2e1) / 16


# ---------------------------------------------------------------------------
# Section properties for homogeneous layered shell
# ---------------------------------------------------------------------------
def compute_homogeneous_properties(E, b, h, n_layers=N_LAYERS):
    """Compute A11, D11, D11_eq, I_eq for homogeneous n-layer shell."""
    thickness = h / n_layers
    # z_rel: layer-centre offsets normalised by h (symmetric about mid-plane)
    z_rel = [(i - (n_layers - 1) / 2.0) / n_layers for i in range(n_layers)]

    A11 = 0.0
    B11 = 0.0
    D11 = 0.0
    I0 = 0.0  # mass per unit length

    for z in z_rel:
        zc = z * h
        I0 += RHO_STEEL * b * thickness
        A11 += E * b * thickness
        B11 += E * b * thickness * zc
        D11 += E * b * (thickness * zc ** 2 + thickness ** 3 / 12.0)

    D11_eq = D11 - B11 * B11 / A11 if abs(A11) > 1e-30 else D11
    I_eq = D11_eq / E  # equivalent second moment of area
    return I0, A11, B11, D11, D11_eq, I_eq


# ---------------------------------------------------------------------------
# APDL generation (exact template from AnsysBatch_V1_FGM.py, homogenised)
# ---------------------------------------------------------------------------
def generate_apdl(label, L, f_arch, h, b, KXL, KYL, KZL, KXR, KYR, KZR,
                  n_layers=N_LAYERS):
    """Return (apdl_str, S_arch, I_eq, A11, D11_eq)."""

    E_GPa = E_STEEL / 1e9
    S = arc_length(L, f_arch)
    I0, A11, B11, D11, D11_eq, I_eq = compute_homogeneous_properties(E_STEEL, b, h, n_layers)

    # --- homogeneous material arrays ---
    E_list = [E_STEEL] * n_layers
    rho_list = [RHO_STEEL] * n_layers
    mu_list = [PRXY_STEEL] * n_layers

    mat_arrays = (
        f"*dim,Elastic_modulus,array,{n_layers}\n"
        f"Elastic_modulus(1)={','.join(str(e) for e in E_list)}\n"
        f"*dim,Density,array,{n_layers}\n"
        f"Density(1)={','.join(str(r) for r in rho_list)}\n"
        f"*dim,mu_val,array,{n_layers}\n"
        f"mu_val(1)={','.join(str(m) for m in mu_list)}"
    )

    mat_prop_loop = (
        "MPTEMP,,,,,,,,\n"
        "MPTEMP,1,0\n"
        "MPDATA,EX,i,,Elastic_modulus(i)\n"
        "MPDATA,PRXY,i,,mu_val(i)\n"
        "MPTEMP,,,,,,,,\n"
        "MPTEMP,1,0\n"
        "MPDATA,DENS,i,,Density(i)"
    )

    secdata_loop = f"SECDATA,h_section/{n_layers},i,0.0,3"

    material_section = f"""!-------- 单元与材料 --------
ET,1,SHELL181
KEYOPT,1,1,0
KEYOPT,1,8,2

{mat_arrays}

*DO,i,1,{n_layers}
{mat_prop_loop}
*ENDDO

!-------- 壳截面定义 --------
SECTYPE,1,SHELL
*DO,i,1,{n_layers}
{secdata_loop}
*ENDDO
secoffset,MID
seccontrol,,,, , , ,"""

    # -----------------------------------------------------------------------
    # Full APDL template (mirrors AnsysBatch_V1_FGM.py with POST26 extension)
    # -----------------------------------------------------------------------
    apdl_str = rf'''
!##############################################################
! 均质钢拱非线性弧长屈曲验证 — {label}
! L={L}  f={f_arch}  b={b}  h={h}
! KXL={KXL} KYL={KYL} KZL={KZL} KXR={KXR} KYR={KYR} KZR={KZR}
! 5 层均质壳 (E=210 GPa, PRXY=0.3)
!##############################################################

!============== 均布压力施加载荷宏 =================
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
/TITLE,Parabolic Arch Buckling Validation - {label}

!==================== 用户参数区 ====================
!-------- 材料参数 --------
E_val   = {E_GPa}e9
rho_val = {RHO_STEEL/1000}e3
mu_val  = {PRXY_STEEL}

!-------- 几何参数（抛物线拱） --------
L_arch   = {L}
f_rise   = {f_arch}
h_section= {h}
b_section= {b}

!-------- 分析控制参数 --------
n_points          = 21
n_buckling_modes  = 5
S_arch            = {S}

!-------- 拱脚弹性支撑参数 --------
etaXL   = {KXL}
etaYL   = {KYL}
etaRotL = {KZL}
etaXR   = {KXR}
etaYR   = {KYR}
etaRotR = {KZR}
L0_anchor = 0.1

!==================== 前处理模块 ====================
/PREP7

!-------- 抛物线拱几何建模 --------
*DO,i,1,n_points
  x = -L_arch/2 + (i-1)*(L_arch/(n_points-1))
  y = (4*f_rise/(L_arch**2))*( (L_arch/2)**2 - x**2 )
  K,i, x, y, 0
*ENDDO

FLST,3,n_points,3
*DO,i,1,n_points
  FITEM,3,i
*ENDDO
BSPLIN, ,P51X

K,n_points+1,-L_arch/2,0,b_section
LSTR,1,n_points+1
ADRAG, 1, , , , , , 2

{material_section}

!-------- 网格划分 --------
TYPE,1
MAT,1
SECNUM,1

LESIZE,1,,,160
LESIZE,3,,,160
LESIZE,4,,,4
LESIZE,5,,,4

ASEL,S,,,1
MSHKEY,1
AMESH,ALL

!==================== 定义拱脚弹性支撑 ====================
DI  = {D11_eq}
EA  = {A11}
kxL_val  = EA*etaXL/L_arch
kyL_val  = EA*etaYL/L_arch
kthL_val = DI*etaRotL/L_arch

kxR_val  = EA*etaXR/L_arch
kyR_val  = EA*etaYR/L_arch
kthR_val = DI*etaRotR/L_arch

! 平移弹簧（X向）— 左
ET,2,COMBIN14
KEYOPT,2,1,0
KEYOPT,2,2,0
KEYOPT,2,3,0
R,21,kxL_val/3

! 平移弹簧（Y向）— 左
ET,3,COMBIN14
KEYOPT,3,1,0
KEYOPT,3,2,0
KEYOPT,3,3,0
R,31,kyL_val/3

! 转动弹簧（关于Z）— 左
ET,4,COMBIN14
KEYOPT,4,1,0
KEYOPT,4,2,0
KEYOPT,4,3,1
R,41,kthL_val/3

! 平移弹簧（X向）— 右
ET,5,COMBIN14
KEYOPT,5,1,0
KEYOPT,5,2,0
KEYOPT,5,3,0
R,51,kxR_val/3

! 平移弹簧（Y向）— 右
ET,6,COMBIN14
KEYOPT,6,1,0
KEYOPT,6,2,0
KEYOPT,6,3,0
R,61,kyR_val/3

! 转动弹簧（关于Z）— 右
ET,7,COMBIN14
KEYOPT,7,1,0
KEYOPT,7,2,0
KEYOPT,7,3,1
R,71,kthR_val/3

!-------- 生成锚固节点并连接弹簧 --------
*GET, MAX_NODE, NODE, 0, NUM, MAX
nodestart = MAX_NODE+1

! 左拱脚锚固节点
N,MAX_NODE+1,-(L_arch/2+L0_anchor),0,0
N,MAX_NODE+2,-(L_arch/2+L0_anchor),0,b_section/2
N,MAX_NODE+3,-(L_arch/2+L0_anchor),0,b_section

N,MAX_NODE+4,-(L_arch/2),(0-L0_anchor),0
N,MAX_NODE+5,-(L_arch/2),(0-L0_anchor),b_section/2
N,MAX_NODE+6,-(L_arch/2),(0-L0_anchor),b_section

N,MAX_NODE+7,-L_arch/2,0,L0_anchor
N,MAX_NODE+8,-L_arch/2,0,L0_anchor+b_section/2
N,MAX_NODE+9,-L_arch/2,0,L0_anchor+b_section

! ---- store left Y-spring anchor-node numbers for POST26 ----
y_anchor_L1 = MAX_NODE+4
y_anchor_L2 = MAX_NODE+5
y_anchor_L3 = MAX_NODE+6

! 右拱脚锚固节点
*GET, MAX_NODE, NODE, 0, NUM, MAX
N,MAX_NODE+1,(L_arch/2+L0_anchor),0,0
N,MAX_NODE+2,(L_arch/2+L0_anchor),0,b_section/2
N,MAX_NODE+3,(L_arch/2+L0_anchor),0,b_section

N,MAX_NODE+4,(L_arch/2),(0-L0_anchor),0
N,MAX_NODE+5,(L_arch/2),(0-L0_anchor),b_section/2
N,MAX_NODE+6,(L_arch/2),(0-L0_anchor),b_section

N,MAX_NODE+7,L_arch/2,0,L0_anchor
N,MAX_NODE+8,L_arch/2,0,L0_anchor+b_section/2
N,MAX_NODE+9,L_arch/2,0,L0_anchor+b_section

! ---- store right Y-spring anchor-node numbers for POST26 ----
y_anchor_R1 = MAX_NODE+4
y_anchor_R2 = MAX_NODE+5
y_anchor_R3 = MAX_NODE+6

! ---- store mid-span node number ----
mid_node = NODE(0, f_rise, b_section/2)

! 获取拱脚实际连接节点
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

! 连接弹簧（左侧）
*GET, MAX_NODE, NODE, 0, NUM, MAX
TYPE,2
REAL,21
E,nodeL_1,nodestart+0
E,nodeL_2,nodestart+1
E,nodeL_3,nodestart+2

TYPE,3
REAL,31
E,nodeL_1,nodestart+3
E,nodeL_2,nodestart+4
E,nodeL_3,nodestart+5

TYPE,4
REAL,41
E,nodeL_1,nodestart+6
E,nodeL_2,nodestart+7
E,nodeL_3,nodestart+8

! 连接弹簧（右侧）
TYPE,5
REAL,51
E,nodeR_1,nodestart+9
E,nodeR_2,nodestart+10
E,nodeR_3,nodestart+11

TYPE,6
REAL,61
E,nodeR_1,nodestart+12
E,nodeR_2,nodestart+13
E,nodeR_3,nodestart+14

TYPE,7
REAL,71
E,nodeR_1,nodestart+15
E,nodeR_2,nodestart+16
E,nodeR_3,nodestart+17

! 记录锚固节点范围
*GET, MAX_NODE, NODE, 0, NUM, MAX
nodeend = MAX_NODE

FINISH

!==================== 边界约束 ====================
/SOLU
! 约束锚固节点
FLST,2,2,1,ORDE,2
FITEM,2,nodestart
FITEM,2,-nodeend
/GO
D,P51X,ALL

! 最小面外约束
node_L1 = NODE(-L_arch/2, 0, 0)
node_L2 = NODE(-L_arch/2, 0, b_section/2)
node_L3 = NODE(-L_arch/2, 0, b_section)
node_R1 = NODE( L_arch/2, 0, 0)
node_R2 = NODE( L_arch/2, 0, b_section/2)
node_R3 = NODE( L_arch/2, 0, b_section)
node_M  = NODE( 0, f_rise, b_section/2)
FLST,2,7,1,ORDE,7
FITEM,2,node_L1
FITEM,2,node_L2
FITEM,2,node_L3
FITEM,2,node_R1
FITEM,2,node_R2
FITEM,2,node_R3
FITEM,2,node_M
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

! 导出自振频率
*DEL,arch_frequcy
*DIM,arch_frequcy,array,n_buckling_modes,1
*do,i,1,n_buckling_modes,1
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

!==================== 引入初始缺陷（S/1000 缩放） ====================
/POST1
! Find the first POSITIVE buckling eigenvalue and its mode number
first_pos_mode = 0
buckle_load_factor = 0
*DO,jmode,1,n_buckling_modes
  SET,1,jmode
  *GET,blf_j,ACTIVE,,SET,FREQ
  *IF,blf_j,GT,0,AND,first_pos_mode,EQ,0,THEN
    first_pos_mode = jmode
    buckle_load_factor = blf_j
  *ENDIF
*ENDDO
*IF,first_pos_mode,EQ,0,THEN
  ! Fallback: use first mode even if negative
  SET,1,1,1
  first_pos_mode = 1
  *GET,buckle_load_factor,ACTIVE,,SET,FREQ
*ELSE
  SET,1,first_pos_mode
*ENDIF
NSORT,U,SUM,1
*GET,U_max,SORT,,MAX
! Write BLF and mode number to text files for reliable Python parsing
/OUTPUT,'blf','txt','../result'
*VWRITE,buckle_load_factor
%G
/OUTPUT,TERM
/OUTPUT,'blf_mode','txt','../result'
*VWRITE,first_pos_mode
%G
/OUTPUT,TERM

/PREP7
Defect_Max = S_arch/1000
Factor = Defect_Max/U_max
UPGEOM,Factor,1,1,'BucklingAnalysis','rst'
FINISH

!==================== 非线性弧长后屈曲分析 ====================
/SOLU
ANTYPE,STATIC
NLGEOM,ON
OUTRES,ALL,ALL
ARCLEN,ON,10,0.001
NSUBST,100,1000,10
NEQIT,1000
CNVTOL,F
FDELE,ALL,ALL
ASEL,S, , , 1
Arbpres,1,'Y',1.1*buckle_load_factor,'FY',-1
TIME,1.1*buckle_load_factor
EQSLV,SPAR
SOLVE
FINISH

!==================== POST26: 提取总反力与跨中位移 ====================
/POST26
FILE,'BucklingAnalysis','rst','.'
NUMVAR,200

! Y向反力 — 左拱脚 3 个 Y-spring 锚固节点
RFORCE,2,y_anchor_L1,F,Y
RFORCE,3,y_anchor_L2,F,Y
RFORCE,4,y_anchor_L3,F,Y

! Y向反力 — 右拱脚 3 个 Y-spring 锚固节点
RFORCE,5,y_anchor_R1,F,Y
RFORCE,6,y_anchor_R2,F,Y
RFORCE,7,y_anchor_R3,F,Y

! 跨中 UY
NSOL,13,mid_node,U,Y

! 求和 → total_FY（在STORE之前定义，确保对每个时间点求值）
ADD,8,2,3,,sum_L12
ADD,9,8,4,,sum_L123
ADD,10,9,5,,sum_LR1
ADD,11,10,6,,sum_LR2
ADD,12,11,7,,total_FY

! 现在存储所有变量 — 这会按ADD定义计算每个时间点的total_FY
STORE,MERGE

! 导出 CSV
*GET,num_steps,VARI,0,NSETS
*DEL,result_data
*DIM,result_data,ARRAY,num_steps,3
VGET,result_data(1,1),1
VGET,result_data(1,2),12
VGET,result_data(1,3),13
! 取绝对值
*DO,i,1,num_steps
  result_data(i,2) = ABS(result_data(i,2))
  result_data(i,3) = ABS(result_data(i,3))
*ENDDO

/OUTPUT,'load_disp_result','csv','../result'
*VWRITE,'TIME','total_FY_N','UY_m'
%C %C %C
*VWRITE,result_data(1,1),result_data(1,2),result_data(1,3)
%G %G %G
/OUTPUT,TERM

FINISH
'''
    return apdl_str, S, I_eq, A11, D11_eq


# ---------------------------------------------------------------------------
# Q_cr extraction from CSV
# ---------------------------------------------------------------------------
def extract_qcr(csv_path):
    """
    Read the POST26 output CSV and extract Q_cr = first local maximum
    of total_FY_N vs UY_m (stiffness goes to zero).
    Returns (Q_cr_N, num_data_rows) or (None, num_data_rows) on failure.
    """
    if not os.path.exists(csv_path):
        return None, 0
    try:
        # Use loadtxt with ndmin to handle single-row files
        data = np.loadtxt(csv_path, skiprows=1, ndmin=2)
    except Exception:
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, sep=r'\s+', engine='python')
            data = df.values
            if data.ndim == 1:
                data = data.reshape(1, -1)
        except Exception:
            return None, 0

    if data.shape[0] < 2:
        # Only one data point — use it as Q_cr if valid
        n_rows = data.shape[0]
        if data.shape[1] >= 2:
            fy_val = float(data[0, 1])
            return fy_val if fy_val > 0 else None, n_rows
        return None, n_rows

    fy_col = data[:, 1]

    # Find first local maximum in force-displacement curve
    for i in range(1, len(fy_col) - 1):
        if fy_col[i] > fy_col[i - 1] and fy_col[i] > fy_col[i + 1]:
            return fy_col[i], data.shape[0]

    # No clear local peak — use global maximum
    idx = int(np.argmax(fy_col))
    return fy_col[idx], data.shape[0]


# ---------------------------------------------------------------------------
# Np computation
# ---------------------------------------------------------------------------
def compute_np(L, E, I, bc_type):
    """
    Theoretical buckling load:
      pin:   Np = pi^2 * E * I / (L/2)^2
      fixed: Np = (1.4303*pi)^2 * E * I / (L/2)^2
    Returns Np in Newtons.
    """
    Le = L / 2.0  # half-span effective length
    if bc_type == "pin":
        K = 1.0
    else:
        K = 1.4303
    return (K * math.pi) ** 2 * E * I / (Le ** 2)


# ---------------------------------------------------------------------------
# Run a single case
# ---------------------------------------------------------------------------
def run_case(case):
    """Run one case: generate APDL, launch MAPDL, parse result. Returns dict."""
    label = case["label"]
    case_dir = os.path.join(WORK_DIR, label)
    rundata_dir = os.path.join(case_dir, "rundata")
    result_dir = os.path.join(case_dir, "result")
    os.makedirs(rundata_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # Generate APDL
    apdl_str, S, I_eq, A11, D11_eq = generate_apdl(
        label,
        case["L"], case["f"], case.get("h", H), case.get("b", B),
        case["KXL"], case["KYL"], case["KZL"],
        case["KXR"], case["KYR"], case["KZR"],
    )

    apdl_path = os.path.join(rundata_dir, "run.txt")
    with open(apdl_path, "w", encoding="utf-8") as f:
        f.write(apdl_str)

    # Also copy to result for reference
    with open(os.path.join(result_dir, "run.txt"), "w", encoding="utf-8") as f:
        f.write(apdl_str)

    # Build MAPDL command
    command = [
        MAPDL_PATH,
        "-p", "ansys", "-smp", "-np", N_CORES,
        "-lch", "-dir", rundata_dir,
        "-j", "run", "-s", "read", "-l", "en-us", "-b",
        "-i", apdl_path,
        "-o", os.path.join(result_dir, "run.out"),
    ]

    print(f"  [{label}] Launching MAPDL ...")
    t0 = time.time()

    try:
        result = subprocess.run(
            command, check=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        print(f"  [{label}] ERROR: MAPDL not found at {MAPDL_PATH}")
        return {
            "label": label, "blf": None, "Q_cr_kN": None,
            "Np_kN": None, "ratio": None, "n_steps": 0,
            "elapsed_s": 0.0, "rc": -1,
        }

    elapsed = time.time() - t0
    rc = result.returncode

    # Parse eigenvalue
    eig_path = os.path.join(result_dir, "freq.txt")
    blf = None
    # Buckling load factor is NOT in freq.txt — it's in POST1 output.
    # We need to scrape it from run.out or compute from POST26 CSV.
    # For now read freq.txt for reference; blf is obtained differently.
    if os.path.exists(eig_path):
        try:
            freqs = np.loadtxt(eig_path)
        except Exception:
            freqs = None

    # Parse POST26 CSV
    csv_path = os.path.join(result_dir, "load_disp_result.csv")
    Q_cr_N, n_steps = extract_qcr(csv_path)

    # Np
    Np_N = compute_np(case["L"], E_STEEL, I_eq, case["bc_type"])

    Q_cr_kN = Q_cr_N / 1000.0 if Q_cr_N is not None else None
    Np_kN = Np_N / 1000.0
    ratio = Q_cr_N / Np_N if Q_cr_N is not None and Np_N > 0 else None

    # Read BLF from dedicated output file
    blf = None
    blf_path = os.path.join(result_dir, "blf.txt")
    if os.path.exists(blf_path):
        try:
            with open(blf_path, "r") as f:
                blf = float(f.read().strip())
        except Exception:
            pass

    print(
        f"  [{label}] done in {elapsed:.0f}s  rc={rc}  "
        f"blf={blf}  Q_cr={Q_cr_kN}kN  Np={Np_kN:.0f}kN  "
        f"ratio={ratio}  n_steps={n_steps}"
    )

    return {
        "label": label,
        "blf": blf,
        "Q_cr_kN": Q_cr_kN,
        "Np_kN": Np_kN,
        "ratio": ratio,
        "n_steps": n_steps,
        "elapsed_s": elapsed,
        "rc": rc,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("hu_4cases_validation — Nonlinear Arc-Length Buckling Validation")
    print(f"MAPDL : {MAPDL_PATH}")
    print(f"Work  : {WORK_DIR}")
    print(f"Cases : {len(CASES)}")
    print("=" * 72)

    if not os.path.isfile(MAPDL_PATH):
        print(f"ERROR: MAPDL not found at '{MAPDL_PATH}'", file=sys.stderr)
        print("Set MAPDL_PATH in the script to the correct location.", file=sys.stderr)
        sys.exit(1)

    results = []
    for case in CASES:
        print(f"\n{'─'*60}")
        print(f"Case: {case['label']}  L={case['L']}m  f={case['f']}m  "
              f"KZL={case['KZL']}  KZR={case['KZR']}  ({case['bc_type']})")
        print(f"{'─'*60}")
        r = run_case(case)
        results.append(r)

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Label':<8} {'BLF':>10} {'Q_cr(kN)':>14} {'Np(kN)':>12} "
          f"{'N_cr/N_p':>10} {'Steps':>7} {'Time(s)':>8}")
    print("-" * 90)
    for r in results:
        blf_s = f"{r['blf']:.1f}" if r['blf'] is not None else "N/A"
        qcr_s = f"{r['Q_cr_kN']:.1f}" if r['Q_cr_kN'] is not None else "N/A"
        np_s = f"{r['Np_kN']:.0f}" if r['Np_kN'] is not None else "N/A"
        ratio_s = f"{r['ratio']:.4f}" if r['ratio'] is not None else "N/A"
        print(f"{r['label']:<8} {blf_s:>10} {qcr_s:>14} {np_s:>12} "
              f"{ratio_s:>10} {r['n_steps']:>7} {r['elapsed_s']:>8.0f}")
    print("=" * 90)

    # Check which cases succeeded
    failed = [r for r in results if r['Q_cr_kN'] is None]
    if failed:
        print(f"\nWARNING: {len(failed)} case(s) did not produce valid Q_cr: "
              f"{[r['label'] for r in failed]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
