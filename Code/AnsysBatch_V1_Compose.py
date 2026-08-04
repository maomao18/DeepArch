"""
V1_Compose.py - V1 data generation with composite 5-layer material (+-15% random)
DeepArch Project
Generates ANSYS APDL commands and runs MAPDL simulations in parallel.
Default: 500 samples, output to D:\tanghao\V1_Compose_Data

Usage: python Code/V1_Compose.py
"""

import concurrent.futures
import random
import subprocess
import os
import math

MAPDL_PATH = r"C:\Program Files\ANSYS Inc\v211\ansys\bin\winx64\MAPDL.exe"


def run_ansys_command(command):
    """Execute an ANSYS MAPDL command and return stdout, stderr, command."""
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        return result.stdout, result.stderr, command
    except subprocess.CalledProcessError as e:
        return e.stdout, e.stderr, command


def task_done(future):
    """Callback: print output and clean up work directory (keep only run.txt)."""
    stdout, stderr, command = future.result()
    print("Task completed.")
    print("Standard Output:\n", stdout)
    print("Standard Error:\n", stderr)

    for i, arg in enumerate(command):
        if arg == "-dir":
            work_dir = command[i + 1]
            break
    else:
        work_dir = None

    if work_dir:
        for filename in os.listdir(work_dir):
            if filename != "run.txt":
                file_path = os.path.join(work_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)
                        print(f"Deleted {file_path}")
                    else:
                        print(f"Skipped {file_path} (not a file)")
                except OSError as e:
                    print(f"Error deleting {file_path}: {e}")


def arc_length(L, f_arch):
    """Compute arc length of a parabolic arch."""
    return (8 * math.sqrt((L ** 2 + 16 * f_arch ** 2) / L ** 2) * math.sqrt(f_arch ** 2 / L ** 4) * L + math.log(1 / L ** 3 * (L ** 3 * math.sqrt((L ** 2 + 16 * f_arch ** 2) / L ** 2) * math.sqrt(f_arch ** 2 / L ** 4) + 4 * f_arch ** 2) * (f_arch ** 2 / L ** 4) ** (-0.1e1 / 0.2e1)) - math.log(1 / L ** 3 * (L ** 3 * math.sqrt((L ** 2 + 16 * f_arch ** 2) / L ** 2) * math.sqrt(f_arch ** 2 / L ** 4) - 4 * f_arch ** 2) * (f_arch ** 2 / L ** 4) ** (-0.1e1 / 0.2e1))) * (f_arch ** 2 / L ** 4) ** (-0.1e1 / 0.2e1) / 16


def compute_materials(E, rho, mu, b, h):
    """
    Composite 5-layer material computation (+-15% random per layer).
    Returns: (E_list, rho_list, mu_list, I0, A11, B11, D11, n_layers)
    """
    n_layers = 5
    E_list, rho_list, mu_list = [], [], []
    for _ in range(n_layers):
        factor = random.uniform(0.85, 1.15)
        E_list.append(factor * E * 1e9)
        rho_list.append(factor * rho * 1e3)
        mu_list.append(factor * mu)

    I0, A11, B11, D11 = 0.0, 0.0, 0.0, 0.0
    thickness = h / n_layers
    for j in range(n_layers):
        z_center = -h / 2.0 + j * thickness + thickness / 2.0
        I0 += rho_list[j] * b * thickness
        A11 += E_list[j] * b * thickness
        B11 += E_list[j] * b * thickness * z_center
        D11 += E_list[j] * b * (thickness * z_center ** 2 + thickness ** 3 / 12.0)

    D11_eq = D11 - (B11 * B11) / A11 if A11 != 0 else D11
    return E_list, rho_list, mu_list, I0, A11, B11, D11, D11_eq, n_layers


def generate_apdl(E, rho, mu, L, f_arch, h, b, S,
                  KXL, KYL, KZL, KXR, KYR, KZR,
                  E_list, rho_list, mu_list,
                  I0, A11, B11, D11, D11_eq, n_layers):
    """Generate the complete APDL input string."""
    # --- Build dynamic material array definitions ---
    mat_arrays = (
        f"*dim,Elastic_modulus,array,{n_layers}\n"
        f"Elastic_modulus(1)={','.join(str(e) for e in E_list)}\n"
        f"*dim,Density,array,{n_layers}\n"
        f"Density(1)={','.join(str(r) for r in rho_list)}\n"
        f"*dim,mu_val,array,{n_layers}\n"
        f"mu_val(1)={','.join(str(m) for m in mu_list)}"
    )

    # --- Build material property loop (V2-style: use loop variable i) ---
    mat_prop_loop = (
        "MPTEMP,,,,,,,,\n"
        "MPTEMP,1,0\n"
        "MPDATA,EX,i,,Elastic_modulus(i)\n"
        "MPDATA,PRXY,i,,mu_val(i)\n"
        "MPTEMP,,,,,,,,\n"
        "MPTEMP,1,0\n"
        "MPDATA,DENS,i,,Density(i)"
    )

    # --- Build SECDATA loop ---
    secdata_loop = f"SECDATA,h_section/{n_layers},i,0.0,3"

    # --- Assemble full material section ---
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

    # --- Full APDL template ---
    apdl_str = rf'''
!##############################################################
! 拱结构屈曲分析程序 (APDL 19.5)
! 功能：包含特征值屈曲分析+弧长法非线性后屈曲分析
! 修改要点（本版）：
! - 将拱脚原"固结/铰接"改为：X向弹性支撑 + Y向弹性支撑 + 关于Z的转动弹性支撑
! - 参考"弹性支撑模型"的做法：在拱脚附近生成锚固节点，通过COMBIN14弹簧连接
! - 为保持平面内屈曲分析，最小化面外约束：仅在三个代表节点施加UZ=0
! 作者：Hao Tang
! 版本：V1_Compose (5-layer +-15%)
! 最后修改：2025-07-27
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
E_val   = {E}e9
rho_val = {rho}e3
mu_val  = {mu}

!-------- 几何参数（抛物线拱） --------
L_arch   = {L}          ! 跨径 (m)
f_rise   = {f_arch}        ! 矢高 (m)
h_section= {h}      ! 壳厚 (m)
b_section= {b}      ! 沿Z方向板宽（拉伸方向宽度，用于生成面积）

!-------- 分析控制参数 --------
n_points          = 21      ! 曲线采样关键点，用于生成样条
n_elements        = 500     ! 建议线单元数（此处已通过LESIZE控制）
n_buckling_modes  = 5
S_arch            = {S}     ! 弧长 (用于缺陷缩放)

!-------- 拱脚弹性支撑参数（本次新增） --------
! 采用无量纲到有量纲的示例换算（与参考弹性脚本一致的思路）：
etaXL   = {KXL}
etaYL   = {KYL}
etaRotL = {KZL}
etaXR   = {KXR}
etaYR   = {KYR}
etaRotR = {KZR}
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

{material_section}

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

!==================== 定义拱脚弹性支撑（新增） ====================
! 1) 定义弹簧单元类型与刚度
!    - COMBIN14（平移）：X向、Y向
!    - COMBIN14（转动）：KEYOPT(3)=1，关于Z的扭转弹簧（用于平面内弯矩传递）
!
! 等效刚度计算（可按需自行给定kx,ky,kth）：
DI  = {D11_eq}
EA  =  {A11}    ! 与同跨同矢高圆弧的等效半径
kxL_val  = EA*etaXL/L_arch                    ! X向平移弹簧刚度
kyL_val  =  EA*etaYL/L_arch                   ! Y向平移弹簧刚度
kthL_val = DI*etaRotL/L_arch                 ! Z向转动弹簧刚度（弯矩-转角）

kxR_val  = EA*etaXR/L_arch                    ! X向平移弹簧刚度
kyR_val  =  EA*etaYR/L_arch                   ! Y向平移弹簧刚度
kthR_val = DI*etaRotR/L_arch                 ! Z向转动弹簧刚度（弯矩-转角）

! 平移弹簧（X向）
ET,2,COMBIN14
KEYOPT,2,1,0
KEYOPT,2,2,0
KEYOPT,2,3,0
R,21,kxL_val/3

! 平移弹簧（Y向）
ET,3,COMBIN14
KEYOPT,3,1,0
KEYOPT,3,2,0
KEYOPT,3,3,0
R,31,kyL_val/3

! 转动弹簧（关于Z）
ET,4,COMBIN14
KEYOPT,4,1,0
KEYOPT,4,2,0
KEYOPT,4,3,1
R,41,kthL_val/3


! 平移弹簧（X向）
ET,5,COMBIN14
KEYOPT,5,1,0
KEYOPT,5,2,0
KEYOPT,5,3,0
R,51,kxR_val/3

! 平移弹簧（Y向）
ET,6,COMBIN14
KEYOPT,6,1,0
KEYOPT,6,2,0
KEYOPT,6,3,0
R,61,kyR_val/3

! 转动弹簧（关于Z）
ET,7,COMBIN14
KEYOPT,7,1,0
KEYOPT,7,2,0
KEYOPT,7,3,1
R,71,kthR_val/3


! 2) 生成锚固节点并连接弹簧（每侧3个：Z=0, b/2, b）
*GET, MAX_NODE, NODE, 0, NUM, MAX
nodestart = MAX_NODE+1

! 左拱脚三处锚固节点（X偏置用于X向弹簧，Y偏置用于Y向弹簧，Z偏置用于转动弹簧）
N,MAX_NODE+1,-(L_arch/2+L0_anchor),0,0
N,MAX_NODE+2,-(L_arch/2+L0_anchor),0,b_section/2
N,MAX_NODE+3,-(L_arch/2+L0_anchor),0,b_section

N,MAX_NODE+4,-(L_arch/2),(0-L0_anchor),0
N,MAX_NODE+5,-(L_arch/2),(0-L0_anchor),b_section/2
N,MAX_NODE+6,-(L_arch/2),(0-L0_anchor),b_section

N,MAX_NODE+7,-L_arch/2,0,L0_anchor
N,MAX_NODE+8,-L_arch/2,0,L0_anchor+b_section/2
N,MAX_NODE+9,-L_arch/2,0,L0_anchor+b_section

! 右拱脚三处锚固节点
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

! 3) 获取拱脚实际连接节点（Z=0, b/2, b）
!    注意：采用几何精确坐标查询
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

! 4) 连接弹簧（左侧）
*GET, MAX_NODE, NODE, 0, NUM, MAX
! 上一节已生成顺序固定，这里按相对序号连接
! X向弹簧：左
TYPE,2
REAL,21
E,nodeL_1,nodestart+0   ! 对应 (-L/2+L0_anchor,0,0)
E,nodeL_2,nodestart+1   ! 对应 (-L/2+L0_anchor,0,b/2)
E,nodeL_3,nodestart+2   ! 对应 (-L/2+L0_anchor,0,b)

! Y向弹簧：左
TYPE,3
REAL,31
E,nodeL_1,nodestart+3   ! 对应 (-L/2,-L0_anchor,0)
E,nodeL_2,nodestart+4   ! 对应 (-L/2,-L0_anchor,b/2)
E,nodeL_3,nodestart+5   ! 对应 (-L/2,-L0_anchor,b)

! 转动弹簧：左（关于Z）
TYPE,4
REAL,41
E,nodeL_1,nodestart+6   ! 对应 (-L/2,0,L0_anchor)
E,nodeL_2,nodestart+7   ! 对应 (-L/2,0,L0_anchor+b/2)
E,nodeL_3,nodestart+8    ! 对应 (-L/2,0,L0_anchor+b)

! 连接弹簧（右侧）
! X向弹簧：右
TYPE,5
REAL,51
E,nodeR_1,nodestart+9    ! 对应 (+L/2+L0_anchor,0,0)
E,nodeR_2,nodestart+10    ! 对应 (+L/2+L0_anchor,0,b/2)
E,nodeR_3,nodestart+11    ! 对应 (+L/2+L0_anchor,0,b)

! Y向弹簧：右
TYPE,6
REAL,61
E,nodeR_1,nodestart+12    ! 对应 (+L/2,-L0_anchor,0)
E,nodeR_2,nodestart+13    ! 对应 (+L/2,-L0_anchor,b/2)
E,nodeR_3,nodestart+14    ! 对应 (+L/2,-L0_anchor,b)

! 转动弹簧：右（关于Z）
TYPE,7
REAL,71
E,nodeR_1,nodestart+15    ! 对应 (+L/2,0,L0_anchor)
E,nodeR_2,nodestart+16    ! 对应 (+L/2,0,L0_anchor+b/2)
E,nodeR_3,nodestart+17      ! 对应 (+L/2,0,L0_anchor+b)

! 记录锚固节点范围（用于后续约束）
*GET, MAX_NODE, NODE, 0, NUM, MAX
nodeend = MAX_NODE

FINISH

!==================== 边界与分析流程 ====================
/SOLU
! 1) 约束锚固节点（将弹簧另一端"接地"）
!    简化处理：对全部锚固节点施加 ALL 约束（也可仅UX/ UY/ ROTZ）
FLST,2,2,1,ORDE,2
FITEM,2,nodestart
FITEM,2,-nodeend
/GO
D,P51X,ALL

! 2) 最小面外约束（仅3个位置节点UZ=0），保持平面内问题稳定
node_L1 = NODE(-L_arch/2, 0, 0)
node_L2 = NODE(-L_arch/2, 0, b_section/2)
node_L3 = NODE(-L_arch/2, 0, b_section)
node_R1 = NODE( L_arch/2, 0, 0)
node_R2 = NODE( L_arch/2, 0, b_section/2)
node_R3 = NODE( L_arch/2, 0, b_section)
node_M = NODE( 0, f_rise, b_section/2)
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

!==================== 引入初始缺陷（S/1000 缩放，与V2一致） ====================
/POST1
SET,1,1,1
NSORT,U,SUM,1
*GET,U_max,SORT,,MAX
*GET,buckle_load_factor,ACTIVE, ,SET,FREQ

/PREP7
Defect_Max = S_arch/1000
Factor = Defect_Max/U_max
UPGEOM,Factor,1,1,'BucklingAnalysis','rst'
FINISH

!==================== 非线性弧长后屈曲分析（参数与V2一致） ====================
/SOLU
ANTYPE,STATIC
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
EQSLV,SPAR
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

  NSOL,2,node_num,U,X
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
    return apdl_str


def create_commands(work_dir, num_samples=500, start_idx=0):
    """Generate simulation commands for ANSYS MAPDL."""
    number_of_cores = '4'
    commands = []

    for i in range(start_idx, start_idx + num_samples):
        # --- Parameter Sampling (V1, random uniform) ---
        while True:
            E = random.uniform(60, 210)           # GPa
            rho = random.uniform(2.7, 8)          # g/cm^3
            mu = random.uniform(0.2, 0.4)
            b_h = random.uniform(3, 20)
            S_over_h = random.uniform(65, 500)    # S/h, lambda is DERIVED
            L = random.uniform(0.5, 3)            # m
            f_L = 1.0 / random.uniform(3, 13)
            f_arch = L * f_L
            S = arc_length(L, f_arch)
            h = S / S_over_h
            b = b_h * h
            if S > b * 10:
                break

        # --- Spring Stiffness (V1) ---
        KXL = random.uniform(0.1, 10)
        KYL = random.uniform(0.1, 10)
        KZL = random.uniform(0.1, 1000)
        KXR = random.uniform(0.1, 10)
        KYR = random.uniform(0.1, 10)
        KZR = random.uniform(0.1, 1000)

        # --- Material Computation ---
        E_list, rho_list, mu_list, I0, A11, B11, D11, D11_eq, n_layers = \
            compute_materials(E, rho, mu, b, h)

        # --- Generate APDL ---
        apdl_str = generate_apdl(
            E, rho, mu, L, f_arch, h, b, S,
            KXL, KYL, KZL, KXR, KYR, KZR,
            E_list, rho_list, mu_list,
            I0, A11, B11, D11, D11_eq, n_layers,
        )

        # --- Create Directory Structure ---
        new_dir = os.path.join(work_dir, f"Compose_load_disp_{i}")
        os.makedirs(new_dir, exist_ok=True)
        os.makedirs(os.path.join(new_dir, "rundata"), exist_ok=True)
        os.makedirs(os.path.join(new_dir, "result"), exist_ok=True)

        # --- Write APDL Files ---
        with open(os.path.join(new_dir, "rundata", "run.txt"), "w") as f:
            f.write(apdl_str)
        with open(os.path.join(new_dir, "result", "run.txt"), "w") as f:
            f.write(apdl_str)

        # --- Compute derived values and write input.txt ---
        S_check = arc_length(L, f_arch)
        ix = math.sqrt(D11 / A11)
        lambda_real = S_check / ix

        with open(os.path.join(new_dir, "result", "input.txt"), "w") as f:
            f.write(str(I0) + "\n")
            f.write(str(A11) + "\n")
            f.write(str(B11) + "\n")
            f.write(str(D11) + "\n")
            f.write(str(L) + "\n")
            f.write(str(f_arch) + "\n")
            f.write(str(lambda_real) + "\n")
            f.write(str(KXL) + "\n")
            f.write(str(KYL) + "\n")
            f.write(str(KZL) + "\n")
            f.write(str(KXR) + "\n")
            f.write(str(KYR) + "\n")
            f.write(str(KZR) + "\n")

        # --- Build ANSYS Command ---
        command = [
            MAPDL_PATH, "-p", "ansys", "-smp", "-np", number_of_cores,
            "-lch", "-dir", os.path.join(new_dir, "rundata"),
            "-j", "run", "-s", "read", "-l", "en-us", "-b",
            "-i", os.path.join(new_dir, "rundata", "run.txt"),
            "-o", os.path.join(new_dir, "result", "run.out"),
        ]
        commands.append(command)
        print(f"Created command: {command}")

    return commands


def main():
    # === Configuration ===
    work_dir = r"./V1_Compose_Data"
    num_samples = 500
    start_idx = 0

    cpu_count = os.cpu_count() or 4
    max_workers = cpu_count // 4

    commands = create_commands(work_dir, num_samples, start_idx)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        # Submit initial batch
        for _ in range(min(max_workers, len(commands))):
            command = commands.pop(0)
            future = executor.submit(run_ansys_command, command)
            future.add_done_callback(task_done)
            futures.append(future)

        # Monitor and submit remaining
        while futures:
            done, not_done = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                futures.remove(future)
                if commands:
                    command = commands.pop(0)
                    new_future = executor.submit(run_ansys_command, command)
                    new_future.add_done_callback(task_done)
                    futures.append(new_future)


if __name__ == '__main__':
    main()
