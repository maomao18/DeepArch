# -*- coding: utf-8 -*-
"""
FE Validation — Hu et al. (2018) / 有限元验证
==============================================
Uses EXACT same APDL template as production data (AnsysBatch.py).
Only substitutes parameter values for validation cases.

Paper: Hu C-F, Pi Y-L, Gao W, Li L. Thin-Walled Structures 2018;129:74-84.

Usage:  python Code/AnsysFEValidation.py
Output: results/fe_validation/
"""

import os, math, subprocess, glob, shutil
import numpy as np, pandas as pd

MAPDL_PATH = r"C:\Program Files\ANSYS Inc\v211\ansys\bin\winx64\MAPDL.exe"
WORK_DIR = r"./results/fe_validation"
os.makedirs(WORK_DIR, exist_ok=True)

# =============================================================================
# Paper validation parameters (SI units)
# =============================================================================
E_val   = 210e9      # Young's modulus (Pa)
rho_val = 7800       # Density (kg/m3)
mu_val  = 0.3        # Poisson's ratio
B_sec   = 0.400      # Section width (m)
D_sec   = 0.045      # Section depth (m)

I_x = B_sec * D_sec**3 / 12.0  # Moment of inertia
Area = B_sec * D_sec            # Cross-sectional area
i_x = math.sqrt(I_x / Area)     # Radius of gyration

# Homogeneous → 5 identical layers
E_list  = [E_val] * 5
rho_list = [rho_val] * 5
mu_list  = [mu_val] * 5

# Equivalent sectional stiffnesses
A11 = E_val * Area
D11 = E_val * I_x

# Validation cases
CASES = [
    # (label, KXL, KYL, KZL, KXR, KYR, KZR, f/L, lam=2f/ix, desc)
    # Pinned: translational stiff (10), rotational free (0)
    ("pin_fL12",   10, 10, 0,    10, 10, 0,    1/12.5, 12, "Pinned f/L=1/12.5"),
    ("pin_fL9",    10, 10, 0,    10, 10, 0,    1/9,    12, "Pinned f/L=1/9"),
    ("pin_fL8",    10, 10, 0,    10, 10, 0,    1/8,    12, "Pinned f/L=1/8"),
    ("pin_fL7",    10, 10, 0,    10, 10, 0,    1/7,    12, "Pinned f/L=1/7"),
    # Fixed: translational stiff (10), rotational stiff (1000)
    ("fixed_fL12", 10, 10, 1000, 10, 10, 1000, 1/12.5, 26, "Fixed f/L=1/12.5"),
    ("fixed_fL9",  10, 10, 1000, 10, 10, 1000, 1/9,    26, "Fixed f/L=1/9"),
    ("fixed_fL8",  10, 10, 1000, 10, 10, 1000, 1/8,    26, "Fixed f/L=1/8"),
    ("fixed_fL7",  10, 10, 1000, 10, 10, 1000, 1/7,    26, "Fixed f/L=1/7"),
]


def arc_length(L: float, f: float) -> float:
    """Arc length of parabola."""
    a = 4.0 * f / (L * L)
    t = math.sqrt(1.0 + (a * L) ** 2)
    return L / 2.0 * t + 1.0 / (2.0 * a) * math.log(a * L + t)


def generate_apdl(label, KXL, KYL, KZL, KXR, KYR, KZR, f_L, lam) -> str:
    """Generate APDL using EXACT production data template."""
    f_rise = lam * i_x / 2.0
    L_arch = f_rise / f_L
    h_section = D_sec   # shell thickness
    b_section = B_sec   # width

    return fr'''
!##############################################################
! 拱结构屈曲分析程序 — 有限元验证 Hu et al. (2018)
! {label}: f/L={f_L:.4f}, lam=2f/ix={lam}, L={L_arch*1000:.1f}mm
! 作者：Hao Tang  |  版本：FE Validation
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
/FILNAME,'FEVal_{label}'
/TITLE,FE Validation — {label}

!==================== 用户参数区 ====================
!-------- 材料参数 (homogeneous steel, SI units) --------
E_val   = {E_val}
rho_val = {rho_val}
mu_val  = {mu_val}

!-------- 几何参数（抛物线拱） --------
L_arch   = {L_arch}       ! 跨径 (m)
f_rise   = {f_rise}       ! 矢高 (m)
h_section= {h_section}    ! 壳厚 (m)
b_section= {b_section}    ! 板宽 (m)

!-------- 分析控制参数 --------
n_points          = 21
imp_factor        = 0.005
n_buckling_modes  = 5

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

!-------- 单元与材料 (homogeneous: 5 identical layers) --------
ET,1,SHELL181
KEYOPT,1,1,0
KEYOPT,1,8,2

*dim,Elastic_modulus,array,5
Elastic_modulus(1)={E_list[0]},{E_list[1]},{E_list[2]},{E_list[3]},{E_list[4]}
*dim,Density,array,5
Density(1)={rho_list[0]},{rho_list[1]},{rho_list[2]},{rho_list[3]},{rho_list[4]}
*dim,mu_val_arr,array,5
mu_val_arr(1)={mu_list[0]},{mu_list[1]},{mu_list[2]},{mu_list[3]},{mu_list[4]}

*DO,i,1,5
MPTEMP,,,,,,,,
MPTEMP,1,0
MPDATA,EX,i,,Elastic_modulus(i)
MPDATA,PRXY,i,,mu_val_arr(i)
MPTEMP,,,,,,,,
MPTEMP,1,0
MPDATA,DENS,i,,Density(i)
*ENDDO

!-------- 壳截面定义 --------
SECTYPE,1,SHELL
*DO,i,1,5
SECDATA,h_section/5,i,0.0,3
*ENDDO
secoffset,MID
seccontrol,,,, , , ,

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
DI  = {D11}
EA  = {A11}
kxL_val  = EA*etaXL/L_arch
kyL_val  = EA*etaYL/L_arch
kthL_val = DI*etaRotL/L_arch

kxR_val  = EA*etaXR/L_arch
kyR_val  = EA*etaYR/L_arch
kthR_val = DI*etaRotR/L_arch

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

! 平移弹簧（X向-右）
ET,5,COMBIN14
KEYOPT,5,1,0
KEYOPT,5,2,0
KEYOPT,5,3,0
R,51,kxR_val/3

! 平移弹簧（Y向-右）
ET,6,COMBIN14
KEYOPT,6,1,0
KEYOPT,6,2,0
KEYOPT,6,3,0
R,61,kyR_val/3

! 转动弹簧（关于Z-右）
ET,7,COMBIN14
KEYOPT,7,1,0
KEYOPT,7,2,0
KEYOPT,7,3,1
R,71,kthR_val/3

! 生成锚固节点并连接弹簧
*GET, MAX_NODE, NODE, 0, NUM, MAX
nodestart = MAX_NODE+1

N,MAX_NODE+1,-(L_arch/2+L0_anchor),0,0
N,MAX_NODE+2,-(L_arch/2+L0_anchor),0,b_section/2
N,MAX_NODE+3,-(L_arch/2+L0_anchor),0,b_section
N,MAX_NODE+4,-(L_arch/2),(0-L0_anchor),0
N,MAX_NODE+5,-(L_arch/2),(0-L0_anchor),b_section/2
N,MAX_NODE+6,-(L_arch/2),(0-L0_anchor),b_section
N,MAX_NODE+7,-L_arch/2,0,L0_anchor
N,MAX_NODE+8,-L_arch/2,0,L0_anchor+b_section/2
N,MAX_NODE+9,-L_arch/2,0,L0_anchor+b_section

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

Xl=-L_arch/2
Yl=0
Xr=L_arch/2
Yr=0
Z0=0
Zm=b_section/2
Z1=b_section
nodeL_1=NODE(Xl,Yl,Z0)
nodeL_2=NODE(Xl,Yl,Zm)
nodeL_3=NODE(Xl,Yl,Z1)
nodeR_1=NODE(Xr,Yr,Z0)
nodeR_2=NODE(Xr,Yr,Zm)
nodeR_3=NODE(Xr,Yr,Z1)

! 连接弹簧（左）
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

! 连接弹簧（右）
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

*GET, MAX_NODE, NODE, 0, NUM, MAX
nodeend = MAX_NODE
FINISH

!==================== 边界与分析流程 ====================
/SOLU
FLST,2,2,1,ORDE,2
FITEM,2,nodestart
FITEM,2,-nodeend
/GO
D,P51X,ALL

! 面外约束
node_L1=NODE(-L_arch/2,0,0)
node_L2=NODE(-L_arch/2,0,b_section/2)
node_L3=NODE(-L_arch/2,0,b_section)
node_R1=NODE(L_arch/2,0,0)
node_R2=NODE(L_arch/2,0,b_section/2)
node_R3=NODE(L_arch/2,0,b_section)
node_M=NODE(0,f_rise,b_section/2)
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

!==================== 静力预分析 ====================
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

!==================== 引入初始缺陷 ====================
/POST1
SET,FIRST
*GET,buckle_load_factor,ACTIVE, ,SET,FREQ
/PREP7
UPGEOM,imp_factor,1,1,'FEVal_{label}','rst'
FINISH

!==================== 非线性弧长后屈曲分析 ====================
/SOLU
NCNV,2,0,0,0,0
ANTYPE,STATIC
LNSRCH,ON
NROPT,UNSYM
NLGEOM,ON
OUTRES,ALL,ALL
ARCLEN,ON
ARCLEN,1,25,1e-9,1
NSUBST,500,1000,50
NEQIT,1000
CNVTOL,F,0.05
FDELE,ALL,ALL
ASEL,S, , , 1
Arbpres,1,'Y',1.1*buckle_load_factor,'FY',-1
TIME,1.1*buckle_load_factor
ONFAIL,CONTINUE
EQSLV,SPAR
SOLVE
FINISH

!==================== 后处理：输出荷载-位移曲线 ====================
/post26
RESET
FILE,'FEVal_{label}','rst','.'
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
FINISH
'''
# (end of generate_apdl)


def run_case(label, KXL, KYL, KZL, KXR, KYR, KZR, f_L, lam):
    """Run one ANSYS validation case (matching production data structure)."""
    case_dir = os.path.join(WORK_DIR, label)
    run_dir = os.path.join(case_dir, "rundata")
    res_dir = os.path.join(case_dir, "result")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    apdl_str = generate_apdl(label, KXL, KYL, KZL, KXR, KYR, KZR, f_L, lam)
    with open(os.path.join(run_dir, "run.txt"), "w") as f:
        f.write(apdl_str)

    command = [
        MAPDL_PATH, "-p", "ansys", "-smp", "-np", "4",
        "-m", "1024", "-db", "512", "-lch",
        "-dir", run_dir, "-j", "run",
        "-s", "read", "-l", "en-us", "-b",
        "-i", os.path.join(run_dir, "run.txt"),
        "-o", os.path.join(res_dir, "run.out"),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    WARNING: ANSYS exit code {result.returncode}")


def collect_results():
    """Parse frequency and buckling results from output files."""
    rows = []
    for case_dir in sorted(glob.glob(os.path.join(WORK_DIR, "*_fL*"))):
        label = os.path.basename(case_dir)
        # Read frequency output
        freq_file = os.path.join(case_dir, "result", "freq.txt")
        freq1 = None
        if os.path.exists(freq_file):
            with open(freq_file) as f:
                lines = f.readlines()
                if lines:
                    freq1 = float(lines[0].strip())

        # Read load-disp_1 (mid-span) to extract buckling load
        ld_file = os.path.join(case_dir, "result", "load-disp_1.csv")
        q_cr = None
        if os.path.exists(ld_file):
            try:
                df = pd.read_csv(ld_file)
                # Load is from TIME column × buckle_load_factor reference
                # The actual load during arc-length is scaled by buckle_load_factor
                # Find peak of TIME (which tracks the load multiplier)
                if 'TIME' in df.columns:
                    q_cr = df['TIME'].max()
            except:
                pass

        # Compute Np for this case
        f_L_val = float(label.split("fL")[-1])
        f_L_ratio = 1.0 / f_L_val
        lam = 12 if label.startswith("pin") else 26
        f = lam * i_x / 2.0
        L = f / f_L_ratio
        Np = math.pi**2 * E_val * I_x / (L / 2.0)**2

        rows.append({
            "case": label,
            "L_m": L,
            "f_m": f,
            "f_L": f_L_ratio,
            "lam": lam,
            "freq_Hz": freq1,
            "Q_cr_TIMEmax": q_cr,
            "Np_Npm": Np,
        })

    return pd.DataFrame(rows)


def main():
    print("=" * 65)
    print("FE Validation — Hu et al. (2018)")
    print(f"APDL: EXACT production template (Arbpres + COMBIN14 + arc-length)")
    print(f"Section: B={B_sec*1000:.0f}mm, D={D_sec*1000:.0f}mm, E={E_val/1e9:.0f}GPa")
    print(f"Pinned: eta_KX=KY=10, eta_KZ=0   Fixed: eta_KX=KY=10, eta_KZ=1000")
    print(f"Np = pi^2 * E * Ix / (L/2)^2")
    print("=" * 65)

    for i, (label, KXL, KYL, KZL, KXR, KYR, KZR, f_L, lam, desc) in enumerate(CASES):
        f_rise = lam * i_x / 2.0
        L_arch = f_rise / f_L
        kx = A11 * KXL / L_arch
        kth = D11 * KZL / L_arch
        print(f"\n[{i+1}/{len(CASES)}] {label}  ({desc})")
        print(f"    L={L_arch*1000:.0f}mm  f={f_rise*1000:.1f}mm  "
              f"kx={kx/1e6:.0f}MN/m  kth={kth/1e3:.0f}kNm/rad")
        run_case(label, KXL, KYL, KZL, KXR, KYR, KZR, f_L, lam)
        print(f"    Done.")

    print("\n" + "=" * 65)
    print("Results")
    print("=" * 65)
    df = collect_results()
    if df.empty:
        print("No results found. Check ANSYS output for errors.")
        return
    print(df[["case", "L_m", "f_L", "freq_Hz", "Q_cr_TIMEmax", "Np_Npm"]].to_string(index=False))

    csv_path = os.path.join(WORK_DIR, "validation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print("Compare Q_cr_TIMEmax / Np_Npm with Hu et al. (2018) Fig. 6.")


if __name__ == "__main__":
    main()
