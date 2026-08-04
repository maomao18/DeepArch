# -*- coding: utf-8 -*-
"""
Mesh Convergence Study — Modal Analysis / 网格收敛性分析 — 模态分析
==================================================================
Computes natural frequencies of a representative parabolic arch at
different mesh densities to verify FE mesh convergence.
在不同网格密度下计算代表性抛物线拱的自振频率，验证有限元网格收敛性。

Usage:  python Code/AnsysMeshConvergence.py
Output: results/mesh_convergence/frequencies.csv
"""

import os
import subprocess
import math

MAPDL_PATH = r"C:\Program Files\ANSYS Inc\v211\ansys\bin\winx64\MAPDL.exe"
WORK_DIR = r"./results/mesh_convergence"
os.makedirs(WORK_DIR, exist_ok=True)

# =============================================================================
# Representative arch geometry / 代表性拱几何参数
# Chosen from the middle of the dataset parameter ranges:
#   L ∈ [0.5, 3],  f/L ∈ [1/13, 1/3],  b/h ∈ [3, 20],  λ ∈ [40, 500]
# =============================================================================
L       = 1.5       # Span / 跨径 (m)
f_L     = 1/6       # Rise-to-span ratio / 矢跨比
f_arch  = L * f_L
b       = 0.3       # Width / 板宽 (m)
h       = 0.04      # Section height / 截面高度 (m)

# Material / 材料 (homogeneous — mesh convergence is geometry-driven)
E_val   = 200e9
rho_val = 7850
mu_val  = 0.3

# Boundary: fixed at both ends / 两端固支
# (Elastic supports would couple with mesh, obscuring convergence;
#  fixed BC gives clean geometry-only convergence.)

# Mesh sizes to test / 沿拱轴线方向单元数
# Factor-of-2 series centered on 160 — the mesh used in the paper / 论文采用网格
PAPER_N_ELEM = 160
ELEMENT_COUNTS = [10, 20, 40, 80, PAPER_N_ELEM, 320]
N_MODES = 10
N_POINTS = 21  # keypoints for spline curve
N_WIDTH = 4    # elements across width — same as paper / 与论文一致

# =============================================================================
# Generate APDL for each mesh density / 为每个网格密度生成APDL
# =============================================================================

def generate_apdl(n_elem: int) -> str:
    """Generate APDL script for modal analysis at given mesh density."""
    paper_mark = "  ← PAPER MESH" if n_elem == PAPER_N_ELEM else ""
    return fr'''
!============================================================
! Mesh Convergence — Modal Analysis  (n_elem = {n_elem}){paper_mark}
! Paper mesh: LESIZE=160 along curve, 4 across width
!============================================================
FINISH
/CLEAR,NOSTART
/uis,msgpop,3
/NERR,0,999999
/FILNAME,'MeshConv_{n_elem}'
/TITLE,Mesh Convergence Study — Modal Analysis

!==================== Preprocessor ====================
/PREP7

! Geometry: parabolic arch spline / 抛物线拱样条
*DO,i,1,{N_POINTS}
  x = -{L}/2 + (i-1)*({L}/({N_POINTS}-1))
  y = (4*{f_arch}/{L}**2)*( ({L}/2)**2 - x**2 )
  K,i, x, y, 0
*ENDDO

FLST,3,{N_POINTS},3
*DO,i,1,{N_POINTS}
  FITEM,3,i
*ENDDO
BSPLIN, ,P51X

K,{N_POINTS}+1,-{L}/2,0,{b}
LSTR,1,{N_POINTS}+1
ADRAG, 1, , , , , , 2

! Element type / 单元类型
ET,1,SHELL181
KEYOPT,1,1,0
KEYOPT,1,8,2

! Material / 材料 (homogeneous for convergence study)
MPTEMP,1,0
MPDATA,EX,1,,{E_val}
MPDATA,PRXY,1,,{mu_val}
MPDATA,DENS,1,,{rho_val}

! Section / 截面
SECTYPE,1,SHELL
SECDATA,{h},1,0.0,3
SECOFFSET,MID

! Meshing / 网格划分
TYPE,1
MAT,1
SECNUM,1
LESIZE,1,,,{n_elem}    ! Arch curve — varying / 拱曲线方向 — 变量
LESIZE,3,,,{n_elem}    ! Auxiliary edge
LESIZE,4,,,{N_WIDTH}   ! Width direction — fixed / 宽度方向 — 固定
LESIZE,5,,,{N_WIDTH}
ASEL,S,,,1
MSHKEY,1
AMESH,ALL

! Boundary conditions: fixed at both ends / 两端固支
! Left end / 左端
NSEL,S,LOC,X,-{L}/2
D,ALL,ALL
! Right end / 右端
NSEL,S,LOC,X,{L}/2
D,ALL,ALL
ALLSEL

FINISH

!==================== Modal Analysis ====================
/SOLU
ANTYPE,2
MODOPT,LANB,{N_MODES}
EQSLV,SPAR
MXPAND,{N_MODES}, , ,0
LUMPM,0
PSTRES,0
MODOPT,LANB,{N_MODES},0,999999, ,OFF
SOLVE
FINISH

!==================== Extract frequencies ====================
/POST1
*DIM,freqs,array,{N_MODES},1
*DO,i,1,{N_MODES},1
  SET,,,1,,,,i,
  *GET,freqs(i),ACTIVE,,SET,FREQ
*ENDDO

*CREATE,scratch,gui
out_path = STRCAT('..\freq_{n_elem}','.txt')
/OUTPUT,out_path,'txt'
*VWRITE,freqs(1,1)
%G
/OUTPUT,TERM
*END
/INPUT,scratch,gui

FINISH
'''
# (end of generate_apdl)


def run_ansys(n_elem: int) -> str:
    """Run ANSYS for one mesh density. Returns stdout."""
    run_dir = os.path.join(WORK_DIR, f"mesh_{n_elem}")
    os.makedirs(run_dir, exist_ok=True)

    apdl_str = generate_apdl(n_elem)
    with open(os.path.join(run_dir, "run.txt"), "w") as f:
        f.write(apdl_str)

    command = [
        MAPDL_PATH,
        "-p", "ansys",
        "-smp", "-np", "4",
        "-m", "1024",
        "-db", "512",
        "-lch",
        "-dir", run_dir,
        "-j", "run",
        "-s", "read",
        "-l", "en-us",
        "-b",
        "-i", os.path.join(run_dir, "run.txt"),
        "-o", os.path.join(run_dir, "run.out"),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout


def collect_results() -> dict:
    """Parse frequency output files for all mesh densities."""
    import glob
    results = {}
    for fpath in sorted(glob.glob(os.path.join(WORK_DIR, "freq_*.txt"))):
        n_elem = int(os.path.basename(fpath).replace("freq_", "").replace(".txt", ""))
        with open(fpath) as f:
            freqs = [float(line.strip()) for line in f if line.strip()]
        results[n_elem] = freqs
    return results


def main():
    print("=" * 70)
    print("Mesh Convergence Study — Modal Analysis")
    print(f"Geometry: L={L}m, f/L={f_L:.3f}, b={b}m, h={h}m")
    print(f"Element counts: {ELEMENT_COUNTS}")
    print(f"Paper mesh: n_elem={PAPER_N_ELEM} (curve) × {N_WIDTH} (width)")
    print(f"Modes extracted: {N_MODES}")
    print("=" * 70)

    # Run sequentially to avoid license contention / 顺序运行避免许可证冲突
    for i, n_elem in enumerate(ELEMENT_COUNTS):
        print(f"\n[{i+1}/{len(ELEMENT_COUNTS)}] Running n_elem={n_elem} ...")
        run_ansys(n_elem)
        print(f"  Done.")

    # Collect results / 收集结果
    print("\n" + "=" * 70)
    print("Results: Natural Frequencies (Hz)")
    print("=" * 70)

    results = collect_results()

    if not results:
        print("ERROR: No frequency files found!")
        return

    # Print table / 打印表格
    n_elem_list = sorted(results.keys())
    header = f"{'n_elem':>8} | " + " | ".join(f"Mode {i+1:>3}" for i in range(N_MODES))
    print(header)
    print("-" * len(header))
    for n in n_elem_list:
        freqs = results[n]
        marker = " ← PAPER" if n == PAPER_N_ELEM else ""
        row = f"{n:>8} | " + " | ".join(f"{f:>8.3f}" for f in freqs[:N_MODES])
        print(row + marker)

    # Relative error vs finest mesh / 相对误差 vs 最细网格
    finest = n_elem_list[-1]
    ref = results[finest]
    print(f"\nRelative error (%) vs n_elem={finest}:")
    print(f"{'n_elem':>8} | " + " | ".join(f"Mode {i+1:>3}" for i in range(N_MODES)))
    print("-" * len(header))
    for n in n_elem_list[:-1]:
        freqs = results[n]
        errors = [abs(f - r) / r * 100 for f, r in zip(freqs[:N_MODES], ref[:N_MODES])]
        row = f"{n:>8} | " + " | ".join(f"{e:>8.2f}%" for e in errors)
        print(row)

    # Save CSV / 保存 CSV
    import pandas as pd
    rows = []
    for n in n_elem_list:
        row = {"n_elem": n}
        for i in range(N_MODES):
            row[f"mode_{i+1}"] = results[n][i]
        rows.append(row)

    csv_path = os.path.join(WORK_DIR, "frequencies.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
