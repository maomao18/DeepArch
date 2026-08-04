"""
V2_FGM.py — V2 LHS Data Generation with FGM 10-Layer Material
=============================================================

Generates ANSYS APDL input files for parabolic arch buckling analysis
using Latin Hypercube Sampling (LHS) with the V2 parameter space.

Material: Functionally Graded Material (FGM), 10 layers.
  - Cos-distribution profiles (types 1 and 2) modulated by gradient e0.
  - Stiffness integration over 10 layers with iterative h/b solve.

V2 design:
  - Span L = 1.0 FIXED (not sampled).
  - Lambda is an INDEPENDENT parameter → iterate h, b to match lambda_target.
  - Rotational springs only (COMBIN14 about Z). No X/Y translational springs.
  - eta >= 0.99 → fixed-end condition (ROTZ=0) instead of springs.
  - APDL template: Arbpres macro, eigenvalue buckling + arc-length post-buckling.

Execution: concurrent.futures.ProcessPoolExecutor with MAPDL.

Default work_dir: ./V2_FGM_Data
"""

import concurrent.futures
import subprocess
import os
import numpy as np
import math
from scipy.stats import qmc

# =============================================================================
# Configuration
# =============================================================================
MAPDL_PATH = r"C:\Program Files\ANSYS Inc\v211\ansys\bin\winx64\MAPDL.exe"
DEFAULT_WORK_DIR = r"./V2_FGM_Data"
DEFAULT_NUM_SAMPLES = 5000
NUM_CORES = "4"
SEED = 42
N_LAYERS = 10
MAX_ITER = 20
TOLERANCE = 0.01
FIXED_THRESHOLD = 0.99

# =============================================================================
# ANSYS Execution Helpers
# =============================================================================


def run_ansys_command(command):
    """Execute an ANSYS MAPDL command."""
    try:
        result = subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return result.stdout, result.stderr, command
    except subprocess.CalledProcessError as e:
        return e.stdout, e.stderr, command


def task_done(future):
    """Callback on task completion: print output and clean up temp files."""
    try:
        stdout, stderr, command = future.result()
    except Exception as exc:
        print(f"Task exception: {exc}")
        return
    print("=" * 50)
    print("Task completed.")
    if stdout:
        print("Standard Output snippet:\n", stdout[-500:])
    if stderr:
        print("Standard Error snippet:\n", stderr[-500:])

    work_dir = None
    for i, arg in enumerate(command):
        if arg == "-dir":
            work_dir = command[i + 1]
            break
    if work_dir and os.path.isdir(work_dir):
        print(f"Cleaning up directory: {work_dir}")
        for filename in os.listdir(work_dir):
            filepath = os.path.join(work_dir, filename)
            if filename != "run.txt" and os.path.isfile(filepath):
                try:
                    os.remove(filepath)
                except OSError:
                    pass


# =============================================================================
# Stiffness & Geometry Helpers
# =============================================================================


def eta_to_rotational_stiffness(eta, DI, L):
    """
    Map [0,1] dimensionless stiffness coefficient to actual rotational stiffness.

    kth = DI/L * eta/(1-eta)
      eta=0   → kth=0      (perfect pin)
      eta=0.5 → kth=DI/L   (moderate)
      eta→1   → kth→inf    (fixed)
    """
    k_base = DI / L
    kth = k_base * eta / (1.0 - eta + 1e-12)
    return kth


def arc_length_parabolic(L, f_L):
    """Parabolic arch arc length given span L and rise-to-span ratio f_L."""
    k = 4.0 * f_L
    sqrt_term = math.sqrt(1.0 + k * k)
    log_term = math.log(k + sqrt_term)
    C = 0.5 * sqrt_term + (1.0 / (2.0 * k)) * log_term
    return L * C


# =============================================================================
# FGM Material Factors (10-layer, cos-distribution, types 1 & 2)
# =============================================================================
# Relative thickness positions (centre of each layer, symmetric about mid-plane)
Z_REL = np.array(
    [-0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45]
)

TYPE1_BASE = np.cos(np.pi * Z_REL)                # cos(pi * z_rel)
TYPE2_BASE = np.cos(np.pi * Z_REL / 2.0 + np.pi / 4.0)  # cos(pi*z_rel/2 + pi/4)


def compute_fgm_factors(choice, e0):
    """
    Compute per-layer E and rho factors for FGM 10-layer material.

    choice=1: factorE = 1 - cos(pi*z_rel) * e0
    choice=2: factorE = 1 - cos(pi*z_rel/2 + pi/4) * e0
    factorRho derived from factorE via empirical power-law relation.
    """
    if choice == 1:
        factorE = 1.0 - TYPE1_BASE * e0
        factorRho = np.power(1.0 - TYPE1_BASE * e0, 0.4347826087) * 1.121 - 0.121
    else:
        factorE = 1.0 - TYPE2_BASE * e0
        factorRho = np.power(1.0 - TYPE2_BASE * e0, 0.4347826087) * 1.121 - 0.121
    return factorE, factorRho


# =============================================================================
# Sample Generation: FGM 10-layer with stiffness iteration
# =============================================================================


def generate_single_sample(sample_params):
    """
    Process one LHS sample into geometry, material, and stiffness parameters.

    LHS dimensions (9):
      0: E          [60, 210] GPa
      1: rho        [2.7, 8]  g/cm^3
      2: b_h        [5, 10]   width-to-height ratio
      3: lambda     [150, 250] slenderness (independent)
      4: f_L        [1/12, 1/2] rise-to-span ratio
      5: e0         [0.1, 0.3] FGM gradient coefficient
      6: choice_cont [0, 1]   discrete → 1 if <0.5 else 2
      7: etaRotL    [0, 1]    left rotational stiffness coefficient
      8: etaRotR    [0, 1]    right rotational stiffness coefficient

    Returns: (is_valid: bool, params: dict)
    """
    (E, rho, b_h, lambda_target, f_L, e0, choice_cont, etaRotL, etaRotR) = sample_params

    choice = 1 if choice_cont < 0.5 else 2
    L = 1.0
    f_arch = L * f_L
    S = arc_length_parabolic(L, f_L)
    ix_target = S / lambda_target

    # FGM material factors (independent of h)
    factorE, factorRho = compute_fgm_factors(choice, e0)
    mu = 0.3
    mu_list = np.full(N_LAYERS, mu)

    # Initial estimate (homogeneous approximation: ix = h / sqrt(12))
    h = math.sqrt(12.0) * ix_target
    b = b_h * h

    # Iterative solve for h, b
    lambda_real = 0.0
    D11_eq = 0.0
    A11_val = 0.0
    I0_val = 0.0
    B11_val = 0.0
    D11_val = 0.0
    ix_eq = 0.0
    E_list = None
    rho_list = None

    for _ in range(MAX_ITER):
        E_list = factorE * E * 1e9
        rho_list = factorRho * rho * 1e3

        h_layer = h / N_LAYERS
        I0_val = 0.0
        A11_val = 0.0
        B11_val = 0.0
        D11_val = 0.0

        for j in range(N_LAYERS):
            z_layer = -h / 2.0 + j * h_layer + h_layer / 2.0
            I0_val += rho_list[j] * b * h_layer
            A11_val += E_list[j] * b * h_layer
            B11_val += E_list[j] * b * h_layer * z_layer
            D11_val += E_list[j] * b * (
                h_layer * z_layer * z_layer + h_layer ** 3 / 12.0
            )

        D11_eq = D11_val - (B11_val * B11_val) / A11_val
        ix_eq = math.sqrt(D11_eq / A11_val)
        lambda_real = S / ix_eq

        error = abs(lambda_real - lambda_target) / lambda_target
        if error < TOLERANCE:
            break

        h = h * (ix_target / ix_eq)
        b = b_h * h

    # Rotational spring stiffness
    kthL = eta_to_rotational_stiffness(etaRotL, D11_eq, L)
    kthR = eta_to_rotational_stiffness(etaRotR, D11_eq, L)
    is_fixed_L = etaRotL >= FIXED_THRESHOLD
    is_fixed_R = etaRotR >= FIXED_THRESHOLD

    valid = abs(lambda_real - lambda_target) / lambda_target < TOLERANCE

    params = {
        "E": E,
        "rho": rho,
        "mu": mu,
        "b": b,
        "h": h,
        "b_h": b_h,
        "lambda_target": lambda_target,
        "lambda_real": lambda_real,
        "f_L": f_L,
        "L": L,
        "f_arch": f_arch,
        "S": S,
        "choice": choice,
        "e0": e0,
        "E_list": list(E_list),
        "rho_list": list(rho_list),
        "mu_list": list(mu_list),
        "I0": I0_val,
        "A11": A11_val,
        "B11": B11_val,
        "D11": D11_val,
        "D11_eq": D11_eq,
        "ix_eq": ix_eq,
        "etaRotL": etaRotL,
        "etaRotR": etaRotR,
        "kthL": kthL,
        "kthR": kthR,
        "is_fixed_L": is_fixed_L,
        "is_fixed_R": is_fixed_R,
        "n_layers": N_LAYERS,
    }
    return valid, params


# =============================================================================
# APDL Template Generation
# =============================================================================


def _make_spring_block(side, eta_val, is_fixed, kth):
    """
    Generate boundary-condition code blocks for one arch support.

    side: "L" or "R"
    Returns dict with keys: et, anchor, connect, rotz_constraint, anchor_constraint
    """
    if is_fixed:
        return {
            "et": f"! {side} fixed: no rotational spring element",
            "anchor": f"! {side} fixed: no anchor nodes",
            "connect": f"! {side} fixed: no spring connection",
            "rotz_constraint": f"""
! {side} fixed-end (ROTZ=0)
D,node{side}_1,ROTZ,0
D,node{side}_2,ROTZ,0
D,node{side}_3,ROTZ,0
""",
            "anchor_constraint": f"! {side} fixed: no anchor nodes to constrain",
        }
    else:
        # Elastic rotational spring
        return {
            "et": f"""
! {side} rotational spring
ET,{2 if side == 'L' else 3},COMBIN14
KEYOPT,{2 if side == 'L' else 3},1,0
KEYOPT,{2 if side == 'L' else 3},2,0
KEYOPT,{2 if side == 'L' else 3},3,1  ! rotational dof
R,{2 if side == 'L' else 3},{kth:.6e}/3  ! stiffness split over 3 nodes
""",
            "anchor": f"""
! {side} arch rotational spring anchor nodes (Z offset)
N,nodestart+{0 if side == 'L' else 3},{'-' if side == 'L' else ''}L_arch/2,0,L0_anchor
N,nodestart+{1 if side == 'L' else 4},{'-' if side == 'L' else ''}L_arch/2,0,L0_anchor+b_section/2
N,nodestart+{2 if side == 'L' else 5},{'-' if side == 'L' else ''}L_arch/2,0,L0_anchor+b_section
""",
            "connect": f"""
! Connect rotational springs ({side})
TYPE,{2 if side == 'L' else 3}
REAL,{2 if side == 'L' else 3}
E,node{side}_1,nodestart+{0 if side == 'L' else 3}
E,node{side}_2,nodestart+{1 if side == 'L' else 4}
E,node{side}_3,nodestart+{2 if side == 'L' else 5}
""",
            "rotz_constraint": f"! {side} elastic: preserve ROTZ dof",
            "anchor_constraint": f"""
! Constrain {side} anchor nodes
D,nodestart+{0 if side == 'L' else 3},ALL,0
D,nodestart+{1 if side == 'L' else 4},ALL,0
D,nodestart+{2 if side == 'L' else 5},ALL,0
""",
        }


def _make_material_block(n_layers, E_list, rho_list, mu_list):
    """Generate APDL material-array and MP definition block for n_layers."""

    E_str = ",".join(f"{v:.6e}" for v in E_list)
    rho_str = ",".join(f"{v:.6e}" for v in rho_list)
    mu_str = ",".join(f"{v:.6f}" for v in mu_list)

    lines = []
    lines.append(f"*dim,Elastic_modulus,array,{n_layers}")
    lines.append(f"Elastic_modulus(1)={E_str}")
    lines.append(f"*dim,Density,array,{n_layers}")
    lines.append(f"Density(1)={rho_str}")
    lines.append(f"*dim,mu_val,array,{n_layers}")
    lines.append(f"mu_val(1)={mu_str}")
    lines.append("")

    lines.append(f"*DO,i,1,{n_layers}")
    lines.append("MPTEMP,,,,,,,,")
    lines.append("MPTEMP,1,0")
    lines.append("MPDATA,EX,i,,Elastic_modulus(i)")
    lines.append("MPDATA,PRXY,i,,mu_val(i)")
    lines.append("MPTEMP,,,,,,,,")
    lines.append("MPTEMP,1,0")
    lines.append("MPDATA,DENS,i,,Density(i)")
    lines.append("*ENDDO")

    return "\n".join(lines)


def _make_secdata_block(n_layers):
    """Generate SECDATA loop for shell section definition."""
    lines = []
    lines.append("SECTYPE,1,SHELL")
    for i in range(1, n_layers + 1):
        lines.append(f"SECDATA,h_section/{n_layers},{i},0.0,3")
    lines.append("secoffset,MID")
    lines.append("seccontrol,,,, , , ,")
    return "\n".join(lines)


def generate_apdl_str(params):
    """Generate complete APDL input file from the parameter dictionary."""
    n_layers = params["n_layers"]

    # Build per-side spring blocks
    L_blk = _make_spring_block("L", params["etaRotL"], params["is_fixed_L"], params["kthL"])
    R_blk = _make_spring_block("R", params["etaRotR"], params["is_fixed_R"], params["kthR"])

    material_block = _make_material_block(
        n_layers, params["E_list"], params["rho_list"], params["mu_list"]
    )
    secdata_block = _make_secdata_block(n_layers)

    return rf"""\
!##############################################################
! V2_FGM — Parabolic Arch Buckling (FGM 10-layer, LHS sampling)
! V2 design: L=1.0 fixed, lambda independent, rotational springs only.
! Material: 10-layer FGM with cos-distribution, types 1/2, gradient e0.
! Author: Hao Tang
!##############################################################

!============== Uniform-pressure macro (Arbpres) ===============
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
!================ Macro end ================

!==================== Initialisation ====================
FINISH
/CLEAR,NOSTART
/uis,msgpop,3
KEYW,PR_SGVOF,1
/NERR,0,999999
/FILNAME,'BucklingAnalysis'
/TITLE,Parabolic Arch Buckling Analysis — V2 FGM

!==================== User Parameters ====================
E_val   = {params['E']}e9
rho_val = {params['rho']}e3
mu_val  = {params['mu']}

L_arch   = {params['L']}
f_rise   = {params['f_arch']:.8f}
h_section= {params['h']:.8f}
b_section= {params['b']:.8f}
S_arch   = {params['S']:.8f}

n_points          = 21
n_elements        = 500
imp_factor        = 0.005
n_buckling_modes  = 5

kthL_val = {params['kthL']:.6e}
kthR_val = {params['kthR']:.6e}
L0_anchor = 0.1

!==================== Preprocessor ====================
/PREP7

!--- Parabolic arch geometry (XY-plane, extruded along Z) ---
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

!--- Element type ---
ET,1,SHELL181
KEYOPT,1,1,0
KEYOPT,1,8,2

!--- Material arrays ---
{material_block}

!--- Shell section ---
{secdata_block}

!--- Meshing ---
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

!==================== Boundary conditions ====================
{L_blk['et']}
{R_blk['et']}

*GET, MAX_NODE, NODE, 0, NUM, MAX
nodestart = MAX_NODE+1

{L_blk['anchor']}
{R_blk['anchor']}

! --- Retrieve arch-foot nodes ---
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

{L_blk['connect']}
{R_blk['connect']}

*GET, MAX_NODE, NODE, 0, NUM, MAX
nodeend = MAX_NODE

FINISH

!==================== Solution ====================
/SOLU

{L_blk['anchor_constraint']}
{R_blk['anchor_constraint']}

! Pinned constraints (UX=UY=0)
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

{L_blk['rotz_constraint']}
{R_blk['rotz_constraint']}

! Minimal out-of-plane restraint
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

!==================== Modal analysis ====================
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

! Export natural frequencies
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

!==================== Static pre-analysis (stress stiffening) ====================
/SOLU
ANTYPE,STATIC
PSTRES,ON
FDELE,ALL,ALL
ASEL,S, , , 1
Arbpres,1,'Y',1,'FY',-1
SOLVE
FINISH

!==================== Eigenvalue buckling ====================
/SOLU
ANTYPE,BUCKLE
BUCOPT,LANB,n_buckling_modes
MXPAND,n_buckling_modes
SOLVE
FINISH

!==================== Geometric imperfection ====================
/POST1
ALLSEL,ALL

SET,1,1,1

*GET,buckle_load_factor,ACTIVE,,SET,FREQ

NSORT,U,SUM,1
*GET,U_max,SORT,,MAX

*STATUS,U_max
*STATUS,buckle_load_factor

Defect_Max = S_arch/1000
Factor     = Defect_Max/U_max

*STATUS,Factor
*STATUS,Defect_Max

FINISH

/PREP7
UPGEOM,Factor,1,1,'BucklingAnalysis','rst'
FINISH

!==================== Nonlinear arc-length post-buckling ====================
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

!==================== Post26 output ====================
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
"""


# =============================================================================
# Command Generation: LHS sampling + APDL file creation
# =============================================================================


def create_commands(work_dir, num_samples=DEFAULT_NUM_SAMPLES):
    """
    1. Generate LHS samples (9-dim).
    2. For each sample, compute FGM stiffness iteratively.
    3. Write APDL input files and build MAPDL command lists.
    """
    bounds = np.array(
        [
            [60, 210],       # 0  E (GPa)
            [2.7, 8],        # 1  rho (g/cm^3)
            [5, 10],         # 2  b_h
            [150, 250],      # 3  lambda_target
            [1.0 / 12, 0.5], # 4  f_L
            [0.1, 0.3],      # 5  e0
            [0, 1],          # 6  choice_cont
            [0, 1],          # 7  etaRotL
            [0, 1],          # 8  etaRotR
        ]
    )
    n_dims = bounds.shape[0]

    print(f"Generating {num_samples} LHS samples ({n_dims}-dim, seed={SEED})...")
    sampler = qmc.LatinHypercube(d=n_dims, seed=SEED)
    sample_unit = sampler.random(n=num_samples)
    samples = qmc.scale(sample_unit, bounds[:, 0], bounds[:, 1])
    print("LHS samples generated.")

    commands = []
    valid_count = 0

    for i in range(num_samples):
        sample_idx = i + 1
        valid, params = generate_single_sample(samples[i])

        if not valid:
            continue

        valid_count += 1
        if valid_count % 100 == 0 or valid_count <= 10:
            print(
                f"Sample {sample_idx}/{num_samples} valid, "
                f"total valid: {valid_count}"
            )

        # Generate APDL
        apdl_str = generate_apdl_str(params)

        # Create directory structure
        sample_dir = os.path.join(work_dir, f"FGM_load_disp_{valid_count}")
        rundata_dir = os.path.join(sample_dir, "rundata")
        result_dir = os.path.join(sample_dir, "result")
        os.makedirs(rundata_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        # Write APDL input
        with open(os.path.join(rundata_dir, "run.txt"), "w") as f:
            f.write(apdl_str)
        with open(os.path.join(result_dir, "run.txt"), "w") as f:
            f.write(apdl_str)

        # Write input parameters
        with open(os.path.join(result_dir, "input.txt"), "w") as f:
            f.write(f"{params['I0']}\n")
            f.write(f"{params['A11']}\n")
            f.write(f"{params['B11']}\n")
            f.write(f"{params['D11_eq']}\n")
            f.write(f"{params['f_L']}\n")
            f.write(f"{params['lambda_real']}\n")
            f.write(f"{params['b_h']}\n")
            f.write(f"{params['etaRotL']}\n")
            f.write(f"{params['etaRotR']}\n")

        # MAPDL command
        command = [
            MAPDL_PATH,
            "-p", "ansys",
            "-smp",
            "-np", NUM_CORES,
            "-lch",
            "-dir", rundata_dir,
            "-j", "run",
            "-s", "read",
            "-l", "en-us",
            "-b",
            "-i", os.path.join(rundata_dir, "run.txt"),
            "-o", os.path.join(result_dir, "run.out"),
        ]
        commands.append(command)

    print(f"\nSample processing done. Total: {num_samples}, valid: {valid_count}")
    return commands


# =============================================================================
# Main
# =============================================================================


def main():
    cpu_count = os.cpu_count() or 4
    max_workers = max(1, cpu_count // 4)
    print(f"CPU cores: {cpu_count}, parallel workers: {max_workers}")

    commands = create_commands(DEFAULT_WORK_DIR, num_samples=DEFAULT_NUM_SAMPLES)

    if not commands:
        print("No valid commands generated. Exiting.")
        return

    print(f"\nSubmitting {len(commands)} tasks...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        # Initial batch
        for _ in range(min(max_workers, len(commands))):
            command = commands.pop(0)
            future = executor.submit(run_ansys_command, command)
            future.add_done_callback(task_done)
            futures.append(future)

        # Feed remaining as tasks complete
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

    print("\nAll tasks submitted.")


if __name__ == "__main__":
    main()
