# -*- coding: utf-8 -*-
"""
FE Validation — Hu et al. (2018) via BEAM188
=============================================
8 cases: pin/fixed x fL12/fL9/fL8/fL7
Section 400x45mm, homogeneous steel E=210GPa.
BEAM188 matches Hu's element type.

Usage: python Code/AnsysFEValidation_Beam.py
"""
import os, math, subprocess
import numpy as np
import pandas as pd

MAPDL_PATH = r"C:\Program Files\ANSYS Inc\v211\ansys\bin\winx64\MAPDL.exe"
WORK_DIR = r"./results/fe_validation_beam"

E_val, b_sec, d_sec = 210e9, 0.400, 0.045
I_sec = b_sec * d_sec**3 / 12
A_sec = b_sec * d_sec
ix = math.sqrt(I_sec / A_sec)
EA = E_val * A_sec
EI = E_val * I_sec

CASES = [
    ("pin_fL12",   12, 1/12.5,    0,    0),
    ("pin_fL9",    12, 1/9,       0,    0),
    ("pin_fL8",    12, 1/8,       0,    0),
    ("pin_fL7",    12, 1/7,       0,    0),
    ("fixed_fL12", 26, 1/12.5, 1000, 1000),
    ("fixed_fL9",  26, 1/9,    1000, 1000),
    ("fixed_fL8",  26, 1/8,    1000, 1000),
    ("fixed_fL7",  26, 1/7,    1000, 1000),
]


def arc_length(L, f):
    a = 4 * f / L**2
    t = math.sqrt(1 + (a*L)**2)
    return L/2 * t + 1/(2*a) * math.log(a*L + t)


def build_apdl(label, lam, fL, etaZL, etaZR):
    f_rise = lam * ix / 2.0
    L_arch = f_rise / fL
    S_arch = arc_length(L_arch, f_rise)
    kx_val = EA * 10 / L_arch
    ky_val = EA * 10 / L_arch
    kthL   = EI * etaZL / L_arch
    kthR   = EI * etaZR / L_arch
    L0 = 0.1

    tpl = """!==================================================
! FE Validation BEAM188 — __LABEL__
! f/L=__FL__, L=__L_MM__mm, lam=2f/ix=__LAM__
!==================================================
FINISH
/CLEAR,NOSTART
/uis,msgpop,3
/NERR,0,999999
/FILNAME,'FEVal_beam___LABEL__'

/PREP7
! Keypoints (__NKP__ points along parabola)
*DO,i,1,__NKP__
  x = -__L__/2 + (i-1)*(__L__/(__NKP__-1))
  y = (4*__F__/__L__**2)*((__L__/2)**2 - x**2)
  K,i, x, y, 0
*ENDDO
FLST,3,__NKP__,3
*DO,i,1,__NKP__$FITEM,3,i$*ENDDO
BSPLIN,,P51X
K,__NKP__+1, 0, __F__+0.1, 0

ET,1,BEAM188
KEYOPT,1,1,1$KEYOPT,1,3,3
MP,EX,1,__E__$MP,PRXY,1,0.3$MP,DENS,1,7800
SECTYPE,1,BEAM,RECT$SECDATA,__B__,__D__,2,2$SECOFFSET,CENT
LSEL,ALL$LATT,1,,1,,__NKP__+1,,1$LESIZE,ALL,,,160$LMESH,ALL

! Spring anchors (1 per end per direction)
N,,-__L__/2-__L0__,0,0$N,,-__L__/2,-__L0__,0$N,,-__L__/2,0,__L0__
nLaX=NODE(-__L__/2-__L0__,0,0)$nLaY=NODE(-__L__/2,-__L0__,0)$nLaR=NODE(-__L__/2,0,__L0__)
N,,__L__/2+__L0__,0,0$N,,__L__/2,-__L0__,0$N,,__L__/2,0,__L0__
nRaX=NODE(__L__/2+__L0__,0,0)$nRaY=NODE(__L__/2,-__L0__,0)$nRaR=NODE(__L__/2,0,__L0__)

! Springs: X, Y, ROTZ (left + right)
ET,2,COMBIN14$KEYOPT,2,3,0$R,21,__KX__$TYPE,2$REAL,21$E,1,nLaX
ET,3,COMBIN14$KEYOPT,3,3,0$R,31,__KY__$TYPE,3$REAL,31$E,1,nLaY
ET,4,COMBIN14$KEYOPT,4,3,1$R,41,__KTHL__$TYPE,4$REAL,41$E,1,nLaR
ET,5,COMBIN14$KEYOPT,5,3,0$R,51,__KX__$TYPE,5$REAL,51$E,161,nRaX
ET,6,COMBIN14$KEYOPT,6,3,0$R,61,__KY__$TYPE,6$REAL,61$E,161,nRaY
ET,7,COMBIN14$KEYOPT,7,3,1$R,71,__KTHR__$TYPE,7$REAL,71$E,161,nRaR

FINISH

/SOLU
D,nLaX,ALL$D,nLaY,ALL$D,nLaR,ALL$D,nRaX,ALL$D,nRaY,ALL$D,nRaR,ALL
! Out-of-plane
D,1,UZ,0$D,1,ROTX,0$D,1,ROTY,0
D,161,UZ,0$D,161,ROTX,0$D,161,ROTY,0
D,81,UZ,0$D,81,ROTX,0$D,81,ROTY,0
ALLSEL
FINISH

! Step 1: Static pre-analysis with unit reference load
/SOLU
ANTYPE,STATIC
PSTRES,ON
FDELE,ALL,ALL
*DO,i,2,160$F,i,FY,-1$*ENDDO
F,1,FY,-0.5$F,161,FY,-0.5
SOLVE
FINISH

! Step 2: Eigenvalue buckling
/SOLU
ANTYPE,BUCKLE
BUCOPT,LANB,5$MXPAND,5
SOLVE
FINISH

! Get eigenvalue
/POST1
SET,1,1,1
*GET,blf,ACTIVE,,SET,FREQ
/OUTPUT,'eigenvalue','txt','../result'
*VWRITE,blf
%G
/OUTPUT,TERM

! Step 3: Imperfection (S/1000)
NSORT,U,SUM,1
*GET,U_max,SORT,,MAX
Defect_Max = __S__/1000
Factor = Defect_Max / U_max
/PREP7
UPGEOM,Factor,1,1,'FEVal_beam___LABEL__','rst'
FINISH

! Step 4: Arc-length nonlinear
/SOLU
ANTYPE,STATIC
NLGEOM,ON
OUTRES,ALL,ALL
ARCLEN,ON
NSUBST,100,1000,50
NEQIT,200
! ref = blf → total reference at critical load
ref = blf
*DO,i,2,160$F,i,FY,-ref$*ENDDO
F,1,FY,-ref/2$F,161,FY,-ref/2
TIME,ref
EQSLV,SPAR
SOLVE
FINISH

! Step 5: Post26 output
/POST26
FILE,'FEVal_beam___LABEL__','rst','.'
STORE,MERGE
NUMVAR,200

NSOL,2,81,U,Y
RFORCE,3,1,F,Y
RFORCE,4,161,F,Y
ADD,5,3,4,,total_FY

*GET,num_steps,VARI,0,NSETS
*DEL,result
*DIM,result,TABLE,num_steps,3
VGET,result(1,0),5
VGET,result(1,1),2
VGET,result(1,2),1

/OUTPUT,'load_disp','csv','../result'
*VWRITE,'total_FY_N','UY_m','TIME'
%C, %C, %C
*VWRITE,result(1,0),result(1,1),result(1,2)
%G, %G, %G
/OUTPUT,TERM
FINISH
"""

    return (tpl.replace("__LABEL__", label).replace("__FL__", str(fL))
            .replace("__LAM__", str(lam)).replace("__L__", str(L_arch))
            .replace("__L_MM__", f"{L_arch*1000:.0f}")
            .replace("__F__", str(f_rise)).replace("__S__", str(S_arch))
            .replace("__NKP__", "21").replace("__E__", str(E_val))
            .replace("__B__", str(b_sec)).replace("__D__", str(d_sec))
            .replace("__KX__", str(kx_val)).replace("__KY__", str(ky_val))
            .replace("__KTHL__", str(kthL)).replace("__KTHR__", str(kthR))
            .replace("__L0__", str(L0)))


def run_case(label, lam, fL, etaZL, etaZR):
    case_dir = os.path.join(WORK_DIR, label)
    run_dir = os.path.join(case_dir, "rundata")
    res_dir = os.path.join(case_dir, "result")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    apdl = build_apdl(label, lam, fL, etaZL, etaZR)
    with open(os.path.join(run_dir, "run.txt"), "w") as f:
        f.write(apdl)

    cmd = [MAPDL_PATH, "-p", "ansys", "-smp", "-np", "4",
           "-dir", run_dir, "-j", "run", "-s", "read", "-l", "en-us", "-b",
           "-i", os.path.join(run_dir, "run.txt"),
           "-o", os.path.join(res_dir, "run.out")]
    return subprocess.run(cmd, capture_output=True, text=True).returncode


def collect_results():
    rows = []
    for label, lam, fL, etaZL, etaZR in CASES:
        f_rise = lam * ix / 2.0
        L_arch = f_rise / fL
        Np = math.pi**2 * E_val * I_sec / (L_arch/2)**2 / 1000  # kN

        res_dir = os.path.join(WORK_DIR, label, "result")

        # Read eigenvalue
        eig_f = os.path.join(res_dir, "eigenvalue.txt")
        blf = None
        if os.path.exists(eig_f):
            with open(eig_f) as f:
                try: blf = float(f.read().strip())
                except: pass

        # Read load-disp CSV
        csv_p = os.path.join(res_dir, "load_disp.csv")
        q_cr, n_pts = None, 0
        if os.path.exists(csv_p):
            df = pd.read_csv(csv_p)
            n_pts = len(df)
            if n_pts > 5:
                fy = df["total_FY_N"].abs() / 1000
                # Peak of force = buckling load (stiffness→0)
                q_cr = fy.max()

        ratio = q_cr / Np if q_cr else None
        rows.append({"case": label, "L_mm": L_arch*1000, "f_L": fL,
                     "blf": blf, "Np_kN": Np,
                     "Q_cr_kN": q_cr, "N_cr_N_p": ratio, "n_pts": n_pts})

    return pd.DataFrame(rows)


# ═══════════════ MAIN ═══════════════
if __name__ == "__main__":
    os.makedirs(WORK_DIR, exist_ok=True)
    print("=" * 65)
    print("FE Validation — Hu et al. (2018) via BEAM188")
    print(f"Section: {b_sec*1000:.0f}x{d_sec*1000:.0f}mm, E={E_val/1e9:.0f}GPa")
    print("=" * 65)

    for i, (label, lam, fL, etaZL, etaZR) in enumerate(CASES):
        f_rise = lam * ix / 2.0
        L_arch = f_rise / fL
        print(f"[{i+1}/8] {label}  L={L_arch*1000:.0f}mm  f/L={fL:.4f}", end=" ", flush=True)
        rc = run_case(label, lam, fL, etaZL, etaZR)
        print("Done" if rc == 0 else f"ANSYS exit {rc}")

    print("\n" + "=" * 65)
    print("Results")
    print("=" * 65)
    df = collect_results()
    print(df.to_string(index=False))
    df.to_csv(os.path.join(WORK_DIR, "validation_results_beam.csv"), index=False)
    print(f"\nSaved: {WORK_DIR}/validation_results_beam.csv")
