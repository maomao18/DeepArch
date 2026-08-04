# Data Generation Scripts — Design Spec

**Date:** 2026-07-29
**Status:** Approved

## Overview

Six standalone scripts for arch buckling data generation, organized in two versions × three material types.

## Script Inventory

| Script | Sampling | λ | Material | Layers |
|---|---|---|---|---|
| `Code/V1_FGM.py` | random uniform | derived (S/ix) | FGM cos-distribution + e0 | 10 |
| `Code/V1_Compose.py` | random uniform | derived (S/ix) | random ±15% per layer | 5 |
| `Code/V1_Homogeneous.py` | random uniform | derived (S/ix) | uniform material | 1 |
| `Code/V2_FGM.py` | LHS | independent param | FGM cos-distribution + e0 | 10 |
| `Code/V2_Compose.py` | LHS | independent param | random ±15% per layer | 5 |
| `Code/V2_Homogeneous.py` | LHS | independent param | uniform material | 1 |

## Shared Components

- APDL template: SHELL181, 160×4 mesh, COMBIN14 springs, Arbpres load macro, arc-length analysis — identical to `AnsysBatch.py` v2.3
- Output structure: `{work_dir}/FGM_load_disp_{i}/` with `rundata/run.txt` and `result/` directories
- ANSYS execution: MAPDL v211, 4 cores
- Post-run cleanup: delete all files except run.txt from rundata

## V1 Parameter Space (matching current training dataset)

| Param | Sampling | Range |
|---|---|---|
| E | random.uniform | [60, 210] GPa |
| rho | random.uniform | [2.7, 8] g/cm³ |
| mu | random.uniform | [0.2, 0.4] |
| b_h | random.uniform | [3, 20] |
| S/h | random.uniform | [65, 500] |
| L | random.uniform | [0.5, 3] m |
| f/L | 1/random.uniform(3, 13) | [0.077, 0.333] |
| KXL, KYL | random.triangular(0.1, 10, 10) | [0.1, 10] |
| KZL, KZR | random.triangular(0.1, 1000, 1000) | [0.1, 1000] |
| KXR, KYR | random.triangular(0.1, 10, 10) | [0.1, 10] |

- Material factor: random ±15% per layer (compose) OR FGM cos distribution (FGM) OR uniform (homogeneous)
- h = S / (S/h), b = b_h × h
- λ = S / ix (derived from section stiffness)
- Validation: S > 10 × b

## V2 Parameter Space (LHS, independent λ)

| Param | Sampling | Range |
|---|---|---|
| E | LHS | [60, 210] GPa |
| rho | LHS | [2.7, 8] g/cm³ |
| b_h | LHS | [5, 10] (narrower than V1) |
| λ | LHS | [150, 250] (independent) |
| f/L | LHS | [1/12, 1/2] |
| e0 | LHS | [0.1, 0.3] |
| choice | LHS (binary) | {1, 2} (FGM distribution type) |
| etaRotL | LHS | [0, 1] |
| etaRotR | LHS | [0, 1] |

- L = 1 fixed
- h, b computed iteratively to match λ_target via FGM stiffness iteration
- Rotational springs: kth = DI/L × eta/(1-eta), fixed (UZ=ROTZ=0) when eta ≥ 0.99
- X/Y hinged (no translational springs)

## Material Models

### FGM (10 layers)
```
z_rel = [-0.45, -0.35, ..., 0.45]  # 10 points across thickness
type1 = cos(π × z_rel)
type2 = cos(π × z_rel/2 + π/4)
E_factor = 1 - type × e0
rho_factor = (1 - type × e0)^0.435 × 1.121 - 0.121
```

### Compose (5 layers)
```
for each layer:
  factor = random.uniform(0.85, 1.15)
  E_i = factor × E × 1e9
  rho_i = factor × rho × 1e3
  mu_i = factor × mu
```

### Homogeneous (1 layer)
```
E_1 = E × 1e9, rho_1 = rho × 1e3, mu_1 = mu
```

## File Structure

```
Code/
  V1_FGM.py           # V1, FGM 10-layer
  V1_Compose.py        # V1, compose 5-layer ±15%
  V1_Homogeneous.py    # V1, homogeneous 1-layer
  V2_FGM.py           # V2 LHS, FGM 10-layer
  V2_Compose.py        # V2 LHS, compose 5-layer ±15%
  V2_Homogeneous.py    # V2 LHS, homogeneous 1-layer
```

Each script is self-contained (no shared imports beyond stdlib + numpy/scipy). Copy-paste the APDL template within each file.

## Output

Same format as current `AnsysBatch.py`:
- `{work_dir}/FGM_load_disp_{i}/rundata/run.txt`
- `{work_dir}/FGM_load_disp_{i}/result/run.txt`
- `{work_dir}/FGM_load_disp_{i}/result/input.txt` (9 params for V2, all params for V1)
- ANSYS output in `result/` directory
