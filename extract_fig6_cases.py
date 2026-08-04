"""
从 Hu (2018) Fig. 6(c)(d) 反推拱的几何参数
根据给定的 2f/ix 和 矢跨比 f/L，计算每个工况的 f, L, S
"""

import numpy as np
import pandas as pd

# ============================================================================
# 论文给定的参数
# ============================================================================
# 截面参数
D = 0.045  # m (45 mm)
B = 0.400  # m (400 mm)

# 截面特性
A = B * D
I = B * D**3 / 12
ix = np.sqrt(I / A)  # 回转半径

print("=" * 80)
print("Hu (2018) Fig. 6(c)(d) 工况几何参数反推")
print("=" * 80)
print(f"\n截面参数:")
print(f"  D = {D*1000:.1f} mm")
print(f"  B = {B*1000:.1f} mm")
print(f"  A = {A*1e6:.2f} mm^2")
print(f"  I = {I*1e9:.6f} mm^4")
print(f"  ix = {ix*1000:.4f} mm")

# ============================================================================
# Fig. 6(c): Pin-ended, λ = 2f/ix = 12
# ============================================================================
print("\n" + "=" * 80)
print("Fig. 6(c): Pin-ended arches (2f/ix = 12)")
print("=" * 80)

lambda_pin = 12
f_pin = lambda_pin * ix / 2  # 从 2f/ix = 12 反推矢高

print(f"\n给定: 2f/ix = {lambda_pin}")
print(f"计算得: f = {f_pin*1000:.3f} mm = {f_pin:.6f} m")

# Fig. 6(c) 的矢跨比范围（横轴）
f_L_ratios = [1/12.5, 1/10, 1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3]

print(f"\n{'矢跨比 f/L':<12} {'跨度 L (mm)':<15} {'矢高 f (mm)':<15} {'拱长 S (mm)':<15}")
print("-" * 80)

cases_pin = []
for f_L in f_L_ratios:
    # 从 f/L 反推跨度
    L = f_pin / f_L

    # 计算抛物线拱长 S
    # 抛物线方程: y = 4f/L² * x * (L-x)
    # 拱长积分: S = ∫√(1 + (dy/dx)²) dx
    # 近似公式: S ≈ L * (1 + 8/3 * (f/L)² - 32/5 * (f/L)⁴)
    S = L * (1 + 8/3 * f_L**2 - 32/5 * f_L**4)

    cases_pin.append({
        'boundary': 'pin-ended',
        'lambda': lambda_pin,
        'f_L': f_L,
        'f_mm': f_pin * 1000,
        'L_mm': L * 1000,
        'S_mm': S * 1000,
        'f_m': f_pin,
        'L_m': L,
        'S_m': S
    })

    print(f"{f_L:<12.4f} {L*1000:<15.2f} {f_pin*1000:<15.3f} {S*1000:<15.2f}")

# ============================================================================
# Fig. 6(d): Fixed, λ = 2f/ix = 26
# ============================================================================
print("\n" + "=" * 80)
print("Fig. 6(d): Fixed arches (2f/ix = 26)")
print("=" * 80)

lambda_fixed = 26
f_fixed = lambda_fixed * ix / 2  # 从 2f/ix = 26 反推矢高

print(f"\n给定: 2f/ix = {lambda_fixed}")
print(f"计算得: f = {f_fixed*1000:.3f} mm = {f_fixed:.6f} m")

print(f"\n{'矢跨比 f/L':<12} {'跨度 L (mm)':<15} {'矢高 f (mm)':<15} {'拱长 S (mm)':<15}")
print("-" * 80)

cases_fixed = []
for f_L in f_L_ratios:
    # 从 f/L 反推跨度
    L = f_fixed / f_L

    # 计算抛物线拱长
    S = L * (1 + 8/3 * f_L**2 - 32/5 * f_L**4)

    cases_fixed.append({
        'boundary': 'fixed',
        'lambda': lambda_fixed,
        'f_L': f_L,
        'f_mm': f_fixed * 1000,
        'L_mm': L * 1000,
        'S_mm': S * 1000,
        'f_m': f_fixed,
        'L_m': L,
        'S_m': S
    })

    print(f"{f_L:<12.4f} {L*1000:<15.2f} {f_fixed*1000:<15.3f} {S*1000:<15.2f}")

# ============================================================================
# 保存到CSV
# ============================================================================
df_pin = pd.DataFrame(cases_pin)
df_fixed = pd.DataFrame(cases_fixed)

df_pin.to_csv('fig6c_pin_cases.csv', index=False)
df_fixed.to_csv('fig6d_fixed_cases.csv', index=False)

print("\n" + "=" * 80)
print("数据已保存:")
print("  - fig6c_pin_cases.csv (Pin-ended)")
print("  - fig6d_fixed_cases.csv (Fixed)")
print("=" * 80)

# ============================================================================
# 打印建模用的参数总结
# ============================================================================
print("\n" + "=" * 80)
print("建模参数总结")
print("=" * 80)

print("\n材料参数:")
print("  E = 210 GPa = 210,000 MPa")

print("\n截面参数:")
print(f"  宽度 B = {B*1000:.1f} mm")
print(f"  深度 D = {D*1000:.1f} mm")
print(f"  截面积 A = {A*1e6:.2f} mm^2")
print(f"  惯性矩 I = {I*1e9:.6f} mm^4")

print("\n加载条件:")
print("  均布竖向荷载 (Uniform vertical load)")

print("\n屈曲模式:")
print("  反对称分岔屈曲 (Antisymmetric bifurcation buckling)")

print("\n" + "=" * 80)
print("注意事项:")
print("=" * 80)
print("1. 拱长 S 使用抛物线拱的近似公式计算")
print("2. 如需更精确的拱长，可使用数值积分")
print("3. Fig. 6(c)(d) 的纵轴是无量纲屈曲载荷，具体形式需查看论文")
print("4. 论文中提到当 f/L >= 1/5 时，线性屈曲理论更准确")
print("=" * 80)
