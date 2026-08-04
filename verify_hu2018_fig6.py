"""
验证 Hu (2018) Fig. 6(c)(d) 的工况
反对称分岔屈曲载荷 vs 矢跨比
"""

import numpy as np
import matplotlib.pyplot as plt

# 基本参数
E = 210e9  # Pa (210 GPa)
D = 0.045  # m (45 mm)
B = 0.400  # m (400 mm)

# 截面特性
A = B * D  # 截面积
I = B * D**3 / 12  # 惯性矩
ix = np.sqrt(I / A)  # 回转半径

print("=" * 60)
print("Hu (2018) Fig. 6(c)(d) Verification Cases")
print("=" * 60)
print(f"Section size: D = {D*1000:.1f} mm, B = {B*1000:.1f} mm")
print(f"Young's modulus: E = {E/1e9:.0f} GPa")
print(f"Section area: A = {A*1e6:.2f} mm^2")
print(f"Moment of inertia: I = {I*1e9:.6f} mm^4")
print(f"Radius of gyration: ix = {ix*1000:.4f} mm")
print("=" * 60)

# 矢跨比范围 (Fig. 6 中测试的范围)
rise_span_ratios = [1/12.5, 1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3]

print("\nCase list - Pin-ended arches (lambda = 2f/ix = 12):")
print("-" * 60)
print(f"{'Case':<8} {'f/L':<10} {'L (m)':<10} {'f (m)':<10} {'lambda':<10}")
print("-" * 60)

# Pin-ended: λ = 2f/ix = 12
lambda_pin = 12
results_pin = []

for i, f_L in enumerate(rise_span_ratios, 1):
    # 从 λ = 2f/ix 计算矢高 f
    f = lambda_pin * ix / 2
    # 从矢跨比计算跨度 L
    L = f / f_L

    results_pin.append({
        'case': i,
        'f_L': f_L,
        'L': L,
        'f': f,
        'lambda': lambda_pin,
        'boundary': 'pin-ended'
    })

    print(f"{i:<8} {f_L:<10.4f} {L:<10.4f} {f:<10.6f} {lambda_pin:<10.1f}")

print("\nCase list - Fixed arches (lambda = 2f/ix = 26):")
print("-" * 60)
print(f"{'Case':<8} {'f/L':<10} {'L (m)':<10} {'f (m)':<10} {'lambda':<10}")
print("-" * 60)

# Fixed: λ = 2f/ix = 26
lambda_fixed = 26
results_fixed = []

for i, f_L in enumerate(rise_span_ratios, 1):
    # 从 λ = 2f/ix 计算矢高 f
    f = lambda_fixed * ix / 2
    # 从矢跨比计算跨度 L
    L = f / f_L

    results_fixed.append({
        'case': i,
        'f_L': f_L,
        'L': L,
        'f': f,
        'lambda': lambda_fixed,
        'boundary': 'fixed'
    })

    print(f"{i:<8} {f_L:<10.4f} {L:<10.4f} {f:<10.6f} {lambda_fixed:<10.1f}")

# 保存工况到CSV
import pandas as pd

df_pin = pd.DataFrame(results_pin)
df_fixed = pd.DataFrame(results_fixed)

df_pin.to_csv('hu2018_fig6_cases_pin.csv', index=False)
df_fixed.to_csv('hu2018_fig6_cases_fixed.csv', index=False)

print("\n" + "=" * 60)
print("Cases saved to:")
print("  - hu2018_fig6_cases_pin.csv (Pin-ended)")
print("  - hu2018_fig6_cases_fixed.csv (Fixed)")
print("=" * 60)

# Key findings from the paper
print("\nKey findings from Fig. 6(c)(d):")
print("-" * 60)
print("1. When f/L >= 1/5, the proposed method deviates from FEM results")
print("2. For f/L >= 1/5 arches, linear buckling (Timoshenko) is more accurate")
print("3. Rise-to-span ratio significantly affects antisymmetric bifurcation buckling")
print("=" * 60)

# Generate input format for deep learning model verification
print("\nGenerating input format for deep learning model...")
print("-" * 60)

for result in results_pin[:4]:  # Show first 4 as examples
    f_L = result['f_L']
    L = result['L']
    f = result['f']

    print(f"\nCase: f/L = {f_L:.4f}")
    print(f"  Span L = {L:.4f} m")
    print(f"  Rise f = {f:.6f} m")
    print(f"  Boundary: {result['boundary']}")
    print(f"  Loading: Uniform vertical load")
    print(f"  Analysis: Antisymmetric bifurcation buckling")

print("\n" + "=" * 60)
print("Tip: Use your deep learning model to predict buckling loads")
print("     Then compare with Hu (2018) Fig. 6(c)(d) results")
print("=" * 60)
