"""
Hu (2018) Fig. 6(c)(d) 验证工况准备
准备好待验证的工况列表，供后续使用训练好的模型进行预测
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_theoretical_buckling_load(f, L, E, I, A, boundary='pin-ended'):
    """
    计算理论屈曲载荷（简化公式，仅供参考）

    根据Timoshenko的线性屈曲理论
    """
    # 对于抛物线拱的反对称分岔屈曲
    # 这是一个简化公式，实际应使用Hu (2018)中的完整公式

    if boundary == 'pin-ended':
        # Pin-ended拱的近似公式
        k = np.pi**2 * E * I / L**2
        q_cr = k / (L * f)  # 临界均布荷载
    else:  # fixed
        # Fixed拱的近似公式
        k = 4 * np.pi**2 * E * I / L**2
        q_cr = k / (L * f)

    return q_cr


def main():
    print("=" * 70)
    print("Hu (2018) Fig. 6(c)(d) 验证工况汇总")
    print("=" * 70)

    # 读取工况
    cases_pin = pd.read_csv('hu2018_fig6_cases_pin.csv')
    cases_fixed = pd.read_csv('hu2018_fig6_cases_fixed.csv')

    # 材料和截面参数
    E = 210e9  # Pa
    D = 0.045  # m
    B = 0.400  # m
    A = B * D
    I = B * D**3 / 12

    print(f"\n截面参数:")
    print(f"  D = {D*1000:.1f} mm, B = {B*1000:.1f} mm")
    print(f"  A = {A*1e6:.2f} mm^2, I = {I*1e9:.6f} mm^4")
    print(f"  E = {E/1e9:.0f} GPa")

    # 选择Fig. 6(c)(d)中关注的矢跨比
    test_ratios = [0.08, 1/9, 0.125, 1/7, 1/6, 0.2, 0.25, 1/3]

    print("\n" + "=" * 70)
    print("Pin-ended arches (lambda = 12)")
    print("=" * 70)
    print(f"{'f/L':<10} {'L(m)':<10} {'f(m)':<10} {'理论q_cr(N/m)':<15}")
    print("-" * 70)

    results_pin = []
    for idx, row in cases_pin.iterrows():
        f_L = row['f_L']
        L = row['L']
        f = row['f']

        # 计算理论屈曲载荷
        q_cr = calculate_theoretical_buckling_load(f, L, E, I, A, 'pin-ended')

        results_pin.append({
            'f_L': f_L,
            'L': L,
            'f': f,
            'theoretical_q_cr': q_cr,
            'boundary': 'pin-ended'
        })

        print(f"{f_L:<10.4f} {L:<10.4f} {f:<10.6f} {q_cr:<15.2e}")

    print("\n" + "=" * 70)
    print("Fixed arches (lambda = 26)")
    print("=" * 70)
    print(f"{'f/L':<10} {'L(m)':<10} {'f(m)':<10} {'理论q_cr(N/m)':<15}")
    print("-" * 70)

    results_fixed = []
    for idx, row in cases_fixed.iterrows():
        f_L = row['f_L']
        L = row['L']
        f = row['f']

        # 计算理论屈曲载荷
        q_cr = calculate_theoretical_buckling_load(f, L, E, I, A, 'fixed')

        results_fixed.append({
            'f_L': f_L,
            'L': L,
            'f': f,
            'theoretical_q_cr': q_cr,
            'boundary': 'fixed'
        })

        print(f"{f_L:<10.4f} {L:<10.4f} {f:<10.6f} {q_cr:<15.2e}")

    # 保存结果
    df_pin = pd.DataFrame(results_pin)
    df_fixed = pd.DataFrame(results_fixed)

    df_pin.to_csv('hu2018_theoretical_pin.csv', index=False)
    df_fixed.to_csv('hu2018_theoretical_fixed.csv', index=False)

    print("\n" + "=" * 70)
    print("理论结果已保存到:")
    print("  - hu2018_theoretical_pin.csv")
    print("  - hu2018_theoretical_fixed.csv")
    print("=" * 70)

    # 创建可视化对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pin-ended
    ax1.plot(df_pin['f_L'], df_pin['theoretical_q_cr']/1e3, 'b-o', label='Theoretical (Timoshenko)')
    ax1.set_xlabel('Rise-to-span ratio (f/L)', fontsize=12)
    ax1.set_ylabel('Critical load q_cr (kN/m)', fontsize=12)
    ax1.set_title('Pin-ended arches (λ=12)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0.05, 0.35)

    # Fixed
    ax2.plot(df_fixed['f_L'], df_fixed['theoretical_q_cr']/1e3, 'r-s', label='Theoretical (Timoshenko)')
    ax2.set_xlabel('Rise-to-span ratio (f/L)', fontsize=12)
    ax2.set_ylabel('Critical load q_cr (kN/m)', fontsize=12)
    ax2.set_title('Fixed arches (λ=26)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0.05, 0.35)

    plt.tight_layout()
    plt.savefig('hu2018_fig6_template.png', dpi=150, bbox_inches='tight')
    print("\n对比图模板已保存: hu2018_fig6_template.png")

    print("\n" + "=" * 70)
    print("下一步操作:")
    print("=" * 70)
    print("1. 使用您的深度学习模型对这些工况进行预测")
    print("2. 从Hu (2018) Fig. 6(c)(d)中提取实验/FEM数据点")
    print("3. 将模型预测、理论值和文献数据绘制在同一图中对比")
    print()
    print("建议使用的模型:")
    print("  - Full_2 (较小模型，1.6M参数)")
    print("  - v2F (较大模型，10.2M参数)")
    print("=" * 70)

    # 打印关键信息供参考
    print("\n关键参数总结:")
    print("-" * 70)
    print(f"Pin-ended: λ = 2f/ix = 12")
    print(f"  矢高 f = {cases_pin['f'].iloc[0]:.6f} m = {cases_pin['f'].iloc[0]*1000:.3f} mm")
    print(f"  跨度范围: {cases_pin['L'].min():.4f} ~ {cases_pin['L'].max():.4f} m")
    print(f"\nFixed: λ = 2f/ix = 26")
    print(f"  矢高 f = {cases_fixed['f'].iloc[0]:.6f} m = {cases_fixed['f'].iloc[0]*1000:.3f} mm")
    print(f"  跨度范围: {cases_fixed['L'].min():.4f} ~ {cases_fixed['L'].max():.4f} m")
    print("=" * 70)


if __name__ == '__main__':
    main()
