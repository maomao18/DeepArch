import pandas as pd

print("=" * 100)
print("Hu (2018) Fig. 6(c)(d) - 拱的几何参数")
print("=" * 100)

# 截面参数
print("\n共同参数 (All cases):")
print("  截面: B=400mm x D=45mm")
print("  材料: E=210 GPa")
print("  加载: 均布竖向荷载")
print("  屈曲模式: 反对称分岔屈曲")

# 读取数据
df_pin = pd.read_csv('fig6c_pin_cases.csv')
df_fixed = pd.read_csv('fig6d_fixed_cases.csv')

# Pin-ended
print("\n" + "=" * 100)
print("Fig. 6(c): PIN-ENDED ARCHES (2f/ix = 12, f = 77.942 mm 固定)")
print("=" * 100)
print(f"{'序号':<6} {'f/L':<10} {'跨度L(mm)':<12} {'矢高f(mm)':<12} {'拱长S(mm)':<12}")
print("-" * 100)
for i, row in df_pin.iterrows():
    print(f"{i+1:<6} {row['f_L']:<10.4f} {row['L_mm']:<12.2f} {row['f_mm']:<12.3f} {row['S_mm']:<12.2f}")

# Fixed
print("\n" + "=" * 100)
print("Fig. 6(d): FIXED ARCHES (2f/ix = 26, f = 168.875 mm 固定)")
print("=" * 100)
print(f"{'序号':<6} {'f/L':<10} {'跨度L(mm)':<12} {'矢高f(mm)':<12} {'拱长S(mm)':<12}")
print("-" * 100)
for i, row in df_fixed.iterrows():
    print(f"{i+1:<6} {row['f_L']:<10.4f} {row['L_mm']:<12.2f} {row['f_mm']:<12.3f} {row['S_mm']:<12.2f}")

print("\n" + "=" * 100)
print("关键点:")
print("=" * 100)
print("1. Pin-ended: 矢高 f = 77.942 mm 保持不变，改变跨度 L")
print("2. Fixed: 矢高 f = 168.875 mm 保持不变，改变跨度 L")
print("3. 每个矢跨比 f/L 对应一个工况")
print("4. 总共 9 个 Pin-ended 工况 + 9 个 Fixed 工况 = 18 个工况")
print()
print("文件已保存:")
print("  - fig6c_pin_cases.csv")
print("  - fig6d_fixed_cases.csv")
print("  - FIG6_GEOMETRY_SUMMARY.txt")
print("=" * 100)
