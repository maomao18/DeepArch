# Hu (2018) Fig. 6(c)(d) 验证指南

## 概述

本指南说明如何使用训练好的深度学习模型验证 Hu (2018) 论文中 Fig. 6(c)(d) 的反对称分岔屈曲载荷预测。

## 论文信息

**论文标题**: In-plane non-linear elastic stability of parabolic arches with different rise-to-span ratios

**作者**: Chang-Fu Hu, Yong-Lin Pi, Wei Gao, Li Li

**发表**: Thin-Walled Structures 129 (2018) 74–84

**研究内容**: Fig. 6(c)(d) 展示了不同矢跨比下抛物线拱的反对称分岔屈曲载荷

## 工况参数

### 几何参数（来自论文）

- **截面尺寸**: D = 45 mm (深度), B = 400 mm (宽度)
- **弹性模量**: E = 210 GPa
- **截面积**: A = 18,000 mm²
- **惯性矩**: I = 3,037.5 mm⁴
- **回转半径**: ix = 12.99 mm

### 修正长细比

- **Pin-ended**: λ = 2f/ix = 12
  - 矢高 f = 77.942 mm
  - 跨度范围: 234 ~ 974 mm

- **Fixed**: λ = 2f/ix = 26
  - 矢高 f = 168.875 mm
  - 跨度范围: 507 ~ 2,111 mm

### 矢跨比范围

Fig. 6(c)(d) 测试的矢跨比：
- f/L = 1/12.5 (0.08)
- f/L = 1/9 (0.111)
- f/L = 1/8 (0.125)
- f/L = 1/7 (0.143)
- f/L = 1/6 (0.167)
- f/L = 1/5 (0.20)
- f/L = 1/4 (0.25)
- f/L = 1/3 (0.333)

## 已生成的文件

1. **hu2018_fig6_cases_pin.csv** - Pin-ended工况列表
2. **hu2018_fig6_cases_fixed.csv** - Fixed工况列表
3. **hu2018_theoretical_pin.csv** - Pin-ended理论值
4. **hu2018_theoretical_fixed.csv** - Fixed理论值
5. **hu2018_fig6_template.png** - 对比图模板

## 验证步骤

### 步骤1: 准备模型输入

对于每个工况，需要准备以下输入：

```python
input_data = {
    'span': L,              # 跨度 (m)
    'rise': f,              # 矢高 (m)
    'rise_span_ratio': f_L, # 矢跨比
    'section_area': A,      # 截面积 (m²)
    'moment_inertia': I,    # 惯性矩 (m⁴)
    'youngs_modulus': E,    # 弹性模量 (Pa)
    'boundary': boundary,   # 'pin-ended' 或 'fixed'
    'load_type': 'uniform_vertical'  # 均布竖向荷载
}
```

### 步骤2: 使用模型预测

推荐使用的模型：
- **Full_2**: 1.6M参数，训练较快，性能良好
- **v2F**: 10.2M参数，更大容量，可能更准确

模型路径：
```
./models/Full_2/buckling_predictor_best.pth
./models/v2F/buckling_predictor_best.pth
```

### 步骤3: 提取论文数据

从 Hu (2018) Fig. 6(c)(d) 中提取以下数据点：
- X轴: 矢跨比 f/L
- Y轴: 无量纲屈曲载荷或临界荷载

**注意**: 论文中可能使用了无量纲化参数，需要确认：
- 无量纲载荷形式
- 归一化方式

### 步骤4: 对比分析

将以下三组数据绘制在同一图中：
1. **FEM结果** (从论文Fig. 6提取)
2. **理论解** (Timoshenko线性屈曲或Hu提出的方法)
3. **深度学习模型预测**

## 论文中的关键发现

### Fig. 6(c)(d) 的主要结论：

1. **当 f/L ≥ 1/5 时**：
   - Hu提出的方法与FEM结果存在偏差
   - 线性屈曲分析(Timoshenko)更接近FEM结果
   - 原因：高矢跨比拱主要承受轴压，弯曲作用较小

2. **当 f/L < 1/5 时**：
   - Hu提出的方法与FEM结果吻合良好
   - 比传统近似方法（Bradford等）更准确

3. **矢跨比的影响**：
   - 矢跨比对反对称分岔屈曲载荷有显著影响
   - 传统方法（不考虑曲线弧微分项）在f/L > 1/12.5时误差较大

## 验证指标

计算以下误差指标：

1. **相对误差**:
   ```
   Error = |Predicted - Reference| / Reference × 100%
   ```

2. **均方根误差 (RMSE)**:
   ```
   RMSE = sqrt(mean((Predicted - Reference)²))
   ```

3. **R²决定系数**:
   ```
   R² = 1 - SS_res / SS_tot
   ```

## 预期结果

根据论文和模型训练结果，预期：

1. **f/L < 1/5 区域**:
   - 模型预测应与FEM结果较好吻合
   - 相对误差 < 10%

2. **f/L ≥ 1/5 区域**:
   - 可能存在较大偏差（如论文所述）
   - 需要使用线性屈曲理论进行对比

## 参考公式

### Timoshenko线性屈曲公式

反对称分岔屈曲临界荷载（近似）：

**Pin-ended**:
```
q_cr = π² × E × I / (L² × f)
```

**Fixed**:
```
q_cr = 4π² × E × I / (L² × f)
```

### 无量纲化参数

论文中可能使用的无量纲参数：
```
q̄ = q × L² / (E × I)
N̄ = N × L² / (E × I)
```

## 使用脚本

### 生成工况列表
```bash
python verify_hu2018_fig6.py
```

### 准备验证数据
```bash
python prepare_hu2018_verification.py
```

### 运行模型预测（需要完善）
```bash
python run_verification_hu2018.py
```

## 注意事项

1. **加载方式**: 论文使用均布竖向荷载，确保模型输入正确
2. **边界条件**: Pin-ended vs Fixed，两种边界条件的结果差异很大
3. **屈曲模式**: Fig. 6(c)(d)专门针对反对称分岔屈曲
4. **数据归一化**: 确保模型输入/输出的归一化与训练时一致

## 下一步工作

1. ✅ 准备工况参数列表
2. ✅ 生成理论对比值
3. ⏳ 实现模型预测功能
4. ⏳ 从论文中提取FEM数据点
5. ⏳ 创建对比图表
6. ⏳ 分析误差来源

## 联系信息

如有问题，请查阅：
- 论文原文: Hu et al. (2018) Thin-Walled Structures
- 模型训练日志: ./logs/Full_2/training_metrics.csv
- 项目根目录: D:\tanghao\deeparch_review

---

**创建日期**: 2024
**最后更新**: 根据Hu (2018) Fig. 6(c)(d)
