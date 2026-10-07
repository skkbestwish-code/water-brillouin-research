# 瑞利–布里渊散射频谱的 FP 成像与反演

固定起始参数 FSR=15 GHz、谱精细度=30。第一版可运行低压高斯极限、高压三洛伦兹极限和明确标注的三高斯算法测试。不是 Tenti S6 实现，尚未对应已标定的实际装置。

运行：

```powershell
python -m pip install -r requirements.txt
python run_simulation.py
python -m unittest discover -s tests -v
```

本机系统 Python 已具备所需科学计算库，可直接运行。输出位于 `output/rb_fp/`，先读其中的 `验证报告.md` 和 `overview.png`。

调整假设：编辑输出中的 `parameters.json` 的副本，然后运行 `python run_simulation.py --config 参数文件.json --output output/新实验`。每次运行会覆盖所选输出目录中的同名结果。FSR、精细度、激光线宽/纵模、焦距、失谐、PSF、像素尺寸、气体温度与散射角可配置。第一版验证压力固定为0.02及20 bar；任意压力的动力学区间需要独立验证的Tenti S6输入。

模块：`simulation/physics.py` 定义谱与 Airy 响应；`simulation/imaging.py` 定义二维PSF、像素积分和径向平均；`simulation/inverse.py` 提供非负正则化恢复及有谱型先验的前向拟合；`run_simulation.py` 执行实验、敏感性和数值验证。

非参数反演目标为入射FP的散射光频谱（已包括激光谱卷积）。激光和光学参数必须标定后，才能把恢复线宽解释为气体本征性质。保留文献原始文件与 `sources/` 为只读。
