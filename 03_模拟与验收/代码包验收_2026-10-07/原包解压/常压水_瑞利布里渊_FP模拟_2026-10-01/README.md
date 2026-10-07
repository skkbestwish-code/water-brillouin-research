# 常压液态水瑞利–布里渊散射的 FP 成像与反演

当前目标是**液态水中散射 → 空气中放置的 FP → 干涉环图 → 入射谱恢复**。FP 腔内介质仍未知，默认空气腔只是参考情景。旧氮气版保留为历史，不能用其数值预测水实验。

先读 `说明_常压水模型方法与结果.md`，再看 `output/water_rb_fp/验证结果.md`、`water_spectra_and_recovery.png` 和 `water_interference_images.png`。

```powershell
python -m pip install -r requirements_water.txt
python -m unittest discover -s tests -v
python run_water_simulation.py
```

`.water_deps` 包含 iapws 1.5.5 及许可证，运行时无需联网取得水物性。本机已有 NumPy、SciPy、Matplotlib。代码使用 Python 3.10 以上语法，本次在 Python 3.14.4 运行。

修改参数时，复制 `output/water_rb_fp/parameters.json`，然后运行：

```powershell
python run_water_simulation.py --config 新参数.json --output output/新参数结果
```

重复运行会覆盖所选输出目录同名结果，请用新输出目录保留比较结果。默认纯水 20 °C、常压、水内 90°；另算 180°、2° 收光角标准差和 10% 额外弹性光。输出包含展开谱、折叠谱、噪声图、径向曲线、非参数反演、条件拟合、参数扫描和验证。

已确认水、常压、He–Ne、标称 FSR 15 GHz、精细度 30、FP 外部空气。其他是参考假设；未知水成分、腔介质及光路已明确标注。没有把独立标定设为运行前提，也未计算实际散射功率或曝光需求。

历史入口 `run_simulation.py`、`output/rb_fp` 和旧说明是气体测试，水版不调用 Gas 或 Tenti 模型。原论文和 sources 保持只读。
