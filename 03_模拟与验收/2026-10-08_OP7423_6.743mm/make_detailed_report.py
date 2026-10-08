"""Detailed Chinese report from checked numerical artifacts and explicit scope notes."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def build(cases,audit,deep,dest):
    m=read(cases/'metrics.json');a=read(audit/'audit_metrics.json');d=read(deep/'deep_review.json')
    dest.mkdir(parents=True,exist_ok=True)
    instrument=m['configuration']['instrument']
    from run_water_simulation import DEFAULTS
    for key in ['water','instrument','camera','laser_fwhm_ghz','laser_modes','gain_electrons','background_electrons','read_noise_electrons','fit_spacing_bounds_ghz']:
        if m['configuration'][key]!=DEFAULTS[key]:
            raise ValueError('Detailed report is for the frozen default scenario; incompatible field: '+key)
    if instrument!=d['baseline']['instrument'] or instrument!=a['configuration']['provenance']['instrument']:
        raise ValueError('Cannot combine reports from different instruments.')
    names={'water_90deg':'20℃、90°','water_180deg':'20℃、180°','water_angular_2deg':'角度标准差2°','water_elastic_10percent':'10%额外弹性光'}
    lines=['# 6.743 mm 石英标准具：模型、误差验收与使用汇报','',
      '验收日期：2026-10-08。对象：本目录最终代码。性质：代码审查与合成数据验证，尚无本装置实测谱。本包统一采用最终修订代码、配置与说明。',
      '', '## 一、先说明结论',
      '', '**本轮发现并修复了实际缺陷；前向数值误差很小，但默认配置没有全部达到此前设定的10⁻⁴前向对照阈值；任意整谱还原仍不整体通过10%目标。带明确水谱范围先验的频移拟合，在中、高计数情景下通过1 MHz随机RMSE目标。**',
      '', '最重要的新发现：180°算例存在约6.241 GHz和8.749 GHz两组近等价解。仅看局部标准差和残差，可能误以为测量唯一且精确。修订后会检查竞争解；本轮默认显式采用B=0.05–7 GHz的物理范围先验，开放到10 GHz的诊断则标记歧义。',
      '', '同样重要的修正：上一轮“数值检查通过”只覆盖频率步长等已列项目。新增积分窗口和空间离散检查后，该结论必须收窄，不能继续写成所有正向误差均低于10⁻⁴。',
      '', '## 二、检查范围、独立性和没有做过的事',
      '', '逐项审查 water_simulation/physics.py、inverse.py、etalon.py，共享的相机成像与非参数反演模块，两个原运行入口、报告生成器和一键入口；复现缺陷后新增回归测试。原有测试继续保留；当前37项测试通过。',
      '', '实际执行：四个完整算例；20/25/30℃无噪声恢复；三档信号尺度各50次条件拟合；频率积分三档；积分窗口三档；空间采样三档；角度求积三档；反演频率网格三档；三类场景各5次非参数反演；级次多解；环心偏移；FSR、焦距、精细度、群折射率、背景和激光等失配诊断。',
      '', '验证层次：①原函数内更换网格检验离散误差；②不同积分路径和完整Sellmeier光程表达式作数值交叉核对；③合成观测反演检验估计器；④厂家图纸、材料模型提供参数依据。仍缺⑤独立实验响应/水谱验证。前四项不自动构成第五项，也没有做全局穷举或严格误差上界证明。',
      '', '## 三、发现的问题、修复及残留限制',
      '', '| 编号/优先级 | 发现与影响 | 本轮处理 |',
      '|---|---|---|',
      '| R01 / P1 | 原多初值只到6.5GHz；180°在8.749GHz存在同样小的残差，原quality_pass均为真，局部约0.255MHz标准误不能表示全局歧义 | 增加8.8GHz起点和竞争解诊断；显式可配置频移范围；不再把竞争解当作唯一精确解 |',
      '| R02 / P1 | 雅可比秩亏时伪逆把零空间贡献置零，可能输出“标准误=0”，零核反例已复现 | 秩亏/病态时不输出伪精度，标准误为null；有竞争解时全局条件标准误也为null，局部标准误另列 |',
      '| R03 / P2 | 反演接受非均匀频率轴，却使用固定步长差分和积分；可产生单位/平滑错误 | 检查均匀递增网格、有限数据、正噪声尺度和正则化参数；暴露正则化边界与未达到噪声准则的告警 |',
      '| R04 / P2 | NaN流速等非法输入能进入计算，原物性验证不完整 | 增加水参数、纵模、激光宽度和积分参数的有限值检查；不是任意输入的全面形式化验证 |',
      '| R05 / P1，结论范围 | 只细化频率步长，没有验证±150GHz截断与空间网格是否满足同一阈值 | 新增并保留未通过项，修正汇报结论；默认快速配置未被偷偷调参以掩盖失败 |',
      '| R06 / 方法限制 | 非参数350×600算子欠定、强平滑，25MHz附近采样会漏掉窄峰细节；提高采样点数也不保证降低恢复误差 | 增加300/600/1200点和噪声重复；明确这是病态问题，未冒称已修复唯一性 |',
      '| R07 / 数据接口限制 | 当前CLI主要生成和拟合合成数据，没有完整的实测图像导入、环心拟合、平场/暗场校正和实测响应核导入流程 | 在使用说明中明确边界和所需数据；没有把模拟CSV冒充已支持的实验接口 |',
      '', '修复位置：竞争解、范围和协方差在 water_simulation/inverse.py；有限参数校验在 water_simulation/physics.py；网格和正则化诊断在 simulation/inverse.py；预设先验由 run_water_simulation.py 与 run_etalon_audit.py 显式传入。tests/test_review.py保存四个复现反例。',
      '', '竞争解判据：两个已收敛候选B相差>50MHz且总Δχ²≤9时告警。这是工程筛查，不是严格的全局置信区间或遍历证明。只检查已找到的极小值。没有找到竞争解，也不能证明全局唯一。',
      '', '默认0.05–7GHz范围适用于此次632.8nm常压纯水场景；同一理论物性在0.01℃与50℃的180°频移约5.909与6.475GHz，7GHz保留了余量。这是独立物理范围先验，不是看模拟误差后选取解。如果激光波长、介质或目标变更，必须重设范围并重新做混叠检查。',
      '', '## 四、前向模型和误差定义',
      '', '链路：水谱S(ν) → 标准具透过T(r,ν) → 照明权重 → 二维高斯PSF → 像素面积积分 → 环形均值y。数值形式为y=KS；K包含Δν积分权重。半径r不是普通频率卷积坐标，FFT只作近轴映射后的周期卷积对照。',
      '', '本文“前向相对L2”是||y_test−y_reference||₂/||y_reference||₂；“整谱相对L2”是单FSR内||S_rec−S_true||₂/||S_true||₂；“频移误差”是B_fit−B_true，以MHz计。三者不可互换；环曲线拟合很好不表示谱还原正确。标准化残差均方使用残差除以逐环带sigma；它不是自动扣除自由度的reduced χ²。',
      '', '图纸FSR=0.5cm⁻¹=14.9896229GHz；石英n=1.45701793、ng=1.47539225。精细度30和峰值透过率50%是厂家下限的边界情景，非实际检测值。反射率94%不能直接等同于有效精细度；未把厚度均匀性上限再次随意叠加为一个已知展宽。',
      '', '水在20℃，石英材料模型也取20℃；改变水温不代表标准具跟随水温变化。真实标准具热漂移尚未建模。图纸厚度和FSR有公差，代码以标称FSR为输入，按ng反推的是有效光学厚度，不强迫三项独立精确自洽。',
      '', '### 4.1 原有频率检查',
      '', '| 检查 | 数值 | 解释 |','|---|---:|---|',
      f"| 核心频率步长2.5→1.25MHz | {a['grid'][1]['relative_l2_to_previous']:.3e} | 尾网格也同步减半 |",
      f"| 1.25→0.625MHz | {a['grid'][2]['relative_l2_to_previous']:.3e} | 仅检验给定积分窗 |",
      f"| 独立标量积分最大绝对差 | {a['quadrature_max_abs_error']:.3e} | 共用谱和响应函数，检验积分路径 |",
      f"| 一阶色散响应vs完整光程相对L2 | {a['linear_dispersion_vs_full_sellmeier_relative_l2']:.3e} | 频偏±12GHz、半径0–1.4mm；仍忽略镀膜色散 |",
      f"| FFT近似vs完整相机链相对L2 | {m['numerical_checks']['fft_vs_unfolded_profile_relative_l2']:.4%} | FFT不能替代完整K |",
      '', '### 4.2 新增积分窗口检查',
      '', '| 频率窗半宽/GHz | 窗内谱面积 | 剖面对±150GHz的相对L2 | 用默认网格拟合的B偏差/MHz |',
      '|---|---:|---:|---:|']
    for row in d['window']:lines.append(f"| {row['limit_ghz']} | {row['area']:.9f} | {row['profile_l2_vs_baseline']:.6%} | {row['B_bias_mhz']:+.7f} |")
    lines+=['', '±150GHz缺失约0.0452%的谱面积。窗口扩展引起的曲线变化超过10⁻⁴；默认前向总精度不能只按频率步长误差宣称。上述更宽窗仍不是无限窗口真值，误差列是数值对照，不是严格上界。B偏差小的原因是该尾部变化在此情景主要表现为很小的宽背景/幅度变化；不能推广至所有谱形。',
      '', '### 4.3 相机空间与角度积分',
      '', '| 径向步长/µm | 每维子像素数 | 剖面对默认值的相对L2 | B偏差/MHz |','|---|---:|---:|---:|']
    for row in d['spatial']:lines.append(f"| {row['radial_step_um']} | {row['subpixels']} | {row['profile_l2_vs_baseline']:.6%} | {row['B_bias_mhz']:+.7f} |")
    lines+=['', f"0.25µm/8子像素与默认1µm/4子像素相比，曲线变化{d['spatial'][-1]['profile_l2_vs_baseline']:.5%}，略超10⁻⁴；本次B偏差仍低于0.0001MHz。径向步长与子像素同时改变，这是组合收敛检查，不能单独归因其中一项。",
      '', f"2°角度标准差算例由15点Gauss–Hermite提高到31、61点，剖面变化最大约{max(x['profile_l2_vs_baseline'] for x in d['angular']):.2e}；该场景15点足够。大孔径、偏轴、180°附近截断角分布尚不能据此概括。",
      '', '综合判断：默认前向计算档没有全部通过10⁻⁴目标；本轮所测离散因素对条件B的影响均低于0.01MHz。高精度配置示例可加宽到±600GHz并细化空间网格，但仍须继续检查更宽窗/更细网格，不能凭更大数值直接授予新精度等级。',
      '', '## 五、反向恢复：整谱与参数分开验收',
      '', '| 场景 | 单次整谱相对L2 | 条件B偏差/MHz | 非参数整谱<10% |','|---|---:|---:|---|']
    for name,row in m['cases'].items():
        p=row['physics'];f=row['parametric'];e=row['nonparametric']['relative_l2_folded']
        lines.append(f"| {names[name]} | {e:.4%} | {1000*(f['parameters']['brillouin_ghz']-p['brillouin_ghz']):+.6f} | {'通过' if e<.1 else '未通过'} |")
    lines+=['', '### 5.1 非参数网格与随机重复',
      '', '| 反演点数 | 间隔/MHz | 同一观测整谱L2 | 真谱采样再插值误差 | 选中α |','|---:|---:|---:|---:|---:|']
    for row in d['inverse_grid']:lines.append(f"| {row['nodes']} | {row['step_mhz']:.3f} | {row['spectral_relative_l2']:.3%} | {row['truth_sampling_interpolation_relative_l2']:.3%} | {row['alpha']:.4g} |")
    lines+=['', '加密到1200点没有继续改善恢复；不是“点越多越准”。原因包括欠定、噪声放大和离散α选择。当前正则项是离散二阶差分平方和，没有按Δν转换成连续积分范数；不同网格的α数值不能直接比较，也未声称此三档已达到反演网格收敛。真谱采样插值列只是表示误差诊断，不是所有重建方法的理论下界。',
      '', '| 场景（每类5次） | 整谱L2均值 | 最小—最大 | ≥10%的次数 |','|---|---:|---:|---:|']
    for key,row in d['nonparametric_noise'].items():lines.append(f"| {key} | {row['mean']:.3%} | {row['min']:.3%}—{row['max']:.3%} | {row['failures_above_10pct']}/5 |")
    lines+=['', '这15次为诊断性重复，不能给出精确失败概率。90°与强弹性光的不合格不是单一随机种子造成的；180°五次均低于10%也不是实测保证。',
      '', f"算子为350×600，零空间维数至少250；数值秩约{a['identifiability']['numerical_rank']}，奇异值比约{a['identifiability']['singular_ratio']:.2e}（依赖阈值）。非负约束和正则化选定一个候选谱，不证明真实谱唯一。",
      '', '### 5.2 条件频移随机精度',
      '', '| 信号尺度/电子 | 重复数 | 偏差/MHz | 标准差/MHz | RMSE/MHz | 观测质控失败 | 局部±1.96SE覆盖真值 |','|---|---:|---:|---:|---:|---:|---:|']
    for g,row in a['monte_carlo'].items():
        summary=row['summary'];s=summary['all_finite_fits'];trials=row['trials']
        valid=[x for x in trials if x.get('conditional_standard_errors',{}).get('brillouin_ghz') is not None]
        coverage=sum(abs(x['spacing_error_mhz'])<=1.96*1000*x['conditional_standard_errors']['brillouin_ghz'] for x in valid)
        lines.append(f"| {g} | {summary['attempts']} | {s['bias_mhz']:+.4f} | {s['std_mhz']:.4f} | {s['rmse_mhz']:.4f} | {summary['quality_failures']} | {coverage}/{len(valid)} |")
    lines+=['', '信号尺度不是相机e⁻/ADU转换增益，也不是直接给定的SNR。标准误和覆盖率都在固定模型、独立像素噪声和准确已知sigma前提下；50次覆盖结果仅作有限样本检查。低计数中央权重触边警告保留，不计为频移目标的自动失败。',
      '', '### 5.3 全局混叠反例',
      '', f"在同一180°合成观测上，放开范围至10GHz得到B={d['alias']['unrestricted']['parameters']['brillouin_ghz']:.9f}GHz，另有B={d['alias']['unrestricted']['competing_solutions'][0]['b_ghz']:.9f}GHz，二者总Δχ²约{d['alias']['unrestricted']['competing_solutions'][0]['delta_chi2']:.3f}。新版quality_pass=False，原因competing_spacing_solutions。",
      '', '二者接近B与FSR−B的周期别名。默认0.05–7GHz范围排除了后一支，因此默认表格里的高精度是**条件于该先验**的。若想仅靠数据确认级次，需要可靠独立频移范围、不同FSR/不同相位的额外测量或其他频率参考；不能只增加拟合迭代次数。',
      '', '## 六、未包含或仅简化处理的实验因素',
      '', '| 因素 | 当前处理 | 可能后果/需要补充 |','|---|---|---|',
      '| 实际FP线型、镀膜相位与偏振 | 理想Airy、标称FSR、有效精细度假设 | FSR/线宽偏差、非对称响应；需对应波长与光束条件的实测响应 |',
      '| 镜片/标准具面形、楔角、实际照射位置 | 未显式映射 | 位置相关线宽与频移；需有效光斑下的响应或面形资料 |',
      '| 光轴倾斜、环心、椭圆、畸变 | 主模型轴对称；新增有限偏心诊断 | 环均值可能丢信息；需保留二维原始帧并标定几何 |',
      '| 非高斯/位置相关PSF | 固定二维高斯sigma=4µm | 引入假线宽；需空间或等效频率响应测量 |',
      '| 暗场/平场与坏点、饱和、非线性、相机预处理 | 主流程没有真实校正 | 光子统计错误或曲线失真；需RAW与多帧校正数据 |',
      '| 噪声协方差与背景参考误差 | 独立泊松+独立高斯；sigma按模拟真值计算 | 实际滤波、共享背景、漂移使独立假设失效；需重复数据估计协方差/噪声模型 |',
      '| 时间漂移、振动、曝光内频率变化 | 未显式动态建模 | 等效展宽/偏移；需时间戳、参考谱、激光/温度稳定性 |',
      '| 水体成分、颗粒、热梯度与真实收光角分布 | 纯水、均匀温度、截断高斯角分布、简单弹性分量 | 谱族失配；需样品信息、温度空间稳定性、光阑与几何资料 |',
      '| 本征线宽物理模型 | 弱阻尼三线模型、体黏度参考值 | 不覆盖所有频率依赖黏弹性/仪器耦合；需外部水谱/文献对照 |',
      '', '不把所有未知量一起自由拟合当作解决方案：峰宽与仪器宽度、全谱平移与标准具相位、幅度与透过率等存在强耦合。数据不足时增加自由度会使不确定性更大。',
      '', '## 七、关键系统误差及提高精度的量级要求',
      '', '上一轮的大幅失配重跑仍成立：FSR误设±1%导致约±20MHz的B偏差，焦距高1%约−49MHz；真精细度40却按30拟合可低估本征线宽约58%，而残差仍很小。合成随机RMSE=0.20MHz不能覆盖这些系统偏差。',
      '', '本轮另用±0.01%的小扰动计算局部灵敏度。下表把全部误差预算暂分给单一项，仅作数量级设计；多项同时存在时必须留余量、考虑协方差及模型偏差。不能直接相加百分比，也不能把未知系统偏差当独立随机项平方和。',
      '', '| 单一误差来源 | 单项B误差约≤1MHz | 单项B误差约≤0.1MHz | 适用条件 |','|---|---:|---:|---|']
    fs=d['local_sensitivity']['fsr_ghz'];fl=d['local_sensitivity']['focal_length_mm'];sl=d['physical_slopes']
    lines += [f"| FSR相对校准误差 | {fs['illustrative_fraction_for_1mhz']:.5%}（约{fs['illustrative_fraction_for_1mhz']*instrument['fsr_ghz']*1000:.2f}MHz） | {fs['illustrative_fraction_for_0p1mhz']:.5%} | 本次相位/视场/90°模型局部结果 |",
      f"| 有效焦距相对误差 | {fl['illustrative_fraction_for_1mhz']:.5%}（100mm时约{fl['illustrative_fraction_for_1mhz']*100*1000:.2f}µm） | {fl['illustrative_fraction_for_0p1mhz']:.5%} | 更应直接标定频率—位置映射，不是只测镜筒尺寸 |",
      f"| 水内散射角误差 | 约{1/sl['B_mhz_per_degree']:.4f}° | 约{.1/sl['B_mhz_per_degree']:.4f}° | 90°附近dB/dθ≈{sl['B_mhz_per_degree']:.2f}MHz/°；与理论B或声速比较时需要 |",
      f"| 温度误差/不稳定性 | 约{1/sl['B_mhz_per_kelvin']:.3f}K | 约{.1/sl['B_mhz_per_kelvin']:.4f}K | 20℃附近dB/dT≈{sl['B_mhz_per_kelvin']:.2f}MHz/K；这是水温效应 |",
      '', '这些要求不是厂家承诺，也不是从本文件可证明的最优极限。频率—半径映射可把有效焦距、像素尺度、倍率和材料折射率的组合直接标定，通常比逐项追求机械尺寸精度更贴近实际需求。',
      '', '线宽目标另算：本模型90°水谱本征FWHM约214MHz。孤立洛伦兹峰近似下，若要求本征线宽相对误差<10%，仪器FWHM不确定度应约<21MHz；要求<1%则约<2.1MHz。真实Airy、激光卷积、相机模糊和参数相关会改变这一估算，必须用实测响应做端到端检验。',
      '', '提高平均次数只降低独立随机误差。理想稳定条件下4帧平均可把0.20MHz随机量级降到约0.10MHz，但不能消除FSR偏差、漂移、谱族错误和级次歧义；本轮未模拟多帧漂移。',
      '', '### 环心偏移补充诊断',
      '', '| 合成光轴偏移/像素 | 前向曲线L2 | B偏差/MHz | 标准化残差均方 |','|---|---:|---:|---:|']
    for row in d['ring_center']:lines.append(f"| {row['offset_pixels']} | {row['profile_relative_l2']:.4%} | {row['B_bias_mhz']:+.6f} | {row['chi2']:.3f} |")
    lines+=['', '这是固定ROI内、径向信号与照明一起偏心的无噪声诊断；不能解释为任何实验都允许1像素环心误差。椭圆、局部遮挡、非均匀响应尚未同时扫描。',
      '', '## 八、需要向厂家和实验采集端取得的数据',
      '', '按优先级排列：P0是进入可信实测拟合前优先补齐的输入；P1用于完整误差预算或线宽目标。无法立即标定时可继续做带假设的灵敏度分析，但不宣称已达到实验精度。',
      '', '| 优先级/提供方 | 数据 | 推荐保存形式 | 用途 |','|---|---|---|',
      '| P0 厂家/标定 | 632.8nm实际FSR、测量不确定度、实测厚度、检测波长与温度 | 检测报告+CSV/JSON | 确定频率尺度与规格适用性 |',
      '| P0 厂家/实验 | 有效透射曲线：频偏、透过率、重复测量；注明光斑、角度、偏振、照射位置 | CSV，不仅是图片或一个精细度数字 | 建真实响应核，区分有效线宽和理想反射率精细度 |',
      '| P0 实验 | 已知频差参考的二维环图；覆盖使用视场，记录参考频率及不确定度 | RAW TIFF/NPY + 频率表 | 标定环心、r²映射、畸变、级次与相位；已知频差可来自经确认的参考源/移频装置 |',
      '| P0 相机/实验 | 相机e⁻/ADU转换增益、读出噪声、线性范围、饱和阈值；采集模式/像素合并 | 参数表+相同曝光的多帧暗场/平场 | 把数值灰度转换成正确噪声权重 |',
      '| P0 实验 | 样品原始二维帧、暗场、空池/光路背景、重复测量 | 保留原始位深，无自动Gamma/锐化/压缩 | 背景去除、坏点与漂移检查；不只保留径向均值 |',
      '| P0 实验 | 水内散射角及不确定度、光阑/收集角、容器折射几何 | 光路图+尺寸/角度表 | 确定水中q与角展宽，不能拿空气外角直接代替 |',
      '| P0 实验 | 水温校准、时间序列、稳定性、样品纯度与处理方式 | CSV+实验记录 | 温度主线与不同批次比较 |',
      '| P1 激光/实验 | 波长、线宽、纵模/边模、时间漂移及曝光时间 | 测量记录或经核对的规格；标记来源 | 避免把激光展宽误认为水谱线宽 |',
      '| P1 厂家/实验 | 实际光斑、标准具温度、偏振、面形/楔角或位置响应、PSF | 图/表/响应数据 | 建模非理想线型、漂移和空间依赖 |',
      '| P1 独立来源 | 同温同角水谱或已知标准样品的可靠对照 | 原始数据+出处+不确定度 | 从自洽验证走向外部物理验证 |',
      '', '最有价值的最终输入是有效响应核K[j,ν]、频率/位置映射及其不确定度、观测噪声或协方差。如果K由实测窄线扫描建立，必须说明输入激光线型是否已含在核中，避免再次卷积激光而重复展宽。还要说明K是否已乘Δν；代码的physical_kernel已包含积分权重。',
      '', '推荐的原始采集记录字段：sample_id、frame_id、time、exposure_ms、camera_gain_setting、temperature_C、temperature_uncertainty_C、wavelength_nm、internal_angle_deg、angle_uncertainty_deg、etalon_temperature_C、reference_frequency_GHz、raw_file、dark_reference_id、background_reference_id、saturation_mask。未知值留空并标记unknown，不能填默认值冒充测量。',
      '', '推荐的径向数据字段：radius_mm、mean_signal_electrons、sigma_mean_electrons、pixel_count、valid_flag，并保存环心/椭圆参数与二维原图链接。若已做背景扣除，sigma必须包含背景参考的不确定度；例如独立N帧样品平均减去M帧暗场平均，其方差包含两个平均量的方差，而非只有样品项。存在相关噪声时还需要协方差矩阵或相应噪声模型。',
      '', '相机标定概念可参考[EMVA 1288官方标准入口](https://www.emva.org/standards-technology/emva-1288/)及[Release 4.0 Linear](https://www.emva.org/wp-content/uploads/EMVA1288Linear_4.0Release.pdf)中的系统增益、暗噪声和线性响应定义。本汇报没有据此宣称当前相机已符合该标准。',
      '', '## 九、代码使用方法',
      '', '### 9.1 环境与一键运行',
      '', '本机验证使用C:/Python314/python.exe；numpy/scipy/matplotlib实际版本见requirements_verified.txt，IAPWS源码与许可证在.water_deps。未在另一台干净机器复验。解压后保留目录结构，不要只拷贝一个脚本。',
      '', '在当前文件夹双击“运行修订与验收.cmd”，或打开终端执行：',
      '', '```console','python -m pip install -r requirements_verified.txt','python run_all.py','```',
      '', '新入口依次执行单元测试、四算例、150次条件拟合验收、深入检查，再生成统一研究汇报。输出进入output/rerun_时间戳/，不覆盖已有结果；出现非零退出码会停止并保留日志。计算时间取决于机器，1200点非参数反演和空间细化较慢。',
      '', '### 9.2 只运行某一部分',
      '', '```console',
      'python -m unittest discover -s tests -v',
      'python run_water_simulation.py --output output/my_cases',
      'python run_etalon_audit.py --output output/my_acceptance',
      'python run_deep_review.py --output output/my_deep',
      'python make_detailed_report.py --cases output/my_cases --audit output/my_acceptance --deep output/my_deep --dest output/my_report',
      '```',
      '', '每个输出目录必须为空或不存在。若已有结果，换一个目录名，不要直接覆盖。',
      '', '### 9.3 更改模拟参数',
      '', '四算例入口支持JSON覆盖，其他两个验收入口是冻结的基准情景。示例：',
      '', '```json','{"water": {"temperature_k": 298.15}, "fit_spacing_bounds_ghz": [0.05, 7.0]}','```',
      '', '```console','python run_water_simulation.py --config 示例参数_25C.json --output output/water25','```',
      '', '常用字段：water.temperature_k为K；scattering_angle_deg为水内角；instrument.fsr_ghz为GHz；finesse是精细度；peak_transmission是0–1比例；focal_length_mm为mm；camera.pixel_um/radial_step_um/psf_sigma_um为µm；gain_electrons是模拟信号尺度。变更波长要同时更新water和instrument的wavelength_nm及石英n/ng；代码仅检查波长一致，不会自动帮任意JSON重算材料参数。',
      '', '修改参数后，不能把默认验收报告直接贴到新配置上。报告生成器会拒绝不匹配的冻结配置；若要对新仪器/新波长完整验收，应同步修改验收脚本情景与先验，再重新冻结判据。高精度数值配置示例仅供继续收敛试验，不代表自动提高实测精度。',
      '', '### 9.4 看哪些输出',
      '', '| 文件 | 内容 |','|---|---|',
      '| parameters.json / audit_config.json | 实际配置、来源与阈值；先看这些再解读数值 |',
      '| metrics.json | 四算例、反演正则路径、拟合状态和模型失配 |',
      '| *_unfolded_spectrum.csv | 宽频谱、积分权重、条件恢复；频率GHz、密度1/GHz |',
      '| *_folded_spectrum.csv | 单FSR真谱与非参数恢复谱 |',
      '| *_radial_profile.csv | 半径、期望、观测、sigma、两种前向回代 |',
      '| *_observed_image_electrons.npy | 模拟含噪二维电子计数；不是相机实测 |',
      '| audit_metrics.json / mc_*_observations.npz | 150次结果、每次观测、sigma、种子与计数 |',
      '| deep_review.json / inverse_grid_curves.npz | 新增前向误差、竞争解、网格与整谱重复 |',
      '| 研究汇报.md / 深入误差图.png | 本次最终说明和图 |',
      '', '### 9.5 如何调用底层拟合器，以及目前不能直接做什么',
      '', '```python','from water_simulation.inverse import fit_water',
      'fit = fit_water(K, frequency_ghz, observed_profile, sigma_profile, water,',
      '                spacing_bounds=(0.05, 7.0))',
      '# 先检查质量和竞争解，再读取参数；success本身不够。',
      'print(fit["quality_pass"], fit["quality_reasons"], fit["competing_solutions"])',
      'print(fit["parameters"], fit["conditional_standard_errors"])','```',
      '', '以上是API说明，K、坐标、观测及sigma必须由正确标定/预处理提供，不是可直接复制运行的实测示例。K形状为环带数×频率点数且已含积分权重；观测与K输出必须同单位；sigma必须是环均值的标准差，不是方差或单像素噪声。振幅范围目前0.1–3，以归一化信号为设计背景；直接塞入原始ADU通常不合适。',
      '', '本版没有“上传一张实验环图就自动得到可靠水谱”的完整入口。实测应用还需实现并验收：RAW导入、背景/平场/饱和处理、环心和几何标定、测得响应核接入、正确噪声权重与不确定度传播。当前结果适合仿真、误差排查和采集方案设计。',
      '', '## 十、验收状态与下一步',
      '', '1. 已修复复现的代码与质量判据缺陷；37项测试通过。',
      '2. 前向默认数值精度：频率步长通过；新增窗口/空间对照未全部低于10⁻⁴；频移离散偏差在所测场景下远小于0.01MHz。',
      '3. 反向频移：有明确先验、中高计数下通过1MHz随机RMSE目标；级次未知时必须报告歧义。',
      '4. 反向整谱：90°与强弹性光未达到10%目标；180°有限合成样本达到，但不能推广为实验保证。',
      '5. 本征线宽与绝对频移：仪器/相位/激光未知时不予实测精度认证。',
      '6. 优先取得对应波长的实测有效响应和频率—位置参考，再配套相机噪声、背景、温度与角度数据；用独立样品或留出数据检验，避免只在同一组标定数据上评价自己。',
      '', '本轮没有完成实测系统验证，也没有签署项目G2整体完成。报告中的误差阈值为事先记录的工程诊断标准，不代表导师批准。',
      '', '## 十一、来源与复现记录',
      '', '- 厂家图纸：证据/op-7423-x-a-1-inch-etalon-various-thickness.pdf，原始文件哈希保留。',
      '- 石英色散：[Malitson 1965](https://opg.optica.org/josa/abstract.cfm?uri=josa-55-10-1205)。',
      '- 水物性：[IAPWS-95官方说明](https://iapws.org/technical-guidance/release/IAPWS-95)、[IAPWS水折射率发布](https://iapws.org/documents/release/Rindex.download)。材料物性来源不等于散射线型的独立实测验证。',
      '- 相机噪声定义：上文EMVA官方来源。采集建议和误差预算为本轮工程推导，不冒称厂家要求。',
      '- 证据/二次修订单元测试.txt记录37项通过结果；最终数据核验.txt记录数据核验，交付文件清单绑定本版。',
      f'- 四算例：{cases.resolve()}；扩展验收：{audit.resolve()}；深入检查：{deep.resolve()}。',
      '', '复现时应保留源码版本、配置、种子、软件版本、原始观测和输出。不确定度不是单独一个百分比；需要注明目标量、先验、噪声条件、标定状态与误差来源。']
    (dest/'研究汇报.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10})
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    axs[0,0].plot([x['limit_ghz'] for x in d['window']],[100*x['profile_l2_vs_baseline'] for x in d['window']],marker='o')
    axs[0,0].axhline(.01,color='gray',ls='--');axs[0,0].set(title='积分窗口误差：不能只看频率步长',xlabel='频率窗半宽 / GHz',ylabel='相对默认剖面的变化 / %')
    axs[0,1].plot([x['nodes'] for x in d['inverse_grid']],[100*x['spectral_relative_l2'] for x in d['inverse_grid']],marker='o',color='#ba633c')
    axs[0,1].axhline(10,color='gray',ls='--');axs[0,1].set(title='反演网格：加密不保证更准确',xlabel='频率点数',ylabel='整谱相对L2误差 / %',ylim=(0,25))
    keys=list(d['nonparametric_noise'])
    means=np.array([d['nonparametric_noise'][x]['mean']*100 for x in keys])
    low=np.array([d['nonparametric_noise'][x]['min']*100 for x in keys]);high=np.array([d['nonparametric_noise'][x]['max']*100 for x in keys])
    axs[1,0].bar(['90°','180°','10%弹性光'],means,color='#477c87',yerr=[means-low,high-means],capsize=4)
    axs[1,0].axhline(10,color='gray',ls='--');axs[1,0].set(title='每类5次整谱重复：均值及范围',ylabel='整谱L2误差 / %')
    f=d['alias']['unrestricted'];solutions=[x for x in f['start_solutions'] if x['start_b'] in [6.5,8.8]]
    axs[1,1].bar([f"B={x['b']:.3f} GHz" for x in solutions],[2*x['cost']/350 for x in solutions],color=['#477c87','#ba633c'])
    axs[1,1].set(title='不限制级次：两个解残差几乎相同',ylabel='平均标准化残差平方',ylim=(0,1.1))
    fig.suptitle('二次审查：数值精度、整谱恢复与全局歧义必须分开',fontsize=15)
    fig.savefig(dest/'深入误差图.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,default=Path('output/final_cases'));p.add_argument('--audit',type=Path,default=Path('output/final_acceptance'));p.add_argument('--deep',type=Path,default=Path('output/deep_review'));p.add_argument('--dest',type=Path,default=Path('.'))
    x=p.parse_args();build(x.cases,x.audit,x.deep,x.dest)
