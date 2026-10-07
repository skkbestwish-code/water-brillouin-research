"""Reproducible liquid-water spectra, FP images and inverse validation.

python run_water_simulation.py --output output/water_rb_fp
python run_water_simulation.py --config output/water_rb_fp/parameters.json

中文说明：主入口脚本。完整链路：
  水谱（中心线 + 布里渊侧峰）→ 展开谱前向 → 折叠进单个 FSR →
  FP 环图（PSF + 像素面积平均 + 泊松/读出噪声）→ 环形均值 →
  非参数反演 + 参数化条件拟合 → 数值收敛、失配敏感性与蒙特卡洛检查。
产物：parameters.json、metrics.json、各算例 CSV/NPY、四组 PNG、
中文报告「验证结果.md」。
"""
import os
# 固定单线程 BLAS/OMP：避免线程调度引起的浮点求和次序差异，
# 保证数值结果与耗时统计可复现。
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,time,sys
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
import scipy
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from water_simulation.physics import (Water,WaterFP,WaterDetector,properties,line_parameters,
    physical_spectrum,folded_spectrum,integration_grid,fp_response,coverage,components)
from water_simulation.inverse import fit_water
from simulation.physics import frequency_grid,airy,illumination
from simulation.imaging import Camera,noisy_profile
from simulation.inverse import reconstruct

# 默认配置：水/仪器/相机参数、激光线宽与纵模、电子学增益/背景/读出噪声、
# 随机种子、各数值网格规模与蒙特卡洛重复次数。
# --config 可提供同结构 JSON 覆盖（嵌套字典浅合并，未知字段报错）。
DEFAULTS=dict(water=asdict(Water()),instrument=asdict(WaterFP()),camera=asdict(Camera()),
    laser_fwhm_ghz=.01,laser_modes=[[0.,1.]],gain_electrons=2000.,background_electrons=20.,
    read_noise_electrons=3.,seed=20261001,folded_grid_size=6000,inverse_grid_size=600,
    physical_core_step_ghz=.0025,physical_tail_step_ghz=.1,physical_limit_ghz=150.,
    monte_carlo_replicates=12,low_gain_electrons=25.)
BLUE='#224e75';ORANGE='#c66926';PINK='#a64678';GRAY='#666666'


# 递归把 numpy 标量/数组转成可 JSON 序列化的 Python 对象。
def clean(obj):
    if isinstance(obj,np.ndarray):return obj.tolist()
    if isinstance(obj,(np.integer,np.floating,np.bool_)):return obj.item()
    if isinstance(obj,dict):return {k:clean(v) for k,v in obj.items()}
    if isinstance(obj,(tuple,list)):return [clean(v) for v in obj]
    return obj


# 输出工具：JSON（UTF-8、禁止 NaN，便于下游机器读取）；CSV 按列堆叠写出。
def write_json(path,obj):path.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csv(path,cols,names):np.savetxt(path,np.column_stack(cols),delimiter=',',header=','.join(names),comments='')
# 从拟合结果中剥离 spectrum/fitted 大数组，只保留参数与诊断摘要。
def fit_summary(fit):return {k:v for k,v in fit.items() if k not in ['spectrum','fitted']}
# 相对 L2 差异 ‖a−b‖/‖b‖。
def rel(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def run(config,out):
    # 主流程：a) 构造对象并校验波长一致；b) 建立宽频/折叠/反演网格与
    # 前向矩阵；c) 四个算例（参考角、180°、2° 角展宽、10% 弹性光）；
    # d) 数值独立核对；e) 物性/视场扫描；f) 失配敏感性；
    # g) 蒙特卡洛噪声重复；h) 输出 JSON/CSV/图/中文报告。
    start=time.perf_counter();out.mkdir(parents=True,exist_ok=True);write_json(out/'parameters.json',config)
    water=Water(**config['water']);cfg=WaterFP(**config['instrument']);camera=Camera(**config['camera'])
    # 散射介质（水）与 FP 必须共享同一个激光真空波长。
    if water.wavelength_nm!=cfg.wavelength_nm:raise ValueError('Water and FP must share the laser vacuum wavelength.')
    detector=WaterDetector(camera)
    # 三套频率网格：展开用宽频细网格（含洛伦兹尾），折叠与反演各用
    # 单 FSR 周期网格（点数不同：6000 用于对比画图，600 用于反演）。
    v,w=integration_grid(config['physical_core_step_ghz'],config['physical_tail_step_ghz'],config['physical_limit_ghz'])
    folded_v=frequency_grid(cfg.fsr_ghz,config['folded_grid_size']);inverse_v=frequency_grid(cfg.fsr_ghz,config['inverse_grid_size'])
    # 三个前向矩阵（环形均值 × 频率），均含照明与积分权重：
    # wide_k 展开网格、inverse_k 反演网格、fold_k 折叠网格。
    print('Building unfolded and folded water forward operators...',flush=True)
    wide_k,radial=detector.physical_kernel(v,w,cfg);inverse_k=detector.kernel(inverse_v,cfg)
    fold_k=detector.kernel(folded_v,cfg)
    laser=config['laser_fwhm_ghz'];modes=config['laser_modes'];gain=config['gain_electrons']
    bg=config['background_electrons'];rn=config['read_noise_electrons'];seed=config['seed']
    metrics=dict(configuration=config,confirmed=dict(target='ambient-pressure liquid water',laser_type='He-Ne',
        nominal_fsr_ghz=15.,nominal_finesse=30.,fp_surroundings='air',fp_cavity_medium='unknown'),
        instrument_derived=dict(fwhm_ghz=cfg.fsr_ghz/cfg.finesse,airy_coefficient=cfg.airy_coefficient,
            ideal_reflectivity=cfg.ideal_reflectivity,cavity_thickness_mm=cfg.cavity_thickness_mm,
            note='Thickness assumes specified group index and neglects mirror phase dispersion.'),
        software=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,iapws='1.5.5'),
        cases={},numerical_checks={},sensitivity={},physical_scans={},coverage_scans=[],monte_carlo={})
    # 四个算例：参考角、180° 背散射（角展宽置 0）、2° 收光角展宽、
    # 10% 额外弹性光。每个算例完整走一遍"真值 → 噪声成像 → 反演/拟合"。
    cases=[]
    for i,(name,ww,title) in enumerate([
        ('water_90deg' if water.scattering_angle_deg==90 else f'water_reference_{water.scattering_angle_deg:g}deg',water,
            f'Pure water: internal {water.scattering_angle_deg:g} deg'),
        ('water_180deg',replace(water,scattering_angle_deg=180,angular_sigma_deg=0),'Pure water: internal 180 deg'),
        ('water_angular_2deg',replace(water,angular_sigma_deg=2),'Water: 2 deg angular sigma'),
        ('water_elastic_10percent',replace(water,elastic_fraction=.1),'Water: 10% extra elastic light')]):
        print('Running '+name+'...',flush=True)
        # 真值：展开绝对谱（宽网格）与折叠周期谱（一个 FSR）。
        source=physical_spectrum(v,ww,laser,modes);folded=folded_spectrum(folded_v,ww,laser,modes)
        expected=wide_k@source;irradiance=radial@source;image=detector.image(irradiance)
        # 电子学噪声：泊松光子散粒 + 高斯读出；y 为去背景后的相对剖面，
        # sigma 为逐 bin 噪声标准差（含散粒与读出两部分）。
        rng=np.random.default_rng(seed+i);observed_image=rng.poisson(gain*image+bg)+rng.normal(0,rn,image.shape)
        y=(detector.annular_mean(observed_image)-bg)/gain
        sigma=np.sqrt((gain*expected+bg+rn**2)/detector.counts)/gain
        # 两条反演路线：非参数非负 Tikhonov（单 FSR 网格）与
        # 条件参数拟合（展开网格、宽频矩阵，使用全部已知先验）。
        inverse=reconstruct(inverse_k,y,sigma,inverse_v)
        fit=fit_water(wide_k,v,y,sigma,ww,laser,modes)
        # 把非参数恢复谱周期插值到折叠网格，便于与折叠真值直接比较。
        recovered=np.interp(folded_v,inverse_v,inverse['spectrum'],period=cfg.fsr_ghz)
        # 汇总：物性/谱参数、视场覆盖、宽窗积分损失、单 FSR 外功率、
        # 非参数与参数化结果、以及有限网格前向一致性
        # （环形均值图像应精确还原矩阵正演 expected）。
        core=abs(v)<cfg.fsr_ghz/2
        p=line_parameters(ww)
        summary=dict(physics=p,water=asdict(ww),coverage=coverage(ww,cfg,camera.roi_mm),
            integrated_area_in_wide_window=float(w@source),wide_window_missing_area=float(1-w@source),
            area_outside_single_fsr=float(w[~core]@source[~core]),
            nonparametric=dict(relative_l2_folded=rel(recovered,folded),recovered_area=float(inverse['spectrum'].sum()*(inverse_v[1]-inverse_v[0])),
                alpha=inverse['alpha'],chi2=inverse['chi2'],discrepancy_reached=inverse['discrepancy_reached'],path=inverse['path'],
                note='Recovers one periodic FSR of incident light, including laser broadening and all folded orders.'),
            parametric=fit_summary(fit),image_operator_max_abs_error=float(np.max(abs(detector.annular_mean(image)-expected))),
            folded_forward_relative_l2=rel(fold_k@folded,expected))
        metrics['cases'][name]=summary
        # 每个算例落盘：展开谱、折叠谱、径向剖面 CSV 与两张像素图 npy。
        csv(out/(name+'_unfolded_spectrum.csv'),[v,w,physical_spectrum(v,ww,0,[[0,1]]),source,fit['spectrum']],
            ['offset_GHz','quadrature_width_GHz','intrinsic_density_per_GHz','incident_density_per_GHz','conditional_fit_density_per_GHz'])
        csv(out/(name+'_folded_spectrum.csv'),[folded_v,folded,recovered],['offset_GHz','folded_incident_density_per_GHz','nonparametric_recovery_per_GHz'])
        csv(out/(name+'_radial_profile.csv'),[detector.bin_radius_mm,expected,y,sigma,inverse['fitted'],fit['fitted']],
            ['radius_mm','expected','observation','sigma','nonparametric_forward_check','conditional_fit'])
        np.save(out/(name+'_expected_image.npy'),image);np.save(out/(name+'_observed_image_electrons.npy'),observed_image)
        cases.append(dict(name=name,water=ww,title=title,source=source,folded=folded,expected=expected,y=y,sigma=sigma,inverse=inverse,fit=fit,image=image,observed=observed_image))
        print(f"  folded spectrum L2 {summary['nonparametric']['relative_l2_folded']:.4f}; fit B {fit['parameters']['brillouin_ghz']:.6f} GHz; chi2 {fit['chi2']:.3f}",flush=True)

    # ---- 数值独立核对 ----：用与宽网格矩阵无关的标量积分交叉验证。
    # High accuracy scalar quadrature is independent of the wide-grid matrix.
    print('Independent quadrature and numerical convergence...',flush=True)
    base=cases[0];p=line_parameters(water)
    # 标量密度函数：单点求值，供 scipy.integrate.quad 使用。
    def scalar_density(x):return float(physical_spectrum(np.array([x]),water,laser,modes)[0])
    # quad 断点：每个分量的中心及 ±2、±10 个 HWHM 处，帮助处理窄尖峰。
    points=[row[0]+j*row[1] for row in components(water) if row[2]>0 for j in [-10,-2,0,2,10]]
    lo=-config['physical_limit_ghz'];hi=-lo
    scalar_area=quad(scalar_density,lo,hi,points=sorted(set(points)),epsabs=1e-9,limit=1500)[0]
    checks=dict(wide_grid_area=float(w@base['source']),independent_quad_area=scalar_area,
        quadrature_area_abs_error=abs(scalar_area-w@base['source']),radial_bins=detector.nbins)
    # 在三个半径处用独立标量积分计算 FP 响应，与矩阵路径对比。
    scalar_response=[]
    for r in [.36,.71,1.1]:
        scalar_response.append(quad(lambda x: scalar_density(x)*float(fp_response([r],[x],cfg)[0,0]),lo,hi,
            points=sorted(set(points)),epsabs=2e-8,limit=2000)[0])
    direct_response=fp_response([.36,.71,1.1],v,cfg)@(w*base['source'])
    checks['independent_response_max_abs_error']=float(np.max(abs(direct_response-scalar_response)))
    # 网格步长减半重算环形剖面，检查离散收敛（相对 L2）。
    vv,ww=integration_grid(config['physical_core_step_ghz']/2,config['physical_tail_step_ghz']/2,config['physical_limit_ghz'])
    finer_k,_=detector.physical_kernel(vv,ww,cfg,False)
    finer_profile=finer_k@physical_spectrum(vv,water,laser,modes)
    checks['half_step_profile_relative_l2']=rel(base['expected'],finer_profile)
    del finer_k
    # Paraxial FFT convolution is a control, not the full camera inverse.
    # 抛物近似下的 FFT 周期卷积对照：半径先映射到等效频率坐标再插值。
    df=folded_v[1]-folded_v[0]
    conv=np.fft.fftshift(np.fft.ifft(np.fft.fft(np.fft.ifftshift(base['folded']))*
        np.fft.fft(np.fft.ifftshift(airy(folded_v,cfg)))).real)*df
    x=cfg.nu0_ghz*cfg.external_index**2/(cfg.cavity_phase_index*cfg.group_index)*(detector.radius_mm/cfg.focal_length_mm)**2/2-cfg.center_detuning_ghz
    fft_profile=detector.profile(np.interp(x,folded_v,conv,period=cfg.fsr_ghz)*illumination(detector.radius_mm,cfg))
    checks['fft_vs_unfolded_profile_relative_l2']=rel(fft_profile,base['expected'])
    # A full-order frequency translation is almost invisible to a periodic FP.
    # Translate the quadrature nodes, too: otherwise a narrow line translated
    # into the coarse tail mesh would be undersampled and mimic order contrast.
    # （整体平移一个 FSR，周期谱应几乎不变——验证折叠/周期约定。）
    shifted_k,_=detector.physical_kernel(v+cfg.fsr_ghz,w,cfg,False)
    shifted_profile=shifted_k@base['source']
    del shifted_k
    checks['shift_one_fsr_profile_relative_l2']=rel(shifted_profile,base['expected'])
    checks['shift_one_fsr_mean_squared_noise_units']=float(np.mean(((shifted_profile-base['expected'])/base['sigma'])**2))
    checks['shift_one_fsr_note']='Residual includes finite integration tails and weak off-axis chromaticity; exact mathematical periodicity is the paraxial limit.'
    metrics['numerical_checks']=checks

    # ---- 扫描 ----：物性扫描只重算谱参数与单 FSR 外功率（不重建全链路）。
    print('Water property, angle, linewidth and FOV scans...',flush=True)
    for name,values,field in [('temperature_C',[10,20,30],'temperature_k'),('water_angle_deg',[60,90,120,150,180],'scattering_angle_deg'),
            ('bulk_viscosity_mPa_s',[1,2.5,5],'bulk_viscosity_pa_s'),('intrinsic_central_fraction',[.0065,.03,.1],'central_fraction_override')]:
        rows=[]
        for value in values:
            # 单位换算：温度 °C→K、体黏度 mPa·s→Pa·s，其余原样使用。
            use=value+273.15 if field=='temperature_k' else value/1000 if field=='bulk_viscosity_pa_s' else value
            scan=replace(water,**{field:use});sp=physical_spectrum(v,scan,laser,modes)
            rows.append(dict(value=value,physics=line_parameters(scan),area_outside_single_fsr=float(w[abs(v)>=cfg.fsr_ghz/2]@sp[abs(v)>=cfg.fsr_ghz/2])))
        metrics['physical_scans'][name]=rows
    # 视场覆盖扫描：腔折射率 × 焦距 × 中心失谐相位 × 水内角度，
    # 检查两侧布里渊峰的环是否仍在 ROI 内。
    for index in [1.,1.46]:
        for f in [50.,100.,150.,250.]:
            for phase in [2.5,7.5]:
                icfg=replace(cfg,cavity_phase_index=index,cavity_group_index=index,focal_length_mm=f,center_detuning_ghz=phase)
                for angle in [90,180]:
                    metrics['coverage_scans'].append(dict(cavity_index=index,focal_length_mm=f,center_detuning_ghz=phase,water_angle_deg=angle,
                        **coverage(replace(water,scattering_angle_deg=angle),icfg,camera.roi_mm)))
    write_json(out/'coverage_scans.json',metrics['coverage_scans'])

    # ---- 失配敏感性 ----：真值不变，用"错误假设"重新拟合基准观测，
    # 量化每个假设对 B 频移/线宽/整谱频移的偏差贡献。
    print('Instrument, laser and angular-model mismatch fits...',flush=True)
    truthB=p['brillouin_ghz'];truthW=p['brillouin_hwhm_ghz']
    # 仪器类失配：相位 +100 MHz、焦距 +1%、精细度 25/35、PSF 假设 0/8 µm、
    # 腔折射率假设 1.46；各用错误假设的前向矩阵重拟合同一组观测。
    for name,new_cfg,new_camera in [('phase_plus_100MHz',replace(cfg,center_detuning_ghz=cfg.center_detuning_ghz+.1),camera),
            ('focal_plus_1percent',replace(cfg,focal_length_mm=cfg.focal_length_mm*1.01),camera),
            ('finesse_25',replace(cfg,finesse=25),camera),('finesse_35',replace(cfg,finesse=35),camera),
            ('psf_assumed_0um',cfg,replace(camera,psf_sigma_um=0)),('psf_assumed_8um',cfg,replace(camera,psf_sigma_um=8)),
            ('cavity_assumed_index_1p46',replace(cfg,cavity_phase_index=1.46,cavity_group_index=1.46),camera)]:
        dd=detector if new_camera==camera else WaterDetector(new_camera)
        kk,_=dd.physical_kernel(v,w,new_cfg,False)
        result=fit_water(kk,v,base['y'],base['sigma'],water,laser,modes)
        metrics['sensitivity'][name]=dict(**fit_summary(result),shift_bias_mhz=1000*(result['parameters']['shift_ghz']-p['shift_ghz']),
            spacing_bias_mhz=1000*(result['parameters']['brillouin_ghz']-truthB),width_bias_mhz=1000*(result['parameters']['brillouin_hwhm_ghz']-truthW))
        del kk
    # 模型类失配：观测由"含展宽/杂散/双模/更宽激光"的真值模型生成，
    # 拟合仍用名义模型（无展宽、无杂散、单模、10 MHz 激光）。
    mismatch_sources=[('unmodeled_water_angle_sigma_2deg',replace(water,angular_sigma_deg=2),laser,modes),
        ('unmodeled_10percent_elastic',replace(water,elastic_fraction=.1),laser,modes),
        ('unmodeled_laser_doublet',water,laser,[[-.16,.5],[.16,.5]]),
        ('unmodeled_laser_FWHM_100MHz',water,.1,modes)]
    for j,(name,ww,lf,lm) in enumerate(mismatch_sources):
        # 用真值模型的期望生成一组新的含噪观测（与算例使用不同种子）。
        e=wide_k@physical_spectrum(v,ww,lf,lm)
        y,sigma=noisy_profile(e,detector.counts,gain,bg,rn,seed+200+j)
        result=fit_water(wide_k,v,y,sigma,water,laser,modes)
        metrics['sensitivity'][name]=dict(**fit_summary(result),shift_bias_mhz=1000*(result['parameters']['shift_ghz']-p['shift_ghz']),
            spacing_bias_mhz=1000*(result['parameters']['brillouin_ghz']-truthB),width_bias_mhz=1000*(result['parameters']['brillouin_hwhm_ghz']-truthW))

    # ---- 蒙特卡洛 ----：参考计数与低计数两种增益下重复噪声实现，
    # 统计 B 间距的偏差与标准差（只反映固定模型下的随机误差）。
    print('Repeated noise trials...',flush=True)
    for name,g in [('reference_counts',gain),('low_counts',config['low_gain_electrons'])]:
        trials=[]
        for j in range(config['monte_carlo_replicates']):
            y,sigma=noisy_profile(base['expected'],detector.counts,g,bg,rn,seed+1000+j)
            result=fit_water(wide_k,v,y,sigma,water,laser,modes)
            trials.append(dict(**fit_summary(result),spacing_error_mhz=1000*(result['parameters']['brillouin_ghz']-truthB)))
        errors=np.array([t['spacing_error_mhz'] for t in trials])
        metrics['monte_carlo'][name]=dict(gain_electrons=g,trials=trials,spacing_bias_mhz=float(errors.mean()),
            spacing_std_mhz=float(errors.std(ddof=1)),all_success=all(t['success'] for t in trials))
    metrics['elapsed_seconds']=time.perf_counter()-start
    # 汇总输出：机器可读 metrics.json、四组图与中文验证报告。
    write_json(out/'metrics.json',metrics)
    draw(out,cases,v,folded_v,inverse_v,detector,cfg,metrics)
    report(out,metrics)
    print('Finished. Metrics and report: '+str(out.resolve()),flush=True)


def draw(out,cases,v,fv,iv,d,cfg,metrics):
    # 图 1（4×3 面板，每算例一行）：
    #   左：展开谱与条件拟合（点线为 ±FSR/2 级次边界）；
    #   中：折叠真值 vs 非参数恢复（一个 FSR 内的周期信息才可唯一恢复）；
    #   右：含噪径向剖面与反演前向校验（半径不是普通卷积坐标，需完整算子）。
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    fig,axs=plt.subplots(4,3,figsize=(15,13),layout='constrained')
    for i,case in enumerate(cases):
        mask=abs(v)<10
        axs[i,0].plot(v[mask],case['source'][mask],color=BLUE,label='Unfolded incident')
        axs[i,0].plot(v[mask],case['fit']['spectrum'][mask],color=ORANGE,ls='--',label='Conditional fit')
        axs[i,0].axvline(-cfg.fsr_ghz/2,color=GRAY,lw=.7,ls=':');axs[i,0].axvline(cfg.fsr_ghz/2,color=GRAY,lw=.7,ls=':')
        axs[i,0].set(xlabel='Physical offset (GHz)',ylabel='Density (1/GHz)',title=case['title'])
        axs[i,1].plot(fv,case['folded'],color=BLUE,label='Folded truth')
        axs[i,1].plot(iv,case['inverse']['spectrum'],color=ORANGE,label='Nonnegative inverse')
        axs[i,1].set(xlabel='One FSR offset (GHz)',ylabel='Density (1/GHz)',title='Periodic information: unfolded spectrum is not unique')
        axs[i,2].plot(d.bin_radius_mm,case['y'],color=GRAY,lw=.8,label='Noisy annular mean')
        axs[i,2].plot(d.bin_radius_mm,case['inverse']['fitted'],color=ORANGE,ls='--',label='Inverse forward check')
        axs[i,2].set(xlabel='Radius (mm)',ylabel='Relative irradiance',title='Measured radius is not a convolution coordinate')
        for a in axs[i]:a.legend(fontsize=8);a.grid(alpha=.15)
    fig.savefig(out/'water_spectra_and_recovery.png');plt.close(fig)
    # 图 2：90°/180° 两个算例的含噪二维干涉环图（青色圆为 ROI 边界）。
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained');extent=[-d.camera.size*d.pixel_mm/2,d.camera.size*d.pixel_mm/2]*2
    for a,case in zip(axs,cases[:2]):
        display=(case['observed']-metrics['configuration']['background_electrons'])/metrics['configuration']['gain_electrons']
        im=a.imshow(display,origin='lower',extent=extent,cmap='magma',vmin=0,vmax=.32)
        a.add_patch(plt.Circle((0,0),d.camera.roi_mm,fill=False,color='cyan',lw=.8))
        a.set(title=case['title']+' / noisy image',xlabel='x (mm)',ylabel='y (mm)');fig.colorbar(im,ax=a,label='Relative signal')
    fig.savefig(out/'water_interference_images.png');plt.close(fig)
    # 图 3：失配敏感性的 B 偏差与残差 χ² 条形图（标签与失配列表按序对应）。
    names=list(metrics['sensitivity']);bias=[metrics['sensitivity'][name]['spacing_bias_mhz'] for name in names]
    chi=[metrics['sensitivity'][name]['chi2'] for name in names]
    labels=['Phase +100 MHz','Focal length +1%','Finesse 25','Finesse 35','PSF assumed zero','PSF assumed 8 um','Cavity index assumed 1.46',
        'Angular sigma omitted: 2 deg','Extra elastic omitted: 10%','Laser doublet omitted: 320 MHz','Laser width assumed 10 / true 100 MHz']
    fig,axs=plt.subplots(1,2,figsize=(15,6),layout='constrained')
    axs[0].barh(labels,bias,color=ORANGE);axs[0].axvline(0,color=GRAY,lw=.7);axs[0].set(xlabel='Brillouin spacing bias (MHz)',title='Synthetic model mismatch')
    axs[1].barh(labels,chi,color=BLUE);axs[1].set(xscale='log',xlabel='Mean squared standardized residual',title='Residuals can hide parameter ambiguity')
    fig.savefig(out/'water_mismatch_sensitivity.png');plt.close(fig)
    # 图 4：一级环半径 vs 焦距（90°/180°、红/蓝分量）与
    # 单 FSR 外功率 vs 水内散射角（洛伦兹尾越级次边界的情况）。
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for angle in [90,180]:
        rows=[r for r in metrics['coverage_scans'] if r['cavity_index']==1 and r['center_detuning_ghz']==7.5 and r['water_angle_deg']==angle]
        for component,color in [('red',ORANGE),('blue',BLUE)]:
            axs[0].plot([r['focal_length_mm'] for r in rows],[next(x['radius_mm'] for x in r['rings'] if x['component']==component and x['order']==0) for r in rows],
                marker='o',color=color,ls='-' if angle==90 else '--',label=f'{angle} deg / {component}')
    axs[0].axhline(d.camera.roi_mm,color=GRAY,ls=':',label='ROI radius');axs[0].set(xlabel='Focal length (mm)',ylabel='First-order radius (mm)',title='Air-cavity example; cavity is still unknown');axs[0].legend(fontsize=8)
    rows=metrics['physical_scans']['water_angle_deg']
    axs[1].plot([r['value'] for r in rows],[100*r['area_outside_single_fsr'] for r in rows],marker='o',color=PINK)
    axs[1].set(xlabel='Water-internal scattering angle (deg)',ylabel='Incident power outside one FSR (%)',title='Lorentzian tails cross order boundaries')
    fig.savefig(out/'water_fov_and_fsr.png');plt.close(fig)


def report(out,m):
    # 生成中文验证报告「验证结果.md」：算例表、数值核对、视场/失配/
    # 蒙特卡洛要点（完整明细以 metrics.json 为准）。
    b=next(iter(m['cases'].values()));back=m['cases']['water_180deg'];p=b['physics'];mc=m['monte_carlo'];cfg=m['configuration']
    lines=['# 常压液态水模拟运行验证结果','',
        '这是合成数据验证；已确认散射介质为水，FP 外部为空气，腔内介质尚未知。',
        f"本次参考：纯水 {cfg['water']['temperature_k']-273.15:g} °C、{cfg['water']['pressure_mpa']:g} MPa、{cfg['water']['wavelength_nm']:g} nm、水内 {cfg['water']['scattering_angle_deg']:g}°；假设腔相位折射率 {cfg['instrument']['cavity_phase_index']:g}、焦距 {cfg['instrument']['focal_length_mm']:g} mm，其他条件见 parameters.json。",
        '',f"IAPWS 参考值：n={p['n_water']:.8f}，ρ={p['rho_kg_m3']:.4f} kg/m³，声速={p['sound_ms']:.4f} m/s；γ={p['gamma']:.8f}。体黏度参考值 {cfg['water']['bulk_viscosity_pa_s']*1000:g} mPa·s 仅为扫描中心值，不是实测值。",'',
        '| 算例 | B 频移 / GHz | B 本征 HWHM / GHz | 非参数折叠谱 L2 | 条件拟合 B / GHz | 条件拟合 χ² | 单 FSR 外功率 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,case in m['cases'].items():
        p=case['physics'];fit=case['parametric']
        lines.append(f"| {name} | {p['brillouin_ghz']:.6f} | {p['brillouin_hwhm_ghz']:.6f} | {case['nonparametric']['relative_l2_folded']:.2%} | {fit['parameters']['brillouin_ghz']:.6f} | {fit['chi2']:.3f} | {case['area_outside_single_fsr']:.2%} |")
    lines+=['','其中 HWHM 是半高半宽，FWHM 为其两倍。中心热谱很窄，非参数重建不保证其本征线宽；中心强度采用密度涨落近似并提供独立光学权重覆盖。',
        '',f"视场半径 {cfg['camera']['roi_mm']:g} mm：参考角度和 180° 两侧峰中心可见性分别为 {b['coverage']['both_side_peaks_visible']}、{back['coverage']['both_side_peaks_visible']}。扫描角度跨度约 {b['coverage']['angular_scan_fsr']:.3f} 个 FSR；不同腔介质、焦距和相位可改变结论，详细见 coverage_scans.json。",'',
        f"宽频积分所失功率：参考算例 {b['wide_window_missing_area']:.4%}，180° {back['wide_window_missing_area']:.4%}；未重新归一化掩盖尾损失。折叠谱正演与展开谱正演差异：参考算例 {b['folded_forward_relative_l2']:.4%}。",
        '',f"频率步长减半后环剖面相对差异 {m['numerical_checks']['half_step_profile_relative_l2']:.3g}；独立标量积分透过率最大绝对差 {m['numerical_checks']['independent_response_max_abs_error']:.3g}；二维像素图环均值与矩阵正演差 {b['image_operator_max_abs_error']:.3g}。",'',
        f"各 {len(mc['reference_counts']['trials'])} 次噪声重复：参考计数下条件拟合 B 误差标准差 {mc['reference_counts']['spacing_std_mhz']:.3f} MHz，低计数下 {mc['low_counts']['spacing_std_mhz']:.3f} MHz。这些只反映固定模型下的随机误差。",'',
        '| 假设失配 | B 偏差 / MHz | HWHM 偏差 / MHz | 整谱频移偏差 / MHz | χ² | 成功 / 边界 |','|---|---:|---:|---:|---:|---|']
    for name,row in m['sensitivity'].items():lines.append(f"| {name} | {row['spacing_bias_mhz']:.3f} | {row['width_bias_mhz']:.3f} | {row['shift_bias_mhz']:.3f} | {row['chi2']:.3f} | {row['success']} / {row['at_bound']} |")
    lines+=['','相位与整谱频移有近乎退化关系；在未知级次、未知激光模式下，只靠一幅环图无法唯一确定展开谱及绝对频移。没有把独立标定作为运行前提，先在参数化情景中检验可辨识性。',
        '',f"模拟计算耗时 {m['elapsed_seconds']:.1f} s（不含画图与打包）。详细参数、正则化路径、拟合边界及软件版本见 metrics.json。"]
    (out/'验证结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    # 命令行：--config 读取 JSON 覆盖默认（拒绝未知字段、嵌套 dict 浅合并），
    # --output 指定输出目录（默认 output/water_rb_fp）。
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path,default=Path('output/water_rb_fp'))
    args=parser.parse_args();config=json.loads(json.dumps(DEFAULTS))
    if args.config:
        supplied=json.loads(args.config.read_text(encoding='utf-8'))
        unknown=set(supplied)-set(config)
        if unknown:raise ValueError('Unknown configuration fields: '+str(unknown))
        for key,value in supplied.items():
            if isinstance(config[key],dict):config[key].update(value)
            else:config[key]=value
    run(config,args.output)
