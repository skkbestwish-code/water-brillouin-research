"""Run the reproducible RB/FP validation suite; figures and metrics are saved.

Usage: python run_simulation.py [--config parameters.json] [--output DIRECTORY]
Only numpy/scipy/matplotlib are needed; sources/ is never modified.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,time
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import scipy
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulation.physics import (Instrument,Gas,frequency_grid,rb_spectrum,gas_parameters,
    gaussian,normalize,convolve_laser,fp_kernel,illumination,airy,resonance_radius_mm)
from simulation.imaging import Camera,Detector,noisy_profile
from simulation.inverse import reconstruct,fit_parametric,wiener_reconstruct

DEFAULTS=dict(instrument=asdict(Instrument()),camera=asdict(Camera()),gas=asdict(Gas()),
              laser_fwhm_ghz=.01,laser_modes=[[0.,1.]],gain_electrons=2000.,
              background_electrons=20.,read_noise_electrons=3.,seed=20260930,
              truth_grid_size=3000,inverse_grid_size=600,fit_grid_size=2400,
              monte_carlo_replicates=12,low_count_gain=25.)
BLUE='#214d74';ORANGE='#c66c28';GRAY='#545454';PINK='#ae5a83'


def jsonable(obj):
    if isinstance(obj,np.ndarray):return obj.tolist()
    if isinstance(obj,(np.floating,np.integer,np.bool_)):return obj.item()
    if isinstance(obj,dict):return {k:jsonable(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [jsonable(v) for v in obj]
    return obj


def summarize(v,s,truth_v,truth,centers=None):
    interpolated=np.interp(truth_v,v,s,period=(v[1]-v[0])*len(v))
    result=dict(relative_l2=float(np.linalg.norm(interpolated-truth)/np.linalg.norm(truth)),
                recovered_area=float(s.sum()*(v[1]-v[0])))
    if centers:
        peaks,_=find_peaks(s,prominence=.02*max(s))
        matched=[]
        for center in centers:
            candidates=[i for i in peaks if abs(v[i]-center)<.3]
            if candidates:
                i=min(candidates,key=lambda i:abs(v[i]-center))
                matched.append(dict(true_ghz=float(center),recovered_ghz=float(v[i]),error_mhz=float(1000*(v[i]-center))))
            else:matched.append(dict(true_ghz=float(center),recovered_ghz=None,error_mhz=None))
        result['matched_peaks']=matched
    return result


def safe_fit(fit):
    return {k:v for k,v in fit.items() if k not in ['spectrum','fitted']}


def save_csv(path,columns,names):
    np.savetxt(path,np.column_stack(columns),delimiter=',',header=','.join(names),comments='')


def convolution_comparison(detector,v,s,cfg):
    df=v[1]-v[0];h=airy(v,cfg)
    conv=np.fft.fftshift(np.fft.ifft(np.fft.fft(np.fft.ifftshift(s))*np.fft.fft(np.fft.ifftshift(h))).real)*df
    x=cfg.nu0_ghz*(detector.radius_mm/cfg.focal_length_mm)**2/2-cfg.center_detuning_ghz
    irradiance=np.interp(x,v,conv,period=cfg.fsr_ghz)*illumination(detector.radius_mm,cfg)
    fft_profile=detector.profile(irradiance)
    exact=detector.kernel(v,cfg)@s
    small=detector.kernel(v,cfg,small_angle=True)@s
    return dict(relative_l2_small_angle=float(np.linalg.norm(small-exact)/np.linalg.norm(exact)),
                relative_l2_fft_interpolation=float(np.linalg.norm(fft_profile-small)/np.linalg.norm(small))),exact,small,fft_profile


def run(config,out):
    started=time.perf_counter();out.mkdir(parents=True,exist_ok=True)
    (out/'parameters.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    cfg=Instrument(**config['instrument']);camera=Camera(**config['camera']);gas=Gas(**config['gas'])
    detector=Detector(camera)
    vtruth=frequency_grid(cfg.fsr_ghz,config['truth_grid_size'])
    vinv=frequency_grid(cfg.fsr_ghz,config['inverse_grid_size'])
    vfit=frequency_grid(cfg.fsr_ghz,config['fit_grid_size'])
    print('Building truth, inverse and fit forward operators...',flush=True)
    ktruth=detector.kernel(vtruth,cfg);kinv=detector.kernel(vinv,cfg);kfit=detector.kernel(vfit,cfg)
    gain=config['gain_electrons'];bg=config['background_electrons'];rn=config['read_noise_electrons'];seed=config['seed']
    cases=[]
    for label,pressure,kind in [('low_pressure',.02,'knudsen'),('high_pressure',20.,'hydrodynamic')]:
        intrinsic,meta=rb_spectrum(vtruth,replace(gas,pressure_bar=pressure),kind)
        cases.append(dict(label=label,title=f'N2 {pressure:g} bar: {kind} limit',kind=kind,intrinsic=intrinsic,meta=meta))
    # An explicit algorithm benchmark, not a kinetic gas model.
    weights=[2/7,5/14,5/14];centers=[.08,-.72,.88]
    toy=sum(w*gaussian(vtruth,mu,.16,cfg.fsr_ghz) for w,mu in zip(weights,centers))
    cases.append(dict(label='three_peak_benchmark',title='Three-Gaussian algorithm benchmark',kind='benchmark',
                      intrinsic=normalize(toy,vtruth),meta=dict(model='algorithm benchmark',sigma_ghz=.16,centers_ghz=centers,weights=weights)))
    metrics=dict(configuration=config,instrument_derived=dict(optical_length_mm=cfg.optical_length_mm,
        airy_coefficient=cfg.airy_coefficient,ideal_reflectivity=cfg.ideal_reflectivity,
        fwhm_ghz=cfg.fsr_ghz/cfg.finesse,nu0_thz=cfg.nu0_ghz/1000,
        reference_ring_radii_mm=[resonance_radius_mm(p,0,cfg) for p in range(3)]),
        software=dict(python=__import__('sys').version,numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__),
        cases={},sensitivity={},monte_carlo={},numerical_checks={},wiener_control={})
    for ci,case in enumerate(cases):
        print(f'Running {case["label"]}: nonnegative inversion and forward fit...',flush=True)
        incident=convolve_laser(case['intrinsic'],vtruth,config['laser_fwhm_ghz'],config['laser_modes'])
        exact=ktruth@incident
        irradiance=(fp_kernel(detector.radius_mm,vtruth,cfg)@incident)*(vtruth[1]-vtruth[0])*illumination(detector.radius_mm,cfg)
        image=detector.image(irradiance)
        agreement=float(np.max(np.abs(detector.annular_mean(image)-exact)))
        rng=np.random.default_rng(seed+100+ci)
        observed_image=rng.poisson(np.maximum(gain*image+bg,0))+rng.normal(0,rn,image.shape)
        # This displayed noisy pixel image is the actual input to the inversion.
        y=(detector.annular_mean(observed_image)-bg)/gain
        sigma=np.sqrt((gain*exact+bg+rn**2)/detector.counts)/gain
        recovery=reconstruct(kinv,y,sigma,vinv)
        fit=None
        if case['kind']!='benchmark':
            fit=fit_parametric(kfit,y,sigma,vfit,case['kind'],config['laser_fwhm_ghz'],gas.gamma,config['laser_modes'])
        phase_centers=case['meta'].get('centers_ghz')
        if case['kind']=='hydrodynamic':
            p=case['meta'];phase_centers=[p['shift_ghz']-p['brillouin_ghz'],p['shift_ghz'],p['shift_ghz']+p['brillouin_ghz']]
        summary=summarize(vinv,recovery['spectrum'],vtruth,incident,phase_centers)
        summary.update(chi2=recovery['chi2'],alpha=recovery['alpha'],converged=recovery['converged'],
                       discrepancy_reached=recovery['discrepancy_reached'],regularization_path=recovery['path'])
        case.update(incident=incident,exact=exact,y=y,sigma=sigma,recovery=recovery,fit=fit,image=image,observed_image=observed_image)
        metrics['cases'][case['label']]=dict(physics=case['meta'],nonparametric=summary,
            parametric=safe_fit(fit) if fit else None,image_operator_max_abs_error=agreement)
        save_csv(out/(case['label']+'_spectrum.csv'),[vtruth,case['intrinsic'],incident,np.interp(vtruth,vinv,recovery['spectrum'],period=cfg.fsr_ghz)],
                 ['frequency_offset_GHz','intrinsic_RB_density_per_GHz','incident_density_per_GHz','recovered_incident_density_per_GHz'])
        save_csv(out/(case['label']+'_profile.csv'),[detector.bin_radius_mm,exact,y,sigma,recovery['fitted']],
                 ['radius_mm','noiseless_mean_intensity','observed_mean_intensity','noise_sigma','reconstructed_profile'])
        np.save(out/(case['label']+'_expected_image.npy'),image)
        np.save(out/(case['label']+'_observed_image_electrons.npy'),observed_image)
        print(f'  shape L2={summary["relative_l2"]:.4f}; chi2={recovery["chi2"]:.3f}; alpha={recovery["alpha"]:g}',flush=True)

    print('Ideal frequency-domain Wiener controls (camera excluded)...',flush=True)
    transfer=np.fft.fft(np.fft.ifftshift(airy(vtruth,cfg)))*(vtruth[1]-vtruth[0])
    control_sigma=.001
    for ci,case in enumerate(cases):
        clean=np.fft.fftshift(np.fft.ifft(np.fft.fft(np.fft.ifftshift(case['incident']))*transfer).real)
        yy=clean+np.random.default_rng(seed+3000+ci).normal(0,control_sigma,len(clean))
        wr=wiener_reconstruct(vtruth,yy,airy(vtruth,cfg),control_sigma)
        sm=summarize(vtruth,wr['spectrum'],vtruth,case['incident'])
        sm.update(regularizer=wr['regularizer'],chi2=wr['chi2'],negative_area=wr['negative_area'],noise_sigma=control_sigma,
                  selection=wr['selection'],gcv=wr['gcv'],regularization_path=wr['path'],
                  note='Ideal uniformly sampled frequency-domain convolution; excludes camera, pixel integration and illumination. Not the observed ring-image inverse.')
        metrics['wiener_control'][case['label']]=sm
        case['wiener_spectrum']=wr['spectrum']
        save_csv(out/(case['label']+'_wiener_control.csv'),[vtruth,case['incident'],yy,wr['spectrum']],
                 ['frequency_offset_GHz','known_incident_density','noisy_convolved_intensity','Wiener_recovered_density'])

    # Sensitivity uses the same high-pressure observation; changing one assumed
    # instrument parameter at a time isolates calibration errors.
    high=cases[1];sensitivity=[('nominal',cfg,detector,gain),
        ('low_counts_25',cfg,detector,config['low_count_gain']),
        ('focal_length_plus_1pct',replace(cfg,focal_length_mm=cfg.focal_length_mm*1.01),detector,gain),
        ('finesse_assumed_25',replace(cfg,finesse=25),detector,gain),
        ('psf_ignored',cfg,Detector(replace(camera,psf_sigma_um=0)),gain),
        ('detuning_plus_0p1GHz',replace(cfg,center_detuning_ghz=cfg.center_detuning_ghz+.1),detector,gain)]
    for name,assumed,det,g in sensitivity:
        print(f'Sensitivity: {name}',flush=True)
        if name=='nominal':rec=high['recovery'];fit=high['fit'];yy=high['y'];ss=high['sigma']
        else:
            yy,ss=noisy_profile(high['exact'],detector.counts,g,bg,rn,seed+1) if g!=gain else (high['y'],high['sigma'])
            inv=kinv if assumed==cfg and det is detector else det.kernel(vinv,assumed)
            fk=kfit if assumed==cfg and det is detector else det.kernel(vfit,assumed)
            rec=reconstruct(inv,yy,ss,vinv)
            fit=fit_parametric(fk,yy,ss,vfit,'hydrodynamic',config['laser_fwhm_ghz'],gas.gamma,config['laser_modes'])
        sm=summarize(vinv,rec['spectrum'],vtruth,high['incident'])
        sm.update(chi2=rec['chi2'],alpha=rec['alpha'],discrepancy_reached=rec['discrepancy_reached'],
                  parametric=safe_fit(fit),assumed_instrument=asdict(assumed),assumed_camera=asdict(det.camera),gain=g)
        metrics['sensitivity'][name]=sm
        save_csv(out/('sensitivity_'+name+'.csv'),[vinv,rec['spectrum']],['frequency_offset_GHz','recovered_incident_density_per_GHz'])

    for ci,case in enumerate(cases[:2]):
        print(f'Monte Carlo conditional fits: {case["label"]}',flush=True)
        rows=[]
        for rep in range(config['monte_carlo_replicates']):
            yy,ss=noisy_profile(case['exact'],detector.counts,gain,bg,rn,seed+1000+ci*100+rep)
            ff=fit_parametric(kfit,yy,ss,vfit,case['kind'],config['laser_fwhm_ghz'],gas.gamma,config['laser_modes'])
            rows.append(ff['parameters'])
        stats={name:dict(mean=float(np.mean([r[name] for r in rows])),std=float(np.std([r[name] for r in rows],ddof=1))) for name in rows[0]}
        metrics['monte_carlo'][case['label']]=dict(replicates=len(rows),parameter_statistics=stats,
            note='Empirical noise scatter, conditional on known laser/instrument and exact limiting line family; not full experimental uncertainty.')
        (out/('monte_carlo_'+case['label']+'.json')).write_text(json.dumps(jsonable(rows),indent=2),encoding='utf-8')

    comparison,exact,small,fft=convolution_comparison(detector,vtruth,cases[0]['incident'],cfg)
    metrics['numerical_checks']['convolution_approximation']=comparison
    # Integration convergence: same physical density sampled on a new fine grid.
    vref=frequency_grid(cfg.fsr_ghz,4500);sref,_=rb_spectrum(vref,replace(gas,pressure_bar=20.),'hydrodynamic')
    sref=convolve_laser(sref,vref,config['laser_fwhm_ghz'],config['laser_modes'])
    yref=detector.kernel(vref,cfg)@sref
    metrics['numerical_checks']['high_pressure_frequency_quadrature_relative_l2']=float(np.linalg.norm(yref-high['exact'])/np.linalg.norm(yref))
    # A different camera quadrature checks subpixel/radial discretization.
    fine_detector=Detector(replace(camera,radial_step_um=camera.radial_step_um/2,subpixels=6))
    yf=fine_detector.kernel(vtruth,cfg)@high['incident']
    metrics['numerical_checks']['camera_quadrature_relative_l2']=float(np.linalg.norm(yf-high['exact'])/np.linalg.norm(yf))
    metrics['runtime_seconds']=time.perf_counter()-started
    (out/'metrics.json').write_text(json.dumps(jsonable(metrics),ensure_ascii=False,indent=2),encoding='utf-8')
    make_figures(out,cases,detector,vinv,vtruth,vfit,cfg,metrics,(exact,small,fft))
    write_report(out,metrics)
    print(f'Done. Results in {out.resolve()}; compute seconds={metrics["runtime_seconds"]:.1f}',flush=True)
    return metrics


def style_axis(ax):
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15,color='#777777')
    ax.tick_params(labelsize=10)


def make_figures(out,cases,detector,vinv,vtruth,vfit,cfg,metrics,comparison):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,'figure.facecolor':'white'})
    fig,axs=plt.subplots(3,3,figsize=(17,12),layout='constrained')
    extent=detector.camera.size*detector.pixel_mm/2
    for row,case in enumerate(cases):
        ax=axs[row,0];ax.plot(vtruth,case['incident'],color=BLUE,lw=1.8,label='Known input')
        ax.plot(vinv,case['recovery']['spectrum'],color=ORANGE,lw=1.7,label='Nonparametric recovery')
        if case['fit']:
            ax.plot(vfit,case['fit']['spectrum'],color=GRAY,lw=1.2,ls='--',label='Conditional forward fit')
        ax.set(xlim=(-2,2),xlabel='Frequency offset (GHz)',ylabel='Spectral density (1/GHz)',title=case['title'])
        ax.legend(fontsize=9);style_axis(ax)
        ax=axs[row,1]
        im=ax.imshow(case['image'],extent=[-extent,extent,-extent,extent],origin='lower',cmap='gray',vmin=0,vmax=max(case['image'].max(),1e-12))
        ax.set(xlabel='Image x (mm)',ylabel='Image y (mm)',title='Expected image, before counting noise')
        fig.colorbar(im,ax=ax,shrink=.7,label='Relative pixel intensity')
        ax=axs[row,2];r=detector.bin_radius_mm
        ax.plot(r,case['exact'],color=BLUE,lw=1.7,label='Exact forward model')
        ax.scatter(r,case['y'],s=5,color='#777777',alpha=.45,label='Noisy annular mean')
        ax.plot(r,case['recovery']['fitted'],color=ORANGE,lw=1.2,ls='--',label='Recovered spectrum re-imaged')
        ax.set(xlabel='Radius (mm)',ylabel='Mean intensity',title=f'Re-image closure: chi2 = {case["recovery"]["chi2"]:.2f}')
        ax.legend(fontsize=9);style_axis(ax)
    fig.suptitle('RB spectrum → imaging FP → spectral recovery | FSR 15 GHz, finesse 30',fontsize=17)
    fig.savefig(out/'overview.png',dpi=150);plt.close(fig)

    fig,axs=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
    for ax,case in zip(axs,cases):
        ax.scatter(detector.bin_radius_mm,(case['recovery']['fitted']-case['y'])/case['sigma'],s=7,color=BLUE)
        ax.axhspan(-1,1,color='#dddddd',alpha=.6);ax.axhline(0,color=GRAY,lw=.8)
        ax.set(xlabel='Radius (mm)',ylabel='Residual / known noise sigma',title=case['title']);style_axis(ax)
    fig.suptitle('Standardized residuals: a small image residual does not prove correct fine linewidth',fontsize=14)
    fig.savefig(out/'residuals.png',dpi=150);plt.close(fig)

    fig,axs=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    for ax,(name,sm) in zip(axs.flat,metrics['sensitivity'].items()):
        data=np.loadtxt(out/('sensitivity_'+name+'.csv'),delimiter=',',skiprows=1)
        ax.plot(vtruth,cases[1]['incident'],color=BLUE,lw=1.5,label='Known high-pressure input')
        ax.plot(data[:,0],data[:,1],color=ORANGE,lw=1.5,label='Nonparametric recovery')
        ax.set(xlim=(-2,2),xlabel='Frequency offset (GHz)',ylabel='Density (1/GHz)',
               title=f'{name.replace("_"," ")}\nshape error {sm["relative_l2"]:.1%}, chi2 {sm["chi2"]:.1f}')
        style_axis(ax)
    axs.flat[0].legend(fontsize=9)
    fig.suptitle('High-pressure recovery sensitivity (one change at a time)',fontsize=16)
    fig.savefig(out/'sensitivity.png',dpi=150);plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    v=np.linspace(-1.5,1.5,1500);axs[0].plot(v,airy(v,cfg),color=BLUE)
    axs[0].axhline(.5,color=GRAY,ls='--',lw=1)
    half_width=cfg.fsr_ghz/(2*cfg.finesse)
    axs[0].axvline(-half_width,color=ORANGE,ls=':',lw=1);axs[0].axvline(half_width,color=ORANGE,ls=':',lw=1)
    axs[0].set(xlabel='Detuning (GHz)',ylabel='FP transmission',title=f'Airy peak: FWHM = {2*half_width:.3f} GHz');style_axis(axs[0])
    for y,label,color,ls in zip(comparison,['Exact angle integral','Small-angle response','FFT convolution in r² coordinate'],[BLUE,ORANGE,GRAY],['-','--',':']):
        axs[1].plot(detector.bin_radius_mm,y,label=label,color=color,ls=ls)
    axs[1].set(xlabel='Radius (mm)',ylabel='Mean intensity',title='Convolution approximation: low-pressure case');axs[1].legend(fontsize=9);style_axis(axs[1])
    fig.savefig(out/'instrument_and_convolution.png',dpi=150);plt.close(fig)

    fig,ax=plt.subplots(figsize=(6,6),layout='constrained')
    ax.imshow(cases[1]['observed_image'],cmap='gray',origin='lower',extent=[-extent,extent,-extent,extent])
    ax.set(xlabel='Image x (mm)',ylabel='Image y (mm)',title='High-pressure image: Poisson + read noise\nAssumed electron scale; not predicted experimental brightness')
    fig.savefig(out/'high_pressure_noisy_image.png',dpi=150);plt.close(fig)

    fig,axs=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
    for ax,case in zip(axs,cases):
        ax.plot(vtruth,case['incident'],color=BLUE,lw=1.6,label='Known input')
        ax.plot(vtruth,case['wiener_spectrum'],color=ORANGE,lw=1.4,label='Wiener control recovery')
        ax.axhline(0,color=GRAY,lw=.8)
        ax.set(xlim=(-2,2),xlabel='Frequency offset (GHz)',ylabel='Spectral density (1/GHz)',title=case['title']);style_axis(ax)
    axs[0].legend(fontsize=9)
    fig.suptitle('Ideal frequency-domain Wiener controls | camera and pixel effects excluded',fontsize=14)
    fig.savefig(out/'wiener_control.png',dpi=150);plt.close(fig)


def write_report(out,m):
    derived=m['instrument_derived'];lo=m['cases']['low_pressure'];hi=m['cases']['high_pressure']
    lp=lo['parametric']['parameters'];hp=hi['parametric']['parameters']
    true_l=lo['physics'];true_h=hi['physics']
    temperature_fit=m['configuration']['gas']['temperature_k']*(lp['sigma_ghz']/true_l['sigma_ghz'])**2
    conv=m['numerical_checks']['convolution_approximation']
    conf=m['configuration'];gc=conf['gas'];ic=conf['instrument'];cc=conf['camera']
    lines=['# 瑞利–布里渊散射 / FP 成像：第一版验证结果','',
       f'这是合成数据的物理与数值验证，尚未使用实验图像。FSR={ic["fsr_ghz"]:g} GHz、FP 谱精细度={ic["finesse"]:g}；空气腔和配置中的 He–Ne 激光谱为启动假设。', '',
       '## 当前假设','',
       f'- 气体摩尔质量{gc["molar_mass_kg"]*1000:g} g/mol（默认氮气）；{gc["temperature_k"]:g} K，{gc["scattering_angle_deg"]:g}° 散射角；散射波矢方向速度 {gc["velocity_ms"]:g} m/s；气体折射率 1。暂不积分接收锥角内的散射角变化。',
       f'- 激光 {ic["wavelength_nm"]:g} nm，每个纵模高斯 FWHM={conf["laser_fwhm_ghz"]*1000:g} MHz；纵模[频移GHz, 权重]={conf["laser_modes"]}。多纵模参数未知时不能把激光贡献归给气体。',
       f'- 成像焦距 {ic["focal_length_mm"]:g} mm，空气腔；中心失谐 {ic["center_detuning_ghz"]:g} GHz，照明包络半径 {ic["illumination_radius_mm"]:g} mm。',
       f'- 相机 {cc["size"]}×{cc["size"]}，{cc["pixel_um"]:g} um 像素，二维高斯 PSF sigma={cc["psf_sigma_um"]:g} um；{cc["subpixels"]}×{cc["subpixels"]} 子像素积分，分析半径 {cc["roi_mm"]:g} mm。',
       f'- 假设计数尺度{conf["gain_electrons"]:g}电子/相对强度单位/像素、背景{conf["background_electrons"]:g}电子、读出噪声{conf["read_noise_electrons"]:g}电子。未计算绝对散射功率与曝光需求。',
       f'- 剪切黏度{gc["shear_viscosity_pa_s"]:g} Pa s、热导率{gc["thermal_conductivity_w_mk"]:g} W/(m K)、体积黏度{gc["bulk_viscosity_pa_s"]:g}为示例；高压谱宽会依赖这些参数。',
       '- 0.02 bar 用 Knudsen 极限；20 bar 用流体力学极限；三高斯谱仅是算法测试。没有实现 Tenti S6，也不对常压作精确谱形预测。', '',
       '## 前向模型','',
       '输入气体本征谱先与激光谱卷积，然后积分 FP 的频率/角度响应。二维相机模糊采用带 Bessel I0 因子和半径面积权重的轴对称高斯卷积；随后积分像素面积，计算圆环平均强度。环总计数和平均强度明确区分。','',
       f'- Airy 系数={derived["airy_coefficient"]:.6f}；理想反射率={derived["ideal_reflectivity"]:.6f}；光学长度={derived["optical_length_mm"]:.6f} mm。',
       f'- 仪器半高全宽={derived["fwhm_ghz"]:.3f} GHz，不能把精细度30直接作为 Airy 系数。',
       f'- 生成、非参数反演、参数拟合分别采用{conf["truth_grid_size"]}、{conf["inverse_grid_size"]}、{conf["fit_grid_size"]}个频率点；避免只在同一个离散算子上演示恢复。','',
       '## 运行结果','',
       '| 测试 | y | 非参数谱形相对 L2 误差 | 重新成像 chi² |',
       '|---|---:|---:|---:|']
    for name,label in [('low_pressure','低压高斯极限'),('high_pressure','高压三洛伦兹极限'),('three_peak_benchmark','三高斯算法测试')]:
        c=m['cases'][name];r=c['nonparametric'];yy=c['physics'].get('y')
        lines.append(f'| {label} | {yy:.3f} | {r["relative_l2"]:.2%} | {r["chi2"]:.3f} |' if yy is not None else f'| {label} | 不适用 | {r["relative_l2"]:.2%} | {r["chi2"]:.3f} |')
    lines+=['', '非参数方法恢复的是“气体谱与激光谱卷积后的入射谱”。它不假设三峰形状，加入非负约束与二阶平滑，按已知噪声选择正则化强度。chi² 是平均标准化残差平方，接近1表示残差与所设噪声相容；谱形仍可能存在明显误差。','',
       '另运行了Wiener反卷积对照：在均匀频率坐标中加入sigma=0.001的独立高斯噪声，再按广义交叉验证（GCV）选择正则化，不使用真实频谱调参。早期仅按chi²≤1选参数，在三峰测试中追逐随机噪声方差而造成严重振铃；固定该失败案例的回归检查后改用GCV。该对照排除了相机、像素和照明影响，不能与带噪环图的恢复误差直接排名；保留负值旁瓣，未裁剪美化。每个控制结果保存为wiener_control CSV，并记录负面积。','',
       '## 有谱型先验的前向拟合','',
       f'- 低压真实频移 {true_l["shift_ghz"]*1000:.3f} MHz，拟合 {lp["shift_ghz"]*1000:.3f} MHz；真实 Gaussian sigma {true_l["sigma_ghz"]:.6f} GHz，拟合 {lp["sigma_ghz"]:.6f} GHz。',
       f'- 在散射角、分子质量、激光谱和仪器函数完全已知的条件下，低压拟合温度 {temperature_fit:.3f} K（真实{gc["temperature_k"]:g} K）。',
       f'- 高压真实 Brillouin 偏移 {true_h["brillouin_ghz"]:.6f} GHz，拟合 {hp["brillouin_ghz"]:.6f} GHz。',
       f'- 高压中心谱线真实 FWHM {2*true_h["rayleigh_hwhm_ghz"]*1000:.3f} MHz，拟合 {2*hp["rayleigh_hwhm_ghz"]*1000:.3f} MHz。',
       f'- 高压侧峰谱线真实 FWHM {2*true_h["brillouin_hwhm_ghz"]*1000:.3f} MHz，拟合 {2*hp["brillouin_hwhm_ghz"]*1000:.3f} MHz。','',
       f'参数拟合假定正确的高斯/三洛伦兹极限谱族及已知谱线权重。其窄线宽估计依赖先验和标定，不能解读为仪器已经具备该宽度的普遍分辨率。Monte Carlo文件记录{conf["monte_carlo_replicates"]}次独立噪声拟合；这些散布仅是条件噪声误差，没有覆盖仪器参数或物理模型误差。','',
       '## 参数误差敏感性','',
       '| 仅改变一个条件 | 高压非参数谱形 L2 误差 | chi² | 达到噪声标准 |',
       '|---|---:|---:|---|']
    for name,sm in m['sensitivity'].items():
        lines.append(f'| {name} | {sm["relative_l2"]:.2%} | {sm["chi2"]:.2f} | {"是" if sm["discrepancy_reached"] else "否"} |')
    lines+=['', '错误焦距、中心失谐、精细度或相机 PSF 会被反演解释成谱形变化。低残差不能单独证明气体线宽正确；应使用窄线参考光标定完整仪器响应。尤其是失谐与整体谱移近似退化：中心失谐误设+100 MHz，恢复谱整体约移-100 MHz，仍能得到良好图样残差。','',
       '## 数值一致性','',
       f'- 小角度卷积近似相对准确角度模型的径向曲线 L2 误差：{conv["relative_l2_small_angle"]:.4%}。',
       f'- FFT 周期卷积相对小角度直接积分的误差：{conv["relative_l2_fft_interpolation"]:.4%}。',
       f'- 高压频率积分从{conf["truth_grid_size"]}点改用4500点，图样变化：{100*m["numerical_checks"]["high_pressure_frequency_quadrature_relative_l2"]:.3e}%。',
       f'- 径向网格从1 um加密到0.5 um，子像素4×4加密到6×6，图样变化：{m["numerical_checks"]["camera_quadrature_relative_l2"]:.5%}。','',
       '## 图和数据','',
       '![输入频谱、干涉环和恢复](overview.png)','',
       '![仪器函数与卷积](instrument_and_convolution.png)','',
       '![高压谱恢复敏感性](sensitivity.png)','',
       '![残差](residuals.png)','',
       '![Wiener理想频率域对照](wiener_control.png)','',
       'metrics.json 包含完整参数、恢复指标、正则化路径和条件拟合误差；CSV保存谱和径向曲线，NPY保存期望图像和带噪电子计数图。基准反演使用这张带噪图的圆环均值作为输入。parameters.json 可以作为下次运行的输入。','',
       '## 对真实实验还需补齐','',
       '气体种类、温度和压力，实际散射角及接收角范围，He–Ne纵模和线宽，FP为空气腔还是固体、有效精细度是否已包含相机，成像焦距/像素尺寸/PSF，参考光环与背景，曝光和计数水平。动力学区间可接入独立验证的Tenti S6频谱，而当前成像与反演模块可以复用。','',
       'FP谱响应具有级次歧义；本例将恢复范围限制在一个FSR中。低压和高压案例的亮度都归一化到相同假设计数水平，不能比较真实散射强弱。','',
       '来源：SPIE 89101E，doi:10.1117/12.2034394；Witschas et al., Applied Optics 49, 4217 (2010), https://elib.dlr.de/64895/1/204030.pdf。']
    (out/'验证报告.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path,default=Path('output/rb_fp'))
    args=parser.parse_args();config=json.loads(json.dumps(DEFAULTS))
    if args.config:
        supplied=json.loads(args.config.read_text(encoding='utf-8'))
        for key,value in supplied.items():
            if key in ['instrument','camera','gas']:config[key].update(value)
            else:config[key]=value
    run(config,args.output)
