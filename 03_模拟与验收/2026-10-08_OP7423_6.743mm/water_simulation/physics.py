"""Liquid-water spectra and FP geometry with distinct optical media.

Hz widths are converted from angular-frequency damping exactly once. The
three-line model is the weak-damping density-fluctuation hydrodynamic limit,
not a full optical/viscoelastic theory for every kind of water.

中文说明：本模块给出液态水散射谱（中心线 + 两个布里渊侧峰）以及
"水中散射 → 空气中 FP → 相机成像" 的几何与仪器模型。散射介质是水，
FP 腔内介质独立设置。频率轴用 GHz、谱密度用 1/GHz；声学阻尼的角频率
只在 Hz 换算处折算一次。三线洛伦兹是弱阻尼密度涨落（流体力学）极限，
不是对所有水体都成立的完整黏弹性理论。
"""
from dataclasses import dataclass,replace
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np
from scipy.special import voigt_profile
from numpy.polynomial.hermite import hermgauss
from simulation.physics import Instrument,airy,illumination,C
from simulation.imaging import Detector

# A local, versioned dependency is included with its GPL license. No network
# access is used at runtime; pip-installed iapws is also supported.
# 本地随包携带的 iapws（含 GPL 许可证），运行时无需联网；若已 pip 安装也兼容。
_deps=Path(__file__).resolve().parent.parent/'.water_deps'
if _deps.is_dir():sys.path.insert(0,str(_deps))
from iapws import IAPWS95


# 液态水散射配置：默认 20 °C、常压、水内 90° 散射。
# scattering_angle_deg 是水内散射角 Θ（不是空气中的收集角）；
# angular_sigma_deg 模拟有限收光孔径造成的角度展宽（高斯近似）；
# velocity_ms 是流速贡献（静止水为 0，不能沿用旧气体默认值）；
# bulk_viscosity_pa_s 是体黏度，直接决定布里渊线宽，不能默认为 0；
# elastic_fraction/elastic_hwhm_ghz 是额外弹性杂散光（颗粒、器壁等）；
# central_fraction_override 可覆盖 (γ-1)/γ 的本征中心分量权重。
@dataclass(frozen=True)
class Water:
    temperature_k: float=293.15
    pressure_mpa: float=.101325
    scattering_angle_deg: float=90.
    angular_sigma_deg: float=0.
    velocity_ms: float=0.
    bulk_viscosity_pa_s: float=.0025
    wavelength_nm: float=632.8
    elastic_fraction: float=0.
    elastic_hwhm_ghz: float=.001
    central_fraction_override: float|None=None

    # 仅允许 0–50 °C、常压范围（0.08–0.2 MPa）的液态水；其余参数须物理合法。
    def __post_init__(self):
        values=[v for v in self.__dict__.values() if v is not None]
        if not np.all(np.isfinite(values)):raise ValueError('Water parameters must be finite.')
        if not 0<self.scattering_angle_deg<=180 or self.angular_sigma_deg<0:
            raise ValueError('Water-internal scattering angle must be in (0,180].')
        if not 273.16<=self.temperature_k<=323.15 or not .08<=self.pressure_mpa<=.2:
            raise ValueError('This study is restricted to ambient-pressure liquid water, 0–50 C.')
        if self.bulk_viscosity_pa_s<0 or self.wavelength_nm<=0 or self.elastic_hwhm_ghz<=0:
            raise ValueError('Invalid transport, wavelength or elastic width.')
        if not 0<=self.elastic_fraction<=1 or (self.central_fraction_override is not None and not 0<=self.central_fraction_override<=1):
            raise ValueError('Spectral fractions must be in [0,1].')


# FP 仪器配置（默认腔内外都是空气）。若为固体标准具，把腔相位/群折射率
# 改成腔内对应值；腔厚由 FSR 反推、镀膜相位色散被忽略。
@dataclass(frozen=True)
class WaterFP(Instrument):
    center_detuning_ghz: float=7.5
    external_index: float=1.
    cavity_phase_index: float=1.
    cavity_group_index: float|None=None
    peak_transmission: float=1.

    def __post_init__(self):
        super().__post_init__()
        if min(self.external_index,self.cavity_phase_index,self.group_index)<=0:
            raise ValueError('FP indices must be positive.')
        values=[self.fsr_ghz,self.finesse,self.wavelength_nm,self.focal_length_mm,
                self.center_detuning_ghz,self.external_index,self.cavity_phase_index,
                self.group_index,self.peak_transmission,self.illumination_radius_mm]
        if not np.all(np.isfinite(values)) or not 0<self.peak_transmission<=1:
            raise ValueError('Finite parameters and peak transmission in (0,1] required.')

    # 群折射率未单独给出时退化为相位折射率（空气腔两者都是 1）。
    @property
    def group_index(self):
        return self.cavity_phase_index if self.cavity_group_index is None else self.cavity_group_index

    # 腔介质中的轴上载频修正项：ν0·n_phase/n_group。
    @property
    def angular_carrier_ghz(self):
        return self.nu0_ghz*self.cavity_phase_index/self.group_index

    # 由 FSR 反推腔厚 d = c/(2·n_g·FSR)，忽略镀膜相位色散。
    @property
    def cavity_thickness_mm(self):
        return C/(2*self.group_index*self.fsr_ghz*1e9)*1e3


# 由 IAPWS-95 计算水物性（密度、声速、cp/cv、γ、剪切黏度、热导率、
# 折射率、热膨胀系数），带缓存；非液态状态直接报错。
@lru_cache(maxsize=128)
def properties(water):
    state=IAPWS95(T=water.temperature_k,P=water.pressure_mpa,l=water.wavelength_nm/1000)
    if state.status!=1 or state.phase!='Liquid':raise ValueError('IAPWS state is not liquid.')
    return {key:float(value) for key,value in dict(rho_kg_m3=state.rho,
        sound_ms=state.w,cp_j_kg_k=state.cp*1000,cv_j_kg_k=state.cv*1000,
        gamma=state.cp/state.cv,shear_viscosity_pa_s=state.mu,
        thermal_conductivity_w_mk=state.k,n_water=state.n,
        thermal_expansion_per_k=state.alfav).items()}


# 计算散射波矢 q = 4π n sin(Θ/2)/λ0 与谱线参数：布里渊频移 q·v_s/2π、
# 瑞利线宽（热扩散主导）、布里渊线宽（剪切+体黏度+内部热模）、
# 本征中心分量权重 (γ-1)/γ（可被 central_fraction_override 覆盖）。
def line_parameters(water):
    p=properties(water)
    q=4*np.pi*p['n_water']*np.sin(np.deg2rad(water.scattering_angle_deg)/2)/(water.wavelength_nm*1e-9)
    dt=p['thermal_conductivity_w_mk']/(p['rho_kg_m3']*p['cp_j_kg_k'])
    center=(p['gamma']-1)/p['gamma'] if water.central_fraction_override is None else water.central_fraction_override
    return dict(**p,q_per_m=float(q),thermal_diffusivity_m2_s=dt,
        brillouin_ghz=float(q*p['sound_ms']/(2*np.pi*1e9)),
        shift_ghz=float(q*water.velocity_ms/(2*np.pi*1e9)),
        rayleigh_hwhm_ghz=float(dt*q*q/(2*np.pi*1e9)),
        brillouin_hwhm_ghz=float(((4*p['shear_viscosity_pa_s']/3+water.bulk_viscosity_pa_s)/p['rho_kg_m3']+
                               (p['gamma']-1)*dt)*q*q/(4*np.pi*1e9)),
        intrinsic_central_fraction=float(center),density_landau_placzek_ratio=p['gamma']-1,
        bulk_viscosity_pa_s=water.bulk_viscosity_pa_s,
        central_weight_model='density-fluctuation approximation; optical thermal coupling not fully modeled')


# 有限收光角分布：用 15 点 Gauss–Hermite 求积近似角度高斯展宽，
# 返回 (角度, 归一化权重)，并剔除采样出的非法角度。
def angle_samples(water):
    if water.angular_sigma_deg==0:return [(water.scattering_angle_deg,1.)]
    nodes,weights=hermgauss(15)
    angles=water.scattering_angle_deg+np.sqrt(2)*water.angular_sigma_deg*nodes
    valid=(angles>0)&(angles<=180)
    weights=weights[valid]/weights[valid].sum()
    return list(zip(angles[valid],weights))


# 规范化激光纵模表 [(频偏 GHz, 权重)]：权重非负、总和归一。
def _modes(modes):
    modes=np.asarray([[0.,1.]] if modes is None else modes,dtype=float)
    if modes.ndim!=2 or modes.shape[1]!=2 or not np.all(np.isfinite(modes)) or np.any(modes[:,1]<0) or modes[:,1].sum()<=0:
        raise ValueError('Laser modes require (offset_GHz, nonnegative_weight).')
    return modes[:,0],modes[:,1]/modes[:,1].sum()


# 水谱分量表：(中心 GHz, 洛伦兹 HWHM GHz, 积分权重)，总面积严格为 1。
# 每个角度样本给出一条中心线 + 一对 ±布里渊侧峰；最后附加一个固定于
# 0 频移的额外弹性分量（杂散/颗粒光，不随角度权重缩放）。
def components(water):
    """(centre GHz, Lorentz HWHM GHz, integrated weight); area exactly one."""
    lines=[]
    for angle,weight in angle_samples(water):
        p=line_parameters(replace(water,scattering_angle_deg=float(angle),angular_sigma_deg=0))
        a=(1-water.elastic_fraction)*weight;c=p['intrinsic_central_fraction']
        lines.append((p['shift_ghz'],p['rayleigh_hwhm_ghz'],a*c))
        for sign in [-1,1]:lines.append((p['shift_ghz']+sign*p['brillouin_ghz'],p['brillouin_hwhm_ghz'],a*(1-c)/2))
    # Additional wall/particle elastic light is stationary in the laser frame.
    lines.append((0.,water.elastic_hwhm_ghz,water.elastic_fraction))
    return lines


# 把分量表与激光线型卷积为 Voigt 谱（高斯激光 ⊗ 洛伦兹本征线），
# 支持多纵模（逐模频偏与权重加权）。
def spectrum_from_lines(v,lines,laser_fwhm_ghz=.01,laser_modes=None):
    if not np.isfinite(laser_fwhm_ghz) or laser_fwhm_ghz<0:raise ValueError('Laser FWHM must be finite and nonnegative.')
    sigma=laser_fwhm_ghz/(2*np.sqrt(2*np.log(2)));offsets,weights=_modes(laser_modes)
    s=np.zeros_like(np.asarray(v,dtype=float))
    for mu,width,area in lines:
        if area==0:continue
        for offset,weight in zip(offsets,weights):
            s+=area*weight*voigt_profile(v-mu-offset,sigma,width)
    return s


def physical_spectrum(v,water,laser_fwhm_ghz=.01,laser_modes=None):
    """Unfolded, absolutely normalized analytical Voigt densities; no truncation renormalization.

    中文：展开（非周期）的绝对归一化 Voigt 密度。落在给定频率窗口外的
    洛伦兹尾部不会通过重归一化被掩盖——宽频积分面积会如实小于 1。"""
    return spectrum_from_lines(np.asarray(v),components(water),laser_fwhm_ghz,laser_modes)


def folded_spectrum(v,water,laser_fwhm_ghz=.01,laser_modes=None):
    """Fourier-series periodic Voigt sum, including all spectral orders.

    The inverse target is this periodic information, not a unique unfolded
    spectrum. Sampling must resolve the narrowest plotted line.

    中文：用傅里叶级数把所有频率级次折进一个 FSR 的周期谱——这才是反演
    尝试恢复的周期信息，仍可能欠定，不能保证唯一恢复。傅里叶系数＝各分量洛伦兹特征函数 × 频移相位，再乘
    激光高斯特征函数与纵模因子；采样必须分辨最窄线型。
    """
    if laser_fwhm_ghz<0:raise ValueError('Laser FWHM must be nonnegative.')
    v=np.asarray(v);df=v[1]-v[0];f=np.fft.fftfreq(len(v),d=df)
    sigma=laser_fwhm_ghz/(2*np.sqrt(2*np.log(2)));offsets,weights=_modes(laser_modes)
    coeff=np.zeros(len(v),dtype=complex)
    for mu,width,area in components(water):
        coeff+=area*np.exp(-2*np.pi*width*abs(f))*np.exp(-2j*np.pi*f*mu)
    modes=sum(weight*np.exp(-2j*np.pi*f*offset) for offset,weight in zip(offsets,weights))
    coeff*=np.exp(-2*np.pi**2*sigma**2*f*f)*modes
    return np.fft.fftshift(np.fft.ifft(coeff).real)/df


# 宽频积分网格：|ν|≤12 GHz 用细步长（覆盖谱峰），外侧到 ±limit 用粗
# 步长（覆盖洛伦兹尾）；返回中点节点与对应积分宽度。
def integration_grid(core_step=.0025,tail_step=.1,limit=150.):
    if not np.all(np.isfinite([core_step,tail_step,limit])) or core_step<=0 or tail_step<=0 or limit<12:raise ValueError('Invalid wide-frequency quadrature.')
    segments=[(-limit,-12,tail_step),(-12,12,core_step),(12,limit,tail_step)]
    nodes=[];widths=[]
    for lo,hi,step in segments:
        if hi==lo:continue
        edges=np.linspace(lo,hi,int(np.ceil((hi-lo)/step))+1)
        nodes.extend((edges[1:]+edges[:-1])/2);widths.extend(np.diff(edges))
    return np.asarray(nodes),np.asarray(widths)


# 由环半径 r = f·tanθ 得外部入射角，Snell 折射进腔体：
# sin²θ_cav = (n_ext/n_cav)²·r²/(f²+r²)；同时返回 cosθ_cav 与稳定的
# 1−cosθ（用 sin²/(1+cos) 形式，避免两个相近量相减）。
def cavity_cosine(radius_mm,cfg):
    r=np.asarray(radius_mm)/cfg.focal_length_mm
    sin2=(cfg.external_index/cfg.cavity_phase_index)**2*r*r/(1+r*r)
    if np.any(sin2>=1):raise ValueError('Ray has no propagating cavity angle.')
    ct=np.sqrt(1-sin2)
    return ct,sin2/(1+ct)


# FP 透过率矩阵 T(半径, 频率)：含中心失谐与轴上载频修正
# （angular_carrier_ghz = ν0·n_phase/n_group）。
# small_angle=True 用抛物近似角→频率换算，否则用严格 cos 形式。
def fp_response(radius_mm,v,cfg,small_angle=False):
    r=np.asarray(radius_mm);v=np.asarray(v)
    if small_angle:
        angular=cfg.nu0_ghz*cfg.external_index**2/(cfg.cavity_phase_index*cfg.group_index)*(r/cfg.focal_length_mm)**2/2
        delta=cfg.center_detuning_ghz+v[None,:]-angular[:,None]
    else:
        ct,loss=cavity_cosine(r,cfg)
        # 对固定外部入射角，d(n*cos(theta_c)*nu)/dnu 不是 n_g*cos(theta_c)。
        # 一阶色散导数，n_g=n 时退化为旧空气/无色散模型。
        spectral_slope=ct+(1-cfg.cavity_phase_index/cfg.group_index)*(1-ct*ct)/ct
        delta=cfg.center_detuning_ghz+spectral_slope[:,None]*v[None,:]-cfg.angular_carrier_ghz*loss[:,None]
    return cfg.peak_transmission*airy(delta,cfg)


# 反解：给定级次 order 与频率偏移，求该成分的共振环半径；
# 不物理（cosθ 越界）时返回 NaN。用于视场/级次覆盖分析。
def ring_radius(order,frequency_offset,cfg):
    # Solve the same first-order dispersive phase as fp_response.
    ratio=cfg.cavity_phase_index/cfg.group_index
    a=cfg.angular_carrier_ghz+frequency_offset*ratio
    b=cfg.center_detuning_ghz-cfg.angular_carrier_ghz+order*cfg.fsr_ghz
    c=frequency_offset*(1-ratio)
    discriminant=b*b-4*a*c
    if discriminant<0 or a<=0:return float('nan')
    ct=(-b+np.sqrt(discriminant))/(2*a)
    if ct<=0 or ct>1:return float('nan')
    sin2=(cfg.cavity_phase_index/cfg.external_index)**2*(1-ct*ct)
    if sin2>=1:return float('nan')
    return float(cfg.focal_length_mm*np.sqrt(sin2/(1-sin2)))


# 视场覆盖检查：列出红移/中心/蓝移三分量在 -2..3 衍射级次内的环半径
# 是否落在 ROI 内，汇总两侧布里渊峰可见性与角度的频率扫描范围（单位 FSR）。
# 只按峰中心频率判断；线尾、相位、照明与收光孔径另有影响。
def coverage(water,cfg,roi_mm):
    p=line_parameters(water);rows=[]
    for name,frequency in [('red',p['shift_ghz']-p['brillouin_ghz']),('center',p['shift_ghz']),('blue',p['shift_ghz']+p['brillouin_ghz'])]:
        for order in range(-2,4):
            r=ring_radius(order,frequency,cfg)
            if np.isfinite(r):rows.append(dict(component=name,frequency_ghz=frequency,order=order,radius_mm=r,inside_roi=bool(r<roi_mm)))
    visible={name:any(row['component']==name and row['inside_roi'] for row in rows) for name in ['red','center','blue']}
    ct,loss=cavity_cosine([roi_mm],cfg)
    return dict(rings=rows,visibility=visible,both_side_peaks_visible=visible['red'] and visible['blue'],
        angular_scan_fsr=float(cfg.angular_carrier_ghz*loss[0]/cfg.fsr_ghz),
        note='Peak centres only; line tails, phase offset, illumination and collection acceptance also affect recorded signal.')


# 在气体版成像 Detector 上叠加 FP 透过率与照明，得到"环形均值"前向算子。
class WaterDetector(Detector):
    # 在细宽频网格上组装环形均值前向矩阵 K(bin × ν)（含积分权重），
    # 按 512 列分块以控制内存峰值；可选同时返回逐半径响应矩阵。
    def physical_kernel(self,v,weights,cfg,return_radial=True):
        k=np.empty((self.nbins,len(v)));radial=np.empty((len(self.radius_mm),len(v))) if return_radial else None
        for start in range(0,len(v),512):
            sl=slice(start,start+512)
            response=fp_response(self.radius_mm,v[sl],cfg)*illumination(self.radius_mm,cfg)[:,None]*weights[sl]
            k[:,sl]=self.operator@response
            if radial is not None:radial[:,sl]=response
        return k,radial

    # 折叠/反演网格上的环形均值算子（含一个步长的 Δν 乘子）。
    def kernel(self,v,cfg,small_angle=False):
        response=fp_response(self.radius_mm,v,cfg,small_angle)*illumination(self.radius_mm,cfg)[:,None]
        return np.asarray(self.operator@response)*(v[1]-v[0])
