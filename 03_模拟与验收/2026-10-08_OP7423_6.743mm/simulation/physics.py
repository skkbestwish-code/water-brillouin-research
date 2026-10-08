"""Physical quantities in SI internally; spectral axes/densities in GHz/1 GHz.

RB line models are limiting models, not a Tenti S6 implementation. A single FSR
spectral prior is used. Lorentzian tails are folded into that window explicitly.

中文说明：内部一律使用 SI 单位，频率轴与谱密度用 GHz 与 1/GHz。
RB 谱是两个极限模型（Knudsen 高斯、流体力学三线洛伦兹）而不是
Tenti S6；动力学区（约 0.3<y<3）会拒绝运行而不是悄悄外推。
谱密度始终折叠在一个 FSR 周期上。
"""
from dataclasses import dataclass, asdict
import numpy as np

# 光速 (m/s)、玻尔兹曼常数、阿伏伽德罗常数。
C = 299792458.0
KB = 1.380649e-23
NA = 6.02214076e23


# 气体版仪器参数（水版见 water_simulation.WaterFP）。
@dataclass(frozen=True)
class Instrument:
    fsr_ghz: float = 15.0
    finesse: float = 30.0
    wavelength_nm: float = 632.8
    focal_length_mm: float = 100.0
    center_detuning_ghz: float = 2.5
    illumination_radius_mm: float = 2.0

    def __post_init__(self):
        if min(self.fsr_ghz,self.wavelength_nm,self.focal_length_mm,self.illumination_radius_mm)<=0 or self.finesse<=1:
            raise ValueError('Positive dimensions and Airy finesse > 1 are required.')

    # 激光真空频率（GHz）。
    @property
    def nu0_ghz(self):
        return C/(self.wavelength_nm*1e-9)/1e9

    # 由 FSR 换算的腔内光学长度（mm）。
    @property
    def optical_length_mm(self):
        return C/(2*self.fsr_ghz*1e9)*1e3

    # 由"谱精细度 = FSR/FWHM"反推的 Airy 系数 A = 1/sin²(π/2F)。
    # 注意 A 不是精细度本身：F=30 时 A≈365。
    @property
    def airy_coefficient(self):
        return 1/np.sin(np.pi/(2*self.finesse))**2

    # 由 A 反推的理想腔镜反射率 R = A/(√(A+1)+1)²。
    @property
    def ideal_reflectivity(self):
        a=self.airy_coefficient
        return a/(a+2+2*np.sqrt(a+1))


# 气体状态（旧基准用；水版散射配置见 water_simulation.Water）。
@dataclass(frozen=True)
class Gas:
    temperature_k: float = 300.0
    pressure_bar: float = .02
    molar_mass_kg: float = 28.0134e-3
    scattering_angle_deg: float = 90.0
    velocity_ms: float = 20.0
    gamma: float = 1.4
    shear_viscosity_pa_s: float = 1.78e-5
    bulk_viscosity_pa_s: float = 0.0
    thermal_conductivity_w_mk: float = .026
    wavelength_nm: float = 632.8

    def __post_init__(self):
        if min(self.temperature_k,self.pressure_bar,self.molar_mass_kg,self.shear_viscosity_pa_s,self.thermal_conductivity_w_mk,self.wavelength_nm)<=0:
            raise ValueError('Temperature, pressure and transport parameters must be positive.')
        if not 0<self.scattering_angle_deg<=180 or self.gamma<=1 or self.bulk_viscosity_pa_s<0:
            raise ValueError('Invalid scattering geometry or heat capacity/viscosity.')


# 以 0 为中心的周期频率网格：size 个点、总跨度 period_ghz。
def frequency_grid(period_ghz, size):
    if period_ghz<=0 or size<8:
        raise ValueError('Invalid frequency grid.')
    return (np.arange(size)-size//2)*(period_ghz/size)


# 截去负值并归一化为面积 1 的密度（气体谱内部使用）。
def normalize(s,v):
    df=float(v[1]-v[0]);area=float(np.sum(s)*df)
    if not np.isfinite(area) or area<=0:
        raise ValueError('Spectrum must have positive finite area.')
    return np.maximum(s,0)/area


# 周期化高斯密度（±3 个周期求和后归一化），Knudsen 极限的谱线。
def gaussian(v,mu,sigma,period):
    if sigma<=0:
        raise ValueError('Gaussian sigma must be positive.')
    s=sum(np.exp(-.5*((v-mu+j*period)/sigma)**2)/(np.sqrt(2*np.pi)*sigma) for j in range(-3,4))
    return normalize(s,v)


# 周期化洛伦兹的精确级数和（对全部混叠求和，避免 cosh(a)−cos(b)
# 两个相近大数相减的精度损失）。
def lorentzian(v,mu,hwhm,period):
    if hwhm<=0:
        raise ValueError('Lorentzian HWHM must be positive.')
    a=2*np.pi*hwhm/period;b=2*np.pi*(v-mu)/period
    # Exact sum of all Lorentzian aliases; avoids subtracting cosh(a)-cos(b).
    s=np.sinh(a)/(period*(2*np.sinh(a/2)**2+2*np.sin(b/2)**2))
    return normalize(s,v)


# 由气体状态计算：波矢 q、热运动速度、声速、密度、cp、热扩散率、
# 动理学参数 y、各线宽（GHz）与三线模型的峰面积权重。
def gas_parameters(g):
    mass=g.molar_mass_kg/NA
    wavevector=4*np.pi*np.sin(np.deg2rad(g.scattering_angle_deg)/2)/(g.wavelength_nm*1e-9)
    thermal_std=np.sqrt(KB*g.temperature_k/mass)
    sound=np.sqrt(g.gamma)*thermal_std
    pressure=g.pressure_bar*1e5
    rho=pressure*mass/(KB*g.temperature_k)
    cp=g.gamma/(g.gamma-1)*KB/mass
    dt=g.thermal_conductivity_w_mk/(rho*cp)
    nu_l=(4*g.shear_viscosity_pa_s/3+g.bulk_viscosity_pa_s)/rho
    return dict(wavevector_m_inv=wavevector,thermal_std_ms=thermal_std,sound_ms=sound,
                density_kg_m3=rho,cp_j_kgk=cp,thermal_diffusivity_m2_s=dt,
                y=pressure/(g.shear_viscosity_pa_s*wavevector*np.sqrt(2)*thermal_std),
                sigma_ghz=wavevector*thermal_std/(2*np.pi*1e9),
                shift_ghz=wavevector*g.velocity_ms/(2*np.pi*1e9),
                brillouin_ghz=wavevector*sound/(2*np.pi*1e9),
                rayleigh_hwhm_ghz=dt*wavevector**2/(2*np.pi*1e9),
                brillouin_hwhm_ghz=.5*(nu_l+(g.gamma-1)*dt)*wavevector**2/(2*np.pi*1e9),
                central_weight=(g.gamma-1)/g.gamma,side_weight=1/(2*g.gamma))


# 按 y 选择 RB 模型：y<0.3 Knudsen 高斯、y>3 流体力学洛伦兹三线；
# 中间动力学区要求外部提供 Tenti S6，直接报错而不是悄悄外推。
# 返回归一化周期谱与元数据。
def rb_spectrum(v,g,kind='auto'):
    p=gas_parameters(g);period=(v[1]-v[0])*len(v)
    if kind=='auto':
        if p['y']<.3:kind='knudsen'
        elif p['y']>3:kind='hydrodynamic'
        else:raise ValueError('Kinetic regime: a validated Tenti S6 input is required.')
    if kind=='knudsen':
        if p['y']>=.3:raise ValueError('Knudsen limit requested outside its y range.')
        s=gaussian(v,p['shift_ghz'],p['sigma_ghz'],period)
    elif kind=='hydrodynamic':
        if p['y']<=3:raise ValueError('Hydrodynamic limit requested outside its y range.')
        s=p['central_weight']*lorentzian(v,p['shift_ghz'],p['rayleigh_hwhm_ghz'],period)
        for sign in [-1,1]:
            s+=p['side_weight']*lorentzian(v,p['shift_ghz']+sign*p['brillouin_ghz'],p['brillouin_hwhm_ghz'],period)
    else:raise ValueError('Unknown RB model.')
    return normalize(s,v),dict(p,model=kind,gas=asdict(g))


# 与激光线型做周期卷积（FFT 实现），保持面积归一化与非负；
# 支持多纵模 [(频偏 GHz, 权重)]。
def convolve_laser(s,v,fwhm_ghz=.01,modes=None):
    if fwhm_ghz<0:raise ValueError('Laser FWHM must be nonnegative.')
    period=(v[1]-v[0])*len(v);modes=modes or [(0.,1.)]
    if fwhm_ghz==0 and modes==[(0.,1.)]:return normalize(s,v)
    if any(w<0 for _,w in modes) or sum(w for _,w in modes)<=0:
        raise ValueError('Laser mode weights must be nonnegative with positive total.')
    sigma=max(fwhm_ghz/(2*np.sqrt(2*np.log(2))),1e-12)
    laser=normalize(sum(w*gaussian(v,mu,sigma,period) for mu,w in modes),v)
    result=np.fft.fftshift(np.fft.ifft(np.fft.fft(np.fft.ifftshift(s))*np.fft.fft(np.fft.ifftshift(laser))).real)*(v[1]-v[0])
    return normalize(np.maximum(result,0),v)


# Airy 透射率 T(Δν) = 1/(1 + A·sin²(πΔν/FSR))，峰值为 1、周期为 FSR。
def airy(detuning_ghz,cfg):
    return 1/(1+cfg.airy_coefficient*np.sin(np.pi*np.asarray(detuning_ghz)/cfg.fsr_ghz)**2)


# 环形均值前向矩阵（半径 × 频率）：每个半径对应 FP 入射角 θ(r)，
# 代入 Airy 透射率；small_angle 用 r²/2 抛物近似，否则用严格
# 1−cosθ 的稳定表达式。
def fp_kernel(radius_mm,frequency_ghz,cfg,small_angle=False):
    r=np.asarray(radius_mm,dtype=float)[:,None];v=np.asarray(frequency_ghz,dtype=float)[None,:]
    z=(r/cfg.focal_length_mm)**2
    if small_angle:
        detuning=cfg.center_detuning_ghz+v-cfg.nu0_ghz*z/2
    else:
        ct=1/np.sqrt(1+z)
        # stable 1-cos(theta), avoiding subtraction of two optical frequencies.
        one_minus_ct=z/(np.sqrt(1+z)*(np.sqrt(1+z)+1))
        detuning=cfg.center_detuning_ghz+v*ct-cfg.nu0_ghz*one_minus_ct
    return airy(detuning,cfg)


# 高斯照明轮廓 exp(−2(r/w)²)，w = illumination_radius_mm。
def illumination(radius_mm,cfg):
    return np.exp(-2*(np.asarray(radius_mm)/cfg.illumination_radius_mm)**2)


# 由衍射级次与频率失谐反解共振环半径；要求 0<cosθ≤1，否则报错。
def resonance_radius_mm(order,detuning_ghz,cfg):
    ct=(cfg.nu0_ghz-cfg.center_detuning_ghz-order*cfg.fsr_ghz)/(cfg.nu0_ghz+detuning_ghz)
    if not 0<ct<=1:raise ValueError('This resonance is outside the physical angular range.')
    return cfg.focal_length_mm*np.sqrt(1/ct**2-1)
