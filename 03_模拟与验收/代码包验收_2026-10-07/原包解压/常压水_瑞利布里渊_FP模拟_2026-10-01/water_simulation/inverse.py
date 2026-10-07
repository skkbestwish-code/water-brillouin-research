"""Conditional water-line forward fitting, using unfolded spectral integration.

中文说明：条件前向拟合——谱族先验（三线洛伦兹 + 额外弹性线）、仪器、
激光与角度分布均固定，只拟合整体频移、侧峰间距/宽度、中心分量与振幅。
报告的误差是在此模型下的条件误差，不是实验总误差；拟合初始化与选解
绝不使用真实谱。
"""
from dataclasses import replace
import numpy as np
from scipy.optimize import least_squares
from .physics import line_parameters,angle_samples,spectrum_from_lines


def fit_water(kernel,v,y,sigma,water,laser_fwhm_ghz=.01,laser_modes=None):
    """Fit shift, side spacing/width, central fraction and amplitude.

    Known angle distribution, elastic fraction, thermal width, laser and FP
    are held fixed. Reported errors are conditional, not total experimental
    uncertainty. No true spectrum is used to initialize or choose the fit.
    """
    # 参数向量 p = [整体频移 shift, 布里渊间距 b, 侧峰 HWHM, 中心权重, 振幅]。
    base=line_parameters(water)
    # 各角度样本相对基准角的波矢比 q/q0：频移按 q 缩放、宽度按 q² 缩放。
    scales=[]
    for angle,weight in angle_samples(water):
        p=line_parameters(replace(water,scattering_angle_deg=float(angle),angular_sigma_deg=0))
        scales.append((p['q_per_m']/base['q_per_m'],weight))
    # 由参数构造参数化入射谱：中心线 + ±b 侧峰（面积按 central 分配），
    # 再加上固定于 0 频移的弹性分量，最后乘振幅并卷积激光。
    def spectrum(p):
        shift,b,width,central,amplitude=p;lines=[]
        for scale,weight in scales:
            area=(1-water.elastic_fraction)*weight
            lines.append((shift*scale,base['rayleigh_hwhm_ghz']*scale**2,area*central))
            for sign in [-1,1]:lines.append(((shift+sign*b)*scale,width*scale**2,area*(1-central)/2))
        lines.append((0.,water.elastic_hwhm_ghz,water.elastic_fraction))
        return amplitude*spectrum_from_lines(v,lines,laser_fwhm_ghz,laser_modes)
    def prediction(p):return kernel@spectrum(p)
    start=[0.,4.2,.12,.03,1.]
    bounds=([-1.,1.,.015,0.,.1],[1.,8.,1.,.7,3.])
    fit=least_squares(lambda p:(prediction(p)-y)/sigma,start,bounds=bounds,
        xtol=1e-10,ftol=1e-10,gtol=1e-10,max_nfev=200)
    names=['shift_ghz','brillouin_ghz','brillouin_hwhm_ghz','intrinsic_central_fraction','amplitude']
    # 局部高斯误差近似：协方差 ≈ (JᵀJ)⁻¹，仅在此模型条件内有效。
    covariance=np.linalg.pinv(fit.jac.T@fit.jac)
    return dict(parameters=dict(zip(names,map(float,fit.x))),
        conditional_standard_errors=dict(zip(names,map(float,np.sqrt(np.diag(covariance))))),
        chi2=float(np.mean(fit.fun**2)),success=bool(fit.success),at_bound=bool(np.any(fit.active_mask)),
        evaluations=int(fit.nfev),fitted=prediction(fit.x),spectrum=spectrum(fit.x),
        note='Conditional on the specified instrument, laser, water angle distribution and elastic fraction; central optical weight remains phenomenological.')
