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


def fit_water(kernel,v,y,sigma,water,laser_fwhm_ghz=.01,laser_modes=None,
              spacing_starts=None,quality_chi2_max=2.,spacing_bounds=(.05,10.),
              ambiguity_delta_chi2=9.):
    """Fit shift, side spacing/width, central fraction and amplitude.

    Known angle distribution, elastic fraction, thermal width, laser and FP
    are held fixed. Reported errors are conditional, not total experimental
    uncertainty. No true spectrum is used to initialize or choose the fit.
    """
    # 参数向量 p = [整体频移 shift, 布里渊间距 b, 侧峰 HWHM, 中心权重, 振幅]。
    kernel=np.asarray(kernel);v=np.asarray(v);y=np.asarray(y);sigma=np.asarray(sigma)
    if v.ndim!=1 or y.ndim!=1 or len(v)<2 or len(y)<5 or kernel.shape!=(len(y),len(v)) or sigma.shape!=y.shape or np.any(sigma<=0) or not all(np.all(np.isfinite(x)) for x in [kernel,v,y,sigma]):
        raise ValueError('Finite compatible arrays and positive sigma required.')
    if len(spacing_bounds)!=2 or not np.all(np.isfinite(spacing_bounds)) or not .05<=spacing_bounds[0]<spacing_bounds[1]<=10:
        raise ValueError('Spacing bounds must lie within [0.05,10] GHz.')
    if not np.isfinite(quality_chi2_max) or quality_chi2_max<=0 or not np.isfinite(ambiguity_delta_chi2) or ambiguity_delta_chi2<0:
        raise ValueError('Invalid quality thresholds.')
    base=line_parameters(water)
    if not spacing_bounds[0]<base['brillouin_ghz']<spacing_bounds[1] or not .00001<base['brillouin_hwhm_ghz']<1:
        raise ValueError('Reference water geometry outside declared fit bounds; check independent prior and configuration.')
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
    lo,hi=spacing_bounds
    starts=([b for b in [.5,4.2,6.5,8.8] if lo<b<hi] or [(lo+hi)/2]) if spacing_starts is None else list(spacing_starts)
    if not starts or not all(np.isfinite(b) and lo<b<hi for b in starts):
        raise ValueError('Spacing starts must be within declared bounds.')
    bounds=([-1.,lo,.00001,0.,.1],[1.,hi,1.,.7,3.])
    fits=[least_squares(lambda p:(prediction(p)-y)/sigma,[0.,b,.12,.03,1.],bounds=bounds,
        xtol=1e-10,ftol=1e-10,gtol=1e-10,max_nfev=200) for b in starts]
    # 数据残差选解，不使用合成真值。
    fit=min([f for f in fits if f.success] or fits,key=lambda f: np.sum(f.fun**2))
    competing=[dict(b_ghz=float(f.x[1]),delta_chi2=float(2*(f.cost-fit.cost))) for f in fits
        if f.success and abs(f.x[1]-fit.x[1])>.05 and 2*(f.cost-fit.cost)<=ambiguity_delta_chi2]
    names=['shift_ghz','brillouin_ghz','brillouin_hwhm_ghz','intrinsic_central_fraction','amplitude']
    # 局部高斯误差近似：协方差 ≈ (JᵀJ)⁻¹，仅在此模型条件内有效。
    _,singular,vt=np.linalg.svd(fit.jac,full_matrices=False)
    rank=int(np.sum(singular>singular[0]*max(fit.jac.shape)*np.finfo(float).eps))
    norm=np.linalg.norm(fit.jac,axis=0)
    normalized_s=np.linalg.svd(fit.jac/np.maximum(norm,1e-300),compute_uv=False)
    condition=float(normalized_s[0]/normalized_s[-1]) if normalized_s[-1]>0 else None
    local_valid=rank==5 and condition is not None and condition<=1e8
    # A pseudoinverse zero in a null direction is NOT zero uncertainty.
    if local_valid:
        covariance=(vt.T/singular**2)@vt
        local_errors=dict(zip(names,map(float,np.sqrt(np.maximum(np.diag(covariance),0)))))
    else:local_errors=dict.fromkeys(names,None)
    bound_mask=(fit.x-np.array(bounds[0])<1e-7)|(np.array(bounds[1])-fit.x<1e-7)
    reasons=[]
    if not fit.success:reasons.append('optimizer_failed')
    if not np.all(np.isfinite(fit.x)):reasons.append('nonfinite_parameters')
    if np.any(bound_mask[[0,1,2,4]]):reasons.append('key_parameter_at_bound')
    if float(np.mean(fit.fun**2))>quality_chi2_max:reasons.append('residual_exceeds_threshold')
    if not local_valid:reasons.append('local_jacobian_ill_conditioned')
    if competing:reasons.append('competing_spacing_solutions')
    return dict(parameters=dict(zip(names,map(float,fit.x))),
        conditional_standard_errors=local_errors if not competing else dict.fromkeys(names,None),
        local_standard_errors=local_errors,competing_solutions=competing,
        spacing_bounds_ghz=list(spacing_bounds),ambiguity_delta_chi2_threshold=ambiguity_delta_chi2,
        chi2=float(np.mean(fit.fun**2)),success=bool(fit.success),at_bound=bool(np.any(bound_mask)),
        optimizer_success=bool(fit.success),quality_pass=not reasons,quality_reasons=reasons,
        quality_warning=['central_weight_at_bound'] if bound_mask[3] else [],
        message=str(fit.message),jacobian_rank=rank,normalized_jacobian_condition=condition,
        start_solutions=[dict(start_b=b,cost=float(f.cost),success=bool(f.success),b=float(f.x[1])) for b,f in zip(starts,fits)],
        evaluations=int(fit.nfev),fitted=prediction(fit.x),spectrum=spectrum(fit.x),
        note='Conditional on the specified instrument, laser, water angle distribution and elastic fraction; central optical weight remains phenomenological.')
