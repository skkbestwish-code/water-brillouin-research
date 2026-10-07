"""Nonnegative Tikhonov inversion. Truth is never used to select alpha.

中文说明：反演含两条路线——周期 Wiener 对照（频域、假定平移不变核）与
非负 Tikhonov+NNLS（完整相机算子，Morozov 偏差准则选 α）。
正则化强度的选择只使用数据与已知噪声尺度，从不使用真值。
"""
import numpy as np
from scipy.optimize import nnls,least_squares
from .physics import gaussian,lorentzian,convolve_laser


# 周期频域 Wiener 对照方法（要求平移不变核；不是完整像素相机反演）。
# 负旁瓣保留并作为 negative_area 报告，而不是截断成"更好看"的正值。
def wiener_reconstruct(v,y,h,noise_sigma,regularizers=None):
    """Periodic frequency-domain control; assumes a shift-invariant kernel.

    This is not the full pixel-camera inverse. Negative sidelobes are retained
    and reported rather than clipped to make the reconstruction look positive.
    """
    if noise_sigma<=0 or len(v)!=len(y) or len(v)!=len(h):
        raise ValueError('Invalid Wiener control data.')
    transfer=np.fft.fft(np.fft.ifftshift(h))*(v[1]-v[0])
    observed=np.fft.fft(np.fft.ifftshift(y))
    regularizers=np.asarray(regularizers if regularizers is not None else np.logspace(-10,-1,19))
    if np.any(regularizers<=0):raise ValueError('Wiener regularizers must be positive.')
    path=[];solutions=[];predictions=[]
    for lam in regularizers:
        recovered=np.conj(transfer)*observed/(abs(transfer)**2+lam)
        s=np.fft.fftshift(np.fft.ifft(recovered).real)
        fitted=np.fft.fftshift(np.fft.ifft(recovered*transfer).real)
        chi2=float(np.mean(((fitted-y)/noise_sigma)**2))
        effective_dof=float(np.sum(abs(transfer)**2/(abs(transfer)**2+lam)))
        gcv=chi2/max(1-effective_dof/len(v),1e-12)**2
        path.append(dict(regularizer=float(lam),chi2=chi2,effective_dof=effective_dof,gcv=gcv))
        solutions.append(s);predictions.append(fitted)
    # 用 GCV 从数据本身选择正则化强度（不需要真谱作参照）：
    # A fixed chi2=1 threshold can chase random noise-variance fluctuations,
    # forcing near-unregularized inversion. GCV penalizes fitted degrees of
    # freedom and chooses from data alone, without a true-spectrum reference.
    index=int(np.argmin([p['gcv'] for p in path]))
    s=solutions[index]
    return dict(spectrum=s,fitted=predictions[index],regularizer=path[index]['regularizer'],chi2=path[index]['chi2'],
                negative_area=float(np.maximum(-s,0).sum()*(v[1]-v[0])),path=path,
                selection='generalized cross-validation',gcv=path[index]['gcv'])


# 非负 Tikhonov 反演：白化 A=K/σ 与 √α·D 增广后用 NNLS 求解；
# 取最大的 α 使 χ² ≤ 目标值（Morozov 偏差准则）；真值不参与选择。
def reconstruct(kernel,y,sigma,v,alphas=None,target_chi2=1.0):
    k=np.asarray(kernel);y=np.asarray(y);sigma=np.asarray(sigma)
    if k.shape!=(len(y),len(v)) or sigma.shape!=y.shape or np.any(sigma<=0):
        raise ValueError('Inconsistent inverse problem or noise scale.')
    n=len(v);df=v[1]-v[0]
    # Periodic boundary follows the single-FSR folded-spectrum convention.
    # 周期边界上的二阶差分正则算子（匹配单 FSR 折叠谱的周期约定）。
    d=(np.roll(np.eye(n),1,axis=1)-2*np.eye(n)+np.roll(np.eye(n),-1,axis=1))/df**2
    a=k/sigma[:,None];b=y/sigma
    alphas=np.asarray(alphas if alphas is not None else np.logspace(-5,3,9))
    if np.any(alphas<=0):raise ValueError('Regularization coefficients must be positive.')
    path=[];solutions=[]
    for alpha in alphas:
        augmented=np.vstack([a,np.sqrt(alpha)*d])
        rhs=np.r_[b,np.zeros(n)]
        s,_=nnls(augmented,rhs,maxiter=5*n)
        residual=(k@s-y)/sigma
        chi2=float(np.mean(residual**2))
        path.append(dict(alpha=float(alpha),chi2=chi2,roughness=float(np.linalg.norm(d@s))))
        solutions.append(s)
    # Morozov discrepancy: largest alpha still fitting the known noise scale.
    eligible=[i for i,p in enumerate(path) if p['chi2']<=target_chi2]
    idx=max(eligible,key=lambda i:path[i]['alpha']) if eligible else int(np.argmin([p['chi2'] for p in path]))
    s=solutions[idx]
    return dict(spectrum=s,fitted=k@s,alpha=path[idx]['alpha'],chi2=path[idx]['chi2'],
                converged=True,path=path,discrepancy_reached=bool(eligible),
                selection='largest alpha with mean squared standardized residual <= target')


# 参数化条件拟合（仅两种气体极限）：谱族与仪器固定，只拟合频移、
# 宽度与振幅；激光卷积作用在本征谱与 K 之前。误差为条件高斯误差。
def fit_parametric(kernel,y,sigma,v,kind,laser_fwhm=.01,gamma=1.4,laser_modes=None):
    """Conditional forward fit, with spectral-family and instrument held fixed.

    Widths are intrinsic gas widths; laser convolution is applied before K.
    Standard errors are local Gaussian errors conditional on this exact model.
    """
    period=(v[1]-v[0])*len(v)
    if kind=='knudsen':
        names=['shift_ghz','sigma_ghz','amplitude'];start=[0.,.9,1.]
        bounds=([-1.5,.1,.1],[1.5,2.,3.])
        def intrinsic(p):return gaussian(v,p[0],p[1],period)
    elif kind=='hydrodynamic':
        names=['shift_ghz','brillouin_ghz','rayleigh_hwhm_ghz','brillouin_hwhm_ghz','amplitude']
        start=[0.,.9,.08,.08,1.];bounds=([-1.5,.2,.002,.002,.1],[1.5,2.,1.,1.,3.])
        def intrinsic(p):
            s=(gamma-1)/gamma*lorentzian(v,p[0],p[2],period)
            for sign in [-1,1]:s+=1/(2*gamma)*lorentzian(v,p[0]+sign*p[1],p[3],period)
            return s
    else:raise ValueError('Parametric fit only supports the two physical limits.')
    def model(p):return p[-1]*(kernel@convolve_laser(intrinsic(p),v,laser_fwhm,laser_modes))
    fit=least_squares(lambda p:(model(p)-y)/sigma,start,bounds=bounds,
                      xtol=1e-10,ftol=1e-10,gtol=1e-10,max_nfev=300)
    cov=np.linalg.pinv(fit.jac.T@fit.jac)
    return dict(parameters=dict(zip(names,map(float,fit.x))),
                conditional_standard_errors=dict(zip(names,map(float,np.sqrt(np.diag(cov))))),
                success=bool(fit.success),message=str(fit.message),evaluations=int(fit.nfev),
                chi2=float(np.mean(fit.fun**2)),fitted=model(fit.x),
                spectrum=fit.x[-1]*convolve_laser(intrinsic(fit.x),v,laser_fwhm,laser_modes),
                at_bound=bool(np.any(fit.active_mask)),kind=kind)
