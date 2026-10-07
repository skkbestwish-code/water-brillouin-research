"""Axisymmetric 2D PSF followed by pixel area integration and annular mean.

The radial Gaussian convolution includes the Bessel factor and area measure;
it is not a 1D radial Gaussian approximation. r=0 uses midpoint quadrature.

中文说明：轴对称二维 PSF → 像素面积积分 → 环形平均的完整成像模型。
径向高斯模糊包含贝塞尔因子与面积测度（不是一维径向高斯近似）；
r=0 用中点求积；像素内用子像素中点采样；常量场严格保持。
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix,eye
from scipy.special import i0e
from .physics import fp_kernel,illumination


# 相机与 ROI 配置；subpixels 是像素内积分的子像素数。
@dataclass(frozen=True)
class Camera:
    size: int = 768
    pixel_um: float = 4.0
    roi_mm: float = 1.4
    radial_step_um: float = 1.0
    psf_sigma_um: float = 4.0
    subpixels: int = 4

    def __post_init__(self):
        if self.size<8 or min(self.pixel_um,self.roi_mm,self.radial_step_um)<=0 or self.psf_sigma_um<0 or self.subpixels<1:
            raise ValueError('Invalid camera dimensions or PSF.')
        if self.roi_mm>self.size*self.pixel_um/2000:
            raise ValueError('Circular ROI must lie inside the sensor.')


# 轴对称成像的算术框架：像素环带划分、PSF 离散、像素装箱与环形平均。
class Detector:
    # 预计算几何：环带划分（labels/counts）、半径网格、二维 PSF 的径向
    # 离散模糊矩阵、像素装箱矩阵，并合成为环形均值算子 operator。
    def __init__(self,camera=Camera()):
        self.camera=camera;self.pixel_mm=camera.pixel_um/1000;dr=camera.radial_step_um/1000
        axis=(np.arange(camera.size)-(camera.size-1)/2)*self.pixel_mm
        self.xx,self.yy=np.meshgrid(axis,axis)
        center_r=np.hypot(self.xx,self.yy)
        self.nbins=int(np.floor(camera.roi_mm/self.pixel_mm+1e-9))
        self.labels=np.floor(center_r/self.pixel_mm).astype(int)
        self.valid=self.labels<self.nbins
        self.counts=np.bincount(self.labels[self.valid],minlength=self.nbins)
        if np.any(self.counts==0):raise ValueError('Empty annular bins.')
        self.bin_radius_mm=(np.arange(self.nbins)+.5)*self.pixel_mm
        maxr=np.sqrt(2)*(camera.size*self.pixel_mm/2)+7*camera.psf_sigma_um/1000+2*dr
        self.radius_mm=(np.arange(int(np.ceil(maxr/dr)))+.5)*dr
        self.blur=self._blur_matrix(dr)
        self.pixel_bins=self._pixel_bin_matrix(dr)
        self.operator=(self.pixel_bins@self.blur).tocsr()

    # 轴对称二维高斯 PSF 的径向离散：权重 ∝ r'·exp(−(r−r')²/2σ²)·I0(r·r'/σ²)
    # （用指数缩放贝塞尔 i0e 防溢出）；行归一化修正中点求积误差。
    def _blur_matrix(self,dr):
        sigma=self.camera.psf_sigma_um/1000;n=len(self.radius_mm)
        if sigma==0:return eye(n,format='csr')
        rows=[];cols=[];data=[];half=int(np.ceil(7*sigma/dr))
        for i,r in enumerate(self.radius_mm):
            js=np.arange(max(0,i-half),min(n,i+half+1));rp=self.radius_mm[js]
            weights=rp*dr/sigma**2*np.exp(-.5*((r-rp)/sigma)**2)*i0e(r*rp/sigma**2)
            weights/=weights.sum()  # correct tiny midpoint quadrature error on constants
            rows.extend([i]*len(js));cols.extend(js);data.extend(weights)
        return csr_matrix((data,(rows,cols)),shape=(n,n))

    # 像素内子像素中点偏移（面积积分用）。
    def _offsets(self):
        return ((np.arange(self.camera.subpixels)+.5)/self.camera.subpixels-.5)*self.pixel_mm

    # 把径向亮度按子像素采样线性插值分配到各环带，再除以环内像素数
    # （即环形平均而非环上总计数），保证常量场映射仍为 1。
    def _pixel_bin_matrix(self,dr):
        nr=len(self.radius_mm);total=np.zeros((self.nbins,nr))
        labels=self.labels[self.valid]
        for dy in self._offsets():
            for dx in self._offsets():
                rr=np.hypot(self.xx[self.valid]+dx,self.yy[self.valid]+dy)
                index=np.clip(rr/dr-.5,0,nr-1);lo=np.floor(index).astype(int);hi=np.minimum(lo+1,nr-1);fraction=index-lo
                total+=np.bincount(labels*nr+lo,weights=1-fraction,minlength=self.nbins*nr).reshape(self.nbins,nr)
                total+=np.bincount(labels*nr+hi,weights=fraction,minlength=self.nbins*nr).reshape(self.nbins,nr)
        total/=self.counts[:,None]*self.camera.subpixels**2
        return csr_matrix(total)

    # 环形均值（稀疏算子乘法）。
    def profile(self,irradiance):
        return np.asarray(self.operator@irradiance)

    # 生成二维像素图：先做径向 PSF 模糊，再逐子像素采样插值（画图/噪声仿真用）。
    def image(self,irradiance):
        blurred=np.asarray(self.blur@irradiance);image=np.zeros_like(self.xx)
        for dy in self._offsets():
            for dx in self._offsets():
                rr=np.hypot(self.xx+dx,self.yy+dy)
                image+=np.interp(rr,self.radius_mm,blurred)
        return image/self.camera.subpixels**2

    # 对任意二维图像直接做环带平均（与 profile 互为验证）。
    def annular_mean(self,image):
        return np.bincount(self.labels[self.valid],weights=np.asarray(image)[self.valid],minlength=self.nbins)/self.counts

    # 常规（气体版）环形均值前向矩阵（含一个步长的 Δν 乘子）。
    def kernel(self,v,cfg,small_angle=False):
        response=fp_kernel(self.radius_mm,v,cfg,small_angle)*illumination(self.radius_mm,cfg)[:,None]
        return np.asarray(self.operator@response)*(v[1]-v[0])


# 电子计数噪声模型（输入是"每像素"相对期望值）：
# 每像素期望电子数 = gain·期望 + 背景；对整条环带做 Poisson(电子数×像素数)
# 加高斯读出噪声，再除以像素数与增益还原为每像素相对观测；
# σ 同样按每像素折算，供卡方/拟合使用。
def noisy_profile(expected,counts,gain,background,read_noise,seed):
    if gain<=0 or background<0 or read_noise<0 or np.any(np.asarray(counts)<=0):
        raise ValueError('Invalid electron-count model.')
    rng=np.random.default_rng(seed);counts=np.asarray(counts)
    electrons=np.maximum(gain*np.asarray(expected)+background,0)
    summed=rng.poisson(electrons*counts)+rng.normal(0,read_noise*np.sqrt(counts))
    observation=(summed/counts-background)/gain
    sigma=np.sqrt((electrons+read_noise**2)/counts)/gain
    return observation,sigma
