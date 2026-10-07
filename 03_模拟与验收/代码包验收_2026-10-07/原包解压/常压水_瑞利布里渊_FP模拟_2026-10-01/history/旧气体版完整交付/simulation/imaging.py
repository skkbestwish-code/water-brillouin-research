"""Axisymmetric 2D PSF followed by pixel area integration and annular mean.

The radial Gaussian convolution includes the Bessel factor and area measure;
it is not a 1D radial Gaussian approximation. r=0 uses midpoint quadrature.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix,eye
from scipy.special import i0e
from .physics import fp_kernel,illumination


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


class Detector:
    def __init__(self,camera=Camera()):
        self.camera=camera;self.pixel_mm=camera.pixel_um/1000;dr=camera.radial_step_um/1000
        axis=(np.arange(camera.size)-(camera.size-1)/2)*self.pixel_mm
        self.xx,self.yy=np.meshgrid(axis,axis)
        center_r=np.hypot(self.xx,self.yy)
        self.nbins=int(np.floor(camera.roi_mm/self.pixel_mm))
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

    def _offsets(self):
        return ((np.arange(self.camera.subpixels)+.5)/self.camera.subpixels-.5)*self.pixel_mm

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

    def profile(self,irradiance):
        return np.asarray(self.operator@irradiance)

    def image(self,irradiance):
        blurred=np.asarray(self.blur@irradiance);image=np.zeros_like(self.xx)
        for dy in self._offsets():
            for dx in self._offsets():
                rr=np.hypot(self.xx+dx,self.yy+dy)
                image+=np.interp(rr,self.radius_mm,blurred)
        return image/self.camera.subpixels**2

    def annular_mean(self,image):
        return np.bincount(self.labels[self.valid],weights=np.asarray(image)[self.valid],minlength=self.nbins)/self.counts

    def kernel(self,v,cfg,small_angle=False):
        response=fp_kernel(self.radius_mm,v,cfg,small_angle)*illumination(self.radius_mm,cfg)[:,None]
        return np.asarray(self.operator@response)*(v[1]-v[0])


def noisy_profile(expected,counts,gain,background,read_noise,seed):
    if gain<=0 or background<0 or read_noise<0 or np.any(np.asarray(counts)<=0):
        raise ValueError('Invalid electron-count model.')
    rng=np.random.default_rng(seed);counts=np.asarray(counts)
    electrons=np.maximum(gain*np.asarray(expected)+background,0)
    summed=rng.poisson(electrons*counts)+rng.normal(0,read_noise*np.sqrt(counts))
    observation=(summed/counts-background)/gain
    sigma=np.sqrt((electrons+read_noise**2)/counts)/gain
    return observation,sigma
