import unittest
import numpy as np
from water_simulation.physics import Water,WaterDetector,integration_grid,physical_spectrum
from water_simulation.etalon import catalog_instrument
from water_simulation.inverse import fit_water
from simulation.imaging import Camera,noisy_profile
from simulation.inverse import reconstruct

class ReviewTests(unittest.TestCase):
    def test_global_alias_is_not_quality_pass(self):
        d=WaterDetector(Camera(size=128,pixel_um=8,roi_mm=.48,radial_step_um=2,subpixels=2))
        from dataclasses import replace
        c=replace(catalog_instrument(),focal_length_mm=35)
        v,w=integration_grid(.005,.1,150);water=Water(scattering_angle_deg=180)
        k,_=d.physical_kernel(v,w,c,False);y=k@physical_spectrum(v,water)
        _,ss=noisy_profile(y,d.counts,2000,20,3,55)
        f=fit_water(k,v,y,ss,water,spacing_starts=[6.5,8.8])
        self.assertFalse(f['quality_pass'])
        self.assertIn('competing_spacing_solutions',f['quality_reasons'])

    def test_singular_fit_does_not_report_zero_uncertainty(self):
        v=np.linspace(-12,12,200);k=np.zeros((8,len(v)))
        f=fit_water(k,v,np.zeros(8),np.ones(8),Water(),spacing_starts=[4.2])
        self.assertIsNone(f['conditional_standard_errors']['brillouin_ghz'])

    def test_nonuniform_inverse_axis_rejected(self):
        v=np.array([0.,.1,.3,.4,.5,.6,.7,.8])
        with self.assertRaises(ValueError):reconstruct(np.eye(8),np.ones(8),np.ones(8),v)

    def test_water_nan_is_rejected(self):
        with self.assertRaises(ValueError):Water(velocity_ms=float('nan'))

if __name__=='__main__':unittest.main()
