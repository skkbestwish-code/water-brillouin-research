import unittest
from dataclasses import replace
import numpy as np
from water_simulation.physics import WaterFP, fp_response, Water, WaterDetector, integration_grid, physical_spectrum, line_parameters
from water_simulation.inverse import fit_water
from simulation.imaging import Camera
from water_simulation.etalon import silica_indices,catalog_instrument
from water_simulation.physics import ring_radius

class EtalonTests(unittest.TestCase):
    def test_catalog_units_and_silica_indices(self):
        n,ng=silica_indices(632.8)
        self.assertAlmostEqual(n,1.45701793,places=8)
        self.assertAlmostEqual(ng,1.47539225,places=8)
        self.assertAlmostEqual(catalog_instrument().fsr_ghz,299792458*.5*100/1e9,places=10)
        self.assertGreater(catalog_instrument().cavity_thickness_mm,6.743)

    def test_dispersive_ring_is_response_maximum(self):
        cfg=catalog_instrument()
        r=ring_radius(0,6.24,cfg)
        self.assertAlmostEqual(float(fp_response([r],[6.24],cfg)[0,0]),.5,places=12)

    def test_invalid_transmission_and_nonfinite_are_rejected(self):
        for t in [0,-1,1.1,float('nan')]:
            with self.assertRaises(ValueError):WaterFP(peak_transmission=t)
        with self.assertRaises(ValueError):WaterFP(cavity_group_index=float('nan'))

    def test_peak_transmission_in_forward_operator(self):
        self.assertIn('peak_transmission',WaterFP.__dataclass_fields__)
        a=WaterFP(); b=replace(a,peak_transmission=.5)
        np.testing.assert_allclose(fp_response([0,.2],[0,1],b),.5*fp_response([0,.2],[0,1],a))

    def test_quality_is_separate_from_optimizer_success(self):
        d=WaterDetector(Camera(size=64,pixel_um=8,roi_mm=.24,radial_step_um=3,subpixels=2))
        v,w=integration_grid(.01,.5,30); water=Water()
        k,_=d.physical_kernel(v,w,WaterFP(focal_length_mm=20),False)
        y=k@physical_spectrum(v,water)
        fit=fit_water(k,v,y+np.linspace(0,.1,len(y)),np.full(len(y),.0001),water)
        self.assertIn('quality_pass',fit)
        self.assertFalse(fit['quality_pass'])

    def test_low_angle_spacing_is_not_forced_to_one_ghz(self):
        d=WaterDetector(Camera(size=64,pixel_um=8,roi_mm=.24,radial_step_um=3,subpixels=2))
        v,w=integration_grid(.0025,.5,30); water=Water(scattering_angle_deg=10)
        k,_=d.physical_kernel(v,w,WaterFP(focal_length_mm=20),False)
        y=k@physical_spectrum(v,water)
        fit=fit_water(k,v,y,np.full(len(y),.0001),water,spacing_starts=[.5,4.2])
        self.assertAlmostEqual(fit['parameters']['brillouin_ghz'],line_parameters(water)['brillouin_ghz'],places=6)

if __name__=='__main__': unittest.main()
