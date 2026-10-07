import unittest
import numpy as np


class PhysicsChecks(unittest.TestCase):
    def test_airy_has_requested_fwhm(self):
        from simulation.physics import Instrument, airy
        cfg = Instrument()
        self.assertAlmostEqual(float(airy(0.0, cfg)), 1.0)
        self.assertAlmostEqual(float(airy(cfg.fsr_ghz / (2*cfg.finesse), cfg)), .5, places=10)
        self.assertAlmostEqual(float(airy(cfg.fsr_ghz, cfg)), 1.0)
        self.assertAlmostEqual(cfg.optical_length_mm, 9.993081933333333)

    def test_thermal_width_is_one_dimensional(self):
        from simulation.physics import Gas, gas_parameters
        p = gas_parameters(Gas())
        expected = np.sqrt(1.380649e-23*300/(28.0134e-3/6.02214076e23))*np.sqrt(2)/632.8e-9/1e9
        self.assertAlmostEqual(p['sigma_ghz'], expected, places=12)
        self.assertAlmostEqual(p['brillouin_ghz']/p['sigma_ghz'], np.sqrt(1.4), places=10)

    def test_spectra_integrate_to_one(self):
        from simulation.physics import Gas, rb_spectrum, frequency_grid
        v = frequency_grid(15., 6000)
        for pressure, kind in [(0.02, 'knudsen'), (20., 'hydrodynamic')]:
            s, meta = rb_spectrum(v, Gas(pressure_bar=pressure), kind)
            self.assertAlmostEqual(float(s.sum()*(v[1]-v[0])), 1., places=9)
            self.assertTrue(np.all(s >= 0))
            if kind == 'hydrodynamic':
                self.assertAlmostEqual(meta['central_weight'], 2/7)
                self.assertAlmostEqual(meta['side_weight'], 5/14)

    def test_kinetic_regime_is_not_silently_modelled(self):
        from simulation.physics import Gas, rb_spectrum, frequency_grid
        with self.assertRaises(ValueError):
            rb_spectrum(frequency_grid(15., 600), Gas(pressure_bar=1.), 'auto')

    def test_exact_rings_and_frequency_direction(self):
        from simulation.physics import Instrument, resonance_radius_mm, fp_kernel
        cfg = Instrument()
        for p in range(3):
            r = resonance_radius_mm(p, .1, cfg)
            self.assertAlmostEqual(float(fp_kernel(np.array([r]), np.array([.1]), cfg)[0,0]), 1., places=8)
        self.assertGreater(resonance_radius_mm(0, .1, cfg), resonance_radius_mm(0, 0., cfg))

    def test_laser_convolution_preserves_mass_and_shift(self):
        from simulation.physics import frequency_grid, gaussian, convolve_laser
        v=frequency_grid(15., 1500)
        s=gaussian(v,.25,.4,15.)
        b=convolve_laser(s,v,.1,[(.2,1.)])
        self.assertAlmostEqual(float(b.sum()*(v[1]-v[0])),1.,places=10)
        self.assertAlmostEqual(float(v[np.argmax(b)]),.45,places=2)


class CameraChecks(unittest.TestCase):
    def test_camera_preserves_constant_field(self):
        from simulation.imaging import Camera, Detector
        d=Detector(Camera(size=64,pixel_um=10.,roi_mm=.25,radial_step_um=2.,psf_sigma_um=10.))
        out=d.profile(np.ones_like(d.radius_mm))
        np.testing.assert_allclose(out,1.,atol=2e-12)

    def test_image_and_annular_operator_agree(self):
        from simulation.imaging import Camera, Detector
        d=Detector(Camera(size=64,pixel_um=10.,roi_mm=.25,radial_step_um=2.,psf_sigma_um=8.))
        s=np.exp(-d.radius_mm**2/.08**2)
        image=d.image(s)
        np.testing.assert_allclose(d.annular_mean(image),d.profile(s),atol=2e-12)

    def test_noise_is_reproducible(self):
        from simulation.imaging import noisy_profile
        expected=np.linspace(.02,.2,30);counts=np.arange(1,31)*10
        a,sa=noisy_profile(expected,counts,2000,20,3,42)
        b,sb=noisy_profile(expected,counts,2000,20,3,42)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(sa,sb)

    def test_gaussian_psf_is_two_dimensional(self):
        from simulation.imaging import Camera,Detector
        d=Detector(Camera(size=64,pixel_um=10.,roi_mm=.25,radial_step_um=1.,psf_sigma_um=10.))
        r=d.radius_mm;source_sigma=.04;camera_sigma=.01
        blurred=d.blur@np.exp(-.5*(r/source_sigma)**2)
        expected=source_sigma**2/(source_sigma**2+camera_sigma**2)*np.exp(-r**2/(2*(source_sigma**2+camera_sigma**2)))
        np.testing.assert_allclose(blurred[r<.2],expected[r<.2],atol=.0005)

    def test_radial_psf_matches_independent_cartesian_convolution(self):
        from scipy.ndimage import gaussian_filter
        from simulation.imaging import Camera,Detector
        d=Detector(Camera(size=128,pixel_um=4.,roi_mm=.2,radial_step_um=1.,psf_sigma_um=10.))
        source=np.exp(-.5*(d.radius_mm/.04)**2)
        actual=d.image(source)
        fine_axis=(np.arange(512)-(512-1)/2)*.001
        xx,yy=np.meshgrid(fine_axis,fine_axis)
        cartesian=gaussian_filter(np.exp(-.5*(xx**2+yy**2)/.04**2),sigma=10.,mode='constant',truncate=7.)
        reference=cartesian.reshape(128,4,128,4).mean(axis=(1,3))
        np.testing.assert_allclose(actual[d.valid],reference[d.valid],atol=.0004)


class InverseChecks(unittest.TestCase):
    def test_wiener_inverts_independent_frequency_integral(self):
        from simulation.physics import Instrument,frequency_grid,gaussian,airy
        from simulation.inverse import wiener_reconstruct
        v=frequency_grid(15.,600);cfg=Instrument();s=gaussian(v,.23,.7,15.)
        h=airy(v,cfg)
        y=airy(v[:,None]-v[None,:],cfg)@s*(v[1]-v[0])
        rec=wiener_reconstruct(v,y,h,1e-5,regularizers=[1e-10])
        self.assertLess(np.linalg.norm(rec['spectrum']-s)/np.linalg.norm(s),.001)

    def test_wiener_selection_does_not_explode_at_noise_threshold(self):
        from simulation.physics import Instrument,frequency_grid,gaussian,airy,convolve_laser
        from simulation.inverse import wiener_reconstruct
        v=frequency_grid(15.,3000);cfg=Instrument()
        s=2/7*gaussian(v,.08,.16,15.)+5/14*gaussian(v,-.72,.16,15.)+5/14*gaussian(v,.88,.16,15.)
        s=convolve_laser(s,v,.01)
        transfer=np.fft.fft(np.fft.ifftshift(airy(v,cfg)))*(v[1]-v[0])
        y=np.fft.fftshift(np.fft.ifft(np.fft.fft(np.fft.ifftshift(s))*transfer).real)
        y+=np.random.default_rng(20263932).normal(0,.001,len(v))
        rec=wiener_reconstruct(v,y,airy(v,cfg),.001)
        self.assertLess(np.linalg.norm(rec['spectrum']-s)/np.linalg.norm(s),.15)

    def test_broad_spectrum_recovery(self):
        from simulation.physics import Instrument,frequency_grid,gaussian,fp_kernel
        from simulation.inverse import reconstruct
        cfg=Instrument();v=frequency_grid(15.,300);r=np.linspace(.02,1.35,400)
        k=fp_kernel(r,v,cfg)*(v[1]-v[0]);s=gaussian(v,.2,.7,15.)
        y=k@s;sig=np.full(y.size,.0002)
        result=reconstruct(k,y,sig,v,alphas=[1e-8,1e-7,1e-6,1e-5])
        self.assertTrue(result['converged'])
        self.assertGreaterEqual(result['spectrum'].min(),0.)
        self.assertLess(np.linalg.norm(result['spectrum']-s)/np.linalg.norm(s),.04)
        self.assertLess(np.linalg.norm(k@result['spectrum']-y)/np.linalg.norm(y),.005)

    def test_parametric_fit_recovers_unknown_shift_and_width(self):
        from simulation.physics import Instrument,frequency_grid,gaussian,fp_kernel
        from simulation.inverse import fit_parametric
        cfg=Instrument();v=frequency_grid(15.,1200);r=np.linspace(.05,1.35,400)
        k=fp_kernel(r,v,cfg)*(v[1]-v[0]);s=gaussian(v,.12,.65,15.)
        y=k@s
        fit=fit_parametric(k,y,np.full(len(y),.0002),v,'knudsen',laser_fwhm=0.)
        self.assertTrue(fit['success'])
        self.assertAlmostEqual(fit['parameters']['shift_ghz'],.12,places=5)
        self.assertAlmostEqual(fit['parameters']['sigma_ghz'],.65,places=5)

    def test_parametric_fit_accounts_for_known_laser_modes(self):
        from simulation.physics import Instrument,frequency_grid,gaussian,fp_kernel,convolve_laser
        from simulation.inverse import fit_parametric
        cfg=Instrument();v=frequency_grid(15.,1200);r=np.linspace(.05,1.35,400)
        modes=[(-.3,.3),(.2,.7)]
        k=fp_kernel(r,v,cfg)*(v[1]-v[0])
        s=convolve_laser(gaussian(v,.12,.65,15.),v,.03,modes)
        fit=fit_parametric(k,k@s,np.full(len(r),.0002),v,'knudsen',laser_fwhm=.03,laser_modes=modes)
        self.assertAlmostEqual(fit['parameters']['shift_ghz'],.12,places=5)
        self.assertAlmostEqual(fit['parameters']['sigma_ghz'],.65,places=5)


if __name__ == '__main__':
    unittest.main()
