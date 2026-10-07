# 水版基准与回归测试：IAPWS 物性基准、谱面积守恒、FP 腔介质独立性、
# 视场覆盖、前向算子一致性、条件拟合可辨识性与非法输入防护。
import unittest
from dataclasses import replace
import numpy as np
from water_simulation.physics import (Water, WaterFP, properties, line_parameters,
    physical_spectrum, folded_spectrum, integration_grid, fp_response,
    ring_radius, coverage, WaterDetector)
from simulation.physics import airy, frequency_grid
from simulation.imaging import Camera
from water_simulation.inverse import fit_water


class WaterTests(unittest.TestCase):
    # IAPWS-95 在 20 °C/常压/632.8 nm 下的基准值：密度、声速、折射率与 γ 范围。
    def test_iapws_room_temperature_and_wavelength(self):
        p=properties(Water())
        self.assertAlmostEqual(p['rho_kg_m3'],998.20715,places=4)
        self.assertAlmostEqual(p['sound_ms'],1482.34617,places=3)
        self.assertAlmostEqual(p['n_water'],1.33210688,places=7)
        self.assertGreater(p['gamma'],1)
        self.assertLess(p['gamma'],1.02)

    # 90° 与 180° 的波矢关系：频移比 √2、布里渊线宽比 2。
    def test_water_wavevector_and_backscatter(self):
        a=line_parameters(Water());b=line_parameters(Water(scattering_angle_deg=180))
        self.assertGreater(a['brillouin_ghz'],4.3)
        self.assertLess(a['brillouin_ghz'],4.5)
        self.assertAlmostEqual(b['brillouin_ghz']/a['brillouin_ghz'],np.sqrt(2))
        self.assertAlmostEqual(b['brillouin_hwhm_ghz']/a['brillouin_hwhm_ghz'],2)

    # 展开谱宽频积分应接近 1（<1），且 |ν|>7.5 GHz 外必须有非零洛伦兹尾
    # 功率——不允许用重归一化掩盖尾部损失。
    def test_physical_area_is_not_hidden_by_normalization(self):
        v,w=integration_grid()
        s=physical_spectrum(v,Water())
        self.assertGreater(float(w@s),.998)
        self.assertLess(float(w@s),1)
        self.assertGreater(float(w[abs(v)>7.5]@s[abs(v)>7.5]),.005)

    # 折叠谱面积严格为 1；额外弹性分量只抬高中心、不改变归一化。
    def test_fold_preserves_area_and_extra_elastic_is_independent(self):
        v=frequency_grid(15,6000)
        a=folded_spectrum(v,Water());b=folded_spectrum(v,Water(elastic_fraction=.2))
        self.assertAlmostEqual(float(a.sum()*(v[1]-v[0])),1,places=10)
        self.assertAlmostEqual(float(b.sum()*(v[1]-v[0])),1,places=10)
        self.assertGreater(b[np.argmin(abs(v))],a[np.argmin(abs(v))])

    # FP 腔介质独立：峰半宽不随腔折射率变化；环半径按 n 缩放；
    # 共振环处透过率严格为 1。
    def test_fp_medium_is_separate_and_fwhm_unchanged(self):
        a=WaterFP();b=replace(a,cavity_phase_index=1.46,cavity_group_index=1.46)
        self.assertAlmostEqual(airy(np.array([.25]),a)[0],.5,places=12)
        self.assertAlmostEqual(ring_radius(0,0,b)/ring_radius(0,0,a),1.46,places=3)
        for cfg in [a,b]:
            r=ring_radius(0,4.4,cfg)
            self.assertAlmostEqual(float(fp_response([r],[4.4],cfg)[0,0]),1,places=10)

    # 250 mm 焦距 + 1.46 腔时蓝移峰被 ROI 裁掉（双峰可见性判据生效）。
    def test_fov_reports_cropped_blue_peak(self):
        a=coverage(Water(scattering_angle_deg=180),WaterFP(),1.4)
        self.assertTrue(a['both_side_peaks_visible'])
        b=coverage(Water(scattering_angle_deg=180),WaterFP(focal_length_mm=250,
            cavity_phase_index=1.46,cavity_group_index=1.46),1.4)
        self.assertFalse(b['both_side_peaks_visible'])

    # 细网格前向矩阵与"先二维成像再环平均"在 2e-14 内一致。
    def test_physical_kernel_and_annular_image_match(self):
        d=WaterDetector(Camera(size=96,pixel_um=8,roi_mm=.35,radial_step_um=2,psf_sigma_um=4))
        v,w=integration_grid(core_step=.02,tail_step=.5,limit=30)
        s=physical_spectrum(v,Water())
        k,radial=d.physical_kernel(v,w,WaterFP(focal_length_mm=35))
        image=d.image(radial@s)
        np.testing.assert_allclose(d.annular_mean(image),k@s,atol=2e-14)

    # 默认相机在浮点舍入下仍得到 350 个环带（回归防护）。
    def test_350_bins_survive_float_roundoff(self):
        d=WaterDetector(Camera(size=720))
        self.assertEqual(d.nbins,350)

    # 非法参数（角度、弹性占比、腔折射率）必须抛 ValueError。
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):Water(scattering_angle_deg=0)
        with self.assertRaises(ValueError):Water(elastic_fraction=1.1)
        with self.assertRaises(ValueError):WaterFP(cavity_phase_index=0)

    # 无噪声合成数据下条件拟合精确恢复间距/宽度/频移（7 位小数）。
    def test_conditional_water_fit_recovers_spacing_and_width(self):
        d=WaterDetector(Camera(size=128,pixel_um=8,roi_mm=.48,radial_step_um=2,psf_sigma_um=2))
        v,w=integration_grid(core_step=.01,tail_step=.5,limit=30)
        water=Water(velocity_ms=15,central_fraction_override=.04)
        k,_=d.physical_kernel(v,w,WaterFP(focal_length_mm=40),False)
        expected=k@physical_spectrum(v,water)
        fit=fit_water(k,v,expected,np.full(len(expected),.001),water)
        p=line_parameters(water)
        self.assertTrue(fit['success'])
        self.assertFalse(fit['at_bound'])
        self.assertAlmostEqual(fit['parameters']['brillouin_ghz'],p['brillouin_ghz'],places=7)
        self.assertAlmostEqual(fit['parameters']['brillouin_hwhm_ghz'],p['brillouin_hwhm_ghz'],places=7)
        self.assertAlmostEqual(fit['parameters']['shift_ghz'],p['shift_ghz'],places=7)

    # 角展宽压低并展宽侧峰，但不改变 IAPWS 给出的水声速基准。
    def test_angular_spread_broadens_sideband_without_moving_water_sound(self):
        v=np.linspace(4,4.8,1601)
        a=physical_spectrum(v,Water());b=physical_spectrum(v,Water(angular_sigma_deg=2))
        self.assertLess(b.max(),a.max())
        self.assertEqual(properties(Water())['sound_ms'],properties(Water(angular_sigma_deg=2))['sound_ms'])


if __name__=='__main__':unittest.main()
