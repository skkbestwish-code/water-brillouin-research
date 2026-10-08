"""OP-7423-6743-1 catalog scenario, not a calibrated instrument.

Malitson 1965, DOI 10.1364/JOSA.55.001205; wavelength in micrometres.
FSR uses the drawing, not a hidden substitution from nominal thickness.
"""
from dataclasses import asdict
import numpy as np
from .physics import WaterFP

def silica_indices(wavelength_nm):
    x=float(wavelength_nm)/1000
    if not np.isfinite(x) or not .21<=x<=3.71:
        raise ValueError('Malitson model restricted to 0.21--3.71 um.')
    b=np.array([.6961663,.4079426,.8974794])
    c=np.array([.0684043,.1162414,9.896161])**2
    n=float(np.sqrt(1+np.sum(b*x*x/(x*x-c))))
    derivative=float(np.sum(-b*x*c/(x*x-c)**2)/n)
    return n,n-x*derivative

def catalog_instrument():
    n,ng=silica_indices(632.8)
    return WaterFP(fsr_ghz=14.9896229,center_detuning_ghz=14.9896229/2,
        cavity_phase_index=n,cavity_group_index=ng,peak_transmission=.5,finesse=30.)

def provenance():
    cfg=catalog_instrument()
    return dict(data_class='synthetic_catalog_boundary_scenario',
        selected_by_user='6.743 mm fused silica OP-7423',
        drawing='op-7423-x-a-1-inch-etalon-various-thickness.pdf; Rev A 2014-08-18',
        catalog=dict(thickness_mm=6.743,fsr_cm_inverse=.5,thickness_and_fsr_tolerance_fraction=.01,
            reflectivity=.94,reflectivity_absolute_tolerance=.02,finesse_lower_exclusive=30,
            peak_transmission_lower_exclusive=.5,clear_aperture_mm=20,diameter_mm=25.4,
            thickness_uniformity_rms_nm_upper=1.5,coating_band_nm=[530,660]),
        theory=dict(index_source='https://opg.optica.org/josa/abstract.cfm?uri=josa-55-10-1205',
            phase_index=cfg.cavity_phase_index,group_index=cfg.group_index,
            model_temperature_c=20,coating_phase_dispersion='omitted',
            effective_thickness_from_nominal_fsr_mm=cfg.cavity_thickness_mm),
        assumptions=dict(finesse='30: conservative boundary, not measured',
            peak_transmission='0.5: conservative boundary, not measured',
            wavelength='632.8 nm: historical reference, not hardware confirmation',
            camera_and_focal_length='historical assumptions',
            center_detuning='FSR/2: chosen phase; physical setting requires adjustment/calibration'),
        instrument=asdict(cfg))
