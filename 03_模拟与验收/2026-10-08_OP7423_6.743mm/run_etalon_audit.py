"""Frozen synthetic acceptance, independent seeds, all trials retained.
Run after run_water_simulation.py. Never overwrites a previous output folder.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,sys,time,hashlib
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
import scipy
from scipy.integrate import quad
from water_simulation.physics import (Water,WaterDetector,physical_spectrum,integration_grid,
    line_parameters,fp_response,components,coverage)
from water_simulation.etalon import catalog_instrument,provenance,silica_indices
from water_simulation.inverse import fit_water as unrestricted_fit_water
from functools import partial
fit_water=partial(unrestricted_fit_water,spacing_bounds=(.05,7.))
from simulation.physics import frequency_grid
from simulation.imaging import Camera,noisy_profile
from run_water_simulation import write_json,fit_summary,rel

THRESHOLDS=dict(forward_relative_l2=1e-4,noiseless_spacing_mhz=.01,
                spacing_rmse_mhz=1.,nonparametric_relative_l2=.1,quality_chi2_max=2.)

def stats(rows):
    valid=[r for r in rows if 'spacing_error_mhz' in r]
    accepted=[r for r in valid if r['quality_pass']]
    def summary(items):
        if not items:return None
        e=np.array([r['spacing_error_mhz'] for r in items])
        return dict(count=len(items),bias_mhz=float(e.mean()),std_mhz=float(e.std(ddof=1)) if len(e)>1 else None,
            rmse_mhz=float(np.sqrt(np.mean(e*e))),max_abs_mhz=float(abs(e).max()),
            rmse_pass=bool(np.sqrt(np.mean(e*e))<THRESHOLDS['spacing_rmse_mhz']))
    return dict(attempts=len(rows),exception_count=len(rows)-len(valid),
        quality_failures=len(rows)-len(accepted),quality_failure_rate=(len(rows)-len(accepted))/len(rows),
        central_boundary_warnings=sum(bool(r.get('quality_warning')) for r in valid),
        all_finite_fits=summary(valid),quality_accepted_fits=summary(accepted))

def run(out):
    if out.exists() and any(out.iterdir()):raise FileExistsError('Select an empty audit output folder.')
    out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();water=Water();cfg=catalog_instrument();camera=Camera()
    d=WaterDetector(camera);v,w=integration_grid();k,_=d.physical_kernel(v,w,cfg,False)
    source=physical_spectrum(v,water);expected=k@source;p=line_parameters(water)
    _,sigma=noisy_profile(expected,d.counts,2000,20,3,2026100800)
    meta=dict(provenance=provenance(),water=asdict(water),camera=asdict(camera),thresholds=THRESHOLDS,fit_spacing_bounds_ghz=[.05,7.],
        software=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__),
        noise=dict(signal_scales=[25,200,2000],background_electrons=20,read_sigma_electrons=3,replicates=50),
        note='Synthetic conditional precision, not measured apparatus accuracy; all finite attempts included in primary statistics.')
    write_json(out/'audit_config.json',meta)
    result=dict(configuration=meta,temperature=[],grid=[],monte_carlo={},mismatch={})
    for t in [20,25,30]:
        ww=replace(water,temperature_k=t+273.15);truth=line_parameters(ww)
        yy=k@physical_spectrum(v,ww);fit=fit_water(k,v,yy,sigma,ww)
        err=1000*(fit['parameters']['brillouin_ghz']-truth['brillouin_ghz'])
        result['temperature'].append(dict(temperature_c=t,truth_ghz=truth['brillouin_ghz'],
            spacing_error_mhz=err,pass_noiseless=abs(err)<THRESHOLDS['noiseless_spacing_mhz'],**fit_summary(fit)))
    print('Noiseless temperature checks done.',flush=True)
    previous=expected
    for factor in [1,2,4]:
        if factor==1:vv,ww,profile=v,w,expected
        else:
            vv,ww=integration_grid(.0025/factor,.1/factor,150)
            kk,_=d.physical_kernel(vv,ww,cfg,False);profile=kk@physical_spectrum(vv,water);del kk
        fit=fit_water(k,v,profile,sigma,water)
        result['grid'].append(dict(factor=factor,nodes=len(vv),relative_l2_to_previous=rel(profile,previous),
            spacing_error_mhz=1000*(fit['parameters']['brillouin_ghz']-p['brillouin_ghz'])))
        previous=profile
    # This tests quadrature independently, not an independent water-spectrum theory.
    points=sorted(set(mu+j*width for mu,width,area in components(water) if area>0 for j in [-10,-2,0,2,10]))
    radii=[.36,.71,1.1]
    independent=[quad(lambda x:float(physical_spectrum(np.array([x]),water)[0]*fp_response([r],[x],cfg)[0,0]),
        -150,150,points=points,epsabs=1e-9,limit=2000)[0] for r in radii]
    result['quadrature_max_abs_error']=float(np.max(abs(fp_response(radii,v,cfg)@(w*source)-independent)))
    # Direct dispersive optical-path expression; independent from fp_response.
    rr=np.linspace(0,1.4,45);vv=np.linspace(-12,12,121)
    nu0=299792458/(632.8e-9)/1e9;n0,ng=silica_indices(632.8)
    nn=np.array([silica_indices(299792458/((nu0+x)*1e9)*1e9)[0] for x in vv])
    external_sin2=rr[:,None]**2/(cfg.focal_length_mm**2+rr[:,None]**2)
    path=(nu0+vv)[None,:]*np.sqrt(nn[None,:]**2-external_sin2)
    detuning=cfg.center_detuning_ghz+(path-nu0*n0)/ng
    exact=.5/(1+(1/np.sin(np.pi/60)**2)*np.sin(np.pi*detuning/cfg.fsr_ghz)**2)
    result['linear_dispersion_vs_full_sellmeier_relative_l2']=rel(fp_response(rr,vv,cfg),exact)
    iv=frequency_grid(cfg.fsr_ghz,600);ik=d.kernel(iv,cfg)
    s=np.linalg.svd(ik,compute_uv=False);rank=int(np.sum(s>s[0]*max(ik.shape)*np.finfo(float).eps))
    result['identifiability']=dict(shape=list(ik.shape),numerical_rank=rank,
        algebraic_nullity_lower_bound=ik.shape[1]-min(ik.shape),numerical_nullity=ik.shape[1]-rank,
        singular_ratio=float(s[0]/s[-1]),note='Threshold-dependent rank; regularization cannot prove unique arbitrary-spectrum recovery.')
    result['coverage']=coverage(water,cfg,camera.roi_mm)
    print('Grid, independent phase and rank checks done.',flush=True)
    for level,gain in enumerate([25,200,2000]):
        rows=[];observations=[];uncertainties=[]
        for j in range(50):
            seed=2026100800+level*1000+j
            y,ss=noisy_profile(expected,d.counts,gain,20,3,seed)
            observations.append(y);uncertainties.append(ss)
            row=dict(seed=seed,trial=j,signal_scale_electrons=gain)
            try:
                fit=fit_water(k,v,y,ss,water)
                row.update(fit_summary(fit))
                row.update(spacing_error_mhz=1000*(fit['parameters']['brillouin_ghz']-p['brillouin_ghz']),
                    intrinsic_hwhm_error_mhz=1000*(fit['parameters']['brillouin_hwhm_ghz']-p['brillouin_hwhm_ghz']))
            except Exception as exc:row.update(quality_pass=False,exception=repr(exc))
            rows.append(row)
            if (j+1)%10==0:print(f'Count scale {gain}: {j+1}/50',flush=True)
        np.savez_compressed(out/f'mc_{gain}_observations.npz',y=np.array(observations),sigma=np.array(uncertainties),
            expected=expected,radius_mm=d.bin_radius_mm,counts=d.counts,seeds=[r['seed'] for r in rows])
        result['monte_carlo'][str(gain)]=dict(summary=stats(rows),trials=rows)
        write_json(out/'audit_metrics.json',result)
    # Same noise-free baseline, fit with an intentionally wrong instrument.
    # This isolates deterministic bias from random sampling fluctuations.
    mismatch_cfg=[('assumed_FSR_minus_1pct',replace(cfg,fsr_ghz=cfg.fsr_ghz*.99)),
        ('assumed_FSR_plus_1pct',replace(cfg,fsr_ghz=cfg.fsr_ghz*1.01)),
        ('assumed_finesse_40',replace(cfg,finesse=40)),('assumed_finesse_50p7567',replace(cfg,finesse=50.75668179456)),
        ('assumed_focal_plus_1pct',replace(cfg,focal_length_mm=101)),
        ('assumed_air_cavity',replace(cfg,cavity_phase_index=1,cavity_group_index=1)),
        ('assumed_group_equals_phase',replace(cfg,cavity_group_index=cfg.cavity_phase_index)),
        ('assumed_peak_transmission_1',replace(cfg,peak_transmission=1))]
    for name,bad in mismatch_cfg:
        kk,_=d.physical_kernel(v,w,bad,False);fit=fit_water(kk,v,expected,sigma,water);del kk
        result['mismatch'][name]=dict(**fit_summary(fit),
            spacing_bias_mhz=1000*(fit['parameters']['brillouin_ghz']-p['brillouin_ghz']),
            width_bias_percent=100*(fit['parameters']['brillouin_hwhm_ghz']/p['brillouin_hwhm_ghz']-1),
            direction='true catalog boundary instrument; fit uses altered assumption')
    # Conversely, actual finesse may be greater than its lower bound.
    for finesse in [40,50.75668179456]:
        truth_cfg=replace(cfg,finesse=finesse)
        kk,_=d.physical_kernel(v,w,truth_cfg,False);yy=kk@source;del kk
        _,ss=noisy_profile(yy,d.counts,2000,20,3,99)
        fit=fit_water(k,v,yy,ss,water)
        result['mismatch'][f'true_finesse_{finesse:g}_assumed_30']=dict(**fit_summary(fit),
            spacing_bias_mhz=1000*(fit['parameters']['brillouin_ghz']-p['brillouin_ghz']),
            width_bias_percent=100*(fit['parameters']['brillouin_hwhm_ghz']/p['brillouin_hwhm_ghz']-1),
            direction='true instrument sharper; fit assumes catalog boundary finesse30')
    for electrons in [-5,-1,1,5]:
        fit=fit_water(k,v,expected+electrons/2000,sigma,water)
        result['mismatch'][f'background_residual_{electrons:+d}e']=dict(**fit_summary(fit),
            spacing_bias_mhz=1000*(fit['parameters']['brillouin_ghz']-p['brillouin_ghz']),
            width_bias_percent=100*(fit['parameters']['brillouin_hwhm_ghz']/p['brillouin_hwhm_ghz']-1))
    result['elapsed_seconds']=time.perf_counter()-start
    result['numerical_pass']=bool(all(x['pass_noiseless'] for x in result['temperature']) and
        max(x['relative_l2_to_previous'] for x in result['grid'])<THRESHOLDS['forward_relative_l2'] and
        result['linear_dispersion_vs_full_sellmeier_relative_l2']<THRESHOLDS['forward_relative_l2'])
    write_json(out/'audit_metrics.json',result)
    print('Audit finished:',result['elapsed_seconds'],'seconds',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('output/acceptance'))
    run(parser.parse_args().output)
