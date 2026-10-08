"""Second-pass model and inverse diagnostics; no experimental claims."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,time
from pathlib import Path
from dataclasses import replace,asdict
from functools import partial
import numpy as np
from numpy.polynomial.hermite import hermgauss
from water_simulation.physics import (Water,WaterDetector,physical_spectrum,folded_spectrum,
    integration_grid,line_parameters,components,spectrum_from_lines)
from water_simulation.etalon import catalog_instrument
from water_simulation.inverse import fit_water
from simulation.imaging import Camera,noisy_profile
from simulation.physics import frequency_grid
from simulation.inverse import reconstruct
from run_water_simulation import write_json,fit_summary,rel

fit_conditional=partial(fit_water,spacing_bounds=(.05,7.))

def judgments(r):
    rows=r['window']+r['spatial']+r['angular']
    return dict(forward_default_all_comparisons_below_1e_4=all(x['profile_l2_vs_baseline']<1e-4 for x in rows),
        conditional_B_discretization_below_0p01mhz=all(abs(x['B_bias_mhz'])<.01 for x in rows),
        note='Pairwise numerical diagnostics, not a bound on true experimental error. No cancellation or quadrature sum assumed.')

def run(out):
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use an empty output folder.')
    out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    cfg=catalog_instrument();water=Water();d=WaterDetector(Camera());v,w=integration_grid()
    k,radial=d.physical_kernel(v,w,cfg);s=physical_spectrum(v,water);base=k@s
    _,sigma=noisy_profile(base,d.counts,2000,20,3,123)
    truth=line_parameters(water)['brillouin_ghz']
    r=dict(data_class='synthetic_secondary_review',baseline=dict(instrument=asdict(cfg),camera=asdict(d.camera),
        water=asdict(water),B_bounds_ghz=[.05,7.]),window=[],spatial=[],angular=[],inverse_grid=[],nonparametric_noise={},alias={},local_sensitivity={})
    # Widen integration support: a different check from halving the node spacing.
    previous=base
    for limit in [150,300,600]:
        vv,ww=integration_grid(limit=limit)
        if limit==150:yy=base;area=float(w@s)
        else:
            kk,_=d.physical_kernel(vv,ww,cfg,False);ss=physical_spectrum(vv,water);yy=kk@ss;area=float(ww@ss);del kk
        f=fit_conditional(k,v,yy,sigma,water)
        r['window'].append(dict(limit_ghz=limit,area=area,profile_l2_vs_baseline=rel(yy,base),
            profile_l2_vs_previous=rel(yy,previous),B_bias_mhz=1000*(f['parameters']['brillouin_ghz']-truth)))
        previous=yy
    print('Window extension done.',flush=True)
    previous=base
    for dr,sub in [(1,4),(.5,6),(.25,8)]:
        if dr==1:yy=base
        else:
            dd=WaterDetector(replace(d.camera,radial_step_um=dr,subpixels=sub));kk,_=dd.physical_kernel(v,w,cfg,False);yy=kk@s;del kk,dd
        f=fit_conditional(k,v,yy,sigma,water)
        r['spatial'].append(dict(radial_step_um=dr,subpixels=sub,profile_l2_vs_previous=rel(yy,previous),
            profile_l2_vs_baseline=rel(yy,base),B_bias_mhz=1000*(f['parameters']['brillouin_ghz']-truth)))
        previous=yy
    print('Spatial refinement done.',flush=True)
    aw=replace(water,angular_sigma_deg=2);baseline_angular=k@physical_spectrum(v,aw)
    previous=baseline_angular
    for order in [15,31,61]:
        nodes,weights=hermgauss(order);angles=90+np.sqrt(2)*2*nodes
        valid=(angles>0)&(angles<=180);weights=weights[valid]/weights[valid].sum();ss=np.zeros_like(v)
        for angle,weight in zip(angles[valid],weights):
            ss+=weight*physical_spectrum(v,replace(water,scattering_angle_deg=float(angle)))
        yy=k@ss;f=fit_conditional(k,v,yy,sigma,aw)
        r['angular'].append(dict(order=order,profile_l2_vs_previous=rel(yy,previous),
            profile_l2_vs_baseline=rel(yy,baseline_angular),B_bias_mhz=1000*(f['parameters']['brillouin_ghz']-truth)))
        previous=yy
    print('Angular quadrature refinement done.',flush=True)
    # Prior-free 180-deg diagnostic: report both minima, not a tiny local SE alone.
    bw=replace(water,scattering_angle_deg=180);yy=k@physical_spectrum(v,bw)
    obs,ss=noisy_profile(yy,d.counts,2000,20,3,20261002)
    unrestricted=fit_water(k,v,obs,ss,bw,spacing_starts=[.5,4.2,6.5,8.8])
    restricted=fit_conditional(k,v,obs,ss,bw)
    r['alias']=dict(truth_ghz=line_parameters(bw)['brillouin_ghz'],unrestricted=fit_summary(unrestricted),restricted=fit_summary(restricted))
    np.savez_compressed(out/'alias_observation.npz',observation=obs,sigma=ss,expected=yy)
    # Actual ring-center displacement: generate shifted 2D image then average about wrong centre.
    # Response radial includes PSF blur, so noise-free geometry bias is isolated.
    blurred=np.asarray(d.blur@(radial@s))
    r['ring_center']=[]
    for pixels in [.25,.5,1.]:
        image=np.zeros_like(d.xx)
        for dx in d._offsets():
            for dy in d._offsets():
                rr=np.hypot(d.xx+dx-pixels*d.pixel_mm,d.yy+dy)
                image+=np.interp(rr,d.radius_mm,blurred)
        image/=d.camera.subpixels**2;yy=d.annular_mean(image)
        f=fit_conditional(k,v,yy,sigma,water)
        r['ring_center'].append(dict(offset_pixels=pixels,profile_relative_l2=rel(yy,base),B_bias_mhz=1000*(f['parameters']['brillouin_ghz']-truth),chi2=f['chi2']))
    # Small, symmetric perturbations, distinct from the earlier +/-1% stress test.
    for field in ['fsr_ghz','focal_length_mm']:
        rows=[]
        for fraction in [-.0001,.0001]:
            altered=replace(cfg,**{field:getattr(cfg,field)*(1+fraction)})
            kk,_=d.physical_kernel(v,w,altered,False);f=fit_conditional(kk,v,base,sigma,water);del kk
            rows.append(dict(fraction=fraction,B_bias_mhz=1000*(f['parameters']['brillouin_ghz']-truth),chi2=f['chi2']))
        derivative=(rows[1]['B_bias_mhz']-rows[0]['B_bias_mhz'])/.0002
        r['local_sensitivity'][field]=dict(rows=rows,mhz_per_fraction=derivative,
            illustrative_fraction_for_1mhz=1/abs(derivative),illustrative_fraction_for_0p1mhz=.1/abs(derivative))
    delta=.01
    bplus=line_parameters(replace(water,scattering_angle_deg=90+delta))['brillouin_ghz']
    bminus=line_parameters(replace(water,scattering_angle_deg=90-delta))['brillouin_ghz']
    tplus=line_parameters(replace(water,temperature_k=293.15+delta))['brillouin_ghz']
    tminus=line_parameters(replace(water,temperature_k=293.15-delta))['brillouin_ghz']
    r['physical_slopes']=dict(B_mhz_per_degree=(bplus-bminus)*1000/(2*delta),B_mhz_per_kelvin=(tplus-tminus)*1000/(2*delta))
    print('Alias, centre and local sensitivity checks done.',flush=True)
    fv=frequency_grid(cfg.fsr_ghz,12000);true_fold=folded_spectrum(fv,water)
    # Use exactly the same noisy observation for inverse-grid comparison.
    obs,ss=noisy_profile(base,d.counts,2000,20,3,2026100800)
    grid_curves=[]
    for count in [300,600,1200]:
        iv=frequency_grid(cfg.fsr_ghz,count);ik=d.kernel(iv,cfg)
        f=reconstruct(ik,obs,ss,iv);rec=np.interp(fv,iv,f['spectrum'],period=cfg.fsr_ghz)
        coarse_truth=folded_spectrum(iv,water)
        # Sampling/interpolation diagnostic, not an asserted lower bound over all possible coefficients.
        representation=np.interp(fv,iv,coarse_truth,period=cfg.fsr_ghz)
        r['inverse_grid'].append(dict(nodes=count,step_mhz=cfg.fsr_ghz/count*1000,spectral_relative_l2=rel(rec,true_fold),
            truth_sampling_interpolation_relative_l2=rel(representation,true_fold),alpha=f['alpha'],chi2=f['chi2'],diagnostics=f['diagnostics']))
        grid_curves.append(rec)
    np.savez_compressed(out/'inverse_grid_curves.npz',frequency_ghz=fv,truth=true_fold,recovered=np.array(grid_curves))
    # Noise repeats of full-spectrum recovery, not conditional parameter fitting.
    iv=frequency_grid(cfg.fsr_ghz,600);ik=d.kernel(iv,cfg)
    for index,(name,ww) in enumerate([('90deg',water),('180deg',bw),('elastic10pct',replace(water,elastic_fraction=.1))]):
        yy=k@physical_spectrum(v,ww);ts=folded_spectrum(fv,ww);rows=[];observations=[]
        for j in range(5):
            seed=2026100900+index*100+j;obs,ss=noisy_profile(yy,d.counts,2000,20,3,seed)
            f=reconstruct(ik,obs,ss,iv);rec=np.interp(fv,iv,f['spectrum'],period=cfg.fsr_ghz)
            rows.append(dict(seed=seed,spectral_relative_l2=rel(rec,ts),alpha=f['alpha'],chi2=f['chi2'],diagnostics=f['diagnostics']))
            observations.append(obs)
        errors=[x['spectral_relative_l2'] for x in rows]
        r['nonparametric_noise'][name]=dict(trials=rows,mean=float(np.mean(errors)),min=float(np.min(errors)),max=float(np.max(errors)),
            failures_above_10pct=sum(x>=.1 for x in errors),note='Only 5 diagnostic repeats; not a precise failure-rate estimate.')
        np.savez_compressed(out/f'nonparametric_{name}.npz',observations=np.array(observations),sigma=ss,expected=yy)
        print('Nonparametric repeats done:',name,flush=True)
    r['elapsed_seconds']=time.perf_counter()-start
    r['judgments']=judgments(r)
    write_json(out/'deep_review.json',r)
    print('Deep review completed',r['elapsed_seconds'],'seconds',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('output/deep_review'))
    run(p.parse_args().output)
