"""只读调用原包，结果另存；验收诊断不替代预注册验证或实验。"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys, json, hashlib, math, platform
from pathlib import Path
from dataclasses import replace
import numpy as np

ROOT=Path(__file__).resolve().parent
PKG=ROOT/'原包解压'/'常压水_瑞利布里渊_FP模拟_2026-10-01'
sys.path.insert(0,str(PKG))
from water_simulation.physics import Water,WaterFP,WaterDetector,properties,line_parameters,physical_spectrum,integration_grid
from water_simulation.inverse import fit_water
from simulation.imaging import Camera
from simulation.physics import frequency_grid
from run_water_simulation import DEFAULTS,clean,fit_summary

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))
def save(name,data):
    (ROOT/name).write_text(json.dumps(clean(data),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

rows=json.loads((PKG/'文件清单.json').read_text(encoding='utf-8-sig'))
checks=[]
for row in rows:
    p=PKG/row['path']
    checks.append(dict(path=row['path'],exists=p.is_file(),matches=p.is_file() and digest(p).lower()==row['sha256'].lower()))
zip_path=ROOT.parents[1]/'05_原始归档'/'常压水_瑞利布里渊_FP模拟_2026-10-01_注释版.zip'
save('原包完整性.json',dict(zip_sha256=digest(zip_path),manifest_count=len(rows),all_match=all(r['matches'] for r in checks),files=checks))

print('构造默认前向算子',flush=True)
water=Water();cfg=WaterFP();det=WaterDetector(Camera())
v,w=integration_grid();k,_=det.physical_kernel(v,w,cfg,False)
source=physical_spectrum(v,water);expected=k@source
sigma=np.sqrt((DEFAULTS['gain_electrons']*expected+DEFAULTS['background_electrons']+DEFAULTS['read_noise_electrons']**2)/det.counts)/DEFAULTS['gain_electrons']
out=dict(data_class='simulation_audit_only',scope='原包之外的本轮诊断；未预注册精度预算，不作为G2整体通过证据',environment=dict(python=sys.version,platform=platform.platform()),temperatures=[],grids=[],background_probes=[],fit_domain_probes=[])

for temp in [20,25,30]:
    ww=replace(water,temperature_k=temp+273.15);p=properties(ww);lp=line_parameters(ww)
    # 独立表达式，不调用 line_parameters 内的 q 路径；物性仍共用同一 IAPWS，不能算独立实测。
    b=2*p['n_water']*p['sound_ms']*math.sin(math.radians(ww.scattering_angle_deg/2))/(ww.wavelength_nm*1e-9)/1e9
    y=k@physical_spectrum(v,ww);fit=fit_water(k,v,y,sigma,ww)
    out['temperatures'].append(dict(temperature_C=temp,n=p['n_water'],sound_m_s=p['sound_ms'],formula_B_GHz=b,code_B_GHz=lp['brillouin_ghz'],formula_abs_difference_GHz=abs(b-lp['brillouin_ghz']),fit=fit_summary(fit),fit_B_error_MHz=(fit['parameters']['brillouin_ghz']-b)*1000))
    print('无噪声温度',temp,'完成',flush=True)

previous=expected
for factor in [1,2,4]:
    vv,weights=integration_grid(.0025/factor,.1/factor,150)
    if factor==1:profile=expected
    else:
        kk,_=det.physical_kernel(vv,weights,cfg,False)
        profile=kk@physical_spectrum(vv,water)
        del kk
    fit=fit_water(k,v,profile,sigma,water)
    out['grids'].append(dict(step_factor=factor,points=len(vv),core_step_GHz=.0025/factor,tail_step_GHz=.1/factor,relative_L2_vs_previous=rel(profile,previous),relative_L2_vs_default=rel(profile,expected),fit_with_default_grid=fit_summary(fit),B_error_MHz=1000*(fit['parameters']['brillouin_ghz']-line_parameters(water)['brillouin_ghz'])))
    previous=profile
    print('频率网格',factor,'完成',flush=True)

for delta in [-5,-1,0,1,5]:
    # 生成端与扣除端背景电子数之差：每像素常数偏置，不是额外弹性谱线。
    y=expected+delta/DEFAULTS['gain_electrons']
    fit=fit_water(k,v,y,sigma,water)
    out['background_probes'].append(dict(residual_background_electrons_per_pixel=delta,fit=fit_summary(fit),B_error_MHz=1000*(fit['parameters']['brillouin_ghz']-line_parameters(water)['brillouin_ghz'])))

for angle in [10,60,90,180]:
    ww=replace(water,scattering_angle_deg=angle)
    y=k@physical_spectrum(v,ww);fit=fit_water(k,v,y,sigma,ww)
    out['fit_domain_probes'].append(dict(angle_deg=angle,true_B_GHz=line_parameters(ww)['brillouin_ghz'],fit=fit_summary(fit)))
inverse_k=det.kernel(frequency_grid(15,600),cfg)
sv=np.linalg.svd(inverse_k,compute_uv=False)
rank=int(np.sum(sv>sv[0]*max(inverse_k.shape)*np.finfo(float).eps))
out['inverse_identifiability']=dict(shape=list(inverse_k.shape),rank_tolerance=float(sv[0]*max(inverse_k.shape)*np.finfo(float).eps),numerical_rank=rank,nullity_at_least=600-350,numerical_nullity=600-rank,condition_ratio=float(sv[0]/sv[-1]),note='线性离散算子欠定；非负和平滑条件下仍不得宣称任意谱唯一恢复。')
save('补充诊断结果.json',out)

old=json.loads((PKG/'output/water_rb_fp/metrics.json').read_text(encoding='utf-8'))
new=json.loads((ROOT/'本轮默认重跑/metrics.json').read_text(encoding='utf-8'))
comparison=[]
for name,c in new['cases'].items():
    comparison.append(dict(case=name,true_B_GHz=c['physics']['brillouin_ghz'],fit_B_GHz=c['parametric']['parameters']['brillouin_ghz'],fit_B_error_MHz=1000*(c['parametric']['parameters']['brillouin_ghz']-c['physics']['brillouin_ghz']),nonparametric_relative_L2=c['nonparametric']['relative_l2_folded'],fit_delta_vs_old_GHz=c['parametric']['parameters']['brillouin_ghz']-old['cases'][name]['parametric']['parameters']['brillouin_ghz']))
save('新旧结果对照.json',dict(cases=comparison,numerical_checks=new['numerical_checks'],monte_carlo={name:dict(replicates=len(mc['trials']),optimizer_failures=sum(not t['success'] for t in mc['trials']),at_bound=sum(t['at_bound'] for t in mc['trials']),mean_error_MHz=mc['spacing_bias_mhz'],std_MHz=mc['spacing_std_mhz']) for name,mc in new['monte_carlo'].items()},sensitivity={name:dict(B_bias_MHz=r['spacing_bias_mhz'],width_bias_MHz=r['width_bias_mhz'],chi2=r['chi2'],success=r['success'],at_bound=r['at_bound']) for name,r in new['sensitivity'].items()}))
print('全部补充诊断完成',flush=True)
