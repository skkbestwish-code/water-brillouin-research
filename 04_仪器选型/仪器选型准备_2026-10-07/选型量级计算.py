"""选型量级计算；理论预测，不是测量或完整装置信号预测。运行：python 选型量级计算.py"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import math
import json
import hashlib

OUT = Path(__file__).resolve().parent
DEP = OUT.parents[1] / '03_模拟与验收/代码包验收_2026-10-07/原包解压/常压水_瑞利布里渊_FP模拟_2026-10-01/.water_deps'
sys.path.insert(0, str(DEP))
import iapws
from iapws import IAPWS95

def shift(nm, angle_deg, temp_c):
    water = IAPWS95(T=temp_c + 273.15, P=0.101325, l=nm/1000)
    return 2 * water.n * water.w * math.sin(math.radians(angle_deg/2)) / (nm*1e-9) / 1e9

rows = []
for nm, angle in [(532,90),(532,180),(632.8,90),(632.8,180),(660,180)]:
    for temp in [20,25,30]:
        water = IAPWS95(T=temp+273.15,P=0.101325,l=nm/1000)
        b = shift(nm,angle,temp)
        rows.append(dict(wavelength_vacuum_nm=nm,internal_angle_deg=angle,temperature_C=temp,
                         refractive_index=water.n,sound_speed_m_s=water.w,Brillouin_shift_GHz=b,
                         d_shift_dT_MHz_K=(shift(nm,angle,temp+.01)-shift(nm,angle,temp-.01))/.02*1000))

result = {
    'status':'纯水物性模型量级计算；不代表物理验证、仪器验收或实验完成',
    'pressure_MPa':0.101325,
    'frequency_convention':'普通频率Hz；角度为水内入射与散射波矢夹角；真空波长',
    'sources':['https://iapws.org/technical-guidance/release/IAPWS-95','https://iapws.org/technical-guidance/release/Rindex'],
    'implementation':dict(iapws_version=iapws.__version__,dependency_path=str(DEP),
                          iapws95_sha256=hashlib.sha256((DEP/'iapws/iapws95.py').read_bytes()).hexdigest(),
                          auxiliary_properties_sha256=hashlib.sha256((DEP/'iapws/_iapws.py').read_bytes()).hexdigest()),
    'water_predictions':rows,
    'plane_air_gap_FP_candidates':[dict(FSR_GHz=f,air_gap_mm=299792458/(2*f*1e9)*1e3,
        effective_finesse=fin,IRF_FWHM_GHz=f/fin) for f in [20,25,30] for fin in [50,100,150]],
    'near_axis_air_gap_angular_shift_532nm_MHz':{str(a):299792458/(532e-9)*(a*1e-3)**2/2/1e6 for a in [.2,.3,.5]},
    'ideal_single_pass_FP_peak_to_minimum_dB':{str(f):10*math.log10(1+1/math.sin(math.pi/(2*f))**2) for f in [30,100]},
    '532nm_90deg_angle_0p1deg_first_order_MHz':shift(532,90,25)*1000*.5*math.radians(.1),
    'warning':'平面空气腔计算不适用于直接套算共焦FP或VIPA；真实透过率、有效精细度、信噪比待测。'
}
assert len(rows)==15 and all(math.isfinite(r['Brillouin_shift_GHz']) for r in rows)
assert abs(shift(532,180,25)/shift(532,90,25)-math.sqrt(2))<1e-12
assert 2*shift(532,180,30)>15
(OUT/'选型量级计算结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'rows':len(rows),'532nm_180deg_30C_GHz':shift(532,180,30),'checks':'有限值、角度公式比例、15 GHz 谱级风险检查通过'},ensure_ascii=False))
