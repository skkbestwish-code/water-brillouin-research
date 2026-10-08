"""Run tests, simulation, acceptance and report in a fresh timestamped directory."""
import os,sys,subprocess,json,hashlib
from pathlib import Path
from datetime import datetime

def main():
    root=Path(__file__).resolve().parent
    out=root/'output'/('rerun_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    source_files=list(root.glob('*.py'))
    for folder in ['water_simulation','simulation','tests']:
        source_files+=list((root/folder).glob('*.py'))
    (out/'run_source_hashes.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_files},indent=2),encoding='utf-8')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONIOENCODING='utf-8')
    jobs=[('tests',['-m','unittest','discover','-s','tests','-v']),
          ('cases',['run_water_simulation.py','--output',str(out/'cases')]),
          ('audit',['run_etalon_audit.py','--output',str(out/'audit')]),
          ('deep',['run_deep_review.py','--output',str(out/'deep')]),
          ('detailed_report',['make_detailed_report.py','--cases',str(out/'cases'),'--audit',str(out/'audit'),'--deep',str(out/'deep'),'--dest',str(out)])]
    for name,args in jobs:
        print('Running',name,flush=True)
        with (out/(name+'.log')).open('w',encoding='utf-8') as log:
            proc=subprocess.Popen([sys.executable,*args],cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8')
            for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            if proc.wait():raise RuntimeError(f'{name} failed; see {out}')
    print('All stages completed. Read:',out/'研究汇报.md')

if __name__=='__main__':main()
