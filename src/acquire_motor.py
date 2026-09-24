"""Version-pinned public MOTOR acquisition. Never overwrite immutable raw files."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,shutil
import requests
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/motor'
FILES=['Appendix 10_MOTOR_Registry_Data_Dictionary.csv','Appendix 11 data analysis code part 1.do',
       'Appendix 12 Data analysis code part 2.do','Appendix 13 MOTOR TRIAL DATASET.dta']
EXPECTED_DTA='01136a9e9ef177dd175f5f260b71eba365379c736ab9a8d80bb36135cbd1a064'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def acquire():
    RAW.mkdir(parents=True,exist_ok=True);manifest=ROOT/'data/manifest.json'
    old=json.loads(manifest.read_text()) if manifest.exists() else None
    rows=[];remote=None
    for name in FILES:
        target=RAW/name;original=ROOT/'work/data/round2/motor'/name
        if not target.exists():
            if original.exists():shutil.copy2(original,target);route='verified local discovery copy'
            else:
                if remote is None:
                    r=requests.get('https://data.mendeley.com/public-api/datasets/bgpmkpcwdt/files',params={'folder_id':'root','version':'1'},timeout=60);r.raise_for_status()
                    remote={x.get('filename') or x.get('name'):x for x in r.json()}
                r=requests.get(remote[name]['content_details']['download_url'],timeout=90);r.raise_for_status();target.write_bytes(r.content);route='official Mendeley public API'
        else:route='existing immutable local file'
        digest=sha(target)
        if name.endswith('.dta') and digest!=EXPECTED_DTA:raise RuntimeError('MOTOR version/hash changed; stop rather than silently substitute data')
        if old:
            prior=next(x for x in old['files'] if x['filename']==name)
            if digest!=prior['sha256']:raise RuntimeError('Immutable source hash changed: '+name)
        target.chmod(0o444)
        rows.append({'filename':name,'path':str(target.relative_to(ROOT)).replace('\\','/'),'bytes':target.stat().st_size,'sha256':digest,'initial_route':route})
    if not old:
        manifest.write_text(json.dumps({'dataset_doi':'10.17632/bgpmkpcwdt.1','version':1,'license':'CC BY 4.0',
            'landing_url':'https://data.mendeley.com/datasets/bgpmkpcwdt/1','created_utc':datetime.now(timezone.utc).isoformat(),'files':rows},indent=2),encoding='utf8')
    print(json.dumps({'raw_files_verified':len(rows),'dta_sha256':EXPECTED_DTA}))
    return RAW
if __name__=='__main__':acquire()
