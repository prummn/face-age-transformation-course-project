"""Build explicit public research bundles; never archive the workspace wholesale."""
import gzip
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from urllib.parse import urlsplit,urlunsplit

ROOT=Path(__file__).resolve().parents[2]
STAGE=ROOT/'deployment/research-release-20261117'
ASSETS=STAGE/'assets'
PAYLOAD=STAGE/'payload'
TAG='research-mobei-2026-11-17'
CHUNK=128*1024**2


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def scrub(obj):
    if isinstance(obj,dict):
        return {k:scrub(v) for k,v in obj.items() if not any(t in k.lower() for t in ['password','private_key','secret_key','api_key','session_id','authorization','embedding'])
                and not (isinstance(v,str) and v.startswith('data:'))}
    if isinstance(obj,list):return [scrub(v) for v in obj if not (isinstance(v,dict) and v.get('name')=='uploaded')]
    if isinstance(obj,str):
        if obj.startswith('data:'):return '[image payload excluded]'
        if obj.startswith(('https://','http://')):
            u=urlsplit(obj)
            if any(t in u.query.lower() for t in ['signature=','token=','credential=']):return urlunsplit((u.scheme,u.netloc,u.path,'',''))
        return obj.replace(str(ROOT)+'/', '').replace(str(ROOT),'[original repository root]')
    return obj


def generated(rel,data):
    path=PAYLOAD/rel;save(path,data);return path


def parts(path,name):
    if path.stat().st_size<=CHUNK:
        dest=ASSETS/name
        if not dest.exists():shutil.copyfile(path,dest)
        assert sha(dest)==sha(path)
        return [dict(name=name,bytes=dest.stat().st_size,sha256=sha(dest),tag=TAG)]
    result=[]
    with path.open('rb') as f:
        i=0
        while data:=f.read(CHUNK):
            dest=ASSETS/f'{name}.part{i:03d}';h=hashlib.sha256(data).hexdigest()
            if not dest.exists():dest.write_bytes(data)
            assert dest.stat().st_size==len(data) and sha(dest)==h
            result.append(dict(name=dest.name,bytes=len(data),sha256=h,tag=TAG));i+=1
    return result


def archive(group,files):
    path=STAGE/(group+'.tar.gz');inventory=[]
    for dest,source in sorted(files.items()):inventory.append(dict(path=dest,bytes=source.stat().st_size,sha256=sha(source)))
    stamp=STAGE/(group+'.inventory.json')
    if path.exists():
        if not stamp.exists() or json.loads(stamp.read_text())!=inventory:
            raise RuntimeError('Existing archive has a different source inventory; preserve it and use a new --stage')
    if not path.exists():
        temp=path.with_suffix('.building')
        with temp.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=1,mtime=0) as gz,tarfile.open(fileobj=gz,mode='w|') as tar:
            for dest,source in sorted(files.items()):
                assert source.is_file() and not source.is_symlink(),str(source)
                info=tar.gettarinfo(str(source),arcname=dest);info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o644
                with source.open('rb') as f:tar.addfile(info,f)
        temp.replace(path)
        save(stamp,inventory)
    item=dict(id=group,group=group,format='tar.gz',filename=path.name,bytes=path.stat().st_size,sha256=sha(path),files=inventory,assets=parts(path,path.name))
    print('BUNDLE_READY',group,len(files),round(item['bytes']/1024**2,2),'MiB',flush=True)
    return item


def collect_evidence():
    records={};index=[]
    runs=['20261113-040430','20261113-040634','20261116-131721','20261116-134401','20261116-140451',
          '20261117-014357','20261117-015703','20261117-075925','20261117-093112','20261117-095522','20261117-095925','20261117-100543','20261117-101216']
    for stamp in runs:
        root=ROOT/'deployment/training-runs'/stamp
        # Numeric reports and protocols only. No images, W+ codes, credentials,
        # session payloads, environment dumps, or private uploaded photographs.
        names=['experiment.json','summary.json','quality-summary.json','paired-summary.json','candidate-manifest.json',
               'source-age-audit.json','artifact-validation.json','native-validation.json','platform-final.json',
               'browser-validation.json','quality-browser-validation.json','runtime-source.json','assets.json',
               'repair-decision.json','repair-numerics-amendment.json','chroma-summary.json','chroma-assessment.json',
               'control-reproduction.json','external-summary.json','external-summary.diagnostic.json',
               'evaluation/report.json','latent-refinement/report.json','latent-constrained/report.json',
               'age-texture/report.json','external-results/report.json','chroma-repair/report.json']
        for sub in [p for p in root.iterdir() if p.is_dir() and not p.is_symlink()]:
            if (sub/'train.jsonl').exists():names += [sub.name+'/'+f for f in ['train.jsonl','completed.json','plan.json']]
        for name in names:
            source=root/name
            if not source.is_file():continue
            text=source.read_text()
            obj=[json.loads(line) for line in text.splitlines()] if source.suffix=='.jsonl' else json.loads(text)
            rel=f'deployment/shared-research/evidence/{stamp}/{name}'
            dest=PAYLOAD/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            sanitized=scrub(obj)
            dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sanitized) if source.suffix=='.jsonl' else json.dumps(sanitized,ensure_ascii=False,indent=2)+'\n')
            records[rel]=dest;index.append(dict(original=str(source.relative_to(ROOT)),original_sha256=sha(source),public_path=rel,public_sha256=sha(dest),sanitized=True))
    rel='deployment/shared-research/evidence/INDEX.json';records[rel]=generated(rel,dict(files=index,
        note='Source paths made relative; credential/session fields, image payloads, and per-uploaded-photo records excluded. Aggregate published metrics retained. Public files have different hashes from raw originals.'))
    return records


def main():
    global STAGE,ASSETS,PAYLOAD
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',type=Path,default=STAGE)
    STAGE=parser.parse_args().stage.resolve();ASSETS=STAGE/'assets';PAYLOAD=STAGE/'payload'
    ASSETS.mkdir(parents=True,exist_ok=True);PAYLOAD.mkdir(parents=True,exist_ok=True)
    notices={}
    source=ROOT/'deployment/data-research/20261113'
    for name in ['ffhq_readme.txt','ffhq_aging_readme.txt']:
        rel='deployment/shared-research/notices/'+name;notices[rel]=source/name
    for path in (ROOT/'integration/models/notices').rglob('*'):
        if path.is_file():notices['deployment/shared-research/notices/'+str(path.relative_to(ROOT/'integration/models/notices'))]=path
    for path in (STAGE/'notices').glob('*'):
        notices['deployment/shared-research/notices/'+path.name]=path
    items=[]
    evidence=collect_evidence();evidence.update(notices);items.append(archive('evidence',evidence))
    demo={}
    run=ROOT/'deployment/demo-hd/20261117-110722'
    for path in (run/'results').rglob('*'):
        if path.is_file() and path.suffix in ['.png','.webp','.json']:demo[str(path.relative_to(ROOT))]=path
    for name in ['browser-validation.json','native-validation.json','selection.json','report.json']:
        rel='deployment/demo-hd/20261117-110722/'+name
        demo[rel]=generated(rel,scrub(json.loads((run/name).read_text())))
    demo.update(notices);items.append(archive('demo',demo))
    data={};folder=ROOT/'deployment/training-data/ffhq-adult-v2';manifests=ROOT/'deployment/training-runs/20261113-040634/manifests'
    allrows=[]
    for split in ['train','eval']:
        rows=[json.loads(line) for line in (manifests/(split+'.jsonl')).read_text().splitlines()];allrows+=rows
        for row in rows:
            for field,hashkey in [('image_path','sha256'),('reference_path','reference_sha256'),('reference512_path','reference512_sha256')]:
                path=Path(row[field]);assert path.is_relative_to(folder) and sha(path)==row[hashkey]
                data[str(path.relative_to(ROOT))]=path
        rel=f'deployment/shared-research/manifests/{split}.portable.jsonl';dest=PAYLOAD/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(''.join(json.dumps(scrub(row),ensure_ascii=False)+'\n' for row in rows));data[rel]=dest
    labels=source/'ffhq_aging_labels.csv';data['deployment/shared-research/data/ffhq_aging_labels.csv']=labels
    plan=json.loads((folder/'collection-plan.json').read_text());ids={r['image_id'] for r in allrows}
    rel='deployment/shared-research/data/collection-subset.json';data[rel]=generated(rel,dict(
        original_plan_sha256=sha(folder/'collection-plan.json'),original_candidates=1760,
        packaged_train=1054,packaged_eval=80,samples=[r for r in plan['samples'] if r['image_id'] in ids],
        modifications='Official FFHQ alignment retained; INTER_AREA256 and512; fixed training/evaluation split. No training on APPA/user photos.',
        license='FFHQ collection CC BY-NC-SA4.0 and per-photo licenses in source_metadata; FFHQ-Aging attribution retained. Noncommercial course research.'))
    data.update(notices);items.append(archive('data',data))
    # Existing original weights stay in their original Release, byte-identical.
    wanted={'face-age/models/sam/sam_ffhq_aging.pt','face-age/out/ft_l1/encoder_step1000.pt',
        'face-age/models/faceattr/convnext_t_fairface.pt','face-age/models/yunet/face_detection_yunet_2023mar.onnx',
        'face-age/models/irse50/model_ir_se50.pth','face-age/torch_cache/hub/checkpoints/alexnet-owt-7be5be79.pth'}
    for row in json.loads((ROOT/'integration/models/weights.json').read_text()):
        if row['path'] not in wanted:continue
        dest='deployment/model-files/'+row['path'];assert sha(ROOT/dest)==row['sha256']
        items.append(dict(id=Path(row['path']).name,group='baseline',format='file',filename=Path(row['path']).name,
            bytes=row['bytes'],sha256=row['sha256'],files=[dict(path=dest,bytes=row['bytes'],sha256=row['sha256'])],
            assets=[dict(a,tag='models-2026-11-11') for a in row['assets']]))
    files=[('baseline','sface.onnx','deployment/gan-research/face_recognition_sface_2021dec.onnx')]
    candidates=[('weights100','20261113-040634',['encoder/encoder.pt','encoder_generator/encoder.pt','encoder_generator/generator_affines.pt']),
        ('weights500','20261116-131721',['id_control/encoder.pt','id_strong/encoder.pt']),
        ('weights800','20261116-134401',['recon_encoder/encoder.pt','recon_generator/encoder.pt','recon_generator/generator_patch.pt'])]
    for group,stamp,names in candidates:
        for name in names:files.append((group,stamp+'-'+name.replace('/','-'),f'deployment/training-runs/{stamp}/{name}'))
    for group,name,relative in files:
        path=ROOT/relative;h=sha(path)
        items.append(dict(id=name,group=group,format='file',filename=name,bytes=path.stat().st_size,sha256=h,
            files=[dict(path=relative,bytes=path.stat().st_size,sha256=h)],assets=parts(path,name),
            status='Evaluation helper' if group=='baseline' else 'Research checkpoint only; not promoted; direct encoder route; pair generator patch with matching encoder.'))
        print('WEIGHT_READY',name,round(path.stat().st_size/1024**2,1),'MiB',flush=True)
    manifest=dict(schema_version=1,repository='prummn/face-age-transformation-course-project',release_tag=TAG,
        source_branch='dev/mobei-gan-portable-20261116',source_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        groups=dict(demo='24 source-attributed HD portraits,120 conditions,raw/preserved240 PNGs; no GPU needed to browse.',
            evidence='Public numeric protocols, metrics, failures and sanitized records; no private/APPA images.',
            baseline='Original Release6 models plus SFace; original files unchanged; required for W+ quality inference.',
            data='Fixed1054 training+80 regression FFHQ,verified native1024,input256,reference512,portable manifests and credits.',
            weights100='Paired100-step encoder and generator-affine experiment; not promoted.',
            weights500='Paired500-step identity control and stronger identity experiment; not promoted.',
            weights800='Paired800-step reconstruction encoder and generator-patch experiment; not promoted.'),items=items)
    path=ROOT/'integration/models/research-20261117.json';save(path,manifest);shutil.copy2(path,ASSETS/'research-20261117.json')
    checks=''.join(f"{sha(p)}  {p.name}\n" for p in sorted(ASSETS.iterdir()) if p.is_file() and p.name!='SHA256SUMS')
    (ASSETS/'SHA256SUMS').write_text(checks)
    print('BUILD_COMPLETE',len(items),'items',len(list(ASSETS.iterdir())),'assets',flush=True)


if __name__=='__main__':main()
