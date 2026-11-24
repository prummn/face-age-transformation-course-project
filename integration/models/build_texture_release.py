"""Package the audited native-texture pilot with explicitly attributed public examples; exclude private portrait bytes."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]
TAG='research-mobei-texture-pilot-2026-11-24'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('stage',type=Path)
    p.add_argument('--pilot-run',type=Path)
    p.add_argument('--public-panels',type=Path)
    p.add_argument('--manifest',type=Path,default=ROOT/'integration/models/texture-pilot-20261124.json')
    a=p.parse_args();run=a.run.resolve();stage=a.stage.resolve()
    for required in ('DONE','summary.json','decision.json','browser-validation.json'):
        if not (run/required).is_file():raise ValueError('Complete the study and browser review first: '+required)
    spec=importlib.util.spec_from_file_location('bundle',Path(__file__).with_name('build_research_release.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    helper.TAG=TAG;helper.STAGE=stage;helper.ASSETS=stage/'assets';helper.PAYLOAD=stage/'payload'
    helper.ASSETS.mkdir(parents=True,exist_ok=True);helper.PAYLOAD.mkdir(exist_ok=True)
    items=[]
    for arm,name in [('control','generator-highres-control.pt'),('patch_gan','generator-highres-patch-gan.pt')]:
        source=run/arm/'generator_patch.pt';dest=helper.ASSETS/name
        if dest.exists() and helper.sha(dest)!=helper.sha(source):raise ValueError('Preserving different staged weight')
        if not dest.exists():shutil.copy2(source,dest)
        digest=helper.sha(dest);size=dest.stat().st_size
        items.append(dict(id=name,group='weights',format='file',filename=name,bytes=size,sha256=digest,
            files=[dict(path='deployment/shared-research/texture-pilot/'+name,bytes=size,sha256=digest)],
            assets=[dict(name=name,bytes=size,sha256=digest,tag=TAG)],
            status='Exploratory high-resolution generator parameter patch, not an encoder or a promoted default.'))
    files={}
    def jsonfile(relative,value):
        path=helper.generated(relative,helper.scrub(value));files[relative]=path
    evidence={}
    for name in ('experiment.json','summary.json','decision.json','common-audit.json','browser-validation.json','platform-final.json','native-change-summary.json'):
        path=run/name
        if path.exists():
            value=json.loads(path.read_text())
            if name=='common-audit.json':value={k:v for k,v in value.items() if k not in ('affected','cross_groups','missing_face')}
            evidence[name]=value
    for arm in ('control','patch_gan'):
        evidence[arm]={name:json.loads((run/arm/name).read_text()) for name in ('plan.json','completed.json')}
        logs=run/arm/'train.jsonl';relative='deployment/shared-research/texture-pilot/'+arm+'-train.json'
        jsonfile(relative,[json.loads(line) for line in logs.read_text().splitlines()])
    if a.pilot_run:
        pilot=a.pilot_run.resolve()
        evidence['initial500']={name:json.loads((pilot/name).read_text()) for name in ('summary.json','decision.json','native-change-summary.json')}
        evidence['initial500']['training']={arm:json.loads((pilot/arm/'completed.json').read_text()) for arm in ('control','patch_gan')}
        for path in (pilot/'source').iterdir():
            if path.is_file():files['deployment/shared-research/texture-pilot/initial500-source/'+path.name]=path
    if a.public_panels:
        panels=a.public_panels.resolve();credits=json.loads((panels/'public-panel-credits.json').read_text())
        for row in credits:
            if row['name'] not in ('ffhq_64376','ffhq_64919') or Path(row['file']).name!=row['file']:raise ValueError('Unexpected public panel')
            files['deployment/shared-research/texture-pilot/'+row['file']]=panels/row['file']
        jsonfile('deployment/shared-research/texture-pilot/public-panel-credits.json',credits)
    jsonfile('deployment/shared-research/texture-pilot/results.json',evidence)
    comparison=ROOT/'deployment/age-quality-final/research-20261124/encoder-and-audit.json'
    jsonfile('deployment/shared-research/texture-pilot/encoder-and-audit.json',json.loads(comparison.read_text()))
    training=[json.loads(line) for line in (run/'data/train.jsonl').read_text().splitlines()]
    source_manifest=[]
    for r in training:
        source_manifest.append({k:r[k] for k in ('name','image_id','reference_sha256','age_interval','source_age','age_kind',
            'age_confidence','publisher_gender','published_demographic_label','source_metadata','identity_group','source_face_short_edge') if k in r})
    jsonfile('deployment/shared-research/texture-pilot/training-sources.json',source_manifest)
    for path in (run/'source').iterdir():
        if path.is_file():files['deployment/shared-research/texture-pilot/source/'+path.name]=path
    files['deployment/shared-research/texture-pilot/SAM-LICENSE.txt']=ROOT/'deployment/model-files/face-age/src/SAM/LICENSE'
    files['deployment/shared-research/texture-pilot/report.md']=ROOT/'integration/quality/20261124_高龄纹理单卡对照结果.md'
    items.append(helper.archive('evidence',files))
    manifest=dict(schema_version=1,repository='prummn/face-age-transformation-course-project',release_tag=TAG,
        source_branch='dev/mobei-gan-portable-20261116',source_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        groups=dict(weights='Two small high-resolution generator research patches; original SAM weights also required.',
                    evidence='Audits, recipes, FFHQ sources, two source-attributed public panels; no private/APPA portraits or W+ codes.'),items=items)
    helper.save(a.manifest,manifest);helper.save(helper.ASSETS/'manifest.json',manifest)
    sums=''.join(helper.sha(path)+'  '+path.name+'\n' for path in sorted(helper.ASSETS.iterdir()) if path.name!='SHA256SUMS')
    (helper.ASSETS/'SHA256SUMS').write_text(sums)
    print(json.dumps(dict(tag=TAG,stage=str(stage),assets=[p.name for p in helper.ASSETS.iterdir()]),indent=2))

if __name__=='__main__':main()
