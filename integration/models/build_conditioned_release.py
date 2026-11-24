"""Package two completed latent/conditional studies; no training photos or W+ codes."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]
TAG='research-mobei-age-conditioned-2026-11-24'
PREFIX='deployment/shared-research/age-conditioned/'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('matched_run',type=Path);p.add_argument('conditional_run',type=Path);p.add_argument('stage',type=Path)
    p.add_argument('--public-panels',type=Path,required=True)
    a=p.parse_args();matched=a.matched_run.resolve();conditional=a.conditional_run.resolve();stage=a.stage.resolve()
    spec=importlib.util.spec_from_file_location('bundle',Path(__file__).with_name('build_research_release.py'))
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    h.TAG=TAG;h.STAGE=stage;h.ASSETS=stage/'assets';h.PAYLOAD=stage/'payload'
    h.ASSETS.mkdir(parents=True,exist_ok=True);h.PAYLOAD.mkdir(exist_ok=True)
    items=[];files={};index=[]
    def generated(relative,value):
        path=h.generated(PREFIX+relative,h.scrub(value));files[PREFIX+relative]=path
    for label,run,arms in [('matched',matched,['direct_latent','matched_wplus']),('conditional',conditional,['control','patch_gan'])]:
        for name in ['DONE','summary.json','decision.json','browser-validation.json','paired-verification.json']:
            if not (run/name).is_file():raise ValueError('Finish study first: '+str(run/name))
        decision=json.loads((run/'decision.json').read_text())
        if not decision.get('visually_reviewed'):raise ValueError('Native review is required before packaging')
        for arm in arms:
            source=run/arm/'generator_patch.pt';completed=json.loads((run/arm/'completed.json').read_text())
            if h.sha(source)!=completed['patch_sha256']:raise ValueError('Checkpoint changed')
            name=label+'-'+arm+'.pt';destination=h.ASSETS/name
            if destination.exists() and h.sha(destination)!=completed['patch_sha256']:raise ValueError('Preserving different staged weight')
            if not destination.exists():shutil.copy2(source,destination)
            size=destination.stat().st_size;digest=h.sha(destination)
            items.append(dict(id=name,group='weights',format='file',filename=name,bytes=size,sha256=digest,
                files=[dict(path=PREFIX+name,bytes=size,sha256=digest)],assets=[dict(name=name,bytes=size,sha256=digest,tag=TAG)],
                status='Unpromoted partial StyleGAN2 generator parameters, not a complete SAM model; conditional patches require age-dependent parameter interpolation.'))
            generated(label+'/'+arm+'-train.json',[json.loads(x) for x in (run/arm/'train.jsonl').read_text().splitlines()])
            for name in ['plan.json','completed.json']:generated(label+'/'+arm+'-'+name,json.loads((run/arm/name).read_text()))
        for name in ['experiment.json','summary.json','decision.json','browser-validation.json','paired-verification.json','native-change-summary.json',
                     'independent-age.json','young-age-invariance.json','platform-final.json','automatic-completion.json','artifact-validation.json','evaluation-correction.json','batch-shape-diagnostic.json']:
            path=run/name
            if path.is_file():generated(label+'/'+name,json.loads(path.read_text()))
        for path in (run/'source').iterdir():
            if path.is_file():files[PREFIX+label+'/source/'+path.name]=path
        for path in (run/'source').iterdir():
            if path.is_file():index.append(dict(study=label,file=path.name,sha256=h.sha(path)))
        if (run/'evaluation-recovery').exists():
            for name in ['provenance.json','receipt.json','platform-final.json']:
                path=run/'evaluation-recovery'/name
                if path.is_file():generated(label+'/evaluation-recovery/'+name,json.loads(path.read_text()))
            for path in (run/'evaluation-recovery/source').iterdir():
                if path.is_file():
                    files[PREFIX+label+'/evaluation-recovery/source/'+path.name]=path
                    index.append(dict(study=label,file='evaluation-recovery/source/'+path.name,sha256=h.sha(path)))
    generated('frozen-source-index.json',index)
    for name in ['20261124_Wplus匹配训练对照.md','20261124_年龄条件纹理与年龄保护.md']:
        files[PREFIX+name]=ROOT/'integration/quality'/name
    for name in ['finish_matched_latent.py','summarize_highres_texture.py','measure_texture_changes.py','check_independent_age.py','build_texture_review.py','verify_texture_review_browser.py',
                 'reevaluate_conditional_texture.py','build_public_texture_panels.py','collect_mcst_originals.py']:
        files[PREFIX+'postprocess/'+name]=ROOT/'integration/quality'/name
    source_rows=[json.loads(x) for x in (conditional/'data/train.jsonl').read_text().splitlines()]
    fields=['name','image_id','reference_sha256','age_interval','source_age','age_kind','age_confidence','publisher_gender','published_demographic_label','source_metadata','identity_group','source_face_short_edge']
    generated('training-sources.json',[{k:r[k] for k in fields if k in r} for r in source_rows])
    canonical=ROOT/'deployment/age-quality-final/commons-canonical-source-20261124'
    for name in ['intake-outcome.json','archive-provenance.json','original-image-inventory.json','original-quality-screen.json']:
        generated('new-source-audit/'+name,json.loads((canonical/name).read_text()))
    files[PREFIX+'SAM-LICENSE.txt']=ROOT/'deployment/model-files/face-age/src/SAM/LICENSE'
    for name in ['ffhq_readme.txt','ffhq_aging_readme.txt']:
        files[PREFIX+'licenses/'+name]=ROOT/'deployment/data-research/20261113'/name
    for notice in (ROOT/'deployment/model-files/face-age/src/SAM/licenses').iterdir():
        if notice.is_file():files[PREFIX+'licenses/'+notice.name]=notice
    credits=json.loads((a.public_panels/'public-panel-credits.json').read_text())
    for row in credits:
        if row['name'] not in ('ffhq_64376','ffhq_64919') or Path(row['file']).name!=row['file']:raise ValueError('Unexpected public portrait')
        path=a.public_panels/row['file'];name=row['file'];digest=h.sha(path);size=path.stat().st_size
        items.append(dict(id=name,group='examples',format='file',filename=name,bytes=size,sha256=digest,
            files=[dict(path=PREFIX+name,bytes=size,sha256=digest)],assets=h.parts(path,name)))
        files[PREFIX+name]=path
    generated('public-panel-credits.json',credits)
    items.append(h.archive('evidence',files))
    manifest=dict(schema_version=1,repository='prummn/face-age-transformation-course-project',release_tag=TAG,source_branch='dev/mobei-gan-portable-20261116',
        source_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        training_source_commits={label:json.loads((run/'experiment.json').read_text())['code_commit'] for label,run in [('matched',matched),('conditional',conditional)]},
        groups=dict(weights='Four unpromoted partial generator checkpoints. Conditional patches must be interpolated by target age; original SAM weights also required.',
                    evidence='Frozen source, training logs, independent metrics, decisions and source/quality audit. No private/APPA portrait bytes or W+ codes.',
                    examples='Two attributed public1024-output comparisons and unscaled native crops, research only.'),items=items)
    h.save(ROOT/'integration/models/age-conditioned-20261124.json',manifest);h.save(h.ASSETS/'manifest.json',manifest)
    (h.ASSETS/'SHA256SUMS').write_text(''.join(h.sha(path)+'  '+path.name+'\n' for path in sorted(h.ASSETS.iterdir()) if path.name!='SHA256SUMS'))
    print(json.dumps(dict(tag=TAG,stage=str(stage),assets=[p.name for p in h.ASSETS.iterdir()]),indent=2))


if __name__=='__main__':main()
