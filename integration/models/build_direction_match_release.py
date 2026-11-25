"""Archive a completed direction-matching study without protected portraits/latents."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]
TAG='research-mobei-direction-match-2026-11-25'
PREFIX='deployment/shared-research/direction-match/'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('stage',type=Path);p.add_argument('--public-panels',type=Path,required=True);p.add_argument('--personalized',type=Path);p.add_argument('--personalized-confirmation',type=Path);p.add_argument('--personalized-panels',type=Path);a=p.parse_args()
    if a.personalized_panels and not a.personalized:raise ValueError('Per-photo panels require completed per-photo study')
    run=a.run.resolve();stage=a.stage.resolve()
    spec=importlib.util.spec_from_file_location('bundle',Path(__file__).with_name('build_research_release.py'));h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    h.TAG=TAG;h.STAGE=stage;h.ASSETS=stage/'assets';h.PAYLOAD=stage/'payload';h.ASSETS.mkdir(parents=True,exist_ok=True);h.PAYLOAD.mkdir(exist_ok=True)
    for name in ['DONE','summary.json','decision.json','paired-verification.json','artifact-validation.json','browser-validation.json','platform-final.json']:
        if not (run/name).is_file():raise ValueError('Complete the study first: '+name)
    decision=json.loads((run/'decision.json').read_text())
    if not decision.get('visually_reviewed'):raise ValueError('Native review required before publication')
    items=[];evidence={}
    def addjson(name,data):evidence[PREFIX+name]=h.generated(PREFIX+name,h.scrub(data))
    for label,arm in [('ft-delta','control'),('original-delta','patch_gan')]:
        source=run/arm/'generator_patch.pt';done=json.loads((run/arm/'completed.json').read_text())
        if h.sha(source)!=done['patch_sha256']:raise ValueError('Checkpoint changed')
        name=label+'-generator.pt';dest=h.ASSETS/name
        if not dest.exists():shutil.copy2(source,dest)
        if h.sha(dest)!=done['patch_sha256']:raise ValueError('Different staged checkpoint')
        size=dest.stat().st_size
        items.append(dict(id=label,group='weights',format='file',filename=name,bytes=size,sha256=done['patch_sha256'],files=[dict(path=PREFIX+name,bytes=size,sha256=done['patch_sha256'])],assets=[dict(name=name,bytes=size,sha256=done['patch_sha256'],tag=TAG)],status='Research partial StyleGAN2 parameters; original SAM required. Never automatically enabled.'))
        addjson(arm+'/train.json',[json.loads(s) for s in (run/arm/'train.jsonl').read_text().splitlines()])
        for name in ['plan.json','completed.json']:addjson(arm+'/'+name,json.loads((run/arm/name).read_text()))
    for name in ['experiment.json','summary.json','paired-effect-summary.json','decision.json','paired-verification.json','artifact-validation.json','browser-validation.json','native-change-summary.json','native-review-observations.json','independent-age.json','platform-final.json','automatic-completion.json','supplemental-publisher-labels.json']:
        if (run/name).is_file():addjson(name,json.loads((run/name).read_text()))
    for path in (run/'source').iterdir():
        if path.is_file():evidence[PREFIX+'source/'+path.name]=path
    for name in ['finish_original_delta_match.py','finish_small_quality.py','summarize_highres_texture.py','summarize_paired_generator_effects.py','check_independent_age.py','measure_texture_changes.py','build_texture_review.py','verify_texture_review_browser.py','build_public_texture_panels.py']:
        evidence[PREFIX+'postprocess/'+name]=ROOT/'integration/quality'/name
    evidence[PREFIX+'report.md']=ROOT/'integration/quality/20261125_原SAM差值匹配生成器训练.md'
    for variant in ['ft','original']:addjson('cache-'+variant+'-index.json',json.loads((run/('train-codes-'+variant)/'index.json').read_text()))
    fields=['name','reference_sha256','identity_group','source_age','age_interval','age_confidence','source_face_short_edge','source_metadata','published_demographic_label']
    rows=[json.loads(s) for s in (run/'data/train.jsonl').read_text().splitlines()]
    addjson('training-sources.json',[{k:r[k] for k in fields if k in r} for r in rows])
    for name in ['ffhq_readme.txt','ffhq_aging_readme.txt']:evidence[PREFIX+'licenses/'+name]=ROOT/'deployment/data-research/20261113'/name
    reserve=ROOT/'deployment/age-quality-final/fresh-hd-reserve-20261125'
    for name in ['collection-report.json','prepared/audit.json','prepared/source-qc-observations.json']:
        if (reserve/name).is_file():addjson('fresh-holdout/'+name,json.loads((reserve/name).read_text()))
    evidence[PREFIX+'fresh-holdout/report.md']=ROOT/'integration/quality/20261125_新增高清验证储备.md'
    commons=ROOT/'deployment/age-quality-final/commons-retry-20261125'
    for name in ['audit.json','records.json','rejected.json','count-amendment.json']:
        if (commons/name).is_file():addjson('commons-intake/'+name,json.loads((commons/name).read_text()))
    evidence[PREFIX+'commons-intake/report.md']=ROOT/'integration/quality/20261125_高龄公开来源重试与质检.md'
    for group,provided in [('per-photo',a.personalized),('per-photo-confirmation',a.personalized_confirmation)]:
        if not provided:continue
        personal=provided.resolve()
        for name in ['DONE','summary.json','decision.json','paired-verification.json','artifact-validation.json','browser-validation.json','platform-final.json']:
            if not (personal/name).is_file():raise ValueError('Complete per-photo diagnostic first: '+name)
        if not json.loads((personal/'decision.json').read_text()).get('visually_reviewed'):raise ValueError('Per-photo native review required')
        for name in ['experiment.json','summary.json','paired-effect-summary.json','decision.json','paired-verification.json','baseline-preflight.json','artifact-validation.json','browser-validation.json','platform-final.json','independent-age.json','native-change-summary.json','native-review-observations.json','automatic-completion.json']:
            if (personal/name).is_file():addjson(group+'/'+name,json.loads((personal/name).read_text()))
        report=json.loads((personal/'evaluation/report.json').read_text())
        addjson(group+'/timing-and-frozen-verification.json',{k:report[k] for k in ['timings','total_seconds','cuda_peak_bytes','frozen_tensors_sha256','reused_baseline_exact']})
        for row in report['records']:
            if Path(row['name']).name!=row['name']:raise ValueError('Unsafe diagnostic record name')
            for route in report['routes'][1:]:addjson(group+'/logs/'+row['name']+'/'+route+'.json',json.loads((personal/'evaluation'/row['name']/(route+'_train.json')).read_text()))
        for path in (personal/'source').iterdir():
            if path.is_file():evidence[PREFIX+group+'/source/'+path.name]=path
        evidence[PREFIX+group+'/report.md']=ROOT/'integration/quality/20261125_逐图生成器适配诊断.md'
    evidence[PREFIX+'licenses/SAM-LICENSE.txt']=ROOT/'deployment/model-files/face-age/src/SAM/LICENSE'
    def addpanels(folder,prefix=''):
        credits=json.loads((folder/'public-panel-credits.json').read_text())
        if len(credits)!=2 or {r['name'] for r in credits}!={'ffhq_64376','ffhq_64919'}:raise ValueError('Expected two attributed public examples')
        for row in credits:
            if Path(row['file']).name!=row['file']:raise ValueError('Unsafe public panel filename')
            path=folder/row['file'];size=path.stat().st_size;digest=h.sha(path);name=prefix+row['file']
            items.append(dict(id=name,group='examples',format='file',filename=name,bytes=size,sha256=digest,files=[dict(path=PREFIX+name,bytes=size,sha256=digest)],assets=h.parts(path,name)))
            row['release_file']=name
        addjson(prefix+'public-panel-credits.json',credits)
    addpanels(a.public_panels)
    if a.personalized_panels:addpanels(a.personalized_panels,'per-photo-')
    items.append(h.archive('evidence',evidence))
    manifest=dict(schema_version=1,repository='prummn/face-age-transformation-course-project',release_tag=TAG,source_branch='dev/mobei-gan-portable-20261116',source_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),training_source_commit=json.loads((run/'experiment.json').read_text())['code_commit'],default_groups=['evidence','examples'],groups=dict(weights='Two research partial generator checkpoints; original SAM required, not enabled automatically. Per-photo patches are deliberately excluded.',evidence='Frozen code, training logs, independent metrics and source audit; no protected portrait or W+ payload.',examples='Attributed public comparisons with unscaled native crops.'),items=items)
    h.save(ROOT/'integration/models/direction-match-20261125.json',manifest);h.save(h.ASSETS/'manifest.json',manifest)
    (h.ASSETS/'SHA256SUMS').write_text(''.join(h.sha(path)+'  '+path.name+'\n' for path in sorted(h.ASSETS.iterdir()) if path.name!='SHA256SUMS'))
    print('DIRECTION_RELEASE_READY',stage,flush=True)


if __name__=='__main__':main()
