"""Build explicit full-native-data and result bundles; no private evaluation images."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]
TAG='research-mobei-native-full-2026-11-24'
PREFIX='deployment/shared-research/native-full/'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['data','results']);p.add_argument('run',type=Path);p.add_argument('stage',type=Path);p.add_argument('--public-panels',type=Path);p.add_argument('--diagnostic-run',type=Path,action='append',default=[]);p.add_argument('--diagnostic-panels',type=Path);a=p.parse_args();run=a.run.resolve();stage=a.stage.resolve()
    spec=importlib.util.spec_from_file_location('bundler',Path(__file__).with_name('build_research_release.py'));h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    h.TAG=TAG;h.STAGE=stage;h.ASSETS=stage/'assets';h.PAYLOAD=stage/'payload';h.ASSETS.mkdir(parents=True,exist_ok=True);h.PAYLOAD.mkdir(exist_ok=True)
    def generated(relative,obj):return h.generated(PREFIX+relative,h.scrub(obj))
    if a.action=='data':
        rows=[json.loads(s) for s in (run/'data/train.jsonl').read_text().splitlines()];files={};portable=[]
        for row in rows:
            r=dict(row)
            if not r.get('training_authorized'):raise ValueError('Training permission field required')
            if r.get('published_demographic_label') not in (None,'Asian'):raise ValueError('Unexpected annotation')
            for field,key,sub in [('reference_path','reference_sha256','native1024'),('image_path','sha256','input256'),('reference512_path','reference512_sha256','reference512')]:
                source=Path(r[field]);dest=PREFIX+'data/'+sub+'/'+r['name']+'.png'
                if h.sha(source)!=r[key]:raise ValueError('Frozen image digest differs')
                files[dest]=source;r[field]=dest
            portable.append(r)
        rel=PREFIX+'data/train.portable.json';files[rel]=generated('data/train.portable.json',portable)
        rel=PREFIX+'data/data-card.json';files[rel]=generated('data/data-card.json',dict(images=len(rows),native_face_min=512,source='Official FFHQ native1024, file/pixel MD5 verified; individual source attribution in portable rows.',
            training_manifest_sha256=h.sha(run/'data/train.jsonl'),manifest_path_change_only=True,new_source_ages='Null means infer once with original frozen FairFace; no invented chronological age.',
            rule='Extended conservative source/hash/similarity grouping excludes protected components; no evaluation/private/APPA image payload.',
            use='Noncommercial course research under FFHQ BY-NC-SA4.0 and individual photo licences; no biometric identity claims or ethnicity self-report claims.'))
        for name in ['ffhq_readme.txt','ffhq_aging_readme.txt']:files[PREFIX+'licenses/'+name]=ROOT/'deployment/data-research/20261113'/name
        files[PREFIX+'data/restore-note.txt']=h.PAYLOAD/'restore-note.txt';files[PREFIX+'data/restore-note.txt'].write_text('Paths in train.portable.json are repository-relative. Prefix each image/reference path with the current repository root. Verify per-file SHA256 before training. This bundle has training images only; it does not contain the original protected evaluation photos or W+ codes.\n')
        item=h.archive('data',files);h.save(stage/'data-item.json',item);print('NATIVE_DATA_BUNDLE_READY',len(rows),item['bytes'],flush=True);return
    for name in ['DONE','summary.json','decision.json','browser-validation.json','paired-verification.json','artifact-validation.json']:
        if not (run/name).is_file():raise ValueError('Finish study first: '+name)
    decision=json.loads((run/'decision.json').read_text())
    if not decision.get('visually_reviewed'):raise ValueError('Native review required')
    items=[json.loads((stage/'data-item.json').read_text())];evidence={}
    def addjson(relative,path):evidence[PREFIX+relative]=generated(relative,json.loads(path.read_text()))
    def addweight(source,name,relative,group):
        digest=h.sha(source);dest=h.ASSETS/name
        if dest.exists() and h.sha(dest)!=digest:raise ValueError('Preserving different staged file')
        if not dest.exists():shutil.copy2(source,dest)
        size=dest.stat().st_size
        items.append(dict(id=name,group=group,format='file',filename=name,bytes=size,sha256=digest,
            files=[dict(path=PREFIX+relative,bytes=size,sha256=digest)],assets=[dict(name=name,tag=TAG,bytes=size,sha256=digest)]))
    for arm in ['native_control','native_gan']:
        source=run/arm/'generator_patch.pt';done=json.loads((run/arm/'completed.json').read_text());digest=h.sha(source)
        if digest!=done['patch_sha256']:raise ValueError('Completed checkpoint changed')
        addweight(source,arm+'.pt',arm+'.pt','weights')
        for step in (1000,2000,3000):
            addweight(run/arm/f'generator_step{step}.pt',f'{arm}_step{step}.pt',f'{arm}/generator_step{step}.pt','checkpoints')
        if arm=='native_gan':addweight(run/arm/'discriminator.pt','native_gan_discriminator.pt','native_gan/discriminator.pt','checkpoints')
        for name in ['plan.json','completed.json']:addjson(arm+'/'+name,run/arm/name)
        evidence[PREFIX+arm+'/train.json']=generated(arm+'/train.json',[json.loads(s) for s in (run/arm/'train.jsonl').read_text().splitlines()])
    for name in ['experiment.json','common-audit.json','supplemental-data-balance.json','supplemental-publisher-labels.json','summary.json','decision.json','browser-validation.json','paired-verification.json','artifact-validation.json','independent-age.json','native-change-summary.json','platform-final.json','automatic-completion.json','evaluation-merge.json']:
        if (run/name).is_file():addjson(name,run/name)
    addjson('train-codes/index.json',run/'train-codes/index.json')
    if (stage/'data-local-restore-validation.json').is_file():
        addjson('data-local-restore-validation.json',stage/'data-local-restore-validation.json')
    for path in (run/'source').iterdir():
        if path.is_file():evidence[PREFIX+'source/'+path.name]=path
    for name in ['finish_native_age.py','summarize_highres_texture.py','check_independent_age.py','measure_texture_changes.py','build_texture_review.py','verify_texture_review_browser.py','expand_native_age_data.py','build_public_texture_panels.py']:
        evidence[PREFIX+'postprocess/'+name]=ROOT/'integration/quality'/name
    evidence[PREFIX+'report.md']=ROOT/'integration/quality/20261124_全量高清数据与RGB纹理训练.md'
    diagnostic_index=[]
    for diagnostic in a.diagnostic_run:
        diagnostic=diagnostic.resolve()
        if not diagnostic.is_relative_to(ROOT/'deployment/training-runs'):raise ValueError('Diagnostic outside research runs')
        d=json.loads((diagnostic/'decision.json').read_text());completion=json.loads((diagnostic/'automatic-completion.json').read_text())
        if not (diagnostic/'DONE').is_file() or not d.get('visually_reviewed') or completion['native_review_pending']:raise ValueError('Finish diagnostic review before archiving')
        for name in ['experiment.json','selection.json','summary.json','decision.json','browser-validation.json','artifact-validation.json','independent-age.json','native-change-summary.json','platform-final.json','automatic-completion.json','evaluation/report.json']:
            addjson('diagnostics/'+diagnostic.name+'/'+name,diagnostic/name)
        for path in (diagnostic/'source').iterdir():
            if path.is_file():evidence[PREFIX+'diagnostics/'+diagnostic.name+'/source/'+path.name]=path
        diagnostic_index.append(dict(run=diagnostic.name,cases=json.loads((diagnostic/'experiment.json').read_text())['cases'],promotion=d['promotion'],no_image_or_latent_payload=True))
    evidence[PREFIX+'diagnostics/index.json']=generated('diagnostics/index.json',diagnostic_index)
    if diagnostic_index:
        evidence[PREFIX+'diagnostics/report.md']=ROOT/'integration/quality/20261125_年龄差值与预训练判别器诊断.md'
        evidence[PREFIX+'diagnostics/discriminator-provenance.json']=generated('diagnostics/discriminator-provenance.json',json.loads((ROOT/'deployment/gan-research/stylegan2ada-ffhq-checkpoint.json').read_text()))
        evidence[PREFIX+'postprocess/finish_small_quality.py']=ROOT/'integration/quality/finish_small_quality.py'
    for name in ['SAM-LICENSE.txt']:
        evidence[PREFIX+'licenses/'+name]=ROOT/'deployment/model-files/face-age/src/SAM/LICENSE'
    for path in (ROOT/'deployment/model-files/face-age/src/SAM/licenses').iterdir():
        if path.is_file():evidence[PREFIX+'licenses/'+path.name]=path
    if not a.public_panels:raise ValueError('Source-attributed public panels required')
    credits=json.loads((a.public_panels/'public-panel-credits.json').read_text())
    for row in credits:
        if row['name'] not in ('ffhq_64376','ffhq_64919') or Path(row['file']).name!=row['file']:raise ValueError('Unexpected public example')
        source=a.public_panels/row['file'];digest=h.sha(source);size=source.stat().st_size;items.append(dict(id=row['file'],group='examples',format='file',filename=row['file'],bytes=size,sha256=digest,files=[dict(path=PREFIX+row['file'],bytes=size,sha256=digest)],assets=h.parts(source,row['file'])))
    evidence[PREFIX+'public-panel-credits.json']=generated('public-panel-credits.json',credits)
    if a.diagnostic_panels:
        diagnostic_credits=json.loads((a.diagnostic_panels/'public-panel-credits.json').read_text())
        for row in diagnostic_credits:
            if row['name'] not in ('ffhq_64376','ffhq_64919') or Path(row['file']).name!=row['file']:raise ValueError('Unexpected diagnostic example')
            source=a.diagnostic_panels/row['file'];name='original-delta-'+row['file'];row['file']=name;digest=h.sha(source);size=source.stat().st_size
            items.append(dict(id=name,group='examples',format='file',filename=name,bytes=size,sha256=digest,files=[dict(path=PREFIX+name,bytes=size,sha256=digest)],assets=h.parts(source,name)))
        evidence[PREFIX+'diagnostics/public-panel-credits.json']=generated('diagnostics/public-panel-credits.json',diagnostic_credits)
    items.append(h.archive('evidence',evidence))
    original=json.loads((ROOT/'integration/models/research-20261117.json').read_text())
    items.extend(item for item in original['items'] if item['group']=='baseline')
    manifest=dict(schema_version=1,repository='prummn/face-age-transformation-course-project',release_tag=TAG,source_branch='dev/mobei-gan-portable-20261116',source_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        training_source_commit=json.loads((run/'experiment.json').read_text())['code_commit'],
        default_groups=['evidence','examples'],
        portable_training_manifest=dict(source=PREFIX+'data/train.portable.json',destination=PREFIX+'data/train.jsonl',format='json-list'),
        groups=dict(data='All712native training images and derived inputs/references with portable paths, hashes and source attribution; no protected evaluation images.',weights='Two final4000step partial StyleGAN2 generator patches; original SAM model required; not automatically activated.',checkpoints='Both arms1000/2000/3000step generator snapshots and trained local discriminator; no optimizer state, not exact restart packages.',baseline='Original SAM, ft_l1, FairFace, YuNet, identity/LPIPS and SFace dependencies; reuse existing releases without duplicate uploads.',evidence='Frozen training/diagnostic code, paired sampling logs, full-data audit, native review decisions, independent metrics and browser validation; no portrait or latent payload.',examples='Source-attributed original/raw75/native-crop comparisons of full native training and optional original SAM displacement diagnostic.'),items=items)
    h.save(ROOT/'integration/models/native-full-20261124.json',manifest);h.save(h.ASSETS/'manifest.json',manifest);(h.ASSETS/'SHA256SUMS').write_text(''.join(h.sha(path)+'  '+path.name+'\n' for path in sorted(h.ASSETS.iterdir()) if path.name!='SHA256SUMS'));print('NATIVE_RELEASE_READY',TAG,len(items),flush=True)


if __name__=='__main__':main()
