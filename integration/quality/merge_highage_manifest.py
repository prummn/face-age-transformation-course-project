"""Export a combined manifest only after one common grouping audit.

Never restore previously rejected old training rows by comparing identity-group
strings from different versions: their grouping universes can differ.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from probe_inversion import sha256,write_json
from train_gan_reconstruction import rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('prior',type=Path);p.add_argument('prepared',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--common-audit',required=True,type=Path)
    a=p.parse_args();prior=a.prior.resolve();prepared=a.prepared.resolve();audit=a.common_audit.resolve();out=a.output.resolve()
    if not (audit/'DONE').is_file():raise ValueError('Common grouping audit must complete')
    report=json.loads((audit/'audit.json').read_text())
    for name,digest in report['files'].items():
        if sha256(audit/name)!=digest:raise ValueError('Audited manifest changed: '+name)
    train,evaluation,external=(rows(audit/(name+'.jsonl')) for name in ('train','eval','external'))
    protected=evaluation+external
    if {r['identity_group'] for r in train}&{r['identity_group'] for r in protected}:raise ValueError('Common train/protected groups overlap')
    if {r['sha256'] for r in train}&{r['sha256'] for r in protected}:raise ValueError('Train/protected image overlap')
    old_train=rows(prior/'manifests/train.jsonl');old_hash={r['sha256'] for r in old_train}
    additions=[r for r in train if r['sha256'] not in old_hash]
    out.mkdir(parents=True,exist_ok=False)
    files={}
    for name,records in [('train.pre-age.jsonl',train),('eval.pre-age.jsonl',evaluation),('external.jsonl',external)]:
        path=out/name;path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records));files[name]=sha256(path)
    if (prepared/'longitudinal.json').is_file():
        shutil.copy2(prepared/'longitudinal.json',out/'longitudinal.json');files['longitudinal.json']=sha256(out/'longitudinal.json')
    result=dict(prior=str(prior),source=str(prepared),common_audit=str(audit),train=len(train),prior_train=len(old_train),
        prior_retained=sum(r['sha256'] in old_hash for r in train),new_unique=len(additions),evaluation=len(evaluation),
        new_holdout=sum(r['cohort']=='new-hd-holdout' for r in evaluation),
        added_by_age=dict(Counter(str(r['age_interval']) for r in additions)),external_alignment_failures=[],files=files,
        notes=['All rows use the same conservative grouping universe. Excluded old training rows stay excluded.',
               'Account/similarity grouping is conservative; it does not establish verified subject-disjoint splits.'])
    write_json(out/'audit.json',result);(out/'DONE').write_text('Exported common-audited manifests\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
