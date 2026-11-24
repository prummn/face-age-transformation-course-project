"""Merge the frozen prior training pool with a verified high-age increment."""
import argparse,json
from collections import Counter
from pathlib import Path
from probe_inversion import sha256,write_json
from train_gan_reconstruction import rows


def main():
 p=argparse.ArgumentParser();p.add_argument('prior',type=Path);p.add_argument('prepared',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 prior=a.prior.resolve(); prepared=a.prepared.resolve(); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
 old_train=rows(prior/'manifests/train.jsonl'); old_eval=rows(prior/'manifests/eval.jsonl'); old_ext=rows(prior/'manifests/external.jsonl')
 fresh=rows(prepared/'train.pre-age.jsonl'); fresh_eval=rows(prepared/'eval.pre-age.jsonl')
 old_hash={r['sha256'] for r in old_train}; old_eval_hash={r['sha256'] for r in old_eval+old_ext}
 additions=[r for r in fresh if r['sha256'] not in old_hash]
 if not additions: raise ValueError('No unique increment')
 if {r['sha256'] for r in additions}&old_eval_hash: raise ValueError('Increment overlaps prior eval/external pixels')
 new_eval=[r for r in fresh_eval if r.get('cohort')=='new-hd-holdout']
 if not new_eval: raise ValueError('Missing new holdout')
 train=old_train+additions; evaluation=old_eval+new_eval
 if {r['sha256'] for r in train}&{r['sha256'] for r in evaluation}: raise ValueError('Train/eval pixel overlap')
 if {r['identity_group'] for r in train}&{r['identity_group'] for r in evaluation}: raise ValueError('Train/eval identity group overlap')
 def write(name,items):
  p=out/name;p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in items));return p
 paths=[write('train.pre-age.jsonl',train),write('eval.pre-age.jsonl',evaluation),write('external.jsonl',old_ext)]
 audit=dict(prior=str(prior),source=str(prepared),train=len(train),prior_train=len(old_train),new_unique=len(additions),evaluation=len(evaluation),new_holdout=len(new_eval),
  added_by_age=dict(Counter(str(r['age_interval']) for r in additions)),added_by_collection=dict(Counter(r.get('collection','original') for r in additions)),
  train_groups_overlapping_prior=sum(r['identity_group'] in {x['identity_group'] for x in old_train} for r in additions),
  files={p.name:sha256(p) for p in paths},notes=['The increment has no pixel or identity overlap with prior evaluation/external sets.','Some new training groups overlap prior training groups under conservative SFace grouping and remain recorded for ablation.'])
 write_json(out/'audit.json',audit);(out/'DONE').write_text('Merged high-age increment with prior training pool\n');print(json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
