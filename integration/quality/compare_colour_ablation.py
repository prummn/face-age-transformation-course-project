#!/usr/bin/env python3
"""CPU-only paired analysis of the authorized 100-step control/colour experiment."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def stats(values):
    x=np.asarray(list(values),dtype=float)
    return {'n':len(x),'mean':float(x.mean()),'median':float(np.median(x)),'p90':float(np.quantile(x,.9))}


def row_index(blob):
    return {(row['key'],row['tag']):row for row in blob['metrics']}


def main(args):
    base=Path(args.experiments).resolve()
    original=Path(json.loads((base/'control'/'plan.json').read_text())['base']).resolve()
    output=Path(args.output).resolve()
    if output==original or original in output.parents:
        raise ValueError('Comparison output must be outside the original read-only face-age project.')
    output.mkdir(parents=True,exist_ok=True)
    blobs={}
    logs={}
    for arm in ['control','colour_only']:
        for step in [0,100]:
            blobs[(arm,step)]=json.loads((base/arm/f'validation_step{step}/metrics.json').read_text())
        logs[arm]=[json.loads(line) for line in (base/arm/'train_log.jsonl').read_text().splitlines() if line]
        if len(logs[arm])!=100 or logs[arm][-1]['step']!=100:
            raise RuntimeError(f'{arm} did not complete exactly 100 steps')
        if not (base/arm/'encoder_step100.pt').is_file():
            raise RuntimeError(f'{arm} checkpoint missing')
    indexes={name:row_index(blob) for name,blob in blobs.items()}
    keys=list(blobs[('control',0)]['keys'])
    if any(blob['keys']!=keys for blob in blobs.values()):
        raise RuntimeError('Validation key order differs between arms or steps')
    reference=set(indexes[('control',0)])
    if len(keys)!=24 or len(reference)!=120 or any(set(index)!=reference for index in indexes.values()):
        raise RuntimeError('Expected exactly the same 24 x 5 paired outputs in every evaluation')
    numeric=['cheek_delta_L','cheek_delta_a','cheek_delta_b','cheek_chroma_shift','diagnostic_id_cos_shared_training_model','diagnostic_age_shared_training_model']
    baseline_difference={metric:max(abs(indexes[('control',0)][key][metric]-indexes[('colour_only',0)][key][metric]) for key in reference) for metric in numeric}
    report={'scope':'Two authorized 100-step experiments only; 24 previously used frozen regression inputs x five outputs. No checkpoint deployed. Age/identity models are shared with training and are secondary diagnostics, not independent quality claims.','keys':keys,'initial_baseline_max_absolute_metric_difference':baseline_difference,'per_arm':{},'paired_colour_minus_control':{},'training':{}}
    for (arm,step),index in indexes.items():
        summary={}
        for tag in ['same_age','a20','a40','a60','a75']:
            rows=[index[(key,tag)] for key in keys]
            summary[tag]={metric:stats(row[metric] for row in rows) for metric in numeric}
            if tag=='same_age':
                summary[tag]['same_age_mae_rgb']=stats(row['same_age_mae_rgb'] for row in rows)
            else:
                target=float(tag[1:])
                summary[tag]['shared_age_target_mae']=stats(abs(row['diagnostic_age_shared_training_model']-target) for row in rows)
        report['per_arm'][f'{arm}_step{step}']=summary
    before=indexes[('control',100)];after=indexes[('colour_only',100)]
    for tag in ['same_age','a20','a40','a60','a75']:
        deltas=[after[(key,tag)]['cheek_chroma_shift']-before[(key,tag)]['cheek_chroma_shift'] for key in keys]
        report['paired_colour_minus_control'][tag]={'chroma_delta':stats(deltas),'n_improved':sum(value<0 for value in deltas),'n_worse':sum(value>0 for value in deltas)}
    # Paired bootstrap resamples input photos, not the four correlated ages.
    # No identity labels exist; different photos may contain the same person.
    per_input=np.array([np.mean([after[(key,tag)]['cheek_chroma_shift']-before[(key,tag)]['cheek_chroma_shift'] for tag in ['a20','a40','a60','a75']]) for key in keys])
    rng=np.random.default_rng(20261110)
    boots=np.array([rng.choice(per_input,len(per_input),replace=True).mean() for _ in range(4000)])
    control_mean=np.mean([before[(key,tag)]['cheek_chroma_shift'] for key in keys for tag in ['a20','a40','a60','a75']])
    colour_mean=np.mean([after[(key,tag)]['cheek_chroma_shift'] for key in keys for tag in ['a20','a40','a60','a75']])
    report['paired_aged_aggregate']={'control_mean_chroma':float(control_mean),'colour_mean_chroma':float(colour_mean),'relative_reduction_percent':float(100*(control_mean-colour_mean)/control_mean),'paired_input_mean_difference':float(per_input.mean()),'paired_bootstrap_ci95':np.quantile(boots,[.025,.975]).tolist(),'n_inputs_improved':int((per_input<0).sum()),'n_inputs':len(keys),'caution':'Exploratory photo-level interval on 24 previously used regression inputs; identity independence is unverified and different photos may show the same person. Not an independent population guarantee.'}
    for arm,log in logs.items():
        report['training'][arm]={'steps':len(log),'skipped':sum(row['skipped'] for row in log),'first10_loss':stats(row['loss'] for row in log[:10]),'last10_loss':stats(row['loss'] for row in log[-10:]),'first10_chroma_raw':stats(row['chroma_raw'] for row in log[:10]),'last10_chroma_raw':stats(row['chroma_raw'] for row in log[-10:]),'step_seconds':stats(row['seconds'] for row in log),'checkpoint_sha256':hashlib.sha256((base/arm/'encoder_step100.pt').read_bytes()).hexdigest()}
    try:
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    except OSError:
        font=ImageFont.load_default()
    titles=['Input','Control same age','Colour same age','Control age 75','Colour age 75']
    for page in range(6):
        sheet=Image.new('RGB',(1280,4*286+28),'#f4f4f4');draw=ImageDraw.Draw(sheet)
        for column,title in enumerate(titles):draw.text((column*256+8,5),title,fill='#111111',font=font)
        for position,key in enumerate(keys[page*4:(page+1)*4]):
            control=Image.open(base/'control'/'validation_step100'/f'{key}_input_sameage_20_40_60_75.png').convert('RGB')
            colour=Image.open(base/'colour_only'/'validation_step100'/f'{key}_input_sameage_20_40_60_75.png').convert('RGB')
            tiles=[control.crop((0,0,256,256)),control.crop((256,0,512,256)),colour.crop((256,0,512,256)),control.crop((1280,0,1536,256)),colour.crop((1280,0,1536,256))]
            y=28+position*286
            for column,tile in enumerate(tiles):sheet.paste(tile,(column*256,y))
            left=before[(key,'a75')];right=after[(key,'a75')]
            draw.text((8,y+259),f"{key}  age75 chroma: {left['cheek_chroma_shift']:.2f} -> {right['cheek_chroma_shift']:.2f}",fill='#111111',font=font)
        sheet.save(output/f'paired_all24_page{page+1}.jpg',quality=94)
    (output/'ablation_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'baseline_metric_difference':baseline_difference,'aged_aggregate':report['paired_aged_aggregate'],'per_target':report['paired_colour_minus_control'],'training_steps':{arm:data['steps'] for arm,data in report['training'].items()}},ensure_ascii=False,indent=2))
    print('COMPARISON_COMPLETE '+str(output))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments',required=True)
    parser.add_argument('--output',required=True)
    main(parser.parse_args())
