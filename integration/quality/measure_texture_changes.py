"""Describe native output changes; these pixel statistics are not realism scores."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import cv2
import numpy as np
from PIL import Image
from probe_inversion import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve();folder=run/'evaluation'
    report=json.loads((folder/'report.json').read_text());base={(r['name'],r['condition']):r for r in report['outputs'] if r['route']=='wplus_base'}
    raw={};changes=[];reconstruction=[];cv2.setNumThreads(2)
    metadata={r['name']:r for r in report['records']}
    def image(path):
        with Image.open(path) as im:im.load();return np.asarray(im.convert('RGB')).astype(np.float32)
    anchors={(x['name'],x['route']):x for x in report['outputs'] if x['condition']=='same'}
    def canonical(row):
        if row['condition']!='same' and row['target']==metadata[row['name']]['source_age']:
            return dict(anchors[row['name'],row['route']],condition=row['condition'],target=row['target'])
        return row
    for row in report['outputs']:
        row=canonical(row)
        if row['condition'] not in ('same','60','75'):continue
        key=(row['name'],row['condition'])
        if key not in raw:raw[key]=image(folder/canonical(base[key])['native'])
        output=image(folder/row['native']);reference=raw[key]
        changes.append(dict(name=row['name'],cohort=row['cohort'],route=row['route'],condition=row['condition'],
            difference_rgb8=float(np.abs(output-reference).mean()),face_difference_rgb8=float(np.abs(output-reference)[300:800,220:804].mean()),
            max_difference_rgb8=float(np.abs(output-reference).max())))
        record=metadata[row['name']]
        if row['condition']=='same' and record.get('source_face_short_edge',record.get('native_face_short_edge',0))>=512:
            real=image(folder/row['name']/'reference.png')
            if real.shape!=(1024,1024,3):continue
            roi=(slice(300,800),slice(220,804))
            gray=cv2.cvtColor(output,cv2.COLOR_RGB2GRAY);truth=cv2.cvtColor(real,cv2.COLOR_RGB2GRAY)
            hp=gray-cv2.GaussianBlur(gray,(0,0),1);hp_truth=truth-cv2.GaussianBlur(truth,(0,0),1)
            reconstruction.append(dict(name=row['name'],route=row['route'],face_l1_rgb8=float(np.abs(output-real)[roi].mean()),
                local_highpass_l1=float(np.abs(hp-hp_truth)[roi].mean())))
    groups=defaultdict(list);same=defaultdict(list)
    for row in changes:groups[row['route']].append(row)
    for row in reconstruction:same[row['route']].append(row)
    result=dict(changes=changes,same_reconstruction=reconstruction,
        average_native_change={route:dict(cases=len({r['name'] for r in rows}),face_rgb8=statistics.mean(r['face_difference_rgb8'] for r in rows),
            full_rgb8=statistics.mean(r['difference_rgb8'] for r in rows),max_rgb8=max(r['max_difference_rgb8'] for r in rows)) for route,rows in groups.items()},
        same_hd_reconstruction={route:dict(cases=len(rows),face_l1=statistics.mean(r['face_l1_rgb8'] for r in rows),
            highpass_l1=statistics.mean(r['local_highpass_l1'] for r in rows)) for route,rows in same.items()},
        caveat='Descriptive output differences and same-age paired pixel errors. No cross-age texture truth, realism classifier or automatic visual improvement decision.')
    write_json(run/'native-change-summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('changes','same_reconstruction')},indent=2))

if __name__=='__main__':main()
