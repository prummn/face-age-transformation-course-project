#!/usr/bin/env python3
"""Same-source age transformation color audit; no ethnicity inferred from pictures.

select/historical/score are CPU-only. generate refuses all physical devices except
ASCEND_RT_VISIBLE_DEVICES=1; logical npu:0 then maps to authorized physical NPU1.
Online physical NPU0 and reserved NPU2/3 are never touched.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, shutil, sys, time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

AGES = [20, 40, 60, 80]
MODELS = {'base':None, 'ft_l1_1000':'out/ft_l1/encoder_step1000.pt', 'ft_l5_250':'out/ft_l5/encoder_step250.pt'}
BASE_DEFAULT = '/data/home/1120231322/scow/face-age'

def sha(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=lambda value:value.item() if isinstance(value,np.generic) else str(value))+'\n')

def rgb(path):
    with Image.open(path) as image:return np.array(image.convert('RGB'))

def lab(image):
    return cv2.cvtColor(cv2.resize(image,(256,256),interpolation=cv2.INTER_AREA).astype(np.float32)/255,cv2.COLOR_RGB2LAB)

def safe_mask(image,yunet):
    """Conservative bilateral cheek patches from source-only landmarks, no skin classifier.

    Excludes eyes/mouth/edge saturation. Occlusion/hair may still invalidate a
    patch: source/mask overlays require visual review, and metrics remain proxies.
    Identical reference mask is used for every model/target of this input.
    """
    image=cv2.resize(image,(256,256),interpolation=cv2.INTER_AREA)
    detector=cv2.FaceDetectorYN.create(str(yunet),'',(256,256),.5,.3,20)
    _,faces=detector.detect(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    if faces is None:raise ValueError('Source landmark detector failed; no metric fallback mask')
    face=max(faces,key=lambda f:f[2]*f[3]);points=face[4:14].reshape(5,2)
    eyes=sorted(points[:2],key=lambda p:p[0]);mouth=sorted(points[3:],key=lambda p:p[0])
    eye_mid=(eyes[0]+eyes[1])*.5;mouth_mid=(mouth[0]+mouth[1])*.5
    eye_dist=float(np.linalg.norm(eyes[1]-eyes[0]));vertical=mouth_mid-eye_mid
    if eye_dist<25 or np.linalg.norm(vertical)<20:raise ValueError('Source landmark geometry too small')
    yy,xx=np.mgrid[:256,:256];regions=[];mask=np.zeros((256,256),bool)
    for side in (0,1):
        center=eyes[side]+vertical*.60
        center[0]+=(-1 if side==0 else 1)*eye_dist*.03
        rx=max(4.,eye_dist*.14);ry=max(4.,np.linalg.norm(vertical)*.16)
        patch=((xx-center[0])/rx)**2+((yy-center[1])/ry)**2<1
        regions.append({'center':center.tolist(),'radius':[rx,ry]});mask|=patch
    reference=lab(image);valid=(reference[:,:,0]>5)&(reference[:,:,0]<95)
    # Exclude only reference near-black/clipped highlights. Never select output pixels
    # by how close they are to the source: that would bias the color comparison.
    mask&=valid
    if mask.sum()<100:raise ValueError('Insufficient valid source cheek pixels')
    return mask,{'regions':regions,'pixels':int(mask.sum()),'method':'source_yunet_bilateral_cheeks_v1','review':'pending_visual_mask_review'}

def select(base,out,count):
    import pyarrow.parquet as pq
    rows=pq.read_table(base/'data/index.parquet').to_pylist()
    pool=[r for r in rows if r['split']=='eval' and r['p_asian']>=.75 and r['p_minor']<.15 and (base/'data/aligned/eval'/f"{r['key']}.png").exists()]
    if count!=10:raise ValueError('This frozen audit requires exactly 10 inputs')
    centers=[25,27,30,35,40,45,50,55,60,70]
    selected=[];used=set();rejected=[];yunet=base/'models/yunet/face_detection_yunet_2023mar.onnx'
    for center in centers:
        for candidate in sorted((r for r in pool if r['key'] not in used),key=lambda r:(abs(r['age_exp']-center),r['key'])):
            key=candidate['key'];source=base/'data/aligned/eval'/f'{key}.png';image=rgb(source)
            try:mask,mask_info=safe_mask(image,yunet)
            except ValueError as error:rejected.append({'key':key,'reason':str(error)});used.add(key);continue
            used.add(key);(out/'inputs').mkdir(parents=True,exist_ok=True);(out/'masks').mkdir(exist_ok=True)
            shutil.copy2(source,out/'inputs'/f'{key}.png');Image.fromarray(mask.astype(np.uint8)*255).save(out/'masks'/f'{key}.png')
            overlay=image.copy();overlay[mask]=(overlay[mask]*.45+np.array([0,255,120])*.55).astype(np.uint8)
            Image.fromarray(overlay).save(out/'masks'/f'{key}_overlay.png')
            selected.append({'key':key,'source':str(source),'input_sha256':sha(source),'selection_age_center':center,
                'metadata':{k:candidate[k] for k in ['split','age_exp','p_asian','p_minor','p_ea','p_sea','license']},'mask':mask_info});break
        else:raise RuntimeError('Not enough mask-valid candidates')
    manifest={'version':1,'count':len(selected),'targets':AGES,'selection':'nearest unused source metadata age_exp to predeclared centers, key tie break; not selected on model output',
      'label_provenance':'Existing caption Asian-cue candidate pool followed by FairFace p_asian>=0.75 and p_minor<0.15. Model-derived dataset selection metadata, NOT verified ethnicity or skin-color truth. No labels inferred from photos during this audit.',
      'age_provenance':'age_exp is FairFace source estimate, NOT chronological age; age-stratified convenience sample is not representative of all 431 inputs.',
      'metrics_limit':'Source-derived small cheek masks are candidate skin regions; human overlay review required. No semantic skin ground truth; shape/lighting changes confound color differences. Correcting color to match this metric is not independent evidence of image quality.',
      'base':str(base),'samples':selected,'rejected_mask_candidates':rejected,'script_sha256':sha(__file__)}
    dump(out/'manifest.json',manifest)
    make_input_sheet(out,manifest)
    print(json.dumps({'manifest':str(out/'manifest.json'),'keys':[r['key'] for r in selected],'source_age_est':[round(r['metadata']['age_exp'],2) for r in selected]},ensure_ascii=False),flush=True)

def make_input_sheet(out,manifest):
    tile=256;sheet=Image.new('RGB',(tile*5,600),'white');draw=ImageDraw.Draw(sheet)
    for index,item in enumerate(manifest['samples']):
        x=(index%5)*tile;y=(index//5)*300
        sheet.paste(Image.open(out/'inputs'/f"{item['key']}.png").convert('RGB'),(x,y))
        draw.text((x+5,y+260),f"{item['key']} source est {item['metadata']['age_exp']:.1f}",fill='black')
    sheet.save(out/'inputs_contact.jpg',quality=95)
    sheet2=Image.new('RGB',(tile*5,600),'white');draw=ImageDraw.Draw(sheet2)
    for index,item in enumerate(manifest['samples']):
        x=(index%5)*tile;y=(index//5)*300
        sheet2.paste(Image.open(out/'masks'/f"{item['key']}_overlay.png").convert('RGB'),(x,y));draw.text((x+5,y+260),item['key']+' mask review',(0,0,0))
    sheet2.save(out/'masks_contact.jpg',quality=95)

def compute(reference,generated,mask):
    left,right=lab(reference)[mask],lab(generated)[mask]
    a=np.median(left,axis=0);b=np.median(right,axis=0);delta=b-a
    return {'source_median_Lab':a.tolist(),'output_median_Lab':b.tolist(),'delta_L':float(delta[0]),'abs_delta_L':float(abs(delta[0])),
      'delta_a':float(delta[1]),'delta_b':float(delta[2]),'chroma_shift_ab':float(np.linalg.norm(delta[1:])),
      'median_color_distance_Lab76':float(np.linalg.norm(delta)),
      'pixelwise_median_Lab76_geometry_confounded':float(np.median(np.linalg.norm(right-left,axis=1))),
      'output_clipped_fraction_in_mask':float(np.mean((right[:,0]<5)|(right[:,0]>95))),'pixels':int(mask.sum())}

def aggregate(records):
    result={}
    for model in sorted(set(r['model'] for r in records)):
        values=[r for r in records if r['model']==model];result[model]={'n_outputs':len(values),'n_sources':len(set(r['key'] for r in values))}
        for metric in ['delta_L','abs_delta_L','chroma_shift_ab','median_color_distance_Lab76']:
            nums=[r['color'][metric] for r in values];result[model][metric]={'mean':float(np.mean(nums)),'median':float(np.median(nums)),'max':float(np.max(nums))}
        for metric in ['identity_cosine','estimated_age_error']:
            nums=[r[metric] for r in values if r.get(metric) is not None]
            if nums:result[model][metric]={'mean':float(np.mean(nums)),'median':float(np.median(nums))}
    paired=[]
    models=list(result)
    if 'base' in models:
        index={(r['model'],r['key'],r['target_age']):r for r in records}
        for model in models:
            if model=='base':continue
            pairs=[(r,index.get((model,r['key'],r['target_age']))) for r in records if r['model']=='base'];pairs=[p for p in pairs if p[1]]
            paired.append({'comparison':model+' minus base','n_pairs':len(pairs),**{metric:{'mean_difference':float(np.mean([b['color'][metric]-a['color'][metric] for a,b in pairs])),
              'improved_count':sum(b['color'][metric]<a['color'][metric] for a,b in pairs)} for metric in ['abs_delta_L','chroma_shift_ab','median_color_distance_Lab76']}})
    return {'models':result,'paired_vs_base':paired,'interpretation':'Descriptive small-sample color proxies only; no ethnicity, actual age, or skin-ground-truth assertion. Review images, masks, identity and aging effectiveness separately.'}

def contacts(out,manifest,models,folder='raw'):
    target=out/'contacts';target.mkdir(exist_ok=True)
    for item in manifest['samples']:
        key=item['key'];tile=256;rowh=294;sheet=Image.new('RGB',(tile*5,rowh*len(models)),'white');draw=ImageDraw.Draw(sheet)
        for row,model in enumerate(models):
            y=row*rowh;sheet.paste(Image.open(out/'inputs'/f'{key}.png').convert('RGB'),(0,y+28));draw.text((5,y+8),model+' | input',(0,0,0))
            for col,age in enumerate(AGES,1):
                p=out/folder/model/f'{key}_a{age}.png'
                if p.exists():sheet.paste(Image.open(p).convert('RGB').resize((tile,tile),Image.Resampling.LANCZOS),(tile*col,y+28))
                draw.text((tile*col+5,y+8),f'target {age}',(0,0,0))
        sheet.save(target/f'{key}_{folder}.jpg',quality=95)

def generate(base,out,identity):
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!='1':raise RuntimeError('Requires explicitly authorized physical NPU1: ASCEND_RT_VISIBLE_DEVICES=1. Refusing default/other devices.')
    import torch,torch_npu
    sys.path.insert(0,str(base/'src'))
    from sam_npu import load_sam
    from sam_losses import IDLoss,AgeEst
    manifest=json.loads((out/'manifest.json').read_text());dev='npu:0';torch.npu.set_device(dev)
    started=time.time();net=load_sam(dev);age_net=AgeEst().to(dev).eval();id_net=IDLoss().to(dev).eval() if identity else None
    records=[];weights={}
    for model,relative in MODELS.items():
        weight=base/relative if relative else base/'models/sam/sam_ffhq_aging.pt'
        weights[model]={'path':str(weight),'size':weight.stat().st_size,'mtime':weight.stat().st_mtime}
        if relative:net.encoder.load_state_dict(torch.load(weight,map_location='cpu'))
        folder=out/'raw'/model;folder.mkdir(parents=True,exist_ok=True)
        with torch.inference_mode():
            for item in manifest['samples']:
                key=item['key'];image=rgb(out/'inputs'/f'{key}.png');mask=rgb(out/'masks'/f'{key}.png')[:,:,0]>0
                x=torch.from_numpy(image.copy()).permute(2,0,1).float().div(127.5).sub(1)[None].to(dev)
                for age in AGES:
                    start=time.monotonic();condition=torch.full((1,1,256,256),age/100,device=dev)
                    y=net(torch.cat([x,condition],1),randomize_noise=False,resize=False);torch.npu.synchronize()
                    if tuple(y.shape)!=(1,3,1024,1024):raise RuntimeError('Native 1024 contract failed')
                    pooled=net.face_pool(y);estimated,_=age_net(pooled);identity_score=float(id_net.cos(pooled,x).item()) if id_net else None
                    generated=((y[0].clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy()
                    filename=folder/f'{key}_a{age}.png';Image.fromarray(generated).save(filename)
                    records.append({'key':key,'model':model,'target_age':age,'source_age_est':item['metadata']['age_exp'],'file':str(filename),
                      'width':1024,'height':1024,'output_kind':'raw_native','identity_cosine':identity_score,'estimated_age':float(estimated.item()*100),
                      'estimated_age_error':abs(float(estimated.item()*100)-age),'seconds_with_metrics_and_save':time.monotonic()-start,'color':compute(image,generated,mask)})
                    dump(out/'records_native1024.json',records)
                print(f'{model} {key}: four native 1024 outputs',flush=True)
        contacts(out,manifest,list(MODELS)[:list(MODELS).index(model)+1])
    dump(out/'summary_native1024.json',aggregate(records));dump(out/'run.json',{'physical_device':1,'logical_device':dev,'visible_devices':os.environ['ASCEND_RT_VISIBLE_DEVICES'],
      'weights':weights,'script_sha256':sha(__file__),'seconds':time.time()-started,'native_resolution':1024,'targets':AGES,'identity_metric_caveat':'IR-SE50 and AgeEst participate in training; these are not independent validation.'})
    print('NATIVE_AUDIT_COMPLETE',flush=True)

def historical(base,out):
    manifest=json.loads((out/'manifest.json').read_text());records=[]
    for model,dirname in [('base','eval_base'),('ft_l1_1000','eval_l1_1000'),('ft_l5_250','eval_l5_250')]:
        hist={Path(r['input']).stem+'_'+r['tag']:r for r in json.loads((base/'out'/dirname/'records.json').read_text())}
        for item in manifest['samples']:
            key=item['key'];reference=rgb(out/'inputs'/f'{key}.png');mask=rgb(out/'masks'/f'{key}.png')[:,:,0]>0
            for age in AGES:
                image=base/'out'/dirname/f'{key}_a{age}.png';old=hist[f'{key}_a{age}']
                records.append({'key':key,'model':model,'target_age':age,'width':256,'height':256,'file':str(image),'color':compute(reference,rgb(image),mask),
                  'identity_cosine':old['id_cos'],'estimated_age':old['age_est'],'estimated_age_error':abs(old['age_est']-age)})
    dump(out/'records_historical256.json',records);dump(out/'summary_historical256.json',aggregate(records));print('CPU_HISTORICAL_COMPLETE',flush=True)

def postprocess(out,module_path,strength):
    spec=importlib.util.spec_from_file_location('source_color_candidate',str(module_path));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    manifest=json.loads((out/'manifest.json').read_text());raw_records=json.loads((out/'records_native1024.json').read_text());records=[]
    for old in raw_records:
        key,model,age=old['key'],old['model'],old['target_age'];source=rgb(out/'inputs'/f'{key}.png');raw=rgb(old['file']);mask=rgb(out/'masks'/f'{key}.png')[:,:,0]>0
        corrected,meta=module.preserve_source_chroma(source,raw,strength=strength)
        filename=out/'corrected'/model/f'{key}_a{age}.png';filename.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(corrected).save(filename)
        raw_l=cv2.cvtColor(raw.astype(np.float32)/255,cv2.COLOR_RGB2LAB)[:,:,0];corrected_l=cv2.cvtColor(corrected.astype(np.float32)/255,cv2.COLOR_RGB2LAB)[:,:,0]
        native_mask=cv2.resize(mask.astype(np.uint8),(raw.shape[1],raw.shape[0]),interpolation=cv2.INTER_NEAREST)>0
        l_diff=corrected_l-raw_l;hf_diff=(corrected_l-cv2.GaussianBlur(corrected_l,(0,0),2))-(raw_l-cv2.GaussianBlur(raw_l,(0,0),2))
        records.append({'key':key,'model':model+'_corrected','target_age':age,'file':str(filename),'width':corrected.shape[1],'height':corrected.shape[0],
          'output_kind':'postprocessed_native1024','color':compute(source,corrected,mask),'correction':meta,
          'delta_L_from_raw_proxy':float(np.median(lab(corrected)[mask,0])-np.median(lab(raw)[mask,0])),
          'native_L_preservation':{'mean_abs_global':float(np.mean(np.abs(l_diff))),'max_abs_global':float(np.max(np.abs(l_diff))),'cheek_rms':float(np.sqrt(np.mean(l_diff[native_mask]**2))),'highfreq_sigma2_global_rms_difference':float(np.sqrt(np.mean(hf_diff**2))),'highfreq_sigma2_cheek_rms_difference':float(np.sqrt(np.mean(hf_diff[native_mask]**2)))},
          'identity_cosine':None,'estimated_age_error':None})
    dump(out/'records_corrected1024.json',records);dump(out/'summary_raw_vs_corrected.json',aggregate(raw_records+records))
    paired=[]
    for model in MODELS:
        a=[r for r in raw_records if r['model']==model];b={(r['key'],r['target_age']):r for r in records if r['model']==model+'_corrected'}
        paired.append({'model':model,'n':len(a),'correction_applied':sum(r['correction']['applied'] for r in b.values()),
          **{metric:{'mean_before':float(np.mean([r['color'][metric] for r in a])), 'mean_after':float(np.mean([b[(r['key'],r['target_age'])]['color'][metric] for r in a])),
            'improved_count':sum(b[(r['key'],r['target_age'])]['color'][metric]<r['color'][metric] for r in a)} for metric in ['abs_delta_L','chroma_shift_ab','median_color_distance_Lab76']}})
    dump(out/'correction_comparison.json',{'method_module':str(module_path),'module_sha256':sha(module_path),'strength':strength,'models':paired,
      'independence':'Scoring uses source-YuNet cheek ellipses; correction uses fixed FFHQ patches. Both derive from same source colors, so reduced error is not independent perceptual quality evidence. Check source masks and visuals; L/texture/age/identity still need review.'})
    for item in manifest['samples']:
        key=item['key'];tile=256;rowh=292;sheet=Image.new('RGB',(tile*5,rowh*4),'white');draw=ImageDraw.Draw(sheet)
        rows=[('raw','base'),('raw','ft_l1_1000'),('corrected','ft_l1_1000'),('raw','ft_l5_250')]
        for row,(folder,model) in enumerate(rows):
            y=row*rowh;sheet.paste(Image.open(out/'inputs'/f'{key}.png').convert('RGB'),(0,y+28));draw.text((4,y+7),f'{model} {folder}',(0,0,0))
            for col,age in enumerate(AGES,1):
                image=out/folder/model/f'{key}_a{age}.png';sheet.paste(Image.open(image).convert('RGB').resize((tile,tile),Image.Resampling.LANCZOS),(col*tile,y+28));draw.text((col*tile+4,y+7),f'target {age}',(0,0,0))
        sheet.save(out/'contacts'/f'{key}_AB.jpg',quality=95)
    ranking=sorted(({'key':item['key'],'mean_raw_l1_chroma_shift':float(np.mean([r['color']['chroma_shift_ab'] for r in raw_records if r['model']=='ft_l1_1000' and r['key']==item['key']])),
      'panel':str(out/'contacts'/f"{item['key']}_AB.jpg")} for item in manifest['samples']),key=lambda r:-r['mean_raw_l1_chroma_shift'])
    dump(out/'worst_l1_chroma_panels.json',ranking[:4]);print('CPU_POSTPROCESS_COMPLETE',flush=True)

def score_corrected(base,out):
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!='1':raise RuntimeError('Metric model execution only allowed on physical NPU1 via ASCEND_RT_VISIBLE_DEVICES=1')
    import torch,torch_npu
    sys.path.insert(0,str(base/'src'));from sam_losses import IDLoss,AgeEst
    torch.npu.set_device('npu:0');idl=IDLoss().to('npu:0').eval();age_net=AgeEst().to('npu:0').eval()
    raw_records=json.loads((out/'records_native1024.json').read_text());pairs=[]
    def tensor(image):
        # Identical quantized RGB / area resize preprocessing for raw and corrected.
        small=cv2.resize(image,(256,256),interpolation=cv2.INTER_AREA)
        return torch.from_numpy(small.copy()).permute(2,0,1).float().div(127.5).sub(1)[None].to('npu:0')
    with torch.inference_mode():
        for record in raw_records:
            key,model,age=record['key'],record['model'],record['target_age'];x=tensor(rgb(out/'inputs'/f'{key}.png'));a=tensor(rgb(record['file']));b=tensor(rgb(out/'corrected'/model/f'{key}_a{age}.png'))
            id_raw=float(idl.cos(a,x).item());id_corrected=float(idl.cos(b,x).item());id_between=float(idl.cos(a,b).item());age_raw=float(age_net(a)[0].item()*100);age_corrected=float(age_net(b)[0].item()*100)
            pairs.append({'key':key,'model':model,'target_age':age,'id_source_raw':id_raw,'id_source_corrected':id_corrected,'id_delta_corrected_minus_raw':id_corrected-id_raw,
              'id_raw_vs_corrected':id_between,'estimated_age_raw':age_raw,'estimated_age_corrected':age_corrected,'estimated_age_delta':age_corrected-age_raw})
    summary={}
    for model in MODELS:
        part=[p for p in pairs if p['model']==model];summary[model]={'n':len(part),**{field:{'mean':float(np.mean([p[field] for p in part])),'min':float(np.min([p[field] for p in part])),'max':float(np.max([p[field] for p in part]))} for field in ['id_source_raw','id_source_corrected','id_delta_corrected_minus_raw','id_raw_vs_corrected','estimated_age_delta']},'id_drop_gt_0_02_count':sum(p['id_delta_corrected_minus_raw']<-.02 for p in part)}
    dump(out/'records_identity_raw_corrected.json',pairs);dump(out/'summary_identity_raw_corrected.json',{'models':summary,'caveats':'IR-SE50/AgeEst are training-loss models, not independent perceptual metrics. Source identity similarity can change from chroma normalization; no real-person identity assertion. All images are the same declared input across targets.', 'preprocessing':'Both raw/corrected PNG RGB uint8 resized to256 with INTER_AREA then normalized; this differs slightly from native tensor face_pool used during generation.'})
    print('NPU1_IDENTITY_AB_COMPLETE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['select','historical','generate','postprocess','score-corrected']);parser.add_argument('--base',type=Path,default=Path(BASE_DEFAULT));parser.add_argument('--out',type=Path,required=True);parser.add_argument('--count',type=int,default=10);parser.add_argument('--identity',action='store_true');parser.add_argument('--correction',type=Path);parser.add_argument('--strength',type=float,default=.65)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(2)
    if args.mode=='select':select(args.base,args.out,args.count)
    elif args.mode=='historical':historical(args.base,args.out)
    elif args.mode=='postprocess':postprocess(args.out,args.correction,args.strength)
    elif args.mode=='score-corrected':score_corrected(args.base,args.out)
    else:generate(args.base,args.out,args.identity)
