"""Build a local data-source audit preview; no model training or inference."""
import argparse
import html
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args();root=args.directory
    esc=lambda value:html.escape(str(value),quote=True)
    blocks=[]
    for row in map(json.loads,(root/'ffhq-pilot32/manifest.jsonl').read_text().splitlines()):
        if row['status']!='verified':continue
        info=row['source_metadata'];image=Path(row['image_path']).resolve().relative_to(root.resolve())
        label='–'.join(map(str,row['age_interval']))
        blocks.append(f'''<article><a href="{esc(image)}" target="_blank"><img loading="lazy" src="{esc(image)}" alt="FFHQ {row['image_id']}"></a><h3>FFHQ {row['image_id']:05d} · {label} 岁档</h3><p>原生1024；源脸短边 {row['source_face_short_edge']} px；人工年龄档置信度 {row['age_confidence']:.2f}。</p><p><a href="{esc(info['photo_url'])}" target="_blank">{esc(info['author'])}</a> · <a href="{esc(info['license_url'])}" target="_blank">{esc(info['license'])}</a></p></article>''')
    ffhq=''.join(blocks);blocks=[]
    for row in map(json.loads,(root/'appa-asian-pilot/manifest.jsonl').read_text().splitlines()):
        if row['status']!='verified':continue
        info=row['annotation'];image=Path(row['files'][1]['path']).resolve().relative_to(root.resolve())
        source=Path(row['files'][0]['path']).resolve().relative_to(root.resolve())
        blocks.append(f'''<article><a href="{esc(source)}" target="_blank"><img loading="lazy" src="{esc(image)}" alt="APPA {esc(info['file'])}"></a><h3>APPA {esc(info['file'])}</h3><p>发布方真实年龄 {esc(info['real_age'])}；表观年龄评分均值 {float(info['apparent_age_avg']):.1f}；{esc(info['num_ratings'])} 人评分。</p><p>发布方扩展标注：Asian；裁脸尺寸 {'×'.join(map(str,row['files'][1]['size']))}。官方 test 集，仅作评估候选。</p></article>''')
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>年龄编辑数据核查</title><style>*{{box-sizing:border-box}}body{{margin:0;background:#f4f4ef;color:#263432;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1400px;margin:auto;padding:24px}}h1{{font-size:28px}}h2{{margin-top:36px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:16px}}article{{background:white;padding:12px;border-radius:10px}}img{{width:100%;aspect-ratio:1;object-fit:contain;background:#e8e8e4}}h3{{font-size:17px;margin:8px 0}}p{{margin:8px 0}}a{{color:#166958}}article p{{font-size:14px}}</style><main><h1>年龄编辑数据来源与样本核查 · 2026-11-13</h1><p>已取得并校验的数据样本。这里没有模型生成图，也没有训练收益结论。点击FFHQ查看原生图，点击APPA裁脸查看原照。</p><nav><a href="#ffhq">32张 FFHQ-Aging</a> · <a href="#appa">12张 APPA-REAL</a> · <a href="ffhq-representatives.jpg">FFHQ代表图</a> · <a href="appa-representatives.jpg">APPA代表图</a></nav><h2 id="ffhq">FFHQ-Aging：高清与高龄训练候选</h2><p>四个成人年龄档各8张；这些是人工表观年龄档，不是精确出生年龄。1024图片和官方文件／像素MD5一致。无身份与亚洲类别标注，尚未划分本项目训练／评估集。基础生成器已经使用FFHQ预训练，因此不能把这里当作独立测试。图片保留各自许可与作者；数据集与年龄标注有非商业研究及署名／相同方式共享条款。</p><div class="grid">{ffhq}</div><h2 id="appa">APPA-REAL：真实年龄与亚洲标注的评估候选</h2><p>从官方test集按真实年龄20–39／40–59／60–74／75–100各取3张。Asian为发布方宽泛类别，不能等同于全部是东亚人脸。裁脸尺寸与质量差异很大，保留困难样例。官方页面和README未给出完整训练／再分发许可，本次只核查公开小样，不将这批test照片用于训练或对外再发布。</p><div class="grid">{''.join(blocks)}</div><p><a href="image-validation.json">图片校验记录</a> · <a href="ffhq-quality-statistics.json">FFHQ候选统计</a> · <a href="appa-asian-statistics.json">APPA亚洲标注统计</a></p></main></html>'''
    (root/'index.html').write_text(page)
    print(root/'index.html')


if __name__=='__main__':main()
