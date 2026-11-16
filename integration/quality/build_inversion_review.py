"""Build a local static viewer for probe_inversion.py results (no inference API)."""
import argparse
from pathlib import Path

PAGE = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>年龄编辑 · 推理对照</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f5f4ef;color:#243234;font:16px/1.6 system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:24px}h1{font-size:28px;margin:0}p{margin:8px 0 20px}.muted{color:#59696c}
.controls{display:flex;gap:16px;flex-wrap:wrap;background:white;padding:16px;border-radius:12px;margin:16px 0}
label{display:grid;font-size:14px;gap:4px}select{font:inherit;padding:8px;border:1px solid #bbc6c7;border-radius:6px;max-width:330px}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}figure{margin:0;min-width:0;background:white;padding:12px;border-radius:12px}
.frame{aspect-ratio:1;overflow:hidden;background:#ddd}.frame img{width:100%;height:100%;object-fit:contain;display:block}
.crop .frame img{transform:scale(1.9);transform-origin:50% 51%}figcaption{font-weight:600;margin-bottom:8px}.detail{font-size:14px;min-height:3em;margin-top:8px}
a{color:#18645b}button{font:inherit;cursor:pointer;border:1px solid #abc1b9;border-radius:6px;padding:6px 12px;background:white}
.compare{max-width:650px;margin:24px auto}.slider{position:relative;aspect-ratio:1;overflow:hidden}.slider img{width:100%;position:absolute;inset:0}
#overlay{clip-path:inset(0 50% 0 0)}input[type=range]{width:100%}summary{cursor:pointer}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px 12px;border-bottom:1px solid #ccd4d0;text-align:left}
@media(max-width:760px){main{padding:14px}.grid{grid-template-columns:1fr}.frame{max-height:75vh}h1{font-size:23px}}
</style><main>
<h1>同一张脸，分别检查重建与年龄编辑</h1>
<p class="muted">独立离线实验，线上仍为原 Release 权重。所有编码器、生成器权重冻结；“潜变量优化”只为每张照片拟合一个 W+，没有训练新模型。原始生成图与可选保色图均保留。</p>
<div class="controls">
<label>照片<select id="sample"></select></label>
<label>目标年龄条件<select id="age"></select></label>
<label>对照方法<select id="route">
<option value="refined_detail">逐照片潜变量优化 · 细节约束</option>
<option value="refined">逐照片潜变量优化 + 原年龄位移</option>
<option value="original">原始 SAM</option>
<option value="fine_direct">直接微调编码器</option>
<option value="freeze_first3">不编辑前 3 个潜变量层</option>
<option value="freeze_first7">不编辑前 7 个潜变量层</option>
<option value="half_delta">年龄位移减半</option>
<option value="refined_half">潜变量优化 + 年龄位移减半</option>
<option value="psp_only">仅 pSp 重建（只在源年龄条件可选）</option>
<option value="psp_delta">pSp 重建 + 原年龄位移</option>
</select></label>
<label>显示<select id="kind"><option value="raw">原始模型输出</option><option value="color">可选原照保色</option></select></label>
<label>查看范围<select id="crop"><option value="full">完整脸图</option><option value="face">放大五官</option></select></label>
</div>
<p id="note"></p>
<div class="grid" id="pictures">
<figure><figcaption>上传／对齐参考</figcaption><div class="frame"><img id="source" alt="原照"></div><div class="detail" id="sourceInfo"></div><a id="sourceDownload" download>下载参考</a></figure>
<figure><figcaption>当前线上路线</figcaption><div class="frame"><img id="baseline" alt="当前混合路线输出"></div><div class="detail" id="baselineInfo"></div><a id="baselineDownload" download>下载 1024 PNG</a></figure>
<figure><figcaption id="candidateTitle">实验对照</figcaption><div class="frame"><img id="candidate" alt="实验输出"></div><div class="detail" id="candidateInfo"></div><a id="candidateDownload" download>下载 1024 PNG</a></figure>
</div>
<div class="compare"><b>滑动比较：左为实验，右为线上路线</b><div class="slider"><img id="underlay" alt="线上路线"><img id="overlay" alt="实验路线"></div><input id="split" aria-label="左右对比位置" type="range" min="0" max="100" value="50"></div>
<details><summary>同龄重建诊断与实验记录</summary><p>RGB 与中央脸部误差按 256 输入计算，单位为 0–255 像素绝对误差；仅衡量同龄重建，不能代替身份评价。年龄诊断沿用训练相关的 FairFace 网络，只作辅助观察。</p><table><thead><tr><th>方法</th><th>全图 MAE</th><th>中央脸部 MAE</th><th>网络参考估龄</th></tr></thead><tbody id="metrics"></tbody></table><p><a href="report.json" target="_blank">完整 JSON 记录</a> · <a id="overview" target="_blank">该照片的全部方法总览</a></p></details>
<p class="muted">只有两张诊断照片；上传图已对齐并重采样，本次没有重复对齐。减小年龄位移可能同时减弱年龄效果。年龄条件与模型估龄都不是照片真实年龄；生成结果不代表未来容貌预测。保色不改变五官结构。</p>
<p id="error" role="alert"></p>
<script>
const $=id=>document.getElementById(id);let report;
const titles={mixed:'当前混合路线',original:'原始 SAM',fine_direct:'直接微调编码器',refined:'逐照片潜变量优化',refined_detail:'潜变量优化 · 细节约束',freeze_first3:'不编辑前 3 层',freeze_first7:'不编辑前 7 层',half_delta:'年龄位移减半',refined_half:'潜变量优化 + 位移减半',psp_only:'仅 pSp',psp_delta:'pSp + 原年龄位移'};
function sample(){return report.samples[+$('sample').value]}
function ages(){const s=sample(),old=$('age').value;$('age').replaceChildren(...s.targets.map(a=>new Option(`${a}${a===s.source_condition?'（源年龄条件／同龄）':''}`,a)));$('age').value=s.targets.includes(+old)?old:s.source_condition;render()}
function render(){const s=sample(),a=+$('age').value,route=$('route').value,kind=$('kind').value;
let base=s.outputs.find(o=>o.route==='mixed'&&o.condition===a),candidate=s.outputs.find(o=>o.route===route&&o.condition===a),fallback=false;
if(!candidate&&a===s.source_condition&&['freeze_first3','freeze_first7','half_delta'].includes(route)){candidate=base;fallback=true}
for(const o of $('route').options){o.disabled=!s.outputs.some(x=>x.route===o.value&&x.condition===a)&&!(a===s.source_condition&&['freeze_first3','freeze_first7','half_delta'].includes(o.value))}
if(!candidate){$('route').value='refined';return render()}
$('source').src=s.name+'/reference.png';$('baseline').src=base[kind];$('candidate').src=candidate[kind];$('underlay').src=base[kind];$('overlay').src=candidate[kind];
$('sourceDownload').href=$('source').src;$('baselineDownload').href=base[kind];$('candidateDownload').href=candidate[kind];
$('candidateTitle').textContent='实验：'+titles[route];$('pictures').classList.toggle('crop',$('crop').value==='face');
$('sourceInfo').textContent=`参考估龄 ${s.reference_age_unrounded.toFixed(1)}；实际源年龄条件 ${s.source_condition}。`;
const info=o=>`共享网络诊断估龄 ${o.reference_age_diagnostic.toFixed(1)}；解码 ${o.decode_ms.toFixed(1)} ms。`+(kind==='color'?(o.color_metadata.applied?'已应用保色。':'未应用保色：'+o.color_metadata.reason+'。'):'');
$('baselineInfo').textContent=info(base);$('candidateInfo').textContent=info(candidate);
const fit=route.startsWith('refined')?`此照片额外拟合 ${report.fitting.steps} 步，耗时 ${(route==='refined_detail'?s.detail_fitting_seconds:s.fitting_seconds).toFixed(1)} 秒；不含首次模型加载。`:'';
$('note').textContent=(fallback?'同龄条件下年龄位移为零，该方法与线上锚点相同。':'')+fit;
$('overview').href=s.name+(report.follow_up?'/detail-overview.jpg':'/overview.jpg');$('metrics').replaceChildren();
for(const o of s.outputs.filter(o=>o.condition===s.source_condition)){
 const tr=document.createElement('tr');for(const text of [titles[o.route]||o.route,o.rgb_mae_0_255.toFixed(2),o.central_face_mae_0_255.toFixed(2),o.reference_age_diagnostic.toFixed(1)]){const td=document.createElement('td');td.textContent=text;tr.append(td)}$('metrics').append(tr)
}}
fetch('report.json').then(r=>{if(!r.ok)throw new Error('记录读取失败');return r.json()}).then(data=>{report=data;report.samples.forEach((s,i)=>$('sample').add(new Option(s.name==='uploaded'?'你上传的人像':'公开验证样例',i)));ages()}).catch(e=>$('error').textContent=e.message);
$('sample').onchange=ages;for(const id of ['age','route','kind','crop'])$(id).onchange=render;
$('split').oninput=()=>$('overlay').style.clipPath=`inset(0 ${100-$('split').value}% 0 0)`;
</script></main></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    if not (args.results / 'DONE').exists():
        raise RuntimeError('Experiment has not completed')
    (args.results / 'index.html').write_text(PAGE)
    print(args.results / 'index.html')


if __name__ == '__main__':
    main()
