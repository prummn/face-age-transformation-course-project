/* Frozen-frame age mirror: same source geometry for every age; no live face tracking claim. */
let mirrorAnimation=0,mirrorFrameIndex=-1;
function clearMirrorFrames(){
 cancelAnimationFrame(mirrorAnimation);mirrorAnimation=0;mirrorFrameIndex=-1;
 for(const page of S.pages||[])if(page.composite)release(page.image);
 for(const id of ['#mirrorBase','#mirrorNext','#mirrorFrozen']){const c=$(id);if(c){c.width=c.height=0;}}
}
function renderMirrorFrame(){
 const active=S.mode==='mirror'&&S.state==='s-grid',host=$('#immersiveMirror');host.hidden=!active;
 if(!active)return;
 const page=S.pages[S.pageIndex];if(!page)return;
 host.classList.toggle('camera-reflection',S.source==='camera'&&S.facing==='user');
 $('#mirrorAge').textContent=page.t===null?'此刻':page.t+' 岁';
 if(mirrorFrameIndex===S.pageIndex)return;
 cancelAnimationFrame(mirrorAnimation);
 const base=$('#mirrorBase'),next=$('#mirrorNext'),w=page.image.width,h=page.image.height;
 if(base.width!==w||base.height!==h){base.width=next.width=w;base.height=next.height=h;mirrorFrameIndex=-1;}
 const g=base.getContext('2d'),n=next.getContext('2d');
 if(mirrorFrameIndex<0){g.drawImage(page.image,0,0);next.style.opacity=0;}
 else{
  // Collapse an interrupted dissolve to its current visible frame.
  const alpha=Number(next.style.opacity)||0;
  if(alpha){g.globalAlpha=alpha;g.drawImage(next,0,0);g.globalAlpha=1;}
  n.clearRect(0,0,w,h);n.drawImage(page.image,0,0);next.style.opacity=0;
  const started=performance.now(),duration=matchMedia('(prefers-reduced-motion: reduce)').matches?0:850;
  const tick=now=>{if(S.state!=='s-grid'||S.mode!=='mirror')return;const t=duration?Math.min(1,(now-started)/duration):1;next.style.opacity=t*t*(3-2*t);if(t<1)mirrorAnimation=requestAnimationFrame(tick);else{g.drawImage(next,0,0);next.style.opacity=0;mirrorAnimation=0;}};
  mirrorAnimation=requestAnimationFrame(tick);
 }
 mirrorFrameIndex=S.pageIndex;
}
const mirrorStyle=document.createElement('style');mirrorStyle.textContent=`
body[data-experience="mirror"][data-view="s-grid"]{overflow:hidden;background:#111}
body[data-experience="mirror"][data-view="s-grid"] .app{width:100%;height:100dvh;max-width:none;margin:0;padding:0;border-radius:0;background:#111}
body[data-experience="mirror"][data-view="s-grid"] header{position:absolute;z-index:12;left:0;right:0;color:white;background:linear-gradient(#0006,transparent)}
body[data-experience="mirror"][data-view="s-grid"] main{padding:0;max-width:none;height:100dvh!important}
body[data-experience="mirror"] #s-grid .page-track,body[data-experience="mirror"] #s-grid .page-dots,body[data-experience="mirror"] #s-grid .page-heading h2,body[data-experience="mirror"] #resultColorView,body[data-experience="mirror"] #bNativeSave,body[data-experience="mirror"] #resultDisclosure{display:none}
#immersiveMirror{position:absolute;inset:0;background:#111;overflow:hidden}
#immersiveMirror canvas{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}
#immersiveMirror.camera-reflection canvas{transform:scaleX(-1)}
.mirror-caption{position:absolute;left:0;right:0;bottom:145px;color:white;text-align:center;text-shadow:0 2px 12px #000}
.mirror-caption span{display:block;font-size:clamp(36px,6vw,64px);font-weight:500}.mirror-caption small{font-size:14px}
body[data-experience="mirror"] #s-grid .page-heading{position:absolute;right:24px;top:85px;z-index:11}
body[data-experience="mirror"] #s-grid .page-heading button{color:white;background:#0005}
body[data-experience="mirror"] #s-grid .page-controls{position:absolute;left:24px;right:24px;bottom:90px;z-index:11;justify-content:space-between}
body[data-experience="mirror"] #s-grid .result-tools{position:absolute;left:24px;right:24px;bottom:24px;z-index:11;display:flex;justify-content:center}
body[data-experience="mirror"] #s-grid .result-tools button{font-size:14px;padding:10px 18px;background:#ffffffd9}
`;document.head.append(mirrorStyle);
if(new URLSearchParams(location.search).get('experience')==='mirror'){
 S.mode='mirror';$('#experienceMode').value='mirror';document.body.dataset.experience='mirror';
 notice('#hudText','开启镜头后，站稳自动定格，轮播不同年龄；离开 5 秒自动清除。');
}

const frozen=document.createElement('canvas');frozen.id='mirrorFrozen';frozen.hidden=true;$('#s-warp').prepend(frozen);
function renderMirrorProcessing(id){
 const c=$('#mirrorFrozen');if(!c)return;c.hidden=!(S.mode==='mirror'&&id==='s-warp'&&S.original);
 if(!c.hidden){c.width=S.original.width;c.height=S.original.height;c.getContext('2d').drawImage(S.original,0,0);c.style.transform=S.source==='camera'&&S.facing==='user'?'scaleX(-1)':'none';}
}
mirrorStyle.textContent+=`
body[data-experience="mirror"][data-view="s-warp"] .app{width:100%;height:100dvh;max-width:none;margin:0;padding:0;border-radius:0;background:#111}
body[data-experience="mirror"][data-view="s-warp"] main{padding:0;max-width:none;height:100dvh}
body[data-experience="mirror"][data-view="s-warp"] header{position:absolute;z-index:12;left:0;right:0;color:white}
body[data-experience="mirror"] #s-warp{position:absolute;inset:0;padding:calc(100dvh - 240px) 24px 24px;color:white;box-sizing:border-box}
#mirrorFrozen{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;z-index:0}
body[data-experience="mirror"] #s-warp>:not(canvas){position:relative;z-index:1;text-shadow:0 1px 6px black}
body[data-experience="mirror"] #s-warp h2,body[data-experience="mirror"] #taskDetails,body[data-experience="mirror"] #s-warp .help-line{display:none}
body[data-experience="mirror"] #taskSummary{background:#0008;padding:10px;border-radius:12px}
`;
mirrorStyle.textContent+=`
body[data-experience="mirror"] #s-grid .page-heading{top:150px}
body[data-experience="mirror"] #s-grid .result-tools button{color:#222}
`;
