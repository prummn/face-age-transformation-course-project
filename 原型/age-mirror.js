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
if(new URLSearchParams(location.search).get('experience')==='mirror'){
 S.mode='mirror';$('#experienceMode').value='mirror';document.body.dataset.experience='mirror';
 notice('#hudText','开启镜头后，站稳自动定格，轮播不同年龄；离开 5 秒自动清除。');
}

const frozen=document.createElement('canvas');frozen.id='mirrorFrozen';frozen.hidden=true;$('#s-warp').prepend(frozen);
function renderMirrorProcessing(id){
 const c=$('#mirrorFrozen');if(!c)return;c.hidden=!(S.mode==='mirror'&&id==='s-warp'&&S.original);
 if(!c.hidden){c.width=S.original.width;c.height=S.original.height;c.getContext('2d').drawImage(S.original,0,0);c.style.transform=S.source==='camera'&&S.facing==='user'?'scaleX(-1)':'none';}
}
// Keep all captions and controls outside the source image stage.
$('#s-grid').append($('#immersiveMirror .mirror-caption'));
