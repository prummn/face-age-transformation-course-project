/* Uploaded photos use a still scene. Camera results use the current video frame. */
let liveTarget=null, previousTarget=null, changedAt=0;
const liveLayers=new Map();

function drawLiveMirror(video,face,now){
 const base=$('#mirrorBase'),w=video.videoWidth,h=video.videoHeight;
 if(!w||!h)return;
 if(base.width!==w||base.height!==h){base.width=w;base.height=h;}
 const g=base.getContext('2d');
 g.drawImage(video,0,0,w,h);
 const source=staticFace();
 if(!face||face.id!==source?.id||!source.eyes||!face.eyes)return;
 const eyes=(f,width,height)=>f.eyes.map(p=>({x:p.x*width,y:p.y*height})).sort((a,b)=>a.x-b.x);
 const [a,b]=eyes(source,S.original.width,S.original.height),[c,d]=eyes(face,w,h);
 const dx=b.x-a.x,dy=b.y-a.y,length=dx*dx+dy*dy;
 if(length<4)return;
 const scaleCos=((d.x-c.x)*dx+(d.y-c.y)*dy)/length;
 const scaleSin=((d.y-c.y)*dx-(d.x-c.x)*dy)/length;
 const t=matchMedia('(prefers-reduced-motion: reduce)').matches?1:clamp((now-changedAt)/650,0,1);
 g.save();
 g.setTransform(scaleCos,scaleSin,-scaleSin,scaleCos,c.x-scaleCos*a.x+scaleSin*a.y,c.y-scaleSin*a.x-scaleCos*a.y);
 function layer(target,opacity){
  if(target===null||opacity<=0)return;
  if(!liveLayers.has(target))liveLayers.set(target,pasteBack(resultImage(target),false));
  g.globalAlpha=opacity;g.drawImage(liveLayers.get(target),0,0);
 }
 layer(previousTarget,1-t);layer(liveTarget,t);
 g.restore();
}
let mirrorAnimation=0,mirrorFrameIndex=-1;
function clearMirrorFrames(){
 for(const layer of liveLayers.values())release(layer);
 liveLayers.clear();previousTarget=liveTarget=null;changedAt=0;
 cancelAnimationFrame(mirrorAnimation);mirrorAnimation=0;mirrorFrameIndex=-1;
 for(const page of S.pages||[])if(page.composite)release(page.image);
 for(const id of ['#mirrorBase','#mirrorNext','#mirrorFrozen']){const c=$(id);if(c){c.width=c.height=0;}}
}
function renderMirrorFrame(){
 const active=S.mode==='mirror'&&S.state==='s-grid',host=$('#immersiveMirror');host.hidden=!active;
 if(!active)return;
 const page=S.pages[S.pageIndex];if(!page)return;
 host.classList.toggle('camera-reflection',S.source==='camera'&&S.facing==='user');
 const age=Number.isFinite(S.age)?S.age:null,source=S.ageInfo?.source;
 const reference=age===null?'当前年龄未设置':source==='model'?`参考估龄约 ${age} 岁`:source==='demo'?`演示占位 ${age} 岁`:`当前 ${age} 岁（手动）`;
 $('#mirrorAge').textContent=page.t===null?'此刻':`目标 ${page.t} 岁`;
 $('#mirrorAgeReference').textContent=reference+' · 年龄外观模拟';
 if(S.source==='camera'&&S.stream){
  if(mirrorFrameIndex!==S.pageIndex){previousTarget=liveTarget;liveTarget=page.t;changedAt=performance.now();}
  mirrorFrameIndex=S.pageIndex;$('#mirrorNext').style.opacity=0;
  const video=$('#video');if(video.readyState>=2)drawLiveMirror(video,selectedFace(),performance.now());
  return;
 }
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
 notice('#hudText','开启镜头后站稳，倒计时结束会自动拍摄。');
 navigator.permissions?.query({name:'camera'}).then(permission=>{
  if(permission.state==='granted'&&!cameraRequest&&!S.stream&&S.state==='s-mirror')startCamera();
 }).catch(()=>{});
}

const frozen=document.createElement('canvas');frozen.id='mirrorFrozen';frozen.hidden=true;$('#s-warp').prepend(frozen);
function renderMirrorProcessing(id){
 const c=$('#mirrorFrozen');if(!c)return;c.hidden=!(S.mode==='mirror'&&id==='s-warp'&&S.original);
 if(!c.hidden){c.width=S.original.width;c.height=S.original.height;c.getContext('2d').drawImage(S.original,0,0);c.style.transform=S.source==='camera'&&S.facing==='user'?'scaleX(-1)':'none';}
}
// Keep all captions and controls outside the source image stage.
$('#s-grid').append($('#immersiveMirror .mirror-caption'));
