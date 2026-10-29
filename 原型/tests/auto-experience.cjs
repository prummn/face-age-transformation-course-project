const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
 let failures=0;
 const cases=[
 ['displayed IDs survive detector reordering and temporary loss',async p=>{
  const result=await p.evaluate(()=>{resetTracking();F.nextId=1;const a={x:.12,y:.2,w:.22,h:.3},b={x:.63,y:.2,w:.22,h:.3};updateTracks([a,b],100);chooseFace(2);updateTracks([{...b},{...a}],300);renderFaceChoices();const ids=F.faces.map(f=>f.id),labels=[...document.querySelectorAll('#faceChoices button')].map(b=>b.textContent);updateTracks([],500);updateTracks([{...b},{...a}],700);const drawn=[];const g={beginPath(){},rect(){},stroke(){},fillRect(){},fillText(t){drawn.push(t);}};drawFaces(g,F.faces,{map:p=>p},F.selectedId,false);return {ids,labels,selected:selectedFace().id,drawn};});
  assert.deepEqual(result,{ids:[1,2],labels:['人脸 1','人脸 2'],selected:2,drawn:['1','2']});
 }],
 ['global matching preserves tracks when greedy matching would steal a closer face',async p=>{
  assert.deepEqual(await p.evaluate(()=>{resetTracking();F.nextId=1;updateTracks([{x:.15,y:.2,w:.24,h:.3},{x:.35,y:.2,w:.24,h:.3}],100);updateTracks([{x:.25,y:.2,w:.24,h:.3},{x:.16,y:.2,w:.24,h:.3}],300);return F.faces.map(f=>[f.id,f.x]);}),[[1,.16],[2,.25]]);
 }],
 ['all faces gate capture; pose, size, movement and loss restart countdown',async p=>{
  const result=await p.evaluate(()=>{
   const c=canvas(384,384),g=c.getContext('2d');for(let y=0;y<384;y+=4)for(let x=0;x<384;x+=4){g.fillStyle=(x+y)%8?'#aaa':'#666';g.fillRect(x,y,4,4);}
   const make=(x)=>({x,y:.2,w:.23,h:.32,eyes:[{x:x+.05,y:.29},{x:x+.18,y:.29}],nose:{x:x+.115,y:.35}});
   resetTracking();F.nextId=1;const base=[make(.15),make(.6)];let now=100;const step=(faces,advance=200)=>{now+=advance;updateTracks(faces.map(f=>({...f})),now);return autoCaptureReadiness(c,now);};
   let value;for(let i=0;i<27;i++)value=step(base);const steady=value.ready;
   const turned=base.map(f=>({...f}));turned[1].nose={x:.81,y:.35};const side=step(turned).ready;
   for(let i=0;i<27;i++)value=step(base);const recovered=value.ready;
   const tiny=[base[0],{...base[1],w:.1}];const small=step(tiny).ready;
   for(let i=0;i<27;i++)value=step(base);const moved=step([{...base[0],x:.18},base[1]]).ready;
   const lost=step([base[0]]).ready;const newArrival=step([...base,make(.38)]).ready;
   return {steady,side,recovered,small,moved,lost,newArrival};
  });assert.deepEqual(result,{steady:true,side:false,recovered:true,small:false,moved:false,lost:false,newArrival:false});
 }],
 ['slow drift and inference gaps never count as continuous stillness',async p=>{
  assert.equal(await p.evaluate(()=>{resetTracking();const c=canvas(384,384),g=c.getContext('2d');for(let y=0;y<384;y+=4)for(let x=0;x<384;x+=4){g.fillStyle=(x+y)%8?'#aaa':'#666';g.fillRect(x,y,4,4);}const f={id:1,x:.2,y:.2,w:.3,h:.4,eyes:[{x:.26,y:.3},{x:.44,y:.3}],nose:{x:.35,y:.38}};for(let i=0;i<27;i++){updateTracks([{...f}],100+i*200);autoCaptureReadiness(c,100+i*200);}if(!autoCaptureReadiness(c,5500).ready)throw Error("invalid stable baseline");resetTracking();let ready=false;for(let i=0;i<50;i++){const dx=i*.003;updateTracks([{...f,x:f.x+dx,eyes:f.eyes.map(p=>({...p,x:p.x+dx})),nose:{...f.nose,x:f.nose.x+dx}}],100+i*200);ready ||= autoCaptureReadiness(c,100+i*200).ready;}resetTracking();for(let i=0;i<15;i++){updateTracks([{...f}],100+i*200);autoCaptureReadiness(c,100+i*200);}const gap=autoCaptureReadiness(c,14000).ready;return ready||gap;}),false);
 }],
 ['default upload uses an honest placeholder and automatically plays age sequence',async p=>{
  await p.locator('#file').setInputFiles('tmp/qa_portrait.jpg');await p.waitForFunction(()=>S.state==='s-grid',null,{timeout:30000});
  assert.equal(await p.evaluate(()=>S.age),30);assert.equal(await p.evaluate(()=>S.ageInfo.source),'demo');assert.match(await p.evaluate(()=>S.pages[0].caption),/演示占位/);assert.ok(await p.evaluate(()=>S.autoTimer));
  assert.equal(await p.locator('#bPlay').isVisible(),true);assert.equal(await p.locator('#bPlay').getAttribute('aria-pressed'),'true');await p.waitForTimeout(600);const before=await p.evaluate(()=>S.pageIndex);await p.waitForFunction(i=>S.pageIndex!==i,before,{timeout:4500});
  await p.locator('#bPlay').click();assert.equal(await p.evaluate(()=>S.autoTimer),0);await p.locator('#bPlay').click();assert.ok(await p.evaluate(()=>S.autoTimer));await p.locator('#pageNext').click();assert.equal(await p.evaluate(()=>S.autoTimer),0);
  await p.screenshot({path:'tmp/auto-results.png',fullPage:true});
 }],
 ['failed or stalled age model falls back without a required age form',async p=>{
  for(const type of ['reject','timeout']){await p.evaluate(async type=>{toWaiting();CONFIG.estimateTimeoutMs=40;S.original=canvas(640,640);S.staticFaces=[{id:1,x:.2,y:.2,w:.4,h:.4,manual:true}];S.staticSelected=1;window.AgeMirrorAdapters={estimate:type==='reject'?async()=>{throw Error('offline');}:()=>new Promise(r=>window.lateAge=r)};await confirmFace();},type);assert.equal(await p.evaluate(()=>S.state),'s-grid');assert.equal(await p.evaluate(()=>S.ageInfo.source),'demo');}
  await p.evaluate(()=>window.lateAge({age:83}));await p.waitForTimeout(100);assert.equal(await p.evaluate(()=>S.age),30);
 }],
 ['mirror keeps the selected identity when another face remains after it leaves',async p=>{
  assert.equal(await p.evaluate(()=>{resetTracking();S.mode='mirror';S.state='s-grid';updateTracks([{x:.1,y:.2,w:.2,h:.3},{x:.6,y:.2,w:.2,h:.3}],100);chooseFace(1);updateTracks([{x:.6,y:.2,w:.2,h:.3}],2000);return F.selectedId===1&&selectedFace()===null&&F.lastSeen<2000;}),true);
 }],
 ['real local detection drives automatic capture of a stable two-face video',async p=>{
  await p.route('**/qa-live.jpg',r=>r.fulfill({path:'tmp/qa_portrait.jpg',contentType:'image/jpeg'}));
  await p.evaluate(async()=>{const img=new Image();img.src='/qa-live.jpg';await new Promise((ok,no)=>{img.onload=ok;img.onerror=no;});window.qaVideo=canvas(640,960);const g=qaVideo.getContext('2d');const draw=()=>{g.fillStyle='#888';g.fillRect(0,0,640,960);g.drawImage(img,245,50,310,400,40,200,240,340);g.drawImage(img,245,50,310,400,360,200,240,340);};draw();window.qaTick=setInterval(draw,80);navigator.mediaDevices.getUserMedia=async()=>qaVideo.captureStream(15);});
  await p.locator('#bStartCamera').click();await p.waitForFunction(()=>F.faces.length===2,null,{timeout:15000});
  console.log('LIVE QUALITY',await p.evaluate(()=>F.faces.map(f=>({id:f.id,box:[f.x,f.y,f.w,f.h],quality:qualityOf(probe,f)}))));
  await p.locator('#bInfo').click();await p.waitForTimeout(5500);assert.equal(await p.evaluate(()=>S.state),'s-mirror','options suspend automatic capture');await p.locator('#bInfoClose').click();
  await p.waitForFunction(()=>S.state==='s-grid',null,{timeout:15000});assert.equal(await p.evaluate(()=>S.source),'camera');assert.equal(await p.evaluate(()=>S.staticFaces.length),2);assert.equal(await p.evaluate(()=>S.stream),null);assert.ok(await p.evaluate(()=>S.autoTimer));
  await p.evaluate(()=>{toWaiting();clearInterval(window.qaTick);});
 }],
 ['clearing during age estimate discards late work',async p=>{
  await p.evaluate(()=>{S.original=canvas(640,640);S.staticFaces=[{id:1,x:.2,y:.2,w:.4,h:.4,manual:true}];S.staticSelected=1;window.AgeMirrorAdapters={estimate:()=>new Promise(r=>window.lateAge=r)};confirmFace();});await p.waitForFunction(()=>!!window.lateAge);await p.evaluate(()=>{toWaiting();window.lateAge({age:83});});await p.waitForTimeout(100);assert.equal(await p.evaluate(()=>S.state),'s-mirror');assert.equal(await p.evaluate(()=>S.results.size),0);
 }]
 ];
 for(const [name,run]of cases){const p=await browser.newPage({viewport:{width:390,height:844}});const errors=[];p.on('pageerror',e=>errors.push(e.message));await p.route('https://**/*',r=>r.abort());try{await p.goto('http://localhost:65512/');await run(p);assert.deepEqual(errors,[]);console.log('PASS',name);}catch(e){failures++;console.log('FAIL',name,e.stack);}await p.close();}
 await browser.close();process.exitCode=failures?1:0;
})().catch(e=>{console.error(e);process.exit(1);});
