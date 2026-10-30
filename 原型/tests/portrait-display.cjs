const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
 let failed=0;
 for(const viewport of [{width:1080,height:1920},{width:720,height:1280},{width:1440,height:2560}]){
  const p=await browser.newPage({viewport});await p.route('https://**/*',r=>r.abort());
  const inside=async selector=>{const r=await p.locator(selector).boundingBox();assert.ok(r&&r.x>=-1&&r.y>=-1&&r.x+r.width<=viewport.width+1&&r.y+r.height<=viewport.height+1,selector+' fits display');};
  const noOverflow=async()=>{const r=await p.evaluate(()=>({state:S.state,doc:[document.documentElement.scrollWidth,document.documentElement.scrollHeight],app:[document.querySelector('#screen').scrollHeight,document.querySelector('#screen').clientHeight],ok:document.documentElement.scrollWidth<=innerWidth+1&&document.documentElement.scrollHeight<=innerHeight+1&&document.querySelector('#screen').scrollHeight<=document.querySelector('#screen').clientHeight+1}));assert.equal(r.ok,true,'no page or inner overflow '+JSON.stringify(r));};
  try{
   await p.goto('http://localhost:65512/');
   const rect=await p.locator('#cameraWell').boundingBox();assert.ok(rect.width>=viewport.width*.98,'portrait camera fills display width');assert.ok(rect.height>=viewport.height*.98,'portrait camera fills display height');
   await inside('#bStartCamera');await inside('#bUpload');await noOverflow();
   assert.ok(await p.locator('header h1').evaluate(el=>parseFloat(getComputedStyle(el).fontSize))>=24,'display title is readable');
   await p.screenshot({path:`tmp/display-${viewport.width}-camera.png`});
   await p.locator('#bInfo').click();await inside('#infoSheet .dialog-card');await inside('#bInfoClose');await p.locator('#bInfoClose').click();
   await p.evaluate(()=>{S.original=canvas(720,960);S.face=canvas();const g=S.face.getContext('2d'),grad=g.createLinearGradient(0,0,256,256);grad.addColorStop(0,'#d1d9dc');grad.addColorStop(1,'#96978b');g.fillStyle=grad;g.fillRect(0,0,256,256);S.age=30;S.ageInfo={age:30,source:'demo'};S.targets=[15,40,55,70];for(const t of S.targets){const c=canvas();c.getContext('2d').drawImage(S.face,0,0);S.results.set(t,c);}buildPages();});
   await inside('#bPlay');await inside('#bRetakeResults');await inside('#resultDisclosure');await noOverflow();
   assert.ok((await p.locator('#pageTrack').boundingBox()).width>=viewport.width*.8,'result uses display width');
   await p.screenshot({path:`tmp/display-${viewport.width}-results.png`});
   await p.evaluate(()=>{S.staticFaces=[{id:1,x:.2,y:.2,w:.3,h:.4}];S.staticSelected=1;show('s-select');drawSource();drawStaticOverlay();});await inside('#bSelectConfirm');await noOverflow();
   await p.evaluate(()=>showInput());await inside('#bContinue');await noOverflow();
   await p.evaluate(()=>{show('s-warp');renderTasks();});await inside('#bCancelGenerate');await noOverflow();
   await p.evaluate(()=>{buildPages();openDetail(15);});await inside('#bSave');await noOverflow();
   await p.evaluate(()=>{S.view='paste';S.transform={inverse:[1,0,0,1,0,0]};prepareComparison();});await inside('#bSave');await noOverflow();
   await p.evaluate(()=>buildOverview());await inside('#bOverviewSave');await noOverflow();
   await p.evaluate(()=>{for(const t of [5,10,80,90]){const c=canvas();S.results.set(t,c);S.pages.push({t,image:c,label:String(t)});}buildOverview();});await inside('#bOverviewSave');await noOverflow();
   // Reevaluate the layout when the physical display rotates, without a reload.
   await p.setViewportSize({width:1920,height:1080});await p.evaluate(()=>toWaiting());assert.ok((await p.locator('#cameraWell').boundingBox()).width<=430,'landscape retains phone preview');
   await p.setViewportSize(viewport);assert.ok((await p.locator('#cameraWell').boundingBox()).width>=viewport.width*.98,'rotation restores portrait layout');
   assert.ok(await p.evaluate(()=>{const g={beginPath(){},rect(){},stroke(){},fillRect(){},fillText(){}};drawFaces(g,[{id:1,x:.1,y:.1,w:.2,h:.3}],{map:p=>p},1,false);return parseFloat(g.font);})>=16,'face ID labels scale with display');
   console.log('PASS portrait display capture, results, secondary pages, dialogs and rotation',viewport);
  }catch(e){failed++;console.log('FAIL',viewport,e.message);}
  await p.close();
 }
 await browser.close();process.exitCode=failed?1:0;
})().catch(e=>{console.error(e);process.exit(1);});
