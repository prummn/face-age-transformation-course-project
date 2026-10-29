const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH ? {executablePath:process.env.CHROME_PATH} : {})});
 const page=await browser.newPage({viewport:{width:390,height:844},isMobile:true,hasTouch:true});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));await page.route('https://**/*',r=>r.abort());
 await page.goto('http://localhost:65512/');await page.waitForTimeout(250);
 await page.locator('#bStartCamera').click();await page.waitForFunction(()=>document.querySelector('#hudText').classList.contains('error'));console.log('PASS camera denial gives upload guidance');
 await page.locator('#file').setInputFiles('tmp/qa_two_faces.jpg');await page.waitForFunction(()=>S.staticFaces.length===2,{},{timeout:60000});
 assert.equal(await page.locator('#bSelectConfirm').isDisabled(),true);
 await page.locator('#staticFaceChoices button').nth(1).click();assert.equal(await page.evaluate(()=>S.staticSelected),2);
 await page.screenshot({path:'tmp/mobile-multiple.png',fullPage:true});await page.locator('#bSelectConfirm').click();
 await page.locator('#currentAgeInput').fill('40');await page.locator('.age-options summary').click();await page.locator('#targetKind').selectOption('delta');await page.locator('#targetAgeInput').fill('-10');
 await page.evaluate(()=>{S.targets=[20,50];renderTargetChips();});await page.locator('#addTarget').click();assert.deepEqual(await page.evaluate(()=>S.targets),[20,30,50]);
 // Real result can fail independently, retry, and time out; it must never become a simulated success.
 await page.evaluate(()=>{window.AgeMirrorAdapters={generate:async({image,targetAge})=>{if(targetAge===30)throw new Error('QA service failure');return image;}};});
 await page.locator('#bContinue').click();await page.waitForFunction(()=>!S.generating&&S.tasks.get(30)?.state==='failed');
 assert.equal(await page.evaluate(()=>S.results.size),2);await page.evaluate(()=>{window.AgeMirrorAdapters.generate=async({image})=>image;});await page.getByRole('button',{name:'重试',exact:true}).click();
 await page.waitForFunction(()=>S.tasks.get(30).state==='done');console.log('PASS multiple faces, selection, relative age, independent failure and retry');
 await page.evaluate(async()=>{showInput();S.targets=[60];CONFIG.timeoutMs=50;window.AgeMirrorAdapters.generate=()=>new Promise(()=>{});await continueFromInput();});
 assert.equal(await page.evaluate(()=>S.tasks.get(60).state),'failed');assert.match(await page.evaluate(()=>S.tasks.get(60).error),/超时/);
 await page.evaluate(()=>{showInput();S.targets=[70];window.AgeMirrorAdapters.generate=()=>new Promise(resolve=>window.qaResolve=resolve);continueFromInput();});await page.waitForFunction(()=>!!window.qaResolve);
 await page.evaluate(()=>{toWaiting();const c=canvas();window.qaResolve(c);});await page.waitForTimeout(120);assert.equal(await page.evaluate(()=>S.results.size),0);console.log('PASS timeout and late result after clear');
 // Track IDs survive ordering changes and a short missed frame, without changing the selected face.
 assert.equal(await page.evaluate(()=>{resetTracking();const a={x:.1,y:.2,w:.2,h:.3},b={x:.65,y:.2,w:.2,h:.3};updateTracks([a,b],100);chooseFace(F.faces[1].id);const chosen=F.selectedId;updateTracks([],250);updateTracks([{...b},{...a}],400);return selectedFace()?.id===chosen&&selectedFace().x>.5;}),true);
 // Simulated camera uses an actual decoded photo and MediaPipe inference, not stubbed landmarks.
 await page.route('**/qa-live.jpg',r=>r.fulfill({path:'tmp/qa_portrait.jpg',contentType:'image/jpeg'}));
 await page.evaluate(async()=>{const img=new Image();img.src='/'+encodeURIComponent('qa-live.jpg');await new Promise((resolve,reject)=>{img.onload=resolve;img.onerror=reject;});window.qaCam=canvas(640,480);const g=qaCam.getContext('2d');g.drawImage(img,0,0,640,480);window.qaCamTimer=setInterval(()=>{g.drawImage(img,0,0,640,480);},80);navigator.mediaDevices.getUserMedia=async()=>qaCam.captureStream(15);});
 await page.locator('#bStartCamera').click();const contour=await page.waitForFunction(()=>S.stream&&F.faces[0]?.oval?.length===36?F.faces[0].oval.length:false,{},{timeout:15000});
 assert.equal(await contour.jsonValue(),36);await page.screenshot({path:'tmp/mobile-live.png',fullPage:true});await page.locator('#bCapture').click();await page.waitForFunction(()=>S.state==='s-input');assert.equal(await page.evaluate(()=>S.stream),null);console.log('PASS simulated live stream, real contours and capture stops phone camera');
 await page.evaluate(()=>{toWaiting();clearInterval(window.qaCamTimer);delete window.AgeMirrorAdapters;});
 await page.evaluate(()=>{window.AgeMirrorAdapters={estimate:()=>new Promise(resolve=>window.qaAgeResolve=resolve)};});
 await page.locator('#file').setInputFiles('tmp/qa_portrait.jpg');await page.waitForFunction(()=>!!window.qaAgeResolve);await page.locator('#currentAgeInput').fill('40');
 await page.evaluate(()=>window.qaAgeResolve({age:30}));await page.waitForTimeout(80);assert.equal(await page.locator('#currentAgeInput').inputValue(),'40');
 await page.evaluate(()=>{let called=0;window.AgeMirrorAdapters={generate:async()=>{called++;return canvas();}};S.targets=[60];continueFromInput();toWaiting();window.qaCalls=()=>called;});await page.waitForTimeout(100);assert.equal(await page.evaluate(()=>qaCalls()),0);
 assert.equal(await page.evaluate(()=>probe.getContext('2d').getImageData(0,0,probe.width,probe.height).data.some((v,i)=>i%4===3&&v)),false);console.log('PASS late estimate preserves manual age; cleared queued job never dispatches; probe pixels erased');
 // No model/network: a user must still be able to draw a region on the photograph.
 const fallback=await browser.newPage({viewport:{width:390,height:844}});await fallback.route('**/assets/**',r=>r.abort());await fallback.route('https://**/*',r=>r.abort());await fallback.goto('http://localhost:65512/');await fallback.locator('#file').setInputFiles('tmp/qa_portrait.jpg');await fallback.waitForFunction(()=>S.manual);
 const box=await fallback.locator('#selectOverlay').boundingBox();await fallback.mouse.move(box.x+box.width*.3,box.y+box.height*.2);await fallback.mouse.down();await fallback.mouse.move(box.x+box.width*.7,box.y+box.height*.65,{steps:10});await fallback.mouse.up();await fallback.locator('#bSelectConfirm').click();assert.equal(await fallback.evaluate(()=>S.state),'s-input');console.log('PASS model unavailable, manual drag and continue');
 assert.deepEqual(errors,[]);await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});
