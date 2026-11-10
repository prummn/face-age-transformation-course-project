#!/usr/bin/env node
'use strict';
/** Repeatable classroom integration acceptance. Real and simulated cases are separated.
 * Run from any directory after integration/launch.py reports ready:
 *   node integration/tests/integration-flow.cjs
 * Optional: BASE_URL, PLAYWRIGHT_MODULE, CHROME_PATH, QA_VIDEO=0, QA_CASE=pattern.
 * Public fixture: https://storage.googleapis.com/mediapipe-assets/portrait.jpg
 * Screenshots, downloaded pictures, video, and JSON stay in ignored tmp/integration-qa.
 * UI actions use Playwright clicks and file inputs; evaluate only observes state/pixels
 * or makes a synthetic two-face fixture, without calling application functions.
 */
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const ROOT = path.resolve(__dirname, '../..');
const BASE = process.env.BASE_URL || 'http://localhost:65512';
const SOURCE = 'https://storage.googleapis.com/mediapipe-assets/portrait.jpg';
const FIXTURE = path.join(ROOT, 'tmp/qa_portrait.jpg');
const RUN = new Date().toISOString().replace(/[:.]/g, '-');
const OUTPUT = path.join(ROOT, 'tmp/integration-qa', RUN);
const report = {startedAt:new Date().toISOString(), baseUrl:BASE, output:OUTPUT,
  scope:'Real browser→same-origin gateway→Ascend API; route-simulated error/race cases separately labelled.',
  exclusions:['Android/RK3288 hardware acceptance','Age accuracy or identity-quality validation','Production P95 and load validation'], cases:[]};
const replay = {prepare:null, transform:null};
let browser;
const hash = value => crypto.createHash('sha256').update(value).digest('hex');
const pngSize = buffer => {
  assert.equal(buffer.subarray(0,8).toString('hex'),'89504e470d0a1a0a','Downloaded file is PNG');
  return {width:buffer.readUInt32BE(16),height:buffer.readUInt32BE(20)};
};
const dataBytes = value => Buffer.from(value.split(',')[1],'base64');
async function bounded(promise,ms,label){let timer;try{return await Promise.race([promise,new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error(label+' timed out')),ms);})]);}finally{clearTimeout(timer);}}
const safeReply = value => {
  if (!value || typeof value !== 'object') return value;
  const result={...value};
  for(const key of ['image','raw_image','aligned_image','native_image','reference_image']) if(result[key]) {
    const bytes=dataBytes(result[key]);result[key]={bytes:bytes.length,sha256:hash(bytes),...pngSize(bytes)};
  }
  if(result.session_id) result.session_id=`sha256:${hash(result.session_id).slice(0,16)}`;
  return result;
};
const responseAt = (response, endpoint) => new URL(response.url()).pathname === endpoint;
async function snapshot(page) {
  return page.evaluate(() => ({view:document.body.dataset.view,backend:document.body.dataset.backend,
    state:typeof S !== 'undefined'?S.state:null,age:typeof S !== 'undefined'?S.age:null,
    ageInfo:typeof S !== 'undefined'?S.ageInfo:null,targets:typeof S !== 'undefined'?[...S.targets]:[],
    results:typeof S !== 'undefined'?[...S.results].map(([age,image])=>({age,width:image.width,height:image.height})):[],
    tasks:typeof S !== 'undefined'?[...S.tasks].map(([age,task])=>({age,...task})):[],
    original:typeof S !== 'undefined'&&S.original?{width:S.original.width,height:S.original.height}:null,
    sessionPresent:!!window.AgeMirrorAPI?.state.session,ready:window.AgeMirrorAPI?.state.ready,
    disclosure:document.querySelector('#resultDisclosure')?.textContent,
    selectedFace:typeof S !== 'undefined'?S.staticSelected:null,
    faces:typeof S !== 'undefined'?S.staticFaces.map(f=>({id:f.id,x:f.x,y:f.y,w:f.w,h:f.h})):[]}));
}
async function capture(page,name) {
  const filename=path.join(OUTPUT,`${name}.png`);await page.screenshot({path:filename,fullPage:true});return filename;
}
async function ready(page) {
  await page.goto(BASE+'/?mode=live',{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>window.AgeMirrorAPI?.state.ready===true,{},{timeout:30000});
  assert.equal(await page.locator('body').getAttribute('data-backend'),'live');
}
async function pause(page) {
  if(await page.locator('#bPlay').getAttribute('aria-pressed')==='true')await page.locator('#bPlay').click();
}
async function download(page,selector,name) {
  const event=page.waitForEvent('download',{timeout:15000});await page.locator(selector).click();
  const downloaded=await event;const filename=path.join(OUTPUT,name);await downloaded.saveAs(filename);
  const bytes=await fs.readFile(filename);return {file:filename,suggestedFilename:downloaded.suggestedFilename(),bytes:bytes.length,sha256:hash(bytes),...pngSize(bytes)};
}
async function clearUI(page) {
  await page.locator('#bInfo').click();await page.locator('#bReset').click();
  if(await page.locator('#clearDialog').isVisible())await page.locator('#bClearConfirm').click();
  await page.locator('#s-mirror').waitFor({state:'visible'});
}
async function mockSession(page) {
  assert.ok(replay.prepare&&replay.transform,'Real successful flow is required before replay-based cases');
  await page.route('**/api/prepare',route=>route.fulfill({status:200,json:{...replay.prepare,session_id:'qa_simulated_session_0123456789'}}));
  await page.route('**/api/sessions/**',route=>route.fulfill({status:200,json:{deleted:true,qa_simulated:true}}));
}
async function runCase(name,mode,execute) {
  if(process.env.QA_CASE&&!name.includes(process.env.QA_CASE))return;
  const item={name,mode,startedAt:new Date().toISOString(),status:'running'};report.cases.push(item);
  const context=await browser.newContext({viewport:{width:1080,height:1920},deviceScaleFactor:1,
    acceptDownloads:true,reducedMotion:'reduce',...(process.env.QA_VIDEO==='0'?{}:{recordVideo:{dir:path.join(OUTPUT,'videos'),size:{width:540,height:960}}})});
  const page=await context.newPage();page.setDefaultTimeout(20000);
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  const started=Date.now();
  try {Object.assign(item,await execute(page,context));assert.deepEqual(errors,[],'No uncaught page errors');item.status='passed';}
  catch(error){item.status='failed';item.error=error.stack||error.message;try{item.state=await snapshot(page);item.failureScreenshot=await capture(page,name+'-failed');}catch(_){};}
  finally {
    item.durationMs=Date.now()-started;item.pageErrors=errors;
    // Cleanup only a session created in this test context; never enumerate/delete other sessions.
    try{const sid=await page.evaluate(()=>window.AgeMirrorAPI?.state.session);if(sid&&!sid.startsWith('qa_simulated'))await context.request.delete(BASE+'/api/sessions/'+encodeURIComponent(sid));}catch(_){}
    const video=page.video();await context.close();if(video)item.video=await video.path();
    await fs.writeFile(path.join(OUTPUT,'report.json'),JSON.stringify(report,null,2));
    console.log(`${item.status.toUpperCase()} [${mode}] ${name} (${item.durationMs} ms)`);
  }
}
async function buildTwoFaceFixture(context) {
  const scratch=await context.newPage();const data='data:image/jpeg;base64,'+(await fs.readFile(FIXTURE)).toString('base64');
  const fixture=await scratch.evaluate(async source=>{
    const img=new Image();img.src=source;await img.decode();
    const c=document.createElement('canvas');c.width=1600;c.height=1100;const g=c.getContext('2d');
    g.fillStyle='#e8e8e8';g.fillRect(0,0,c.width,c.height);
    // Near-face crop keeps both detections in the detector's useful scale range.
    // Original full-portrait composite missed the smaller right instance; retained
    // as a limitation in report rather than interpreting fixture repair as a fix.
    const sx=img.width*.2,sy=0,sw=img.width*.6,sh=img.height*.63;
    const w1=740,h1=w1*sh/sw,w2=555,h2=w2*sh/sw;
    g.drawImage(img,sx,sy,sw,sh,35,30,w1,h1);g.save();g.translate(1530,160);g.scale(-1,1);g.drawImage(img,sx,sy,sw,sh,0,0,w2,h2);g.restore();
    return {data:c.toDataURL('image/png'),width:c.width,height:c.height,originalSize:{width:img.width,height:img.height}};
  },data);await scratch.close();const filename=path.join(OUTPUT,'two-faces-synthetic.png');await fs.writeFile(filename,dataBytes(fixture.data));
  return {file:filename,width:fixture.width,height:fixture.height,
    note:'Synthetic near-face duplicate of one public portrait: larger left instance and smaller mirrored right instance. Tests spatial selection, not identity recognition or different-person model quality.',
    knownBoundary:'Earlier full-portrait 1600x1100 fixture (left height 1040, right height 760) detected only the larger left face and automatically generated it. Original evidence: tmp/integration-qa/2026-11-10T05-53-33-154Z/report.json case 02. Near cropping changes test input; it does not fix that detection limitation.'};
}
(async()=>{
  await fs.mkdir(OUTPUT,{recursive:true});
  try{await fs.access(FIXTURE);}catch(_){const response=await fetch(SOURCE);assert.ok(response.ok,'Public fixture download');await fs.mkdir(path.dirname(FIXTURE),{recursive:true});await fs.writeFile(FIXTURE,Buffer.from(await response.arrayBuffer()));}
  report.fixture={file:FIXTURE,source:SOURCE,sha256:hash(await fs.readFile(FIXTURE))};
  browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});

  await runCase('01-real-upload-four-targets-export-clear','REAL',async(page,context)=>{
    await ready(page);const health=await context.request.get(BASE+'/api/health');assert.equal(health.status(),200);const healthInfo=await health.json();assert.equal(healthInfo.ready,true);assert.equal(healthInfo.native_size,1024);
    const transforms=[];const tasks=[];page.on('response',response=>{if(responseAt(response,'/api/transform'))tasks.push(response.json().then(body=>{transforms.push({status:response.status(),body});}).catch(()=>{}));});
    const preparation=page.waitForResponse(response=>responseAt(response,'/api/prepare'));
    await page.locator('#file').setInputFiles(FIXTURE);const preparedResponse=await preparation;assert.equal(preparedResponse.status(),200);const prepared=await preparedResponse.json();replay.prepare=prepared;
    assert.equal(prepared.age.source,'model');assert.ok(prepared.age.age>=1&&prepared.age.age<=99);assert.equal(prepared.alignment,'sam_ffhq');
    await page.locator('#s-grid').waitFor({state:'visible',timeout:180000});await Promise.all(tasks);await pause(page);
    const state=await snapshot(page);assert.equal(state.ageInfo.source,'model');assert.equal(state.results.length,4);assert.equal(state.tasks.filter(t=>t.state==='done').length,4);
    assert.equal(transforms.length,4);for(const result of transforms){assert.equal(result.status,200);assert.equal(result.body.meta.output_kind,'native');assert.deepEqual(pngSize(dataBytes(result.body.image)),{width:1024,height:1024});}
    const unique=new Set(transforms.map(result=>hash(dataBytes(result.body.image))));assert.equal(unique.size,4,'Four ages must not duplicate a single image');
    for(const result of state.results)assert.deepEqual({width:result.width,height:result.height},{width:1024,height:1024});
    replay.transform=transforms[0].body;const resultScreenshot=await capture(page,'01-real-four-targets');
    await page.locator('#pageDots button').nth(1).click();const native=await download(page,'#bNativeSave','native1024.png');assert.deepEqual({width:native.width,height:native.height},{width:1024,height:1024});
    await page.getByRole('button',{name:`比较 ${state.targets[0]} 岁`,exact:true}).click();await page.locator('#s-detail').waitFor({state:'visible'});
    const crop=await page.locator('#cmpA').evaluate(c=>({width:c.width,height:c.height}));assert.deepEqual(crop,{width:1024,height:1024});
    await page.locator('#bPasteView').click();const full=await page.locator('#cmpA').evaluate(c=>({width:c.width,height:c.height}));assert.deepEqual(full,state.original);
    await page.locator('#split').focus();await page.keyboard.press('Home');await page.keyboard.press('ArrowRight');assert.equal(await page.locator('#split').inputValue(),'1');
    const comparison=await download(page,'#bSave','whole-image-comparison.png');assert.equal(comparison.width,state.original.width);assert.ok(comparison.height>state.original.height);
    const compareScreenshot=await capture(page,'01-real-whole-image-comparison');await page.locator('#bBack').click();
    await page.locator('#bInfo').click();await page.locator('#bOverview').click();await page.locator('#s-overview').waitFor({state:'visible'});assert.equal(await page.locator('#overviewGrid .overview-card').count(),5);
    const overview=await download(page,'#bOverviewSave','four-age-overview.png');const overviewScreenshot=await capture(page,'01-real-overview');
    const deletion=page.waitForResponse(response=>response.request().method()==='DELETE'&&new URL(response.url()).pathname.endsWith(prepared.session_id));
    await clearUI(page);assert.equal((await deletion).status(),200);
    const cleared=await snapshot(page);assert.equal(cleared.results.length,0);assert.equal(cleared.sessionPresent,false);assert.equal(cleared.original,null);
    const expired=await context.request.post(BASE+'/api/transform',{data:{session_id:prepared.session_id,target_age:40,request_id:'qa-after-clear'}});assert.equal(expired.status(),410);
    return {health:healthInfo,preparation:safeReply(prepared),state,transforms:transforms.map(r=>({status:r.status,body:safeReply(r.body)})),downloads:[native,comparison,overview],screenshots:[resultScreenshot,compareScreenshot,overviewScreenshot],clear:{state:cleared,postClearStatus:expired.status()}};
  });

  await runCase('02-real-smaller-second-face-selection','REAL',async(page,context)=>{
    const fixture=await buildTwoFaceFixture(context);await ready(page);await page.locator('#file').setInputFiles(fixture.file);
    await page.waitForFunction(()=>document.querySelectorAll('#staticFaceChoices button').length>=2,{},{timeout:45000});
    const detected=await snapshot(page);assert.ok(detected.faces.length>=2,'Both synthetic face instances must be detected');
    const right=detected.faces.reduce((a,b)=>(a.x+a.w/2)>(b.x+b.w/2)?a:b);const left=detected.faces.reduce((a,b)=>(a.x+a.w/2)<(b.x+b.w/2)?a:b);
    assert.ok(right.w*right.h<left.w*left.h,'Right instance must be smaller, detecting largest-face fallback');assert.ok(right.x>0.5);
    const index=detected.faces.findIndex(face=>face.id===right.id);await page.locator('#staticFaceChoices button').nth(index).click();
    const selectScreenshot=await capture(page,'02-selected-smaller-right-face');const preparation=page.waitForResponse(response=>responseAt(response,'/api/prepare'));
    await page.locator('#bSelectConfirm').click();const response=await preparation;assert.equal(response.status(),200);const prepared=await response.json();
    const request=response.request().postDataJSON();assert.equal(request.face.id,right.id);assert.equal(prepared.meta.face_id,right.id);
    const [x,y,w,h]=prepared.meta.detected_bbox;assert.ok((x+w/2)/prepared.transform.sourceWidth>0.5,'Backend must align selected right face, not largest left face');
    assert.ok(Math.abs((x+w/2)/prepared.transform.sourceWidth-(right.x+right.w/2))<right.w,'Backend chosen center overlaps UI selected face');
    await page.locator('#s-grid').waitFor({state:'visible',timeout:180000});await pause(page);const completed=await snapshot(page);assert.equal(completed.results.length,4);
    const resultScreenshot=await capture(page,'02-real-smaller-face-results');await clearUI(page);
    return {fixture,selected:right,other:left,requestFace:request.face,preparation:safeReply(prepared),state:completed,screenshots:[selectScreenshot,resultScreenshot]};
  });

  await runCase('03-prepare-outage-never-falls-back-to-demo','SIMULATED_FAILURE',async page=>{
    await ready(page);let transforms=0;page.on('request',request=>{if(new URL(request.url()).pathname==='/api/transform')transforms++;});
    await page.route('**/api/prepare',route=>route.fulfill({status:503,json:{detail:'QA 模拟：生成服务暂未连接，请重新连接。'}}));
    await page.locator('#file').setInputFiles(FIXTURE);await page.waitForFunction(()=>document.querySelector('#selectNotice')?.classList.contains('error')&&document.querySelector('#selectNotice').textContent.includes('QA 模拟'),{},{timeout:45000});
    const state=await snapshot(page);assert.equal(state.state,'s-select');assert.equal(state.results.length,0);assert.equal(transforms,0);assert.notEqual(state.ageInfo?.source,'demo');
    return {injection:'Only /api/prepare returns artificial HTTP 503. No model-quality conclusion.',state,notice:await page.locator('#selectNotice').textContent(),screenshot:await capture(page,'03-simulated-prepare-outage')};
  });

  await runCase('04-one-target-fails-preserve-successful-results','SIMULATED_FAILURE',async page=>{
    await ready(page);await mockSession(page);let requests=0;
    await page.route('**/api/transform',route=>{const request=route.request().postDataJSON();requests++;return route.fulfill(requests===2?{status:503,json:{detail:'QA 模拟：此目标生成失败。'}}:{status:200,json:{...replay.transform,request_id:request.request_id,meta:{...replay.transform.meta,target_age:request.target_age}}});});
    await page.locator('#file').setInputFiles(FIXTURE);await page.waitForFunction(()=>typeof S!=='undefined'&&!S.generating&&[...S.tasks.values()].some(t=>t.state==='failed')&&S.tasks.size===4,{},{timeout:45000});
    const state=await snapshot(page);assert.equal(state.results.length,3);assert.equal(state.tasks.filter(t=>t.state==='failed').length,1);assert.equal(state.ageInfo.source,'model');assert.equal(requests,4);
    await page.locator('#bShowResults').click();await pause(page);assert.equal(await page.locator('#pageTrack .age-page').count(),4);
    return {injection:'Replay of prior real response; second target artificially returns HTTP 503. Repeated replay images are not quality evidence.',state,screenshot:await capture(page,'04-simulated-partial-failure')};
  });

  await runCase('05-cancel-before-prepare-reply-discards-late-session','SIMULATED_RACE',async page=>{
    assert.ok(replay.prepare);await ready(page);let release,entered;const hold=new Promise(resolve=>release=resolve);const requested=new Promise(resolve=>entered=resolve);let deletes=0;
    await page.route('**/api/sessions/**',route=>{deletes++;return route.fulfill({status:200,json:{deleted:true,qa_simulated:true}});});
    await page.route('**/api/prepare',async route=>{entered();await hold;await route.fulfill({status:200,json:{...replay.prepare,session_id:'qa_simulated_late_prepare_012345'}}).catch(()=>{});});
    let before;try{await page.locator('#file').setInputFiles(FIXTURE);await bounded(requested,45000,'Prepare request');await page.locator('#bCancelGenerate').click();before=await snapshot(page);}finally{release();}
    await page.waitForTimeout(1200);const after=await snapshot(page);assert.equal(after.results.length,0);assert.equal(after.sessionPresent,false);assert.notEqual(after.state,'s-grid');assert.notEqual(after.ageInfo?.source,'demo');assert.equal(deletes,1,'Late server-side preparation session must receive DELETE exactly once');
    return {injection:'Artificially delayed prepare reply. User cancels via visible button before response; no stale generated result may appear.',before,after,deletes,screenshot:await capture(page,'05-simulated-late-prepare-cancel')};
  });

  await runCase('06-clear-during-transform-discards-late-result','SIMULATED_RACE',async page=>{
    await ready(page);await mockSession(page);let release,entered;const hold=new Promise(resolve=>release=resolve);const requested=new Promise(resolve=>entered=resolve);
    await page.route('**/api/transform',async route=>{const request=route.request().postDataJSON();entered();await hold;await route.fulfill({status:200,json:{...replay.transform,request_id:request.request_id,meta:{...replay.transform.meta,target_age:request.target_age}}}).catch(()=>{});});
    try{await page.locator('#file').setInputFiles(FIXTURE);await bounded(requested,45000,'Transform request');await clearUI(page);}finally{release();}await page.waitForTimeout(1200);
    const state=await snapshot(page);assert.equal(state.state,'s-mirror');assert.equal(state.results.length,0);assert.equal(state.sessionPresent,false);assert.equal(state.original,null);
    return {injection:'Artificially delayed transform reply replayed after user clears. Tests abort/stale-display protection, not backend in-flight NPU cancellation.',state,screenshot:await capture(page,'06-simulated-clear-late-result')};
  });

  await runCase('07-show-partial-results-then-include-late-targets','SIMULATED_PROGRESS',async page=>{
    await ready(page);await mockSession(page);let release,entered,requests=0;const hold=new Promise(resolve=>release=resolve);const secondStarted=new Promise(resolve=>entered=resolve);
    await page.route('**/api/transform',async route=>{const request=route.request().postDataJSON();requests++;if(requests===2){entered();await hold;}await route.fulfill({status:200,json:{...replay.transform,request_id:request.request_id,meta:{...replay.transform.meta,target_age:request.target_age}}}).catch(()=>{});});
    let partial;try{await page.locator('#file').setInputFiles(FIXTURE);await bounded(secondStarted,45000,'Second target');await page.locator('#bShowResults').click();await pause(page);partial=await snapshot(page);assert.equal(partial.results.length,1);assert.equal(await page.locator('#pageTrack .age-page').count(),2);}finally{release();}
    await page.waitForFunction(()=>typeof S!=='undefined'&&!S.generating&&S.results.size===4&&document.querySelectorAll('#pageTrack .age-page').length===5,{},{timeout:15000});
    const complete=await snapshot(page);assert.equal(complete.results.length,4);assert.equal(complete.tasks.filter(t=>t.state==='done').length,4);
    return {injection:'Real response replayed with second target held until the user clicks 查看已完成. Tests UI progress updates only; repeated images are not quality evidence.',partial,complete,screenshot:await capture(page,'07-simulated-partial-results-updated')};
  });

  report.finishedAt=new Date().toISOString();report.summary={passed:report.cases.filter(c=>c.status==='passed').length,failed:report.cases.filter(c=>c.status==='failed').length};
  await fs.writeFile(path.join(OUTPUT,'report.json'),JSON.stringify(report,null,2));
  await fs.writeFile(path.join(ROOT,'tmp/integration-qa/latest.json'),JSON.stringify({run:RUN,report:path.join(OUTPUT,'report.json'),...report.summary},null,2));
  console.log(JSON.stringify({report:path.join(OUTPUT,'report.json'),...report.summary}));
  await browser.close();process.exitCode=report.summary.failed?1:0;
})().catch(async error=>{report.fatal=error.stack||error.message;await fs.mkdir(OUTPUT,{recursive:true});await fs.writeFile(path.join(OUTPUT,'report.json'),JSON.stringify(report,null,2));console.error(error);if(browser)await browser.close();process.exitCode=1;});
