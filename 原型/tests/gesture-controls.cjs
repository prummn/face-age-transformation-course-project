const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
 const page=await browser.newPage({viewport:{width:390,height:844}});await page.route('https://**/*',r=>r.abort());await page.goto('http://localhost:65512/');
 const openHand=(x)=>({x,open:true});const closedHand=(x)=>({x,open:false});
 try{
  assert.equal(await page.evaluate(()=>typeof processGesture), 'function');
  assert.equal(await page.evaluate(()=>{resetGestures();return processGesture({x:.15,open:true},0)}),'none');
  assert.equal(await page.evaluate(()=>{processGesture({x:.22,open:true},120);processGesture({x:.36,open:true},260);return processGesture({x:.52,open:true},480)}),'next');
  assert.equal(await page.evaluate(()=>{resetGestures();const out=[processGesture({x:.8,open:true},0),processGesture({x:.62,open:true},120),processGesture({x:.44,open:true},260),processGesture({x:.25,open:true},480)];return out.includes('prev')}),true);
  assert.equal(await page.evaluate(()=>{resetGestures();processGesture({x:.5,open:true},0);processGesture({x:.5,open:true},400);return processGesture({x:.5,open:true},800)}),'toggle');
  assert.equal(await page.evaluate(()=>{const a=processGesture({x:.5,open:true},1600);const b=processGesture({x:.5,open:true},2200);return [a,b]}).then(x=>x[0]),'none','same held palm is latched');
  assert.equal(await page.evaluate(()=>{processGesture({x:.5,open:false},2400);processGesture({x:.5,open:true},2500);return processGesture({x:.5,open:true},3300)}),'toggle','closing hand rearms palm gesture');
  assert.equal(await page.evaluate(()=>{resetGestures();const out=[processGesture({x:.2,open:true},0),processGesture({x:.45,open:true},180),processGesture({x:.7,open:true},360),processGesture({x:.8,open:true},540)];return out.includes('next')}),true,'wave threshold ignores a slow short drift');
  assert.deepEqual(await page.evaluate(()=>{resetGestures();return [handOpen({landmarks:[...Array(21)].map((_,i)=>({x:.5,y:[8,12,16,20].includes(i)?.1:[6,10,14,18].includes(i)?.4:.5}))}),handOpen(null)]}),[true,false]);
  assert.equal(await page.evaluate(()=>initHandDetector()),true,'local hand model initializes');
  assert.equal(await page.evaluate(()=>HAND.ready),true);
  await page.evaluate(()=>{toWaiting();S.face=canvas();S.age=30;S.ageInfo={age:30,source:'demo'};S.targets=[15,40];for(const t of S.targets)S.results.set(t,canvas());buildPages();stopPlay();goPage(0);});
  assert.equal(await page.evaluate(()=>{resetGestures();processGesture({x:.1,open:true},0);processGesture({x:.3,open:true},120);return processGesture({x:.5,open:true},260)}),'next');
  assert.equal(await page.evaluate(()=>S.pageIndex),1);
  await page.evaluate(()=>startPlay());assert.equal(await page.evaluate(()=>S.autoTimer>0),true);
  assert.equal(await page.evaluate(()=>{resetGestures();processGesture({x:.5,open:true},1000);processGesture({x:.5,open:true},1400);return processGesture({x:.5,open:true},1800)}),'toggle');
  assert.equal(await page.evaluate(()=>S.autoTimer),0);
  console.log('PASS gesture recognition state machine, cooldown, rearm and palm classifier');
 }catch(e){console.error(e);process.exitCode=1;}
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});
