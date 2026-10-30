const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
 const page=await browser.newPage({viewport:{width:390,height:844}});await page.route('https://**/*',r=>r.abort());await page.goto('http://localhost:65512/');
 try{
  assert.equal(await page.evaluate(()=>typeof processGesture),'function');
  assert.equal(await page.evaluate(()=>{S.state='s-mirror';resetGestures();return processGesture({x:.5,open:true},0)}),'none');
  assert.equal(await page.evaluate(()=>{processGesture({x:.5,open:true},400);return processGesture({x:.5,open:true},800)}),'arm');
  assert.equal(await page.evaluate(()=>S.gestureArmed),true);
  assert.equal(await page.evaluate(()=>{const a=processGesture({x:.5,open:true},1600);const b=processGesture({x:.5,open:true},2200);return [a,b]}).then(x=>x[0]),'none','held palm is latched');
  assert.equal(await page.evaluate(()=>{processGesture({x:.5,open:false},2400);processGesture({x:.5,open:true},2500);return processGesture({x:.5,open:true},3300)}),'arm','closing hand rearms palm gesture');
  assert.equal(await page.evaluate(()=>{resetGestures();S.gestureArmed=false;processGesture({x:.2,open:true},0);processGesture({x:.45,open:true},180);return processGesture({x:.7,open:true},360)}),'none','horizontal motion does not become a page action');
  assert.deepEqual(await page.evaluate(()=>{resetGestures();return [handOpen({landmarks:[...Array(21)].map((_,i)=>({x:.5,y:[8,12,16,20].includes(i)?.1:[6,10,14,18].includes(i)?.4:.5}))}),handOpen(null)]}),[true,false]);
  assert.equal(await page.evaluate(()=>initHandDetector()),true,'local hand model initializes');
  assert.equal(await page.evaluate(()=>HAND.ready),true);
  assert.equal(await page.evaluate(()=>typeof drawHandOverlay),'function');
  await page.evaluate(()=>{S.state='s-grid';S.pages=[{t:null,image:canvas()}];resetGestures();});
  assert.equal(await page.evaluate(()=>{processGesture({x:.1,open:true},0);processGesture({x:.4,open:true},120);return processGesture({x:.7,open:true},260)}),'none');
  console.log('PASS pre-capture palm arming, rearm, no page gesture, hand model and overlay API');
 }catch(e){console.error(e);process.exitCode=1;}
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});
