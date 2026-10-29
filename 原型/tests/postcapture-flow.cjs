const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
 let failures=0;
 const cases=[
  ['single uploaded face goes directly to demo results without age input',async page=>{
   await page.locator('#file').setInputFiles('tmp/qa_portrait.jpg');await page.waitForFunction(()=>S.state==='s-grid');
   assert.equal(await page.evaluate(()=>S.state),'s-grid');assert.equal(await page.evaluate(()=>S.ageInfo.source),'demo');assert.equal(await page.locator('#targetChips').isVisible(),false);
  }],
  ['successful age adapter proceeds to preset results without user confirmation',async page=>{
   await page.evaluate(async()=>{S.original=canvas(640,640);S.original.getContext('2d').fillRect(0,0,640,640);S.staticFaces=[{id:1,x:.2,y:.2,w:.5,h:.5,manual:true}];S.staticSelected=1;window.AgeMirrorAdapters={estimate:async()=>({age:32})};await confirmFace();});
   assert.equal(await page.evaluate(()=>S.state),'s-grid');assert.ok(await page.evaluate(()=>S.results.size)>=4);assert.equal(await page.locator('#bEditAge').isVisible(),false);
   await page.locator('#bInfo').click();assert.equal(await page.locator('#bEditAge').isVisible(),true);
  }]
 ];
 for(const [name,run] of cases){const page=await browser.newPage({viewport:{width:390,height:844}});await page.route('https://**/*',r=>r.abort());await page.goto('http://localhost:65512/');try{await run(page);console.log('PASS',name);}catch(e){failures++;console.log('FAIL',name,e.message);}await page.close();}
 await browser.close();process.exitCode=failures?1:0;
})().catch(e=>{console.error(e);process.exit(1);});
