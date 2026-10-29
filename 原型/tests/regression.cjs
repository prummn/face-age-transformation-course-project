const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const {pathToFileURL}=require('node:url');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH ? {executablePath:process.env.CHROME_PATH} : {})});
 const page=await browser.newPage({viewport:{width:390,height:844}});
 await page.route('https://**/*',r=>r.abort());
 await page.goto(pathToFileURL(process.cwd()+'/原型/人脸年龄变换原型_v5.html').href);
 const cases=[
  ['empty target must not change selected ages',async()=>{const r=await page.evaluate(()=>{S.targets=[20,40];document.querySelector('#targetAgeInput').value='';addTargetFromInput();return S.targets;});assert.deepEqual(r,[20,40]);}],
  ['maximum eight selected ages',async()=>{const r=await page.evaluate(()=>{S.targets=[10,20,30,40,50,60,70,80];document.querySelector('#targetAgeInput').value='85';addTargetFromInput();return S.targets.length;});assert.equal(r,8);}],
  ['clear must erase pixels as well as references',async()=>{const a=await page.evaluate(()=>{const c=document.querySelector('#previewCv');c.getContext('2d').fillRect(0,0,40,40);toWaiting();return c.getContext('2d').getImageData(0,0,1,1).data[3];});assert.equal(a,0);}],
 ];
 let failures=0;for(const [name,run] of cases){try{await run();console.log('PASS',name);}catch(e){failures++;console.log('FAIL',name,e.message);}}
 await browser.close();process.exitCode=failures?1:0;
})();
