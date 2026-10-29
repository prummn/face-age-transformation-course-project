const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH ? {executablePath:process.env.CHROME_PATH} : {})});
 const page=await browser.newPage({viewport:{width:390,height:844},isMobile:true,hasTouch:true,acceptDownloads:true});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));page.on('console',msg=>{if(msg.type()==='error')console.log('CONSOLE',msg.text().slice(0,180));});
 await page.route('https://**/*',r=>r.abort());
 await page.goto('http://localhost:65512/');await page.waitForTimeout(250);
 await page.screenshot({path:'tmp/mobile-capture.png',fullPage:true});
 await page.locator('#file').setInputFiles('tmp/qa_portrait.jpg');
 await page.waitForFunction(()=>DET.ready&&S.staticFaces.length>0,{},{timeout:60000});
 console.log('DETECTION',await page.evaluate(()=>({kind:DET.kind,faces:S.staticFaces.length,points:S.staticFaces[0].oval.length})));
 assert.equal(await page.evaluate(()=>DET.kind),'mesh');
 await page.screenshot({path:'tmp/mobile-selection.png',fullPage:true});
 await page.locator('#bSelectConfirm').click();
 await page.locator('#currentAgeInput').fill('25');
 await page.locator('#bContinue').click();
 await page.waitForFunction(()=>S.state==='s-grid');
 assert.equal(await page.locator('.age-page').count(),6);
 const touch=await page.context().newCDPSession(page);const bounds=await page.locator('#pageTrack').boundingBox();const y=Math.min(bounds.y+120,700);
 await touch.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:320,y}]});
 for(let x=300;x>=65;x-=25){await touch.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x,y}]});await page.waitForTimeout(20);}
 await touch.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});await page.waitForTimeout(600);assert.ok(await page.evaluate(()=>S.pageIndex)>0,'touch swipe changes page');
 await page.evaluate(()=>goPage(0));await page.waitForTimeout(600);
 await page.screenshot({path:'tmp/mobile-pages.png',fullPage:true});
 await page.locator('#pageNext').click();await page.waitForTimeout(650);
 assert.equal(await page.evaluate(()=>S.pageIndex),1);
 await page.locator('.age-page').nth(1).locator('.page-photo').click();
 await page.locator('#bPasteView').click();
 await page.locator('#split').fill('70');
 await page.screenshot({path:'tmp/mobile-compare.png',fullPage:true});
 const dl=page.waitForEvent('download');await page.locator('#bSave').click();const d=await dl;await d.saveAs('tmp/export-compare.png');assert.ok(fs.statSync('tmp/export-compare.png').size>10000);
 await page.locator('#bBack').click();await page.locator('#bOverview').click();
 assert.equal(await page.locator('.overview-card').count(),6);
 await page.screenshot({path:'tmp/mobile-overview.png',fullPage:true});
 for(const viewport of [{width:320,height:568},{width:844,height:390},{width:1280,height:900}]){
  await page.setViewportSize(viewport);assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'overflow '+JSON.stringify(viewport));
 }
 await page.setViewportSize({width:390,height:844});
 await page.locator('#bInfo').click();await page.locator('#bReset').click();await page.locator('#bClearConfirm').click();
 assert.equal(await page.evaluate(()=>S.original),null);assert.equal(await page.evaluate(()=>comparisonImages),null);
 assert.equal(await page.evaluate(()=>[...document.querySelectorAll('canvas')].some(c=>c.width&&c.height&&c.getContext('2d').getImageData(0,0,c.width,c.height).data.some((v,i)=>i%4===3&&v))),false);
 assert.deepEqual(errors,[]);
 console.log('PASS flow: upload, real contour, age pages, navigation, pasteback, PNG, overview, responsive, clear');
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});
