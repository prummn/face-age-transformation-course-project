const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
 const page=await browser.newPage();let failed=0;
 for(const viewport of [{width:320,height:568},{width:390,height:844},{width:430,height:932},{width:844,height:390},{width:1440,height:900}]){
  await page.setViewportSize(viewport);await page.goto('http://localhost:65512/');await page.waitForTimeout(300);
  try{const rect=await page.locator('#cameraWell').boundingBox();assert.ok(rect.height>=viewport.height*.9,'mirror should fill at least 90% of viewport height');assert.ok(rect.width>=viewport.width*.94,'mirror should span viewport width');assert.ok(Math.abs(rect.x+rect.width/2-viewport.width/2)<=1,'mirror centered');const button=await page.locator('#bStartCamera').boundingBox();assert.ok(button.y>=0&&button.y+button.height<=viewport.height,'camera action on first screen');assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'no horizontal overflow');assert.equal(await page.locator('#experienceMode').isVisible(),false,'settings not on capture screen');console.log('PASS immersive layout',viewport);}catch(e){failed++;console.log('FAIL',viewport,e.message);}
 }
 await browser.close();process.exitCode=failed?1:0;
})().catch(e=>{console.error(e);process.exit(1);});
