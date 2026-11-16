"""Browser checks of real completed artifacts, using the existing loopback gateway."""
import argparse,json,struct
from pathlib import Path
from playwright.sync_api import sync_playwright

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--url',default='http://127.0.0.1:65512/training-review/');a=p.parse_args()
    report=json.loads((a.run/'evaluation/report.json').read_text());expected={cohort:sum(r['cohort']==cohort for r in report['records']) for cohort in {r['cohort'] for r in report['records']}}
    errors=[];checks=[]
    with sync_playwright() as w:
        browser=w.chromium.launch(executable_path='/tmp/face-age-chrome/chrome',args=['--no-sandbox','--disable-dev-shm-usage'])
        page=browser.new_page(viewport={'width':1440,'height':1000})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(a.url);page.wait_for_function("document.querySelectorAll('#pictures figure').length===5")
        for cohort,count in expected.items():
            page.select_option('#cohort',cohort);assert page.locator('#sample option').count()==count
            page.select_option('#age','75');page.wait_for_function("[...document.querySelectorAll('#pictures img')].every(i=>i.complete&&i.naturalWidth>0)")
            assert page.locator('#pictures figure').count()==5
            checks.append(dict(cohort=cohort,samples=count,images_loaded=True))
        page.select_option('#cohort','diagnostic');page.select_option('#sample','uploaded');page.select_option('#age','same')
        page.wait_for_function("[...document.querySelectorAll('#pictures img')].every(i=>i.complete&&i.naturalWidth>0)")
        page.select_option('#candidate','encoder');page.locator('#split').evaluate("el=>{el.value='25';el.dispatchEvent(new Event('input',{bubbles:true}))}")
        assert '75%' in page.locator('#overlay').get_attribute('style')
        link=page.locator('#pictures figure').nth(4).locator('a').get_attribute('href')
        response=page.request.get(a.url+link);assert response.ok
        raw=response.body();assert raw[:8]==b'\x89PNG\r\n\x1a\n' and struct.unpack('>II',raw[16:24])==(1024,1024)
        page.select_option('#crop','face');assert 'crop' in page.locator('#pictures').get_attribute('class')
        page.select_option('#crop','full');page.screenshot(path=str(a.run/'browser-desktop.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        assert not errors,errors
        browser.close()
    (a.run/'browser-validation.json').write_text(json.dumps(dict(url=a.url,checks=checks,raw_download_size=[1024,1024],slider=True,crop=True,mobile_no_overflow=True,page_errors=errors),ensure_ascii=False,indent=2)+'\n')
    print('BROWSER_REVIEW_VERIFIED',checks)
if __name__=='__main__':main()
