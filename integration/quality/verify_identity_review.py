"""Verify completed identity-pilot output and the static comparison page on CPU."""
import argparse
import json
from pathlib import Path
import struct
from playwright.sync_api import sync_playwright


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--url',default='http://127.0.0.1:65512/identity-review/')
    args=parser.parse_args();run=args.run
    report=json.loads((run/'evaluation/report.json').read_text())
    errors=[];checks=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/tmp/face-age-chrome/chrome',args=['--no-sandbox','--disable-dev-shm-usage'])
        page=browser.new_page(viewport={'width':1440,'height':1000})
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(args.url)
        page.wait_for_function("document.querySelectorAll('#images figure').length===4")
        assert page.locator('#sample option').count()==104
        names=['uploaded']+[next(r['name'] for r in report['records'] if r['cohort']==cohort) for cohort in ['appa_asian_test','ffhq_regression']]
        for name in names:
            page.select_option('#sample',name)
            for target in ['same','20','40','60','75']:
                page.select_option('#age',target)
                page.wait_for_function("[...document.querySelectorAll('#images img')].every(i=>i.complete&&i.naturalWidth>0)")
                assert page.locator('#images figure').count()==4
            checks.append(dict(name=name,all_ages_displayed=True))
        page.select_option('#sample','uploaded');page.select_option('#age','75')
        page.wait_for_function("[...document.querySelectorAll('#images img')].every(i=>i.complete&&i.naturalWidth>0)")
        link=page.locator('#images figure').nth(3).locator('a').get_attribute('href')
        with page.expect_download() as download:
            page.locator('#images figure').nth(3).locator('a').click()
        path=Path(download.value.path());data=path.read_bytes()
        assert data[:8]==b'\x89PNG\r\n\x1a\n' and struct.unpack('>II',data[16:24])==(1024,1024)
        page.screenshot(path=str(run/'browser-desktop.png'),full_page=True)
        assert page.locator('#metrics tr').count()==9
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        assert not errors,errors
        browser.close()
    (run/'browser-validation.json').write_text(json.dumps(dict(url=args.url,samples=104,checks=checks,
        download=link,download_size=[1024,1024],summary_rows=9,mobile_no_overflow=True,page_errors=errors),ensure_ascii=False,indent=2)+'\n')
    print('IDENTITY_REVIEW_BROWSER_VERIFIED',checks)


if __name__=='__main__':
    main()
