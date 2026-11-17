"""Verify the actual HD gallery, source filters, color modes and native downloads."""
import argparse
import json
from pathlib import Path
import struct
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--url',default='http://127.0.0.1:65512/hd-demo.html');a=p.parse_args()
    data=json.loads((a.run/'results/demo.json').read_text());errors=[];checked=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path='/tmp/face-age-chrome/chrome',args=['--no-sandbox','--disable-dev-shm-usage'])
        page=browser.new_page(viewport={'width':1440,'height':1080});page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(a.url);page.wait_for_selector('#viewer:visible');assert page.locator('.person').count()==24
        for row in data['people']:
            page.select_option('#ageFilter',row['ageGroup']);page.select_option('#genderFilter',row['gender'])
            assert page.locator('.person').count()==2
            page.locator(f'.person[data-id="{row["id"]}"]').click()
            page.wait_for_function("document.querySelector('#personTitle').textContent.includes("+json.dumps(row['id'])+")")
            for c in ['same','20','40','60','75']:
                page.locator(f'[data-condition="{c}"]').click()
                for color in ['raw','preserved']:
                    page.locator(f'[data-color="{color}"]').click()
                    page.wait_for_function("[...document.querySelectorAll('.portraits img')].every(i=>i.complete&&i.naturalWidth===1024)")
                    assert page.locator('#downloadResult').get_attribute('href').endswith(f'/{c}_{color}.png')
            checked.append(row['id'])
        page.select_option('#ageFilter','');page.select_option('#genderFilter','');assert page.locator('.person').count()==24
        page.locator('[data-condition="60"]').click();page.locator('[data-color="raw"]').click()
        with page.expect_download() as d:page.locator('#downloadResult').click()
        raw=Path(d.value.path()).read_bytes();assert raw[:8]==b'\x89PNG\r\n\x1a\n' and struct.unpack('>II',raw[16:24])==(1024,1024)
        page.locator('#zoomResult').click();assert page.locator('#zoom').evaluate('(e)=>e.open')
        page.wait_for_function("document.querySelector('#zoomImage').naturalWidth===1024")
        assert page.locator('#zoomImage').evaluate('(e)=>e.getBoundingClientRect().width')==1024
        page.locator('#closeZoom').click();assert not page.locator('#zoom').evaluate('(e)=>e.open')
        page.locator('#play').click();page.wait_for_timeout(3800)
        assert page.locator('[data-condition="75"]').get_attribute('aria-pressed')=='true'
        page.locator('#play').click()
        page.screenshot(path=str(a.run/'browser-desktop.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.locator('#zoomSource').click();assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.locator('#closeZoom').click();page.screenshot(path=str(a.run/'browser-mobile.png'),full_page=True)
        page.goto(urljoin(a.url,'./'));page.locator('#bInfo').click();page.get_by_role('link',name='查看高清人像样例 →').click()
        page.wait_for_selector('#viewer:visible');assert '/hd-demo.html' in page.url
        assert not errors,errors;browser.close()
    report=dict(url=a.url,samples=checked,conditions=5,color_variants=2,native_download=[1024,1024],
                zoom_native_pixels=True,age_gender_filters=True,autoplay=True,mobile_no_overflow=True,main_demo_entry=True,page_errors=errors)
    (a.run/'browser-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
