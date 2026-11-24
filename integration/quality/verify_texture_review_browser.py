import argparse,json
from pathlib import Path
import subprocess
from playwright.sync_api import sync_playwright
p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('run',type=Path);p.add_argument('--texture',action='store_true');a=p.parse_args()
with sync_playwright() as pw:
    browser=pw.chromium.launch(executable_path='/tmp/face-age-chrome/chrome',args=['--no-sandbox','--disable-dev-shm-usage'])
    page=browser.new_page(viewport={'width':1400,'height':950},accept_downloads=True);errors=[]
    page.on('pageerror',lambda err:errors.append(str(err)))
    response=page.goto(a.url,wait_until='domcontentloaded');assert response.status==200
    page.wait_for_function("document.querySelectorAll('#grid img').length===4 && [...document.querySelectorAll('#grid img')].every(i=>i.complete && i.naturalWidth>0)")
    assert page.locator('#age').input_value()=='75'
    page.locator('#age').select_option('60')
    page.locator('#color').select_option('color' if a.texture else 'preserved')
    if a.texture:
        page.locator('#region').select_option('cheek')
        page.wait_for_function("[...document.querySelectorAll('#grid img')].slice(1).every(i=>i.complete && i.naturalWidth===256 && i.src.endsWith('_cheek.png'))")
    with page.expect_download() as pending:page.locator('#grid figure').nth(1).locator('a').click()
    download=pending.value;path=a.run/'browser-download.png';download.save_as(path)
    subprocess.run(['/usr/bin/python3','-c',"from PIL import Image;import sys;im=Image.open(sys.argv[1]);im.load();assert im.format=='PNG' and im.size==(1024,1024)",str(path)],check=True)
    if a.texture:
        page.locator('#subject').select_option('appa_new_006975');page.locator('#age').select_option('60')
        canonical=page.locator('#grid figure').nth(1).locator('a').get_attribute('href')
        assert canonical.endswith('_same_color.png')
        page.locator('#age').select_option('same')
        assert page.locator('#grid figure').nth(1).locator('a').get_attribute('href')==canonical
    page.locator('#subject').select_option(index=1)
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.wait_for_function("[...document.querySelectorAll('#grid img')].every(i=>i.complete && i.naturalWidth>0)")
    assert not errors,errors
    result=dict(url=a.url,status=200,desktop=True,photo_switch=True,age_switch=True,color_switch=True,
        native_crops=a.texture,source_equals_target_anchor=a.texture,download_size=[1024,1024],mobile_width=390,overflow=False,errors=errors)
    (a.run/'browser-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    browser.close()
