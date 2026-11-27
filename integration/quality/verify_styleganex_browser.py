"""Check the actual strength/age review page and a native PNG download."""
import argparse
import json
from pathlib import Path
import subprocess

from playwright.sync_api import sync_playwright

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('url'); p.add_argument('run',type=Path); args = p.parse_args()
with sync_playwright() as pw:
    browser = pw.chromium.launch(executable_path='/tmp/face-age-chrome/chrome',
        args=['--no-sandbox','--disable-dev-shm-usage'])
    page = browser.new_page(viewport={'width':1400,'height':950},accept_downloads=True)
    errors = []; page.on('pageerror',lambda error:errors.append(str(error)))
    response = page.goto(args.url,wait_until='domcontentloaded'); assert response.status == 200
    page.wait_for_function("document.querySelectorAll('#grid img').length===3 && [...document.querySelectorAll('#grid img')].every(x=>x.complete&&x.naturalWidth>0)")
    assert page.locator('#subject option').count() == 8
    assert page.locator('#scale option').count() == 7
    assert page.locator('#sam').input_value() == '75'
    assert page.locator('#scale').input_value() == 'scale_1'
    page.locator('#sam').select_option('same'); page.locator('#scale').select_option('scale_0')
    assert '同龄' in page.locator('#grid figure').nth(1).inner_text()
    assert '强度0' in page.locator('#grid figure').nth(2).inner_text()
    for scale in ['scale_-1','scale_-0.5','scale_0','scale_0.5','scale_1','scale_1.5','scale_2']:
        page.locator('#scale').select_option(scale)
        page.wait_for_function("[...document.querySelectorAll('#grid img')].every(x=>x.complete&&x.naturalWidth>0)")
    page.locator('#color').select_option('color'); page.locator('#region').select_option('mouth')
    page.wait_for_function("[...document.querySelectorAll('#grid img')].every(x=>x.complete&&x.naturalWidth===256)")
    with page.expect_download() as pending: page.locator('#grid figure').nth(2).locator('a').click()
    filename = args.run/'browser-download.png'; pending.value.save_as(filename)
    subprocess.run(['/usr/bin/python3','-c',
        "from PIL import Image;import sys;im=Image.open(sys.argv[1]);im.load();assert im.format=='PNG' and im.size==(1024,1024)",str(filename)],check=True)
    page.locator('#subject').select_option(index=1)
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.wait_for_function("[...document.querySelectorAll('#grid img')].every(x=>x.complete&&x.naturalWidth>0)")
    assert not errors,errors
    result = dict(url=args.url,status=200,cases=8,strengths=7,all_strength_switches=True,
        sam_age_switch=True,color_switch=True,native_crops=True,native_download=[1024,1024],
        mobile_width=390,overflow=False,errors=errors)
    (args.run/'browser-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result)); browser.close()
