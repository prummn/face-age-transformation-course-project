"""Verify real SFE/AgeTransGAN comparison artifacts in Chromium, without GPU."""
import argparse
import json
from pathlib import Path
import subprocess

from playwright.sync_api import sync_playwright

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('profile',choices=['sfe','agetrans']);p.add_argument('url');p.add_argument('run',type=Path)
a=p.parse_args();sfe=a.profile=='sfe';count=4 if sfe else 3
report=json.loads((a.run/'evaluation/report.json').read_text());cases=len(report['records'])
values=['scale_'+format(s,'g') for s in report['scales']] if sfe else ['group_'+str(g) for g in report['groups']]
with sync_playwright() as pw:
    browser=pw.chromium.launch(executable_path='/tmp/face-age-chrome/chrome',args=['--no-sandbox','--disable-dev-shm-usage'])
    page=browser.new_page(viewport={'width':1500,'height':950},accept_downloads=True)
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    response=page.goto(a.url,wait_until='domcontentloaded');assert response.status==200
    page.wait_for_function(f"document.querySelectorAll('#grid img').length==={count} && [...document.querySelectorAll('#grid img')].every(x=>x.complete&&x.naturalWidth>0)")
    assert page.locator('#subject option').count()==cases
    assert page.locator('#scale option').count()==len(values)
    for value in values:
        page.locator('#scale').select_option(value)
        page.wait_for_function("[...document.querySelectorAll('#grid img')].every(x=>x.complete&&x.naturalWidth>0)")
    page.locator('#sam').select_option('same')
    assert '同龄' in page.locator('#grid figure').nth(1).inner_text()
    page.locator('#color').select_option('color');page.locator('#region').select_option('mouth')
    page.wait_for_function("[...document.querySelectorAll('#grid img')].every(x=>x.complete&&x.naturalWidth===256)")
    with page.expect_download() as pending:page.locator('#grid figure').nth(2).locator('a').click()
    path=a.run/'browser-download.png';pending.value.save_as(path)
    subprocess.run(['/usr/bin/python3','-c',"from PIL import Image;import sys;im=Image.open(sys.argv[1]);im.load();assert im.format=='PNG' and im.size==(1024,1024)",str(path)],check=True)
    page.set_viewport_size({'width':390,'height':844})
    for index in range(cases):
        page.locator('#subject').select_option(index=index)
        page.wait_for_function("[...document.querySelectorAll('#grid img')].every(x=>x.complete&&x.naturalWidth>0)")
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    assert not errors,errors
    result=dict(profile=a.profile,url=a.url,status=200,cases=cases,conditions=len(values),all_conditions_switch=True,
        all_cases_switch=True,sam_age_switch=True,color_switch=True,native_crops=True,native_download=[1024,1024],mobile_width=390,overflow=False,errors=errors)
    (a.run/'browser-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));browser.close()
