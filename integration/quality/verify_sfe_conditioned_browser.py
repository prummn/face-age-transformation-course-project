"""Check condition/case selection, raw/color/crops and native downloads in Chromium."""
import argparse
import json
from pathlib import Path
import subprocess

from playwright.sync_api import sync_playwright

p=argparse.ArgumentParser(description=__doc__);p.add_argument('url');p.add_argument('run',type=Path);a=p.parse_args()
report=json.loads((a.run/'evaluation/report.json').read_text())
dense=json.loads((a.run/'experiment.json').read_text())['profile'] in ['official_sfe_dense_younger','official_sfe_glasses_projected','e4e_source_eyeglass_channels','e4e_eyeglass_correction_sweep']
projection=json.loads((a.run/'experiment.json').read_text())['profile']=='official_sfe_glasses_projected'
attribute=json.loads((a.run/'experiment.json').read_text())['profile'] in ['e4e_source_eyeglass_channels','e4e_eyeglass_correction_sweep']
count=len(report['routes'])+1 if attribute else (3 if dense and not projection else 4)
with sync_playwright() as pw:
    browser=pw.chromium.launch(executable_path='/tmp/face-age-chrome/chrome',args=['--no-sandbox','--disable-dev-shm-usage'])
    page=browser.new_page(viewport={'width':1500,'height':950},accept_downloads=True)
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    response=page.goto(a.url,wait_until='domcontentloaded');assert response.status==200
    page.wait_for_function(f"document.querySelectorAll('#grid img').length==={count} && [...document.querySelectorAll('#grid img')].every(x=>x.complete&&x.naturalWidth>0)")
    cases=page.locator('#subject option').count();conditions=page.locator('#condition option').count()
    assert cases==len(report['records'])
    assert conditions==(len(report['scales']) if dense else len(report['targets'])+1)
    for index in range(conditions):
        page.locator('#condition').select_option(index=index)
        page.wait_for_function("[...document.querySelectorAll('#grid img')].every(x=>x.complete&&x.naturalWidth>0)")
    page.locator('#color').select_option('color');page.locator('#region').select_option('eye')
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
    result=dict(url=a.url,status=200,cases=cases,conditions=conditions,all_conditions_switch=True,all_cases_switch=True,
        color_switch=True,native_crop=True,download=[1024,1024],mobile_width=390,errors=errors)
    (a.run/'browser-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));browser.close()
