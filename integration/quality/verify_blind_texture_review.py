"""Exercise the real route-hidden form; generated answers are synthetic only."""
import argparse
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen
from urllib.error import HTTPError

from playwright.sync_api import sync_playwright


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('url');p.add_argument('output',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    errors=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path='/tmp/face-age-chrome/chrome',
                                  args=['--no-sandbox','--disable-dev-shm-usage'])
        page=browser.new_page(viewport={'width':1440,'height':1100},accept_downloads=True)
        page.on('pageerror',lambda e:errors.append(str(e)))
        r=page.goto(a.url,wait_until='domcontentloaded');assert r.status==200
        page.wait_for_function("document.querySelectorAll('#images img').length===3 && [...document.querySelectorAll('#images img')].every(i=>i.complete&&i.naturalWidth===1024)")
        body=page.content()
        for text in ['personalized_reconstruction','personalized_locality','wplus_base','private-key.json']:
            assert text not in body
        page.locator('#save').click()
        assert '评审编号' in page.locator('#notice').inner_text()
        page.locator('#rater').fill('browser-automation-test')
        page.locator('#save').click()
        assert '三个选择' in page.locator('#notice').inner_text()
        page.locator('input[name=texture][value=tie]').check()
        page.locator('input[name=identity][value=left]').check()
        page.locator('input[name=age][value=uncertain]').check()
        page.locator('#artifactLeft').check()
        page.locator('#comment').fill('SYNTHETIC BROWSER CHECK - not a human rating')
        page.locator('#save').click()
        assert '第 2 / 55' in page.locator('#progress').inner_text()
        page.locator('#prev').click()
        assert page.locator('input[name=texture][value=tie]').is_checked()
        assert page.locator('#artifactLeft').is_checked()
        page.reload(wait_until='domcontentloaded')
        assert page.locator('input[name=identity][value=left]').is_checked()
        page.locator('#region').select_option('eyes')
        page.wait_for_function("[...document.querySelectorAll('#images img')].every(i=>i.complete&&i.naturalWidth===584&&i.naturalHeight===180)")
        page.locator('#region').select_option('mouth')
        page.wait_for_function("[...document.querySelectorAll('#images img')].every(i=>i.complete&&i.naturalWidth===344&&i.naturalHeight===185)")
        assert page.evaluate("[...document.querySelectorAll('#images img')].every(i=>i.getBoundingClientRect().width<=i.naturalWidth)")
        with tempfile.TemporaryDirectory(prefix='synthetic-blind-browser-') as tmp:
            with page.expect_download() as pending:page.locator('#export').click()
            path=Path(tmp)/'synthetic.json';pending.value.save_as(path)
            answer=json.loads(path.read_text())
            assert answer['rater']=='browser-automation-test' and len(answer['answers'])==1
            result=next(iter(answer['answers'].values()))
            assert result['texture']=='tie' and result['identity']=='left' and result['age']=='uncertain'
            assert result['artifact_left'] and 'SYNTHETIC' in result['comment']
        page.screenshot(path=str(a.output/'desktop.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        assert not errors,errors
        browser.close()
    try:
        with urlopen(a.url.rstrip('/')+'/private-key.json',timeout=5):
            raise AssertionError('Private key was served')
    except HTTPError as e:assert e.code==404
    receipt=dict(status=200,route_names_hidden=True,private_key_http_status=404,
                 native_1024=True,unresized_crops=True,no_display_upscaling=True,required_fields=True,
                 answers_persist_reload=True,back_navigation=True,json_download=True,
                 mobile_width=390,horizontal_overflow=False,errors=errors,
                 synthetic_only=True,human_ratings_created=0)
    (a.output/'browser-validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
