"""Check the real result gallery, native downloads and mobile layout on CPU."""
import argparse
import json
from pathlib import Path
import struct
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--url', default='http://127.0.0.1:65512/quality-sprint/')
    args = parser.parse_args()
    data = json.loads((args.run / 'review-data.json').read_text())
    errors, checks = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/tmp/face-age-chrome/chrome',
            args=['--no-sandbox', '--disable-dev-shm-usage'])
        page = browser.new_page(viewport={'width':1440, 'height':1000})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.wait_for_function("document.querySelectorAll('#images figure').length>1")
        assert page.locator('#family option').count() == len(data['families'])
        for key, family in data['families'].items():
            page.select_option('#family', key)
            assert page.locator('#sample option').count() == len(family['records'])
            cohorts = sorted({r['cohort'] for r in family['records']})
            names = list(dict.fromkeys((['uploaded'] if any(r['name']=='uploaded' for r in family['records']) else []) + [next(r['name'] for r in family['records'] if r['cohort']==c) for c in cohorts]))
            for name in names:
                page.select_option('#sample', name)
                for target in ['same', '20', '40', '60', '75']:
                    page.select_option('#age', target)
                    page.wait_for_function("[...document.querySelectorAll('#images img')].every(i=>i.complete&&i.naturalWidth>0)")
                    assert page.locator('#images figure').count() == len(family['routes']) + 1
            with page.expect_download() as download:
                page.locator('#images figure').nth(len(family['routes'])).locator('a').click()
            raw = Path(download.value.path()).read_bytes()
            assert raw[:8] == b'\x89PNG\r\n\x1a\n' and struct.unpack('>II', raw[16:24]) == (1024,1024)
            checks.append(dict(family=key, sample_count=len(family['records']), representatives=names,
                               conditions=5, native_download=[1024,1024]))
        page.select_option('#sample', 'uploaded' if page.locator('#sample option[value="uploaded"]').count() else family['records'][0]['name'])
        page.screenshot(path=str(args.run/'quality-browser-desktop.png'), full_page=True)
        page.set_viewport_size({'width':390, 'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        assert not errors, errors
        browser.close()
    report = dict(url=args.url, checks=checks, mobile_no_overflow=True, page_errors=errors)
    (args.run/'quality-browser-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    main()
