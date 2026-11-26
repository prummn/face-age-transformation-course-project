"""Exercise the completed six-arm/three-arm regression gallery in Chromium."""
import argparse
import json
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--url', default='http://127.0.0.1:65512/age-objective-review/')
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path='/tmp/face-age-chrome/chrome',
            args=['--no-sandbox', '--disable-dev-shm-usage'])
        page = browser.new_page(viewport={'width': 1440, 'height': 1000}, accept_downloads=True)
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        response = page.goto(args.url, wait_until='domcontentloaded')
        assert response.status == 200

        def loaded(count, width=512):
            page.wait_for_function('''([count,width]) => {
                const images = [...document.querySelectorAll('#grid img')];
                return images.length === count && images.every(image =>
                    image.complete && image.naturalWidth === width);
            }''', arg=[count, width])

        assert page.locator('#subject option').count() == 35
        assert page.locator('#age').input_value() == '75'
        loaded(6)
        for age in ['60', '20', 'same', '75']:
            page.locator('#age').select_option(age)
            loaded(6)
        page.locator('#color').select_option('color')
        loaded(6)
        for region in ['eye', 'cheek', 'mouth']:
            page.locator('#region').select_option(region)
            loaded(6, 256)
        with page.expect_download() as pending:
            page.locator('#grid figure').last.locator('a').click()
        downloaded = args.run / 'browser-download.png'
        pending.value.save_as(downloaded)
        with Image.open(downloaded) as image:
            image.load()
            assert image.format == 'PNG' and image.size == (1024, 1024)

        page.locator('#subject').select_option('appa_new_006975')
        page.locator('#region').select_option('full')
        page.locator('#age').select_option('60')
        loaded(4)
        anchor = page.locator('#grid figure').last.locator('a').get_attribute('href')
        assert anchor.endswith('_same_color.png')
        page.locator('#age').select_option('same')
        assert page.locator('#grid figure').last.locator('a').get_attribute('href') == anchor
        loaded(4)
        page.locator('#age').select_option('75')
        page.locator('#color').select_option('raw')
        loaded(4)
        assert page.locator('#credit a').count() == 1
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        assert not errors, errors
        result = dict(url=args.url, status=200, cases=35, diagnostic_figures=6,
                      confirmation_figures=4, age_switch=True, color_switch=True,
                      native_regions=['eye', 'cheek', 'mouth'], download_size=[1024, 1024],
                      source_equals_target_anchor=True, source_credit=True,
                      mobile_width=390, overflow=False, errors=errors)
        (args.run / 'browser-validation.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))
        browser.close()


if __name__ == '__main__':
    main()
