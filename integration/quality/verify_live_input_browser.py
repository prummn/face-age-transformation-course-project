"""Exercise a public portrait through fake camera hardware and the REAL SAM API.

Only the camera device is synthetic. No API/model responses are mocked. Run
with deployment/control-venv/bin/python; system Python creates the Y4M fixture.
"""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:65512/')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    photo=ROOT/'deployment/age-quality-final/expanded-ffhq/native1024/64376.png'
    fixture=args.out/'public-camera.y4m'
    subprocess.run(['/usr/bin/python3','-c','''
import sys,cv2,numpy as np
from PIL import Image
p=Image.open(sys.argv[1]).convert('RGB').resize((700,700))
im=Image.new('RGB',(1920,1080),(120,120,120));im.paste(p,(610,190))
im.save(sys.argv[2]+'.png')
im.resize((640,360)).save(sys.argv[2]+'.low.png')
with open(sys.argv[2],'wb') as f:
 f.write(b'YUV4MPEG2 W1920 H1080 F10:1 Ip A1:1 C420jpeg\\nFRAME\\n')
 f.write(cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2YUV_I420).tobytes())
''',str(photo),str(fixture)],check=True)
    report=dict(real_api=True,synthetic_camera=True,photo=str(photo),errors=[])
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path='/tmp/face-age-chrome/chrome',args=[
            '--no-sandbox','--disable-dev-shm-usage','--use-fake-device-for-media-stream',
            '--use-fake-ui-for-media-stream','--use-file-for-fake-video-capture='+str(fixture.resolve())])
        context=browser.new_context(viewport={'width':1400,'height':950},accept_downloads=True,permissions=['camera'])
        page=context.new_page();page.on('pageerror',lambda error:report['errors'].append(str(error)))
        page.add_init_script('''window.testCameraConstraints=[];
const realGet=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
navigator.mediaDevices.getUserMedia=constraints=>{window.testCameraConstraints.push(constraints);return realGet(constraints);};''')
        session=None
        try:
            page.goto(args.url,wait_until='domcontentloaded')
            page.wait_for_function('window.AgeMirrorAPI?.state.ready===true')
            page.evaluate("defaultTargets=()=>[75]; $('#autoCapture').checked=false;")
            page.evaluate('startCamera()')
            page.wait_for_function('F.faces.length>0 && DET.ready',timeout=90000)
            report['camera']=page.evaluate('''() => ({constraints:window.testCameraConstraints,
                width:$('#video').videoWidth,height:$('#video').videoHeight,
                facePixels:qualityOf(probe,selectedFace(),{width:$('#video').videoWidth,height:$('#video').videoHeight}).facePixels})''')
            assert report['camera']['width']==1920 and report['camera']['height']==1080
            gate=page.evaluate('''() => {
                const f=selectedFace();
                const native=qualityOf(probe,f,{width:1920,height:1080});
                const small=qualityOf(probe,{...f,w:.1,h:.1},{width:640,height:480});
                return {native: native.facePixels,small:small.facePixels,
                    nativeWarning:native.warnings.some(s=>s.includes('像素')),
                    smallWarning:small.warnings.some(s=>s.includes('像素'))};
            }''')
            assert gate['native']>=256 and not gate['nativeWarning']
            assert gate['small']<256 and gate['smallWarning'];report['native_pixel_gate']=gate
            started=time.monotonic()
            page.evaluate('capture()')
            # Any existing non-size warning intentionally opens the confirmation page.
            if page.evaluate("S.state==='s-input'"):page.locator('#bContinue').click()
            page.wait_for_function("S.state==='s-grid' && S.results.has(75)",timeout=120000)
            report['camera_to_result_seconds']=time.monotonic()-started
            session=page.evaluate('AgeMirrorAPI.state.session')
            report['result']=page.evaluate('''() => ({source:[S.original.width,S.original.height],
                age:S.age,quality:S.quality,output:[S.results.get(75).width,S.results.get(75).height],
                meta:S.results.get(75).ageMeta})''')
            assert report['result']['output']==[1024,1024]
            data=page.evaluate('''() => ({aligned:S.face.toDataURL(),reference:S.referenceFace.toDataURL(),
                raw:S.results.get(75).ageRaw.toDataURL(),color:S.results.get(75).toDataURL()})''')
            for name,url in data.items():(args.out/(name+'.png')).write_bytes(base64.b64decode(url.split(',',1)[1]))
            page.evaluate("stopPlay(); S.mode='mirror';buildPages();goPage(1);setColorView('raw');goPage(1)")
            assert page.evaluate('S.pages[1].image.width')==1920
            assert '1024' in page.locator('#bNativeSave').text_content()
            page.locator('#bCompareNow').click()
            assert page.evaluate("S.state==='s-detail' && S.view==='crop'")
            with page.expect_download() as pending:page.locator('#bDetailNativeSave').click()
            pending.value.save_as(args.out/'mirror-native-download.png')
            # The mirror's scene is1920, but its native model download must stay1024.
            subprocess.run(['/usr/bin/python3','-c',
                'from PIL import Image;import sys;im=Image.open(sys.argv[1]);im.load();assert im.size==(1024,1024) and im.format=="PNG";assert im.convert("RGB").tobytes()==Image.open(sys.argv[2]).convert("RGB").tobytes()',
                str(args.out/'mirror-native-download.png'),str(args.out/'raw.png')],check=True)
            report['mirror_native_download']=[1024,1024]
            report['download_matches_raw_rgb']=True
            page.locator('#bPasteView').click();assert page.evaluate("S.view==='paste'")
            page.locator('#bCropView').click()
            report['crop_and_paste_comparison']=True
            page.screenshot(path=str(args.out/'camera-real-result.png'))
            page.evaluate('toWaiting()');page.wait_for_timeout(500)
            try:
                req=Request(args.url.rstrip('/')+'/api/transform',data=json.dumps(dict(session_id=session,target_age=75)).encode(),headers={'Content-Type':'application/json'})
                with urlopen(req,timeout=10) as response:status=response.status
            except Exception as error:status=getattr(error,'code',None)
            assert status==410;report['after_clear_status']=status;session=None
            page.evaluate("S.mode='phone';show('s-mirror')")
            page.locator('#file').set_input_files(str(fixture)+'.low.png')
            page.wait_for_function("S.state==='s-input' && !S.preparing && !!AgeMirrorAPI.state.session",timeout=90000)
            session=page.evaluate('AgeMirrorAPI.state.session')
            assert page.evaluate("S.quality.facePixels<256 && !S.generating && S.results.size===0")
            report['low_quality_confirmation']=page.evaluate('({facePixels:S.quality.facePixels,warnings:S.quality.warnings,automatic_generation:S.generating})')
            page.evaluate('toWaiting()');page.wait_for_timeout(500);session=None
            assert not report['errors'],report['errors']
            report['passed']=True
        finally:
            if session:
                try:
                    with urlopen(Request(args.url.rstrip('/')+'/api/sessions/'+session,method='DELETE'),timeout=10):pass
                except Exception:pass
            browser.close()
            (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
