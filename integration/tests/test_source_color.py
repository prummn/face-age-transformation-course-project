"""Numerical safeguards, not a substitute for visual comparison on real faces."""
import sys
import unittest
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source_color import masks, preserve_source_chroma


class SourceColourTests(unittest.TestCase):
    def setUp(self):
        self.source = np.full((256,256,3), (164,126,100), dtype=np.uint8)
        self.generated = np.full((1024,1024,3),(178,121,109),dtype=np.uint8)

    def test_disabled_and_no_drift_preserve_exact_pixels(self):
        out,meta=preserve_source_chroma(self.source,self.generated,0)
        np.testing.assert_array_equal(out,self.generated)
        self.assertFalse(meta['applied'])
        out,meta=preserve_source_chroma(self.source,self.source)
        np.testing.assert_array_equal(out,self.source)
        self.assertFalse(meta['applied'])

    def test_correction_reduces_cheek_chroma_without_changing_background(self):
        source=self.source.copy(); raw=self.generated.copy()
        out,meta=preserve_source_chroma(source,raw)
        self.assertTrue(meta['applied'])
        np.testing.assert_array_equal(source,self.source)
        np.testing.assert_array_equal(raw,self.generated)
        self.assertEqual(out.shape,(1024,1024,3))
        np.testing.assert_array_equal(out[:100],raw[:100])
        cheeks,_=masks(1024)
        src=cv2.cvtColor(source.astype(np.float32)/255,cv2.COLOR_RGB2LAB)[0,0]
        before=cv2.cvtColor(raw.astype(np.float32)/255,cv2.COLOR_RGB2LAB)[cheeks].mean(axis=0)
        after=cv2.cvtColor(out.astype(np.float32)/255,cv2.COLOR_RGB2LAB)[cheeks].mean(axis=0)
        self.assertLess(np.linalg.norm(after[1:]-src[1:]),np.linalg.norm(before[1:]-src[1:])*.5)
        self.assertLess(abs(float(after[0]-before[0])),.5)

    def test_luminance_texture_and_unreliable_source_guard(self):
        raw=self.generated.copy()
        raw[::4]=np.clip(raw[::4].astype(int)-12,0,255)
        out,meta=preserve_source_chroma(self.source,raw)
        before=cv2.cvtColor(raw.astype(np.float32)/255,cv2.COLOR_RGB2LAB)[...,0]
        after=cv2.cvtColor(out.astype(np.float32)/255,cv2.COLOR_RGB2LAB)[...,0]
        self.assertLess(float(np.mean(abs(after-before))),.3)
        out,meta=preserve_source_chroma(np.full_like(self.source,255),raw)
        np.testing.assert_array_equal(out,raw)
        self.assertFalse(meta['applied'])

    def test_dark_or_inconsistent_reference_is_not_forced_to_match(self):
        dark=np.full_like(self.source,18)
        out,meta=preserve_source_chroma(dark,self.generated)
        self.assertEqual(meta['reason'],'low_light_reference')
        np.testing.assert_array_equal(out,self.generated)
        mixed=self.source.copy(); mixed[:,:128]=(70,90,180)
        out,meta=preserve_source_chroma(mixed,self.generated)
        self.assertEqual(meta['reason'],'asymmetric_reference')
        np.testing.assert_array_equal(out,self.generated)


if __name__=='__main__':
    unittest.main()
