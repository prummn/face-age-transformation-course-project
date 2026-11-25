"""Regression checks for source-date interpretation and blinded vote decoding."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from expand_documented_seniors import dates, file_mime
from build_blind_texture_review import score


class SourceEvidenceTests(unittest.TestCase):
    def test_photo_dates_and_explicit_uncertainty(self):
        for value in ['Taken on\u00a016 May 2019','2019-05-16 12:04','16 May 2019']:
            lower,upper,_=dates(value)
            self.assertEqual(lower.isoformat(),'2019-05-16')
            self.assertEqual(lower,upper)
        self.assertEqual(dates('2019')[1].isoformat(),'2019-12-31')
        for value in ['2019-02-30','circa 2019','unknown 2019-05-16',
                      'date of the article 2019-05-16','before 2019']:
            self.assertIsNone(dates(value))

    def test_legacy_mime_does_not_depend_on_query_suffix(self):
        self.assertEqual(file_mime({'url':'https://example.org/a.JPG?utm=1'}),'image/jpeg')
        self.assertIsNone(file_mime({'url':'https://example.org/a.svg'}))
        self.assertEqual(file_mime({'mime':'image/png','url':'https://example.org/a.jpg'}),'image/png')


class BlindScoringTests(unittest.TestCase):
    def test_swapped_repeat_does_not_change_preference_or_sample_size(self):
        keys={
            'one':dict(subject='one',candidate='candidate',left='candidate',right='wplus_base',repeat_of=None),
            'repeat':dict(subject='one',candidate='candidate',left='wplus_base',right='candidate',repeat_of='one')}
        with tempfile.TemporaryDirectory(prefix='synthetic-review-test-') as tmp:
            out=Path(tmp)
            (out/'private-key.json').write_text(json.dumps(dict(study_id='fixture',trial_key=keys,
                protocol=dict(raters_required=2,limitations=['Synthetic test fixture']))))
            answers={tid:dict(texture='left' if key['left']=='candidate' else 'right',identity='tie',
                             age='uncertain',artifact_left=key['left']=='wplus_base',
                             artifact_right=key['right']=='wplus_base') for tid,key in keys.items()}
            answer=out/'answer.json'
            answer.write_text(json.dumps(dict(study_id='fixture',rater='synthetic',answers=answers)))
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream):score(out,[answer])
            result=json.loads(stream.getvalue());d=result['comparisons']['candidate']
            self.assertEqual(d['texture']['votes'],{'candidate':1})
            self.assertEqual(d['serious_artifacts']['wplus_base'],{'flagged':1,'reviewed':1})
            self.assertEqual(result['repeats'][0]['agreement'],dict(texture=True,identity=True,age=True))
            self.assertFalse(result['minimum_raters_met'])
            self.assertFalse(result['automatic_promotion'])
            with self.assertRaises(ValueError),contextlib.redirect_stdout(io.StringIO()):score(out,[answer,answer])
            answers['tampered']=answers['one']
            answer.write_text(json.dumps(dict(study_id='fixture',rater='synthetic',answers=answers)))
            with self.assertRaises(ValueError),contextlib.redirect_stdout(io.StringIO()):score(out,[answer])


if __name__=='__main__':unittest.main()
