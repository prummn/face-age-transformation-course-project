"""Guard against label-conflict and train/validation contamination in AAF intake."""
import json
from pathlib import Path
import tempfile
import unittest

from audit_aaf import screen_age_candidates


def row(name, age, split, original=None, aligned=None, multi=False):
    return dict(name=name,publisher_age=age,publisher_gender='female',split=split,
                original=dict(pixel_sha256=original or 'original-'+name),
                aligned=dict(pixel_sha256=aligned or 'aligned-'+name),
                face_qc=dict(multiple_prominent_faces=multi),training_admitted=False)


class ExactAgeCandidateTests(unittest.TestCase):
    def test_conflicts_and_validation_content_cannot_enter_training(self):
        records=[
            row('conflict-train',78,'train',original='conflict'),
            row('conflict-val',46,'val',original='conflict'),
            row('protected-train',70,'train',original='protected'),
            row('protected-val',70,'val',original='protected'),
            row('same-train-a',67,'train',original='same-train'),
            row('same-train-b',67,'train',original='same-train'),
            # A conflict bridged through original, then through aligned pixels
            # must quarantine all three records, including the indirect member.
            row('bridge-a',60,'train',original='bridge'),
            row('bridge-b',60,'train',original='bridge',aligned='bridge-aligned'),
            row('bridge-c',80,'val',aligned='bridge-aligned'),
            row('multiple-faces',72,'train',multi=True),
            row('child',10,'train'),
            row('unique-train',75,'train'),
            row('unique-val',76,'val')]
        with tempfile.TemporaryDirectory(prefix='aaf-intake-test-') as tmp:
            out=Path(tmp);result=screen_age_candidates(records,out)
            kept={s:[json.loads(line) for line in (out/f'age-{s}.candidates.jsonl').read_text().splitlines()] for s in ['train','val']}
            self.assertEqual({r['name'] for r in kept['train']},{'same-train-a','unique-train'})
            self.assertEqual({r['name'] for r in kept['val']},{'protected-val','unique-val'})
            self.assertEqual(result['rejections_by_reason']['exact_group_label_conflict'],5)
            self.assertEqual(result['exact_train_val_overlap_after_screen'],0)
            self.assertFalse(result['approximate_identity_isolation_done'])
            self.assertEqual(result['training_admitted'],0)
            self.assertTrue(all(not r['training_admitted'] and not r['texture_training_eligible'] for rows in kept.values() for r in rows))


if __name__=='__main__':unittest.main()
