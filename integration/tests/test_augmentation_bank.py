import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'quality'))
from augmentation_bank import ParentSampler, digest, read_bank


class AugmentationBankTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.png'; self.source.write_bytes(b'original')
        self.variant = self.root / 'variant.png'; self.variant.write_bytes(b'variant')
        self.parent = dict(key='train-a', image_path=str(self.source), duplicate_candidate_group='group-a', pseudo_age=40.25, pseudo_age_source='estimate_NOT_truth', sampling_weight=1)
        evaluation_path=self.root / 'evaluation.png'; evaluation_path.write_bytes(b'evaluation')
        self.evaluation = [dict(key='eval-a', duplicate_candidate_group='group-eval',sha256=digest(evaluation_path))]
        self.row = dict(key='train-a__light', parent_key='train-a', image_path=str(self.variant), source_image_path=str(self.source), source_sha256=digest(self.source), sha256=digest(self.variant), duplicate_candidate_group='group-a', pseudo_age=40.25, pseudo_age_source='estimate_NOT_truth', split='train_augmentation', recipe='light')
        self.manifest = self.root / 'bank.jsonl'

    def load(self, row=None):
        self.manifest.write_text(json.dumps(row or self.row) + '\n')
        return read_bank(self.manifest, [self.parent], self.evaluation)

    def test_group_and_age_cannot_be_forged_for_evaluation_or_new_labels(self):
        self.assertEqual(list(self.load()), ['train-a'])
        for field, value in [('parent_key', 'eval-a'), ('duplicate_candidate_group', 'group-eval'), ('pseudo_age', 60), ('pseudo_age_source', 'verified'), ('split', 'eval')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load({**self.row, field: value})

    def test_modified_variant_or_source_is_rejected(self):
        self.variant.write_bytes(b'changed')
        with self.assertRaises(ValueError): self.load()
        self.variant.write_bytes(b'variant'); self.source.write_bytes(b'changed')
        with self.assertRaises(ValueError): self.load()

    def test_hash_collision_with_evaluation_cannot_hide_behind_new_keys(self):
        for field in ['sha256', 'source_sha256']:
            with self.subTest(field=field):
                self.evaluation[0]['sha256']=self.row[field]
                with self.assertRaises(ValueError): self.load()

    def test_parent_sampling_is_identical_despite_variant_counts_and_probability(self):
        bank = self.load()
        parents = [self.parent, {**self.parent, 'key': 'train-b', 'sampling_weight': 3}]
        bank['train-b'] = [{**self.row, 'key': f'b-{index}'} for index in range(9)]
        control = ParentSampler(parents, bank, 0, 123)
        augmented = ParentSampler(parents, bank, .25, 123)
        changed = 0
        for _ in range(100):
            left, unused = control.sample(2)
            right, variants = augmented.sample(2)
            self.assertEqual([r['key'] for r in left], [r['key'] for r in right])
            self.assertFalse(any(unused))
            changed += sum(r is not None for r in variants)
        self.assertGreater(changed, 0)
        self.assertLess(changed, 200)


if __name__ == '__main__':
    unittest.main()
