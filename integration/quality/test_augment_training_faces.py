"""CPU numerical, provenance, leakage and overwrite checks; no model loading."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, JpegImagePlugin

import augment_training_faces as aug


def texture():
    yy, xx = np.mgrid[:256, :256]
    light = 8 * np.sin(xx / 6.) + 8 * np.cos(yy / 7.)
    return np.rint(np.stack([155 + light, 120 + light, 100 + light], axis=-1)).astype(np.uint8)


class AugmentationChecks(unittest.TestCase):
    def test_seed_and_bytes_are_reproducible_and_order_independent(self):
        source = texture()
        for kind in aug.KINDS:
            seed = aug.stable_seed(20261110, 'parent1', kind)
            a, recipe_a = aug.candidate(source, kind, seed)
            aug.candidate(source, kind, aug.stable_seed(20261110, 'unrelated', kind))
            b, recipe_b = aug.candidate(source, kind, seed)
            self.assertEqual(a, b)
            self.assertEqual(recipe_a, recipe_b)
            self.assertNotEqual(seed, aug.stable_seed(20261110, 'parent2', kind))

    def test_illumination_preserves_geometry_and_decoded_chroma_within_gates(self):
        source = texture()
        encoded, recipe = aug.candidate(source, 'illumination', 123)
        result = aug.decode(encoded)
        metrics = aug.measure(source, result)
        self.assertEqual(result.shape, source.shape)
        self.assertFalse(recipe['nominal_ab_edit'])
        self.assertGreater(metrics['L_abs_mean'], .5)
        self.assertEqual(aug.reject_reasons(metrics, 'illumination'), [])

    def test_jpeg_has_no_chroma_subsampling_and_passes_mild_quality_gate(self):
        source = texture()
        encoded, recipe = aug.candidate(source, 'jpeg', 123)
        with Image.open(io.BytesIO(encoded)) as image:
            self.assertEqual(JpegImagePlugin.get_sampling(image), 0)
        self.assertIn(recipe['quality'], [96, 97, 98])
        self.assertEqual(aug.reject_reasons(aug.measure(source, aug.decode(encoded)), 'jpeg'), [])

    def test_color_blur_exposure_and_geometry_regressions_are_rejected(self):
        source = texture()
        red = source.copy(); red[:, :, 0] = np.clip(red[:, :, 0].astype(int) + 30, 0, 255)
        self.assertIn('max_cheek_ab_shift_above_limit', aug.reject_reasons(aug.measure(source, red), 'illumination'))
        blurred = cv2.GaussianBlur(source, (0, 0), 10)
        self.assertIn('L_laplacian_ratio_below_limit', aug.reject_reasons(aug.measure(source, blurred), 'jpeg'))
        white = np.full_like(source, 255)
        reasons = aug.reject_reasons(aug.measure(source, white), 'illumination')
        self.assertIn('L_abs_p99_above_limit', reasons)
        self.assertIn('new_luminance_clipped_fraction_above_limit', reasons)
        with self.assertRaisesRegex(ValueError, 'Geometry'):
            aug.measure(source, source[:128])

    def fixture(self, directory):
        rows = []
        for index, key in enumerate(('a', 'b', 'eval')):
            path = directory / (key + '.png')
            Image.fromarray(texture() + index).save(path)
            rows.append({'key': key, 'image_path': str(path), 'duplicate_candidate_group': 'g_' + key,
                         'pseudo_age': 64.1, 'pseudo_age_source': 'original_full_pseudo_label_not_ground_truth',
                         'split': 'train' if key != 'eval' else 'frozen_regression_eval', 'sha256': aug.sha256(path)})
        return rows[:2], rows[2:]

    def test_manifest_leakage_and_eval_mutation_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root); train, evaluation = self.fixture(directory)
            aug.validate_manifests(train, evaluation, 2)
            leaked = [dict(train[0], duplicate_candidate_group='g_eval'), train[1]]
            with self.assertRaisesRegex(ValueError, 'group overlap'):
                aug.validate_manifests(leaked, evaluation, 2)
            with self.assertRaisesRegex(ValueError, 'key overlap'):
                aug.validate_manifests([dict(train[0], key='eval'), train[1]], evaluation, 2)
            with self.assertRaisesRegex(ValueError, 'Duplicate train'):
                aug.validate_manifests([train[0], train[0]], evaluation, 2)
            leaked_bytes = [dict(train[0], image_path=evaluation[0]['image_path']), train[1]]
            with self.assertRaisesRegex(ValueError, 'content hash overlap'):
                aug.validate_manifests(leaked_bytes, evaluation, 2)
            (directory / 'eval.png').write_bytes(b'mutated')
            with self.assertRaisesRegex(ValueError, 'Frozen eval image changed'):
                aug.validate_manifests(train, evaluation, 2)

    def test_bank_inherits_provenance_keeps_sources_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root); inputs = directory / 'sources'; inputs.mkdir()
            train, evaluation = self.fixture(inputs)
            train_file, eval_file = directory / 'train.jsonl', directory / 'eval.jsonl'
            train_file.write_text(''.join(json.dumps(r) + '\n' for r in train))
            eval_file.write_text(''.join(json.dumps(r) + '\n' for r in evaluation))
            args = SimpleNamespace(train_manifest=str(train_file), eval_manifest=str(eval_file),
                                   out=str(directory / 'bank'), protected_root=[str(inputs)],
                                   expected_parent_count=2, limit=None, seed=20261110)
            with contextlib.redirect_stdout(io.StringIO()):
                aug.build(args)
            accepted = aug.read_manifest(Path(args.out) / 'accepted_variants.jsonl')
            rejected = aug.read_manifest(Path(args.out) / 'rejected_variants.jsonl')
            self.assertGreaterEqual(len(accepted), 2)
            self.assertEqual(len(accepted) + len(rejected), 4)
            parents = {r['key']: r for r in train}
            for row in accepted:
                original = parents[row['parent_key']]
                self.assertEqual(row['pseudo_age_source'], original['pseudo_age_source'])
                self.assertEqual(row['duplicate_candidate_group'], original['duplicate_candidate_group'])
                self.assertEqual(row['source_sha256'], original['sha256'])
                self.assertEqual(aug.sha256(row['image_path']), row['sha256'])
                self.assertEqual(aug.sha256(row['source_image_path']), row['source_sha256'])
            for row in rejected:
                self.assertNotIn('image_path', row)
                self.assertTrue(row['reject_reasons'])
            self.assertEqual(aug.sha256(evaluation[0]['image_path']), evaluation[0]['sha256'])
            with self.assertRaisesRegex(ValueError, 'already exists'):
                aug.build(args)
            # An encoded candidate matching eval content must be rejected even
            # when its parent key/group and small numerical shifts are valid.
            original_candidate = aug.candidate
            eval_bytes = Path(evaluation[0]['image_path']).read_bytes()
            def candidate_with_leak(source, kind, seed):
                encoded, recipe = original_candidate(source, kind, seed)
                return (eval_bytes, recipe) if kind == 'illumination' else (encoded, recipe)
            args.out = str(directory / 'leak_guard_bank')
            with patch.object(aug, 'candidate', side_effect=candidate_with_leak), contextlib.redirect_stdout(io.StringIO()):
                aug.build(args)
            leak_rejections = aug.read_manifest(Path(args.out) / 'rejected_variants.jsonl')
            self.assertEqual(sum('candidate_content_hash_overlaps_eval' in r['reject_reasons'] for r in leak_rejections), 2)
            args.out = str(inputs / 'forbidden')
            with self.assertRaisesRegex(ValueError, 'protected'):
                aug.build(args)


if __name__ == '__main__':
    cv2.setNumThreads(2)
    unittest.main()
