"""Validate train-only variants and sample parents without duplicating people.

This helper uses only the standard library. An augmentation is not a new
identity, a verified age label, or a high-resolution ground-truth target.
"""
import hashlib
import json
import random
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_bank(path, training, evaluation):
    if not path:
        return {}
    parents = {row['key']: row for row in training}
    eval_keys = {row['key'] for row in evaluation}
    eval_groups = {row['duplicate_candidate_group'] for row in evaluation}
    eval_hashes = {row['sha256'] for row in evaluation}
    source_hashes = {}
    bank = {}
    seen = set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        parent_key = row['parent_key']
        if parent_key not in parents or parent_key in eval_keys:
            raise ValueError('Augmentation parent is not an eligible training input: ' + parent_key)
        parent = parents[parent_key]
        if row['key'] in seen or row['key'] in parents or row['key'] in eval_keys:
            raise ValueError('Augmentation key is duplicate or collides with an original key')
        seen.add(row['key'])
        if row['split'] != 'train_augmentation':
            raise ValueError('Augmentation must have a train-only split')
        if row['duplicate_candidate_group'] != parent['duplicate_candidate_group'] or row['duplicate_candidate_group'] in eval_groups:
            raise ValueError('Augmentation group differs from parent or overlaps evaluation')
        for field in ['pseudo_age', 'pseudo_age_source']:
            if row[field] != parent[field]:
                raise ValueError('Augmentation must inherit the original pseudo-age metadata')
        original = Path(parent['image_path']).resolve()
        if Path(row['source_image_path']).resolve() != original:
            raise ValueError('Augmentation source path differs from parent')
        variant = Path(row['image_path']).resolve()
        if variant == original:
            raise ValueError('Augmentation must not replace its original')
        if parent_key not in source_hashes:
            source_hashes[parent_key] = digest(original)
        if source_hashes[parent_key] != row['source_sha256'] or digest(variant) != row['sha256']:
            raise ValueError('Augmentation or source hash differs from the recorded bank')
        if source_hashes[parent_key] in eval_hashes or row['sha256'] in eval_hashes:
            raise ValueError('Augmentation or parent has the same file content as frozen evaluation')
        bank.setdefault(parent_key, []).append(row)
    if not bank:
        raise ValueError('Augmentation bank has no accepted training variants')
    for variants in bank.values():
        variants.sort(key=lambda row: row['key'])
    return bank


class ParentSampler:
    """Independent RNGs keep parent choices identical between both arms."""
    def __init__(self, rows, bank, probability, seed):
        if not 0 <= probability <= 1:
            raise ValueError('Augmentation probability must be between zero and one')
        if probability and not bank:
            raise ValueError('Nonzero augmentation probability requires a verified bank')
        self.rows, self.bank, self.probability = rows, bank, probability
        self.weights = [row['sampling_weight'] for row in rows]
        self.source_rng = random.Random(seed)
        self.augmentation_rng = random.Random(seed + 1000003)

    def sample(self, size):
        parents = self.source_rng.choices(self.rows, weights=self.weights, k=size)
        variants = []
        for parent in parents:
            candidates = self.bank.get(parent['key'], [])
            use = self.augmentation_rng.random() < self.probability
            variants.append(self.augmentation_rng.choice(candidates) if use and candidates else None)
        return parents, variants
