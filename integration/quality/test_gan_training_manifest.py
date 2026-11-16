"""Training split guards prevent accidental evaluation leakage and photo misuse."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from train_gan_reconstruction import validate
from probe_inversion import sha256


class ManifestGuards(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        for name in ['a','b']:(self.root/name).write_bytes(name.encode())
        self.records=[dict(image_path=str(self.root/name),sha256=sha256(self.root/name),identity_group=name,
                           source='synthetic manifest validation fixture',training_authorized=True,
                           source_age=30,age_kind='pseudo',group_kind='conservative_source_similarity_component') for name in ['a','b']]
        self.args=Namespace(train_manifest=self.root/'train.jsonl',eval_manifest=self.root/'eval.jsonl',
                            output=self.root/'output',steps=100,batch_size=2)
    def check(self,a=None,b=None):
        self.args.train_manifest.write_text(json.dumps(a or self.records[0])+'\n')
        self.args.eval_manifest.write_text(json.dumps(b or self.records[1])+'\n')
        return validate(self.args)
    def test_valid_disjoint_manifests(self):
        self.assertEqual(tuple(map(len,self.check())),(1,1))
    def test_identity_overlap(self):
        b=dict(self.records[1],identity_group='a')
        with self.assertRaisesRegex(ValueError,'identity overlap'):self.check(b=b)
    def test_missing_identity_group(self):
        a=dict(self.records[0],identity_group=None)
        with self.assertRaisesRegex(ValueError,'identity_group'):self.check(a=a)
    def test_missing_authorization(self):
        a=dict(self.records[0],training_authorized=False)
        with self.assertRaisesRegex(ValueError,'authorization'):self.check(a=a)
    def test_content_changed(self):
        (self.root/'a').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.check()
    def test_reject_undefined_group_provenance(self):
        a=dict(self.records[0],group_kind='filename')
        with self.assertRaisesRegex(ValueError,'group_kind'):self.check(a=a)
    def test_highres_reference_hash(self):
        a=dict(self.records[0],reference512_path=str(self.root/'b'),reference512_sha256='bad')
        with self.assertRaisesRegex(ValueError,'reference hash'):self.check(a=a)
    def test_reject_missing_source_age_kind(self):
        a=dict(self.records[0],age_kind='interval_midpoint_truth')
        with self.assertRaisesRegex(ValueError,'age_kind'):self.check(a=a)
    def test_reject_existing_output(self):
        self.args.output.mkdir()
        with self.assertRaisesRegex(ValueError,'new experiment'):self.check()


if __name__=='__main__':unittest.main()
