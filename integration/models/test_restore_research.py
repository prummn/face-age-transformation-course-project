"""Artifact integrity / traversal / non-overwrite checks, no network or models."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from restore_research import configure,digest,install,installed,target


class RestoreResearchTests(unittest.TestCase):
    def archive(self,folder,name,body,declared=None):
        archive=folder/'sample.tar.gz'
        with tarfile.open(archive,'w:gz') as t:
            info=tarfile.TarInfo(name);info.size=len(body);t.addfile(info,io.BytesIO(body))
        return archive,dict(format='tar.gz',files=[dict(path=declared or name,bytes=len(body),sha256=hashlib.sha256(body).hexdigest())])

    def test_real_content_and_idempotence(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);blob,item=self.archive(root,'deployment/reports/result.json',b'{"real":true}\n')
            self.assertFalse(installed(root,item));install(root,item,blob);self.assertTrue(installed(root,item));install(root,item,blob)
            self.assertEqual((root/'deployment/reports/result.json').read_bytes(),b'{"real":true}\n')

    def test_different_existing_result_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);blob,item=self.archive(root,'result.txt',b'expected');(root/'result.txt').write_bytes(b'other work')
            with self.assertRaises(ValueError):installed(root,item)
            self.assertEqual((root/'result.txt').read_bytes(),b'other work')

    def test_tampered_content_never_extracts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);blob,item=self.archive(root,'result.txt',b'wrong');item['files'][0]['sha256']='0'*64
            with self.assertRaises(ValueError):install(root,item,blob)
            self.assertFalse((root/'result.txt').exists())

    def test_traversal_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as d,tempfile.TemporaryDirectory() as outside:
            root=Path(d);blob,item=self.archive(root,'../escape',b'x','allowed.txt')
            with self.assertRaises(ValueError):install(root,item,blob)
            (root/'link').symlink_to(outside,target_is_directory=True)
            with self.assertRaises(ValueError):target(root,'link/escape')
            for name in ['/etc/passwd','../escape','a/../../b','a\\b']:
                with self.assertRaises(ValueError):target(root,name)

    def test_manifest_relocation_preserves_image_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);image=root/'face.png';image.write_bytes(b'fake content for path mapping test')
            folder=root/'deployment/shared-research/manifests';folder.mkdir(parents=True)
            row={}
            for k,h in [('image_path','sha256'),('reference_path','reference_sha256'),('reference512_path','reference512_sha256')]:row[k]='face.png';row[h]=digest(image)
            for split in ['train','eval']:(folder/(split+'.portable.jsonl')).write_text(json.dumps(row)+'\n')
            configure(root,['data'])
            restored=json.loads((folder/'train.jsonl').read_text());self.assertEqual(restored['image_path'],str(image));self.assertEqual(restored['sha256'],row['sha256'])
            self.assertNotEqual(digest(folder/'train.jsonl'),digest(folder/'train.portable.jsonl'))


if __name__=='__main__':unittest.main()
