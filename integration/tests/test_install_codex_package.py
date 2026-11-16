"""Package integrity/layout regressions; no downloads or real API keys."""
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[1]/'install_codex_package.py'
spec=importlib.util.spec_from_file_location('codex_package_installer',SCRIPT)
installer=importlib.util.module_from_spec(spec);spec.loader.exec_module(installer)


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.target='x86_64-unknown-linux-musl'
        self.prefix='package/vendor/'+self.target+'/'
        manifest=dict(version='0.167.1',target=self.target,entrypoint='bin/codex',
                      resourcesDir='codex-resources',pathDir='codex-path')
        self.files={self.prefix+'codex-package.json':json.dumps(manifest).encode(),
                    self.prefix+'bin/codex':b'#!/bin/sh\nexit 0\n',
                    self.prefix+'bin/codex-code-mode-host':b'#!/bin/sh\nexit 0\n',
                    self.prefix+'codex-path/rg':b'#!/bin/sh\nexit 0\n',
                    self.prefix+'codex-resources/resource.txt':b'fixture resource'}
        self.archive=self.root/'official-fixture.tgz'
        with tarfile.open(self.archive,'w:gz') as t:
            for name,data in self.files.items():
                info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o755
                t.addfile(info,io.BytesIO(data))
        integrity=base64.b64encode(hashlib.sha512(self.archive.read_bytes()).digest()).decode()
        self.metadata=dict(name='@openai/codex',version='0.167.1-linux-x64',
            dist=dict(tarball='https://registry.npmjs.org/@openai/codex/-/codex-0.167.1-linux-x64.tgz',integrity='sha512-'+integrity))

    def test_missing_code_host_rejected_even_when_main_binary_exists(self):
        installer.extract_verified(self.archive,self.metadata,self.root/'payload')
        root=self.root/'payload/package/vendor'/self.target
        installer.layout(root,self.target)
        (root/'bin/codex-code-mode-host').unlink()
        with self.assertRaisesRegex(ValueError,'Incomplete executable'):installer.layout(root,self.target)

    def test_reject_integrity_mismatch_before_extracting(self):
        changed=dict(self.metadata,dist=dict(self.metadata['dist'],integrity='sha512-bad'))
        with self.assertRaisesRegex(ValueError,'SHA512 mismatch'):
            installer.extract_verified(self.archive,changed,self.root/'payload')
        self.assertFalse((self.root/'payload').exists())

    def test_old_wrapper_replaced_by_complete_layout_with_backup(self):
        home=self.root/'home';old=home/'.local/bin/codex';old.parent.mkdir(parents=True)
        old.write_text('#!/bin/sh\n# old daemon-free wrapper\n');old.chmod(0o755)
        calls=[]
        def download(url,path):
            calls.append(url)
            if url.endswith('0.167.1-linux-x64'):Path(path).write_text(json.dumps(self.metadata))
            else:Path(path).write_bytes(self.archive.read_bytes())
        with patch.dict(os.environ,{'HOME':str(home)}),patch.object(installer.platform,'machine',return_value='x86_64'),patch.object(installer,'download',side_effect=download):
            result=installer.ensure_package()
            self.assertEqual(result,old)
            self.assertTrue(old.is_symlink())
            self.assertTrue((old.resolve().parent/'codex-code-mode-host').is_file())
            backups=list(old.parent.glob('codex.before-full-package-*'))
            self.assertEqual(len(backups),1)
            self.assertIn('old daemon-free wrapper',backups[0].read_text())
            self.assertEqual(len(calls),2)
            installer.ensure_package()
            self.assertEqual(len(calls),2,'Complete package should not download again')


if __name__=='__main__':unittest.main()
