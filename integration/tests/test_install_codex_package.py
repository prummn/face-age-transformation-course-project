"""Package integrity/layout regressions; no downloads or real API keys."""
import base64
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import threading
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
        def download(url,path,**kwargs):
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

    def test_failed_install_keeps_partial_download_for_next_run(self):
        home = self.root / 'home'
        archive_bytes = self.archive.read_bytes()
        prefix_size = len(archive_bytes) // 2
        archive_paths = []

        def transfer(url, path, **kwargs):
            if url.endswith('0.167.1-linux-x64'):
                Path(path).write_text(json.dumps(self.metadata))
                return
            archive_paths.append(path)
            if len(archive_paths) == 1:
                path.write_bytes(archive_bytes[:prefix_size])
                raise installer.DownloadError('simulated interrupted connection')
            self.assertTrue(kwargs['resume'])
            self.assertEqual(path.read_bytes(), archive_bytes[:prefix_size])
            with path.open('ab') as output:
                output.write(archive_bytes[prefix_size:])

        with (patch.dict(os.environ, {'HOME': str(home)}),
              patch.object(installer.platform, 'machine', return_value='x86_64'),
              patch.object(installer, 'download', side_effect=transfer),
              patch.object(installer.shutil, 'which', return_value=None)):
            with self.assertRaises(installer.DownloadError):
                installer.ensure_package()
            self.assertTrue(archive_paths[0].is_file(), 'Temporary-directory cleanup must retain partial bytes')
            self.assertFalse((home / '.local/bin/codex').exists())
            result = installer.ensure_package()
            self.assertTrue(result.is_symlink())
            self.assertEqual(archive_paths[0], archive_paths[1])
            self.assertFalse(archive_paths[0].exists(), 'Verified partial file becomes the complete archive')
            self.assertTrue(archive_paths[0].with_suffix('').is_file())

    def test_damaged_partial_is_restarted_and_verified(self):
        cache = self.root / 'cache/platform.tgz'
        cache.parent.mkdir()
        partial = cache.with_name(cache.name + '.part')
        partial.write_bytes(b'corrupt prefix')
        calls = []

        def transfer(url, path, **kwargs):
            calls.append(path)
            if len(calls) == 1:
                with path.open('ab') as output:
                    output.write(self.archive.read_bytes())
            else:
                self.assertFalse(path.exists(), 'A damaged prefix must be removed before retrying')
                path.write_bytes(self.archive.read_bytes())

        with patch.object(installer, 'download', side_effect=transfer):
            installer.cached_archive(self.metadata['dist']['tarball'], self.metadata, cache)
        self.assertEqual(len(calls), 2)
        self.assertEqual(cache.read_bytes(), self.archive.read_bytes())
        with patch.object(installer, 'download') as transfer:
            installer.cached_archive(self.metadata['dist']['tarball'], self.metadata, cache)
        transfer.assert_not_called()


@contextmanager
def interrupted_server(payload, interruptions=1, ignore_range=False):
    state = {'requests': [], 'interruptions': interruptions}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass

        def do_GET(self):
            requested = self.headers.get('Range')
            state['requests'].append(requested)
            offset = int(requested.removeprefix('bytes=').removesuffix('-')) if requested else 0
            if ignore_range:
                offset = 0
            self.send_response(206 if offset else 200)
            if offset:
                self.send_header('Content-Range', f'bytes {offset}-{len(payload) - 1}/{len(payload)}')
            self.send_header('Content-Length', str(len(payload) - offset))
            self.end_headers()
            end = offset + 512 if len(state['requests']) <= state['interruptions'] else len(payload)
            try:
                self.wfile.write(payload[offset:end])
                self.wfile.flush()
            except OSError:
                pass  # curl closes an ignored Range response before reading it.
            self.close_connection = True

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with patch.dict(os.environ, {'NO_PROXY': '127.0.0.1', 'no_proxy': '127.0.0.1'}):
            yield f'http://127.0.0.1:{server.server_port}/package.tgz', state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class DownloadTests(unittest.TestCase):
    def test_real_curl_resumes_after_partial_transfer(self):
        payload = b'official-package-fixture' * 4096
        with tempfile.TemporaryDirectory() as folder, interrupted_server(payload) as (url, state):
            path = Path(folder) / 'platform.tgz.part'
            with patch.object(installer.time, 'sleep'):
                installer.download(url, path, resume=True)
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(state['requests'], [None, 'bytes=512-'])

    def test_real_curl_keeps_bytes_after_exhausted_retries_and_resumes_on_rerun(self):
        payload = b'official-package-fixture' * 4096
        with tempfile.TemporaryDirectory() as folder, interrupted_server(payload, interruptions=6) as (url, state):
            path = Path(folder) / 'platform.tgz.part'
            with patch.object(installer.time, 'sleep'):
                with self.assertRaisesRegex(installer.DownloadError, 'Partial file kept'):
                    installer.download(url, path, resume=True)
                offset = path.stat().st_size
                self.assertEqual(path.read_bytes(), payload[:offset])
                self.assertEqual(offset, 6 * 512)
                state['interruptions'] = 0
                installer.download(url, path, resume=True)
            self.assertEqual(state['requests'][-1], f'bytes={offset}-')
            self.assertEqual(path.read_bytes(), payload)

    def test_real_curl_restarts_if_server_ignores_range(self):
        payload = b'official-package-fixture' * 4096
        with tempfile.TemporaryDirectory() as folder, interrupted_server(payload, ignore_range=True) as (url, state):
            path = Path(folder) / 'platform.tgz.part'
            with patch.object(installer.time, 'sleep'):
                installer.download(url, path, resume=True)
            self.assertEqual(state['requests'], [None, 'bytes=512-', None])
            self.assertEqual(path.read_bytes(), payload)


if __name__=='__main__':unittest.main()
