"""Install the complete official Linux platform package, preserving relay/session data.

The npm platform archive includes native binaries and resources and can be
installed without Node.js. Do not move its main executable out of that layout.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.parse import urlsplit


VERSION = '0.167.1'


def file_hash(path, algorithm='sha256'):
    result = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result


def layout(root, target):
    """Reject the old one-binary and daemon-free-wrapper installations."""
    manifest = json.loads((root / 'codex-package.json').read_text())
    if manifest.get('version') != VERSION or manifest.get('target') != target:
        raise ValueError('Unexpected Codex package version/architecture')
    resolved = root.resolve()

    def member(name):
        path = (root / name).resolve()
        if not path.is_relative_to(resolved):
            raise ValueError('Package manifest path escapes its root')
        return path

    entry = member(manifest['entrypoint'])
    binaries = [entry, entry.parent / 'codex-code-mode-host']
    # The official npm platform package embeds exec-server as a CLI subcommand;
    # the standalone installer may additionally ship a dedicated executable.
    if manifest.get('execServer'):
        binaries.append(member(manifest['execServer']))
    for binary in binaries:
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError('Incomplete executable set: ' + str(binary))
    resources = member(manifest['resourcesDir'])
    search = member(manifest['pathDir']) / 'rg'
    if not resources.is_dir() or not search.is_file() or not os.access(search, os.X_OK):
        raise ValueError('Missing package resources or bundled rg')
    return entry, binaries


def existing_package(executable, target):
    if not executable:
        return None
    entry = Path(executable).resolve()
    for parent in list(entry.parents)[:3]:
        try:
            actual, _ = layout(parent, target)
            if actual == entry:
                return actual
        except (OSError, ValueError, KeyError):
            pass
    return None


def extract_verified(archive, metadata, destination):
    integrity = metadata['dist']['integrity']
    if not integrity.startswith('sha512-'):
        raise ValueError('Official package SHA512 integrity is required')
    if base64.b64encode(file_hash(archive, 'sha512').digest()).decode() != integrity[7:]:
        raise ValueError('Codex package SHA512 mismatch')
    with tarfile.open(archive, 'r:gz') as package:
        members = package.getmembers()
        if len(members) > 512 or sum(m.size for m in members) > 1536 * 1024**2:
            raise ValueError('Unexpected platform archive size')
        for info in members:
            parts = PurePosixPath(info.name).parts
            if not parts or parts[0] != 'package' or '..' in parts or '\\' in info.name:
                raise ValueError('Invalid package member path')
            path = destination.joinpath(*parts)
            if info.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif info.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                with package.extractfile(info) as source, path.open('wb') as output:
                    shutil.copyfileobj(source, output)
                path.chmod(info.mode & 0o777)
            else:
                raise ValueError('Unexpected link/device in platform package')


def download(url, path):
    subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                    '--retry', '2', '--connect-timeout', '15', '--max-time', '600',
                    '--output', str(path), url], check=True, stdout=sys.stderr)


def ensure_package(archive=None, metadata_file=None):
    if platform.system() != 'Linux':
        raise ValueError('Only Linux is supported by this installer')
    cpu = {'x86_64': 'x64', 'aarch64': 'arm64', 'arm64': 'arm64'}.get(platform.machine())
    if cpu is None:
        raise ValueError('Unsupported CPU architecture')
    arch = 'x86_64' if cpu == 'x64' else 'aarch64'
    target = arch + '-unknown-linux-musl'
    version = VERSION + '-linux-' + cpu
    link = Path.home() / '.local/bin/codex'
    found = existing_package(str(link) if link.exists() else shutil.which('codex'), target)
    if found and archive is None:
        print('Reusing complete Codex package: ' + str(found.parent.parent), file=sys.stderr)
        return found
    packages = Path.home() / '.local/share/face-age-codex/packages'
    packages.mkdir(parents=True, exist_ok=True)
    final = packages / version
    root = final / 'package/vendor' / target
    try:
        entry, _ = layout(root, target)
    except (OSError, ValueError, KeyError):
        print('Installing complete Codex platform package ' + version + ' (about166MiB download).', file=sys.stderr)
        with tempfile.TemporaryDirectory(prefix='.install-', dir=packages) as temporary:
            stage = Path(temporary)
            if archive is None:
                metadata_file = stage / 'registry-metadata.json'
                download('https://registry.npmjs.org/@openai%2Fcodex/' + version, metadata_file)
            elif metadata_file is None:
                raise ValueError('--archive requires official registry --metadata for integrity verification')
            metadata = json.loads(Path(metadata_file).read_text())
            if metadata.get('name') != '@openai/codex' or metadata.get('version') != version:
                raise ValueError('Registry package identity mismatch')
            url = metadata['dist']['tarball']
            parsed = urlsplit(url)
            if parsed.scheme != 'https' or parsed.netloc != 'registry.npmjs.org':
                raise ValueError('Unexpected registry archive URL')
            if archive is None:
                archive = stage / 'platform.tgz'
                download(url, archive)
            payload = stage / 'payload'
            extract_verified(archive, metadata, payload)
            staged_root = payload / 'package/vendor' / target
            _, binaries = layout(staged_root, target)
            # Check both tool host executables before publishing the installation.
            for binary in binaries:
                flag = '--version' if binary.name == 'codex' else '--help'
                subprocess.run([str(binary), flag], check=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, timeout=20)
            subprocess.run([str(binaries[0]), 'exec-server', '--help'], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=20)
            receipt = dict(version=version, source=url, integrity=metadata['dist']['integrity'],
                           binaries={str(p.relative_to(payload)): file_hash(p).hexdigest() for p in binaries})
            (payload / 'installation.json').write_text(json.dumps(receipt, indent=2) + '\n')
            if final.exists():
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                final.rename(packages / (version + '.incomplete-' + stamp))
            payload.rename(final)
            entry, _ = layout(root, target)
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.is_symlink() or link.resolve() != entry:
        temp = link.with_name('.codex-package-link-' + str(os.getpid()))
        temp.symlink_to(entry)
        if link.exists() or link.is_symlink():
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            shutil.copy2(link, link.with_name('codex.before-full-package-' + stamp), follow_symlinks=False)
        temp.replace(link)
    print('Complete package verified: CLI, code-mode host, execution service, resources and rg.', file=sys.stderr)
    return link


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, help='Reuse an official cached platform archive')
    p.add_argument('--metadata', type=Path, help='Its official npm registry metadata, including SHA512')
    a = p.parse_args()
    print(ensure_package(a.archive, a.metadata))


if __name__ == '__main__':
    main()
