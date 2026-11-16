"""Exercise private relay persistence/config migration without network or real keys."""
import os
from pathlib import Path
import stat
import json
import platform
import shutil
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'setup_codex_cli.sh'


class RelaySetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        fixtures = self.root / 'repository/integration'
        fixtures.mkdir(parents=True)
        for name in ['setup_codex_cli.sh', 'install_codex_package.py']:
            shutil.copy2(SCRIPT.parent / name, fixtures / name)
        # Runtime execution is checked separately with the real packaged CLI;
        # these tests isolate config/key persistence and launch orchestration.
        (fixtures / 'verify_codex_cli.py').write_text('print("fixture runtime check")\n')
        self.script = fixtures / 'setup_codex_cli.sh'
        package = self.root / 'complete-package'
        (package / 'bin').mkdir(parents=True)
        (package / 'codex-resources').mkdir()
        (package / 'codex-path').mkdir()
        arch = 'x86_64' if platform.machine()=='x86_64' else 'aarch64'
        (package / 'codex-package.json').write_text(json.dumps(dict(version='0.167.1',target=arch+'-unknown-linux-musl',entrypoint='bin/codex',resourcesDir='codex-resources',pathDir='codex-path')))
        for name in ['bin/codex','bin/codex-code-mode-host','codex-path/rg']:
            f = package / name
            f.write_text('#!/bin/sh\ncase "$1" in --version) echo codex-test;; features) exit 0;; *) exit 2;; esac\n')
            f.chmod(0o755)
        binary = self.root / '.local/bin/codex'
        binary.parent.mkdir(parents=True)
        binary.symlink_to(package/'bin/codex')
        self.env = dict(os.environ, HOME=str(self.root), CODEX_HOME=str(self.root / '.codex'),
                        PATH=str(binary.parent) + ':' + os.environ['PATH'],
                        SPACETIMEAI_API_KEY="test-only key with 'quotes' and $literal")
        self.config = self.root / '.codex/config.toml'
        self.config.parent.mkdir()

    def run_setup(self, *args):
        return subprocess.run(['bash', str(self.script), *args], env=self.env, capture_output=True,
                              text=True, timeout=30)

    def test_preserves_other_settings_backs_up_and_keeps_secret_private(self):
        original = '''model = "previous"
model_provider = "old"
approval_policy = "on-request"
[mcp_servers.keep_me]
command = "example"
[model_providers.spacetimeai]
base_url = "https://old.invalid"
wire_api = "chat"
'''
        self.config.write_text(original)
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        updated = self.config.read_text()
        self.assertIn('model = "gpt-6.1-sol"', updated)
        self.assertIn('wire_api = "responses"', updated)
        self.assertIn('base_url = "https://spacetimeai.cc"', updated)
        self.assertIn('[mcp_servers.keep_me]', updated)
        self.assertIn('approval_policy = "on-request"', updated)
        self.assertNotIn('old.invalid', updated)
        backups = list(self.config.parent.glob('config.toml.before-face-age-*'))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(), original)
        keyfile = self.root / '.config/face-age-codex/env.sh'
        self.assertEqual(stat.S_IMODE(keyfile.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(keyfile.parent.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(self.config.stat().st_mode), 0o600)
        key = self.env['SPACETIMEAI_API_KEY']
        self.assertNotIn(key, updated + result.stdout + result.stderr)
        for name in ['.bashrc', '.zshrc', '.profile']:
            self.assertNotIn(key, (self.root / name).read_text())
        loaded = subprocess.run(['bash', '-c', '. "$HOME/.config/face-age-codex/env.sh"; printf %s "$SPACETIMEAI_API_KEY"'],
                                env=dict(self.env, SPACETIMEAI_API_KEY=''),
                                capture_output=True, text=True, check=True)
        self.assertEqual(loaded.stdout, key)

    def test_rerun_reuses_private_key_without_duplicating_profile_hooks(self):
        self.assertEqual(self.run_setup().returncode, 0)
        self.env.pop('SPACETIMEAI_API_KEY')
        self.assertEqual(self.run_setup().returncode, 0)
        for name in ['.bashrc', '.zshrc', '.profile']:
            self.assertEqual((self.root / name).read_text().count('# Face-age project:'), 1)

    def test_unusual_existing_config_is_left_intact(self):
        original = 'instructions = """a multiline setting"""\n'
        self.config.write_text(original)
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.config.read_text(), original)

    def test_start_uses_supported_daemon_free_flag_and_passes_handoff_prompt(self):
        binary = self.root / '.local/bin/codex'
        binary.write_text('''#!/bin/sh
case "$1" in
  --version) echo codex-test;;
  features) exit 0;;
  --no-daemon)
    case "$2" in *integration/迁移与研究接手提示词_20261116.md*) echo HANDOFF_OK;; *) exit 3;; esac;;
  *) echo 'no complete local package' >&2; exit 2;;
esac
''')
        result = self.run_setup('--start')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('HANDOFF_OK', result.stdout)

    def test_resume_reuses_saved_key_and_passes_session_id(self):
        self.assertEqual(self.run_setup().returncode, 0)
        self.env.pop('SPACETIMEAI_API_KEY')
        binary = self.root / '.local/bin/codex'
        binary.write_text('''#!/bin/sh
case "$1" in
  --version) echo codex-test;;
  features) exit 0;;
  --no-daemon)
    [ "$2" = resume ] && [ "$3" = test-session-id ] && echo RESUME_OK;;
  *) exit 2;;
esac
''')
        result = self.run_setup('--resume', 'test-session-id')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('RESUME_OK', result.stdout)


if __name__ == '__main__':
    unittest.main()
