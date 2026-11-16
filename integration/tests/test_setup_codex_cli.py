"""Exercise private relay persistence/config migration without network or real keys."""
import os
from pathlib import Path
import stat
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
        binary = self.root / '.local/bin/codex'
        binary.parent.mkdir(parents=True)
        binary.write_text('#!/bin/sh\ncase "$1" in --version) echo codex-test;; features) exit 0;; *) exit 2;; esac\n')
        binary.chmod(0o755)
        self.env = dict(os.environ, HOME=str(self.root), CODEX_HOME=str(self.root / '.codex'),
                        PATH=str(binary.parent) + ':' + os.environ['PATH'],
                        SPACETIMEAI_API_KEY="test-only key with 'quotes' and $literal")
        self.config = self.root / '.codex/config.toml'
        self.config.parent.mkdir()

    def run_setup(self, *args):
        return subprocess.run(['bash', str(SCRIPT), *args], env=self.env, capture_output=True,
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

    @unittest.skipUnless(shutil.which('cc'), 'ELF repair fixture requires a C compiler')
    def test_repairs_previous_single_binary_install_and_reuses_key(self):
        self.assertEqual(self.run_setup().returncode, 0)
        self.env.pop('SPACETIMEAI_API_KEY')
        source = self.root / 'codex-fixture.c'
        source.write_text(r'''
#include <stdio.h>
#include <string.h>
int main(int argc, char **argv) {
    int daemon_free = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--version")) { puts("codex-test"); return 0; }
        if (!strcmp(argv[i], "features")) return 0;
        if (!strcmp(argv[i], "--no-daemon")) daemon_free++;
    }
    if (daemon_free == 1) { puts("DAEMON_FREE_OK"); return 0; }
    fputs("no complete local package\n", stderr);
    return 2;
}
''')
        binary = self.root / '.local/bin/codex'
        subprocess.run(['cc', str(source), '-o', str(binary)], check=True, capture_output=True)
        before = subprocess.run([str(binary), 'test prompt'], capture_output=True, text=True)
        self.assertNotEqual(before.returncode, 0)
        result = self.run_setup('--start')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('DAEMON_FREE_OK', result.stdout)
        self.assertTrue(binary.read_text().startswith('#!/bin/sh'))
        later = subprocess.run([str(binary), 'later session'], capture_output=True, text=True)
        self.assertEqual(later.returncode, 0)
        self.assertIn('DAEMON_FREE_OK', later.stdout)
        # A rerun retains the launcher and does not move another native binary.
        self.assertEqual(self.run_setup().returncode, 0)
        self.assertEqual(len(list((self.root / '.local/lib/face-age-codex').glob('codex-standalone-*'))), 1)


if __name__ == '__main__':
    unittest.main()
