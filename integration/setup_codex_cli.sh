#!/usr/bin/env bash
# Reproduce MoBei's relay configuration without storing credentials in the repo.
set -euo pipefail

start_session=0
case "${1:-}" in
  '') ;;
  --start) start_session=1 ;;
  *) echo 'Usage: bash integration/setup_codex_cli.sh [--start]' >&2; exit 2 ;;
esac

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" != Linux ]]; then
  echo 'This installer targets Linux servers.' >&2
  exit 1
fi
for dependency in python3 curl tar; do
  if ! command -v "$dependency" >/dev/null; then
    echo "Missing prerequisite: $dependency. Install it with your server package manager." >&2
    exit 1
  fi
done

codex_bin="$(command -v codex || true)"
if [[ -z "$codex_bin" && -x "$HOME/.local/bin/codex" ]]; then
  codex_bin="$HOME/.local/bin/codex"
fi
if [[ -z "$codex_bin" ]]; then
  case "$(uname -m)" in
    x86_64) codex_arch=x86_64 ;;
    aarch64|arm64) codex_arch=aarch64 ;;
    *) echo 'Supported architectures: x86_64 and aarch64.' >&2; exit 1 ;;
  esac
  # Match the working CLI on the original server. No Node.js or sudo needed.
  codex_version=0.167.1
  codex_asset="codex-${codex_arch}-unknown-linux-musl"
  download_dir="$(mktemp -d)"
  trap 'rm -rf -- "$download_dir"' EXIT
  echo "Installing Codex CLI $codex_version to ~/.local/bin ..."
  curl --fail --location --silent --show-error --retry 2 --connect-timeout 15 \
    --output "$download_dir/codex.tar.gz" \
    "https://github.com/openai/codex/releases/download/rust-v${codex_version}/${codex_asset}.tar.gz"
  tar -xzf "$download_dir/codex.tar.gz" -C "$download_dir" -- "$codex_asset"
  mkdir -p "$HOME/.local/bin"
  install -m 755 "$download_dir/$codex_asset" "$HOME/.local/bin/codex"
  codex_bin="$HOME/.local/bin/codex"
fi
"$codex_bin" --version

relay_dir="$HOME/.config/face-age-codex"
if [[ -z "${SPACETIMEAI_API_KEY:-}" && -f "$relay_dir/env.sh" ]]; then
  # This file was created by this script and is private to the current user.
  source "$relay_dir/env.sh"
fi
if [[ -z "${SPACETIMEAI_API_KEY:-}" ]]; then
  if [[ ! -r /dev/tty ]]; then
    echo 'Set SPACETIMEAI_API_KEY in the environment, or rerun in an interactive terminal.' >&2
    exit 1
  fi
  IFS= read -r -s -p 'SpacetimeAI API Key (hidden): ' SPACETIMEAI_API_KEY < /dev/tty
  printf '\n'
fi
if [[ -z "$SPACETIMEAI_API_KEY" ]]; then
  echo 'The relay API key cannot be empty.' >&2
  exit 1
fi
export SPACETIMEAI_API_KEY
export PATH="$HOME/.local/bin:$PATH"

python3 - <<'PY'
import os
from pathlib import Path
import re
import shlex
import shutil
import tempfile
from datetime import datetime, timezone

key = os.environ['SPACETIMEAI_API_KEY']
if '\n' in key or '\r' in key:
    raise SystemExit('API key must be a single line')
root = Path(os.environ.get('CODEX_HOME', str(Path.home() / '.codex')))
root.mkdir(parents=True, exist_ok=True)
config = root / 'config.toml'
original = config.read_text() if config.exists() else ''
# Retain unrelated settings, profiles, tools and trusted projects. Replace only
# the three global selection keys and our provider's table. Refuse unusual
# multiline TOML before editing it instead of guessing its section boundaries.
if '"""' in original or ("'" * 3) in original:
    raise SystemExit('Config contains multiline TOML; preserve it and adapt the three model settings manually.')
try:
    import tomllib
except ImportError:
    tomllib = None  # Python3.10 is supported; the CLI also validates the TOML.
if tomllib:
    tomllib.loads(original)
selection = {'model', 'model_provider', 'model_reasoning_effort'}
kept = []
top_level = True
skip_provider = False
for line in original.splitlines(keepends=True):
    stripped = line.strip()
    if stripped.startswith('['):
        top_level = False
        table = stripped.split('#', 1)[0].strip()
        skip_provider = table in (
            '[model_providers.spacetimeai]',
            '[model_providers."spacetimeai"]',
            "[model_providers.'spacetimeai']",
        ) or table.startswith('[model_providers.spacetimeai.')
    if skip_provider:
        continue
    match = re.match(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=', line)
    if top_level and match and match.group(1) in selection:
        continue
    kept.append(line)
updated = '''model = "gpt-6.1-sol"
model_provider = "spacetimeai"
model_reasoning_effort = "xhigh"
''' + ''.join(kept).rstrip() + '''

[model_providers.spacetimeai]
name = "SpacetimeAI"
base_url = "https://spacetimeai.cc"
env_key = "SPACETIMEAI_API_KEY"
wire_api = "responses"
requires_openai_auth = false
'''
if tomllib:
    tomllib.loads(updated)
if config.exists() and original != updated:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    shutil.copy2(config, root / ('config.toml.before-face-age-' + stamp))
    (root / ('config.toml.before-face-age-' + stamp)).chmod(0o600)

def private_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
        stream.write(text)
        temp = Path(stream.name)
    temp.chmod(0o600)
    temp.replace(path)

private_write(config, updated)
relay = Path.home() / '.config/face-age-codex'
relay.mkdir(parents=True, exist_ok=True)
relay.chmod(0o700)
private_write(relay / 'env.sh', 'export SPACETIMEAI_API_KEY=' + shlex.quote(key) + '\n')
hook = '''
# Face-age project: local Codex executable and private relay key.
export PATH="$HOME/.local/bin:$PATH"
[ ! -f "$HOME/.config/face-age-codex/env.sh" ] || . "$HOME/.config/face-age-codex/env.sh"
'''
# Interactive Bash/Zsh and the relevant login files all load the same private
# file. The API key itself never appears in shell history or in these profiles.
profiles = [Path.home() / name for name in ['.bashrc', '.zshrc', '.profile']]
profiles += [p for name in ['.bash_profile', '.bash_login'] if (p := Path.home() / name).exists()]
for profile in profiles:
    old = profile.read_text() if profile.exists() else ''
    if '# Face-age project: local Codex executable and private relay key.' not in old:
        with profile.open('a') as stream:
            stream.write(hook)
print('Configured https://spacetimeai.cc / gpt-6.1-sol / xhigh / Responses.')
print('Saved a private relay key outside the repository; existing Codex config was backed up if changed.')
PY

# Parses the actual installed CLI config without sending a model request.
"$codex_bin" features list >/dev/null
echo 'Codex CLI is ready. New terminal sessions can run: codex'
echo 'No ChatGPT login is required for this provider. Existing approval settings were preserved.'
if [[ "$start_session" == 1 ]]; then
  cd "$repo_dir"
  exec "$codex_bin" '先阅读 integration/迁移与研究接手提示词_20261116.md，并按其中的迁移和研究交接继续执行。先核查新机环境和已提供文件，恢复真实模型与基线；不要把已有失败实验当成效果提升。'
fi
