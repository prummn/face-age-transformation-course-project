#!/usr/bin/env bash
# Scope the documented YiCloud developer egress proxy to one command only.
set -euo pipefail
network_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${FACE_AGE_HTTP_PROXY:-}" ]]; then
  HTTP_PROXY="$FACE_AGE_HTTP_PROXY"
else
  HTTP_PROXY="$(python3 "$network_script_dir/select_yicloud_proxy.py")"
fi
export HTTP_PROXY
export HTTPS_PROXY="$HTTP_PROXY" http_proxy="$HTTP_PROXY" https_proxy="$HTTP_PROXY"
export NO_PROXY="localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,.svc,.cluster.local,.shzhisuan.local,gate.yicloud.com.cn,registry.gate.yicloud.com.cn"
export no_proxy="$NO_PROXY"
exec "$@"
