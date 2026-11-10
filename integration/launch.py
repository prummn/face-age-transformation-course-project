"""Start the classroom prototype, its SSH tunnel, and the isolated Ascend API."""
import argparse
import json
import shlex
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from urllib.error import URLError
from urllib.request import urlopen

from gateway import make_server

REMOTE_DIR = "/data/home/1120231322/scow/face-age-integration-20261110"


def health(url):
    try:
        with urlopen(url + "/api/health", timeout=3) as response:
            return json.load(response)
    except (OSError, URLError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="bit-ascend")
    parser.add_argument("--remote-dir", default=REMOTE_DIR)
    parser.add_argument("--remote-port", default=8765, type=int)
    parser.add_argument("--port", default=65512, type=int)
    parser.add_argument("--upstream", help="Use an existing local API; skip SSH and remote startup")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    tunnel = None
    server = None
    try:
        existing = health(f"http://127.0.0.1:{args.port}")
        if existing and existing.get("ready") and existing.get("version") == "1.2.0" and existing.get("engine") == "SAM / StyleGAN2":
            url = f"http://localhost:{args.port}/"
            print(f"整合演示已在运行：{url}", flush=True)
            if not args.no_browser:
                webbrowser.open(url)
            return 0
        # Reserve the page port before starting remote work.
        server = make_server(args.port, args.upstream or "http://127.0.0.1:1")
        if args.upstream:
            upstream = args.upstream.rstrip("/")
        else:
            print("正在通过 SSH 启动昇腾生成服务…", flush=True)
            subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", args.host,
                            "bash " + shlex.quote(args.remote_dir + "/start_backend.sh")],
                           check=True, timeout=35)
            with socket.socket() as free:
                free.bind(("127.0.0.1", 0))
                tunnel_port = free.getsockname()[1]
            upstream = f"http://127.0.0.1:{tunnel_port}"
            tunnel = subprocess.Popen([
                "ssh", "-N", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15",
                "-o", "ServerAliveCountMax=3", "-L",
                f"127.0.0.1:{tunnel_port}:127.0.0.1:{args.remote_port}", args.host])
        server.RequestHandlerClass.keywords["upstream"] = upstream
        print("正在检查模型状态；首次加载可能需要几分钟。", flush=True)
        deadline = time.monotonic() + 300
        last_status = None
        while time.monotonic() < deadline:
            if tunnel and tunnel.poll() is not None:
                raise RuntimeError("SSH 隧道已断开。请先检查 ssh bit-ascend 是否可用。")
            result = health(upstream)
            if result and result.get("ready"):
                print("真实模型已就绪，原生输出 1024×1024。", flush=True)
                break
            status = result.get("status", "加载中") if result else "等待服务连接"
            if status != last_status:
                print(status, flush=True)
                last_status = status
            if result and result.get("status") in ("error", "failed"):
                raise RuntimeError("模型加载失败，请检查服务器 integration/logs/backend.log。")
            time.sleep(2)
        else:
            raise RuntimeError("模型仍未就绪，请检查服务器日志后重新启动。")
        url = f"http://localhost:{args.port}/"
        print(f"演示入口：{url}\n保持本窗口开启；按 Ctrl+C 结束本机页面与 SSH 隧道。", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        if tunnel:
            def monitor_tunnel():
                tunnel.wait()
                print("SSH 连接已结束，请重新运行启动程序以恢复演示。", flush=True)
                server.shutdown()
            threading.Thread(target=monitor_tunnel, daemon=True).start()
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\n本机演示已结束。", flush=True)
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        print(f"启动未完成：{exc}", file=sys.stderr)
        return 1
    finally:
        if server:
            server.server_close()
        if tunnel:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
