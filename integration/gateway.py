"""Local, same-origin bridge from the existing prototype to the Ascend API."""
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

PROTOTYPE = Path(__file__).resolve().parents[1] / "原型"
MAX_BODY = 24 * 1024 * 1024


class Gateway(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".mjs": "text/javascript", ".wasm": "application/wasm"}

    def __init__(self, *args, upstream, **kwargs):
        self.upstream = upstream.rstrip("/")
        super().__init__(*args, directory=str(PROTOTYPE), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def json_error(self, status, message):
        raw = json.dumps({"detail": message}, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def proxy(self):
        path = urlsplit(self.path).path
        allowed = (path == "/api/health" and self.command == "GET"
                   or path in ("/api/prepare", "/api/transform") and self.command == "POST"
                   or path.startswith("/api/sessions/") and self.command == "DELETE")
        if not allowed:
            self.json_error(404, "接口不存在。")
            return
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            self.json_error(403, "请从本机演示页面操作。")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.json_error(400, "请求长度无效。")
            return
        if length < 0 or length > MAX_BODY:
            self.json_error(413, "图片过大，请换一张较小的照片。")
            return
        self.connection.settimeout(30)
        try:
            body = self.rfile.read(length) if length else None
            if body is not None and len(body) != length:
                self.json_error(400, "图片传输未完成，请重试。")
                return
            req = Request(self.upstream + path, data=body, method=self.command,
                          headers={"Content-Type": "application/json"})
            try:
                response = urlopen(req, timeout=150 if self.command == "POST" else 8)
            except HTTPError as exc:
                response = exc
            with response:
                data = response.read()
                self.send_response(response.status)
                self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass  # User cancelled or closed this request.
        except (URLError, TimeoutError, OSError):
            try:
                self.json_error(503, "生成服务暂未连接。请检查 SSH 连接后点击重新连接。")
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_GET(self):
        if self.path.startswith("/api/"):
            return self.proxy()
        if urlsplit(self.path).path == "/":
            self.send_response(302)
            self.send_header("Location", "/" + quote("人脸年龄变换原型_v5.html") + "?mode=live")
            self.end_headers()
            return
        try:
            super().do_GET()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        self.proxy()

    def do_DELETE(self):
        self.proxy()


def make_server(port=65512, upstream="http://127.0.0.1:18765"):
    return ThreadingHTTPServer(("127.0.0.1", port), partial(Gateway, upstream=upstream))
