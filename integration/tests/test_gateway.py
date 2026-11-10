import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gateway import make_server


class FakeAPI(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ready":true}')

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(422 if body == b"{}" else 200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


class GatewayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = ThreadingHTTPServer(("127.0.0.1", 0), FakeAPI)
        cls.gateway = make_server(0, f"http://127.0.0.1:{cls.api.server_port}")
        cls.base = f"http://127.0.0.1:{cls.gateway.server_port}"
        for server in (cls.api, cls.gateway):
            threading.Thread(target=server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        for server in (cls.api, cls.gateway):
            server.shutdown()
            server.server_close()

    def test_root_is_explicit_live_mode(self):
        with urlopen(self.base) as response:
            self.assertIn("?mode=live", response.url)
            self.assertIn("no-store", response.headers["Cache-Control"])
            response.read()

    def test_health_and_unicode_payload_are_proxied(self):
        with urlopen(self.base + "/api/health") as response:
            self.assertTrue(json.load(response)["ready"])
        raw = json.dumps({"image": "中文测试"}, ensure_ascii=False).encode()
        with urlopen(Request(self.base + "/api/prepare", data=raw)) as response:
            self.assertEqual(response.read(), raw)

    def test_upstream_validation_error_is_preserved(self):
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(self.base + "/api/prepare", data=b"{}"))
        self.assertEqual(error.exception.code, 422)
        self.assertEqual(error.exception.read(), b"{}")
        error.exception.close()

    def test_cross_origin_and_unknown_api_rejected(self):
        for path, headers, status in (
            ("/api/prepare", {"Origin": "https://example.org"}, 403),
            ("/api/unknown", {}, 404),
        ):
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(self.base + path, data=b"{}", headers=headers))
            self.assertEqual(error.exception.code, status)
            error.exception.close()

    def test_unavailable_service_is_explicit(self):
        import socket
        with socket.socket() as free:
            free.bind(("127.0.0.1", 0))
            unused = free.getsockname()[1]
        server = make_server(0, f"http://127.0.0.1:{unused}")
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with self.assertRaises(HTTPError) as error:
                urlopen(f"http://127.0.0.1:{server.server_port}/api/health")
            self.assertEqual(error.exception.code, 503)
            self.assertIn("detail", json.load(error.exception))
            error.exception.close()
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
