"""Serve only the prototype directory; no dependencies beyond Python 3."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote


class PrototypeHandler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".mjs": "text/javascript", ".wasm": "application/wasm"}

    def do_GET(self):
        if self.path == "/":
            self.send_response(302)
            self.send_header("Location", quote("人脸年龄变换原型_v5.html"))
            self.end_headers()
        else:
            super().do_GET()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", default=65512, type=int)
    args = parser.parse_args()
    handler = partial(PrototypeHandler, directory=str(Path(__file__).resolve().parent))
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    print(f"Prototype: http://{args.bind}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
