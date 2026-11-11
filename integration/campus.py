"""Expose the running classroom gateway on a selected campus LAN address."""
import argparse
import ssl
import threading
from contextlib import ExitStack
from functools import partial
from http.server import ThreadingHTTPServer
from urllib.parse import quote

from gateway import Gateway


class CampusGateway(Gateway):
    def do_GET(self):
        if self.path == "/":
            self.send_response(302)
            self.send_header("Location", "/" + quote("人脸年龄变换原型_v5.html") + "?mode=live&experience=mirror")
            self.end_headers()
        else:
            super().do_GET()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True, help="LAN address assigned to this computer")
    parser.add_argument("--cert", help="TLS certificate for the LAN hostname or IP")
    parser.add_argument("--key", help="TLS private key")
    args = parser.parse_args()
    if bool(args.cert) != bool(args.key):
        parser.error("Supply both --cert and --key for camera access.")
    handler = partial(CampusGateway, upstream="http://127.0.0.1:65512")
    with ExitStack() as stack:
        http = stack.enter_context(ThreadingHTTPServer((args.address, 8088), handler))
        if args.cert:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(args.cert, args.key)
            https = stack.enter_context(ThreadingHTTPServer((args.address, 8443), handler))
            https.socket = context.wrap_socket(https.socket, server_side=True)
            threading.Thread(target=https.serve_forever, daemon=True).start()
            stack.callback(https.shutdown)
        try:
            http.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
