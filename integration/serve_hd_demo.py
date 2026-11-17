"""Serve saved real-model HD demo results on loopback without allocating a GPU."""
import argparse
from urllib.request import urlopen
from gateway import make_server


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=65512)
    port=p.parse_args().port;url=f'http://127.0.0.1:{port}/hd-demo.html'
    try:
        with urlopen(url,timeout=2) as response:
            if response.status==200 and b'hd-demo.js' in response.read(16384):
                print('Existing gallery: '+url);return
    except OSError:pass
    with make_server(port) as server:
        print('Saved real-model gallery: '+url+' (no GPU allocated; Ctrl+C stops this gateway)',flush=True)
        try:server.serve_forever()
        except KeyboardInterrupt:pass


if __name__=='__main__':main()
