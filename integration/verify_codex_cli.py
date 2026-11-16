"""Verify the real CLI's command host through a local mock Responses endpoint.

No relay/model request or real key is used. The only command reads a temporary
test file; user config, sessions and approval settings are not modified.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import threading


def verify(binary):
    binary = str(Path(binary).resolve())
    outputs = []
    with tempfile.TemporaryDirectory(prefix='codex-tool-check-') as folder:
        root = Path(folder)
        reference = root / 'probe.txt'
        reference.write_text('CODEX_FILE_READ_OK\n')
        command = 'cat ' + shlex.quote(str(reference))
        code = 'text(await tools.exec_command(' + json.dumps(dict(cmd=command, workdir=folder, max_output_tokens=100)) + '));'

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"data":[]}')

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                received = [i for i in request.get('input', []) if i.get('type') == 'custom_tool_call_output']
                outputs.extend(received)
                tool_call = not received
                if tool_call:
                    item = dict(type='custom_tool_call', id='ct_local', call_id='call_local',
                                name='exec', namespace='functions', input=code)
                else:
                    item = dict(type='message', id='msg_local', role='assistant', status='completed',
                                content=[dict(type='output_text', text='CHECK_COMPLETE', annotations=[])])
                response = dict(id='resp_local', object='response', status='completed',
                                model='gpt-6.1-sol', output=[item],
                                usage=dict(input_tokens=1, output_tokens=1, total_tokens=2))
                events = [dict(type='response.created', response=dict(response, status='in_progress', output=[]))]
                if tool_call:
                    events += [dict(type='response.output_item.added', output_index=0, item=dict(item, input='')),
                               dict(type='response.custom_tool_call_input.delta', item_id=item['id'], output_index=0, delta=code),
                               dict(type='response.custom_tool_call_input.done', item_id=item['id'], output_index=0, input=code)]
                else:
                    events += [dict(type='response.output_item.added', output_index=0, item=dict(item, content=[])),
                               dict(type='response.content_part.added', item_id=item['id'], output_index=0, content_index=0,
                                    part=dict(type='output_text', text='', annotations=[])),
                               dict(type='response.output_text.delta', item_id=item['id'], output_index=0,
                                    content_index=0, delta='CHECK_COMPLETE')]
                events += [dict(type='response.output_item.done', output_index=0, item=item),
                           dict(type='response.completed', response=response)]
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                for event in events:
                    self.wfile.write(('event: ' + event['type'] + '\ndata: ' + json.dumps(event) + '\n\n').encode())
                self.wfile.flush()

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        config = root / '.codex'
        config.mkdir()
        (config / 'config.toml').write_text('''model = "gpt-6.1-sol"
model_provider = "local_install_check"
approval_policy = "never"
[model_providers.local_install_check]
name = "Local install check"
base_url = "http://127.0.0.1:''' + str(server.server_port) + '''"
env_key = "CODEX_LOCAL_INSTALL_CHECK_KEY"
wire_api = "responses"
requires_openai_auth = false
''')
        env = dict(os.environ, HOME=folder, CODEX_HOME=str(config),
                   CODEX_LOCAL_INSTALL_CHECK_KEY='local-test-only',
                   NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
        try:
            result = subprocess.run([binary, '--no-daemon', 'exec', '--skip-git-repo-check', '--ephemeral',
                                     '-s', 'danger-full-access', '--json', 'Read the local test file using the command tool.'],
                                    env=env, cwd=folder, stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True, timeout=45)
        finally:
            server.shutdown()
            server.server_close()
        if result.returncode or 'CODEX_FILE_READ_OK' not in json.dumps(outputs):
            raise RuntimeError('CLI command-host check failed.\n' + result.stderr[-1800:] + '\n' + json.dumps(outputs)[-2000:])
    return dict(cli_version='0.167.1',command_host=True,actual_file_read=True,
                api='loopback mock only; no real model/relay request',user_settings_changed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.binary), ensure_ascii=False))


if __name__ == '__main__':
    main()
