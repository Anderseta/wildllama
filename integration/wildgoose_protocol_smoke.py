"""Exercise the built protocol server without loading models or using personal data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main(binary, temporary):
    binary, temporary = binary.resolve(), temporary.resolve()
    temporary.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='wildllama-protocol-', dir=temporary) as folder:
        root = Path(folder).resolve()
        assert root.is_relative_to(temporary) and root.name.startswith('wildllama-protocol-')
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(('OLLAMA_', 'LLAMA_', 'WILDLLAMA_', 'GGML_'))}
        env.update(OLLAMA_HOST='127.0.0.1:0', OLLAMA_MODELS=str(root / 'models'),
                   OLLAMA_NO_CLOUD='1', USERPROFILE=str(root / 'home'),
                   TEMP=str(root), TMP=str(root), TMPDIR=str(root))
        (root / 'home').mkdir()
        log = root / 'server.log'
        with log.open('wb') as output:
            process = subprocess.Popen([str(binary), 'serve'], env=env, stdout=output, stderr=output,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            try:
                address = None
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    text = log.read_text('utf8', errors='replace')
                    match = re.search(r'Listening on (127\.0\.0\.1:\d+)', text)
                    if match:
                        address = 'http://' + match[1]
                        break
                    if process.poll() is not None:
                        raise AssertionError(text[-3000:])
                    time.sleep(.05)
                assert address, log.read_text('utf8', errors='replace')[-3000:]

                def call(path, payload=None, method=None):
                    request = Request(address + path,
                        data=json.dumps(payload).encode() if payload is not None else None,
                        headers={'Content-Type': 'application/json', 'X-Request-ID': 'contract-smoke-123'},
                        method=method)
                    try:
                        response = urlopen(request, timeout=3)
                    except HTTPError as error:
                        response = error
                    with response:
                        assert response.headers.get('X-Request-ID') == 'contract-smoke-123'
                        return response.status, json.load(response)

                status, version = call('/api/version')
                protocol = version['wildllama']
                assert status == 200 and protocol['protocol_version'] == 1
                assert all(protocol['features'][name] for name in
                           ('tool_call_ids', 'request_id', 'structured_errors', 'request_admission'))
                maximum = protocol['limits']['embedding_items']
                status, error = call('/api/embed', {'model': 'absent', 'input': ['x'] * (maximum + 1)})
                assert status == 413 and error['code'] == 'embedding_input_limit', error
                assert error['retryable'] is False and error['request_id'] == 'contract-smoke-123'
                status, activity = call('/api/activity')
                assert status == 200 and activity['admission']['active'] == 0, activity
                assert not list((root / 'models').rglob('sha256-*')), 'Smoke test loaded or downloaded weights'
                assert 'download_cleanup' in version['capabilities']
                metadata, blobs = root / 'models' / 'metadata', root / 'models' / 'blobs'
                metadata.mkdir(exist_ok=True)
                blobs.mkdir(exist_ok=True)
                pending, other = 'huggingface.co/test/pending:q4', 'huggingface.co/test/other:q4'
                first, shared = 'sha256:' + 'a' * 64, 'sha256:' + 'b' * 64
                for name, digests in [(pending, [first, shared]), (other, [shared])]:
                    key = hashlib.sha256(name.lower().encode()).hexdigest()
                    (metadata / f'download-pin-{key}.json').write_text(json.dumps({'model': name, 'digests': digests}), encoding='utf8')
                partial = blobs / (first.replace(':', '-') + '-partial')
                progress = blobs / (first.replace(':', '-') + '-partial-0')
                retained = blobs / shared.replace(':', '-')
                for artifact in [partial, progress, retained]:
                    artifact.write_bytes(b'fixture only')
                status, cleanup = call('/api/download', {'model': pending}, method='DELETE')
                assert status == 200 and cleanup['removed_files'] == 2 and cleanup['retained_blobs'] == 1, cleanup
                assert not partial.exists() and not progress.exists() and retained.exists()
                status, cleanup = call('/api/download', {'model': pending}, method='DELETE')
                assert status == 200 and cleanup['removed_files'] == 0
                print(json.dumps({'protocol': 1, 'version': version['version'], 'correlation': True,
                                  'oversized_embedding_rejected_before_model': True, 'weights_loaded': False,
                                  'offline_download_cleanup': True, 'shared_download_preserved': True}))
            finally:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--temporary-dir', type=Path, required=True)
    args = parser.parse_args()
    main(args.binary, args.temporary_dir)
