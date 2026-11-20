"""Exercise full and incremental slot saves with a small local GGUF on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def expand(encoded, base):
    header = b"WLDDEL01" + hashlib.sha256(base).hexdigest().encode()
    assert encoded.startswith(header)
    offset = len(header)
    restored = bytearray()
    while encoded[offset]:
        kind = encoded[offset]
        length, = struct.unpack_from("<I", encoded, offset + 1)
        offset += 5
        assert 0 < length <= 65536
        if kind == 1:
            restored.extend(encoded[offset:offset + length])
            offset += length
        else:
            assert kind == 2
            start, = struct.unpack_from("<Q", encoded, offset)
            assert start + length <= len(base)
            offset += 8
            restored.extend(base[start:start + length])
    assert offset + 1 == len(encoded)
    return restored


def check(server, model, directory):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    def request(path, payload=None):
        body = json.dumps(payload).encode() if payload is not None else None
        req = Request(f"http://127.0.0.1:{port}{path}", data=body,
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=45) as response:
            return json.load(response)

    def slot(action, **options):
        return request("/slots/0?action=" + action,
                       {"filename": "resume.bin", "handoff_auth": "local-test", **options})

    def complete(tokens):
        return request("/completion", {"prompt": tokens, "id_slot": 0, "cache_prompt": True,
                                       "n_predict": 2, "temperature": 0, "seed": 3})

    args = [str(server), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
            "-ngl", "0", "-c", "4096", "-np", "1", "-t", "2", "-tb", "2",
            "--no-warmup", "--slot-save-path", str(directory)]
    with (directory / "runner.log").open("w") as log:
        proc = subprocess.Popen(args, cwd=server.parent, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, "WILDLLAMA_IMAGE_HANDOFF_AUTH": "local-test"},
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            deadline = time.monotonic() + 30
            while True:
                try:
                    request("/health")
                    break
                except OSError:
                    if proc.poll() is not None or time.monotonic() >= deadline:
                        details = (directory / "runner.log").read_text(errors="replace")
                        raise RuntimeError("Small runner did not start:\n" + details[-2500:])
                    time.sleep(0.1)

            tokens = [3 + i % 18 for i in range(1800)]
            complete(tokens)
            slot("save")
            path = directory / "resume.bin"
            base = path.read_bytes()
            path.rename(directory / "resume.bin.base")
            tokens.extend(list(range(7, 18)) * 20)
            complete(tokens)
            slot("save")
            full = path.read_bytes()
            slot("save", image_budget=128 << 20, checkpoint_base=hashlib.sha256(base).hexdigest())
            delta = path.read_bytes()
            restored = expand(delta, base)
            assert restored == full
            assert len(delta) < len(full), "the fixture shared no blocks with its keyframe"

            def resume(data):
                path.write_bytes(data)
                slot("erase")
                assert slot("restore")["n_read"] == len(full)
                return complete(tokens + [7, 8])["content"]

            reference = resume(full)
            assert reference == resume(restored)
            path.write_bytes(delta)
            slot("erase")
            options = dict(image_budget=128 << 20, checkpoint_base=hashlib.sha256(base).hexdigest())
            assert slot("restore", **options)["n_read"] == len(full)
            assert complete(tokens + [7, 8])["content"] == reference
            assert path.read_bytes() == delta
            for damaged in (delta[:-1], delta + b"extra"):
                path.write_bytes(damaged)
                try:
                    slot("restore", **options)
                except HTTPError as error:
                    assert error.code == 400
                else:
                    raise AssertionError("accepted invalid incremental state")
            assert resume(full) == reference
            print(json.dumps({"base_bytes": len(base), "full_bytes": len(full), "delta_bytes": len(delta)}))
        finally:
            proc.terminate()
            proc.wait(timeout=15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path, help="Small dense CPU fixture with at least 22 token IDs")
    parser.add_argument("--temp-dir", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(dir=args.temp_dir) as temp:
        check(args.server.resolve(), args.model.resolve(), Path(temp).resolve())
