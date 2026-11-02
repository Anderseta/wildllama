"""Check native prompt response selection using a tiny CPU-only fixture."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--server", type=Path, required=True)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--context-metadata-key", default="llama.context_length",
                    help="context key overridden only inside this isolated synthetic fixture process")
args = parser.parse_args()
if not 0 < args.model.stat().st_size <= 16 * 1024 * 1024:
    parser.error("only tiny synthetic fixtures up to 16 MiB are accepted")
args.output.mkdir(parents=True, exist_ok=True)
(args.output / "temp").mkdir(exist_ok=True)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
fields = ["content", "stop", "stop_type", "timings", "completion_probabilities", "prompt_progress"]


@contextmanager
def running():
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("OLLAMA_", "LLAMA_", "WILDLLAMA_", "GGML_"))
    }
    cpu_library = args.server.parent / ("ggml-cpu.dll" if os.name == "nt" else "libggml-cpu.so")
    environment.update(CUDA_VISIBLE_DEVICES="-1", GGML_BACKEND_PATH=str(cpu_library),
                       OMP_NUM_THREADS="1", TEMP=str(args.output / "temp"), TMP=str(args.output / "temp"))
    command = [str(args.server), "-m", str(args.model), "-c", "2048", "-b", "64", "-ub", "64",
               "-np", "1", "-t", "1", "-tb", "1", "-ngl", "0", "--device", "none",
               "-fa", "off", "-ctk", "f32", "-ctv", "f32", "--no-warmup", "--no-webui",
               "--host", "127.0.0.1", "--port", str(port)]
    command += ["--override-kv", args.context_metadata_key + "=int:2048"]
    with (args.output / "native-http.log").open("wb") as log:
        process = subprocess.Popen(command, cwd=args.server.parent, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 30
            while True:
                assert process.poll() is None, "CPU fixture exited; see native-http.log"
                try:
                    with opener.open(url + "/health", timeout=1) as response:
                        if json.load(response).get("status") == "ok":
                            break
                except OSError:
                    pass
                assert time.monotonic() < deadline, "CPU fixture startup timeout"
                time.sleep(0.05)
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def completion(url, stream=False, selected=None):
    request = dict(prompt=[20, 21, 22, 23] * 384, n_predict=4, temperature=0, seed=1234,
                   cache_prompt=False, stream=stream, ignore_eos=True, n_probs=2)
    if selected is not None:
        request["response_fields"] = selected
    request = urllib.request.Request(url + "/completion", json.dumps(request).encode(), {"Content-Type": "application/json"})
    with opener.open(request, timeout=30) as response:
        data = response.read()
    if not stream:
        result = json.loads(data)
        assert "error" not in result, result
        return [result], len(data)
    events = [json.loads(line[6:]) for line in data.decode().splitlines() if line.startswith("data: ")]
    assert events and events[-1].get("stop") is True, events
    assert all("error" not in event for event in events), events
    return events, len(data)


def consumed(events):
    result = []
    for event in events:
        selected = {key: event[key] for key in fields if key in event}
        if "timings" in selected:
            selected["timings"] = {key: value for key, value in selected["timings"].items()
                                   if key in ("prompt_n", "cache_n", "predicted_n", "draft_n", "draft_n_accepted")}
        result.append(selected)
    return result


with running() as url:
    results = []
    for stream in (False, True):
        full, full_bytes = completion(url, stream=stream)
        reduced, reduced_bytes = completion(url, stream=stream, selected=fields)
        assert full[-1]["prompt"], "external native default stopped echoing the prompt"
        assert "prompt" not in reduced[-1], "unused prompt leaked into filtered terminal event"
        assert consumed(full) == consumed(reduced), "field selection changed content, stop, probabilities or token counts"
        assert reduced_bytes < full_bytes, "terminal field selection did not reduce IPC payload"
        explicit, _ = completion(url, stream=stream, selected=fields + ["prompt"])
        assert explicit[-1]["prompt"] == full[-1]["prompt"], "explicit prompt request changed detokenization"
        nested, _ = completion(url, stream=stream, selected=fields + ["prompt/unknown"])
        assert "prompt/unknown" not in nested[-1], "invalid nested string path changed behavior"
        results.append({"stream": stream, "prompt_tokens": 1536, "full_bytes": full_bytes,
                        "selected_bytes": reduced_bytes, "saved_bytes": full_bytes - reduced_bytes})
report = {"status": "passed", "device": "CPU only", "fixture_bytes": args.model.stat().st_size,
          "synthetic_context_override": args.context_metadata_key + "=int:2048",
          "equivalent_consumed_fields": True, "default_and_explicit_prompt_preserved": True, "cases": results}
(args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
print(json.dumps(report))
