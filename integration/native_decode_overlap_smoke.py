#!/usr/bin/env python3
"""Check opt-in ordinary-decode/CPU-output overlap using tiny local fixtures.

Both controls use the SAME binary and sampling settings. This is not an
autonomous GPU-token-feedback test. Use a byte-vocabulary diagnostic Llama GGUF
with EOS ID 2, no trained weights, and at most 64 MiB. No download is performed.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import time
import urllib.request


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def stable(value):
    if isinstance(value, dict):
        result = {k: stable(v) for k, v in value.items() if k not in ("timings", "tokens_cached")}
        if "timings" in value:
            result["__timings_present"] = True
        return result
    if isinstance(value, list):
        return [stable(v) for v in value]
    return value


def sse(server, body, cancel_after=None):
    request = urllib.request.Request(server.url + "/completion", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    events = []
    first_content_ms = None
    started = time.perf_counter()
    with server.opener.open(request, timeout=45) as response:
        require("text/event-stream" in response.headers.get("Content-Type", ""), "Missing SSE response")
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line or line.startswith(":"):
                continue
            require(line.startswith("data: "), "Unexpected SSE framing")
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            require(type(event) is dict and "error" not in event, "SSE request failed")
            # Client-observed TTFT includes request, prefill, generation and
            # framing/JSON. Empty progress/terminal events do not count.
            if first_content_ms is None and isinstance(event.get("content"), str) and event["content"]:
                first_content_ms = (time.perf_counter() - started) * 1000
            events.append(event)
            if cancel_after and len(events) >= cancel_after:
                break
    require(events, "Empty SSE response")
    if not cancel_after:
        require(events[-1].get("stop") is True, "SSE request lost its terminal event")
    elapsed = (time.perf_counter() - started) * 1000
    require(first_content_ms is None or 0 < first_content_ms <= elapsed, "Invalid first-content latency")
    return events, elapsed, dict(first_nonempty_content_ms=first_content_ms, wall_ms=elapsed)


def wait_idle(server):
    deadline = time.monotonic() + 15
    while True:
        rows = server.request("/slots")
        require(type(rows) is list, "Missing actual slot-state response")
        if not any(row.get("is_processing", False) for row in rows):
            return
        require(time.monotonic() < deadline, "Cancellation left a request active")
        time.sleep(.03)


def actual_markers(server, start):
    trace = server.log_text(start)
    return {name: trace.count("decode-overlap " + name) for name in ("queued", "consumed", "discarded")}



def configured_server(launcher, args, label, option, verbosity):
    server = launcher.NativeServer(args.executable, args.model, args.output_dir / label, args.gpu)
    server.env["WILDLLAMA_DECODE_OVERLAP"] = option
    # Independently selected graph budget is identical for both controls.
    server.env["WILDLLAMA_GRAPH_VARIANTS_MIB"] = "128"
    server.command[server.command.index("--ctx-size") + 1] = "2048"
    server.command[server.command.index("--log-verbosity") + 1] = str(verbosity)
    server.command.extend(("--slots", "--cpu-mask", "14", "--cpu-strict", "1"))
    return server


def quiet_timing(args, launcher, provenance, identity, body, report, raw_requests):
    # Correctness/positive-path servers have already exited. Fresh controls
    # exclude DEBUG messages so additional overlap diagnostics do not bias I/O.
    servers, proofs, windows, warmups = {}, {}, [], []
    expected = None
    with ExitStack() as stack:
        for label, option in (("disabled", "0"), ("enabled", "1")):
            server = configured_server(launcher, args, "timing-" + label, option, 4)
            stack.enter_context(server)
            servers[label] = server
        for label, server in servers.items():
            for warmup_index in range(3):
                events, elapsed, metrics = sse(server, body)
                if expected is None:
                    expected = stable(events)
                require(stable(events) == expected and metrics["first_nonempty_content_ms"] is not None,
                        "Quiet timing warmup changed output or omitted first content")
                phase = "cold-first-completion" if warmup_index == 0 else "warmup"
                row = dict(label=label, warmup_index=warmup_index, phase=phase, **metrics)
                warmups.append(row)
                raw_requests.append(dict(case="quiet-warmup", request=body, events=events, **row))
            # Actual own-process module proof, outside every timed window.
            modules = provenance.verify_loaded_payload(server, identity)
            (server.directory / "modules.json").write_text(json.dumps(modules, indent=2) + "\n", encoding="utf-8")
            proofs[label] = dict(command=server.command, observed_modules=modules)
        for cycle in range(args.cycles):
            order = ("disabled", "enabled", "enabled", "disabled") if cycle % 2 == 0 else (
                "enabled", "disabled", "disabled", "enabled")
            for label in order:
                events, elapsed, metrics = sse(servers[label], body)
                require(stable(events) == expected and math.isfinite(elapsed) and elapsed > 0 and
                        metrics["first_nonempty_content_ms"] is not None,
                        "Quiet timing changed output or returned invalid elapsed time")
                terminal = events[-1]
                generated = terminal["tokens_predicted"]
                native_tps = terminal["timings"]["predicted_per_second"]
                require(generated == body["n_predict"] and math.isfinite(native_tps) and native_tps > 0,
                        "Quiet timing omitted the requested generated tokens or native throughput")
                window = dict(cycle=cycle, label=label, **metrics, generated_tokens=generated,
                              end_to_end_tokens_per_second=generated * 1000.0 / elapsed,
                              native_reported_tokens_per_second=native_tps)
                windows.append(window)
                raw_requests.append(dict(case="quiet-timing", request=body, events=events, **window))
        for label, server in servers.items():
            require(server.process.poll() is None, "Quiet timing server exited")
            require(not any(actual_markers(server, 0).values()), "DEBUG markers leaked into quiet timed controls")
            after = provenance.verify_loaded_payload(server, identity)
            require(after["validated_required_modules"] == proofs[label]["observed_modules"]["validated_required_modules"],
                    "Quiet server loaded core/backend identity changed")
            proofs[label]["observed_modules_after"] = after
    for label, server in servers.items():
        proofs[label]["log_sha256"] = sha(server.logfile)
    medians = {label: statistics.median(w["wall_ms"] for w in windows if w["label"] == label) for label in servers}
    native_tps = {label: statistics.median(w["native_reported_tokens_per_second"] for w in windows
                  if w["label"] == label) for label in servers}
    wall_tps = {label: statistics.median(w["end_to_end_tokens_per_second"] for w in windows
                if w["label"] == label) for label in servers}
    ttft = {label: statistics.median(w["first_nonempty_content_ms"] for w in windows if w["label"] == label)
            for label in servers}
    cold = {label: next(w["first_nonempty_content_ms"] for w in warmups
                       if w["label"] == label and w["warmup_index"] == 0) for label in servers}
    warm = {label: [w["first_nonempty_content_ms"] for w in warmups
                   if w["label"] == label and w["warmup_index"] > 0] for label in servers}
    report["timing"] = dict(cycles=args.cycles, warmups_per_control=3, windows=windows, warmup_requests=warmups,
        median_first_nonempty_content_ms=ttft, cold_first_completion_ttft_ms=cold, warmup_ttft_ms=warm,
        ttft_definition="Client-observed time to first parsed SSE event containing a nonempty content string; excludes empty progress events",
        cold_first_completion_definition="First completion after a fresh server is health-ready; excludes model load. Native initialization may already run capability probes, so this is not fully cold CUDA startup.",
        median_wall_ms=medians, median_native_reported_tokens_per_second=native_tps,
        median_end_to_end_tokens_per_second=wall_tps,
        latency_change_percent=(medians["enabled"] / medians["disabled"] - 1) * 100,
        fresh_servers=True, log_verbosity=4, debug_io_excluded=True, actual_loaded_core_module_pins=True,
        payload_identity=identity, servers=proofs,
        scope="Tiny random-weight SSE latency and actual native-reported throughput, same binary; not trained-model TPS")


def main():
    require(__debug__, "Do not use python -O")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--cycles", type=int, default=0,
                        help="Optional paired ABBA SSE latency cycles after correctness checks")
    args = parser.parse_args()
    args.executable, args.model, args.output_dir = (p.resolve() for p in
        (args.executable, args.model, args.output_dir))
    require(args.executable.is_file() and args.model.is_file(), "Explicit existing native payload and fixture required")
    require(0 < args.model.stat().st_size <= 64 * 1024 * 1024, "Only local fixtures up to 64 MiB")
    require(args.output_dir.drive.upper() != "C:", "All outputs must stay outside C:")
    require(args.cycles == 0 or 2 <= args.cycles <= 20, "Use zero or 2-20 bounded timing cycles")
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()), "Use a fresh output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    launcher_path = Path(__file__).with_name("native_execution_features_smoke.py")
    spec = importlib.util.spec_from_file_location("overlap_native_launcher", launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    provenance_path = Path(__file__).with_name("native_payload_provenance.py")
    provenance_spec = importlib.util.spec_from_file_location("overlap_payload_provenance", provenance_path)
    provenance = importlib.util.module_from_spec(provenance_spec)
    provenance_spec.loader.exec_module(provenance)
    identity = provenance.manifest(args.executable)
    payload = args.executable.parent
    names = ("llama-server-impl", "llama", "llama-common", "ggml", "ggml-base", "ggml-cpu")
    core_files = [args.executable] + [payload / (("" if os.name == "nt" else "lib") + name +
                                                (".dll" if os.name == "nt" else ".so")) for name in names]
    require(all(p.is_file() for p in core_files), "Use the complete shared native payload")
    pinned = {str(p): sha(p) for p in core_files}
    report = {"status": "running", "gpu": args.gpu, "fixture_bytes": args.model.stat().st_size,
              "fixture_sha256": sha(args.model), "harness_sha256": sha(Path(__file__)),
              "launcher_sha256": sha(launcher_path), "provenance_harness_sha256": sha(provenance_path),
              "actual_loaded_core_module_pins": False, "default_enabled": False,
              "autonomous_gpu_feedback": False, "cases": [], "servers": {}}
    raw_requests = []
    try:
        servers = {}
        with ExitStack() as stack:
            for label, option in (("disabled", "0"), ("enabled", "1")):
                server = configured_server(launcher, args, label, option, 5)
                if str(server.backend) not in pinned:
                    core_files.append(server.backend)
                    pinned[str(server.backend)] = sha(server.backend)
                stack.enter_context(server)
                servers[label] = server
            ascii_tokens = servers["disabled"].request("/tokenize", {"content": "A", "add_special": False})["tokens"]
            # SentencePiece fixtures can add a dummy space prefix even with
            # add_special=false. Select the actual ASCII byte by round trip,
            # rather than assuming the whole encoded string has one token.
            ascii_candidates = [token for token in dict.fromkeys(ascii_tokens)
                if servers["disabled"].request("/detokenize", {"tokens": [token]})["content"] == "A"]
            require(len(ascii_candidates) == 1, "Diagnostic fixture must expose one exact ASCII A token")
            ascii_token = ascii_candidates[0]
            report["forced_ascii_token"] = ascii_token
            prompt = servers["disabled"].request("/tokenize", {"content": "Ola! teste de fluxo.", "add_special": True})["tokens"]
            base = dict(prompt=prompt, n_predict=32, temperature=0.0, seed=1234,
                        samplers=["temperature"], repeat_penalty=1.0, frequency_penalty=0.0,
                        presence_penalty=0.0, dry_multiplier=0.0, xtc_probability=0.0,
                        backend_sampling=True, n_probs=0, cache_prompt=False,
                        stream=True, return_tokens=True, ignore_eos=True,
                        logit_bias=[[ascii_token, 100.0]])
            cases = [
                ("ordinary", {}, True),
                ("timer-disabled-zero", {"t_max_predict_ms": 0}, True),
                ("time-controller-positive", {"t_max_predict_ms": 60000}, False),
                ("limit-one", {"n_predict": 1}, False),
                ("limit-two", {"n_predict": 2}, True),
                ("per-token-timings", {"n_predict": 4, "timings_per_token": True}, True),
                ("eos", {"ignore_eos": False, "logit_bias": [[2, 100.0]]}, False),
                ("stop-word", {"stop": ["A"]}, False),
                ("cpu-sampling", {"backend_sampling": False}, False),
                ("history", {"repeat_penalty": 1.1}, False),
                ("stochastic", {"temperature": .7}, False),
                ("probabilities", {"n_probs": 5}, False),
                ("indentation-control", {"n_indent": 1}, False),
                ("non-streaming", {"stream": False}, False),
                ("grammar", {"grammar": 'root ::= "AAA"\n', "grammar_first": True,
                             "ignore_eos": False}, False),
            ]
            for name, change, expected in cases:
                body = dict(base, **change)
                observed, markers = {}, {}
                for label, server in servers.items():
                    start = server.logfile.stat().st_size
                    if body["stream"]:
                        events, elapsed, metrics = sse(server, body)
                    else:
                        started = time.perf_counter()
                        events = [server.request("/completion", body)]
                        elapsed = (time.perf_counter() - started) * 1000
                        metrics = dict(first_nonempty_content_ms=None, wall_ms=elapsed, non_streaming=True)
                    wait_idle(server)
                    observed[label] = stable(events)
                    markers[label] = actual_markers(server, start)
                    raw_requests.append(dict(case=name, label=label, request=body, events=events,
                                             wall_ms=elapsed, client_latency=metrics, markers=markers[label]))
                    should_run = args.gpu and label == "enabled" and expected
                    require(bool(markers[label]["queued"]) == should_run,
                            f"Unexpected actual decode-overlap admission: {name}/{label}: {markers[label]}")
                    require(markers[label]["queued"] == markers[label]["consumed"] and
                            markers[label]["discarded"] == 0, "Normal request lost or duplicated an ahead step")
                require(observed["disabled"] == observed["enabled"], f"SSE content/tokens/settings/stop changed: {name}")
                report["cases"].append(dict(case=name, exact_sse=True, actual_markers=markers))
            # Closing an active stream must drain/trim a pending ordinary step;
            # a subsequent request checks that no future output/token leaked.
            recovery = {}
            for label, server in servers.items():
                start = server.logfile.stat().st_size
                events, _, canceled_latency = sse(server, dict(base, n_predict=1024), cancel_after=2)
                wait_idle(server)
                markers = actual_markers(server, start)
                require(markers["queued"] == markers["consumed"] + markers["discarded"],
                        "Canceled stream retained a live or unaccounted ahead step")
                if args.gpu and label == "enabled":
                    require(markers["discarded"] > 0, "Cancel case did not actually exercise ahead rollback")
                followup, _, recovery_latency = sse(server, dict(base, n_predict=16))
                wait_idle(server)
                recovery[label] = stable(followup)
                raw_requests.append(dict(case="cancel", label=label, canceled_events=events,
                                         markers=markers, recovery=followup, client_latency=canceled_latency,
                                         recovery_latency=recovery_latency))
                report["cases"].append(dict(case="cancel-" + label, actual_markers=markers,
                                            recovery_terminal=True))
            require(recovery["disabled"] == recovery["enabled"], "Cancellation changed next-request state")
            if args.cycles:
                # Prove the exact timed body really admits overlap with verbose
                # diagnostics, before launching fresh quiet timing controls.
                timed_body = dict(base, n_predict=128)
                reference = None
                for label, server in servers.items():
                    start = server.logfile.stat().st_size
                    events, elapsed, metrics = sse(server, timed_body)
                    wait_idle(server)
                    markers = actual_markers(server, start)
                    should_run = args.gpu and label == "enabled"
                    require(bool(markers["queued"]) == should_run and
                            markers["queued"] == markers["consumed"] and markers["discarded"] == 0,
                            "Exact timed body did not exercise the expected actual path")
                    if reference is None:
                        reference = stable(events)
                    require(stable(events) == reference, "Verbose timing-body control changed output")
                    raw_requests.append(dict(case="timing-body-positive-control", label=label,
                        request=timed_body, events=events, wall_ms=elapsed, client_latency=metrics, markers=markers))
            for label, server in servers.items():
                require(server.process.poll() is None, "Native server exited")
                # Outside timed windows, enumerate this exact owned process and
                # require every core/backend DLL to come from the pinned bundle.
                modules = provenance.verify_loaded_payload(server, identity)
                (args.output_dir / label / "modules.json").write_text(json.dumps(modules, indent=2) + "\n", encoding="utf-8")
                report["servers"][label] = dict(command=server.command,
                    backend_sha256=sha(server.backend), actual_markers=actual_markers(server, 0),
                    observed_modules=modules)
            report["actual_loaded_core_module_pins"] = True
        for label, server in servers.items():
            # Shutdown may append diagnostics; pin the completed log only after
            # the owned process and its output stream have both closed.
            report["servers"][label]["log_sha256"] = sha(server.logfile)
        if args.cycles:
            quiet_timing(args, launcher, provenance, identity, timed_body, report, raw_requests)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        provenance.unchanged(identity)
        report["payload_identity"] = identity
        report["payload_sha256"] = {str(p): sha(p) for p in core_files}
        require(report["payload_sha256"] == pinned, "Native core changed during validation")
        require(sha(args.model) == report["fixture_sha256"], "Fixture changed during validation")
        responses = args.output_dir / "responses.json"
        responses.write_text(json.dumps(raw_requests, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["responses_sha256"] = sha(responses)
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
