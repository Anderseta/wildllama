#!/usr/bin/env python3
"""CPU-only regression for mixed neural verification and bounded prompt slices.

Uses an explicit existing random-weight fixture <=4 MiB as BOTH target and draft.
No download/build/installation or production endpoint is used. Forced printable
tokens test transport/ownership/stop/cancel contracts, not language quality or
speed. The independent native hybrid test checks computed state and logits.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True


def require(value, message):
    if not value:
        raise AssertionError(message)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(executed=False, scope="No server starts without --execute")))
        return
    args.executable, args.model, args.output_dir = (
        path.resolve() for path in (args.executable, args.model, args.output_dir))
    require(args.executable.is_file() and args.model.is_file(), "Explicit existing executable and tiny fixture required")
    require(0 < args.model.stat().st_size <= 4 * 1024 * 1024, "Only the untrained tiny native fixture is allowed")
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()), "Select a fresh output directory")
    if os.name == "nt":
        require(all(path.drive.upper() in ("G:", "F:") for path in
                    (args.executable, args.model, args.output_dir)), "Payload, fixture and output must stay on G: or F:")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = Path(__file__).parent
    helpers = module(base / "native_adaptive_prefill_smoke.py", "portable_scheduling_helpers")
    launcher = module(base / "native_execution_features_smoke.py", "portable_scheduling_launcher")
    model_sha = helpers.sha(args.model)
    report = dict(status="running", scope="Tiny CPU neural-draft scheduling correctness; no model TPS claim",
                  fixture_bytes=args.model.stat().st_size, fixture_sha256=model_sha, cases=[], model_downloads=0)
    controls = {}
    try:
        for setting in ("0", "1"):
            label = "enabled" if setting == "1" else "disabled"
            server = launcher.NativeServer(args.executable, args.model, args.output_dir / label, False)
            for name, value in (("--ctx-size", "8192"), ("--batch-size", "512"), ("--ubatch-size", "128"),
                                ("--parallel", "2"), ("--spec-type", "draft-simple")):
                server.command[server.command.index(name) + 1] = value
            server.command.extend(("--kv-unified", "--slots", "--model-draft", str(args.model),
                                   "--spec-draft-n-max", "3", "--spec-draft-n-min", "1", "--spec-draft-p-min", "0",
                                   "--n-gpu-layers-draft", "0", "--spec-draft-device", "none"))
            server.env.update(WILDLLAMA_ADAPTIVE_PREFILL=setting, WILDLLAMA_PREFILL_LATENCY_US="100",
                              WILDLLAMA_PREFILL_TRACE="1", WILDLLAMA_GRAPH_VARIANTS_MIB="128")
            with server:
                forced, a, b = helpers.diagnostic_bodies(server)
                pair = helpers.paired(server, a, b, label, forced_token=forced)
                markers = helpers.wait_positive_markers(server, pair["log_start"], setting == "1")
                require("draft acceptance =" in server.log_text(), "The neural draft did not execute")
                if setting == "1":
                    require(any(row.get("speculative") == 1 and row.get("neural") == 1 and
                                row.get("generation_rows", 0) > 1 for row in markers["active"]),
                            "No actual intact neural verification/prompt scheduling hit was observed")
                    for row in markers["memory"]:
                        require(row.get("draft_scope_restored") == 1 and
                                row.get("draft_primary_growth_bytes") == 0 and
                                row.get("draft_primary_gpu_growth_bytes") == 0 and
                                row.get("draft_secondary_growth_bytes") == 0,
                                "Adapted prompt slices enlarged or leaked draft graph storage")
                    canceled = helpers.paired(server, a, dict(b, return_progress=True), "cancel-pending", True, forced)
                    report["cases"].append(dict(case="cancel-pending-neural-prompt", result=canceled))
                    recovery = helpers.paired(server, a, b, "reuse-after-cancel", forced_token=forced)
                    report["cases"].append(dict(case="reuse-after-cancel", result=recovery))
                pair["markers"] = markers
                controls[label] = pair
                report["cases"].append(dict(case=label + "-neural-prompt-contention", result=pair))
        for slot in ("a", "b"):
            require(controls["disabled"]["semantics"][slot] == controls["enabled"]["semantics"][slot],
                    "Same-setting neural scheduling changed request/SSE/stop semantics")
        target_peaks = helpers.compare_memory_peaks(controls["disabled"]["markers"], controls["enabled"]["markers"])
        report["target_memory"] = target_peaks
        draft_fields = ("draft_primary_peak_bytes", "draft_primary_gpu_peak_bytes", "draft_secondary_peak_bytes")
        draft_peaks = {label: {field: max(row[field] for row in controls[label]["markers"]["memory"])
                               for field in draft_fields} for label in ("disabled", "enabled")}
        require(all(draft_peaks["enabled"][field] <= draft_peaks["disabled"][field] for field in draft_fields),
                "Adapted scheduling exceeded disabled-control draft memory peaks")
        report.update(status="passed", forced_streams_exact=True, intact_neural_verification_hit=True,
                      draft_memory=draft_peaks, configured_context=8192, slots=2, batch=512, ubatch=128,
                      gpu_tests=0, pretrained_model_tests=0)
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        require(helpers.sha(args.model) == model_sha, "Diagnostic fixture changed")
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(status="passed", cases=len(report["cases"]), intact_neural_verification_hit=True)))


if __name__ == "__main__":
    main()
