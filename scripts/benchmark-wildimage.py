#!/usr/bin/env python3
"""Compare local WildImage runners/profiles with a fixed seed and pixel parity.

Pass a saved image configuration or a single image profile JSON object, a prompt
file and optional ordered references. Set width/height to the effective
canvas if Wildllama derived it from a reference. This exercises the native
runner directly, outside Wildllama's model scheduler and disk conditioning cache.
The default is a dry run. An actual run needs NVIDIA's nvidia-smi and Pillow.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
from pathlib import Path


NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
PHASE_STARTS = (
    ("encoding", re.compile(r"\bgenerate_image \d+x\d+", re.I)),
    ("sampling", re.compile(r"\bgenerating image:", re.I)),
    ("decoding", re.compile(r"\bdecoding \d+(?:/\d+)? latents?", re.I)),
    ("saving", re.compile(r"\bgenerate_image completed|\bimages saved", re.I)),
)
NATIVE_TIMES = re.compile(
    r"(encode_first_stage|get_learned_condition|sampling|decode_first_stage) "
    r"completed, taking ([\d.]+)s", re.I
)


def absolute_path(value):
    path = Path(value)
    if not path.is_absolute() or any(character in str(path) for character in "\x00\r\n"):
        raise ValueError(f"expected an absolute path: {value!r}")
    return path.resolve()


def load_profile(path):
    profile = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(profile, dict):
        raise ValueError("the profile file must contain one image profile object")
    if "profiles" in profile:
        profiles = profile["profiles"]
        if not isinstance(profiles, list) or not profiles or not all(isinstance(item, dict) for item in profiles):
            raise ValueError("the saved image configuration must contain a nonempty profiles list")
        default_id = profile.get("default_profile_id")
        selected = next((item for item in profiles if default_id and item.get("id") == default_id), None)
        if selected is None and len(profiles) != 1:
            raise ValueError("the configuration has no matching default_profile_id; provide one profile object")
        profile = selected if selected is not None else profiles[0]
    for key in ("diffusion_model", "text_encoder", "vae"):
        profile[key] = absolute_path(profile[key])
    if profile.get("vision_encoder"):
        profile["vision_encoder"] = absolute_path(profile["vision_encoder"])
    for key in ("width", "height"):
        if type(profile[key]) is not int or not 64 <= profile[key] <= 4096 or profile[key] % 32:
            raise ValueError(f"{key} must be a multiple of 32 from 64 to 4096")
    if type(profile["steps"]) is not int or not 1 <= profile["steps"] <= 100:
        raise ValueError("steps must be an integer from 1 to 100")
    if not 0 <= profile["cfg_scale"] <= 30:
        raise ValueError("cfg_scale must be from 0 to 30")
    for key, default, maximum in (("max_ram_mib", 0, 1048576), ("vram_reserve_mib", 256, 1048576),
                                  ("cpu_threads", 0, 1024)):
        value = profile.get(key, default)
        if type(value) is not int or not 0 <= value <= maximum:
            raise ValueError(f"{key} must be an integer from 0 to {maximum}")
    profile["loras"] = [
        {"path": absolute_path(item["path"]), "scale": item["scale"]}
        for item in profile.get("loras") or [] if item["scale"] != 0
    ]
    for item in profile["loras"]:
        if not -4 <= item["scale"] <= 4 or item["path"].suffix.lower() not in {".gguf", ".safetensors"}:
            raise ValueError("active LoRAs must have scales from -4 to 4 and use GGUF or safetensors")
    return profile


def runner_command(runner, profile, options, directory, variant):
    backend = "cuda0"
    if profile.get("vae_on_cpu"):
        backend += ",vae=cpu"
    if profile.get("encoder_on_cpu"):
        backend += ",te=cpu"
    staged = profile.get("stage_components", False)
    args = [
        str(runner), "--diffusion-model", str(profile["diffusion_model"]),
        "--llm", str(profile["text_encoder"]), "--vae", str(profile["vae"]),
        "--prompt-file", str(directory / "prompt.txt"),
        "--negative-prompt-file", str(directory / "negative.txt"),
        "--width", str(profile["width"]), "--height", str(profile["height"]),
        "--steps", str(profile["steps"]), "--cfg-scale", str(profile["cfg_scale"]),
        "--sampling-method", profile["sampler"], "--seed", str(options.seed),
        "--output", str(directory / "image.png"), "--backend", backend,
        "--stage-components", "on" if staged else "off",
        "--max-vram", str(options.vram_mib / 1024),
        "--flash-attn", "on" if profile.get("flash_attention", False) else "off",
        "--log-level", "verbose",
    ]
    if variant != "stock":
        args += ["--max-ram", str(profile.get("max_ram_mib", 0) / 1024),
                 "--vram-reserve", str(profile.get("vram_reserve_mib", 256) / 1024)]
    if variant == "ram-offload":
        args += ["--params-backend", "cpu", "--offload-to-cpu", "--auto-fit", "off"]
    elif profile.get("offload_to_cpu"):
        # Mirror Wildllama's first attempt. A direct run measures this attempt
        # in isolation, without the application's automatic offload retries.
        placement = backend
        if staged and variant != "stock":
            encoder = "cpu" if profile.get("encoder_on_cpu") else "disk"
            vae = "cpu" if profile.get("vae_on_cpu") else "disk"
            placement = f"cuda0,te={encoder},vae={vae}"
        args += ["--params-backend", placement, "--auto-fit", "off"]
        if not options.segmented:
            args += ["--disable-segmented-compute"]
    elif staged:
        args += ["--params-backend", "disk", "--auto-fit", "off"]
    else:
        args += ["--auto-fit", "on"]
    if options.minimax_attention_budget_mib:
        args += ["--model-args", f"minimax_h3_attn_chunk_budget={options.minimax_attention_budget_mib / 1024}"]
    for field in ("sigmas", "scheduler", "cpu_threads"):
        if profile.get(field):
            flag = "threads" if field == "cpu_threads" else field
            args += [f"--{flag}", str(profile[field])]
    if profile.get("vae_tiling"):
        args += ["--vae-tiling"]
    if options.reference:
        vision = profile.get("vision_encoder", profile["text_encoder"])
        if vision != profile["text_encoder"]:
            args += ["--llm_vision", str(vision)]
        for reference in options.reference:
            args += ["-r", str(reference)]
    if profile["loras"]:
        args += ["--lora-model-dir", str(directory / "loras"), "--lora-apply-mode", "at_runtime"]
    if variant == "sage" or profile.get("sage_attention"):
        args += ["--sage-attn"]
    return args


def prepare_inputs(directory, profile, prompt, negative):
    tags = []
    if profile["loras"]:
        library = directory / "loras"
        library.mkdir()
        for index, item in enumerate(profile["loras"]):
            name = f"lora-{index}{item['path'].suffix.lower()}"
            target = library / name
            try:
                os.link(item["path"], target)
            except OSError:
                shutil.copyfile(item["path"], target)
            tags.append(f"\n<lora:{name}:{item['scale']}>")
    (directory / "prompt.txt").write_text(prompt + "".join(tags), encoding="utf-8")
    (directory / "negative.txt").write_text(negative, encoding="utf-8")


def gpu_read(nvidia_smi, device):
    result = subprocess.run(
        [nvidia_smi, "-i", str(device), "--query-gpu=uuid,memory.used,memory.free,memory.total",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=5, creationflags=NO_WINDOW,
    )
    uuid, used, free, total = next(csv.reader(result.stdout.splitlines()))
    return {"uuid": uuid.strip(), "used_mib": int(used), "free_mib": int(free), "total_mib": int(total)}


def windows_process_memory(pid):
    """Optional standard-library-only counters, independent of GPU telemetry."""
    if os.name != "nt":
        return None, lambda: None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "peak_working_set", "working_set", "peak_paged", "paged",
                "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile", "private",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    kernel.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x410, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())

    def read():
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not kernel.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {name + "_mib": round(getattr(counters, name) / 1024**2, 2)
                for name in ("working_set", "peak_working_set", "private")}

    return read, lambda: kernel.CloseHandle(handle)


def monitor_gpu(nvidia_smi, device, start, interval, stop, samples, errors, memory_read, memory_samples):
    while not stop.is_set():
        try:
            reading = gpu_read(nvidia_smi, device)
        except (OSError, ValueError, StopIteration, subprocess.SubprocessError) as error:
            errors.append(str(error))
            return
        samples.append((time.perf_counter() - start, reading["used_mib"]))
        if memory_read:
            try:
                memory_samples.append((time.perf_counter() - start, memory_read()))
            except OSError:
                # The native process may exit between its last log line and sampling.
                memory_read = None
        stop.wait(interval)


def gpu_summary(initial, samples, phases, errors):
    phase_peaks = {}
    for elapsed, used in samples:
        phase = next((name for started, name in reversed(phases) if elapsed >= started), "loading")
        phase_peaks[phase] = max(used, phase_peaks.get(phase, 0))
    peak = max((used for _, used in samples), default=None)
    return {
        "scope": "device-wide, including display and other processes",
        "uuid": initial["uuid"], "total_mib": initial["total_mib"],
        "before_mib": initial["used_mib"], "sampled_peak_mib": peak,
        "peak_delta_mib": None if peak is None else peak - initial["used_mib"],
        "phase_sampled_peaks_mib": phase_peaks, "samples": len(samples), "errors": errors,
    }


def memory_summary(samples, phases):
    phase_peaks = {}
    for elapsed, values in samples:
        phase = next((name for started, name in reversed(phases) if elapsed >= started), "loading")
        peak = phase_peaks.setdefault(phase, {})
        for name in ("working_set_mib", "private_mib"):
            peak[name] = max(peak.get(name, 0), values[name])
    return {
        "scope": "native runner only; working set is resident RAM, private is committed memory (not necessarily resident)",
        "peak_working_set_mib": max((values["peak_working_set_mib"] for _, values in samples), default=None),
        "phase_sampled_peaks": phase_peaks, "samples": len(samples),
    }


def run_trial(runner, profile, options, directory, nvidia_smi, variant, prompt, negative):
    directory.mkdir()
    prepare_inputs(directory, profile, prompt, negative)
    command = runner_command(runner, profile, options, directory, variant)
    initial = gpu_read(nvidia_smi, options.gpu_index)
    reserve = profile.get("vram_reserve_mib", 256)
    if initial["free_mib"] < options.vram_mib + reserve:
        raise ValueError(f"GPU has {initial['free_mib']} MiB free; need target plus {reserve} MiB headroom")
    env = os.environ.copy()
    for name in ("WILDIMAGE_CONDITION_CACHE_DIR", "WILDIMAGE_CONDITION_NAMESPACE", "WILDIMAGE_CONDITION_CACHE_BYTES"):
        env.pop(name, None)
    env["CUDA_VISIBLE_DEVICES"] = initial["uuid"]
    env["CUDA_CACHE_PATH"] = str(directory.parent / "cuda-cache")
    env["TMP"] = env["TEMP"] = str(directory)
    Path(env["CUDA_CACHE_PATH"]).mkdir(exist_ok=True)
    samples, memory_samples, errors, phases, native = [], [], [], [(0.0, "loading")], {}
    phase_cursor = 0
    stop, timed_out = threading.Event(), threading.Event()
    start = time.perf_counter()
    with (directory / "runner.log").open("w", encoding="utf-8", buffering=1) as log:
        child = subprocess.Popen(command, cwd=runner.parent, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                 creationflags=NO_WINDOW)
        memory_read, memory_close = None, lambda: None
        try:
            memory_read, memory_close = windows_process_memory(child.pid)
        except OSError as error:
            errors.append(f"RAM telemetry unavailable: {error}")

        def enforce_timeout():
            if child.poll() is None:
                timed_out.set()
                try:
                    child.kill()
                except ProcessLookupError:
                    pass

        timer = threading.Timer(options.timeout_seconds, enforce_timeout)
        monitor = threading.Thread(target=monitor_gpu, args=(
            nvidia_smi, initial["uuid"], start, options.interval_ms / 1000, stop, samples, errors,
            memory_read, memory_samples,
        ), daemon=True)
        timer.start()
        monitor.start()
        try:
            for line in child.stdout:
                elapsed = time.perf_counter() - start
                log.write(f"{elapsed:9.3f} {line}")
                for index in range(phase_cursor, len(PHASE_STARTS)):
                    phase, pattern = PHASE_STARTS[index]
                    if pattern.search(line):
                        phases.append((elapsed, phase))
                        phase_cursor = index + 1
                        break
                marker = NATIVE_TIMES.search(line)
                if marker:
                    native[marker.group(1).lower()] = float(marker.group(2))
            code = child.wait()
            elapsed = time.perf_counter() - start
        finally:
            timer.cancel()
            if child.poll() is None:
                child.kill()
                child.wait()
            child.stdout.close()
            stop.set()
            monitor.join()
            memory_close()

    phase_times = {}
    for (begin, name), (end, _) in zip(phases, phases[1:] + [(elapsed, "exit")]):
        phase_times[name] = round(end - begin, 4)
    return {
        "command": command, "seconds": round(elapsed, 4), "exit_code": code,
        "timed_out": timed_out.is_set(), "log": str(directory / "runner.log"),
        "wall_phase_seconds": phase_times, "native_phase_seconds": native,
        "gpu": gpu_summary(initial, samples, phases, errors),
        "memory": memory_summary(memory_samples, phases),
    }


def pixel_metrics(left, right, image_chops):
    if left.size != right.size:
        return {"exact": False, "reason": "dimensions differ"}
    histogram = image_chops.difference(left, right).histogram()
    count = len(left.getbands()) * left.width * left.height
    errors = [(index % 256, occurrences) for index, occurrences in enumerate(histogram)
              if index % 256 and occurrences]
    different = sum(occurrences for _, occurrences in errors)
    mse = sum(value * value * occurrences for value, occurrences in errors) / count
    return {
        "exact": different == 0, "different_channel_values": different,
        "mean_absolute_channel_error": sum(value * occurrences for value, occurrences in errors) / count,
        "max_channel_error": max((value for value, _ in errors), default=0),
        "psnr_db": None if not mse else 10 * math.log10(255 ** 2 / mse),
    }


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, type=absolute_path)
    parser.add_argument("--baseline-profile", required=True, type=absolute_path,
                        help="saved image configuration (uses default_profile_id) or one profile JSON object")
    parser.add_argument("--candidate-profile", type=absolute_path,
                        help="saved image configuration or one profile JSON object for the candidate")
    parser.add_argument("--candidate-runner", type=absolute_path)
    parser.add_argument("--stock-runner", type=absolute_path,
                        help="optional upstream build at the same commit, using the baseline profile")
    parser.add_argument("--sage", action="store_true",
                        help="force SageAttention in an extra variant; standard runs use the profile setting")
    parser.add_argument("--ram-offload", action="store_true", help="test CUDA with model weights backed by RAM as an extra variant")
    parser.add_argument("--segmented", action="store_true", help="allow bounded native workspace segments while retaining GPU weights")
    parser.add_argument("--minimax-attention-budget-mib", type=int, choices=[32, 64, 128], default=0,
                        help="explicit MiniMax H3 attention-head workspace budget; does not change precision")
    parser.add_argument("--work-dir", required=True, type=absolute_path,
                        help="existing directory in which to create a separate benchmark folder")
    parser.add_argument("--prompt", required=True, type=absolute_path, help="UTF-8 prompt file")
    parser.add_argument("--negative-prompt", type=absolute_path, help="optional UTF-8 negative prompt file")
    parser.add_argument("--reference", type=absolute_path, action="append", default=[])
    parser.add_argument("--vram-mib", required=True, type=int, help="explicit managed VRAM target for every variant")
    parser.add_argument("--gpu-index", type=int, default=0, help="physical nvidia-smi device index")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=1200, help="hard deadline per native runner process")
    parser.add_argument("--interval-ms", type=int, default=500, help="minimum delay between GPU observations")
    parser.add_argument("--run", action="store_true", help="execute generation; omit to print the plan without GPU access")
    parser.add_argument("--require-pixel-parity", action="store_true", help="fail if variants differ from their paired baseline")
    options = parser.parse_args()
    if (not 0 <= options.seed <= 2 ** 63 - 1 or options.gpu_index < 0 or options.repetitions < 1
            or options.vram_mib < 64 or options.interval_ms < 100 or options.timeout_seconds < 1
            or len(options.reference) > 4):
        parser.error("invalid seed, GPU index, repetitions, memory target, sampling interval, timeout or reference count")
    return options


def main():
    options = arguments()
    baseline = load_profile(options.baseline_profile)
    candidate = load_profile(options.candidate_profile) if options.candidate_profile else baseline
    candidate_runner = options.candidate_runner or options.runner
    cases = [("baseline", options.runner, baseline)]
    if options.candidate_profile or options.candidate_runner:
        cases.append(("candidate", candidate_runner, candidate))
    if options.sage:
        cases.append(("sage", candidate_runner, candidate))
    if options.ram_offload:
        cases.append(("ram-offload", candidate_runner, candidate))
    if options.stock_runner:
        cases.append(("stock", options.stock_runner, baseline))
    if not options.run:
        print(json.dumps({
            "mode": "dry-run", "repetitions": options.repetitions, "prompt_file": str(options.prompt),
            "cases": {name: runner_command(runner, profile, options, options.work_dir / "<run>" / name, name)
                      for name, runner, profile in cases},
        }, ensure_ascii=False, indent=2))
        return 0

    from PIL import Image, ImageChops

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi or not options.work_dir.is_dir():
        raise ValueError("--run requires nvidia-smi and an existing --work-dir")
    prompt = options.prompt.read_text(encoding="utf-8-sig")
    negative = options.negative_prompt.read_text(encoding="utf-8-sig") if options.negative_prompt else ""
    if not prompt.strip() or any("<lora:" in value.lower() for value in (prompt, negative)):
        raise ValueError("enter a prompt and select any LoRAs through the profile")
    for reference in options.reference:
        if not reference.is_file():
            raise ValueError(f"missing reference image: {reference}")
    for _, runner, profile in cases:
        assets = [runner, profile["diffusion_model"], profile["text_encoder"], profile["vae"]]
        assets += [item["path"] for item in profile["loras"]]
        if options.reference and profile.get("vision_encoder"):
            assets.append(profile["vision_encoder"])
        if missing := next((path for path in assets if not path.is_file()), None):
            raise ValueError(f"missing local runner or profile asset: {missing}")

    root = Path(tempfile.mkdtemp(prefix="wildimage-benchmark-", dir=options.work_dir))
    report = {"seed": options.seed, "trials": [], "comparisons": []}
    print(f"Benchmark output: {root}", flush=True)
    failed = False
    for repeat in range(1, options.repetitions + 1):
        images = {}
        ordered = cases if repeat % 2 else reversed(cases)
        for name, runner, profile in ordered:
            directory = root / f"{repeat:02d}-{name}"
            print(f"{repeat}/{options.repetitions}: {name}", flush=True)
            trial = run_trial(runner, profile, options, directory, nvidia_smi, name, prompt, negative)
            trial.update(repetition=repeat, case=name)
            report["trials"].append(trial)
            try:
                with Image.open(directory / "image.png", formats=("PNG",)) as image:
                    if image.size != (profile["width"], profile["height"]):
                        raise ValueError(f"unexpected PNG dimensions {image.size}")
                    rgba = image.convert("RGBA")
                images[name] = rgba
                trial["pixel_sha256_rgba8"] = hashlib.sha256(rgba.tobytes()).hexdigest()
            except (OSError, ValueError) as error:
                trial["output_error"] = str(error)
            failed |= trial["exit_code"] != 0 or "output_error" in trial or bool(trial["gpu"]["errors"])
        if "baseline" in images:
            for name, image in images.items():
                if name == "baseline":
                    continue
                metrics = pixel_metrics(images["baseline"], image, ImageChops)
                report["comparisons"].append({"repetition": repeat, "case": name, **metrics})
                failed |= options.require_pixel_parity and not metrics["exact"]
        report["median_seconds"] = {}
        for name, *_ in cases:
            times = [trial["seconds"] for trial in report["trials"] if trial["case"] == name
                     and trial["exit_code"] == 0 and "output_error" not in trial]
            if times:
                report["median_seconds"][name] = statistics.median(times)
        (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(root / "report.json")
    return int(failed)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, ImportError, subprocess.SubprocessError) as error:
        raise SystemExit(f"WildImage benchmark: {error}") from error
