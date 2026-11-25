"""Local Wildllama narration worker. Each neural stage owns its CUDA context."""
import argparse
import json
import io
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import wave

from speech_plan import speech_plan, caption_times, preview_chunk
from voice_engines import QWEN, INDEX, QwenVoice, IndexVoice, transcribe

ALIGN = "Qwen3-ForcedAligner-0.6B"


def progress(percent, message):
    print("@narration " + json.dumps({"percent": percent, "message": message}), flush=True)


def ff(*args):
    run = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
                         capture_output=True)
    if run.returncode:
        raise RuntimeError(run.stderr.decode("utf8", "replace")[-1600:])


def audio_input(path):
    return ["-protocol_whitelist", "file,pipe", "-i", str(path)]


def load_model(cls, directory):
    import torch
    gpu = torch.cuda.is_available()
    return cls.from_pretrained(str(directory), device_map="cuda:0" if gpu else "cpu",
                               dtype=torch.bfloat16 if gpu and torch.cuda.is_bf16_supported() else
                               torch.float16 if gpu else torch.float32, attn_implementation="sdpa",
                               local_files_only=True)


def read_reference(path, work):
    import soundfile as sf
    ref = work / "reference.wav"
    ff(*audio_input(path), "-vn", "-t", "61", "-ac", "1", "-ar", "24000", ref)
    samples, sr = sf.read(ref, dtype="float32")
    ref.unlink()
    if not sr <= len(samples) <= sr * 60:
        raise ValueError("Use uma amostra de voz de 1 a 60 segundos.")
    return samples, sr


def voice_filter(cfg, rate):
    pitch = 2 ** (cfg["pitch"] / 12)
    tempo = cfg["speed"] / pitch
    filters = [f"asetrate={round(rate * pitch)}", "aresample=24000"]
    # Keep each time-stretch in the high-quality 0.5–2 range.
    while tempo < 0.5 or tempo > 2:
        step = 0.5 if tempo < 0.5 else 2
        filters.append(f"atempo={step}")
        tempo /= step
    filters.append(f"atempo={tempo:.8f}")
    body, brightness = cfg.get("body", 0), cfg.get("brightness", 0)
    if body or brightness:
        filters.append("aformat=sample_fmts=fltp")
        if body:
            filters.append(f"bass=f=300:g={body}")
        if brightness:
            filters.append(f"treble=f=3200:g={brightness}")
        if max(body, brightness) > 0:
            filters.append("alimiter=limit=0.98:level=0:latency=1")
    return ",".join(filters)


def preview_filter(cfg, rate):
    return voice_filter(cfg, rate) + ",atrim=duration=12,afade=t=out:st=11.94:d=0.06"


def remix_stream():
    """Remix the retained dry WAV through pipes, without loading a voice model."""
    cfg = json.loads(sys.stdin.buffer.readline(512))
    source = sys.stdin.buffer.read(2 * 1024 * 1024 + 1)
    if len(source) > 2 * 1024 * 1024:
        raise ValueError("Amostra de prévia inválida.")
    with wave.open(io.BytesIO(source), "rb") as audio:
        rate = audio.getframerate()
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "wav", "-i", "pipe:0",
           "-af", preview_filter(cfg, rate), "-ac", "1", "-f", "s16le", "pipe:1"]
    run = subprocess.run(cmd, input=source, capture_output=True, timeout=15)
    if run.returncode or not run.stdout:
        raise RuntimeError(run.stderr.decode("utf8", "replace")[-1200:] or "Prévia sem áudio.")
    with wave.open(sys.stdout.buffer, "wb") as result:
        result.setparams((1, 2, 24000, len(run.stdout)//2, "NONE", "not compressed"))
        result.writeframes(run.stdout)


def finish_audio(source, target, cfg, rate, preview=False):
    filters = preview_filter(cfg, rate) if preview else voice_filter(cfg, rate)
    ff("-i", source, "-af", filters, "-ac", "1", "-c:a", "pcm_s16le", target)


def synth(root, cfg, work, preview=False):
    progress(1, "Iniciando o motor de voz")
    import numpy as np
    import soundfile as sf
    index = cfg.get("engine") == "index"
    progress(2, "Carregando " + (INDEX if index else QWEN))
    chunks = [preview_chunk(cfg)] if preview else speech_plan(cfg)
    engine = IndexVoice(root) if index else QwenVoice(root, load_model)
    for i, chunk in enumerate(chunks):
        settings = cfg.get("directions", {}).get(str(chunk["paragraph"]), cfg.get("directions", {}).get("default", {}))
        if not index and cfg.get("voiceProfile") and not settings.get("reference"):
            settings = {**settings, "voiceProfile": cfg["voiceProfile"]}
        reference = cfg["reference"] if index else settings.get("reference") or cfg["reference"]
        transcript = settings.get("transcript","") if settings.get("reference") else cfg.get("transcript","")
        pct = 8 + int(58 * i / len(chunks))
        message = f"Narrando trecho {i+1} de {len(chunks)}"
        progress(pct, message)
        def report(fraction, detail):
            progress(pct + int(52 * fraction / len(chunks)), f"{message} · {detail}")
        audio, rate = engine.speak(chunk["spoken"], reference, transcript, settings, work, read_reference, report)
        report(1, "aplicando os ajustes de voz")
        pcm = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not pcm.size or not np.isfinite(pcm).all():
            raise RuntimeError("A síntese produziu áudio inválido.")
        if preview:
            # Twelve seconds at up to 2x speed, plus filter latency.
            source = work / "source.wav"
            sf.write(source, pcm[:rate * 26], rate)
            finish_audio(source, work / "preview.wav", cfg, rate, preview=True)
        else:
            sf.write(work / "segment.wav", pcm, rate)
            finish_audio(work / "segment.wav", work / f"{i}.wav", cfg, rate)
    if not preview:
        (work / "segment.wav").unlink()
    progress(66, "Narração pronta")


def align(root, cfg, work):
    progress(67, "Iniciando o motor de alinhamento")
    import soundfile as sf
    from qwen_asr import Qwen3ForcedAligner
    progress(68, "Carregando o alinhador")
    model = load_model(Qwen3ForcedAligner, root / ALIGN)
    chunks = speech_plan(cfg)
    offset, captions = 0, []
    with wave.open(str(work / "speech.wav"), "wb") as combined:
        combined.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        for i, chunk in enumerate(chunks):
            progress(69 + int(13 * i / len(chunks)), f"Sincronizando trecho {i + 1} de {len(chunks)}")
            p = work / f"{i}.wav"
            samples, sr = sf.read(p, dtype="float32")
            marks = list(model.align(audio=(samples, sr), text=chunk["spoken"], language="English")[0])
            if not marks:
                raise RuntimeError(f"Sem alinhamento para o trecho {i + 1}.")
            captions.extend(caption_times(chunk["units"], marks, len(samples)/sr, offset/sr))
            with wave.open(str(p), "rb") as part:
                combined.writeframes(part.readframes(part.getnframes()))
            offset += len(samples)
            if chunk["paragraph_end"]:
                silence = round(sr * cfg.get("pauseMs", 350) / 1000)
                combined.writeframes(bytes(2 * silence))
                offset += silence
            p.unlink()
    save_json(work / "words.json", captions)
    progress(82, "Texto sincronizado")


def ass_time(seconds):
    t = round(seconds * 100)
    return f"{t // 360000}:{t // 6000 % 60:02d}:{t // 100 % 60:02d}.{t % 100:02d}"


def write_captions(work, cfg):
    words = json.loads((work / "words.json").read_text("utf8"))
    width, height = cfg["width"], cfg["height"]
    size = max(28, round(min(width, height) * .061))
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,40,40,{height // 12},1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    group = []
    def flush():
        if not group:
            return
        # Neutralise ASS override syntax in user text.
        labels = [str(w[2]).replace("\\", "＼").replace("{", "｛").replace("}", "｝") for w in group]
        for j, word in enumerate(group):
            if cfg["captions"] == "phrase" and j:
                break
            if cfg["captions"] == "phrase":
                start, end, display = group[0][0], group[-1][1], labels
            else:
                start = word[0]
                end = group[j + 1][0] if j + 1 < len(group) else word[1]
                display = labels.copy()
                display[j] = r"{\c&H72E7FF&}" + display[j] + r"{\r}"
            if ass_time(start) != ass_time(end) and end > start:
                lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{' '.join(display)}\n")
        group.clear()
    for w in words:
        if group and (sum(len(v[2])+1 for v in group) + len(w[2]) > (36 if width < height else 54)
                      or w[0] - group[-1][1] > .48):
            flush()
        group.append(w)
        if re.search(r"[.!?]$", w[2]):
            flush()
    flush()
    (work / "captions.ass").write_text("".join(lines), encoding="utf-8-sig")


def render(cfg, work):
    write_captions(work, cfg)
    bg = Path(cfg["background"])
    w, h, fps = cfg["width"], cfg["height"], cfg["fps"]
    with wave.open(str(work / "speech.wav"), "rb") as audio:
        duration = audio.getnframes() / audio.getframerate()
    static = bg.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    source = ["-loop", "1", "-framerate", str(fps)] if static else ["-stream_loop", "-1"]
    filt = f"fps={fps},scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,ass=captions.ass"
    progress(83, "Montando o vídeo com legendas")
    high = 83
    for encoder in ("h264_nvenc", "libx264"):
        args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *source, *audio_input(bg),
                "-i", "speech.wav", "-map", "0:v:0", "-map", "1:a:0", "-vf", filt, "-t", f"{duration:.8f}",
                "-c:v", encoder, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                "-progress", "pipe:1", "video.mp4"]
        with (work / "ffmpeg.log").open("wb") as log, subprocess.Popen(args, cwd=work, stdout=subprocess.PIPE,
                                                                        stderr=log, text=True) as proc:
            for line in proc.stdout:
                if line.startswith("out_time_us="):
                    try:
                        current = min(97, 83 + int(14 * int(line.split("=")[1]) / 1e6 / duration))
                    except ValueError:
                        continue
                    if current > high:
                        high = current
                        progress(high, "Codificando o vídeo")
            if not proc.wait():
                break
        if encoder == "libx264":
            raise RuntimeError((work / "ffmpeg.log").read_text("utf8", errors="replace")[-1500:])
    progress(98, "Salvando vídeo e miniatura")
    ff("-ss", f"{math.floor(duration * fps / 2) / fps:.8f}", "-i", work / "video.mp4", "-frames:v", "1",
       "-vf", "scale=320:-2", work / "poster.jpg")
    return duration


def save_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), "utf8")
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["run", "synth", "align", "transcribe", "preview", "preview-remix", "save-voice", "read"])
    parser.add_argument("job", type=Path)
    parser.add_argument("models", type=Path)
    parser.add_argument("runtime", type=Path)
    opts = parser.parse_args()
    job = opts.job.resolve()
    if opts.stage == "transcribe":
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as folder:
            text = transcribe(opts.models, str(job), Path(folder), read_reference, load_model, progress)
            print("@transcript " + json.dumps(text), flush=True)
        return
    cfg = json.loads((job / "request.json").read_text("utf-8-sig"))
    work = job / "working"
    if opts.stage == "save-voice":
        progress(8, "Carregando o Qwen3-TTS")
        voice = QwenVoice(opts.models, load_model)
        voice.save_voice(job / "voice.safetensors", cfg["reference"], cfg.get("transcript", ""),
                         job, read_reference, progress)
        progress(95, "Dados da voz preparados")
        return
    if opts.stage == "read":
        work.mkdir()
        try:
            synth(opts.models, cfg, work)
            progress(90, "Montando o áudio da leitura")
            with wave.open(str(job / "speech.wav"), "wb") as result:
                result.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                for i, chunk in enumerate(speech_plan(cfg)):
                    part = work / f"{i}.wav"
                    with wave.open(str(part), "rb") as src:
                        while data := src.readframes(65536):
                            result.writeframesraw(data)
                    part.unlink()
                    if chunk["paragraph_end"]:
                        result.writeframesraw(bytes(2 * round(24000 * cfg.get("pauseMs", 350) / 1000)))
            return
        finally:
            shutil.rmtree(work, ignore_errors=True)
    if opts.stage == "preview-remix":
        source = job / "source.wav"
        with wave.open(str(source), "rb") as audio:
            rate = audio.getframerate()
        return finish_audio(source, job / "preview.wav", cfg, rate, preview=True)
    if opts.stage == "preview":
        return synth(opts.models, cfg, job, preview=True)
    if opts.stage == "synth":
        return synth(opts.models, cfg, work)
    if opts.stage == "align":
        return align(opts.models, cfg, work)
    start = time.monotonic()
    work.mkdir(exist_ok=False)
    try:
        for stage, env in (("synth", "index" if cfg.get("engine") == "index" else "tts"), ("align", "aligner")):
            py = opts.runtime / env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            subprocess.run([str(py), "-u", __file__, stage, str(job), str(opts.models), str(opts.runtime)], check=True)
        duration = render(cfg, work)
        for name in ("video.mp4", "poster.jpg"):
            os.replace(work / name, job / name)
        save_json(job / "record.json", dict(id=cfg["id"], created_at=cfg["created_at"], frames=math.ceil(duration * cfg["fps"]),
                  fps=cfg["fps"], generation_seconds=round(time.monotonic()-start, 2), path=str(job.parent / cfg["id"] / "video.mp4"),
                  details=dict(prompt=cfg["text"], model=INDEX if cfg.get("engine") == "index" else QWEN, pause_ms=cfg.get("pauseMs",350),
                               pronunciations=cfg.get("pronunciations",[]),
                               directions={k: {**{a:v for a,v in d.items() if a in ("emotion","strength")},
                                               "reference":Path(d.get("reference","")).name} for k,d in cfg.get("directions",{}).items()}, width=cfg["width"], height=cfg["height"],
                               speed=cfg["speed"], pitch=cfg["pitch"], body=cfg.get("body", 0),
                               brightness=cfg.get("brightness", 0), captions=cfg["captions"],
                               reference=cfg.get("voiceName") or Path(cfg["reference"]).name, background=Path(cfg["background"]).name)))
        progress(100, "Vídeo salvo")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    try:
        if sys.argv[1:] == ["preview-pipe"]:
            remix_stream()
        else:
            main()
    except Exception as error:
        if sys.argv[1:] != ["preview-pipe"] and not isinstance(error, subprocess.CalledProcessError):
            print("@failure " + str(error).replace("\n", " ")[:1500], flush=True)
        raise
