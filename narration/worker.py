"""Local narration worker and session-scoped voice synthesis."""
import argparse
import json
import io
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


def read_reference(path, _work):
    import numpy as np
    sr = 24000
    proc = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", *audio_input(path),
                           "-vn", "-t", "61", "-ac", "1", "-ar", str(sr), "-f", "s16le", "pipe:1"],
                          capture_output=True, timeout=60)
    if proc.returncode:
        raise ValueError(proc.stderr.decode("utf8", "replace")[-1200:])
    samples = np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768
    if not sr <= len(samples) <= sr * 60:
        raise ValueError("Use uma amostra de voz de 1 a 60 segundos.")
    return samples, sr


def volume_filters(cfg):
    volume = cfg.get("volume", 100)
    if type(volume) not in (float, int) or not 0 <= volume <= 300:
        raise ValueError("Use um volume entre 0% e 300%.")
    if volume == 100:
        return []
    filters = [f"volume={volume / 100:.8f}"]
    if volume > 100:
        filters.append("alimiter=limit=0.98:level=0:latency=1")
    return filters


def voice_filter(cfg, rate, *, apply_volume=True, frames=None):
    pitch = 2 ** (cfg["pitch"] / 12)
    tempo = cfg["speed"] / pitch
    resonance = cfg.get("resonance", 0)
    preserve = cfg.get("preserveFormants", False)
    de_ess, compression = cfg.get("deEss", 0), cfg.get("compression", 0)
    if (type(preserve) is not bool or any(type(v) not in (int, float) or not low <= v <= high
            for v, low, high in ((resonance, -5, 5), (de_ess, 0, 100), (compression, 0, 100)))):
        raise ValueError("Ajustes de timbre inválidos.")
    if resonance or (preserve and pitch != 1):
        # Resampling shifts the spectral envelope. Rubber Band then sets the requested
        # pitch and tempo independently, preserving that new envelope.
        shift = 2 ** (resonance / 12) * (1 if preserve else pitch)
        source_rate = round(rate * shift)
        shift = source_rate / rate
        filters = ["apad=pad_dur=0.25", f"asetrate={source_rate}", "aresample=24000",
                   f"rubberband=pitch={pitch/shift:.8f}:tempo={cfg['speed']/shift:.8f}:formant=preserved:pitchq=quality"]
        if frames is not None:
            length = max(1, round(frames * 24000 / (rate * cfg["speed"])))
            filters.extend([f"apad=whole_len={length}", f"atrim=end_sample={length}"])
    else:
        filters = [f"asetrate={round(rate * pitch)}", "aresample=24000"]
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
    if de_ess:
        filters.append(f"deesser=i={de_ess * .007:.5f}:m=0.5:s=o")
    if compression:
        filters.append(f"acompressor=threshold=0.16:ratio={1 + compression * .035:.5f}"
                       f":attack=12:release=140:makeup={1 + compression * .0035:.5f}")
    if max(body, brightness) > 0 or compression or resonance or (preserve and pitch != 1):
        filters.append("alimiter=limit=0.98:level=0:latency=1")
    if apply_volume:
        filters.extend(volume_filters(cfg))
    return ",".join(filters)


def preview_filter(cfg, rate, frames):
    return voice_filter(cfg, rate, frames=frames) + ",atrim=duration=12,afade=t=out:st=11.94:d=0.06"


def remix_stream():
    """Remix the retained dry WAV through pipes, without loading a voice model."""
    cfg = json.loads(sys.stdin.buffer.readline(512))
    source = sys.stdin.buffer.read(2 * 1024 * 1024 + 1)
    if len(source) > 2 * 1024 * 1024:
        raise ValueError("Amostra de prévia inválida.")
    with wave.open(io.BytesIO(source), "rb") as audio:
        rate, frames = audio.getframerate(), audio.getnframes()
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "wav", "-i", "pipe:0",
           "-af", preview_filter(cfg, rate, frames), "-ac", "1", "-f", "s16le", "pipe:1"]
    run = subprocess.run(cmd, input=source, capture_output=True, timeout=15)
    if run.returncode or not run.stdout:
        raise RuntimeError(run.stderr.decode("utf8", "replace")[-1200:] or "Prévia sem áudio.")
    with wave.open(sys.stdout.buffer, "wb") as result:
        result.setparams((1, 2, 24000, len(run.stdout)//2, "NONE", "not compressed"))
        result.writeframes(run.stdout)


def finish_audio(source, target, cfg, rate, preview=False):
    with wave.open(str(source), "rb") as audio:
        frames = audio.getnframes()
    filters = preview_filter(cfg, rate, frames) if preview else voice_filter(cfg, rate, frames=frames)
    ff("-i", source, "-af", filters, "-ac", "1", "-c:a", "pcm_s16le", target)


def make_voice(root, index):
    return IndexVoice(root) if index else QwenVoice(root, load_model)


def spoken_chunks(chunks, engine, index, cfg, work):
    import numpy as np
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
        yield i, pcm, rate, chunk["paragraph_end"]


def finish_segment(i, pcm, rate, paragraph_end, cfg, reading, work):
    import soundfile as sf
    dry = io.BytesIO()
    sf.write(dry, pcm, rate, format="WAV", subtype="PCM_16")
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "wav",
                           "-i", "pipe:0", "-af", voice_filter(cfg, rate, apply_volume=reading is not None,
                                                               frames=len(pcm)),
                           "-ac", "1", "-f", "s16le", "pipe:1"], input=dry.getvalue(), capture_output=True)
    if proc.returncode or not proc.stdout:
        raise RuntimeError(proc.stderr.decode("utf8", "replace")[-1200:] or "Leitura sem áudio.")
    if reading is None:
        with wave.open(str(work / f"{i}.wav"), "wb") as segment:
            segment.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            segment.writeframes(proc.stdout)
    else:
        reading.writeframesraw(proc.stdout)
        if paragraph_end:
            reading.writeframesraw(bytes(2 * round(24000 * cfg.get("pauseMs", 350) / 1000)))


def synth(root, cfg, work, preview=False, engine=None, reading=None):
    progress(1, "Iniciando o motor de voz")
    index = cfg.get("engine") == "index"
    progress(2, ("Reutilizando " if engine is not None else "Carregando ") + (INDEX if index else QWEN))
    chunks = [preview_chunk(cfg)] if preview else speech_plan(cfg)
    if engine is None:
        engine = make_voice(root, index)
    if index:
        engine.references.clear()  # Normalized files live in the current job directory.
    spoken = spoken_chunks(chunks, engine, index, cfg, work)
    if preview:
        import soundfile as sf
        _, pcm, rate, _ = next(spoken)
        # Twelve seconds at up to 2x speed, plus filter latency.
        source = work / "source.wav"
        sf.write(source, pcm[:rate * 26], rate)
        finish_audio(source, work / "preview.wav", cfg, rate, preview=True)
    elif len(chunks) == 1 or not str(getattr(engine.model, "device", "")).startswith("cuda"):
        for chunk in spoken:
            finish_segment(*chunk, cfg, reading, work)
    else:
        # Overlap the previous CPU audio filter with synthesis of the next chunk.
        # One writer owns the WAV; at most one filtered chunk can be outstanding.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as writer:
            pending = None
            for chunk in spoken:
                if pending is not None:
                    pending.result()
                pending = writer.submit(finish_segment, *chunk, cfg, reading, work)
            pending.result()
    progress(66, "Narração pronta")


def align(root, cfg, work):
    progress(67, "Iniciando o motor de alinhamento")
    import numpy as np
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
            # Synthesis writes mono PCM16. Read once for both alignment and the master WAV.
            with wave.open(str(p), "rb") as part:
                sr = part.getframerate()
                pcm = part.readframes(part.getnframes())
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
            marks = list(model.align(audio=(samples, sr), text=chunk["spoken"], language="English")[0])
            if not marks:
                raise RuntimeError(f"Sem alinhamento para o trecho {i + 1}.")
            captions.extend(caption_times(chunk["units"], marks, len(samples)/sr, offset/sr))
            combined.writeframesraw(pcm)
            offset += len(samples)
            if chunk["paragraph_end"]:
                silence = round(sr * cfg.get("pauseMs", 350) / 1000)
                combined.writeframesraw(bytes(2 * silence))
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


def still_background(bg):
    if bg.suffix.lower() != ".png":
        return bg.suffix.lower() in (".jpg", ".jpeg", ".bmp")
    # APNG declares its animation before the first IDAT. Skip metadata without decoding it.
    with bg.open("rb") as stream:
        if stream.read(8) != b"\x89PNG\r\n\x1a\n":
            return False
        while len(header := stream.read(8)) == 8:
            kind = header[4:]
            if kind == b"acTL":
                return False
            if kind == b"IDAT":
                return True
            stream.seek(int.from_bytes(header[:4], "big") + 4, 1)
    return False


def render(cfg, work):
    write_captions(work, cfg)
    bg = Path(cfg["background"])
    w, h, fps = cfg["width"], cfg["height"], cfg["fps"]
    with wave.open(str(work / "speech.wav"), "rb") as audio:
        duration = audio.getnframes() / audio.getframerate()
    static = still_background(bg)
    source = ["-framerate", str(fps)] if static else ["-stream_loop", "-1"]
    filt = f"fps={fps},scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    if static:
        # Repeat the scaled frame. Animated captions still render at each output timestamp.
        filt += ",loop=loop=-1:size=1:start=0,setpts=N"
    filt += ",ass=captions.ass"
    # Align with audible speech even when the user mutes the finished video.
    gain = volume_filters(cfg)
    audio_args = ["-af", ",".join(gain)] if gain else []
    progress(83, "Montando o vídeo com legendas")
    high = 83
    for encoder in ("h264_nvenc", "libx264"):
        frames = 0
        args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *source, *audio_input(bg),
                "-i", "speech.wav", "-map", "0:v:0", "-map", "1:a:0", "-vf", filt, *audio_args, "-t", f"{duration:.8f}",
                "-c:v", encoder, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                "-progress", "pipe:1", "video.mp4"]
        with (work / "ffmpeg.log").open("wb") as log, subprocess.Popen(args, cwd=work, stdout=subprocess.PIPE,
                                                                        stderr=log, text=True) as proc:
            for line in proc.stdout:
                if line.startswith("frame="):
                    frames = int(line.partition("=")[2])
                elif line.startswith("out_time_us="):
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
    if frames < 1:
        raise RuntimeError("O codificador não produziu quadros.")
    ff("-ss", f"{(frames // 2) / fps:.8f}", "-i", work / "video.mp4", "-frames:v", "1",
       "-vf", "scale=320:-2", work / "poster.jpg")
    return frames


def save_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), "utf8")
    os.replace(tmp, path)


def read_aloud(root, cfg, job, engine=None):
    work = job / "working"
    work.mkdir()
    try:
        with wave.open(str(job / "speech.wav"), "wb") as result:
            result.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            synth(root, cfg, work, engine=engine, reading=result)
        progress(90, "Leitura pronta")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def voice_session(root, kind):
    if kind not in ("tts", "index"):
        raise ValueError("Motor de voz inválido.")
    engine = None
    # The parent owns stdin. An idle worker exits on EOF even if the app crashes.
    for line in sys.stdin:
        request = json.loads(line)
        stage = request["stage"]
        job = Path(request["dir"])
        cfg = json.loads((job / "request.json").read_text("utf-8-sig"))
        if (cfg.get("engine") == "index") != (kind == "index"):
            raise ValueError("O motor carregado é diferente do solicitado.")
        if engine is None:
            progress(2, "Carregando " + (INDEX if kind == "index" else QWEN))
            engine = make_voice(root, kind == "index")
        if stage == "preview":
            synth(root, cfg, job, preview=True, engine=engine)
        elif stage == "read":
            read_aloud(root, cfg, job, engine)
        elif stage == "synth":
            work = job / "working"
            work.mkdir()
            synth(root, cfg, work, engine=engine)
        elif stage == "save-voice" and kind == "tts":
            engine.save_voice(job / "voice.safetensors", cfg["reference"], cfg.get("transcript", ""),
                              job, read_reference, progress)
        else:
            raise ValueError("Operação de voz inválida.")
        print("@voice-done " + json.dumps({"device": str(engine.model.device)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["run", "synth", "align", "transcribe", "preview", "preview-remix", "save-voice", "read", "finish-video"])
    parser.add_argument("job", type=Path)
    parser.add_argument("models", type=Path)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--elapsed", type=float, default=0)
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
        return read_aloud(opts.models, cfg, job)
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
    start = time.monotonic() - opts.elapsed
    stages = [("align", "aligner")]
    if opts.stage != "finish-video":
        work.mkdir(exist_ok=False)
        stages.insert(0, ("synth", "index" if cfg.get("engine") == "index" else "tts"))
    try:
        for stage, env in stages:
            py = opts.runtime / env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            subprocess.run([str(py), "-u", __file__, stage, str(job), str(opts.models), str(opts.runtime)], check=True)
        frames = render(cfg, work)
        for name in ("video.mp4", "poster.jpg"):
            os.replace(work / name, job / name)
        save_json(job / "record.json", dict(id=cfg["id"], created_at=cfg["created_at"], frames=frames,
                  fps=cfg["fps"], generation_seconds=round(time.monotonic()-start, 2), path=str(job.parent / cfg["id"] / "video.mp4"),
                  details=dict(prompt=cfg["text"], model=INDEX if cfg.get("engine") == "index" else QWEN, pause_ms=cfg.get("pauseMs",350),
                               pronunciations=cfg.get("pronunciations",[]),
                               directions={k: {**{a:v for a,v in d.items() if a in ("emotion","strength")},
                                               "reference":Path(d.get("reference","")).name} for k,d in cfg.get("directions",{}).items()}, width=cfg["width"], height=cfg["height"],
                               speed=cfg["speed"], pitch=cfg["pitch"], body=cfg.get("body", 0),
                               brightness=cfg.get("brightness", 0), volume=cfg.get("volume", 100),
                               resonance=cfg.get("resonance", 0), preserveFormants=cfg.get("preserveFormants", False),
                               deEss=cfg.get("deEss", 0), compression=cfg.get("compression", 0), captions=cfg["captions"],
                               reference=cfg.get("voiceName") or Path(cfg["reference"]).name, background=Path(cfg["background"]).name)))
        progress(100, "Vídeo salvo")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    try:
        if sys.argv[1:] == ["preview-pipe"]:
            remix_stream()
        elif len(sys.argv) == 4 and sys.argv[1] == "voice-session":
            voice_session(Path(sys.argv[2]), sys.argv[3])
        else:
            main()
    except Exception as error:
        if sys.argv[1:] != ["preview-pipe"] and not isinstance(error, subprocess.CalledProcessError):
            print("@failure " + str(error).replace("\n", " ")[:1500], flush=True)
        raise
