"""Local Wildllama narration worker. Each neural stage owns its CUDA context."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import wave

TTS = "Qwen3-TTS-12Hz-1.7B-Base"
ALIGN = "Qwen3-ForcedAligner-0.6B"


def progress(percent, message):
    print("@narration " + json.dumps({"percent": percent, "message": message}), flush=True)


def split_text(text, limit=340):
    parts, line = [], ""
    for word in text.split():
        if line and (len(line) + len(word) > limit or
                     (len(line) > limit // 2 and re.search(r"""[.!?]["')]*$""", line))):
            parts.append(line)
            line = ""
        line = (line + " " + word).strip()
    if line:
        parts.append(line)
    if not parts or any(len(part) > limit * 2 for part in parts):
        raise ValueError("O texto precisa conter palavras separadas por espaços.")
    return parts


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


def synth(root, cfg, work):
    progress(1, "Iniciando o motor de voz")
    import numpy as np
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel
    progress(2, "Carregando o modelo de voz")
    ref = work / "reference.wav"
    ff(*audio_input(cfg["reference"]), "-vn", "-t", "61", "-ac", "1", "-ar", "24000", ref)
    ref_audio, sr = sf.read(ref, dtype="float32")
    ref.unlink()
    if len(ref_audio) > sr * 60:
        raise ValueError("Selecione uma amostra de voz de até 60 segundos.")
    if len(ref_audio) < sr:
        raise ValueError("Use uma amostra de voz com pelo menos um segundo.")
    model = load_model(Qwen3TTSModel, root / TTS)
    transcript = cfg.get("transcript", "").strip()
    progress(6, "Lendo a voz de referência")
    voice = model.create_voice_clone_prompt(ref_audio=(ref_audio, sr), ref_text=transcript or None,
                                           x_vector_only_mode=not transcript)
    chunks = split_text(cfg["text"])
    for i, text in enumerate(chunks):
        progress(8 + int(58 * i / len(chunks)), f"Narrando trecho {i + 1} de {len(chunks)}")
        # Count completed acoustic blocks without moving a tensor off the GPU.
        blocks = 0
        def count_block(_module, _inputs, _outputs):
            nonlocal blocks
            blocks += 1
            if blocks % 24 == 0:
                progress(8 + int(58 * i / len(chunks)),
                         f"Narrando trecho {i + 1} de {len(chunks)} · {blocks} blocos de áudio")
        handle = model.model.talker.register_forward_hook(count_block)
        try:
            audios, rate = model.generate_voice_clone(text=text, language="English", voice_clone_prompt=voice)
        finally:
            handle.remove()
        pcm = np.asarray(audios[0], dtype=np.float32).reshape(-1)
        if not pcm.size or not np.isfinite(pcm).all():
            raise RuntimeError("A síntese produziu áudio inválido.")
        sf.write(work / "segment.wav", pcm, rate)
        pitch = 2 ** (cfg["pitch"] / 12)
        ff("-i", work / "segment.wav", "-af",
           f"asetrate={round(rate * pitch)},aresample=24000,atempo={cfg['speed'] / pitch:.8f}",
           "-ac", "1", "-c:a", "pcm_s16le", work / f"{i}.wav")
    (work / "segment.wav").unlink()
    progress(66, "Narração pronta")


def align(root, cfg, work):
    progress(67, "Iniciando o motor de alinhamento")
    import soundfile as sf
    from qwen_asr import Qwen3ForcedAligner
    progress(68, "Carregando o alinhador")
    model = load_model(Qwen3ForcedAligner, root / ALIGN)
    chunks = split_text(cfg["text"])
    offset, captions = 0, []
    with wave.open(str(work / "speech.wav"), "wb") as combined:
        combined.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        for i, text in enumerate(chunks):
            progress(69 + int(13 * i / len(chunks)), f"Sincronizando trecho {i + 1} de {len(chunks)}")
            p = work / f"{i}.wav"
            samples, sr = sf.read(p, dtype="float32")
            marks = list(model.align(audio=(samples, sr), text=text, language="English")[0])
            if not marks:
                raise RuntimeError(f"Sem alinhamento para o trecho {i + 1}.")
            # The English aligner strips punctuation. Recover it from the original
            # whitespace-delimited words whenever its token boundaries match.
            words = [w for w in text.split() if any(ch.isalnum() for ch in w)]
            for index, mark in enumerate(marks):
                start = max(0., min(len(samples) / sr, mark.start_time))
                end = max(start, min(len(samples) / sr, mark.end_time))
                label = words[index] if len(words) == len(marks) else mark.text
                captions.append([offset / sr + start, offset / sr + end, label])
            with wave.open(str(p), "rb") as part:
                combined.writeframes(part.readframes(part.getnframes()))
            offset += len(samples)
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
    parser.add_argument("stage", choices=["run", "synth", "align"])
    parser.add_argument("job", type=Path)
    parser.add_argument("models", type=Path)
    parser.add_argument("runtime", type=Path)
    opts = parser.parse_args()
    job = opts.job.resolve()
    cfg = json.loads((job / "request.json").read_text("utf-8-sig"))
    work = job / "working"
    if opts.stage == "synth":
        return synth(opts.models, cfg, work)
    if opts.stage == "align":
        return align(opts.models, cfg, work)
    start = time.monotonic()
    work.mkdir(exist_ok=False)
    try:
        for stage, env in (("synth", "tts"), ("align", "aligner")):
            py = opts.runtime / env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            subprocess.run([str(py), "-u", __file__, stage, str(job), str(opts.models), str(opts.runtime)], check=True)
        duration = render(cfg, work)
        for name in ("video.mp4", "poster.jpg"):
            os.replace(work / name, job / name)
        save_json(job / "record.json", dict(id=cfg["id"], created_at=cfg["created_at"], frames=math.ceil(duration * cfg["fps"]),
                  fps=cfg["fps"], generation_seconds=round(time.monotonic()-start, 2), path=str(job.parent / cfg["id"] / "video.mp4"),
                  details=dict(prompt=cfg["text"], model=TTS, width=cfg["width"], height=cfg["height"],
                               speed=cfg["speed"], pitch=cfg["pitch"], captions=cfg["captions"],
                               reference=Path(cfg["reference"]).name, background=Path(cfg["background"]).name)))
        progress(100, "Vídeo salvo")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if not isinstance(error, subprocess.CalledProcessError):
            print("@failure " + str(error).replace("\n", " ")[:1500], flush=True)
        raise
