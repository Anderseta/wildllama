"""Qwen voice prompts, expression vectors and optional IndexTTS2 synthesis."""
from dataclasses import replace
from functools import lru_cache

QWEN = "Qwen3-TTS-12Hz-1.7B-Base"
INDEX = "IndexTTS-2"
ASR = "Qwen3-ASR-0.6B"
EMOTIONS = {"happy": 1.5, "sad": 2.5, "angry": 2.5}


def transcribe(root, reference, work, read_reference, load_model):
    from qwen_asr import Qwen3ASRModel
    samples = read_reference(reference, work)
    asr = load_model(Qwen3ASRModel, root / ASR)
    result = asr.transcribe(audio=samples, language=None)
    text = result[0].text.strip() if result else ""
    if not text:
        raise ValueError("O ASR não reconheceu fala na amostra.")
    return text


class QwenVoice:
    def __init__(self, root, load_model):
        from qwen_tts import Qwen3TTSModel
        self.model = load_model(Qwen3TTSModel, root / QWEN)
        self.root = root
        self.prompt = lru_cache(maxsize=4)(self._prompt)
        self.saved_prompt = lru_cache(maxsize=1)(self._saved_prompt)
        self.vectors = {}

    def _prompt(self, reference, transcript, work, read_reference):
        samples = read_reference(reference, work)
        return self.model.create_voice_clone_prompt(ref_audio=samples, ref_text=transcript or None,
                                                   x_vector_only_mode=not transcript)

    def save_voice(self, target, reference, transcript, work, read_reference):
        from safetensors.torch import save_file
        item, = self.prompt(reference, transcript, work, read_reference)
        tensors = {"speaker": item.ref_spk_embedding.detach().cpu().contiguous()}
        if item.icl_mode:
            tensors["codes"] = item.ref_code.detach().cpu().contiguous()
        save_file(tensors, str(target), metadata={"format": "wildllama-qwen-voice-v1", "model": QWEN,
                  "mode": "icl" if item.icl_mode else "speaker", "text": item.ref_text or ""})

    def _saved_prompt(self, file):
        from safetensors import safe_open
        from qwen_tts import VoiceClonePromptItem
        with safe_open(str(file), framework="pt", device="cpu") as voice:
            meta = voice.metadata() or {}
            if (meta.get("format") != "wildllama-qwen-voice-v1" or meta.get("model") != QWEN
                    or meta.get("mode") not in ("icl", "speaker")):
                raise ValueError("A voz salva é incompatível com este Qwen.")
            icl = meta["mode"] == "icl"
            return [VoiceClonePromptItem(ref_spk_embedding=voice.get_tensor("speaker"),
                    ref_code=voice.get_tensor("codes") if icl else None, x_vector_only_mode=not icl,
                    icl_mode=icl, ref_text=meta["text"] if icl else None)]

    def speak(self, text, reference, transcript, settings, work, read_reference, on_blocks):
        import torch
        profile = settings.get("voiceProfile")
        prompt = self.saved_prompt(profile) if profile else self.prompt(reference, transcript, work, read_reference)
        emotion, strength = settings.get("emotion"), settings.get("strength", 0)
        if emotion in EMOTIONS and strength > 0:
            if emotion not in self.vectors:
                file = self.root / "emotion-vectors" / f"tau_{emotion}_avg4spk.pt"
                if not file.is_file():
                    raise ValueError("Instale os vetores experimentais de expressão do Qwen.")
                self.vectors[emotion] = torch.load(file, map_location="cpu", weights_only=True)["tau"]
            base = prompt[0].ref_spk_embedding
            delta = self.vectors[emotion].to(base).reshape_as(base)
            prompt = [replace(prompt[0], ref_spk_embedding=base + EMOTIONS[emotion] * strength * delta)]
        blocks = 0
        def count(_module, _inputs, _outputs):
            nonlocal blocks
            blocks += 1
            if blocks % 24 == 0:
                on_blocks(blocks)
        hook = self.model.model.talker.register_forward_hook(count)
        try:
            audios, rate = self.model.generate_voice_clone(text=text, language="English", voice_clone_prompt=prompt)
            return audios[0], rate
        finally:
            hook.remove()


class IndexVoice:
    def __init__(self, root):
        import torch
        from indextts.infer_v2 import IndexTTS2
        directory = root / INDEX
        cache = directory / "hf_cache"
        aux = {"w2v_bert": str(cache/"w2v-bert-2.0"), "semantic_codec": str(cache/"semantic_codec_model.safetensors"),
               "campplus": str(cache/"campplus_cn_common.bin"), "bigvgan": str(cache/"bigvgan")}
        self.model = IndexTTS2(cfg_path=str(directory/"config.yaml"), model_dir=str(directory),
                               use_fp16=torch.cuda.is_available(), use_cuda_kernel=False, use_qwen_emo=False, aux_paths=aux)
        self.references = {}

    def wav(self, path, work, read_reference):
        import soundfile as sf
        if path not in self.references:
            samples, rate = read_reference(path, work)
            file = work / f"index-reference-{len(self.references)}.wav"
            sf.write(file, samples, rate)
            self.references[path] = str(file)
        return self.references[path]

    def speak(self, text, reference, transcript, settings, work, read_reference, on_blocks):
        import soundfile as sf
        base = self.wav(reference, work, read_reference)
        style = settings.get("reference")
        emotion = settings.get("emotion")
        strength = settings.get("strength", 1)
        options = {}
        if style:
            options = {"emo_audio_prompt": self.wav(style, work, read_reference), "emo_alpha": strength}
        elif emotion in EMOTIONS and strength > 0:
            vector = [0.] * 8
            vector[{"happy": 0, "angry": 1, "sad": 2}[emotion]] = 1.
            options = {"emo_vector": vector, "emo_alpha": strength}
        output = str(work / "index-output.wav")
        self.model.infer(spk_audio_prompt=base, text=text, output_path=output, verbose=False, **options)
        return sf.read(output, dtype="float32")
