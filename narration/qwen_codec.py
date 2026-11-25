"""Fixed-length residual code groups for the pinned Qwen3 12 Hz decoder."""
from importlib.metadata import version
from types import SimpleNamespace


def optimize_codec(talker):
    # The outer talker still uses the upstream sampler, EOS and repetition rules.
    # Only its internal 15-group residual prediction has a known length and no EOS.
    if version("qwen-tts") != "0.1.1" or version("transformers") != "4.57.3":
        return
    import torch
    from transformers import GenerationConfig, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
    predictor = talker.code_predictor
    if next(predictor.parameters()).device.type != "cuda" or predictor.model.has_sliding_layers:
        return
    groups = talker.config.num_code_groups - 1
    if groups != 15:
        return
    config = predictor.generation_config
    defaults = GenerationConfig()
    # Keep upstream behavior if this checkpoint adds stopping criteria, processors or a cache policy.
    allowed = {"_from_model_config", "transformers_version", "max_length", "bos_token_id", "pad_token_id"}
    if any(key not in allowed and getattr(defaults, key, None) != value
           for key, value in config.to_diff_dict().items()):
        return
    original = predictor.generate
    signature = {"inputs_embeds", "max_new_tokens", "do_sample", "top_p", "top_k", "temperature",
                 "output_hidden_states", "return_dict_in_generate"}

    def predict(embeds, kw):
        sample = kw["do_sample"]
        warpers = []
        if sample and kw["temperature"] != 1:
            warpers.append(TemperatureLogitsWarper(kw["temperature"]))
        if sample and kw["top_k"]:
            warpers.append(TopKLogitsWarper(kw["top_k"]))
        if sample and kw["top_p"] < 1:
            warpers.append(TopPLogitsWarper(kw["top_p"]))
        cache, token, sequence = None, None, []
        for step in range(groups):
            out = predictor(inputs_embeds=embeds if step == 0 else None, input_ids=token,
                            past_key_values=cache, use_cache=True, generation_steps=step,
                            output_hidden_states=False, return_dict=True)
            scores = out.logits[:, -1, :].float()
            for warp in warpers:
                scores = warp(None, scores)
            token = (torch.multinomial(torch.softmax(scores, dim=-1), 1) if sample
                     else scores.argmax(dim=-1, keepdim=True))
            sequence.append(token)
            cache = out.past_key_values
        return torch.cat(sequence, dim=1)

    # One graph for the session. Its input and output are tiny; weights are shared with the model.
    # Replay submits the 15 dependent predictions together, including the original CUDA sampler.
    graph = static = codes = graph_key = None
    attempted = False

    def generate(*args, **kw):
        nonlocal graph, static, codes, graph_key, attempted
        embeds = kw.get("inputs_embeds")
        if (args or kw.keys() != signature or kw["max_new_tokens"] != groups or embeds is None or
                not kw["return_dict_in_generate"] or embeds.shape[:2] != (1, 2)):
            return original(*args, **kw)
        key = (embeds.device, embeds.dtype, tuple(embeds.shape), kw["do_sample"],
               kw["top_p"], kw["top_k"], kw["temperature"])
        if graph is not None and key == graph_key:
            static.copy_(embeds)
            graph.replay()
            return SimpleNamespace(sequences=codes.clone())
        if attempted or not torch.is_inference_mode_enabled():
            return SimpleNamespace(sequences=predict(embeds, kw))
        attempted = True
        if torch.cuda.mem_get_info(embeds.device)[0] < 1024**3:
            return SimpleNamespace(sequences=predict(embeds, kw))
        with torch.cuda.device(embeds.device):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                static = embeds.clone()
                first = predict(static, kw)
            torch.cuda.current_stream().wait_stream(stream)
            # Capturing must not consume a second sample from the caller's random sequence.
            try:
                with torch.random.fork_rng(devices=[embeds.device]):
                    candidate = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(candidate, stream=stream):
                        codes = predict(static, kw)
                graph, graph_key = candidate, key
            except RuntimeError:
                # The eager decoder is valid on drivers that reject graph capture.
                static = codes = None
            return SimpleNamespace(sequences=first)

    predictor.generate = generate
