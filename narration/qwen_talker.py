"""CUDA graphs for the fixed-shape part of Qwen3-TTS 12 Hz single-token decoding."""
from functools import wraps
from importlib.metadata import version
import logging


def _rotary_coefficients(values, sections, interleaved):
    """Select Qwen's temporal/spatial coefficients once for all decoder layers."""
    import torch
    if interleaved:
        half = values[..., :values.shape[-1] // 2]
        merged = half[0].clone()
        stride = len(sections)
        for axis, width in enumerate(sections[1:], 1):
            indices = slice(axis, width * stride, stride)
            merged[..., indices] = half[axis, ..., indices]
        return torch.cat((merged, merged), dim=-1).unsqueeze(1)
    parts = values.split(sections * 2, dim=-1)
    return torch.cat([part[i % 3] for i, part in enumerate(parts)], dim=-1).unsqueeze(1)


class _DecoderGraphs:
    def __init__(self, model, embed, rotary):
        import torch
        from qwen_tts.core.models.modeling_qwen3_tts import rotate_half
        self.stream = torch.cuda.current_stream(embed.device)
        recording = torch.cuda.Stream(device=embed.device)
        pool = torch.cuda.graph_pool_handle()
        self.history = torch.empty((len(model.layers) + 1, *embed.shape), device=embed.device, dtype=embed.dtype)
        self.incoming = torch.empty_like(embed)
        self.rotary = tuple(torch.empty_like(v) for v in rotary)
        self.graphs = []
        self.attention = []
        recording.wait_stream(self.stream)
        try:
            with torch.cuda.stream(recording):
                for i in range(len(model.layers) + 1):
                    graph = torch.cuda.CUDAGraph()
                    # Kernels are warm from eager decoding. Capture in one private pool,
                    # without flushing the process allocator or synchronizing per layer.
                    graph.capture_begin(pool=pool)
                    try:
                        if i == 0:
                            rope = model.layers[0].self_attn.rope_scaling
                            cos, sin = (_rotary_coefficients(v, rope["mrope_section"], rope["interleaved"])
                                        for v in self.rotary)
                        else:
                            prev = model.layers[i - 1]
                            hidden = self.history[i - 1] + prev.self_attn.o_proj(self.incoming)
                            hidden = hidden + prev.mlp(prev.post_attention_layernorm(hidden))
                            self.history[i].copy_(model.norm(hidden) if i == len(model.layers) else hidden)
                        if i < len(model.layers):
                            layer = model.layers[i]
                            attention = layer.self_attn
                            shape = (1, 1, -1, attention.head_dim)
                            hidden = layer.input_layernorm(self.history[i])
                            q = attention.q_norm(attention.q_proj(hidden).view(shape)).transpose(1, 2)
                            k = attention.k_norm(attention.k_proj(hidden).view(shape)).transpose(1, 2)
                            v = attention.v_proj(hidden).view(shape).transpose(1, 2)
                            q = (q * cos) + (rotate_half(q) * sin)
                            k = (k * cos) + (rotate_half(k) * sin)
                            self.attention.append((attention, q, k, v))
                    finally:
                        graph.capture_end()
                    self.graphs.append(graph)
        finally:
            self.stream.wait_stream(recording)

    def decode(self, embed, rotary, cache, cache_position, mask, want_history):
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        from transformers.modeling_outputs import BaseModelOutputWithPast
        self.history[0].copy_(embed)
        for target, source in zip(self.rotary, rotary):
            target.copy_(source)
        cache_args = {"cos": rotary[0], "sin": rotary[1], "cache_position": cache_position}
        for graph, (attention, q, k, v) in zip(self.graphs, self.attention):
            graph.replay()
            k, v = cache.update(k, v, attention.layer_idx, cache_args)
            # Keep SDPA's actual sequence length, kernel selection and original mask.
            output, _ = sdpa_attention_forward(attention, q, k, v, mask, dropout=0.0,
                                               scaling=attention.scaling, sliding_window=attention.sliding_window)
            self.incoming.copy_(output.reshape_as(self.incoming))
        self.graphs[-1].replay()
        # Generation holds past_hidden across calls; never expose reusable graph scratch.
        history = self.history.clone().unbind() if want_history else None
        last = history[-1] if want_history else self.history[-1].clone()
        return BaseModelOutputWithPast(last_hidden_state=last, past_key_values=cache, hidden_states=history)


def optimize_talker(talker):
    if version("qwen-tts") != "0.1.1" or version("transformers") != "4.57.3":
        return
    import torch
    from transformers.cache_utils import DynamicCache
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerModel, create_causal_mask
    model = talker.model
    cfg = model.config
    weight = next(model.parameters())
    if (type(model) is not Qwen3TTSTalkerModel or weight.device.type != "cuda"
            or weight.dtype not in (torch.bfloat16, torch.float16) or model.training
            or cfg._attn_implementation != "sdpa" or cfg.sliding_window is not None
            or not cfg.use_return_dict or cfg.output_attentions
            or any(p.device != weight.device for p in model.parameters())
            or any(layer.self_attn.rope_scaling != model.layers[0].self_attn.rope_scaling
                   for layer in model.layers)):
        return
    original = model.forward
    fields = {"input_ids", "inputs_embeds", "position_ids", "past_key_values", "cache_position",
              "attention_mask", "use_cache", "output_attentions", "output_hidden_states"}
    plan = None
    warmed = disabled = False

    @wraps(original)
    def forward(*args, **kw):
        nonlocal plan, warmed, disabled
        embed = kw.get("inputs_embeds")
        pos = kw.get("position_ids")
        cache = kw.get("past_key_values")
        step = kw.get("cache_position")
        if (disabled or args or not kw.keys() <= fields or kw.get("input_ids") is not None
                or not kw.get("use_cache") or kw.get("output_attentions") or model.training
                or not torch.is_inference_mode_enabled() or embed is None
                or embed.shape != (1, 1, cfg.hidden_size) or embed.device != weight.device
                or embed.dtype != weight.dtype or pos is None or pos.shape != (3, 1, 1)
                or pos.device != embed.device or step is None or step.shape != (1,)
                or step.device != embed.device or type(cache) is not DynamicCache or cache.offloading
                or len(cache.layers) != len(model.layers) or not cache.get_seq_length()
                or (plan is not None and torch.cuda.current_stream(embed.device) != plan.stream)):
            return original(*args, **kw)
        if not warmed:
            warmed = True
            return original(*args, **kw)
        if plan is None:
            with torch.cuda.device(embed.device):
                if torch.cuda.is_current_stream_capturing():
                    return original(*args, **kw)
                if torch.cuda.mem_get_info()[0] < 1024**3:
                    disabled = True
                    return original(*args, **kw)
                rotary = model.rotary_emb(embed, pos)
                try:
                    plan = _DecoderGraphs(model, embed, rotary)
                except (RuntimeError, NotImplementedError) as error:
                    disabled = True
                    logging.getLogger(__name__).warning("Qwen TTS using native decoding: %s", error)
            if disabled:
                return original(*args, **kw)
        else:
            rotary = model.rotary_emb(embed, pos)
        mask = create_causal_mask(config=cfg, input_embeds=embed, attention_mask=kw.get("attention_mask"),
                                  cache_position=step, past_key_values=cache, position_ids=pos[0])
        want_history = kw.get("output_hidden_states")
        return plan.decode(embed, rotary, cache, step, mask,
                           cfg.output_hidden_states if want_history is None else want_history)

    model.forward = forward
