"""Numerical regression for the pinned CUDA decoder, with a small random model."""
from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def cuda_runtime():
    try:
        if version("qwen-tts") != "0.1.1" or version("transformers") != "4.57.3":
            return False
        import torch
        return torch.cuda.is_available()
    except (ImportError, PackageNotFoundError):
        return False


@unittest.skipUnless(cuda_runtime(), "requires the Qwen narration CUDA runtime")
class TalkerGraphTests(unittest.TestCase):
    def setUp(self):
        import torch
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerModel
        cfg = Qwen3TTSTalkerConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                                   vocab_size=64, text_vocab_size=64, text_hidden_size=128, head_dim=32,
                                   num_attention_heads=4, num_key_value_heads=2,
                                   rope_scaling={"rope_type": "default", "interleaved": True,
                                                 "mrope_section": [8, 4, 4]})
        cfg._attn_implementation = "sdpa"
        torch.manual_seed(123)
        self.model = Qwen3TTSTalkerModel(cfg).cuda().half().eval()
        self.native = self.model.forward
        self.inputs = [torch.randn(1, length, 128, device="cuda", dtype=torch.float16)
                       for length in (5, 1, 1, 1, 1)]

    def decode(self, forward, histories):
        import torch
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        states = []
        with torch.inference_mode():
            for embed in self.inputs:
                start = cache.get_seq_length()
                pos = torch.arange(start, start + embed.shape[1], device="cuda")
                # Different rotary axes also exercise interleaving.
                pos3 = torch.stack([pos, pos + 1, pos + 2]).unsqueeze(1)
                mask = torch.ones(1, start + embed.shape[1], dtype=torch.long, device="cuda")
                mask[:, 0] = 0
                result = forward(inputs_embeds=embed, position_ids=pos3, attention_mask=mask,
                                 past_key_values=cache, cache_position=pos, use_cache=True,
                                 output_hidden_states=histories)
                self.assertEqual(cache.get_seq_length(), mask.shape[1])
                if histories:
                    states.extend(result.hidden_states)
                else:
                    self.assertIsNone(result.hidden_states)
                    states.append(result.last_hidden_state)
        # Compare after all replays to catch aliasing with later tokens.
        return states + [tensor for layer in cache.layers for tensor in (layer.keys, layer.values)]

    def assert_exact(self, first, second):
        import torch
        self.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_replay_across_requests_with_and_without_history(self):
        import qwen_talker
        qwen_talker.optimize_talker(SimpleNamespace(model=self.model))
        with patch.object(qwen_talker._DecoderGraphs, "decode", autospec=True,
                          side_effect=qwen_talker._DecoderGraphs.decode) as replay:
            for histories in (True, False):
                self.assert_exact(self.decode(self.native, histories), self.decode(self.model, histories))
            self.assertGreater(replay.call_count, 1)

    def test_capture_failure_falls_back_without_appending_a_token_twice(self):
        import qwen_talker
        qwen_talker.optimize_talker(SimpleNamespace(model=self.model))
        with patch.object(qwen_talker, "_DecoderGraphs", side_effect=RuntimeError("test capture failure")) as capture:
            self.assert_exact(self.decode(self.native, True), self.decode(self.model, True))
            capture.assert_called_once()

    def test_low_memory_stays_native(self):
        import qwen_talker
        qwen_talker.optimize_talker(SimpleNamespace(model=self.model))
        with patch("torch.cuda.mem_get_info", return_value=(1024, 4096)), patch.object(qwen_talker, "_DecoderGraphs") as capture:
            self.assert_exact(self.decode(self.native, True), self.decode(self.model, True))
            capture.assert_not_called()
