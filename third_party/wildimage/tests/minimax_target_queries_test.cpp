// Link against the patched native image runner and its GGML backend.
#include "model/diffusion/minimax_h3.hpp"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include <chrono>
#include <cstdio>
#include <random>

using Clock = std::chrono::steady_clock;

struct Measurement {
    std::vector<float> output;
    size_t workspace;
    double milliseconds;
};

static Measurement run(ggml_backend_t backend, int references, bool grouped, bool flash,
                       bool target_queries, bool split, int target_tokens, bool change_reference = false) {
    MiniMaxH3::Config config;
    config.hidden_size = 256;
    config.num_attention_heads = 4;
    config.attention_head_dim = 128;
    config.ffn_hidden_size = 512;
    config.time_embed_dim = 96;
    MiniMaxH3::LowRankShape ranks;
    if (split) {
        ranks.q = ranks.k = ranks.v = 64;
    }
    MiniMaxH3::TransformerBlock block(config, ranks);
    auto data = ggml_init({4 << 20, nullptr, true});
    String2TensorStorage types;
    if (!split) {
        types["attn.qkv_proj.weight"].type = GGML_TYPE_Q5_K;
        types["attn.out_proj.weight"].type = GGML_TYPE_Q5_K;
    }
    block.init(data, types);
    std::vector<GGMLBlock*> children;
    block.get_all_blocks(children);
    for (auto* child : children) {
        if (auto* attention = dynamic_cast<MiniMaxH3::Attention*>(child)) {
            attention->head_chunk_budget_gib = grouped ? 0.0001f : 0.f;
        }
    }
    std::vector<MiniMaxH3::TokenModulationSpan> spans{{0, 33, 0}};
    for (int i = 0; i < references; ++i) {
        auto start = spans.back().end;
        spans.push_back({start, start + 257 + i, 1});
    }
    auto start = spans.back().end;
    spans.push_back({start, start + target_tokens, 2});
    const int64_t length = spans.back().end;
    auto x = ggml_new_tensor_2d(data, GGML_TYPE_F32, config.hidden_size, length);
    auto t = ggml_new_tensor_1d(data, GGML_TYPE_F32, config.time_embed_dim);
    auto pe = ggml_new_tensor_4d(data, GGML_TYPE_F32, 2, 2, 48, length);
    auto weights = ggml_backend_alloc_ctx_tensors(data, backend);
    GGML_ASSERT(weights);
    std::map<std::string, ggml_tensor*> params;
    block.get_param_tensors(params);
    std::mt19937 rng(9105);
    auto fill = [&](ggml_tensor* tensor, float bias, float scale) {
        std::vector<float> values(ggml_nelements(tensor));
        for (float& value : values) {
            value = bias + scale * std::uniform_real_distribution<float>(-1.f, 1.f)(rng);
        }
        if (tensor == x && change_reference) {
            GGML_ASSERT(references > 0);
            auto first = values.begin() + spans[1].start * config.hidden_size;
            auto last = values.begin() + spans[1].end * config.hidden_size;
            std::transform(first, last, first, [](float value) { return -value; });
        }
        if (tensor->type == GGML_TYPE_Q5_K) {
            std::vector<uint8_t> packed(ggml_nbytes(tensor));
            size_t written = ggml_quantize_chunk(tensor->type, values.data(), packed.data(), 0,
                                                 tensor->ne[1], tensor->ne[0], nullptr);
            GGML_ASSERT(written == packed.size());
            ggml_backend_tensor_set(tensor, packed.data(), 0, written);
        } else {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
        }
    };
    for (auto& [name, tensor] : params) {
        bool norm = name.find("norm") != std::string::npos;
        fill(tensor, norm ? 1.f : 0.f, norm ? 0.2f : 0.06f);
    }
    fill(x, 0.f, 1.f);
    fill(t, 0.f, 1.f);
    std::vector<float> positions(ggml_nelements(pe));
    for (int64_t pos = 0; pos < length; ++pos) {
        for (int channel = 0; channel < 48; ++channel) {
            const float phase = float(pos) / (channel + 1);
            float* r = &positions[4 * (48 * pos + channel)];
            r[0] = r[3] = std::cos(phase);
            r[2] = std::sin(phase);
            r[1] = -r[2];
        }
    }
    ggml_backend_tensor_set(pe, positions.data(), 0, ggml_nbytes(pe));
    auto ctx = ggml_init({8 << 20, nullptr, true});
    GGMLRunnerContext runner;
    runner.ggml_ctx = ctx;
    runner.backend = backend;
    runner.flash_attn_enabled = flash;
    runner.graph = ggml_new_graph_custom(ctx, 4096, false);
    auto output = block.forward(&runner, x, t, spans, pe, spans.size() - 1, target_queries);
    GGML_ASSERT(output->ne[0] == config.hidden_size && output->ne[1] == target_tokens);
    ggml_build_forward_expand(runner.graph, output);
    auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(alloc, runner.graph));
    std::vector<double> times;
    for (int i = 0; i < 12; ++i) {
        auto begin = Clock::now();
        GGML_ASSERT(ggml_backend_graph_compute(backend, runner.graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);
        if (i >= 3) {
            times.push_back(std::chrono::duration<double, std::milli>(Clock::now() - begin).count());
        }
    }
    std::sort(times.begin(), times.end());
    Measurement result{{}, ggml_gallocr_get_buffer_size(alloc, 0), times[times.size() / 2]};
    result.output.resize(ggml_nelements(output));
    ggml_backend_tensor_get(output, result.output.data(), 0, ggml_nbytes(output));
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(weights);
    ggml_free(data);
    return result;
}

int main(int argc, char** argv) {
    std::string mode = argc == 2 ? argv[1] : "";
    if (argc > 2 || (!mode.empty() && mode != "--require-exact" && mode != "--baseline" && mode != "--target")) {
        return 2;
    }
    auto backend = ggml_backend_cuda_init(0);
    GGML_ASSERT(backend);
    if (mode == "--baseline" || mode == "--target") {
        // Separate processes let an external device-memory sampler compare peaks.
        for (int i = 0; i < 12; ++i) {
            run(backend, 2, true, true, mode == "--target", false, 256);
        }
        ggml_backend_free(backend);
        return 0;
    }
    bool valid = true;
    for (int refs : {0, 1, 2}) {
        for (bool flash : {false, true}) {
            for (bool grouped : {false, true}) {
                const bool split = refs == 1;
                const int target_tokens = refs == 1 ? 259 : 256;
                for (bool reverse : {false, true}) {
                    auto a = run(backend, refs, grouped, flash, reverse, split, target_tokens);
                    auto b = run(backend, refs, grouped, flash, !reverse, split, target_tokens);
                    if (reverse) {
                        std::swap(a, b);
                    }
                    const bool same = std::memcmp(a.output.data(), b.output.data(), a.output.size() * sizeof(float)) == 0;
                    float max_error = 0.f;
                    double squared_error = 0.0;
                    for (size_t i = 0; i < a.output.size(); ++i) {
                        GGML_ASSERT(std::isfinite(a.output[i]) && std::isfinite(b.output[i]));
                        const float difference = std::abs(a.output[i] - b.output[i]);
                        max_error = std::max(max_error, difference);
                        squared_error += double(difference) * difference;
                        // These bounded fixtures use the production FP16 attention path.
                        valid &= difference <= std::ldexp(1.f, -11) * std::max(1.f, std::abs(a.output[i]));
                    }
                    // Changing the number of query rows can change CUDA reduction
                    // order. Report bitwise parity separately from numerical accuracy.
                    if (mode == "--require-exact") {
                        valid &= same;
                    }
                    valid &= b.workspace <= a.workspace;
                    if (refs == 2 && flash && grouped && !reverse) {
                        auto changed = run(backend, refs, grouped, flash, true, split, target_tokens, true);
                        float reference_effect = 0.f;
                        for (size_t i = 0; i < changed.output.size(); ++i) {
                            reference_effect = std::max(reference_effect, std::abs(b.output[i] - changed.output[i]));
                        }
                        GGML_ASSERT(reference_effect > 0.0001f);
                    }
                    const double rmse = std::sqrt(squared_error / a.output.size());
                    std::printf("refs=%d flash=%d grouped=%d reverse=%d exact=%d max_abs=%g rmse=%g ms=%.5f,%.5f workspace=%zu,%zu\n",
                                refs, flash, grouped, reverse, same, max_error, rmse, a.milliseconds, b.milliseconds, a.workspace, b.workspace);
                }
            }
        }
    }
    ggml_backend_free(backend);
    return valid ? 0 : 1;
}
