// Link to the same native objects as wan_backend_benchmark. No model or video is needed.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <vector>
#include "core/probe/backend_probe.h"
#include "model/diffusion/wan.hpp"

static bool compare_outputs(BackendProbeGraph& baseline, BackendProbeGraph& candidate,
                            const char* label, int queries, double max_error) {
    if (!ggml_are_same_shape(baseline.output, candidate.output)) return false;
    std::vector<float> a(ggml_nelements(baseline.output)), b(a.size());
    ggml_backend_tensor_get(baseline.output, a.data(), 0, a.size() * sizeof(float));
    ggml_backend_tensor_get(candidate.output, b.data(), 0, b.size() * sizeof(float));
    double diff = 0, signal = 0, peak = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return false;
        const double err = a[i] - b[i];
        diff += err * err;
        signal += double(a[i]) * a[i];
        peak = std::max(peak, std::abs(err));
    }
    const double relative = std::sqrt(diff / signal);
    std::printf("%s queries=%d relative_rms=%.6f max_abs=%.6f\n", label, queries, relative, peak);
    return relative < .05 && peak < max_error;
}

static bool compare_attention(ggml_backend_t gpu, int tokens, int heads, int group) {
    BackendProbeGraph baseline(gpu, 512), candidate(gpu, 512);
    for (auto* graph : {&baseline, &candidate}) {
        GGMLRunnerContext ctx{};
        ctx.backend = gpu;
        ctx.ggml_ctx = graph->ctx;
        ctx.flash_attn_enabled = true;
        auto q = ggml_new_tensor_4d(graph->ctx, GGML_TYPE_F32, 128, heads, tokens, 1);
        auto k = ggml_dup_tensor(graph->ctx, q);
        auto v = ggml_dup_tensor(graph->ctx, q);
        auto pe = ggml_new_tensor_4d(graph->ctx, GGML_TYPE_F32, 2, 2, 64, tokens);
        graph->inputs = {q, k, v};
        graph->output = graph == &baseline ? Rope::attention(&ctx, q, k, v, pe, nullptr)
                                           : WAN::attention_by_head(&ctx, q, k, v, pe, group, true);
        if (!graph->supported() || !graph->allocate() || !graph->initialize_f32_inputs()) return false;
        std::vector<float> rotation(ggml_nelements(pe));
        for (size_t i = 0; i < rotation.size(); i += 4) {
            const float angle = float(i / 4 % 509) * .013f;
            rotation[i] = rotation[i + 3] = std::cos(angle);
            rotation[i + 1] = -std::sin(angle);
            rotation[i + 2] = std::sin(angle);
        }
        ggml_backend_tensor_set(pe, rotation.data(), 0, ggml_nbytes(pe));
        if (graph->run() < 0) return false;
    }
    return compare_outputs(baseline, candidate, "self", tokens, .02);
}

// One salient token competes with many near-uniform logits. A distant channel
// has a large value at that same token. Distinct peaks/signs in the two I2V
// branches catch concatenated softmaxes, skipped references and layout errors.
static bool compare_cross(ggml_backend_t gpu, int d, int queries, int text_len, int image_len) {
    constexpr int heads = 3;
    BackendProbeGraph baseline(gpu, 512), candidate(gpu, 512);
    for (auto* graph : {&baseline, &candidate}) {
        GGMLRunnerContext ctx{};
        ctx.backend = gpu;
        ctx.ggml_ctx = graph->ctx;
        ctx.flash_attn_enabled = true;
        auto input = [&](int tokens) {
            auto t = ggml_new_tensor_3d(graph->ctx, GGML_TYPE_F32, d * heads, tokens, 1);
            graph->inputs.push_back(t);
            return t;
        };
        auto q = input(queries), k = input(text_len), v = input(text_len);
        auto ik = image_len ? input(image_len) : nullptr;
        auto iv = image_len ? input(image_len) : nullptr;
        if (graph == &baseline) {
            auto attend = [&](ggml_tensor* keys, ggml_tensor* values) {
                return ggml_ext_attention_ext(&ctx, q, keys, values, heads, nullptr, false, true);
            };
            graph->output = attend(k, v);
            if (image_len) graph->output = ggml_add(graph->ctx, graph->output, attend(ik, iv));
        } else {
            graph->output = WAN::wan_sage_cross(&ctx, heads, q, k, v, ik, iv);
        }
        if (!graph->supported() || !graph->allocate() || !graph->initialize_f32_inputs()) return false;
        for (auto* t : graph->inputs) {
            std::vector<float> data(ggml_nelements(t));
            ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
            for (int pos = 0; pos < t->ne[1]; ++pos) {
                for (int h = 0; h < heads; ++h) {
                    auto row = data.data() + (pos * heads + h) * d;
                    const int peak = int(t->ne[1]) - 1 - h;
                    if (t == q) row[h] = 8.f;
                    else if (t == k || t == ik) row[h] = 5.f + (pos == peak ? 20.f : 0.f);
                    else if (pos == peak) row[d - 1 - h] = t == v ? 6.f : -2.f;
                }
            }
            ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
        }
        if (graph->run() < 0) return false;
    }
    return compare_outputs(baseline, candidate, image_len ? "reference+text" : "text", queries, .08);
}

// Check support without allocating a full long-video attention graph. Wan groups
// the 40 heads into four-head operations at 1024x768x109 (86,016 query positions).
static bool check_long_attention(ggml_backend_t gpu, int queries, int keys, int heads, bool expected) {
    BackendProbeGraph graph(gpu, 32);
    GGMLRunnerContext ctx{};
    ctx.backend = gpu;
    ctx.ggml_ctx = graph.ctx;
    auto q = ggml_new_tensor_3d(graph.ctx, GGML_TYPE_F32, 128, queries, heads);
    auto k = ggml_new_tensor_3d(graph.ctx, GGML_TYPE_F32, 128, keys, heads);
    auto v = ggml_new_tensor_3d(graph.ctx, GGML_TYPE_F16, 128, keys, heads);
    auto out = ggml_ext_sage_attn(&ctx, q, k, v, 1.f / std::sqrt(128.f), GGML_SAGE_ATTN_2);
    const bool supported = out && ggml_backend_supports_op(gpu, out);
    std::printf("SageAttention query=%d keys=%d heads=%d support=%d\n", queries, keys, heads, int(supported));
    return supported == expected;
}

// Compare direct output with the original FP16 round-trip and FP32 addition.
// For I2V, key counts can differ and the second softmax must recompute its K/V scales.
static bool compare_sage_output(ggml_backend_t gpu, int d, int queries, int text_len, int image_len,
                              int query_heads, int kv_heads, int rounds = 0) {
    BackendProbeGraph reference(gpu, 128), candidate(gpu, 128);
    for (auto* graph : {&reference, &candidate}) {
        GGMLRunnerContext ctx{};
        ctx.backend = gpu;
        ctx.ggml_ctx = graph->ctx;
        auto input = [&](int tokens, int heads) {
            auto* tensor = ggml_new_tensor_3d(graph->ctx, GGML_TYPE_F32, d, tokens, heads);
            graph->inputs.push_back(tensor);
            return tensor;
        };
        auto* q = input(queries, query_heads);
        auto* k = input(text_len, kv_heads);
        auto* v = ggml_cast(graph->ctx, input(text_len, kv_heads), GGML_TYPE_F16);
        auto* ik = image_len ? input(image_len, kv_heads) : nullptr;
        auto* iv = image_len ? ggml_cast(graph->ctx, input(image_len, kv_heads), GGML_TYPE_F16) : nullptr;
        const float scale = 1.f / std::sqrt(float(d));
        if (graph == &reference) {
            graph->output = ggml_ext_sage_attn(&ctx, q, k, v, scale, GGML_SAGE_ATTN_2);
            if (image_len) {
                auto* image = ggml_ext_sage_attn(&ctx, q, ik, iv, scale, GGML_SAGE_ATTN_2);
                if (!graph->output || !image) return false;
                graph->output = ggml_add(graph->ctx, graph->output, image);
            }
        } else {
            graph->output = image_len ? WAN::wan_sage_attention_pair(&ctx, q, k, v, ik, iv, scale)
                                      : ggml_ext_sage_attn(&ctx, q, k, v, scale, GGML_SAGE_ATTN_2);
            if (graph->output) ggml_set_name(graph->output, "wan.sage_fp8");
        }
        if (!graph->output) return false;
        if (!graph->supported() || !graph->allocate() || !graph->initialize_f32_inputs() ||
            graph->run() < 0. || graph->run() < 0.) return false;
    }
    const size_t elements = static_cast<size_t>(ggml_nelements(reference.output));
    std::vector<float> expected(elements), actual(elements);
    ggml_backend_tensor_get(reference.output, expected.data(), 0, ggml_nbytes(reference.output));
    ggml_backend_tensor_get(candidate.output, actual.data(), 0, ggml_nbytes(candidate.output));
    const bool exact = std::all_of(actual.begin(), actual.end(), [](float v) { return std::isfinite(v); }) &&
                       std::memcmp(expected.data(), actual.data(), elements * sizeof(float)) == 0;
    const char* kind = image_len ? "pair" : "self";
    std::printf("Sage %s D=%d Q=%d keys=%d image=%d heads=%d/%d exact=%d\n",
                kind, d, queries, text_len, image_len, query_heads, kv_heads, int(exact));
    if (!exact || !rounds) return exact;

    std::vector<double> old_times(rounds), new_times(rounds);
    for (int i = 0; i < rounds; ++i) {
        if (i % 2 == 0) {
            old_times[i] = reference.run();
            new_times[i] = candidate.run();
        } else {
            new_times[i] = candidate.run();
            old_times[i] = reference.run();
        }
        if (old_times[i] < 0. || new_times[i] < 0.) return false;
    }
    std::sort(old_times.begin(), old_times.end());
    std::sort(new_times.begin(), new_times.end());
    std::printf("Sage %s Q=%d timed rounds=%d original=%.3f ms optimized=%.3f ms\n",
                kind, queries, rounds, old_times[rounds / 2], new_times[rounds / 2]);
    return true;
}

int main(int argc, char** argv) {
    const bool benchmark = argc == 3 && std::strcmp(argv[2], "--benchmark-pair") == 0;
    if (argc != 2 && !benchmark) return 2;
    ggml_backend_load_all_from_path(argv[1]);
    auto gpu = ggml_backend_init_by_name("CUDA0", nullptr);
    if (!gpu) return 3;
    sd_ggml_register_extensions(gpu);
    bool ok = compare_attention(gpu, 256, 4, 4);
    ok &= compare_attention(gpu, 137, 5, 2);
    ok &= compare_attention(gpu, 4097, 1, 1);
    ok &= compare_cross(gpu, 128, 129, 77, 0);
    ok &= compare_cross(gpu, 128, 257, 77, 37);
    ok &= compare_cross(gpu, 64, 127, 16385, 61);
    ok &= check_long_attention(gpu, 65536, 65536, 4, true);
    ok &= check_long_attention(gpu, 86016, 86016, 4, true);
    ok &= check_long_attention(gpu, 86016, 257, 4, true);
    ok &= check_long_attention(gpu, 1, 1, 65536, false);
    ok &= compare_sage_output(gpu, 64, 137, 137, 0, 3, 3);
    ok &= compare_sage_output(gpu, 128, 257, 257, 0, 4, 4);
    ok &= compare_sage_output(gpu, 64, 133, 129, 33, 3, 3);
    ok &= compare_sage_output(gpu, 128, 257, 77, 259, 4, 2);
    if (benchmark) {
        ok &= compare_sage_output(gpu, 128, 4096, 4096, 0, 4, 4, 9);
        ok &= compare_sage_output(gpu, 128, 8192, 129, 257, 4, 4, 9);
        ok &= compare_sage_output(gpu, 128, 32768, 129, 257, 4, 4, 9);
    }
    ggml_backend_free(gpu);
    return ok ? 0 : 1;
}
