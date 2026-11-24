// Link to the same native objects as wan_backend_benchmark. No model or video is needed.
#include <cmath>
#include <cstdio>
#include <vector>
#include "core/probe/backend_probe.h"
#include "model/diffusion/wan.hpp"

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
    double relative = std::sqrt(diff / signal);
    std::printf("tokens=%d heads=%d group=%d relative_rms=%.6f max_abs=%.6f\n", tokens, heads, group, relative, peak);
    return relative < .05 && peak < .02;
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    ggml_backend_load_all_from_path(argv[1]);
    auto gpu = ggml_backend_init_by_name("CUDA0", nullptr);
    if (!gpu) return 3;
    sd_ggml_register_extensions(gpu);
    const bool ok = compare_attention(gpu, 256, 4, 4) && compare_attention(gpu, 137, 5, 2);
    ggml_backend_free(gpu);
    return ok ? 0 : 1;
}
