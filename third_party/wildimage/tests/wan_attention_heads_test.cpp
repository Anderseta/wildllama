#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include "ggml-cpu.h"
#include "model/diffusion/wan.hpp"

// Use distinct head/position values, an incomplete final group and both
// attention paths. No downloaded weights are needed.
int main() {
    auto backend = sd_backend_cpu_init();
    if (!backend) return 1;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend_guard(backend, ggml_backend_free);
    for (bool flash : {false, true}) {
        auto c = ggml_init({16 << 20, nullptr, true});
        if (!c) return 2;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> arena(c, ggml_free);
        GGMLRunnerContext ctx{};
        ctx.ggml_ctx = c;
        ctx.backend = backend;
        ctx.flash_attn_enabled = flash;
        auto input = [&]() { return ggml_new_tensor_4d(c, GGML_TYPE_F32, 64, 5, 19, 1); };
        auto q = input(), k = input(), v = input();
        auto pe = ggml_new_tensor_4d(c, GGML_TYPE_F32, 2, 2, 32, 19);
        auto reference = Rope::attention(&ctx, q, k, v, pe, nullptr);
        auto result = WAN::attention_by_head(&ctx, q, k, v, pe, 2);
        if (!ggml_are_same_shape(result, reference)) return 3;
        auto graph = ggml_new_graph(c);
        ggml_build_forward_expand(graph, reference);
        ggml_build_forward_expand(graph, result);
        auto buffer = ggml_backend_alloc_ctx_tensors(c, backend);
        if (!buffer) return 4;
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> storage(buffer, ggml_backend_buffer_free);
        int seed = 3;
        for (auto tensor : {q, k, v}) {
            std::vector<float> data(ggml_nelements(tensor));
            for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
                data[i] = static_cast<float>((i * 37 + i / 11 + seed) % 113 - 56) / 64.f;
            }
            ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
            seed += 19;
        }
        std::vector<float> rotation(ggml_nelements(pe));
        for (int i = 0; i < 32 * 19; ++i) {
            float angle = (i % 73) / 40.f;
            rotation[4 * i] = rotation[4 * i + 3] = std::cos(angle);
            rotation[4 * i + 1] = -std::sin(angle);
            rotation[4 * i + 2] = std::sin(angle);
        }
        ggml_backend_tensor_set(pe, rotation.data(), 0, ggml_nbytes(pe));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 4;
        std::vector<float> expected(ggml_nelements(reference)), actual(expected.size());
        ggml_backend_tensor_get(reference, expected.data(), 0, ggml_nbytes(reference));
        ggml_backend_tensor_get(result, actual.data(), 0, ggml_nbytes(result));
        double maximum = 0;
        for (int64_t i = 0; i < ggml_nelements(reference); ++i) {
            const auto a = expected[i], b = actual[i];
            if (!std::isfinite(a) || !std::isfinite(b)) return 5;
            maximum = std::max(maximum, std::abs(double(a) - b));
        }
        std::printf("flash=%d heads=5 group=2 max_abs=%g\n", flash, maximum);
        if (maximum > 1e-5) return 6;
    }
    return 0;
}
