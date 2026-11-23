#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "core/ggml_extensions.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"

namespace sd_cuda {
    void register_rope(ggml_backend_dev_t device);
}

// Compare the actual CUDA outputs, including the incomplete last launch block,
// a head with partial rotation, and a strided view into a longer sequence.
static bool check(ggml_backend_t backend, int tokens, int heads, int rotary_pairs, int interleaved, int batch) {
    auto ctx = ggml_init({2 << 20, nullptr, true});
    if (!ctx) return false;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> arena(ctx, ggml_free);
    auto source = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, heads, tokens + 3, batch);
    auto keys = ggml_view_4d(ctx, source, 128, heads, tokens, batch, source->nb[1], source->nb[2], source->nb[3], source->nb[2]);
    auto positions = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, rotary_pairs, tokens);
    ggml_tensor* inputs[] = {keys, positions};
    const int32_t layout = interleaved;
    auto rotated = ggml_ext_op_create(ctx, sd_ggml_rope_op(), inputs, 2, &layout);
    auto fused = ggml_ext_op_create(ctx, sd_ggml_rope_f16_op(), inputs, 2, &layout);
    if (!rotated || !fused || !ggml_backend_supports_op(backend, fused)) return false;
    auto reference = ggml_cast(ctx, rotated, GGML_TYPE_F16);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, reference);
    ggml_build_forward_expand(graph, fused);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return false;
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> storage(buffer, ggml_backend_buffer_free);
    uint32_t seed = 53;
    for (auto tensor : {source, positions}) {
        std::vector<float> values(ggml_nelements(tensor));
        for (auto& value : values) {
            seed = seed * 1664525u + 1013904223u;
            value = (int(seed >> 16) - 32768) / 32768.f;
        }
        ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
    }
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
    std::vector<uint16_t> a(ggml_nelements(reference)), b(ggml_nelements(fused));
    ggml_backend_tensor_get(reference, a.data(), 0, ggml_nbytes(reference));
    ggml_backend_tensor_get(fused, b.data(), 0, ggml_nbytes(fused));
    return a == b;
}

int main() {
    auto backend = ggml_backend_cuda_init(0);
    if (!backend) return 1;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> guard(backend, ggml_backend_free);
    sd_cuda::register_rope(ggml_backend_get_device(backend));
    if (!check(backend, 37, 5, 64, 1, 1) || !check(backend, 8193, 2, 64, 1, 1) ||
        !check(backend, 19, 3, 32, 1, 1) || !check(backend, 13, 2, 64, 0, 2)) return 2;
    std::puts("Wan CUDA rotary keys match the original FP16 values.");
    return 0;
}
