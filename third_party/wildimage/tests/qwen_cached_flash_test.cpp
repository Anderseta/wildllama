#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include "model/diffusion/qwen_image_2_1.hpp"

void wildimage_register_fp8(ggml_backend_dev_t);

static void check(bool ok) {
    if (!ok) throw std::runtime_error("Qwen cached flash regression");
}

static void benchmark(ggml_backend_t backend, ggml_cgraph* const* graphs, const char* label, int tokens, int heads) {
    std::vector<double> times[2];
    for (int i = 0; i < 9; ++i) {
        for (int k : {i % 2, 1 - i % 2}) {
            auto start = std::chrono::steady_clock::now();
            for (int rep = 0; rep < 12; ++rep) check(ggml_backend_graph_compute(backend, graphs[k]) == GGML_STATUS_SUCCESS);
            ggml_backend_synchronize(backend);
            times[k].push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 12);
        }
    }
    for (auto& t : times) std::sort(t.begin(), t.end());
    std::printf("%s, %d tokens / %d heads, median ms: %.3f -> %.3f\n", label, tokens, heads, times[0][4], times[1][4]);
}

static void compare(ggml_backend_t backend, int prefix, int tokens, int heads) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> arena(ggml_init({2 << 20, nullptr, true}), ggml_free);
    auto a = arena.get();
    check(a != nullptr);
    GGMLRunnerContext ctx{};
    ctx.ggml_ctx = a;
    ctx.backend = backend;
    ctx.flash_attn_enabled = true;
    constexpr int width = 128;
    auto q = ggml_new_tensor_3d(a, GGML_TYPE_F32, width, tokens, heads);
    auto k = ggml_dup_tensor(a, q);
    auto v = ggml_new_tensor_3d(a, GGML_TYPE_F32, width, heads, tokens);
    if (prefix == 13) {
        auto backing = ggml_new_tensor_1d(a, GGML_TYPE_F32, ggml_nelements(v) + 1);
        v = ggml_view_3d(a, backing, width, heads, tokens, v->nb[1], v->nb[2], sizeof(float));
    }
    auto pk = ggml_new_tensor_2d(a, GGML_TYPE_F16, width * heads, prefix);
    auto pv = ggml_dup_tensor(a, pk);
    auto baseline_k = ggml_reshape_3d(a, ggml_cast(a, pk, GGML_TYPE_F32), width, prefix, heads);
    auto baseline_v = ggml_reshape_3d(a, ggml_cast(a, pv, GGML_TYPE_F32), width, heads, prefix);
    bool used_flash = false;
    auto reference = ggml_ext_attention_ext(&ctx, q, ggml_concat(a, baseline_k, k, 1), ggml_concat(a, baseline_v, v, 2),
                                            heads, nullptr, true, true, 1.f, &used_flash);
    auto candidate = Qwen::QwenImage21Attention::cached_flash(&ctx, q, k, v, pk, pv);
    check(used_flash && candidate && ggml_are_same_shape(reference, candidate));
    ctx.attn_scale = 2.f;
    check(!Qwen::QwenImage21Attention::cached_flash(&ctx, q, k, v, pk, pv));
    ctx.attn_scale = 1.f;
    ctx.sage_attn_enabled = true;
    check(!Qwen::QwenImage21Attention::cached_flash(&ctx, q, k, v, pk, pv));
    ctx.sage_attn_enabled = false;
    ctx.flash_attn_enabled = false;
    check(!Qwen::QwenImage21Attention::cached_flash(&ctx, q, k, v, pk, pv));
    ctx.flash_attn_enabled = true;
    check(!Qwen::QwenImage21Attention::cached_flash(&ctx, q, k, v, baseline_k, baseline_v));
    ctx.attn_scale = 0.f;
    ctx.sage_attn_enabled = true;
    check(!Qwen::QwenImage21Attention::cached_flash(&ctx, q, k, v, pk, pv));
    ctx.sage_attn_enabled = false;
    check(!Qwen::QwenImage21Attention::cached_flash(&ctx, q, k, v, baseline_k, pv));
    auto encoded_v = ggml_cast(a, ggml_permute(a, v, 0, 2, 1, 3), GGML_TYPE_F16);
    auto prefix_v = ggml_reshape_3d(a, pv, width, heads, prefix);
    auto original_v = ggml_concat(a, ggml_cont(a, ggml_permute(a, prefix_v, 0, 2, 1, 3)), encoded_v, 1);
    auto packed_v = wildimage_flash_values::pack(a, backend, pv, v);
    check(packed_v && ggml_are_same_shape(original_v, packed_v));
    check(!wildimage_flash_values::pack(a, backend, pv, pk));
    ggml_cgraph* value_graphs[] = {ggml_new_graph(a), ggml_new_graph(a)};
    ggml_build_forward_expand(value_graphs[0], original_v);
    ggml_build_forward_expand(value_graphs[1], packed_v);
    ggml_cgraph* graphs[] = {ggml_new_graph(a), ggml_new_graph(a)};
    ggml_build_forward_expand(graphs[0], reference);
    ggml_build_forward_expand(graphs[1], candidate);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> storage(ggml_backend_alloc_ctx_tensors(a, backend), ggml_backend_buffer_free);
    check(storage != nullptr);
    uint32_t rng = 751;
    for (auto t : {q, k, v, pk, pv}) {
        std::vector<float> values(ggml_nelements(t));
        for (auto& value : values) {
            rng = 1664525u * rng + 1013904223u;
            value = float(int(rng >> 16) - 32768) / 32768.f;
        }
        if (t->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> half(values.size());
            ggml_fp32_to_fp16_row(values.data(), half.data(), half.size());
            ggml_backend_tensor_set(t, half.data(), 0, ggml_nbytes(t));
        } else {
            ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t));
        }
    }
    for (auto graph : graphs) check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> before(ggml_nelements(reference)), after(before.size());
    ggml_backend_tensor_get(reference, before.data(), 0, ggml_nbytes(reference));
    ggml_backend_tensor_get(candidate, after.data(), 0, ggml_nbytes(candidate));
    check(std::memcmp(before.data(), after.data(), ggml_nbytes(reference)) == 0);
    // Also check the representation itself: Flash Attention may hide a rounding
    // or stride error in a low-weight token. Repeat after replacing input data.
    std::vector<ggml_fp16_t> expected(ggml_nelements(packed_v)), actual(expected.size());
    for (int i = 0; i < 2; ++i) {
        for (auto graph : value_graphs) check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(original_v, expected.data(), 0, ggml_nbytes(original_v));
        ggml_backend_tensor_get(packed_v, actual.data(), 0, ggml_nbytes(packed_v));
        check(expected == actual);
        if (i == 0) {
            std::vector<float> fresh(ggml_nelements(v), -0.0f);
            ggml_backend_tensor_set(v, fresh.data(), 0, ggml_nbytes(v));
        }
    }
    if (tokens == 1024) {
        benchmark(backend, value_graphs, "value pack", tokens, heads);
    }
    if (tokens == 1024) {
        benchmark(backend, graphs, "cached attention", tokens, heads);
    }
}

int main() {
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cuda(ggml_backend_cuda_init(0), ggml_backend_free);
    check(cuda != nullptr);
    wildimage_register_fp8(ggml_backend_get_device(cuda.get()));
    compare(cuda.get(), 1, 9, 2);
    compare(cuda.get(), 13, 129, 4);
    compare(cuda.get(), 257, 256, 8);
    compare(cuda.get(), 57, 1024, 8);
    std::puts("Qwen CUDA cached attention matches original values: PASS");
}
