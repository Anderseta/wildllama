#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
#include "core/wildimage_scaled_projection.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

void wildimage_register_fp8(ggml_backend_dev_t);

namespace {
void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

void trial(ggml_backend_t gpu, int width, int rows, bool residual, int prefix, bool unaligned, bool benchmark) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({2 << 20, nullptr, true}), ggml_free);
    auto* arena = ctx.get();
    const int projected_width = residual ? width : 2 * width;
    auto proj = ggml_new_tensor_2d(arena, GGML_TYPE_F32, projected_width, rows);
    if (unaligned) {
        auto storage = ggml_new_tensor_1d(arena, GGML_TYPE_F32, int64_t(projected_width) * rows + 1);
        proj = ggml_view_2d(arena, storage, projected_width, rows, projected_width * sizeof(float), sizeof(float));
    }
    auto scale = ggml_new_tensor_1d(arena, GGML_TYPE_F32, 1);
    auto scaled = ggml_mul(arena, proj, scale);
    auto x = ggml_new_tensor_2d(arena, GGML_TYPE_F32, width, rows);
    auto tgt = ggml_new_tensor_1d(arena, GGML_TYPE_F32, width);
    auto pre = ggml_dup_tensor(arena, tgt);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
    check(cpu && !wildimage_projection::fuse(arena, cpu.get(), scaled), "CPU should use its original graph");
    check(!wildimage_projection::fuse(arena, gpu, ggml_add(arena, scaled, scale)), "unexpected projection operation fused");
    ggml_tensor *old_path, *new_path;
    if (residual) {
        ggml_tensor* mod;
        if (prefix) {
            auto p = ggml_view_2d(arena, scaled, width, prefix, scaled->nb[1], 0);
            auto t = ggml_view_2d(arena, scaled, width, rows - prefix, scaled->nb[1], prefix * scaled->nb[1]);
            mod = ggml_concat(arena, ggml_mul(arena, p, pre), ggml_mul(arena, t, tgt), 1);
        } else {
            mod = ggml_mul(arena, scaled, tgt);
        }
        old_path = ggml_add(arena, x, mod);
        new_path = wildimage_projection::fuse(arena, gpu, scaled, x, tgt, pre, prefix);
        check(!wildimage_projection::fuse(arena, gpu, scaled, x, tgt, pre, rows), "invalid prefix accepted");
    } else {
        old_path = ggml_swiglu(arena, scaled);
        new_path = wildimage_projection::fuse(arena, gpu, scaled);
    }
    check(new_path && ggml_are_same_shape(old_path, new_path), "expected fused projection");
    check(!wildimage_projection::fuse(arena, gpu, proj), "accepted an unscaled input");
    auto channel = ggml_new_tensor_1d(arena, GGML_TYPE_F32, projected_width);
    check(!wildimage_projection::fuse(arena, gpu, ggml_mul(arena, proj, channel)), "accepted a channelwise scale");
    auto stock = ggml_new_graph(arena), fused = ggml_new_graph(arena);
    ggml_build_forward_expand(stock, old_path);
    ggml_build_forward_expand(fused, new_path);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> memory(ggml_backend_alloc_ctx_tensors(arena, gpu), ggml_backend_buffer_free);
    check(memory != nullptr, "GPU allocation failed");
    uint32_t rng = 47;
    for (auto t : {proj, x, tgt, pre}) {
        std::vector<float> values(ggml_nelements(t));
        for (auto& v : values) {
            rng = rng * 1664525u + 1013904223u;
            v = float(int(rng >> 16) - 32768) / 361.75f;
        }
        ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t));
    }
    std::vector<float> before(ggml_nelements(old_path)), after(before.size());
    for (float factor : {0.013479f, 0.f, -0.7513f, 1.f}) {
        ggml_backend_tensor_set(scale, &factor, 0, sizeof(factor));
        for (auto* g : {stock, fused}) check(ggml_backend_graph_compute(gpu, g) == GGML_STATUS_SUCCESS, "graph execution failed");
        ggml_backend_tensor_get(old_path, before.data(), 0, ggml_nbytes(old_path));
        ggml_backend_tensor_get(new_path, after.data(), 0, ggml_nbytes(new_path));
        check(!std::memcmp(before.data(), after.data(), ggml_nbytes(old_path)), residual ? "residual differs" : "SwiGLU differs");
    }
    if (benchmark) {
        std::array<std::vector<double>, 2> sample;
        for (int round = 0; round < 9; ++round) {
            for (int k : {round % 2, 1 - round % 2}) {
                const auto start = std::chrono::steady_clock::now();
                for (int i = 0; i < 8; ++i)
                    check(ggml_backend_graph_compute(gpu, k ? fused : stock) == GGML_STATUS_SUCCESS, "benchmark failed");
                ggml_backend_synchronize(gpu);
                sample[k].push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 8);
            }
        }
        for (auto& times : sample) std::sort(times.begin(), times.end());
        std::printf("%s width=%d rows=%d prefix=%d: median %.4f -> %.4f ms\n",
                    residual ? "residual" : "SwiGLU", width, rows, prefix, sample[0][4], sample[1][4]);
    }
}
}

int main() {
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> gpu(ggml_backend_cuda_init(0), ggml_backend_free);
    check(gpu != nullptr, "CUDA backend unavailable");
    wildimage_register_fp8(ggml_backend_get_device(gpu.get()));
    trial(gpu.get(), 15, 3, false, 0, true, false);
    trial(gpu.get(), 16, 3, true, 2, true, false);
    trial(gpu.get(), 12288, 1024, false, 0, false, true);
    trial(gpu.get(), 4096, 1024, true, 0, false, true);
    trial(gpu.get(), 4096, 1024, true, 41, false, true);
    std::puts("Qwen scaled projection CUDA parity: PASS");
}
