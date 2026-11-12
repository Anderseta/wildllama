#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#include "wildimage_fp8.h"
#include <cuda_fp8.h>
#include <map>
#include <vector>
#include <cmath>
#include <cstdio>
#include <stdexcept>

void wildimage_register_fp8(ggml_backend_dev_t);
struct Context {
    ggml_context* ggml_ctx;
    ggml_backend_t backend;
    std::map<ggml_tensor*, ggml_tensor*> fp8_pack_cache;
};
static void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
static void trial(ggml_backend_t gpu, ggml_backend_t cpu, int k, int m, int n, bool unaligned = false) {
    auto ctx = ggml_init({ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(128, false), nullptr, true});
    require(ctx != nullptr, "context allocation");
    Context c{ctx, gpu, {}};
    auto x = unaligned ? ggml_view_2d(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k*n+1), k, n, k*sizeof(float), sizeof(float)) :
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    auto w = ggml_new_tensor_2d(ctx, GGML_TYPE_F8_E4M3, k, m);
    auto baseline = ggml_mul_mat(ctx, w, x);
    auto first = wildimage_fp8::linear(&c, x, w, false, 1.f);
    auto second = wildimage_fp8::linear(&c, x, w, false, 1.f);
    require(first && second && first->src[1] == second->src[1] && c.fp8_pack_cache.size() == 1, "shared packing");
    require(!wildimage_fp8::linear(&c, x, w, true, 1.f), "precise mode fallback");
    require(!wildimage_fp8::linear(&c, x, w, false, .5f), "scale fallback");
    auto bad_weight = ggml_view_2d(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F8_E4M3, k*m+16), k, m, k, 1);
    require(!wildimage_fp8::linear(&c, x, bad_weight, false, 1.f), "unaligned weight fallback");
    c.backend = cpu;
    require(!wildimage_fp8::linear(&c, x, w, false, 1.f), "CPU fallback");
    c.backend = gpu;
    auto graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, baseline);
    ggml_build_forward_expand(graph, first);
    ggml_build_forward_expand(graph, second);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    require(buffer != nullptr, "GPU allocation");
    std::vector<uint8_t> weights(k*m);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = __nv_fp8_e4m3(float(int(i%97)-48)/64.f).__x;
    ggml_backend_tensor_set(w, weights.data(), 0, weights.size());
    std::vector<float> input(k*n), original(m*n), actual(m*n), other(m*n);
    // Repeated execution changes activations after graph capture, including zeros.
    for (int repeat = 0; repeat < 6; ++repeat) {
        for (size_t i = 0; i < input.size(); ++i) input[i] = repeat == 4 ? 0.f : std::sin(float(i*3+repeat*37))*(repeat+1);
        ggml_backend_tensor_set(x, input.data(), 0, input.size()*sizeof(float));
        require(ggml_backend_graph_compute(gpu, graph) == GGML_STATUS_SUCCESS, "graph compute");
        ggml_backend_tensor_get(baseline, original.data(), 0, original.size()*sizeof(float));
        ggml_backend_tensor_get(first, actual.data(), 0, actual.size()*sizeof(float));
        ggml_backend_tensor_get(second, other.data(), 0, other.size()*sizeof(float));
        require(original == actual && original == other, "FP8 numerical parity / fresh activation");
    }
    std::printf("PASS K=%d M=%d N=%d unaligned=%d: exact FP32 parity, shared packing, replay and fallbacks\n", k, m, n, unaligned);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}
int main() {
    try {
        auto gpu = ggml_backend_cuda_init(0), cpu = ggml_backend_cpu_init();
        require(gpu && cpu, "backends");
        wildimage_register_fp8(ggml_backend_get_device(gpu));
        trial(gpu, cpu, 128, 128, 1);
        trial(gpu, cpu, 256, 128, 17);
        trial(gpu, cpu, 3072, 3072, 512);
        trial(gpu, cpu, 8192, 3072, 256);
        trial(gpu, cpu, 4096, 4096, 512);
        trial(gpu, cpu, 4096, 16384, 256);
        trial(gpu, cpu, 256, 128, 17, true);
        ggml_backend_free(gpu); ggml_backend_free(cpu);
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
}
