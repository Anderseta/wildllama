#define SD_CUDA_I8_CUTLASS
#include <cstdio>
#include <stdexcept>
#include <vector>
#include "core/cuda/int8.cu"
#include "core/ggml_extensions.cpp"
#include "ggml-cuda.h"

// This standalone test registers the CUDA kernels directly.
namespace sd_cpu {
    void register_extensions(ggml_backend_dev_t) {}
}
static void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
static void check_cuda(cudaError_t status) {
    require(status == cudaSuccess, cudaGetErrorString(status));
}

static void quantize_trial(ggml_backend_t gpu, int k, int rows) {
    auto ctx             = ggml_init({ggml_tensor_overhead() * 32, nullptr, true});
    auto backing         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, int64_t(k) * (rows + 4));
    auto x               = ggml_view_2d(ctx, backing, k, rows, size_t(k) * sizeof(float), 0);
    const int32_t group = 256;
    ggml_tensor* inputs[] = {x};
    auto out             = ggml_ext_op_create(ctx, sd_ggml_convrot_op(), inputs, 1, &group);
    ggml_tensor* nodes[] = {out};
    void* fresh          = nullptr;
    size_t fresh_bytes   = 0;
    require(sd_cuda::sd_i8_cuda::prepare(gpu, nodes, 1, &fresh, &fresh_bytes, nullptr) == GGML_STATUS_SUCCESS, "fresh prepare");
    require(static_cast<sd_cuda::sd_i8_cuda::Plan*>(fresh)->output == SIZE_MAX, "fresh output reserved copy scratch");
    auto allocation = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    require(allocation, "allocation");
    void* separate = out->data;
    std::vector<float> input(size_t(k) * rows);
    const size_t meaningful = size_t(k) * ((rows + 3) / 4 * 4) + rows * sizeof(float);
    std::vector<unsigned char> baseline(meaningful), actual(baseline.size());
    for (int repeat = 0; repeat < 3; ++repeat) {
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = repeat == 2 ? 0.f : std::sin(float(i + repeat * 53)) * (1 + repeat);
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
        auto scratch = ggml_backend_alloc_buffer(gpu, fresh_bytes);
        ggml_ext_execution_context e{GGML_EXT_ABI_VERSION, gpu, scratch};
        require(sd_cuda::sd_i8_cuda::execute(&e, nodes, 1, fresh, nullptr) == GGML_STATUS_SUCCESS, "fresh execution");
        check_cuda(cudaDeviceSynchronize());
        ggml_backend_tensor_get(out, baseline.data(), 0, baseline.size());
        ggml_backend_buffer_free(scratch);
        out->data          = x->data;
        void* aliased      = nullptr;
        size_t alias_bytes = 0;
        require(sd_cuda::sd_i8_cuda::prepare(gpu, nodes, 1, &aliased, &alias_bytes, nullptr) == GGML_STATUS_SUCCESS, "alias prepare");
        require(alias_bytes >= baseline.size(), "missing alias scratch");
        scratch     = ggml_backend_alloc_buffer(gpu, alias_bytes);
        e.workspace = scratch;
        require(sd_cuda::sd_i8_cuda::execute(&e, nodes, 1, aliased, nullptr) == GGML_STATUS_SUCCESS, "alias execution");
        check_cuda(cudaDeviceSynchronize());
        check_cuda(cudaMemcpy(actual.data(), out->data, actual.size(), cudaMemcpyDeviceToHost));
        if (actual != baseline) {
            size_t at = 0;
            while (at < actual.size() && actual[at] == baseline[at])
                ++at;
            std::fprintf(stderr, "mismatch k=%d rows=%d repeat=%d byte=%zu expected=%u actual=%u\n", k, rows, repeat, at, unsigned(baseline[at]), unsigned(actual[at]));
            throw std::runtime_error("alias changed packed activations/scales");
        }
        sd_cuda::sd_i8_cuda::destroy(aliased, nullptr);
        ggml_backend_buffer_free(scratch);
        // A stale no-alias plan must fail safely instead of writing beyond its scratch.
        scratch     = ggml_backend_alloc_buffer(gpu, fresh_bytes);
        e.workspace = scratch;
        require(sd_cuda::sd_i8_cuda::execute(&e, nodes, 1, fresh, nullptr) == GGML_STATUS_FAILED, "stale alias plan accepted");
        ggml_backend_buffer_free(scratch);
        out->data = separate;
    }
    std::printf("PASS ConvRot k=%d rows=%d fresh=%zu exact packed bytes, zero input, aliased output and stale plan\n", k, rows, fresh_bytes);
    sd_cuda::sd_i8_cuda::destroy(fresh, nullptr);
    ggml_backend_buffer_free(allocation);
    ggml_free(ctx);
}
int main() {
    try {
        auto gpu = ggml_backend_cuda_init(0);
        require(gpu, "GPU");
        for (int k : {256, 4096, 12288})
            for (int rows : {1, 17, 64})
                quantize_trial(gpu, k, rows);
        ggml_backend_free(gpu);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
