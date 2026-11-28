#include "core/wildimage_fp8.h"
#include "core/wildimage_flash_values.h"
#include "core/wildimage_scaled_projection.h"
#include "core/cuda/runtime.h"
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <algorithm>
#include <new>

namespace wildimage_fp8 {
// Identical global absolute maximum and E4M3 rounding to the native FP8 path.
// Contiguous float4 loads avoid per-element 64-bit divisions/stride arithmetic.
__device__ __forceinline__ float4 load4(const float* x, int64_t index) {
    if ((reinterpret_cast<uintptr_t>(x) & 15) == 0) return reinterpret_cast<const float4*>(x)[index];
    x += 4 * index;
    return make_float4(x[0], x[1], x[2], x[3]);
}
__global__ void maximum(const float* x, float* out, int64_t count) {
    float value = 0.f;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += int64_t(blockDim.x) * gridDim.x) {
        const auto v = load4(x, i);
        value = fmaxf(value, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    for (int delta = 16; delta; delta >>= 1) value = fmaxf(value, __shfl_down_sync(0xffffffff, value, delta));
    __shared__ float warps[8];
    if ((threadIdx.x & 31) == 0) warps[threadIdx.x >> 5] = value;
    __syncthreads();
    if (threadIdx.x < 32) {
        value = threadIdx.x < 8 ? warps[threadIdx.x] : 0.f;
        for (int delta = 16; delta; delta >>= 1) value = fmaxf(value, __shfl_down_sync(0xffffffff, value, delta));
        if (threadIdx.x == 0) atomicMax(reinterpret_cast<unsigned*>(out), __float_as_uint(value));
    }
}
__global__ void quantize(const float* x, uint32_t* out, const float* amax, float* scale, int64_t count) {
    const float a = *amax, inv = a > 0.f ? 448.f / a : 1.f;
    if (blockIdx.x == 0 && threadIdx.x == 0) *scale = a > 0.f ? a / 448.f : 1.f;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += int64_t(blockDim.x) * gridDim.x) {
        const auto v = load4(x, i);
        const __nv_fp8_e4m3 a0(v.x * inv), a1(v.y * inv), a2(v.z * inv), a3(v.w * inv);
        out[i] = uint32_t(a0.__x) | (uint32_t(a1.__x) << 8) | (uint32_t(a2.__x) << 16) | (uint32_t(a3.__x) << 24);
    }
}
static bool supports(ggml_backend_dev_t, const ggml_tensor* node, void*) {
    ggml_ext_tensor_desc shape;
    auto op = ggml_ext_op_get(node);
    const bool ok = op == pack_op() ? pack_shape(node->src, 1, nullptr, &shape) :
        op == matmul_op() && matmul_shape(node->src, 2, ggml_ext_op_attributes(node), &shape);
    if (ok && op == matmul_op()) {
        auto weight = node->src[0];
        size_t offset = 0;
        while (!weight->data && weight->view_src) { offset += weight->view_offs; weight = weight->view_src; }
        if ((reinterpret_cast<uintptr_t>(weight->data) + offset) % 16) return false;
    }
    return ok && node->type == shape.type && std::memcmp(node->ne, shape.ne, sizeof(shape.ne)) == 0 && ggml_is_contiguous(node);
}
struct Plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    ~Plan() {
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (op) cublasLtMatmulDescDestroy(op);
    }
};
static ggml_status prepare(ggml_backend_t, ggml_tensor* const* nodes, int, void** state, size_t* workspace, void*) {
    *workspace = 0;
    if (ggml_ext_op_get(nodes[0]) == pack_op()) return GGML_STATUS_SUCCESS;
    auto p = new(std::nothrow) Plan;
    if (!p) return GGML_STATUS_ALLOC_FAILED;
    *state = p;
    Shape s; std::memcpy(&s, ggml_ext_op_attributes(nodes[0]), sizeof(s));
    auto m = nodes[0]->ne[0]; cublasOperation_t trans = CUBLAS_OP_T;
    if (cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&p->a, CUDA_R_8F_E4M3, s.k, m, s.k) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&p->b, CUDA_R_8F_E4M3, s.k, s.n, s.k) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&p->c, CUDA_R_32F, m, s.n, m) != CUBLAS_STATUS_SUCCESS) return GGML_STATUS_FAILED;
    return GGML_STATUS_SUCCESS;
}
static void destroy(void* state, void*) { delete static_cast<Plan*>(state); }
static ggml_status execute(const ggml_ext_execution_context* e, ggml_tensor* const* nodes, int, void* state, void*) {
    ggml_ext_gpu_context context;
    if (!sd_cuda::cuda_context(e->backend, &context)) return GGML_STATUS_FAILED;
    const auto stream = static_cast<cudaStream_t>(context.stream);
    auto node = nodes[0];
    if (ggml_ext_op_get(node) == pack_op()) {
        const int64_t elements = ggml_nelements(node->src[0]);
        auto stats = reinterpret_cast<float*>(static_cast<char*>(node->data) + elements);
        if (cudaMemsetAsync(stats, 0, 32, stream) != cudaSuccess) return GGML_STATUS_FAILED;
        const int blocks = int(std::min<int64_t>((elements / 4 + 255) / 256, 1024));
        maximum<<<blocks,256,0,stream>>>(static_cast<const float*>(node->src[0]->data), stats, elements / 4);
        quantize<<<blocks,256,0,stream>>>(static_cast<const float*>(node->src[0]->data), static_cast<uint32_t*>(node->data), stats, stats + 4, elements / 4);
        return cudaGetLastError() == cudaSuccess ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
    }
    struct Handle {
        cublasLtHandle_t h = nullptr;
        Handle() { cublasLtCreate(&h); }
        ~Handle() { if (h) cublasLtDestroy(h); }
    };
    // cuBLASLt's handle is device-independent; this cache owns no model tensors.
    static thread_local Handle handle;
    if (!handle.h) return GGML_STATUS_FAILED;
    auto p = static_cast<Plan*>(state); Shape s;
    std::memcpy(&s, ggml_ext_op_attributes(node), sizeof(s));
    const void* scale = static_cast<const char*>(node->src[1]->data) + s.k * s.n + 16;
    if (cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale, sizeof(scale)) != CUBLAS_STATUS_SUCCESS) return GGML_STATUS_FAILED;
    const float alpha = 1.f, beta = 0.f;
    auto status = cublasLtMatmul(handle.h, p->op, &alpha, node->src[0]->data, p->a, node->src[1]->data, p->b,
        &beta, node->data, p->c, node->data, p->c, nullptr, nullptr, 0, stream);
    return status == CUBLAS_STATUS_SUCCESS ? GGML_STATUS_SUCCESS :
        status == CUBLAS_STATUS_ALLOC_FAILED ? GGML_STATUS_ALLOC_FAILED : GGML_STATUS_FAILED;
}
}
namespace wildimage_projection {
template <bool residual>
__device__ __forceinline__ float finish(float a, float b, float c, float scale) {
    const float value = __fmul_rn(a, scale);
    if constexpr (residual) return __fadd_rn(c, __fmul_rn(value, b));
    const float up = __fmul_rn(b, scale);
    return __fmul_rn(value / (1.f + expf(-value)), up);
}

template <bool residual, class T>
__global__ void scaled_projection(const T* projection, const float* weight_scale, const T* skip,
                                  const T* target_gate, const T* prefix_gate, T* out,
                                  int64_t width, int64_t rows, int64_t prefix) {
    const int64_t col = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= width) return;
    const float scale = *weight_scale;
    for (int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
        const int64_t idx = row * width + col;
        T a, b, c{};
        if constexpr (residual) {
            a = projection[idx];
            b = (row < prefix ? prefix_gate : target_gate)[col];
            c = skip[idx];
        } else {
            a = projection[2 * row * width + col];
            b = projection[(2 * row + 1) * width + col];
        }
        if constexpr (sizeof(T) == sizeof(float)) {
            out[idx] = finish<residual>(a, b, c, scale);
        } else {
            out[idx] = make_float4(finish<residual>(a.x, b.x, c.x, scale), finish<residual>(a.y, b.y, c.y, scale),
                                   finish<residual>(a.z, b.z, c.z, scale), finish<residual>(a.w, b.w, c.w, scale));
        }
    }
}

static bool supports(ggml_backend_dev_t, const ggml_tensor* node, void*) {
    const bool add = ggml_ext_op_get(node) == operation(true);
    const int count = add ? (node->src[4] ? 5 : 4) : 2;
    ggml_ext_tensor_desc shape;
    return infer(node->src, count, add ? ggml_ext_op_attributes(node) : nullptr, &shape) && matrix(node) &&
           node->ne[0] <= INT32_MAX && !std::memcmp(node->ne, shape.ne, sizeof(shape.ne));
}

template <bool add, class T>
static void launch(ggml_tensor* node, int64_t prefix, cudaStream_t stream) {
    const int64_t width = node->ne[0] / (sizeof(T) / sizeof(float));
    auto input = [&](int i) { return node->src[i] ? static_cast<const T*>(node->src[i]->data) : nullptr; };
    const dim3 grid(unsigned((width + 255) / 256), unsigned(std::min<int64_t>(node->ne[1], 65535)));
    scaled_projection<add><<<grid, 256, 0, stream>>>(input(0), static_cast<const float*>(node->src[1]->data),
        input(2), input(3), input(4), static_cast<T*>(node->data), width, node->ne[1], prefix);
}

static ggml_status execute(const ggml_ext_execution_context* execution, ggml_tensor* const* nodes, int, void*, void*) {
    ggml_ext_gpu_context context;
    if (!sd_cuda::cuda_context(execution->backend, &context)) return GGML_STATUS_FAILED;
    auto* node = nodes[0];
    auto stream = static_cast<cudaStream_t>(context.stream);
    uintptr_t alignment = reinterpret_cast<uintptr_t>(node->data);
    for (int i : {0, 2, 3, 4}) {
        if (node->src[i]) alignment |= reinterpret_cast<uintptr_t>(node->src[i]->data);
    }
    const bool vectorized = !(alignment % alignof(float4)) && node->ne[0] % 4 == 0;
    if (ggml_ext_op_get(node) == operation(true)) {
        int64_t prefix;
        std::memcpy(&prefix, ggml_ext_op_attributes(node), sizeof(prefix));
        if (vectorized) launch<true, float4>(node, prefix, stream);
        else launch<true, float>(node, prefix, stream);
    } else {
        if (vectorized) launch<false, float4>(node, 0, stream);
        else launch<false, float>(node, 0, stream);
    }
    return cudaGetLastError() == cudaSuccess ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}
}
namespace wildimage_flash_values {
// A warp writes one 128-channel row. Copy cached half bits verbatim and
// round fresh values once, directly into their final attention layout.
template<bool aligned>
__global__ void pack_rows(const half* prefix, const float* target, half* output,
                          int64_t heads, int64_t prefix_len, int64_t total) {
    const int64_t token = int64_t(blockIdx.x) * blockDim.y + threadIdx.y;
    if (token >= total) return;
    const int64_t head = blockIdx.y, col = 4 * threadIdx.x;
    half* dst = output + (head * total + token) * 128 + col;
    if (token < prefix_len) {
        const half* src = prefix + (token * heads + head) * 128 + col;
        if constexpr (aligned) {
            *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(src);
        } else {
            for (int i = 0; i < 4; ++i) dst[i] = src[i];
        }
    } else {
        const float* src = target + ((token - prefix_len) * heads + head) * 128 + col;
        if constexpr (aligned) {
            const float4 v = *reinterpret_cast<const float4*>(src);
            auto* d = reinterpret_cast<half2*>(dst);
            d[0] = __floats2half2_rn(v.x, v.y);
            d[1] = __floats2half2_rn(v.z, v.w);
        } else {
            for (int i = 0; i < 4; ++i) dst[i] = __float2half_rn(src[i]);
        }
    }
}

static bool supports(ggml_backend_dev_t, const ggml_tensor* node, void*) {
    ggml_ext_tensor_desc out;
    return shape(node->src, 2, nullptr, &out) && node->type == out.type &&
           !std::memcmp(node->ne, out.ne, sizeof(out.ne)) && ggml_is_contiguous(node);
}

static ggml_status execute(const ggml_ext_execution_context* e, ggml_tensor* const* nodes, int, void*, void*) {
    ggml_ext_gpu_context context;
    if (!sd_cuda::cuda_context(e->backend, &context)) return GGML_STATUS_FAILED;
    auto node = nodes[0];
    auto p = static_cast<const half*>(node->src[0]->data);
    auto t = static_cast<const float*>(node->src[1]->data);
    auto out = static_cast<half*>(node->data);
    const int64_t prefix = node->src[0]->ne[1], total = node->ne[1], heads = node->ne[2];
    const dim3 block(32, 8), grid(unsigned((total + 7) / 8), unsigned(heads));
    const auto stream = static_cast<cudaStream_t>(context.stream);
    if (!(uintptr_t(p) % alignof(uint2)) && !(uintptr_t(t) % alignof(float4)) && !(uintptr_t(out) % alignof(uint2))) {
        pack_rows<true><<<grid, block, 0, stream>>>(p, t, out, heads, prefix, total);
    } else {
        pack_rows<false><<<grid, block, 0, stream>>>(p, t, out, heads, prefix, total);
    }
    return cudaGetLastError() == cudaSuccess ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}
}

void wildimage_register_fp8(ggml_backend_dev_t device) {
    const auto info = sd_cuda::cuda_device_info(device);
    int major = 0, minor = 0;
    if (info.device < 0 || info.kind != GGML_EXT_GPU_CUDA ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, info.device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, info.device) != cudaSuccess || major * 10 + minor < 89) return;
    ggml_ext_kernel_desc desc{};
    desc.abi_version = GGML_EXT_ABI_VERSION;
    desc.op = GGML_OP_CUSTOM; desc.priority = 100;
    desc.supports = wildimage_fp8::supports;
    desc.prepare = wildimage_fp8::prepare; desc.execute = wildimage_fp8::execute; desc.destroy = wildimage_fp8::destroy;
    for (auto op : {wildimage_fp8::pack_op(), wildimage_fp8::matmul_op()}) {
        desc.custom_op = op;
        desc.name = op == wildimage_fp8::pack_op() ? "wildimage.fp8.pack.cuda.v1" : "wildimage.fp8.matmul.cuda.v1";
        GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
    }
    ggml_ext_kernel_desc epilogue{};
    epilogue.abi_version = GGML_EXT_ABI_VERSION;
    epilogue.op = GGML_OP_CUSTOM;
    epilogue.priority = 100;
    epilogue.supports = wildimage_projection::supports;
    epilogue.execute = wildimage_projection::execute;
    for (bool residual : {false, true}) {
        epilogue.custom_op = wildimage_projection::operation(residual);
        epilogue.name = residual ? "wildimage.scaled_gated_residual.cuda.v1" : "wildimage.scaled_swiglu.cuda.v1";
        GGML_ASSERT(ggml_ext_register_kernel_ex(device, &epilogue, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
    }
    ggml_ext_kernel_desc values{};
    values.abi_version = GGML_EXT_ABI_VERSION;
    values.op = GGML_OP_CUSTOM;
    values.priority = 100;
    values.name = "wildimage.flash_values.cuda.v1";
    values.custom_op = wildimage_flash_values::operation();
    values.supports = wildimage_flash_values::supports;
    values.execute = wildimage_flash_values::execute;
    GGML_ASSERT(ggml_ext_register_kernel_ex(device, &values, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
}
