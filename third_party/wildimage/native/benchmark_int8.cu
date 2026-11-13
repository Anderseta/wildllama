#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include "core/cuda/i8_gemm.cu"
#include "ggml-cuda.h"

static void check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
template <typename T>
static T* upload(const std::vector<T>& data) {
    T* out = nullptr;
    check(cudaMalloc(&out, data.size() * sizeof(T)));
    check(cudaMemcpy(out, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
    return out;
}
template <class Config>
static void measure(ggml_backend_dev_t device, int rows, int n, int k, bool swiglu, const char* name, int8_t* x, int8_t* w, float* xs, float* ws, float* out, std::vector<float>& expected) {
    sd_i8_cutlass::Plan p{};
    p.device = sd_i8_cutlass::device_info(device);
    p.rows   = rows;
    p.n      = n;
    p.k      = k;
    p.swiglu = swiglu;
    if (!sd_i8_cutlass::configure_tile<Config>(p)) {
        std::printf("unsupported %s\n", name);
        return;
    }
    void* scratch = nullptr;
    check(cudaMalloc(&scratch, std::max<size_t>(256, p.scratch_offset + p.kernel_bytes)));
    auto run = [&] {
        if (p.launch(p, x, w, xs, ws, out, scratch, nullptr) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("GEMM launch failed");
    };
    for (int i = 0; i < 4; ++i)
        run();
    std::vector<float> actual(size_t(rows) * n / (swiglu ? 2 : 1));
    check(cudaMemcpy(actual.data(), out, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
    if (expected.empty())
        expected = actual;
    if (actual != expected)
        throw std::runtime_error(std::string("non-exact GEMM: ") + name);
    cudaEvent_t start, end;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&end));
    std::vector<float> times;
    for (int trial = 0; trial < 5; ++trial) {
        check(cudaEventRecord(start));
        for (int i = 0; i < 10; ++i)
            run();
        check(cudaEventRecord(end));
        check(cudaEventSynchronize(end));
        float elapsed;
        check(cudaEventElapsedTime(&elapsed, start, end));
        times.push_back(elapsed / 10.f);
    }
    std::sort(times.begin(), times.end());
    std::printf("rows=%d n=%d k=%d swiglu=%d tile=%s ms=%.4f occupancy=%d scratch=%zu exact=1\n",
                rows, n, k, swiglu, name, times[2], p.occupancy, p.scratch_offset + p.kernel_bytes);
    std::fflush(stdout);
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(end));
    check(cudaFree(scratch));
}
static void shape(ggml_backend_dev_t device, int rows, int n, int k, bool swiglu) {
    std::vector<int8_t> hx(size_t(rows) * k), hw(size_t(n) * k);
    std::vector<float> hxs((rows + 3) / 4 * 4), hws(n);
    for (size_t i = 0; i < hx.size(); ++i)
        hx[i] = int(i % 251) - 125;
    for (size_t i = 0; i < hw.size(); ++i)
        hw[i] = int((i * 17) % 255) - 127;
    for (size_t i = 0; i < hxs.size(); ++i)
        hxs[i] = (i % 17 + 1) * .0003f;
    for (size_t i = 0; i < hws.size(); ++i)
        hws[i] = (i % 31 + 1) * .0002f;
    auto x = upload(hx), w = upload(hw);
    auto xs = upload(hxs), ws = upload(hws);
    float* out;
    check(cudaMalloc(&out, size_t(rows) * n * sizeof(float)));
    std::vector<float> expected;
    using namespace sd_i8_cutlass;
    measure<TileConfig<128, 256, 64>>(device, rows, n, k, swiglu, "128x256", x, w, xs, ws, out, expected);
    measure<TileConfig<128, 128, 64>>(device, rows, n, k, swiglu, "128x128", x, w, xs, ws, out, expected);
    measure<TileConfig<64, 128, 32>>(device, rows, n, k, swiglu, "64x128", x, w, xs, ws, out, expected);
    measure<TileConfig<16, 128, 16>>(device, rows, n, k, swiglu, "16x128", x, w, xs, ws, out, expected);
    check(cudaFree(x));
    check(cudaFree(w));
    check(cudaFree(xs));
    check(cudaFree(ws));
    check(cudaFree(out));
}
int main() {
    try {
        auto gpu = ggml_backend_cuda_init(0);
        if (!gpu)
            throw std::runtime_error("no GPU");
        auto device = ggml_backend_get_device(gpu);
        for (int rows : {32, 1024, 4096, 6144}) {
            shape(device, rows, 4096, 4096, false);
            shape(device, rows, 12288, 4096, false);
            shape(device, rows, 4096, 12288, false);
            shape(device, rows, 24576, 4096, true);
        }
        ggml_backend_free(gpu);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
