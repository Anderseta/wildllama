// Offline Wan backend probe. Link with the native media objects of the tested build.
// Args: runtime_directory query_tokens|conv key_tokens|width heads|channels fixture_file|-
// An optional fixture contains the FP32 output in GGML order and .inN files for cuDNN.
// The convolution case times the production direct GGML operation in both slots.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include "core/probe/backend_probe.h"
#include "ggml-ext.h"
#include "ggml.h"
void sd_ggml_register_extensions(ggml_backend_t);
int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: wan-probe RUNTIME Q|conv K|WIDTH HEADS|CHANNELS FILE|-\n");
        return 2;
    }
    const bool conv = std::strcmp(argv[2], "conv") == 0;
    const int nq = conv ? 1 : std::atoi(argv[2]), nk = std::atoi(argv[3]), heads = std::atoi(argv[4]);
    if (nq < 1 || nq > 8192 || nk < nq || nk > 90000 || heads < 1 || heads > (conv ? 192 : 40) ||
        (conv && nk > 128))
        return 2;
    ggml_backend_load_all_from_path(argv[1]);
    auto backend = ggml_backend_init_by_name("CUDA0", nullptr);
    if (!backend)
        return 3;
    sd_ggml_register_extensions(backend);
    auto dev = ggml_backend_get_device(backend);
    BackendProbeCase test;
    test.build = [&](BackendProbeGraph& g, bool cute) {
        if (conv) {
            auto x = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, nk, nk, 4, heads);
            auto w = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, 3, 3, 3, heads * heads);
            g.inputs = {x, w};
            g.output = ggml_conv_3d_direct(g.ctx, w, x, 1, 1, 1, 1, 1, 1, 1, 1, 1, heads, 1, heads);
            return;
        }
        auto q = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, 128, nq, heads);
        auto k = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F16, 128, nk, heads);
        auto v = ggml_dup_tensor(g.ctx, k);
        g.inputs = {q, k, v};
        g.output = ggml_flash_attn_ext(g.ctx, q, k, v, nullptr, 1.f / std::sqrt(128.f), 0, 0);
        ggml_prec_set_acc(g.output, GGML_PREC_F32);
        ggml_set_name(g.output, cute ? "wan.bench.cute" : "wan.bench.native");
        if (!ggml_ext_dev_supports_native_op(dev, g.output))
            g.output = nullptr;
    };
    test.initialize = [&](BackendProbeGraph& g) {
        unsigned seed = 425;
        std::vector<float> block(16384);
        std::vector<ggml_fp16_t> half(block.size());
        for (size_t j = 0; j < g.inputs.size(); ++j) {
            auto* x = g.inputs[j];
            std::ofstream fixture;
            if (std::strcmp(argv[5], "-"))
                fixture.open(std::string(argv[5]) + ".in" + std::to_string(j), std::ios::binary);
            for (int64_t at = 0; at < ggml_nelements(x); at += block.size()) {
                int n = static_cast<int>(std::min<int64_t>(block.size(), ggml_nelements(x) - at));
                for (int i = 0; i < n; ++i) {
                    seed = 1664525U * seed + 1013904223U;
                    block[i] = (int(seed >> 21) - 1024) / 512.f;
                }
                if (fixture.is_open())
                    fixture.write(reinterpret_cast<const char*>(block.data()), n * sizeof(float));
                if (x->type == GGML_TYPE_F16)
                    ggml_fp32_to_fp16_row(block.data(), half.data(), n);
                ggml_backend_tensor_set(
                    x, x->type == GGML_TYPE_F32 ? static_cast<void*>(block.data()) : half.data(),
                    at * ggml_type_size(x->type), n * ggml_type_size(x->type));
            }
        }
        return true;
    };
    double maximum = 0, squares = 0;
    size_t differences = 0;
    test.compare = [&](const BackendProbeGraph& native, const BackendProbeGraph& cute) {
        size_t n = ggml_nelements(cute.output);
        std::vector<float> a(n), b(n);
        ggml_backend_tensor_get(native.output, a.data(), 0, n * sizeof(float));
        ggml_backend_tensor_get(cute.output, b.data(), 0, n * sizeof(float));
        if (std::strcmp(argv[5], "-")) {
            std::ofstream f(argv[5], std::ios::binary);
            f.write(reinterpret_cast<const char*>(b.data()), n * sizeof(float));
            if (!f)
                return false;
        }
        for (size_t i = 0; i < n; ++i) {
            if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
                return false;
            double error = std::abs(a[i] - b[i]);
            maximum = std::max(maximum, error);
            squares += error * error;
            differences += a[i] != b[i];
        }
        return true;
    };
    BackendProbeOptions opts;
    opts.min_free_device_bytes =
        2ULL * (uint64_t(4) * 128 * nq + uint64_t(4) * 128 * nk + uint64_t(4) * 128 * nq) * heads +
        (2ULL << 30);
    opts.time_budget_ms = 25000;
    auto result = sd_run_backend_probe(backend, 1, {test}, opts);
    std::printf(
        "op=%s q=%d k=%d heads=%d reference_ms=%.4f candidate_ms=%.4f different=%zu max_abs=%.8g rms=%.8g "
        "completed=%d %s\n",
        conv ? "conv3d" : "attention", nq, nk, heads, result.reference_ms, result.candidate_ms, differences,
        maximum, std::sqrt(squares / (conv ? double(nk) * nk * 4 * heads : double(nq) * heads * 128)),
        result.completed, result.error.c_str());
    ggml_backend_free(backend);
    return result.completed ? 0 : 1;
}
