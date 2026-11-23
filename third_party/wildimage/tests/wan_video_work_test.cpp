#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include "ggml-alloc.h"
#include "model/diffusion/wan.hpp"
#include "model/vae/wan_vae.hpp"

static void check(bool ok) {
    if (!ok) throw std::runtime_error("Wan video work regression");
}

template<class A, class B>
static void compare(const char* label, bool benchmark, A original, B candidate) {
    using namespace std::chrono;
    std::vector<double> times[2];
    const auto measure = [&](auto fn, int variant) {
        auto start = steady_clock::now();
        auto output = fn();
        times[variant].push_back(duration<double, std::milli>(steady_clock::now() - start).count());
        return output;
    };
    for (int i = 0; i < (benchmark ? 5 : 1); ++i) {
        decltype(original()) left, right;
        if (i % 2) {
            right = measure(candidate, 1);
            left = measure(original, 0);
        } else {
            left = measure(original, 0);
            right = measure(candidate, 1);
        }
        check(left.size() == right.size() && !left.empty() &&
              std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) == 0);
    }
    if (benchmark) {
        for (auto& samples : times) std::sort(samples.begin(), samples.end());
        std::printf("%s (ms): %.3f -> %.3f\n", label, times[0][2], times[1][2]);
    }
}

static void grid(int t, int h, int w, int pt, int ph, int pw, int bs, int offset, bool timing) {
    const std::vector<int> dims{44, 42, 42};
    compare("position grid", timing, [&] {
        return Rope::embed_nd(Rope::gen_vid_ids(t, h, w, pt, ph, pw, bs, offset), bs, 10000.f, dims);
    }, [&] {
        return Rope::gen_wan_pe(t, h, w, pt, ph, pw, bs, 10000, dims, offset);
    });
}

static void latents(std::vector<int64_t> shape, bool timing) {
    using Tensor = sd::Tensor<float>;
    const size_t channel_axis = shape.size() - 2;
    std::vector<int64_t> stats_shape(shape.size(), 1);
    stats_shape[channel_axis] = shape[channel_axis];
    Tensor input(shape), mean(stats_shape), deviation(stats_shape);
    for (int i = 0; i < mean.numel(); ++i) {
        mean.data()[i] = (i - 7) * .077f;
        deviation.data()[i] = 1.119f + i * .08f;
    }
    for (int64_t i = 0; i < input.numel(); ++i) input.data()[i] = ((i % 2333) - 833) * .00387f;
    compare("VAE normalize", timing, [&] {return (((input - mean) * 1.f) / deviation).values();}, [&] {
        return WAN::WanVAERunner::unit_scale_latents<true>(input, mean, deviation).values();
    });
    compare("VAE denormalize", timing, [&] {return ((input * deviation) / 1.f + mean).values();}, [&] {
        return WAN::WanVAERunner::unit_scale_latents<false>(input, mean, deviation).values();
    });
}

static void cuda_modulation(ggml_backend_t backend, int width, int rows, bool timing) {
    auto arena = ggml_init({2 << 20, nullptr, true});
    check(arena != nullptr);
    GGMLRunnerContext ctx{};
    ctx.ggml_ctx = arena;
    ctx.backend = backend;
    auto x = ggml_new_tensor_2d(arena, GGML_TYPE_F32, width, rows);
    auto embedding = ggml_new_tensor_1d(arena, GGML_TYPE_F32, width * 4);
    auto s = ggml_view_1d(arena, embedding, width, width * sizeof(float));
    auto h = ggml_view_1d(arena, embedding, width, width * sizeof(float) * 3);
    auto old = ggml_add(arena, ggml_add(arena, x, ggml_mul(arena, x, s)), h);
    auto fused = WAN::modulate_norm(&ctx, x, s, h);
    check(fused->op == GGML_OP_CUSTOM);
    ggml_cgraph* graphs[]{ggml_new_graph(arena), ggml_new_graph(arena)};
    ggml_build_forward_expand(graphs[0], old);
    ggml_build_forward_expand(graphs[1], fused);
    auto buffer = ggml_backend_alloc_ctx_tensors(arena, backend);
    check(buffer != nullptr);
    std::vector<float> data(ggml_nelements(x));
    for (size_t i = 0; i < data.size(); ++i) data[i] = (int(i % 827) - 390) * .002871f;
    ggml_backend_tensor_set(x, data.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(embedding, data.data(), 0, ggml_nbytes(embedding));
    std::vector<float> ref(data.size());
    for (int k : {0, 1}) {
        check(ggml_backend_graph_compute(backend, graphs[k]) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(k ? fused : old, k ? data.data() : ref.data(), 0, ggml_nbytes(x));
    }
    check(std::memcmp(data.data(), ref.data(), ggml_nbytes(x)) == 0);
    if (timing) {
        std::vector<double> times[2];
        for (int pair = 0; pair < 5; ++pair) {
            for (int k : {pair % 2, 1 - pair % 2}) {
                auto start = std::chrono::steady_clock::now();
                for (int n = 0; n < 16; ++n) check(ggml_backend_graph_compute(backend, graphs[k]) == GGML_STATUS_SUCCESS);
                ggml_backend_synchronize(backend);
                times[k].push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 16);
            }
        }
        for (auto& samples : times) std::sort(samples.begin(), samples.end());
        std::printf("CUDA modulation %dx%d (ms): %.3f -> %.3f\n", width, rows, times[0][2], times[1][2]);
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(arena);
}

// Build two tensors with different storage types and physical/name ordering.
// The raw target uploads directly, while half-to-float still needs conversion.
static void mapped_upload(ggml_backend_t backend) {
    const auto dir = std::filesystem::temp_directory_path() /
                     ("wan-upload-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    check(std::filesystem::create_directory(dir));
    const auto path = dir / "source.safetensors";
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code ec; std::filesystem::remove(path, ec); std::filesystem::remove(path.parent_path(), ec); }
    } cleanup{path};
    std::vector<float> source(1 << 20);
    std::vector<uint16_t> halves(1024, 0x3e00);  // 1.5
    for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<float>(i) / 64.f;
    {
        std::ofstream file(path, std::ios::binary);
        std::string header = R"({"z.weight":{"dtype":"F32","shape":[1048576],"data_offsets":[0,4194304]},"a.weight":{"dtype":"F16","shape":[1024],"data_offsets":[4194304,4196352]}})";
        header.resize((header.size() + 7) / 8 * 8, ' ');
        const uint64_t length = header.size();
        for (int i = 0; i < 8; ++i) file.put(static_cast<char>(length >> (8 * i)));
        file.write(header.data(), header.size());
        file.write(reinterpret_cast<const char*>(source.data()), source.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(halves.data()), halves.size() * sizeof(uint16_t));
        check(file.good());
    }
    ModelLoader loader;
    check(loader.init_from_file(path.string()));
    loader.set_n_threads(2);
    auto arena = ggml_init({1 << 20, nullptr, true});
    check(arena != nullptr);
    auto direct = ggml_new_tensor_1d(arena, GGML_TYPE_F32, source.size());
    auto converted = ggml_new_tensor_1d(arena, GGML_TYPE_F32, halves.size());
    auto storage = ggml_backend_alloc_ctx_tensors(arena, backend);
    check(storage != nullptr);
    const std::set<std::string> names{"z.weight", "a.weight"};
    uint64_t staging_peak[2]{};
    for (int direct_upload : {0, 1}) {
#if defined(_WIN32)
        check(_putenv_s("WILDLLAMA_VIDEO_PROGRESS", direct_upload ? "1" : "") == 0);
#else
        check((direct_upload ? setenv("WILDLLAMA_VIDEO_PROGRESS", "1", 1) : unsetenv("WILDLLAMA_VIDEO_PROGRESS")) == 0);
#endif
        auto& budget = wildimage_ram::budget();
        check(budget.configure(32 << 20, {2 * wildimage_ram::GiB, 2 * wildimage_ram::GiB}, true));
        check(loader.load_tensors([&](const TensorStorage& src, ggml_tensor** dst) {
            *dst = src.name == "z.weight" ? direct : converted;
            return true;
        }, true, &names, false));
        staging_peak[direct_upload] = budget.peak();
        std::vector<float> received(source.size());
        ggml_backend_tensor_get(direct, received.data(), 0, ggml_nbytes(direct));
        check(source == received);
        received.resize(halves.size());
        ggml_backend_tensor_get(converted, received.data(), 0, ggml_nbytes(converted));
        check(std::all_of(received.begin(), received.end(), [](float v) {return v == 1.5f;}));
    }
    check(staging_peak[0] >= source.size() * sizeof(float) && staging_peak[1] < 16384);
    std::printf("Mapped upload managed host peak (bytes): %llu -> %llu\n",
                static_cast<unsigned long long>(staging_peak[0]), static_cast<unsigned long long>(staging_peak[1]));
    ggml_backend_buffer_free(storage);
    ggml_free(arena);
}

int main(int argc, char** argv) {
    const bool timing = argc > 1;
    grid(1, 1, 1, 1, 1, 1, 1, 0, false);
    grid(7, 15, 23, 2, 2, 3, 2, 39, false);
    grid(13, 96, 128, 1, 2, 2, 1, 0, timing);
    latents({17, 13, 48, 2}, false);
    latents({96, 128, 13, 16, 1}, timing);
    if (argc > 2) {
        ggml_backend_load_all_from_path(argv[2]);
        auto backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        check(backend && sd_backend_is(backend, "CUDA"));
        sd_ggml_register_extensions(backend);
        cuda_modulation(backend, 513, 73, false);
        cuda_modulation(backend, 5120, 2048, timing);
        mapped_upload(backend);
        ggml_backend_free(backend);
    }
    std::puts("All compared float values match byte for byte.");
}
