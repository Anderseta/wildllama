#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "model/vae/wan_vae.hpp"

// Standalone native regression using only the Wan 2.1 VAE. Supply its path,
// the frame count and either "compare" (whole clip vs blocks) or "long".
int main(int argc, char** argv) {
    if (argc != 4) return 2;
    const int frames = std::atoi(argv[2]);
    if (frames < 5 || frames > 241 || (frames - 1) % 4 != 0) return 2;
    const bool compare = std::strcmp(argv[3], "compare") == 0;
    if (!compare && std::strcmp(argv[3], "long") != 0) return 2;
    auto backend = sd_backend_cpu_init();
    if (!backend) return 3;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend_guard(backend, ggml_backend_free);
    auto manager = std::make_shared<ModelManager>();
    ModelLoader loader;
    if (!loader.init_from_file_and_convert_name(argv[1], "vae.") || !manager->set_loader(std::move(loader))) return 4;
    WAN::WanVAERunner vae(backend, manager->loader().get_tensor_storage_map(), "first_stage_model", false, VERSION_WAN2, manager);
    struct ReleaseVAE {
        WAN::WanVAERunner& vae;
        ModelManager& manager;
        ~ReleaseVAE() {
            vae.runner_end();
            manager.unregister_param_tensors(ModelComponent::VAE);
        }
    } release{vae, *manager};
    if (!manager->register_runner_params(ModelComponent::VAE, vae, ModelManager::ResidencyMode::ParamBackend, backend, backend) ||
        !manager->validate_registered_tensors()) return 5;
    sd::Tensor<float> input({32, 32, frames, 3, 1});
    for (int64_t i = 0; i < input.numel(); ++i) input[i] = static_cast<float>((i * 37 + i / 19) % 257) / 128.f - 1.f;
    const auto actual = vae._compute(6, input, false);
    if (actual.shape() != std::vector<int64_t>({4, 4, 1 + (frames - 1) / 4, 16, 1})) return 6;
    for (auto value : actual.values()) if (!std::isfinite(value)) return 7;
    double max_abs = 0;
    if (compare) {
        auto reference = vae.compute([&]() { return vae.build_graph(input, false); }, 6);
        if (!reference || reference->numel() != actual.numel()) return 8;
        for (int64_t i = 0; i < actual.numel(); ++i) {
            if (!std::isfinite((*reference)[i])) return 9;
            max_abs = std::max(max_abs, std::abs(double(actual[i]) - (*reference)[i]));
        }
        if (max_abs > 1e-4) { std::fprintf(stderr, "maximum latent difference: %g\n", max_abs); return 10; }
    }
    std::printf("PASS: frames=%d latent_frames=%lld compared=%d max_abs=%g\n", frames, (long long)actual.shape()[2], (int)compare, max_abs);
    return 0;
}
