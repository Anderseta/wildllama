#include <cstdio>
#include <vector>
#include "ggml-backend.h"
#include "stable-diffusion.h"

// Explicit large-model test; supply weights and the matching runtime directory.
// It runs twice in one context to verify that phase eviction remains reloadable.
int main(int argc, char** argv) {
    sd_set_log_callback([](sd_log_level_t, const char* text, void*) {
        std::fputs(text, stderr);
        std::fflush(stderr);
    },
                        nullptr);
    if (argc != 5) {
        std::fprintf(stderr, "Usage: test-image-staging DIFFUSION ENCODER VAE RUNTIME_DIRECTORY\n");
        return 2;
    }
    ggml_backend_load_all_from_path(argv[4]);
    if (!sd_set_memory_budget("0", true, "0.25")) {
        std::fputs("FAIL: cannot configure memory budget\n", stderr);
        return 1;
    }
    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    params.diffusion_model_path      = argv[1];
    params.llm_path                  = argv[2];
    params.vae_path                  = argv[3];
    params.backend                   = "cuda0";
    params.params_backend            = "cuda0,te=disk,vae=disk";
    params.max_vram                  = "12.8";
    params.auto_fit                  = false;
    params.disable_segmented_compute = true;
    params.flash_attn_mode           = "on";
    auto ctx                         = new_sd_ctx(&params);
    if (!ctx) {
        std::fputs("FAIL: cannot create image context\n", stderr);
        return 1;
    }
    sd_img_gen_params_t image;
    sd_img_gen_params_init(&image);
    image.prompt          = "A red ceramic teapot on a wooden table, soft natural window light.";
    image.negative_prompt = "blurred";
    image.width = image.height           = 512;
    image.seed                           = 41723;
    image.sample_params.sample_steps     = 3;
    image.sample_params.guidance.txt_cfg = 3.f;
    image.sample_params.sample_method    = EULER_SAMPLE_METHOD;
    image.sample_params.scheduler        = SIMPLE_SCHEDULER;
    std::vector<unsigned char> expected;
    bool ok = true;
    for (int i = 0; i < 2 && ok; ++i) {
        sd_image_generation_result_t* result = nullptr;
        ok                                   = generate_image(ctx, &image, &result) && result && result->count == 1;
        if (ok) {
            auto& output = result->items[0].image;
            std::vector<unsigned char> actual(output.data, output.data + size_t(output.width) * output.height * output.channel);
            if (i == 0)
                expected = actual;
            else
                ok = actual == expected;
        }
        free_sd_image_generation_result(result);
    }
    free_sd_ctx(ctx);
    std::puts(ok ? "PASS: phase eviction, same-context weight reload and identical decoded pixels" : "FAIL: phase eviction/reload");
    return ok ? 0 : 1;
}
