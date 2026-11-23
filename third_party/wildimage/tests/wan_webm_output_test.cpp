// Link against the native CLI media objects and its configured libwebp.
#include "media_io.h"
#include "webp/encode.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static void check(bool value) {
    if (!value) throw std::runtime_error("Wan WebM packet regression");
}

static void staged_video(bool enabled) {
#ifdef _WIN32
    check(_putenv_s("WILDLLAMA_VIDEO_PROGRESS", enabled ? "1" : "") == 0);
#else
    check((enabled ? setenv("WILDLLAMA_VIDEO_PROGRESS", "1", 1) : unsetenv("WILDLLAMA_VIDEO_PROGRESS")) == 0);
#endif
}

int main(int argc, char**) {
    const bool bench = argc > 1;
    const uint32_t w = bench ? 640 : 176, h = bench ? 480 : 112;
    const int count = bench ? 25 : 7, fps = 16, quality = 90;
    std::vector<std::vector<uint8_t>> inputs(count), packets(count);
    std::vector<sd_image_t> frames;
    for (int f = 0; f < count; ++f) {
        const uint32_t channels = bench ? 3 : f % 3 == 0 ? 1 : f % 3 == 1 ? 3 : 4;
        inputs[f].resize(size_t(w) * h * channels);
        std::vector<uint8_t> rgb(size_t(w) * h * 3);
        for (size_t p = 0; p < size_t(w) * h; ++p) {
            for (uint32_t c = 0; c < channels; ++c)
                inputs[f][p * channels + c] = static_cast<uint8_t>(p * (c + 1) + p / w * 3 + f * 23);
            for (uint32_t c = 0; c < 3; ++c)
                rgb[p * 3 + c] = inputs[f][p * channels + (channels == 1 ? 0 : c)];
        }
        frames.push_back({w, h, channels, inputs[f].data()});
        uint8_t* reference = nullptr;
        const size_t bytes = WebPEncodeRGB(rgb.data(), w, h, w * 3, quality, &reference);
        check(reference && bytes > 20 && !std::memcmp(reference + 12, "VP8 ", 4));
        size_t n = 0;
        for (unsigned i = 0; i < 4; ++i) n |= size_t(reference[16+i]) << (i*8);
        check(n && n <= bytes-20);
        packets[f].assign(reference+20, reference+20+n);
        WebPFree(reference);
    }
    std::vector<float> samples(8000 * count / fps, 0.125f);
    sd_audio_t audio{8000, 1, samples.size(), samples.data()};
    std::vector<double> times[2];
    for (int pass = 0; pass < (bench ? 3 : 1); ++pass) {
        for (int parallel : {pass % 2, 1 - pass % 2}) {
            staged_video(parallel != 0);
            const auto start = std::chrono::steady_clock::now();
            const auto clip = create_webm_from_sd_images_to_vector(frames.data(), count, fps, quality, &audio, "metadata");
            times[parallel].push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
            auto at = clip.begin();
            for (const auto& packet : packets) {
                at = std::search(at, clip.end(), packet.begin(), packet.end());
                check(at != clip.end());
                at += packet.size();
            }
        }
    }
    for (auto& v : times) std::sort(v.begin(), v.end());
    std::printf("Serial and bounded encode, %d frames (s): %.4f -> %.4f\n", count, times[0][times[0].size()/2], times[1][times[1].size()/2]);
    staged_video(true);
    --frames[1].width;
    check(create_webm_from_sd_images_to_vector(frames.data(), count, fps, quality).empty());
    ++frames[1].width;
    check(sd_set_memory_budget("0.001", true, "0"));
    check(!create_webm_from_sd_images_to_vector(frames.data(), count, fps, quality).empty());
    check(sd_get_memory_failure_code() == 0);
}
