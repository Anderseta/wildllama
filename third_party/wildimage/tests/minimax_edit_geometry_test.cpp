// Link against the patched MiniMax image runner. No model weights are required.
#include "core/wildimage_edit_crop.h"
#include "model/diffusion/minimax_h3.hpp"
#include <chrono>
#include <cstdio>

static bool test_crop() {
    std::vector<uint8_t> pixels(13 * 9 * 3);
    for (size_t i = 0; i < pixels.size(); ++i) pixels[i] = static_cast<uint8_t>(i);
    const auto original = pixels;
    sd_image_t image{13, 9, 3, pixels.data()};
    for (const auto* bad : {"", "1,2,0,3", "4,4,10,1", "1,1,1,1,2", "-1,0,3,4"}) {
        if (wildimage_crop_reference(image, bad) || original != pixels) return false;
    }
    if (!wildimage_crop_reference(image, "4,3,9,5") || image.width != 9 || image.height != 5) return false;
    for (size_t y = 0; y < 5; ++y) {
        if (std::memcmp(pixels.data() + 9 * y * 3, original.data() + ((3 + y) * 13 + 4) * 3, 27)) return false;
    }
    return true;
}

int main() {
    if (!test_crop()) return 1;
    using namespace MiniMaxH3;
    std::vector<sd::Tensor<float>> refs{sd::Tensor<float>({64, 48, 1, 1}), sd::Tensor<float>({48, 32, 1, 1})};
    std::vector<MiniMaxH3ReferenceBlock> blocks{{MiniMaxH3ReferenceKind::IMAGE, 0, -1}, {MiniMaxH3ReferenceKind::IMAGE, 1, -1}};
    const sd::Tensor<int32_t> empty;
    auto layout = [&](int step, bool emit) {
        const float t = step / 6.f;
        return build_layout(192, 1, 128, 96, 0, refs, {}, empty, blocks, empty, t, 0, (step + 1) / 6.f, 0, emit);
    };
    const auto initial = layout(0, true);
    if (initial.positions.empty()) return 2;
    for (int s = 0; s < 6; ++s) {
        const auto full = layout(s, true);
        const auto reuse = layout(s, false);
        if (full.positions != initial.positions || !reuse.positions.empty() || full.timesteps != reuse.timesteps ||
            full.endpoints != reuse.endpoints || full.segments.size() != reuse.segments.size() ||
            full.sequence_segments.size() != reuse.sequence_segments.size() || (s && full.timesteps == initial.timesteps)) return 3;
        for (size_t i = 0; i < full.segments.size(); ++i) {
            const auto &a = full.segments[i], &b = reuse.segments[i];
            if (a.start != b.start || a.end != b.end || a.modulation_row != b.modulation_row) return 4;
        }
        for (size_t i = 0; i < full.sequence_segments.size(); ++i) {
            const auto &a = full.sequence_segments[i], &b = reuse.sequence_segments[i];
            if (a.start != b.start || a.end != b.end || a.kind != b.kind || a.source_index != b.source_index) return 5;
        }
    }
    std::swap(blocks[0], blocks[1]);
    if (layout(0, true).positions == initial.positions) return 6;
    using Clock = std::chrono::steady_clock;
    double elapsed[2]{};
    size_t rows = 0;
    for (int repetition = 0; repetition < 200; ++repetition) {
        for (int run = 0; run < 2; ++run) {
            int cached = (repetition + run) % 2;
            auto start = Clock::now();
            for (int s = 0; s < 6; ++s) rows += layout(s, !cached || s == 0).positions.size();
            elapsed[cached] += std::chrono::duration<double, std::micro>(Clock::now() - start).count();
        }
    }
    std::printf("PASS: six steps, two references; host layout us/six steps: rebuilt=%.2f reused=%.2f samples=%zu\n",
                elapsed[0] / 200, elapsed[1] / 200, rows);
    return 0;
}
