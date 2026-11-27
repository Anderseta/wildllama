#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include "model/diffusion/qwen_image_2_1.hpp"

namespace {
using Shape = std::pair<int64_t, int64_t>;
const std::vector<int> axes = {16, 56, 56};

void check(bool ok) {
    if (!ok) throw std::runtime_error("Qwen Image 2.1 position regression");
}

// Independent reference in the original matrix layout. Slots occupy one
// encoder token per four reference latents; target positions follow the text.
Rope::Embedding reference(const std::vector<int32_t>& slots, const std::vector<Shape>& images, bool target_only) {
    Rope::Embedding e;
    int64_t position = 0;
    auto image = [&](Shape shape, bool emit) {
        const auto [h, w] = shape;
        if (emit) {
            e.positions.append_image(static_cast<int>(h), static_cast<int>(w));
            for (int64_t y = 0; y < h; ++y)
                for (int64_t x = 0; x < w; ++x)
                    e.ids.push_back({float(position), float(y - (h - h / 2)), float(x - (w - w / 2))});
        }
        position += std::max(h, w);
    };
    for (size_t i = 0; i < slots.size();) {
        int tag = slots[i++];
        if (tag) {
            while (i < slots.size() && slots[i] == tag) ++i;
            image(images.at(tag - 1), !target_only);
        } else {
            if (!target_only) {
                e.positions.append_tokens(1);
                e.ids.push_back(std::vector<float>(3, float(position)));
            }
            ++position;
        }
    }
    image(images.back(), true);
    e.values = Rope::embed_nd(e.ids, 1, 10000.f, axes, e.layout, &e.frequencies);
    return e;
}

void compare(const std::vector<int32_t>& slots, const std::vector<Shape>& shapes, bool implicit, bool timing = false) {
    sd::Tensor<int32_t> tags;
    if (!implicit) tags = sd::Tensor<int32_t>({int64_t(slots.size())}, slots);
    auto layout = Qwen::QwenImage21Layout::build(slots.size(), tags, shapes);
    for (bool target_only : {false, true}) {
        for (int circular = 0; circular < (timing ? 1 : 4); ++circular) {
            std::vector<double> times[2];
            for (int i = 0; i < (timing ? 9 : 1); ++i) {
                Rope::Embedding expected;
                sd::Tensor<float> actual;
                for (int k : {i % 2, 1 - i % 2}) {
                    const auto start = std::chrono::steady_clock::now();
                    if (k == 0) {
                        expected = reference(slots, shapes, target_only);
                        Rope::apply_circular(expected, circular & 1, circular & 2);
                    } else {
                        actual = layout.position_tensor(target_only, axes, 128, circular & 1, circular & 2);
                    }
                    times[k].push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
                }
                check(actual.values().size() == expected.values.size());
                check(std::memcmp(actual.data(), expected.values.data(), expected.values.size() * sizeof(float)) == 0);
            }
            if (timing) {
                for (auto& v : times) std::sort(v.begin(), v.end());
                std::printf("position matrix, target only=%d, median ms: %.3f -> %.3f\n", target_only, times[0][4], times[1][4]);
            }
        }
    }
}
}

int main() {
    compare({}, {{1, 1}}, true);
    compare({0, 0, 0}, {{3, 7}}, true);
    compare({0, 1, 1, 1, 0, 0, 2, 2, 2, 0}, {{3, 4}, {2, 6}, {7, 5}}, false);
    compare(std::vector<int32_t>(40), {{64, 64}}, true, true);
    auto key = [](Shape shape, size_t text, bool cached, bool circular_x) {
        return Qwen::QwenImage21Layout::build(text, {}, {shape}).position_key(cached, axes, 128, circular_x, false);
    };
    check(!(key({3, 4}, 4, true, false) == key({4, 3}, 4, true, false)));
    check(!(key({3, 4}, 4, true, false) == key({3, 4}, 5, true, false)));
    check(!(key({3, 4}, 4, true, false) == key({3, 4}, 4, false, false)));
    check(!(key({3, 4}, 4, true, false) == key({3, 4}, 4, true, true)));
    auto layout = Qwen::QwenImage21Layout::build(40, {}, {{64, 64}});
    auto& budget = wildimage_ram::budget();
    constexpr size_t bytes = 64 * 64 * 128 * 2 * sizeof(float);
    check(budget.configure(bytes, {2 * wildimage_ram::GiB, 2 * wildimage_ram::GiB}, true));
    {
        auto matrix = layout.position_tensor(true, axes, 128, false, false);
        check(matrix.values().size() * sizeof(float) == bytes && budget.peak() == bytes);
    }
    check(budget.used() == 0);
    check(budget.configure(bytes - 1, {2 * wildimage_ram::GiB, 2 * wildimage_ram::GiB}, true));
    bool refused = false;
    try {
        layout.position_tensor(true, axes, 128, false, false);
    } catch (const wildimage_ram::Exceeded&) {
        refused = true;
    }
    check(refused && budget.used() == 0);
    std::puts("Qwen Image 2.1 position regression: PASS");
}
