#pragma once
#include "stable-diffusion.h"
#include <charconv>
#include <cstring>
#include <limits>
#include <string>

// Compact the decoded raster in place. SDImageOwner keeps the allocation and
// releases it normally. No full-sized crop buffer or intermediate PNG is needed.
inline bool wildimage_crop_reference(sd_image_t& image, const std::string& spec) {
    uint32_t values[4];
    const char* next = spec.data();
    const char* end = next + spec.size();
    for (int i = 0; i < 4; ++i) {
        auto result = std::from_chars(next, end, values[i]);
        if (result.ec != std::errc{} || (i == 3 ? result.ptr != end : result.ptr == end || *result.ptr != ',')) return false;
        next = result.ptr + (i != 3);
    }
    const auto [x, y, w, h] = values;
    if (!image.data || !image.channel || !w || !h || x >= image.width || y >= image.height ||
        w > image.width - x || h > image.height - y ||
        image.width > std::numeric_limits<size_t>::max() / image.height / image.channel) return false;
    const size_t stride = size_t(image.width) * image.channel;
    const size_t row = size_t(w) * image.channel;
    for (uint32_t i = 0; i < h; ++i) {
        std::memmove(image.data + i * row, image.data + (size_t(y) + i) * stride + size_t(x) * image.channel, row);
    }
    image.width = w;
    image.height = h;
    return true;
}
