#include "model/vae/vae_tiling.hpp"
#include "runtime/tiling.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

// This fixture runs the host-side assembly without a GPU or model weights.
void log_printf(sd_log_level_t, const char*, int, const char*, ...) {}

void pretty_progress(int, int, float) {}

static void check_spatial(bool wrap_x, bool wrap_y, bool single) {
    sd::Tensor<float> img({11, 9, 3, 2});
    for (int64_t i = 0; i < img.numel(); ++i) {
        img[i] = static_cast<float>(i);
    }
    int count = 0;
    auto identity = [&](const sd::Tensor<float>& tile) {
        ++count;
        if (single) {
            assert(tile.data() == img.data());
        }
        const int offset = static_cast<int>(tile[0]);
        const int w = static_cast<int>(tile.shape()[0]);
        const int h = static_cast<int>(tile.shape()[1]);
        for (int g = 0; g < 6; ++g) {
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const int origin = g * 99 + ((y + offset / 11) % 9) * 11 + (x + offset % 11) % 11;
                    assert(tile[(g * h + y) * w + x] == img[origin]);
                }
            }
        }
        return tile;
    };
    auto out = process_tiles_2d(img, 11, 9, 1, single ? 32 : 7, single ? 32 : 6, 0.25f,
                                wrap_x, wrap_y, identity, true);
    assert(single ? count == 1 : count > 1);
    assert(out.shape() == img.shape());
    for (int64_t i = 0; i < out.numel(); ++i) {
        assert(std::abs(out[i] - img[i]) < 0.001f);
    }
}

static uint64_t assembly_peak(bool direct) {
    auto& budget = wildimage_ram::budget();
    assert(budget.configure(32ULL << 20, {2 * wildimage_ram::GiB, 2 * wildimage_ram::GiB}, true));
    sd::Tensor<float> input({32, 16, 21, 3, 1});
    const auto tiles = make_vae_temporal_tile_plan(21, {4, 0, 1});
    auto decoded = process_vae_temporal_tiles(input, tiles, [](const auto& tile, const auto& span) {
        auto shape = tile.shape();
        shape[2] = 4 * shape[2] - (span.first ? 3 : 0);
        return sd::Tensor<float>(std::move(shape));
    }, direct ? 81 : 0);
    assert(decoded.shape()[2] == 81);
    return budget.peak();
}

static void check_wan_decode_plan() {
    constexpr uint64_t gib = uint64_t(1) << 30;
    sd_tiling_params_t defaults = {true, true, 0, 0, .5f, 0, 0, nullptr};
    auto params = defaults;
    assert(!sd_plan_wan_decode_tiles(params, 128, 96, 8, 5 * gib, 4 * gib));
    assert(!sd_plan_wan_decode_tiles(params, 128, 96, 8, 12 * gib, gib - 1));
    assert(!sd_plan_wan_decode_tiles(params, 32, 24, 8, 12 * gib, 4 * gib));
    assert(params.tile_size_w == 0 && params.extra_tiling_args == nullptr);
    assert(sd_plan_wan_decode_tiles(params, 128, 96, 8, 12 * gib, 4 * gib));
    assert(params.tile_size_w == 256 && params.tile_size_h == 512);
    assert(params.target_overlap == .5f && params.extra_tiling_args == nullptr);
    assert(!sd_plan_wan_decode_tiles(params, 96, 128, 8, 12 * gib, 4 * gib));
    params = defaults;
    assert(sd_plan_wan_decode_tiles(params, 96, 128, 8, 8 * gib, 4 * gib));
    assert(params.tile_size_w == 512 && params.tile_size_h == 256);
    assert(std::strcmp(params.extra_tiling_args, "temporal_tile_frames=2") == 0);
    params = defaults;
    params.extra_tiling_args = "temporal_tile_frames=1";
    assert(!sd_plan_wan_decode_tiles(params, 96, 128, 8, 12 * gib, 4 * gib));
}

static void check_sparse_reference(int frames) {
    using Tensor = sd::Tensor<float>;
    Tensor first({11, 9, 1, 3, 1});
    for (int64_t i = 0; i < first.numel(); ++i) first[i] = float(i - 97) / 128.f;
    first[0] = -0.0f;
    Tensor dense({11, 9, frames, 3, 1});
    sd::ops::slice_assign(&dense, 2, 0, 1, first);
    const auto plan = make_vae_temporal_tile_plan(frames, {32, 0, 1});
    auto compare = [&](const Tensor& source, bool sparse) {
        return process_tiles_2d(source, 11, 9, 1, 7, 6, .25f, true, true, [&](const Tensor& tile) {
            if (sparse) {
                return assemble_vae_temporal_tiles(plan, [&](const VAETemporalTile& span) {
                    return wan_reference_window(tile, span.start, span.end);
                }, frames);
            }
            return tile;
        }, true);
    };
    const auto actual = compare(first, true);
    const auto expected = compare(dense, false);
    assert(actual.shape() == expected.shape());
    assert(std::memcmp(actual.data(), expected.data(), actual.numel() * sizeof(float)) == 0);
}

static void check_streamed_spatial_decode() {
    using Tensor = sd::Tensor<float>;
    Tensor input({19, 15, 9, 3, 2});
    for (int64_t i = 0; i < input.numel(); ++i) input[i] = float(i % 131 - 64) / 31.f;
    const auto plan = make_vae_temporal_tile_plan(9, {4, 0, 1});
    auto decode = [](const Tensor& tile, const VAETemporalTile& span) {
        auto shape = tile.shape();
        shape[0] *= 2;
        shape[1] *= 2;
        shape[2] = shape[2] * 4 - (span.first ? 3 : 0);
        Tensor out(shape);
        const auto src_plane = tile.shape()[0] * tile.shape()[1];
        const auto dst_plane = shape[0] * shape[1];
        for (int64_t i = 0; i < out.numel(); ++i) {
            const auto pix = i % dst_plane;
            const auto plane = i / dst_plane;
            const auto f = (plane % shape[2] + (span.first ? 3 : 0)) / 4;
            out[i] = tile[((plane / shape[2]) * tile.shape()[2] + f) * src_plane +
                          (pix / shape[0] / 2) * tile.shape()[0] + (pix % shape[0]) / 2];
        }
        return out;
    };
    auto dense = [&](const Tensor& tile) { return process_vae_temporal_tiles(tile, plan, decode, 33); };
    auto run = [&](const TileChunkProducer& producer) {
        return process_tiles_2d(input, 38, 30, 2, 9, 8, .25f, true, true, dense, true, producer);
    };
    auto stream = [&](const Tensor& tile, const TileChunkSink& sink) {
        int64_t at = 0;
        for (const auto& span : plan.tiles) {
            auto out = decode(sd::ops::slice(tile, 2, span.start, span.end), span);
            if (!sink(out, at, 33)) return false;
            at += out.shape()[2];
        }
        return true;
    };
    const auto expected = run({});
    const auto actual = run(stream);
    assert(!actual.empty() && actual.shape() == expected.shape());
    assert(std::memcmp(expected.data(), actual.data(), actual.numel() * sizeof(float)) == 0);
    assert(run([&](const Tensor& tile, const TileChunkSink& sink) {
        return stream(tile, [&](const Tensor& out, int64_t at, int64_t total) {
            return at == 0 && sink(out, at, total);
        });
    }).empty());
    assert(run([](const Tensor&, const TileChunkSink&) { return true; }).empty());
    assert(run([](const Tensor& tile, const TileChunkSink& sink) {
        return sink(Tensor({tile.shape()[0] * 2, tile.shape()[1] * 2}), 0, 1);
    }).empty());
}

int main() {
    check_wan_decode_plan();
    ggml_time_init();
    check_streamed_spatial_decode();
    check_sparse_reference(5);
    check_sparse_reference(33);
    check_sparse_reference(109);
    const auto previous_peak = assembly_peak(false);
    const auto direct_peak = assembly_peak(true);
    assert(direct_peak < previous_peak);
    std::printf("Temporal assembly (bytes): previous %llu, direct %llu\n",
                static_cast<unsigned long long>(previous_peak), static_cast<unsigned long long>(direct_peak));
    check_spatial(false, false, true);
    check_spatial(false, false, false);
    check_spatial(true, false, false);
    check_spatial(true, true, false);
    using Tensor = sd::Tensor<float>;
    auto& budget = wildimage_ram::budget();
    assert(budget.configure(1 << 20, {2 * wildimage_ram::GiB, 2 * wildimage_ram::GiB}, true));
    Tensor input({3, 2, 9, 3, 2});
    for (int64_t i = 0; i < input.numel(); ++i) {
        input[i] = static_cast<float>(i);
    }
    auto decode = [](const Tensor& in, const VAETemporalTile& tile) {
        auto shape = in.shape();
        shape[2] = in.shape()[2] * 4 - (tile.first ? 3 : 0);
        Tensor out(shape);
        for (int64_t group = 0; group < 6; ++group) {
            for (int64_t frame = 0; frame < shape[2]; ++frame) {
                const int64_t source_frame = (frame + (tile.first ? 3 : 0)) / 4;
                for (int64_t px = 0; px < 6; ++px) {
                    out[(group * shape[2] + frame) * 6 + px] = in[(group * in.shape()[2] + source_frame) * 6 + px];
                }
            }
        }
        return out;
    };
    const int64_t frames = 33;
    const auto multiple = make_vae_temporal_tile_plan(9, {4, 0, 1});
    const auto expected = decode(input, {0, 0, 9, 0, true, true});
    const auto result = process_vae_temporal_tiles(input, multiple, decode, frames);
    assert(expected.shape() == result.shape());
    assert(std::memcmp(expected.data(), result.data(), result.numel() * sizeof(float)) == 0);
    assert(process_vae_temporal_tiles(input, multiple, decode, frames - 1).empty());
    assert(process_vae_temporal_tiles(input, multiple, decode, frames + 1).empty());
    auto wrong_shape = [&](const Tensor& in, const VAETemporalTile& tile) {
        auto out = decode(in, tile);
        if (!tile.first) {
            auto shape = out.shape();
            std::swap(shape[3], shape[4]);
            out.reshape_(shape);
        }
        return out;
    };
    assert(process_vae_temporal_tiles(input, multiple, wrong_shape, frames).empty());
    auto cancel = [&](const Tensor& in, const VAETemporalTile& tile) {
        return tile.first ? decode(in, tile) : Tensor{};
    };
    assert(process_vae_temporal_tiles(input, multiple, cancel, frames).empty());
    const auto single = make_vae_temporal_tile_plan(9, {9, 0, 1});
    auto check_borrow = [&](const Tensor& in, const VAETemporalTile& tile) {
        assert(in.data() == input.data());
        return decode(in, tile);
    };
    assert(process_vae_temporal_tiles(input, single, check_borrow, frames).values() == expected.values());
}
