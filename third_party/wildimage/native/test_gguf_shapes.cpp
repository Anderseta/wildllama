#include "model_io/gguf_io.h"
#include "model_io/gguf_reader_ext.h"
#include <cstdarg>
#include <cstdio>
#include <stdexcept>

void log_printf(sd_log_level_t, const char*, int, const char*, ...) {}
static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template<class T> static void write(std::ofstream& f, T v) { f.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
static void text(std::ofstream& f, const std::string& s) { write<uint64_t>(f, s.size()); f.write(s.data(), s.size()); }
static void fixture(const std::string& path, const std::vector<int64_t>& original, int64_t physical = 24, bool truncated = false, bool wide = false) {
    std::ofstream f(path, std::ios::binary);
    f.write("GGUF", 4); write<uint32_t>(f, 3); write<uint64_t>(f, 1); write<uint64_t>(f, original.empty() ? 0 : 1);
    if (!original.empty()) {
        text(f, "comfy.gguf.orig_shape.visual.patch_embed.proj.weight");
        write<uint32_t>(f, 9); write<uint32_t>(f, 11); write<uint64_t>(f, original.size());
        for (auto dim : original) write<int64_t>(f, dim);
    }
    text(f, "visual.patch_embed.proj.weight"); write<uint32_t>(f, wide ? 5 : 1);
    write<int64_t>(f, physical);
    if (wide) for (int i = 0; i < 4; ++i) write<int64_t>(f, 1);
    write<uint32_t>(f, GGML_TYPE_F32); write<uint64_t>(f, 0);
    while (static_cast<size_t>(f.tellp()) % 32) f.put(0);
    for (int i = 0; i < (truncated ? 1 : physical); ++i) write<float>(f, float(i));
}
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const auto path = (std::filesystem::path(argv[1]) / "shape-fixture.gguf").string();
    try {
        for (bool wide : {false, true}) {
            fixture(path, {3, 8}, 24, false, wide);
            std::vector<TensorStorage> tensors;
            std::string error;
            require(read_gguf_file(path, tensors, &error), error.c_str());
            require(tensors.size() == 1 && tensors[0].ne[0] == 8 && tensors[0].ne[1] == 3 && tensors[0].nbytes() == 96, "flattened shape");
        }
        fixture(path, {2, 3, 1, 2, 2});
        GGUFReader reader;
        require(reader.load(path), "5D vision shape");
        require(reader.tensors()[0].shape == std::vector<int64_t>({2,2,1,6}), "vision convolution layout");
        for (auto shape : {std::vector<int64_t>{3,9}, {0,24}, {-1,24}, {INT64_MAX,2}, {1,1,1,1,1,24}}) {
            fixture(path, shape);
            std::vector<TensorStorage> tensors;
            require(!read_gguf_file(path, tensors), "invalid original shape accepted");
        }
        fixture(path, {3,8}, 24, true);
        std::vector<TensorStorage> tensors;
        require(!read_gguf_file(path, tensors), "truncated tensor accepted");
        fixture(path, {});
        require(read_gguf_file(path, tensors) && tensors[0].ne[0] == 24, "legacy GGUF changed");
        std::filesystem::remove(path);
        std::puts("PASS: original shapes, 5D vision, fallback, invalid metadata, truncation and legacy GGUF");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        std::filesystem::remove(path);
        return 1;
    }
}
