#pragma once

#ifndef SD_USE_UPSTREAM_GGML
#include "ggml-ext.h"
#include <cstdlib>
#include <cstring>
#include <cstdint>

// Explicit graph storage, not a data-pointer cache. Every graph execution
// recomputes the activation; sibling projections share its packed output.
namespace wildimage_fp8 {
struct Shape { int64_t k, n; };
inline bool pack_shape(ggml_tensor* const* src, int count, const void*, ggml_ext_tensor_desc* out) {
    if (count != 1 || !src[0] || src[0]->type != GGML_TYPE_F32 || !ggml_is_contiguous(src[0]) ||
        src[0]->ne[0] <= 0 || src[0]->ne[0] % 16 || src[0]->ne[1] <= 0 ||
        src[0]->ne[2] != 1 || src[0]->ne[3] != 1 || src[0]->ne[0] > (INT64_MAX - 32) / src[0]->ne[1]) return false;
    *out = {GGML_TYPE_I8, {src[0]->ne[0] * src[0]->ne[1] + 32, 1, 1, 1}};
    return true;
}
inline ggml_ext_op_t pack_op() {
    static const auto op = [] {
        const ggml_ext_op_desc desc{GGML_EXT_ABI_VERSION, "wildimage.fp8.pack.v1", 0, pack_shape};
        return ggml_ext_register_op(&desc);
    }();
    return op;
}
inline bool matmul_shape(ggml_tensor* const* src, int count, const void* attrs, ggml_ext_tensor_desc* out) {
    if (count != 2 || !attrs || !src[0] || !src[1]) return false;
    Shape s; std::memcpy(&s, attrs, sizeof(s));
    const auto* w = src[0]; const auto* packed = src[1];
    if (w->type != GGML_TYPE_F8_E4M3 || !ggml_is_contiguous(w) || w->ne[0] != s.k || s.k <= 0 || s.k % 16 ||
        w->ne[1] <= 0 || w->ne[1] % 16 || w->ne[2] != 1 || w->ne[3] != 1 || s.n <= 0 ||
        s.k > (INT64_MAX - 32) / s.n || packed->type != GGML_TYPE_I8 || !ggml_is_contiguous(packed) ||
        packed->ne[0] != s.k * s.n + 32 || packed->ne[1] != 1 || packed->ne[2] != 1 || packed->ne[3] != 1) return false;
    *out = {GGML_TYPE_F32, {w->ne[1], s.n, 1, 1}};
    return true;
}
inline ggml_ext_op_t matmul_op() {
    static const auto op = [] {
        const ggml_ext_op_desc desc{GGML_EXT_ABI_VERSION, "wildimage.fp8.matmul.v1", sizeof(Shape), matmul_shape};
        return ggml_ext_register_op(&desc);
    }();
    return op;
}
template<class Context>
inline ggml_tensor* linear(Context* ctx, ggml_tensor* x, ggml_tensor* w, bool force_f32, float scale) {
    static const bool disabled = [] { const char* v = std::getenv("WILDIMAGE_DISABLE_FP8_PACK"); return v && std::atoi(v) != 0; }();
    if (disabled || force_f32 || scale != 1.f || w->type != GGML_TYPE_F8_E4M3 || !ctx->backend) return nullptr;
    ggml_tensor* input[] = {x}; ggml_ext_tensor_desc shape;
    if (!pack_shape(input, 1, nullptr, &shape)) return nullptr;
    auto found = ctx->fp8_pack_cache.find(x);
    auto packed = found == ctx->fp8_pack_cache.end() ? ggml_ext_op_create(ctx->ggml_ctx, pack_op(), input, 1, nullptr) : found->second;
    ggml_tensor* inputs[] = {w, packed}; const Shape dims{x->ne[0], x->ne[1]};
    if (!matmul_shape(inputs, 2, &dims, &shape)) return nullptr;
    auto result = ggml_ext_op_create(ctx->ggml_ctx, matmul_op(), inputs, 2, &dims);
    if (!ggml_backend_supports_op(ctx->backend, packed) || !ggml_backend_supports_op(ctx->backend, result)) return nullptr;
    ctx->fp8_pack_cache.emplace(x, packed);
    return result;
}
}
#endif
