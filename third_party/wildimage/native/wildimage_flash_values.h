#pragma once

#ifndef SD_USE_UPSTREAM_GGML
#include "ggml-ext.h"
#include <cstring>

namespace wildimage_flash_values {
// The persisted prefix is a flattened [channel, head, token] FP16 array.
// The target has the same order in FP32. Flash Attention reads [channel, token, head].
inline bool shape(ggml_tensor* const* src, int count, const void*, ggml_ext_tensor_desc* out) {
    if (count != 2 || !src[0] || !src[1]) return false;
    const auto* p = src[0];
    const auto* t = src[1];
    const int64_t heads = t->ne[1], tokens = t->ne[2];
    if (p->type != GGML_TYPE_F16 || t->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(p) || !ggml_is_contiguous(t) ||
        t->ne[0] != 128 || heads < 1 || heads > 65535 || tokens < 1 || t->ne[3] != 1 ||
        p->ne[0] != 128 * heads || p->ne[1] < 1 || p->ne[2] != 1 || p->ne[3] != 1 ||
        tokens > INT32_MAX || p->ne[1] > INT32_MAX - tokens) return false;
    *out = {GGML_TYPE_F16, {128, p->ne[1] + tokens, heads, 1}};
    return true;
}

inline ggml_ext_op_t operation() {
    static const auto op = [] {
        const ggml_ext_op_desc desc{GGML_EXT_ABI_VERSION, "wildimage.flash_values.v1", 0, shape};
        return ggml_ext_register_op(&desc);
    }();
    return op;
}

inline ggml_tensor* pack(ggml_context* arena, ggml_backend_t device, ggml_tensor* prefix, ggml_tensor* target) {
    ggml_tensor* inputs[] = {prefix, target};
    auto node = ggml_ext_op_create(arena, operation(), inputs, 2, nullptr);
    return node && ggml_backend_supports_op(device, node) ? node : nullptr;
}
}
#endif
