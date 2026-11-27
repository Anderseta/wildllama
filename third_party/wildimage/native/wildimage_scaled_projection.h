#pragma once

#ifndef SD_USE_UPSTREAM_GGML
#include "ggml-ext.h"
#include <cstring>

namespace wildimage_projection {
inline bool matrix(const ggml_tensor* t) {
    return t && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) &&
           t->ne[0] > 0 && t->ne[1] > 0 && t->ne[2] == 1 && t->ne[3] == 1;
}

inline bool infer(ggml_tensor* const* src, int count, const void* attr, ggml_ext_tensor_desc* out) {
    if ((count != 2 && count != 4 && count != 5) || !matrix(src[0]) ||
        !matrix(src[1]) || !ggml_is_scalar(src[1])) return false;
    const auto width = src[0]->ne[0], rows = src[0]->ne[1];
    if (count == 2) {
        if (attr || width % 2) return false;
        *out = {GGML_TYPE_F32, {width / 2, rows, 1, 1}};
        return true;
    }
    int64_t prefix;
    if (!attr) return false;
    std::memcpy(&prefix, attr, sizeof(prefix));
    if (prefix < 0 || prefix >= rows || (prefix != 0) != (count == 5) ||
        !matrix(src[2]) || !ggml_are_same_shape(src[0], src[2])) return false;
    for (int i = 3; i < count; ++i) {
        if (!matrix(src[i]) || src[i]->ne[0] != width || src[i]->ne[1] != 1) return false;
    }
    *out = {GGML_TYPE_F32, {width, rows, 1, 1}};
    return true;
}

inline ggml_ext_op_t operation(bool residual) {
    static const auto gate = [] {
        const ggml_ext_op_desc desc{GGML_EXT_ABI_VERSION, "wildimage.scaled_swiglu.v1", 0, infer};
        return ggml_ext_register_op(&desc);
    }();
    static const auto add = [] {
        const ggml_ext_op_desc desc{GGML_EXT_ABI_VERSION, "wildimage.scaled_gated_residual.v1", sizeof(int64_t), infer};
        return ggml_ext_register_op(&desc);
    }();
    return residual ? add : gate;
}

// Only replace a final, out-of-place scalar multiply. Adapter additions,
// channelwise scales and other layouts keep the model's regular graph.
inline ggml_tensor* fuse(ggml_context* arena, ggml_backend_t device, ggml_tensor* scaled,
                         ggml_tensor* residual = nullptr, ggml_tensor* target_gate = nullptr,
                         ggml_tensor* prefix_gate = nullptr, int64_t prefix = 0) {
    if (!device || scaled->op != GGML_OP_MUL || scaled->view_src || !ggml_is_scalar(scaled->src[1])) return nullptr;
    ggml_tensor* inputs[] = {scaled->src[0], scaled->src[1], residual, target_gate, prefix_gate};
    const int count = residual ? (prefix ? 5 : 4) : 2;
    auto node = ggml_ext_op_create(arena, operation(residual != nullptr), inputs, count, residual ? &prefix : nullptr);
    return node && ggml_backend_supports_op(device, node) ? node : nullptr;
}
}
#endif
