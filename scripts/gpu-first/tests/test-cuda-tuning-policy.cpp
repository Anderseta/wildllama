#include "ggml-cuda/resident-tuning-policy.h"
#include "ggml-cuda/dense-reuse.h"

#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace tuning = wildllama_cuda_tuning;

static void require(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

static tuning::policy fixture() {
    tuning::policy p;
    p.entries[0] = {{GGML_TYPE_Q8_0, 256, 128, 1}, GGML_CUDA_TUNING_MMVQ, GGML_CUDA_TUNING_MMQ, 5, 1.2, 4096};
    p.count = 1;
    return p;
}

static void persistence() {
    const std::string identity(64, 'a');
    std::ostringstream encoded;
    require(tuning::serialize(encoded, identity, fixture()), "serialize failed");
    tuning::policy loaded;
    std::istringstream source(encoded.str());
    require(tuning::parse(source, identity, loaded), "policy round trip failed");
    require(loaded.lookup({GGML_TYPE_Q8_0, 256, 128, 1}, GGML_CUDA_TUNING_MMVQ, 4096) == GGML_CUDA_TUNING_MMQ,
        "validated policy was not consumed");
    require(loaded.lookup({GGML_TYPE_Q8_0, 512, 128, 1}, GGML_CUDA_TUNING_MMVQ, 4096) == GGML_CUDA_TUNING_DEFAULT,
        "different shape bypassed original dispatch");
    require(loaded.lookup({GGML_TYPE_Q8_0, 256, 128, 1}, GGML_CUDA_TUNING_MMQ, 4096) == GGML_CUDA_TUNING_DEFAULT,
        "different baseline bypassed original dispatch");
    require(loaded.hits == 1 && loaded.misses == 2, "lookup counters incorrect");
    require(loaded.lookup({GGML_TYPE_Q8_0, 256, 128, 1}, GGML_CUDA_TUNING_MMVQ, 4095) == GGML_CUDA_TUNING_DEFAULT,
        "policy grew workspace instead of retaining original dispatch");
    std::istringstream wrong_driver(encoded.str());
    require(!tuning::parse(wrong_driver, std::string(64, 'b'), loaded), "wrong fingerprint reused policies");
    require(loaded.count == 0, "failed load retained stale policy");
    std::istringstream oversized(std::string(tuning::max_file_bytes + 1, 'x'));
    require(!tuning::parse(oversized, identity, loaded), "oversized cache accepted");
    for (const auto & broken : {
        "wildllama-cuda-policy 2 " + identity + " bitwise-finite\n",
        "wildllama-cuda-policy 1 " + identity + " tolerant\n",
        encoded.str() + "garbage",
        encoded.str() + "8 256 128 1 1 2 2 1.2 4096\n",
        encoded.str() + "8 256 128 1 1 2 5 nan 4096\n",
        encoded.str() + "8 256 128 1 1 2 5 1.2 4096\n"}) {
        std::istringstream bad(broken);
        require(!tuning::parse(bad, identity, loaded), "invalid/duplicate cache accepted");
    }
    auto too_many = fixture();
    too_many.count = tuning::max_entries + 1;
    std::ostringstream ignored;
    require(!tuning::serialize(ignored, identity, too_many), "entry bound ignored");
    require(!tuning::fingerprint_valid("short") && !tuning::fingerprint_valid(std::string(64, 'G')),
        "non SHA-256 identity accepted");
}

static void validation() {
    float a[] = {1.f, -2.f, 0.f}, b[] = {1.f, -2.f, 0.f};
    require(tuning::finite_bitwise_equal(a, b, 3), "identical finite values rejected");
    b[1] = std::nextafter(b[1], 0.f);
    require(!tuning::finite_bitwise_equal(a, b, 3), "one ULP silently accepted");
    b[1] = a[1]; b[2] = -0.f;
    require(!tuning::finite_bitwise_equal(a, b, 3), "different zero sign silently accepted");
    a[0] = b[0] = std::numeric_limits<float>::infinity();
    require(!tuning::finite_bitwise_equal(a, b, 3), "identical infinities accepted");
    a[0] = b[0] = std::numeric_limits<float>::quiet_NaN();
    require(!tuning::finite_bitwise_equal(a, b, 3), "identical NaNs accepted");
    double mv[] = {1.3, 1.25, 1.2}, mq[] = {1., 1., 1.};
    double gain;
    require(tuning::repeatable_gain(mv, mq, 3, GGML_CUDA_TUNING_MMVQ, gain) && gain == 1.2,
        "repeatable improvement rejected");
    mq[2] = 1.19;
    require(!tuning::repeatable_gain(mv, mq, 3, GGML_CUDA_TUNING_MMVQ, gain), "noisy average was approved");
    mq[2] = 0;
    require(!tuning::repeatable_gain(mv, mq, 3, GGML_CUDA_TUNING_MMVQ, gain), "zero duration approved");
    mq[2] = std::numeric_limits<double>::infinity();
    require(!tuning::repeatable_gain(mv, mq, 3, GGML_CUDA_TUNING_MMVQ, gain), "infinite duration approved");
    require(!tuning::repeatable_gain(mv, mq, 2, GGML_CUDA_TUNING_MMVQ, gain), "too few pairs approved");
}

static void iq3_and_ffn_admission() {
    require(ggml_cuda_iq3_input_reuse_setting(nullptr) && ggml_cuda_iq3_input_reuse_setting("") &&
            ggml_cuda_iq3_input_reuse_setting("1"), "IQ3 automatic/explicit enable rejected");
    require(!ggml_cuda_iq3_input_reuse_setting("0") && !ggml_cuda_iq3_input_reuse_setting("false") &&
            !ggml_cuda_iq3_input_reuse_setting("invalid"), "IQ3 opt-out/malformed setting enabled reuse");
    for (const auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
            GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
            GGML_TYPE_IQ3_S}) {
        require(ggml_cuda_dense_q8_1_input(type), "validated Q8_1 activation format rejected");
        require(ggml_cuda_dense_reuse_type(type, false, true) == (type == GGML_TYPE_IQ3_S),
            "default IQ3 mode changed other formats");
        require(ggml_cuda_dense_reuse_type(type, true, false) == (type != GGML_TYPE_IQ3_S),
            "IQ3 opt-out changed legacy eligibility");
        require(!ggml_cuda_dense_reuse_type(type, false, false), "both opt-outs ignored");
    }
    for (const auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ4_NL}) {
        require(!ggml_cuda_dense_q8_1_input(type), "unvalidated activation format admitted");
    }
    require(tuning::supported_shape(256, 128, 1) && tuning::supported_shape(16384, 16384, 8),
        "original shape boundary regressed");
    require(tuning::supported_shape(5120, 17408, 1) && tuning::supported_shape(17408, 5120, 8),
        "17408-wide FFN rejected");
    for (const auto shape : {tuning::key{GGML_TYPE_IQ3_S, 16640, 5120, 1},
            {GGML_TYPE_IQ3_S, 5120, 17536, 1}, {GGML_TYPE_IQ3_S, 5120, 17408, 0},
            {GGML_TYPE_IQ3_S, 5120, 17408, 9}, {GGML_TYPE_IQ3_S, 5119, 17408, 1},
            {GGML_TYPE_IQ3_S, std::numeric_limits<int64_t>::max(), 17408, 1}}) {
        require(!tuning::supported_shape(shape.k, shape.m, shape.n), "unsupported shape admitted");
    }
    auto p = fixture();
    p.entries[0].shape = {GGML_TYPE_IQ3_S, 5120, 17408, 1};
    const std::string identity(64, 'a');
    std::ostringstream bytes;
    require(tuning::serialize(bytes, identity, p), "IQ3_S FFN policy serialization failed");
    tuning::policy loaded;
    std::istringstream input(bytes.str());
    require(tuning::parse(input, identity, loaded), "IQ3_S FFN policy round trip failed");
    require(loaded.lookup(p.entries[0].shape, GGML_CUDA_TUNING_MMVQ, 4095) == GGML_CUDA_TUNING_DEFAULT,
        "IQ3_S policy grew workspace");
    require(loaded.lookup(p.entries[0].shape, GGML_CUDA_TUNING_MMVQ, 4096) == GGML_CUDA_TUNING_MMQ,
        "IQ3_S policy with available workspace rejected");
}

int main() {
    try { persistence(); validation(); iq3_and_ffn_admission(); }
    catch (const std::exception & error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
    std::puts("PASS: bounded policy persistence, fingerprint invalidation, real lookup, bitwise/finite validation, repeatable gain");
    return 0;
}
