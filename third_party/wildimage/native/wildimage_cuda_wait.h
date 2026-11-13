#pragma once

#include <cstdlib>
#include <cstring>
#include <cstdio>

// Used only by the NVIDIA extension. Scheduling changes CPU waits, not kernels,
// precision or GPU placement. Keep an override for latency-sensitive workloads.
inline void wildimage_configure_cuda_wait(ggml_backend_dev_t device) {
    const char* mode = std::getenv("WILDIMAGE_CUDA_WAIT");
    if (mode == nullptr || mode[0] == '\0') {
        return; // The host chooses a tested policy; embedded consumers stay unchanged.
    }
    unsigned int schedule;
    if (std::strcmp(mode, "blocking") == 0) {
        schedule = cudaDeviceScheduleBlockingSync;
    } else if (std::strcmp(mode, "spin") == 0) {
        schedule = cudaDeviceScheduleSpin;
    } else if (std::strcmp(mode, "yield") == 0) {
        schedule = cudaDeviceScheduleYield;
    } else if (std::strcmp(mode, "auto") == 0) {
        return;
    } else {
        std::fprintf(stderr, "wildimage: ignoring unknown CUDA wait policy: %s\n", mode);
        return;
    }
    const auto info = sd_cuda::cuda_device_info(device);
    int previous = -1;
    unsigned int flags = 0;
    if (info.device < 0 || cudaGetDevice(&previous) != cudaSuccess) {
        return;
    }
    auto status = cudaSetDevice(info.device);
    if (status == cudaSuccess) {
        status = cudaGetDeviceFlags(&flags);
    }
    if (status == cudaSuccess) {
        status = cudaSetDeviceFlags((flags & ~cudaDeviceScheduleMask) | schedule);
    }
    // Do not reset a context or fail generation if a driver rejects the policy.
    if (status != cudaSuccess) {
        std::fprintf(stderr, "wildimage: CUDA wait policy unchanged: %s\n", cudaGetErrorString(status));
        (void)cudaGetLastError();
    } else {
        std::fprintf(stderr, "wildimage: CUDA%d CPU wait policy: %s\n", info.device, mode);
    }
    if (previous != info.device) {
        (void)cudaSetDevice(previous);
    }
}
