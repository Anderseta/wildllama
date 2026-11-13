#include <cstdio>
#include <cstdlib>

using ggml_backend_dev_t = void*;
using cudaError_t = int;
constexpr int cudaSuccess = 0;
constexpr unsigned cudaDeviceScheduleSpin = 1, cudaDeviceScheduleYield = 2;
constexpr unsigned cudaDeviceScheduleBlockingSync = 4, cudaDeviceScheduleMask = 7;
static int current = 0, selected = 1, writes = 0, fail = 0;
static unsigned flags = 0x18;
namespace sd_cuda {
struct Info { int device; };
Info cuda_device_info(ggml_backend_dev_t) { return {selected}; }
}
int cudaGetDevice(int* value) { *value = current; return 0; }
int cudaSetDevice(int value) { current = value; return 0; }
int cudaGetDeviceFlags(unsigned* value) { *value = flags; return 0; }
int cudaSetDeviceFlags(unsigned value) { ++writes; if (!fail) flags = value; return fail; }
int cudaGetLastError() { return 0; }
const char* cudaGetErrorString(int) { return "simulated unsupported policy"; }
#include "wildimage_cuda_wait.h"

static void policy(const char* mode) {
#ifdef _WIN32
    _putenv_s("WILDIMAGE_CUDA_WAIT", mode);
#else
    setenv("WILDIMAGE_CUDA_WAIT", mode, 1);
#endif
    wildimage_configure_cuda_wait(nullptr);
}
#define CHECK(condition) do { if (!(condition)) { std::fprintf(stderr, "failed: %s\n", #condition); return 1; } } while (0)
int main() {
    policy(""); policy("auto"); policy("invalid");
    CHECK(writes == 0 && current == 0 && flags == 0x18);
    policy("blocking");
    CHECK(writes == 1 && current == 0 && flags == 0x1c);
    policy("spin");
    CHECK(writes == 2 && current == 0 && flags == 0x19);
    policy("yield");
    CHECK(writes == 3 && current == 0 && flags == 0x1a);
    fail = 1;
    policy("blocking");
    CHECK(writes == 4 && current == 0 && flags == 0x1a);
    selected = -1;
    policy("blocking");
    CHECK(writes == 4 && current == 0 && flags == 0x1a);
    std::puts("CUDA wait policy: PASS (flags, device restoration, fallback)");
}
