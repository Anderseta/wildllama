#ifndef WILDIMAGE_RAM_BUDGET_H
#define WILDIMAGE_RAM_BUDGET_H

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <cstring>
#include <cerrno>
#include <stdexcept>
#include <string>
#include <utility>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#pragma push_macro("WIN32_LEAN_AND_MEAN")
#pragma push_macro("small")
#pragma push_macro("near")
#pragma push_macro("far")
#pragma push_macro("interface")
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#pragma pop_macro("interface")
#pragma pop_macro("far")
#pragma pop_macro("near")
#pragma pop_macro("small")
#pragma pop_macro("WIN32_LEAN_AND_MEAN")
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

namespace wildimage_ram {
constexpr uint64_t GiB = uint64_t(1) << 30;
struct Memory { uint64_t total = 0, available = 0; };
inline uint64_t saturating_add(uint64_t a,uint64_t b) {return b>UINT64_MAX-a ? UINT64_MAX : a+b;}

inline Memory system_memory() {
#if defined(_WIN32)
    MEMORYSTATUSEX state{}; state.dwLength = sizeof(state);
    if (GlobalMemoryStatusEx(&state)) return {state.ullTotalPhys, state.ullAvailPhys};
#elif defined(__linux__)
    std::ifstream input("/proc/meminfo");
    std::string key, unit; uint64_t value = 0; Memory memory;
    while (input >> key >> value >> unit) {
        if (unit == "kB" && value <= UINT64_MAX / 1024) {
            if (key == "MemTotal:") memory.total = value * 1024;
            if (key == "MemAvailable:") memory.available = value * 1024;
        }
    }
    return memory;
#elif defined(__APPLE__)
    Memory memory; size_t size = sizeof(memory.total);
    sysctlbyname("hw.memsize", &memory.total, &size, nullptr, 0);
    const auto host = mach_host_self(); vm_size_t page = 0;
    vm_statistics64_data_t state{}; mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_page_size(host, &page) == KERN_SUCCESS &&
        host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&state), &count) == KERN_SUCCESS)
        memory.available = (uint64_t(state.free_count) + state.inactive_count) * page;
    mach_port_deallocate(mach_task_self(), host); return memory;
#endif
    return {};
}

inline uint64_t effective_limit(uint64_t requested, Memory memory) {
    const uint64_t reserve = std::max(GiB, memory.total / 20);
    const uint64_t available = std::min(memory.total, memory.available);
    const uint64_t automatic = available > reserve ? available - reserve : 0;
    return requested ? std::min(requested, automatic) : automatic;
}

inline bool parse_gib(const char* amount,uint64_t& bytes) {
    if(!amount || !*amount) return false;
    errno=0; char* end=nullptr; const double value=std::strtod(amount,&end);
    // 2^34 GiB is 2^64 bytes, which cannot be represented even where
    // long double has only double precision (MSVC).
    if(errno==ERANGE || end==amount || !end || *end || !std::isfinite(value) || value<0 || value>=17179869184.0) return false;
    const double scaled=value*GiB;
    if(value>0 && scaled<1) return false;
    bytes=static_cast<uint64_t>(scaled); return true;
}

class Exceeded : public std::runtime_error {
public: Exceeded() : std::runtime_error("Wildimage managed RAM budget exhausted") {}
};

class Budget {
    std::atomic<uint64_t> used_{0}, peak_{0};
    std::atomic<uint64_t> limit_{0};
    std::atomic<bool> configured_{false};
    std::mutex configure_mutex_;
public:
    std::atomic<bool> ram_failure{false}, gpu_failure{false};
    std::atomic<uint64_t> gpu_reserve_bytes{GiB/4};
    std::atomic<bool> stage_components{false};
    bool configure(uint64_t requested, Memory memory, bool stage,uint64_t gpu_reserve=GiB/4) {
        std::lock_guard<std::mutex> lock(configure_mutex_);
        if (used_.load()) return false;
        limit_ = effective_limit(requested, memory);
        configured_ = true; stage_components = stage;
        gpu_reserve_bytes=gpu_reserve;
        ram_failure = false; gpu_failure = false; peak_ = 0;
        return limit_.load() > 0;
    }
    uint64_t limit() {
        if (!configured_.load()) configure(0, system_memory(), false);
        return limit_.load();
    }
    uint64_t used() const { return used_.load(); }
    uint64_t peak() const { return peak_.load(); }
    uint64_t remaining() { const auto cap=limit(), used=used_.load(); return used < cap ? cap-used : 0; }
    bool try_acquire(uint64_t bytes) {
        if (!bytes) return true;
        const auto cap = limit(); auto used = used_.load();
        for (;;) {
            if (used > cap || bytes > cap-used) return false;
            if (used_.compare_exchange_weak(used, used+bytes)) break;
        }
        auto peak=peak_.load(); while (peak < used+bytes && !peak_.compare_exchange_weak(peak,used+bytes)) {}
        return true;
    }
    void release(uint64_t bytes) { if (bytes) used_.fetch_sub(bytes); }
    void fail() {
        ram_failure = true;
        std::fprintf(stderr,"WILDIMAGE_RAM_BUDGET_EXHAUSTED: managed %.2f MiB, limit %.2f MiB\n", used()/1048576.,limit()/1048576.);
    }
    int failure_code() const { return ram_failure.load() ? 76 : gpu_failure.load() ? 75 : 0; }
};
inline Budget& budget() { static Budget instance; return instance; }

class Reservation {
    Budget* owner_ = nullptr; uint64_t bytes_ = 0;
public:
    Reservation() = default;
    explicit Reservation(uint64_t bytes) { reset(bytes); }
    Reservation(const Reservation& other) { reset(other.bytes_, other.owner_); }
    Reservation(Reservation&& other) noexcept { swap(other); }
    Reservation& operator=(Reservation other) { swap(other); return *this; }
    ~Reservation() { if (owner_) owner_->release(bytes_); }
    void swap(Reservation& other) noexcept { std::swap(owner_,other.owner_); std::swap(bytes_,other.bytes_); }
    bool try_reset(uint64_t bytes, Budget* owner=nullptr) {
        if (!owner) owner=&budget();
        if (owner_ == owner && bytes <= bytes_) { owner->release(bytes_-bytes); bytes_=bytes; return true; }
        const auto added = owner_ == owner ? bytes-bytes_ : bytes;
        if (!owner->try_acquire(added)) return false;
        if (owner_ && owner_ != owner) owner_->release(bytes_);
        owner_=owner; bytes_=bytes; return true;
    }
    void reset(uint64_t bytes=0, Budget* owner=nullptr) {
        if (!try_reset(bytes, owner)) { (owner ? owner : &budget())->fail(); throw Exceeded(); }
    }
    uint64_t bytes() const { return bytes_; }
};

inline uint64_t tensor_bytes(uint64_t elements, uint64_t size) {
    if (size && elements > UINT64_MAX/size) { budget().fail(); throw Exceeded(); }
    return elements*size;
}
inline void gpu_failure() { budget().gpu_failure=true; }
inline int classify_exit(int status,int memory_code) { return status && (memory_code==75 || memory_code==76) ? memory_code : status; }

template<class T> struct Allocator {
    using value_type=T;
    Allocator()=default;
    template<class U> Allocator(const Allocator<U>&) noexcept {}
    T* allocate(size_t n) {
        const auto bytes=tensor_bytes(n,sizeof(T));
        if(!budget().try_acquire(bytes)) throw Exceeded();
        try { return static_cast<T*>(::operator new(static_cast<size_t>(bytes))); }
        catch(...) { budget().release(bytes); throw; }
    }
    void deallocate(T* ptr,size_t n) noexcept { ::operator delete(ptr); budget().release(uint64_t(n)*sizeof(T)); }
    template<class U> bool operator==(const Allocator<U>&) const noexcept {return true;}
    template<class U> bool operator!=(const Allocator<U>&) const noexcept {return false;}
};

struct RawRegistry { std::mutex mutex; std::map<void*,size_t> sizes; };
inline RawRegistry& raw_registry() { static RawRegistry registry; return registry; }
inline void* raw_malloc(size_t bytes) noexcept {
    if(!budget().try_acquire(bytes)) { budget().fail(); return nullptr; }
    void* ptr=std::malloc(bytes);
    if(!ptr) { budget().release(bytes); budget().fail(); return nullptr; }
    try { auto& registry=raw_registry(); std::lock_guard<std::mutex> lock(registry.mutex); registry.sizes.emplace(ptr,bytes); }
    catch(...) { std::free(ptr); budget().release(bytes); return nullptr; }
    return ptr;
}
inline void raw_free(void* ptr) noexcept {
    if(!ptr) return;
    size_t bytes=0;
    { auto& registry=raw_registry(); std::lock_guard<std::mutex> lock(registry.mutex);
      auto found=registry.sizes.find(ptr); if(found!=registry.sizes.end()) {bytes=found->second; registry.sizes.erase(found);} }
    std::free(ptr); budget().release(bytes);
}
inline void* raw_realloc(void* ptr,size_t bytes) noexcept {
    if(!ptr) return raw_malloc(bytes);
    if(!bytes) {raw_free(ptr); return nullptr;}
    size_t old=0;
    { auto& registry=raw_registry(); std::lock_guard<std::mutex> lock(registry.mutex);
      auto found=registry.sizes.find(ptr); if(found==registry.sizes.end()) return nullptr; old=found->second; }
    auto* next=raw_malloc(bytes); if(!next) return nullptr;
    std::memcpy(next,ptr,std::min(old,bytes)); raw_free(ptr); return next;
}
}
#endif
