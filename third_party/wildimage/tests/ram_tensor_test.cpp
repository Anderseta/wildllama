#include "core/tensor.hpp"
#include <cassert>
#include <cstdlib>
#include <new>

static size_t watched_size=0, minimum_charge=0, watched_calls=0, fail_size=0;
void* operator new(size_t size) {
    if(size==watched_size) {assert(wildimage_ram::budget().used()>=minimum_charge); ++watched_calls;}
    if(size==fail_size && fail_size) {fail_size=0; throw std::bad_alloc();}
    if(auto* ptr=std::malloc(size)) return ptr;
    throw std::bad_alloc();
}
void operator delete(void* ptr) noexcept {std::free(ptr);}
void operator delete(void* ptr,size_t) noexcept {std::free(ptr);}

int main() {
    using namespace wildimage_ram;
    auto& memory=budget();
    assert(memory.configure(512,{2*GiB,2*GiB},true));
    {
        watched_size=256; minimum_charge=256;
        sd::Tensor<float> large({64});
        assert(watched_calls==1); watched_size=0;
        sd::Tensor<float> small({1});
        assert(memory.used()==260);
        watched_size=4; minimum_charge=264; watched_calls=0;
        large=small;
        assert(watched_calls==1); watched_size=0;
        assert(memory.used()==8 && large.values().capacity()==1);
        watched_size=64; minimum_charge=72; watched_calls=0;
        large.resize({16});
        assert(watched_calls==1); watched_size=0;
        assert(memory.used()==68);
        const auto before=memory.used();
        watched_size=4000; watched_calls=0;
        try {large.resize({1000}); assert(false);} catch(const Exceeded&) {}
        assert(watched_calls==0); watched_size=0;
        assert(memory.used()==before && large.numel()==16);
        fail_size=64;
        try {sd::Tensor<float> failed({16}); assert(false);} catch(const std::bad_alloc&) {}
        assert(memory.used()==before);
        sd::Tensor<float> copy(large);
        assert(memory.used()==132);
        copy=std::move(large);
        assert(memory.used()==68 && large.empty());
        std::vector<float> external(8,7.f);
        auto from=sd::Tensor<float>::from_vector(external);
        assert(memory.used()==100 && from[0]==7.f);
        auto* raw=raw_malloc(64); assert(raw && memory.used()==164);
        assert(!raw_malloc(400)); assert(memory.used()==164);
        raw_free(raw); assert(memory.used()==100);
    }
    assert(memory.used()==0 && memory.peak()<=512);
    memory.ram_failure=false;
    {
        std::vector<uint8_t,Allocator<uint8_t>> payload(32);
        const auto before=memory.used();
        try {payload.reserve(1024); assert(false);} catch(const Exceeded&) {}
        assert(memory.used()==before && payload.size()==32 && !memory.ram_failure);
    }
    assert(memory.used()==0);
    {
        auto* full=static_cast<uint8_t*>(raw_malloc(512));
        assert(full && memory.used()==512); full[0]=17;
        auto* smaller=static_cast<uint8_t*>(raw_realloc(full,128));
        assert(smaller==full && smaller[0]==17 && memory.used()==512);
        assert(raw_realloc(smaller,0)==nullptr && memory.used()==0);
    }
}
