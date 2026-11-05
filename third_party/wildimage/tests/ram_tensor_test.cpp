#include "core/tensor.hpp"
#include <cassert>

int main() {
    using namespace wildimage_ram;
    auto& memory=budget();
    assert(memory.configure(512,{2*GiB,2*GiB},true));
    {
        sd::Tensor<float> large({64}), small({1});
        assert(memory.used()==260);
        large=small;
        assert(memory.used()==8 && large.values().capacity()==1);
        large.resize({16});
        assert(memory.used()==68);
        const auto before=memory.used();
        try {large.resize({1000}); assert(false);} catch(const Exceeded&) {}
        assert(memory.used()==before && large.numel()==16);
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
}
