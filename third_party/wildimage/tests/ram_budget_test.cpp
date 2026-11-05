#include "../native/wildimage_ram_budget.h"
#include <cassert>
#include <thread>
#include <vector>

int main() {
    using namespace wildimage_ram;
    uint64_t bytes=999;
    assert(parse_gib("0.25",bytes) && bytes==GiB/4);
    assert(parse_gib("0",bytes) && bytes==0);
    assert(parse_gib("0.0009765625",bytes) && bytes==(1ULL<<20));
    assert(!parse_gib("17179869184",bytes));
    assert(!parse_gib("1e400",bytes) && !parse_gib("1e-400",bytes));
    assert(!parse_gib("nan",bytes) && !parse_gib("-1",bytes));
    assert(!parse_gib("0.00000000000001",bytes));
    assert(effective_limit(0,{32*GiB,24*GiB}) == 24*GiB-32*GiB/20);
    assert(effective_limit(4*GiB,{32*GiB,24*GiB}) == 4*GiB);
    assert(effective_limit(30*GiB,{32*GiB,24*GiB}) == 24*GiB-32*GiB/20);
    assert(effective_limit(0,{0,0}) == 0);
    assert(effective_limit(8*GiB,{8*GiB,GiB}) == 0);
    Budget memory;
    Budget unavailable;
    assert(!unavailable.configure(0,{0,0},false));
    assert(!unavailable.try_acquire(1) && unavailable.used()==0);
    assert(memory.configure(128,{2*GiB,2*GiB},true));
    assert(memory.gpu_reserve_bytes==GiB/4);
    Budget zero_headroom;
    assert(zero_headroom.configure(128,{2*GiB,2*GiB},true,0));
    assert(zero_headroom.gpu_reserve_bytes==0);
    assert(zero_headroom.limit()==128 && zero_headroom.gpu_reserve_bytes==0);
    Budget custom_headroom;
    assert(custom_headroom.configure(128,{2*GiB,2*GiB},true,GiB/2));
    assert(custom_headroom.limit()==128 && custom_headroom.gpu_reserve_bytes==GiB/2);
    assert(saturating_add(UINT64_MAX-1,2)==UINT64_MAX);
    assert(saturating_add(saturating_add(UINT64_MAX-512,256),512)==UINT64_MAX);
    assert(saturating_add(1024,GiB/4)==1024+GiB/4);
    {
        Reservation first; assert(first.try_reset(80,&memory));
        Reservation second; assert(!second.try_reset(49,&memory));
        assert(second.try_reset(48,&memory));
        assert(memory.used()==128);
        Reservation moved(std::move(second));
        first.reset(32,&memory); assert(memory.used()==80);
        {Reservation copy(moved); assert(memory.used()==128);}
        assert(memory.used()==80);
        assert(!memory.configure(512,{2*GiB,2*GiB},true));
        assert(memory.limit()==128);
        assert(!moved.try_reset(UINT64_MAX,&memory));
        assert(memory.used()==80);
    }
    assert(memory.used()==0 && memory.peak()==128);
    std::vector<std::thread> threads;
    for(int i=0;i<8;++i) threads.emplace_back([&] {
        for(int j=0;j<10000;++j) {
            Reservation token;
            if(token.try_reset(32,&memory)) assert(memory.used()<=128);
        }
    });
    for(auto& thread:threads) thread.join();
    assert(memory.used()==0 && memory.peak()<=128);
    memory.gpu_failure=true; assert(memory.failure_code()==75);
    memory.ram_failure=true; assert(memory.failure_code()==76);
    assert(classify_exit(0,75)==0 && classify_exit(0,76)==0);
    assert(classify_exit(1,75)==75 && classify_exit(1,76)==76 && classify_exit(1,0)==1);
    return 0;
}
