#include "../native/wildimage_ram_budget.h"
#include <cassert>
#include <thread>
#include <vector>

int main() {
    using namespace wildimage_ram;
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
