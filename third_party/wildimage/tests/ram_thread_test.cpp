#include "core/parallel.h"
#include "core/wildimage_ram_budget.h"
#include <cassert>
#include <thread>
#include <vector>

int main() {
    using namespace wildimage_ram;
    auto& memory=budget();
    assert(memory.configure(128,{2*GiB,2*GiB},true));
    {
        ThreadFailures required;
        std::thread worker([&] {required.run([] {std::vector<uint8_t,Allocator<uint8_t>> denied(1024);});});
        worker.join();
        assert(required.failed && memory.failure_code()==76 && memory.used()==0);
        try {required.rethrow(); assert(false);} catch(const Exceeded&) {}
    }
    memory.ram_failure=false;
    {
        std::unique_ptr<ThreadFailures> optional;
        {OptionalWork scope; optional=std::make_unique<ThreadFailures>();}
        std::thread worker([&] {optional->run([] {Reservation denied(1024);});});
        worker.join();
        assert(optional->failed && memory.failure_code()==0 && memory.used()==0);
        try {optional->rethrow(); assert(false);} catch(const Exceeded&) {}
        {Reservation actual(64); assert(memory.used()==64);}
        gpu_failure(); assert(memory.failure_code()==75);
    }
    memory.gpu_failure=false;
    {
        std::unique_ptr<ThreadFailures> optional;
        {OptionalWork scope; optional=std::make_unique<ThreadFailures>();}
        std::thread worker([&] {optional->run([] {throw std::bad_alloc();});});
        worker.join(); assert(optional->failed && memory.failure_code()==0);
        ThreadFailures required;
        std::thread failed([&] {required.run([] {throw std::bad_alloc();});});
        failed.join(); assert(required.failed && memory.failure_code()==76);
    }
    memory.ram_failure=false;
    sd::ParallelExecutor executor(2);
    sd::ParallelScope parallel(&executor);
    {
        OptionalWork optional;
        try {sd::parallel_for(0,4,1,[](int64_t,int64_t) {Reservation denied(1024);}); assert(false);}
        catch(const Exceeded&) {}
    }
    assert(memory.failure_code()==0 && memory.used()==0);
    try {sd::parallel_for(0,4,1,[](int64_t,int64_t) {Reservation denied(1024);}); assert(false);}
    catch(const Exceeded&) {}
    assert(memory.failure_code()==76 && memory.used()==0);
    memory.ram_failure=false;
    {
        std::atomic<bool> ready{false};
        std::thread required([&] {while(!ready.load()) std::this_thread::yield(); memory.fail();});
        std::thread optional([&] {OptionalWork scope; ready=true; memory.fail(); gpu_failure();});
        required.join(); optional.join();
        assert(memory.failure_code()==76); // Optional completion never clears concurrent fatal state.
    }
}
