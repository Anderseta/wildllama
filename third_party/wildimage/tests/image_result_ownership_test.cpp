#include "resource_owners.hpp"
#include "resource_owners.hpp"
#include <cstdlib>

// Result pixels belong to the native host allocator on both sides of the DLL.
// Reconfiguring the budget is only possible once every live charge is released.
int main() {
    for (int i = 0; i < 10; ++i) {
        if (!sd_set_memory_budget("0.015625", true, "0")) return 1;
        auto* result = static_cast<sd_image_generation_result_t*>(std::calloc(1, sizeof(sd_image_generation_result_t)));
        if (!result) return 2;
        result->items = static_cast<sd_image_result_t*>(std::calloc(1, sizeof(sd_image_result_t)));
        if (!result->items) return 2;
        result->count = 1;
        result->items[0].image = {512, 512, 3, static_cast<uint8_t*>(sd_host_malloc(512 * 512 * 3))};
        if (!result->items[0].image.data) return 2;
        free_sd_image_generation_result(result);
        {
            SDImageVec images;
            images.push_back(sd_image_t{64, 64, 3, static_cast<uint8_t*>(sd_host_malloc(64 * 64 * 3))});
            if (!images[0].data) return 2;
        }
    }
    return sd_set_memory_budget("0.015625", true, "0") ? 0 : 1;
}
