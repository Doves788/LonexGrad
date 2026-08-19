


#ifndef lonexgrad_RANDOM_H_
#define lonexgrad_RANDOM_H_

#include <lonexgrad/export.h>
#include <lonexgrad/lonex_macros.h>

#include <cstdint>
#include <vector>

namespace lonexgrad {



class LONEXGRAD_EXPORT Generator {
public:
    



    LONEXGRAD_HOST_DEVICE Generator(uint64_t seed);

    


    LONEXGRAD_HOST_DEVICE static auto from_state(uint64_t state) -> Generator;

    


    LONEXGRAD_HOST_DEVICE auto operator()() noexcept -> uint64_t;

    


    LONEXGRAD_HOST_DEVICE void set_state(uint64_t state);

private:
    LONEXGRAD_HOST_DEVICE Generator(uint64_t s, bool is_state);

    uint64_t state;
};

LONEXGRAD_EXPORT void shuffle(std::vector<int> &data, Generator &gen);





LONEXGRAD_EXPORT void set_default_generator_seed(uint64_t seed);





LONEXGRAD_EXPORT auto get_default_generator() -> Generator &;

}    

#endif    
