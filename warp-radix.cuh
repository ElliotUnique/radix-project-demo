#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// A stable warp-level MSD-radix sort (32-bit) that resolves with a single sweep-through pass and returns the 
// index that sorts each key. The caller must guarantee all 32 lanes are convergent and active upon entry.
__device__ uint32_t warpRadix(uint32_t key,
                              int32_t start_bit,
                              int32_t N)
{
    // Setup: equal_mask records which lanes are still unresolved against my key to prevent accidentally reading from
    // lanes already known to rank higher than my key in the sort. It starts as a full mask to be narrowed down. lt_mask 
    // records lanes known to rank below me in the sort, and starts as an empty mask to be built up.
    if (N > 32 || N < 1) {return threadIdx.x & 31;}
    int32_t shift = 32 - N;
    uint32_t equal_mask = 0xFFFFFFFF >> shift;
    uint32_t lt_mask = 0u;

    // Main body: The relative ordering is determined by each thread broadcasting its key's bit via a ballot each pass.
    // If my key's current bit is a 0, lt_mask can't be updated. If my bit is a 1, I can add to my lt_mask all the
    // bit_ballot lanes with a 0. Any lanes that reported a different bit to mine must be eliminated from equal_mask 
    // as their relation to my key is now resolved.
    start_bit = (start_bit <= 31) ? start_bit : 31;
    #pragma unroll
    for (int32_t bit = start_bit; bit >= 0; --bit) 
    {
        bool bit_val = (key >> bit) & 1u;
        uint32_t bit_ballot = __ballot_sync(0xFFFFFFFF, bit_val);

        lt_mask |= bit_val ? equal_mask & ~bit_ballot : 0u;
        equal_mask &= bit_val ? bit_ballot : ~bit_ballot;
    }

    // Exit point: I mask out my own key's lane from equal_mask and any others with a higher initial index to mine, and
    // add them to lt_mask. This acts as a tie-break that enforces stability in the presence of duplicate keys. The number 
    // of set lanes in lt_mask then gives the final sorted index.
    uint32_t laneID = threadIdx.x & 31;
    lt_mask |= equal_mask & ((1u << laneID) - 1);
    return __popc(lt_mask);
}