#pragma once
#include <cassert>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ void mainPassLogic(uint32_t&, int32_t&, uint32_t&, uint32_t&, int32_t, uint32_t*);
__device__ int32_t narrowBucketMask(bool, bool, bool, bool, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t&);
__device__ void submitMessages(uint32_t, uint32_t, uint32_t*);
__device__ uint32_t retrieveMessages(int32_t, int32_t, int32_t, uint32_t, uint32_t*);
__device__ void submitKeys(uint32_t, int32_t, uint32_t, uint32_t, uint32_t, int32_t, uint32_t*, int32_t);
__device__ void constructMetadata(int32_t&, uint32_t&, uint32_t&, uint32_t, uint32_t, int32_t);
__device__ __forceinline__ uint32_t bitRangeU32(int32_t, int32_t);

// Returns the value of key's ith bit (LSB to MSB). Guards against negative inputs.
__device__ int32_t ithBit(uint32_t key, int32_t bit) {return (bit >= 0) ? ((key >> bit) & 1u) : 0;}


// Block-level entry point. Sorts 32-bit keys from MSB to LSB in 8 passes.  
__global__ void blockRadixKernel(uint32_t* d_buffer,
                                 int32_t start_bit,
                                 size_t N)
{
    if (N > blockDim.x || (blockDim.x & (warpSize - 1)) != 0u) {return;}
    extern __shared__ uint32_t s_mem[];
    start_bit = (start_bit <= 31) ? start_bit : 31;

    // Initialise thread states: starting bucket_base 0, bucket_mask and bucket_warp_mask include all
    // threads/warps that carry keys as all start in same bucket. If any thread has no key, it is given a 
    // bucket_base sentinel value of -1, and the mask bit-fields only represent the owner's lane/warp.
    // These choices stop keyless threads from mutating the sort and simplify assertion logic.
    int32_t tid = threadIdx.x;
    int32_t num_warps = blockDim.x / warpSize;
    bool key_bearing = tid < N;
    uint32_t key = key_bearing ? d_buffer[tid] : 0u;
    int32_t bucket_base = (key_bearing) ? 0 : -1;
    uint32_t key_mask = __ballot_sync(0xFFFFFFFF, key_bearing);
    uint32_t bucket_mask =  (key_bearing) ? key_mask : (1u << (tid & 31));
    uint32_t bucket_warp_mask = (key_bearing) ? bitRangeU32(0, num_warps - 1) : (1u << (tid / warpSize));

    // Pass loop: descends all 32 bits top to bottom, 4 bits per pass
    for (int32_t nibble_start = start_bit; nibble_start >= 0; nibble_start -= 4)
    {
        mainPassLogic(key, bucket_base, bucket_mask, bucket_warp_mask, nibble_start, s_mem);
    }

    // Stability layer
    if (__popc(bucket_warp_mask) > 1) // Impure path
    {
        bucket_base = threadIdx.x;
    }
    else if (__popc(bucket_mask) > 1) // Pure path
    {
        int32_t stability_offset = __popc(bucket_mask & ((1u << (threadIdx.x & 31)) - 1));
        bucket_base += stability_offset;
    }

    assert(bucket_base < (int32_t)N);
    assert(key_bearing ? bucket_base >= 0 : bucket_base == -1);
    if (key_bearing)
    {
        d_buffer[bucket_base] = key;
    }
    return;
}

// Processes each MSD pass: subdivides buckets using either warp intrinsics if the bucket is pure, or by 
// sharing data across warps if impure. Keys in an impure bucket are redistributed to keep bucket 
// members contiguous in threadIdx.x space, which bounds the number of impure buckets per warp to two or 
// fewer, and guarantees that they will always straddle contiguous neighbouring warps.
__device__ void mainPassLogic(uint32_t& key,
                              int32_t& bucket_base,
                              uint32_t& bucket_mask,
                              uint32_t& bucket_warp_mask,
                              int32_t nibble_start,
                              uint32_t* s_mem)
{
    // Intra-warp Phase: Determine who my "messengers" are and let them read my data.

    // Establish my context - does my bucket straddle multiple warps? If it straddles any warp below
    // mine, the straddle "direction" is negative. Positive if it only straddles warps above. If
    // the bucket is pure and straddles no warp boundaries, the direction is 0.
    uint32_t my_warp_mask = 1u << (threadIdx.x >> 5);
    int32_t bucket_warps_below = __ffs(bucket_warp_mask) < __ffs(my_warp_mask);
    int32_t bucket_warps_above = __clz(bucket_warp_mask) < __clz(my_warp_mask);
    int32_t straddle_dir = bucket_warps_above - 2 * bucket_warps_below;
    uint32_t straddles_above = __ballot_sync(0xFFFFFFFFu, (straddle_dir > 0));
    uint32_t straddles_below = __ballot_sync(0xFFFFFFFFu, (straddle_dir < 0));
    
    uint32_t b0_ballot = __ballot_sync(0xFFFFFFFFu, ithBit(key, nibble_start - 0));
    uint32_t b1_ballot = __ballot_sync(0xFFFFFFFFu, ithBit(key, nibble_start - 1));
    uint32_t b2_ballot = __ballot_sync(0xFFFFFFFFu, ithBit(key, nibble_start - 2));
    uint32_t b3_ballot = __ballot_sync(0xFFFFFFFFu, ithBit(key, nibble_start - 3));

    // Compute who I need to carry messages for: if my lane is in [0, 15], I service threads /w
    // buckets that straddle below, else I service those that straddle above.
    uint32_t sub_bucket_mask = (threadIdx.x & 16) ? straddles_above : straddles_below;
    int32_t anchor_lane = (sub_bucket_mask != 0) ? __ffs(sub_bucket_mask) - 1 : 0;
    uint32_t msg_warp_mask = __shfl_sync(0xFFFFFFFFu, bucket_warp_mask, anchor_lane);
    if (sub_bucket_mask == 0u) {msg_warp_mask = 0u;}

    // If no one's bucket straddles multiple warps, this warp is pure and thus resolves sort w/
    // register ops and warp intrinsics. Pure threads from impure warps are needed for the messenger
    // phase and cannot take this route.
    if ((straddles_above | straddles_below) == 0)
    {
        bucket_base += narrowBucketMask(ithBit(key, nibble_start - 0),
                                        ithBit(key, nibble_start - 1),
                                        ithBit(key, nibble_start - 2),
                                        ithBit(key, nibble_start - 3),
                                        b0_ballot,
                                        b1_ballot,
                                        b2_ballot,
                                        b3_ballot,
                                        bucket_mask);
        __syncthreads(); // Must still contribute to block-level syncing, else may cause a deadlock
        __syncthreads();
        __syncthreads();
        __syncthreads();
        return;
    }

    // Messaging phase: From the context I serve (i.e. straddles_above or straddles_below), I count how
    // many threads have a nibble value less than or equal to my nibble_slot. I then submit this number
    // to shared memory and accumulate the counts submitted by messengers from the other warps. Across  
    // all 16 messengers, these counts form a prefix summed population histogram of the nibble values.
    // All threads act as messengers unless the *entire* warp is pure, even if their own personal bucket 
    // happens to be pure.
    uint32_t nibble_slot = threadIdx.x & 15u;
    uint32_t msg_lte_count = narrowBucketMask((nibble_slot & 8u),
                                              (nibble_slot & 4u),
                                              (nibble_slot & 2u),
                                              (nibble_slot & 1u),
                                              b0_ballot,
                                              b1_ballot,
                                              b2_ballot,
                                              b3_ballot,
                                              sub_bucket_mask);
    msg_lte_count += __popc(sub_bucket_mask); 

    __syncthreads();

    if (msg_warp_mask != 0)
    {
        submitMessages(msg_lte_count, msg_warp_mask, s_mem);
    }

    int32_t my_row = __popc(msg_warp_mask & (my_warp_mask - 1));
    int32_t final_row = __popc(msg_warp_mask) - 1;

    __syncthreads();

    uint32_t prefix_above = retrieveMessages(my_row + 1, 
                                             final_row - my_row, 
                                             nibble_slot, 
                                             msg_warp_mask, 
                                             s_mem);
    uint32_t prefix_below = retrieveMessages(0, 
                                             my_row, 
                                             nibble_slot, 
                                             msg_warp_mask, 
                                             s_mem);
    uint32_t sub_bucket_ub = prefix_above + prefix_below + msg_lte_count;

   
    if (straddle_dir == 0) // If my own bucket is pure, resolve my nibble for this pass
    {
        bucket_base += narrowBucketMask(ithBit(key, nibble_start - 0),
                                        ithBit(key, nibble_start - 1),
                                        ithBit(key, nibble_start - 2),
                                        ithBit(key, nibble_start - 3),
                                        b0_ballot,
                                        b1_ballot,
                                        b2_ballot,
                                        b3_ballot,
                                        bucket_mask);
    }

    __syncthreads();

    // Key-shuffle phase: Threads end their messenger duties and submit their own keys (if impure) to 
    // shared memory to redistribute among the context. Keys are ordered by nibble value first then 
    // relative index, and picked up by threads in index order. This process is interleaved with pre-
    // emptive metadata construction ahead of the arrival of the new keys.
    submitKeys(
        key, straddle_dir, sub_bucket_mask, prefix_below, sub_bucket_ub, bucket_base, s_mem, nibble_start);

    // Binary search of my messengers to find the lower bound of my future key's bucket. Messengers hold sub-
    // bucket upper bounds, so the search space is across {0} Union {messenger 0, ..., messenger 14}, not
    // {messenger 0, ..., messenger 15}.
    uint32_t context_pos = threadIdx.x - bucket_base;
    uint32_t sub_bucket_floor = 0;
    int32_t messenger_base = (straddle_dir < 0) ? 0 : 16;
    int32_t lo = 0, hi = 15;
    for (int32_t i = 0; i < 4; ++i)
    {
        uint32_t mid = (lo + hi) >> 1;
        uint32_t probe_lb = __shfl_sync(0xFFFFFFFFu, sub_bucket_ub, messenger_base + mid);
        if (probe_lb > context_pos)
        {
            hi = mid;
        }
        else
        {
            sub_bucket_floor = probe_lb;
            lo = mid + 1;
        }
    }
    uint32_t sub_bucket_ceiling = __shfl_sync(0xFFFFFFFFu, sub_bucket_ub, messenger_base + lo);
    assert((context_pos < __shfl_sync(0xFFFFFFFFu, sub_bucket_ub, messenger_base + 15)) || (straddle_dir == 0));

    __syncthreads();

    if (straddle_dir == 0) // The procedures below will corrupt the metadata of pure/keyless threads. Exit now.
    {
        assert(bucket_warp_mask == (1u << (threadIdx.x >> 5u)));
        assert((bucket_mask & (1u << (threadIdx.x & 31u))) > 0u);
        return;
    }

    assert(sub_bucket_floor <= context_pos && // Gated from pure/keyless threads w/ garbage values
           sub_bucket_ceiling > context_pos);

    key = s_mem[threadIdx.x];

    // Final metadata reconstruction step while new key is arriving.
    constructMetadata(bucket_base,
                      bucket_mask,
                      bucket_warp_mask,
                      sub_bucket_floor,
                      sub_bucket_ceiling,
                      straddle_dir);
}

// Narrows my bucket-membership mask by eliminating lanes whose nibble is not equal to my own.
// Returns the number of lanes in my bucket with a nibble value less than my own.
__device__ int32_t narrowBucketMask(bool b0,
                                    bool b1,
                                    bool b2,
                                    bool b3,
                                    uint32_t b0_ballot,
                                    uint32_t b1_ballot,
                                    uint32_t b2_ballot,
                                    uint32_t b3_ballot,
                                    uint32_t& bucket_mask)
{
    uint32_t lt_mask = 0u;
    lt_mask |= b0 ? (bucket_mask & ~b0_ballot) : 0u; // b0 is the top bit of the nibble
    bucket_mask &= b0 ? b0_ballot : ~b0_ballot;
    lt_mask |= b1 ? (bucket_mask & ~b1_ballot) : 0u;
    bucket_mask &= b1 ? b1_ballot : ~b1_ballot;
    lt_mask |= b2 ? (bucket_mask & ~b2_ballot) : 0u;
    bucket_mask &= b2 ? b2_ballot : ~b2_ballot;
    lt_mask |= b3 ? (bucket_mask & ~b3_ballot) : 0u;
    bucket_mask &= b3 ? b3_ballot : ~b3_ballot;

    return __popc(lt_mask);
}

// Writes my message (an element of the bucket histogram's prefix-sum) to a row-major table 
// in shared memory with dimensions 16 * num-warps-in-context. The table start is anchored 
// at the midpoint of the lowest warp in the context.
__device__ void submitMessages(uint32_t msg_lte_count, 
                               uint32_t msg_warp_mask, 
                               uint32_t* s_mem)
{
    int32_t anchor_warp = __ffs(msg_warp_mask) - 1;
    int32_t base_idx = anchor_warp * warpSize + 16;
    int32_t my_warp_idx = threadIdx.x / warpSize;
    int32_t my_offset = (my_warp_idx - anchor_warp) * 16 + (threadIdx.x & 15);
    assert(base_idx + my_offset < blockDim.x);
    s_mem[base_idx + my_offset] = msg_lte_count;
}

// Sums other warp's prefix element submissions down the table row that corresponds to my 
// nibble_slot. Accumulates up to 8 entries at a time before combining them to break up the
// dependency chain.
__device__ uint32_t retrieveMessages(int32_t start_row,
                                     int32_t row_count,
                                     int32_t nibble_slot,
                                     uint32_t msg_warp_mask,
                                     uint32_t* s_mem)
{
    if (msg_warp_mask == 0)
        return 0;
    int32_t anchor_warp = __ffs(msg_warp_mask) - 1;
    int32_t base_idx = anchor_warp * warpSize + 16 + nibble_slot + (start_row << 4);

    // Linear, segmented reduction, broken into a "body" and a "tail". The body reduces 
    // in multiples of 8 and only deploys when there are more than 8 elements to a table column. 
    // The tail handles the remainder.
    uint32_t t[8];
    int32_t acc_0 = 0, acc_1 = 0;
    int32_t row = 0;
#pragma unroll 1
    for (; row + 8 <= row_count; row += 8)
    {
#pragma unroll
        for (int32_t i = 0; i < 8; ++i)
        {
            assert(base_idx + ((row + i) << 4) < blockDim.x);
            t[i] = s_mem[base_idx + ((row + i) << 4)];
        }
        acc_0 += t[0] + t[1] + t[2] + t[3];
        acc_1 += t[4] + t[5] + t[6] + t[7];
    }

    int32_t tail = row_count - row;
#pragma unroll
    for (int32_t i = 0; i < 8; ++i)
    {
        assert(!(i < tail) || (base_idx + ((row + i) << 4) < blockDim.x));
        t[i] = (i < tail) ? s_mem[base_idx + ((row + i) << 4)] : 0u;
    }
    acc_0 += t[0] + t[1] + t[2] + t[3];
    acc_1 += t[4] + t[5] + t[6] + t[7];

    return acc_0 + acc_1;
}

// Computes my key's destination index then scatters it: 
// Scatter offset = my sub-bucket's floor + same-nibble keys ranked below me across whole context. 
// The destination index is then scatter offset + my bucket's base.
__device__ void submitKeys(uint32_t key,
                           int32_t straddle_dir,
                           uint32_t sub_bucket_mask,
                           uint32_t prefix_below,
                           uint32_t sub_bucket_ub,
                           int32_t bucket_base,
                           uint32_t* s_mem,
                           int32_t nibble_start)
{
    uint32_t my_nibble = (ithBit(key, nibble_start - 0) << 3) |
                         (ithBit(key, nibble_start - 1) << 2) |
                         (ithBit(key, nibble_start - 2) << 1) |
                         (ithBit(key, nibble_start - 3) << 0);
    int32_t prev_slot = (my_nibble == 0) ? my_nibble : my_nibble - 1; // No sub-bucket below slot 0
    int32_t messenger_base = (straddle_dir < 0) ? 0 : 16;

    uint32_t sub_bucket_floor = __shfl_sync(0xFFFFFFFFu, sub_bucket_ub, messenger_base + prev_slot);
    uint32_t half_bucket_lo = __shfl_sync(0xFFFFFFFFu, prefix_below, messenger_base + prev_slot);
    uint32_t half_bucket_hi = __shfl_sync(0xFFFFFFFFu, prefix_below, messenger_base + my_nibble);
    uint32_t my_bucket_mask = __shfl_sync(0xFFFFFFFFu, sub_bucket_mask, messenger_base + my_nibble);

    if (straddle_dir == 0) {return;} // Shuffle syncs are done; pure threads can exit

    sub_bucket_floor = (my_nibble > 0) ? sub_bucket_floor : 0;
    uint32_t same_nibble_below = (my_nibble > 0) ? half_bucket_hi - half_bucket_lo : half_bucket_hi; // No sub-bucket below slot 0
    int32_t same_nibble_self = __popc(my_bucket_mask & ((1u << (threadIdx.x & 31)) - 1));
    int32_t scatter_offset = sub_bucket_floor + same_nibble_below + same_nibble_self;
    assert(threadIdx.x >= static_cast<uint32_t>(bucket_base)); // Invariant: bucket_base = context anchor in index space.
                                                               // If triggered, the error is likely from a previous shuffle.
    int32_t write_idx = bucket_base + scatter_offset;
    assert(write_idx >= 0 && static_cast<uint32_t>(write_idx) < blockDim.x);
    s_mem[write_idx] = key;
}

// Derive the bucket_base and membership masks for the incoming key ahead of the next pass.
__device__ void constructMetadata(int32_t& bucket_base,
                                  uint32_t& bucket_mask,
                                  uint32_t& bucket_warp_mask,
                                  uint32_t sub_bucket_floor,
                                  uint32_t sub_bucket_ceiling,
                                  int32_t straddle_dir)
{
    assert(straddle_dir != 0); // If triggered, check correctness of pure thread early exit above key collection

    int32_t sub_bucket_start = bucket_base + sub_bucket_floor;
    int32_t sub_bucket_end = bucket_base + sub_bucket_ceiling - 1;
    int32_t my_warp_idx = threadIdx.x >> 5;
    int32_t warp_base = my_warp_idx << 5;

    // New mask is the intersection between my bucket_base's span and my warp's 32 lanes
    int32_t lane_lo = max(0, sub_bucket_start - warp_base);
    int32_t lane_hi = min(31, sub_bucket_end - warp_base);
    bucket_mask = bitRangeU32(lane_lo, lane_hi);

    int32_t warp_lo = sub_bucket_start >> 5;
    int32_t warp_hi = sub_bucket_end >> 5;
    bucket_warp_mask = bitRangeU32(warp_lo, warp_hi); 

    bucket_base += sub_bucket_floor;
}

// Returns a bit-field with bits set between positions a and b (inclusive)
__device__ __forceinline__ uint32_t bitRangeU32(int32_t a, int32_t b)
{
    // If asserts trigger: call sites at end of constructMetadata 
    assert(a <= b);
    assert(a >= 0 && b <= 31);

    uint32_t hi = (1u << b) | ((1u << b) - 1u);
    uint32_t lo = (1u << a) - 1u;
    return hi & ~lo;
}