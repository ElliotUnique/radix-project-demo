# Parallel MSD-Radix Sort Prototype

## Algorithm Overview
The algorithm is predicated on encoding the bucket structure of MSD-radix by associating each key with an integer that counts how many keys must precede it in the sort. This forms the authoritative sorting mechanism rather than iteratively adjusting the physical location of keys throughout the sort. The physical movement of keys is instead performed tactically to isolate buckets within single warps where they can resolve in register space, thereby controlling the scale of cross-warp data traffic and the overall memory footprint.

Each pass consists of three phases:
1. **Intra-warp Phase:** Threads read their key's bits and share their values across the warp.
2. **Messenger Phase:** Lanes are organised into messenger groups that interpret the bit values from phase one and deliver them to an allotted shared memory region. Messengers then return with data submitted by other warps.
3. **Key-shuffle Phase:** If a bucket's population intersects more than one warp, the constituent keys are scattered to contiguous memory in sub-bucket order (derived from the data in phase two) and picked up by threads in thread-index order. At the same time, threads construct the metadata of their new key ahead of its arrival.

## Repo Layout
This repo presents two self-contained `.cuh` headers containing the algorithmic code and one `.cu` demo file that `#include`s both.

- **warp-radix.cuh:** The source code for a single-warp algorithm that sweeps through the keys' bits in one pass and outputs their sorted index. This preceded the algorithm outlined in the overview and its adapted logic drives the intra-warp phase.
- **block-radix.cuh:** The source code for the algorithm outlined in the overview. Sorts up to 1024 keys, using shared memory to exchange data between warps and redistribute keys.
- **radix-demo.cu:** An interactive terminal-based demo that runs the two algorithms on randomly generated keysets and tests the output for monotonicity, stability and permutation validity. This harness uses each key's upper 10 bits to carry its original index for the stability test.
- **CMakeLists.txt:** The build configuration for the demo.
- **RATIONALE.md:** A discussion of the core architectural decisions.

## Scope
Both algorithms accept one key per thread and work on 32-bit unsigned integers. `block-radix.cuh` defines a kernel that operates across one block and scatters sorted keys back into the input array, while `warp-radix.cuh` defines a single-warp device function that returns each key's sorted index. They synchronise exclusively through `__syncthreads()` gates and full-mask warp collectives that enforce lane convergence; the sort logic is not reliant on Pascal's lockstep behaviour, though the algorithms haven't been tested on hardware that supports independent scheduling.

## Build & Run
- **Prerequisites:** CMake v3.20 + Ninja, validated on CUDA v11.8 targeting sm_61 (Pascal). To target a different architecture, change the `CMAKE_CUDA_ARCHITECTURES` setting in `CMakeLists.txt` from `61` to match your compute capability.
- **Configure:** `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release` or `=Debug`. `Release` defines `NDEBUG` which compiles out all assertions, `Debug` preserves them. If configured as Debug, note that assertions increase register occupancy which may cause launch errors for large N.
- **Build:** `cmake --build build`
- **Run:** `.\build\demo` (Windows) or `./build/demo` (Linux) via a terminal. Launches a selection menu prompting the user to choose an algorithm, set a key range, toggle input/output printing (off by default) or run validation. 

For a discussion of the core architectural decisions, see [RATIONALE.md](RATIONALE.md).