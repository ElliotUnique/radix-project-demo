#include "block-radix.cuh"
#include "warp-radix.cuh"
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>
#include <stdio.h>
#include <stdint.h>

#define CUDACHECK(cmd) do { \
    cudaError_t e = cmd; \
    if (e != cudaSuccess) { \
        printf("CUDA Error: %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

bool mainMenu(bool& running, uint32_t& lb, uint32_t& ub, bool& ver, bool& p_in, bool& p_out);
bool runValidation(uint32_t lb, uint32_t ub, bool ver, bool p_in, bool p_out);
__global__ void warpRadixKernel(uint32_t* d_buffer, size_t N);
void generateKeys(uint32_t* keys, uint32_t lb, uint32_t ub, size_t N);
void validateSort(uint32_t* keys, bool ver, size_t N);
bool checkPermutation(uint32_t* keys, size_t N);
bool checkMonotonicity(uint32_t* keys, size_t N);
bool checkStability(uint32_t* keys, size_t N);
void nextScreen() {for (int32_t _ = 0; _ < 10; ++_) {printf("\n\n\n\n");}}
void printKeys(const bool print, const uint32_t* keys, uint32_t N, const char* title)
{
    if (print)
    {    
        printf("\n\n%s\n", title);
        for (uint32_t i = 0; i < N; ++i) {printf("%u, ", keys[i] & ((1u << 22) - 1));}
    }
}
void dumpBadInput(int32_t validity)
{
    if (validity != 1)
    {
        int32_t c;
        while ((c = getchar()) != '\n' && c != EOF){}
        printf("\nInvalid selection");
    }
}

int main()
{
    bool menu = true;
    bool running = true;
    uint32_t lb = 0;
    uint32_t ub = 15;
    bool ver = 1;
    bool p_in = false;
    bool p_out = false;
    nextScreen();
    while (running)
    {
        if (menu)
        {
            menu = mainMenu(running, lb, ub, ver, p_in, p_out);
        }
        else
        {
            menu = runValidation(lb, ub, ver, p_in, p_out);
        }
    }
    return 0;
}

bool mainMenu(bool& running, uint32_t& lb, uint32_t& ub, bool& ver, bool& p_in, bool& p_out)
{
    printf("\nMain Menu\n--------\n[1] Run validation\n[2] Choose key lower bound (%u)\n[3] Choose key upper bound (%u)\n[4] Change algorithm       (%s)\n[5] Display input array    (%s)\n[6] Display output array   (%s)\n[0] Quit\nEnter: ",
           lb, ub,
           ver ? "blockRadix" : "warpRadix",
           p_in ? "ON" : "OFF", 
           p_out ? "ON" : "OFF");
    uint32_t user_input = ~0u;
    dumpBadInput(scanf("%u", &user_input));
    nextScreen();
    switch (user_input) 
    {
        case 1:
        {
            nextScreen();
            return false;
        }
        case 2:
        {
            uint32_t bound = lb;
            printf("\nChoose a value from the interval [0, %u], where %u is the currently selected upper bound\nEnter: ", ub, ub);
            dumpBadInput(scanf("%u", &bound));
            if (bound <= ub) 
            {
                lb = bound;
            }
            break;
        }
        case 3:
        {
            uint32_t bound = ub;
            printf("\nChoose a value from the interval [%u, 2^22], where %u is the currently selected lower bound\nEnter: ", lb, lb);
            dumpBadInput(scanf("%u", &bound));
            if ((bound >= lb) && (bound <= ((1 << 22) - 1))) 
            {
                ub = bound;
            }
            break;
        }
        case 4:
        {
            ver = !ver;
            break;
        }
        case 5:
        {
            p_in = !p_in;
            break;
        }
        case 6:
        {
            p_out = !p_out;
            break;
        }
        case 0:
        {
            running = false;
            break;
        }
    }
    nextScreen();
    return true;
}

bool runValidation(uint32_t lb, uint32_t ub, bool ver, bool p_in, bool p_out)
{
    // Select N and generate keyset
    uint32_t max_N = ver ? 1024u : 32u;
    printf("\n\nValidating %s:\nSelect a value for N from the interval [1, %u], or select [0] to quit to menu.\nEnter: ",
           ver ? "block radix" : "warp radix",
           static_cast<uint32_t>(max_N));
    uint32_t user_input = ~0u;
    dumpBadInput(scanf("%u", &user_input));
    
    nextScreen();
    if (user_input == 0) {return 1;}
    else if (user_input > max_N) {printf("Defaulting to N = %u", static_cast<uint32_t>(max_N));}

    uint32_t N = (user_input > max_N) ? max_N : user_input;

    uint32_t keys[1024] = {0};
    generateKeys(keys, lb, ub, N);
    printKeys(p_in, keys, N, "Unsorted keys:");

    // Initialise device memory and launch sorting kernel, rearrange keys, then validate results
    uint32_t* d_buffer = nullptr;
    size_t N_size = N * sizeof(uint32_t);
    CUDACHECK(cudaMalloc((void**)&d_buffer, N_size));
    CUDACHECK(cudaMemcpy(d_buffer, keys, N_size, cudaMemcpyHostToDevice));

    if (ver)
    {
        uint32_t block_size = 32u * ((N + 31u) / 32u);
        size_t smem_bytes = block_size * sizeof(uint32_t);
        blockRadixKernel<<<1, block_size, smem_bytes>>>(d_buffer, 21, N);
    }
    else {warpRadixKernel<<<1, 32>>>(d_buffer, N);}

    cudaError_t launch_error = cudaGetLastError();
    cudaError_t exec_error = cudaDeviceSynchronize();
    if (launch_error != cudaSuccess) {printf("LAUNCH: %s\n", cudaGetErrorString(launch_error));}
    if (exec_error   != cudaSuccess) {printf("EXEC:   %s\n", cudaGetErrorString(exec_error));}

    CUDACHECK(cudaMemcpy(keys, d_buffer, N_size, cudaMemcpyDeviceToHost));
    printKeys(p_out, keys, N, "Sorted keys:");
    validateSort(keys, ver, N);

    CUDACHECK(cudaFree(d_buffer));
    return 0;
}

// Draws keys from a user-defined uniform distribution and packs each key's initial index into the upper 10 bits
// to allow stability-checking
void generateKeys(uint32_t* keys, uint32_t lb, uint32_t ub, size_t N)
{
    std::random_device seed;
    std::mt19937 gen(seed());
    std::uniform_int_distribution<uint32_t> dist(lb, ub);

    for (uint32_t i = 0; i < N; ++i)
    {
        uint32_t key = dist(gen);
        keys[i] = (key & ((1 << 22) - 1)) | (i << 22);
    }
}

// Runs the warp-level radix sort (up to 32 keys)
__global__ void warpRadixKernel(uint32_t* d_buffer, size_t N)
{
    int32_t tid = threadIdx.x;
    bool key_bearing = (tid < N);
    uint32_t key = key_bearing ? d_buffer[tid] : 0u;

    int32_t sorted_idx = warpRadix(key, 21, N);

    if (key_bearing)
    {
        d_buffer[sorted_idx] = key;
    }
}

// Tests sort for monotonicity and stability and prints a declaration of the results
void validateSort(uint32_t* keys, bool ver, size_t N)
{
    bool permuted = checkPermutation(keys, N);
    bool monotonic = checkMonotonicity(keys, N);
    bool stable = checkStability(keys, N);
    printf("\n\n%s %s (N = %u):\nPermutation test:    %s\nMonotonicity test:   %s\nStability test:      %s",
           (ver) ? "Block-Radix" : "Warp-Radix",
           (permuted && monotonic && stable) ? "Validated" : "Invalidated",
           static_cast<uint32_t>(N),
           (permuted) ? "passed" : "   failed",
           (monotonic) ? "passed" : "  failed",
           (stable) ? "passed" : "     failed");
}

// Checks that all numbers between 0 and N - 1 are present in the bit-packed indices to
// catch any duplicated or lost keys that might pass the other two tests.
bool checkPermutation(uint32_t* keys, size_t N)
{
    int32_t headcount[1024] = {0};
    for (size_t i = 0; i < N; ++i)
    {
        uint32_t idx = keys[i] >> 22;
        ++headcount[idx];
    }
    for (size_t i = 0; i < N; ++i)
    {
        if (headcount[i] != 1)
        {
            return false;
        }
    }
    return true;
}

// Checks that consecutive keys appear in the right relative order, ignoring the upper 10 bits which
// encode the initial indices
bool checkMonotonicity(uint32_t* keys, size_t N)
{
    for (size_t i = 0; i + 1 < N; ++i)
    {
        bool unsorted = (keys[i] & ((1u << 22) - 1)) > (keys[i + 1] & ((1u << 22) - 1));
        if (unsorted)
        {
            return false;
        }
    }
    return true;
}

// Checks that identical keys (in the bottom 22 bits) appear in an order with strictly increasing
// initial indices.
bool checkStability(uint32_t* keys, size_t N)
{
    for (size_t i = 0; i + 1 < N; ++i)
    {
        bool equal_keys = (keys[i] << 10) == (keys[i + 1] << 10);
        bool strictly_increasing_idxs = (keys[i] >> 22) < (keys[i + 1] >> 22);
        if (equal_keys && !strictly_increasing_idxs)
        {
            return false;
        }
    }
    return true;
}


