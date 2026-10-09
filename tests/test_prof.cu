#include <iostream>
#include <thread>
#include <chrono>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>


#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

int main() {
    std::cout << "Booting NVIDIA Driver and initializing CUDA Context..." << std::endl;
    cudaFree(0);
    std::cout << "Starting Simulateed LLM Workload..." << std::endl;

    std::cout << "Simulating Prefill Phase (50MB Transfer)..." << std::endl;
    nvtxRangePushA("Phase_Prefill");
    
    size_t prefill_size = 50 * 1024 * 1024; // 50 MB

    void* d_prefill_data;
    void* h_prefill_data = malloc(prefill_size);
    CHECK_CUDA(cudaMalloc(&d_prefill_data, prefill_size));
    
    CHECK_CUDA(cudaMemcpy(d_prefill_data, h_prefill_data, prefill_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    nvtxRangePop(); 

    std::cout << "Simulating Decode Phase (100 iterative tokens)..." << std::endl;
    nvtxRangePushA("Phase_Decode");
    
    size_t chunk_size = 5 * 1024 * 1024; // 5 MB chunk 
    void* d_chunk_data;
    void* h_chunk_data = malloc(chunk_size);
    CHECK_CUDA(cudaMalloc(&d_chunk_data, chunk_size));

    for (int i = 0; i < 1000; i++) {
        nvtxRangePushA("Fetch_5MB_Chunk");
        CHECK_CUDA(cudaMemcpy(d_chunk_data, h_chunk_data, chunk_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaDeviceSynchronize()); // Wait for physical PCIe transfer to finish
        nvtxRangePop();
        
        // Simulate attention computation where GPU does math
        std::this_thread::sleep_for(std::chrono::microseconds(500)); 
    }
    
    nvtxRangePop(); 

    cudaFree(d_prefill_data);
    cudaFree(d_chunk_data);
    free(h_prefill_data);
    free(h_chunk_data);

    std::cout << "Complete..." << std::endl;
    return 0;
}