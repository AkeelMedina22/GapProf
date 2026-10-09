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
    std::cout << "Starting Simulated LLM Workload (Asynchronous)..." << std::endl;
    
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();;

    std::cout << "Simulating Prefill Phase (50MB Transfer)..." << std::endl;
    nvtxRangePushA("Phase_Prefill");
    
    size_t prefill_size = 50 * 1024 * 1024; // 50 MB

    void* d_prefill_data;
    void* h_prefill_data;

    // Pinned memory is required for async DMA transfers
    CHECK_CUDA(cudaMallocHost(&h_prefill_data, prefill_size));
    CHECK_CUDA(cudaMalloc(&d_prefill_data, prefill_size));
    
    CHECK_CUDA(cudaMemcpy(d_prefill_data, h_prefill_data, prefill_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    nvtxRangePop(); 

    std::cout << "Simulating Decode Phase (100 iterative async tokens)..." << std::endl;
    nvtxRangePushA("Phase_Decode");
    
    size_t chunk_size = 5 * 1024 * 1024; // 5 MB chunk 
    void* d_chunk_data;
    void* h_chunk_data;
    CHECK_CUDA(cudaMallocHost(&h_chunk_data, chunk_size)); 
    CHECK_CUDA(cudaMalloc(&d_chunk_data, chunk_size));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    for (int i = 0; i < 1000; i++) {
        nvtxRangePushA("Fetch_5MB_Chunk");
        
        // Async DMA copy
        CHECK_CUDA(cudaMemcpyAsync(d_chunk_data, h_chunk_data, chunk_size, cudaMemcpyHostToDevice, stream));
        cudaDeviceSynchronize();
        
        // NVTX range ends instantly after copy. 
        nvtxRangePop();
        
        std::this_thread::sleep_for(std::chrono::microseconds(500)); 
        
    }
    
    // Ensure all async operations are complete
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaStreamDestroy(stream));
    
    nvtxRangePop(); 

    cudaFree(d_prefill_data);
    cudaFree(d_chunk_data);
    cudaFreeHost(h_prefill_data);
    cudaFreeHost(h_chunk_data);

    std::cout << "Complete..." << std::endl;
    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    auto Duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime-startTime);
    std::cout << "Duration " << Duration.count() << " microseconds." << std::endl;

    return 0;
}