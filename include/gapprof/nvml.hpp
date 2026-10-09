#pragma once

#include <thread>
#include <atomic>
#include <fstream>
#include <mutex>
#include <nvml.h>

namespace gapprof {
class NvmlSampler {
public:
    NvmlSampler();
    ~NvmlSampler();

    void start();
    void stop();

private:
    void worker();

    nvmlDevice_t device;
    unsigned long long start_energy_mj = 0;

    // NVML_FI_DEV_POWER_INSTANT was added in driver R535+, Older drivers return NVML_ERROR_NOT_SUPPORTED.
    bool field_api_supported = false;

    // GPU only or GH200 + GPU
    unsigned int power_scope_id = 0;

    std::thread poll_thread;
    std::atomic<bool> keep_polling{false};
};
} // namespace gapprof
