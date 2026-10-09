#pragma once

#include <iostream>
#include <nvml.h>
#include <cupti.h>

// Failure log will not kill host process
#define GAPPROF_NVML_LOG(call) \
    do { \
        nvmlReturn_t status = call; \
        if (status != NVML_SUCCESS) { \
            std::cerr << "[GapProf Warning] NVML Error: " << nvmlErrorString(status) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while(0)

#define GAPPROF_CUPTI_LOG(call, errstr) \
    do { \
        CUptiResult _status = call; \
        if (_status != CUPTI_SUCCESS) { \
            const char* errstr_ptr; \
            cuptiGetResultString(_status, &errstr_ptr); \
            std::cerr << "[GapProf Error] CUPTI Error (" << errstr << "): " << errstr_ptr \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while(0)
