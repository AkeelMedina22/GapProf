#pragma once

#include <cstdint>
#include <cstddef>

namespace gapprof {

enum class EventSource : uint8_t {
    RUNTIME_CALLBACK,
    NVTX_CALLBACK,
    ACTIVITY_MEMCPY,
    ACTIVITY_KERNEL,
    ACTIVITY_MARKER,
    SYNC_POINT,
    NVML_SAMPLE
};

enum class MarkerFlag : uint8_t {
    UNKNOWN       = 0,
    INSTANTANEOUS = 1,
    START         = 2,
    END           = 3
};

constexpr size_t GAPPROF_NAME_LEN = 256;

struct ProfileEvent {
    char marker_name[GAPPROF_NAME_LEN]; // kernel/marker/API name, or a fixed literal for SYNC/NVML rows. see report for examples
    EventSource source;                 // event source struct
    bool is_push;                       // RUNTIME_CALLBACK/NVTX_CALLBACK only, true=enter/push, false=exit/pop
    uint64_t timestamp_ns;              // GPU-side (CUPTI) timestamp, 0 if not applicable
    uint64_t gpu_end_ns;                // ACTIVITY_MEMCPY/ACTIVITY_KERNEL only
    uint32_t correlation_id;            // CUPTI correlation ID linking a runtime call to its GPU op
    uint64_t bytes;                     // ACTIVITY_MEMCPY only
    uint8_t copy_kind;                  // ACTIVITY_MEMCPY only
    uint8_t src_kind;                   // ACTIVITY_MEMCPY only
    uint8_t dst_kind;                   // ACTIVITY_MEMCPY only
    MarkerFlag marker_flag;             // ACTIVITY_MARKER only
    uint32_t marker_id;                 // ACTIVITY_MARKER only
    uint64_t host_timestamp_ns;         // source-stamped host time, 0 if not applicable
    double power_mw_instant;            // NVML_SAMPLE only, instantaneous power draw
};

inline const char* source_to_string(EventSource s) {
    switch (s) {
        case EventSource::RUNTIME_CALLBACK: return "RUNTIME_CB";
        case EventSource::NVTX_CALLBACK:    return "NVTX_CB";
        case EventSource::ACTIVITY_MEMCPY:  return "ACTIVITY_MEMCPY";
        case EventSource::ACTIVITY_KERNEL:  return "ACTIVITY_KERNEL";
        case EventSource::ACTIVITY_MARKER:  return "ACTIVITY_MARKER";
        case EventSource::SYNC_POINT:       return "SYNC";
        case EventSource::NVML_SAMPLE:      return "NVML";
        default:                            return "UNKNOWN";
    }
}

} // namespace gapprof
