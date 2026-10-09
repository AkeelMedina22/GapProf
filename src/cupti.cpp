/*
CUPTI tutorial:
https://eunomia.dev/others/cupti-tutorial/callback_event/
*/


#include "gapprof/cupti.hpp"
#include "gapprof/error.hpp"
#include "gapprof/gapprof.hpp"
#include "gapprof/events.hpp"
#include <chrono>
#include <iostream>
#include <cstring>
#include <vector>
#include <cupti.h>
#include <nvtx3/nvToolsExt.h>
#include <generated_nvtx_meta.h>

namespace {

// CUPTI assigns a distinct cbid to every cudaMemcpy* variant
// e.g. CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020 = 41, from
// cupti_runtime_cbid.h as per https://docs.nvidia.com/cupti/12.9/main/main.html
// Make a static table of memcpy cbid's to avoid doing string search every callback
const std::vector<bool>& memcpy_cbid_table() {
    static const std::vector<bool> table = [] {
        std::vector<bool> t(CUPTI_RUNTIME_TRACE_CBID_SIZE, false);
        for (uint32_t id = 0; id < CUPTI_RUNTIME_TRACE_CBID_SIZE; ++id) {
            const char* name = nullptr;
            if (cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, id, &name) == CUPTI_SUCCESS &&
                name != nullptr && strstr(name, "Memcpy") != nullptr) {
                t[id] = true;
            }
        }
        return t;
    }();
    return table;
}

const char* copy_kind_name(uint8_t kind) {
    switch (kind) {
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:    return "HtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:    return "DtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:    return "DtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:    return "HtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:    return "PtoP";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:    return "HtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:    return "AtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:    return "AtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:    return "AtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:    return "DtoA";
        default:                                 return "Unknown";
    }
}

gapprof::MarkerFlag classify_marker_flags(uint32_t flags) {
    if (flags & CUPTI_ACTIVITY_FLAG_MARKER_END)           return gapprof::MarkerFlag::END;
    if (flags & CUPTI_ACTIVITY_FLAG_MARKER_START)         return gapprof::MarkerFlag::START;
    if (flags & CUPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS) return gapprof::MarkerFlag::INSTANTANEOUS;
    return gapprof::MarkerFlag::UNKNOWN;
}

void safe_copy_name(char* dst, size_t dst_len, const char* src, const char* fallback) {
    const char* s = (src != nullptr) ? src : fallback;
    std::strncpy(dst, s, dst_len - 1);
    dst[dst_len - 1] = '\0';
}
} // anonymous namespace


namespace gapprof {

std::atomic<int> active_buffers{0};

static void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
    constexpr size_t BUF_SIZE = 8 * 1024 * 1024;  // 8 MB
    constexpr size_t ALIGN = 8;
    *size = BUF_SIZE;
    *buffer = static_cast<uint8_t*>(std::aligned_alloc(ALIGN, BUF_SIZE));
    *maxNumRecords = 0;
    if (!*buffer) {
        *size = 0;
        return;
    }
}

static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId,
                                     uint8_t* buffer, size_t size, size_t validSize) {
    CUpti_Activity* record = nullptr;
    active_buffers.fetch_add(1, std::memory_order_acq_rel);

    int records_in_buffer = 0;
    while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
        records_in_buffer++;
        switch (record->kind){
            case CUPTI_ACTIVITY_KIND_MEMCPY: {
                #ifdef USE_TRACY
                ZoneScopedN("CUPTI:Activity_Memcpy");
                #endif
                auto* m = reinterpret_cast<CUpti_ActivityMemcpy5*>(record);
                ProfileEvent ev{};
                ev.source = EventSource::ACTIVITY_MEMCPY;
                ev.timestamp_ns = m->start;
                ev.gpu_end_ns = m->end;
                ev.correlation_id = m->correlationId;
                ev.bytes = m->bytes;
                ev.copy_kind = m->copyKind;
                ev.src_kind = m->srcKind;
                ev.dst_kind = m->dstKind;
                snprintf(ev.marker_name, GAPPROF_NAME_LEN, "MEMCPY_%s", copy_kind_name(m->copyKind));
                emit(ev);
                break;
            }
            case CUPTI_ACTIVITY_KIND_KERNEL:
            case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
                #ifdef USE_TRACY
                ZoneScopedN("CUPTI:Activity_Concurrent_Kernel");
                #endif
                // Version 9 should be applicable for A100s and H200s.
                auto* k = reinterpret_cast<CUpti_ActivityKernel9*>(record);
                ProfileEvent ev{};
                ev.source = EventSource::ACTIVITY_KERNEL;
                ev.timestamp_ns = k->start;
                ev.gpu_end_ns = k->end;
                ev.correlation_id = k->correlationId;
                safe_copy_name(ev.marker_name, GAPPROF_NAME_LEN, k->name, "KERNEL_UNKNOWN");
                emit(ev);
                break;
            }

            case CUPTI_ACTIVITY_KIND_MARKER: {
                #ifdef USE_TRACY
                ZoneScopedN("CUPTI:Activity_Marker");
                #endif
                auto* mk = reinterpret_cast<CUpti_ActivityMarker2*>(record);
                ProfileEvent ev{};
                ev.source = EventSource::ACTIVITY_MARKER;
                ev.timestamp_ns = mk->timestamp;
                ev.marker_id = mk->id;
                ev.marker_flag = classify_marker_flags(mk->flags);
                safe_copy_name(ev.marker_name, GAPPROF_NAME_LEN, mk->name,
                               (ev.marker_flag == MarkerFlag::END) ? "NVTX_END" : "NVTX_UNKNOWN");
                emit(ev);
                break;
            }

            // Future Work (vLLM)
            case CUPTI_ACTIVITY_KIND_MARKER_DATA:
                break;

            // Future Work
            case CUPTI_ACTIVITY_KIND_DEVICE:
            case CUPTI_ACTIVITY_KIND_CONTEXT:
                break;
            default:
                break;
        }
    }
    std::free(buffer);
    active_buffers.fetch_sub(1, std::memory_order_acq_rel);
}

CuptiTracker::CuptiTracker(){
    GAPPROF_CUPTI_LOG(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)runtimeCallback, this), "Subscribe");
}

CuptiTracker::~CuptiTracker() {
    stop();
}

/* Background worker thread that periodically generates correlation markers.
Because the GPU and CPU clocks drift, we need periodic dual-timestamps to interpolate accurate host times. */
void CuptiTracker::sync_worker() {
    while (keep_syncing.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        emit_sync_point("SYNC_HEARTBEAT");
    }
}

// Called synchronously by CUDA driver whenever an NVTX hook occurs
void CUPTIAPI CuptiTracker::runtimeCallback(void* userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void* cbdata) {
    // Hardware Operations (PCIe memory transfers)
    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
        const auto& is_memcpy_cbid = memcpy_cbid_table();
        if (static_cast<uint32_t>(cbid) < is_memcpy_cbid.size() && is_memcpy_cbid[cbid]) {
            #ifdef USE_TRACY
            ZoneScopedN("CUPTI:Callback_Runtime_API");
            #endif
            auto cbInfo = (const CUpti_CallbackData*)cbdata;
            ProfileEvent event = {};
            event.source = EventSource::RUNTIME_CALLBACK;
            cuptiGetTimestamp(&event.timestamp_ns);
            safe_copy_name(event.marker_name, GAPPROF_NAME_LEN, cbInfo->functionName, "RUNTIME_UNKNOWN");
            event.is_push = (cbInfo->callbackSite == CUPTI_API_ENTER);
            event.correlation_id = cbInfo->correlationId;            emit(event);
        }
    }
    // NVTX
    else if (domain == CUPTI_CB_DOMAIN_NVTX) {
        auto nvtxInfo = (const CUpti_NvtxData*)cbdata;
        ProfileEvent event = {};
        event.source = EventSource::NVTX_CALLBACK;
        cuptiGetTimestamp(&event.timestamp_ns);

        if (cbid == CUPTI_CBID_NVTX_nvtxRangePushA) {
            #ifdef USE_TRACY
            ZoneScopedN("CUPTI:Callback_NVTX_Push");
            #endif
            auto params = (nvtxRangePushA_params*)nvtxInfo->functionParams;
            const char* msg = (params->message != nullptr) ? params->message : "NVTX_EVENT";
            safe_copy_name(event.marker_name, GAPPROF_NAME_LEN, msg, "NVTX_EVENT");
            event.is_push = true;
            emit(event);
        }
        else if (cbid == CUPTI_CBID_NVTX_nvtxRangePop) {
            #ifdef USE_TRACY
            ZoneScopedN("CUPTI:Callback_NVTX_Pop");
            #endif
            // Pop dosen't carry a name, assigned a generic name
            safe_copy_name(event.marker_name, GAPPROF_NAME_LEN, "NVTX_POP", "NVTX_POP");
            event.is_push = false;
            emit(event);
        }
    }
}

void CuptiTracker::start() {
    if (!subscriber) {
        std::cerr << "[GapProf Error] CUPTI subscription failed at construction; "
                     "telemetry disabled, workload will run unprofiled.\n";
        return;
    }

    (void)memcpy_cbid_table(); // build table once at startup

    GAPPROF_CUPTI_LOG(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API), "Enable Runtime API");
    GAPPROF_CUPTI_LOG(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_NVTX), "Enable NVTX");

    GAPPROF_CUPTI_LOG(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted), "Activity Callbacks");
    GAPPROF_CUPTI_LOG(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE), "Enable Device");
    GAPPROF_CUPTI_LOG(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT), "Enable Context");
    GAPPROF_CUPTI_LOG(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY), "Enable Memcpy");
    GAPPROF_CUPTI_LOG(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL), "Enable Concurrent Kernel");
    GAPPROF_CUPTI_LOG(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER), "Enable Marker");
    GAPPROF_CUPTI_LOG(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER_DATA), "Enable Marker Data");

    keep_syncing.store(true, std::memory_order_relaxed);
    sync_thread = std::thread(&CuptiTracker::sync_worker, this);
}

void CuptiTracker::stop() {
    bool was_running = keep_syncing.exchange(false, std::memory_order_relaxed);
    if (!was_running) return;

    if (sync_thread.joinable()) {
        sync_thread.join();
    }

    cuptiEnableDomain(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
    cuptiEnableDomain(0, subscriber, CUPTI_CB_DOMAIN_NVTX);
    cuptiUnsubscribe(subscriber);

    GAPPROF_CUPTI_LOG(cuptiActivityFlushAll(0), "Flush");
}

// Sample CPU and GPU clocks as close together as possible to establish correlation point
void CuptiTracker::emit_sync_point(const char* marker_name) {
    #ifdef USE_TRACY
    ZoneScopedN("CUPTI:Sync_Point");
    #endif
    uint64_t gpu_ts = 0;

    // Host -> GPU -> Host
    auto t1 = std::chrono::steady_clock::now().time_since_epoch();
    cuptiGetTimestamp(&gpu_ts);
    auto t2 = std::chrono::steady_clock::now().time_since_epoch();

    // Calculate median
    uint64_t cpu_ts = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 + (t2 - t1) / 2).count();

    ProfileEvent ev{};
    ev.source = EventSource::SYNC_POINT;
    safe_copy_name(ev.marker_name, GAPPROF_NAME_LEN, marker_name, "SYNC");
    ev.timestamp_ns = gpu_ts;
    ev.host_timestamp_ns = cpu_ts;

    emit(ev);
}

} // namespace gapprof
