#pragma once

#include <cupti.h>
#include <vector>
#include <string>
#include <mutex>
#include <thread>
#include <atomic>

namespace gapprof {

extern std::atomic<int> active_buffers;

class CuptiTracker {
public:
    CuptiTracker();
    ~CuptiTracker();

    void start();
    void stop();
    void emit_sync_point(const char* marker_name);

private:

    CUpti_SubscriberHandle subscriber = nullptr;

    std::thread sync_thread;
    std::atomic<bool> keep_syncing{false};
    void sync_worker();

    static void CUPTIAPI runtimeCallback(void* userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void* cbdata);
};
} // namespace gapprof
