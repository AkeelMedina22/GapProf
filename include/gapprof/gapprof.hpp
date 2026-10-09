#pragma once

#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include "gapprof/events.hpp"
#include "gapprof/mpsc_queue.hpp"

namespace gapprof {
    class NvmlSampler;
    class CuptiTracker;

    class Profiler {
    public:
        static Profiler& get();
        void initialize();
        void shutdown();

        Profiler(const Profiler&) = delete;
        void operator=(const Profiler&) = delete;

        MpscQueue<ProfileEvent, 65536>* get_queue() const { return event_queue.get(); }

    private:
        Profiler();
        ~Profiler();

        std::vector<ProfileEvent> buffer;

        std::unique_ptr<NvmlSampler> nvml_sampler;
        std::unique_ptr<CuptiTracker> cupti_tracker;
        std::unique_ptr<MpscQueue<ProfileEvent, 65536>> event_queue;

        std::thread event_thread;
        std::atomic<bool> keep_consuming{false};
        void event_worker();

        std::atomic<bool> is_initialized{false};
        std::atomic<bool> is_shutdown{false};
    };

    inline void emit(const ProfileEvent& ev) {
        if (auto* q = Profiler::get().get_queue()) q->push(ev);
    }
} // namespace gapprof
