#include "gapprof/gapprof.hpp"
#include "gapprof/nvml.hpp"
#include "gapprof/cupti.hpp"
#include "gapprof/config.hpp"
#include "gapprof/error.hpp"
#include "gapprof/events.hpp"
#include "gapprof/mpsc_queue.hpp"
#include "gapprof/writer.hpp"
#include <chrono>
#include <iostream>

namespace gapprof {

// Thread-safe Meyers Singleton pattern
Profiler& Profiler::get() {
    static Profiler instance;
    return instance;
}

Profiler::Profiler() {
    buffer.reserve(100000);
}

Profiler::~Profiler() {
    // Fallback in case of early exit
    shutdown();
}

void Profiler::initialize() {
    bool expected = false;
    if (!is_initialized.compare_exchange_strong(expected, true)) return;

    // queue must be initialized before CUPTI
    event_queue = std::make_unique<MpscQueue<ProfileEvent, 65536>>();

    nvml_sampler = std::make_unique<NvmlSampler>();
    cupti_tracker = std::make_unique<CuptiTracker>();

    // first sync-point for clock interpolation
    cupti_tracker->emit_sync_point("CALIBRATION_START");

    keep_consuming.store(true, std::memory_order_relaxed);
    event_thread = std::thread(&Profiler::event_worker, this);

    cupti_tracker->start();
    nvml_sampler->start();

    std::cout << "[GapProf] Telemetry online. Writing to "
              << config::get_output_filename() << "\n";
}

void Profiler::shutdown() {
    bool expected = false;
    if (!is_shutdown.compare_exchange_strong(expected, true)) return;
    if (!is_initialized) return;

    // Force flush any GPU activity records
    cupti_tracker->stop();

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (active_buffers.load(std::memory_order_acquire) > 0) {
        if (std::chrono::steady_clock::now() > deadline) {
            std::cerr << "[GapProf] Warning: CUPTI Activity API Buffer Drain Timeout\n";
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // final sync-point before consumer is stopped
    if (cupti_tracker) {
        cupti_tracker->emit_sync_point("CALIBRATION_END");
    }

    // final nvml-sample before consumer is stopped
    if (nvml_sampler) nvml_sampler->stop();

    // consumer event threads stopped
    keep_consuming.store(false, std::memory_order_relaxed);
    if (event_thread.joinable()) {
        event_thread.join();
    }

    // flush to csv, report drops
    Writer writer(config::get_output_filename());
    for (const auto& r : buffer) {
        writer.write(r);
    }
    writer.finish(event_queue ? event_queue->dropped() : 0);
}

void Profiler::event_worker() {
    ProfileEvent event = {};
    int idle_spins = 0;

    while (keep_consuming.load(std::memory_order_relaxed)) {
        if (event_queue && event_queue->pop(event)) {
            buffer.push_back(event);
            idle_spins = 0;
        } else {
            if (idle_spins < 100) {
                idle_spins++;
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
        }
    }

    // Drain queue on exit
    while (event_queue && event_queue->pop(event)) buffer.push_back(event);
}

} // namespace gapprof
