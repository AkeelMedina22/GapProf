#include "gapprof/nvml.hpp"
#include "gapprof/error.hpp"
#include "gapprof/config.hpp"
#include "gapprof/gapprof.hpp"
#include <chrono>
#include <cstring>
#include <iostream>

namespace gapprof {

constexpr int FIELD_IDX_POWER_INSTANT = 0;
constexpr int FIELD_COUNT             = 1;

NvmlSampler::NvmlSampler(){
    // Init NVML
    GAPPROF_NVML_LOG(nvmlInit());
    GAPPROF_NVML_LOG(nvmlDeviceGetHandleByIndex(0, &device)); // single-GPU

    power_scope_id = static_cast<unsigned int>(config::get_env_int("GAPPROF_POWER_SCOPE", 0));

    nvmlFieldValue_t probe[1] = {};
    probe[0].fieldId = NVML_FI_DEV_POWER_INSTANT;
    probe[0].scopeId = power_scope_id;

    nvmlReturn_t probe_status = nvmlDeviceGetFieldValues(device, 1, probe);
    if (probe_status == NVML_SUCCESS && probe[0].nvmlReturn == NVML_SUCCESS) {
        field_api_supported = true;
        std::cout << "[GapProf] NVML field-ID API: POWER_INSTANT available (scopeId="
                << power_scope_id << ")\n";
    } else {
        std::cerr << "[GapProf] NVML field-ID API not available ("
                << nvmlErrorString(probe_status) << ", field="
                << nvmlErrorString(probe[0].nvmlReturn)
                << "). Warning: Hot loop legacy fallback removed. No telemetry will be recorded.\n";
        field_api_supported = false;
        if (power_scope_id != 0) {
            std::cerr << "[GapProf Error] GAPPROF_POWER_SCOPE=" << power_scope_id
                    << " requested but field-ID API unavailable. "
                    << "Module-scope power requires driver R535+ on Grace hardware.\n";
        }
    }

    // Capture a baseline energy reading to calculate the total session energy (Executed once)
    GAPPROF_NVML_LOG(nvmlDeviceGetTotalEnergyConsumption(device, &start_energy_mj));
}

NvmlSampler::~NvmlSampler() {
    stop();
}

void NvmlSampler::start() {
    keep_polling.store(true, std::memory_order_relaxed);
    poll_thread = std::thread(&NvmlSampler::worker, this);
}

void NvmlSampler::stop() {
    bool was_running = keep_polling.exchange(false, std::memory_order_relaxed);
    if (!was_running) return;

    if (poll_thread.joinable()) {
        poll_thread.join();
    }
    unsigned long long end_energy_mj = 0;
    // Captured once on teardown for cross-validation
    GAPPROF_NVML_LOG(nvmlDeviceGetTotalEnergyConsumption(device, &end_energy_mj));
    std::cout << "\n[GapProf] Session Complete. Total Energy: "
              << (end_energy_mj - start_energy_mj) << " mJ\n";
    GAPPROF_NVML_LOG(nvmlShutdown());
}

void NvmlSampler::worker() {
    int poll_us = config::get_poll_interval_us();

    nvmlFieldValue_t fields[FIELD_COUNT] = {};
    fields[FIELD_IDX_POWER_INSTANT].fieldId = NVML_FI_DEV_POWER_INSTANT;
    fields[FIELD_IDX_POWER_INSTANT].scopeId = power_scope_id;

    while (keep_polling.load(std::memory_order_relaxed)) {
        double power_mw_instant_d = 0.0;
        bool valid_sample = false;
        uint64_t host_now = 0;

        if (field_api_supported) {
            auto t1 = std::chrono::steady_clock::now().time_since_epoch();
            nvmlReturn_t status_fields = nvmlDeviceGetFieldValues(device, FIELD_COUNT, fields);
            auto t2 = std::chrono::steady_clock::now().time_since_epoch();
            host_now = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 + (t2 - t1) / 2).count();

            if (status_fields == NVML_SUCCESS &&
                fields[FIELD_IDX_POWER_INSTANT].nvmlReturn == NVML_SUCCESS) {

                const auto& fv = fields[FIELD_IDX_POWER_INSTANT];
                switch (fv.valueType) {
                    case NVML_VALUE_TYPE_UNSIGNED_INT:
                        power_mw_instant_d = static_cast<double>(fv.value.uiVal);
                        valid_sample = true;
                        break;
                    case NVML_VALUE_TYPE_UNSIGNED_LONG:
                        power_mw_instant_d = static_cast<double>(fv.value.ulVal);
                        valid_sample = true;
                        break;
                    case NVML_VALUE_TYPE_UNSIGNED_LONG_LONG:
                        power_mw_instant_d = static_cast<double>(fv.value.ullVal);
                        valid_sample = true;
                        break;
                    case NVML_VALUE_TYPE_DOUBLE:
                        power_mw_instant_d = fv.value.dVal;
                        valid_sample = true;
                        break;
                    default:
                        break;
                }
            }
        }

        if (valid_sample) {
            ProfileEvent rec{};
            rec.source = EventSource::NVML_SAMPLE;
            std::strncpy(rec.marker_name, "POWER_POLL", GAPPROF_NAME_LEN - 1);
            rec.host_timestamp_ns = host_now;
            rec.power_mw_instant = power_mw_instant_d;

            emit(rec);
        }

        std::this_thread::sleep_for(std::chrono::microseconds(poll_us));
    }
}

} // namespace gapprof
