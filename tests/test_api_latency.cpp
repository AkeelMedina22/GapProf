#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <nvml.h>

#define NVML_CHECK(call) \
    do { \
        nvmlReturn_t status = call; \
        if (status != NVML_SUCCESS) { \
            std::cerr << "NVML Error: " << nvmlErrorString(status) << std::endl; \
            exit(1); \
        } \
    } while(0)

void print_stats(const std::string& name, std::vector<double>& times) {
    if (times.empty()) {
        std::cerr << "Error: No data recorded for " << name << "\n";
        return;
    }

    std::sort(times.begin(), times.end());
    
    double p50 = times[times.size() * 0.50];
    double p90 = times[times.size() * 0.90];
    double p99 = times[times.size() * 0.99];

    std::cout << std::left << std::setw(30) << name 
              << " | p50: " << std::fixed << std::setprecision(5) << std::setw(7) << p50 << " ms"
              << " | p90: " << std::fixed << std::setprecision(5) << std::setw(7) << p90 << " ms"
              << " | p99: " << std::fixed << std::setprecision(5) << std::setw(7) << p99 << " ms"
              << " | Max: " << std::fixed << std::setprecision(5) << times.back() << " ms\n";
}

int main() {
    NVML_CHECK(nvmlInit());
    nvmlDevice_t device;
    NVML_CHECK(nvmlDeviceGetHandleByIndex(0, &device));

    const int ITERATIONS = 1000;
    std::vector<double> legacy_times, field_times, energy_times;

    std::cout << "Benchmarking NVML API Latency (" << ITERATIONS << " iterations)...\n";
    std::cout << std::string(80, '-') << "\n";
 
    // legacy power function
    unsigned int power = 0;
    for (int i = 0; i < ITERATIONS; ++i) { 
        nvmlDeviceGetPowerUsage(device, &power);
    }
    for (int i = 0; i < ITERATIONS; ++i) {
        power = 0;
        auto t1 = std::chrono::high_resolution_clock::now();
        nvmlDeviceGetPowerUsage(device, &power);
        auto t2 = std::chrono::high_resolution_clock::now();
        legacy_times.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
    }
    
    // field-ID power 
    nvmlFieldValue_t fields[1] = {};
    for (int i = 0; i < ITERATIONS; ++i) {
        fields[0].fieldId = NVML_FI_DEV_POWER_INSTANT;
    }
    nvmlDeviceGetFieldValues(device, 1, fields);
    for (int i = 0; i < ITERATIONS; ++i) {
        auto t1 = std::chrono::high_resolution_clock::now();
        nvmlDeviceGetFieldValues(device, 1, fields);
        auto t2 = std::chrono::high_resolution_clock::now();
        field_times.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
    }


    // energy
    unsigned long long energy = 0;
    for (int i = 0; i < ITERATIONS; ++i) {
        nvmlDeviceGetTotalEnergyConsumption(device, &energy);
    }
    for (int i = 0; i < ITERATIONS; ++i) {
        energy = 0;
        auto t1 = std::chrono::high_resolution_clock::now();
        nvmlDeviceGetTotalEnergyConsumption(device, &energy);
        auto t2 = std::chrono::high_resolution_clock::now();
        energy_times.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
    }

    print_stats("nvmlDeviceGetFieldValues", field_times);
    print_stats("nvmlDeviceGetPowerUsage", legacy_times);
    print_stats("nvmlDeviceGetTotalEnergy", energy_times);
    
    std::cout << std::string(80, '-') << "\n";
    NVML_CHECK(nvmlShutdown());
    return 0;
}