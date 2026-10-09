#include <iostream>
#include <chrono>
#include <thread>
#include "nvml.h"

void run_monitor(nvmlDevice_t device, bool use_field_api, int sleep_interval_ms) {
    unsigned int lastPower = 0;
    int changesCaptured = 0;
    
    nvmlFieldValue_t fieldValues[1];
    if (use_field_api) {
        fieldValues[0].fieldId = NVML_FI_DEV_POWER_INSTANT;
        fieldValues[0].scopeId = 0;
    }

    auto lastChangeTime = std::chrono::high_resolution_clock::now();

    while (changesCaptured < 15) {
        unsigned int currentPower = 0;
        bool sampleValid = false;

        if (use_field_api) {
            nvmlReturn_t result = nvmlDeviceGetFieldValues(device, 1, fieldValues);
            if (result == NVML_SUCCESS && fieldValues[0].nvmlReturn == NVML_SUCCESS) {
                currentPower = fieldValues[0].value.uiVal;
                sampleValid = true;
            }
        } else {
            nvmlReturn_t result = nvmlDeviceGetPowerUsage(device, &currentPower);
            if (result == NVML_SUCCESS) {
                sampleValid = true;
            }
        }

        if (sampleValid) {
            if (lastPower != 0 && currentPower != lastPower) {
                auto now = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> delta = now - lastChangeTime;
                
                if (delta.count() > 0.1) {
                    std::cout << "  Value changed: " << lastPower << "mW -> " << currentPower 
                              << "mW | Real Delta: " << delta.count() << " ms\n";
                    changesCaptured++;
                    lastChangeTime = now;
                }
            }
            lastPower = currentPower;
        }

        if (sleep_interval_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_interval_ms));
        }
    }
}

int main() {
    const int sleep_interval_ms = 15;

    nvmlInit();
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(0, &device);

    std::cout << "Monitoring via nvmlDeviceGetPowerUsage()\n";
    run_monitor(device, false, sleep_interval_ms);

    std::cout << "Monitoring via NVML_FI_DEV_POWER_INSTANT\n";
    run_monitor(device, true, sleep_interval_ms);

    nvmlShutdown();
    return 0;
}