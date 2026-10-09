#pragma once

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace gapprof {
    namespace config {
        // read integer env variables with a fallback default
        inline int get_env_int(const char* name, int default_value) {
            if (const char* env_p = std::getenv(name)) {
                try {
                    return std::stoi(env_p);
                } catch (const std::exception&) {
                    std::cerr << "[GapProf Warning] " << name << "=\"" << env_p
                              << "\" is not a valid integer; using default "
                              << default_value << ".\n";
                    return default_value;
                }
            }
            return default_value;
        }

        inline int get_poll_interval_us() {
            return get_env_int("GAPPROF_POLL_US", 10000);
        }

        inline std::string get_output_filename() {
            if (const char* env_p = std::getenv("GAPPROF_OUTPUT_CSV")) {
                return std::string(env_p);
            }
            return "gapprof_results.csv";
        }

    } // namespace config
} // namespace gapprof
