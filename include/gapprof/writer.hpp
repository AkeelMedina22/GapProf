#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include "gapprof/events.hpp"

namespace gapprof {

class Writer {
public:
    explicit Writer(const std::string& filename);

    void write(const ProfileEvent& ev);
    void finish(uint64_t dropped_events);

private:
    std::string filename_;
    std::ofstream file_;
    uint64_t rows_written_ = 0;
};

} // namespace gapprof
