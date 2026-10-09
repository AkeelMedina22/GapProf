#include "gapprof/writer.hpp"
#include <iostream>

namespace {

void write_escaped(std::ostream& os, const char* s) {
    os << '"';
    for (const char* p = s; *p != '\0'; ++p) {
        if (*p == '"') os << '"'; 
        os << *p;
    }
    os << '"';
}

const char* event_type_string(const gapprof::ProfileEvent& ev) {
    using gapprof::EventSource;
    using gapprof::MarkerFlag;
    switch (ev.source) {
        case EventSource::RUNTIME_CALLBACK:
        case EventSource::NVTX_CALLBACK:
            return ev.is_push ? "START" : "END";
        case EventSource::ACTIVITY_MEMCPY:
        case EventSource::ACTIVITY_KERNEL:
            return "GPU_OP";
        case EventSource::ACTIVITY_MARKER:
            switch (ev.marker_flag) {
                case MarkerFlag::START:         return "START";
                case MarkerFlag::END:           return "END";
                case MarkerFlag::INSTANTANEOUS: return "INSTANT";
                default:                        return "UNKNOWN";
            }
        case EventSource::SYNC_POINT:
            return "SYNC_POINT";
        case EventSource::NVML_SAMPLE:
            return "SAMPLE";
        default:
            return "OTHER";
    }
}

} // anonymous namespace

namespace gapprof {

Writer::Writer(const std::string& filename)
    : filename_(filename), file_(filename) {
    if (file_.is_open()) {
        file_ << "Event_Name,Event_Source,Event_Type,Correlation_ID,"
                 "GPU_Start_ns,GPU_End_ns,Host_Timestamp_ns,"
                 "Bytes,Copy_Kind,Src_Kind,Dst_Kind,"
                 "Energy_mJ,Power_mW_Legacy,Power_mW_Instant,"
                 "Marker_Flag,Marker_ID\n";
    } else {
        std::cerr << "[GapProf Error] Failed to open " << filename_ << " for write.\n";
    }
}

void Writer::write(const ProfileEvent& ev) {
    if (!file_.is_open()) return;

    write_escaped(file_, ev.marker_name);
    file_ << "," << source_to_string(ev.source)
          << "," << event_type_string(ev)
          << "," << ev.correlation_id
          << "," << ev.timestamp_ns
          << "," << ev.gpu_end_ns
          << "," << ev.host_timestamp_ns
          << "," << ev.bytes
          << "," << static_cast<int>(ev.copy_kind)
          << "," << static_cast<int>(ev.src_kind)
          << "," << static_cast<int>(ev.dst_kind)
          << "," << 0.0 // Energy_mJ (not used anymore due to severe latency)
          << "," << 0.0 // Power_mW_Legacy (not used anymore)
          << "," << ev.power_mw_instant
          << "," << static_cast<int>(ev.marker_flag)
          << "," << ev.marker_id << "\n";
    ++rows_written_;
}

void Writer::finish(uint64_t dropped_events) {
    if (file_.is_open()) {
        file_.close();
        std::cout << "[GapProf] Flushed " << rows_written_ << " records to " << filename_ << "\n";
    } else {
        std::cerr << "[GapProf Error] Failed to open " << filename_ << " for write.\n";
    }

    if (dropped_events > 0) {
        std::cerr << "[GapProf Warning] " << dropped_events << " event(s) were dropped (queue full).\n";
    } else {
        std::cout << "[GapProf] Zero events dropped.\n";
    }
}

} // namespace gapprof
