#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "gapprof/mpsc_queue.hpp"
#include "gapprof/mpmc_queue.hpp"
#include "gapprof/mutex_queue.hpp"

namespace {

constexpr size_t NAME_LEN = 96;

// measured: mean 38544, max 40786
constexpr size_t MEASURED_BURST = 38544;
constexpr int MEASURED_BURSTS = 12;

struct BenchEvent {
    uint64_t timestamp_ns;
    uint64_t gpu_end_ns;
    uint64_t bytes;
    uint32_t correlation_id;
    uint32_t source;
    double   power_mw;
    char     marker_name[NAME_LEN];
};

// representative llama.cpp kernel names, cycled so the strncpy length varies
const char* const KERNEL_NAMES[] = {
    "mul_mat_vec_q",
    "flash_attn_ext_f16",
    "void mul_mat_vec_q<(ggml_type)8, 4, 4>(const void *, const void *, float *)",
    "MEMCPY_HtoD",
    "void flash_attn_ext_f16<128, 128, 32, 3, 128, (bool)1, (bool)0>(char *)",
    "rms_norm_f32",
};
constexpr int N_NAMES = sizeof(KERNEL_NAMES) / sizeof(KERNEL_NAMES[0]);

using clk = std::chrono::steady_clock;

inline double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

struct Result {
    double burst_s = 0;      // mean wall time of one burst
    double offered = 0;      // records/s during a burst
    double drained = 0;      // records/s the consumer achieved
    double drop_pct = 0;
    uint64_t dropped = 0;
    uint64_t popped = 0;
};

// nanoseconds of real work per record in bufferCompleted, measured
double g_record_ns = 0.0;
bool g_csv_consumer = true;

inline void spin_ns(double ns) {
    if (ns <= 0) return;
    auto t0 = clk::now();
    while (std::chrono::duration<double, std::nano>(clk::now() - t0).count() < ns) {}
}

template <typename Queue>
Result run(size_t burst, int n_bursts, std::FILE* out) {
    auto owned = std::make_unique<Queue>();
    Queue& queue = *owned;
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> popped{0};
    std::atomic<int> burst_id{0};
    std::atomic<int> drained_upto{-1};

    // consumer formats a csv row per record, as the daemon does
    auto consumer = [&]() {
        BenchEvent ev;
        char line[512];
        uint64_t n = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            while (queue.pop(ev)) {
                if (!g_csv_consumer) { ++n; continue; }
                int len = std::snprintf(line, sizeof(line),
                    "%s,%u,%u,%lu,%lu,%lu,%.1f\n",
                    ev.marker_name, ev.source, ev.correlation_id,
                    (unsigned long)ev.timestamp_ns,
                    (unsigned long)ev.gpu_end_ns,
                    (unsigned long)ev.bytes, ev.power_mw);
                std::fwrite(line, 1, size_t(len), out);
                ++n;
            }
            drained_upto.store(burst_id.load(std::memory_order_acquire),
                               std::memory_order_release);
        }
        while (queue.pop(ev)) ++n;
        popped.store(n, std::memory_order_relaxed);
    };

    std::thread cons(consumer);
    double total_burst_s = 0;

    for (int b = 0; b < n_bursts; ++b) {
        auto t0 = clk::now();
        BenchEvent ev{};
        for (size_t i = 0; i < burst; ++i) {
            // mirrors bufferCompleted: field copies then safe_copy_name
            ev.timestamp_ns = uint64_t(i);
            ev.gpu_end_ns = uint64_t(i) + 1500;
            ev.correlation_id = uint32_t(i);
            ev.source = uint32_t(i & 3);
            const char* name = KERNEL_NAMES[i % N_NAMES];
            std::strncpy(ev.marker_name, name, NAME_LEN - 1);
            ev.marker_name[NAME_LEN - 1] = '\0';
            spin_ns(g_record_ns);
            queue.push(ev);
        }
        total_burst_s += secs(t0, clk::now());

        // real bursts are seconds apart, so the queue drains fully between
        burst_id.store(b, std::memory_order_release);
        while (drained_upto.load(std::memory_order_acquire) < b) {
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }

    stop.store(true, std::memory_order_relaxed);
    cons.join();

    Result r;
    r.dropped = queue.dropped();
    r.popped = popped.load();
    r.burst_s = total_burst_s / n_bursts;
    double offered_total = double(burst) * n_bursts;
    r.offered = offered_total / total_burst_s;
    r.drained = double(r.popped) / total_burst_s;
    r.drop_pct = 100.0 * double(r.dropped) / offered_total;
    return r;
}

void header(const std::string& what) {
    std::cout << "\n  " << std::left << std::setw(12) << what
              << std::setw(12) << "queue"
              << std::right
              << std::setw(11) << "burst ms"
              << std::setw(13) << "offered M/s"
              << std::setw(13) << "drained M/s"
              << std::setw(11) << "dropped %" << "\n";
    std::cout << "  " << std::string(72, '-') << "\n";
}

void row(const std::string& label, const std::string& q, const Result& r) {
    std::cout << "  " << std::left << std::setw(12) << label
              << std::setw(12) << q << std::right << std::fixed
              << std::setw(11) << std::setprecision(2) << r.burst_s * 1000
              << std::setw(13) << std::setprecision(2) << r.offered / 1e6
              << std::setw(13) << std::setprecision(2) << r.drained / 1e6
              << std::setw(11) << std::setprecision(1) << r.drop_pct << "\n";
}

template <size_t Cap>
void both(const std::string& label, size_t burst, int n, std::FILE* out) {
    row(label, "mutex", run<gapprof::MutexQueue<BenchEvent, Cap>>(burst, n, out));
    row(label, "mpsc", run<gapprof::MpscQueue<BenchEvent, Cap>>(burst, n, out));
    row(label, "mpmc", run<gapprof::MpmcQueue<BenchEvent, Cap>>(burst, n, out));
}

} // namespace

int main(int argc, char** argv) {
    bool cap_sweep = false;
    const char* path = "/dev/null";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--capacity") cap_sweep = true;
        else if (a == "--csv" && i + 1 < argc) path = argv[++i];
        else if (a == "--record-ns" && i + 1 < argc) g_record_ns = std::atof(argv[++i]);
        else if (a == "--noop-consumer") g_csv_consumer = false;
    }

    std::FILE* out = std::fopen(path, "w");
    if (!out) { std::perror("fopen"); return 1; }
    static char iobuf[1 << 20];
    std::setvbuf(out, iobuf, _IOFBF, sizeof(iobuf));

    std::cout << "burst response benchmark\n"
              << "  payload        " << sizeof(BenchEvent) << " bytes\n"
              << "  consumer       csv row per record -> " << path << "\n"
              << "  producer work  " << g_record_ns << " ns per record"
              << (g_record_ns > 0 ? "\n" : "   <-- UNCALIBRATED, see --record-ns\n")
              << "  measured burst " << MEASURED_BURST
              << " records, " << MEASURED_BURSTS << " completions per run\n";

    if (!cap_sweep) {
        std::cout << "\n[1] burst size at the deployed capacity of 65536 slots\n";
        header("burst");
        both<4096>("1024", 1024, MEASURED_BURSTS, out);
        both<4096>("4096", 4096, MEASURED_BURSTS, out);
        both<4096>("16384", 16384, MEASURED_BURSTS, out);
        both<4096>("38544", MEASURED_BURST, MEASURED_BURSTS, out);
        both<4096>("65536", 65536, MEASURED_BURSTS, out);
    } else {
        std::cout << "\n[2] queue capacity at the measured burst of "
                  << MEASURED_BURST << " records\n";
        header("capacity");
        both<1024>("1024", MEASURED_BURST, MEASURED_BURSTS, out);
        both<4096>("4096", MEASURED_BURST, MEASURED_BURSTS, out);
        both<16384>("16384", MEASURED_BURST, MEASURED_BURSTS, out);
        both<65536>("65536", MEASURED_BURST, MEASURED_BURSTS, out);
    }

    std::fclose(out);
    std::cout << "\n";
    return 0;
}