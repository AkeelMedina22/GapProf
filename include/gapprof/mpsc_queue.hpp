#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#ifdef USE_TRACY
#include "Tracy.hpp"
#endif

namespace gapprof {

// Bounded MPSC queue - Dmitry Vyukov
template <typename T, size_t Capacity>
class MpscQueue {
private:
    struct Slot {
        std::atomic<size_t> seq;
        T item;
    };

    Slot slots_[Capacity];

    alignas(64) std::atomic<size_t> head_{0};      // producers
    alignas(64) std::atomic<size_t> tail_{0};      // single consumer
    alignas(64) std::atomic<uint64_t> dropped_{0}; // failed-push counter

public:
    MpscQueue() {
        for (size_t i = 0; i < Capacity; ++i) {
            slots_[i].seq.store(i, std::memory_order_relaxed);
        }
    }

    bool push(const T& item) {
        #ifdef USE_TRACY
        ZoneScopedN("Gapprof:MSPC_Push");
        #endif
        size_t pos = head_.load(std::memory_order_relaxed);
        for (;;) {
            Slot& s = slots_[pos % Capacity];
            size_t seq = s.seq.load(std::memory_order_acquire);
            intptr_t dif = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);

            if (dif == 0) {
                if (head_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    s.item = item;
                    s.seq.store(pos + 1, std::memory_order_release);
                    return true;
                }
                // CAS failed: pos was refreshed with the current head, retry.
            } else if (dif < 0) {
                dropped_.fetch_add(1, std::memory_order_relaxed);
                return false; // full
            } else {
                pos = head_.load(std::memory_order_relaxed);
            }
        }
    }

    bool pop(T& item) { // single consumer only
        #ifdef USE_TRACY
        ZoneScopedN("Gapprof:MSPC_Pop");
        #endif
        size_t pos = tail_.load(std::memory_order_relaxed);
        Slot& s = slots_[pos % Capacity];
        size_t seq = s.seq.load(std::memory_order_acquire);
        intptr_t dif = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
        if (dif < 0) return false; // empty

        item = s.item;
        s.seq.store(pos + Capacity, std::memory_order_release); 
        tail_.store(pos + 1, std::memory_order_relaxed);
        return true;
    }

    uint64_t dropped() const {
        return dropped_.load(std::memory_order_relaxed);
    }
};

} // namespace gapprof
