#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#ifdef USE_TRACY
#include "Tracy.hpp"
#endif

namespace gapprof {

// Bounded MPMC queue - Dmitry Vyukov
template <typename T, size_t Capacity>
class MpmcQueue {
    static_assert(Capacity >= 2, "capacity must be at least 2");
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "capacity must be a power of two so the index is a mask");

private:
    struct Slot {
        std::atomic<size_t> seq;
        T item;
    };

    static constexpr size_t MASK = Capacity - 1;

    Slot slots_[Capacity];

    alignas(64) std::atomic<size_t> head_{0};      // producers
    alignas(64) std::atomic<size_t> tail_{0};      // consumers
    alignas(64) std::atomic<uint64_t> dropped_{0}; // failed-push counter

public:
    MpmcQueue() {
        for (size_t i = 0; i < Capacity; ++i) {
            slots_[i].seq.store(i, std::memory_order_relaxed);
        }
    }

    bool push(const T& item) {
        #ifdef USE_TRACY
        ZoneScopedN("Gapprof:MPMC_Push");
        #endif
        Slot* s;
        size_t pos = head_.load(std::memory_order_relaxed);
        for (;;) {
            s = &slots_[pos & MASK];
            size_t seq = s->seq.load(std::memory_order_acquire);
            intptr_t dif = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);

            if (dif == 0) {
                if (head_.compare_exchange_weak(pos, pos + 1,
                                                std::memory_order_relaxed)) {
                    break;
                }
                // CAS failed: pos was refreshed with the current head, retry.
            } else if (dif < 0) {
                dropped_.fetch_add(1, std::memory_order_relaxed);
                return false; // full
            } else {
                pos = head_.load(std::memory_order_relaxed);
            }
        }
        // the slot is ours, no other producer can touch it until we publish
        s->item = item;
        s->seq.store(pos + 1, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        #ifdef USE_TRACY
        ZoneScopedN("Gapprof:MPMC_Pop");
        #endif
        Slot* s;
        size_t pos = tail_.load(std::memory_order_relaxed);
        for (;;) {
            s = &slots_[pos & MASK];
            size_t seq = s->seq.load(std::memory_order_acquire);
            intptr_t dif = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);

            if (dif == 0) {
                if (tail_.compare_exchange_weak(pos, pos + 1,
                                                std::memory_order_relaxed)) {
                    break;
                }
            } else if (dif < 0) {
                return false; // empty
            } else {
                pos = tail_.load(std::memory_order_relaxed);
            }
        }

        item = s->item;
        s->seq.store(pos + Capacity, std::memory_order_release);
        return true;
    }

    uint64_t dropped() const {
        return dropped_.load(std::memory_order_relaxed);
    }
};

} // namespace gapprof