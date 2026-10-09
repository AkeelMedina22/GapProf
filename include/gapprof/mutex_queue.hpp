#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace gapprof {
template <typename T, size_t Capacity>
class MutexQueue {
public:
    bool push(const T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (count_ == Capacity) {
            dropped_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        buffer_[head_] = item;
        head_ = (head_ + 1) % Capacity;
        ++count_;
        return true;
    }

    bool pop(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (count_ == 0) {
            return false;
        }
        item = buffer_[tail_];
        tail_ = (tail_ + 1) % Capacity;
        --count_;
        return true;
    }

    uint64_t dropped() const {
        return dropped_.load(std::memory_order_relaxed);
    }

private:
    T buffer_[Capacity];
    size_t head_ = 0;
    size_t tail_ = 0;
    size_t count_ = 0;
    std::mutex mutex_;
    std::atomic<uint64_t> dropped_{0};
};

} // namespace gapprof