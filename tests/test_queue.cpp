#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <string>

#include "gapprof/mpsc_queue.hpp"

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "CHECK FAILED: " #cond << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)


void test_single_thread() {
    std::cout << "[Test 1] Running Single-Threaded Test..." << std::endl;
    gapprof::MpscQueue<int, 3> queue;
    int val = 0;
    CHECK(queue.pop(val) == false && "Queue should be empty");
    CHECK(queue.push(10) == true);
    CHECK(queue.push(20) == true);
    CHECK(queue.push(30) == true);
    CHECK(queue.push(40) == false && "Queue should be full and drop the 4th item");
    CHECK(queue.dropped() == 1 && "Drop counter should record exactly one drop");
    CHECK(queue.pop(val) == true && val == 10 && "First out should be 10");
    CHECK(queue.pop(val) == true && val == 20 && "Second out should be 20");
    CHECK(queue.pop(val) == true && val == 30 && "Third out should be 30");
    CHECK(queue.pop(val) == false && "Queue should be empty again");
    std::cout << "  -> Passed!" << std::endl;
}

void test_multi_thread() {
    std::cout << "[Test 2] Running Multi-Thread+Producer Test..." << std::endl;
    const int NUM_PRODUCERS = 8;
    const int ITEMS_PER_PRODUCER = 10000;
    gapprof::MpscQueue<int, ITEMS_PER_PRODUCER*(NUM_PRODUCERS+2)> queue;
    std::atomic<bool> start_flag{false};
    std::atomic<bool> producers_done{false};
    std::atomic<long long> total_pushed_sum{0};
    long long total_popped_sum = 0;
    int items_popped = 0;

    auto producer_task = [&](int id) {
        while (!start_flag.load(std::memory_order_acquire)) { std::this_thread::yield(); }
        long long local_sum = 0;
        for (int i = 1; i <= ITEMS_PER_PRODUCER; ++i) {
            int payload = (id * 100000) + i;
            queue.push(payload); 
            local_sum += payload;
        }
        total_pushed_sum.fetch_add(local_sum, std::memory_order_relaxed);
    };

    auto consumer_task = [&]() {
        int val;
        while (!producers_done.load(std::memory_order_acquire)) {
            while (queue.pop(val)) {
                total_popped_sum += val;
                items_popped++;
            }
            std::this_thread::yield();
        }
    };

    std::vector<std::thread> producers;
    for (int i = 0; i < NUM_PRODUCERS; ++i) {
        producers.emplace_back(producer_task, i + 1);
    }
    std::thread consumer(consumer_task);
    start_flag.store(true, std::memory_order_release);
    for (auto& t : producers) { t.join(); }
    producers_done.store(true, std::memory_order_release); 
    consumer.join();

    CHECK(items_popped == (NUM_PRODUCERS * ITEMS_PER_PRODUCER));
    CHECK(total_popped_sum == total_pushed_sum.load());
    CHECK(queue.dropped() == 0);
    std::cout << "  -> Passed!" << std::endl;
}

int main() {
    std::cout << "Starting Tests for MPSC Lock Free Queue\n" << std::endl;

    test_single_thread();
    test_multi_thread();

    std::cout << "All Tests Successful\n" << std::endl;

    return 0;
}