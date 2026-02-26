#pragma once
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>
#include <iostream>
#include <string.h>
#include <sys/ioctl.h>

// Wrapper for the syscall
static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

class PerfProfiler {
    struct Event {
        int fd;
        uint64_t id;
    };
    std::vector<Event> events;
    bool running = false;

public:
    // Your 6 specific features
    struct Features {
        long long cycles;       // PAPI_TOT_CYC
        long long instructions; // PAPI_TOT_INS
        long long l1_dcm;      // PAPI_L1_DCM
        long long l2_dcm;      // PAPI_L2_DCM
        long long l3_tcm;      // PAPI_L3_TCM
        long long br_msp;      // PAPI_BR_MSP
    };

    PerfProfiler() {
        // 1. Total Cycles
        add_event(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
        // 2. Total Instructions
        add_event(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
        // 3. L1 Data Cache Miss
        add_event(PERF_TYPE_HW_CACHE, 
                  (PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
        // 4. L2 Data Cache Miss (Generic approximation, architecture specific)
        // Note: Generic PERF doesn't always have a clean L2 event. 
        // This uses the generic LL (Last Level) if L2 is last, or specific raw codes if needed.
        // For standard x86/ARM, we often approximate or use raw codes. 
        // Here we use L1D Read Miss as a placeholder or you must lookup your CPU's raw code (e.g. Intel 0x24)
        add_event(PERF_TYPE_HW_CACHE, 
                  (PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)); 
        // 5. L3/LL Cache Miss
        add_event(PERF_TYPE_HW_CACHE, 
                  (PERF_COUNT_HW_CACHE_LL) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
        // 6. Branch Misprediction
        add_event(PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);
    }

    void start() {
        if (running) return;
        for (auto& e : events) {
            ioctl(e.fd, PERF_EVENT_IOC_RESET, 0);
            ioctl(e.fd, PERF_EVENT_IOC_ENABLE, 0);
        }
        running = true;
    }

    Features stop_and_read() {
        Features f = {0};
        if (!running) return f;
        
        std::vector<long long> counts;
        for (auto& e : events) {
            ioctl(e.fd, PERF_EVENT_IOC_DISABLE, 0);
            long long count;
            read(e.fd, &count, sizeof(long long));
            counts.push_back(count);
        }
        
        f.cycles = counts[0];
        f.instructions = counts[1];
        f.l1_dcm = counts[2];
        f.l2_dcm = counts[3];
        f.l3_tcm = counts[4];
        f.br_msp = counts[5];
        
        running = false;
        return f;
    }

private:
    void add_event(uint32_t type, uint64_t config) {
        struct perf_event_attr pe;
        memset(&pe, 0, sizeof(struct perf_event_attr));
        pe.type = type;
        pe.size = sizeof(struct perf_event_attr);
        pe.config = config;
        pe.disabled = 1;
        pe.exclude_kernel = 1; // User space only
        pe.exclude_hv = 1;

        // pid=0 (current thread), cpu=-1 (any cpu), group_fd=-1 (group leader)
        int fd = perf_event_open(&pe, 0, -1, -1, 0);
        if (fd == -1) {
             std::cerr << "Error opening perf event: " << config << std::endl;
        }
        events.push_back({fd, config});
    }
};