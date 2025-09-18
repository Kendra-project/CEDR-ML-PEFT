#pragma once

#include <plog/Log.h>
#include <cstdint>
#include <map>
#include <string>

#if defined(USEPERF)
#include <linux/perf_event.h>    /* Definition of PERF_* constants */
#include <linux/hw_breakpoint.h> /* Definition of HW_* constants */
#include <sys/syscall.h>         /* Definition of SYS_* constants */
#include <unistd.h>
#endif

namespace PerfMon {
enum {
  // 0 is not specified here and is used for the default logger instance
  LoggerId = 1
};

  void initPerfLog(const std::string &filename = "perf_stats.csv");
} // namespace PerfMon

#if defined(USEPERF)
static std::vector<unsigned long> PERF_TYPES{
                                              /*0*/ PERF_TYPE_HARDWARE,
                                              /*1*/ PERF_TYPE_SOFTWARE,
                                              /*2*/ PERF_TYPE_TRACEPOINT,
                                              /*3*/ PERF_TYPE_HW_CACHE,
                                              /*4*/ PERF_TYPE_RAW,
                                              /*5*/ PERF_TYPE_BREAKPOINT};

/*Update following define and 3 vectors to add new perf event counters*/
/*Refer to this manual for more pref related information: https://man7.org/linux/man-pages/man2/perf_event_open.2.html*/
#define PERF_EVENT_NUM 9
static std::vector<int> perf_event_types{0, 0, 0, 0, 3, 3, 3, 3, 3}; // Indexed from PERF_TYPES
static std::vector<unsigned long> perf_events{
                                            PERF_COUNT_HW_INSTRUCTIONS,
                                            PERF_COUNT_HW_CACHE_REFERENCES,
                                            PERF_COUNT_HW_CACHE_MISSES,
                                            PERF_COUNT_HW_BRANCH_MISSES,
                                            (PERF_COUNT_HW_CACHE_DTLB) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16),
                                            (PERF_COUNT_HW_CACHE_ITLB) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16),
                                            (PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16),
                                            (PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16),
                                            (PERF_COUNT_HW_CACHE_L1I) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)
                                            };
static std::vector<std::string> perf_events_string{"INSTRUCTIONS", "CACHE_REFERENCES", "CACHE_MISSES", "BRANCH_MISSES", "DTLB_READ_MISS", "ITLB_READ_MISS", "L1D_READ_ACCESS", "L1D_READ_MISS", "L1I_READ_MISS"};

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,int cpu, int group_fd, unsigned long flags){
  return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

struct read_perf_event{
  uint64_t event_count;
  struct {
    long long perf_val;
    uint64_t id;
  } perf_vals [PERF_EVENT_NUM];
};

#endif
