#include "threads.hpp"
#include <dlfcn.h>
#include <sched.h>
#include <stdlib.h>
#include <linux/sched.h>
#include <sys/syscall.h>
#include <stdint.h>
#include <thread>
#include <unistd.h>
#include <fcntl.h>      // NEW: For file control options
#include <sys/stat.h>   // NEW: For mkfifo
#include <iostream>    // NEW: For debugging output
#include <signal.h> // Required for signal handling

#define gettid() syscall(__NR_gettid)

// NEW: Include your profiler
#include "PerfProfiler.hpp"

#if defined(USEPAPI)
#include "performance_monitor.hpp"
#include <papi/papi.h>
#endif

#include <plog/Log.h>
#include <string>
#include <sstream>      // NEW: For string formatting

#define ON_CPU 0
#define ON_ACC 1

struct sched_attr{
  uint32_t size;              /* Size of this structure */
  uint32_t sched_policy;      /* Policy (SCHED_*) */
  uint64_t sched_flags;       /* Flags */
  int32_t sched_nice;         /* Nice value (SCHED_OTHER, SCHED_BATCH) */
  uint32_t sched_priority;    /* Static priority (SCHED_FIFO, SCHED_RR) */
  /* Remaining fields are for SCHED_DEADLINE */
  uint64_t sched_runtime;
  uint64_t sched_deadline;
  uint64_t sched_period;
};

int sched_setattr(pid_t pid, const struct sched_attr *attr, unsigned int flags){
  return syscall(__NR_sched_setattr, pid, attr, flags);
}

// NEW: Helper to ensure pipe exists
const char* PIPE_PATH = "/tmp/cedr_features_pipe";

void *hardware_thread(void *ptr) {
  pthread_t self = pthread_self();
  pid_t self_tid = gettid();
  clockid_t clock_id;
  if (pthread_getcpuclockid(self, &clock_id) != 0) {
    LOG_FATAL << "Not able to get CLOCK ID";
    exit(1);
  }

  auto *thread_arg = (pthread_arg *)ptr;
  auto *worker_thread = thread_arg->thread;
  auto *thread_lock = thread_arg->thread_lock;
  auto *cedr_config = worker_thread->cedr_config;

  void *run_args[MAX_ARGS];
  unsigned long long last_busy = 0, last_avail = 0, total_idle_time = 0;

  int cpu_name = sched_getcpu();
  LOG_DEBUG << "Starting thread " << self << " as resource name " << worker_thread->resource_name << " and type " << resource_type_names[(uint8_t) worker_thread->thread_resource_type]
            << " on cpu id " << cpu_name;

  // ---------------------------------------------------------
  // NEW: Initialize PerfProfiler and Pipe
  // ---------------------------------------------------------
  // 1. Safety: Ignore SIGPIPE. 
  // If Python closes the pipe, we don't want CEDR to crash on the next write.
  signal(SIGPIPE, SIG_IGN);

  // ... (setup code remains the same) ...

  PerfProfiler profiler; // Creates the 6 events for this thread
  
  // // Create pipe if it doesn't exist (safe if already exists)
  mkfifo(PIPE_PATH, 0666);
  
  // // Open pipe for non-blocking write. 
  // // We re-check this inside the loop in case the Python reader restarts.
  // int pipe_fd = open(PIPE_PATH, O_WRONLY | O_NONBLOCK);
  int pipe_fd = open(PIPE_PATH, O_WRONLY);
  // int pipe_fd = -1; // Initialize to -1 (not opened yet)
  const char* LOG_FILE_PATH = "cedr_metrics.csv";
  
  // O_CREAT: Create file if missing
  // O_WRONLY: Write only
  // O_APPEND: Add to the end (essential for multiple threads!)
  int log_fd = open(LOG_FILE_PATH, O_CREAT | O_WRONLY | O_APPEND, 0666);

  if (log_fd == -1) {
      LOG_ERROR << "Failed to open metrics log file!";
  }
  // ---------------------------------------------------------

#if defined(USEPAPI)
  int papiRet, papiEventSet = PAPI_NULL;
  long long papiValues[cedr_config->getPAPICounters().size()];

  if (cedr_config->getUsePAPI() && PAPI_is_initialized()) {
    PAPI_create_eventset(&papiEventSet);

    for (const auto &papiEvent : cedr_config->getPAPICounters()) {
      if ((papiRet = PAPI_add_named_event(papiEventSet, papiEvent.c_str())) != PAPI_OK) {
        LOG_ERROR << "PAPI error " << papiRet << ": " << std::string(PAPI_strerror(papiRet));
      }
    }
  }
#endif

  LOG_VERBOSE << "Entering loop for thread " << self;
  while (true) {
    pthread_mutex_lock(thread_lock);
    if ((!worker_thread->todo_task_dequeue.empty())) {
      auto *task = worker_thread->todo_task_dequeue.front();
      worker_thread->todo_task_dequeue.pop_front();
      worker_thread->task = task;
      worker_thread->resource_state = 1;
      const std::vector<void *> &args = task->args;
      void *task_run_func = task->actual_run_func;
      pthread_mutex_unlock(thread_lock);

      void (*run_func)(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, 
                       void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
      *reinterpret_cast<void **>(&run_func) = task_run_func;

      if (args.size() >= MAX_ARGS) {
        LOG_ERROR << "Task " << worker_thread->task->task_name << " has too many arguments.";
      } else {
        for (unsigned int idx = 0; idx < MAX_ARGS; idx++) {
          if (idx < args.size()) {
            run_args[idx] = task->args.at(idx);
          } else if (idx == args.size() && worker_thread->thread_resource_type != resource_type::cpu) {
            run_args[idx] = (void *) worker_thread->resource_cluster_idx;
          }
        }

        LOG_VERBOSE << "About to dispatch " << worker_thread->task->task_name;

        // -----------------------------------------------------
        // NEW: Start Profiling
        // -----------------------------------------------------
        profiler.start();
        // -----------------------------------------------------
        std::cout << "Started profiling for task: " << worker_thread->task->task_name << std::endl; // For debugging

        clock_gettime(CLOCK_MONOTONIC_RAW, &(worker_thread->task->start));
        last_busy = (worker_thread->task->start.tv_sec * SEC2NANOSEC) + (worker_thread->task->start.tv_nsec);

        // --- EXECUTE TASK ---
        (run_func)(run_args[0], run_args[1], run_args[2], run_args[3], run_args[4], run_args[5], run_args[6],
                   run_args[7], run_args[8], run_args[9], run_args[10], run_args[11], run_args[12], run_args[13], 
                   run_args[14], run_args[15], run_args[16], run_args[17], run_args[18], run_args[19]);
        // --------------------

        // -----------------------------------------------------
        // NEW: Stop Profiling & Send Data
        // -----------------------------------------------------
        auto stats = profiler.stop_and_read();
        
        if (log_fd != -1) {
          std::stringstream ss;
          // Optional: Add Thread ID or Task Name to identify who wrote what
          // ss << worker_thread->resource_name << "," << worker_thread->task->task_name << ","; 
        
          ss << stats.cycles << "," << stats.instructions << "," 
            << stats.l1_dcm << "," << stats.l2_dcm << "," 
            << stats.l3_tcm << "," << stats.br_msp << "\n";
            
          std::string data = ss.str();
          
          // write() is atomic for small buffers (POSIX standard)
          // enabling O_APPEND ensures threads don't overwrite each other
          write(log_fd, data.c_str(), data.size());
      }
        // Re-open pipe if needed (e.g. if Python reader wasn't ready earlier)
        if (pipe_fd == -1) {
             pipe_fd = open(PIPE_PATH, O_WRONLY | O_NONBLOCK);
        }

        std::cout << "Stopped profiling for task: " << worker_thread->task->task_name << std::endl; // For debugging  

// 3. Write Data (Blocking)
        // If pipe_fd is valid, we write. If the buffer is full, 
        // this line will now WAIT until Python reads some data.
        if (pipe_fd != -1) {
             std::stringstream ss;
             ss << stats.cycles << "," << stats.instructions << "," 
                << stats.l1_dcm << "," << stats.l2_dcm << "," 
                << stats.l3_tcm << "," << stats.br_msp << "\n";
             
             std::string data = ss.str();
             
             // Check write result to re-open if connection was lost
             if (write(pipe_fd, data.c_str(), data.size()) == -1) {
                 // If write fails (e.g. Python restarted), close and try to reopen
                 close(pipe_fd);
                 // Blocking open again - waits for new Python connection
                 pipe_fd = open(PIPE_PATH, O_WRONLY); 
                 // Try write one more time (optional)
                 if (pipe_fd != -1) write(pipe_fd, data.c_str(), data.size());
             }
        }
        // -----------------------------------------------------

        cedr_barrier_t *barrier = task->kernel_barrier;
        pthread_mutex_lock(barrier->mutex);
        (*(barrier->completion_ctr))++;
        pthread_cond_signal(barrier->cond);
        pthread_mutex_unlock(barrier->mutex);
        clock_gettime(CLOCK_MONOTONIC_RAW, &(worker_thread->task->end));
        last_avail = (worker_thread->task->end.tv_sec * SEC2NANOSEC) + (worker_thread->task->end.tv_nsec);
      }

      LOG_VERBOSE << "Successfully executed " << worker_thread->task->task_name;
      pthread_mutex_lock(thread_lock);
      worker_thread->completed_task_dequeue.push_back(task);
      worker_thread->resource_state = 0;
      pthread_mutex_unlock(thread_lock);
    } else {
      if (worker_thread->resource_state == 3) {
        pthread_mutex_unlock(thread_lock);
        break;
      }
      pthread_mutex_unlock(thread_lock);
      last_avail = 0;
      last_busy = 0;
    }
    sched_yield();
  }
  
  // Cleanup pipe fd
  if (pipe_fd != -1) close(pipe_fd);
  // End of function
  if (log_fd != -1) close(log_fd);

  return nullptr;
}

// ... (Rest of the file: spawn_worker_thread, initializeThreads, etc. remains unchanged) ...

void spawn_worker_thread(ConfigManager &cedr_config, pthread_t *pthread_handles, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex,
                         uint8_t res_type, uint32_t global_idx, uint32_t cluster_idx) {
  const unsigned int processor_count = std::thread::hardware_concurrency();
  int ret;

  auto *thread_argument = (pthread_arg *) calloc(1, sizeof(pthread_arg));  
  
  hardware_thread_handle[global_idx].task = nullptr;
  hardware_thread_handle[global_idx].resource_state = 0;

  hardware_thread_handle[global_idx].resource_name = resource_type_names[res_type] + std::to_string(cluster_idx + 1);
  hardware_thread_handle[global_idx].thread_resource_type = (resource_type) res_type;
  hardware_thread_handle[global_idx].resource_cluster_idx = cluster_idx;
  hardware_thread_handle[global_idx].thread_avail_time = 0; 

  hardware_thread_handle[global_idx].cedr_config = &cedr_config;

  pthread_mutex_init(&(resource_mutex[global_idx]), nullptr);
  thread_argument->thread = &(hardware_thread_handle[global_idx]);
  thread_argument->thread_lock = &(resource_mutex[global_idx]);

  ret = pthread_create(&(pthread_handles[global_idx]), nullptr, hardware_thread, (void *) thread_argument);
  if (ret != 0) {
    LOG_FATAL << "Worker thread creation failed for thread " << global_idx+1;
    exit(1);
  }

  if (processor_count > 1) {
    int cpu_idx = 1 + (global_idx % (processor_count-1));
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(cpu_idx, &cpu_set);
    pthread_setaffinity_np(pthread_handles[global_idx], sizeof(cpu_set_t), &cpu_set);
  }
  
  if (!cedr_config.getLoosenThreadPermissions() && processor_count > 1) {
    struct sched_param sp;
    sp.sched_priority = 99;
    ret = pthread_setschedparam(pthread_handles[global_idx], SCHED_RR, &sp);
    if (ret != 0) {
      LOG_FATAL << "Unable to set pthread scheduling policy.";
      exit(1);
    }
  }
}

void initializeThreads(ConfigManager &cedr_config, pthread_t *resource_handle, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex) { 
  pthread_t current_thread = pthread_self();
  cpu_set_t scheduler_affinity;
  CPU_ZERO(&scheduler_affinity);
  CPU_SET(0, &scheduler_affinity);
  pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &scheduler_affinity);
  
  uint32_t global_idx = 0;
  const unsigned int processor_count = std::thread::hardware_concurrency();

  for (uint8_t res_type = 0; res_type < (uint8_t) resource_type::NUM_RESOURCE_TYPES; res_type++) {
    const unsigned int workers_to_spawn = cedr_config.getResourceArray()[res_type];
    for (uint32_t cluster_idx = 0; cluster_idx < workers_to_spawn; cluster_idx++) {
      spawn_worker_thread(cedr_config, resource_handle, hardware_thread_handle, resource_mutex, res_type, global_idx, cluster_idx);
      global_idx++;
    }
  }
}

void cleanupThreads(ConfigManager &cedr_config) { LOG_WARNING << "CleanupHardware not implemented"; }