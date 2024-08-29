#pragma once

#include "header.hpp"
#include <mutex>
#include <list>
#include <atomic>
#include <string>

class ConfigManager {
  // When unit testing, we need the flexibility to tweak the various config fields individually
#if defined(CEDR_UNIT_TESTING)
public:
#else
private:
#endif
  unsigned int resource_array[(uint8_t) resource_type::NUM_RESOURCE_TYPES]{};

  bool cache_schedules;
  bool enable_queueing;
  bool use_papi;
  bool loosen_thread_permissions;
  bool fixed_periodic_injection;
  bool exit_when_idle;
  unsigned int total_num_of_resources;

#if defined(ROLLING_MEAN_EXEC_HISTORY)
  // For each (api, resource) pair, we have a pair of atomic counters
  // The first atomic counter is the cumulative time spent running that api on that resource
  // The second atomic counter is the number of those tasks that have run
  std::map<std::pair<api_types, resource_type>, 
           std::pair<std::atomic<uint64_t>, std::atomic<uint64_t>>> exec_history = {};
#endif

#if defined(ROLLING_MEDIAN_EXEC_HISTORY)
  size_t exec_history_limit;
  std::map<std::pair<api_types, resource_type>, std::deque<uint64_t>> exec_history = {};
  std::mutex exec_history_mtx[api_types::NUM_API_TYPES][resource_type::NUM_RESOURCE_TYPES] = {};
#endif

  uint64_t exec_times[api_types::NUM_API_TYPES][resource_type::NUM_RESOURCE_TYPES] = {0};

  // used by LUT
  unsigned int resource_count[resource_type::NUM_RESOURCE_TYPES] = {0};
  unsigned int resource_to_global_id[resource_type::NUM_RESOURCE_TYPES][10] = {0}; // Assuming max 10 resources per type.. Parameterize it.


  std::list<std::string> PAPI_Counters;

  std::string scheduler;

  // TODO: the logic for respecting this is not implemented
  unsigned long max_parallel_jobs;
  unsigned long random_seed;

#if defined(ENABLE_IL_ORACLE_LOGGING)
  ////////////////////////////////////
  // IL-Sched
  ////////////////////////////////////
  std::list<struct_ILOracle_task> task_list { };
  ////////////////////////////////////
#endif

  std::list<std::string> dash_binary_paths;

public:
  ConfigManager();
  ~ConfigManager();

  void parseConfig(const std::string &filename);

  unsigned int getTotalResources();

  unsigned int *getResourceArray();

  bool getCacheSchedules() const;
  bool getEnableQueueing() const;
  bool getUsePAPI() const;
  bool getLoosenThreadPermissions() const;
  bool getFixedPeriodicInjection() const;
  bool getExitWhenIdle() const;

#if !defined(STATIC_API_COSTS)
  void pushDashExecTime(const api_types api, const resource_type resource, uint64_t time);
#endif
  uint64_t getDashExecTime(const api_types api, const resource_type resource);

  std::list<std::string> &getPAPICounters();

  unsigned int getResourceCount(uint8_t resource);
  void setResourceCount(uint8_t resource, uint32_t resource_id);

  unsigned int getResourceToGlobalID(uint8_t resource,  uint32_t resource_id);
  void setResourceToGlobalID(uint8_t resource,  uint32_t resource_id, uint32_t global_idx);

  std::string &getScheduler();
  unsigned long getMaxParallelJobs() const;
  unsigned long getRandomSeed() const;

#if defined(ENABLE_IL_ORACLE_LOGGING)
  ////////////////////////////////////
  // IL-Sched
  ////////////////////////////////////
  void pushILOracle(struct_ILOracle_task task_info);
  std::list<struct_ILOracle_task> getILOracleTaskList();
  ////////////////////////////////////  
#endif  

  std::list<std::string> &getDashBinaryPaths();

};
