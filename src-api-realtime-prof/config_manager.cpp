#include "config_manager.hpp"
#include <algorithm>
#include <nlohmann/json.hpp>
#include <fstream>

ConfigManager::ConfigManager() {
  // Populate the config with all of our "default" configuration options
#if defined(__aarch64__)
  resource_array[(uint8_t) resource_type::cpu] = 3;
  resource_array[(uint8_t) resource_type::fft] = 0;
  resource_array[(uint8_t) resource_type::mmult] = 0;
  resource_array[(uint8_t) resource_type::gpu] = 0;
#else
  resource_array[(uint8_t) resource_type::cpu] = 3;
  resource_array[(uint8_t) resource_type::fft] = 0;
  resource_array[(uint8_t) resource_type::mmult] = 0;
  resource_array[(uint8_t) resource_type::gpu] = 0;
#endif

  cache_schedules = false;
  enable_queueing = true;
  use_papi = false;
  use_perf = false;
  loosen_thread_permissions = true;
  fixed_periodic_injection = true;
  exit_when_idle = false;
  total_num_of_resources = 3;
  log_traces = true;
  periodic_logs = false;
  periodic_logging_threshold = 10000;
  logging_period = 1000000000; //in nanoseconds
  overwrite_periodic_log = true; // generates a single AD features file without accumlating the previous log periods
  app_logging_threshold = 100000000000; // 100 secconds, written in nanoseconds
  PAPI_Counters = {};

#if defined(ROLLING_MEDIAN_EXEC_HISTORY)
  exec_history_limit = 25;
#endif

  for (int api = 0; api < api_types::NUM_API_TYPES; api++) {
    for (int platform = 0; platform < resource_type::NUM_RESOURCE_TYPES; platform++) {
      exec_times[api][platform] = 0;
#if defined(ROLLING_MEDIAN_EXEC_HISTORY)
      exec_history[std::make_pair((api_types) api, (resource_type) platform)] = {};
#endif
    }
  }

  scheduler = "SIMPLE";
  max_parallel_jobs = 12;
  random_seed = 0;
  dash_binary_paths = {"./libdash-rt/libdash-rt.so"};
}

ConfigManager::~ConfigManager() = default;

void ConfigManager::parseConfig(const std::string &filename) {
  std::ifstream if_stream(filename);
  if (!if_stream.is_open()) {
    LOG_ERROR << "Failed to open configuration input file " << filename;
    return;
  }
  LOG_VERBOSE << "Successfully opened configuration input file " << filename;

  nlohmann::json j;
  if_stream >> j;
  if_stream.close();

  if (j.find("Worker Threads") != j.end()) {
    LOG_VERBOSE << "Overwriting default values for number of worker threads";
    total_num_of_resources = 0;
    auto threads_config = j["Worker Threads"];
    for (auto &resource_type_name : resource_type_names) {
      if (threads_config.contains(resource_type_name)) {
        unsigned int resource_count = threads_config[resource_type_name];
        LOG_DEBUG << "Config > Worker Threads contains key '" << resource_type_name << "', assigning config value to " << resource_count;
        resource_array[(uint8_t) resource_type_map.at(resource_type_name)] = resource_count;
        total_num_of_resources += resource_count;
      } else {
        LOG_DEBUG << "Config > Worker Threads does not contain key '" << resource_type_name << "', skipping...";
      }
    }
  }

  if (j.find("Features") != j.end()) {
    LOG_VERBOSE << "Overwriting default values for chosen features";
    auto features_config = j["Features"];

    if (features_config.contains("Cache Schedules")) {
      LOG_DEBUG << "Config > Features contains key 'Cache Schedules', assigning config value to " << (bool)features_config["Cache Schedules"];
      cache_schedules = features_config["Cache Schedules"];
    }
    if (features_config.contains("Enable Queueing")) {
      LOG_DEBUG << "Config > Features contains key 'Enable Queueing', assigning config value to " << (bool)features_config["Enable Queueing"];
      enable_queueing = features_config["Enable Queueing"];
    }
    if (features_config.contains("Use PAPI")) {
      LOG_DEBUG << "Config > Features contains key 'Use PAPI', assigning config value to " << (bool)features_config["Use PAPI"];
      use_papi = features_config["Use PAPI"];
    }
    if (features_config.contains("Use PERF")) {
      LOG_DEBUG << "Config > Features contains key 'Use PERF', assigning config value to " << (bool)features_config["Use PERF"];
      use_perf = features_config["Use PERF"];
    }    
    if (features_config.contains("Loosen Thread Permissions")) {
      LOG_DEBUG << "Config > Features contains key 'Loosen Thread Permissions', assigning config value to " << (bool)features_config["Loosen Thread Permissions"];
      loosen_thread_permissions = features_config["Loosen Thread Permissions"];
    }
    if (features_config.contains("Fixed Periodic Injection")) {
      LOG_DEBUG << "Config > Features contains key 'Fixed Periodic Injection', assigning config value to " << (bool)features_config["Fixed Periodic Injection"];
      fixed_periodic_injection = features_config["Fixed Periodic Injection"];
    }
    if (features_config.contains("Exit When Idle")) {
      LOG_DEBUG << "Config > Features contains key 'Exit When Idle', assigning config value to " << (bool)features_config["Exit When Idle"];
      exit_when_idle = features_config["Exit When Idle"];
    }
    if (features_config.contains("Enable Log Traces")) {
      LOG_DEBUG << "Config > Features contains key 'Enable Log Traces', assigning config value to " << (bool)features_config["Enable Log Traces"];
      log_traces = features_config["Enable Log Traces"];
    }
    if (features_config.contains("Log Periodically")) {
      LOG_DEBUG << "Config > Features contains key 'Log Periodically', assigning config value to " << (bool)features_config["Log Periodically"];
      periodic_logs = features_config["Log Periodically"];
    }
    if (features_config.contains("Periodic Logging Threshold")) {
      LOG_DEBUG << "Config > Features contains key 'Periodic Logging Threshold', assigning config value to " << (unsigned int)features_config["Periodic Logging Threshold"];
      periodic_logging_threshold = features_config["Periodic Logging Threshold"];
    }
    if (features_config.contains("Logging Period (ms)")) {
      LOG_DEBUG << "Config > Features contains key 'Logging Period (ms)', assigning config value to " << (uint64_t)features_config["Logging Period (ms)"];
      logging_period = features_config["Logging Period (ms)"] ;
      logging_period = logging_period * 1000000;
    }
    if (features_config.contains("App Logging Threshold (ms)")) {
      LOG_DEBUG << "Config > Features contains key 'App Logging Threshold (ms)', assigning config value to " << (uint64_t)features_config["App Logging Threshold (ms)"];
      app_logging_threshold = features_config["App Logging Threshold (ms)"] ;
      app_logging_threshold = app_logging_threshold * 1000000;
    }    
    if (features_config.contains("Overwriting Periodic Log File")) {
      LOG_DEBUG << "Config > Features contains key 'App Logging Threshold (ms)', assigning config value to " << (bool)features_config["Overwriting Periodic Log File"];
      overwrite_periodic_log = features_config["Overwriting Periodic Log File"] ;
    }     
  }

  if (j.find("PAPI Counters") != j.end()) {
    std::list<std::string> papi_list = j["PAPI Counters"].get<std::list<std::string>>();
    std::stringstream sstr = {};

    // Why am I still printing lists of strings like this in 2022 T.T
    // There must be a better way. Someone, please commit a better way.
    sstr << "[";
    std::for_each(papi_list.begin(), papi_list.end(), [&sstr](const std::string& binary) { sstr << "\"" << binary << "\"" << ", "; });
    sstr << "]";
    std::string papi_list_str = sstr.str();
    papi_list_str = papi_list_str.replace(papi_list_str.find_last_of(','), 2, "");

    LOG_DEBUG << "Config contains key 'PAPI Counters', assigning config value to " << papi_list_str;
    PAPI_Counters = papi_list;
  }

#if defined(STATIC_API_COSTS)
  if (j.find("DASH API Costs") != j.end()) {
    LOG_VERBOSE << "Overwriting default values for DASH API Costs";
    auto dash_api_costs = j["DASH API Costs"];
    for (int api = 0; api < api_types::NUM_API_TYPES; api++){
      for (int platform = 0; platform < resource_type::NUM_RESOURCE_TYPES; platform++){
        const std::string function_name = (std::string(api_type_names[api]) + '_' + std::string(resource_type_names[platform])).c_str();
        if (dash_api_costs.contains(function_name)) {
          uint64_t temp = dash_api_costs[function_name]; 
          exec_times[api][platform] = temp;
        }
        LOG_VERBOSE << "Value of dash_exec_times for api " << function_name << " is " << exec_times[api][platform];
      }
    }
  }
#endif

  if (j.find("Max Parallel Jobs") != j.end()) {
    LOG_DEBUG << "Config contains key 'Max Parallel Jobs', assigning config value to " << (unsigned long)j["Max Parallel Jobs"];
    max_parallel_jobs = j["Max Parallel Jobs"];
  }

#if defined(ROLLING_MEDIAN_EXEC_HISTORY)
  if (j.find("Execution History Limit") != j.end()) {
    LOG_DEBUG << "Config contains key 'Execution History Limit', assigning config value to " << (size_t)j["Execution History Limit"];
    exec_history_limit = (size_t) j["Execution History Limit"];
  }
#endif

  if (j.find("Scheduler") != j.end()) {
    LOG_DEBUG << "Config contains key 'Scheduler', assigning config value to " << (std::string)j["Scheduler"];
    scheduler = j["Scheduler"];
  }

  if (j.find("Random Seed") != j.end()) {
    LOG_DEBUG << "Config contains key 'Random Seed', assigning config value to " << (unsigned long)j["Random Seed"];
    random_seed = j["Random Seed"];
  }

  if (j.find("DASH Binary Paths") != j.end()) {
    std::list<std::string> binary_list = j["DASH Binary Paths"].get<std::list<std::string>>();
    std::stringstream sstr = {};

    // Why am I still printing lists of strings like this in 2022 T.T
    // There must be a better way. Someone, please commit a better way.
    sstr << "[";
    std::for_each(binary_list.begin(), binary_list.end(), [&sstr](const std::string& binary) { sstr << "\"" << binary << "\"" << ", "; });
    sstr << "]";
    std::string binary_list_str = sstr.str();
    binary_list_str = binary_list_str.replace(binary_list_str.find_last_of(','), 2, "");

    LOG_DEBUG << "Config contains key 'DASH Binary Paths', assigning config value to " << binary_list_str;
    dash_binary_paths = binary_list;
  }
}

unsigned int ConfigManager::getTotalResources() {
  return total_num_of_resources;
}

unsigned int *ConfigManager::getResourceArray() { return resource_array; }

bool ConfigManager::getCacheSchedules() const { return cache_schedules; }
bool ConfigManager::getEnableQueueing() const { return enable_queueing; }
bool ConfigManager::getUsePAPI() const { return use_papi; }
bool ConfigManager::getLoosenThreadPermissions() const { return loosen_thread_permissions; }
bool ConfigManager::getFixedPeriodicInjection() const { return fixed_periodic_injection; }
bool ConfigManager::getExitWhenIdle() const { return exit_when_idle; }
bool ConfigManager::getLogTraces() const { return log_traces; }
bool ConfigManager::getPeriodicLogs() const { return periodic_logs; }
bool ConfigManager::getOverwritePeriodicLog() const { return overwrite_periodic_log; }

unsigned int ConfigManager::getPeriodicLoggingThreshold() const { return periodic_logging_threshold; }
uint64_t ConfigManager::getLoggingPeriod() const { return logging_period; }
uint64_t ConfigManager::getAppLoggingThreshold() const { return app_logging_threshold; }
bool ConfigManager::getUsePERF() const { return use_perf; }

std::list<std::string> &ConfigManager::getPAPICounters() { return PAPI_Counters; }

#if !defined(STATIC_API_COSTS)
void ConfigManager::pushDashExecTime(const api_types api, const resource_type resource, uint64_t time) {
  #if defined(ROLLING_MEAN_EXEC_HISTORY)
    auto &ctrs = exec_history[std::make_pair(api, resource)];
    std::atomic<uint64_t> &cumulative_exec = ctrs.first;
    std::atomic<uint64_t> &times_run = ctrs.second;

    cumulative_exec += time;
    times_run       += 1;
    uint64_t new_mean = cumulative_exec / times_run;
    LOG_VERBOSE << "Updating mean computation time for api " 
              << std::string(api_type_names[api]) << " on resource " 
              << std::string(resource_type_names[resource]) << " to " 
              << new_mean;

    exec_times[api][resource] = new_mean;
  #endif

  #if defined(ROLLING_MEDIAN_EXEC_HISTORY)
    const std::lock_guard<std::mutex> lock(exec_history_mtx[api][resource]);

    std::deque<uint64_t> &times = exec_history[std::make_pair(api, resource)];
    times.push_front(time);
    if (times.size() >= exec_history_limit) {
      times.pop_back();
    }
    auto med = times.begin() + times.size()/2;
    std::nth_element(times.begin(), med, times.end());
    uint64_t new_median = times[times.size()/2];
    LOG_VERBOSE << "Updating median computation time for api " 
                << std::string(api_type_names[api]) << " on resource " 
                << std::string(resource_type_names[resource]) << " to " 
                << new_median;
    exec_times[api][resource] = new_median;
  #endif
}
#endif // !defined(STATIC_API_COSTS)

uint64_t ConfigManager::getDashExecTime(const api_types api, const resource_type resource) {
  return exec_times[api][resource];
}

unsigned int ConfigManager::getResourceCount(uint8_t resource) {return resource_count[(resource_type) resource]; }
void ConfigManager::setResourceCount(uint8_t resource, uint32_t resource_id) {resource_count[(resource_type) resource] = resource_id;}

unsigned int ConfigManager::getResourceToGlobalID(uint8_t resource,  uint32_t resource_id) {return resource_to_global_id[(resource_type) resource][resource_id]; }
void ConfigManager::setResourceToGlobalID(uint8_t resource,  uint32_t resource_id, uint32_t global_idx) {resource_to_global_id[(resource_type) resource][resource_id] = global_idx;}


std::string &ConfigManager::getScheduler() { return scheduler; }
unsigned long ConfigManager::getMaxParallelJobs() const { return max_parallel_jobs; }
unsigned long ConfigManager::getRandomSeed() const { return random_seed; }
std::list<std::string> &ConfigManager::getDashBinaryPaths() { return dash_binary_paths; }

#if defined(ENABLE_IL_ORACLE_LOGGING)
////////////////////////////////////
// IL-Sched
////////////////////////////////////
void ConfigManager::pushILOracle(struct_ILOracle_task task_info) {
    task_list.push_back(task_info);
}

std::list<struct_ILOracle_task> ConfigManager::getILOracleTaskList() {
    return task_list;
}

////////////////////////////////////
#endif
