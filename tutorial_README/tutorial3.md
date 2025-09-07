# Class Tutorial #3: Scheduling and Design Space Exploration

Table of Contents:
1. [Running multiple applications with CEDR](#1-running-multiple-applications-with-cedr)
   1. [Running multiple instances of a single application](#111-running-multiple-instances-of-a-single-application)
   2. [Running multiple applications with multiple instances](#112-running-multiple-applications-with-multiple-instances)
2. [Integration and Evaluation of EFT Scheduler](#2-integration-and-evaluation-of-eft-scheduler)
   1. [Initialization](#21-initialization)
   2. [EFT heuristic - Task-to-PE mapping](#22-eft-heuristic---task-to-pe-mapping)
   3. [EFT heuristic - Task-to-PE assignment](#23-eft-heuristic---task-to-pe-assignment)
   4. [Final checks](#24-final-checks)
   5. [Full EFT in C/C++](#25-full-eft-in-cc)
   6. [Adding EFT as a scheduling option](#26-adding-eft-as-a-scheduling-option)
   7. [Enabling EFT for CEDR](#27-enabling-eft-for-cedr)
   8. [Running CEDR with EFT Scheduler](#28-running-cedr-with-eft-scheduler)
3. [Perform simple design space exploration (DSE) on x86](#3-design-space-exploration)

## 1. Running Multiple Applications with CEDR

In this section, we will use two applications ([radar correlator](/applications/radar_correlator/) and [pulse doppler](/applications/pulse_doppler/) that were created as part of assignment 1.

If not already there, copy both applications and their input folders to the `build` folder.

```bash
# Assuming you are in the `build` folder
cp -r ../applications/radar_correlator/{radar_correlatorr_api-x86.so,input} ../applications/pulse_doppler/{pulse_doppler_api-x86.so,input} ./
```

### 1.1.1. Running Multiple Instances of a Single Application

[Return to top](#class-tutorial-3-scheduling-and-design-space-exploration)

Launch CEDR with your desired configuration (edit [daemon_config.json](/daemon_config.json)) and submit [radar correlator](/applications/radar_correlator/) applications with 10 instances and an injection period of 100 microseconds.

Here, we use `-l NONE` to disable logs related to CEDR and track the outputs of each application running for functional verification. 

```bash
./cedr -c ./daemon_config.json -l NONE &
./sub_dag -a ./radar_correlator_api-x86.so -n 10 -p 100
```

Verify the outputs by checking the results of each of the 10 applications. Later, examine the log files to see the execution logs.

```bash
vim log_dir/experiment0/timing_trace.log
vim log_dir/experiment0/appruntime_trace.log
```

### 1.1.2. Running Multiple Applications with Multiple Instances

[Return to top](#class-tutorial-3-scheduling-and-design-space-exploration)

Next, we will launch CEDR with your desired configuration and submit two applications with varying numbers of instances and injection rates.

```bash
./cedr -c ./daemon_config.json -l NONE &
./sub_dag -a ./radar_correlator_api-x86.so,./pulse_doppler_api-x86.so -n 10,1 -p 100,0
```

Verify the outputs by checking the outputs of each 10 [radar correlator](/applications/radar_correlator/) applications and the single [pulse doppler](/applications/pulse_doppler/) application. Later, examine the log files to see the execution logs for both applications.

```bash
vim log_dir/experiment1/timing_trace.log
vim log_dir/experiment1/appruntime_trace.log
```

## 2. Integration and Evaluation of EFT Scheduler

[Return to top](#class-tutorial-3-scheduling-and-design-space-exploration)

Now navigate to [scheduler.cpp](/src-api/scheduler.cpp). This file contains various schedulers already tailored to work with CEDR. In this part of the tutorial, we will add the Earliest Finish Time(EFT) scheduler to CEDR. EFT heuristic schedules all the tasks in the `read queue` one by one based on the earliest expected finish time of the task on the available resources (processing elements -- PE). 

First, we will write the EFT scheduler as a C/C++ function. We will utilize the available variables for all the schedulers in CEDR. A list of the useful variables and their explanations can be found in the bulleted list below.

* ***cedr_config:*** Information about the current configuration of CEDR
* ***ready_queue:*** Tasks ready to be scheduled at the time of the scheduling event
* ***hardware_thread_handle:*** List of hardware threads that manage the available resources (PEs)
* ***resource_mutex:*** Mutex protection for the available resources (PEs)
* ***free_resource_count:*** Number of free resources (PEs) at the time of scheduling event, only useful if PE queues are disabled -- Not used in the tutorial

Based on the available variables, we will construct the prototype of the EFT scheduler as shown:

```C
int scheduleEFT(ConfigManager &cedr_config, std::deque<task_nodes *> &ready_queue, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex, uint32_t &free_resource_count)
```

Leveraging the available variables, we will implement the EFT in C/C++ step by step. First, we will handle the initializations, then the main loop body for task-to-PE mapping based on the heuristic, followed by the actual PE assignment for the task, and end with final checks. After implementing the scheduler, we will add the EFT scheduler as one of the schedulers for CEDR and enable it in the runtime config.

### 2.1. Initialization

[Return to top](#class-tutorial-3-scheduling-and-design-space-exploration)

Here we will initialize some of the required fields for the EFT heuristic. We will use this function's start time as the reference current time while computing the expected finish time of a task on the PEs.

```C
  unsigned int tasks_scheduled = 0; // Number of tasks scheduled so far
  int eft_resource = 0; // ID of the PE that will be assigned to the task
  unsigned long long earliest_estimated_availtime = 0; // Estimated finish time initialization
  bool task_allocated; // Task assigned to PE successfully or not

 /* Get current time in nanosecond scale */
  struct timespec curr_timespec {};
  clock_gettime(CLOCK_MONOTONIC_RAW, &curr_timespec);
  long long curr_time = curr_timespec.tv_nsec + curr_timespec.tv_sec * SEC2NANOSEC;

  long long avail_time; // Current available time of the PEs
  long long task_exec_time; // Estimated execution time of the task

  unsigned int total_resources = cedr_config.getTotalResources(); // Total Number of PEs available
```


### 2.2. EFT heuristic - Task-to-PE mapping

[Return to top](#class-tutorial-3-scheduling-and-design-space-exploration)

Here, we will have a double-nested loop, where the outer loop will traverse all the tasks in the ready queue using the `ready_queue` variable. The inner loop will traverse all the PEs in the current runtime by indexing the `hardware_thread_handle`. 

```C
  // For loop to iterate over all tasks in Ready queue
  for (auto itr = ready_queue.begin(); itr != ready_queue.end();) {
 earliest_estimated_availtime = ULLONG_MAX;
    // For each task, iterate over all PEs to find the earliest finishing one
    for (int i = total_resources - 1; i >= 0; i--) {
 auto resourceType = hardware_thread_handle[i].thread_resource_type; // FFT, ZIP, GEMM, etc.
 avail_time = hardware_thread_handle[i].thread_avail_time; // Based on estimated execution times of the tasks in the `todo_queue` of the PE
 task_exec_time = cedr_config.getDashExecTime((*itr)->task_type, resourceType); // Estimated execution time of the task
 auto finishTime = (curr_time >= avail_time) ? curr_time + task_exec_time : avail_time + task_exec_time; // estimated finish time of the task on the PE at i^th index
 auto resourceIsSupported = ((*itr)->supported_resources[(uint8_t) resourceType]); // Check if the current PE support execution of this task
 /* Check if the PE supports the task and if the estimated finish time is earlier than what is found so far */
      if (resourceIsSupported && finishTime < earliest_estimated_availtime) {
 earliest_estimated_availtime = finishTime;
 eft_resource = i;
 }
 }
```

### 2.3. EFT heuristic - Task-to-PE assignment

[Return to top](#class-tutorial-3-scheduling-and-design-space-exploration)

Here, we will utilize a built-in function that does final checks before assigning the task to the given PE. Based on this assignment, it also handles the actual queue management and modification of any required field. Details of this function can be found [here](https://github.com/UA-RCL/CEDR/blob/ece506/src-api/scheduler.cpp#L17-L64).

```C
    // Attempt to assign task on earliest finishing PE
 task_allocated = attemptToAssignTaskToPE(
 cedr_config, // Current configuration of the CEDR
 (*itr), // Task that is being scheduled
 &hardware_thread_handle[eft_resource], // PE that is mapped to the task based on the heuristic
 &resource_mutex[eft_resource], // Mutex protection for the PE's todo queue
 eft_resource // ID of the mapped PE
 );
```

### 2.4. Final checks

[Return to top](#class-tutorial-3-scheduling-and-design-space-exploration)

In the last part, we will check whether the task assignment to the given PE was successful and move on to the next task in the `ready queue`.

```C
    if (task_allocated) { // If task allocated successfully
 tasks_scheduled++; // Increment the number of scheduled tasks
 itr = ready_queue.erase(itr); // Remove the task from ready_queue
 /* If queueing is disabled, decrement free resource count*/
      if (!cedr_config.getEnableQueueing()) {
 free_resource_count--;
        if (free_resource_count == 0)
          break;
 }
 } else { // If task is not allocated successfully
 itr++; // Go to the next task in ready_queue
 }
 }
  return tasks_scheduled;
```

### 2.5. Full EFT in C/C++

[Return to top](#class-tutorial-3-scheduling-and-design-space-exploration)

Now, collecting all the steps, we will have the EFT function written in C/C++ that is tailored to CEDR, as shown:

```C
int scheduleEFT(ConfigManager &cedr_config, std::deque<task_nodes *> &ready_queue, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex, uint32_t &free_resource_count) {

  unsigned int tasks_scheduled = 0; // Number of tasks scheduled so far
  int eft_resource = 0; // ID of the PE that will be assigned to the task
  unsigned long long earliest_estimated_availtime = 0; // Estimated finish time initialization
  bool task_allocated; // Task assigned to PE successfully or not

 /* Get current time in nanosecond scale */
  struct timespec curr_timespec {};
  clock_gettime(CLOCK_MONOTONIC_RAW, &curr_timespec);
  long long curr_time = curr_timespec.tv_nsec + curr_timespec.tv_sec * SEC2NANOSEC;

  long long avail_time; // Current available time of the PEs
  long long task_exec_time; // Estimated execution time of the task

  unsigned int total_resources = cedr_config.getTotalResources(); // Total Number of PEs available

  // For loop to iterate over all tasks in Ready queue
  for (auto itr = ready_queue.begin(); itr != ready_queue.end();) {
 earliest_estimated_availtime = ULLONG_MAX;
    // For each task, iterate over all PEs to find the earliest finishing one
    for (int i = total_resources - 1; i >= 0; i--) {
 auto resourceType = hardware_thread_handle[i].thread_resource_type; // FFT, ZIP, GEMM, etc.
 avail_time = hardware_thread_handle[i].thread_avail_time; // Based on estimated execution times of the tasks in the `todo_queue` of the PE
 task_exec_time = cedr_config.getDashExecTime((*itr)->task_type, resourceType); // Estimated execution time of the task
 auto finishTime = (curr_time >= avail_time) ? curr_time + task_exec_time : avail_time + task_exec_time; // estimated finish time of the task on the PE at i^th index
 auto resourceIsSupported = ((*itr)->supported_resources[(uint8_t) resourceType]); // Check if the current PE support execution of this task
 /* Check if the PE supports the task and if the estimated finish time is earlier than what is found so far */
      if (resourceIsSupported && finishTime < earliest_estimated_availtime) {
 earliest_estimated_availtime = finishTime;
 eft_resource = i;
 }
 }

    // Attempt to assign task on earliest finishing PE
 task_allocated = attemptToAssignTaskToPE(
 cedr_config, // Current configuration of the CEDR
 (*itr), // Task that is being scheduled
 &hardware_thread_handle[eft_resource], // PE that is mapped to the task based on the heuristic
 &resource_mutex[eft_resource], // Mutex protection for the PE's todo queue
 eft_resource // ID of the mapped PE
 );

    if (task_allocated) { // If task allocated successfully
 tasks_scheduled++; // Increment the number of scheduled tasks
 itr = ready_queue.erase(itr); // Remove the task from ready_queue
 /* If queueing is disabled, decrement free resource count*/
      if (!cedr_config.getEnableQueueing()) {
 free_resource_count--;
        if (free_resource_count == 0)
          break;
 }
 } else { // If task is not allocated successfully
 itr++; // Go to the next task in ready_queue
 }
 }
  return tasks_scheduled;
}
```

### 2.6. Adding EFT as a scheduling option

[Return to top](#class-tutorial-3-scheduling-and-design-space-exploration)

Now, the only thing left is to ensure CEDR can run this function during scheduling events. To do this in the same [scheduler.cpp](/src-api/scheduler.cpp) file, we go to the end and update the [performScheduling](https://github.com/UA-RCL/CEDR/blob/ece506/src-api/scheduler.cpp#L406) function. In the function where `sched_policy` is checked, we add another `else if` segment that checks whether the scheduling policy is `EFT`. If it is, we will call the function we just created.

```C
else if (sched_policy == "EFT") {
 tasks_scheduled += scheduleEFT(cedr_config, ready_queue, hardware_thread_handle, resource_mutex, free_resource_count);
 }
```

After adding EFT as one of the scheduling heuristic options to CEDR, we will need to rebuild CEDR in the `build` directory. First, navigate to [root directory](./), then follow the steps below to rebuild CEDR with EFT.

```bash
cd build
make -j
```

### 2.7. Enabling EFT for CEDR

[Return to top](#class-tutorial-3-scheduling-and-design-space-exploration)

In the [daemon_config.json](/daemon_config.json) file, we updated the ["Scheduler"](https://github.com/UA-RCL/CEDR/blob/ece506/daemon_config.json#L35) field to be "EFT" before running CEDR with the updated daemon config file.

```JSON
    "Scheduler": "EFT",
```

### 2.8. Running CEDR with EFT Scheduler

[Return to top](#class-tutorial-3-scheduling-and-design-space-exploration)

Using the same methods as in Section 3.2, we will run CEDR and see the use of the EFT scheduler. After running the following command, we will see that the scheduler to be used is selected as EFT in the displayed logs.

```bash
./cedr -c ./daemon_config.json -l VERBOSE | grep -E "(Scheduler|scheduler)" &
```

<pre>
[...] DEBUG [312918] [ConfigManager::parseConfig@136] Config contains key 'Scheduler', assigning config value to <b>EFT</b>
</pre>

After submitting the application with `sub_dag`, we will see that the newly added EFT scheduler is used during the scheduling event. 

```bash
./sub_dag -a ./radar_correlator_api-x86.so -n 1 -p 0
```

<pre>
[...] DEBUG [312918] [performScheduling@475] Ready queue non-empty, performing task scheduling using <b>EFT</b> scheduler.
[...] DEBUG [312918] [performScheduling@475] Ready queue non-empty, performing task scheduling using <b>EFT</b> scheduler.
[...] DEBUG [312918] [performScheduling@475] Ready queue non-empty, performing task scheduling using <b>EFT</b> scheduler.
</pre>

Once everything is completed, we will terminate CEDR with `kill_daemon`.

```bash
./kill_daemon
```

## 3. Design Space Exploration

CEDR comes with some scripts that make design-space exploration (DSE) rapid and easy. Now, we will go over the flow and define how to perform DSE step by step. First, navigate to the folder where we accommodate API based CEDR scripts from [root directory](/).

```bash
cd scripts/scripts-API/run_scripts
```

We will initially run [daemon_generator.py](/scripts/scripts-API/run_scripts/daemon_generator.py) file to generate `daemon_config.json` files for our sweeps. We can modify the following code portion to denote scheduler types and hardware compositions. We set schedulers to `SIMPLE, ETF, and MET`, and for hardware compositions that will be swept, we select 4 CPUs at maximum, as the x86 system does not have any accelerators. If there were an accelerator, we would also set the maximum number of accelerators we would like to sweep up to. 

```python
SCHEDS = ["SIMPLE", "ETF", "MET"]
CPUS = 3
FFTS = 0
MMULTS = 0
ZIPS = 0
GPUS = 0
```

Then, we can see that the number of each processing element starts from `0` all the way up to `maximum number of that processing element`, looking at nested loops between Lines 26-31 (except CPU starts from 1). By changing the boundaries of the for loops, we can control the starting point of the sweep for each processing element. For this experiment, we will keep the file the same.  

Next, we need to configure [run_cedr.sh](/scripts/scripts-API/run_scripts/run_cedr.sh) and [run_sub_dag.sh](/scripts/scripts-API/run_scripts/run_subdag.sh), which will concurrently run CEDR and submit applications. In `run_cedr.sh`, we need to set the following fields to be identical to the daemon config generator. Periodicity denotes the delay between injecting each application instance in microseconds.

```bash
declare -a SCHEDS=("SIMPLE" "MET" "ETF")
CPUS=3
FFTS=0
MMULTS=0
ZIPS=0
GPUS=0
######################################################################

# Number of distinct period values for each workload. Use this value from the bash script that
# runs sub_dag
PERIODCOUNT=2
PERIODS=("1734" "2313")

declare -a WORKLOADS=("HIGH" )
```

In the case of `run_sub_dag.sh`, we need to set the following fields identically as before. In this context, we define two variables: `APPS`, which stores the applications to be swept, and `INSTS`, which specifies the number of each application to be submitted during each sweep. 

```bash
#### All possible schedulers, and max number of allowed resources ####
declare -a SCHEDS=("SIMPLE" "MET" "ETF")
declare -a CPUS=3
declare -a FFTS=0
declare -a MMULTS=0
declare -a ZIPS=0
declare -a GPUS=0
######################################################################

APPS=("radar_correlator_api-x86.so")
INSTS=("5")

declare -a PERIODS=("1734" "2313")
PERIODCOUNT=2

declare -a WORKLOADS=("HIGH")
```

After getting everything ready, we can move scripts and configuration files to the [build](/build/) folder to start the DSE. We also need to create a folder named `schedsweep_daemon_configs` in the `build` folder to store configuration files. 

```bash
python3 daemon_generator.py
mkdir ../../../build/schedsweep_daemon_configs/
cp daemon_config*.json ../../../build/schedsweep_daemon_configs/
cp *.sh ../../../build
```

Navigate back to the `build` directory and remove all earlier log files in the `log_dir` directory.
```bash
cd ../../../build
rm -rf log_dir/*
``` 

First, execute `run_cedr.sh` to run CEDR with the DSE configurations in the first terminal. Then, open a new terminal and run `run_sub_dag.sh` to dynamically submit applications based on workload composition. If you are in a Docker environment, execute the first commands on a separate terminal to be able to pull up a second terminal running on the same Docker container.

```bash
docker ps
# Take note of the number below the "CONTAINER ID" column
docker exec -it <CONTAINER ID> /bin/bash
```

```bash
bash run_cedr.sh    # Execute on the first terminal
bash run_sub_dag.sh # Execute on the second terminal
```

After both scripts terminate, there should be a folder named `HIGH` in the `log_dir` containing as many files as there are trials. Each folder should have log files for each hardware composition, scheduler, and injection rate. To plot all the DSE results in a 3D format, first navigate to `scripts/scripts-API/` from [root directory](/).

```bash
cd ../scripts/scripts-API/
```

There are two scripts named `makedataframe.py` and `plt3dplot_inj.py` for plotting a 3D diagram. For each DSE experiment, the following lines in [makedataframe.py](/scripts/scripts-API/makedataframe.py) should be modified.

```python
corelist = [' cpu1', ' cpu2', ' cpu3']  # Line 38

############# Edit parameters here ####################
# Starting from line 179
CPUS=3
FFTS=0
MMULTS=0
ZIPS=0

SCHEDS=["SIMPLE", "MET", "ETF"]

if WORKLOAD == 'HIGH':
    # Use following INJ_RATES and PERIODS for High latency workload data
 INJ_RATES=[10, 20]
 PERIODS=[1734, 2313]
elif WORKLOAD == 'LOW':
    print('Low workload is not specified for this setup')
else:
    print('Wrong workload type ', WORKLOAD, ' chosen, please choose either "HIGH" or "LOW"!')
    exit()

INJ_COUNT=int(args.injectionRateCount)
TRIALS=int(args.trial)
corelist = [' cpu1', ' cpu2', ' cpu3']  # Edit here

#######################################################
```

To learn about the input arguments of `makedataframe.py`, execute the script with the `-h` option. Then, execute the `makedataframe.py` script using the given arguments below for the DSE experiment in this tutorial. Other DSE experiments may require a different set of input arguments. 

```bash
python3 makedataframe.py -h
python3 makedataframe.py -i ../../build/log_dir/ -w HIGH -o dataframe.csv -t 2 -r 2
```

Modify the following lines in the [plt3dplot_inj.py](scripts/scripts-API/plt3dplot_inj.py).

```python
### Configuration specification ###
### Starting from line 27
CPUS = 3
FFTS = 0
MMULTS = 0
ZIPS = 0
GPUS = 0
WORKLOAD = 'High'
TRIALS = 2
schedlist = {'SIMPLE':1, 'MET':2, 'ETF':3}
schedmarkerlist = {'SIMPLE':'o', 'MET':'o', 'ETF':'o'}
schednamelist = ['RR', 'MET', 'ETF']
```

Execute the script using the following commands. 

```bash
python3 plt3dplot_inj.py <input file name> <metric>
python3 plt3dplot_inj.py dataframe.csv CUMU  # Accumulates execution time of each API call
python3 plt3dplot_inj.py dataframe.csv EXEC  # Application execution time
python3 plt3dplot_inj.py dataframe.csv SCHED # Scheduling overhead
```