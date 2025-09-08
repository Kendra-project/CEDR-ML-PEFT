# Class Tutorial #1: API Integration and Verification

Table of Contents:
1. [Introduce the existing FFT API in a sample application](#1-introducing-cedr-apis-to-baseline-c-applications)
     1. [Demonstrate functional verification of the FFT API](#11-cpu-based-preliminary-validation-of-the-api-based-code)
2. [Extend the system with a new ZIP API signature](#2-introducing-a-new-api-call)
3. [Update the sample application to use the new ZIP API](#3-modify-sample-application-to-include-zip-api)
     1. [Perform functional verification](#31-zip-api-verification-with-cedr)
     2. [Validate API behavior using Gantt chart analysis](#32-gantt-chart-validation)

## 1. Introducing CEDR APIs to Baseline C++ Applications

[Return to top](#class-tutorial-1-api-integration-and-verification)

Move to the `api_example` folder in the [applications](/applications/api_example/) folder from [root directory](/)
```bash
cd applications/api_example
```

Look at the [non-API version](/applications/api_example/api_example_non_api.cpp) of the sample application and locate possible places for adding API calls to the application
 - Forward FFT call: Line 26

Create a new file to add CEDR API calls and modify the sample application to include DASH_FFT API calls.
```bash
cp api_example_non_api.cpp api_example_api.cpp
```
    
Make sure the [dash.h](/libdash/dash.h) is included in the application
```c
#include "dash.h"
```

Change the `api_example_api.cpp` to include `DASH_FFT` calls
```c
<gsl_fft_wrapper((dash_cmplx_flt_type*) input, (dash_cmplx_flt_type*) output, size, forwardTrans);
>DASH_FFT_flt((dash_cmplx_flt_type*) input, (dash_cmplx_flt_type*) output, size, forwardTrans);
```

Build the sample application without API calls and observe the output
```bash
make non-api
./api_example_non_api-x86.out
```

Build the sample application with API calls (standalone execution outside CEDR) and compare the output against the non-API version
```bash
make api
./api_example_api-x86.out
```

This will also create the sample application shared object to be used with CEDR.

Copy the shared object to the CEDR build folder
```bash
cp api_example_api-x86.so ../../build
```

### 1.1. CPU-based Preliminary Validation of the API Based Code

[Return to top](#class-tutorial-1-api-integration-and-verification)

Move back to the CEDR build folder containing CEDR binaries
```bash
cd ../../build
```

Copy the daemon configuration file from the [root repository base directory](/) to the build folder
```bash
cp ../daemon_config.json ./
```

Observe the contents of the `daemon_config.json`
  * *Worker Threads*: Specify the number of PEs for each resource
  * *Features*: Enable/Disable CEDR features
    * *Cache Schedules*: Enable/**Disable** schedule caching
    * *Enable Queueing*: **Enable**/Disable Queueing
    * *Use PAPI*: Enable/**Disable** PAPI based performance counters
    * *Loosen Thread Permissions*: Thread permission setup (**Enable**/Disable)
    * *Fixed Periodic Injection*: Inject jobs with fixed injection rate
    * *Exit When Idle*: Kill CEDR once all running applications are terminated
  * *Scheduler*: Type of scheduler to use (SIMPLE, RANDOM, EFT, ETF, MET, or HEFT_RT)
  * *Random Seed*: Seed to be used on random operations
  * *DASH Binary Path*: List of paths to the libdash shared objects

Modify the `daemon_config.json` file to set the number of CPUs to 3 (or any other number).

<pre>
"Worker Threads": {
 "cpu": <b>3</b>,
 "gemm": 0,
 "gpu": 0
},
</pre>

Run CEDR using the config file. <u>Read the next lines before running!</u>
```bash
./cedr -c ./daemon_config.json
```

Push CEDR to the background or open another terminal and navigate to the same build folder ([root repository](/)/build)
  * Option 1:
    * Path 1: `Ctrl+z` and run `bg` on the terminal
    * Path 2: Run CEDR with `./cedr -c ./daemon_config.json &`. Use of `&` starts the job specified on the left-hand side as a background process.
    
    ***Now this terminal can be used for running the next steps.***
  * Option 2: Open a second terminal and go to the CEDR build directory, <code>cd ${repo_root}/build</code>.
    
    ***Now this second terminal can be used for running the next steps.***

Run `sub_dag` to submit application(s) to CEDR

```bash
./sub_dag -a ./api_example_api-x86.so -n 1 -p 0
```

  * We use `-a` to specify the application shared object to be submitted
  * We use `-n` to specify the number of instances that are to be submitted
  * We use `-p` to specify the injection rate (waiting period between two instances in microseconds) when `-n` is more than 1
    * This is an optional argument; if not set, it will be set to 0 by default

Look at the results and terminate CEDR using `kill_daemon`

```bash
./kill_daemon
```

Now, observe the log files generated in `log_dir/experiment0`. *NOTE:* Number at the end of the `experiment` folder will increment with each CEDR execution (i.e., `experiment0`, `experiment1`, `experiment2`, ...).
* `appruntime_trace.log` stores the execution time of the application.
* `schedule_trace.log` tracks the ready queue size and overhead of each scheduling decision.
* `timing_trace.log` stores the computing resource and execution time of the API call.

## 2. Introducing a New API Call

[Return to top](#class-tutorial-1-api-integration-and-verification)

In this section of the tutorial, we will demonstrate integration of a new API call to the CEDR. We will use `DASH_ZIP` API call as an example. 

Navigate to the libdash folder [root directory/libdash](/libdash/).
```bash
cd ../libdash
```

Add the API function definitions to the `dash.h`.

```c
void DASH_ZIP_flt(dash_cmplx_flt_type* input_1, dash_cmplx_flt_type* input_2, dash_cmplx_flt_type* output, size_t size, zip_op_t op);
void DASH_ZIP_flt_nb(dash_cmplx_flt_type** input_1, dash_cmplx_flt_type** input_2, dash_cmplx_flt_type** output, size_t* size, zip_op_t* op, cedr_barrier_t* kernel_barrier);

void DASH_ZIP_int(dash_cmplx_int_type* input_1, dash_cmplx_int_type* input_2, dash_cmplx_int_type* output, size_t size, zip_op_t op);
void DASH_ZIP_int_nb(dash_cmplx_int_type** input_1, dash_cmplx_int_type** input_2, dash_cmplx_int_type** output, size_t* size, zip_op_t* op, cedr_barrier_t* kernel_barrier);
```

There are 4 different function definitions here:
  1. DASH_ZIP_flt: Supports blocking ZIP calls for `dash_cmplx_flt_type`
  2. DASH_ZIP_flt_nb: Supports non-blocking ZIP calls for `dash_cmplx_flt_type`
  3. DASH_ZIP_int: Supports blocking ZIP calls for `dash_cmplx_int_type`
  4. DASH_ZIP_int_nb: Supports non-blocking ZIP calls for `dash_cmplx_int_type`

Add supported ZIP operation enums to `dash_types.h`
```c
typedef enum zip_op {
 ZIP_ADD = 0,
 ZIP_SUB = 1,
 ZIP_MULT = 2,
 ZIP_DIV = 3
} zip_op_t;
```

Add CPU implementation of the ZIP to [libdash/cpu](/libdash/cpu/) as `zip.cpp`. For simplicity, we copy an existing implementation.
```bash
cp ../original_files/zip.cpp cpu/
```

In `zip.cpp`, we also have the `enqueue_kernel` call in the API definition, which is how the task for this API will be sent to CEDR. A prototype of the `enqueue_kernel` function is given in line 12, and `enqueue_kernel` is used in non-blocking versions of the function (lines 83 and 113). The prototype is the same for all the APIs created for CEDR. The first argument must be the function name, the second argument must be the precision to be used, and the third argument shows how many inputs are needed for the calling function (for ZIP, this is 6). Now let's look at the ZIP-specific `enqueue_kernel` call on line 83.

```c
enqueue_kernel("DASH_ZIP", "flt", 6, input_1, input_2, output, size, op, kernel_barrier);
```

In this sample `enqueue kernel` call, we have 4 important portions:
  * ***"DASH_ZIP"***: Name of the API call
  * ***"flt"***: Type of the inputs on the API call
  * ***6***: Number of variables for the API call
  * Variables:
    * ***input_1***: First input of the ZIP
    * ***input_2***: Second input of the ZIP
    * ***output***: Output of the ZIP
    * ***size***: Array size for inputs and output
    * ***op***: ZIP operation type (ADD, SUB, MUL, or DIV)
    * ***kernel_barrier***: Contains configuration information of barriers for blocking and non-blocking implementation

In the `zip.cpp`, we need to fill the bodies of the four function definitions that are used so the application will call `enqueue_kernel` properly and hand off the task to CEDR for scheduling:

1. DASH_ZIP_flt
2. DASH_ZIP_flt_nb
3. DASH_ZIP_int
4. DASH_ZIP_int_nb

We also implement two additional functions that contain the implementation of CPU-based ZIP operations. Functions are created with `_cpu` suffix so that CEDR can identify the functions correctly for CPU execution:

1. DASH_ZIP_flt_cpu: `dash_cmplx_flt_type`
2. DASH_ZIP_int_cpu: `dash_cmplx_int_type`

Having included API implementation, we should introduce the new API call to the system by updating the CEDR header file ([../src-api/include/header.hpp](/src-api/include/header.hpp)):

<pre>
enum api_types {DASH_FFT = 0, DASH_GEMM = 1, DASH_FIR = 2, DASH_SpectralOpening = 3, DASH_CIC = 4, DASH_BPSK = 5, DASH_QAM16 = 6, DASH_CONV_2D = 7, DASH_CONV_1D = 8, <b>DASH_ZIP = 9,</b> NUM_API_TYPES = <b>10</b>};

static const char *api_type_names[] = {"DASH_FFT", "DASH_GEMM", "DASH_FIR", "DASH_SpectralOpening", "DASH_CIC", "DASH_BPSK", "DASH_QAM16", "DASH_CONV_2D", "DASH_CONV_1D"<b>, "DASH_ZIP"</b>};
...
static const std::map<std::string, api_types> api_types_map = { {api_type_names[api_types::DASH_FFT], api_types::DASH_FFT},
 {api_type_names[api_types::DASH_GEMM], api_types::DASH_GEMM},
 {api_type_names[api_types::DASH_FIR], api_types::DASH_FIR},
 {api_type_names[api_types::DASH_SpectralOpening], api_types::DASH_SpectralOpening},
 {api_type_names[api_types::DASH_CIC], api_types::DASH_CIC},
 {api_type_names[api_types::DASH_BPSK], api_types::DASH_BPSK},
 {api_type_names[api_types::DASH_QAM16], api_types::DASH_QAM16},
 {api_type_names[api_types::DASH_CONV_2D], api_types::DASH_CONV_2D},
 {api_type_names[api_types::DASH_CONV_1D], api_types::DASH_CONV_1D},
 <b>{api_type_names[api_types::DASH_ZIP], api_types::DASH_ZIP}</b>};
</pre>

### Building CEDR with ZIP API

Navigate to the build folder, re-generate the files, and check the `libdash-rt.so` shared object to verify the new ZIP-based function calls.
```bash
cd ../build
cmake ..
make -j $(nproc)
nm -D libdash-rt/libdash-rt.so | grep -E '*_ZIP_*'
```


### 3. Modify Sample Application to Include ZIP API

After rebuilding CEDR with ZIP API, move to the `api_example` folder in the [applications](/applications/api_example/)
```bash
cd ../applications/api_example
```

Look at the [latest version](/applications/api_example/api_example_api.cpp) of the sample application created in [step 1](#1-introducing-cedr-apis-to-baseline-c-applications) and locate possible places for adding ZIP API calls to the application
 - Multiplication: Lines 69 to 72

Change the `api_example_api.cpp` to include `DASH_ZIP` calls
```cpp
<for (i = 0; i < size; i ++) {
<  C[i].re = A[i].re * B[i].re - A[i].im * B[i].im;
<  C[i].im = A[i].re * B[i].im + A[i].im * B[i].re;
<}
>DASH_ZIP_flt(A, B, C, size, ZIP_MULT);
```

Build sample applications with ZIP API calls, standalone as well as the shared object, and compare the output against other versions
```bash
make api
./api_example_api-x86.out
```

## 3.1. ZIP API verification with CEDR

[Return to top](#class-tutorial-1-api-integration-and-verification)

Copy the shared object to the CEDR build folder and then run `cedr`, `sub_dag`, and `kill_daemon`.
```bash
cp api_example_api-x86.so ../../build
cd ../../build
cp ../daemon_config.json ./
./cedr -c ./daemon_config.json & # With print logs enabled, you can see `DASH_ZIP` API being located by CEDR now
./sub_dag -a ./api_example_api-x86.so -n 1 -p 0 # Verifiy the output!
./kill_daemon
```

Let's check the `timing_trace.log` for ZIP API calls.
```bash
cat log_dir/experiment1/timing_trace.log | grep -E '*ZIP*'
```

## 3.2. Gantt Chart Validation

[Return to top](#class-tutorial-1-api-integration-and-verification)

We can generate a Gantt chart showing the distribution of tasks to the processing elements. Navigate the `scripts/` folder from the [root directory](/) and run the `gantt_k-nk.py` script.

```bash
cd ../scripts/
python3 gantt_k-nk.py ../build/log_dir/experiment1/timing_trace.log
```
