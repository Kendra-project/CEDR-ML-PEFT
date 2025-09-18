# HOPPERFISH: Holistic Profiling with Portable Extensible and Robust Framework Intended for Systems with Heterogeneity

> **TL;DR.** This branch shows how to: (1) build CEDR with HOPPERFISH’s profiling enabled, (2) build & run example applications, and (3) run the neural network experiments with the provided datasets & pretrained models. HOPPERFISH is built **on top of CEDR**; for full CEDR usage/config details, see the official tutorial: <https://github.com/UA-RCL/CEDR/blob/tutorial/CEDR_tutorial.md>.

---

## 1) What is in this branch?

**Scope of this README** (only the parts you need to reproduce the paper’s results):

- **CEDR‑API build & flow** used by HOPPERFISH 
- **Applications**: build & runable examples used in profiling for anomaly detection
- **Profiling**: current offline end‑to‑end flow (PERF-based counters); real‑time and GPU‑extended profiling coming soon
- **Neural networks**: datasets, pretrained models, and experiment scripts (AE demo)



---

## 2) Prerequisites


To set up all prerequisites, simply:

- Run `install_dependencies.sh` from the repository root, **or**
- Use the Docker image and instructions at <https://github.com/UA-RCL/CEDR/blob/tutorial/CEDR_tutorial.md>
- Install Python packages by running `pip install -r neural_networks/AE_demo/requirements.txt`


**Linux PERF installation:**
To install PERF (performance counter profiling tool) on Ubuntu or most Linux systems, run:

```bash
sudo apt-get update
sudo apt-get install linux-tools-$(uname -r)
```

If you encounter issues, you may also need:

```bash
sudo apt-get install linux-tools-common linux-tools-generic
```

After installation, verify with:

```bash
perf --version
```

---
This will cover all required system and Python dependencies for building and running HOPPERFISH and CEDR.

## 3) Build CEDR + HOPPERFISH (Offline Profiling)

HOPPERFISH integrates into the CEDR build via CMake. The default flow here enables the **offline** end‑to‑end profiling path using **PERF-based hardware counters**.


Artifacts will include the CEDR runtime libraries/binaries and example apps with HOPPERFISH hooks compiled in.

> **Coming soon**: a `src-api-realtime-prof` build switch that enables **real‑time (sampling) profiling** and **extended GPU hardware counters**.

For **detailed CEDR run/config instructions** (platform and scheduler configuration and app submission), follow the upstream tutorial: <https://github.com/UA-RCL/CEDR/blob/tutorial/CEDR_tutorial.md>.

---



## 4) Build and Run Example Application with CEDR + HOPPERFISH (x86)

### Step 4.1: Build CEDR for x86

```bash
# From the repository root
mkdir build
cd build
cmake -DUsePERF=ON ..
make -j $(nproc)
```

### Step 4.2: Configure the CEDR Daemon

Copy the default config and edit as needed:

```bash
cp ../daemon_config.json ./
```

For example, edit `daemon_config.json` to set 3 CPUs and enable kill when idle:

```json
{
  "Worker Threads": {
    "cpu": 3,
    "fft": 0,
    "gemm": 0,
    "gpu": 0
  },
  "Features": {
    "Cache Schedules": false,
    "Enable Queueing": true,
    "Use PAPI": false,
    "Use PERF": true,
    "Loosen Thread Permissions": true,
    "Fixed Periodic Injection": true,
    "Exit When Idle": true
  },
  "Scheduler": "MET",
  "Random Seed": 0,
  "DASH Binary Paths": [ "./libdash-rt/libdash-rt.so" ]
}
```

### Step 4.3: Build the Radar Correlator Application

```bash
cd ../applications/APIApps/radar_correlator
make  # Builds the main shared object for CEDR API-based profiling (FFT-API based radar correlator)
# Other CEDR-based options:
#   make zip      # Builds radar_correlator that uses both ZIP and FFT API calls
#   make noapi    # Builds FFT-based radar correlator without API calls
cp radar_correlator-x86.so ../../build/
cp -r input/ ../../build/
cd ../../build
```

### Step 4.4: Run CEDR and Submit the Application

Start the CEDR daemon (in one terminal):

```bash
./cedr -c ./daemon_config.json -l NONE &
```

In the same terminal, submit the radar correlator for 10 samples with an injection period of 10000 microseconds between each instance:

```bash
./sub_dag -a ./radar_correlator_fft-x86.so -n 10 -p 10000
```

After execution, terminate CEDR (if you set ```"Exit When Idle": false```):

```bash
./kill_daemon
```

### Step 4.5: Check Profiling Output

Profiling results and logs are in `log_dir/experiment0/`.
Hopperfish raw profiled features are in `log_dir/experiment0/ad_features.csv`.

#### Example Output (ad_features.csv):

```csv
app_id,app_name,app_size,app_runtime,app_task_count,max_in_ready_queue,max_in_todo_queue,ref_arrival_time,app_diff_arrival_time,app_lifetime,app_api_min,app_api_max,app_api_mean,cpu1_todo_queue,cpu2_todo_queue,cpu3_todo_queue,cpu1,cpu2,cpu3,API_INSTRUCTIONS,API_CACHE_REFERENCES,API_CACHE_MISSES,API_BRANCH_MISSES,API_DTLB_READ_MISS,API_ITLB_READ_MISS,API_L1D_READ_ACCESS,API_L1D_READ_MISS,API_L1I_READ_MISS,APP_INSTRUCTIONS,APP_CACHE_REFERENCES,APP_CACHE_MISSES,APP_BRANCH_MISSES,APP_DTLB_READ_MISS,APP_ITLB_READ_MISS,APP_L1D_READ_ACCESS,APP_L1D_READ_MISS,APP_L1I_READ_MISS
0,radar_correlator-x86,20136,887906,3,1,1,3643303305971128,0,1203079,37678,56274,44930,1,1,0,0.001721,0.002129,0.001751,249789,8039,1548,1294,121,0,0,0,0,229110,20848,6748,3824,306,0,0,0,0
1,radar_correlator-x86,20136,462633,3,1,1,3643303315972903,10001775,637497,8838,17221,12296,1,1,1,0.001721,0.002129,0.001751,930,4380,623,948,49,4,0,0,0,936865,16260,4477,2198,104,0,0,0,0
2,radar_correlator-x86,20136,527481,3,1,1,3643303325972855,9999952,710554,9164,44084,21060,1,1,1,0.001721,0.002129,0.001751,1178,30,804,1037,35,40,21313,0,0,947257,11973,5470,2077,103,0,0,0,0
3,radar_correlator-x86,20136,797330,3,1,1,3643303335974802,10001947,1079996,12342,13764,13163,1,1,1,0.001721,0.002129,0.001751,1426,44,0,350,37,36,48278,761,0,738523,12851,7000,2930,133,0,0,0,0
4,radar_correlator-x86,20136,773195,3,1,1,3643303345973130,9998328,1010873,10057,13395,11984,1,1,1,0.001721,0.002129,0.001751,162051,18,1,15,0,1,48268,1001,1013,773234,16595,6496,3149,132,0,0,0,0
5,radar_correlator-x86,20136,702960,3,1,1,3643303355973847,10000717,1012989,9404,13776,10951,1,1,1,0.001721,0.002129,0.001751,239760,2916,0,12,0,13,48285,996,1553,641901,12912,6546,2929,132,0,0,0,0
6,radar_correlator-x86,20136,617535,3,1,1,3643303365974510,10000663,784626,9913,27473,16611,1,1,1,0.001721,0.002129,0.001751,239760,4297,549,36,0,0,15753,944,1455,947523,13510,5141,1774,76,0,0,0,0
7,radar_correlator-x86,20136,410146,3,1,1,3643303375977580,10003070,551401,9200,14186,10984,1,1,1,0.001721,0.002129,0.001751,239760,4415,899,632,0,0,0,308,1374,948002,13547,6333,1723,81,0,0,0,0
8,radar_correlator-x86,20136,753674,3,1,1,3643303385976327,9998747,1075620,13024,18480,15633,1,1,1,0.001721,0.002129,0.001751,239760,3744,1079,963,18,0,0,0,481,557398,12369,6204,2748,131,0,0,0,0
9,radar_correlator-x86,20136,538677,3,1,1,3643303395976633,10000306,716898,8894,13576,11164,1,1,1,0.001721,0.002129,0.001751,78825,4356,920,940,13,0,0,0,0,852053,16877,6317,1947,104,0,0,0,0
```



---

## 5) Neural Networks: Datasets & Pretrained Models (AE Demo)



For instructions on running the AE (Autoencoder) demo, see the README in:

`neural_networks/AE_demo/README.md` ([AE DEMO](https://github.com/UA-RCL/CEDR/blob/hopperfish/neural_networks/AE_demo/README.md))

This README provides details on datasets, pretrained models, experiment scripts, and usage examples for anomaly detection. Follow the steps there to set up your environment and run the demo.

Additional useful scripts outside AE_demo:


- `global_model_preprocessing.py`: Aggregates and processes experiment CSVs from different devices and schedulers, computes summary statistics (min/max/mean) for todo queues and processing elements, encodes scheduler/device info, and produces a unified dataset for global model training and cross-experiment analysis.
- `feature_analysis.py`: Provides functions for evaluating anomaly detection and classification results, including accuracy, false positive/negative rates, and per-application metrics. It generates plots and summary tables to help interpret model performance across different applications, schedulers, and platforms.
- `models.py`: Implements neural network architectures (AutoEncoder, VAE) and utility functions for anomaly detection, including model creation, training, and inference. It supports both standard and HLS4ML-compatible models for hardware deployment.



## 6) Citing

If you use this repository or parts of it in academic work, please cite:

- **CEDR** project: <https://ua-rcl.github.io/projects/cedr/>
- **HOPPERFISH**: [Placeholder for ACM TACO citation, add citation here when available]

---

**Questions?** Please open a GitHub issue on this branch with your environment (OS, compiler, Python, CUDA) and a minimal repro, or email mustafaghanim@arizona.edu for direct support.

