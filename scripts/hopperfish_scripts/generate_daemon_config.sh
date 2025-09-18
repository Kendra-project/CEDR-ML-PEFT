#!/bin/bash

if [[ $# -ne 6 ]]
then
        echo "There are $# argument(s)!"
        echo "Usage 'bash generate_daemon_config.sh CPU# FFT# GEMM# GPU# zip# scheduler'"
        exit
fi

echo "Generating deamon config with $1 cpu(s), $2 fft(s), $3 gemm(s), $4 gpu(s), $5 zip(s) and $6 scheduler!"

echo "{
   \"Worker Threads\": {
        \"cpu\": $1,
        \"fft\": $2,
        \"gemm\": $3,
        \"gpu\": $4,
	      \"zip\": $5
    },

    \"Features\": {
        \"Cache Schedules\": false,
        \"Enable Queueing\": true,
        \"Use PAPI\": false,
        \"Use PERF\": true,
        \"Loosen Thread Permissions\": true,
        \"Fixed Periodic Injection\": true,
        \"Exit When Idle\": true

    },

    \"PAPI Counters\": [
        \"perf::INSTRUCTIONS\",
        \"perf::BRANCHES\",
        \"perf::BRANCH-MISSES\",
        \"perf::L1-DCACHE-LOADS\",
        \"perf::L1-DCACHE-LOAD-MISSES\"
    ],

    \"DASH API Costs\": {
        \"DASH_FFT_cpu\": 32606,
        \"DASH_FFT_fft\": 0,
        \"DASH_FFT_gpu\": 0,
        \"DASH_GEMM_cpu\": 7442,
        \"DASH_GEMM_mmult\": 0,
        \"DASH_GEMM_gpu\": 0,
        \"DASH_ZIP_cpu\": 14883,
        \"DASH_ZIP_zip\": 0,
        \"DASH_ZIP_gpu\": 0
    },

    \"Scheduler\": \"$6\",
    \"Random Seed\": 0,
    \"DASH Binary Path\": [ \"./libdash-rt/libdash-rt.so\" ]
}" > daemon_config.json

