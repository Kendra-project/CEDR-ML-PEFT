#!/bin/bash

if [[ $# -ne 7 ]]
then
        echo "There are $# argument(s)!"
        echo "Usage 'bash generate_daemon_config.sh CPU# FFT# ZIP# GEMM# GPU# Conv_2D# scheduler_name'"
        exit
fi

echo "Generating deamon config with $1 cpu(s), $2 fft(s), $3 zip(s), $4 gemm(s), $5 gpu(s),  $6 conv_2d(s), and $7 scheduler!"

echo "{
   \"Worker Threads\": {
        \"cpu\": $1,
        \"fft\": $2,
        \"zip\": $3,
        \"gemm\": $4,
        \"gpu\": $5,
        \"conv_2d\": $6
    },

    \"Features\": {
        \"Fixed Periodic Injection\": true,
        \"Exit When Idle\": true
    },

    \"DASH API Costs\": {
            \"DASH_FFT_cpu\": 118620,
            \"DASH_FFT_fft\": 107910,
            \"DASH_ZIP_cpu\": 246570,
            \"DASH_ZIP_zip\": 151200,
            \"DASH_CONV_2D_cpu\": 100,
            \"DASH_CONV_2D_conv_2d\": 100,
            \"DASH_GEMM_cpu\": 100,
            \"DASH_GEMM_mmult\": 100
    },

    \"Scheduler\": \"$7\",
    \"Random Seed\": 0,
    \"DASH Binary Path\": [ \"./libdash-rt/libdash-rt.so\" ]
}" > daemon_config.json

