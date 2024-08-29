#!/bin/bash

# TODO: READ RESOURCE AND SCHEDULER LISTS, PERIODCOUNT AS INPUT ARGUMENT FROM A FILE
# Leave a copy of this input configuration file into the generated trace file base directory to be used
# later by makedataframe.
#### All possible schedulers, and max number of allowed resources ####
declare -a SCHEDS=("EFT" "ETF" "MET") # "HEFT_RT")
CPUS=2
FFTS=3
######################################################################

# Number of distinct period values for each workload. Use this value from the bash script that
# runs sub_dag

INSTS=("5,5")

APPS=("apps/pulse_doppler-phased-nb-riscv64.so,apps/wifi-tx-nb-riscv64.so" "object")

PERIODCOUNT=15
PERIODS=("8687500,8687500" "4343750,4343750" "2895833,2895833" "1737500,1737500" "115833,1158333"  "868750,868750" "579167,579167" "434375,434375" "289583,289583" "217188,217188" "173750,173750" "86875,86875" "43438,43438" "21719,21719" "17375,17375")

# Two different workloads, "High" has Pulse doppler and WiFi-TX; "Low" has Radar correlator and Temporal mitigation
declare -a WORKLOADS=("HIGH" "LOW")

if [ -d ./log_dir/experiment0 ];
then
  echo directory is still there!
  exit
fi

STAGE_DIR=./log_dir_sweep/ # Edit this
if [ ! -d ${STAGE_DIR} ];
then
  echo No stage directory!
  exit
fi
if [ ! -d ${STAGE_DIR}/log_dir ];
then
  mkdir ${STAGE_DIR}/log_dir
fi

FILE=launchfile

counter=0

for w in {0..0}; do
  for trial in {4..5}; do
    for ((period=0; period<$PERIODCOUNT; period++)); do
      for (( fft=0; fft<=$FFTS; fft++ )); do
        for (( cpu=$CPUS; cpu<=$CPUS; cpu++ )); do
          for sched in "${SCHEDS[@]}"; do
            log_check=3
            # while [ $? -ne 0 ]; do
            while [ $log_check -eq 3 ]; do
              if [ -d ./log_dir/experiment0 ]; then
                echo rmdir ./log_dir/experiment0
                rmdir ./log_dir/experiment0
              fi
              config=c${cpu}_f${fft}-${sched}
              echo Running the config: daemon_config-${config}.json
              echo trial ${trial} run "$counter" cpu ${cpu} fft ${fft} sched ${sched}
              if [ ! -f ./daemon_configs/daemon_config-${config}.json ]; then
                echo No daemon_configs/daemon_config-${config}.json
                exit
              fi
              echo ./cedr -c ./daemon_configs/daemon_config-${config}.json -l NONE 
              ./cedr -c ./daemon_configs/daemon_config-${config}.json -l NONE > /dev/null 2>&1 &
              sleep 10
              echo ./sub_dag -a "${APPS[$w]}" -n "${INSTS[$w]}" -p "${PERIODS[(${w} * ${PERIODCOUNT}) + ${period}]}"
              ./sub_dag -a "${APPS[$w]}" -n "${INSTS[$w]}" -p "${PERIODS[(${w} * ${PERIODCOUNT}) + ${period}]}"
              wait
              log_check=`ls -al ./log_dir/experiment0/ | wc -l`
            done
            echo cp -r log_dir/experiment0 ${STAGE_DIR}/log_dir/${config}-p"${PERIODS[(${w} * ${PERIODCOUNT}) + ${period}]}"
            cp -r log_dir/experiment0 ${STAGE_DIR}/log_dir/${config}-p"${PERIODS[(${w} * ${PERIODCOUNT}) + ${period}]}"
            echo rm -rf log_dir/experiment0
            rm -rf log_dir/experiment0
            ((counter++))
          done
        done
      done
    done
    mkdir ${STAGE_DIR}/log_dir/trial_${trial}
    mv ${STAGE_DIR}/log_dir/c* ${STAGE_DIR}/log_dir/trial_${trial}
  done
  mv ${STAGE_DIR}/log_dir/trial_* ${STAGE_DIR}/log_dir/${WORKLOADS[$w]}
done
