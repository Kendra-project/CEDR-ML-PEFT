#!/bin/bash

# TODO: READ RESOURCE AND SCHEDULER LISTS, PERIODCOUNT AS INPUT ARGUMENT FROM A FILE
# Leave a copy of this input configuration file into the generated trace file base directory to be used
# later by makedataframe.
#### All possible schedulers, and max number of allowed resources ####
declare -a SCHEDS=("EFT")
CPUS=2
FFTS=3
######################################################################

# Number of distinct period values for each workload. Use this value from the bash script that
# runs sub_dag

INSTS=("10" "10" "10")
INJ_RATES=("0" "0" "0")
declare -a APPS=("./apps/pulse_doppler-phased-nb-riscv64_print.so" "./apps/SAR_nb-riscv64_print.so" "./apps/track_nb-riscv64_print.so")

rm -rf ./sweep
rm -rf ./log_dir

STAGE_DIR=./sweep/ # Edit this
if [ ! -d ${STAGE_DIR} ];
then
  echo No stage directory! Creating a new one.
  mkdir ${STAGE_DIR}
fi

FILE=launchfile

counter=0

# For little configuration
for (( app=0; app<=0; app++ )); do
    for sched in "${SCHEDS[@]}"; do
      for trial in {1..2}; do
          
        config=c2_f3-${sched}
        echo Running the config: daemon_config-${config}.json
          
        echo trial ${trial} run "$counter" app "${APPS[$app]}" sched ${sched}
          
        if [ ! -f ./daemon_configs/daemon_config-${config}.json ]; then
          echo No daemon_configs/daemon_config-${config}.json
          exit
        fi
              
        echo ./cedr_little -c ./daemon_configs/daemon_config-${config}.json -l NONE 
        ./cedr_little -c ./daemon_configs/daemon_config-${config}.json -l NONE & > output.txt #/dev/null 2>&1 &
        sleep 10
          
        echo ./sub_dag -a "${APPS[$app]}" -n "${INSTS[$app]}" -p "${INJ_RATES[$app]}"
        ./sub_dag -a "${APPS[$app]}" -n "${INSTS[$app]}" -p "${INJ_RATES[$app]}"
          
        wait
        log_check=`ls -al ./log_dir/ | wc -l`
          
        #done
        ((counter++))
      done
      mkdir -p ${STAGE_DIR}/little/"${APPS[$app]}"/${sched}
      mv log_dir/* ${STAGE_DIR}/little/"${APPS[$app]}"/${sched}
    done
done

# For big configuration
for (( app=0; app<=0; app++ )); do
    for sched in "${SCHEDS[@]}"; do
      for trial in {1..5}; do

        config=c2_f3-${sched}
        echo Running the config: daemon_config-${config}.json

        echo trial ${trial} run "$counter" app "${APPS[$app]}" sched ${sched}

        if [ ! -f ./daemon_configs/daemon_config-${config}.json ]; then
          echo No daemon_configs/daemon_config-${config}.json
          exit
        fi

        echo ./cedr_1big -c ./daemon_configs/daemon_config-${config}.json -l NONE
        ./cedr_1big -c ./daemon_configs/daemon_config-${config}.json -l NONE & > output.txt #/dev/null 2>&1 &
        sleep 10

        echo ./sub_dag -a "${APPS[$app]}" -n "${INSTS[$app]}" -p "${INJ_RATES[$app]}"
        ./sub_dag -a "${APPS[$app]}" -n "${INSTS[$app]}" -p "${INJ_RATES[$app]}"

        wait
        log_check=`ls -al ./log_dir/ | wc -l`

        #done
        ((counter++))
      done
      mkdir -p ${STAGE_DIR}/big/"${APPS[$app]}"/${sched}
      mv log_dir/* ${STAGE_DIR}/big/"${APPS[$app]}"/${sched}
    done
done

