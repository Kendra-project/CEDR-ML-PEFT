#!/bin/bash

# TODO: READ RESOURCE AND SCHEDULER LISTS, PERIODCOUNT AS INPUT ARGUMENT FROM A FILE
# Leave a copy of this input configuration file into the generated trace file base directory to be used
# later by makedataframe.
#### All possible schedulers, and max number of allowed resources ####
declare -a SCHEDS=("SIMPLE" "MET" "ETF") 
CPUS=3
FFTS=2
MMULTS=0
ZIPS=2
GPUS=0
CONV_2DS=0
######################################################################

# Number of distinct period values for each workload.
PERIODCOUNT=2
PERIODS=("1734" "2313" )

APPS=("./radar_correlator_zip-aarch64.so")
INSTS=("5")

declare -a WORKLOADS=("HIGH" )

counter=0

for w in {0..0};do
  for trial in {1..2}; do
    for ((period=0; period<$PERIODCOUNT; period++)); do
      for (( conv_2d=0; conv_2d<=$CONV_2DS; conv_2d++ )); do
        for (( gpu=0; gpu<=$GPUS; gpu++ )); do
          for (( zip=0; zip<=$ZIPS; zip++ )); do
            for (( mmult=0; mmult<=$MMULTS; mmult++ )); do
              for (( fft=0; fft<=$FFTS; fft++ )); do
                for (( cpu=$CPUS; cpu<=$CPUS; cpu++ )); do # Use to fix the CPU count to MAX
                #for (( cpu=1; cpu<=$CPUS; cpu++ )); do     # Use to sweep number of CPUs
                  for sched in "${SCHEDS[@]}"; do
                    idx=$(( cpu + fft + zip + gpu + mmult + conv_2d - 1 ))
                    bash generate_daemon_config.sh $cpu $fft $zip $mmult $gpu $conv_2d $sched
                    #echo t${trial}-"$counter": cpu ${cpu} fft ${fft} mmult ${mmult} zip ${zip} gpu ${gpu} conv_2d ${conv_2d} sched ${sched} with ${PERIODS[(${w} * ${PERIODCOUNT}) + ${period}]}
                    echo t${trial}-"$counter": running... \`./sub_dag -a "${APPS[$w]}" -n "${INSTS[$w]}" -p "${PERIODS[(${w} * ${PERIODCOUNT}) + ${period}]}"\`
                    ./cedr -c ./daemon_config.json -l NONE > /dev/null 2>&1 &
                    sleep 5
                    ./sub_dag -a "${APPS[$w]}" -n "${INSTS[$w]}" -p "${PERIODS[(${w} * ${PERIODCOUNT}) + ${period}]}"
                    wait
                    # Rename log file
                    mv log_dir/experiment0 log_dir/c${cpu}_f${fft}_m${mmult}_z${zip}_sched-${sched}-p"${PERIODS[(${w} * ${PERIODCOUNT}) + ${period}]}"
                    ((counter++))
                  done
                done
              done
            done
          done
        done
      done
    done
    mkdir log_dir/trial_${trial}
    mv log_dir/c* log_dir/trial_${trial}
  done
  mkdir log_dir/${WORKLOADS[$w]}
  mv log_dir/trial_* log_dir/${WORKLOADS[$w]}
done
