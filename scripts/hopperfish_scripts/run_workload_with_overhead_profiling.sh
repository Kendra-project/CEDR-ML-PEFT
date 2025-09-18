#!/bin/bash

SCALE_FACTOR=3
DATASET_SEED=400
pkill -f cedr

declare -a SCHEDS=("SIMPLE")
cpu="11"
fft="0"
mmult="0"
gpu="0"
zip="0"

if [ -z "$1" ]; then
  echo "Error: Application path not provided. Usage: $0 <app_folder>"
  exit 1
fi
APPS_PATH="$1"

echo "The used app class is: $APPS_PATH"
OUTPUT_BASE_DIR="./log_dir"

echo "Please enter a name for the subdirectory (e.g., train, val): "
read SUB_DIR_NAME

echo "Please enter a maximum amount of time to wait before killing CEDR in seconds, enter -1 to wait for CEDR till the end."
read ROUND_WAIT


N=10
app_instances_min=2
app_instances_max=5
app_injection_rate_min=100000
app_injection_rate_max=200000
min_delay=1
max_delay=5

rm -f outputs.log
rm -rf ./log_dir 
rm -f perf_output.txt mem_output.txt smaps_output.txt 

mkdir -p "$OUTPUT_BASE_DIR/$SUB_DIR_NAME/SIMPLE"

declare -A cumulative_csv_files
for sched in "${SCHEDS[@]}"; do
  cumulative_csv_files["$sched"]="$OUTPUT_BASE_DIR/$SUB_DIR_NAME/$sched/cumulative_ad_features.csv"
  touch "${cumulative_csv_files["$sched"]}"
done
declare -A app_instances_count
declare -a apps=($(find "$APPS_PATH" -name "*.so" -type f -exec basename {} \;))

get_seeded_random() {
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
}

for sched in "${SCHEDS[@]}"; do
  bash generate_daemon_config.sh "$cpu" "$fft" "$mmult" "$gpu" "$zip" "$sched"
  for n in $(seq 1 $N); do
    RANDOM=$(($n + $DATASET_SEED))
    SHUFFLE_SEED=$(($n + $DATASET_SEED))

    echo -e "\n========== ROUND $n BEGIN ==========" >> outputs.log
    echo "Scheduler: $sched - Round: $n" >> outputs.log

    rm -f perf_output.txt mem_output.txt smaps_output.txt
    sudo perf stat -e cycles,instructions,cache-references,cache-misses,bus-cycles \
        -o perf_output.txt --append \
        ./cedr -c ./daemon_config.json -l NONE >> outputs.log 2>&1 &
    CEDR_PID=$!

    sleep 1.0
    echo "CEDR PROCESS ID IS $CEDR_PID" >> outputs.log

    MEM_LOG=mem_output.txt
    SMAPS_LOG=smaps_output.txt
    (
      echo "timestamp(s),rss(bytes),vsz(bytes)" > $MEM_LOG
      echo "timestamp(s),smaps_Rss(kB),smaps_Pss(kB),smaps_Private_Clean(kB),smaps_Private_Dirty(kB)" > $SMAPS_LOG
      while kill -0 $CEDR_PID 2>/dev/null; do
        ts=$(date +%s.%3N)
        output=$(ps -o rss=,vsz= -p $CEDR_PID)
        rss=$(echo $output | awk '{print $1}')
        vsz=$(echo $output | awk '{print $2}')
        if [[ -n "$rss" && -n "$vsz" && "$rss" -gt 0 ]]; then
          echo "$ts,$((rss * 1024)),$((vsz * 1024))" >> $MEM_LOG
        fi
        if [[ -r /proc/$CEDR_PID/smaps_rollup ]]; then
          rss_val=$(grep "Rss:" /proc/$CEDR_PID/smaps_rollup | awk '{print $2}')
          pss_val=$(grep "Pss:" /proc/$CEDR_PID/smaps_rollup | awk '{print $2}')
          pc_val=$(grep "Private_Clean:" /proc/$CEDR_PID/smaps_rollup | awk '{print $2}')
          pd_val=$(grep "Private_Dirty:" /proc/$CEDR_PID/smaps_rollup | awk '{print $2}')
          echo "$ts,$rss_val,$pss_val,$pc_val,$pd_val" >> $SMAPS_LOG
        fi
        sleep 0.05
      done
    ) &
    MEM_PID=$!

    shuffled_apps=($(shuf --random-source=<(get_seeded_random $SHUFFLE_SEED) -e "${apps[@]}"))
    for i in "${!shuffled_apps[@]}"; do
      app_instances=$(( (RANDOM) % (app_instances_max - app_instances_min + 1) + app_instances_min ))
      app_injection_rate=$(( (RANDOM*SCALE_FACTOR) % (app_injection_rate_max - app_injection_rate_min + 1) + app_injection_rate_min ))
      command="./sub_dag -a $APPS_PATH/${shuffled_apps[$i]} -n $app_instances -p $app_injection_rate"
      echo "Submitting command for ${shuffled_apps[$i]}: $command" >> outputs.log
      eval $command >> outputs.log 2>&1
      app_instances_count["${shuffled_apps[$i]}"]=$((${app_instances_count["${shuffled_apps[$i]}"]} + $app_instances))
    done

    for app in "${apps[@]}"; do
      instances=${app_instances_count["$app"]}
      echo "Total instances of $app submitted : $instances" >> outputs.log
    done

    if [ "$ROUND_WAIT" -ne -1 ]; then
      sleep "$ROUND_WAIT"
      if kill -0 $CEDR_PID 2>/dev/null; then
        ./kill_daemon
        wait $CEDR_PID
      fi
    else
      SECONDS=0
      while kill -0 $CEDR_PID 2>/dev/null; do
        echo "Waiting... Elapsed time: ${SECONDS}s"
        sleep 1
        ((SECONDS++))
      done
    fi

    kill $MEM_PID 2>/dev/null

    LATEST_CSV=$(find ./log_dir/experiment* -type f -name '*.csv' -printf '%T+ %p\n' | sort -r | head -n 1 | cut -d' ' -f2)
    if [ ! -z "$LATEST_CSV" ]; then
      if [ $n -gt 1 ]; then
        tail -n +2 "$LATEST_CSV" >> "${cumulative_csv_files["$sched"]}"
      else
        cat "$LATEST_CSV" >> "${cumulative_csv_files["$sched"]}"
      fi
    else
      echo "No new CSV file found for round $n." >> outputs.log
    fi

    echo -e "\n==== Perf Metrics Collected ====" >> outputs.log
    cat perf_output.txt >> outputs.log

    echo -e "\n==== Memory Footprint Summary (RSS from ps) ====" >> outputs.log
    tail -n +2 $MEM_LOG | awk -F',' '
      BEGIN {sum=0; count=0; max=0}
      {rss=$2; sum+=rss; count++; if(rss>max) max=rss}
      END {
        print "Average RSS: " int(sum/count) " bytes"
        print "Peak RSS: " max " bytes"
      }' >> outputs.log

    echo -e "\n==== Memory Footprint Summary (Rss from smaps) ====" >> outputs.log
    tail -n +2 $SMAPS_LOG | awk -F',' '
      BEGIN {maxrss=0; maxpss=0; maxdirty=0}
      {
        if($2 > maxrss) maxrss=$2;
        if($3 > maxpss) maxpss=$3;
        if($5 > maxdirty) maxdirty=$5;
      }
      END {
        print "Peak smaps_RSS: " maxrss " KB"
        print "Peak smaps_PSS: " maxpss " KB"
        print "Peak Private_Dirty: " maxdirty " KB"
      }' >> outputs.log

    echo -e "=========== ROUND $n END ===========\n" >> outputs.log

  done
done
