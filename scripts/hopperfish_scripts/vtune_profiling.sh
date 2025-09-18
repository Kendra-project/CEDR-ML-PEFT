-#!/usr/bin/env bash
# vtune_profiling.sh: always runs CEDR under VTune and perf, and logs all output to outputs.log

# ----------------------------------------------------------------------------
# CONFIGURATION PARAMETERS
# ----------------------------------------------------------------------------
SCALE_FACTOR=3
DATASET_SEED=400

# ----------------------------------------------------------------------------
# PREP: kill any previous CEDR instances
# ----------------------------------------------------------------------------
pkill -x cedr 2>/dev/null || true

# Scheduler list and resource settings
declare -a SCHEDS=("SIMPLE")
cpu="11"
fft="0"
mmult="0"
gpu="0"
zip="0"

# ----------------------------------------------------------------------------
# ARGUMENT & USER PROMPTS
# ----------------------------------------------------------------------------
if [ -z "$1" ]; then
  echo "Error: Application path not provided. Usage: $0 <app_folder>"
  exit 1
fi
APPS_PATH="$1"
OUTPUT_BASE_DIR="./log_dir"

echo "Application folder: $APPS_PATH"
read -p "Subdirectory name (e.g., train, val): " SUB_DIR_NAME
read -p "Time to wait before killing CEDR in seconds (-1 to wait until completion): " ROUND_WAIT

# ----------------------------------------------------------------------------
# SETUP VTune ENVIRONMENT (always enabled)
# ----------------------------------------------------------------------------
source /opt/intel/oneapi/setvars.sh || {
  echo "Error: Failed to source Intel oneAPI environment."
  exit 1
}
echo "VTune profiling enabled."
VTUNE_BASE_DIR="./vtune_results"
rm -rf "$VTUNE_BASE_DIR" 2>/dev/null || sudo rm -rf "$VTUNE_BASE_DIR"
mkdir -p "$VTUNE_BASE_DIR"

# ----------------------------------------------------------------------------
# CLEANUP PREVIOUS OUTPUTS
# ----------------------------------------------------------------------------
rm -f outputs.log perf_output.txt mem_output.txt smaps_output.txt
rm -rf "$OUTPUT_BASE_DIR" 2>/dev/null || sudo rm -rf "$OUTPUT_BASE_DIR"
mkdir -p "$OUTPUT_BASE_DIR/$SUB_DIR_NAME/${SCHEDS[0]}"

# ----------------------------------------------------------------------------
# EXPERIMENT PARAMETERS & APP LIST
# ----------------------------------------------------------------------------
N=10
app_instances_min=2
app_instances_max=5
app_rate_min=100000
app_rate_max=200000

declare -A app_instances_count
mapfile -t apps < <(find "$APPS_PATH" -name '*.so' -printf '%f\n')

get_seeded_random() {
  openssl enc -aes-256-ctr -pass pass:"$1" -nosalt </dev/zero 2>/dev/null
}

trap 'echo "Interrupted; killing all child processes..." >> outputs.log; kill 0; exit 1' SIGINT SIGTERM

# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
for sched in "${SCHEDS[@]}"; do
  bash generate_daemon_config.sh "$cpu" "$fft" "$mmult" "$gpu" "$zip" "$sched"

  for n in $(seq 1 $N); do
    RANDOM=$((n + DATASET_SEED))
    SHUFFLE_SEED=$((n + DATASET_SEED))

    echo -e "\n========== ROUND $n BEGIN ==========" >> outputs.log
    echo "Scheduler: $sched - Round: $n" >> outputs.log

    rm -f perf_output.txt mem_output.txt smaps_output.txt
    ROUND_DIR="$VTUNE_BASE_DIR/round_$n"
    mkdir -p "$ROUND_DIR"

    vtune -collect-with runsa \
      -knob event-config="$(paste -sd, hopperfish_vtune_events.cfg)" \
      -knob enable-driverless-collection=true \
      -knob sampling-interval=1000 \
      -result-dir "$ROUND_DIR" \
      -quiet -no-summary \
      -- ./cedr -c ./daemon_config.json -l NONE >> outputs.log 2>&1 &
    CEDR_PID=$!

    sleep 1.0
    echo "CEDR PROCESS ID IS $CEDR_PID" >> outputs.log

    # Run perf stat in parallel
    perf stat -e cycles,instructions,cache-references,cache-misses,bus-cycles \
      -p $CEDR_PID -o perf_output.txt --append >> outputs.log 2>&1 &

    (
      echo "timestamp(s),rss(bytes),vsz(bytes)" > mem_output.txt
      echo "timestamp(s),smaps_Rss(kB),smaps_Pss(kB),smaps_Private_Clean(kB),smaps_Private_Dirty(kB)" > smaps_output.txt
      while kill -0 $CEDR_PID 2>/dev/null; do
        ts=$(date +%s.%3N)
        output=$(ps -o rss=,vsz= -p $CEDR_PID)
        rss=$(echo $output | awk '{print $1}')
        vsz=$(echo $output | awk '{print $2}')
        if [[ -n "$rss" && -n "$vsz" && "$rss" -gt 0 ]]; then
          echo "$ts,$((rss * 1024)),$((vsz * 1024))" >> mem_output.txt
        fi
        if [[ -r /proc/$CEDR_PID/smaps_rollup ]]; then
          rss_val=$(grep "Rss:" /proc/$CEDR_PID/smaps_rollup | awk '{print $2}')
          pss_val=$(grep "Pss:" /proc/$CEDR_PID/smaps_rollup | awk '{print $2}')
          pc_val=$(grep "Private_Clean:" /proc/$CEDR_PID/smaps_rollup | awk '{print $2}')
          pd_val=$(grep "Private_Dirty:" /proc/$CEDR_PID/smaps_rollup | awk '{print $2}')
          echo "$ts,$rss_val,$pss_val,$pc_val,$pd_val" >> smaps_output.txt
        fi
        sleep 0.05
      done
    ) &
    MEM_PID=$!

    sleep 3.0

    mapfile -t shuffled_apps < <(shuf --random-source=<(get_seeded_random "$SHUFFLE_SEED") -e "${apps[@]}")
    for app in "${shuffled_apps[@]}"; do
      inst=$(( RANDOM % (app_instances_max - app_instances_min + 1) + app_instances_min ))
      rate=$(( RANDOM % (app_rate_max - app_rate_min + 1) + app_rate_min ))
      cmd="./sub_dag -a $APPS_PATH/$app -n $inst -p $rate"
      echo "Submitting command for $app: $cmd" >> outputs.log
      eval $cmd >> outputs.log 2>&1
      app_instances_count["$app"]=$((app_instances_count["$app"] + inst))
    done

    for app in "${apps[@]}"; do
      echo "Total instances of $app submitted : ${app_instances_count["$app"]}" >> outputs.log
    done

    if (( ROUND_WAIT != -1 )); then
      sleep "$ROUND_WAIT"
      kill "$CEDR_PID" 2>/dev/null && wait "$CEDR_PID"
    else
      SECONDS=0
      while kill -0 $CEDR_PID 2>/dev/null; do
        echo "Waiting... Elapsed time: ${SECONDS}s"
        sleep 1
        ((SECONDS++))
      done
    fi

    kill "$MEM_PID" 2>/dev/null

    echo -e "\n==== Perf Metrics Collected ====" >> outputs.log
    cat perf_output.txt >> outputs.log

    echo -e "\n==== Memory Footprint Summary (RSS from ps) ====" >> outputs.log
    tail -n +2 mem_output.txt | awk -F',' '
      BEGIN {sum=0; count=0; max=0}
      {rss=$2; sum+=rss; count++; if(rss>max) max=rss}
      END {
        print "Average RSS: " int(sum/count) " bytes"
        print "Peak RSS: " max " bytes"
      }' >> outputs.log

    echo -e "\n==== Memory Footprint Summary (Rss from smaps) ====" >> outputs.log
    tail -n +2 smaps_output.txt | awk -F',' '
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
