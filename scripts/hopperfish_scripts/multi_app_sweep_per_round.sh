#!/bin/bash

# Define the random seed for this script:
#RANDOM=1 #1 is the seed used for training, 2 is for val, and 3 is for testing
SCALE_FACTOR=3
DATASET_SEED=400 # 0 for training, 100 for val, 200 for testing normal, 300 for testing anomaly, 400 for testing mixed. 
# Define the list of applications and their parameters

# Kill any existing CEDR processes
pkill -f cedr


#declare -a SCHEDS=("SIMPLE" "MET" "EFT") 
declare -a SCHEDS=("SIMPLE") 
cpu="11"
fft="0"
mmult="0"
gpu="0"
zip="0"


#APPS_PATH="./val_anomaly_apps"
# Access the variable_name passed as an argument
APPS_PATH="$1"

if [ -z "$1" ]; then
  echo "Error: Application path not provided. Usage: ./multi_app_sweep_per_round.sh <app_folder>"
  exit 1
fi
echo "The used app class is: $APPS_PATH"

OUTPUT_BASE_DIR="./log_dir"

# Prompt user for subdirectory name
echo "Please enter a name for the subdirectory (e.g., train, val): "
read SUB_DIR_NAME

echo "Please enter a maximum amount of time to wait before kiling CEDR in seconds, enter -1 to wait for CEDR till the end."
read ROUND_WAIT


# Create the subdirectory and scheduler-specific subdirectories
mkdir -p "$OUTPUT_BASE_DIR/$SUB_DIR_NAME/SIMPLE"
mkdir -p "$OUTPUT_BASE_DIR/$SUB_DIR_NAME/MET"
mkdir -p "$OUTPUT_BASE_DIR/$SUB_DIR_NAME/EFT"

# Define the cumulative CSV file path for each scheduler
declare -A cumulative_csv_files
for sched in "${SCHEDS[@]}"; do
  cumulative_csv_files["$sched"]="$OUTPUT_BASE_DIR/$SUB_DIR_NAME/$sched/cumulative_ad_features.csv"
  touch "${cumulative_csv_files["$sched"]}"
done

# Define the number of experiments to run (10 for training, 5 for each validation/test set)
N=1
app_instances_min=2 # 2 for normal train and normal val/test, 1 for anomaly/mixed test
app_instances_max=5 # 5 for normal train and normal val/test, 1 for anomaly/mixed test
app_injection_rate_min=100000
app_injection_rate_max=200000



# Specify the minimum and maximum sleep delays
min_delay=1 #for train and normal val/test is 5, for anomaly and mixed test is 10
max_delay=5 # for train and normal val/test is 10,  for anomaly and mixed test is 15


# Create the folder for reproducible experiments if it doesn't exist


# Randomly generate sleep delays within the given range

if [ -f "outputs.log" ]; then
    rm outputs.log
fi

# Initialize the counters for each application type
declare -A app_instances_count
for app in "${apps[@]}"; do
  app_instances_count["$app"]=0
done


declare -a apps=($(find "$APPS_PATH" -name "*.so" -type f -exec basename {} \;))

get_seeded_random()
{
	seed="$1"
	openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
		</dev/zero 2>/dev/null
}
for sched in "${SCHEDS[@]}"; do
  bash generate_daemon_config.sh "$cpu" "$fft" "$mmult" "$gpu" "$zip" "$sched"
  #sleep 2.0 # 2s delay    
	# Iterate over the number of experiments


	for n in $(seq 1 $N); do
		#delay_1=$(shuf -i $(printf "%.0f" "$min_delay")-$(printf "%.0f" "$max_delay") -n 1)
		#delay_2=$(shuf -i $(printf "%.0f" "$min_delay")-$(printf "%.0f" "$max_delay") -n 1)
		# Set the random seed to the current round index

		RANDOM=$(($n + $DATASET_SEED))
		SHUFFLE_SEED=$(($n + $DATASET_SEED))
		#delay_1=$(( RANDOM % (max_delay - min_delay + 1) + min_delay ))  
		#delay_2=$(( RANDOM % (max_delay - min_delay + 1) + min_delay ))	
		#./cedr -c ./daemon_config.json -l NONE >> outputs.log & #> /dev/null 2>&1 &
		./cedr -c ./daemon_config.json & #>
		sleep 1.0 # 1s delay    
		CEDR_PID=$!
		echo "CEDR PROCESS ID IS $CEDR_PID"				
		#echo "Delay 1 = $delay_1"
		#echo "Delay 2 = $delay_2"
    # Print the current round and scheduler type
    echo "Scheduler: $sched - Round: $n"
		# Shuffle the order of applications randomly for each experiment
		#shuffled_apps=($(shuf -e "${apps[@]}"))
		#shuffled_apps=($(shuf --random-source=<(echo -n "$RANDOM") -e "${apps[@]}"))
  	# Generate a consistent shuffle of apps based on RANDOM_SEED
		# Generate random numbers and pair them with apps
   # Reinitialize shuffled_apps array for each round

		shuffled_apps=($(shuf --random-source=<(get_seeded_random $SHUFFLE_SEED) -e "${apps[@]}"))
		# Iterate over the shuffled list of applications and submit the command for each one
		for i in "${!shuffled_apps[@]}"; do

		  # Generate random instance numbers and injection rates for the current application
		  #app_instances=$(shuf -i ${app_instances_min}-${app_instances_max} -n 1)
		  #app_injection_rate=$(shuf -i ${app_injection_rate_min}-${app_injection_rate_max} -n 1)
  			app_instances=$(( (RANDOM) % (app_instances_max - app_instances_min + 1) + app_instances_min ))
    			app_injection_rate=$(( (RANDOM*SCALE_FACTOR) % (app_injection_rate_max - app_injection_rate_min + 1) + app_injection_rate_min ))
		  # Construct the command for running sub_dag for the current application
		 	 command="./sub_dag -a $APPS_PATH/${shuffled_apps[$i]} -n $app_instances -p $app_injection_rate"
		
		  	echo "Submitting command for ${shuffled_apps[$i]}: $command"

		  # Run the command
		  eval $command
		  #sleep "$delay_1"

		  # Update the counter for the current application type
		  app_instances_count["${shuffled_apps[$i]}"]=$((${app_instances_count["${shuffled_apps[$i]}"]} + $app_instances))
		done

		#sleep "$delay_2"
		# Print the total number of instances of each application type to the terminal and the experiment script
		for app in "${apps[@]}"; do
			instances=${app_instances_count["$app"]}
			echo "Total instances of $app submitted : $instances"

		done	
    if [ "$ROUND_WAIT" -ne -1 ]; then
      echo "All submissions are done. Waiting for $ROUND_WAIT seconds before checking CEDR status."
      # Wait for the specified duration
      sleep "$ROUND_WAIT"
      # Check if CEDR is still running before trying to kill it
      if kill -0 $CEDR_PID 2>/dev/null; then
        echo "Forcibly terminating CEDR after $ROUND_WAIT seconds."
        ./kill_daemon
         wait $CEDR_PID	
      else
        echo "CEDR process has already terminated."
      fi
    else  
			echo "All submissions are done, waiting for CEDR to get idle"

			# Initialize the timer
			SECONDS=0

			# Wait for CEDR to terminate and display the timer
			while kill -0 $CEDR_PID 2>/dev/null; do
				echo "Waiting... Elapsed time: ${SECONDS}s"
				sleep 1
				((SECONDS++))
			done

			echo "CEDR process has been terminated. Total wait time: ${SECONDS}s"
		fi	
		
	  # Find the latest CSV file generated in ./log_dir/
    LATEST_CSV=$(find ./log_dir/experiment* -type f -name '*.csv' -printf '%T+ %p\n' | sort -r | head -n 1 | cut -d' ' -f2)
		# or you can use: LATEST_CSV=$(find ./log_dir/experiment* -type f -name '*.csv' | xargs ls -ltr | tail -n 1 | awk '{ print $(NF) }')
    # Append CSV content to the cumulative file, removing header if not the first round
    if [ ! -z "$LATEST_CSV" ]; then
      if [ $n -gt 1 ]; then
        # Skip the header row for rounds greater than 1
        tail -n +2 "$LATEST_CSV" >> "${cumulative_csv_files["$sched"]}"
      else
        # Include the header row for the first round
        cat "$LATEST_CSV" >> "${cumulative_csv_files["$sched"]}"
      fi
      echo "Appended $LATEST_CSV to ${cumulative_csv_files["$sched"]}"
    else
      echo "No new CSV file found for round $n."
    fi	
		
	done


done


  # Wait for a new CSV file to be generated in /log_dir/
  #echo "Waiting for new CSV file to be generated in ./log_dir/..."
  #LATEST_CSV=""
	#while [ -z "$LATEST_CSV" ]; do
  #LATEST_CSV=$(find ./log_dir/ -type f -name '*.csv' -printf '%T+ %p\n' | sort -r | head -n 1 | cut -d' ' -f2)
	    #sleep 3 # Check every 3 seconds
	#done
  # Move the latest CSV file to the appropriate subdirectory
  #if [ ! -z "$LATEST_CSV" ]; then
  #    mv "$LATEST_CSV" "$OUTPUT_BASE_DIR/$SUB_DIR_NAME/$sched/"
  #    echo "Moved $LATEST_CSV to $OUTPUT_BASE_DIR/$SUB_DIR_NAME/$sched/"
  #else
  #    echo "No new CSV file found."
  #fi









