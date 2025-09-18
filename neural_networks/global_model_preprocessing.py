import pandas as pd
import os
import warnings

# Disable all warnings
warnings.filterwarnings("ignore")

scheduler = {"EFT"}
device  = {"OptiPlex"}
all_data = []
data_class ="train"
N = 13 # number of the first unchanged columns and their headers
P = 18 # number of the PERF features (at the end of each dataframe)
DATASET_PATH ="./datasets" 
# Create the binary coding map
binary_map = {
    "SIMPLE": "11",
    "MET": "01",
    "EFT": "00",
}
for i, task in enumerate((scheduler)):
    print(task)

for d in sorted(device):

    for sch in sorted(scheduler):  
        print("Device is : ", d)
        print("Scheduler is : ", sch)
        data_path = f"{DATASET_PATH}/{d}/{data_class}/{sch}"
        print(data_path)
        files = [os.path.join(root, f) for root, dirs, files in os.walk(data_path) for f in files if f.endswith('.csv')]
        
        print("Explored files are:",  files)  # Check the files list
        data = []
        for file in files:
            df = pd.read_csv(file)
            # Replace spaces and special characters with underscores
            df.columns = df.columns.str.replace('[^A-Za-z0-9]+', '_')

            if data_class in file and sch in file:
                todo_queue_cols = [col for col in df.columns if 'todo_queue' in col and 'max_in_todo_queue' not in col]
                if len(todo_queue_cols) > 0:
                    #header_cols = ['app_id', 'app_name', 'app_size', 'app_runtime', 'app_task_count', 'max_in_ready_queue',
                    #                'ref_arrival_time','app_diff_arrival_time', 
                    #                'app_lifetime','app_api_min','app_api_max','app_api_mean']
                    uncahnged_header_cols = df.columns[:N].tolist()
                    perf_header_cols = df.columns[-P:].tolist()
                    unchanged_cols = [col for col in df.columns if col not in todo_queue_cols and col not in uncahnged_header_cols]
                    todo_queue_min = df[todo_queue_cols].min(axis=1).astype(str)
                    todo_queue_max = df[todo_queue_cols].max(axis=1).astype(str)
                    todo_queue_mean = df[todo_queue_cols].mean(axis=1).astype(str)
                    last_todo_queue_col = todo_queue_cols[-1]
                    start_index = df.columns.get_loc(last_todo_queue_col) + 1
                    PE_cols = df.columns[start_index:-P]
                    PE_min = df[PE_cols].min(axis=1).astype(str)
                    PE_max = df[PE_cols].max(axis=1).astype(str)
                    PE_mean = df[PE_cols].mean(axis=1).astype(str)
                    PE_cols_str = df[PE_cols].columns.astype(str)  # Convert column names to strings
                    CPU_count = PE_cols_str.str.contains('cpu').sum()
                    GPU_count = PE_cols_str.str.contains('gpu').sum()
                    FFT_count = PE_cols_str.str.contains('fft').sum()
                    GEMM_count = PE_cols_str.str.contains('gemm').sum()
                    ZIP_count = PE_cols_str.str.contains('zip').sum()
                    scheduler_code_1 = binary_map[sch][0]
                    scheduler_code_2 = binary_map[sch][1]
                    #print("scheduler_code_1:" ,scheduler_code_1)
                    #print("scheduler_code_2:" ,scheduler_code_2)
              
                    #scheduler_code = binary_map[sch]
                    # Create a list of count values with the same length as the DataFrame
                    #count_data = pd.DataFrame({'CPU_count': [CPU_count] * len(df)})
                    # Create a list of count values with the same length as the DataFrame
                    count_data = pd.DataFrame({
                        'CPU_count': [CPU_count] * len(df),
                        'GPU_count': [GPU_count] * len(df),
                        'FFT_count': [FFT_count] * len(df),
                        'GEMM_count': [GEMM_count] * len(df),
                        'ZIP_count': [ZIP_count] * len(df),
                        'scheduler_code_1': [scheduler_code_1] * len(df),
                        'scheduler_code_2': [scheduler_code_2] * len(df)
                    })

                    #print(count_data)
                    row = pd.concat([df[uncahnged_header_cols],df[perf_header_cols], todo_queue_min, todo_queue_max, todo_queue_mean, PE_min, PE_max,
                                    PE_mean,count_data], axis=1, ignore_index=True)
                    row.columns = uncahnged_header_cols + perf_header_cols +  ['todo_queue_min', 'todo_queue_max', 'todo_queue_mean', 
                            'PE_min', 'PE_max', 'PE_mean','CPU_count','GPU_count','FFT_count','GEMM_count','ZIP_count', 'scheduler_code_1', 'scheduler_code_2']
                    row["scheduler"] = "RR" if sch == "SIMPLE" else sch
                    row["device"] = d
                    data.append(row)

        combined_data = pd.concat(data, axis=0, ignore_index=True)

        all_data.append(combined_data)

    final_data = pd.concat(all_data, axis=0, ignore_index=True)
    print(final_data)
    final_data.to_csv(f"{DATASET_PATH}/combined_{data_class}.csv", index=False, header=True)






    # One-hot encode the scheduler and device columns
    #combined_data = pd.get_dummies(combined_data, columns=['scheduler'])
    #combined_data = pd.get_dummies(combined_data, columns=['device'])
# Apply one-hot encoding to the specified columns
  #  encoded_data = pd.get_dummies(combined_data[['scheduler', 'device']], columns=['scheduler', 'device'], drop_first=False, prefix='', prefix_sep='')
  #  combined_data = pd.concat([combined_data, encoded_data], axis=1)


#expected_columns = combined_data.columns.tolist()
#print(expected_columns)
#final_data.fillna(0, inplace=True)

# Define the two columns to move to the end
#columns_to_move = ['scheduler', 'device']

# Remove the two columns from the column names
#column_names = [col for col in expected_columns if col not in columns_to_move]

# Append the two columns to the end
#column_names += columns_to_move

# Reindex the combined_data DataFrame with the updated column order
#final_data = final_data.reindex(columns=column_names)
# Identify the expected columns

# Reindex and fill missing columns with zeros