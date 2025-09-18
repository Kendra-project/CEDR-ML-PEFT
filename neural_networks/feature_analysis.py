import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
import dataframe_image as dfi
import re

# this is for LOF
def calculate_n_neighbors(training_data, contamination):
    n_samples = len(training_data)
    n_outliers = int(n_samples * contamination)
    min_n_neighbors = n_outliers + 1
    max_n_neighbors = 50  # Adjust as per your requirements
    n_neighbors = int(n_samples ** 0.5)
    return max(min_n_neighbors, min(n_neighbors, max_n_neighbors))

def plot_application_accuracy(labels, predictions, app_names, plot_title='Detection Accuracy'):
    """
    Plots the accuracy of each application in the Testing set.

    Parameters:
        labels (array-like): Array of true labels.
        predictions (array-like): Array of predicted labels.
        app_names (array-like): Array of application names.

    Returns:
        None.
    """
  
    unique_apps = np.unique(app_names)
    app_accuracy = {}
    for app in unique_apps:
        app_indices = np.where(app_names == app)[0]
        app_labels = labels[app_indices]
        app_predictions = predictions[app_indices]
        accuracy = 100 * np.mean(app_labels == app_predictions)
        app_accuracy[app] = accuracy
    plt.figure(figsize=(15, 6))
    plt.bar(range(len(app_accuracy)), list(app_accuracy.values()), align='center')
    plt.xticks(range(len(app_accuracy)), list(app_accuracy.keys()), rotation=60)
    plt.title(plot_title)
    plt.xlabel('Application')
    plt.ylabel('Testing Accuracy (%)')
    plt.show()


def calculate_fpr_fnr(test_labels,classified):

    # Initialize the counts
    tn = 0
    fp = 0
    fn = 0
    tp = 0

    # Calculate the counts
    for true_label, predicted_label in zip(test_labels, classified):
        if true_label == 0 and predicted_label == 0:
            tn += 1
        elif true_label == 0 and predicted_label == 1:
            fp += 1
        elif true_label == 1 and predicted_label == 0:
            fn += 1
        elif true_label == 1 and predicted_label == 1:
            tp += 1
    if tn + fp != 0:
        fpr = fp / (fp + tn)
    else:
        fpr = float('nan')  # handle the case when the denominator is zero

    if fn + tp != 0:
        fnr = fn / (fn + tp)
    else:
        fnr = float('nan')  # handle the case when the denominator is zero

    return fpr,fnr        


def compute_metrics(labels, predictions, app_names,results_file='results.csv', plot_metric = 'FNR (Miss Rate) (%)',
    plot_label = 'Anomaly'):
    unique_apps = np.unique(app_names)
    num_apps = len(unique_apps)
    rows = []
    new_app_names = {'DoS_fft-':'ReExF','DoS_zip-':'ReExZ','track_nb-':'LD','SAR_nb-':'SAR',
                     'availtime_zip-':'AvMa','fft_pe_analysis-':'TTA','pulse_doppler-nb-':'PD','radar_correlator-':'RC'
                     ,'radar_correlator-zip-':'RCFZ','radar_correlator_v1-':'RCM','temporal_mitigation-':'TM','wifi-tx-nb-':'WiFi TX','spectre_v1-':'Spectre-V1'
                     ,'temporal_mitigation-noapi-':'SaTM','radar_correlator-noapi-':'SaRC','SAR-noapi-':'SaSAR',
                     'udma_read-fft-':'DMA-R-FFT','udma_read-gemm-':'DMA-R-GEMM','udma_read-zip-':'DMA-R-ZIP',
                     'udma_write-fft-':'DMA-W-FFT','udma_write-gemm-':'DMA-W-GEMM','udma_write-zip-':'DMA-W-ZIP'
                    } 
    plt.figure(figsize=(15, 15))
   
    for i, app in enumerate(unique_apps):
        app_indices = np.where(app_names == app)[0]
        app_labels = labels[app_indices]
        app_predictions = predictions[app_indices]
        tp = np.sum(np.logical_and(app_labels == 1, app_predictions == 1))
        tn = np.sum(np.logical_and(app_labels == 0, app_predictions == 0))
        fp = np.sum(np.logical_and(app_labels == 0, app_predictions == 1))
        fn = np.sum(np.logical_and(app_labels == 1, app_predictions == 0))
        # Extract scheduler and platform names from app name
        scheduler = re.search(r'_([A-Z]+)_', app).group(1)
        platform = re.search(r'_([^_]+)$', app).group(1)
        app = re.search(r'(.+?)_[A-Z]+_[^_]+$', app).group(1)
        #print(app)
        if app in new_app_names:
                app = new_app_names[app]
        


        if app_labels[0] == 0:
            label = "Normal"
        else:
            label = "Anomaly"   
        
        row = {
            'App': app,
            'Scheduler': scheduler,
            'Platform': platform,
            'Label': label,
            'Total': len(app_labels),
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'ACC (%)': 100 * ((tp+tn) / len(app_labels)),
            'FPR (%)': 100 * (fp / (fp+tn)),
            'FNR (Miss Rate) (%)': 100 * (fn / (fn+tp))
        }
        rows.append(row)
        '''
        plt.subplot(4, 4, i % 16 + 1)
        plt.title(app)
        plt.imshow([[tp, fp], [fn, tn]], cmap=plt.cm.Blues, interpolation='nearest',aspect='auto')
        plt.xticks([1,0], ['Actual Normal', 'Actual Anomaly'])
        plt.yticks([1,0], ['Predicted Normal', 'Predicted Anomaly'])
        plt.colorbar()
        plt.subplots_adjust(hspace=0.5)  
        plt.subplots_adjust(wspace=0.5)
        '''      
    
    df = pd.DataFrame(rows, columns=['App', 'Label', 'Total','Scheduler', 'Platform', 'TP', 'TN', 'FP', 'FN','ACC (%)','FPR (%)','FNR (Miss Rate) (%)'])
    print(df.to_string(index=False))
    df.to_csv(results_file, index=False)
    return 
    # Set the width of the lines
    line_width = 2

    # Filter the dataframe for applications with "Normal" label
    filtered_df = df[df['Label'] == plot_label]

    # Get unique apps, schedulers, and platforms
    apps = filtered_df['App'].unique()
    schedulers = filtered_df['Scheduler'].unique()
    platforms = ['Jetson','ZCU102','OptiPlex']
    #print(platforms)
    # Set colors for lines
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    # Create subplots
    fig, ax = plt.subplots(figsize=(10, 6))
    # Define line styles for each platform
    line_styles = {
        'ZCU102': '-',
        'OptiPlex': '--',
        'Jetson': ':'
    }


    # Plotting for each scheduler-platform combination
    for scheduler in schedulers:
        for platform in platforms:
            #print("Platform is: ", platform)
            # Filter the dataframe for the specific scheduler and platform
            subset_df = filtered_df[(filtered_df['Scheduler'] == scheduler) & (filtered_df['Platform'] == platform)]
            
            # Get the FPR values for the app
            y_values = subset_df[plot_metric]
                # Smooth line transition by increasing data points
            # Smooth line transition by increasing data points
            x_smooth = np.linspace(0, len(apps) - 1, 100)
            y_smooth = np.interp(x_smooth, np.arange(len(apps)), y_values)
            
            # Plot the smoothed line with specific line style
            ax.plot(x_smooth, y_smooth, label=f"{scheduler} - {platform}", linewidth=line_width, linestyle=line_styles[platform])

    # Set the x-axis ticks and labels
    ax.set_xticks(range(len(apps)))
    ax.set_xticklabels(apps)
        # Set labels and title
    ax.set_xlabel('App Name', fontsize=20)
    ax.set_ylabel(plot_metric,fontsize=20)
    #ax.set_title('FPR for Applications with "Normal" Label')

    # Set legend
    #ax.legend(title='Scheduler - Platform',fontsize=18,title_fontsize=18)
    # Set the font size for tick labels on the x-axis and y-axis
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    # Rotate x-axis labels if needed
    plt.xticks(rotation=75)

    # Show the plot
    plt.tight_layout()
    plt.savefig('All_Classes' +  plot_label +  '_Applications_'+ plot_metric, dpi=500, bbox_inches='tight')
    plt.show()

    # Set legend
    #ax.legend(title='Scheduler - Platform', fontsize=20, title_fontsize=20)

    # Create a separate figure for the legend
    #fig_legend = plt.figure(figsize=(6, 4))
    #legend = fig_legend.legend(*ax.get_legend_handles_labels(), title='Scheduler - Platform', fontsize=20, title_fontsize=20)
    #fig_legend.canvas.draw()

    # Save the legend figure
    #fig_legend.savefig('Legend.png', dpi=500, bbox_inches='tight')
    #plt.tight_layout()  
    #plt.show()

    #df_styled =  df.style.background_gradient() #adding a gradient based on values in cell

    #dfi.export(df_styled,"mytable.png")


# Example usage:
#labels = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 1])
#predictions = np.array([1, 0, 1, 0, 1, 0, 1, 1, 1, 1])
#app_names = np.array(['app1', 'app1', 'app1', 'app2', 'app2', 'app2', 'app3', 'app3', 'app4', 'app4'])

#df = compute_metrics(labels, predictions, app_names)
#print(df.to_string(index=False))


def preprocess_real(device,scheduler,LOG_PATH):

    all_data = []

    N = 13 # number of the first unchanged columns and their headers
    P = 18 # number of the PERF features (at the end of each dataframe)
   
    # Create the binary coding map
    binary_map = {
        "SIMPLE": "11",
        "MET": "01",
        "EFT": "00",
    }


    data_path = f"{LOG_PATH}"
    files = [os.path.join(root, f) for root, dirs, files in os.walk(data_path) for f in files if f.endswith('.csv')]
    
    #print("Explored files are:",  files)  # Check the files list
    data = []
    for file in files:
        df = pd.read_csv(file)
        # Replace spaces and special characters with underscores
        df.columns = df.columns.str.replace('[^A-Za-z0-9]+', '_')
        todo_queue_cols = [col for col in df.columns if 'todo_queue' in col and 'max_in_todo_queue' not in col]
        if len(todo_queue_cols) > 0:
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
            scheduler_code_1 = binary_map[scheduler][0]
            scheduler_code_2 = binary_map[scheduler][1]
            count_data = pd.DataFrame({
                'CPU_count': [CPU_count] * len(df),
                'GPU_count': [GPU_count] * len(df),
                'FFT_count': [FFT_count] * len(df),
                'GEMM_count': [GEMM_count] * len(df),
                'ZIP_count': [ZIP_count] * len(df),
                'scheduler_code_1': [scheduler_code_1] * len(df),
                'scheduler_code_2': [scheduler_code_2] * len(df)
            })

            row = pd.concat([df[uncahnged_header_cols],df[perf_header_cols], todo_queue_min, todo_queue_max, todo_queue_mean, PE_min, PE_max,
                        PE_mean,count_data], axis=1, ignore_index=True)
            row.columns = uncahnged_header_cols + perf_header_cols +  ['todo_queue_min', 'todo_queue_max', 'todo_queue_mean', 
                'PE_min', 'PE_max', 'PE_mean','CPU_count','GPU_count','FFT_count','GEMM_count','ZIP_count', 'scheduler_code_1', 'scheduler_code_2']
            row["scheduler"] = "RR" if scheduler == "SIMPLE" else scheduler
            row["device"] = device
            data.append(row)

            combined_data = pd.concat(data, axis=0, ignore_index=True)


        #print(final_data)
    return combined_data 

def preprocess_global(data_class,device,scheduler, DATASET_PATH ="./datasets"):

    all_data = []

    N = 13 # number of the first unchanged columns and their headers
    P = 18 # number of the PERF features (at the end of each dataframe)
   
    # Create the binary coding map
    binary_map = {
        "SIMPLE": "11",
        "MET": "01",
        "EFT": "00",
    }
    for d in sorted(device):

        for sch in sorted(scheduler):  
            #print("Device is : ", d)
            #print("Scheduler is : ", sch)
            data_path = f"{DATASET_PATH}/{d}/{data_class}/{sch}"
            files = [os.path.join(root, f) for root, dirs, files in os.walk(data_path) for f in files if f.endswith('.csv')]
            
            #print("Explored files are:",  files)  # Check the files list
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
        #print(final_data)
    return final_data 

def generate_test_labels(dfs):
    test = []
    test_labels = []
    for key, df in dfs.items():

        if key == "test":
            #print(df.columns)

            df = df.drop(df.index[0])   
            # Set testidation labels based on application name
            labels = (df.iloc[:, 1].str.contains('DoS|availtime|analysis|spectre|udma') == True).astype(int)
            # Append labels to test_labels list
            test_labels.extend(labels.tolist())
            #test_app_names = df.iloc[:, 1].values
            df.iloc[:, 1] = df.iloc[:, 1].str.replace('x86', '').str.replace('aarch64', '').str.replace('jetson', '')
            test_app_names = df.iloc[:, 1].values     
            test_scheduler_names = df.iloc[:, -2].values
            test_device_names = df.iloc[:, -1].values
            #print("test applications count = ", len(test_app_names))
            df = df.drop(df.columns[:2], axis=1)
            df = df.drop(df.columns[-1], axis=1)
            df = df.drop(df.columns[-1], axis=1)
            test.append(df)     


    #print('First lines of test data ..')
    test_df = pd.concat(test, axis=0)
    test_df = test_df.astype(float)

    outputs = {
        'test_df': test_df,
        'test_labels': test_labels,
        'test_app_names':test_app_names,
        'test_device_names':test_device_names,
        'test_scheduler_names':test_scheduler_names,
    }  
    return outputs  

def split_train_val_test(dfs):
    test = []
    train = []
    val = []
    test_labels = []
    val_labels = []
    train_labels = []
    for key, df in dfs.items():
        #file_path = os.path.join(data_path, file)
        #drop the first row:
        #test_global
        if key == "test":
            #print(df.columns)

            df = df.drop(df.index[0])   
            # Set testidation labels based on application name
            labels = (df.iloc[:, 1].str.contains('DoS|availtime|analysis|spectre|udma') == True).astype(int)
            # Append labels to test_labels list
            test_labels.extend(labels.tolist())
            #test_app_names = df.iloc[:, 1].values
            df.iloc[:, 1] = df.iloc[:, 1].str.replace('x86', '').str.replace('aarch64', '').str.replace('jetson', '')
            test_app_names = df.iloc[:, 1].values     
            test_scheduler_names = df.iloc[:, -2].values
            test_device_names = df.iloc[:, -1].values
            #print("test applications count = ", len(test_app_names))
            df = df.drop(df.columns[:2], axis=1)
            df = df.drop(df.columns[-1], axis=1)
            df = df.drop(df.columns[-1], axis=1)
            test.append(df)


        elif key == "train":
            labels =  (df.iloc[:, 1].str.contains('DoS|availtime|analysis|spectre|udma') == True).astype(int)
            train_labels.extend(labels.tolist())
            df.iloc[:, 1] = df.iloc[:, 1].str.replace('x86', '').str.replace('aarch64', '').str.replace('jetson', '')
            train_app_names = df.iloc[:, 1].values     
            train_scheduler_names = df.iloc[:, -2].values
            train_device_names = df.iloc[:, -1].values
            df = df.drop(df.columns[:2], axis=1)

            df = df.drop(df.columns[-1], axis=1)
            df = df.drop(df.columns[-1], axis=1)

            train.append(df)

    
        elif key == "val":
            labels =  (df.iloc[:, 1].str.contains('DoS|availtime|analysis|spectre|udma') == True).astype(int)
            val_labels.extend(labels.tolist())
            df.iloc[:, 1] = df.iloc[:, 1].str.replace('x86', '').str.replace('aarch64', '').str.replace('jetson', '')
            val_app_names = df.iloc[:, 1].values     
            val_scheduler_names = df.iloc[:, -2].values
            val_device_names = df.iloc[:, -1].values
            df = df.drop(df.columns[:2], axis=1)

            df = df.drop(df.columns[-1], axis=1)
            df = df.drop(df.columns[-1], axis=1)

            val.append(df)      


    #print('First lines of test data ..')
    test_df = pd.concat(test, axis=0)
    #print(test_df.head())
    #print('First lines of train data ..')
    train_df = pd.concat(train, axis=0)
    #print(train_df.head())
    #print('First lines of val data ..')
    val_df = pd.concat(val, axis=0)
    #print(val_df.head())
    train_df = train_df.astype(float)
    test_df = test_df.astype(float)
    val_df = val_df.astype(float)

    outputs = {
        'train_df': train_df,
        'train_labels': train_labels,
        'val_df': val_df,
        'val_labels': val_labels,
        'test_df': test_df,
        'test_labels': test_labels,
        'test_app_names':test_app_names,
        'test_device_names':test_device_names,
        'test_scheduler_names':test_scheduler_names,
        'train_app_names':train_app_names,
        'train_device_names':train_device_names,
        'train_scheduler_names':train_scheduler_names      
    }  
    return outputs
