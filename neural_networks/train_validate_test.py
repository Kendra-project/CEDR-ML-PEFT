import warnings

# Disable all warnings
warnings.filterwarnings("ignore")

import numpy as np
import csv
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import re
from numpy import random
from sklearn.metrics import accuracy_score, precision_score, recall_score,fbeta_score

from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from models  import *
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from feature_analysis import *
from sklearn.metrics import confusion_matrix
from scipy.stats import norm
import random
from keras import callbacks
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from numpy import nan
import logging

from datetime import datetime




from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error
MODEL_TYPE = "AE" # AE, OCSVM, iForest,LOF, VAE, AE_HLS4ML
HYPERTUNE = True
STD_DEV_K = 2
MODEL_NAME ="AE"
RESULTS_CLASS = 'test_normal_anomaly_mixed'
SELECTED_FEATURES = "full_stack_features" #hardware_features app_features micro_features
EXPORT_MODEL = True
IMPORT_MODEL = False
PLOT_ENABLED = False
# Construct the MODEL_PATH
MODEL_PATH = "./saved_models/new_model" 
# Create the directory if it does not exist
os.makedirs(MODEL_PATH, exist_ok=True)



print(f"Directory '{MODEL_PATH}' is ensured to exist.")


# File to store the current run index
run_index_file = "run_index.txt"

# Initialize or load the run index
if os.path.exists(run_index_file):
    with open(run_index_file, "r") as f:
        run_index = int(f.read().strip())
else:
    run_index = 0

# Increment the run index for the current run
run_index += 1

# Save the updated run index back to the file
with open(run_index_file, "w") as f:
    f.write(str(run_index))

# Configure logging to append to a log file
log_file = "output.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),  # Append to log file
        logging.StreamHandler()  # Optionally log to console
    ]
)

# Define a function to log a new run with separation
def log_run(run_index, message, separator="-" * 50):
    logging.info(separator)
    logging.info(f"Run Index: {run_index}")
    logging.info(f"Timestamp: {datetime.now()}")
    logging.info(separator)
    logging.info(message)
    logging.info(separator + "\n")

# Example usage
hyperparameters = f"""
Hyperparameters and Settings:
MODEL_TYPE: {MODEL_TYPE}
HYPERTUNE: {HYPERTUNE}
STD_DEV_K: {STD_DEV_K}
MODEL_NAME: {MODEL_NAME}
RESULTS_CLASS: {RESULTS_CLASS}
SELECTED_FEATURES: {SELECTED_FEATURES}
EXPORT_MODEL: {EXPORT_MODEL}
IMPORT_MODEL: {IMPORT_MODEL}
PLOT_ENABLED: {PLOT_ENABLED}
MODEL_PATH: {MODEL_PATH}
"""
log_run(run_index, hyperparameters)



devices =  {"OptiPlex","Jetson","ZCU102"}
train_global = preprocess_global(data_class="train",device = devices,scheduler = {"SIMPLE","MET","EFT"})
test_global_normal = preprocess_global(data_class="test_normal",device =devices,scheduler ={"SIMPLE","MET","EFT"})
test_global_anomaly = preprocess_global(data_class="test_anomaly",device = devices,scheduler ={"SIMPLE","MET","EFT"})
test_global_mixed = preprocess_global(data_class="test_mixed",device = devices,scheduler ={"SIMPLE","MET","EFT"})
test_global =   pd.concat([test_global_normal, test_global_anomaly, test_global_mixed], axis=0)

# Shuffle the data points in-place
test_global = test_global.sample(frac=1, random_state=42).reset_index(drop=True)
val_global = preprocess_global(data_class="val",device = devices,scheduler ={"SIMPLE","MET","EFT"})
#print("test_global is: ",test_global )

dfs = {"train": train_global,"val": val_global, "test": test_global}
data = split_train_val_test(dfs)
train_df = data['train_df']
train_labels = data['train_labels']
val_df = data['val_df']
val_labels = data['val_labels']
test_df = data['test_df']
test_labels = data['test_labels']
test_app_names = data['test_app_names']
test_device_names = data['test_device_names']
test_scheduler_names = data['test_scheduler_names']
#print(train_df.head())  # Displays the first 5 rows along with column names
#print(train_df.columns)

app_features = ['app_size', 'app_runtime','app_task_count']
runtime_features = ['max_in_ready_queue',
                    'max_in_todo_queue','app_lifetime','app_diff_arrival_time',
                    'app_api_min','app_api_max','app_api_mean','todo_queue_min',
                    'todo_queue_max','todo_queue_mean','scheduler_code_1','scheduler_code_2']
hardware_features =['PE_min','PE_max','PE_mean','CPU_count','GPU_count','FFT_count','GEMM_count','ZIP_count']

# Assuming train_df is your DataFrame, get all column names
all_columns = train_df.columns.tolist()

# Define the micro features as columns that start with 'API_' or 'APP_'
micro_features = [col for col in all_columns if col.startswith('API_') or col.startswith('APP_')]

# Print the micro features
#print(micro_features)

# Display the first few rows of the sub-dataframe
#print(app_features.head())

# Dynamically combine feature subsets (e.g., hardware + micro + runtime + app)
hardware_plus_micro = hardware_features + micro_features
hardware_plus_micro_plus_runtime = hardware_features + micro_features + runtime_features
full_stack_features = hardware_features + micro_features + runtime_features + app_features
run_plus_app = runtime_features + app_features

# Dynamically replace the selected features using eval
train_df = train_df[eval(SELECTED_FEATURES)]
val_df = val_df[eval(SELECTED_FEATURES)]
test_df = test_df[eval(SELECTED_FEATURES)] 
# Display the first few rows to confirm
print(train_df.head())
#exit()
# apply log normalization to each feature
for col in train_df.columns:
      train_df[col] = np.power(np.log2(np.abs(train_df[col]) + 1),1)  # add 1 to avoid taking the log of 0

# apply log normalization to each feature
for col in val_df.columns:
      val_df[col] = np.power(np.log2(np.abs(val_df[col]) + 1),1)  # add 1 to avoid taking the log of 0

# apply log normalization to each feature
log_start = time.time()
for col in test_df.columns:
      test_df[col] = np.power(np.log2(np.abs(test_df[col]) + 1),1)  # add 1 to avoid taking the log of 0
log_end= time.time()
test_log_time = log_end - log_start
# show the first few rows of the normalized data
print('First few rows of the preprocessed test data:')
print(test_df[:5])
print('First few rows of the preprocessed train data:')
print(test_df[:5])
test_df = pd.DataFrame(test_df)
train_df = pd.DataFrame(train_df)
val_df = pd.DataFrame(val_df)

# convert dataframes to arrays
train = train_df.to_numpy()
val = val_df.to_numpy()
test = test_df.to_numpy()

# Save the arrays into an NPZ file
#np.savez('test_data_zcu102_threadfish1.npz', inputs=test, labels=test_labels)

#print("Test data and labels have been saved to 'test_data.npz'.")
#exit()
# Initialize the MinMaxScaler
#scaler = MinMaxScaler()

# Fit the scaler to the training data and transform training data
#train = scaler.fit_transform(train)

# Apply the same transformation to validation and test data
#val = scaler.transform(val)
#test = scaler.transform(test)
# train data only contains normal data
train_labels = [0]*len(train)
# data preparing is done, now let's train 
input_shape = train.shape[1:]



print("shape of train is = "+str(train.shape[0])+"x"+str(train.shape[1]))
print("shape of val is = "+str(val.shape[0])+"x"+str(val.shape[1]))

train = tf.convert_to_tensor(train, dtype=tf.float32)
val = tf.convert_to_tensor(val, dtype=tf.float32)
test = tf.convert_to_tensor(test, dtype=tf.float32)

print("shape of test is = "+str(test.shape[0])+"x"+str(test.shape[1]))



#history = autoencoder.fit(train, train, epochs=256, batch_size=16, shuffle=True, verbose = 0)
#optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  # Adjust learning rate as needed
#autoencoder.compile(optimizer=optimizer, loss='mse')

# Create an instance of the custom callback

if(MODEL_TYPE == "VAE"):
      # Initialize the MinMaxScaler
      scaler = MinMaxScaler()

      # Fit the scaler to the training data and transform training data
      t0 = time.time()
      train = scaler.fit_transform(train)

      # Apply the same transformation to validation and test data
      val = scaler.transform(val)
      test = scaler.transform(test)
      infernece_overhead = time.time() - t0    
      if(not IMPORT_MODEL): 
            tf.random.set_seed(123)  # Replace 123 with your desired random seed testue
            input_dim = train.shape[1]
            intermediate_dim = int(input_dim / 2)
            latent_dim = int(input_dim / 3)
            learning_rate = 0.01
            model = create_vae(input_dim, intermediate_dim, latent_dim)
      
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

            earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                          mode="min", patience=20,
                                          restore_best_weights=True)
            #tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
            start_time = time.time()
            history = model.fit(train, train, epochs=500, batch_size=64, shuffle=True, verbose=0,
                                    validation_data=(val, val),
                                    callbacks=[earlystopping]
                                    )
            end_time = time.time()
            train_time = int((end_time - start_time) * 1000)
      #optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  # Adjust learning rate as needed
      # Save the model to a directory
      #tf.saved_model.save(autoencoder, model_path + "/" + device + "_" + PEs + "_" + scheduler + "_model")
            print("Elapsed training time: {:.2f} ms".format((train_time)))
if(MODEL_TYPE == "AE" or MODEL_TYPE == "AE_HLS4ML"):

      if not IMPORT_MODEL:
            tf.random.set_seed(123)  # Replace 123 with your desired random seed testue
            if(MODEL_TYPE == "AE"):
                  model = create_AE(input_shape)
            else: 
                  model = create_AE_HLS4ML(input_shape)      
            earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                          mode="min", patience=20,
                                          restore_best_weights=True)
            #tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
            start_time = time.time()
            history = model.fit(train, train, epochs=500, batch_size=64, shuffle=True, verbose=0,
                                    validation_data=(val, val),
                                    callbacks=[earlystopping]
                                    )
            end_time = time.time()
            train_time = ((end_time - start_time) * 1000)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  # Adjust learning rate as needed
            model.compile(optimizer=optimizer, loss='mse')
            # Save the model to a directory
            #tf.saved_model.save(autoencoder, model_path + "/" + device + "_" + PEs + "_" + scheduler + "_model")
            print("Elapsed training time: {:.2f} ms".format((train_time)))
            if(PLOT_ENABLED):
                  # Access training loss values
                  train_loss = history.history['loss']

                  # Access validation loss values
                  val_loss = history.history['val_loss']

                  # Create a list of epochs
                  epochs = range(1, len(train_loss) + 1)

                  # Plot epoch versus training loss and validation loss
                  plt.plot(epochs, train_loss, 'b', label='Training Loss')
                  plt.plot(epochs, val_loss, 'r', label='Validation Loss')
                  # Find the epoch with the minimum validation loss
                  optimal_epoch = np.argmin(val_loss) + 1

                  # Add a marker for the optimal epoch
                  plt.scatter(optimal_epoch, val_loss[optimal_epoch-1], color='g')
                  offset = 5 # Adjust the offset as desired
                  plt.text(optimal_epoch, val_loss[optimal_epoch-1] + offset, f'Epoch: {optimal_epoch}\nLoss: {val_loss[optimal_epoch-1]:.4f}', ha='center', va='bottom', color='g')
                  #legend_label = f'Optimal Epoch\nEpoch: {optimal_epoch}, Loss: {val_loss[optimal_epoch-1]:.4f}'
                  #existing_legend_labels = [line.get_label() for line in plt.gca().get_lines()]
                  #all_legend_labels = existing_legend_labels + [legend_label]
                  plt.xlabel('Epoch')
                  plt.ylabel('Loss')
                  plt.legend()
                  plt.tight_layout()
                  plt.savefig('Epoch vs Training and Validation Loss.png', dpi=500, bbox_inches='tight')
                  #plt.show()  
                  print("Min. Val Loss = {:.2f}".format(val_loss[optimal_epoch-1]))
                  print("Min. Val Loss Epoch = {:.2f}".format(optimal_epoch-1))


if(MODEL_TYPE =="AE" or MODEL_TYPE =="VAE" or MODEL_TYPE=="AE_HLS4ML"):
             
      if(EXPORT_MODEL):
            if(MODEL_TYPE=="AE_HLS4ML"): 
                  model.save_weights(MODEL_PATH + '/' + 'weights.h5')
            else:
                  model.save(MODEL_PATH)
      elif(IMPORT_MODEL):
            model = tf.keras.models.load_model(MODEL_PATH)
            model.summary()
            total_params = model.count_params()
            print("Total number of parameters:", total_params)
   
      print("Used model is :", MODEL_TYPE)      
      reconstructions_train = model(train)
      reconstructions_val = model(val)
      #for std_dev_k in range(1,4):
      train_loss = tf.keras.losses.mse(reconstructions_train,train)
      val_loss = tf.keras.losses.mse(reconstructions_val,val)
      selected_mean = np.mean(val_loss) #(np.mean(train_loss) + np.mean(val_loss))/2
      selected_std = np.std(val_loss) #(np.std(train_loss) + np.std(val_loss))/2
      threshold = (selected_mean + STD_DEV_K * selected_std) 
      # Save the threshold testue to a file
      #np.save(model_path + "/" + device + "_" + PEs + "_" + scheduler + "_threshold", threshold)
      print("updated threshold = {:.2f}".format(threshold))
      z_score = (threshold - selected_mean) / selected_std
      percentile = norm.cdf(z_score) * 100
      print("Percentile = {:.2f}".format(percentile))

      #prediction = detect_anomaly(model, test, reconstruction_threshold = threshold)


      print("There are many "+str(len(test_labels)) + " data points predicted")
      start_time = time.time()

      classified,test_outputs = detect_anomaly(model, test, reconstruction_threshold = threshold)
      end_time = time.time()
elif(MODEL_TYPE == "OCSVM"):
      # Train the one-class SVM model
      if(HYPERTUNE):

            best_score = -float('inf')  # or -float('inf') if you're maximizing a score

            # Define ranges for nu and gamma
            nu_values = [0.1, 0.2, 0.3,0.4,0.5,0.6,0.7]
            gamma_values = ['scale', 'auto', 0.1, 0.01]

            for nu in nu_values:
                  print('current nu is: ', nu)
                  for gamma in gamma_values:
                        print('current gamma is: ', gamma)
                        # Train the model on the train set
                        svm_model = OneClassSVM(nu=nu, kernel='rbf', gamma=gamma)
                        svm_model.fit(train)

                        # Evaluate on the validation set
                        val_predictions = svm_model.predict(val)
                        train_predictions = svm_model.predict(train)
                        val_predictions = np.where(val_predictions == -1, 1, 0)
                 
                        val_score =  accuracy_score(val_labels,val_predictions)  # Define your metric function

                        # Update the best parameters if current score is better
                        if val_score > best_score:  # or > if maximizing
                              best_score = val_score
                              best_nu = nu
                              best_gamma = gamma

            # Train final model with best parameters on the combined training and validation set
            print('best gamma is: ', best_gamma)
            print('best nu is: ', best_nu)               


#Evaluate final model performance
      best_nu = 0.1
      best_gamma = 'scale'      
      svm_model = OneClassSVM(nu=best_nu, kernel='rbf', gamma=best_gamma)
      #svm_model = OneClassSVM(kernel='linear')
      svm_model.fit(train)
      start_time = time.time()
      predictions = svm_model.predict(test)
      end_time = time.time()
      classified = np.where(predictions == -1, 1, 0)

elif(MODEL_TYPE == "iForest"):
# Train the Isolation Forest model
      if(HYPERTUNE):
            best_score = -float('inf')
            cont_values = ['auto',0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
            for cont in cont_values:
            # Evaluate on the validation set8
                  print('current cont is : ', cont)
                  isoforest_model = IsolationForest(contamination=cont, random_state=42)
                  isoforest_model.fit(train)
                  val_predictions = isoforest_model.predict(val)
                  val_predictions = np.where(val_predictions == -1, 1, 0)
                  val_score = accuracy_score(val_labels,val_predictions)
 
                  # Update the best parameters if current score is better
                  if val_score > best_score:  # or > if maximizing
                        best_score = val_score
                        best_cont = cont

            # Train final model with best parameters on the combined training and validation set
      print('best cont is: ', best_cont)
      best_cont = 0.05
      isoforest_model = IsolationForest(contamination=best_cont, random_state=42)
      isoforest_model.fit(train)
      # Predict anomalies on the test data using Isolation Forest
      start_time = time.time()
      predictions = isoforest_model.predict(test)
      end_time = time.time()
      classified = np.where(predictions == -1, 1, 0)
      test_outputs = classified

if(MODEL_TYPE=="VAE"):

      test_time = (((end_time - start_time) + infernece_overhead )*1000)      
else:
      
      test_time = (((end_time - start_time) )* 1000)      

print("Elapsed test time: {:.4f} ms".format((test_time) ))

accuracy = accuracy_score(test_labels,classified)
recall = recall_score(test_labels,classified)
precision = precision_score(test_labels,classified)
f05_score = fbeta_score(test_labels, classified, beta=0.5)

f1_score = 2 * (precision * recall) / (precision + recall)
fpr,fnr = calculate_fpr_fnr(test_labels,classified)
f2_score = fbeta_score(test_labels, classified, beta=2)

print("testing accuracy = {:.2f}".format(accuracy*100))
print("testing recall = {:.2f}".format(recall*100))
print("testing precision = {:.2f}".format(precision*100))
print("testing F0.5 Score:{:.2f}".format(f05_score * 100))
print("testing F1 score: {:.2f}".format(f1_score*100))
print("F2 Score:{:.2f}".format(f2_score * 100))
print("test FPR: {:.2f}".format(fpr * 100))
print("test FNR: {:.2f}".format(fnr * 100))




results_file = RESULTS_CLASS+'_{}.csv'.format(MODEL_NAME)


#app_sched_dev = test_app_names + '_' + test_scheduler_names + '_' + test_device_names
#compute_metrics(np.array(test_labels),np.array(classified),np.array(app_sched_dev),results_file=results_file)


#print("This is for the model: ",MODEL_NAME)

#csv_file_path = f'./test_normal_anomaly_mixed_{MODEL_NAME}.csv'

# Read the CSV data into a DataFrame
#df = pd.read_csv(csv_file_path)

# Keep only the necessary columns
#df = df[['App', 'Label', 'ACC (%)']]

# Convert ACC (%) to a numeric type for averaging
#df['ACC (%)'] = pd.to_numeric(df['ACC (%)'], errors='coerce')

# Group by 'App' and 'Label' and calculate the mean of 'ACC (%)'
#average_acc = df.groupby(['App', 'Label'])['ACC (%)'].mean().reset_index()

# Print or save the result
#print(average_acc)

# Optionally, save to a new CSV file
#average_acc.to_csv(f'average_accuracy_per_app_{MODEL_NAME}.csv', index=False)
#compute_metrics(np.array(test_labels),np.array(classified),np.array(app_sched_dev),results_file=results_file
#      ,plot_metric = 'FPR (%)',
#plot_label = 'Normal')

# All thresholds in one figure:
exit()

if(PLOT_ENABLED and MODEL_TYPE=="AE" ):

  plt.scatter(range(len(test_outputs)), test_outputs, c=['r' if label == 1 else 'b' for label in test_labels], s=10)

  # Plot the threshold lines with Z-score values 1, 2, and 3
  y_values = [3.26, 5.02, 6.68]  # Corresponding y-values for the threshold lines
  colors = ['orange', 'g', 'purple']  # Different colors for each line
  labels = [f'$\delta$:84.13th percentile', f'$\delta$:97.72th percentile', f'$\delta$:99.87th percentile']  # Labels for each line
  y_offset = 0.35
  # Get current x-axis limits
  current_xlim = plt.xlim()

  # Set new x-axis limits
  # Increase the upper limit by a desired amount, e.g., 10% more
  new_x_max = current_xlim[1] + (current_xlim[1] - current_xlim[0]) * 0.1
  plt.xlim(current_xlim[0], new_x_max)

  for y, color, z_score, label in zip(y_values, colors, [1, 2, 3], labels):
      plt.axhline(y=y, color=color, linestyle='--',linewidth=2, label=f'Z-score = {z_score}')
      plt.text( plt.xlim()[1], y=y+y_offset, s=label, verticalalignment='center', horizontalalignment='right', fontsize=12, color=color)


  # Set plot labels and title
  plt.xlabel('Data Point Index', fontsize=20)
  plt.ylabel('MSE Reconstruction Loss', fontsize=20)

  # Increase font size of x and y axes
  plt.tick_params(axis='x', labelsize=20)
  plt.tick_params(axis='y', labelsize=20)

  # Adjust x-axis to show values every 100 points
  plt.xticks(range(0, len(test_outputs), 500))
  # Adjust y-axis to show values every 5 points
  min_y, max_y = min(test_outputs), max(test_outputs)  # Replace with your actual min and max if needed
  plt.yticks(range(int(min_y), int(max_y), 5))
  # Add a legend
  above_threshold = plt.Line2D([], [], color='r', marker='o', linestyle='None', markersize=3, label='Actual Anomaly')
  below_threshold = plt.Line2D([], [], color='b', marker='o', linestyle='None', markersize=3, label='Actual Normal')
  legend_handles = [above_threshold, below_threshold] + [plt.Line2D([], [], color=c, linestyle='--', label=f'Z-score = {z}') for c, z in zip(colors, [1, 2, 3])]
  legend = plt.legend(handles=legend_handles, fontsize=12, loc='upper right')
  legend.get_frame().set_alpha(1.0)  # Set legend box opacity to 1.0

  #plt.tight_layout()
  plt.savefig('Model Prediction Visual.png', dpi=500)
  # Show the plot
  #plt.show()





# For a single threshold:


if(PLOT_ENABLED):
    # Assuming 'test_outputs' contains the predicted values and 'test_labels' contains the actual labels
    # Plot the points with colors based on actual labels and above/below threshold
    plt.scatter(range(len(test_outputs)), test_outputs, c=['r' if label == 1 else 'b' for label in test_labels], s=10)
    # Plot the threshold line (3.26,5.02,6.68) 
    plt.axhline(y=threshold, color='g', linestyle='--')

    # Set plot labels and title
    plt.xlabel('Data Point Index', fontsize=20)
    plt.ylabel('MSE Reconstruction Loss', fontsize=20)
    # Increase font size of x and y axes
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    # Add a legend
    above_threshold = plt.Line2D([], [], color='r', marker='o', linestyle='None', markersize=5, label='Actual Anomaly')
    below_threshold = plt.Line2D([], [], color='b', marker='o', linestyle='None', markersize=5, label='Actual Normal')
    #threshold_line = plt.Line2D([], [], color='g', linestyle='--', label=f'Threshold: {round(percentile,2)}th percentile')
    threshold_line = plt.Line2D([], [], color='g', linestyle='--', label=f'Z-score = {STD_DEV_K}')

    legend = plt.legend(handles=[above_threshold, below_threshold, threshold_line], fontsize=12, loc='upper right')
    legend.get_frame().set_alpha(1.0)  # Set legend box opacity to 1.0

    plt.tight_layout()
    plt.savefig('Model Prediction Visual.png', dpi=500, bbox_inches='tight')
    # Show the plot
    plt.show()




    app_sched_dev = test_app_names + '_' + test_scheduler_names + '_' + test_device_names
    results_file = RESULTS_CLASS+'_{}_{}.csv'.format(STD_DEV_K, percentile)
    compute_metrics(np.array(test_labels),np.array(classified),np.array(app_sched_dev),results_file=results_file)
    compute_metrics(np.array(test_labels),np.array(classified),np.array(app_sched_dev),results_file=results_file
            ,plot_metric = 'FPR (%)',
    plot_label = 'Normal')

