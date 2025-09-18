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
from models  import AutoEncoder,create_AE,detect_anomaly
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
from sklearn.preprocessing import MinMaxScaler


# Define headers
headers = [
    "Model",
    #"Scheduler",
    "Platform",
    "Accuracy",
    "Recall",
    "Precision",
    "F0.5 Score",
    "F1 Score",
    "F2 Score",
    "FPR",
    "FNR",
]
  
MODEL_TYPE = "iForest" # AE, OCSVM, iForest,LOF
MODEL_NAME = "iForest"
STD_DEV_K = 3
RESULTS_CLASS = 'test_normal_anomaly_mixed'
MODEL_PATH = "./saved_models/threadfish_v1_" + MODEL_TYPE
EXPORT_MODEL = False
IMPORT_MODEL = False

train_global = preprocess_global(data_class="train",device = {"OptiPlex","ZCU102","Jetson"},scheduler={"SIMPLE","MET","EFT"})
val_global = preprocess_global(data_class="val",device = {"OptiPlex","ZCU102","Jetson"},scheduler={"SIMPLE","MET","EFT"})

# Writing to CSV
results_file = "sch_based_results.csv"
with open(results_file, mode="a", newline='') as file:
      writer = csv.writer(file)
      writer.writerow(headers)
for sch in ["SIMPLE", "MET", "EFT"]:
    
#for dev in ["OptiPlex", "Jetson", "ZCU102"]:
      #print ("scheduler-platform pair is: ",sch,"-",dev)
      test_global_normal = preprocess_global(data_class="test_normal", device = {"OptiPlex", "Jetson", "ZCU102"},scheduler={sch})
      test_global_anomaly = preprocess_global(data_class="test_anomaly",device = {"OptiPlex", "Jetson", "ZCU102"},scheduler={sch})
      test_global_mixed = preprocess_global(data_class="test_mixed",device = {"OptiPlex", "Jetson", "ZCU102"},scheduler={sch})
      test_global =  pd.concat([test_global_normal, test_global_anomaly, test_global_mixed], axis=0)
      # Shuffle the data points in-place
      #test_global = test_global.sample(frac=1, random_state=42).reset_index(drop=True)
      train_global = preprocess_global(data_class="train",device = {"OptiPlex", "Jetson", "ZCU102"},scheduler={"SIMPLE","MET","EFT"})
      #print(train_global)

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

      # apply log normalization to each feature
      for col in test_df.columns:
            test_df[col] = np.power(np.log2(np.abs(test_df[col]) + 1),1)  # add 1 to avoid taking the log of 0

                  # apply log normalization to each feature
      for col in train_df.columns:
            train_df[col] = np.power(np.log2(np.abs(train_df[col]) + 1),1)  # add 1 to avoid taking the log of 0

      # apply log normalization to each feature
      for col in val_df.columns:
            val_df[col] = np.power(np.log2(np.abs(val_df[col]) + 1),1)  # add 1 to avoid taking the log of 0

      # show the first few rows of the normalized data
      #print('First few rows of the preprocessed test data:')
      #print(test_df[:5])
      #print('First few rows of the preprocessed train data:')
      #print(test_df[:5])
      test_df = pd.DataFrame(test_df)
      train_df = pd.DataFrame(train_df)
      val_df = pd.DataFrame(val_df)

      # convert dataframes to arrays
      train = train_df.to_numpy()
      val = val_df.to_numpy()
      test = test_df.to_numpy()

      # train data only contains normal data
      train_labels = [0]*len(train)
      # data preparing is done, now let's train 
      input_shape = train.shape[1:]



      #print("shape of train is = "+str(train.shape[0])+"x"+str(train.shape[1]))
      #print("shape of val is = "+str(val.shape[0])+"x"+str(val.shape[1]))

      train = tf.convert_to_tensor(train, dtype=tf.float32)
      val = tf.convert_to_tensor(val, dtype=tf.float32)
      test = tf.convert_to_tensor(test, dtype=tf.float32)

      print("shape of test is = "+str(test.shape[0])+"x"+str(test.shape[1]))



      #history = autoencoder.fit(train, train, epochs=256, batch_size=16, shuffle=True, verbose = 0)
      #optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  # Adjust learning rate as needed
      #autoencoder.compile(optimizer=optimizer, loss='mse')

      # Create an instance of the custom callback

      #autoencoder.summary()
      #total_params = autoencoder.count_params()


      #print("Total number of parameters:", total_params)
      if(MODEL_TYPE == "VAE"):
            scaler = MinMaxScaler()
            train = scaler.fit_transform(train)

            # Apply the same transformation to validation and test data
            val = scaler.transform(val)
            test = scaler.transform(test)           
      if(MODEL_TYPE == "AE" or MODEL_TYPE=="VAE"):
            model = tf.keras.models.load_model(MODEL_PATH)
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

            start_time = time.time()

            print("There are many "+str(len(test_labels)) + " data points predicted")
            classified,test_outputs = detect_anomaly(model, test, reconstruction_threshold = threshold)
      elif(MODEL_TYPE == "OCSVM"):
      # Train the one-class SVM model

            svm_model = OneClassSVM(nu=0.10, kernel='rbf', gamma='scale')
            svm_model.fit(train)
            predictions = svm_model.predict(test)
            classified = np.where(predictions == -1, 1, 0)

      elif(MODEL_TYPE == "iForest"):
      # Train the Isolation Forest model

            isoforest_model = IsolationForest(contamination=0.05, random_state=42)
            isoforest_model.fit(train)
            # Predict anomalies on the test data using Isolation Forest
            predictions = isoforest_model.predict(test)
            classified = np.where(predictions == -1, 1, 0)

      end_time = time.time()

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
      # Define the values
      values = [
      MODEL_NAME,
      sch,
      #dev,
      "{:.2f}".format(accuracy * 100),
      "{:.2f}".format(recall * 100),
      "{:.2f}".format(precision * 100),
      "{:.2f}".format(f05_score * 100),
      "{:.2f}".format(f1_score * 100),
      "{:.2f}".format(f2_score * 100),
      "{:.2f}".format(fpr * 100),
      "{:.2f}".format(fnr * 100),
      ]
      with open(results_file, mode="a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(values)
