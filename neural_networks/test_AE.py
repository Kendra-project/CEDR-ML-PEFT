
import os
import warnings
import logging

# Suppress TensorFlow INFO, WARNING, and ERROR messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Disable all Python warnings
warnings.filterwarnings("ignore")

# Suppress pandas FutureWarnings (optional but safe)
import pandas as pd
pd.options.mode.chained_assignment = None  # Disable SettingWithCopyWarning
pd.options.display.max_columns = None

# Suppress TensorFlow C++ backend logs completely
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(logging.FATAL)

import numpy as np
import tensorflow as tf
from scipy.stats import norm
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
from feature_analysis import preprocess_global, split_train_val_test, calculate_fpr_fnr

# Settings
MODEL_PATH = "./saved_models/threadfish_v1_AE"
STD_DEV_K = 2
Z_SCORE = 2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Preprocess with all features
devices = {"OptiPlex", "Jetson", "ZCU102"}
test_global_normal = preprocess_global(data_class="test_normal", device=devices, scheduler={"SIMPLE", "MET", "EFT"})
test_global_anomaly = preprocess_global(data_class="test_anomaly", device=devices, scheduler={"SIMPLE", "MET", "EFT"})
test_global_mixed = preprocess_global(data_class="test_mixed", device=devices, scheduler={"SIMPLE", "MET", "EFT"})
test_global = pd.concat([test_global_normal, test_global_anomaly, test_global_mixed], axis=0)
test_global = test_global.sample(frac=1, random_state=42).reset_index(drop=True)

dfs = {"train": preprocess_global(data_class="train", device=devices, scheduler={"SIMPLE", "MET", "EFT"}), 
       "val": preprocess_global(data_class="val", device=devices, scheduler={"SIMPLE", "MET", "EFT"}), 
       "test": test_global}
data = split_train_val_test(dfs)

# Use full features
train_df = data['train_df']
val_df = data['val_df']
test_df = data['test_df']
test_labels = data['test_labels']

# Normalize
for df in [train_df, val_df, test_df]:
    for col in df.columns:
        df[col] = np.power(np.log2(np.abs(df[col]) + 1), 1)

# Convert to Tensor
train = tf.convert_to_tensor(train_df.values, dtype=tf.float32)
val = tf.convert_to_tensor(val_df.values, dtype=tf.float32)
test = tf.convert_to_tensor(test_df.values, dtype=tf.float32)

# Load Model
model = tf.keras.models.load_model(MODEL_PATH)
model.summary()
print("Total number of parameters:", model.count_params())

# Inference
reconstructions_train = model(train)
reconstructions_val = model(val)
reconstructions_test = model(test)

# Compute reconstruction errors
train_loss = tf.keras.losses.mse(train, reconstructions_train)
val_loss = tf.keras.losses.mse(val, reconstructions_val)
test_loss = tf.keras.losses.mse(test, reconstructions_test)

# Threshold using Z-score
mean_val = np.mean(val_loss)
std_val = np.std(val_loss)
threshold = mean_val + STD_DEV_K * std_val
print("Threshold (Z=2): {:.4f}".format(threshold))

# Predict
classified = (test_loss > threshold).numpy().astype(int)

# Metrics
accuracy = accuracy_score(test_labels, classified)
recall = recall_score(test_labels, classified)
precision = precision_score(test_labels, classified)
f05_score = fbeta_score(test_labels, classified, beta=0.5)
f1_score = 2 * (precision * recall) / (precision + recall)
f2_score = fbeta_score(test_labels, classified, beta=2)
fpr, fnr = calculate_fpr_fnr(test_labels, classified)

print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"F0.5 Score:{f05_score*100:.2f}%")
print(f"F1 Score:  {f1_score*100:.2f}%")
print(f"F2 Score:  {f2_score*100:.2f}%")
print(f"FPR:       {fpr*100:.2f}%")
print(f"FNR:       {fnr*100:.2f}%")
