import os
import warnings
import logging
import numpy as np
import pandas as pd
# Suppress TensorFlow INFO, WARNING, and ERROR messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
import tensorflow as tf
from datetime import datetime
from scipy.stats import norm
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
from feature_analysis import preprocess_global, split_train_val_test, calculate_fpr_fnr


pd.options.mode.chained_assignment = None
pd.options.display.max_columns = None
# ----------------------- Logging Setup -----------------------

tf.get_logger().setLevel(logging.FATAL)

# Configure log file
LOG_FILE = "ae_experiment_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler()
    ]
)

# ---------------------- Experiment Setup ---------------------
MODEL_PATH = "./AE_model_binary"
STD_DEV_K = 2
DATASET_PATH = "./datasets"
RUN_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Log config
logging.info("===== AutoEncoder Inference Run =====")
logging.info(f"Timestamp: {RUN_TIME}")
logging.info(f"Model path: {MODEL_PATH}")
logging.info(f"Z-score threshold multiplier (STD_DEV_K): {STD_DEV_K}")
logging.info(f"Dataset path: {DATASET_PATH}")

# --------------------- Data Preprocessing ---------------------
devices = {"OptiPlex", "Jetson", "ZCU102"}
test_global_normal = preprocess_global("test_normal", devices, {"SIMPLE", "MET", "EFT"}, DATASET_PATH)
test_global_anomaly = preprocess_global("test_anomaly", devices, {"SIMPLE", "MET", "EFT"}, DATASET_PATH)
test_global_mixed = preprocess_global("test_mixed", devices, {"SIMPLE", "MET", "EFT"}, DATASET_PATH)

test_global = pd.concat([test_global_normal, test_global_anomaly, test_global_mixed], axis=0)
test_global = test_global.sample(frac=1, random_state=42).reset_index(drop=True)

dfs = {
    "train": preprocess_global("train", devices, {"SIMPLE", "MET", "EFT"}, DATASET_PATH),
    "val": preprocess_global("val", devices, {"SIMPLE", "MET", "EFT"}, DATASET_PATH),
    "test": test_global
}
data = split_train_val_test(dfs)
train_df, val_df, test_df = data['train_df'], data['val_df'], data['test_df']
test_labels = data['test_labels']

# Normalize
for df in [train_df, val_df, test_df]:
    for col in df.columns:
        df[col] = np.power(np.log2(np.abs(df[col]) + 1), 1)

# Convert to Tensors
train = tf.convert_to_tensor(train_df.values, dtype=tf.float32)
val = tf.convert_to_tensor(val_df.values, dtype=tf.float32)
test = tf.convert_to_tensor(test_df.values, dtype=tf.float32)

# ------------------------ Model Loading ------------------------
model = tf.keras.models.load_model(MODEL_PATH)
model.summary()
logging.info(f"Total number of model parameters: {model.count_params()}")

# ----------------------- Inference Phase -----------------------
reconstructions_train = model(train)
reconstructions_val = model(val)
reconstructions_test = model(test)

# Reconstruction loss
train_loss = tf.keras.losses.mse(train, reconstructions_train)
val_loss = tf.keras.losses.mse(val, reconstructions_val)
test_loss = tf.keras.losses.mse(test, reconstructions_test)

# Z-score threshold
mean_val = np.mean(val_loss)
std_val = np.std(val_loss)
threshold = mean_val + STD_DEV_K * std_val
z_score = (threshold - mean_val) / std_val
percentile = norm.cdf(z_score) * 100

logging.info(f"Threshold (Z={STD_DEV_K}): {threshold:.4f}")
logging.info(f"Percentile = {percentile:.2f}%")

# ---------------------- Evaluation Phase -----------------------
classified = (test_loss > threshold).numpy().astype(int)

accuracy = accuracy_score(test_labels, classified)
recall = recall_score(test_labels, classified)
precision = precision_score(test_labels, classified)
f05_score = fbeta_score(test_labels, classified, beta=0.5)
f1_score = 2 * (precision * recall) / (precision + recall)
f2_score = fbeta_score(test_labels, classified, beta=2)
fpr, fnr = calculate_fpr_fnr(test_labels, classified)

# Log results
logging.info(f"Accuracy:     {accuracy * 100:.2f}%")
logging.info(f"Recall:       {recall * 100:.2f}%")
logging.info(f"Precision:    {precision * 100:.2f}%")
logging.info(f"F0.5 Score:   {f05_score * 100:.2f}%")
logging.info(f"F1 Score:     {f1_score * 100:.2f}%")
logging.info(f"F2 Score:     {f2_score * 100:.2f}%")
logging.info(f"FPR:          {fpr * 100:.2f}%")
logging.info(f"FNR:          {fnr * 100:.2f}%")
logging.info("====================================================\n")
