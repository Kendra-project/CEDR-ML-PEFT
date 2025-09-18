import os
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Suppress TensorFlow INFO/WARNING/ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
import tensorflow as tf
from datetime import datetime
from scipy.stats import norm
from sklearn.metrics import accuracy_score
from feature_analysis import preprocess_global, split_train_val_test
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
import matplotlib.patches as mpatches


# ------------------------------- Config --------------------------------
OUTPUT_DIR   = "./scenario_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
tf.get_logger().setLevel(logging.FATAL)

LOG_FILE     = "ae_experiment_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, mode='a'), logging.StreamHandler()]
)

MODEL_PATH   = "./AE_demo/AE_model_binary"
STD_DEV_K    = 2
DATASET_PATH = "./AE_demo/datasets"
DEVICES      = {"OptiPlex","Jetson","ZCU102"}
SCHEDULERS   = {"SIMPLE","MET","EFT"}
RUN_TIME     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Log configuration
logging.info("===== AutoEncoder Inference Run =====")
logging.info(f"Timestamp: {RUN_TIME}")
logging.info(f"Model path: {MODEL_PATH}")
logging.info(f"Z-score multiplier (STD_DEV_K): {STD_DEV_K}")
logging.info(f"Dataset path: {DATASET_PATH}")

# ------------------------------ Data Prep -------------------------------
def load_and_drop(split_name):
    df = preprocess_global(split_name, DEVICES, SCHEDULERS, DATASET_PATH)
    if 0 in df.index:
        df = df.drop(index=0).reset_index(drop=True)
    return df

# Load and combine test splits
dfs = {
    "train": load_and_drop("train"),
    "val":   load_and_drop("val"),
    "test": pd.concat([
        load_and_drop("test_mixed"),
        load_and_drop("test_anomaly"),
        load_and_drop("test_normal")
    ], axis=0)
}
dfs["test"] = dfs["test"].sample(frac=1, random_state=42).reset_index(drop=True)

# Split into features + metadata
data = split_train_val_test(dfs)
train_df, val_df, test_df = data['train_df'], data['val_df'], data['test_df']
test_labels = np.array(data['test_labels'])
test_apps   = np.array(data['test_app_names'])

logging.info(f"Loaded windows: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

# Apply exact log2 normalization
for df in (train_df, val_df, test_df):
    df[:] = np.log2(np.abs(df) + 1)

# Convert to tensors
t_train = tf.convert_to_tensor(train_df.values, dtype=tf.float32)
t_val   = tf.convert_to_tensor(val_df.values,   dtype=tf.float32)
t_test  = tf.convert_to_tensor(test_df.values,  dtype=tf.float32)

# ----------------------------- Model Load -------------------------------
model = tf.keras.models.load_model(MODEL_PATH)
logging.info(f"Loaded model with {model.count_params()} parameters")

# -------------------------- Inference Phase -----------------------------
recon_train = model(t_train, training=False)
recon_val   = model(t_val,   training=False)
recon_test  = model(t_test,  training=False)

def sample_mse(orig, recon):
    o = orig.numpy() if hasattr(orig, 'numpy') else orig
    r = recon.numpy() if hasattr(recon, 'numpy') else recon
    return np.mean((o - r)**2, axis=1)

mse_val  = sample_mse(t_val,   recon_val)
mse_test = sample_mse(t_test,  recon_test)

# Compute threshold on raw MSE
mean_val  = mse_val.mean()
std_val   = mse_val.std()
threshold = mean_val + STD_DEV_K * std_val
threshold_log = np.log2(threshold + 1)
logging.info(f"Threshold MSE = {threshold:.6f} (Z={STD_DEV_K})")

# ------------------------- Evaluation Phase ----------------------------
preds = (mse_test > threshold).astype(int)
overall_acc = accuracy_score(test_labels, preds) * 100
logging.info(f"Overall test accuracy: {overall_acc:.2f}%")

# ------------------------- Application Mapping -------------------------
def map_app(prefix):
    if prefix.startswith('radar_correlator-noapi'): return 'SaRC'
    if prefix.startswith('radar_correlator-zip'):    return 'RCFZ'
    if prefix.startswith('radar_correlator_v1'):    return 'RCM'
    if prefix.startswith('radar_correlator'):       return 'RC'
    if prefix.startswith('udma_read'):              return 'BuR'
    if prefix.startswith('udma_write'):             return 'BuW'
    if prefix.startswith('SAR_nb'):                  return 'SAR'
    if prefix.startswith('SAR-noapi'):               return 'SaSAR'
    if prefix.startswith('track_nb'):                return 'LD'
    if prefix.startswith('wifi'):                    return 'WiFi-TX'
    if prefix.startswith('availtime_zip'):           return 'AvMa'
    if prefix.startswith('DoS_'):                    return 'ReEx'
    if prefix.startswith('spectre_v1'):              return 'Spectre-v1'
    if prefix.startswith('temporal_mitigation-noapi'): return 'SaTM'
    if prefix.startswith('temporal_mitigation'):       return 'TM'
    if prefix.startswith('fft_pe_analysis'):           return 'TTA'
    if prefix.startswith('pulse_doppler-noapi'):       return 'SaPD'
    if prefix.startswith('pulse_doppler'):             return 'PD'
    return prefix

apps_prefix = pd.Series(test_apps).str.rstrip('-')
mapped_apps = apps_prefix.map(map_app)

# Unique codes
codes = sorted(mapped_apps.unique())


# ---------------------- Plot: Average Accuracy per App ----------------------
accs = [accuracy_score(test_labels[mapped_apps==c], preds[mapped_apps==c]) * 100 for c in codes]




#---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(len(codes)*0.6, 4))
bars = ax.bar(codes, accs)
# annotate exact accuracy in red at bar center
for idx, val in enumerate(accs):
    ax.text(idx, val/2, f"{int(round(val))}%", ha='center', va='center', color='red')
ax.set_ylabel('Average Accuracy (%)')
ax.set_xlabel('Application')

ax.set_xticklabels(codes, rotation=45, ha='right')
plt.tight_layout()
acc_path = os.path.join(OUTPUT_DIR,'accuracy_per_app.png')
plt.savefig(acc_path, dpi=300)
plt.close()
logging.info(f"Saved accuracy per app plot: {acc_path}")

# ----------------------- Plot: Violin MSE per App -----------------------
fig, ax = plt.subplots(figsize=(len(codes)*0.6, 4))
positions = np.arange(len(codes))
mse_groups = [np.log2(mse_test[mapped_apps==c] + 1) for c in codes]
parts = ax.violinplot(mse_groups, positions=positions, widths=0.6,
                      showmeans=False, showmedians=True)
# style violins
for pc in parts['bodies']:
    pc.set_facecolor('#8888FF')
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)
# median line
parts['cmedians'].set_color('red')
# threshold
th_line = ax.axhline(y=threshold_log, color='green', linestyle='--', label='Threshold')

ax.set_xticks(positions)
ax.set_xticklabels(codes, rotation=45, ha='right')
ax.set_ylabel('log2(MSE+1)')
ax.set_xlabel('Application')

# legend
ax.legend([parts['cmedians'], th_line], ['Median','Threshold'],
          loc='upper center', bbox_to_anchor=(0.5,1.15), ncol=2, frameon=False)
# remove top border
ax.spines['top'].set_visible(False)
plt.tight_layout()
violin_path = os.path.join(OUTPUT_DIR,'mse_violin_per_app.png')
plt.savefig(violin_path, dpi=300)
plt.close()
logging.info(f"Saved violin MSE plot: {violin_path}")

logging.info("===== Run Complete =====")