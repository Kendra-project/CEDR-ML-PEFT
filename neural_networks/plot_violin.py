import os
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score
from feature_analysis import preprocess_global, split_train_val_test
import matplotlib.patches as mpatches
import math

# ------------------------------- Config --------------------------------
OUTPUT_DIR = "./scenario_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.FATAL)
warnings.filterwarnings("ignore")

LOG_FILE = os.path.join(OUTPUT_DIR, "ae_app_violin.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, mode='a'), logging.StreamHandler()]
)

MODEL_PATH   = "./AE_demo/AE_model_binary"
STD_DEV_K    = 2
DATASET_PATH = "./AE_demo/datasets"
DEVICES      = {"OptiPlex", "Jetson", "ZCU102"}
SCHEDULERS   = {"SIMPLE", "MET", "EFT"}

# --------------------------- Plot Appearance ---------------------------
FONT_SIZE     = 14
plt.rcParams.update({'font.size': FONT_SIZE})

NORMAL_COLOR   = '#1f77b4'   # blue
ANOMALY_COLOR  = '#d62728'   # red
MEDIAN_COLOR   = '#000000'   # black
STAR_COLOR     = 'tab:orange'
STAR_X_OFFSET  = -0.35   # <<<<<<<<<<<<<<<<<<<<<<<< requested offset!
WIDTH          = 0.3
CATEGORY_SPACING = 1.2   # horizontal gap (not critical here)
X_TICK_PAD      = 8      # moves x‐tick labels away from the axis

logging.info("===== AE Violin + Accuracy Plot Run =====")
logging.info(f"Model: {MODEL_PATH}, Z={STD_DEV_K}, Data: {DATASET_PATH}")

# ------------------------------ Data Prep -------------------------------
def load_and_drop(split_name):
    df = preprocess_global(split_name, DEVICES, SCHEDULERS, DATASET_PATH)
    if 0 in df.index:
        df = df.drop(index=0).reset_index(drop=True)
    return df

dfs = {
    "train": load_and_drop("train"),
    "val":   load_and_drop("val"),
    "test":  pd.concat([
        load_and_drop("test_mixed"),
        load_and_drop("test_anomaly"),
        load_and_drop("test_normal")
    ], axis=0)
}
dfs["test"] = dfs["test"].sample(frac=1, random_state=42).reset_index(drop=True)

data            = split_train_val_test(dfs)
train_df, val_df, test_df = data['train_df'], data['val_df'], data['test_df']
test_labels     = np.array(data['test_labels'])
test_app_names  = np.array(data['test_app_names'])

logging.info(f"Loaded windows: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

# apply log2 transform
for df in (train_df, val_df, test_df):
    df[:] = np.log2(np.abs(df) + 1)

t_val  = tf.convert_to_tensor(val_df.values,   dtype=tf.float32)
t_test = tf.convert_to_tensor(test_df.values,  dtype=tf.float32)

# ----------------------------- Model Load -------------------------------
model = tf.keras.models.load_model(MODEL_PATH)
logging.info(f"Loaded AE with {model.count_params()} parameters")

# -------------------------- Inference Phase -----------------------------
def sample_mse(orig, recon):
    o = orig.numpy() if hasattr(orig, 'numpy') else orig
    r = recon.numpy() if hasattr(recon, 'numpy') else recon
    return np.mean((o - r) ** 2, axis=1)

recon_val  = model(t_val,   training=False)
recon_test = model(t_test,  training=False)

mse_val    = sample_mse(t_val,   recon_val)
mse_test   = sample_mse(t_test,  recon_test)

mean_val   = mse_val.mean()
std_val    = mse_val.std()
threshold  = mean_val + STD_DEV_K * std_val
logging.info(f"Threshold MSE = {threshold:.6f} (Z={STD_DEV_K})")

preds = (mse_test > threshold).astype(int)

# ------------------------ Application Mapping ---------------------------
def map_app(prefix):
    if prefix.startswith('radar_correlator-noapi'):    return 'SaRC'
    if prefix.startswith('radar_correlator-zip'):      return 'RCZ'
    if prefix.startswith('radar_correlator_v1'):       return 'RCM'
    if prefix.startswith('radar_correlator'):          return 'RC'
    if prefix.startswith('udma_read'):                 return 'BuR'
    if prefix.startswith('udma_write'):                return 'BuW'
    if prefix.startswith('SAR_nb'):                    return 'SAR'
    if prefix.startswith('SAR-noapi'):                 return 'SaSAR'
    if prefix.startswith('track_nb'):                  return 'LD'
    if prefix.startswith('wifi'):                      return 'WiFi-TX'
    if prefix.startswith('availtime_zip'):             return 'AvMa'
    if prefix.startswith('DoS_'):                      return 'ReEx'
    if prefix.startswith('spectre_v1'):                return 'Spectre-v1'
    if prefix.startswith('temporal_mitigation-noapi'): return 'SaTM'
    if prefix.startswith('temporal_mitigation'):       return 'TM'
    if prefix.startswith('fft_pe_analysis'):           return 'TTA'
    if prefix.startswith('pulse_doppler-noapi'):       return 'SaPD'
    if prefix.startswith('pulse_doppler'):             return 'PD'
    return prefix

apps_prefix  = pd.Series(test_app_names).str.rstrip('-')
mapped_apps  = apps_prefix.map(map_app)
codes        = sorted(mapped_apps.unique())

accs = [
    accuracy_score(test_labels[mapped_apps == c],
                   preds      [mapped_apps == c]) * 100
    for c in codes
]

# ------------------- Violin + Accuracy Plot ------------------------
positions     = np.arange(len(codes))
threshold_log = np.log2(threshold + 1)
VIOLIN_HALF_WIDTH = WIDTH / 2

def draw_violin(ax, data, x, facecolor):
    parts = ax.violinplot([data],
                          positions=[x],
                          widths=WIDTH,
                          showmeans=False,
                          showmedians=True,
                          showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(facecolor)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    if 'cmedians' in parts:
        parts['cmedians'].set_color(MEDIAN_COLOR)
        parts['cmedians'].set_linewidth(2)

fig, ax1 = plt.subplots(figsize=(len(codes)*0.8, 5.2))
ax2       = ax1.twinx()
ax2.set_ylim(0, 110)
ax2.set_ylabel('Accuracy (%)', fontsize=FONT_SIZE)
# ---- Add splitting lines here ----
for i in range(len(codes) + 1):
    xpos = i - 0.5
    ax1.axvline(
        xpos, color='gray', linestyle='--', alpha=0.6, linewidth=1, zorder=0
    )

for i, c in enumerate(codes):
    norm_vals = np.log2(mse_test[(mapped_apps == c) & (test_labels == 0)] + 1)
    ano_vals  = np.log2(mse_test[(mapped_apps == c) & (test_labels == 1)] + 1)

    # --- FIXED ALIGNMENT LOGIC ---
    if len(norm_vals) and len(ano_vals):
        draw_violin(ax1, norm_vals, i - VIOLIN_HALF_WIDTH, NORMAL_COLOR)
        draw_violin(ax1, ano_vals,  i + VIOLIN_HALF_WIDTH, ANOMALY_COLOR)
    elif len(norm_vals):
        draw_violin(ax1, norm_vals, i, NORMAL_COLOR)
    elif len(ano_vals):
        draw_violin(ax1, ano_vals, i, ANOMALY_COLOR)

    acc = accs[i]
    ax2.scatter(i + STAR_X_OFFSET, acc, marker='*', s=100, color=STAR_COLOR, zorder=3)
    ax2.text(i + STAR_X_OFFSET, acc + 3.5, f"{round(acc)}",
             ha='center', va='bottom', fontsize=FONT_SIZE, color=STAR_COLOR)

ax1.set_xticks(positions)
ax1.set_xticklabels(codes, rotation=45, ha='right')
ax1.tick_params(axis='x', pad=X_TICK_PAD)
ax1.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
ax1.set_ylabel('log₂(MSE + 1)', fontsize=FONT_SIZE)
ax1.set_xlabel('Application', fontsize=FONT_SIZE, labelpad=15)
ax1.axhline(y=threshold_log, color='green', linestyle='--')

# -------------------- Legend (with Median) -----------------------------
legend_handles = [
    mpatches.Patch(color=NORMAL_COLOR,  label='Actual Normal'),
    mpatches.Patch(color=ANOMALY_COLOR, label='Actual Anomaly'),
    plt.Line2D([], [], color=MEDIAN_COLOR, linestyle='-', linewidth=2, label='MSE Median'),
    plt.Line2D([], [], marker='*', color=STAR_COLOR, linestyle='None', markersize=10, label='Accuracy'),
    plt.Line2D([], [], color='green', linestyle='--', label='Threshold'),
]
ax1.legend(handles=legend_handles,
           loc='upper center',
           bbox_to_anchor=(0.5, 1.2),
           ncol=3,
           frameon=False,
           borderaxespad=0,
           fontsize=FONT_SIZE)

fig.subplots_adjust(top=0.88, bottom=0.28)
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Save
out_path = os.path.join(OUTPUT_DIR, 'mse_and_acc_violin_per_app.png')
out_path_pdf = os.path.join(OUTPUT_DIR, 'mse_and_acc_violin_per_app.pdf')

plt.savefig(out_path, dpi=600, bbox_inches='tight')
plt.savefig(out_path_pdf,  bbox_inches='tight')
logging.info(f"Saved plot to {out_path} and {out_path_pdf}")

plt.close()
