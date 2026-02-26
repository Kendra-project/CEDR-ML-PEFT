import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque

# ==============================================================================
# SECTION 1: DATA PREPROCESSING LOGIC
# ==============================================================================

papi_counters = [
    'PAPI_TOT_CYC',
    'PAPI_TOT_INS',
    'PAPI_L1_DCM',
    'PAPI_L2_DCM',
    'PAPI_L3_TCM',
    'PAPI_BR_MSP'
]

PIPE_PATH = "/tmp/cedr_features_pipe"

def load_and_preprocess_data(jsonl_filepath, plot=True):
    print(f"Loading data from {jsonl_filepath}...")
    try:
        df = pd.read_json(jsonl_filepath, lines=True)
    except ValueError: 
        # Fallback if it's not JSON lines (e.g. standard JSON)
        df = pd.read_json(jsonl_filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {jsonl_filepath}.")
        return None, None, None

    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').set_index('timestamp')
    
    df = df.ffill().fillna(0)

    # --- FIX: AUTO-DETECT CUMULATIVE VS DELTA ---
    # We check PAPI_TOT_CYC. If it decreases frequently, it's NOT cumulative.
    # We use a threshold: if > 10% of steps are negative diffs, it's fluctuating (Delta).
    negative_steps = (df['PAPI_TOT_CYC'].diff() < 0).sum()
    is_cumulative = negative_steps < (len(df) * 0.1)

    delta_df = pd.DataFrame(index=df.index)
    
    if is_cumulative:
        print("[INFO] Detected CUMULATIVE counters. Applying .diff() conversion.")
        for col in papi_counters:
            if col in df.columns:
                delta_df[f'delta_{col}'] = df[col].diff().fillna(0)
    else:
        print("[INFO] Detected DELTA counters (fluctuating). Using values as-is.")
        for col in papi_counters:
            if col in df.columns:
                delta_df[f'delta_{col}'] = df[col]

    # Ensure no NaNs or Infinities
    delta_df = delta_df.replace([np.inf, -np.inf], 0).fillna(0)

    # --- FIX: SCALER WARNINGS ---
    # Fit scaler on numpy array (values) instead of DataFrame to avoid feature name mismatch during inference
    scaler = StandardScaler()
    raw_values = delta_df.values
    scaled_data = scaler.fit_transform(raw_values)
    scaled_data = scaled_data.astype(np.float32)

    final_feature_names = delta_df.columns.tolist()
    print("Data scaling complete.")
    return scaled_data, final_feature_names, scaler


def plot_prediction_vs_actual(history_actuals, history_preds, feature_names):
    """
    Plot comparison between actual and predicted values for each feature.
    """
    y_true = np.array(history_actuals)
    y_pred = np.array(history_preds)

    plt.figure(figsize=(14, 8))
    num_features = len(feature_names)

    for i in range(num_features):
        plt.subplot(num_features, 1, i + 1)
        plt.plot(y_true[:, i], label='Actual', color='blue')
        plt.plot(y_pred[:, i], label='Predicted', color='orange', linestyle='--')
        plt.title(f'Feature: {feature_names[i]}')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()

    plt.tight_layout()
    plt.show()

def plot_results(y_true, y_pred, feature_names):
    """
    Plots the Actual vs Predicted values in real units.
    """
    print("\n[INFO] Generating plots...")
    print(f"[DEBUG] y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
    try:
        # We pick the 3 most important features to plot to avoid clutter
        # Usually: Cycles, Instructions, and L1 Cache Misses
        plot_features = ['delta_PAPI_TOT_CYC', 'delta_PAPI_TOT_INS', 'delta_PAPI_L1_DCM']
        
        # Create a figure
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 12), sharex=True)
        
        # Determine x-axis range (limit to first 200 points for clarity)
        limit = min(200, len(y_true))
        x = range(limit)

        for i, target_feat in enumerate(plot_features):
            # Find index of this feature in the list
            if target_feat not in feature_names: continue
            idx = feature_names.index(target_feat)
            
            # Plot Actual
            axes[i].plot(x, y_true[:limit, idx], label='Actual (Real)', color='blue', linewidth=2)
            # Plot Predicted
            axes[i].plot(x, y_pred[:limit, idx], label='Predicted (AI)', color='orange', linestyle='--', linewidth=2)
            
            axes[i].set_title(f"{target_feat} (Real Value)", fontsize=14)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.xlabel("Time Step (100ms intervals)")
        plt.tight_layout()
        plt.savefig('real_vs_pred.png')
        print("[INFO] Plot saved to 'real_vs_pred.png'")
        plt.show() # Shows interactive plot if supported
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")

def real_time_inference_loop(model, device, scaler, sequence_length=20, input_file=None):
    source_name = "FIFO Pipe"
    if input_file:
        source_name = f"File ({input_file})"
        if not os.path.exists(input_file):
            print(f"[ERROR] Input file not found: {input_file}")
            return

    if not input_file and not os.path.exists(PIPE_PATH):
        try:
            os.mkfifo(PIPE_PATH)
        except FileExistsError:
            pass

    print(f"\n[INFO] Starting Inference Stream from: {source_name}")
    print(f"[INFO] Buffer filling... (need {sequence_length} samples before first prediction)")
    
    input_buffer = deque(maxlen=sequence_length)
    history_actuals = [] # Stores unscaled actuals
    history_preds = []   # Stores unscaled predictions
    
    last_pred_unscaled = None # We will store the REAL value here
    step_count = 0 

    model.eval()
    
    try:
        file_handle = open(input_file, "r") if input_file else open(PIPE_PATH, "r")
        
        with file_handle as source:
            while True:
                line = source.readline()
                if not line: 
                    if input_file: break 
                    continue

                try:
                    # 1. Parse CSV (These are REAL values)
                    values = [float(x) for x in line.strip().split(',')]
                    if len(values) != 6: continue
                    
                    # 2. Scale for the Model
                    values_array = np.array(values).reshape(1, -1)
                    scaled_values = scaler.transform(values_array).flatten()

                    # 3. Compare PREVIOUS prediction vs CURRENT actual
                    if last_pred_unscaled is not None:
                        # Store for final plotting
                        history_actuals.append(values)
                        history_preds.append(last_pred_unscaled)
                        
                        step_count += 1
                        
                        # Print in Real Units (Integers)
                        # We print specifically Cycles (Index 0) and Instructions (Index 1)
                        if step_count % 10 == 0: # Print every 10 steps for visibility
                            real_cyc = int(values[0])
                            pred_cyc = int(last_pred_unscaled[0])
                            diff = real_cyc - pred_cyc
                            print(f"Step {step_count}: Real CYC={real_cyc:<10} | Pred CYC={pred_cyc:<10} | Diff={diff}")

                    # 4. Update Buffer
                    input_buffer.append(scaled_values)
                    
                    # 5. Make NEW Prediction
                    if len(input_buffer) == sequence_length:
                        input_tensor = torch.tensor(np.array(input_buffer), dtype=torch.float32).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            prediction_scaled, _ = model(input_tensor)
                        

                        # --- CRITICAL FIX: INVERSE TRANSFORM IMMEDIATELY ---
                        pred_scaled_numpy = prediction_scaled.cpu().numpy().reshape(1, -1)
                        last_pred_unscaled = scaler.inverse_transform(pred_scaled_numpy).flatten()
                        print(f"[DEBUG] Made prediction at step {step_count+1}")
                        print(f"[DEBUG] Scaled Prediction: {pred_scaled_numpy}")
                        print(f"[DEBUG] Unscaled Prediction: {last_pred_unscaled}")
                except ValueError:
                    continue
                    
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        print("[INFO] Running final evaluation on captured stream...")
        if len(history_actuals) > 0:
            y_true = np.array(history_actuals)
            y_pred = np.array(history_preds)
            
            # Use the global feature names (ensure they are passed or accessible)
            # If feature_names isn't global, we default to the list
            feats = ['PAPI_TOT_CYC','PAPI_TOT_INS','PAPI_L1_DCM','PAPI_L2_DCM','PAPI_L3_TCM','PAPI_BR_MSP']
            
            # Print Metrics
            run_realtime_evaluation_report(y_true, y_pred, feats)
            
            # Generate Visual Plot
            # plot_results(y_true, y_pred, feats)
            plot_prediction_vs_actual(history_actuals, history_preds, feats)
        else:
            print("[WARN] No complete predictions made.")

def run_realtime_evaluation_report(y_true, y_pred, feature_names):
    print("\n--- Stream Performance Report ---")
    feature_maes = []
    feature_r2s = []
    for i in range(len(feature_names)):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        feature_maes.append((feature_names[i], mae))
        feature_r2s.append((feature_names[i], r2))

    print("\n--- MAE (Mean Absolute Error) ---")
    feature_maes.sort(key=lambda x: x[1], reverse=True)
    for feature_name, mae in feature_maes:
        print(f"  - {feature_name:<30}: {mae:.4f}")

    print("\n--- R2 (R-squared) ---")
    feature_r2s.sort(key=lambda x: x[1], reverse=True)
    for feature_name, r2 in feature_r2s:
        print(f"  - {feature_name:<30}: {r2:.4f}")

# ==============================================================================
# SECTION 2: MODEL DEFINITION (Unchanged)
# ==============================================================================
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        self.v.data.normal_(mean=0.0, std=1.0 / torch.sqrt(torch.tensor(hidden_size)))
    def forward(self, hidden_states):
        attn_energies = torch.tanh(self.attn(hidden_states))
        attn_energies = attn_energies.matmul(self.v)
        attn_weights = torch.softmax(attn_energies, dim=1)
        context_vector = torch.bmm(hidden_states.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)
        return context_vector, attn_weights

class AttentionBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
        super(AttentionBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True, dropout=dropout_prob)
        self.attention = Attention(self.hidden_size)
        self.fc = nn.Linear(self.hidden_size * 2, self.output_size)
        self.dropout_layer = nn.Dropout(dropout_prob)
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        context_vector, attn_weights = self.attention(lstm_out)
        context_vector = self.dropout_layer(context_vector)
        out = self.fc(context_vector)
        return out, attn_weights

# ==============================================================================
# SECTION 3: TRAINING HELPERS (Unchanged)
# ==============================================================================
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        seq = data[i:(i + sequence_length)]
        target = data[i + sequence_length]
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    print("\n--- Starting Model Training ---")
    best_val_loss = float('inf'); best_model_path = 'best_model.pth'
    for epoch in range(num_epochs):
        model.train(); train_loss_total = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred, _ = model(X_batch)
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss_total += loss.item()
        
        avg_train_loss = train_loss_total / len(train_loader)
        
        model.eval(); val_loss_total = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred, _ = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss_total += loss.item()
        
        avg_val_loss = val_loss_total / len(val_loader)
        print(f"Epoch [{epoch+1:03d}/{num_epochs:03d}] | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss; torch.save(model.state_dict(), best_model_path)
    
    model.load_state_dict(torch.load(best_model_path))
    return model

def evaluate_model(model, val_loader, scaler, feature_names, device):
    model.eval()
    all_y_true, all_y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_pred, _ = model(X_batch)
            all_y_true.append(y_batch.cpu().numpy())
            all_y_pred.append(y_pred.cpu().numpy())
    y_true_scaled = np.concatenate(all_y_true)
    y_pred_scaled = np.concatenate(all_y_pred)
    y_true_unscaled = scaler.inverse_transform(y_true_scaled)
    y_pred_unscaled = scaler.inverse_transform(y_pred_scaled)
    run_realtime_evaluation_report(y_true_unscaled, y_pred_unscaled, feature_names)

# ==============================================================================
# SECTION 4: MAIN (Updated)
# ==============================================================================
if __name__ == "__main__":
    JSONL_FILE_PATH = 'synthetic_output.json' # User defined input
    SEQUENCE_LENGTH = 10
    TRAIN_SPLIT_RATIO = 0.8
    BATCH_SIZE = 64
    NUM_EPOCHS = 20 
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    LEARNING_RATE = 0.001
    DROPOUT_PROB = 0.2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    scaled_data, feature_names, scaler = load_and_preprocess_data(JSONL_FILE_PATH, plot=False)

    if scaled_data is not None:
        N_FEATURES = len(feature_names)
        X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=(1.0 - TRAIN_SPLIT_RATIO), shuffle=False)

        train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

        model = AttentionBiLSTM(N_FEATURES, HIDDEN_SIZE, NUM_LAYERS, N_FEATURES, DROPOUT_PROB).to(device)
        model = train_model(model, train_loader, val_loader, nn.MSELoss(), optim.Adam(model.parameters(), lr=LEARNING_RATE), NUM_EPOCHS, device)
        
        print("\n" + "="*50)
        print(" TRAINING COMPLETE. Select Inference Mode:")
        print(" 1. Real-Time Pipe (Connect to CEDR via FIFO)")
        print(" 2. File Replay (Test on existing CSV/Log)")
        print("="*50)
        
        # mode = input("Enter 1 or 2: ").strip()
        test_file = "input_data.csv"  # Predefined test file for mode 2
        real_time_inference_loop(model, device, scaler, SEQUENCE_LENGTH, input_file=test_file)
        # if mode == '1':
        #     real_time_inference_loop(model, device, scaler, SEQUENCE_LENGTH, input_file=None)
        # elif mode == '2':
        #     # test_file = input("Enter path to test CSV file (e.g., 'input_data.csv'): ").strip()
        #     real_time_inference_loop(model, device, scaler, SEQUENCE_LENGTH, input_file=test_file)