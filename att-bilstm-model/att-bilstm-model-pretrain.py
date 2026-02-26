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
from matplotlib.ticker import MaxNLocator
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

# --- NEW: Real-Time Inference with Evaluation ---
def real_time_inference_loop(model, device, scaler, sequence_length=20):
    if not os.path.exists(PIPE_PATH):
        try:
            os.mkfifo(PIPE_PATH)
        except FileExistsError:
            pass

    print(f"\n[INFO] Listening for CEDR data on {PIPE_PATH}...")
    print("[INFO] Mode: BLOCKING. CEDR will pause if this script is too slow.")
    
    input_buffer = deque(maxlen=sequence_length)
    history_actuals = []
    history_preds = []
    last_prediction_scaled = None
    step_count = 0 

    model.eval()
    
    try:
        with open(PIPE_PATH, "r") as pipe:
            while True:
                line = pipe.readline()
                if not line: continue 
                
                try:
                    values = [float(x) for x in line.strip().split(',')]
                    if len(values) != 6: continue
                    
                    # Scaling
                    values_array = np.array(values).reshape(1, -1)
                    scaled_values = scaler.transform(values_array).flatten()

                    # Store & Validate
                    if last_prediction_scaled is not None:
                        pred_unscaled = scaler.inverse_transform(last_prediction_scaled.reshape(1, -1)).flatten()
                        history_actuals.append(values)
                        history_preds.append(pred_unscaled)
                        
                        step_count += 1
                        
                        # OPTIMIZATION: Only print every 100 steps to speed up processing
                        if step_count % 100 == 0:
                            print(f"Step {step_count}: Real CYC={values[0]:.0f} vs Pred={pred_unscaled[0]:.0f}")

                    input_buffer.append(scaled_values)
                    
                    if len(input_buffer) == sequence_length:
                        input_tensor = torch.tensor(np.array(input_buffer), dtype=torch.float32).unsqueeze(0).to(device)
                        with torch.no_grad():
                            prediction_scaled, _ = model(input_tensor)
                        last_prediction_scaled = prediction_scaled.cpu().numpy().flatten()
                        
                except ValueError:
                    continue

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user. Running final evaluation on captured stream...")
        
        if len(history_actuals) > 0:
            # Convert lists to numpy arrays
            y_true = np.array(history_actuals)
            y_pred = np.array(history_preds)
            
            # Re-use your existing plotting/metrics functions
            # feature_names comes from the global scope or main
            run_realtime_evaluation_report(y_true, y_pred, papi_counters)
        else:
            print("[WARN] No complete predictions made (buffer didn't fill?).")

def run_realtime_evaluation_report(y_true, y_pred, feature_names):
    """
    Evaluates the data collected during the real-time stream.
    """
    print("\n--- Real-Time Stream Performance Report ---")

    feature_maes = []
    feature_r2s = []
    
    # Calculate metrics per feature
    for i in range(len(feature_names)):
        feature_name = feature_names[i]
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        feature_maes.append((feature_name, mae))
        feature_r2s.append((feature_name, r2))

    print("\n--- MAE (Mean Absolute Error) ---")
    feature_maes.sort(key=lambda x: x[1], reverse=True)
    for feature_name, mae in feature_maes:
        print(f"  - {feature_name:<30}: {mae:.4f}")

    print("\n--- R2 (R-squared) ---")
    feature_r2s.sort(key=lambda x: x[1], reverse=True)
    for feature_name, r2 in feature_r2s:
        print(f"  - {feature_name:<30}: {r2:.4f}")

    # Plotting
    try:
        plot_results(y_true, y_pred, feature_names)
        plot_error_distribution(y_true, y_pred, feature_names)
        print("[INFO] Real-time plots generated.")
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")

# --- 1.2 Generate Dummy Data ---
def generate_dummy_data(num_rows=2000):
    print(f"Generating {num_rows} rows of 'smart' dummy data...")

    t = np.arange(num_rows)
    base_activity = (np.sin(t * 0.1) * 5e7 + np.sin(t * 0.05) * 2e7 + 8e7) 

    delta_ins = base_activity + np.random.normal(0, 1e6, num_rows)
    delta_ins[delta_ins < 0] = 0 

    delta_br_msp = (np.sin(t * 0.5) * 2e5 + 2e5) + np.random.normal(0, 1e4, num_rows)

    cpi_factor = 1.2
    branch_penalty = 15
    delta_cyc = (delta_ins * cpi_factor) + (delta_br_msp * branch_penalty) + np.random.normal(0, 1e6, num_rows)

    l1_miss_rate = 0.05
    l2_miss_rate = 0.3 
    l3_miss_rate = 0.5 

    delta_l1 = (delta_ins * l1_miss_rate) + np.random.normal(0, 1e4, num_rows)
    delta_l1[delta_l1 < 0] = 0

    delta_l2 = (delta_l1 * l2_miss_rate) + np.random.normal(0, 1e3, num_rows)
    delta_l2[delta_l2 < 0] = 0

    delta_l3 = (delta_l2 * l3_miss_rate) + np.random.normal(0, 1e2, num_rows)
    delta_l3[delta_l3 < 0] = 0

    df_data = {
        'timestamp': pd.date_range(start='1/1/2025', periods=num_rows, freq='100ms'),
        'PAPI_TOT_CYC': delta_cyc.cumsum().astype(np.int64),
        'PAPI_TOT_INS': delta_ins.cumsum().astype(np.int64),
        'PAPI_L1_DCM': delta_l1.cumsum().astype(np.int64),
        'PAPI_L2_DCM': delta_l2.cumsum().astype(np.int64),
        'PAPI_L3_TCM': delta_l3.cumsum().astype(np.int64),
        'PAPI_BR_MSP': delta_br_msp.cumsum().astype(np.int64)
    }
    return pd.DataFrame(df_data)

# --- 1.3 Visualization Functions ---

def plot_feature_lines(df):
    print(f"Generating line plots of 'delta' features...")
    try:
        fig, axes = plt.subplots(nrows=len(df.columns), ncols=1, figsize=(12, 12), sharex=True)
        for i, col in enumerate(df.columns):
            sns.lineplot(data=df, x=df.index, y=col, ax=axes[i], label=col)
            axes[i].legend(loc='upper right')
        axes[0].set_title('Line Plots of Key "Delta" Features', fontsize=16)
        axes[-1].set_xlabel('Timestamp')
        plt.tight_layout()
        plt.savefig('feature_line_plots.png')
        # plt.show() 
        print("Line plots saved.")
    except Exception as e:
        print(f"Warning: Could not generate line plots. Error: {e}")

def plot_correlation_heatmap(df):
    print(f"Generating correlation heatmap...")
    try:
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='RdYlGn', ax=ax)
        ax.set_title('Feature Correlation Heatmap', pad=20, fontsize=16)
        plt.tight_layout()
        plt.savefig('feature_heatmap.png')
        # plt.show()
        print("Heatmap saved.")
    except Exception as e:
        print(f"Warning: Could not generate heatmap plot. Error: {e}")

def plot_feature_distributions(df):
    print(f"Generating feature distribution histograms...")
    try:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
        axes = axes.flatten()
        for i, col in enumerate(df.columns):
            sns.histplot(df[col], kde=True, ax=axes[i], bins=30)
            axes[i].set_title(f'Distribution of {col}')
        fig.suptitle('Feature Distribution Histograms', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('feature_distributions.png')
        # plt.show()
        print("Histograms saved.")
    except Exception as e:
        print(f"Warning: Could not generate distribution plots. Error: {e}")

def plot_feature_scatter(df):
    print(f"Generating scatter plot (INS vs CYC)...")
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        sample_df = df.sample(n=min(500, len(df)))
        sns.regplot(data=sample_df, x='delta_PAPI_TOT_INS', y='delta_PAPI_TOT_CYC',
                    ax=ax, line_kws={"color": "red"})
        ax.set_title('Scatter Plot: Instructions vs. Cycles', fontsize=16)
        plt.tight_layout()
        plt.savefig('feature_scatter_plot.png')
        # plt.show()
        print("Scatter plot saved.")
    except Exception as e:
        print(f"Warning: Could not generate scatter plot. Error: {e}")

def load_and_preprocess_data(jsonl_filepath, plot=True):
    print(f"Loading data from {jsonl_filepath}...")
    try:
        df = pd.read_json(jsonl_filepath, lines=True)
    except FileNotFoundError:
        print(f"Error: File not found at {jsonl_filepath}.")
        return None, None, None

    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').set_index('timestamp')
    print(f"Original data shape: {df.shape}")

    df = df.ffill().fillna(0)

    # Convert all counters to deltas for training
    delta_df = pd.DataFrame(index=df.index)
    for col in papi_counters:
        if col in df.columns:
            delta_df[f'delta_{col}'] = df[col].diff().fillna(0)
        else:
            print(f"Warning: Counter column '{col}' not found.")

    print(f"Data shape after pre-processing (deltas): {delta_df.shape}")

    if plot:
        plot_feature_lines(delta_df)
        plot_correlation_heatmap(delta_df)
        plot_feature_distributions(delta_df)
        plot_feature_scatter(delta_df)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(delta_df)
    scaled_data = scaled_data.astype(np.float32)

    final_feature_names = delta_df.columns.tolist()
    print("Data scaling complete.")
    return scaled_data, final_feature_names, scaler

# ==============================================================================
# SECTION 2: MODEL DEFINITION
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

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_prob if num_layers > 1 else 0 
        )
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
# SECTION 3: TRAINING & EVALUATION LOGIC
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
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    print("\n--- Starting Model Training ---")
    start_time = time.time()
    best_val_loss = float('inf')
    best_model_path = 'best_model.pth'

    for epoch in range(num_epochs):
        model.train()
        train_loss_total = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred, _ = model(X_batch)
            loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
        avg_train_loss = train_loss_total / len(train_loader)

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred, _ = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss_total += loss.item()
        avg_val_loss = val_loss_total / len(val_loader)

        print(f"Epoch [{epoch+1:03d}/{num_epochs:03d}] | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)

    print(f"Training finished in {(time.time() - start_time):.2f}s. Loaded best model.")
    model.load_state_dict(torch.load(best_model_path))
    return model

def plot_results(y_true, y_pred, feature_names):
    print("\n--- Visualizing Results ---")
    try:
        # Use first 3 features or available features
        plot_count = min(3, len(feature_names))
        fig, axes = plt.subplots(nrows=plot_count, ncols=1, figsize=(15, 4*plot_count), sharex=True)
        if plot_count == 1: axes = [axes]
        
        plot_len = min(200, len(y_true))
        for i in range(plot_count):
            axes[i].plot(y_true[:plot_len, i], label='Actual', marker='o', markersize=4)
            axes[i].plot(y_pred[:plot_len, i], label='Predicted', linestyle='--', marker='x', markersize=4)
            axes[i].set_title(f'Prediction: {feature_names[i]}')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig('results_plot.png')
        print("Results plot saved.")
    except Exception as e:
        print(f"Warning: Plot failed: {e}")

def plot_error_distribution(y_true, y_pred, feature_names):
    print("\n--- Visualizing Error Distribution ---")
    try:
        errors = y_true - y_pred
        cols = min(3, len(feature_names))
        rows = (len(feature_names) + cols - 1) // cols
        
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 4*rows))
        axes = axes.flatten()

        for i in range(len(feature_names)):
            sns.histplot(errors[:, i], kde=True, ax=axes[i], bins=30)
            axes[i].set_title(f'Error Dist: {feature_names[i]}')

        plt.tight_layout()
        plt.savefig('error_distribution.png')
        print("Error plot saved.")
    except Exception as e:
        print(f"Warning: Plot failed: {e}")

def evaluate_model(model, val_loader, scaler, feature_names, device):
    print("\n--- Starting Final Evaluation ---")
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

    # Re-use the realtime report function for consistency
    run_realtime_evaluation_report(y_true_unscaled, y_pred_unscaled, feature_names)


# ==============================================================================
# SECTION 4: MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    
    # --- Check Command Line Arguments to switch modes? ---
    # For now, we will perform TRAINING first, then wait for USER to confirm Real-time
    
    JSONL_FILE_PATH = 'smart_papi_log.jsonl'
    DUMMY_DATA_ROWS = 2000
    SEQUENCE_LENGTH = 20
    TRAIN_SPLIT_RATIO = 0.8
    BATCH_SIZE = 64
    NUM_EPOCHS = 20 # Reduced for speed in demo
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    LEARNING_RATE = 0.001
    DROPOUT_PROB = 0.2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 0. Generate Data ---
    if not os.path.exists(JSONL_FILE_PATH):
        dummy_df = generate_dummy_data(DUMMY_DATA_ROWS)
        dummy_df.to_json(JSONL_FILE_PATH, orient='records', lines=True)

    # --- 1. Load Data ---
    scaled_data, feature_names, scaler = load_and_preprocess_data(JSONL_FILE_PATH, plot=False)

    if scaled_data is not None:
        N_FEATURES = len(feature_names)
        X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=(1.0 - TRAIN_SPLIT_RATIO), shuffle=False)

        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = AttentionBiLSTM(N_FEATURES, HIDDEN_SIZE, NUM_LAYERS, N_FEATURES, DROPOUT_PROB).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # --- 2. Train ---
        model = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device)
        
        # --- 3. Evaluate on Static Data ---
        print("\n[INFO] Initial evaluation on synthetic test set:")
        evaluate_model(model, val_loader, scaler, feature_names, device)

        # --- 4. Switch to Real-Time Mode ---
        print("\n" + "="*50)
        print(" TRAINING COMPLETE. ")
        print(" Ready to connect to CEDR via FIFO: " + PIPE_PATH)
        print(" Make sure CEDR is running and writing to this pipe.")
        print("="*50)
        
        # Uncomment this to auto-start, or user can toggle flag
        # user_input = input("Start Real-Time Inference now? (y/n): ")
        # if user_input.lower() == 'y':
        real_time_inference_loop(model, device, scaler, SEQUENCE_LENGTH)

    else:
        print("Data loading failed.")