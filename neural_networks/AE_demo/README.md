
# ✅ Quick Start: Run Inference with Pretrained AutoEncoder

```bash
python3 test_AE.py
```

This command:
- Loads the pretrained AutoEncoder from `./AE_model_binary`
- Applies the model on preprocessed test data from `./datasets`
- Computes anomaly detection metrics (accuracy, F1, FPR, etc.)
- Logs results to both console and `ae_experiment_log.txt`

---

## 📁 Project Structure

```
.
├── test_AE.py               # Main inference script
├── models.py                # AutoEncoder model definition
├── feature_analysis.py      # Preprocessing and utility functions
├── AE_model_binary/         # Trained AutoEncoder model (SavedModel format)
├── datasets/                # Folder containing CSV files for all dataset splits
├── requirements.txt         # Python dependencies
└── ae_experiment_log.txt    # Auto-generated experiment log
```

---

## 🧪 Datasets

Your dataset is expected to be stored in `./datasets/` with the following files:

- `train_*.csv`
- `val_*.csv`
- `test_normal_*.csv`
- `test_anomaly_*.csv`
- `test_mixed_*.csv`

Each file should contain extracted and engineered features for the anomaly detection task. The `feature_analysis.py` handles loading and normalization of these files using:

```python
preprocess_global(data_class="train", device={"OptiPlex", "Jetson", "ZCU102"}, scheduler={"SIMPLE", "MET", "EFT"}, DATASET_PATH="./datasets")
```

---

## 🧠 Model Architecture

The AutoEncoder is defined in `models.py` via the `create_AE(input_shape)` function, using a shallow fully connected architecture with batch normalization and dropout.

- Encoder: Dense(16) → BN → Dropout → Dense(8)
- Decoder: Dense(16) → BN → Dropout → Dense(original_dim)

The model is trained to reconstruct input data and compute anomaly scores via **Mean Squared Error (MSE)**.

---

## 🧮 Inference and Scoring

- The script uses a Z-score thresholding strategy on validation reconstruction loss:
  
  ```python
  threshold = mean_val + STD_DEV_K * std_val
  ```

- All test samples with MSE > threshold are classified as **anomalies (1)**.

- Reported metrics:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F0.5 / F1 / F2 scores**
  - **False Positive Rate (FPR)**
  - **False Negative Rate (FNR)**

---

## 📊 Logs and Reproducibility

Every experiment run logs:
- Timestamp
- Model path
- Z-score threshold and percentile
- All metric results

To:
```text
ae_experiment_log.txt
```

This enables simple reproducibility and comparison across runs.

---

## 📦 Install Requirements

To install only the needed dependencies for this project:

```bash
pip install -r requirements.txt
```

If you need a clean environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🧑‍🔬 For Training & Validation (Extendable)

While this release includes only inference, you can extend the codebase to perform training by:

1. Modifying `test_AE.py` to import and call `create_AE()` from `models.py`
2. Feeding it:
   ```python
   model.fit(train, train, validation_data=(val, val), epochs=..., ...)
   ```
3. Saving the model:
   ```python
   model.save('./AE_model_binary')
   ```

For now, the trained model is already included, so this is optional.
