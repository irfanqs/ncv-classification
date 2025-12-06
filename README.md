# 🧠 NCV Classification System

Sistem klasifikasi tingkat keparahan Carpal Tunnel Syndrome (CTS) dari data **Nerve Conduction Velocity (NCV)** menggunakan deep learning.

## 📊 Perbedaan dengan EMG

| Aspek | EMG | NCV |
|-------|-----|-----|
| **Data Type** | Time-series signal (kontinu) | Tabular data (diskrit) |
| **Features** | Raw waveform | Latency, amplitude, velocity |
| **Preprocessing** | STFT → Spectrogram | Normalization only |
| **Model Input** | 2D image (256×256) | 1D vector (n_features) |
| **Architecture** | 2D CNN | 1D CNN |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn openpyxl
```

### 2. Prepare Data

Struktur folder yang diperlukan:

```
data/NCV/
├── non_cts/
│   ├── patient1.csv
│   ├── patient2.csv
│   └── ...
├── mild/
│   ├── patient10.csv
│   └── ...
├── moderate/
│   ├── patient20.csv
│   └── ...
└── severe/
    ├── patient30.csv
    └── ...
```

**Format CSV:**

```csv
distal_latency,amplitude,conduction_velocity,f_wave_latency
3.5,8.2,52.3,28.1
```

atau

```csv
peak_latency,amplitude,conduction_velocity
2.8,15.4,58.7
```

### 3. Configure

Edit `ncv_config.json`:

```json
{
  "data": {
    "base_directory": "data/NCV"  // UPDATE PATH INI
  },
  "features": {
    "motor_features": [
      "distal_latency",
      "amplitude", 
      "conduction_velocity",
      "f_wave_latency"
    ],
    "sensory_features": [
      "peak_latency",
      "amplitude",
      "conduction_velocity"
    ],
    "use_motor": true,
    "use_sensory": true
  }
}
```

### 4. Run Pipeline

```bash
python ncv_main.py
```

atau dengan custom config:

```bash
python ncv_main.py --config my_config.json --data-dir /path/to/data
```

---

## 📁 File Structure

```
NCV_Classification/
├── ncv_config.json           # Configuration
├── ncv_main.py               # Main pipeline
├── ncv_models.py             # CNN, LSTM, CNN-LSTM models
├── ncv_data_loader.py        # Load CSV/Excel files
├── ncv_preprocessing.py      # Normalization & augmentation
├── ncv_train.py              # Training logic
├── ncv_evaluate.py           # Evaluation & metrics
│
├── data/NCV/                 # Your data here
│   ├── non_cts/
│   ├── mild/
│   ├── moderate/
│   └── severe/
│
└── experiments_ncv/          # Output (auto-generated)
    └── ncv_classification_YYYYMMDD_HHMMSS/
        ├── models/           # Trained .h5 files
        ├── plots/            # Confusion matrices
        ├── results/          # CSV metrics
        └── FINAL_REPORT.md   # Summary
```

---

## 🏗️ Model Architectures

### 1. **1D CNN**
```
Input (n_features, 1)
  ↓
Conv1D (64 filters) → BatchNorm → ReLU → MaxPool → Dropout
  ↓
Conv1D (128 filters) → BatchNorm → ReLU → MaxPool → Dropout
  ↓
Conv1D (256 filters) → BatchNorm → ReLU → GlobalAvgPool → Dropout
  ↓
Dense (128) → Dropout → Dense (64) → Dropout
  ↓
Dense (4) → Softmax [non_cts, mild, moderate, severe]
```

### 2. **LSTM**
```
Input (n_features, 1)
  ↓
Bidirectional LSTM (64 units)
  ↓
Bidirectional LSTM (32 units)
  ↓
Dense (128) → Dropout → Dense (64) → Dropout
  ↓
Dense (4) → Softmax
```

### 3. **CNN-LSTM Hybrid**
```
Input (n_features, 1)
  ↓
Conv1D (64) → BatchNorm → ReLU → MaxPool
  ↓
Conv1D (128) → BatchNorm → ReLU → MaxPool
  ↓
Bidirectional LSTM (64 units)
  ↓
Bidirectional LSTM (32 units)
  ↓
Dense (128) → Dense (4) → Softmax
```

---

## ⚙️ Configuration Options

### Data Configuration

```json
"data": {
  "base_directory": "data/NCV",
  "classes": ["non_cts", "mild", "moderate", "severe"],
  "file_extensions": [".csv", ".xlsx", ".txt"],
  "max_files_per_class": null  // null = use all files
}
```

### Feature Configuration

```json
"features": {
  "motor_features": ["distal_latency", "amplitude", "conduction_velocity"],
  "sensory_features": ["peak_latency", "amplitude", "conduction_velocity"],
  "use_motor": true,
  "use_sensory": true,
  "normalization": "standard"  // "standard", "minmax", or "robust"
}
```

### Training Configuration

```json
"training": {
  "hyperparameters": {
    "epochs": 100,
    "batch_size": 16,
    "learning_rate": 0.001,
    "optimizer": "adam"  // "adam", "adamw", "sgd"
  },
  "callbacks": {
    "early_stopping": {
      "enabled": true,
      "patience": 15
    }
  }
}
```

### Cross Validation

```json
"cross_validation": {
  "enabled": true,
  "n_splits": 5,
  "stratify": true
}
```

---

## 📊 Output & Results

### What You Get

```
experiments_ncv/ncv_classification_20241206_143022/
├── FINAL_REPORT.md              # Comprehensive report
├── config.json                  # Saved configuration
├── models/
│   ├── ncv_cnn_best.h5
│   ├── ncv_lstm_best.h5
│   └── ncv_cnn_lstm_best.h5
├── plots/
│   ├── all_confusion_matrices.png
│   └── model_comparison.png
└── results/
    ├── model_comparison.csv
    └── cv_results.csv
```

### Example Output

```
MODEL COMPARISON
================================================================================
Model            Accuracy  Precision  Recall  F1-Score
ncv_cnn_lstm     0.8250    0.8180     0.8250  0.8200
ncv_cnn          0.8000    0.7950     0.8000  0.7970
ncv_lstm         0.7750    0.7680     0.7750  0.7710
```

---

## 🎯 Performance Targets

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Accuracy | 70% | 80% | 85%+ |
| Precision | 0.70 | 0.80 | 0.85+ |
| Recall | 0.70 | 0.80 | 0.85+ |
| F1-Score | 0.70 | 0.80 | 0.85+ |

**Note:** NCV data biasanya lebih mudah diklasifikasi dibanding EMG karena features sudah extracted.

---

## 🔧 Troubleshooting

### Error: "No valid NCV data loaded"

**Penyebab:**
- Folder structure salah
- File format tidak sesuai
- Column names tidak match

**Solusi:**
1. Cek struktur folder: `data/NCV/non_cts/`, `mild/`, dll
2. Pastikan file CSV/Excel valid
3. Update `motor_features` dan `sensory_features` di config sesuai column names

### Error: "KeyError: 'distal_latency'"

**Penyebab:** Column names di CSV tidak match dengan config

**Solusi:**
```python
# Cek column names di CSV
import pandas as pd
df = pd.read_csv('data/NCV/non_cts/patient1.csv')
print(df.columns)

# Update config.json sesuai column names
```

### Low Accuracy (<60%)

**Penyebab:**
- Data terlalu sedikit
- Features tidak informatif
- Imbalanced classes

**Solusi:**
1. Tambah data (minimum 20 samples per class)
2. Cek feature importance
3. Enable data augmentation di preprocessing
4. Adjust class weights (sudah auto-enabled)

### Out of Memory

**Solusi:**
```json
"training": {
  "hyperparameters": {
    "batch_size": 8  // reduce dari 16
  }
}
```

---

## 📈 Advanced Usage

### Custom Features

Edit `ncv_config.json`:

```json
"features": {
  "motor_features": [
    "distal_latency",
    "amplitude",
    "conduction_velocity",
    "f_wave_latency",
    "your_custom_feature"  // ADD YOUR FEATURE
  ]
}
```

### Hyperparameter Tuning

```json
"training": {
  "hyperparameters": {
    "epochs": 150,           // increase for better convergence
    "batch_size": 32,        // larger = faster but more memory
    "learning_rate": 0.0005  // lower = more stable
  }
}
```

### Disable Cross Validation (Faster)

```json
"cross_validation": {
  "enabled": false
}
```

---

## 🆚 Comparison: EMG vs NCV

| Feature | EMG Pipeline | NCV Pipeline |
|---------|-------------|--------------|
| Data loading | `emg_data_loader.py` | `ncv_data_loader.py` |
| Preprocessing | STFT + Spectrogram | Normalization only |
| Feature extraction | `feature_extraction.py` | Built-in (from CSV) |
| Model input | 2D (256, 256, 1) | 1D (n_features, 1) |
| CNN type | 2D Conv | 1D Conv |
| Training time | 2-3 hours | 30-60 minutes |
| Expected accuracy | 60-75% | 70-85% |

---

## 📚 Example Workflow

### Scenario 1: Quick Test (10 minutes)

```bash
# 1. Prepare minimal data (5 files per class)
# 2. Edit config
{
  "data": {"max_files_per_class": 5},
  "training": {"hyperparameters": {"epochs": 20}},
  "cross_validation": {"enabled": false}
}

# 3. Run
python ncv_main.py
```

### Scenario 2: Full Training (1 hour)

```bash
# 1. Use all data
# 2. Default config (100 epochs, CV enabled)
# 3. Run
python ncv_main.py
```

### Scenario 3: Production Model (2 hours)

```bash
# 1. Full data + augmentation
# 2. Increase epochs to 150
# 3. Enable cross validation
# 4. Train all 3 models
python ncv_main.py
```

---

## 🔬 Technical Details

### Data Augmentation

```python
# Gaussian noise augmentation
X_augmented = X_original + noise
noise ~ N(0, 0.05 * std(X))
```

### Normalization Methods

- **Standard (Z-score):** `(X - mean) / std` → Best for normal distribution
- **MinMax:** `(X - min) / (max - min)` → Best for bounded features
- **Robust:** Uses median & IQR → Best for outliers

### Class Weights

Automatically computed untuk handle imbalanced data:

```python
weight[class_i] = n_samples / (n_classes * n_samples_class_i)
```

---

## 📝 Citation

```bibtex
@software{ncv_cts_classification_2024,
  title = {NCV Classification for CTS Severity Detection},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/ncv-classification}
}
```

---

## 📞 Support

Stuck? Check:
1. This README troubleshooting section
2. `experiments_ncv/*/FINAL_REPORT.md`
3. Config file: `ncv_config.json`

---

**Version:** 1.0.0  
**Status:** Production-ready  
**Expected Accuracy:** 70-85% for 4-class CTS classification from NCV data

---

## 🎓 Next Steps

1. ✅ Prepare your NCV data in CSV format
2. ✅ Update `ncv_config.json` with your data path
3. ✅ Run `python ncv_main.py`
4. ✅ Check results in `experiments_ncv/`
5. ✅ Compare CNN vs LSTM vs CNN-LSTM
6. ✅ Use best model for prediction

**Happy Training! 🚀**
