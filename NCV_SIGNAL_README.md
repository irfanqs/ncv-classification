# 🧠 NCV Signal Classification System

## Overview

Sistem klasifikasi CTS dari **NCV signal waveforms** (raw data). Berbeda dengan sistem NCV tabular yang membutuhkan features yang sudah di-extract, sistem ini bisa langsung process raw signal files.

---

## 🆚 Comparison: Two NCV Systems

| Aspect | NCV Tabular System | NCV Signal System (NEW) |
|--------|-------------------|------------------------|
| **Input** | Tabular CSV (features) | Signal CSV (waveforms) |
| **Data Format** | `latency,amplitude,velocity` | Raw trace data |
| **Preprocessing** | Normalization only | Feature extraction + normalization |
| **Use Case** | Pre-extracted features | Raw signal files |
| **Files** | `ncv_main.py` | `ncv_signal_main.py` |
| **Config** | `ncv_config.json` | `ncv_signal_config.json` |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install tensorflow numpy pandas scikit-learn scipy matplotlib seaborn
```

### 2. Check Your Data

Make sure you have NCV signal files in:
```
data/Full_Data/
├── non_cts/
├── mild/
├── moderate/
└── severe/
```

Or use `data/Motorik/` or `data/Sensorik/`

### 3. Run Quick Test

```bash
python test_ncv_signal.py
```

This will:
- Auto-detect available data
- Use 10 files per class
- Train 1 model (Dense NN)
- Take ~10-15 minutes

### 4. Run Full Pipeline

```bash
python ncv_signal_main.py
```

This will:
- Use all available data
- Train 3 models (Dense NN, CNN, LSTM)
- Take ~30-60 minutes

---

## 📊 How It Works

### Pipeline Flow

```
NCV Signal CSV Files
  ↓
Parse & Load Waveforms
  ↓
Bandpass Filter (10-2500 Hz)
  ↓
Feature Extraction:
  - Latency (onset detection)
  - Amplitude (peak-to-peak)
  - Statistical (RMS, mean, std, skewness, kurtosis)
  - Frequency (FFT-based)
  ↓
Normalization (StandardScaler)
  ↓
Train Models:
  - Dense Neural Network
  - 1D CNN
  - LSTM
  ↓
Evaluation & Results
```

### Feature Extraction

From each waveform, we extract **11 features**:

1. **Latency** - Onset time (ms)
2. **Amplitude** - Peak-to-peak (μV)
3. **RMS** - Root mean square
4. **Peak Value** - Maximum absolute value
5. **Mean Value** - Mean of absolute values
6. **Std Value** - Standard deviation
7. **Skewness** - Distribution skewness
8. **Kurtosis** - Distribution kurtosis
9. **Dominant Frequency** - Peak frequency (Hz)
10. **Mean Frequency** - Weighted mean frequency
11. **Spectral Energy** - Total spectral power

---

## ⚙️ Configuration

Edit `ncv_signal_config.json`:

```json
{
  "data": {
    "base_directory": "data/Full_Data",  // or Motorik, Sensorik
    "max_files_per_class": null          // null = use all
  },
  "signal_processing": {
    "sampling_rate": 12804,
    "apply_bandpass_filter": true,
    "bandpass_lowcut": 10,
    "bandpass_highcut": 2500
  },
  "training": {
    "hyperparameters": {
      "epochs": 100,
      "batch_size": 16,
      "learning_rate": 0.001
    }
  }
}
```

---

## 🏗️ Model Architectures

### 1. Dense Neural Network
```
Input (11 features)
  ↓
Dense(128) → BatchNorm → ReLU → Dropout
  ↓
Dense(64) → BatchNorm → ReLU → Dropout
  ↓
Dense(32) → BatchNorm → ReLU → Dropout
  ↓
Dense(4) → Softmax
```

### 2. 1D CNN
```
Input (11 features) → Reshape(11, 1)
  ↓
Conv1D(64) → BatchNorm → ReLU → MaxPool → Dropout
  ↓
Conv1D(128) → BatchNorm → ReLU → GlobalAvgPool → Dropout
  ↓
Dense(128) → Dropout → Dense(64)
  ↓
Dense(4) → Softmax
```

### 3. LSTM
```
Input (11 features) → Reshape(11, 1)
  ↓
Bidirectional LSTM(64)
  ↓
Bidirectional LSTM(32)
  ↓
Dense(128) → Dropout → Dense(64)
  ↓
Dense(4) → Softmax
```

---

## 📈 Expected Performance

| Metric | Target |
|--------|--------|
| **Accuracy** | 70-85% |
| **Training Time** | 30-60 minutes |
| **Memory Usage** | 2-4 GB |
| **GPU** | Optional |

---

## 📁 File Structure

```
NCV_Classification/
├── ncv_signal_main.py              # Main pipeline
├── test_ncv_signal.py              # Quick test
├── ncv_signal_config.json          # Configuration
│
├── ncv_signal_data_loader.py       # Load signal files
├── ncv_signal_preprocessing.py     # Feature extraction
├── ncv_signal_models.py            # Model architectures
│
└── experiments_ncv_signal/         # Results (auto-generated)
    └── ncv_signal_classification_YYYYMMDD_HHMMSS/
        ├── FINAL_REPORT.md
        ├── models/
        ├── plots/
        └── results/
```

---

## 🎯 Usage Examples

### Example 1: Quick Test (10 min)
```bash
python test_ncv_signal.py
```

### Example 2: Full Training with Full_Data
```bash
python ncv_signal_main.py --data-dir data/Full_Data
```

### Example 3: Train with Motorik Only
```bash
python ncv_signal_main.py --data-dir data/Motorik
```

### Example 4: Train with Sensorik Only
```bash
python ncv_signal_main.py --data-dir data/Sensorik
```

---

## 🔧 Troubleshooting

### Issue: "No valid NCV signal data loaded"
**Solution:** Check data folder structure
```bash
ls data/Full_Data/mild/  # Should show .csv files
```

### Issue: "Error parsing file"
**Solution:** Check CSV format - should have "Trace Data" section

### Issue: Low accuracy (<60%)
**Solution:**
- Increase data (use all files, not just 10)
- Increase epochs to 150
- Try different data folder (Full_Data vs Motorik vs Sensorik)

### Issue: Out of memory
**Solution:** Reduce batch_size in config
```json
"batch_size": 8  // reduce from 16
```

---

## 📊 Output

After running, check:

```
experiments_ncv_signal/ncv_signal_classification_YYYYMMDD_HHMMSS/
├── FINAL_REPORT.md              # Results summary
├── models/
│   ├── ncv_signal_dense_best.h5
│   ├── ncv_signal_cnn_best.h5
│   └── ncv_signal_lstm_best.h5
├── plots/
│   ├── all_confusion_matrices.png
│   └── model_comparison.png
└── results/
    └── model_comparison.csv
```

---

## 🆚 When to Use Which System?

### Use NCV Signal System (This) When:
- ✅ You have raw NCV waveform files
- ✅ Files contain "Trace Data" sections
- ✅ You want automatic feature extraction
- ✅ You have Full_Data, Motorik, or Sensorik folders

### Use NCV Tabular System When:
- ✅ You have pre-extracted features
- ✅ CSV files with latency, amplitude, velocity columns
- ✅ You want faster training (no feature extraction)
- ✅ You have data in `data/NCV/` folder

---

## 🎓 Technical Details

### Signal Processing
- **Sampling Rate:** 12,804 Hz
- **Bandpass Filter:** 10-2500 Hz (4th order Butterworth)
- **Normalization:** Z-score (StandardScaler)

### Feature Extraction
- **Latency:** Threshold-based onset detection (10% of peak)
- **Amplitude:** Peak-to-peak measurement
- **Statistical:** Moments and distribution metrics
- **Frequency:** FFT-based spectral analysis

### Training
- **Optimizer:** Adam
- **Loss:** Sparse categorical crossentropy
- **Callbacks:** Early stopping, LR reduction
- **Class Weights:** Automatic balancing

---

## 📝 Next Steps

1. ✅ Run quick test: `python test_ncv_signal.py`
2. ✅ Check results in `experiments_ncv_signal_test/`
3. ✅ Run full pipeline: `python ncv_signal_main.py`
4. ✅ Compare with tabular system if you have both data types

---

**Version:** 1.0.0  
**Status:** Production Ready  
**Expected Accuracy:** 70-85%

**Ready to start? Run:**
```bash
python test_ncv_signal.py
```

**Good luck! 🚀**
