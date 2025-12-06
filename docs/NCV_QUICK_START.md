# ⚡ NCV Classification - Quick Start Guide

## 🎯 3 Langkah Utama

### 1️⃣ **SETUP (5 menit)**

```bash
# Install dependencies
pip install -r ncv_requirements.txt

# Generate sample data (untuk testing)
python create_sample_ncv_data.py --samples 15
```

### 2️⃣ **CONFIGURE (2 menit)**

Edit `ncv_config.json`:

```json
{
  "data": {
    "base_directory": "data/NCV"  // UPDATE jika perlu
  }
}
```

### 3️⃣ **RUN (30-60 menit)**

```bash
python ncv_main.py
```

---

## 📊 Apa yang Terjadi?

```
Step 1: Load NCV data dari CSV
  ↓
Step 2: Preprocessing (normalization + augmentation)
  ↓
Step 3: Train 3 models (CNN, LSTM, CNN-LSTM)
  ↓
Step 4: Evaluate & compare models
  ↓
Step 5: Cross validation (5-fold)
  ↓
Generate report & plots
```

---

## ⏱️ Timeline

| Step | Durasi | Deskripsi |
|------|--------|-----------|
| Load data | < 1 min | Read CSV files |
| Preprocessing | < 1 min | Normalize features |
| Train CNN | 10-15 min | 100 epochs |
| Train LSTM | 10-15 min | 100 epochs |
| Train CNN-LSTM | 10-15 min | 100 epochs |
| Evaluation | < 1 min | Test set metrics |
| Cross validation | 10-15 min | 5-fold CV |
| **TOTAL** | **30-60 min** | Full pipeline |

---

## 📁 Format Data

### Option 1: Motor Only

```csv
distal_latency,amplitude,conduction_velocity,f_wave_latency
3.5,8.2,52.3,28.1
```

### Option 2: Sensory Only

```csv
peak_latency,amplitude,conduction_velocity
2.8,15.4,58.7
```

### Option 3: Combined (Recommended)

```csv
distal_latency,amplitude,conduction_velocity,f_wave_latency,sensory_peak_latency,sensory_amplitude,sensory_conduction_velocity
3.5,8.2,52.3,28.1,2.8,15.4,58.7
```

---

## 📊 Expected Output

```
experiments_ncv/ncv_classification_20241206_143022/
├── FINAL_REPORT.md
├── models/
│   ├── ncv_cnn_best.h5          ← Best CNN model
│   ├── ncv_lstm_best.h5         ← Best LSTM model
│   └── ncv_cnn_lstm_best.h5     ← Best Hybrid model
├── plots/
│   ├── all_confusion_matrices.png
│   └── model_comparison.png
└── results/
    ├── model_comparison.csv
    └── cv_results.csv
```

### Console Output:

```
================================================================================
MODEL COMPARISON
================================================================================
Model            Accuracy  Precision  Recall  F1-Score
ncv_cnn_lstm     0.8250    0.8180     0.8250  0.8200
ncv_cnn          0.8000    0.7950     0.8000  0.7970
ncv_lstm         0.7750    0.7680     0.7750  0.7710
```

---

## 🎯 Target Performance

- **Minimum acceptable:** 70% accuracy
- **Good performance:** 80% accuracy
- **Excellent performance:** 85%+ accuracy

---

## 🔧 Common Issues

### Issue 1: "No valid NCV data loaded"

**Fix:**
```bash
# Check folder structure
ls data/NCV/
# Should show: non_cts  mild  moderate  severe

# Generate sample data
python create_sample_ncv_data.py
```

### Issue 2: "KeyError: 'distal_latency'"

**Fix:** Update config dengan column names yang sesuai

```python
# Check your CSV columns
import pandas as pd
df = pd.read_csv('data/NCV/non_cts/patient1.csv')
print(df.columns)

# Update ncv_config.json accordingly
```

### Issue 3: Low accuracy (<60%)

**Fix:**
1. Increase data: minimum 20 samples per class
2. Enable augmentation (already default)
3. Increase epochs to 150
4. Try different normalization methods

### Issue 4: Out of memory

**Fix:**
```json
"training": {
  "hyperparameters": {
    "batch_size": 8  // reduce from 16
  }
}
```

---

## 🚀 Advanced Usage

### Custom Data Path

```bash
python ncv_main.py --data-dir /path/to/your/ncv/data
```

### Custom Config

```bash
python ncv_main.py --config my_custom_config.json
```

### Generate More Sample Data

```bash
python create_sample_ncv_data.py --samples 50 --output-dir data/NCV_large
```

---

## 📈 Optimization Tips

### For Better Accuracy:

1. **More data:** 30+ samples per class
2. **Feature engineering:** Add derived features
3. **Hyperparameter tuning:**
   ```json
   {
     "learning_rate": 0.0005,
     "epochs": 150,
     "batch_size": 32
   }
   ```

### For Faster Training:

1. **Disable CV:**
   ```json
   "cross_validation": {"enabled": false}
   ```

2. **Reduce epochs:**
   ```json
   "hyperparameters": {"epochs": 50}
   ```

3. **Increase batch size:**
   ```json
   "hyperparameters": {"batch_size": 32}
   ```

---

## 🆚 NCV vs EMG Comparison

| Aspect | NCV | EMG |
|--------|-----|-----|
| **Data type** | Tabular (CSV) | Time-series (TXT) |
| **Preprocessing** | Simple (normalize) | Complex (STFT) |
| **Training time** | 30-60 min | 2-3 hours |
| **Expected accuracy** | 70-85% | 60-75% |
| **Model complexity** | Lower | Higher |
| **Easier to interpret** | ✅ Yes | ❌ No |

---

## 📚 File Descriptions

| File | Purpose |
|------|---------|
| `ncv_main.py` | Main pipeline (run this) |
| `ncv_config.json` | Configuration |
| `ncv_models.py` | Model architectures |
| `ncv_data_loader.py` | Load CSV/Excel |
| `ncv_preprocessing.py` | Normalization |
| `ncv_train.py` | Training logic |
| `ncv_evaluate.py` | Evaluation metrics |
| `create_sample_ncv_data.py` | Generate test data |

---

## ✅ Checklist

- [ ] Install dependencies (`pip install -r ncv_requirements.txt`)
- [ ] Prepare data atau generate sample (`python create_sample_ncv_data.py`)
- [ ] Update config (`ncv_config.json`)
- [ ] Run pipeline (`python ncv_main.py`)
- [ ] Check results (`experiments_ncv/*/FINAL_REPORT.md`)
- [ ] Compare models (CNN vs LSTM vs CNN-LSTM)
- [ ] Use best model for production

---

## 🎓 Next Steps After Training

1. **Analyze results:**
   - Read `FINAL_REPORT.md`
   - Check confusion matrices
   - Compare model performance

2. **Use best model:**
   ```python
   from tensorflow import keras
   model = keras.models.load_model('experiments_ncv/.../models/ncv_cnn_lstm_best.h5')
   
   # Predict new data
   prediction = model.predict(new_data)
   ```

3. **Improve performance:**
   - Add more data
   - Feature engineering
   - Hyperparameter tuning

---

**Ready to start? Run:**

```bash
python create_sample_ncv_data.py && python ncv_main.py
```

**Happy Training! 🚀**
