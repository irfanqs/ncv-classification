# 📚 NCV Classification - Complete Index

## 🎯 Start Here

Baru pertama kali? Mulai dari sini:

1. **[NCV_QUICK_START.md](NCV_QUICK_START.md)** - Panduan 3 langkah untuk memulai
2. **[NCV_README.md](NCV_README.md)** - Dokumentasi lengkap
3. **[EMG_vs_NCV_COMPARISON.md](EMG_vs_NCV_COMPARISON.md)** - Perbandingan EMG vs NCV

---

## 📁 File Structure

### 🔧 Core Files (Yang Anda Jalankan)

| File | Purpose | When to Use |
|------|---------|-------------|
| **ncv_main.py** | Main pipeline | Run full training |
| **test_ncv_pipeline.py** | Quick test | Test installation |
| **create_sample_ncv_data.py** | Generate sample data | Testing/demo |

### ⚙️ Configuration

| File | Purpose |
|------|---------|
| **ncv_config.json** | Main configuration |
| **ncv_requirements.txt** | Python dependencies |

### 📚 Documentation

| File | Content |
|------|---------|
| **NCV_README.md** | Complete documentation |
| **NCV_QUICK_START.md** | Quick start guide |
| **EMG_vs_NCV_COMPARISON.md** | EMG vs NCV comparison |
| **NCV_INDEX.md** | This file |

### 🧩 Module Files (Internal)

| File | Purpose |
|------|---------|
| **ncv_models.py** | CNN, LSTM, CNN-LSTM architectures |
| **ncv_data_loader.py** | Load CSV/Excel files |
| **ncv_preprocessing.py** | Normalization & augmentation |
| **ncv_train.py** | Training & cross-validation |
| **ncv_evaluate.py** | Evaluation & metrics |

---

## 🚀 Quick Commands

### First Time Setup
```bash
# Install dependencies
pip install -r ncv_requirements.txt

# Test installation
python test_ncv_pipeline.py
```

### Generate Sample Data
```bash
# Default: 10 samples per class
python create_sample_ncv_data.py

# Custom: 50 samples per class
python create_sample_ncv_data.py --samples 50 --output-dir data/NCV_large
```

### Run Pipeline
```bash
# Default config
python ncv_main.py

# Custom config
python ncv_main.py --config my_config.json

# Custom data directory
python ncv_main.py --data-dir /path/to/data
```

---

## 📊 Data Format

### Required Folder Structure
```
data/NCV/
├── non_cts/
│   ├── patient001.csv
│   └── patient002.csv
├── mild/
│   ├── patient010.csv
│   └── patient011.csv
├── moderate/
│   ├── patient020.csv
│   └── patient021.csv
└── severe/
    ├── patient030.csv
    └── patient031.csv
```

### CSV Format Options

**Option 1: Motor only**
```csv
distal_latency,amplitude,conduction_velocity,f_wave_latency
3.5,8.2,52.3,28.1
```

**Option 2: Sensory only**
```csv
peak_latency,amplitude,conduction_velocity
2.8,15.4,58.7
```

**Option 3: Combined (recommended)**
```csv
distal_latency,amplitude,conduction_velocity,f_wave_latency,sensory_peak_latency,sensory_amplitude,sensory_conduction_velocity
3.5,8.2,52.3,28.1,2.8,15.4,58.7
```

---

## 🎓 Learning Path

### Beginner (1-2 hours)
1. Read [NCV_QUICK_START.md](NCV_QUICK_START.md)
2. Run `python test_ncv_pipeline.py`
3. Check output in `experiments_ncv_test/`

### Intermediate (1 day)
1. Read [NCV_README.md](NCV_README.md)
2. Generate sample data: `python create_sample_ncv_data.py --samples 30`
3. Run full pipeline: `python ncv_main.py`
4. Analyze results in `FINAL_REPORT.md`

### Advanced (3-5 days)
1. Read [EMG_vs_NCV_COMPARISON.md](EMG_vs_NCV_COMPARISON.md)
2. Prepare your real NCV data
3. Customize `ncv_config.json`
4. Experiment with hyperparameters
5. Compare model architectures
6. Deploy best model

---

## 🔧 Configuration Guide

### Minimal Config (Quick Test)
```json
{
  "data": {
    "base_directory": "data/NCV",
    "max_files_per_class": 5
  },
  "training": {
    "hyperparameters": {
      "epochs": 20,
      "batch_size": 8
    }
  },
  "cross_validation": {
    "enabled": false
  }
}
```

### Full Config (Production)
```json
{
  "data": {
    "base_directory": "data/NCV",
    "max_files_per_class": null
  },
  "training": {
    "hyperparameters": {
      "epochs": 100,
      "batch_size": 16
    }
  },
  "cross_validation": {
    "enabled": true,
    "n_splits": 5
  }
}
```

---

## 📊 Expected Results

### Performance Targets

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Accuracy | 70% | 80% | 85%+ |
| Training time | 30 min | 45 min | 60 min |
| Memory usage | 2 GB | 4 GB | 8 GB |

### Model Comparison

Typical results:
```
Model            Accuracy  F1-Score
ncv_cnn_lstm     0.825     0.820
ncv_cnn          0.800     0.797
ncv_lstm         0.775     0.771
```

---

## 🆘 Troubleshooting

### Common Issues

| Issue | Solution | Reference |
|-------|----------|-----------|
| Dependencies missing | `pip install -r ncv_requirements.txt` | NCV_QUICK_START.md |
| No data found | Check folder structure | NCV_README.md |
| Low accuracy | Increase data, tune hyperparameters | NCV_README.md |
| Out of memory | Reduce batch_size | NCV_QUICK_START.md |
| Column name error | Update config features | NCV_README.md |

---

## 🎯 Use Cases

### Research
- Compare CNN vs LSTM vs Hybrid
- Study feature importance
- Analyze cross-validation results
- Publish findings

### Clinical Deployment
- Train on hospital data
- Deploy best model
- Real-time prediction
- Integration with EMR

### Education
- Learn deep learning
- Understand medical AI
- Practice data science
- Build portfolio

---

## 📈 Workflow Examples

### Example 1: Quick Test (10 min)
```bash
python test_ncv_pipeline.py
```

### Example 2: Sample Data Training (30 min)
```bash
python create_sample_ncv_data.py --samples 20
python ncv_main.py
```

### Example 3: Full Production (1 hour)
```bash
# 1. Prepare real data in data/NCV/
# 2. Update ncv_config.json
# 3. Run pipeline
python ncv_main.py

# 4. Check results
cat experiments_ncv/*/FINAL_REPORT.md
```

---

## 🔗 Related Files

### EMG System (Original)
- `main.py` - EMG pipeline
- `readme.md` - EMG documentation
- `QUICK_START.md` - EMG quick start

### Comparison
- `EMG_vs_NCV_COMPARISON.md` - Detailed comparison

---

## 📞 Support Checklist

Before asking for help:

- [ ] Read [NCV_QUICK_START.md](NCV_QUICK_START.md)
- [ ] Run `python test_ncv_pipeline.py`
- [ ] Check [NCV_README.md](NCV_README.md) troubleshooting
- [ ] Verify data format and folder structure
- [ ] Check `ncv_config.json` settings
- [ ] Review error messages in console

---

## 🎉 Success Checklist

You're successful when:

- [ ] `test_ncv_pipeline.py` runs without errors
- [ ] Pipeline completes and generates `FINAL_REPORT.md`
- [ ] Accuracy > 70% on test set
- [ ] Confusion matrices look reasonable
- [ ] Best model saved in `models/` folder
- [ ] Can load and use model for prediction

---

## 🚀 Next Steps

After successful training:

1. **Analyze Results**
   - Read `FINAL_REPORT.md`
   - Check confusion matrices
   - Compare model performance

2. **Improve Performance**
   - Add more data
   - Feature engineering
   - Hyperparameter tuning

3. **Deploy Model**
   ```python
   from tensorflow import keras
   model = keras.models.load_model('path/to/best_model.h5')
   prediction = model.predict(new_data)
   ```

4. **Compare with EMG**
   - Read [EMG_vs_NCV_COMPARISON.md](EMG_vs_NCV_COMPARISON.md)
   - Consider ensemble approach

---

## 📚 Documentation Map

```
NCV_INDEX.md (You are here)
├── NCV_QUICK_START.md
│   ├── 3-step guide
│   ├── Timeline
│   └── Common issues
│
├── NCV_README.md
│   ├── Complete documentation
│   ├── Architecture details
│   ├── Configuration options
│   └── Advanced usage
│
└── EMG_vs_NCV_COMPARISON.md
    ├── Pipeline comparison
    ├── Performance comparison
    └── When to use which
```

---

**Ready to start? Run:**

```bash
python test_ncv_pipeline.py
```

**Good luck! 🚀**
