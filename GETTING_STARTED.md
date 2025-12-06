# 🚀 Getting Started - NCV Classification

## Quick Start (3 Steps)

### 1. Install Dependencies
```bash
cd NCV_Classification
pip install -r requirements.txt
```

### 2. Generate Sample Data (for testing)
```bash
python utils/create_sample_ncv_data.py
```

### 3. Run Pipeline
```bash
# Quick test (10 minutes)
python test_ncv_pipeline.py

# Full pipeline (30-60 minutes)
python ncv_main.py
```

---

## Project Structure

```
NCV_Classification/
├── README.md                    # Main documentation
├── GETTING_STARTED.md          # This file
├── requirements.txt            # Python dependencies
├── ncv_config.json             # Configuration
├── ncv_main.py                 # Main pipeline
├── test_ncv_pipeline.py        # Quick test script
│
├── src/                        # Source code
│   ├── ncv_data_loader.py      # Load CSV files
│   ├── ncv_preprocessing.py    # Normalization
│   ├── ncv_models.py           # CNN, LSTM, Hybrid
│   ├── ncv_train.py            # Training logic
│   └── ncv_evaluate.py         # Evaluation
│
├── utils/                      # Utilities
│   └── create_sample_ncv_data.py
│
├── data/                       # Data folder
│   └── NCV/
│       ├── non_cts/
│       ├── mild/
│       ├── moderate/
│       └── severe/
│
└── docs/                       # Documentation
    ├── NCV_INDEX.md
    ├── NCV_QUICK_START.md
    ├── NCV_CHECKLIST.md
    ├── NCV_SUMMARY.txt
    ├── EMG_vs_NCV_COMPARISON.md
    └── ARCHITECTURE_DIAGRAM.txt
```

---

## Data Format

### NCV Data (for this system)
Place your NCV CSV files in `data/NCV/` with this structure:

```
data/NCV/
├── non_cts/
│   ├── patient001.csv
│   └── patient002.csv
├── mild/
├── moderate/
└── severe/
```

### NCV Signal Data (reference only)
The folder also contains NCV signal datasets (`Full_Data/`, `Motorik/`, `Sensorik/`) with raw waveforms. These are **NOT** directly usable with the current NCV classification system (which expects tabular features). See `data/NCV_SIGNAL_DATA_INFO.md` for details.

**CSV Format:**
```csv
distal_latency,amplitude,conduction_velocity,f_wave_latency
3.5,8.2,52.3,28.1
```

---

## Configuration

Edit `ncv_config.json` to customize:

```json
{
  "data": {
    "base_directory": "data/NCV"
  },
  "training": {
    "hyperparameters": {
      "epochs": 100,
      "batch_size": 16
    }
  }
}
```

---

## Expected Output

After running, check:
```
experiments_ncv/ncv_classification_YYYYMMDD_HHMMSS/
├── FINAL_REPORT.md              # Results summary
├── models/                      # Trained models (.h5)
├── plots/                       # Confusion matrices
└── results/                     # CSV metrics
```

---

## Performance Targets

- **Minimum:** 70% accuracy
- **Good:** 80% accuracy
- **Excellent:** 85%+ accuracy

---

## Documentation

- **README.md** - Complete documentation
- **docs/NCV_QUICK_START.md** - Quick guide
- **docs/NCV_INDEX.md** - Navigation
- **docs/NCV_CHECKLIST.md** - Complete checklist
- **docs/EMG_vs_NCV_COMPARISON.md** - EMG vs NCV comparison

---

## Troubleshooting

### Issue: "No valid NCV data loaded"
**Solution:** Generate sample data first
```bash
python utils/create_sample_ncv_data.py
```

### Issue: "ModuleNotFoundError"
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: Low accuracy (<60%)
**Solution:** 
- Increase training data (30+ samples per class)
- Increase epochs to 150
- Try different normalization methods

---

## Next Steps

1. ✅ Read [README.md](README.md) for complete documentation
2. ✅ Run `python test_ncv_pipeline.py` to verify setup
3. ✅ Prepare your NCV data in CSV format
4. ✅ Update `ncv_config.json` with your data path
5. ✅ Run `python ncv_main.py` for full training
6. ✅ Check results in `experiments_ncv/`

---

**Ready? Run:**
```bash
python test_ncv_pipeline.py
```

**Good luck! 🚀**
