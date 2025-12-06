# 📊 Data Update Summary

## ✅ EMG Data Successfully Copied

**Date:** December 6, 2024

---

## What Was Done

Copied three EMG datasets from the main project into `NCV_Classification/data/`:

1. ✅ **Full_Data/** - Complete EMG recordings
2. ✅ **Motorik/** - Motor nerve EMG data
3. ✅ **Sensorik/** - Sensory nerve EMG data

---

## Current Data Structure

```
NCV_Classification/data/
├── README.md                  # Updated with new structure
├── EMG_DATA_INFO.md          # New: EMG data documentation
│
├── NCV/                      # For NCV classification (CSV format)
│   ├── non_cts/
│   ├── mild/
│   ├── moderate/
│   └── severe/
│
├── Full_Data/                # EMG reference data
│   ├── non_cts/
│   ├── mild/
│   ├── moderate/
│   └── severe/
│
├── Motorik/                  # Motor EMG reference
│   ├── non_cts/
│   ├── mild/
│   ├── moderate/
│   └── severe/
│
└── Sensorik/                 # Sensory EMG reference
    ├── non_cts/
    ├── mild/
    ├── moderate/
    └── severe/
```

---

## Important Notes

### ⚠️ EMG Data is Reference Only

The copied EMG datasets (`Full_Data/`, `Motorik/`, `Sensorik/`) are **NOT directly compatible** with the NCV classification system because:

| Aspect | EMG Data | NCV Data |
|--------|----------|----------|
| **Format** | Time-series signal | Tabular features |
| **File Type** | .txt/.csv (raw signal) | .csv (measurements) |
| **Content** | Raw waveform values | Latency, amplitude, velocity |
| **Processing** | Requires STFT/spectrogram | Direct normalization |

### ✅ For NCV Classification

Use the `data/NCV/` folder with CSV files containing:
- `distal_latency`
- `amplitude`
- `conduction_velocity`
- `f_wave_latency`
- etc.

Generate sample data:
```bash
python utils/create_sample_ncv_data.py
```

### 📚 For EMG Classification

Use the main project's EMG system (parent folder) which has:
- `main.py` - EMG pipeline
- `main_menu.py` - Interactive menu
- Complete STFT/spectrogram preprocessing

---

## File Counts

Check data availability:

```bash
# NCV data (should be empty or sample data)
ls -1 data/NCV/non_cts/ | wc -l

# EMG data (reference)
ls -1 data/Full_Data/non_cts/ | wc -l
ls -1 data/Motorik/non_cts/ | wc -l
ls -1 data/Sensorik/non_cts/ | wc -l
```

---

## Updated Documentation

The following files were updated to reflect the new structure:

1. ✅ `data/README.md` - Added EMG data section
2. ✅ `data/EMG_DATA_INFO.md` - New file with EMG details
3. ✅ `STRUCTURE.txt` - Updated folder structure
4. ✅ `GETTING_STARTED.md` - Added EMG data note

---

## Usage Recommendations

### Scenario 1: NCV Classification (This System)
```bash
# Generate sample NCV data
python utils/create_sample_ncv_data.py

# Run NCV pipeline
python ncv_main.py
```

### Scenario 2: EMG Classification (Main Project)
```bash
# Go to parent folder
cd ..

# Run EMG pipeline
python main_menu.py
# Select option 2 (Motorik) or 3 (Sensorik)
```

### Scenario 3: Convert EMG to NCV
If you want to extract NCV features from EMG signals:
1. Implement feature extraction (latency, amplitude, velocity)
2. Save as CSV in NCV format
3. Place in `data/NCV/` folder

---

## Data Privacy Reminder

⚠️ **Important:**
- EMG datasets may contain patient information
- Ensure HIPAA/GDPR compliance
- Do not share without authorization
- Remove PII before external use

---

## Next Steps

1. **For NCV work:**
   - Focus on `data/NCV/` folder
   - Generate sample data or prepare real NCV measurements
   - Run `python ncv_main.py`

2. **For EMG work:**
   - Use parent project's EMG system
   - Data is already in correct format
   - Run `python main_menu.py` from parent folder

3. **For reference:**
   - EMG data is available in `Full_Data/`, `Motorik/`, `Sensorik/`
   - See `data/EMG_DATA_INFO.md` for details

---

## Summary

✅ **Completed:**
- EMG data copied to NCV_Classification/data/
- Documentation updated
- Structure clarified
- Usage guidelines provided

⚠️ **Remember:**
- NCV system uses `data/NCV/` (tabular CSV)
- EMG data is reference only (requires different system)
- Both systems are separate and independent

---

**Status:** ✅ Complete  
**Date:** December 6, 2024
