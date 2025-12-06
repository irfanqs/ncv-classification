# 📊 NCV Signal Data Information

## Available NCV Signal Datasets

This folder contains three types of **NCV signal data** (raw waveforms) that were copied from the main project:

### 1. Full_Data/
**Description:** Complete NCV recordings (Motor + Sensory combined)

**Structure:**
```
Full_Data/
├── non_cts/    # Healthy patients
├── mild/       # Mild CTS
├── moderate/   # Moderate CTS
└── severe/     # Severe CTS
```

**File Format:** `.csv` files containing raw NCV signal waveforms

**Content Example:**
```csv
Test Name,NCV
Test Item,Right Median Motor
Patient Name,"Tn Adi Hari S"
Patient ID,1013777
Test Date,30/09/2025
Trace Label,"Wrist : ","Elbow : "
Sensitivity (μV/Div),5000.00,5000.00
...
Trace Data (μV),-2104.977123349,-49.347757545
,-26415.810196730,-114.155730271
...
```

**Use Case:** 
- Raw NCV waveform analysis
- Feature extraction (latency, amplitude, velocity)
- Signal processing research

---

### 2. Motorik/
**Description:** Motor nerve NCV recordings only

**Structure:**
```
Motorik/
├── non_cts/
├── mild/
├── moderate/
└── severe/
```

**File Format:** `.csv` or `.txt` files

**Use Case:**
- Motor nerve conduction analysis
- Focused on motor pathway
- Faster processing (single channel)

---

### 3. Sensorik/
**Description:** Sensory nerve NCV recordings only

**Structure:**
```
Sensorik/
├── non_cts/
├── mild/
├── moderate/
└── severe/
```

**File Format:** `.csv` or `.txt` files

**Use Case:**
- Sensory nerve conduction analysis
- Focused on sensory pathway
- Complementary to motor data

---

## Data Characteristics

### Signal Properties
- **Type:** NCV waveform recordings (CMAP/SNAP)
- **Format:** CSV with metadata header + trace data
- **Sampling:** ~0.078 ms/sample (12,804 Hz)
- **Samples:** ~640 samples per trace
- **Duration:** ~50 ms per recording
- **Channels:** Wrist and Elbow stimulation points

### File Naming Convention
Typically: `PatientID_NerveType_Side.csv`

Example:
- `1013777_Median Motor_R.csv`
- `108198_Ulnar Sensory_L.csv`

---

## Using NCV Signal Data with NCV Classification System

⚠️ **Important:** These NCV signal datasets are **NOT directly compatible** with the current NCV classification system.

### Why?
- **These files:** Raw NCV waveform data (time-series signals)
- **NCV system expects:** Tabular feature data (latency, amplitude, velocity already extracted)

### To Use These NCV Signal Files:
You need to:
1. **Parse the CSV files** to extract waveform data
2. **Detect response onset** → Calculate latency
3. **Measure peak amplitude** → Extract amplitude
4. **Calculate conduction velocity** → Distance/latency
5. **Save as tabular CSV** in NCV format
6. **Place in `data/NCV/`** folder

Example output format:
```csv
distal_latency,amplitude,conduction_velocity,f_wave_latency
3.5,8.2,52.3,28.1
```

### Or Use Signal Processing System:
The main project (parent folder) has signal processing capabilities that can handle these waveform files.

---

## Data Statistics

To check data statistics:

```bash
# Count files per class
ls -1 Full_Data/non_cts/ | wc -l
ls -1 Full_Data/mild/ | wc -l
ls -1 Full_Data/moderate/ | wc -l
ls -1 Full_Data/severe/ | wc -l
```

---

## Converting EMG to NCV Features

If you want to extract NCV features from EMG signals, you'll need to:

1. **Load EMG signal**
2. **Detect response onset** (latency calculation)
3. **Measure peak amplitude**
4. **Calculate conduction velocity** (distance/latency)
5. **Save as CSV** in NCV format

Example output CSV:
```csv
distal_latency,amplitude,conduction_velocity,f_wave_latency
3.5,8.2,52.3,28.1
```

---

## Recommendations

### For NCV Classification:
- Use `data/NCV/` folder
- Generate sample data: `python utils/create_sample_ncv_data.py`
- Or prepare real NCV measurements in CSV format

### For EMG Classification:
- Use the main project's EMG system
- These datasets are already in the correct format
- See parent folder's `main.py` or `main_menu.py`

---

## Data Privacy

⚠️ **Important:**
- These datasets may contain patient information
- Ensure HIPAA/GDPR compliance
- Do not share or publish without proper authorization
- Remove PII before any external use

---

## Next Steps

1. **For NCV work:** Focus on `data/NCV/` folder
2. **For EMG work:** Use parent project's EMG system
3. **For conversion:** Implement EMG→NCV feature extraction
4. **For testing:** Generate sample NCV data

---

**Note:** This data was copied from the main project for reference. The NCV classification system is designed for tabular NCV measurements, not raw EMG signals.
