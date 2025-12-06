# Data Directory

## Structure

This folder contains multiple data types for CTS classification:

```
data/
├── NCV/            # NCV data (CSV format) - for NCV classification
│   ├── non_cts/
│   ├── mild/
│   ├── moderate/
│   └── severe/
│
├── Full_Data/      # Complete EMG data (all channels)
│   ├── non_cts/
│   ├── mild/
│   ├── moderate/
│   └── severe/
│
├── Motorik/        # Motor nerve EMG data
│   ├── non_cts/
│   ├── mild/
│   ├── moderate/
│   └── severe/
│
└── Sensorik/       # Sensory nerve EMG data
    ├── non_cts/
    ├── mild/
    ├── moderate/
    └── severe/
```

## Data Types

### NCV Data (for NCV Classification)
Place your NCV CSV files in the `NCV/` folder structure

## CSV Format

Your CSV files should contain NCV measurements with the following columns:

### Motor Nerve Features
- `distal_latency` (ms) - Distal motor latency
- `amplitude` (mV) - Compound muscle action potential amplitude
- `conduction_velocity` (m/s) - Motor nerve conduction velocity
- `f_wave_latency` (ms) - F-wave latency

### Sensory Nerve Features
- `peak_latency` (ms) - Sensory peak latency
- `amplitude` (μV) - Sensory nerve action potential amplitude
- `conduction_velocity` (m/s) - Sensory nerve conduction velocity

## Example CSV Files

### Example 1: Motor Only
```csv
distal_latency,amplitude,conduction_velocity,f_wave_latency
3.5,8.2,52.3,28.1
```

### Example 2: Sensory Only
```csv
peak_latency,amplitude,conduction_velocity
2.8,15.4,58.7
```

### Example 3: Combined (Recommended)
```csv
distal_latency,amplitude,conduction_velocity,f_wave_latency,sensory_peak_latency,sensory_amplitude,sensory_conduction_velocity
3.5,8.2,52.3,28.1,2.8,15.4,58.7
```

## Clinical Reference Ranges

### Non-CTS (Normal)
- Motor distal latency: 2.5-3.5 ms
- Motor amplitude: 8.0-15.0 mV
- Motor conduction velocity: 50.0-65.0 m/s
- Sensory peak latency: 2.0-3.0 ms
- Sensory amplitude: 15.0-50.0 μV
- Sensory conduction velocity: 50.0-65.0 m/s

### Mild CTS
- Motor distal latency: 3.5-4.5 ms (slightly prolonged)
- Motor amplitude: 6.0-10.0 mV (slightly reduced)
- Motor conduction velocity: 45.0-52.0 m/s (slightly slow)
- Sensory amplitude: 10.0-20.0 μV (reduced)

### Moderate CTS
- Motor distal latency: 4.5-6.0 ms (prolonged)
- Motor amplitude: 4.0-7.0 mV (reduced)
- Motor conduction velocity: 38.0-47.0 m/s (slow)
- Sensory amplitude: 5.0-12.0 μV (significantly reduced)

### Severe CTS
- Motor distal latency: 6.0-8.0 ms (very prolonged)
- Motor amplitude: 2.0-5.0 mV (very reduced)
- Motor conduction velocity: 30.0-40.0 m/s (very slow)
- Sensory amplitude: 2.0-8.0 μV (very reduced or absent)

## Data Requirements

### Minimum Requirements
- At least 10 files per class (40 total)
- Valid CSV format with headers
- No missing values (or will be imputed)
- Consistent column names across files

### Recommended
- 30+ files per class (120+ total)
- Balanced classes (similar number of samples)
- Clean data (no outliers or errors)
- Multiple measurements per patient (if available)

## Available Data

### NCV Signal Data (Raw Waveforms - Reference)
- **Full_Data/**: Complete NCV recordings (Motor + Sensory combined)
- **Motorik/**: Motor nerve NCV recordings only
- **Sensorik/**: Sensory nerve NCV recordings only

These folders contain **raw NCV waveform files** (.csv) with signal traces. They require feature extraction before use with the classification system.

**File Format Example:**
```csv
Test Name,NCV
Test Item,Right Median Motor
Patient ID,1013777
Trace Data (μV),-2104.977,-49.347
,-26415.810,-114.155
...
```

### NCV Tabular Data (For NCV Classification)
The `NCV/` folder is for **extracted NCV features** in tabular CSV format (latency, amplitude, velocity).

## Generate Sample NCV Data

If you don't have real NCV data yet, generate sample data for testing:

```bash
python utils/create_sample_ncv_data.py --samples 20 --output-dir data/NCV
```

This will create:
- 20 motor files per class
- 20 sensory files per class
- 20 combined files per class
- Total: 240 files (60 per class)

## Data Privacy

⚠️ **Important**: 
- Remove all patient identifiable information (PII)
- Use anonymous patient IDs only
- Comply with HIPAA/GDPR regulations
- Do not commit real patient data to version control

## Troubleshooting

### Issue: "No valid NCV data loaded"
**Cause**: No CSV files found or incorrect folder structure

**Solution**:
1. Check folder structure matches exactly
2. Verify CSV files exist in each class folder
3. Generate sample data for testing

### Issue: "KeyError: 'distal_latency'"
**Cause**: Column names in CSV don't match config

**Solution**:
1. Check your CSV column names
2. Update `ncv_config.json` to match your columns
3. Ensure column names are lowercase and use underscores

### Issue: "ValueError: could not convert string to float"
**Cause**: Non-numeric values in CSV

**Solution**:
1. Check for text in numeric columns
2. Remove or fix invalid values
3. Ensure proper CSV formatting

## Next Steps

1. Place your CSV files in the appropriate folders
2. Update `ncv_config.json` with correct feature names
3. Run `python test_ncv_pipeline.py` to verify
4. Run `python ncv_main.py` for full training

For more information, see [GETTING_STARTED.md](../GETTING_STARTED.md)
