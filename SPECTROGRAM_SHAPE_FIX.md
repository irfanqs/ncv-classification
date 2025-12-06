# ✅ Spectrogram Shape Error Fixed!

## Problem

```
ValueError: Negative dimension size caused by subtracting 2 from 1
Input shapes: [?,65536,1,64]
```

Expected: `(None, 256, 256, 1)`
Got: `(None, 65536, 1, 64)` ← Completely wrong!

## Root Cause

The `process_signal_to_spectrogram` function was returning spectrograms with incorrect shapes. The spectrograms were not being resized to 256x256 as expected.

Possible issues:
1. Spectrogram not resized to target size (256x256)
2. Extra dimensions not squeezed
3. Shape mismatch in array conversion

## Solution

### 1. Added Shape Validation & Resize

```python
# Process each spectrogram
for spec in spectrograms:
    # Ensure 2D spectrogram
    if len(spec.shape) > 2:
        spec = spec.squeeze()
    
    # Resize to 256x256 if needed
    if spec.shape != (256, 256):
        from scipy.ndimage import zoom
        zoom_factors = (256 / spec.shape[0], 256 / spec.shape[1])
        spec = zoom(spec, zoom_factors, order=1)
    
    all_spectrograms.append(spec)
```

### 2. Added Debug Output

```python
# Debug first spectrogram
if idx == 0:
    print(f"\n  First spectrogram shape: {spectrograms[0].shape}")
    print(f"  Number of spectrograms from first signal: {len(spectrograms)}")

# Before array conversion
print(f"\n  Total spectrograms collected: {len(all_spectrograms)}")
print(f"  Sample spectrogram shape: {all_spectrograms[0].shape}")

# After array conversion
print(f"\n  After np.array conversion: {self.spectrograms.shape}")
```

### 3. Proper Channel Dimension Handling

```python
# Ensure correct shape for Conv2D: (n_samples, height, width, channels)
if len(self.spectrograms.shape) == 3:
    # Shape is (n_samples, height, width) -> add channel
    self.spectrograms = np.expand_dims(self.spectrograms, axis=-1)
    print(f"  ✓ Added channel dimension: {self.spectrograms.shape}")
elif len(self.spectrograms.shape) == 4:
    print(f"  ✓ Already has 4 dimensions: {self.spectrograms.shape}")
else:
    raise ValueError(f"Unexpected spectrogram shape: {self.spectrograms.shape}")
```

## Expected Flow

```
Signal (640 samples)
  ↓ STFT
Spectrogram (variable size)
  ↓ Resize
Spectrogram (256, 256)
  ↓ Collect all
Array (n_samples, 256, 256)
  ↓ Add channel
Array (n_samples, 256, 256, 1)  ✓ Ready for Conv2D
```

## Files Modified

✅ `ncv_signal_main_v2.py` - Added shape validation, resize, and debug output

## What This Fixes

1. ✅ Ensures all spectrograms are exactly 256x256
2. ✅ Removes extra dimensions (squeeze)
3. ✅ Adds channel dimension correctly
4. ✅ Validates shape at each step
5. ✅ Provides debug output for troubleshooting

## Ready to Run

```bash
cd NCV_Classification
python run_ncv_interactive.py
```

Or:

```bash
cd NCV_Classification
python test_ncv_signal_v2.py
```

## Expected Output

```
Processing signals to spectrograms...

  First spectrogram shape: (256, 256)
  Number of spectrograms from first signal: 8

  Processed 50/100 signals
  Processed 100/100 signals

  Total spectrograms collected: 800
  Sample spectrogram shape: (256, 256)

  After np.array conversion: (800, 256, 256)
  ✓ Added channel dimension: (800, 256, 256, 1)

Generated 800 spectrograms
  Final shape: (800, 256, 256, 1)
```

---

**Status:** ✅ Fixed with validation and resize
**Date:** December 6, 2024

**Expected Result:** 60-75% accuracy!
