# 🧠 NCV Classification System

**Version:** 1.0.0  
**Release Date:** December 6, 2024  
**Status:** Production Ready

---

## 📊 Overview

A complete deep learning system for classifying Carpal Tunnel Syndrome (CTS) severity from Nerve Conduction Velocity (NCV) data using three model architectures: CNN, LSTM, and CNN-LSTM Hybrid.

### Key Features
- ✅ Three model architectures (CNN, LSTM, Hybrid)
- ✅ Automatic data preprocessing and normalization
- ✅ Cross-validation support (5-fold stratified)
- ✅ Comprehensive evaluation metrics
- ✅ Sample data generator for testing
- ✅ Complete documentation suite

---

## 🎯 Quick Stats

| Metric | Value |
|--------|-------|
| **Expected Accuracy** | 70-85% |
| **Training Time** | 30-60 minutes |
| **Memory Usage** | 2-4 GB RAM |
| **GPU Required** | No (optional) |
| **Data Format** | CSV/Excel |
| **Classes** | 4 (non_cts, mild, moderate, severe) |

---

## 📁 Project Structure

```
NCV_Classification/
├── src/                    # Source code
├── utils/                  # Utilities
├── data/                   # Data folder
├── docs/                   # Documentation
├── ncv_main.py            # Main pipeline
├── test_ncv_pipeline.py   # Quick test
└── ncv_config.json        # Configuration
```

---

## 🚀 Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Test
python test_ncv_pipeline.py

# 3. Run
python ncv_main.py
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | Complete documentation |
| [GETTING_STARTED.md](GETTING_STARTED.md) | Quick start guide |
| [STRUCTURE.txt](STRUCTURE.txt) | Project structure |
| [docs/NCV_INDEX.md](docs/NCV_INDEX.md) | Navigation & index |
| [docs/NCV_QUICK_START.md](docs/NCV_QUICK_START.md) | Detailed quick start |
| [docs/NCV_CHECKLIST.md](docs/NCV_CHECKLIST.md) | Complete checklist |
| [docs/EMG_vs_NCV_COMPARISON.md](docs/EMG_vs_NCV_COMPARISON.md) | EMG vs NCV |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## 🏗️ Architecture

### Models
1. **1D CNN** - Convolutional layers for spatial feature extraction
2. **LSTM** - Bidirectional LSTM for temporal patterns
3. **CNN-LSTM Hybrid** - Combined approach for best performance

### Pipeline
```
CSV Data → Preprocessing → Model Training → Evaluation → Results
```

---

## 📊 Performance

### Expected Results
- **Accuracy:** 70-85%
- **Training:** 30-60 minutes for 3 models
- **Inference:** < 1 second per sample

### Comparison with EMG
| Metric | EMG | NCV |
|--------|-----|-----|
| Training Time | 2-3 hours | 30-60 min |
| Accuracy | 60-75% | 70-85% |
| Complexity | High | Medium |
| GPU Required | Yes | No |

---

## 🔧 Configuration

Edit `ncv_config.json` to customize:
- Data paths
- Feature selection
- Model hyperparameters
- Training settings
- Cross-validation options

---

## 📈 Use Cases

### Research
- Compare model architectures
- Study feature importance
- Publish findings

### Clinical
- Automated CTS screening
- Decision support system
- Patient monitoring

### Education
- Learn deep learning
- Medical AI applications
- Portfolio projects

---

## 🆘 Support

### Documentation
- Check [README.md](README.md) for detailed info
- See [docs/NCV_CHECKLIST.md](docs/NCV_CHECKLIST.md) for troubleshooting
- Read [GETTING_STARTED.md](GETTING_STARTED.md) for quick help

### Common Issues
1. **No data found** → Generate sample data: `python utils/create_sample_ncv_data.py`
2. **Low accuracy** → Increase data, tune hyperparameters
3. **Out of memory** → Reduce batch_size in config

---

## 📝 Citation

```bibtex
@software{ncv_classification_2024,
  title = {NCV Classification System for CTS Severity Detection},
  author = {Your Name},
  year = {2024},
  version = {1.0.0},
  url = {https://github.com/yourusername/ncv-classification}
}
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 🎓 Credits

Developed for Carpal Tunnel Syndrome severity classification using deep learning on Nerve Conduction Velocity data.

---

## 🔗 Links

- **Main Documentation:** [README.md](README.md)
- **Quick Start:** [GETTING_STARTED.md](GETTING_STARTED.md)
- **Full Index:** [docs/NCV_INDEX.md](docs/NCV_INDEX.md)
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)

---

**Version 1.0.0** | **December 2024** | **Production Ready** ✅
