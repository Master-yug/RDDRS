# Respiratory Disease Classification Model

**Model-Only Package (No Deployment Infrastructure)**

This directory contains only the model code and weights for the BiLSTM with Attention respiratory disease classifier, without any deployment-related infrastructure (API, Docker, scripts, etc.).

## Contents

- `model/` - Model package directory
  - `architecture.py` - BiLSTM with Attention model architecture
  - `inference.py` - High-level inference interface
  - `__init__.py` - Package initialization
  - `model.pth` - Trained model weights (3.8 MB)
  - `scaler.pkl` - StandardScaler for feature preprocessing
  - `label_encoder.pkl` - Label encoder for class names
  - `feature_names.pkl` - List of 373 feature names
- `requirements.txt` - Minimal dependencies (torch, numpy, scikit-learn only)

## Model Performance

- **Accuracy**: 87.96%
- **Precision**: 87.78%
- **Recall**: 87.96%
- **F1-Score**: 87.84%
- **ROC-AUC**: 96.93%

## Classes

1. **COVID-19** - SARS-CoV-2 infection
2. **healthy_cough** - Normal respiratory function
3. **infection** - General respiratory infections
4. **obstructive_disease** - Chronic obstructive conditions

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Inference

```python
from model import load_classifier
import numpy as np

# Load the classifier
classifier = load_classifier(model_dir='model/')

# Prepare features (373 values)
features = np.random.randn(373)  # Replace with actual features

# Make prediction
result = classifier.predict_single(features)

print(f"Predicted class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
print("\nProbabilities:")
for class_name, prob in result['probabilities'].items():
    print(f"  {class_name}: {prob:.2%}")
```

### Batch Prediction

```python
# Predict for multiple samples
features_list = [np.random.randn(373) for _ in range(10)]
results = classifier.predict_batch(features_list)

for i, result in enumerate(results):
    print(f"Sample {i+1}: {result['predicted_class']} ({result['confidence']:.2%})")
```

### Get Model Information

```python
info = classifier.get_model_info()
print(f"Architecture: {info['architecture']}")
print(f"Total parameters: {info['total_parameters']:,}")
print(f"Device: {info['device']}")
```

## Model Architecture

```
Input (373 features)
    │
    ▼
Bidirectional LSTM (2 layers, hidden_dim=128)
    │
    ▼
Attention Mechanism (additive)
    │
    ▼
Classification Head (256→128→64→4)
    │
    ▼
Output (4 classes)
```

### Architecture Details

- **Input dimension**: 373 audio features
- **Hidden dimension**: 128 per direction (256 total)
- **LSTM layers**: 2 with dropout
- **Attention**: Additive (Bahdanau-style)
- **Dropout**: 0.1
- **Total parameters**: ~470,725

## Input Specifications

- **Features**: 373 audio features (MFCCs, spectral features, temporal features)
- **Data type**: Float32
- **Preprocessing**: StandardScaler (zero mean, unit variance)
- **Format**: NumPy array, Python list, or dictionary

## Output Format

```python
{
    'predicted_class': 'COVID-19',           # Predicted disease class
    'predicted_label': 0,                    # Class index (0-3)
    'confidence': 0.8542,                    # Confidence score (0-1)
    'probabilities': {                       # All class probabilities
        'COVID-19': 0.8542,
        'healthy_cough': 0.0823,
        'infection': 0.0512,
        'obstructive_disease': 0.0123
    },
    'all_probabilities': array([...])       # NumPy array of probabilities
}
```

## Device Support

The model automatically detects and uses GPU if available (CUDA), otherwise falls back to CPU.

```python
# Force CPU
classifier = load_classifier(model_dir='model/', device='cpu')

# Force GPU
classifier = load_classifier(model_dir='model/', device='cuda')
```

## Advanced Usage

### Direct Model Access

```python
from model import create_model
import torch

# Load model directly
model = create_model('model/model.pth', device='cuda')
model.eval()

# Forward pass
x = torch.randn(1, 373)
with torch.no_grad():
    logits = model(x)
    probabilities = torch.softmax(logits, dim=1)
```

### Attention Weights (Model Interpretability)

```python
result = classifier.predict_with_explanation(features)
print(f"Prediction: {result['predicted_class']}")
print(f"Attention weights shape: {result['attention_weights'].shape}")
```

## Testing the Model

```bash
# Test from model directory
cd model
python architecture.py  # Test architecture
python inference.py     # Test inference
```

## Notes

- This package contains **only** the model code and weights
- No API server, Docker files, or deployment scripts included
- For deployment-ready version, see `../finalv3/`
- Model files are required and must be in the `model/` directory
- Ensure all 373 features are provided in the correct order

## Related Documentation

For complete deployment infrastructure including:
- FastAPI REST API
- Docker containerization  
- Production deployment guides
- API documentation

See the parent directory: `../finalv3/`

## Requirements

- Python 3.9+
- PyTorch 1.9+
- NumPy 1.21+
- scikit-learn 1.0+

## License

MIT License

---

**Last Updated**: December 14, 2025  
**Model Version**: 1.0.0  
**Purpose**: Model code only (no deployment infrastructure)
