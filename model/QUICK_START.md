# Quick Start Guide - finalv3p2

Get started with the respiratory disease classification model in under 5 minutes!

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

## Installation

### Step 1: Navigate to the Directory

```bash
cd COPILOT/finalv3p2
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs only 3 packages:
- `torch` - PyTorch deep learning framework
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning utilities

## Basic Usage

### Example 1: Single Prediction

```python
from model import load_classifier
import numpy as np

# Load the classifier
classifier = load_classifier(model_dir='model/')

# Prepare your features (373 values)
# Replace this with your actual audio features
features = np.random.randn(373)

# Make prediction
result = classifier.predict_single(features)

# Print results
print(f"Predicted Disease: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
print("\nAll Probabilities:")
for disease, probability in result['probabilities'].items():
    print(f"  {disease}: {probability:.2%}")
```

**Output:**
```
Predicted Disease: COVID-19
Confidence: 85.42%

All Probabilities:
  COVID-19: 85.42%
  healthy_cough: 8.23%
  infection: 5.12%
  obstructive_disease: 1.23%
```

### Example 2: Batch Prediction

```python
from model import load_classifier
import numpy as np

# Load classifier
classifier = load_classifier()

# Multiple samples
samples = [
    np.random.randn(373),  # Sample 1
    np.random.randn(373),  # Sample 2
    np.random.randn(373),  # Sample 3
]

# Batch prediction (more efficient)
results = classifier.predict_batch(samples)

# Print results
for i, result in enumerate(results, 1):
    print(f"Sample {i}: {result['predicted_class']} ({result['confidence']:.1%})")
```

**Output:**
```
Sample 1: COVID-19 (85.4%)
Sample 2: healthy_cough (72.3%)
Sample 3: infection (68.9%)
```

### Example 3: Using the Example Script

```bash
python example_usage.py
```

This runs a comprehensive demo showing:
- Model loading
- Model information
- Single prediction
- Batch prediction
- Prediction with attention weights

## Model Information

### Get Model Details

```python
from model import load_classifier

classifier = load_classifier()
info = classifier.get_model_info()

print(f"Architecture: {info['architecture']}")
print(f"Parameters: {info['total_parameters']:,}")
print(f"Classes: {', '.join(info['class_names'])}")
print(f"Accuracy: {info['performance_metrics']['accuracy']:.2%}")
```

### Disease Classes

The model classifies into 4 categories:

1. **COVID-19** - SARS-CoV-2 viral infection
2. **healthy_cough** - Normal respiratory function
3. **infection** - General respiratory infections
4. **obstructive_disease** - Chronic obstructive conditions (COPD, asthma)

## Performance Metrics

- **Accuracy**: 87.96%
- **Precision**: 87.78%
- **Recall**: 87.96%
- **F1-Score**: 87.84%
- **ROC-AUC**: 96.93%

## Input Requirements

### Feature Format

The model expects **373 audio features** extracted from cough recordings:

- **Temporal Features** (3): duration, silence_ratio, zero_crossing_rate
- **MFCC Features** (80): 20 MFCCs × 4 statistics (mean, std, min, max)
- **Spectral Features** (290): spectral centroid, bandwidth, rolloff, contrast, chroma, etc.

### Input Formats Supported

**1. NumPy Array** (recommended):
```python
features = np.array([0.5, 0.2, 0.1, ...])  # 373 values
```

**2. Python List**:
```python
features = [0.5, 0.2, 0.1, ...]  # 373 values
```

**3. Dictionary** (with feature names):
```python
features = {
    'duration': 0.5,
    'silence_ratio': 0.2,
    'zero_crossing_rate_mean': 0.1,
    # ... 370 more features
}
```

## Advanced Usage

### Device Selection (CPU/GPU)

**Automatic** (default):
```python
classifier = load_classifier()  # Uses GPU if available
```

**Force CPU**:
```python
classifier = load_classifier(device='cpu')
```

**Force GPU**:
```python
classifier = load_classifier(device='cuda')
```

### Model Interpretability

Get attention weights to understand which features the model focuses on:

```python
result = classifier.predict_with_explanation(features)

print(f"Prediction: {result['predicted_class']}")
print(f"Attention shape: {result['attention_weights'].shape}")
# Use attention weights for feature importance analysis
```

### Direct Model Access

For advanced users who want full control:

```python
from model import create_model
import torch

# Load model
model = create_model('model/model.pth', device='cpu')
model.eval()

# Prepare input tensor
x = torch.randn(1, 373)  # Batch of 1

# Forward pass
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    
print(f"Probabilities: {probs}")
```

## Integration Examples

### Flask Web App

```python
from flask import Flask, request, jsonify
from model import load_classifier
import numpy as np

app = Flask(__name__)
classifier = load_classifier()

@app.route('/predict', methods=['POST'])
def predict():
    features = np.array(request.json['features'])
    result = classifier.predict_single(features)
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000)
```

### Jupyter Notebook

```python
# Cell 1: Import and load
from model import load_classifier
classifier = load_classifier()

# Cell 2: Make predictions
features = np.random.randn(373)
result = classifier.predict_single(features)

# Cell 3: Visualize
import matplotlib.pyplot as plt
plt.bar(result['probabilities'].keys(), 
        result['probabilities'].values())
plt.xticks(rotation=45)
plt.title('Disease Probabilities')
plt.show()
```

### Command Line Script

```python
#!/usr/bin/env python
import sys
import numpy as np
from model import load_classifier

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <features_file.npy>")
        sys.exit(1)
    
    # Load features from file
    features = np.load(sys.argv[1])
    
    # Load classifier and predict
    classifier = load_classifier()
    result = classifier.predict_single(features)
    
    # Output
    print(f"Disease: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")

if __name__ == '__main__':
    main()
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

### Issue: "FileNotFoundError: Model file not found"
**Solution**: Ensure you're in the correct directory:
```bash
cd COPILOT/finalv3p2
python your_script.py
```

### Issue: "Expected 373 features, got X"
**Solution**: Verify your feature array has exactly 373 values:
```python
print(f"Number of features: {len(features)}")
```

### Issue: "CUDA out of memory"
**Solution**: Force CPU usage:
```python
classifier = load_classifier(device='cpu')
```

## Next Steps

1. **Extract Audio Features**: Use audio processing libraries (librosa, pyAudioAnalysis) to extract features from cough recordings
2. **Integrate into Application**: Use the examples above to integrate the model into your application
3. **Batch Processing**: For multiple samples, use `predict_batch()` for better performance
4. **Custom Pipeline**: Build your own preprocessing and postprocessing pipelines

## Additional Resources

- **README.md** - Comprehensive documentation
- **EXTRACTION_SUMMARY.md** - Details on what's included/excluded
- **example_usage.py** - Complete working examples
- **model/architecture.py** - Model architecture details
- **model/inference.py** - Inference implementation

## Support

For questions or issues:
- Check the README.md for detailed documentation
- Review example_usage.py for working code
- See EXTRACTION_SUMMARY.md for architecture details

---

**Happy Coding!** 🚀

*Last Updated: December 14, 2025*
