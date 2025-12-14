"""
Example Usage Script for Respiratory Disease Classification Model

This script demonstrates how to use the model for predictions.
"""

from model import load_classifier
import numpy as np


def main():
    print("=" * 70)
    print("Respiratory Disease Classification - Example Usage")
    print("=" * 70)
    
    # Load the classifier
    print("\n1. Loading classifier...")
    classifier = load_classifier(model_dir='model/', device='cpu')
    print("   ✓ Classifier loaded successfully")
    
    # Get model information
    print("\n2. Model Information:")
    info = classifier.get_model_info()
    print(f"   Architecture: {info['architecture']}")
    print(f"   Input features: {info['num_features']}")
    print(f"   Classes: {', '.join(info['class_names'])}")
    print(f"   Total parameters: {info['total_parameters']:,}")
    print(f"   Device: {info['device']}")
    
    # Single prediction example
    print("\n3. Single Prediction Example:")
    print("   Generating random features (373 values)...")
    features = np.random.randn(373)
    
    result = classifier.predict_single(features)
    print(f"   Predicted class: {result['predicted_class']}")
    print(f"   Confidence: {result['confidence']:.2%}")
    print("   Probabilities:")
    for class_name, prob in result['probabilities'].items():
        bar = '█' * int(prob * 40)
        print(f"     {class_name:20s}: {prob:6.2%} {bar}")
    
    # Batch prediction example
    print("\n4. Batch Prediction Example (5 samples):")
    batch_features = [np.random.randn(373) for _ in range(5)]
    results = classifier.predict_batch(batch_features)
    
    for i, result in enumerate(results):
        print(f"   Sample {i+1}: {result['predicted_class']:20s} (confidence: {result['confidence']:.2%})")
    
    # Model with explanation
    print("\n5. Prediction with Attention Explanation:")
    result_with_attention = classifier.predict_with_explanation(features)
    print(f"   Predicted class: {result_with_attention['predicted_class']}")
    print(f"   Confidence: {result_with_attention['confidence']:.2%}")
    print(f"   Attention weights shape: {result_with_attention['attention_weights'].shape}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nNote: Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        print("\nAnd that model files are present in the 'model/' directory.")
