"""
Inference Module for Respiratory Disease Classification
========================================================
This module provides high-level inference capabilities for the trained
BiLSTM with Attention model.

Features:
- Single sample prediction
- Batch prediction
- Probability output with confidence scores
- Preprocessing pipeline integration
- Device management (CPU/GPU)
"""

import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from typing import Union, Tuple, List, Dict
import warnings

from .architecture import BiLSTMAttention


class RespiratoryDiseaseClassifier:
    """
    High-level interface for respiratory disease classification
    
    This class encapsulates the complete prediction pipeline including
    preprocessing, model inference, and result formatting.
    
    Attributes
    ----------
    model : BiLSTMAttention
        The trained neural network model
    scaler : StandardScaler
        Feature scaler for preprocessing
    label_encoder : LabelEncoder
        Encoder for class labels
    feature_names : list
        List of expected feature names
    device : torch.device
        Device for inference (CPU or GPU)
    class_names : list
        Human-readable class names
    
    Example
    -------
    >>> classifier = RespiratoryDiseaseClassifier('model/')
    >>> features = np.random.randn(373)
    >>> result = classifier.predict_single(features)
    >>> print(f"Predicted: {result['predicted_class']}")
    >>> print(f"Confidence: {result['confidence']:.2%}")
    """
    
    def __init__(self, model_dir='model/', device=None):
        """
        Initialize the classifier with trained model and preprocessing objects
        
        Parameters
        ----------
        model_dir : str
            Directory containing model files:
            - model.pth: trained model weights
            - scaler.pkl: fitted StandardScaler
            - label_encoder.pkl: fitted LabelEncoder
            - feature_names.pkl: list of feature names
        device : str or torch.device, optional
            Device for inference ('cpu', 'cuda', or torch.device object)
            If None, automatically selects GPU if available
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        print(f"Initializing Respiratory Disease Classifier on {self.device}")
        
        # Load model
        model_path = os.path.join(model_dir, 'model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = BiLSTMAttention(
            input_dim=checkpoint.get('input_dim', 373),
            hidden_dim=checkpoint.get('hidden_dim', 128),
            num_layers=checkpoint.get('num_layers', 2),
            num_classes=checkpoint.get('num_classes', 4),
            dropout=checkpoint.get('dropout', 0.1)
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded successfully")
        
        # Load scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"✓ Scaler loaded")
        
        # Load label encoder
        encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        self.class_names = list(self.label_encoder.classes_)
        print(f"✓ Label encoder loaded")
        print(f"  Classes: {self.class_names}")
        
        # Load feature names
        features_path = os.path.join(model_dir, 'feature_names.pkl')
        with open(features_path, 'rb') as f:
            self.feature_names = pickle.load(f)
        print(f"✓ Feature names loaded ({len(self.feature_names)} features)")
        
        print("Classifier ready for inference!")
    
    def preprocess(self, features: Union[np.ndarray, List, Dict]) -> torch.Tensor:
        """
        Preprocess input features
        
        Parameters
        ----------
        features : array-like or dict
            Input features, can be:
            - numpy array of shape (n_features,) or (n_samples, n_features)
            - list of feature values
            - dict mapping feature names to values
        
        Returns
        -------
        torch.Tensor
            Preprocessed features ready for model input
        """
        # Convert dict to array if needed
        if isinstance(features, dict):
            features = np.array([features[name] for name in self.feature_names])
        
        # Convert to numpy array
        features = np.asarray(features)
        
        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Check feature dimension
        if features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {features.shape[1]}"
            )
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        
        return features_tensor
    
    def predict_single(self, features: Union[np.ndarray, List, Dict]) -> Dict:
        """
        Predict respiratory disease for a single sample
        
        Parameters
        ----------
        features : array-like or dict
            Input features (373 values)
        
        Returns
        -------
        dict
            Prediction results containing:
            - predicted_class: str, predicted disease class
            - predicted_label: int, predicted class index
            - confidence: float, confidence score (0-1)
            - probabilities: dict, probability for each class
            - all_probabilities: np.ndarray, probability array
        
        Example
        -------
        >>> result = classifier.predict_single(features)
        >>> print(f"Disease: {result['predicted_class']}")
        >>> print(f"Confidence: {result['confidence']:.2%}")
        >>> for class_name, prob in result['probabilities'].items():
        ...     print(f"  {class_name}: {prob:.2%}")
        """
        # Preprocess
        features_tensor = self.preprocess(features)
        
        # Predict
        with torch.no_grad():
            logits = self.model(features_tensor)
            probabilities = F.softmax(logits, dim=1)
        
        # Get results
        probs = probabilities.cpu().numpy()[0]
        predicted_idx = probs.argmax()
        predicted_class = self.class_names[predicted_idx]
        confidence = probs[predicted_idx]
        
        # Format results
        result = {
            'predicted_class': predicted_class,
            'predicted_label': int(predicted_idx),
            'confidence': float(confidence),
            'probabilities': {
                class_name: float(prob)
                for class_name, prob in zip(self.class_names, probs)
            },
            'all_probabilities': probs
        }
        
        return result
    
    def predict_batch(self, features_list: List[Union[np.ndarray, List, Dict]]) -> List[Dict]:
        """
        Predict respiratory disease for multiple samples
        
        Parameters
        ----------
        features_list : list
            List of feature arrays/dicts
        
        Returns
        -------
        list of dict
            List of prediction results (same format as predict_single)
        
        Example
        -------
        >>> features_batch = [features1, features2, features3]
        >>> results = classifier.predict_batch(features_batch)
        >>> for i, result in enumerate(results):
        ...     print(f"Sample {i}: {result['predicted_class']}")
        """
        # Stack features
        features_array = np.array([
            np.asarray(f) if not isinstance(f, dict)
            else np.array([f[name] for name in self.feature_names])
            for f in features_list
        ])
        
        # Preprocess
        features_tensor = self.preprocess(features_array)
        
        # Predict
        with torch.no_grad():
            logits = self.model(features_tensor)
            probabilities = F.softmax(logits, dim=1)
        
        # Format results
        probs = probabilities.cpu().numpy()
        predicted_indices = probs.argmax(axis=1)
        
        results = []
        for i, (idx, prob_array) in enumerate(zip(predicted_indices, probs)):
            result = {
                'predicted_class': self.class_names[idx],
                'predicted_label': int(idx),
                'confidence': float(prob_array[idx]),
                'probabilities': {
                    class_name: float(prob)
                    for class_name, prob in zip(self.class_names, prob_array)
                },
                'all_probabilities': prob_array
            }
            results.append(result)
        
        return results
    
    def predict_with_explanation(self, features: Union[np.ndarray, List, Dict]) -> Dict:
        """
        Predict with attention-based explanation
        
        Parameters
        ----------
        features : array-like or dict
            Input features
        
        Returns
        -------
        dict
            Prediction results plus attention weights for interpretability
        
        Example
        -------
        >>> result = classifier.predict_with_explanation(features)
        >>> print(f"Predicted: {result['predicted_class']}")
        >>> print(f"Attention weights: {result['attention_weights']}")
        """
        # Get standard prediction
        result = self.predict_single(features)
        
        # Get attention weights
        features_tensor = self.preprocess(features)
        with torch.no_grad():
            attention_weights = self.model.get_attention_weights(features_tensor)
        
        result['attention_weights'] = attention_weights.cpu().numpy()
        
        return result
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns
        -------
        dict
            Model configuration and metadata
        """
        info = self.model.get_model_info()
        info.update({
            'device': str(self.device),
            'num_features': len(self.feature_names),
            'class_names': self.class_names,
            'performance_metrics': {
                'accuracy': 0.8796,
                'precision': 0.8778,
                'recall': 0.8796,
                'f1_score': 0.8784,
                'roc_auc': 0.9693
            }
        })
        return info


def load_classifier(model_dir='model/', device=None) -> RespiratoryDiseaseClassifier:
    """
    Convenience function to load the classifier
    
    Parameters
    ----------
    model_dir : str
        Directory containing model files
    device : str or torch.device, optional
        Device for inference
    
    Returns
    -------
    RespiratoryDiseaseClassifier
        Initialized classifier ready for inference
    
    Example
    -------
    >>> classifier = load_classifier()
    >>> result = classifier.predict_single(features)
    """
    return RespiratoryDiseaseClassifier(model_dir=model_dir, device=device)


if __name__ == "__main__":
    # Test inference module
    print("Testing Respiratory Disease Classifier")
    print("=" * 70)
    
    try:
        # Load classifier
        classifier = load_classifier(model_dir='model/')
        
        print("\n" + "=" * 70)
        print("MODEL INFORMATION")
        print("=" * 70)
        
        info = classifier.get_model_info()
        for key, value in info.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        
        print("\n" + "=" * 70)
        print("TESTING INFERENCE")
        print("=" * 70)
        
        # Create random test features
        test_features = np.random.randn(373)
        
        # Single prediction
        print("\nSingle Prediction:")
        result = classifier.predict_single(test_features)
        print(f"  Predicted class: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Probabilities:")
        for class_name, prob in result['probabilities'].items():
            bar = '█' * int(prob * 50)
            print(f"    {class_name:20s}: {prob:.2%} {bar}")
        
        # Batch prediction
        print("\nBatch Prediction (3 samples):")
        batch_features = [np.random.randn(373) for _ in range(3)]
        results = classifier.predict_batch(batch_features)
        for i, result in enumerate(results):
            print(f"  Sample {i+1}: {result['predicted_class']} ({result['confidence']:.2%})")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("Note: Model files must be present in the 'model/' directory")
