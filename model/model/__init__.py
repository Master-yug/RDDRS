"""
Model Package for Respiratory Disease Classification
=====================================================

This package provides the BiLSTM with Attention model and inference
capabilities for respiratory disease classification.

Modules:
- architecture: Model definition and architecture
- inference: High-level prediction interface

Quick Start:
>>> from model import load_classifier
>>> classifier = load_classifier()
>>> result = classifier.predict_single(features)
>>> print(result['predicted_class'])
"""

from .architecture import BiLSTMAttention, create_model
from .inference import RespiratoryDiseaseClassifier, load_classifier

__version__ = '1.0.0'
__all__ = [
    'BiLSTMAttention',
    'create_model',
    'RespiratoryDiseaseClassifier',
    'load_classifier'
]
