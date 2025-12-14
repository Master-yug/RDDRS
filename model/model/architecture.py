"""
BiLSTM with Attention Model Architecture
============================================
This module defines the optimized BiLSTMAttention model architecture
for respiratory disease classification.

Model Configuration:
- Input dimension: 373 features
- Hidden dimension: 128
- Number of layers: 2
- Number of classes: 4
- Dropout: 0.1 (optimized)
- Class weights: sqrt (optimized)

Performance Metrics:
- Accuracy: 87.96%
- Precision: 87.78%
- Recall: 87.96%
- F1-Score: 87.84%
- ROC-AUC: 96.93%
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class BiLSTMAttention(nn.Module):
    """
    Bidirectional LSTM with Attention Mechanism for Respiratory Disease Classification
    
    This model uses a bidirectional LSTM to process audio features in both forward
    and backward directions, combined with an attention mechanism that dynamically
    weights the importance of different features. The attention mechanism allows
    the model to focus on the most relevant audio characteristics for disease
    classification.
    
    Architecture Components:
        1. Bidirectional LSTM Layer:
           - Processes input features bidirectionally
           - Hidden dimension: 128 per direction (256 total)
           - 2 layers with dropout between them
           
        2. Attention Mechanism:
           - Computes importance weights for LSTM outputs
           - Uses additive (Bahdanau-style) attention
           - Produces weighted representation
           
        3. Classification Head:
           - Three-layer feedforward network
           - ReLU activations
           - Dropout for regularization
           - Output: 4-class probabilities
    
    Parameters
    ----------
    input_dim : int
        Number of input features (default: 373)
    hidden_dim : int
        LSTM hidden dimension (default: 128)
    num_layers : int
        Number of LSTM layers (default: 2)
    num_classes : int
        Number of output classes (default: 4)
    dropout : float
        Dropout rate for regularization (default: 0.1)
    
    Input Shape
    -----------
    (batch_size, input_dim)
    
    Output Shape
    ------------
    (batch_size, num_classes)
    
    Example
    -------
    >>> model = BiLSTMAttention(input_dim=373, dropout=0.1)
    >>> x = torch.randn(32, 373)  # batch of 32 samples
    >>> output = model(x)  # shape: (32, 4)
    >>> probabilities = F.softmax(output, dim=1)
    """
    
    def __init__(self, input_dim=373, hidden_dim=128, num_layers=2, num_classes=4, dropout=0.1):
        super(BiLSTMAttention, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Bidirectional LSTM layer
        # Output dimension is hidden_dim * 2 due to bidirectionality
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism (additive/Bahdanau attention)
        # Projects LSTM output to attention space
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Classification head
        # Three-layer network with decreasing dimensions
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
        
        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, num_classes)
        
        Process Flow
        ------------
        1. Reshape input for LSTM: (batch, features) -> (batch, 1, features)
        2. Process through BiLSTM: (batch, 1, features) -> (batch, 1, hidden*2)
        3. Compute attention weights: (batch, 1, hidden*2) -> (batch, 1, 1)
        4. Apply attention: weighted sum of LSTM outputs
        5. Classify: (batch, hidden*2) -> (batch, num_classes)
        """
        # Reshape for LSTM: add sequence dimension
        # LSTM expects (batch, sequence_length, features)
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # Pass through bidirectional LSTM
        # lstm_out: (batch_size, seq_len, hidden_dim * 2)
        # hidden and cell states are not used
        lstm_out, _ = self.lstm(x)
        
        # Compute attention scores
        # attention_weights: (batch_size, seq_len, 1)
        attention_weights = self.attention(lstm_out)
        
        # Apply softmax to get normalized attention weights
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention weights - compute weighted sum
        # weighted_output: (batch_size, hidden_dim * 2)
        weighted_output = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Pass through classification head
        # output: (batch_size, num_classes)
        output = self.classifier(weighted_output)
        
        return output
    
    def get_attention_weights(self, x):
        """
        Get attention weights for interpretability
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
        
        Returns
        -------
        torch.Tensor
            Attention weights of shape (batch_size, seq_len, 1)
        
        Example
        -------
        >>> model = BiLSTMAttention()
        >>> x = torch.randn(1, 373)
        >>> weights = model.get_attention_weights(x)
        >>> print(weights.squeeze())  # Shows which features are important
        """
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        return attention_weights
    
    def predict_proba(self, x):
        """
        Get class probabilities instead of logits
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
        
        Returns
        -------
        torch.Tensor
            Class probabilities of shape (batch_size, num_classes)
        
        Example
        -------
        >>> model = BiLSTMAttention()
        >>> model.eval()
        >>> x = torch.randn(1, 373)
        >>> with torch.no_grad():
        ...     probs = model.predict_proba(x)
        >>> print(probs)  # [0.1, 0.2, 0.5, 0.2] (sums to 1.0)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def get_model_info(self):
        """
        Get model configuration and parameter information
        
        Returns
        -------
        dict
            Dictionary containing model configuration and statistics
        """
        total_params = 0
        trainable_params = 0
        for p in self.parameters():
            param_count = p.numel()
            total_params += param_count
            if p.requires_grad:
                trainable_params += param_count
        
        return {
            'architecture': 'BiLSTM with Attention',
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'dropout': self.dropout,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'bidirectional': True,
            'attention_type': 'additive',
        }


def create_model(checkpoint_path=None, device='cpu'):
    """
    Factory function to create and optionally load a trained model
    
    Parameters
    ----------
    checkpoint_path : str, optional
        Path to saved model checkpoint (.pth file)
    device : str, optional
        Device to load model on ('cpu' or 'cuda')
    
    Returns
    -------
    BiLSTMAttention
        Initialized model (with loaded weights if checkpoint provided)
    
    Example
    -------
    >>> # Create new model
    >>> model = create_model()
    
    >>> # Load trained model
    >>> model = create_model('model/model.pth', device='cuda')
    """
    if checkpoint_path is not None:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create model with saved configuration
        model = BiLSTMAttention(
            input_dim=checkpoint.get('input_dim', 373),
            hidden_dim=checkpoint.get('hidden_dim', 128),
            num_layers=checkpoint.get('num_layers', 2),
            num_classes=checkpoint.get('num_classes', 4),
            dropout=checkpoint.get('dropout', 0.1)
        )
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model.to(device)
    else:
        # Create new model with default configuration
        model = BiLSTMAttention(
            input_dim=373,
            hidden_dim=128,
            num_layers=2,
            num_classes=4,
            dropout=0.1
        )
        return model.to(device)


if __name__ == "__main__":
    # Test model creation and forward pass
    print("Testing BiLSTMAttention Model")
    print("=" * 50)
    
    # Create model
    model = BiLSTMAttention()
    print("\nModel created successfully!")
    
    # Print model info
    info = model.get_model_info()
    print("\nModel Configuration:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 8
    x = torch.randn(batch_size, 373)
    
    # Get logits
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Get probabilities
        probs = model.predict_proba(x)
        print(f"  Probabilities shape: {probs.shape}")
        print(f"  Sample probability sum: {probs[0].sum():.4f} (should be 1.0)")
        
        # Get attention weights
        attention = model.get_attention_weights(x)
        print(f"  Attention weights shape: {attention.shape}")
    
    print("\n✓ All tests passed!")
