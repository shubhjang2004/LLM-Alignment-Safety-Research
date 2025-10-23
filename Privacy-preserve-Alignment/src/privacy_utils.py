

"""
Differential privacy utilities
"""

import torch
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

def make_private(model, optimizer, data_loader, epochs, target_epsilon, target_delta=1e-5, max_grad_norm=1.0):
    """
    Make a model differentially private using Opacus
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        data_loader: DataLoader
        epochs: Number of training epochs
        target_epsilon: Target privacy budget
        target_delta: Target delta for (ε, δ)-DP
        max_grad_norm: Maximum gradient norm for clipping
        
    Returns:
        private_model, private_optimizer, private_data_loader, privacy_engine
    """
    # Fix model to be compatible with Opacus
    model = ModuleValidator.fix(model)
    
    # Create privacy engine
    privacy_engine = PrivacyEngine()
    
    # Make private
    model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
    )
    
    print(f"   Privacy engine configured:")
    print(f"   Target ε: {target_epsilon}")
    print(f"   Target δ: {target_delta}")
    print(f"   Max grad norm: {max_grad_norm}")
    
    return model, optimizer, data_loader, privacy_engine

def get_privacy_spent(privacy_engine, delta=1e-5):
    """Get privacy budget spent during training"""
    epsilon = privacy_engine.get_epsilon(delta)
    return epsilon

class PrivacyMetrics:
    """Track privacy metrics during training"""
    
    def __init__(self, privacy_engine, delta=1e-5):
        self.privacy_engine = privacy_engine
        self.delta = delta
        self.epsilon_history = []
    
    def update(self):
        """Update privacy metrics"""
        epsilon = self.privacy_engine.get_epsilon(self.delta)
        self.epsilon_history.append(epsilon)
    
    def get_current_epsilon(self):
        """Get current epsilon"""
        return self.privacy_engine.get_epsilon(self.delta)
    
    def summary(self):
        """Get summary of privacy spent"""
        return {
            'final_epsilon': self.get_current_epsilon(),
            'delta': self.delta,
            'epsilon_history': self.epsilon_history
        }