"""
Custom CNN Feature Extractor for Quoridor
Processes the 9x9x6 board state into a feature vector for PPO
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class QuoridorCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for the Quoridor 9x9x6 observation.
    
    Architecture:
        Conv2d(6 -> 32, 3x3, padding=1) -> ReLU
        Conv2d(32 -> 64, 3x3, padding=1) -> ReLU  
        Conv2d(64 -> 64, 3x3, padding=1) -> ReLU
        Flatten -> Linear(64 * 9 * 9 -> 256) -> ReLU
    
    Output: 256-dimensional feature vector
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Input shape: (batch, 9, 9, 6) -> need to transpose to (batch, 6, 9, 9)
        n_input_channels = observation_space.shape[2]  # 6 channels
        
        self.cnn = nn.Sequential(
            # Layer 1: 6 -> 32 channels
            nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Layer 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Layer 3: 64 -> 64 channels
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Flatten: 64 * 9 * 9 = 5184
            nn.Flatten(),
        )
        
        # Compute flattened size
        with torch.no_grad():
            # Sample input (channels last -> channels first)
            sample = torch.zeros(1, n_input_channels, 9, 9)
            n_flatten = self.cnn(sample).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Process observations through CNN.
        
        Args:
            observations: Tensor of shape (batch, 9, 9, 6) - channels last
            
        Returns:
            Feature tensor of shape (batch, 256)
        """
        # Transpose from (batch, H, W, C) to (batch, C, H, W)
        x = observations.permute(0, 3, 1, 2)
        
        # Pass through CNN and linear layers
        return self.linear(self.cnn(x))


class QuoridorResidualCNN(BaseFeaturesExtractor):
    """
    Alternative CNN with residual connections for deeper feature extraction.
    Can be useful for learning more complex board patterns.
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[2]
        
        # Initial conv to expand channels
        self.initial = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # Residual blocks
        self.res_block1 = self._make_residual_block(64, 64)
        self.res_block2 = self._make_residual_block(64, 64)
        
        # Final processing
        self.final = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),  # Reduce to 3x3
            nn.Flatten(),
        )
        
        self.linear = nn.Sequential(
            nn.Linear(128 * 3 * 3, features_dim),
            nn.ReLU(),
        )
    
    def _make_residual_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a residual block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Transpose from (batch, H, W, C) to (batch, C, H, W)
        x = observations.permute(0, 3, 1, 2)
        
        # Initial conv
        x = self.initial(x)
        
        # Residual blocks
        identity = x
        x = self.res_block1(x)
        x = torch.relu(x + identity)
        
        identity = x
        x = self.res_block2(x)
        x = torch.relu(x + identity)
        
        # Final processing
        x = self.final(x)
        return self.linear(x)
