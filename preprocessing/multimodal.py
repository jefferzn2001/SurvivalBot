import torch
import torch.nn as nn
from typing import Tuple, Optional, Any
import logging

# Import modality encoders
from .vision_encoder import VisionEncoder
from .audio_encoder import AudioEncoder

logger = logging.getLogger(__name__)

class MultiModalEncoder(nn.Module):
    """
    Multi-modal encoder for processing different data types:
    - Camera images - CNN encoder + normalization  
    - Audio data - CNN encoder + normalization
    - IMU/sensor data - linear projection + normalization
    
    NOTE: Currently only vision and audio encoders are implemented.
    """
    
    def __init__(self, device: str = 'cuda'):
        super().__init__()
        self.device = device
        
        # Initialize encoders
        self.vision_encoder = VisionEncoder(device=device)
        self.audio_encoder = AudioEncoder(device=device)
        
        # Get feature dimensions
        self.vision_dim = self.vision_encoder.get_feature_dim()
        self.audio_dim = self.audio_encoder.get_feature_dim()
        
        # Projection to unified dimension
        projection_dim = 512
        self.vision_projection = nn.Linear(self.vision_dim, projection_dim).to(device)
        self.audio_projection = nn.Linear(self.audio_dim, projection_dim).to(device)
        
        # Normalization layers
        self.vision_norm = nn.LayerNorm(projection_dim).to(device)
        self.audio_norm = nn.LayerNorm(projection_dim).to(device)
        
        logger.info(f"MultiModalEncoder initialized:")
        logger.info(f"  Vision: {self.vision_dim} -> {projection_dim}")
        logger.info(f"  Audio: {self.audio_dim} -> {projection_dim}")
        logger.info(f"  Device: {device}")
        
        self.to(device)
    
    def process_vision(self, image: torch.Tensor) -> torch.Tensor:
        """
        Process camera image data.
        
        Args:
            image: Camera image tensor
            
        Returns:
            torch.Tensor: Encoded vision features
        """
        return self.vision_encoder(image)
    
    def process_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Process audio data.
        
        Args:
            audio: Audio tensor
            
        Returns:
            torch.Tensor: Encoded audio features
        """
        return self.audio_encoder(audio)
    
    def forward(self, 
                vision_data: Optional[torch.Tensor] = None,
                audio_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through multi-modal encoder.
        
        Args:
            vision_data: Optional camera image
            audio_data: Optional audio data
            
        Returns:
            torch.Tensor: Concatenated multi-modal features
        """
        features = []
        
        if vision_data is not None:
            vision_features = self.process_vision(vision_data)
            vision_proj = self.vision_projection(vision_features)
            vision_norm = self.vision_norm(vision_proj)
            features.append(vision_norm)
        
        if audio_data is not None:
            audio_features = self.process_audio(audio_data)
            audio_proj = self.audio_projection(audio_features)
            audio_norm = self.audio_norm(audio_proj)
            features.append(audio_norm)
        
        if not features:
            raise ValueError("At least one modality must be provided")
        
        # Concatenate all features
        return torch.cat(features, dim=-1)
    
    def get_feature_dim(self) -> int:
        """
        Get total feature dimension.
        
        Returns:
            int: Total feature dimension
        """
        # Currently vision (512) + audio (512) when both present
        return 1024  # Max possible when all modalities present
    
    def get_trainable_parameters(self):
        """
        Get all trainable parameters for optimization.
        
        Returns:
            List of trainable parameters
        """
        trainable_params = []
        
        # Vision encoder parameters
        trainable_params.extend(self.vision_encoder.parameters())
        trainable_params.extend(self.vision_projection.parameters())
        trainable_params.extend(self.vision_norm.parameters())
        
        # Audio encoder parameters  
        trainable_params.extend(self.audio_encoder.parameters())
        trainable_params.extend(self.audio_projection.parameters())
        trainable_params.extend(self.audio_norm.parameters())
        
        return trainable_params

# Data structure for multi-modal input
class MultiModalInput:
    def __init__(self, 
                 vision_data: Optional[torch.Tensor] = None,
                 audio_data: Optional[torch.Tensor] = None):
        """
        Multi-modal input container.
        
        Args:
            vision_data: Camera image tensor
            audio_data: Audio tensor
        """
        self.vision_data = vision_data
        self.audio_data = audio_data
    
    def to(self, device: str):
        """Move all data to specified device."""
        if self.vision_data is not None:
            self.vision_data = self.vision_data.to(device)
        if self.audio_data is not None:
            self.audio_data = self.audio_data.to(device)
        return self
    
    def has_vision(self) -> bool:
        return self.vision_data is not None
    
    def has_audio(self) -> bool:
        return self.audio_data is not None 