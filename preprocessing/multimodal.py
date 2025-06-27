import torch
import torch.nn as nn
import logging
from typing import Optional, List, Union, Dict, Any
from PIL import Image

from .vision_encoder import VisionEncoder
from .audio_encoder import AudioEncoder
from .ros_preprocessor import ROSPreprocessor

logger = logging.getLogger(__name__)

class MultiModalPreprocessor(nn.Module):
    """
    Combined preprocessor for handling multiple input modalities:
    - Vision (CLIP) - frozen encoder + trainable projection + normalization
    - Audio (CLAP) - frozen encoder + trainable projection + normalization
    - ROS2 messages - trainable projection + normalization (skipped for now)
    """
    
    def __init__(self, device="cuda", projection_vision=64, projection_audio=16, use_normalization=True):
        super().__init__()
        self.device = device
        self.projection_vision = projection_vision
        self.projection_audio = projection_audio
        self.use_normalization = use_normalization
        
        logger.info("Initializing MultiModal Preprocessor")
        
        # Initialize encoders (these will be frozen)
        self.vision_encoder = VisionEncoder(device=device)
        self.audio_encoder = AudioEncoder(device=device)
        # self.ros_preprocessor = ROSPreprocessor(device=device)  # Skipped for now
        
        # Initialize trainable projection layers
        self.vision_projection = nn.Linear(512, projection_vision)  # CLIP outputs 512
        self.audio_projection = nn.Linear(512, projection_audio)   # CLAP outputs 512
        
        # Initialize normalization layers (optional)
        if use_normalization:
            self.vision_norm = nn.LayerNorm(projection_vision)
            self.audio_norm = nn.LayerNorm(projection_audio)
            # self.ros_norm = nn.LayerNorm(projection_dim)  # Skipped for now
        
        # Freeze the encoders but keep projections trainable
        self._freeze_encoders()
        
        logger.info("MultiModal Preprocessor initialized successfully")
    
    def _freeze_encoders(self):
        """Freeze encoder parameters to prevent gradient flow."""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        logger.info("Encoders frozen for feature extraction only")
    
    def process_vision(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """
        Process vision input through CLIP + projection + normalization.
        
        Args:
            images: Single PIL image or list of PIL images
            
        Returns:
            torch.Tensor: Projected and normalized vision features (B, projection_dim)
        """
        with torch.no_grad():  # No gradients through frozen encoder
            features = self.vision_encoder(images)
        
        # Gradients can flow through projection and normalization
        projected = self.vision_projection(features)
        
        if self.use_normalization:
            projected = self.vision_norm(projected)
        
        return projected
    
    def process_audio(self, audio: Any) -> torch.Tensor:
        """
        Process audio input through CLAP + projection + normalization.
        
        Args:
            audio: Audio input (torch tensor or list of torch tensors)
            
        Returns:
            torch.Tensor: Projected and normalized audio features (B, projection_dim)
        """
        with torch.no_grad():  # No gradients through frozen encoder
            features = self.audio_encoder(audio)
        
        # Gradients can flow through projection and normalization
        projected = self.audio_projection(features)
        
        if self.use_normalization:
            projected = self.audio_norm(projected)
        
        return projected
    
    def process_ros_data(self, msg: Any) -> torch.Tensor:
        """
        Process ROS2 message data.
        
        Args:
            msg: ROS2 message
            
        Returns:
            torch.Tensor: Processed and normalized message data
        """
        return self.ros_preprocessor(msg)
    
    def combine_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Combine features from different modalities.
        
        Args:
            features: List of feature tensors from different modalities
            
        Returns:
            torch.Tensor: Combined feature tensor
        """
        # Default: concatenate along feature dimension
        return torch.cat(features, dim=-1)
    
    def get_total_feature_dim(self) -> int:
        """
        Get the total dimension of combined features.
        
        Returns:
            int: Total feature dimension
        """
        total_dim = 0
        if hasattr(self, 'vision_projection'):
            total_dim += self.projection_dim
        if hasattr(self, 'audio_projection'):
            total_dim += self.projection_dim
        # if hasattr(self, 'ros_projection'):  # Skipped for now
        #     total_dim += self.projection_dim
        return total_dim
    
    def get_trainable_parameters(self):
        """
        Get trainable parameters for optimizer.
        
        Returns:
            List: List of trainable parameter groups
        """
        trainable_params = []
        
        # Add projection layers
        trainable_params.extend(self.vision_projection.parameters())
        trainable_params.extend(self.audio_projection.parameters())
        # trainable_params.extend(self.ros_projection.parameters())  # Skipped for now
        
        # Add normalization layers if used
        if self.use_normalization:
            trainable_params.extend(self.vision_norm.parameters())
            trainable_params.extend(self.audio_norm.parameters())
            # trainable_params.extend(self.ros_norm.parameters())  # Skipped for now
        
        return trainable_params
    
    def __call__(
        self,
        vision: Optional[Union[Image.Image, List[Image.Image]]] = None,
        audio: Optional[Any] = None,
        ros_data: Optional[Any] = None
    ) -> torch.Tensor:
        """
        Process all available modalities and combine features.
        Gradients can flow through projection layers back to actor/critic.
        
        Args:
            vision: Optional vision input
            audio: Optional audio input
            ros_data: Optional ROS2 message (skipped for now)
            
        Returns:
            torch.Tensor: Combined feature tensor
        """
        features = []
        
        if vision is not None:
            try:
                vision_features = self.process_vision(vision)
                features.append(vision_features)
            except Exception as e:
                logger.error(f"Error processing vision input: {e}")
        
        if audio is not None:
            try:
                audio_features = self.process_audio(audio)
                features.append(audio_features)
            except Exception as e:
                logger.error(f"Error processing audio input: {e}")

        if not features:
            raise ValueError("No valid inputs provided for processing")
        
        return self.combine_features(features) 