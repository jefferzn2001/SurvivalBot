import torch
import logging
from typing import Optional, List, Union, Dict, Any
from PIL import Image

from .vision_encoder import VisionEncoder
from .audio_encoder import AudioEncoder
from .ros_preprocessor import ROSPreprocessor

logger = logging.getLogger(__name__)

class MultiModalPreprocessor:
    """
    Combined preprocessor for handling multiple input modalities:
    - Vision (CLIP)
    - Audio (CLAP - to be implemented)
    - ROS2 messages
    """
    
    def __init__(self, device="cuda"):
        self.device = device
        logger.info("Initializing MultiModal Preprocessor")
        
        # Initialize encoders
        self.vision_encoder = VisionEncoder(device=device)
        self.audio_encoder = AudioEncoder(device=device)
        self.ros_preprocessor = ROSPreprocessor(device=device)
        
        logger.info("MultiModal Preprocessor initialized successfully")
    
    def process_vision(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """
        Process vision input through CLIP.
        
        Args:
            images: Single PIL image or list of PIL images
            
        Returns:
            torch.Tensor: Normalized vision features
        """
        return self.vision_encoder(images)
    
    def process_audio(self, audio: Any) -> torch.Tensor:
        """
        Process audio input through CLAP (to be implemented).
        
        Args:
            audio: Audio input (format TBD)
            
        Returns:
            torch.Tensor: Normalized audio features
        """
        return self.audio_encoder(audio)
    
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
    
    def __call__(
        self,
        vision: Optional[Union[Image.Image, List[Image.Image]]] = None,
        audio: Optional[Any] = None,
        ros_data: Optional[Any] = None
    ) -> torch.Tensor:
        """
        Process all available modalities and combine features.
        
        Args:
            vision: Optional vision input
            audio: Optional audio input
            ros_data: Optional ROS2 message
            
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
        
        if ros_data is not None:
            try:
                ros_features = self.process_ros_data(ros_data)
                features.append(ros_features)
            except Exception as e:
                logger.error(f"Error processing ROS data: {e}")
        
        if not features:
            raise ValueError("No valid inputs provided for processing")
        
        return self.combine_features(features) 