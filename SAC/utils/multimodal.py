import torch
import torch.nn as nn
from typing import Tuple, Optional, Any
import logging
import os
import glob
import pandas as pd

# Import modality encoders
from .vision_encoder import VisionEncoder
# from .audio_encoder import AudioEncoder  # COMMENTED OUT

logger = logging.getLogger(__name__)

def find_latest_raw_state_csv(data_dir=None):
    """
    Find the latest raw_state.csv in the data/ directory.
    Checks both session subdirectories and the root data directory.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')
    data_dir = os.path.abspath(data_dir)
    
    # Check session subdirectories first
    session_files = glob.glob(os.path.join(data_dir, 'data_*/raw_state.csv'))
    
    # Also check root data directory
    root_file = os.path.join(data_dir, 'raw_state.csv')
    if os.path.exists(root_file):
        session_files.append(root_file)
    
    if not session_files:
        raise FileNotFoundError("No raw_state.csv found in any session directory or root data directory.")
    
    session_files.sort()
    return session_files[-1]

def load_latest_states(data_dir=None):
    """
    Load the latest raw_state.csv as a DataFrame.
    """
    path = find_latest_raw_state_csv(data_dir)
    df = pd.read_csv(path)
    return df, path

class MultiModalEncoder(nn.Module):
    """
    Multi-modal encoder for processing different data types:
    - Camera images - CNN encoder + normalization  
    - IMU/sensor data - linear projection + normalization
    
    NOTE: Audio encoder is currently commented out.
    """
    
    def __init__(self, device: str = 'cuda', data_dir=None, vision_encoder=None):
        super().__init__()
        self.device = device
        self.data_dir = data_dir
        
        # Initialize encoders
        if vision_encoder is not None:
            self.vision_encoder = vision_encoder
        else:
            self.vision_encoder = VisionEncoder(device=device)
        # self.audio_encoder = AudioEncoder(device=device)  # COMMENTED OUT
        
        # Get feature dimensions
        self.vision_dim = 512  # CLIP default
        # self.audio_dim = self.audio_encoder.get_feature_dim()  # COMMENTED OUT
        
        # Projection to unified dimension
        projection_dim = 64  # project vision features to 64 dims
        self.vision_projection = nn.Linear(self.vision_dim, projection_dim).to(device)
        # self.audio_projection = nn.Linear(self.audio_dim, projection_dim).to(device)  # COMMENTED OUT
        
        # Normalization layers
        self.vision_norm = nn.LayerNorm(projection_dim).to(device)
        # self.audio_norm = nn.LayerNorm(projection_dim).to(device)  # COMMENTED OUT
        
        logger.info(f"MultiModalEncoder initialized:")
        logger.info(f"  Vision: {self.vision_dim} -> {projection_dim}")
        # logger.info(f"  Audio: {self.audio_dim} -> {projection_dim}")
        logger.info(f"  Device: {device}")
        
        self.to(device)
    
    def get_numeric_features(self, row):
        """Extract numeric features from a raw_state row as a torch tensor."""
        # Continuous numeric features  
        continuous_features = [
            'soc',
            'temperature', 'humidity', 'pressure',
            'roll', 'pitch',
            'encoder_left', 'encoder_right',
            'ldr_left', 'ldr_right'
        ]
        
        # Extract continuous values
        values = [float(row[k]) for k in continuous_features]
        
        # Add bumper categorical features (4 separate binary indicators)
        # Check if we have individual bumper columns or combined bumper_hit
        if 'bumper_top' in row.index:
            # Individual bumper data from raw_data processing
            bumper_values = [
                float(row['bumper_top']),
                float(row['bumper_bottom']), 
                float(row['bumper_left']),
                float(row['bumper_right'])
            ]
        else:
            # Fallback: decode from bumper_hit if available
            bumper_hit = float(row.get('bumper_hit', 0))
            # For now, assume bumper_hit=0 means no bumpers, >0 means some bumper hit
            # This is a simplification - in real data you'd decode the actual bumper pattern
            bumper_values = [
                1.0 if bumper_hit > 0 else 0.0,  # top (placeholder)
                0.0,  # bottom
                0.0,  # left  
                0.0   # right
            ]
        
        # Add VLM categorical action (0-5) as one-hot encoding
        action = int(row.get('action', 0))
        action_onehot = [0.0] * 6  # 6 actions: 0,1,2,3,4,5
        if 0 <= action <= 5:
            action_onehot[action] = 1.0
            
        # Combine all features
        all_values = values + bumper_values + action_onehot
        
        return torch.tensor(all_values, dtype=torch.float32, device=self.device)

    def process_vision(self, row, raw_state_path) -> torch.Tensor:
        """
        Process camera image data from a row in raw_state.csv.
        Args:
            row: DataFrame row from raw_state.csv
            raw_state_path: Path to the raw_state.csv file
        Returns:
            torch.Tensor: Encoded vision features
        """
        image = self.vision_encoder.load_image_from_row(row, raw_state_path)
        vision_features = self.vision_encoder(image)  # (1, 512)
        vision_proj = self.vision_projection(vision_features)
        vision_norm = self.vision_norm(vision_proj)
        numeric_tensor = self.get_numeric_features(row).unsqueeze(0)  # (1, N)
        # Concatenate vision (64) + numeric
        combined = torch.cat([vision_norm, numeric_tensor], dim=-1)  # (1, 64+N)
        return combined
    
    # def process_audio(self, audio: torch.Tensor) -> torch.Tensor:
    #     """
    #     Process audio data.
    #     Args:
    #         audio: Audio tensor
    #     Returns:
    #         torch.Tensor: Encoded audio features
    #     """
    #     return self.audio_encoder(audio)
    
    def forward(self, row, raw_state_path) -> torch.Tensor:
        """
        Forward pass through multi-modal encoder for a single state row.
        Args:
            row: DataFrame row from raw_state.csv
            raw_state_path: Path to the raw_state.csv file
        Returns:
            torch.Tensor: Vision features (projected and normalized)
        """
        return self.process_vision(row, raw_state_path)
    
    def get_feature_dim(self) -> int:
        """
        Get total feature dimension.
        Returns:
            int: Total feature dimension
        """
        # 64 vision + 10 continuous + 4 bumper + 6 action_onehot = 84
        return 64 + 10 + 4 + 6
    
    def get_trainable_parameters(self):
        """
        Get all trainable parameters for optimization.
        Returns:
            List of trainable parameters
        """
        trainable_params = []
        # Vision encoder parameters
        trainable_params.extend(self.vision_encoder.clip_model.parameters())
        trainable_params.extend(self.vision_projection.parameters())
        trainable_params.extend(self.vision_norm.parameters())
        # Audio encoder parameters (commented out)
        # trainable_params.extend(self.audio_encoder.parameters())
        # trainable_params.extend(self.audio_projection.parameters())
        # trainable_params.extend(self.audio_norm.parameters())
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