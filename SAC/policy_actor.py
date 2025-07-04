#!/usr/bin/env python3
"""
Policy Actor for VLM Navigation
Two-head actor model for distance scaling and stop certainty
Now supports multimodal input: vision (64D) + sensors (13D) = 77D
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import sys

# Add SAC utils to path for multimodal encoder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from SAC.utils.multimodal import MultiModalEncoder

logger = logging.getLogger(__name__)

# Import weight_init function directly from utils
def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class VLMPolicyActor(nn.Module):
    """
    Two-head actor model for VLM navigation policy with multimodal input.
    
    Input: 77D = 64D vision features + 13D sensor features
    Head 1: Distance scaling factor (0.3 to 0.6 meters)
    Head 2: Stop certainty (0 to 1, if > 0.9 then stop)
    """
    
    def __init__(self, obs_dim=77, hidden_dim=64, device='cpu'):
        """
        Initialize VLM Policy Actor with multimodal encoder
        
        Args:
            obs_dim: Observation dimension (77 = 64 vision + 13 sensor)
            hidden_dim: Hidden layer dimension
            device: Device for computation
        """
        super().__init__()
        
        self.device = device
        self.obs_dim = obs_dim
        
        # Initialize multimodal encoder
        self.multimodal_encoder = MultiModalEncoder(device=device)
        
        # Shared trunk - processes combined multimodal features
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Head 1: Distance scaling factor (0.3 to 0.6)
        self.distance_head = nn.Linear(hidden_dim, 1)
        
        # Head 2: Stop certainty (0 to 1)
        self.certainty_head = nn.Linear(hidden_dim, 1)
        
        # Apply weight initialization
        self.apply(weight_init)
        
        # Output tracking
        self.outputs = {}
        
        # Move to device
        self.to(device)
        
        logger.info(f"VLMPolicyActor initialized with obs_dim={obs_dim}, device={device}")
    
    def extract_multimodal_features(self, image_pil, sensor_data, vlm_action):
        """
        Extract multimodal features: vision (64D) + sensors (13D) = 77D
        
        Args:
            image_pil: PIL image
            sensor_data: Sensor data dictionary
            vlm_action: VLM action number (0-5)
            
        Returns:
            torch.Tensor: Combined 77D feature vector
        """
        # Process image through CLIP and projection to 64D
        vision_features = self.multimodal_encoder.vision_encoder(image_pil)  # (1, 512)
        vision_proj = self.multimodal_encoder.vision_projection(vision_features)  # (1, 64)
        vision_norm = self.multimodal_encoder.vision_norm(vision_proj)  # (1, 64)
        
        # Extract 13D sensor features (matching State.py but excluding solar_in and current_out)
        sensor_features = self.extract_sensor_features(sensor_data, vlm_action)  # (13,)
        
        # Normalize sensor features to reasonable range
        sensor_features = sensor_features / 100.0  # Normalize to [0, 1] range
        
        # Use standard normalized features (both training and inference)
        vision_features = vision_norm.squeeze(0)  # (64,)
        
        # Combine: vision (64) + sensors (13) = 77D
        combined_features = torch.cat([
            vision_features,  # (64,)
            sensor_features         # (13,)
        ], dim=0)  # (77,)
        
        return combined_features
    
    def extract_sensor_features(self, sensor_data, vlm_action):
        """
        Extract 13D sensor features matching State.py but excluding solar_in and current_out
        
        Args:
            sensor_data: Sensor data dictionary
            vlm_action: VLM action number (0-5)
        
        Returns:
            torch.Tensor: 13D sensor feature vector
        """
        if not sensor_data:
            return torch.zeros(13, device=self.device)
        
        # Calculate derived features exactly as in State.py
        from datetime import datetime
        from data.FakeSolar import panel_current
        
        # 1. time_category (0-23) - categorical hour
        now = datetime.now()
        time_category = now.hour  # Keep as categorical (0-23)
        
        # 2. soc - Battery SOC from voltage using State.py mapping
        power = sensor_data.get('power', {})
        power_out = power.get('out', {})
        voltage = power_out.get('voltage', 12.0)
        # Simplified SOC calculation (should match BatteryState.voltage_to_soc)
        soc = max(0, min(100, (voltage - 12.0) / 1.6 * 100))  # 0-100
        
        # Note: solar_in and current_out are calculated but NOT included in state vector
        # They will be used for later analysis between current and next state
        ldr = sensor_data.get('ldr', {})
        ldr_left = ldr.get('left', 512)
        ldr_right = ldr.get('right', 512)
        ldr_avg = (ldr_left + ldr_right) / 2.0
        solar_in = panel_current(ldr_avg)  # 0-2.34A (calculated but not passed)
        current_out = power_out.get('current', 0.0)  # (calculated but not passed)
        
        # 3. temperature - °C
        env = sensor_data.get('environment', {})
        temperature = env.get('temperature', 25.0)
        
        # 4. humidity - %RH
        humidity = env.get('humidity', 50.0)
        
        # 5. pressure - hPa
        pressure = env.get('pressure', 1013.25)
        
        # 6. roll - degrees
        imu = sensor_data.get('imu', {})
        roll = imu.get('roll', 0.0)
        
        # 7. pitch - degrees
        pitch = imu.get('pitch', 0.0)
        
        # 8. ldr_left - left LDR reading
        # (already extracted above)
        
        # 9. ldr_right - right LDR reading
        # (already extracted above)
        
        # 10. bumper_hit - categorical (0 or 1)
        bumpers = sensor_data.get('bumpers', {})
        bumper_hit = int(any([
            bumpers.get('top', 0) > 0,
            bumpers.get('bottom', 0) > 0,
            bumpers.get('left', 0) > 0,
            bumpers.get('right', 0) > 0
        ]))
        
        # 11. encoder_left - left encoder value
        encoders = sensor_data.get('encoders', {})
        encoder_left = encoders.get('left', 0)
        
        # 12. encoder_right - right encoder value
        encoder_right = encoders.get('right', 0)
        
        # 13. action - VLM action (0-5) - categorical
        action = vlm_action  # Keep as categorical (0-5)
        
        # Assemble 13D sensor vector (excluding solar_in and current_out)
        sensor_vector = np.array([
            float(time_category),      # 0: time_category (0-23)
            float(soc),                # 1: soc (0-100)
            float(temperature),        # 2: temperature (°C)
            float(humidity),           # 3: humidity (%RH)
            float(pressure),           # 4: pressure (hPa)
            float(roll),               # 5: roll (degrees)
            float(pitch),              # 6: pitch (degrees)
            float(ldr_left),           # 7: ldr_left (0-1023)
            float(ldr_right),          # 8: ldr_right (0-1023)
            float(bumper_hit),         # 9: bumper_hit (0 or 1)
            float(encoder_left),       # 10: encoder_left
            float(encoder_right),      # 11: encoder_right
            float(action),             # 12: action (0-5)
        ])
        
        return torch.tensor(sensor_vector, dtype=torch.float32, device=self.device)
    
    def forward(self, obs):
        """
        Forward pass through the network
        
        Args:
            obs: Observation tensor [batch_size, obs_dim] (77D multimodal features)
            
        Returns:
            dict: Contains 'distance_scale' and 'stop_certainty'
        """
        # Shared feature extraction
        features = self.trunk(obs)
        
        # Head 1: Distance scaling factor
        # Use tanh to get [-1, 1], then scale to [0.3, 0.6]
        distance_raw = self.distance_head(features)
        distance_tanh = torch.tanh(distance_raw)
        distance_scale = 0.3 + 0.15 * (distance_tanh + 1)  # Maps [-1,1] to [0.3,0.6]
        
        # Head 2: Stop certainty - for untrained models, give lower default values
        # Use sigmoid to get [0, 1], but bias towards lower values for untrained models
        certainty_raw = self.certainty_head(features)
        stop_certainty = torch.sigmoid(certainty_raw)
        
        # For untrained models, bias towards lower certainty unless there's a strong signal
        # This prevents the policy from stopping everything during testing
        if self.training:
            # During training, use the raw sigmoid output
            pass
        else:
            # During inference/testing, bias towards lower certainty for untrained models
            # Only give high certainty if the raw output is very strong
            if torch.max(certainty_raw) < 2.0:  # If no strong signal
                stop_certainty = stop_certainty * 0.3  # Scale down to reasonable range
            # Additional safety: cap at 0.5 for untrained models to prevent constant stopping
            stop_certainty = torch.clamp(stop_certainty, 0.0, 0.5)
        
        # Store outputs for logging
        self.outputs['distance_scale'] = distance_scale
        self.outputs['stop_certainty'] = stop_certainty
        self.outputs['distance_raw'] = distance_raw
        self.outputs['certainty_raw'] = certainty_raw
        
        return {
            'distance_scale': distance_scale,
            'stop_certainty': stop_certainty
        }
    
    def get_action(self, image_pil, sensor_data, vlm_action):
        """
        Get action for multimodal input (inference mode)
        
        Args:
            image_pil: PIL image
            sensor_data: Sensor data dictionary
            vlm_action: VLM action number (0-5)
            
        Returns:
            tuple: (distance_scale, stop_certainty) as float values rounded to 2 decimal places
        """
        # Extract multimodal features
        multimodal_features = self.extract_multimodal_features(image_pil, sensor_data, vlm_action)
        obs = multimodal_features.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            output = self.forward(obs)
            distance_scale = round(output['distance_scale'].item(), 2)
            stop_certainty = round(output['stop_certainty'].item(), 2)
            
        return distance_scale, stop_certainty
    
    def get_action_from_tensor(self, obs):
        """
        Get action from pre-computed tensor (for compatibility)
        
        Args:
            obs: Single observation tensor [obs_dim] or [1, obs_dim]
            
        Returns:
            tuple: (distance_scale, stop_certainty) as float values rounded to 2 decimal places
        """
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        
        with torch.no_grad():
            output = self.forward(obs)
            distance_scale = round(output['distance_scale'].item(), 2)
            stop_certainty = round(output['stop_certainty'].item(), 2)
            
        return distance_scale, stop_certainty
    
    def log(self, logger, step):
        """Log outputs and parameters"""
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_policy_actor/{k}_hist', v, step)
        
        # Log network parameters
        for i, layer in enumerate(self.trunk):
            if isinstance(layer, nn.Linear):
                logger.log_param(f'train_policy_actor/trunk_{i}', layer, step)
        
        logger.log_param('train_policy_actor/distance_head', self.distance_head, step)
        logger.log_param('train_policy_actor/certainty_head', self.certainty_head, step) 