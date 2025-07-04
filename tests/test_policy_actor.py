#!/usr/bin/env python3
"""
Unit tests for VLM Policy Actor
Tests the two-head actor model for distance scaling and stop certainty
"""

import pytest
import torch
import numpy as np
import os
import sys

# Add SAC directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SAC')))

from policy_actor import VLMPolicyActor


class TestVLMPolicyActor:
    """Test cases for VLM Policy Actor"""
    
    def test_actor_initialization(self):
        """Test if actor initializes correctly"""
        obs_dim = 18
        actor = VLMPolicyActor(obs_dim=obs_dim)
        
        assert actor.obs_dim == obs_dim
        assert actor.distance_head is not None
        assert actor.certainty_head is not None
        assert len(actor.trunk) == 6  # 3 Linear + 3 ReLU layers
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shapes and ranges"""
        obs_dim = 18
        batch_size = 4
        actor = VLMPolicyActor(obs_dim=obs_dim)
        
        # Create dummy observation
        obs = torch.randn(batch_size, obs_dim)
        
        # Forward pass
        output = actor.forward(obs)
        
        # Check output structure
        assert isinstance(output, dict)
        assert 'distance_scale' in output
        assert 'stop_certainty' in output
        
        # Check shapes
        assert output['distance_scale'].shape == (batch_size, 1)
        assert output['stop_certainty'].shape == (batch_size, 1)
        
        # Check value ranges
        distance_scale = output['distance_scale'].detach().numpy()
        stop_certainty = output['stop_certainty'].detach().numpy()
        
        # Distance scale should be in [0.3, 0.6]
        assert np.all(distance_scale >= 0.3)
        assert np.all(distance_scale <= 0.6)
        
        # Stop certainty should be in [0, 1]
        assert np.all(stop_certainty >= 0.0)
        assert np.all(stop_certainty <= 1.0)
    
    def test_get_action_single_observation(self):
        """Test get_action method with single observation"""
        obs_dim = 18
        actor = VLMPolicyActor(obs_dim=obs_dim)
        
        # Create single observation
        obs = torch.randn(obs_dim)
        
        # Get action
        distance_scale, stop_certainty = actor.get_action(obs)
        
        # Check types and ranges
        assert isinstance(distance_scale, float)
        assert isinstance(stop_certainty, float)
        assert 0.3 <= distance_scale <= 0.6
        assert 0.0 <= stop_certainty <= 1.0
    
    def test_get_action_batch_observation(self):
        """Test get_action method with batch observation"""
        obs_dim = 18
        actor = VLMPolicyActor(obs_dim=obs_dim)
        
        # Create batch observation
        obs = torch.randn(1, obs_dim)
        
        # Get action
        distance_scale, stop_certainty = actor.get_action(obs)
        
        # Check types and ranges
        assert isinstance(distance_scale, float)
        assert isinstance(stop_certainty, float)
        assert 0.3 <= distance_scale <= 0.6
        assert 0.0 <= stop_certainty <= 1.0
    
    def test_state_feature_extraction(self):
        """Test state feature extraction from sensor data"""
        # This would test the VLMPolicyNavigation.extract_state_features method
        # but we'll create a simplified version here
        
        # Mock sensor data
        sensor_data = {
            'ldr': {'left': 400, 'right': 600},
            'power': {
                'in': {'voltage': 14.0, 'current': 2.0},
                'out': {'voltage': 12.6, 'current': 1.5}
            },
            'imu': {'roll': 5.0, 'pitch': -2.0, 'yaw': 45.0},
            'environment': {'temperature': 25.0, 'humidity': 60.0, 'pressure': 1013.25},
            'encoders': {'left': 1000, 'right': 1200},
            'bumpers': {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        }
        
        # Import the navigation class to test feature extraction
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'zeroMQscripts')))
        from VLM_policy import VLMPolicyNavigation
        
        # Create a dummy navigation instance
        nav = VLMPolicyNavigation.__new__(VLMPolicyNavigation)
        nav.obs_dim = 18
        
        # Extract features
        features = nav.extract_state_features(sensor_data)
        
        # Check feature vector
        assert isinstance(features, np.ndarray)
        assert features.shape == (18,)
        assert np.all(np.isfinite(features))  # No NaN or inf values
    
    def test_distance_scaling_math(self):
        """Test the distance scaling math"""
        obs_dim = 18
        actor = VLMPolicyActor(obs_dim=obs_dim)
        
        # Test with extreme inputs to verify scaling
        obs_extreme_neg = torch.full((1, obs_dim), -10.0)  # Should push tanh to -1
        obs_extreme_pos = torch.full((1, obs_dim), 10.0)   # Should push tanh to +1
        
        output_neg = actor.forward(obs_extreme_neg)
        output_pos = actor.forward(obs_extreme_pos)
        
        # Distance scaling should still be in valid range
        assert 0.3 <= output_neg['distance_scale'].item() <= 0.6
        assert 0.3 <= output_pos['distance_scale'].item() <= 0.6
        
        # The outputs should be different (showing the scaling works)
        assert output_neg['distance_scale'].item() != output_pos['distance_scale'].item()
    
    def test_model_save_load(self):
        """Test model save and load functionality"""
        obs_dim = 18
        actor = VLMPolicyActor(obs_dim=obs_dim)
        
        # Create dummy observation
        obs = torch.randn(1, obs_dim)
        
        # Get initial output
        output1 = actor.forward(obs)
        
        # Save model
        temp_path = "/tmp/test_policy_actor.pth"
        torch.save(actor.state_dict(), temp_path)
        
        # Create new actor and load
        actor2 = VLMPolicyActor(obs_dim=obs_dim)
        actor2.load_state_dict(torch.load(temp_path))
        
        # Get output from loaded model
        output2 = actor2.forward(obs)
        
        # Outputs should be identical
        assert torch.allclose(output1['distance_scale'], output2['distance_scale'])
        assert torch.allclose(output1['stop_certainty'], output2['stop_certainty'])
        
        # Clean up
        os.remove(temp_path)
    
    def test_evaluation_mode(self):
        """Test that model works in evaluation mode"""
        obs_dim = 18
        actor = VLMPolicyActor(obs_dim=obs_dim)
        
        # Set to evaluation mode
        actor.eval()
        
        # Create observation
        obs = torch.randn(1, obs_dim)
        
        # Should work without gradients
        with torch.no_grad():
            output = actor.forward(obs)
            distance_scale, stop_certainty = actor.get_action(obs)
        
        # Check outputs are valid
        assert 0.3 <= output['distance_scale'].item() <= 0.6
        assert 0.0 <= output['stop_certainty'].item() <= 1.0
        assert 0.3 <= distance_scale <= 0.6
        assert 0.0 <= stop_certainty <= 1.0


def test_expected_use_case():
    """Test the expected use case workflow"""
    # Create actor
    obs_dim = 18
    actor = VLMPolicyActor(obs_dim=obs_dim)
    actor.eval()
    
    # Simulate normalized state features (as would come from extract_state_features)
    state_features = np.array([
        0.5,    # time_category / 24.0
        0.8,    # soc / 100.0
        0.6,    # ldr_avg / 1024.0
        0.3,    # solar_in / 5.0
        0.1,    # current_out / 10.0
        0.2,    # temperature normalized
        0.6,    # humidity / 100.0
        0.3,    # pressure normalized
        0.05,   # roll / 180.0
        -0.02,  # pitch / 180.0
        0.4,    # ldr_left / 1024.0
        0.6,    # ldr_right / 1024.0
        0.0,    # bumper_hit
        0.1,    # encoder_left / 10000.0
        0.12,   # encoder_right / 10000.0
        0.7,    # power_in_voltage / 20.0
        0.4,    # power_in_current / 5.0
        0.84    # power_out_voltage / 15.0
    ])
    
    # Convert to tensor
    state_tensor = torch.FloatTensor(state_features)
    
    # Get policy decision
    distance_scale, stop_certainty = actor.get_action(state_tensor)
    
    # Verify outputs
    assert 0.3 <= distance_scale <= 0.6
    assert 0.0 <= stop_certainty <= 1.0
    
    # Test stop decision logic
    stop_threshold = 0.9
    should_stop = stop_certainty > stop_threshold
    assert isinstance(should_stop, bool)
    
    print(f"✅ Expected use case test passed:")
    print(f"   Distance scale: {distance_scale:.3f}m")
    print(f"   Stop certainty: {stop_certainty:.3f}")
    print(f"   Should stop: {should_stop}")


if __name__ == '__main__':
    # Run the expected use case test
    test_expected_use_case()
    
    # Run pytest if available
    try:
        pytest.main([__file__, '-v'])
    except ImportError:
        print("pytest not available, running manual tests...")
        
        # Manual test runner
        test_class = TestVLMPolicyActor()
        
        test_methods = [
            test_class.test_actor_initialization,
            test_class.test_forward_pass,
            test_class.test_get_action_single_observation,
            test_class.test_get_action_batch_observation,
            test_class.test_distance_scaling_math,
            test_class.test_model_save_load,
            test_class.test_evaluation_mode
        ]
        
        for test_method in test_methods:
            try:
                test_method()
                print(f"✅ {test_method.__name__} passed")
            except Exception as e:
                print(f"❌ {test_method.__name__} failed: {e}") 