#!/usr/bin/env python3
"""
Quick test script to verify that the MultiModalEncoder can load the latest
raw_state.csv, encode observations, and interact with the ReplayBuffer.

Run:
    python tests/test_encoder_replay.py
"""
# Add project root to PYTHONPATH for direct script execution
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pandas as pd
from SAC.utils.multimodal import load_latest_states, MultiModalEncoder
from SAC.replay_buffer import ReplayBuffer

# Mock VisionEncoder for testing without CLIP
class MockVisionEncoder:
    def __init__(self, device='cuda'):
        self.device = device
        
    def load_image_from_row(self, row, raw_state_path):
        # Return a mock PIL image (just create a small RGB image)
        from PIL import Image
        return Image.new('RGB', (224, 224), color='red')
    
    def __call__(self, images):
        # Return mock 512-D features (different for each call to show encoding diversity)
        # Use row timestamp as seed for reproducible but varied features
        seed = hash(str(images)) % 1000000
        torch.manual_seed(seed)
        return torch.randn(1, 512, device=self.device)

def normalize_observations(obs_list):
    """
    Normalize observations to have zero mean and unit variance.
    
    Args:
        obs_list: List of observation vectors
        
    Returns:
        normalized_obs_list, mean, std, constant_features
    """
    obs_array = np.array(obs_list)
    mean = np.mean(obs_array, axis=0)
    std = np.std(obs_array, axis=0)
    
    # Identify constant features (std very close to 0)
    constant_threshold = 1e-8
    constant_features = np.where(std < constant_threshold)[0]
    
    # For constant features, don't normalize (keep original values)
    # For variable features, apply standard normalization
    std_safe = np.where(std < constant_threshold, 1.0, std)
    normalized = (obs_array - mean) / std_safe
    
    # For constant features, just center around 0
    for const_idx in constant_features:
        normalized[:, const_idx] = obs_array[:, const_idx] - mean[const_idx]
    
    return normalized, mean, std, constant_features

def export_replay_buffer_to_csv(replay_buffer, output_path="SAC/replay_buffer.csv"):
    """Export replay buffer contents to CSV for inspection"""
    data = []
    
    for i in range(len(replay_buffer)):
        obs = replay_buffer.obses[i]
        action = replay_buffer.actions[i]
        reward = replay_buffer.rewards[i]
        next_obs = replay_buffer.next_obses[i]
        not_done = replay_buffer.not_dones[i]
        not_done_no_max = replay_buffer.not_dones_no_max[i]
        
        # Create row with all features
        row = {
            'transition_id': i,
            'reward': reward[0],
            'not_done': not_done[0],
            'not_done_no_max': not_done_no_max[0],
            'action_distance': action[0],
            'action_stop_certainty': action[1]
        }
        
        # Add observation features (vision + numeric + bumper + action)
        for j in range(64):  # Vision features
            row[f'obs_vision_{j}'] = obs[j]
        
        # Continuous numeric features (10 features)
        continuous_feature_names = [
            'soc', 'temperature', 'humidity', 'pressure', 
            'roll', 'pitch', 'encoder_left', 'encoder_right', 'ldr_left', 'ldr_right'
        ]
        for j, name in enumerate(continuous_feature_names):
            row[f'obs_{name}'] = obs[64 + j]
            
        # Bumper categorical features (4 features)
        bumper_names = ['bumper_top', 'bumper_bottom', 'bumper_left', 'bumper_right']
        for j, name in enumerate(bumper_names):
            row[f'obs_{name}'] = obs[64 + 10 + j]
            
        # Action one-hot encoding (6 features)
        action_names = ['action_0', 'action_1', 'action_2', 'action_3', 'action_4', 'action_5']
        for j, name in enumerate(action_names):
            row[f'obs_{name}'] = obs[64 + 10 + 4 + j]
            
        # Add next_obs features with same structure
        for j in range(64):  # Vision features
            row[f'next_obs_vision_{j}'] = next_obs[j]
        for j, name in enumerate(continuous_feature_names):
            row[f'next_obs_{name}'] = next_obs[64 + j]
        for j, name in enumerate(bumper_names):
            row[f'next_obs_{name}'] = next_obs[64 + 10 + j]
        for j, name in enumerate(action_names):
            row[f'next_obs_{name}'] = next_obs[64 + 10 + 4 + j]
            
        data.append(row)
    
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"📊 Replay buffer exported to: {output_path}")
    return df

# Main test function 
def test_with_mock_vision():
    """Test function that uses mock vision encoder with full analysis"""
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")
    
    # Load latest raw_state.csv
    df, raw_state_path = load_latest_states()
    print(f"📁 Loaded {len(df)} state rows from: {raw_state_path}")
    print(f"📊 Expected transitions: {len(df) - 1}")
    
    # Create encoder with mock vision
    mock_vision = MockVisionEncoder(device=device)
    encoder = MultiModalEncoder(device=device, vision_encoder=mock_vision)
    
    obs_dim = encoder.get_feature_dim()
    print(f"🧠 Observation dimension: {obs_dim} (64 vision + 10 continuous + 4 bumper + 6 action)")

    # Process all possible transitions
    num_transitions = len(df) - 1
    print(f"🔄 Processing {num_transitions} transitions...")
    
    # First pass: collect all observations for normalization
    all_obs = []
    transition_data = []
    
    for i in range(num_transitions):
        curr_row = df.iloc[i]
        next_row = df.iloc[i + 1]

        try:
            # Get raw observations
            obs_vec = encoder(curr_row, raw_state_path).squeeze(0).detach().cpu().numpy()
            next_vec = encoder(next_row, raw_state_path).squeeze(0).detach().cpu().numpy()
            
            all_obs.extend([obs_vec, next_vec])
            
            # Store transition data
            act_id = int(curr_row['action'])
            transition_data.append({
                'i': i,
                'curr_row': curr_row,
                'next_row': next_row,
                'obs_vec': obs_vec,
                'next_vec': next_vec,
                'act_id': act_id
            })
            
            # Show detailed encoding for first few transitions
            if i < 3:
                print(f"\n🔍 Transition {i+1} encoding details:")
                print(f"   VLM Action: {act_id}")
                print(f"   Vision features (first 5): {obs_vec[:5]}")
                print(f"   Continuous features: {obs_vec[64:74]}")  # 10 continuous
                print(f"   Bumper features (TBLR): {obs_vec[74:78]}")  # 4 bumper
                print(f"   Action one-hot: {obs_vec[78:84]}")  # 6 action
                print(f"   Feature range: [{obs_vec.min():.3f}, {obs_vec.max():.3f}]")
                
                # Show temperature/humidity issue
                temp_val = obs_vec[65]  # temperature is index 1 in continuous
                humid_val = obs_vec[66]  # humidity is index 2 in continuous
                print(f"   ⚠️  Temperature: {temp_val:.3f}, Humidity: {humid_val:.3f}")
                
        except Exception as e:
            print(f"❌ Failed to encode transition {i+1}: {e}")
            
    print(f"\n📈 Successfully encoded {len(transition_data)} transitions")
    
    # Normalize observations
    print("🔧 Normalizing observations...")
    obs_only = [td['obs_vec'] for td in transition_data]
    next_obs_only = [td['next_vec'] for td in transition_data]
    
    all_obs_combined = obs_only + next_obs_only
    normalized_obs, obs_mean, obs_std, constant_features = normalize_observations(all_obs_combined)
    
    # Split back into obs and next_obs
    n_trans = len(transition_data)
    normalized_obs_only = normalized_obs[:n_trans]
    normalized_next_obs_only = normalized_obs[n_trans:]
    
    print(f"📊 Normalization stats:")
    print(f"   Original obs range: [{np.min(all_obs_combined):.3f}, {np.max(all_obs_combined):.3f}]")
    print(f"   Normalized obs range: [{np.min(normalized_obs):.3f}, {np.max(normalized_obs):.3f}]")
    print(f"   Mean magnitude: {np.mean(np.abs(obs_mean)):.3f}")
    print(f"   Std magnitude: {np.mean(obs_std):.3f}")
    
    # Report constant features
    if len(constant_features) > 0:
        print(f"   ⚠️  {len(constant_features)} constant features detected:")
        feature_names = ['vision'] * 64 + ['soc', 'temperature', 'humidity', 'pressure', 'roll', 'pitch', 'encoder_left', 'encoder_right', 'ldr_left', 'ldr_right'] + ['bumper_top', 'bumper_bottom', 'bumper_left', 'bumper_right'] + ['action_0', 'action_1', 'action_2', 'action_3', 'action_4', 'action_5']
        for idx in constant_features:
            if idx < len(feature_names):
                if idx < 64:
                    print(f"      - {feature_names[idx]}_{idx}: constant vision feature")
                else:
                    print(f"      - {feature_names[idx]}: {obs_mean[idx]:.3f} (constant)")
            else:
                print(f"      - Feature {idx}: {obs_mean[idx]:.3f} (constant)")
    else:
        print(f"   ✅ All features are variable")

    # Set up replay buffer
    capacity = max(num_transitions, 100)  # At least 100 for good sampling
    rb = ReplayBuffer(obs_shape=(obs_dim,), action_shape=(2,), capacity=capacity, device=device)

    # Add all transitions to replay buffer with normalized observations
    print(f"\n🗃️ Adding {len(transition_data)} transitions to ReplayBuffer...")
    for i, td in enumerate(transition_data):
        # Use normalized observations
        obs_vec = normalized_obs_only[i]
        next_vec = normalized_next_obs_only[i]
        
        # Map discrete action to continuous action space
        act_id = td['act_id']
        distance_scale = (act_id > 0) * 1.0  # 1 if moving, 0 if turn in place
        stop_certainty = 1.0 if act_id == 0 else 0.0
        action_vec = np.array([distance_scale, stop_certainty], dtype=np.float32)

        # Dummy reward (could be computed from LDR improvement, battery, etc.)
        reward = np.array([0.1], dtype=np.float32)  # Small positive reward
        
        # Episode termination flags:
        # not_done: False if episode terminated (robot crashed, stuck, etc.)
        # not_done_no_max: False if episode terminated due to max steps (timeout)
        # For this data, assume no termination except at the very end
        done = (i == len(transition_data) - 1)  # Only last transition is done
        done_no_max = done  # Same for this simple case
        
        rb.add(obs_vec, action_vec, reward, next_vec, done, done_no_max)
        
        if i < 3:
            print(f"   ✅ Added transition {i+1}: action={act_id}, done={done}")

    print(f"🗃️ ReplayBuffer filled with {len(rb)} transitions (capacity {rb.capacity})")

    # Export to CSV
    export_df = export_replay_buffer_to_csv(rb)
    
    # Sample a batch and show results
    sample_batch_size = min(8, len(rb))
    batch = rb.sample(sample_batch_size)
    names = ["obs", "action", "reward", "next_obs", "not_done", "not_done_no_max"]
    
    print(f"\n🎲 Sampled batch of {sample_batch_size}:")
    for name, tensor in zip(names, batch):
        print(f"   {name:<15}: {tuple(tensor.shape)} on {tensor.device}")
    
    print(f"\n📝 Explanation of fields:")
    print(f"   • obs/next_obs: 84-D normalized state (64 vision + 10 continuous + 4 bumper + 6 action)")
    print(f"   • action: [distance_scale, stop_certainty] continuous mapping")
    print(f"   • reward: Placeholder rewards")
    print(f"   • not_done: 1.0 if episode continues, 0.0 if terminated")
    print(f"   • not_done_no_max: 1.0 if natural continuation, 0.0 if timeout")
    print(f"   • Categorical features: bumper (TBLR binary), VLM action (0-5 one-hot)")
    
    print(f"\n✅ Test completed successfully!")
    print(f"📊 Check SAC/replay_buffer.csv for detailed buffer contents")
    
    return rb, export_df


def main(device="cuda", capacity=10000, batch_size=32):
    # Use comprehensive test 
    rb, df = test_with_mock_vision()
    return rb, df


if __name__ == "__main__":
    replay_buffer, buffer_df = main() 