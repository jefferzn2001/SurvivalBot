# VLM Multimodal Policy Navigation Pipeline

## Overview

The VLM Multimodal Policy Navigation system combines Vision Language Model (VLM) reasoning with learned multimodal policy decisions for autonomous robot navigation. The system processes camera images through CLIP vision encoder and combines with sensor data for enhanced policy decisions.

## Architecture Components

### 1. Core Components
- **VLM_policy.py**: Main navigation system with multimodal integration
- **policy_actor.py**: Neural network policy with CLIP vision processing
- **vision_encoder.py**: CLIP-based vision feature extraction
- **multimodal.py**: Multimodal feature fusion and processing
- **policy_data_visualization.py**: Visualization and analysis tools

### 2. Data Files Generated
- **raw_data.csv**: All sensor data with timestamps and policy decisions
- **reasoning.csv**: VLM decisions and policy outputs for each cycle
- **state.csv**: Exact 77-dimensional multimodal state vectors passed to policy
- **images/**: Original camera images
- **annotated/**: Annotated images with action labels

## Exact Pipeline Flow

### Navigation Cycle Workflow

```
1. 📸 Image Capture
   ├── Capture camera frame
   ├── Save to images/cycle_XXX_timestamp.jpg
   └── Create annotated version

2. 🧠 VLM Processing (Async)
   ├── Send image + goal to Gemini VLM
   ├── Get reasoning text
   └── Extract action number (0-5)

3. 🎯 Multimodal Policy Inference
   ├── Wait for VLM action
   ├── Load image as PIL format
   ├── Process image through CLIP → 512D → projection → 64D vision features
   ├── Extract 13 sensor features (matching State.py but excluding solar_in/current_out)
   ├── Combine: 64D vision + 13D sensors = 77D state
   ├── Mark policy_inference_active = True
   ├── Save 77D state vector to state.csv
   ├── Run multimodal policy neural network inference
   ├── Get distance_scale (0.3-0.6m) and stop_certainty (0-1)
   └── Mark policy_inference_active = False

4. 🚀 Action Execution
   ├── Check if stop_certainty > 0.9 (stop if true)
   ├── Execute VLM action with policy-scaled distance
   ├── Save reasoning to reasoning.csv
   └── Wait for action completion
```

### Multimodal State Vector (77 Features)

The policy receives a 77-dimensional state vector combining vision and sensor data:

```python
# Vision Features (64D): CLIP image features projected to 64D
vision_features = [
    vision_0, vision_1, ..., vision_63    # 64D: CLIP → projection → LayerNorm
]

# Sensor Features (13D): Raw sensor readings matching State.py (excluding solar_in/current_out)
sensor_features = [
    time_category,         # 0: Hour of day (0-23) - categorical
    soc,                   # 1: Battery SOC (0-100) from voltage
    temperature,           # 2: Temperature (°C)
    humidity,              # 3: Humidity (%RH)
    pressure,              # 4: Pressure (hPa)
    roll,                  # 5: Roll (degrees)
    pitch,                 # 6: Pitch (degrees)
    ldr_left,              # 7: Left LDR reading (0-1023)
    ldr_right,             # 8: Right LDR reading (0-1023)
    bumper_hit,            # 9: Any bumper hit (0 or 1) - categorical
    encoder_left,          # 10: Left encoder value
    encoder_right,         # 11: Right encoder value
    action,                # 12: VLM action (0-5) - categorical
]

# Total: 64 + 13 = 77 dimensions
```

### Key State Features (Matching State.py)

#### Categorical Features
- **time_category**: Hour of day (0-23) - NOT normalized
- **bumper_hit**: Binary collision indicator (0 or 1)
- **action**: VLM action choice (0-5)

#### Calculated Features
- **soc**: Battery State of Charge (0-100%) calculated from voltage using BatteryState.voltage_to_soc()

#### Raw Sensor Features
- **temperature, humidity, pressure**: Environmental readings
- **roll, pitch**: IMU orientation (degrees)
- **ldr_left, ldr_right**: Individual light sensor readings (0-1023)
- **encoder_left, encoder_right**: Wheel encoder values

#### Excluded Features (Calculated but NOT Passed to Policy)
- **solar_in**: Solar panel current (0-2.34A) calculated using FakeSolar.py panel_current() formula
- **current_out**: Battery output current (A) during movement
- **ldr_avg**: NOT passed as separate feature (calculated internally for solar_in)
- **power_in_voltage, power_in_current**: NOT included in state vector
- **power_out_voltage**: NOT included in state vector

**Note**: `solar_in` and `current_out` are calculated during feature extraction but excluded from the policy state vector. They will be used for later analysis to compare maximum values between current and next states.

### Multimodal Policy Actor Network

```
Input: 77D multimodal state vector
├── Vision Processing: Image → CLIP (512D) → Linear (64D) → LayerNorm
├── Sensor Processing: 13D raw sensor features (matching State.py, excluding solar_in/current_out)
├── Feature Fusion: Concatenate all features → 77D
├── Shared Trunk (3 hidden layers, 64 units each)
├── Distance Head: Linear → Tanh → Scale to [0.3, 0.6]m
└── Certainty Head: Linear → Sigmoid → [0, 1]
```

### Vision Processing Pipeline

```
Camera Image (BGR) → PIL Image (RGB) → CLIP Preprocessing
                                    ↓
CLIP Vision Encoder (frozen) → 512D features → L2 Normalization
                                    ↓
Linear Projection → 64D features → LayerNorm → Combined with sensors
```

### Data Recording

#### Raw Data (raw_data.csv)
- **Frequency**: Every sensor callback (~10Hz)
- **Timing**: policy_inference column = 1 only during inference, 0 otherwise
- **Content**: All sensor data + policy decisions + timing markers

#### State Data (state.csv)
- **Frequency**: Once per navigation cycle
- **Content**: Exact 77D multimodal state vector passed to policy
- **Structure**: 64 vision features + 13 sensor features + metadata
- **Purpose**: Training data and policy analysis

#### Reasoning Data (reasoning.csv)
- **Frequency**: Once per navigation cycle
- **Content**: VLM action, policy outputs, reasoning text

## Key Timing Markers

### Policy Inference Timing
- `policy_inference_active` flag controls timing marker
- Raw data shows `policy_inference = 1` only during inference
- Visualization shows purple dashed lines for exact inference timing

### Action Execution
- `action_state` tracks current robot state (idle, 0-5)
- Distance scaling applied: `total_distance = distance_scale` (not multiplied)
- Actions 1-5 use scaled distance, Action 0 is turn-only

## Usage

### Running Navigation
```bash
python zeroMQscripts/VLM_policy.py \
    --server-ip 10.102.225.181 \
    --goal "Max Sunlight Location" \
    --model-path SAC/trained_multimodal_policy.pth \
    --stop-threshold 0.9 \
    --device cuda
```

### Visualization
```bash
# Latest session
python data/policy_data_visualization.py

# Specific session
python data/policy_data_visualization.py --session data/policy_data_20241204_135029

# Summary only
python data/policy_data_visualization.py --summary
```

## Output Files Structure

```
data/policy_data_YYYYMMDD_HHMMSS/
├── raw_data.csv          # All sensor data with timing
├── reasoning.csv         # VLM decisions and policy outputs
├── state.csv            # 77D multimodal policy input states
├── images/              # Original camera images
│   └── cycle_XXX_timestamp.jpg
└── annotated/           # Annotated images with labels
    └── cycle_XXX_timestamp.jpg
```

## Key Differences from Standard VLM Navigation

1. **Multimodal Integration**: Camera images processed through CLIP vision encoder
2. **Enhanced State**: 77D state vector (64D vision + 13D sensors)
3. **Policy Integration**: VLM action becomes input to multimodal policy network
4. **Variable Distance**: Policy scales movement distance (0.3-0.6m) based on visual and sensor context
5. **Stop Decisions**: Policy can override VLM with stop certainty using visual information
6. **Enhanced Logging**: 77D multimodal state vectors and timing markers
7. **Vision Features**: Rich visual representations for better spatial understanding

## Technical Details

### CLIP Vision Processing
- **Model**: `openai/clip-vit-base-patch32`
- **Input**: 224x224 RGB images
- **Output**: 512D normalized features
- **Projection**: Linear layer to 64D + LayerNorm
- **Frozen**: CLIP weights are frozen, only projection layer is trainable

### Device Support
- **CPU**: Default mode for inference
- **CUDA**: Accelerated mode for faster vision processing
- **Memory**: ~2GB VRAM for CLIP + policy actor

### Performance Considerations
- **Vision Processing**: ~50-100ms per image on GPU
- **Policy Inference**: ~5-10ms per decision
- **Total Cycle Time**: ~2-5 seconds including VLM processing

## Troubleshooting

### Common Issues
- **CUDA Memory**: Reduce batch size or use CPU if GPU memory insufficient
- **CLIP Loading**: Ensure internet connection for model download on first run
- **State Dimension**: Verify 77D state vector (64 vision + 13 sensors)
- **Image Loading**: Check image paths and PIL compatibility
- **Missing VLM Action**: Ensure VLM completes before policy inference

### Validation
```bash
# Test multimodal policy actor
python -c "
from SAC.policy_actor import VLMPolicyActor
from PIL import Image
import torch
import numpy as np

# Create test data
actor = VLMPolicyActor(obs_dim=77, device='cpu')
image = Image.new('RGB', (224, 224), color='red')
sensor_data = {'ldr': {'left': 500, 'right': 600}, 'power': {'out': {'voltage': 12.5}}}
vlm_action = 3

# Test inference
distance, certainty = actor.get_action(image, sensor_data, vlm_action)
print(f'Distance: {distance:.3f}m, Certainty: {certainty:.3f}')
assert 0.3 <= distance <= 0.6
assert 0.0 <= certainty <= 1.0
print('✅ Multimodal policy actor working correctly')
"
```

## Performance Metrics

The system tracks:
- **Vision Processing Time**: CLIP encoding and projection duration
- **Policy Inference Time**: Multimodal neural network forward pass
- **Distance Scaling Distribution**: Range and variance of scaled distances
- **Stop Certainty Patterns**: When and why policy decides to stop based on visual cues
- **Action Execution Timing**: Complete action cycle duration
- **Power Efficiency**: Solar input vs. power consumption during navigation
- **Visual Feature Quality**: CLIP embedding statistics and diversity

This multimodal pipeline enables sophisticated visual-sensor fusion for robot navigation while maintaining the high-level reasoning capabilities of the VLM system. The integration of CLIP vision features provides rich spatial understanding that enhances policy decisions beyond pure sensor-based approaches. 