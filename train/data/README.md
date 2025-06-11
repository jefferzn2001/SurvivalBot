# Training Data Directory

This directory contains neural network training data collected from the SurvivalBot system.

## Data Structure

The data collection node saves data in the following format:

```
train/data/
├── session_YYYYMMDD_HHMMSS/          # Session directory
│   ├── images/                        # Raw RGB camera images
│   │   ├── img_000001_timestamp.jpg   # Individual image files
│   │   ├── img_000002_timestamp.jpg
│   │   └── ...
│   └── data/                          # Training data files
│       ├── batch_YYYYMMDD_HHMMSS.csv  # CSV format (human readable)
│       ├── batch_YYYYMMDD_HHMMSS.pkl  # Pickle format (fast loading)
│       └── batch_YYYYMMDD_HHMMSS.pt   # PyTorch tensor format
```

## Data Contents

Each data point contains:

### Input Features:
- **RGB Image**: 640x480x3 camera image (normalized to [0,1])
- **Sensor Data**: IMU (x,y,z), encoders (left,right), battery, temperature
- **VLM Action**: Categorical action chosen by VLM (1-5)
- **Random Distance**: Scaling factor (1.0 + random 0-3 meters)

### Target Labels:
- **State of Charge**: Battery level prediction 3 seconds in the future
- **Policy Distance**: Optimal distance scaling factor for policy learning

## Data Collection

Data is collected by running:
```bash
ros2 launch survival_bot_nodes vlm_navigation_random.launch.py
```

The data collection node automatically saves batches every 30 seconds to prevent data loss.

## Usage in Training

Load the data using `train.py`:
```python
from train import SurvivalBotDataset, load_training_data

# Load data
dataset = load_training_data('train/data/session_20241210_143022')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
```

Install dependencies with:
```bash
pip install -r requirements-dev.txt