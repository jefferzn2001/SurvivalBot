#!/usr/bin/env python3
"""
SurvivalBot Neural Network Training Script
Trains a model to predict state of charge and optimal distance scaling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import cv2
import os
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

class SurvivalBotDataset(Dataset):
    """Dataset for SurvivalBot training data"""
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Path to session directory (e.g., 'data/session_20241210_143022')
            transform: Optional transform to be applied on images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load all CSV files in the data directory
        data_files = list(self.data_dir.glob('data/batch_*.csv'))
        if not data_files:
            raise ValueError(f"No data files found in {data_dir}")
        
        # Combine all CSV files
        dfs = []
        for file in data_files:
            df = pd.read_csv(file)
            dfs.append(df)
        
        self.df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(self.df)} samples from {len(data_files)} files")
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),  # Standard size for CNN
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """Get a single training sample"""
        row = self.df.iloc[idx]
        
        # Load image
        image_path = self.data_dir / 'images' / row['image_filename']
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        image = self.transform(image)
        
        # Extract sensor data (flatten all sensor columns)
        sensor_cols = [col for col in self.df.columns if col.startswith('sensor_')]
        sensor_data = torch.tensor([row[col] for col in sensor_cols], dtype=torch.float32)
        
        # VLM action (categorical, convert to one-hot)
        vlm_action = int(row.get('vlm_action', 3))  # Default to action 3 if missing
        vlm_action_onehot = torch.zeros(5)
        if 1 <= vlm_action <= 5:
            vlm_action_onehot[vlm_action - 1] = 1.0
        
        # Random distance scaling factor
        random_distance = torch.tensor(row.get('random_distance', 1.0), dtype=torch.float32)
        
        # Target labels (mock for now - you'll need to compute these)
        # State of charge 3 seconds in the future
        current_battery = row.get('sensor_battery', 12.0)
        future_soc = torch.tensor(current_battery - 0.1, dtype=torch.float32)  # Mock: slight decrease
        
        # Optimal distance scaling (for policy learning)
        optimal_distance = torch.tensor(2.0, dtype=torch.float32)  # Mock: 2 meters optimal
        
        return {
            'image': image,
            'sensor_data': sensor_data,
            'vlm_action': vlm_action_onehot,
            'random_distance': random_distance,
            'future_soc': future_soc,
            'optimal_distance': optimal_distance
        }

class SurvivalBotCNN(nn.Module):
    """CNN model for SurvivalBot state prediction and policy learning"""
    
    def __init__(self, sensor_dim=10, num_vlm_actions=5):
        super(SurvivalBotCNN, self).__init__()
        
        # CNN for RGB image processing
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        
        # Feature dimensions
        self.cnn_output_dim = 256
        self.sensor_dim = sensor_dim
        self.vlm_action_dim = num_vlm_actions
        self.random_distance_dim = 1
        
        # Combined feature dimension
        self.combined_dim = (self.cnn_output_dim + self.sensor_dim + 
                           self.vlm_action_dim + self.random_distance_dim)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Output heads
        self.soc_head = nn.Linear(128, 1)  # State of charge prediction
        self.policy_head = nn.Linear(128, 1)  # Optimal distance prediction
        
    def forward(self, image, sensor_data, vlm_action, random_distance):
        """Forward pass"""
        # Process image through CNN
        cnn_features = self.cnn(image)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)  # Flatten
        
        # Concatenate all features
        combined_features = torch.cat([
            cnn_features,
            sensor_data,
            vlm_action,
            random_distance.unsqueeze(1)
        ], dim=1)
        
        # Process through fully connected layers
        features = self.fc(combined_features)
        
        # Output predictions
        soc_pred = self.soc_head(features)
        distance_pred = self.policy_head(features)
        
        return soc_pred, distance_pred

class PolicyDistanceSelector:
    """Policy for selecting optimal distance scaling factor"""
    
    def __init__(self, model, num_candidates=10):
        self.model = model
        self.num_candidates = num_candidates
        
    def select_distance(self, image, sensor_data, vlm_action):
        """Select optimal distance using argmax over candidate distances"""
        self.model.eval()
        
        # Generate candidate distances (1 + 0-3 random scaling)
        candidate_distances = torch.tensor([
            1.0 + i * 3.0 / (self.num_candidates - 1) 
            for i in range(self.num_candidates)
        ], dtype=torch.float32)
        
        # Evaluate each candidate
        with torch.no_grad():
            batch_size = image.size(0)
            best_distances = []
            
            for b in range(batch_size):
                img_batch = image[b:b+1].repeat(self.num_candidates, 1, 1, 1)
                sensor_batch = sensor_data[b:b+1].repeat(self.num_candidates, 1)
                vlm_batch = vlm_action[b:b+1].repeat(self.num_candidates, 1)
                
                # Get predictions for all candidates
                _, distance_scores = self.model(img_batch, sensor_batch, vlm_batch, candidate_distances)
                
                # Select best distance (argmax)
                best_idx = torch.argmax(distance_scores)
                best_distances.append(candidate_distances[best_idx])
        
        return torch.stack(best_distances)

def load_training_data(data_dir):
    """Load training dataset"""
    return SurvivalBotDataset(data_dir)

def train_model(data_dir, epochs=100, batch_size=16, learning_rate=0.001):
    """Train the SurvivalBot model"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Load dataset
    dataset = load_training_data(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    sensor_dim = len([col for col in dataset.df.columns if col.startswith('sensor_')])
    model = SurvivalBotCNN(sensor_dim=sensor_dim).to(device)
    
    # Loss functions
    soc_criterion = nn.MSELoss()
    distance_criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Training loop
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch in progress_bar:
            # Move data to device
            image = batch['image'].to(device)
            sensor_data = batch['sensor_data'].to(device)
            vlm_action = batch['vlm_action'].to(device)
            random_distance = batch['random_distance'].to(device)
            future_soc = batch['future_soc'].to(device)
            optimal_distance = batch['optimal_distance'].to(device)
            
            # Forward pass
            soc_pred, distance_pred = model(image, sensor_data, vlm_action, random_distance)
            
            # Compute losses
            soc_loss = soc_criterion(soc_pred.squeeze(), future_soc)
            distance_loss = distance_criterion(distance_pred.squeeze(), optimal_distance)
            total_loss = soc_loss + distance_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            progress_bar.set_postfix({'Loss': total_loss.item():.4f})
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'checkpoints/model_epoch_{epoch+1}.pth'
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
    
    # Save final model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/survival_bot_final.pth')
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()
    
    return model

def test_policy_selector(model_path, data_dir):
    """Test the policy distance selector"""
    # Load model
    model = SurvivalBotCNN()
    model.load_state_dict(torch.load(model_path))
    
    # Create policy selector
    policy = PolicyDistanceSelector(model)
    
    # Test with sample data
    dataset = load_training_data(data_dir)
    sample = dataset[0]
    
    # Add batch dimension
    image = sample['image'].unsqueeze(0)
    sensor_data = sample['sensor_data'].unsqueeze(0)
    vlm_action = sample['vlm_action'].unsqueeze(0)
    
    # Select optimal distance
    optimal_distance = policy.select_distance(image, sensor_data, vlm_action)
    print(f"Selected optimal distance: {optimal_distance.item():.2f} meters")
    
    return policy

if __name__ == "__main__":
    # Example usage
    data_dir = "data/session_20241210_143022"  # Replace with actual session directory
    
    if os.path.exists(data_dir):
        print("Starting training...")
        model = train_model(data_dir, epochs=50, batch_size=8)
        print("Training completed!")
        
        # Test policy selector
        print("Testing policy selector...")
        policy = test_policy_selector('models/survival_bot_final.pth', data_dir)
    else:
        print(f"Data directory {data_dir} not found.")
        print("Run data collection first:")
        print("ros2 launch survival_bot_nodes vlm_navigation_random.launch.py")
        print("\nMake sure you have installed all dependencies:")
        print("pip install -r ../requirements-dev.txt") 