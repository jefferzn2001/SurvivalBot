#!/usr/bin/env python3
"""
State Processing for Robot Navigation
Processes raw data into clean state representations at camera capture points
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from FakeSolar import panel_current
import glob

class BatteryState:
    """Battery state estimation based on voltage during idle state"""
    
    # Voltage to SOC mapping (calibrated values)
    VOLTAGE_SOC_MAP = {
        13.6: 100,  # Full charge
        13.4: 90,
        13.3: 80,
        13.2: 70,   # 13.21V should be around here
        13.1: 60,
        13.0: 50,
        12.9: 40,
        12.8: 30,
        12.7: 20,
        12.6: 10,
        12.0: 0     # Empty
    }
    
    @classmethod
    def voltage_to_soc(cls, voltage: float) -> int:
        """
        Convert battery voltage to State of Charge (SOC) percentage.
        Uses linear interpolation between known voltage points.
        Only valid during idle state (no load).
        
        Args:
            voltage: Battery voltage (V)
            
        Returns:
            Estimated SOC percentage (0-100)
        """
        voltages = sorted(cls.VOLTAGE_SOC_MAP.keys(), reverse=True)  # Sort high to low
        
        # Handle out of range
        if voltage >= voltages[0]:
            return cls.VOLTAGE_SOC_MAP[voltages[0]]
        if voltage <= voltages[-1]:
            return cls.VOLTAGE_SOC_MAP[voltages[-1]]
            
        # Find surrounding voltage points
        for i in range(len(voltages) - 1):
            v1, v2 = voltages[i], voltages[i + 1]
            if v1 >= voltage >= v2:
                soc1, soc2 = cls.VOLTAGE_SOC_MAP[v1], cls.VOLTAGE_SOC_MAP[v2]
                
                # Linear interpolation
                soc = soc1 + (voltage - v1) * (soc2 - soc1) / (v2 - v1)
                return int(round(soc))
        
        return 0  # Fallback

class StateProcessor:
    """Process raw data into clean state representations"""
    
    def __init__(self, raw_state_path="raw_state.csv"):
        """
        Initialize state processor
        
        Args:
            raw_state_path: Path to output raw state CSV
        """
        self.raw_state_path = raw_state_path
        self.raw_state = self._create_raw_state_df()
        
    def _create_raw_state_df(self) -> pd.DataFrame:
        """Create new raw state dataframe"""
        return pd.DataFrame(columns=[
            'timestamp',           # Original timestamp
            'time_category',       # Hour category (0-23)
            'soc',                # Battery SOC from voltage
            'ldr_avg',            # Average of left and right LDR
            'solar_in',           # Estimated solar panel input current (A)
            'current_out',        # Average output current during movement (A)
            'temperature',        # °C at state instance
            'humidity',           # %RH at state instance
            'pressure',           # hPa at state instance
            'roll',              # degrees at state instance
            'pitch',             # degrees at state instance
            'ldr_left',          # Left LDR at state instance
            'ldr_right',         # Right LDR at state instance
            'bumper_hit',        # Whether any bumper was hit (1) or not (0)
            'encoder_left',      # Left encoder at last movement
            'encoder_right',     # Right encoder at last movement
            'action',            # Next action taken (0-5)
            'image_file'         # Image filename
        ])
    
    def _get_time_category(self, timestamp: float) -> int:
        """Convert decimal hour timestamp to hour category (0-23)"""
        hour = int(timestamp)  # Integer part is the hour
        return hour % 24  # Ensure it's 0-23
    
    def _calculate_movement_current(self, raw_df, start_idx, end_idx) -> float:
        """Calculate average output current during movement period"""
        # Get data between start and end index where motion isn't 'stop'
        movement_data = raw_df.loc[start_idx:end_idx]
        movement_data = movement_data[movement_data['motion'] != 'stop']
        
        if movement_data.empty:
            return 0.0
            
        return movement_data['power_out_current'].mean()
    
    def _check_bumper_activation(self, raw_df, start_idx, end_idx) -> int:
        """Check if any bumper was activated between start and end index"""
        bumper_data = raw_df.loc[start_idx:end_idx]
        # Check if any bumper (top, bottom, left, right) was activated
        bumper_hit = (
            (bumper_data['bumper_top'] > 0) |
            (bumper_data['bumper_bottom'] > 0) |
            (bumper_data['bumper_left'] > 0) |
            (bumper_data['bumper_right'] > 0)
        ).any()
        return 1 if bumper_hit else 0

    def _get_last_encoder_values(self, raw_df, start_idx, end_idx) -> tuple:
        """Get encoder values from the last moving point"""
        movement_data = raw_df.loc[start_idx:end_idx]
        # Get the last row where motion wasn't 'stop'
        last_moving = movement_data[movement_data['motion'] != 'stop']
        if not last_moving.empty:
            last_row = last_moving.iloc[-1]
            return last_row['encoder_left'], last_row['encoder_right']
        return 0, 0  # Default if no movement found
    
    def process_raw_data(self, raw_data_path: str):
        """
        Process raw data file and extract state information at state_instance=1 points
        
        Args:
            raw_data_path: Path to raw data CSV
        """
        print("Processing raw data...")
        
        # Clear existing raw state
        self.raw_state = self._create_raw_state_df()
        
        # Load raw data
        raw_df = pd.read_csv(raw_data_path)
        
        # Find all state instances (where state_instance = 1)
        state_instances = raw_df[raw_df['state_instance'] == 1].copy()
        
        print(f"\nFound {len(state_instances)} state instances")
        
        for idx, row in state_instances.iterrows():
            try:
                # Calculate LDR average and solar input
                ldr_avg = (float(row['ldr_left']) + float(row['ldr_right'])) / 2.0
                solar_in = panel_current(ldr_avg)
                
                # Calculate battery SOC from voltage - ensure proper float conversion
                voltage = float(row['power_out_voltage'])
                soc = BatteryState.voltage_to_soc(voltage)
                
                # Get state at this instance
                state = {
                    'timestamp': row['timestamp'],
                    'time_category': self._get_time_category(row['timestamp']),
                    'soc': soc,
                    'ldr_avg': ldr_avg,
                    'solar_in': solar_in,
                    'temperature': row['temperature'],
                    'humidity': row['humidity'],
                    'pressure': row['pressure'],
                    'roll': row['imu_roll'],
                    'pitch': row['imu_pitch'],
                    'ldr_left': row['ldr_left'],
                    'ldr_right': row['ldr_right']
                }
                
                # Find the next action taken after this state instance
                next_actions = raw_df[
                    (raw_df.index > idx) & 
                    (raw_df['action_state'].str.match(r'[0-5]', na=False))  # Match any single digit 0-5
                ]
                
                if not next_actions.empty:
                    next_action = int(next_actions.iloc[0]['action_state'])
                    state['action'] = next_action
                    
                    # Find the next state instance to bound our calculations
                    next_state_instance = raw_df[
                        (raw_df.index > idx) & 
                        (raw_df['state_instance'] == 1)
                    ]
                    
                    if not next_state_instance.empty:
                        end_idx = next_state_instance.index[0]
                    else:
                        end_idx = raw_df.index[-1]
                    
                    # Calculate average output current during movement
                    state['current_out'] = self._calculate_movement_current(
                        raw_df, 
                        next_actions.index[0],  # Start from when action begins
                        end_idx                 # Until next state instance or end
                    )
                    
                    # Check for bumper activation
                    state['bumper_hit'] = self._check_bumper_activation(
                        raw_df,
                        next_actions.index[0],  # Start from when action begins
                        end_idx                 # Until next state instance or end
                    )
                    
                    # Get last encoder values
                    enc_left, enc_right = self._get_last_encoder_values(
                        raw_df,
                        next_actions.index[0],  # Start from when action begins
                        end_idx                 # Until next state instance or end
                    )
                    state['encoder_left'] = enc_left
                    state['encoder_right'] = enc_right
                    
                else:
                    print(f"Warning: No action found after state instance at timestamp {row['timestamp']}")
                    continue
                
                # Extract cycle number and timestamp for image matching
                state['image_file'] = f"cycle_{row['timestamp']}.jpg"  # Simplified - adjust if needed
                
                print(f"\nState recorded at {row['timestamp']}:")
                print(f"  Time Category: {state['time_category']}")
                print(f"  Battery SOC: {state['soc']}% (Voltage: {voltage:.2f}V)")
                print(f"  LDR Avg: {state['ldr_avg']:.1f}")
                print(f"  Solar In: {state['solar_in']:.2f}A")
                print(f"  Current Out: {state['current_out']:.2f}A")
                print(f"  Bumper Hit: {state['bumper_hit']}")
                print(f"  Last Encoders: L={state['encoder_left']} R={state['encoder_right']}")
                print(f"  IMU: roll={state['roll']:.2f}° pitch={state['pitch']:.2f}°")
                print(f"  Next Action: {state['action']}")
                
                # Append to raw state
                self.raw_state = pd.concat([
                    self.raw_state,
                    pd.DataFrame([state])
                ], ignore_index=True)
                
            except Exception as e:
                print(f"Error processing state instance at index {idx}: {e}")
                print(f"Row data: {row}")
    
    def save_raw_state(self):
        """Save raw state to CSV"""
        self.raw_state.to_csv(self.raw_state_path, index=False)
        print(f"\nRaw state saved to: {self.raw_state_path}")
        
        # Display all columns in summary
        pd.set_option('display.max_columns', None)
        print("\nRaw State Summary:")
        print("-" * 40)
        print(f"Total states recorded: {len(self.raw_state)}")
        if not self.raw_state.empty:
            print("\nFirst few entries:")
            print(self.raw_state.head().to_string())
            
            print("\nColumns and their statistics:")
            print(self.raw_state.describe().to_string())

def main():
    """Process raw data into state representations"""
    processor = StateProcessor()
    
    # Automatically find the latest raw_data.csv in the data/ directory
    data_dir = os.path.dirname(os.path.abspath(__file__))
    session_dirs = glob.glob(os.path.join(data_dir, "data_*/raw_data.csv"))
    if not session_dirs:
        print("Error: No raw_data.csv files found in any session directory.")
        return
    # Sort by session timestamp in folder name
    session_dirs.sort()
    raw_data_path = session_dirs[-1]
    print(f"Processing raw data from: {raw_data_path}")

    try:
        processor.process_raw_data(raw_data_path)
        processor.save_raw_state()
        
    except FileNotFoundError:
        print(f"Error: Could not find raw data file: {raw_data_path}")
        print("Please provide the correct path to your raw_data.csv file")
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    main()
