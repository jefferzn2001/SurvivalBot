#!/usr/bin/env python3
"""
Policy Data Visualization Script for VLM Policy Navigation Sessions
Creates multi-panel, research-quality graphs showing VLM decisions, policy decisions, 
distance scaling, stop certainty, and sensor data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
import argparse
import numpy as np
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.lines import Line2D

# Use a modern, clean style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context('talk')
sns.set_palette('colorblind')


def list_available_sessions(data_dir):
    """
    List all available policy session directories, sorted by timestamp (most recent last).
    """
    sessions = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item.startswith('policy_data_'):
            sessions.append(item_path)
    # Sort by folder name (timestamp)
    return sorted(sessions)


def load_session_data(session_path):
    """
    Load raw data, reasoning data, and state data from a policy session directory.
    """
    raw_data_file = os.path.join(session_path, 'raw_data.csv')
    reasoning_file = os.path.join(session_path, 'reasoning.csv')
    state_file = os.path.join(session_path, 'state.csv')
    
    if not os.path.exists(raw_data_file):
        print(f"❌ Raw data file not found: {raw_data_file}")
        return None, None, None
    if not os.path.exists(reasoning_file):
        print(f"❌ Reasoning file not found: {reasoning_file}")
        return None, None, None
    
    try:
        raw_data = pd.read_csv(raw_data_file)
        reasoning = pd.read_csv(reasoning_file)
        
        # State data is optional (might not exist in older sessions)
        state_data = None
        if os.path.exists(state_file):
            state_data = pd.read_csv(state_file)
            print(f"✅ State data loaded: {len(state_data)} entries")
        else:
            print(f"⚠️ State file not found: {state_file}")
        
        return raw_data, reasoning, state_data
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None


def convert_timestamp_to_datetime(raw_data):
    """
    Convert timestamp column from decimal hours to datetime objects.
    """
    df = raw_data.copy()
    base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    df['datetime'] = base_date + pd.to_timedelta(df['timestamp'] * 3600, unit='s')
    return df


def convert_reasoning_timestamp(reasoning_data):
    """
    Convert reasoning timestamp to datetime objects.
    """
    df = reasoning_data.copy()
    df['datetime'] = pd.to_datetime(df['timestamp'])
    return df


def create_policy_visualization(raw_data, reasoning_data, session_name, output_path=None, state_data=None):
    """
    Create multi-panel, research-quality visualization of the policy navigation session.
    """
    # Convert timestamps
    raw_data = convert_timestamp_to_datetime(raw_data)
    reasoning_data = convert_reasoning_timestamp(reasoning_data)
    
    # Compute seconds since start for x-axis
    t0 = raw_data['datetime'].iloc[0]
    raw_data['seconds_since_start'] = (raw_data['datetime'] - t0).dt.total_seconds()
    reasoning_data['seconds_since_start'] = (reasoning_data['datetime'] - t0).dt.total_seconds()
    
    # Create figure with subplots (4 panels instead of 6)
    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True, gridspec_kw={'hspace': 0.3})
    fig.suptitle(f'VLM Policy Navigation Session Analysis\nSession: {session_name}', fontsize=22, fontweight='bold', y=0.98)
    
    # Get time range for x-axis
    time_min = 0
    time_max = raw_data['seconds_since_start'].max()
    
    # 1. VLM Decisions, Policy Inference, and Distance/Certainty (Top Panel)
    ax1 = axes[0]
    
    # Policy inference timing - show when policy decisions are made
    for i, row in reasoning_data.iterrows():
        # VLM action vertical line (crimson, thicker) - shows when VLM action is received
        ax1.axvline(x=row['seconds_since_start'], color='crimson', alpha=0.8, linewidth=3, zorder=2)
        
        # Policy inference vertical line (blue, thinner) - shows when policy inference happens
        # Add small offset to separate from VLM action line
        policy_time = row['seconds_since_start'] + 0.5  # 0.5 second offset
        ax1.axvline(x=policy_time, color='blue', alpha=0.8, linewidth=2, linestyle='-', zorder=1)
        
        # VLM action annotation (top)
        ax1.annotate(f"VLM:A{row['action']}",
                     xy=(row['seconds_since_start'], 0.85),
                     xytext=(0, 10), textcoords='offset points',
                     fontsize=12, fontweight='bold', ha='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='crimson', alpha=0.8), zorder=3)
        
        # Policy distance annotation (middle) - changed from "P" to "Δ"
        distance = row.get('distance_scale', 0.3)
        ax1.annotate(f"δ:{distance:.2f}m",
                     xy=(policy_time, 0.65),
                     xytext=(0, 10), textcoords='offset points',
                     fontsize=10, fontweight='bold', ha='center',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.8), zorder=3)
        
        # Stop certainty annotation (bottom)
        certainty = row.get('stop_certainty', 0.0)
        certainty_color = 'red' if certainty > 0.9 else 'lightgreen'
        ax1.annotate(f"Stop:{certainty:.2f}",
                     xy=(policy_time, 0.45),
                     xytext=(0, 10), textcoords='offset points',
                     fontsize=9, fontweight='bold', ha='center',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor=certainty_color, alpha=0.7), zorder=3)
    
    # Bumper event colors
    bumper_colors = {
        'bumper_top': 'deepskyblue',
        'bumper_bottom': 'orange',
        'bumper_left': 'limegreen',
        'bumper_right': 'magenta',
    }
    bumper_labels = {
        'bumper_top': 'UP',
        'bumper_bottom': 'DOWN',
        'bumper_left': 'LEFT',
        'bumper_right': 'RIGHT',
    }
    
    for bumper, color in bumper_colors.items():
        if bumper in raw_data.columns:
            events = raw_data[raw_data[bumper] > 0]
            for _, event in events.iterrows():
                ax1.axvline(x=event['seconds_since_start'], color=color, alpha=0.8, linewidth=2, linestyle='--', zorder=1)
                ax1.annotate(bumper_labels[bumper],
                             xy=(event['seconds_since_start'], 0.25),
                             xytext=(0, 10), textcoords='offset points',
                             fontsize=10, fontweight='bold', ha='center', color=color,
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.7), zorder=4)
    
    ax1.set_ylabel('Policy Inference & Actions')
    ax1.set_yticks([])
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Legend for policy inference, VLM actions, and bumpers
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, linestyle='-', label='Policy Inference'),
        Line2D([0], [0], color='crimson', lw=3, linestyle='-', label='VLM Action'),
        Line2D([0], [0], color='lightblue', lw=2, label='Distance Scale'),
        Line2D([0], [0], color='lightgreen', lw=2, label='Stop Certainty')
    ]
    for color, label in zip(bumper_colors.values(), bumper_labels.values()):
        legend_elements.append(Line2D([0], [0], color=color, lw=2, linestyle='--', label=f'Bumper {label}'))
    ax1.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=11, frameon=True)
    
    # 2. Power Data (Second Panel)
    ax2 = axes[1]
    if 'power_out_current' in raw_data.columns and 'Solar_In' in raw_data.columns:
        ax2.plot(raw_data['seconds_since_start'], raw_data['power_out_current'],
                 label='Power Out Current (A)', color=sns.color_palette()[0], linewidth=2)
        ax2.plot(raw_data['seconds_since_start'], raw_data['Solar_In'],
                 label='Solar In (A)', color=sns.color_palette()[1], linewidth=2)
        ax2.legend(frameon=True, fontsize=12, loc='center left', bbox_to_anchor=(1.01, 0.5))
    else:
        ax2.text(0.5, 0.5, 'Power data missing', ha='center', va='center', fontsize=14, color='red')
    ax2.set_ylabel('Current (A)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Encoder Data (Third Panel)
    ax3 = axes[2]
    if 'encoder_left' in raw_data.columns and 'encoder_right' in raw_data.columns:
        ax3.plot(raw_data['seconds_since_start'], raw_data['encoder_left'],
                 label='Encoder Left', color=sns.color_palette()[2], linewidth=2)
        ax3.plot(raw_data['seconds_since_start'], raw_data['encoder_right'],
                 label='Encoder Right', color=sns.color_palette()[3], linewidth=2)
        ax3.legend(frameon=True, fontsize=12, loc='center left', bbox_to_anchor=(1.01, 0.5))
    else:
        ax3.text(0.5, 0.5, 'Encoder data missing', ha='center', va='center', fontsize=14, color='red')
    ax3.set_ylabel('Encoder Value')
    ax3.grid(True, alpha=0.3)
    
    # 4. Action States (Fourth Panel)
    ax4 = axes[3]
    action_states = raw_data['action_state'].copy()
    state_mapping = {'idle': 0, '0': 1}
    for i in range(1, 6):
        state_mapping[str(i)] = i + 1
    numeric_states = action_states.map(state_mapping)
    ax4.step(raw_data['seconds_since_start'], numeric_states, where='post',
             label='Action State', color='black', linewidth=2)
    for state_num in range(7):
        if state_num in numeric_states.values:
            ax4.axhline(y=state_num, color='gray', alpha=0.2, linestyle='--', linewidth=1)
    ax4.set_ylabel('Action State')
    ax4.set_ylim(-0.5, 6.5)
    ax4.set_yticks(range(7))
    ax4.set_yticklabels(['Idle', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5'])
    ax4.grid(True, alpha=0.3)
    ax4.legend(['Action State'], frameon=True, fontsize=12, loc='center left', bbox_to_anchor=(1.01, 0.5))
    
    # --- Smart X-axis scaling ---
    total_seconds = time_max - time_min
    if total_seconds < 120:  # Less than 2 minutes
        major_tick = 1 if total_seconds < 20 else 5 if total_seconds < 60 else 10
        ax4.xaxis.set_major_locator(plt.MultipleLocator(major_tick))
        ax4.xaxis.set_minor_locator(plt.MultipleLocator(1))
        ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
        x_label = 'Time (seconds)'
    elif total_seconds < 3600:  # Less than 1 hour
        major_tick = 0.5 * 60 if total_seconds < 600 else 1 * 60 if total_seconds < 1800 else 2 * 60
        ax4.xaxis.set_major_locator(plt.MultipleLocator(major_tick))
        ax4.xaxis.set_minor_locator(plt.MultipleLocator(30))
        ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x//60)}:{int(x%60):02d}'))
        x_label = 'Time (minutes)'
    else:
        major_tick = 10 * 60
        ax4.xaxis.set_major_locator(plt.MultipleLocator(major_tick))
        ax4.xaxis.set_minor_locator(plt.MultipleLocator(60))
        ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x//3600)}:{int((x%3600)//60):02d}'))
        x_label = 'Time (hours:minutes)'
    
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=13)
    for ax in axes:
        ax.tick_params(axis='y', labelsize=13)
        ax.set_xlim(time_min, time_max)
    ax4.set_xlabel(x_label, fontsize=14)
    
    # Add a timestamp to the figure
    fig.text(0.99, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
             ha='right', va='bottom', fontsize=11, color='gray')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Policy plot saved to: {output_path}")
    plt.show()


def print_policy_summary(raw_data, reasoning_data, session_name):
    """
    Print a summary of policy decisions and performance.
    """
    print(f"\n📊 Policy Session Summary: {session_name}")
    print("=" * 60)
    
    # Convert timestamps and calculate seconds_since_start
    raw_data = convert_timestamp_to_datetime(raw_data)
    reasoning_data = convert_reasoning_timestamp(reasoning_data)
    
    # Compute seconds since start
    t0 = raw_data['datetime'].iloc[0]
    raw_data['seconds_since_start'] = (raw_data['datetime'] - t0).dt.total_seconds()
    
    # Basic stats
    total_time = raw_data['seconds_since_start'].max()
    num_cycles = len(reasoning_data)
    
    print(f"Total Session Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Number of VLM Cycles: {num_cycles}")
    
    # Policy distance stats
    if 'distance_scale' in raw_data.columns:
        distances = raw_data['distance_scale'].dropna()
        print(f"\nPolicy Distance Statistics:")
        print(f"  Mean Distance: {distances.mean():.3f}m")
        print(f"  Min Distance: {distances.min():.3f}m")
        print(f"  Max Distance: {distances.max():.3f}m")
        print(f"  Std Distance: {distances.std():.3f}m")
    
    # Stop certainty stats
    if 'stop_certainty' in raw_data.columns:
        certainties = raw_data['stop_certainty'].dropna()
        stop_decisions = (certainties > 0.9).sum()
        print(f"\nStop Certainty Statistics:")
        print(f"  Mean Certainty: {certainties.mean():.3f}")
        print(f"  Max Certainty: {certainties.max():.3f}")
        print(f"  Stop Decisions: {stop_decisions} ({stop_decisions/len(certainties)*100:.1f}% of data points)")
    
    # Action distribution
    if not reasoning_data.empty:
        action_counts = reasoning_data['action'].value_counts().sort_index()
        print(f"\nAction Distribution:")
        for action, count in action_counts.items():
            print(f"  Action {action}: {count} times ({count/num_cycles*100:.1f}%)")
    
    # Bumper events
    bumper_columns = ['bumper_top', 'bumper_bottom', 'bumper_left', 'bumper_right']
    bumper_events = 0
    for col in bumper_columns:
        if col in raw_data.columns:
            bumper_events += (raw_data[col] > 0).sum()
    print(f"\nBumper Events: {bumper_events}")
    
    # Power performance
    if 'Solar_In' in raw_data.columns and 'power_out_current' in raw_data.columns:
        solar_avg = raw_data['Solar_In'].mean()
        power_avg = raw_data['power_out_current'].mean()
        net_current = solar_avg - power_avg
        print(f"\nPower Performance:")
        print(f"  Average Solar Input: {solar_avg:.3f}A")
        print(f"  Average Power Output: {power_avg:.3f}A")
        print(f"  Net Current: {net_current:.3f}A")


def main():
    parser = argparse.ArgumentParser(description='Visualize VLM Policy Navigation Session Data')
    parser.add_argument('--session', type=str, help='Specific session directory path')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory path')
    parser.add_argument('--output', type=str, help='Output path for the plot')
    parser.add_argument('--list', action='store_true', help='List available policy sessions')
    parser.add_argument('--summary', action='store_true', help='Print policy summary statistics')
    args = parser.parse_args()

    # List available sessions if requested
    if args.list:
        sessions = list_available_sessions(args.data_dir)
        if sessions:
            print("📁 Available policy sessions:")
            for i, session in enumerate(sessions, 1):
                session_name = os.path.basename(session)
                print(f"  {i}. {session_name}")
        else:
            print("❌ No policy session directories found")
        return

    # Determine session path
    if args.session:
        session_path = args.session
        if not os.path.exists(session_path):
            print(f"❌ Session path does not exist: {session_path}")
            return
    else:
        # Use most recent session
        sessions = list_available_sessions(args.data_dir)
        if not sessions:
            print("❌ No policy session directories found")
            return
        session_path = sessions[-1]  # Most recent
        print(f"📁 Using most recent policy session: {os.path.basename(session_path)}")

    # Load data
    raw_data, reasoning_data, state_data = load_session_data(session_path)
    if raw_data is None or reasoning_data is None:
        print("❌ Failed to load session data")
        return

    session_name = os.path.basename(session_path)
    
    # Print summary if requested
    if args.summary:
        print_policy_summary(raw_data, reasoning_data, session_name)
        return

    # Create output path if not specified
    if not args.output:
        args.output = f"{session_name}_policy_analysis.png"

    # Create visualization
    print(f"📊 Creating policy visualization for session: {session_name}")
    create_policy_visualization(raw_data, reasoning_data, session_name, args.output, state_data)
    
    # Always print summary after visualization
    print_policy_summary(raw_data, reasoning_data, session_name)


if __name__ == '__main__':
    main() 