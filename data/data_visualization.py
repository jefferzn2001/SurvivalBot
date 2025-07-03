#!/usr/bin/env python3
"""
Data Visualization Script for VLM Navigation Sessions
Creates multi-panel, research-quality graphs showing camera events, power data, and LDR readings.
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
    List all available session directories, sorted by timestamp (most recent last).
    """
    sessions = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item.startswith('data_'):
            sessions.append(item_path)
    # Sort by folder name (timestamp)
    return sorted(sessions)


def load_session_data(session_path):
    """
    Load raw data and reasoning data from a session directory.
    """
    raw_data_file = os.path.join(session_path, 'raw_data.csv')
    reasoning_file = os.path.join(session_path, 'reasoning.csv')
    if not os.path.exists(raw_data_file):
        print(f"❌ Raw data file not found: {raw_data_file}")
        return None, None
    if not os.path.exists(reasoning_file):
        print(f"❌ Reasoning file not found: {reasoning_file}")
        return None, None
    try:
        raw_data = pd.read_csv(raw_data_file)
        reasoning = pd.read_csv(reasoning_file)
        return raw_data, reasoning
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None


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


def create_visualization(raw_data, reasoning_data, session_name, output_path=None):
    """
    Create multi-panel, research-quality visualization of the navigation session.
    """
    # Convert timestamps
    raw_data = convert_timestamp_to_datetime(raw_data)
    reasoning_data = convert_reasoning_timestamp(reasoning_data)
    
    # Compute seconds since start for x-axis
    t0 = raw_data['datetime'].iloc[0]
    raw_data['seconds_since_start'] = (raw_data['datetime'] - t0).dt.total_seconds()
    reasoning_data['seconds_since_start'] = (reasoning_data['datetime'] - t0).dt.total_seconds()
    
    # Create figure with subplots
    fig, axes = plt.subplots(5, 1, figsize=(18, 17), sharex=True, gridspec_kw={'hspace': 0.25})
    fig.suptitle(f'VLM Navigation Session Analysis\nSession: {session_name}', fontsize=22, fontweight='bold', y=0.98)
    
    # Get time range for x-axis
    time_min = 0
    time_max = raw_data['seconds_since_start'].max()
    
    # 1. VLM Decisions and Bumper Events (Top Panel)
    ax1 = axes[0]
    for i, row in reasoning_data.iterrows():
        ax1.axvline(x=row['seconds_since_start'], color='crimson', alpha=0.7, linewidth=2, zorder=2)
        ax1.annotate(f"A{row['action']}",
                     xy=(row['seconds_since_start'], 0.5),
                     xytext=(0, 10), textcoords='offset points',
                     fontsize=13, fontweight='bold', ha='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='gold', alpha=0.8), zorder=3)
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
    bumper_legend_handles = []
    for bumper, color in bumper_colors.items():
        if bumper in raw_data.columns:
            events = raw_data[raw_data[bumper] > 0]
            for _, event in events.iterrows():
                line = ax1.axvline(x=event['seconds_since_start'], color=color, alpha=0.8, linewidth=2, linestyle='--', zorder=1)
                ax1.annotate(bumper_labels[bumper],
                             xy=(event['seconds_since_start'], 0.7),
                             xytext=(0, 10), textcoords='offset points',
                             fontsize=11, fontweight='bold', ha='center', color=color,
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.7), zorder=4)
                if len(bumper_legend_handles) < 4:
                    bumper_legend_handles.append((line, bumper_labels[bumper]))
    ax1.set_ylabel('VLM Decision')
    ax1.set_yticks([])
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    # Legend for VLM and bumpers (outside plot)
    legend_elements = [Line2D([0], [0], color='crimson', lw=2, label='VLM Action')]
    for color, label in zip(bumper_colors.values(), bumper_labels.values()):
        legend_elements.append(Line2D([0], [0], color=color, lw=2, linestyle='--', label=f'Bumper {label}'))
    ax1.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=12, frameon=True)
    
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
    
    # 3. Net Current (Third Panel)
    ax3 = axes[2]
    if 'power_out_current' in raw_data.columns and 'Solar_In' in raw_data.columns:
        # Calculate net current: Solar_In (positive) - power_out_current (negative)
        net_current = raw_data['Solar_In'] - raw_data['power_out_current']
        # Calculate cumulative net current starting at 0
        cumulative_net = net_current.cumsum()
        ax3.plot(raw_data['seconds_since_start'], cumulative_net,
                 label='Cumulative Net Current', color='red', linewidth=2)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax3.legend(frameon=True, fontsize=12, loc='center left', bbox_to_anchor=(1.01, 0.5))
    else:
        ax3.text(0.5, 0.5, 'Power data missing', ha='center', va='center', fontsize=14, color='red')
    ax3.set_ylabel('Net Current (A)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Encoder Data (Fourth Panel)
    ax4 = axes[3]
    if 'encoder_left' in raw_data.columns and 'encoder_right' in raw_data.columns:
        ax4.plot(raw_data['seconds_since_start'], raw_data['encoder_left'],
                 label='Encoder Left', color=sns.color_palette()[2], linewidth=2)
        ax4.plot(raw_data['seconds_since_start'], raw_data['encoder_right'],
                 label='Encoder Right', color=sns.color_palette()[3], linewidth=2)
        ax4.legend(frameon=True, fontsize=12, loc='center left', bbox_to_anchor=(1.01, 0.5))
    else:
        ax4.text(0.5, 0.5, 'Encoder data missing', ha='center', va='center', fontsize=14, color='red')
    ax4.set_ylabel('Encoder Value')
    ax4.grid(True, alpha=0.3)
    
    # 5. Action States (Fifth Panel)
    ax5 = axes[4]
    action_states = raw_data['action_state'].copy()
    state_mapping = {'idle': 0, '0': 1}
    for i in range(1, 6):
        state_mapping[str(i)] = i + 1
    numeric_states = action_states.map(state_mapping)
    ax5.step(raw_data['seconds_since_start'], numeric_states, where='post',
             label='Action State', color='black', linewidth=2)
    for state_num in range(7):
        if state_num in numeric_states.values:
            ax5.axhline(y=state_num, color='gray', alpha=0.2, linestyle='--', linewidth=1)
    ax5.set_ylabel('Action State')
    ax5.set_ylim(-0.5, 6.5)
    ax5.set_yticks(range(7))
    ax5.set_yticklabels(['Idle', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5'])
    ax5.grid(True, alpha=0.3)
    ax5.legend(['Action State'], frameon=True, fontsize=12, loc='center left', bbox_to_anchor=(1.01, 0.5))
    
    # --- Smart X-axis scaling ---
    total_seconds = time_max - time_min
    if total_seconds < 120:  # Less than 2 minutes
        major_tick = 1 if total_seconds < 20 else 5 if total_seconds < 60 else 10
        ax5.xaxis.set_major_locator(plt.MultipleLocator(major_tick))
        ax5.xaxis.set_minor_locator(plt.MultipleLocator(1))
        ax5.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
        x_label = 'Time (seconds)'
    elif total_seconds < 3600:  # Less than 1 hour
        major_tick = 0.5 * 60 if total_seconds < 600 else 1 * 60 if total_seconds < 1800 else 2 * 60
        ax5.xaxis.set_major_locator(plt.MultipleLocator(major_tick))
        ax5.xaxis.set_minor_locator(plt.MultipleLocator(30))
        ax5.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x//60)}:{int(x%60):02d}'))
        x_label = 'Time (minutes)'
    else:
        major_tick = 10 * 60
        ax5.xaxis.set_major_locator(plt.MultipleLocator(major_tick))
        ax5.xaxis.set_minor_locator(plt.MultipleLocator(60))
        ax5.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x//3600)}:{int((x%3600)//60):02d}'))
        x_label = 'Time (hours:minutes)'
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=13)
    for ax in axes:
        ax.tick_params(axis='y', labelsize=13)
        ax.set_xlim(time_min, time_max)
    ax5.set_xlabel(x_label, fontsize=14)
    
    # Add a timestamp to the figure
    fig.text(0.99, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
             ha='right', va='bottom', fontsize=11, color='gray')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize VLM Navigation Session Data')
    parser.add_argument('--session', type=str, help='Specific session directory path')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory path')
    parser.add_argument('--output', type=str, help='Output path for the plot')
    parser.add_argument('--list', action='store_true', help='List available sessions')
    args = parser.parse_args()

    # List available sessions if requested
    if args.list:
        sessions = list_available_sessions(args.data_dir)
        if sessions:
            print("📁 Available sessions:")
            for i, session in enumerate(sessions, 1):
                session_name = os.path.basename(session)
                print(f"  {i}. {session_name}")
        else:
            print("❌ No session directories found")
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
            print("❌ No session directories found")
            return
        session_path = sessions[-1]  # Most recent
        print(f"📁 Using most recent session: {os.path.basename(session_path)}")

    # Load data
    raw_data, reasoning_data = load_session_data(session_path)
    if raw_data is None or reasoning_data is None:
        print("❌ Failed to load session data")
        return

    # Create output path if not specified
    if not args.output:
        session_name = os.path.basename(session_path)
        args.output = f"{session_name}_analysis.png"

    # Create visualization
    print(f"📊 Creating visualization for session: {os.path.basename(session_path)}")
    create_visualization(raw_data, reasoning_data, os.path.basename(session_path), args.output)

if __name__ == '__main__':
    main() 