#!/bin/bash
# AI-SHA Brain — quick launch script
# Usage: ./launch.sh [extra launch args]
# Example: ./launch.sh enable_stt:=true audio_device:=plughw:0,0

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

source /opt/ros/jazzy/setup.bash
source "$SCRIPT_DIR/install/setup.bash"
export PYTHONPATH="$SCRIPT_DIR/venv/lib/python3.12/site-packages:$PYTHONPATH"

echo "AI-SHA Brain launching..."
ros2 launch aisha_brain aisha_launch.py "$@"
