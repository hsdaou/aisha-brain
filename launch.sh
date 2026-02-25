#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# AI-SHA Brain — multi-device launch script
#
# Usage:
#   ./launch.sh                            # all nodes locally (default)
#   ./launch.sh --target rpi               # Pi 5:  tts + stt
#   ./launch.sh --target jetson            # Jetson: brain + admin + action
#   ./launch.sh --target ws                # Workstation: whatsapp_listener
#   ./launch.sh --target all               # all nodes locally
#
# Extra ROS2 launch args are passed through:
#   ./launch.sh --target rpi enable_stt:=true audio_device:=plughw:1,0
#   ./launch.sh --target ws  dev_mode:=true
# ──────────────────────────────────────────────────────────────────────────────
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Parse --target argument ───────────────────────────────────────────────────
TARGET="all"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            TARGET="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ── Source ROS2 and workspace ─────────────────────────────────────────────────
source /opt/ros/jazzy/setup.bash
source "$SCRIPT_DIR/install/setup.bash"
export PYTHONPATH="$SCRIPT_DIR/venv/lib/python3.12/site-packages:$PYTHONPATH"

# ── Add Piper binary to PATH if installed locally ────────────────────────────
if [ -d "$SCRIPT_DIR/piper" ]; then
    export PATH="$SCRIPT_DIR/piper:$PATH"
fi

# ── Select launch file and FastDDS profile ────────────────────────────────────
CONFIG_DIR="$(ros2 pkg prefix aisha_brain)/share/aisha_brain/config"

case "$TARGET" in
    rpi)
        LAUNCH_FILE="rpi_launch.py"
        export FASTRTPS_DEFAULT_PROFILES_FILE="$CONFIG_DIR/fastdds_rpi.xml"
        ;;
    jetson)
        LAUNCH_FILE="jetson_launch.py"
        export FASTRTPS_DEFAULT_PROFILES_FILE="$CONFIG_DIR/fastdds_jetson.xml"
        ;;
    ws)
        LAUNCH_FILE="ws_launch.py"
        export FASTRTPS_DEFAULT_PROFILES_FILE="$CONFIG_DIR/fastdds_ws.xml"
        ;;
    all)
        LAUNCH_FILE="aisha_launch.py"
        # No FastDDS profile — local-only uses default multicast discovery
        ;;
    *)
        echo "ERROR: Unknown target '$TARGET'. Use: rpi, jetson, ws, or all"
        exit 1
        ;;
esac

echo "AI-SHA Brain launching [target=$TARGET] ..."
if [ -n "${FASTRTPS_DEFAULT_PROFILES_FILE:-}" ]; then
    echo "  FastDDS profile: $FASTRTPS_DEFAULT_PROFILES_FILE"
fi

ros2 launch aisha_brain "$LAUNCH_FILE" "${EXTRA_ARGS[@]}"
