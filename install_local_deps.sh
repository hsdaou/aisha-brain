#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# AI-SHA Brain — dependency installer
#
# Sets up a Python venv, installs pip packages, downloads the Piper TTS
# binary + default voice model.  Works on both x86_64 and aarch64.
#
# Usage:  bash install_local_deps.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
PIPER_VERSION="2023.11.14-2"
VOICE_MODEL="en_US-amy-low"
ARCH="$(uname -m)"

# ── 1. Architecture check ────────────────────────────────────────────────────
case "$ARCH" in
    x86_64)  PIPER_TARBALL="piper_linux_x86_64.tar.gz" ;;
    aarch64) PIPER_TARBALL="piper_linux_aarch64.tar.gz" ;;
    *)
        echo "ERROR: Unsupported architecture: $ARCH (need x86_64 or aarch64)"
        exit 1
        ;;
esac
echo "Architecture: $ARCH"

# ── 2. System packages ───────────────────────────────────────────────────────
echo ""
echo "=== Installing system packages ==="
sudo apt-get update -qq
sudo apt-get install -y -qq alsa-utils curl wget python3-venv > /dev/null

# ── 3. Python venv ───────────────────────────────────────────────────────────
echo ""
echo "=== Setting up Python venv ==="
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q

echo "Installing Python packages into venv ..."
pip install -r "$SCRIPT_DIR/src/aisha_brain/requirements.txt" -q
echo "Python packages installed."

# ── 4. Piper TTS binary ──────────────────────────────────────────────────────
echo ""
echo "=== Piper TTS binary ==="
PIPER_DIR="$SCRIPT_DIR/piper"
if [ -x "$PIPER_DIR/piper" ]; then
    echo "Piper binary already installed at $PIPER_DIR/piper"
else
    echo "Downloading Piper TTS $PIPER_VERSION for $ARCH ..."
    PIPER_URL="https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/${PIPER_TARBALL}"
    wget -q --show-progress -O "/tmp/${PIPER_TARBALL}" "$PIPER_URL"
    # The tarball extracts to a piper/ directory
    tar -xzf "/tmp/${PIPER_TARBALL}" -C "$SCRIPT_DIR"
    rm -f "/tmp/${PIPER_TARBALL}"
    chmod +x "$PIPER_DIR/piper"
    echo "Piper binary installed at $PIPER_DIR/piper"
fi

# ── 5. Voice model ───────────────────────────────────────────────────────────
echo ""
echo "=== Piper voice model ==="
VOICE_DIR="$PIPER_DIR/voices"
if [ -f "$VOICE_DIR/${VOICE_MODEL}.onnx" ]; then
    echo "Voice model already downloaded: $VOICE_DIR/${VOICE_MODEL}.onnx"
else
    echo "Downloading voice model: $VOICE_MODEL ..."
    mkdir -p "$VOICE_DIR"
    VOICE_BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/low"
    wget -q --show-progress -O "$VOICE_DIR/${VOICE_MODEL}.onnx" \
        "${VOICE_BASE_URL}/en_US-amy-low.onnx"
    wget -q --show-progress -O "$VOICE_DIR/${VOICE_MODEL}.onnx.json" \
        "${VOICE_BASE_URL}/en_US-amy-low.onnx.json"
    echo "Voice model saved to $VOICE_DIR/"
fi

# ── 6. Summary ───────────────────────────────────────────────────────────────
echo ""
echo "========================================="
echo "  AI-SHA Dependencies Installed"
echo "========================================="
echo "  Arch:   $ARCH"
echo "  Venv:   $VENV_DIR"
echo "  Piper:  $PIPER_DIR/piper"
echo "  Voice:  $VOICE_DIR/${VOICE_MODEL}.onnx"
echo ""
echo "Next steps:"
echo "  cd $SCRIPT_DIR"
echo "  colcon build --packages-select aisha_brain"
echo "  ./launch.sh"
echo ""
