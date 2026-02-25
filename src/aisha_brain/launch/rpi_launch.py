"""AI-SHA Pi 5 Launch — Audio I/O nodes only.

Runs tts_node and (optionally) stt_node on the Raspberry Pi 5
with I2S HAT audio.

Usage:
  ros2 launch aisha_brain rpi_launch.py
  ros2 launch aisha_brain rpi_launch.py enable_stt:=true
  ros2 launch aisha_brain rpi_launch.py audio_device:=plughw:1,0

Set the FastDDS profile externally before launching:
  export FASTRTPS_DEFAULT_PROFILES_FILE=.../config/fastdds_rpi.xml
"""
import os
import signal
import subprocess
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_AISHA_EXECUTABLES = ['tts_node', 'stt_node']


def _kill_existing_nodes():
    """Kill any lingering tts/stt node processes before launching fresh."""
    killed = []
    for proc in subprocess.run(
        ['pgrep', '-a', '-f', '|'.join(_AISHA_EXECUTABLES)],
        capture_output=True, text=True
    ).stdout.splitlines():
        parts = proc.split(None, 1)
        if len(parts) < 2:
            continue
        pid_str, cmd = parts
        if int(pid_str) == os.getpid():
            continue
        try:
            os.kill(int(pid_str), signal.SIGTERM)
            killed.append(pid_str)
        except ProcessLookupError:
            pass
    if killed:
        import time as _time
        _time.sleep(1.5)
        for pid_str in killed:
            try:
                os.kill(int(pid_str), signal.SIGKILL)
            except ProcessLookupError:
                pass
        print(f'[rpi_launch] Cleaned up {len(killed)} stale node(s)')


def generate_launch_description():
    _kill_existing_nodes()

    def create_nodes(context):
        nodes = [
            # TTS Node — Piper TTS + aplay via I2S HAT
            Node(
                package='aisha_brain',
                executable='tts_node',
                name='ai_sha_tts',
                output='screen',
                parameters=[{
                    'voice_model': context.perform_substitution(
                        LaunchConfiguration('voice_model')),
                    'audio_device': context.perform_substitution(
                        LaunchConfiguration('audio_device')),
                }]
            ),
        ]

        # STT Node — optional (default: enabled on Pi)
        if context.perform_substitution(LaunchConfiguration('enable_stt')) == 'true':
            nodes.append(Node(
                package='aisha_brain',
                executable='stt_node',
                name='ai_sha_stt',
                output='screen',
                parameters=[{
                    'whisper_model': context.perform_substitution(
                        LaunchConfiguration('whisper_model')),
                    'whisper_device': context.perform_substitution(
                        LaunchConfiguration('whisper_device')),
                    'whisper_compute_type': context.perform_substitution(
                        LaunchConfiguration('whisper_compute_type')),
                    'input_device': context.perform_substitution(
                        LaunchConfiguration('stt_input_device')),
                    'language': 'en',
                    'silence_threshold': 0.02,
                    'chunk_duration': 5.0,
                    'sample_rate': 16000,
                }]
            ))

        return nodes

    return LaunchDescription([
        # I2S HAT is typically card 0 on a Pi 5 with no other audio
        DeclareLaunchArgument('audio_device',    default_value='plughw:0,0'),
        DeclareLaunchArgument('voice_model',     default_value='en_US-amy-low.onnx'),
        DeclareLaunchArgument('enable_stt',      default_value='true'),
        DeclareLaunchArgument('whisper_model',   default_value='base'),
        DeclareLaunchArgument('whisper_device',  default_value='cpu'),
        DeclareLaunchArgument('whisper_compute_type', default_value='int8'),
        DeclareLaunchArgument('stt_input_device', default_value=''),

        OpaqueFunction(function=create_nodes),
    ])
