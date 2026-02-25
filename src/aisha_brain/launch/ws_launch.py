"""AI-SHA Workstation Launch — WhatsApp relay and optional full-stack dev mode.

By default, launches only the whatsapp_listener (relay node).
Pass dev_mode:=true to launch ALL nodes locally (same as aisha_launch.py).

Usage:
  ros2 launch aisha_brain ws_launch.py                       # relay only
  ros2 launch aisha_brain ws_launch.py dev_mode:=true        # all nodes (dev)

Set the FastDDS profile externally before launching:
  export FASTRTPS_DEFAULT_PROFILES_FILE=.../config/fastdds_ws.xml
"""
import os
import signal
import subprocess
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# ── Defaults (same as aisha_launch.py) ────────────────────────────────────────
ALLOWED_NUMBER = '971509726902'
MONITORED_JID  = '269655394504744'

_AISHA_EXECUTABLES = ['whatsapp_listener']


def _kill_existing_nodes():
    """Kill stale whatsapp_listener processes."""
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
        print(f'[ws_launch] Cleaned up {len(killed)} stale node(s)')


def generate_launch_description():
    _kill_existing_nodes()

    def create_nodes(context):
        dev_mode = context.perform_substitution(LaunchConfiguration('dev_mode'))

        if dev_mode == 'true':
            # In dev mode, delegate to the full aisha_launch.py
            launch_dir = os.path.dirname(os.path.abspath(__file__))
            return [IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(launch_dir, 'aisha_launch.py')
                )
            )]

        # Production mode: whatsapp_listener only
        allowed_number = context.perform_substitution(
            LaunchConfiguration('allowed_number'))
        monitored_jid = context.perform_substitution(
            LaunchConfiguration('monitored_jid'))

        return [
            Node(
                package='aisha_brain',
                executable='whatsapp_listener',
                name='ai_sha_listener',
                output='screen',
                parameters=[{
                    'allowed_number': str(allowed_number),
                    'monitored_jid': str(monitored_jid),
                    'wa_reply_delay': 0.0,
                    'echo_mute_secs': 8.0,
                }]
            ),
        ]

    return LaunchDescription([
        DeclareLaunchArgument('dev_mode',       default_value='false',
                              description='Launch all nodes locally (dev/testing)'),
        DeclareLaunchArgument('allowed_number', default_value=ALLOWED_NUMBER),
        DeclareLaunchArgument('monitored_jid',  default_value=MONITORED_JID),

        OpaqueFunction(function=create_nodes),
    ])
