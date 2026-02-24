"""AI-SHA Brain Launch File

Launches all aisha_brain nodes. Aligned with the Jetson Orin Nano ROS2
architecture (see ros2_architecture.pdf).

Key topic alignment with the Jetson architecture:
  /speech/text       <- stt_node publishes (Faster-Whisper)
  /user_speech       <- whatsapp_listener publishes (alias, brain also listens here)
  /robot_speech      <- admin_node / brain_node publish (tts_node + whatsapp_listener subscribe)
  /speaker/playing   <- tts_node publishes Bool; stt_node subscribes to mute mic
  /detection/objects_simple  <- detection_node (YOLOv8) publishes; brain_node subscribes

Usage:
  ros2 launch aisha_brain aisha_launch.py
  ros2 launch aisha_brain aisha_launch.py enable_stt:=true
  ros2 launch aisha_brain aisha_launch.py enable_stt:=true whisper_model:=small whisper_device:=cuda
  ros2 launch aisha_brain aisha_launch.py enable_whatsapp:=false

NOTE: To change allowed_number, edit ALLOWED_NUMBER below (or override with
  ros2 launch ... allowed_number:=971XXXXXXXXX — but see comment on that param).
"""
import launch.conditions
import subprocess
import os
import signal
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# ── Hardcoded defaults (edit here rather than CLI to avoid int-typing bugs) ───
ALLOWED_NUMBER = '971509726902'   # Authorized WhatsApp number (string, not int)
MONITORED_JID  = '269655394504744'  # Second chat JID to monitor (for testing from own phone)

# ── Node executables to kill on re-launch ─────────────────────────────────────
_AISHA_EXECUTABLES = [
    'brain_node', 'admin_node', 'tts_node', 'stt_node',
    'action_node', 'whatsapp_listener',
]


def _kill_existing_nodes():
    """Kill any lingering aisha_brain node processes before launching fresh.

    Without this, re-launching without stopping first accumulates zombie
    instances that all subscribe to the same topics, causing duplicate
    message processing and multiple WhatsApp replies.
    """
    killed = []
    for proc in subprocess.run(
        ['pgrep', '-a', '-f', '|'.join(_AISHA_EXECUTABLES)],
        capture_output=True, text=True
    ).stdout.splitlines():
        parts = proc.split(None, 1)
        if len(parts) < 2:
            continue
        pid_str, cmd = parts
        # Skip ourselves (the launch process)
        if int(pid_str) == os.getpid():
            continue
        try:
            os.kill(int(pid_str), signal.SIGTERM)
            killed.append(pid_str)
        except ProcessLookupError:
            pass
    if killed:
        import time as _time
        _time.sleep(1.5)   # give processes time to exit cleanly
        # SIGKILL any survivors
        for pid_str in killed:
            try:
                os.kill(int(pid_str), signal.SIGKILL)
            except ProcessLookupError:
                pass   # already gone — good
        print(f'[aisha_launch] Cleaned up {len(killed)} stale node(s): PIDs {", ".join(killed)}')


def _sync_knowledge_base():
    """Sync the source-tree knowledge base into the installed share directory.

    colcon copies ChromaDB binary files at build time, but build_knowledge.py
    is typically run AFTER colcon build (it requires the venv + model download).
    This means the installed KB can lag behind the source KB.

    We sync at launch time so the running nodes always get the latest vectors,
    without needing a full colcon rebuild after every KB update.
    """
    import shutil
    try:
        from ament_index_python.packages import get_package_share_directory
        install_kb = os.path.join(
            get_package_share_directory('aisha_brain'), 'aisha_knowledge_db'
        )
    except Exception:
        return  # Not in a colcon workspace — nothing to sync

    # Locate the source KB relative to this launch file:
    # launch/ -> package root -> aisha_knowledge_db/
    src_kb = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'aisha_knowledge_db'
    )
    if not os.path.isdir(src_kb):
        print(f'[aisha_launch] WARNING: source KB not found at {src_kb}. '
              'Run build_knowledge.py first.')
        return

    # Compare chunk counts to decide if a sync is needed
    try:
        import chromadb as _cdb
        src_count = _cdb.PersistentClient(path=src_kb).get_collection('school_info').count()
        try:
            dst_count = _cdb.PersistentClient(path=install_kb).get_collection('school_info').count()
        except Exception:
            dst_count = -1

        if src_count != dst_count:
            print(f'[aisha_launch] KB mismatch (src={src_count} chunks, '
                  f'install={dst_count} chunks) — syncing...')
            if os.path.exists(install_kb):
                shutil.rmtree(install_kb)
            shutil.copytree(src_kb, install_kb)
            print(f'[aisha_launch] KB synced: {src_count} chunks → {install_kb}')
        else:
            print(f'[aisha_launch] KB up to date ({src_count} chunks).')
    except Exception as e:
        print(f'[aisha_launch] KB sync skipped: {e}')


def generate_launch_description():
    # Kill any stale instances from a previous launch before starting fresh.
    # This prevents duplicate subscribers that cause multiple WA replies.
    _kill_existing_nodes()

    # Sync knowledge base from source tree to install tree if chunk counts differ.
    # Ensures the LLM always gets the latest vectors without a full colcon rebuild.
    _sync_knowledge_base()

    # OpaqueFunction lets us read LaunchConfiguration as real Python strings,
    # preventing ROS2 from auto-casting all-digit values to INTEGER in params files.
    def create_nodes(context):
        # Read string args safely
        allowed_number = context.perform_substitution(
            LaunchConfiguration('allowed_number')
        )
        enable_whatsapp_str = context.perform_substitution(
            LaunchConfiguration('enable_whatsapp')
        )

        nodes = [
            # 1. Brain Node (Intent Router)
            Node(
                package='aisha_brain',
                executable='brain_node',
                name='ai_sha_brain',
                output='screen',
                parameters=[{
                    'ollama_url': 'http://127.0.0.1:11434/api/generate',
                    'router_model': context.perform_substitution(LaunchConfiguration('router_model')),
                    'router_timeout': 30,
                }]
            ),

            # 2. Admin Node (Knowledge Base RAG)
            Node(
                package='aisha_brain',
                executable='admin_node',
                name='ai_sha_admin',
                output='screen',
                parameters=[{
                    'ollama_url': 'http://127.0.0.1:11434',
                    'llm_model': context.perform_substitution(LaunchConfiguration('llm_model')),
                    'llm_timeout': 120.0,
                    'similarity_top_k': 6,
                }]
            ),

            # 3. TTS Node — publishes /speaker/playing to mute STT during playback
            Node(
                package='aisha_brain',
                executable='tts_node',
                name='ai_sha_tts',
                output='screen',
                parameters=[{
                    'voice_model': context.perform_substitution(LaunchConfiguration('voice_model')),
                    'audio_device': context.perform_substitution(LaunchConfiguration('audio_device')),
                }]
            ),

            # 4. Action Node
            Node(
                package='aisha_brain',
                executable='action_node',
                name='ai_sha_action',
                output='screen'
            ),
        ]

        # 5. STT Node — optional, only when enable_stt:=true
        if context.perform_substitution(LaunchConfiguration('enable_stt')) == 'true':
            nodes.append(Node(
                package='aisha_brain',
                executable='stt_node',
                name='ai_sha_stt',
                output='screen',
                parameters=[{
                    'whisper_model': context.perform_substitution(LaunchConfiguration('whisper_model')),
                    'whisper_device': context.perform_substitution(LaunchConfiguration('whisper_device')),
                    'whisper_compute_type': context.perform_substitution(LaunchConfiguration('whisper_compute_type')),
                    'input_device': context.perform_substitution(LaunchConfiguration('stt_input_device')),
                    'language': 'en',
                    'silence_threshold': 0.02,
                    'chunk_duration': 5.0,
                    'sample_rate': 16000,
                }]
            ))

        # 6. WhatsApp Listener — optional, enabled by default
        # allowed_number is passed as a native Python str (not via LaunchConfiguration
        # substitution into params YAML) to prevent ROS2 auto-casting it to INTEGER.
        if enable_whatsapp_str == 'true':
            nodes.append(Node(
                package='aisha_brain',
                executable='whatsapp_listener',
                name='ai_sha_listener',
                output='screen',
                parameters=[{
                    'allowed_number': str(allowed_number),   # str() ensures STRING type
                    'monitored_jid': str(context.perform_substitution(
                        LaunchConfiguration('monitored_jid'))),
                    'wa_reply_delay': 0.0,
                    'echo_mute_secs': 8.0,
                }]
            ))

        return nodes

    return LaunchDescription([
        # ── Launch Arguments ──────────────────────────────────────────────────
        DeclareLaunchArgument('audio_device',    default_value='plughw:1,0'),
        DeclareLaunchArgument('voice_model',     default_value='en_US-amy-low.onnx'),
        DeclareLaunchArgument('enable_stt',      default_value='false'),
        DeclareLaunchArgument('whisper_model',   default_value='base'),
        DeclareLaunchArgument('whisper_device',  default_value='cpu'),
        DeclareLaunchArgument('whisper_compute_type', default_value='int8'),
        DeclareLaunchArgument('stt_input_device', default_value=''),
        DeclareLaunchArgument('router_model',    default_value='gemma3:270m'),
        DeclareLaunchArgument('llm_model',       default_value='llama3.2'),
        DeclareLaunchArgument('enable_whatsapp', default_value='true'),
        DeclareLaunchArgument('allowed_number',  default_value=ALLOWED_NUMBER),
        DeclareLaunchArgument('monitored_jid',   default_value=MONITORED_JID),

        OpaqueFunction(function=create_nodes),
    ])
