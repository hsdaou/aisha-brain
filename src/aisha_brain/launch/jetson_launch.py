"""AI-SHA Jetson Orin Nano Launch — AI / vision / reasoning nodes.

Runs brain_node (intent router), admin_node (RAG knowledge base),
and action_node on the Jetson. Ollama must be running locally on
the Jetson for LLM inference.

Usage:
  ros2 launch aisha_brain jetson_launch.py
  ros2 launch aisha_brain jetson_launch.py llm_model:=llama3.2:1b

Set the FastDDS profile externally before launching:
  export FASTRTPS_DEFAULT_PROFILES_FILE=.../config/fastdds_jetson.xml
"""
import os
import shutil
import signal
import subprocess
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_AISHA_EXECUTABLES = ['brain_node', 'admin_node', 'action_node']


def _kill_existing_nodes():
    """Kill any lingering AI node processes before launching fresh."""
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
        print(f'[jetson_launch] Cleaned up {len(killed)} stale node(s)')


def _sync_knowledge_base():
    """Sync the source-tree KB into the installed share directory.

    colcon copies ChromaDB files at build time, but build_knowledge.py
    is typically run AFTER colcon build. Sync at launch time so the
    running admin_node always gets the latest vectors.
    """
    try:
        from ament_index_python.packages import get_package_share_directory
        install_kb = os.path.join(
            get_package_share_directory('aisha_brain'), 'aisha_knowledge_db'
        )
    except Exception:
        return

    src_kb = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'aisha_knowledge_db'
    )
    if not os.path.isdir(src_kb):
        print(f'[jetson_launch] WARNING: source KB not found at {src_kb}')
        return

    try:
        import chromadb as _cdb
        src_count = _cdb.PersistentClient(path=src_kb).get_collection('school_info').count()
        try:
            dst_count = _cdb.PersistentClient(path=install_kb).get_collection('school_info').count()
        except Exception:
            dst_count = -1

        if src_count != dst_count:
            print(f'[jetson_launch] KB mismatch (src={src_count}, install={dst_count}) — syncing...')
            if os.path.exists(install_kb):
                shutil.rmtree(install_kb)
            shutil.copytree(src_kb, install_kb)
            print(f'[jetson_launch] KB synced: {src_count} chunks')
        else:
            print(f'[jetson_launch] KB up to date ({src_count} chunks)')
    except Exception as e:
        print(f'[jetson_launch] KB sync error: {e}')


def generate_launch_description():
    _kill_existing_nodes()
    _sync_knowledge_base()

    def create_nodes(context):
        return [
            # 1. Brain Node (Intent Router)
            Node(
                package='aisha_brain',
                executable='brain_node',
                name='ai_sha_brain',
                output='screen',
                parameters=[{
                    'ollama_url': 'http://127.0.0.1:11434/api/generate',
                    'router_model': context.perform_substitution(
                        LaunchConfiguration('router_model')),
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
                    'llm_model': context.perform_substitution(
                        LaunchConfiguration('llm_model')),
                    'llm_timeout': 120.0,
                    'similarity_top_k': 15,
                }]
            ),

            # 3. Action Node
            Node(
                package='aisha_brain',
                executable='action_node',
                name='ai_sha_action',
                output='screen'
            ),
        ]

    return LaunchDescription([
        DeclareLaunchArgument('router_model', default_value='gemma3:270m'),
        DeclareLaunchArgument('llm_model',    default_value='llama3.2'),

        OpaqueFunction(function=create_nodes),
    ])
