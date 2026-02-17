import os

import launch.conditions
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Default: knowledge DB lives next to the source package
    _pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _default_kb = os.path.join(_pkg_dir, 'aisha_knowledge_db')

    return LaunchDescription([
        # --- Launch arguments (override with ros2 launch ... key:=value) ---
        DeclareLaunchArgument('audio_device', default_value='plughw:1,0',
                              description='ALSA device for TTS output'),
        DeclareLaunchArgument('voice_model', default_value='en_US-amy-low.onnx',
                              description='Piper TTS voice model path'),
        DeclareLaunchArgument('router_model', default_value='gemma3:270m',
                              description='Ollama model for intent routing'),
        DeclareLaunchArgument('llm_model', default_value='llama3.2',
                              description='Ollama model for knowledge QA'),
        DeclareLaunchArgument('knowledge_db_path', default_value=_default_kb,
                              description='Path to ChromaDB knowledge base'),
        DeclareLaunchArgument('whisper_model', default_value='ggml-base.en.bin',
                              description='Whisper.cpp model path for STT'),
        DeclareLaunchArgument('enable_stt', default_value='false',
                              description='Enable microphone STT input (set true on robot)'),

        # 1. WhatsApp Listener (Input Gateway)
        Node(
            package='aisha_brain',
            executable='whatsapp_listener',
            name='ai_sha_listener',
            output='screen',
            parameters=[{
                'allowed_number': '971509726902',
            }]
        ),
        # 2. The Brain (Intent Router)
        Node(
            package='aisha_brain',
            executable='brain_node',
            name='ai_sha_brain',
            output='screen',
            parameters=[{
                'ollama_url': 'http://127.0.0.1:11434/api/generate',
                'router_model': LaunchConfiguration('router_model'),
                'router_timeout': 30,
            }]
        ),
        # 3. The Admin Node (Knowledge Base)
        Node(
            package='aisha_brain',
            executable='admin_node',
            name='ai_sha_admin',
            output='screen',
            parameters=[{
                'knowledge_db_path': LaunchConfiguration('knowledge_db_path'),
                'ollama_url': 'http://127.0.0.1:11434',
                'llm_model': LaunchConfiguration('llm_model'),
                'llm_timeout': 120.0,
            }]
        ),
        # 4. The Action Node (WhatsApp, Calendar)
        Node(
            package='aisha_brain',
            executable='action_node',
            name='ai_sha_action',
            output='screen'
        ),
        # 5. The TTS Node (Voice Output)
        Node(
            package='aisha_brain',
            executable='tts_node',
            name='ai_sha_tts',
            output='screen',
            parameters=[{
                'voice_model': LaunchConfiguration('voice_model'),
                'audio_device': LaunchConfiguration('audio_device'),
            }]
        ),
        # 6. STT Node (Microphone Input) — enable with: enable_stt:=true
        # Requires whisper.cpp and arecord. Set whisper_model to model path.
        Node(
            package='aisha_brain',
            executable='stt_node',
            name='ai_sha_stt',
            output='screen',
            parameters=[{
                'whisper_model': LaunchConfiguration('whisper_model'),
                'audio_device': LaunchConfiguration('audio_device'),
            }],
            condition=launch.conditions.IfCondition(LaunchConfiguration('enable_stt'))
        ),
    ])
