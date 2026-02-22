"""AI-SHA Brain Launch File

Launches all aisha_brain nodes. Aligned with the Jetson Orin Nano ROS2
architecture (see ros2_architecture.pdf).

Key topic alignment with the Jetson architecture:
  /speech/text       <- stt_node publishes (Faster-Whisper)
  /user_speech       <- whatsapp_listener publishes (alias, brain also listens here)
  /tts_text          <- brain_node / admin_node publish (Jetson standard)
  /robot_speech      <- admin_node / brain_node also publish (local alias)
  /speaker/playing   <- tts_node publishes Bool; stt_node subscribes to mute mic
  /detection/objects_simple  <- detection_node (YOLOv8) publishes; brain_node subscribes

Usage:
  ros2 launch aisha_brain aisha_launch.py
  ros2 launch aisha_brain aisha_launch.py enable_stt:=true
  ros2 launch aisha_brain aisha_launch.py enable_stt:=true whisper_model:=small whisper_device:=cuda
  ros2 launch aisha_brain aisha_launch.py enable_whatsapp:=false
"""
import launch.conditions
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        # ── Launch Arguments ──────────────────────────────────────────────────

        # Audio / TTS
        DeclareLaunchArgument('audio_device', default_value='plughw:1,0',
                              description='ALSA device for audio output (TTS)'),
        DeclareLaunchArgument('voice_model', default_value='en_US-amy-low.onnx',
                              description='Piper TTS voice model filename or full path'),

        # STT (Faster-Whisper)
        DeclareLaunchArgument('enable_stt', default_value='false',
                              description='Enable microphone STT (requires faster-whisper + sounddevice)'),
        DeclareLaunchArgument('whisper_model', default_value='base',
                              description='Faster-Whisper model size: tiny | base | small | medium'),
        DeclareLaunchArgument('whisper_device', default_value='cpu',
                              description='Inference device: cpu | cuda (use cuda on Jetson with CUDA)'),
        DeclareLaunchArgument('whisper_compute_type', default_value='int8',
                              description='Compute type: int8 (CPU) | float16 (GPU/CUDA)'),
        DeclareLaunchArgument('stt_input_device', default_value='',
                              description='sounddevice input device name/index (empty = system default)'),

        # LLM / Routing
        DeclareLaunchArgument('router_model', default_value='gemma3:270m',
                              description='Ollama model for intent routing'),
        DeclareLaunchArgument('llm_model', default_value='llama3.2',
                              description='Ollama model for knowledge base QA'),

        # WhatsApp
        DeclareLaunchArgument('enable_whatsapp', default_value='true',
                              description='Enable WhatsApp listener (requires mudslide auth)'),
        DeclareLaunchArgument('allowed_number', default_value='971509726902',
                              description='Authorized WhatsApp number (digits only, no + or spaces)'),

        # ── Nodes ─────────────────────────────────────────────────────────────

        # 1. Brain Node (Intent Router)
        #    Subscribes: /speech/text, /user_speech, /detection/objects_simple
        #    Publishes:  /admin_task, /nav_goal, /action_request, /tts_text, /robot_speech
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

        # 2. Admin Node (Knowledge Base RAG)
        #    Subscribes: /admin_task
        #    Publishes:  /robot_speech (tts_node also listens on /tts_text alias)
        Node(
            package='aisha_brain',
            executable='admin_node',
            name='ai_sha_admin',
            output='screen',
            parameters=[{
                'ollama_url': 'http://127.0.0.1:11434',
                'llm_model': LaunchConfiguration('llm_model'),
                'llm_timeout': 120.0,
                'similarity_top_k': 6,
            }]
        ),

        # 3. TTS Node (Piper)
        #    Subscribes: /tts_text, /robot_speech
        #    Publishes:  /speaker/playing (Bool) — mutes STT microphone during playback
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

        # 4. STT Node (Faster-Whisper) — enabled with enable_stt:=true
        #    Publishes:  /speech/text (std_msgs/String)
        #    Subscribes: /speaker/playing (Bool) — mutes mic during TTS playback
        Node(
            package='aisha_brain',
            executable='stt_node',
            name='ai_sha_stt',
            output='screen',
            parameters=[{
                'whisper_model': LaunchConfiguration('whisper_model'),
                'whisper_device': LaunchConfiguration('whisper_device'),
                'whisper_compute_type': LaunchConfiguration('whisper_compute_type'),
                'input_device': LaunchConfiguration('stt_input_device'),
                'language': 'en',
                'silence_threshold': 0.02,
                'chunk_duration': 5.0,
                'sample_rate': 16000,
            }],
            condition=launch.conditions.IfCondition(LaunchConfiguration('enable_stt'))
        ),

        # 5. Action Node (WhatsApp send, Calendar)
        #    Subscribes: /action_request
        Node(
            package='aisha_brain',
            executable='action_node',
            name='ai_sha_action',
            output='screen'
        ),

        # 6. WhatsApp Listener — enabled with enable_whatsapp:=true (default)
        #    Publishes: /user_speech (brain_node listens on this alias)
        Node(
            package='aisha_brain',
            executable='whatsapp_listener',
            name='ai_sha_listener',
            output='screen',
            parameters=[{
                'allowed_number': LaunchConfiguration('allowed_number'),
            }],
            condition=launch.conditions.IfCondition(LaunchConfiguration('enable_whatsapp'))
        ),
    ])
