import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import os
import re


class TTSNode(Node):
    # Common Piper model directories across platforms
    _MODEL_SEARCH_PATHS = [
        '/usr/share/piper-voices',
        os.path.expanduser('~/.local/share/piper-voices'),
        '/opt/piper/models',
    ]

    def __init__(self):
        super().__init__('ai_sha_tts')
        self.subscription = self.create_subscription(String, '/robot_speech', self.speak, 10)

        self.declare_parameter('voice_model', 'en_US-amy-low.onnx')
        self.declare_parameter('audio_device', 'plughw:1,0')

        model_param = self.get_parameter('voice_model').get_parameter_value().string_value
        self.audio_device = self.get_parameter('audio_device').get_parameter_value().string_value

        # Resolve model path: if absolute and exists, use it; otherwise search
        self.model = self._resolve_model_path(model_param)

        self.get_logger().info('AI-SHA TTS Node Active (Piper)')

    def _resolve_model_path(self, model_param):
        """Find the voice model file, searching common locations if needed."""
        # Already an absolute path that exists
        if os.path.isabs(model_param) and os.path.exists(model_param):
            self.get_logger().info(f'Voice model: {model_param}')
            return model_param

        # Relative path that exists from cwd
        if os.path.exists(model_param):
            resolved = os.path.abspath(model_param)
            self.get_logger().info(f'Voice model: {resolved}')
            return resolved

        # Search common directories
        for search_dir in self._MODEL_SEARCH_PATHS:
            candidate = os.path.join(search_dir, model_param)
            if os.path.exists(candidate):
                self.get_logger().info(f'Voice model found: {candidate}')
                return candidate

        # Not found — piper may still find it via its own search path
        self.get_logger().warn(
            f'Voice model not found: {model_param} '
            f'(searched {self._MODEL_SEARCH_PATHS}). '
            f'Passing name to piper directly.'
        )
        return model_param

    def _sanitize_text(self, text):
        """Remove characters that could break the subprocess pipeline."""
        text = re.sub(r'[^\w\s.,!?\';\-:/()@]', '', text)
        return text.strip()[:2000]

    def speak(self, msg):
        text = self._sanitize_text(msg.data)
        if not text:
            return

        self.get_logger().info(f'Speaking: {text[:80]}...' if len(text) > 80 else f'Speaking: {text}')

        try:
            piper = subprocess.Popen(
                ['piper', '--model', self.model, '--output_raw'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            aplay = subprocess.Popen(
                ['aplay', '-D', self.audio_device, '-r', '22050', '-f', 'S16_LE', '-c', '1', '-t', 'raw'],
                stdin=piper.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            piper.stdin.write(text.encode('utf-8'))
            piper.stdin.close()
            piper.stdout.close()
            aplay.wait(timeout=60)
            piper.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.get_logger().error('TTS timed out')
            for p in [piper, aplay]:
                try:
                    p.kill()
                except Exception:
                    pass
        except FileNotFoundError as e:
            self.get_logger().error(f'TTS binary not found: {e}')
        except Exception as e:
            self.get_logger().error(f'TTS error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = TTSNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
