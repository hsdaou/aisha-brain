import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import threading
import queue
import os


class STTNode(Node):
    """Speech-to-Text node using whisper.cpp for local transcription.

    Listens on the microphone, transcribes speech, and publishes
    the text to /user_speech for the brain to process.

    Requires:
      - whisper.cpp built and on PATH (or provide full path via parameter)
      - A GGML whisper model (e.g., ggml-base.en.bin)
      - arecord (ALSA) for audio capture
    """

    def __init__(self):
        super().__init__('ai_sha_stt')
        self.publisher_ = self.create_publisher(String, '/user_speech', 10)

        self.declare_parameter('whisper_model', 'ggml-base.en.bin')
        self.declare_parameter('audio_device', 'plughw:1,0')
        self.declare_parameter('language', 'en')
        self.declare_parameter('silence_threshold', 0.3)
        self.declare_parameter('record_seconds', 5)

        self.whisper_model = self.get_parameter('whisper_model').get_parameter_value().string_value
        self.audio_device = self.get_parameter('audio_device').get_parameter_value().string_value
        self.language = self.get_parameter('language').get_parameter_value().string_value
        self.silence_threshold = self.get_parameter('silence_threshold').get_parameter_value().double_value
        self.record_seconds = self.get_parameter('record_seconds').get_parameter_value().integer_value

        # Thread-safe queue for publishing from main thread
        self._msg_queue = queue.SimpleQueue()
        self._publish_timer = self.create_timer(0.1, self._publish_pending)

        self._tmp_wav = '/tmp/aisha_stt_input.wav'

        if not self._check_dependencies():
            self.get_logger().error(
                'STT dependencies missing. Install whisper.cpp and ensure arecord is available. '
                'Node will idle until dependencies are met.'
            )
            return

        self.get_logger().info('AI-SHA STT Node Active (whisper.cpp)')

        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()

    def _check_dependencies(self):
        """Verify whisper.cpp and arecord are available."""
        try:
            subprocess.run(['which', 'whisper-cpp'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Also try the common binary name
            try:
                subprocess.run(['which', 'main'], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.get_logger().warn('whisper-cpp binary not found on PATH')
                return False

        try:
            subprocess.run(['which', 'arecord'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.get_logger().warn('arecord not found')
            return False

        if not os.path.exists(self.whisper_model):
            self.get_logger().warn(f'Whisper model not found: {self.whisper_model}')
            return False

        return True

    def _publish_pending(self):
        """Drain the queue from the main executor thread."""
        while not self._msg_queue.empty():
            try:
                text = self._msg_queue.get_nowait()
                msg = String()
                msg.data = text
                self.publisher_.publish(msg)
                self.get_logger().info(f'Heard: {text}')
            except queue.Empty:
                break

    def _listen_loop(self):
        """Continuously record and transcribe audio."""
        import time

        while rclpy.ok():
            try:
                # Record audio chunk
                subprocess.run([
                    'arecord',
                    '-D', self.audio_device,
                    '-f', 'S16_LE',
                    '-r', '16000',
                    '-c', '1',
                    '-d', str(self.record_seconds),
                    self._tmp_wav
                ], capture_output=True, timeout=self.record_seconds + 5)

                if not os.path.exists(self._tmp_wav):
                    continue

                # Transcribe with whisper.cpp
                result = subprocess.run([
                    'whisper-cpp',
                    '-m', self.whisper_model,
                    '-l', self.language,
                    '-nt',  # no timestamps
                    '-f', self._tmp_wav
                ], capture_output=True, text=True, timeout=30)

                if result.returncode == 0:
                    text = result.stdout.strip()
                    # Filter out empty/noise transcriptions
                    if text and len(text) > 2 and text.lower() not in ('[blank_audio]', '(silence)'):
                        self._msg_queue.put(text)

            except subprocess.TimeoutExpired:
                self.get_logger().warn('Audio capture or transcription timed out')
            except FileNotFoundError:
                self.get_logger().error('STT binary not found, stopping listener')
                break
            except Exception as e:
                self.get_logger().error(f'STT error: {e}')
                time.sleep(1)


def main(args=None):
    rclpy.init(args=args)
    node = STTNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
