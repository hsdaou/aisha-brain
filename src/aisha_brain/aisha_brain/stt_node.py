import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import threading
import queue
import subprocess
import tempfile
import os
import time
import numpy as np


class STTNode(Node):
    """Speech-to-Text node using Faster-Whisper (CTranslate2) for local transcription.

    Records audio via arecord (ALSA), transcribes with Faster-Whisper, and
    publishes text to /speech/text (Jetson architecture standard topic).

    Feedback prevention: subscribes to /speaker/playing (Bool).
    When True, microphone capture is paused so the robot does not transcribe
    its own TTS output.

    Target architecture topics (ros2_architecture.pdf):
      Publishes:  /speech/text      (std_msgs/String)
      Subscribes: /speaker/playing  (std_msgs/Bool)

    Requires:
      - faster-whisper Python package  (pip install faster-whisper)
      - arecord (ALSA)                 (apt install alsa-utils)
    """

    def __init__(self):
        super().__init__('ai_sha_stt')

        # Publish to architecture-standard topic
        self.publisher_ = self.create_publisher(String, '/speech/text', 10)

        # Feedback prevention: mute mic while TTS is playing
        self._is_speaker_playing = False
        self.create_subscription(Bool, '/speaker/playing', self._on_speaker_playing, 10)

        # Parameters
        self.declare_parameter('whisper_model', 'base')
        self.declare_parameter('whisper_device', 'cpu')
        self.declare_parameter('whisper_compute_type', 'int8')
        self.declare_parameter('language', 'en')
        self.declare_parameter('silence_threshold', 0.02)   # fraction of max int16 (0.02 = 655 RMS)
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('chunk_duration', 5.0)       # seconds per capture chunk
        self.declare_parameter('audio_device', 'plughw:1,0')  # ALSA capture device

        self.whisper_model_size = self.get_parameter('whisper_model').get_parameter_value().string_value
        self.whisper_device = self.get_parameter('whisper_device').get_parameter_value().string_value
        self.compute_type = self.get_parameter('whisper_compute_type').get_parameter_value().string_value
        self.language = self.get_parameter('language').get_parameter_value().string_value
        self.silence_threshold = self.get_parameter('silence_threshold').get_parameter_value().double_value
        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        self.chunk_duration = self.get_parameter('chunk_duration').get_parameter_value().double_value
        self.audio_device = self.get_parameter('audio_device').get_parameter_value().string_value

        self._msg_queue = queue.SimpleQueue()
        self._publish_timer = self.create_timer(0.1, self._publish_pending)
        self._tmp_wav = '/tmp/aisha_stt_chunk.wav'
        self._model = None

        if not self._check_arecord():
            self.get_logger().error('arecord not found. Install with: sudo apt install alsa-utils')
            return

        if not self._load_model():
            self.get_logger().error(
                'faster-whisper not available. '
                'Install with: pip install faster-whisper --break-system-packages'
            )
            return

        self.get_logger().info(
            f'AI-SHA STT Active — faster-whisper/{self.whisper_model_size} '
            f'on {self.whisper_device} ({self.compute_type}), '
            f'mic={self.audio_device}, chunk={self.chunk_duration}s'
        )
        threading.Thread(target=self._listen_loop, daemon=True).start()

    # ── Startup checks ────────────────────────────────────────────────────────

    def _check_arecord(self) -> bool:
        try:
            subprocess.run(['which', 'arecord'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _load_model(self) -> bool:
        try:
            from faster_whisper import WhisperModel
            self.get_logger().info(f'Loading faster-whisper "{self.whisper_model_size}"...')
            self._model = WhisperModel(
                self.whisper_model_size,
                device=self.whisper_device,
                compute_type=self.compute_type,
            )
            self.get_logger().info('faster-whisper model ready.')
            return True
        except ImportError:
            return False
        except Exception as e:
            self.get_logger().error(f'Failed to load faster-whisper model: {e}')
            return False

    # ── Speaker playing feedback prevention ───────────────────────────────────

    def _on_speaker_playing(self, msg: Bool):
        self._is_speaker_playing = msg.data
        state = 'muted (TTS playing)' if msg.data else 'listening'
        self.get_logger().debug(f'STT mic: {state}')

    # ── ROS publish (called from timer in executor thread) ────────────────────

    def _publish_pending(self):
        while not self._msg_queue.empty():
            try:
                text = self._msg_queue.get_nowait()
                msg = String()
                msg.data = text
                self.publisher_.publish(msg)
                self.get_logger().info(f'STT → /speech/text: "{text}"')
            except queue.Empty:
                break

    # ── Audio capture + transcription loop ───────────────────────────────────

    def _rms(self, wav_path: str) -> float:
        """Compute RMS energy of a WAV file (16-bit PCM, skip 44-byte header)."""
        try:
            with open(wav_path, 'rb') as f:
                f.read(44)  # skip WAV header
                raw = f.read()
            if not raw:
                return 0.0
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            return float(np.sqrt(np.mean(samples ** 2)))
        except Exception:
            return 0.0

    def _listen_loop(self):
        self.get_logger().info('STT: microphone capture started')

        # Noise phrases that whisper commonly hallucinates on silence
        noise_phrases = {
            '', '.', '...', 'you', 'thank you', 'thanks for watching',
            'thanks for watching.', 'thank you for watching.',
            'thank you for watching', 'bye', 'bye.',
        }

        while rclpy.ok():
            try:
                # Skip while TTS is playing (feedback prevention)
                if self._is_speaker_playing:
                    time.sleep(0.1)
                    continue

                # Record one chunk of audio via ALSA
                result = subprocess.run([
                    'arecord',
                    '-D', self.audio_device,
                    '-f', 'S16_LE',
                    '-r', str(self.sample_rate),
                    '-c', '1',
                    '-d', str(int(self.chunk_duration)),
                    self._tmp_wav,
                ], capture_output=True, timeout=self.chunk_duration + 5)

                # Skip if speaker became active during recording
                if self._is_speaker_playing:
                    continue

                if not os.path.exists(self._tmp_wav):
                    continue

                # Energy-based VAD: skip silent chunks to avoid hallucinations
                rms = self._rms(self._tmp_wav)
                threshold = self.silence_threshold * 32768.0
                if rms < threshold:
                    continue

                # Transcribe with Faster-Whisper
                segments, _info = self._model.transcribe(
                    self._tmp_wav,
                    language=self.language,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_speech_duration_ms=300),
                )

                text = ' '.join(seg.text.strip() for seg in segments).strip()

                if text and text.lower() not in noise_phrases and len(text) > 2:
                    self._msg_queue.put(text)

            except subprocess.TimeoutExpired:
                self.get_logger().warn('STT: audio capture timed out, retrying')
            except Exception as e:
                self.get_logger().error(f'STT error: {e}')
                time.sleep(1.0)


def main(args=None):
    rclpy.init(args=args)
    node = STTNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
