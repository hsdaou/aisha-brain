import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import threading
import queue
import subprocess
import tempfile
import os
import re
import time
import numpy as np


# ── Wake word variants ────────────────────────────────────────────────────────
# Whisper may transcribe "ARIA" / "Hey ARIA" in various ways depending on
# accent, noise, and model size.  We match against these known variants.
# Order matters: longer prefixes are checked first so "hey aria" is stripped
# before "aria" alone.
_WAKE_PREFIXES = [
    'hey aria',  'hey arya',  'hey area',  'hey ariya',
    'hi aria',   'hi arya',   'hi area',
    'aria',      'arya',      'area',      'ariya',
]

# Compiled regex: match any wake prefix at the start of the string,
# optionally followed by a comma / period / colon / space.
_WAKE_RE = re.compile(
    r'^(?:' + '|'.join(re.escape(p) for p in _WAKE_PREFIXES) + r')'
    r'[\s,.:;!?\-]*',
    re.IGNORECASE,
)


class STTNode(Node):
    """Speech-to-Text node using Faster-Whisper (CTranslate2) for local transcription.

    Records audio via arecord (ALSA), transcribes with Faster-Whisper, and
    publishes text to /speech/text (Jetson architecture standard topic).

    Wake word support:
      When enabled (default), only speech preceded by "ARIA" or "Hey ARIA"
      is published. After a wake word trigger, a listening window stays
      open (default 15 s) so follow-up sentences don't need the wake word
      again. The window is also extended when the robot responds (TTS).

    Feedback prevention: subscribes to /speaker/playing (Bool).
    When True, microphone capture is paused so the robot does not transcribe
    its own TTS output.

    Target architecture topics (ros2_architecture.pdf):
      Publishes:  /speech/text      (std_msgs/String)
      Subscribes: /speaker/playing  (std_msgs/Bool)
                  /robot_speech     (std_msgs/String)  — extends wake window

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

        # Wake word parameters
        self.declare_parameter('wake_word_enabled', True)
        self.declare_parameter('wake_word_timeout', 15.0)    # seconds of continued listening

        self.whisper_model_size = self.get_parameter('whisper_model').get_parameter_value().string_value
        self.whisper_device = self.get_parameter('whisper_device').get_parameter_value().string_value
        self.compute_type = self.get_parameter('whisper_compute_type').get_parameter_value().string_value
        self.language = self.get_parameter('language').get_parameter_value().string_value
        self.silence_threshold = self.get_parameter('silence_threshold').get_parameter_value().double_value
        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        self.chunk_duration = self.get_parameter('chunk_duration').get_parameter_value().double_value
        self.audio_device = self.get_parameter('audio_device').get_parameter_value().string_value

        self.wake_word_enabled = self.get_parameter('wake_word_enabled').get_parameter_value().bool_value
        self.wake_word_timeout = self.get_parameter('wake_word_timeout').get_parameter_value().double_value

        # Wake word state: timestamp until which we accept speech without wake word
        self._wake_active_until: float = 0.0

        # Extend listening window when the robot responds (conversation continuation)
        if self.wake_word_enabled:
            self.create_subscription(String, '/robot_speech', self._on_robot_speech, 10)

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

        wake_status = (
            f'wake_word=ARIA (timeout={self.wake_word_timeout}s)'
            if self.wake_word_enabled else 'wake_word=disabled (open mic)'
        )
        self.get_logger().info(
            f'AI-SHA STT Active — faster-whisper/{self.whisper_model_size} '
            f'on {self.whisper_device} ({self.compute_type}), '
            f'mic={self.audio_device}, chunk={self.chunk_duration}s, '
            f'{wake_status}'
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

    # ── Robot speech callback — extend listening window ───────────────────────

    def _on_robot_speech(self, msg: String):
        """When the robot speaks, extend the wake-word listening window.

        This allows natural conversation continuation: after the robot answers,
        the user can ask a follow-up without repeating the wake word.
        """
        if msg.data.strip():
            self._wake_active_until = time.monotonic() + self.wake_word_timeout
            self.get_logger().debug(
                f'Wake window extended (robot spoke) — '
                f'listening for {self.wake_word_timeout}s'
            )

    # ── Wake word detection ──────────────────────────────────────────────────

    def _check_wake_word(self, text: str) -> tuple:
        """Check if text contains the wake word and strip it.

        Returns:
            (wake_triggered, cleaned_text):
                wake_triggered: True if wake word was found in this utterance
                cleaned_text:   text with the wake word prefix removed
        """
        match = _WAKE_RE.match(text)
        if match:
            cleaned = text[match.end():].strip()
            return True, cleaned
        return False, text

    def _should_publish(self, text: str) -> tuple:
        """Decide whether to publish this transcription based on wake word state.

        Returns:
            (should_publish, text_to_publish)
        """
        if not self.wake_word_enabled:
            return True, text

        now = time.monotonic()

        # Check if the transcription contains the wake word
        triggered, cleaned = self._check_wake_word(text)

        if triggered:
            # Wake word found — activate listening window
            self._wake_active_until = now + self.wake_word_timeout
            self.get_logger().info(
                f'Wake word detected! Listening for {self.wake_word_timeout}s'
            )
            if cleaned:
                # "Hey ARIA, what are the fees?" → publish "what are the fees?"
                return True, cleaned
            else:
                # Just "Hey ARIA" alone — activate window, don't publish empty
                return False, ''

        # No wake word in this utterance — check if listening window is active
        if now < self._wake_active_until:
            remaining = self._wake_active_until - now
            self.get_logger().debug(
                f'Wake window active ({remaining:.1f}s left) — passing through'
            )
            # Extend the window on continued speech
            self._wake_active_until = now + self.wake_word_timeout
            return True, text

        # No wake word, window expired — discard
        self.get_logger().debug(
            f'Discarded (no wake word): "{text[:60]}..."'
            if len(text) > 60 else f'Discarded (no wake word): "{text}"'
        )
        return False, ''

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
                    # Apply wake word filter
                    should_pub, pub_text = self._should_publish(text)
                    if should_pub and pub_text:
                        self._msg_queue.put(pub_text)

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
