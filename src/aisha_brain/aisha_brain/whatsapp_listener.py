import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from std_msgs.msg import String
import subprocess
import threading
import queue
import time
import json
import os


class WhatsAppListener(Node):
    """Listen for incoming WhatsApp messages and reply with the brain's answers.

    Incoming flow:
      wa_listener.js  →  JSON lines on stdout
      → whatsapp_listener  →  /user_speech  →  brain_node  →  admin_node
      → /robot_speech  →  whatsapp_listener (auto-reply)  →  mudslide send

    The node tracks the JID of the last sender so it can route the brain's
    answer back to them via WhatsApp. Both TTS output (/robot_speech) and
    the WhatsApp reply happen in parallel.

    Runaway-loop prevention:
      When the bot sends a WA reply, wa_listener.js sees it as a new
      incoming message with fromMe=True. We suppress those echoes by
      setting _mute_incoming for a short window after each send.

    Parameters:
      allowed_number  (str)   Authorized sender phone number digits
      wa_reply_delay  (float) Seconds to wait before sending WA reply (0 = immediate)
      echo_mute_secs  (float) Seconds to mute incoming fromMe messages after a send
    """

    def __init__(self):
        super().__init__('ai_sha_listener')

        # Publish incoming text for the brain to process
        self.publisher_ = self.create_publisher(String, '/user_speech', 10)

        # Subscribe to brain's answers to echo them back via WhatsApp
        self.create_subscription(String, '/robot_speech', self._on_robot_speech, 10)

        # Force string type on allowed_number so ROS2 doesn't auto-cast it to INTEGER
        # when the value is all digits (e.g. '971509726902' from a launch file arg).
        self.declare_parameter(
            'allowed_number', '971509726902',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        self.declare_parameter('wa_reply_delay', 0.0)
        self.declare_parameter('echo_mute_secs', 8.0)

        self.allowed_number = self.get_parameter('allowed_number').get_parameter_value().string_value
        self.wa_reply_delay = self.get_parameter('wa_reply_delay').get_parameter_value().double_value
        self.echo_mute_secs = self.get_parameter('echo_mute_secs').get_parameter_value().double_value

        # Resolve wa_listener.js
        try:
            from ament_index_python.packages import get_package_share_directory
            self._listener_script = os.path.join(
                get_package_share_directory('aisha_brain'),
                'wa_listener.js'
            )
        except Exception:
            self._listener_script = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'wa_listener.js'
            )

        # Thread-safe queue: monitor thread enqueues, timer callback publishes
        self._msg_queue = queue.SimpleQueue()
        self._publish_timer = self.create_timer(0.1, self._publish_pending)

        # Sender tracking (protected by lock)
        self._last_sender_num: str = ''  # digits-only, e.g. '971509726902'
        self._last_from_me: bool = False
        self._reply_lock = threading.Lock()

        # Echo-loop prevention: timestamp of last outgoing WA send
        self._last_send_time: float = 0.0

        # Incoming deduplication: WhatsApp/Baileys can deliver the same message
        # twice with different sender JIDs (e.g. phone JID vs linked-device JID).
        # Track last incoming text+time to drop duplicates within this window.
        self._last_incoming_text: str = ''
        self._last_incoming_time: float = 0.0
        self._incoming_dedup_secs: float = 5.0  # drop same text arriving within 5 s

        # Outgoing deduplication: don't send the same answer text twice within this window.
        # Guards against multiple admin_node instances publishing identical /robot_speech msgs.
        self._last_sent_text: str = ''
        self._last_sent_time: float = 0.0
        self._reply_dedup_secs: float = 10.0   # suppress duplicate outgoing texts for 10 s

        self.get_logger().info(
            f'WhatsApp Listener Online — authorized: {self.allowed_number} | '
            f'auto-reply enabled | echo_mute={self.echo_mute_secs}s'
        )

        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    # ── Incoming pipeline ─────────────────────────────────────────────────────

    def _publish_pending(self):
        while not self._msg_queue.empty():
            try:
                text = self._msg_queue.get_nowait()
                msg = String()
                msg.data = text
                self.publisher_.publish(msg)
            except queue.Empty:
                break

    def _monitor_loop(self):
        if not os.path.exists(self._listener_script):
            self.get_logger().error(
                f'wa_listener.js not found at {self._listener_script}.'
            )
            return

        while rclpy.ok():
            try:
                process = subprocess.Popen(
                    ['node', self._listener_script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )

                threading.Thread(
                    target=self._log_stderr, args=(process,), daemon=True
                ).start()

                for line in process.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        sender = data.get('from', '')
                        text = data.get('message', '').strip()
                        from_me = data.get('fromMe', False)

                        if not text:
                            continue

                        # ── Echo-loop prevention ────────────────────────────
                        # After we send a WA reply, wa_listener.js echoes it
                        # back as fromMe=True. Suppress those echoes.
                        if from_me:
                            secs_since_send = time.time() - self._last_send_time
                            if secs_since_send < self.echo_mute_secs:
                                self.get_logger().debug(
                                    f'Suppressed fromMe echo ({secs_since_send:.1f}s after send)'
                                )
                                continue

                        # ── Authorization check ─────────────────────────────
                        is_authorized = from_me or (self.allowed_number in sender)
                        if not is_authorized:
                            self.get_logger().warn(f'Ignored unauthorized: {sender}')
                            continue

                        # ── Incoming deduplication ──────────────────────────
                        # Baileys sometimes delivers the same message twice with
                        # different JIDs (phone number JID vs linked-device JID).
                        # Drop identical text that arrives within the dedup window.
                        now = time.time()
                        if (text == self._last_incoming_text and
                                now - self._last_incoming_time < self._incoming_dedup_secs):
                            self.get_logger().warn(
                                f'Dropped duplicate incoming ({now - self._last_incoming_time:.2f}s): '
                                f'{text[:60]}'
                            )
                            continue
                        self._last_incoming_text = text
                        self._last_incoming_time = now

                        self.get_logger().info(
                            f'{"[me]" if from_me else "[in]"} {sender}: {text}'
                        )

                        with self._reply_lock:
                            self._last_sender_num = sender.split('@')[0] if '@' in sender else sender
                            self._last_from_me = from_me

                        self._msg_queue.put(text)

                    except json.JSONDecodeError:
                        self.get_logger().warn(f'Unexpected wa_listener output: {line}')

                process.wait()
                if rclpy.ok():
                    self.get_logger().warn('wa_listener.js exited, restarting in 5s...')
                    time.sleep(5)

            except FileNotFoundError:
                self.get_logger().error('node not found — cannot start WhatsApp listener')
                break
            except Exception as e:
                self.get_logger().error(f'Monitor error: {e}, restarting in 5s...')
                time.sleep(5)

    def _log_stderr(self, process):
        for line in process.stderr:
            line = line.strip()
            if not line:
                continue
            if 'ERROR' in line:
                self.get_logger().error(f'[wa_listener] {line}')
            else:
                self.get_logger().info(f'[wa_listener] {line}')

    # ── Outgoing reply ────────────────────────────────────────────────────────

    def _on_robot_speech(self, msg: String):
        """Send the brain's answer back to the WhatsApp sender."""
        answer = msg.data.strip()
        if not answer:
            return

        with self._reply_lock:
            sender_num = self._last_sender_num
            from_me = self._last_from_me

        if not sender_num:
            return  # No WA message received yet (e.g. manual ros2 topic pub)

        # ── Outgoing deduplication ─────────────────────────────────────────────
        # If multiple admin_node instances publish the same answer to /robot_speech,
        # only send the first one. Suppress identical text within the dedup window.
        now = time.time()
        if (answer == self._last_sent_text and
                now - self._last_sent_time < self._reply_dedup_secs):
            self.get_logger().warn(
                f'Suppressed duplicate outgoing reply ({now - self._last_sent_time:.1f}s ago): '
                f'{answer[:60]}'
            )
            return
        self._last_sent_text = answer
        self._last_sent_time = now

        recipient = 'me' if from_me else sender_num

        if self.wa_reply_delay > 0:
            time.sleep(self.wa_reply_delay)

        threading.Thread(
            target=self._send_whatsapp,
            args=(recipient, answer),
            daemon=True
        ).start()

    def _send_whatsapp(self, recipient: str, message: str):
        try:
            short = message[:80] + '...' if len(message) > 80 else message
            self.get_logger().info(f'WA reply → {recipient}: {short}')

            # Record send time BEFORE sending so the mute window starts immediately
            self._last_send_time = time.time()

            result = subprocess.run(
                ['npx', 'mudslide', 'send', recipient, message],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                self.get_logger().info('WA reply sent ✓')
            else:
                self.get_logger().error(f'WA reply failed: {result.stderr.strip()}')
        except subprocess.TimeoutExpired:
            self.get_logger().error('WA reply timed out')
        except FileNotFoundError:
            self.get_logger().error('npx/mudslide not found')
        except Exception as e:
            self.get_logger().error(f'WA reply error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = WhatsAppListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
