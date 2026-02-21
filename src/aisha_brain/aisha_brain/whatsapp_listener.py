import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import threading
import queue
import time
import json
import os


class WhatsAppListener(Node):
    """Listen for incoming WhatsApp messages via wa_listener.js.

    Publishes authorized messages to /user_speech so the brain can process them.
    Uses a Node.js script (wa_listener.js) that reuses mudslide's auth credentials
    and outputs JSON lines: {"from": "971XXXXXXXXX", "message": "..."}
    """

    def __init__(self):
        super().__init__('ai_sha_listener')
        self.publisher_ = self.create_publisher(String, '/user_speech', 10)

        self.declare_parameter('allowed_number', '971509726902')
        self.allowed_number = self.get_parameter('allowed_number').get_parameter_value().string_value

        # Resolve wa_listener.js — try ament share dir first, then alongside this file
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

        self.get_logger().info(f'WhatsApp Listener Online - authorized: {self.allowed_number}')

        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _publish_pending(self):
        """Drain the queue from the main executor thread (thread-safe publish)."""
        while not self._msg_queue.empty():
            try:
                text = self._msg_queue.get_nowait()
                msg = String()
                msg.data = text
                self.publisher_.publish(msg)
            except queue.Empty:
                break

    def _monitor_loop(self):
        """Run wa_listener.js and parse its JSON-line output."""
        if not os.path.exists(self._listener_script):
            self.get_logger().error(
                f'wa_listener.js not found at {self._listener_script}. '
                'WhatsApp listener cannot start.'
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

                # Log stderr (connection status) in a background thread
                threading.Thread(
                    target=self._log_stderr,
                    args=(process,),
                    daemon=True
                ).start()

                # Read JSON lines from stdout
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

                        # Accept if: sent by the linked device owner (fromMe=true)
                        # OR received from the configured allowed number.
                        # WhatsApp @lid JIDs don't contain the phone number, so
                        # fromMe is the reliable way to identify the owner's messages.
                        if from_me or self.allowed_number in sender:
                            self.get_logger().info(f'Authorized message from {sender}: {text}')
                            self._msg_queue.put(text)
                        else:
                            self.get_logger().warn(f'Ignored message from unauthorized: {sender}')

                    except json.JSONDecodeError:
                        self.get_logger().warn(f'Unexpected output from wa_listener.js: {line}')

                process.wait()
                if rclpy.ok():
                    self.get_logger().warn('wa_listener.js exited, restarting in 5s...')
                    time.sleep(5)

            except FileNotFoundError:
                self.get_logger().error('node not found - cannot start WhatsApp listener')
                break
            except Exception as e:
                self.get_logger().error(f'Monitor error: {e}, restarting in 5s...')
                time.sleep(5)

    def _log_stderr(self, process):
        """Forward wa_listener.js stderr to ROS2 logger."""
        for line in process.stderr:
            line = line.strip()
            if not line:
                continue
            if 'ERROR' in line:
                self.get_logger().error(f'[wa_listener] {line}')
            else:
                self.get_logger().info(f'[wa_listener] {line}')


def main(args=None):
    rclpy.init(args=args)
    node = WhatsAppListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
