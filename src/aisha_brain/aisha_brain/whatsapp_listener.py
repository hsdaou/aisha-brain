import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import threading
import queue
import time


class WhatsAppListener(Node):
    def __init__(self):
        super().__init__('ai_sha_listener')
        self.publisher_ = self.create_publisher(String, '/user_speech', 10)

        self.declare_parameter('allowed_number', '971509726902')
        self.allowed_number = self.get_parameter('allowed_number').get_parameter_value().string_value

        # Thread-safe queue: monitor thread enqueues, timer callback publishes
        self._msg_queue = queue.SimpleQueue()
        self._publish_timer = self.create_timer(0.1, self._publish_pending)

        self.get_logger().info(f'WhatsApp Listener Online - authorized: {self.allowed_number}')

        self.monitor_thread = threading.Thread(target=self.monitor_whatsapp, daemon=True)
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

    def monitor_whatsapp(self):
        while rclpy.ok():
            try:
                process = subprocess.Popen(
                    ['npx', 'mudslide', 'monitor'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )

                while rclpy.ok():
                    line = process.stdout.readline()
                    if not line:
                        break

                    if "Message from" in line:
                        try:
                            parts = line.split(": ", 1)
                            if len(parts) < 2:
                                continue
                            sender_info = parts[0]
                            message_text = parts[1].strip()
                            sender_number = sender_info.split("from ")[1].strip()

                            if self.allowed_number in sender_number:
                                self.get_logger().info(f'Authorized: {message_text}')
                                self._msg_queue.put(message_text)
                            else:
                                self.get_logger().warn(f'Ignored from unauthorized: {sender_number}')
                        except (IndexError, ValueError):
                            pass

                process.wait()
                self.get_logger().warn('mudslide monitor exited, restarting in 5s...')
                time.sleep(5)

            except FileNotFoundError:
                self.get_logger().error('npx/mudslide not found - WhatsApp listener cannot start')
                break
            except Exception as e:
                self.get_logger().error(f'Monitor error: {e}, restarting in 5s...')
                time.sleep(5)


def main(args=None):
    rclpy.init(args=args)
    node = WhatsAppListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
