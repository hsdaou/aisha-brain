import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import re


class ActionNode(Node):
    def __init__(self):
        super().__init__('ai_sha_action')
        self.subscription = self.create_subscription(
            String, '/action_request', self.handle_action, 10)
        self.speech_pub = self.create_publisher(String, '/robot_speech', 10)
        self.get_logger().info('AI-SHA Action Node Online')

    def _say(self, text):
        """Publish feedback to TTS."""
        msg = String()
        msg.data = text
        self.speech_pub.publish(msg)

    def handle_action(self, msg):
        command = msg.data.strip()
        if not command:
            return
        command_lower = command.lower()
        self.get_logger().info(f'Action Request: {command}')

        if "whatsapp" in command_lower or "message" in command_lower:
            self.send_whatsapp(command)
        elif "calendar" in command_lower or "schedule" in command_lower:
            self._say("Calendar integration is coming soon.")
            self.get_logger().info("Calendar integration not yet implemented")
        else:
            self._say("I'm not sure how to handle that action yet.")
            self.get_logger().warn(f'Unrecognized action: {command}')

    def send_whatsapp(self, text):
        try:
            # Extract phone number (UAE format: 971XXXXXXXXX, 10-12 digits)
            phone_match = re.search(r'\b(971\d{8,10})\b', text)
            if not phone_match:
                self._say("I couldn't find a valid phone number. Please include a number starting with 971.")
                self.get_logger().warn("No valid phone number found in command")
                return

            phone = phone_match.group(1)

            # Extract message content
            message = ""
            text_lower = text.lower()
            for keyword in ["saying", "say", "that says", "message"]:
                if keyword in text_lower:
                    idx = text_lower.index(keyword) + len(keyword)
                    message = text[idx:].strip().strip('"').strip("'")
                    break

            if not message:
                message = "Message from AI-SHA"

            self.get_logger().info(f'Sending to {phone}: {message[:50]}')

            result = subprocess.run(
                ["npx", "mudslide", "send", phone, message],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                self._say(f"Message sent successfully.")
                self.get_logger().info("WhatsApp message sent")
            else:
                self._say("I had trouble sending that message. Please try again.")
                self.get_logger().error(f'mudslide error: {result.stderr}')

        except subprocess.TimeoutExpired:
            self._say("The message is taking too long to send. Please check the connection.")
            self.get_logger().error("WhatsApp send timed out")
        except FileNotFoundError:
            self._say("WhatsApp messaging tool is not installed.")
            self.get_logger().error("npx/mudslide not found")
        except Exception as e:
            self._say("Something went wrong sending the message.")
            self.get_logger().error(f'WhatsApp error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ActionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
