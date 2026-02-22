import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import requests
from collections import deque


class BrainNode(Node):
    def __init__(self):
        super().__init__('ai_sha_brain')

        # Subscribe to both architecture-standard and local-alias speech topics
        self.create_subscription(String, '/speech/text', self.listener_callback, 10)   # Jetson architecture
        self.create_subscription(String, '/user_speech', self.listener_callback, 10)   # local alias / WhatsApp

        # Subscribe to vision detection for context-aware routing
        self.create_subscription(String, '/detection/objects_simple', self._on_detection, 10)
        self._last_detection = {}   # most recent detection payload (parsed JSON)

        # Publishers — publish to both standard and alias topics
        self.admin_pub = self.create_publisher(String, '/admin_task', 10)
        self.nav_pub = self.create_publisher(String, '/nav_goal', 10)
        self.action_pub = self.create_publisher(String, '/action_request', 10)
        self.speech_pub = self.create_publisher(String, '/tts_text', 10)       # Jetson architecture
        self.speech_pub_alias = self.create_publisher(String, '/robot_speech', 10)  # local alias

        # Subscribe to robot speech so we can record answers into history
        self.speech_sub = self.create_subscription(String, '/tts_text', self._record_answer, 10)
        self.speech_sub_alias = self.create_subscription(String, '/robot_speech', self._record_answer, 10)
        self._last_user_input = ''

        # Parameters
        self.declare_parameter('ollama_url', 'http://127.0.0.1:11434/api/generate')
        self.declare_parameter('router_model', 'gemma3:270m')
        self.declare_parameter('router_timeout', 30)

        self.ollama_url = self.get_parameter('ollama_url').get_parameter_value().string_value
        self.router_model = self.get_parameter('router_model').get_parameter_value().string_value
        self.router_timeout = self.get_parameter('router_timeout').get_parameter_value().integer_value

        # Conversation memory: stores (user_text, robot_answer) tuples, last 5 exchanges
        self.history = deque(maxlen=5)

        # Check Ollama connectivity at startup
        self._check_ollama()

        self.get_logger().info('AI-SHA Brain: Router Active')

    def _check_ollama(self):
        try:
            r = requests.get(
                self.ollama_url.replace('/api/generate', '/api/tags'),
                timeout=5
            )
            models = [m['name'] for m in r.json().get('models', [])]
            if self.router_model not in models:
                self.get_logger().warn(f'Router model {self.router_model} not found. Available: {models}')
            else:
                self.get_logger().info(f'Ollama OK. Model: {self.router_model}')
        except Exception as e:
            self.get_logger().error(f'Ollama unreachable at {self.ollama_url}: {e}')

    def classify_intent(self, text):
        """Keyword-first routing: fast, deterministic, works on any hardware.
        Falls back to LLM only for ambiguous inputs."""
        keyword_result = self._keyword_classify(text)
        if keyword_result is not None:
            return keyword_result

        # Ambiguous input — use LLM
        return self._llm_classify(text)

    def _keyword_classify(self, text):
        """Fast deterministic classification using keyword patterns."""
        text_lower = text.lower()

        nav_keywords = [
            'go to', 'navigate to', 'move to', 'come to', 'come here',
            'take me to', 'follow me', 'drive to', 'walk to',
            'head to', 'bring me to', 'lead me to',
        ]
        action_keywords = [
            'whatsapp', 'send a message', 'send message', 'text my',
            'call my', 'email', 'remind me', 'set a reminder',
            'send to', 'message my',
        ]

        for kw in nav_keywords:
            if kw in text_lower:
                self.get_logger().info(f'Keyword match "{kw}" -> NAV')
                return {"intent": "NAV"}
        for kw in action_keywords:
            if kw in text_lower:
                self.get_logger().info(f'Keyword match "{kw}" -> ACTION')
                return {"intent": "ACTION"}

        # Questions are almost always ADMIN
        if text.rstrip().endswith('?'):
            self.get_logger().info('Question mark detected -> ADMIN')
            return {"intent": "ADMIN"}

        question_starters = [
            'what ', 'when ', 'where ', 'how ', 'who ', 'which ',
            'is there', 'are there', 'do you', 'can you tell',
            'tell me about', 'i want to know',
        ]
        for qs in question_starters:
            if text_lower.startswith(qs) or text_lower.startswith(qs.lstrip()):
                self.get_logger().info(f'Question pattern "{qs}" -> ADMIN')
                return {"intent": "ADMIN"}

        # No confident match — return None to trigger LLM
        return None

    def _llm_classify(self, text):
        """LLM-based classification for ambiguous inputs."""
        prompt = f"""Classify this school robot request into exactly one: ADMIN, NAV, or ACTION.

ADMIN = questions about school info, fees, schedule, academics, facilities
NAV = physical movement: go somewhere, navigate, come here
ACTION = send message, whatsapp, call, email, reminder

"{text}"
JSON:"""

        try:
            r = requests.post(self.ollama_url, json={
                "model": self.router_model,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.0}
            }, timeout=self.router_timeout)
            result = json.loads(r.json()['response'])
            intent = result.get("intent", "").upper()
            if intent in ("ADMIN", "NAV", "ACTION"):
                self.get_logger().info(f'LLM classified -> {intent}')
                return {"intent": intent}
            self.get_logger().warn(f'LLM returned unknown intent "{intent}", defaulting to ADMIN')
        except requests.ConnectionError:
            self.get_logger().error('Ollama connection failed')
        except requests.Timeout:
            self.get_logger().error('Ollama timed out')
        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().error(f'Failed to parse LLM response: {e}')

        # Final fallback
        self.get_logger().info('Defaulting -> ADMIN')
        return {"intent": "ADMIN"}

    def _on_detection(self, msg):
        """Receive latest detection context from detection_node (YOLOv8).
        Stores the parsed payload so routing can use object/gesture context."""
        try:
            self._last_detection = json.loads(msg.data)
        except (json.JSONDecodeError, Exception):
            self._last_detection = {'raw': msg.data}

    def _say(self, text):
        """Publish a message to TTS on both standard and alias topics."""
        msg = String()
        msg.data = text
        self.speech_pub.publish(msg)
        self.speech_pub_alias.publish(msg)

    def _record_answer(self, msg):
        """Store the robot's answer paired with the last user input into history."""
        answer = msg.data.strip()
        if self._last_user_input and answer:
            self.history.append((self._last_user_input, answer))
            self._last_user_input = ''

    def listener_callback(self, msg):
        user_input = msg.data.strip()
        if not user_input:
            return

        self.get_logger().info(f'Heard: {user_input}')
        self._last_user_input = user_input

        decision = self.classify_intent(user_input)
        intent = decision.get("intent", "ADMIN")

        out_msg = String()
        if intent == "ADMIN":
            self.get_logger().info("Route -> ADMIN (Knowledge Base)")
            # Include recent conversation history so admin_node can handle follow-ups
            history_list = [{"user": u, "assistant": a} for u, a in self.history]
            out_msg.data = json.dumps({"details": user_input, "history": history_list})
            self.admin_pub.publish(out_msg)

        elif intent == "NAV":
            self.get_logger().info("Route -> NAV")
            out_msg.data = user_input
            self.nav_pub.publish(out_msg)
            self._say("Navigation is not yet available. I'll be able to move around soon.")

        else:
            self.get_logger().info("Route -> ACTION")
            out_msg.data = user_input
            self.action_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
