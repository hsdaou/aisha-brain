import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import requests
import time
import threading
from collections import deque


class BrainNode(Node):
    def __init__(self):
        super().__init__('ai_sha_brain')

        # ── Deduplication: ignore identical messages within this window ─────────
        # /speech/text and /user_speech can both fire for the same WA message.
        self._last_processed_text: str = ''
        self._last_processed_time: float = 0.0
        self._debounce_secs: float = 3.0   # ignore duplicates within 3 s

        # ── Input subscriptions ────────────────────────────────────────────────
        # Both topics feed the same callback — /speech/text is the Jetson
        # architecture standard; /user_speech is the local alias used by
        # whatsapp_listener and manual ros2 topic pub testing.
        self.create_subscription(String, '/speech/text', self.listener_callback, 10)
        self.create_subscription(String, '/user_speech', self.listener_callback, 10)

        # Vision context from detection_node (YOLOv8) — stored for future routing
        self.create_subscription(String, '/detection/objects_simple', self._on_detection, 10)
        self._last_detection = {}

        # ── Output publishers ──────────────────────────────────────────────────
        self.admin_pub  = self.create_publisher(String, '/admin_task', 10)
        self.nav_pub    = self.create_publisher(String, '/nav_goal', 10)
        self.action_pub = self.create_publisher(String, '/action_request', 10)
        # Single canonical speech output bus — tts_node and whatsapp_listener
        # both subscribe to /robot_speech. Do NOT also publish to /tts_text to
        # avoid double-processing by whatsapp_listener.
        self.speech_pub = self.create_publisher(String, '/robot_speech', 10)

        # ── History: record (user, answer) pairs for follow-up context ─────────
        # Subscribe to /robot_speech to capture what the brain outputs.
        # Use a deduplication flag so we only record once per answer.
        self._history_sub = self.create_subscription(
            String, '/robot_speech', self._record_answer, 10
        )
        self._last_user_input = ''
        self.history = deque(maxlen=5)   # (user_text, robot_answer) tuples

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('ollama_url', 'http://127.0.0.1:11434/api/generate')
        self.declare_parameter('router_model', 'gemma3:270m')
        self.declare_parameter('router_timeout', 30)

        self.ollama_url     = self.get_parameter('ollama_url').get_parameter_value().string_value
        self.router_model   = self.get_parameter('router_model').get_parameter_value().string_value
        self.router_timeout = self.get_parameter('router_timeout').get_parameter_value().integer_value

        self._check_ollama()
        self.get_logger().info('AI-SHA Brain: Router Active')

    # ── Startup ────────────────────────────────────────────────────────────────

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
            self.get_logger().error(f'Ollama unreachable: {e}')

    # ── Intent classification ──────────────────────────────────────────────────

    def classify_intent(self, text):
        keyword_result = self._keyword_classify(text)
        if keyword_result is not None:
            return keyword_result
        return self._llm_classify(text)

    def _keyword_classify(self, text):
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

        return None

    def _llm_classify(self, text):
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

        self.get_logger().info('Defaulting -> ADMIN')
        return {"intent": "ADMIN"}

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _on_detection(self, msg):
        try:
            self._last_detection = json.loads(msg.data)
        except Exception:
            self._last_detection = {'raw': msg.data}

    def _say(self, text):
        """Publish direct brain responses to /robot_speech."""
        msg = String()
        msg.data = text
        self.speech_pub.publish(msg)

    def _record_answer(self, msg):
        """Record answer into history when brain sees /robot_speech."""
        answer = msg.data.strip()
        if self._last_user_input and answer:
            self.history.append((self._last_user_input, answer))
            self._last_user_input = ''   # clear so next message starts fresh

    # ── Main routing callback ──────────────────────────────────────────────────

    def listener_callback(self, msg):
        """ROS2 subscription callback — returns immediately, work done in thread.

        Blocking the executor here (e.g. with requests.post to Ollama) would
        freeze ALL subscriptions on this node for the duration of the LLM call.
        We deduplicate synchronously (cheap), then hand off to a daemon thread.
        """
        user_input = msg.data.strip()
        if not user_input:
            return

        # ── Deduplication (cheap — done in callback thread) ────────────────────
        now = time.time()
        if (user_input == self._last_processed_text and
                now - self._last_processed_time < self._debounce_secs):
            self.get_logger().debug(
                f'Deduplicated (within {self._debounce_secs}s): {user_input[:60]}'
            )
            return

        self._last_processed_text = user_input
        self._last_processed_time = now

        # ── Hand off to background thread (non-blocking) ───────────────────────
        # classify_intent() may call requests.post(Ollama) which can take
        # several seconds. Running it in a daemon thread keeps the ROS2
        # executor free to handle other callbacks (STT, WhatsApp, vision, etc.)
        threading.Thread(
            target=self._route,
            args=(user_input,),
            daemon=True
        ).start()

    def _route(self, user_input: str):
        """Classify intent and publish — runs in a background daemon thread."""
        self.get_logger().info(f'Heard: {user_input}')
        self._last_user_input = user_input

        decision = self.classify_intent(user_input)
        intent = decision.get("intent", "ADMIN")

        out_msg = String()
        if intent == "ADMIN":
            self.get_logger().info("Route -> ADMIN (Knowledge Base)")
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
