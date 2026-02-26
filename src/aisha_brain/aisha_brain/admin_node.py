import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import os
import time
import threading
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


SYSTEM_PROMPT = """You are AI-SHA, the administrative assistant robot for the International School of Choueifat (ISC) in Sharjah.

Your job is strictly limited to answering administrative questions about ISC-Sharjah using the provided school knowledge base.

Key facts you must know:
- SLO stands for Student Life Organization (also called SABIS Student Life Organization). It is a student-run leadership and community organization with departments like Academic, Discipline, Sports, Arts, Community Service, and more.
- A prefect is a student leader in the SLO.
- The school phone is +971 6 558 2211 and email is info@iscsharjah.sabis.net.

STRICT RULES — follow these without exception:
1. You ONLY answer questions about: schedules, exams, fees, admissions, school facilities, staff contacts, school events, and other ISC-Sharjah administrative information.
2. You MUST REFUSE any request for academic help, tutoring, or homework assistance. This includes: explaining concepts, solving equations, writing essays, answering trivia, summarizing books, or any general knowledge question not related to school administration. Respond firmly: "I am an administrative assistant. Please ask your teacher for academic help."
3. Always use the retrieved school knowledge to answer. Do not invent information.
4. If a follow-up question refers to something mentioned earlier in the conversation, use that context.
5. Keep answers concise and friendly (2-4 sentences for most questions).
6. For exam schedule questions: if the context contains exam entries, list EXACTLY what is found (grade, subject, date, time). If a specific grade or subject is NOT listed in the context, say explicitly: "No exam has been announced for [grade/subject] yet. The only announced exams are: [list what IS in the context]. For the full schedule, contact the school at +971 6 558 2211."
7. If the answer is genuinely not in the knowledge base, say so politely and suggest contacting the school office at +971 6 558 2211.
8. Do not reveal that you are built on an LLM or that you use a knowledge base.
"""


class AdminNode(Node):
    def __init__(self):
        super().__init__('ai_sha_admin')
        self.subscription = self.create_subscription(String, '/admin_task', self.handle_query, 10)
        # Publish to /robot_speech — tts_node and whatsapp_listener both subscribe here
        self.speech_publisher = self.create_publisher(String, '/robot_speech', 10)

        # Resolve KB path via ament_index so it works regardless of where
        # Python loads this file from (site-packages vs source tree).
        try:
            from ament_index_python.packages import get_package_share_directory
            default_kb_path = os.path.join(
                get_package_share_directory('aisha_brain'),
                'aisha_knowledge_db'
            )
        except Exception:
            # Fallback for running outside a colcon workspace
            default_kb_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'aisha_knowledge_db'
            )
        self.declare_parameter('knowledge_db_path', default_kb_path)
        self.declare_parameter('ollama_url', 'http://127.0.0.1:11434')
        self.declare_parameter('llm_model', 'llama3.2')
        self.declare_parameter('llm_timeout', 120.0)
        self.declare_parameter('similarity_top_k', 15)

        kb_path = self.get_parameter('knowledge_db_path').get_parameter_value().string_value
        ollama_url = self.get_parameter('ollama_url').get_parameter_value().string_value
        llm_model = self.get_parameter('llm_model').get_parameter_value().string_value
        llm_timeout = self.get_parameter('llm_timeout').get_parameter_value().double_value
        similarity_top_k = self.get_parameter('similarity_top_k').get_parameter_value().integer_value

        # ── Deduplication: prevent the same question from being processed twice ──
        # Identical /admin_task messages within this window are silently dropped.
        # This guards against duplicate publishes from brain_node's dual subs.
        self._last_query_text: str = ''
        self._last_query_time: float = 0.0
        self._query_debounce_secs: float = 5.0

        self.get_logger().info(f'Connecting to Knowledge Base at: {kb_path}')
        self.index = None
        self.embed_model = None
        self.llm = None
        self.similarity_top_k = similarity_top_k

        try:
            db = chromadb.PersistentClient(path=kb_path)
            chroma_collection = db.get_or_create_collection("school_info")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

            self.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.llm = Ollama(model=llm_model, base_url=ollama_url, request_timeout=llm_timeout)

            self.index = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=self.embed_model
            )
            self.get_logger().info(f'Knowledge Base Online (top_k={similarity_top_k})')

        except Exception as e:
            self.get_logger().error(f'Failed to initialize knowledge base: {e}')
            self.get_logger().error('AdminNode will return fallback responses until KB is available')

    def _build_messages(self, history: list, user_question: str, context_str: str) -> list:
        """Build a list of ChatMessage objects with system prompt, history, and current question."""
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=SYSTEM_PROMPT + f"\n\nRelevant school knowledge:\n{context_str}"
            )
        ]
        # Add conversation history
        for turn in history:
            messages.append(ChatMessage(role=MessageRole.USER, content=turn.get('user', '')))
            messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=turn.get('assistant', '')))

        # Add current question
        messages.append(ChatMessage(role=MessageRole.USER, content=user_question))
        return messages

    def handle_query(self, msg):
        """ROS2 subscription callback — returns immediately, work done in thread.

        self.llm.chat() is a synchronous blocking call that can take 5-120 seconds.
        Running it here would freeze the entire node executor, blocking all other
        subscriptions for the duration. We parse and deduplicate (cheap), then
        hand the real work off to a daemon thread.
        """
        try:
            data = json.loads(msg.data)
            user_question = data.get("details", "").strip()
            history = data.get("history", [])

            if not user_question:
                return

            # ── Deduplication (cheap — done in callback thread) ────────────────
            now = time.time()
            if (user_question == self._last_query_text and
                    now - self._last_query_time < self._query_debounce_secs):
                self.get_logger().warn(
                    f'Duplicate /admin_task ignored (within {self._query_debounce_secs}s): '
                    f'{user_question[:60]}'
                )
                return
            self._last_query_text = user_question
            self._last_query_time = now

        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid JSON on /admin_task: {msg.data}')
            return

        # ── Hand off to background thread (non-blocking) ───────────────────────
        # Vector retrieval + LLM inference can take many seconds. Running in a
        # daemon thread keeps the ROS2 executor free for other callbacks.
        threading.Thread(
            target=self._process_query,
            args=(user_question, history),
            daemon=True
        ).start()

    def _process_query(self, user_question: str, history: list):
        """Run RAG retrieval + LLM inference — called from a background thread."""
        try:
            self.get_logger().info(f'Query: {user_question}')
            if history:
                self.get_logger().info(f'  with {len(history)} prior turn(s) of context')

            if self.index is None:
                self._publish("I'm sorry, my knowledge base is not available right now. Please try again later.")
                return

            # Step 1: Retrieve relevant context from the vector store
            self.get_logger().info('Retrieving context from knowledge base...')
            retriever = self.index.as_retriever(
                similarity_top_k=self.similarity_top_k,
                embed_model=self.embed_model
            )

            # Build retrieval query: prepend last user turn for follow-up context
            retrieval_query = user_question
            if history:
                last_user = history[-1].get('user', '')
                if last_user and last_user.lower() != user_question.lower():
                    retrieval_query = f"{last_user} {user_question}"

            nodes = retriever.retrieve(retrieval_query)
            context_str = "\n\n".join(
                f"[Source: {n.metadata.get('file_name', 'knowledge base')}]\n{n.get_content()}"
                for n in nodes
            ) if nodes else "No specific context found."

            self.get_logger().info(f'Retrieved {len(nodes)} chunks for query')

            # Step 2: Build chat messages with system prompt + history + context
            messages = self._build_messages(history, user_question, context_str)

            # Step 3: Call LLM (blocking — safe here because we are in a daemon thread)
            self.get_logger().info(f'Calling Ollama ({self.llm.model})...')
            response = self.llm.chat(messages)
            answer = str(response.message.content).strip()

            if not answer:
                answer = "I could not find information about that. Could you rephrase your question?"

            self.get_logger().info(f'Answer: {answer[:120]}...' if len(answer) > 120 else f'Answer: {answer}')
            self._publish(answer)
            self.get_logger().info('Published to /robot_speech')

        except Exception as e:
            self.get_logger().error(f'Query error: {type(e).__name__}: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            self._publish("I encountered an error processing your question. Please try again.")

    def _publish(self, text: str):
        out_msg = String()
        out_msg.data = text
        self.speech_publisher.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = AdminNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
