import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


class AdminNode(Node):
    def __init__(self):
        super().__init__('ai_sha_admin')
        self.subscription = self.create_subscription(String, '/admin_task', self.handle_query, 10)
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

        kb_path = self.get_parameter('knowledge_db_path').get_parameter_value().string_value
        ollama_url = self.get_parameter('ollama_url').get_parameter_value().string_value
        llm_model = self.get_parameter('llm_model').get_parameter_value().string_value
        llm_timeout = self.get_parameter('llm_timeout').get_parameter_value().double_value

        self.get_logger().info(f'Connecting to Knowledge Base at: {kb_path}')
        self.query_engine = None

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
            self.query_engine = self.index.as_query_engine(llm=self.llm)
            self.get_logger().info('Knowledge Base Online')

        except Exception as e:
            self.get_logger().error(f'Failed to initialize knowledge base: {e}')
            self.get_logger().error('AdminNode will return fallback responses until KB is available')

    def handle_query(self, msg):
        try:
            data = json.loads(msg.data)
            user_question = data.get("details", "")
            if not user_question:
                return

            self.get_logger().info(f'Query: {user_question}')

            if self.query_engine is None:
                answer = "I'm sorry, my knowledge base is not available right now. Please try again later."
            else:
                response = self.query_engine.query(user_question)
                answer = str(response).strip()
                if not answer:
                    answer = "I could not find information about that. Could you rephrase your question?"

            out_msg = String()
            out_msg.data = answer
            self.speech_publisher.publish(out_msg)
            self.get_logger().info(f'Answer: {answer[:100]}...' if len(answer) > 100 else f'Answer: {answer}')

        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid JSON on /admin_task: {msg.data}')
        except Exception as e:
            self.get_logger().error(f'Query error: {e}')
            out_msg = String()
            out_msg.data = "I encountered an error processing your question. Please try again."
            self.speech_publisher.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = AdminNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
