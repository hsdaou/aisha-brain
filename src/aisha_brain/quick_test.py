import os
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# 1. Setup local models
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = Ollama(model="llama3.2", request_timeout=120.0)

# 2. Load the index we just built
storage_context = StorageContext.from_defaults(persist_dir="./aisha_knowledge_db")
index = load_index_from_storage(storage_context, embed_model=embed_model)

# 3. Ask a specific question from the PDF
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("Based on the school calendar, when are the holidays in March?")

print("\n" + "="*30)
print(f"🤖 AI-SHA RESPONSE:\n{response}")
print("="*30)
