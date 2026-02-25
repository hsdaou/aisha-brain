"""Build the AI-SHA knowledge base from local files only.

Usage:
    python build_knowledge.py
"""

import os
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(_PKG_DIR, 'aisha_knowledge_db')
DATA_FOLDER = os.path.join(_PKG_DIR, 'aisha_raw_data')

# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index():
    """Ingest all files in DATA_FOLDER into a local ChromaDB vector store."""
    try:
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
        from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        import chromadb
    except ImportError as e:
        print(f"ERROR: Missing dependencies. Please run the install_local_deps.sh script. Details: {e}")
        sys.exit(1)

    if not os.path.exists(DATA_FOLDER):
        print(f"ERROR: Data folder not found at {DATA_FOLDER}")
        sys.exit(1)

    print(f'Ingesting data from {DATA_FOLDER} ...')

    # Initialize ChromaDB locally
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        db.delete_collection('school_info')
        print('Cleared old knowledge base.')
    except Exception:
        pass  # Collection may not exist yet (fresh build)

    chroma_collection = db.get_or_create_collection('school_info')
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Use a local HuggingFace embedding model (downloads once if not cached, then completely offline)
    embed_model = HuggingFaceEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    # Read local documents
    print('Reading local markdown/text files...')
    documents = SimpleDirectoryReader(DATA_FOLDER).load_data()

    if not documents:
        print(f"No documents found in {DATA_FOLDER}")
        sys.exit(0)

    # MarkdownNodeParser splits on headers so each section becomes its own chunk
    # with the header included as context.
    # Sections larger than 512 tokens are further split by SentenceSplitter.
    md_parser = MarkdownNodeParser()
    sentence_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)

    print(f'Building index from {len(documents)} documents...')
    # Build vector store index locally
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[md_parser, sentence_splitter],
    )
    
    print(f'Successfully ingested {len(documents)} source documents.')
    print(f'Knowledge base saved to local database at: {CHROMA_PATH}')


if __name__ == '__main__':
    if '--scrape' in sys.argv:
        print("WARNING: Scraping is strictly prohibited by AI-SHA constraints.")
        print("Ignoring --scrape flag. Only local files will be processed.")
    
    print('Building knowledge index from existing local files...')
    build_index()

