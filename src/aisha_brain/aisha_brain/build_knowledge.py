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

    # ── Preprocess: split each markdown ## section into its own Document ────────
    # This prevents chunks from crossing section boundaries (e.g. Grade 8
    # exam entries leaking into a Grade 9 chunk).  Each section's header is
    # prepended to every bullet line so the embedding always carries the
    # grade/topic context even in small chunks.
    from llama_index.core import Document
    new_docs = []
    for doc in documents:
        file_name = doc.metadata.get('file_name', '')
        if file_name.endswith('.md'):
            lines = doc.get_content().split('\n')
            # Gather the top-level title (# ...) as global context
            global_title = ''
            sections = []          # list of (header, [lines])
            current_header = ''
            current_lines = []

            for line in lines:
                if line.startswith('# ') and not line.startswith('## '):
                    global_title = line[2:].strip()
                elif line.startswith('## '):
                    # Flush previous section
                    if current_header and current_lines:
                        sections.append((current_header, current_lines))
                    current_header = line[3:].strip()
                    current_lines = []
                else:
                    current_lines.append(line)
            # Flush last section
            if current_header and current_lines:
                sections.append((current_header, current_lines))

            if sections:
                for header, sec_lines in sections:
                    # Inject header into every bullet line for embedding context
                    tagged = []
                    for sl in sec_lines:
                        if sl.startswith('- **'):
                            tagged.append(f"- [{header}] **" + sl[4:])
                        else:
                            tagged.append(sl)
                    section_text = f"# {global_title}\n## {header}\n" + '\n'.join(tagged)
                    new_docs.append(Document(
                        text=section_text.strip(),
                        metadata={**doc.metadata, 'section': header}
                    ))
            else:
                # No ## sections found — keep as-is
                new_docs.append(doc)
        else:
            new_docs.append(doc)
    documents = new_docs
    print(f'Expanded to {len(documents)} section-level documents.')

    # Chunk size 256 tokens with 32 overlap — large enough to keep several
    # exam entries together with their grade header, small enough to stay
    # focused.  Each chunk is guaranteed to be from a single grade section.
    sentence_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=32)

    print(f'Building index from {len(documents)} documents...')
    # Build vector store index locally
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[sentence_splitter],
    )
    
    print(f'Successfully ingested {len(documents)} source documents.')
    print(f'Knowledge base saved to local database at: {CHROMA_PATH}')


if __name__ == '__main__':
    if '--scrape' in sys.argv:
        print("WARNING: Scraping is strictly prohibited by AI-SHA constraints.")
        print("Ignoring --scrape flag. Only local files will be processed.")
    
    print('Building knowledge index from existing local files...')
    build_index()

