import asyncio
import os
import sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

# 1. SETUP: Paths relative to the package directory (works from any cwd)
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(_PKG_DIR, "aisha_knowledge_db")
DATA_FOLDER = os.path.join(_PKG_DIR, "aisha_raw_data")

# 2. TARGET LIST: The specific pages with the real data
TARGET_URLS = [
    "https://iscsharjah.sabis.net/",
    "https://iscsharjah.sabis.net/admissions/tuition-fees",
    "https://iscsharjah.sabis.net/student-life/slo",
    "https://iscsharjah.sabis.net/student-life/look-inside"
]

async def scrape_and_save():
    # Lazy import — crawl4ai is only needed when scraping, not when rebuilding
    from crawl4ai import AsyncWebCrawler

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    print(f"Starting Multi-Page Crawl...")
    async with AsyncWebCrawler(verbose=True) as crawler:
        for url in TARGET_URLS:
            print(f"   ... Fetching: {url}")
            result = await crawler.arun(url=url)

            # Create a filename from the URL (e.g., 'tuition-fees.md')
            filename = url.split("/")[-1]
            if filename == "": filename = "homepage"

            # Save raw Markdown to disk
            with open(f"{DATA_FOLDER}/{filename}.md", "w", encoding="utf-8") as f:
                f.write(result.markdown)
    print("Scraping Complete. Files saved to aisha_raw_data/")

def build_index():
    print(f"Ingesting Data from {DATA_FOLDER} ...")

    # Initialize Database
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    # Delete existing collection to rebuild from scratch
    try:
        db.delete_collection("school_info")
        print("Cleared old knowledge base.")
    except ValueError:
        pass
    chroma_collection = db.get_or_create_collection("school_info")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Setup Embedding Model
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # LOAD EVERYTHING: .md files, .txt files, .pdf files from the data folder
    documents = SimpleDirectoryReader(DATA_FOLDER).load_data()

    # Create Index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model
    )
    print(f"Indexed {len(documents)} documents. Knowledge Base saved to {CHROMA_PATH}")

async def main():
    await scrape_and_save()  # Step 1: Get content from Web
    build_index()            # Step 2: Read Web content + Manual PDFs

if __name__ == "__main__":
    # Usage:
    #   python build_knowledge.py          -> rebuild index only (from existing files)
    #   python build_knowledge.py --scrape -> scrape website first, then rebuild
    if "--scrape" in sys.argv:
        asyncio.run(main())
    else:
        print("Rebuilding index from existing files (use --scrape to re-crawl website)")
        build_index()
