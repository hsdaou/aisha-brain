"""Build and optionally scrape the AI-SHA knowledge base.

Usage:
    python build_knowledge.py           # rebuild index from existing files only
    python build_knowledge.py --scrape  # scrape ISC-Sharjah website first, then rebuild
"""

import os
import re
import sys
import time

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(_PKG_DIR, 'aisha_knowledge_db')
DATA_FOLDER = os.path.join(_PKG_DIR, 'aisha_raw_data')

# ---------------------------------------------------------------------------
# Full ISC-Sharjah site map — 27 pages (expanded from the original 4)
# ---------------------------------------------------------------------------
TARGET_URLS = [
    # General / About
    ('General Info',         'https://iscsharjah.sabis.net/contact-us'),
    ('About School',         'https://iscsharjah.sabis.net/our-school/member-of-the-sabis-network'),
    ('Mission & Values',     'https://iscsharjah.sabis.net/our-school/mission-statement'),
    ('Campus Facilities',    'https://iscsharjah.sabis.net/our-school/campus-facilities'),
    ('School Calendar',      'https://iscsharjah.sabis.net/our-school/school-calendar'),
    ('Downloads',            'https://iscsharjah.sabis.net/our-school/downloads'),
    # Academic Life
    ('Academic Overview',    'https://iscsharjah.sabis.net/academic-life/studying-at-sharjah'),
    ('Kindergarten',         'https://iscsharjah.sabis.net/academic-life/kindergarten'),
    ('Lower School',         'https://iscsharjah.sabis.net/academic-life/lower-school'),
    ('Middle School',        'https://iscsharjah.sabis.net/academic-life/middle-school'),
    ('High School',          'https://iscsharjah.sabis.net/academic-life/high-school'),
    ('Student Support',      'https://iscsharjah.sabis.net/academic-life/student-support-service'),
    ('Education Technology', 'https://iscsharjah.sabis.net/academic-life/education-technology'),
    # Student Life (SLO)
    ('What is SLO',          'https://iscsharjah.sabis.net/student-life/slo'),
    ('Role of SLO',          'https://iscsharjah.sabis.net/student-life/role'),
    ('SLO Honor Code',       'https://iscsharjah.sabis.net/student-life/honor-code'),
    ('SLO Inside Look',      'https://iscsharjah.sabis.net/student-life/look-inside'),
    ('The Prefect',          'https://iscsharjah.sabis.net/student-life/prefect'),
    # Extra-Curricular
    ('School Activities',    'https://iscsharjah.sabis.net/extra-curricular-activities/school-activities'),
    ('After School',         'https://iscsharjah.sabis.net/extra-curricular-activities/after-school-activities'),
    ('Weekend Activities',   'https://iscsharjah.sabis.net/extra-curricular-activities/weekend-activities'),
    ('Regional & Intl',      'https://iscsharjah.sabis.net/extra-curricular-activities/regional-international-experiences'),
    # Admissions
    ('Prospective Families', 'https://iscsharjah.sabis.net/admissions/prospective-families'),
    ('Admissions Policy',    'https://iscsharjah.sabis.net/admissions/admission-policy'),
    ('Admissions Process',   'https://iscsharjah.sabis.net/admissions/admissions-process'),
    ('Tuition & Fees',       'https://iscsharjah.sabis.net/admissions/tuition-fees'),
    ('Privacy Notice',       'https://iscsharjah.sabis.net/admissions/general-privacy-notice'),
]

HEADERS = {
    'User-Agent': 'AI-SHA-Bot/1.0 (ISC-Sharjah internal assistant)'
}


# ---------------------------------------------------------------------------
# Scraping helpers
# ---------------------------------------------------------------------------

def _clean_text(text):
    """Collapse whitespace and drop nav-residue short lines."""
    text = re.sub(r'[ \t]+', ' ', text)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) < 15:
            continue
        if line.lower() in ('learn more', 'visit site', 'menu', 'key links', 'find us on:'):
            continue
        lines.append(line)
    return '\n'.join(lines)


def _extract_main_content(soup):
    """Strip nav/footer noise and return the main page text."""
    for tag in soup.find_all(['script', 'style', 'nav', 'footer',
                               'iframe', 'noscript', 'svg']):
        tag.decompose()

    for div in soup.find_all('div', class_=re.compile(r'mega-menu|network|overlay', re.I)):
        div.decompose()

    for tag in soup.find_all(True):
        classes = ' '.join(tag.get('class', []))
        tag_id = tag.get('id', '')
        if re.search(r'menu|sidebar|breadcrumb|cookie|banner', classes + tag_id, re.I):
            tag.decompose()

    main = soup.find('main') or soup.find(id='main-content') or soup.find('article')
    if main:
        return _clean_text(main.get_text(separator='\n'))

    body = soup.find('body')
    return _clean_text(body.get_text(separator='\n')) if body else ''


def _chunk_text(text, max_chunk_chars=500):
    """Split text into embedding-friendly chunks."""
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    current = ''

    for para in paragraphs:
        if len(current) + len(para) + 1 <= max_chunk_chars:
            current = f'{current}\n{para}'.strip()
        else:
            if current:
                chunks.append(current)
            if len(para) > max_chunk_chars:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                sub = ''
                for s in sentences:
                    if len(sub) + len(s) + 1 <= max_chunk_chars:
                        sub = f'{sub} {s}'.strip()
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = s
                current = sub
            else:
                current = para

    if current:
        chunks.append(current)

    return [c for c in chunks if len(c) > 30]


def scrape_and_save():
    """Scrape all TARGET_URLS and save clean markdown to DATA_FOLDER."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print('ERROR: Missing dependencies. Run: pip install requests beautifulsoup4 lxml')
        sys.exit(1)

    os.makedirs(DATA_FOLDER, exist_ok=True)
    print(f'Scraping {len(TARGET_URLS)} pages...')

    for category, url in TARGET_URLS:
        try:
            print(f'  [{category}] {url}')
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, 'lxml')
            text = _extract_main_content(soup)

            if not text:
                print('    WARNING: No content extracted (page may require JS rendering)')
                continue

            filename = url.rstrip('/').split('/')[-1] or 'homepage'
            out_path = os.path.join(DATA_FOLDER, f'{filename}.md')
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(f'# {category}\n\n{text}\n')

            chunks = _chunk_text(text)
            print(f'    -> {len(chunks)} chunks saved to {filename}.md')
            time.sleep(0.5)

        except Exception as e:
            print(f'    FAILED: {e}')

    print('Scraping complete.')


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index():
    """Ingest all files in DATA_FOLDER into a ChromaDB vector store."""
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    import chromadb

    print(f'Ingesting data from {DATA_FOLDER} ...')

    db = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        db.delete_collection('school_info')
        print('Cleared old knowledge base.')
    except Exception:
        pass  # Collection may not exist yet (fresh build)

    chroma_collection = db.get_or_create_collection('school_info')
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = HuggingFaceEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')
    documents = SimpleDirectoryReader(DATA_FOLDER).load_data()

    # Use a small chunk size so each grade-entry / calendar section gets
    # its own embedding vector for precise retrieval.
    # chunk_size=256 tokens ≈ 2-3 sentences; overlap=32 preserves context.
    splitter = SentenceSplitter(chunk_size=256, chunk_overlap=32)

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
    )
    print(f'Indexed {len(documents)} source documents. Knowledge base saved to {CHROMA_PATH}')


if __name__ == '__main__':
    if '--scrape' in sys.argv:
        scrape_and_save()
    else:
        print('Rebuilding index from existing files (use --scrape to re-crawl website)')
    build_index()
