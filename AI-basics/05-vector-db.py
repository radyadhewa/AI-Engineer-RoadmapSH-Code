"""
Vector DB usage flow:

Content -> Embeddings -> Vector Embeddings -> Vector Database
query -> Embeddings -> Vector Embeddings -> Search Vector Database -> query results
"""

import os
import typing as t
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

import spacy
nlp = spacy.load("en_core_web_sm")
stopwords = nlp.Defaults.stop_words

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "faiss_meta.json"

# sample data
Document = t.Dict[str, t.Any]
def example_documents() -> t.List[Document]:
    """Create a small sample corpus with metadata to demonstrate functionality."""
    docs = [
        {"id": "doc1", "text": "Nasi goreng is Indonesia's iconic fried rice, often served with kecap manis, a fried egg, and prawn crackers.", "metadata": {"topic": "food", "tags": ["nasi-goreng", "cuisine", "street-food"]}},
        {"id": "doc2", "text": "Bali is known for its beaches, temples, and terraced rice paddies; it's a top destination for both relaxation and cultural experiences.", "metadata": {"topic": "travel", "tags": ["bali", "beaches", "temples"]}},
        {"id": "doc3", "text": "Traditional arts such as wayang kulit (shadow puppetry) and gamelan music are central to many Indonesian cultural ceremonies.", "metadata": {"topic": "culture", "tags": ["wayang-kulit", "gamelan", "tradition"]}},
        {"id": "doc4", "text": "Jakarta's street food scene offers delights like sate, bakso, and gorengan, reflecting the nation's diverse regional flavors.", "metadata": {"topic": "street-food", "tags": ["jakarta", "sate", "bakso"]}},
        {"id": "doc5", "text": "Komodo National Park is famous for its Komodo dragons and rugged islands, attracting eco-tourists interested in wildlife and marine biodiversity.", "metadata": {"topic": "nature", "tags": ["komodo", "ecotourism", "wildlife"]}},
    ]
    return docs

# preprocess text data
def preprocess_text(text: str):
    doc = nlp(str(text))
    preprocessed = []
    for token in doc:
        if token not in stopwords or not token.is_punct or token.like_num or token.is_space:
            preprocessed.append(token.lemma_.lower().strip())
    return " ".join(preprocessed)

# Preprocess documents
docs = example_documents()
texts = [preprocess_text(d["text"]) for d in docs]

# Embedding generation
model = SentenceTransformer(EMBED_MODEL)
vector = model.encode(texts, convert_to_numpy=True).astype("float32")

# FAISS database setup and store
dim = vector.shape[1]

index: t.Any = faiss.IndexFlatL2(dim)
index.add(vector)

# search query to faiss
def search_faiss(query: str, top_k: int = 3):
    query_vec = model.encode([preprocess_text(query)], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vec, top_k)
    results = []
    for idx in indices[0]:
        if idx < len(docs):
            results.append(docs[idx])
    return results

# example search
query = "Where to go in Indonesia?"
results = search_faiss(query)
print(f"Search Results for query: '{query}'")
for res in results:
    print(f"- {res['id']}: {res['text']}")
