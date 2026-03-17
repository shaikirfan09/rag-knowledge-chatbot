from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

print("Loading knowledge file...")

loader = TextLoader("data/knowledge.txt")
documents = loader.load()

print("Creating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Connecting to Qdrant server...")

client = QdrantClient(
    host="qdrant",
    port=6333
)

print("Storing documents in vector database...")

Qdrant.from_documents(
    documents,
    embeddings,
    host="qdrant",
    port=6333,
    collection_name="knowledge_base"
)

print("Knowledge successfully stored in Qdrant!")