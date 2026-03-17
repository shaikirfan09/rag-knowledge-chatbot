from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import tempfile


# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect Qdrant
client = QdrantClient(
    host="qdrant",
    port=6333
)

vectorstore = Qdrant(
    client=client,
    collection_name="knowledge_base",
    embeddings=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})


# Load LLM
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def ingest_document(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(path)
        docs = loader.load()

    else:
        text = open(path).read()
        docs = [{"page_content": text}]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(docs)

    Qdrant.from_documents(
        split_docs,
        embeddings,
        host="qdrant",
        port=6333,
        collection_name="knowledge_base"
    )


def ask_question(question):

    docs = retriever.invoke(question)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer clearly using the context.
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=150
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    sources = [doc.page_content[:100] for doc in docs]

    return answer, sources