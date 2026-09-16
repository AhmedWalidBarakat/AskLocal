"""
Ingestion pipeline: loads PDFs from this folder, splits them into chunks,
embeds each chunk locally via Ollama, and stores the vectors in a local
Chroma database (./chroma_db). Run this once whenever you add/change PDFs.
"""

import glob
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DOCS_FOLDER = os.path.dirname(os.path.abspath(__file__))
DB_FOLDER = os.path.join(DOCS_FOLDER, "chroma_db")
EMBED_MODEL = "nomic-embed-text"


def load_documents():
    pdf_paths = glob.glob(os.path.join(DOCS_FOLDER, "*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {DOCS_FOLDER}")

    documents = []
    for path in pdf_paths:
        print(f"Loading {os.path.basename(path)}...")
        loader = PyPDFLoader(path)
        documents.extend(loader.load())
    return documents


def main():
    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    print(f"Embedding with '{EMBED_MODEL}' and saving to {DB_FOLDER} ...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    batch_size = 50
    vectordb = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        if vectordb is None:
            vectordb = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=DB_FOLDER,
            )
        else:
            vectordb.add_documents(batch)
        print(f"  Embedded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    print("Done! Vector database is ready at ./chroma_db")


if __name__ == "__main__":
    main()
