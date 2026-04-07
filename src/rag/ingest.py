import os
import structlog
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logger = structlog.get_logger()

DOCS_DIR = "data/clinical_docs"
DB_DIR = "data/chroma_db"

def build_vector_store():
    if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
        logger.error("Missing PDF files", path=DOCS_DIR)
        print(f"Error: Missing documents at for vector store")
        return
    
    logger.info("Loading PDF from directory", directory=DOCS_DIR)
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    documents = loader.load()

    logger.info("Chunking Texts")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents=documents)

    logger.info("Initializing document embedding model")
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    logger.info("Generating document embeddings and saving to chroma")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_DIR
    )

    logger.info("Successfully built vector database")
    print(f"Success! Clinical PDFs embedded locally and saved to vector store")


if __name__ == "__main__":
    build_vector_store()