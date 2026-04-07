from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.utilities import SQLDatabase
import structlog

logger = structlog.get_logger()

@tool
def query_clinical_guidelines(query: str) -> str:
    """
    Use this tool when the user asks a general medical or clinical question 
    (e.g., 'What is a good resting heart rate?', 'How much sleep do I need?').
    Do NOT use this tool to query the user's personal data.
    """

    logger.info("tool invoked: clinical_rag", query=query)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vector_store = Chroma(persist_directory='data/chroma_db', embedding_function=embeddings)

    docs = vector_store.similarity_search(query=query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

db = SQLDatabase.from_uri("sqlite:///data/health.db")

@tool
def query_user_health_data(sql_query: str) -> str:
    """
    Use this tool to execute a SQLite query against the user's health database.
    The database has ONE table named 'daily_biometrics'.
    Columns: user_id (TEXT), date (TEXT), resting_heart_rate (REAL), total_sleep_minutes (INTEGER), calories (REAL).
    
    Input MUST be a valid, raw SQLite query string (e.g., "SELECT AVG(resting_heart_rate) FROM daily_biometrics").
    Return the raw tabular data.
    """

    logger.info("tool invoked: sql execution")

    try:
        result = db.run(sql_query)
        return str(result) if result else "Query returned no results."
    except Exception as e:
        logger.error("SQL execution failed", error=str(e))
        return f"Error executing SQL: {e}. Please rewrite the query and try again."
