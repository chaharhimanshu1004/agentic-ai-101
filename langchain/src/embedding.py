from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
import os
from dotenv import load_dotenv


load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    output_dimensionality=1536,
)


def embed_chunks(chunks: list[Document]) -> list[list[float]]:
    vectors = embeddings.embed_documents([c.page_content for c in chunks])
    print(f"Generated {len(vectors)} embeddings, dimension {len(vectors[0])}")
    return vectors
